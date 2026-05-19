"""Monte Carlo cricket simulations for live win and over-run forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np


OUTCOMES = np.array([0, 1, 2, 4, 6, -1])
BASE_PROBABILITIES = np.array([0.28, 0.31, 0.14, 0.13, 0.07, 0.07])


@dataclass(frozen=True)
class SimulationState:
    """State needed to simulate a chase from the current ball."""

    score: int
    wickets: int
    target: int
    balls_left: int
    pressure: float = 50.0
    batter_settle_score: float = 50.0
    bowler_fatigue: float = 30.0


def poisson_pmf(k: int, lam: float) -> float:
    """Return P(X=k) for runs-per-over modeling."""

    if k < 0:
        return 0.0
    if lam <= 0:
        raise ValueError("lambda must be positive")
    return float(exp(-lam) * pow(lam, k) / factorial(k))


def ball_probabilities(pressure: float, batter_settle_score: float, bowler_fatigue: float) -> np.ndarray:
    """Adjust blueprint base ball probabilities for live context."""

    probs = BASE_PROBABILITIES.astype(float).copy()
    pressure_factor = max(0.0, min(1.0, pressure / 100.0))
    settle_factor = max(0.0, min(1.0, batter_settle_score / 100.0))
    fatigue_factor = max(0.0, min(1.0, bowler_fatigue / 100.0))

    probs[0] += pressure_factor * 0.05
    probs[5] += pressure_factor * 0.04
    probs[3] += settle_factor * 0.03
    probs[4] += settle_factor * 0.03
    probs[3] += fatigue_factor * 0.025
    probs[4] += fatigue_factor * 0.02
    probs[1] -= pressure_factor * 0.03
    probs = np.clip(probs, 0.01, None)
    return probs / probs.sum()


def simulate_chase(state: SimulationState, simulations: int = 10_000, seed: int | None = 42) -> dict[str, float]:
    """Run stochastic chase simulations and return win probability."""

    if state.balls_left < 0:
        raise ValueError("balls_left cannot be negative")
    if not 0 <= state.wickets <= 10:
        raise ValueError("wickets must be between 0 and 10")
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)
    probs = ball_probabilities(state.pressure, state.batter_settle_score, state.bowler_fatigue)
    wins = 0
    final_scores: list[int] = []
    for _ in range(simulations):
        score = state.score
        wickets = state.wickets
        for _ball in range(state.balls_left):
            outcome = int(rng.choice(OUTCOMES, p=probs))
            if outcome == -1:
                wickets += 1
                if wickets >= 10:
                    break
            else:
                score += outcome
                if score >= state.target:
                    break
        wins += int(score >= state.target)
        final_scores.append(score)

    return {
        "win_probability": round(wins / simulations, 4),
        "projected_score_mean": round(float(np.mean(final_scores)), 2),
        "projected_score_p10": round(float(np.percentile(final_scores, 10)), 2),
        "projected_score_p90": round(float(np.percentile(final_scores, 90)), 2),
    }


def predict_over_total(lambda_runs: float = 8.2, max_runs: int = 24) -> dict[int, float]:
    """Return run total distribution for an over using Poisson probabilities."""

    if max_runs <= 0:
        raise ValueError("max_runs must be positive")
    raw = {k: poisson_pmf(k, lambda_runs) for k in range(max_runs + 1)}
    total = sum(raw.values())
    return {k: round(v / total, 5) for k, v in raw.items()}

