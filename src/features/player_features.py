"""Player-level cricket features: form, phase scores, and matchups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PlayerForm:
    """Compact player form output consumed by prediction explainers."""

    player: str
    form_score: float
    last_n_average: float
    innings_used: int


def player_form_score(scorecard: pd.DataFrame, player: str, last_n: int = 5) -> PlayerForm:
    """Calculate a Bayesian-smoothed player batting form score from recent innings."""

    required = {"player", "runs"}
    missing = required - set(scorecard.columns)
    if missing:
        raise ValueError(f"scorecard is missing columns: {sorted(missing)}")

    recent = scorecard.loc[scorecard["player"] == player].tail(last_n)
    if recent.empty:
        return PlayerForm(player=player, form_score=50.0, last_n_average=0.0, innings_used=0)

    last_avg = float(recent["runs"].mean())
    career_avg = float(scorecard.loc[scorecard["player"] == player, "runs"].mean())
    prior_weight = 3.0
    smoothed = (last_avg * len(recent) + career_avg * prior_weight) / (len(recent) + prior_weight)
    return PlayerForm(
        player=player,
        form_score=round(max(0.0, min(100.0, smoothed * 1.35)), 2),
        last_n_average=round(last_avg, 2),
        innings_used=int(len(recent)),
    )


def head_to_head_average(deliveries: pd.DataFrame, batter: str, bowler: str) -> dict[str, float]:
    """Return batter-vs-bowler runs per ball, boundary rate, and wicket rate."""

    required = {"batter", "bowler", "runs"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries is missing columns: {sorted(missing)}")
    subset = deliveries.loc[(deliveries["batter"] == batter) & (deliveries["bowler"] == bowler)]
    if subset.empty:
        return {"runs_per_ball": 0.0, "boundary_rate": 0.0, "wicket_rate": 0.0, "balls": 0.0}
    wicket_col = "is_wicket" if "is_wicket" in subset.columns else None
    return {
        "runs_per_ball": round(float(subset["runs"].mean()), 3),
        "boundary_rate": round(float(subset["runs"].isin([4, 6]).mean()), 3),
        "wicket_rate": round(float(subset[wicket_col].mean()), 3) if wicket_col else 0.0,
        "balls": float(len(subset)),
    }


def phase_average(deliveries: pd.DataFrame, batter: str, phase: str) -> float:
    """Calculate batter average runs per ball for powerplay, middle, or death overs."""

    phase_ranges = {
        "powerplay": (0, 6),
        "middle": (6, 15),
        "death": (15, 20),
    }
    if phase not in phase_ranges:
        raise ValueError("phase must be one of: powerplay, middle, death")
    required = {"batter", "over", "runs"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries is missing columns: {sorted(missing)}")
    start, end = phase_ranges[phase]
    subset = deliveries.loc[
        (deliveries["batter"] == batter)
        & (deliveries["over"].astype(float) >= start)
        & (deliveries["over"].astype(float) < end)
    ]
    if subset.empty:
        return 0.0
    return round(float(subset["runs"].mean()), 3)

