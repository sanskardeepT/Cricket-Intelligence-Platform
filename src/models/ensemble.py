"""Meta-learner and calibrated fallback prediction logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Prediction:
    """A complete prediction payload shared by API and frontend."""

    probability: float
    confidence: float
    label: str
    model_votes: dict[str, float]


def bayesian_update(prior: float, evidence_given_win: float, evidence_probability: float) -> float:
    """Update a win prior with Bayes theorem."""

    if not 0 <= prior <= 1:
        raise ValueError("prior must be between 0 and 1")
    if not 0 < evidence_probability <= 1:
        raise ValueError("evidence_probability must be in (0, 1]")
    if not 0 <= evidence_given_win <= 1:
        raise ValueError("evidence_given_win must be between 0 and 1")
    return max(0.0, min(1.0, evidence_given_win * prior / evidence_probability))


class StackingEnsemble:
    """Logistic Regression stacker over specialized model probabilities."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)
        self.columns: list[str] = []

    def fit(self, model_probabilities: pd.DataFrame, target: pd.Series) -> "StackingEnsemble":
        """Fit the meta-learner."""

        if model_probabilities.empty:
            raise ValueError("model_probabilities cannot be empty")
        self.columns = list(model_probabilities.columns)
        self.model.fit(model_probabilities, target)
        return self

    def predict(self, model_probabilities: pd.DataFrame) -> np.ndarray:
        """Predict stacked win probability."""

        if not self.columns:
            raise RuntimeError("ensemble must be fitted before prediction")
        return self.model.predict_proba(model_probabilities[self.columns])[:, 1]


def heuristic_live_prediction(
    elo_prior: float,
    monte_carlo_probability: float,
    pressure: float,
    required_run_rate: float,
    current_run_rate: float,
) -> Prediction:
    """Production fallback when trained artifacts are not present yet."""

    pressure_penalty = (pressure - 50.0) / 220.0
    rate_edge = (current_run_rate - required_run_rate) / 28.0 if required_run_rate else 0.04
    probability = 0.42 * elo_prior + 0.45 * monte_carlo_probability + 0.13 * (0.5 + rate_edge)
    probability -= pressure_penalty
    probability = round(max(0.02, min(0.98, probability)), 4)
    confidence = round(abs(probability - 0.5) * 2.0, 4)
    return Prediction(
        probability=probability,
        confidence=confidence,
        label="win" if probability >= 0.5 else "loss",
        model_votes={
            "elo_prior": round(elo_prior, 4),
            "monte_carlo": round(monte_carlo_probability, 4),
            "run_rate_edge": round(0.5 + rate_edge, 4),
        },
    )


def average_probabilities(probabilities: Iterable[float]) -> float:
    """Average valid model probabilities."""

    values = [float(p) for p in probabilities if np.isfinite(float(p))]
    if not values:
        raise ValueError("at least one probability is required")
    return round(float(np.mean(values)), 4)

