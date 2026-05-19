"""Runtime inference for trained win-probability artifacts."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api.schemas import LivePredictionRequest
from src.features.match_features import balls_bowled_from_overs, current_run_rate, required_run_rate
from src.features.pressure_index import PressureInputs, pressure_index


DEFAULT_ARTIFACT_PATH = Path("artifacts/models/win_probability_baseline.joblib")


@dataclass(frozen=True)
class ArtifactPrediction:
    """Prediction returned by a trained artifact."""

    probability: float
    confidence: float
    label: str
    model_name: str


def _stable_code(value: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha1(value.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def artifact_path() -> Path:
    """Return configured model artifact path."""

    return Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_ARTIFACT_PATH)))


@lru_cache(maxsize=2)
def load_artifact(path: str | None = None) -> dict[str, Any] | None:
    """Load a trained artifact from disk, returning None when absent."""

    candidate = Path(path) if path else artifact_path()
    if not candidate.exists():
        return None
    payload = joblib.load(candidate)
    if not {"model", "feature_columns"}.issubset(payload):
        raise ValueError(f"invalid model artifact: {candidate}")
    return payload


def live_feature_row(request: LivePredictionRequest, feature_columns: list[str]) -> pd.DataFrame:
    """Adapt a live API request into the trained feature-column layout."""

    balls_bowled = balls_bowled_from_overs(request.overs)
    balls_left = max(120 - balls_bowled, 0)
    runs_needed = max(request.target - request.score, 0)
    crr = current_run_rate(request.score, balls_bowled)
    rrr = required_run_rate(request.target, request.score, balls_left)
    pressure = pressure_index(
        PressureInputs(
            runs_needed=runs_needed,
            balls_left=max(balls_left, 1),
            wickets_lost=request.wickets,
        )
    )
    over = int(request.overs)
    ball = round((request.overs - over) * 10)
    values: dict[str, float] = {
        "innings": 2.0,
        "over": float(over),
        "ball": float(ball),
        "legal_ball_index": float(balls_bowled),
        "team_score": float(request.score),
        "wickets_lost": float(request.wickets),
        "balls_left": float(balls_left),
        "current_run_rate": float(crr),
        "required_run_rate": float(rrr),
        "runs_needed": float(runs_needed),
        "pressure_index": float(pressure),
        "recent_runs_6": float(max(0.0, crr)),
        "recent_wickets_12": float(request.wickets),
        "is_powerplay": float(over < 6),
        "is_middle_overs": float(6 <= over < 15),
        "is_death_overs": float(over >= 15),
        "batting_team_code": float(_stable_code(request.batting_team)),
        "bowling_team_code": float(_stable_code(request.bowling_team)),
        "venue_code": float(_stable_code(request.venue)),
    }
    return pd.DataFrame([{column: values.get(column, 0.0) for column in feature_columns}])


def predict_with_artifact(request: LivePredictionRequest, path: str | None = None) -> ArtifactPrediction | None:
    """Predict live win probability with a trained artifact when available."""

    payload = load_artifact(path)
    if payload is None:
        return None
    feature_columns = list(payload["feature_columns"])
    row = live_feature_row(request, feature_columns)
    probability = float(payload["model"].predict_proba(row)[:, 1][0])
    probability = round(max(0.02, min(0.98, probability)), 4)
    return ArtifactPrediction(
        probability=probability,
        confidence=round(abs(probability - 0.5) * 2.0, 4),
        label="win" if probability >= 0.5 else "loss",
        model_name=str(payload.get("model_name", "trained_artifact")),
    )

