"""Train and serve a real next-ball outcome model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.schemas import LivePredictionRequest
from src.data.preprocessor import load_deliveries_csv
from src.features.match_features import balls_bowled_from_overs, current_run_rate, required_run_rate
from src.features.pressure_index import PressureInputs, pressure_index


DEFAULT_BALL_ARTIFACT_PATH = Path("artifacts/models/next_ball_outcome.joblib")
OUTCOME_ORDER = ["Dot ball", "Single", "Two runs", "Four", "Six", "Wicket", "Other runs"]


@dataclass(frozen=True)
class BallTrainingSummary:
    """Result of training the next-ball outcome model."""

    best_model: str
    artifact_path: str
    metrics_path: str
    train_rows: int
    test_rows: int
    features: int
    test_accuracy: float
    test_log_loss: float


def _code_map(series: pd.Series) -> dict[str, int]:
    values = sorted(series.fillna("unknown").astype(str).unique())
    return {value: code for code, value in enumerate(values)}


def _map_codes(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    return series.fillna("unknown").astype(str).map(mapping).fillna(-1).astype(int)


def _target_by_match(deliveries: pd.DataFrame) -> pd.Series:
    innings_scores = deliveries.groupby(["match_id", "innings"], sort=False)["runs"].sum()
    targets: dict[Any, int] = {}
    for match_id in deliveries["match_id"].dropna().unique():
        first_score = innings_scores.get((match_id, 1), np.nan)
        targets[match_id] = int(first_score + 1) if pd.notna(first_score) else 0
    return deliveries["match_id"].map(targets).fillna(0).astype(int)


def _rolling_previous_sum(frame: pd.DataFrame, group_cols: list[str], value_col: str, window: int) -> pd.Series:
    return (
        frame.groupby(group_cols, group_keys=False, sort=False)[value_col]
        .apply(lambda series: series.shift(1).rolling(window=window, min_periods=1).sum())
        .reindex(frame.index)
        .fillna(0.0)
        .astype(float)
    )


def _label_outcome(row: pd.Series) -> str:
    if int(row.get("is_wicket", 0)) == 1:
        return "Wicket"
    runs = int(row.get("runs", 0))
    if runs == 0:
        return "Dot ball"
    if runs == 1:
        return "Single"
    if runs == 2:
        return "Two runs"
    if runs == 4:
        return "Four"
    if runs == 6:
        return "Six"
    return "Other runs"


def build_ball_outcome_frame(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Build pre-ball features and current-ball outcome labels."""

    required = {"match_id", "innings", "over", "ball", "batting_team", "bowling_team", "runs"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries missing columns: {sorted(missing)}")

    df = deliveries.copy()
    df["innings"] = df["innings"].astype(int)
    df["over"] = df["over"].astype(int)
    df["ball"] = df["ball"].astype(int)
    if "is_wicket" not in df.columns:
        df["is_wicket"] = 0
    df = df.sort_values(["match_id", "innings", "over", "ball"], kind="stable").reset_index(drop=True)
    group_key = ["match_id", "innings"]
    df["target"] = _target_by_match(df)
    df["pre_score"] = df.groupby(group_key, sort=False)["runs"].cumsum() - df["runs"]
    df["pre_wickets"] = df.groupby(group_key, sort=False)["is_wicket"].cumsum() - df["is_wicket"]
    df["legal_ball_index"] = (df["over"] * 6 + df["ball"].clip(lower=1, upper=6)).clip(lower=0)
    df["balls_bowled_before"] = (df["legal_ball_index"] - 1).clip(lower=0, upper=120)
    df["balls_left_before"] = (120 - df["balls_bowled_before"]).clip(lower=0)
    df["current_run_rate"] = np.where(
        df["balls_bowled_before"] > 0,
        df["pre_score"] * 6 / df["balls_bowled_before"],
        0.0,
    )
    df["runs_needed"] = np.where(df["innings"] == 2, np.maximum(df["target"] - df["pre_score"], 0), 0)
    df["required_run_rate"] = np.where(
        (df["innings"] == 2) & (df["balls_left_before"] > 0),
        df["runs_needed"] * 6 / df["balls_left_before"],
        0.0,
    )
    df["pressure_index"] = [
        pressure_index(
            PressureInputs(
                runs_needed=float(runs_needed),
                balls_left=max(int(balls_left), 1),
                wickets_lost=int(min(max(wickets, 0), 10)),
            )
        )
        for runs_needed, balls_left, wickets in zip(
            df["runs_needed"], df["balls_left_before"], df["pre_wickets"], strict=False
        )
    ]
    df["recent_runs_6"] = _rolling_previous_sum(df, group_key, "runs", 6)
    df["recent_wickets_12"] = _rolling_previous_sum(df, group_key, "is_wicket", 12)
    df["match_progress"] = (df["balls_bowled_before"] / 120).astype(float)
    df["wicket_pressure"] = (df["pre_wickets"].clip(lower=0, upper=10) / 10).astype(float)
    df["run_rate_delta"] = np.where(df["innings"] == 2, df["required_run_rate"] - df["current_run_rate"], 0.0)
    df["target_pressure"] = np.where(
        (df["innings"] == 2) & (df["target"] > 0),
        df["runs_needed"] / df["target"],
        0.0,
    )
    category_maps = {
        "batting_team_code": _code_map(df["batting_team"]),
        "bowling_team_code": _code_map(df["bowling_team"]),
        "venue_code": _code_map(df["venue"] if "venue" in df.columns else pd.Series("unknown", index=df.index)),
    }
    df["batting_team_code"] = _map_codes(df["batting_team"], category_maps["batting_team_code"])
    df["bowling_team_code"] = _map_codes(df["bowling_team"], category_maps["bowling_team_code"])
    venue_source = df["venue"] if "venue" in df.columns else pd.Series("unknown", index=df.index)
    df["venue_code"] = _map_codes(venue_source, category_maps["venue_code"])
    features = df[
        [
            "innings",
            "over",
            "ball",
            "balls_bowled_before",
            "pre_score",
            "pre_wickets",
            "balls_left_before",
            "current_run_rate",
            "required_run_rate",
            "runs_needed",
            "pressure_index",
            "recent_runs_6",
            "recent_wickets_12",
            "match_progress",
            "wicket_pressure",
            "run_rate_delta",
            "target_pressure",
            "batting_team_code",
            "bowling_team_code",
            "venue_code",
        ]
    ].replace([np.inf, -np.inf], 0).fillna(0)
    labels = df.apply(_label_outcome, axis=1)
    metadata = {"category_maps": category_maps, "outcome_order": OUTCOME_ORDER}
    return features, labels, metadata


def _candidate_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_ball_outcome_model(
    deliveries_path: str | Path,
    artifact_dir: str | Path = "artifacts/models",
    test_fraction: float = 0.2,
    folds: int = 3,
) -> BallTrainingSummary:
    """Train a multiclass next-ball outcome model from cleaned deliveries."""

    deliveries = load_deliveries_csv(deliveries_path)
    features, labels, metadata = build_ball_outcome_frame(deliveries)
    split_index = max(1, int(len(features) * (1.0 - test_fraction)))
    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = labels.iloc[:split_index], labels.iloc[split_index:]
    models = _candidate_models()
    splitter = TimeSeriesSplit(n_splits=min(folds, max(2, len(x_train) - 2)))
    scores: list[dict[str, float | str]] = []
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        fold_scores: list[float] = []
        for train_idx, validation_idx in splitter.split(x_train):
            fold_model = _candidate_models()[name]
            fold_model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            predictions = fold_model.predict(x_train.iloc[validation_idx])
            fold_scores.append(float(accuracy_score(y_train.iloc[validation_idx], predictions)))
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        predictions = model.predict(x_test)
        scores.append(
            {
                "name": name,
                "cv_accuracy": round(float(np.mean(fold_scores)), 4),
                "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "test_log_loss": round(float(log_loss(y_test, probabilities, labels=list(model.classes_))), 4),
            }
        )
        fitted[name] = model

    best = max(scores, key=lambda item: (float(item["test_accuracy"]), -float(item["test_log_loss"])))
    artifact_output = Path(artifact_dir)
    artifact_output.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_output / "next_ball_outcome.joblib"
    metrics_path = artifact_output / "next_ball_metrics.json"
    best_model = fitted[str(best["name"])]
    joblib.dump(
        {
            "model": best_model,
            "model_name": best["name"],
            "feature_columns": list(x_train.columns),
            "feature_defaults": x_train.mean(numeric_only=True).to_dict(),
            "category_maps": metadata["category_maps"],
            "outcome_order": OUTCOME_ORDER,
            "metrics": best,
        },
        artifact_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": best["name"],
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "features": len(x_train.columns),
                "outcomes": OUTCOME_ORDER,
                "metrics": scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BallTrainingSummary(
        best_model=str(best["name"]),
        artifact_path=str(artifact_path),
        metrics_path=str(metrics_path),
        train_rows=len(x_train),
        test_rows=len(x_test),
        features=len(x_train.columns),
        test_accuracy=float(best["test_accuracy"]),
        test_log_loss=float(best["test_log_loss"]),
    )


def ball_artifact_path() -> Path:
    """Return configured next-ball model artifact path."""

    return Path(os.getenv("BALL_MODEL_ARTIFACT_PATH", str(DEFAULT_BALL_ARTIFACT_PATH)))


@lru_cache(maxsize=2)
def load_ball_artifact(path: str | None = None) -> dict[str, Any] | None:
    candidate = Path(path) if path else ball_artifact_path()
    if not candidate.exists():
        return None
    payload = joblib.load(candidate)
    if not {"model", "feature_columns"}.issubset(payload):
        raise ValueError(f"invalid ball model artifact: {candidate}")
    return payload


def live_ball_feature_row(
    request: LivePredictionRequest,
    feature_columns: list[str],
    feature_defaults: dict[str, float] | None = None,
    category_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    balls_bowled = balls_bowled_from_overs(request.overs)
    balls_left = max(120 - balls_bowled, 0)
    runs_needed = max(request.target - request.score, 0)
    crr = current_run_rate(request.score, balls_bowled)
    rrr = required_run_rate(request.target, request.score, balls_left)
    pressure = pressure_index(
        PressureInputs(runs_needed=runs_needed, balls_left=max(balls_left, 1), wickets_lost=request.wickets)
    )
    over = int(request.overs)
    ball = round((request.overs - over) * 10) + 1
    values: dict[str, float] = {
        "innings": 2.0,
        "over": float(over),
        "ball": float(min(ball, 6)),
        "balls_bowled_before": float(balls_bowled),
        "pre_score": float(request.score),
        "pre_wickets": float(request.wickets),
        "balls_left_before": float(balls_left),
        "current_run_rate": float(crr),
        "required_run_rate": float(rrr),
        "runs_needed": float(runs_needed),
        "pressure_index": float(pressure),
        "recent_runs_6": float(crr),
        "recent_wickets_12": float(request.wickets),
        "match_progress": float(min(max(balls_bowled, 0), 120) / 120),
        "wicket_pressure": float(min(max(request.wickets, 0), 10) / 10),
        "run_rate_delta": float(rrr - crr),
        "target_pressure": float(runs_needed / request.target) if request.target > 0 else 0.0,
    }
    categories = category_maps or {}
    for feature_name, raw_value in {
        "batting_team_code": request.batting_team,
        "bowling_team_code": request.bowling_team,
        "venue_code": request.venue,
    }.items():
        mapping = categories.get(feature_name, {})
        if raw_value in mapping:
            values[feature_name] = float(mapping[raw_value])
    defaults = feature_defaults or {}
    return pd.DataFrame([{column: values.get(column, float(defaults.get(column, 0.0))) for column in feature_columns}])


def predict_next_ball_with_artifact(request: LivePredictionRequest, path: str | None = None) -> dict[str, object] | None:
    """Return next-ball probabilities from the trained artifact when available."""

    payload = load_ball_artifact(path)
    if payload is None:
        return None
    feature_columns = list(payload["feature_columns"])
    row = live_ball_feature_row(
        request,
        feature_columns,
        payload.get("feature_defaults", {}),
        payload.get("category_maps", {}),
    )
    model = payload["model"]
    classes = list(model.classes_)
    probabilities = model.predict_proba(row)[0]
    raw_distribution = {label: float(prob) for label, prob in zip(classes, probabilities, strict=False)}
    ordered_labels = [label for label in OUTCOME_ORDER if label in raw_distribution]
    distribution = {label: round(raw_distribution[label], 4) for label in ordered_labels}
    likely = max(distribution, key=distribution.get)
    return {
        "most_likely": likely,
        "distribution": distribution,
        "reasons": [f"Trained on {payload.get('model_name', 'next-ball')} IPL delivery outcomes."],
        "model_name": str(payload.get("model_name", "next_ball_artifact")),
        "source": "trained_artifact",
    }
