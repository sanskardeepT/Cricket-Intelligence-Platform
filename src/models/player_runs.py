"""Train and serve batter runs prediction models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.schemas import PlayerRunsRequest
from src.data.preprocessor import load_deliveries_csv


DEFAULT_PLAYER_RUNS_ARTIFACT_PATH = Path("artifacts/models/player_runs.joblib")
PLAYER_RUN_FEATURE_COLUMNS = [
    "innings",
    "batter_code",
    "batting_team_code",
    "bowling_team_code",
    "venue_code",
    "career_avg_prior",
    "recent_avg_5",
    "recent_avg_10",
    "career_balls_avg_prior",
    "recent_strike_rate_5",
    "venue_avg_prior",
    "opposition_avg_prior",
    "innings_used_prior",
]


@dataclass(frozen=True)
class PlayerRunsTrainingSummary:
    """Result of training the player runs model."""

    best_model: str
    artifact_path: str
    metrics_path: str
    train_rows: int
    test_rows: int
    features: int
    test_mae: float
    test_rmse: float


def _code_map(series: pd.Series) -> dict[str, int]:
    values = sorted(series.fillna("unknown").astype(str).unique())
    return {value: code for code, value in enumerate(values)}


def _rolling_prior_mean(frame: pd.DataFrame, group_cols: list[str], value_col: str, window: int | None = None) -> pd.Series:
    def calc(series: pd.Series) -> pd.Series:
        shifted = series.shift(1)
        if window is None:
            return shifted.expanding(min_periods=1).mean()
        return shifted.rolling(window=window, min_periods=1).mean()

    return (
        frame.groupby(group_cols, group_keys=False, sort=False)[value_col]
        .apply(calc)
        .reindex(frame.index)
        .astype(float)
    )


def build_player_runs_frame(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Build batter-innings feature rows and target runs."""

    required = {"match_id", "innings", "batting_team", "bowling_team", "batter", "runs", "batter_runs", "ball"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries missing columns: {sorted(missing)}")

    df = deliveries.copy()
    if "start_date" in df.columns:
        df["sort_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    else:
        df["sort_date"] = pd.NaT
    group_cols = ["match_id", "innings", "batting_team", "bowling_team", "batter"]
    innings = (
        df.groupby(group_cols, sort=False)
        .agg(
            venue=("venue", "first") if "venue" in df.columns else ("match_id", "first"),
            start_date=("sort_date", "first"),
            runs=("batter_runs", "sum"),
            balls=("ball", "count"),
        )
        .reset_index()
    )
    innings["venue"] = innings["venue"].fillna("unknown").astype(str)
    innings["start_date"] = pd.to_datetime(innings["start_date"], errors="coerce")
    innings = innings.sort_values(["start_date", "match_id", "innings"], kind="stable").reset_index(drop=True)
    innings["strike_rate"] = np.where(innings["balls"] > 0, innings["runs"] * 100 / innings["balls"], 0.0)
    global_avg = float(innings["runs"].mean()) if len(innings) else 0.0
    global_balls = float(innings["balls"].mean()) if len(innings) else 0.0
    global_sr = float(innings["strike_rate"].mean()) if len(innings) else 0.0

    innings["career_avg_prior"] = _rolling_prior_mean(innings, ["batter"], "runs").fillna(global_avg)
    innings["recent_avg_5"] = _rolling_prior_mean(innings, ["batter"], "runs", 5).fillna(global_avg)
    innings["recent_avg_10"] = _rolling_prior_mean(innings, ["batter"], "runs", 10).fillna(global_avg)
    innings["career_balls_avg_prior"] = _rolling_prior_mean(innings, ["batter"], "balls").fillna(global_balls)
    innings["recent_strike_rate_5"] = _rolling_prior_mean(innings, ["batter"], "strike_rate", 5).fillna(global_sr)
    innings["venue_avg_prior"] = _rolling_prior_mean(innings, ["batter", "venue"], "runs").fillna(
        innings["career_avg_prior"]
    )
    innings["opposition_avg_prior"] = _rolling_prior_mean(innings, ["batter", "bowling_team"], "runs").fillna(
        innings["career_avg_prior"]
    )
    innings["innings_used_prior"] = innings.groupby("batter", sort=False).cumcount().astype(float)

    category_maps = {
        "batter_code": _code_map(innings["batter"]),
        "batting_team_code": _code_map(innings["batting_team"]),
        "bowling_team_code": _code_map(innings["bowling_team"]),
        "venue_code": _code_map(innings["venue"]),
    }
    for source_col, feature_col in [
        ("batter", "batter_code"),
        ("batting_team", "batting_team_code"),
        ("bowling_team", "bowling_team_code"),
        ("venue", "venue_code"),
    ]:
        innings[feature_col] = innings[source_col].astype(str).map(category_maps[feature_col]).fillna(-1).astype(int)

    features = innings[PLAYER_RUN_FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0)
    labels = innings["runs"].astype(float)
    return features, labels, {"category_maps": category_maps}


def _candidate_models() -> dict[str, Any]:
    return {
        "ridge": Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_player_runs_model(
    deliveries_csv: str | Path,
    artifact_dir: str | Path = "artifacts/models",
    test_fraction: float = 0.2,
    folds: int = 4,
) -> PlayerRunsTrainingSummary:
    """Train expected batter-runs regression from cleaned deliveries."""

    deliveries = load_deliveries_csv(deliveries_csv)
    features, labels, metadata = build_player_runs_frame(deliveries)
    split_index = max(1, int(len(features) * (1.0 - test_fraction)))
    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = labels.iloc[:split_index], labels.iloc[split_index:]
    splitter = TimeSeriesSplit(n_splits=min(folds, max(2, len(x_train) - 2)))
    scores: list[dict[str, float | str]] = []
    fitted: dict[str, Any] = {}

    for name, model in _candidate_models().items():
        fold_errors: list[float] = []
        for train_idx, validation_idx in splitter.split(x_train):
            fold_model = _candidate_models()[name]
            fold_model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            fold_errors.append(float(mean_absolute_error(y_train.iloc[validation_idx], fold_model.predict(x_train.iloc[validation_idx]))))
        model.fit(x_train, y_train)
        predictions = np.clip(model.predict(x_test), 0, None)
        rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
        scores.append(
            {
                "name": name,
                "cv_mae": round(float(np.mean(fold_errors)), 3),
                "test_mae": round(float(mean_absolute_error(y_test, predictions)), 3),
                "test_rmse": round(rmse, 3),
            }
        )
        fitted[name] = model

    best = min(scores, key=lambda item: (float(item["test_mae"]), float(item["test_rmse"])))
    artifact_output = Path(artifact_dir)
    artifact_output.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_output / "player_runs.joblib"
    metrics_path = artifact_output / "player_runs_metrics.json"
    joblib.dump(
        {
            "model": fitted[str(best["name"])],
            "model_name": best["name"],
            "feature_columns": list(x_train.columns),
            "feature_defaults": x_train.mean(numeric_only=True).to_dict(),
            "category_maps": metadata["category_maps"],
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
                "metrics": scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PlayerRunsTrainingSummary(
        best_model=str(best["name"]),
        artifact_path=str(artifact_path),
        metrics_path=str(metrics_path),
        train_rows=len(x_train),
        test_rows=len(x_test),
        features=len(x_train.columns),
        test_mae=float(best["test_mae"]),
        test_rmse=float(best["test_rmse"]),
    )


def player_runs_artifact_path() -> Path:
    return Path(os.getenv("PLAYER_RUNS_ARTIFACT_PATH", str(DEFAULT_PLAYER_RUNS_ARTIFACT_PATH)))


@lru_cache(maxsize=2)
def load_player_runs_artifact(path: str | None = None) -> dict[str, Any] | None:
    candidate = Path(path) if path else player_runs_artifact_path()
    if not candidate.exists():
        return None
    payload = joblib.load(candidate)
    if not {"model", "feature_columns"}.issubset(payload):
        raise ValueError(f"invalid player runs artifact: {candidate}")
    return payload


def _category_code(payload: dict[str, Any], feature_name: str, raw_value: str, default: float) -> float:
    mapping = payload.get("category_maps", {}).get(feature_name, {})
    if isinstance(mapping, dict) and raw_value in mapping:
        return float(mapping[raw_value])
    return default


def player_runs_feature_row(request: PlayerRunsRequest, payload: dict[str, Any]) -> pd.DataFrame:
    defaults = payload.get("feature_defaults", {})
    values = {
        "innings": float(request.innings),
        "batter_code": _category_code(payload, "batter_code", request.batter, float(defaults.get("batter_code", 0.0))),
        "batting_team_code": _category_code(
            payload, "batting_team_code", request.batting_team, float(defaults.get("batting_team_code", 0.0))
        ),
        "bowling_team_code": _category_code(
            payload, "bowling_team_code", request.bowling_team, float(defaults.get("bowling_team_code", 0.0))
        ),
        "venue_code": _category_code(payload, "venue_code", request.venue, float(defaults.get("venue_code", 0.0))),
    }
    for column in payload["feature_columns"]:
        values.setdefault(column, float(defaults.get(column, 0.0)))
    return pd.DataFrame([{column: values[column] for column in payload["feature_columns"]}])


def predict_player_runs(request: PlayerRunsRequest, path: str | None = None) -> dict[str, object]:
    """Predict expected batter runs using the trained artifact."""

    payload = load_player_runs_artifact(path)
    if payload is None:
        return {
            "batter": request.batter,
            "expected_runs": 24.0,
            "range": {"p10": 6.0, "p90": 52.0},
            "model_name": "fallback_player_prior",
            "source": "fallback",
        }
    row = player_runs_feature_row(request, payload)
    expected = max(0.0, float(payload["model"].predict(row)[0]))
    return {
        "batter": request.batter,
        "expected_runs": round(expected, 2),
        "range": {"p10": round(max(0.0, expected - 18.0), 2), "p90": round(expected + 28.0, 2)},
        "model_name": str(payload.get("model_name", "player_runs_artifact")),
        "source": "trained_artifact",
    }
