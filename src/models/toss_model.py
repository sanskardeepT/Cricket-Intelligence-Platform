"""Train and serve a toss-decision model from Cricsheet match info."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.schemas import TossRequest


DEFAULT_TOSS_ARTIFACT_PATH = Path("artifacts/models/toss_decision.joblib")
TOSS_FEATURE_COLUMNS = [
    "venue_dew_factor",
    "pitch_deterioration",
    "captain_field_tendency",
    "chase_success_rate",
    "venue_code",
    "toss_winner_code",
    "season_code",
]


@dataclass(frozen=True)
class TossTrainingSummary:
    """Result of training the toss-decision model."""

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


def _parse_info_csv(raw_text: str) -> dict[str, Any]:
    info: dict[str, Any] = {"teams": []}
    for row in csv.reader(raw_text.splitlines()):
        if len(row) < 3 or row[0] != "info":
            continue
        key = row[1]
        value = row[2]
        if key == "team":
            info["teams"].append(value)
        elif key in {"season", "date", "venue", "city", "toss_winner", "toss_decision", "winner"}:
            info[key] = value
    return info


def load_cricsheet_match_info(zip_path: str | Path) -> pd.DataFrame:
    """Load match-level metadata from Cricsheet `_info.csv` files."""

    rows: list[dict[str, Any]] = []
    with ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith("_info.csv"):
                continue
            match_id = Path(name).stem.replace("_info", "")
            info = _parse_info_csv(archive.read(name).decode("utf-8", errors="replace"))
            toss_decision = str(info.get("toss_decision", "")).lower()
            if toss_decision not in {"bat", "field"}:
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "season": str(info.get("season", "unknown")),
                    "date": str(info.get("date", "")),
                    "venue": str(info.get("venue", "unknown")),
                    "city": str(info.get("city", "unknown")),
                    "toss_winner": str(info.get("toss_winner", "unknown")),
                    "toss_decision": toss_decision,
                    "winner": str(info.get("winner", "unknown")),
                }
            )
    if not rows:
        raise ValueError(f"no toss info rows found in {zip_path}")
    frame = pd.DataFrame(rows)
    frame["sort_date"] = pd.to_datetime(frame["date"].str.replace("/", "-", regex=False), errors="coerce")
    return frame.sort_values(["sort_date", "match_id"], kind="stable").reset_index(drop=True)


def _first_innings_frame(deliveries_csv: str | Path) -> pd.DataFrame:
    deliveries = pd.read_csv(deliveries_csv, usecols=["match_id", "innings", "batting_team", "runs"])
    deliveries["match_id"] = deliveries["match_id"].astype(str)
    first_scores = deliveries.loc[deliveries["innings"] == 1].groupby("match_id", sort=False)["runs"].sum()
    innings_teams = (
        deliveries.groupby(["match_id", "innings"], sort=False)["batting_team"]
        .first()
        .unstack(fill_value="unknown")
        .rename(columns={1: "first_batting_team", 2: "second_batting_team"})
    )
    frame = innings_teams.reset_index()
    frame["first_innings_score"] = frame["match_id"].map(first_scores).fillna(0).astype(float)
    return frame


def build_toss_training_frame(zip_path: str | Path, deliveries_csv: str | Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Build toss model features using only historical priors before each match."""

    matches = load_cricsheet_match_info(zip_path)
    innings = _first_innings_frame(deliveries_csv)
    matches = matches.merge(innings, on="match_id", how="left")
    matches["chasing_team_won"] = (
        matches["winner"].astype(str) == matches["second_batting_team"].fillna("unknown").astype(str)
    ).astype(float)
    matches["venue_avg_first_score_prior"] = (
        matches.groupby("venue", group_keys=False, sort=False)["first_innings_score"]
        .apply(lambda series: series.shift(1).expanding(min_periods=1).mean())
        .reindex(matches.index)
        .fillna(matches["first_innings_score"].mean())
    )
    matches["venue_chase_success_prior"] = (
        matches.groupby("venue", group_keys=False, sort=False)["chasing_team_won"]
        .apply(lambda series: series.shift(1).expanding(min_periods=1).mean())
        .reindex(matches.index)
        .fillna(matches["chasing_team_won"].mean())
    )
    matches["captain_field_tendency_prior"] = (
        (matches["toss_decision"] == "field")
        .astype(float)
        .groupby(matches["toss_winner"], group_keys=False, sort=False)
        .apply(lambda series: series.shift(1).expanding(min_periods=1).mean())
        .reindex(matches.index)
        .fillna((matches["toss_decision"] == "field").mean())
    )
    venue_map = _code_map(matches["venue"])
    toss_winner_map = _code_map(matches["toss_winner"])
    season_map = _code_map(matches["season"])
    features = pd.DataFrame(
        {
            "venue_dew_factor": (matches["venue_chase_success_prior"] * 100).clip(0, 100),
            "pitch_deterioration": ((185 - matches["venue_avg_first_score_prior"]) * 1.2).clip(0, 100),
            "captain_field_tendency": (matches["captain_field_tendency_prior"] * 100).clip(0, 100),
            "chase_success_rate": (matches["venue_chase_success_prior"] * 100).clip(0, 100),
            "venue_code": matches["venue"].map(venue_map).fillna(-1).astype(int),
            "toss_winner_code": matches["toss_winner"].map(toss_winner_map).fillna(-1).astype(int),
            "season_code": matches["season"].map(season_map).fillna(-1).astype(int),
        }
    ).replace([np.inf, -np.inf], 0).fillna(0)
    metadata = {
        "category_maps": {
            "venue_code": venue_map,
            "toss_winner_code": toss_winner_map,
            "season_code": season_map,
        }
    }
    return features[TOSS_FEATURE_COLUMNS], matches["toss_decision"].astype(str), metadata


def _candidate_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, learning_rate=0.06, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_toss_model(
    zip_path: str | Path,
    deliveries_csv: str | Path,
    artifact_dir: str | Path = "artifacts/models",
    test_fraction: float = 0.2,
    folds: int = 3,
) -> TossTrainingSummary:
    """Train a toss bat/field decision model."""

    features, labels, metadata = build_toss_training_frame(zip_path, deliveries_csv)
    split_index = max(1, int(len(features) * (1.0 - test_fraction)))
    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = labels.iloc[:split_index], labels.iloc[split_index:]
    splitter = TimeSeriesSplit(n_splits=min(folds, max(2, len(x_train) - 2)))
    scores: list[dict[str, float | str]] = []
    fitted: dict[str, Any] = {}

    for name, model in _candidate_models().items():
        fold_scores: list[float] = []
        for train_idx, validation_idx in splitter.split(x_train):
            fold_model = _candidate_models()[name]
            fold_model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            fold_scores.append(float(accuracy_score(y_train.iloc[validation_idx], fold_model.predict(x_train.iloc[validation_idx]))))
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
    artifact_path = artifact_output / "toss_decision.joblib"
    metrics_path = artifact_output / "toss_metrics.json"
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
    return TossTrainingSummary(
        best_model=str(best["name"]),
        artifact_path=str(artifact_path),
        metrics_path=str(metrics_path),
        train_rows=len(x_train),
        test_rows=len(x_test),
        features=len(x_train.columns),
        test_accuracy=float(best["test_accuracy"]),
        test_log_loss=float(best["test_log_loss"]),
    )


def toss_artifact_path() -> Path:
    return Path(os.getenv("TOSS_MODEL_ARTIFACT_PATH", str(DEFAULT_TOSS_ARTIFACT_PATH)))


@lru_cache(maxsize=2)
def load_toss_artifact(path: str | None = None) -> dict[str, Any] | None:
    candidate = Path(path) if path else toss_artifact_path()
    if not candidate.exists():
        return None
    payload = joblib.load(candidate)
    if not {"model", "feature_columns"}.issubset(payload):
        raise ValueError(f"invalid toss model artifact: {candidate}")
    return payload


def _category_code(payload: dict[str, Any], feature_name: str, raw_value: str, default: float) -> float:
    mapping = payload.get("category_maps", {}).get(feature_name, {})
    if isinstance(mapping, dict) and raw_value in mapping:
        return float(mapping[raw_value])
    return default


def toss_feature_row(request: TossRequest, payload: dict[str, Any]) -> pd.DataFrame:
    defaults = payload.get("feature_defaults", {})
    values = {
        "venue_dew_factor": float(request.venue_dew_factor),
        "pitch_deterioration": float(request.pitch_deterioration),
        "captain_field_tendency": float(request.captain_field_tendency),
        "chase_success_rate": float(request.chase_success_rate),
        "venue_code": _category_code(payload, "venue_code", request.venue, float(defaults.get("venue_code", 0.0))),
        "toss_winner_code": _category_code(
            payload, "toss_winner_code", request.toss_winner, float(defaults.get("toss_winner_code", 0.0))
        ),
        "season_code": float(defaults.get("season_code", 0.0)),
    }
    columns = list(payload["feature_columns"])
    return pd.DataFrame([{column: values.get(column, float(defaults.get(column, 0.0))) for column in columns}])


def predict_toss_with_artifact(request: TossRequest, path: str | None = None) -> dict[str, object] | None:
    """Predict toss decision from trained artifact when available."""

    payload = load_toss_artifact(path)
    if payload is None:
        return None
    row = toss_feature_row(request, payload)
    model = payload["model"]
    probabilities = model.predict_proba(row)[0]
    distribution = {label.upper(): round(float(prob), 4) for label, prob in zip(model.classes_, probabilities, strict=False)}
    decision = max(distribution, key=distribution.get)
    field_score = distribution.get("FIELD", 0.0) * 100
    return {
        "decision": decision,
        "field_score": round(field_score, 2),
        "confidence": round(abs(distribution[decision] - 0.5) * 2, 3),
        "probabilities": distribution,
        "model_name": str(payload.get("model_name", "toss_artifact")),
        "source": "trained_artifact",
        "reasons": [
            {"factor": "Venue Dew Factor", "weight": 0.35, "value": request.venue_dew_factor},
            {"factor": "Pitch Deterioration", "weight": 0.25, "value": request.pitch_deterioration},
            {"factor": "Captain Field Tendency", "weight": 0.25, "value": request.captain_field_tendency},
            {"factor": "Chase Success Rate", "weight": 0.15, "value": request.chase_success_rate},
        ],
    }
