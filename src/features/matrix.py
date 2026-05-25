"""Build train-ready feature matrices with time-series safe ordering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessor import load_deliveries_csv
from src.features.match_features import add_basic_delivery_features
from src.features.pressure_index import PressureInputs, pressure_index


FEATURE_COLUMNS = [
    "innings",
    "over",
    "ball",
    "legal_ball_index",
    "team_score",
    "wickets_lost",
    "balls_left",
    "current_run_rate",
    "required_run_rate",
    "runs_needed",
    "pressure_index",
    "recent_runs_6",
    "recent_wickets_12",
    "is_powerplay",
    "is_middle_overs",
    "is_death_overs",
    "batting_team_code",
    "bowling_team_code",
    "venue_code",
    "match_progress",
    "wicket_pressure",
    "run_rate_delta",
    "target_pressure",
    "batting_team_recent_rpb_180",
    "bowling_team_recent_rpb_allowed_180",
    "batting_team_recent_wicket_rate_180",
    "bowling_team_recent_wicket_rate_180",
    "batter_recent_rpb_60",
    "batter_recent_boundary_rate_60",
    "batter_recent_dismissal_rate_60",
    "bowler_recent_rpb_allowed_60",
    "bowler_recent_boundary_rate_allowed_60",
    "bowler_recent_wicket_rate_60",
    "venue_recent_rpb_240",
    "venue_recent_boundary_rate_240",
    "venue_recent_wicket_rate_240",
]


@dataclass(frozen=True)
class FeatureBuildSummary:
    """Metadata returned after writing train/test feature files."""

    rows: int
    train_rows: int
    test_rows: int
    features: int
    output_dir: str


def _code_series(series: pd.Series) -> pd.Series:
    codes, _ = pd.factorize(series.fillna("unknown").astype(str), sort=True)
    return pd.Series(codes, index=series.index, dtype="int64")


def _code_map(series: pd.Series) -> dict[str, int]:
    values = sorted(series.fillna("unknown").astype(str).unique())
    return {value: code for code, value in enumerate(values)}


def _rolling_mean_by(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int,
    default: float,
) -> pd.Series:
    """Previous-ball rolling mean for form features, excluding the current delivery."""

    values = (
        frame.groupby(group_cols, group_keys=False, sort=False)[value_col]
        .apply(lambda series: series.shift(1).rolling(window=window, min_periods=1).mean())
        .reindex(frame.index)
    )
    return values.fillna(default).astype(float)


def _target_by_match(deliveries: pd.DataFrame) -> pd.Series:
    """Infer second-innings targets from first-innings final scores."""

    innings_scores = deliveries.groupby(["match_id", "innings"], sort=False)["runs"].sum()
    target_map: dict[Any, int] = {}
    for match_id in deliveries["match_id"].dropna().unique():
        first_score = innings_scores.get((match_id, 1), np.nan)
        target_map[match_id] = int(first_score + 1) if pd.notna(first_score) else 0
    return deliveries["match_id"].map(target_map).fillna(0).astype(int)


def _winner_label(deliveries: pd.DataFrame) -> pd.Series:
    """Create a supervised win label for the batting team at each row."""

    if "winner" in deliveries.columns:
        return (deliveries["batting_team"].astype(str) == deliveries["winner"].astype(str)).astype(int)

    match_innings_scores = deliveries.groupby(["match_id", "innings"], sort=False)["runs"].sum()
    labels: list[int] = []
    for _, row in deliveries.iterrows():
        match_id = row["match_id"]
        innings = int(row.get("innings", 1))
        if innings == 1:
            first = match_innings_scores.get((match_id, 1), 0)
            second = match_innings_scores.get((match_id, 2), -1)
            labels.append(int(first > second))
        else:
            first = match_innings_scores.get((match_id, 1), 0)
            second = match_innings_scores.get((match_id, 2), 0)
            labels.append(int(second >= first + 1))
    return pd.Series(labels, index=deliveries.index, dtype="int64")


def build_feature_frame(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and win labels from cleaned deliveries."""

    required = {"match_id", "innings", "over", "ball", "batting_team", "bowling_team", "runs"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries missing columns: {sorted(missing)}")

    df = deliveries.copy()
    df["innings"] = df["innings"].astype(int)
    df["over"] = df["over"].astype(int)
    df["ball"] = df["ball"].astype(int)
    df = df.sort_values(["match_id", "innings", "over", "ball"], kind="stable").reset_index(drop=True)
    df = add_basic_delivery_features(df)
    df["target"] = _target_by_match(df)
    df["runs_needed"] = np.where(df["innings"] == 2, np.maximum(df["target"] - df["team_score"], 0), 0)
    df["required_run_rate"] = np.where(
        (df["innings"] == 2) & (df["balls_left"] > 0),
        df["runs_needed"] * 6 / df["balls_left"],
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
        for runs_needed, balls_left, wickets in zip(df["runs_needed"], df["balls_left"], df["wickets_lost"], strict=False)
    ]
    group_key = ["match_id", "innings"]
    df["recent_runs_6"] = (
        df.groupby(group_key, sort=False)["runs"]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=group_key, drop=True)
    )
    df["recent_wickets_12"] = (
        df.groupby(group_key, sort=False)["is_wicket"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=group_key, drop=True)
    )
    df["is_powerplay"] = (df["over"] < 6).astype(int)
    df["is_middle_overs"] = ((df["over"] >= 6) & (df["over"] < 15)).astype(int)
    df["is_death_overs"] = (df["over"] >= 15).astype(int)
    df["batting_team_code"] = _code_series(df["batting_team"])
    df["bowling_team_code"] = _code_series(df["bowling_team"])
    venue_source = df["venue"] if "venue" in df.columns else pd.Series("unknown", index=df.index)
    df["venue_code"] = _code_series(venue_source)
    df["boundary"] = df["runs"].isin([4, 6]).astype(int)
    global_rpb = float(df["runs"].mean()) if len(df) else 0.0
    global_boundary_rate = float(df["boundary"].mean()) if len(df) else 0.0
    global_wicket_rate = float(df["is_wicket"].mean()) if len(df) else 0.0

    df["match_progress"] = (df["legal_ball_index"].clip(lower=0, upper=120) / 120).astype(float)
    df["wicket_pressure"] = (df["wickets_lost"].clip(lower=0, upper=10) / 10).astype(float)
    df["run_rate_delta"] = np.where(
        df["innings"] == 2,
        df["required_run_rate"] - df["current_run_rate"],
        0.0,
    )
    df["target_pressure"] = np.where(
        (df["innings"] == 2) & (df["target"] > 0),
        df["runs_needed"] / df["target"],
        0.0,
    )
    df["batting_team_recent_rpb_180"] = _rolling_mean_by(df, ["batting_team"], "runs", 180, global_rpb)
    df["bowling_team_recent_rpb_allowed_180"] = _rolling_mean_by(df, ["bowling_team"], "runs", 180, global_rpb)
    df["batting_team_recent_wicket_rate_180"] = _rolling_mean_by(
        df, ["batting_team"], "is_wicket", 180, global_wicket_rate
    )
    df["bowling_team_recent_wicket_rate_180"] = _rolling_mean_by(
        df, ["bowling_team"], "is_wicket", 180, global_wicket_rate
    )
    if "batter" in df.columns:
        df["batter_recent_rpb_60"] = _rolling_mean_by(df, ["batter"], "runs", 60, global_rpb)
        df["batter_recent_boundary_rate_60"] = _rolling_mean_by(
            df, ["batter"], "boundary", 60, global_boundary_rate
        )
        df["batter_recent_dismissal_rate_60"] = _rolling_mean_by(
            df, ["batter"], "is_wicket", 60, global_wicket_rate
        )
    else:
        df["batter_recent_rpb_60"] = global_rpb
        df["batter_recent_boundary_rate_60"] = global_boundary_rate
        df["batter_recent_dismissal_rate_60"] = global_wicket_rate
    if "bowler" in df.columns:
        df["bowler_recent_rpb_allowed_60"] = _rolling_mean_by(df, ["bowler"], "runs", 60, global_rpb)
        df["bowler_recent_boundary_rate_allowed_60"] = _rolling_mean_by(
            df, ["bowler"], "boundary", 60, global_boundary_rate
        )
        df["bowler_recent_wicket_rate_60"] = _rolling_mean_by(
            df, ["bowler"], "is_wicket", 60, global_wicket_rate
        )
    else:
        df["bowler_recent_rpb_allowed_60"] = global_rpb
        df["bowler_recent_boundary_rate_allowed_60"] = global_boundary_rate
        df["bowler_recent_wicket_rate_60"] = global_wicket_rate
    if "venue" in df.columns:
        df["venue_recent_rpb_240"] = _rolling_mean_by(df, ["venue"], "runs", 240, global_rpb)
        df["venue_recent_boundary_rate_240"] = _rolling_mean_by(
            df, ["venue"], "boundary", 240, global_boundary_rate
        )
        df["venue_recent_wicket_rate_240"] = _rolling_mean_by(
            df, ["venue"], "is_wicket", 240, global_wicket_rate
        )
    else:
        df["venue_recent_rpb_240"] = global_rpb
        df["venue_recent_boundary_rate_240"] = global_boundary_rate
        df["venue_recent_wicket_rate_240"] = global_wicket_rate

    features = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0)
    labels = _winner_label(df)
    return features, labels


def write_feature_matrices(
    deliveries_path: str | Path,
    output_dir: str | Path = "data/features",
    test_fraction: float = 0.2,
) -> FeatureBuildSummary:
    """Write X/y train-test CSVs using chronological row order."""

    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    deliveries = load_deliveries_csv(deliveries_path)
    features, labels = build_feature_frame(deliveries)
    if len(features) < 2:
        raise ValueError("at least two delivery rows are required")

    split_index = max(1, int(len(features) * (1.0 - test_fraction)))
    if split_index >= len(features):
        split_index = len(features) - 1
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    features.iloc[:split_index].to_csv(output / "X_train.csv", index=False)
    features.iloc[split_index:].to_csv(output / "X_test.csv", index=False)
    labels.iloc[:split_index].rename("win_label").to_csv(output / "y_train.csv", index=False)
    labels.iloc[split_index:].rename("win_label").to_csv(output / "y_test.csv", index=False)
    metadata_payload = {
        "feature_columns": FEATURE_COLUMNS,
        "category_maps": {
            "batting_team_code": _code_map(deliveries["batting_team"]),
            "bowling_team_code": _code_map(deliveries["bowling_team"]),
            "venue_code": _code_map(deliveries["venue"] if "venue" in deliveries.columns else pd.Series("unknown")),
        },
    }
    (output / "feature_metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    metadata = pd.DataFrame(
        [
            {
                "rows": len(features),
                "train_rows": split_index,
                "test_rows": len(features) - split_index,
                "features": len(FEATURE_COLUMNS),
                "test_fraction": test_fraction,
            }
        ]
    )
    metadata.to_csv(output / "metadata.csv", index=False)
    return FeatureBuildSummary(
        rows=len(features),
        train_rows=split_index,
        test_rows=len(features) - split_index,
        features=len(FEATURE_COLUMNS),
        output_dir=str(output),
    )
