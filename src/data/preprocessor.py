"""Cricket data cleaning pipeline for Cricsheet and Kaggle-style CSVs."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


DELIVERY_COLUMN_ALIASES = {
    "striker": "batter",
    "batsman": "batter",
    "runs_off_bat": "batter_runs",
    "total_runs": "runs",
}


def _derive_over_and_ball(ball_column: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split Cricsheet ball notation like 15.2 into over=15 and delivery=2.

    Cricsheet can encode extra deliveries as .7, .8, etc. after wides/no-balls.
    We preserve that delivery sequence here; downstream feature code derives
    legal-ball indices separately.
    """

    ball_as_float = pd.to_numeric(ball_column, errors="coerce")
    if ball_as_float.isna().any():
        raise ValueError("ball column contains values that cannot be parsed as cricket ball notation")
    derived_over = ball_as_float.astype(int)
    fractional = (ball_as_float - derived_over).round(1)
    derived_ball = (fractional * 10).round().astype(int)
    if not (derived_ball >= 1).all():
        raise ValueError("ball notation must use positive delivery values like .1, .2, ...")
    return derived_over, derived_ball


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.rename(columns={k: v for k, v in DELIVERY_COLUMN_ALIASES.items() if k in frame.columns})

    if "over" not in df.columns:
        if "ball" in df.columns:
            df["over"], df["ball"] = _derive_over_and_ball(df["ball"])
        else:
            raise ValueError("missing required over/ball information")

    if "runs" not in df.columns:
        if {"batter_runs", "extras"}.issubset(df.columns):
            df["runs"] = df["batter_runs"].fillna(0) + df["extras"].fillna(0)
        elif "batter_runs" in df.columns:
            df["runs"] = df["batter_runs"].fillna(0)
        else:
            raise ValueError("could not infer runs column")

    if "is_wicket" not in df.columns:
        wicket_cols = [col for col in ["player_dismissed", "wicket_type"] if col in df.columns]
        df["is_wicket"] = df[wicket_cols].notna().any(axis=1).astype(int) if wicket_cols else 0

    return df


def load_deliveries_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize a single deliveries CSV."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    df = _normalize_columns(df)

    required = {"match_id", "over", "ball", "batting_team", "bowling_team", "runs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source.name} is missing columns: {sorted(missing)}")
    return df


def load_cricsheet_zip(zip_path: str | Path) -> pd.DataFrame:
    """Load all delivery rows from a Cricsheet CSV zip into one DataFrame."""

    archive = Path(zip_path)
    if not archive.exists():
        raise FileNotFoundError(archive)
    frames: list[pd.DataFrame] = []
    with ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv") or name.endswith("_info.csv"):
                continue
            with zf.open(name) as handle:
                raw = pd.read_csv(handle)
            raw["match_id"] = Path(name).stem
            frames.append(_normalize_columns(raw))
    if not frames:
        raise ValueError(f"no delivery CSV files found in {archive}")
    return pd.concat(frames, ignore_index=True)


def save_processed_deliveries(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Validate and save cleaned deliveries as CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _normalize_columns(frame)
    cleaned.to_csv(output, index=False)
    return output
