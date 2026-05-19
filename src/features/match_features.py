"""Match-level feature engineering for IPL, T20I, and ODI predictions."""

from __future__ import annotations

from dataclasses import dataclass
from math import pow
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchState:
    """A compact live match state used by API, models, and explainers."""

    batting_team: str
    bowling_team: str
    venue: str
    score: int
    wickets: int
    overs: float
    target: int | None = None
    innings_balls: int = 120
    batting_elo: float = 1500.0
    bowling_elo: float = 1500.0


def balls_bowled_from_overs(overs: float) -> int:
    """Convert cricket over notation, e.g. 12.3, into legal balls bowled."""

    whole_overs = int(overs)
    balls = round((overs - whole_overs) * 10)
    if balls < 0 or balls > 5:
        raise ValueError("overs must use cricket notation, e.g. 12.5 not 12.8")
    return whole_overs * 6 + balls


def current_run_rate(score: int, balls_bowled: int) -> float:
    """Calculate current run rate."""

    if balls_bowled < 0:
        raise ValueError("balls_bowled cannot be negative")
    if balls_bowled == 0:
        return 0.0
    return round(score * 6.0 / balls_bowled, 2)


def required_run_rate(target: int | None, score: int, balls_left: int) -> float:
    """Calculate required run rate for a chase; return 0 when not chasing."""

    if target is None:
        return 0.0
    if balls_left <= 0:
        return float("inf") if score < target else 0.0
    return round(max(target - score, 0) * 6.0 / balls_left, 2)


def elo_win_probability(team_a_elo: float, team_b_elo: float) -> float:
    """Return ELO win probability using the exact blueprint formula."""

    return 1.0 / (1.0 + pow(10.0, (team_b_elo - team_a_elo) / 400.0))


def build_match_features(state: MatchState) -> dict[str, float | str]:
    """Build deterministic match features from a live or pre-match state."""

    balls_bowled = balls_bowled_from_overs(state.overs)
    balls_left = max(state.innings_balls - balls_bowled, 0)
    crr = current_run_rate(state.score, balls_bowled)
    rrr = required_run_rate(state.target, state.score, balls_left)
    runs_needed = max((state.target or state.score) - state.score, 0)

    return {
        "batting_team": state.batting_team,
        "bowling_team": state.bowling_team,
        "venue": state.venue,
        "score": float(state.score),
        "wickets": float(state.wickets),
        "balls_bowled": float(balls_bowled),
        "balls_left": float(balls_left),
        "current_run_rate": crr,
        "required_run_rate": rrr,
        "runs_needed": float(runs_needed),
        "elo_prior": round(elo_win_probability(state.batting_elo, state.bowling_elo), 4),
    }


def add_basic_delivery_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Add reusable ball-by-ball features to a Cricsheet/Kaggle deliveries frame."""

    required = {"match_id", "innings", "over", "ball", "batting_team", "bowling_team", "runs"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries is missing columns: {sorted(missing)}")

    df = deliveries.copy()
    df["legal_ball_index"] = (df["over"].astype(int) * 6 + df["ball"].astype(int)).clip(lower=0)
    df["team_score"] = df.groupby(["match_id", "innings"])["runs"].cumsum()
    wicket_col = "is_wicket" if "is_wicket" in df.columns else None
    df["wickets_lost"] = df.groupby(["match_id", "innings"])[wicket_col].cumsum() if wicket_col else 0
    df["balls_left"] = np.maximum(120 - df["legal_ball_index"], 0)
    df["current_run_rate"] = np.where(
        df["legal_ball_index"] > 0,
        df["team_score"] * 6 / df["legal_ball_index"],
        0.0,
    )
    return df


def team_form_index(match_results: pd.DataFrame, team: str, window: int = 5) -> float:
    """Return recent win percentage for a team using only past rows supplied."""

    required = {"team", "won"}
    missing = required - set(match_results.columns)
    if missing:
        raise ValueError(f"match_results is missing columns: {sorted(missing)}")
    recent = match_results.loc[match_results["team"] == team].tail(window)
    if recent.empty:
        return 50.0
    return round(float(recent["won"].mean() * 100.0), 2)


def normalize_feature_mapping(values: Mapping[str, float]) -> dict[str, float]:
    """Return finite float features safe for model inference."""

    clean: dict[str, float] = {}
    for key, value in values.items():
        number = float(value)
        clean[key] = 0.0 if not np.isfinite(number) else number
    return clean

