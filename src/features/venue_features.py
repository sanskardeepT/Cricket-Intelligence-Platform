"""Venue DNA features for toss, scoring, dew, and pitch behavior."""

from __future__ import annotations

import pandas as pd


def venue_dna(matches: pd.DataFrame, venue: str) -> dict[str, float | str]:
    """Build a venue profile from historical match rows."""

    required = {"venue", "first_innings_score", "chasing_team_won"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing columns: {sorted(missing)}")
    subset = matches.loc[matches["venue"] == venue]
    if subset.empty:
        return {
            "venue": venue,
            "avg_first_innings_score": 165.0,
            "chase_success_rate": 50.0,
            "dew_factor": 50.0,
            "pitch_type": "balanced",
        }
    dew = float(subset["humidity"].mean()) if "humidity" in subset.columns else 55.0
    avg_score = float(subset["first_innings_score"].mean())
    chase = float(subset["chasing_team_won"].mean() * 100.0)
    pitch_type = "batting" if avg_score >= 180 else "bowling" if avg_score < 145 else "balanced"
    return {
        "venue": venue,
        "avg_first_innings_score": round(avg_score, 2),
        "chase_success_rate": round(chase, 2),
        "dew_factor": round(max(0.0, min(100.0, dew)), 2),
        "pitch_type": pitch_type,
    }

