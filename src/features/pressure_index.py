"""Pressure index calculations for live cricket states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PressureInputs:
    """Inputs required to quantify chase pressure on a 0-100 scale."""

    runs_needed: float
    balls_left: int
    wickets_lost: int
    innings_balls: int = 120


def pressure_index(inputs: PressureInputs) -> float:
    """Return the blueprint pressure score clipped to 0-100.

    Formula from the project blueprint:
    ((runs_needed / balls_left) * 0.45 + (wickets_lost / 10) * 0.30
    + (1 - balls_left / innings_balls) * 0.25) * 100.
    """

    if inputs.balls_left <= 0:
        return 100.0
    if inputs.innings_balls <= 0:
        raise ValueError("innings_balls must be positive")
    if inputs.runs_needed < 0:
        raise ValueError("runs_needed cannot be negative")
    if not 0 <= inputs.wickets_lost <= 10:
        raise ValueError("wickets_lost must be between 0 and 10")

    score = (
        (inputs.runs_needed / inputs.balls_left) * 0.45
        + (inputs.wickets_lost / 10.0) * 0.30
        + (1.0 - inputs.balls_left / inputs.innings_balls) * 0.25
    ) * 100.0
    return round(max(0.0, min(100.0, score)), 2)


def pressure_bucket(score: float) -> str:
    """Classify a numeric pressure score into a human-readable band."""

    if score < 30:
        return "low"
    if score < 60:
        return "medium"
    if score < 80:
        return "high"
    return "extreme"

