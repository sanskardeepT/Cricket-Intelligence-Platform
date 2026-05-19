"""Toss decision prediction with weighted scientific reasons."""

from __future__ import annotations


def explain_toss_decision(
    venue_dew_factor: float,
    pitch_deterioration: float,
    captain_field_tendency: float,
    chase_success_rate: float,
) -> dict[str, object]:
    """Predict bat/field using blueprint toss weights."""

    for value in [venue_dew_factor, pitch_deterioration, captain_field_tendency, chase_success_rate]:
        if not 0 <= value <= 100:
            raise ValueError("all toss factors must be 0-100 percentages")
    field_score = (
        venue_dew_factor * 0.35
        + (100.0 - pitch_deterioration) * 0.25
        + captain_field_tendency * 0.25
        + chase_success_rate * 0.15
    )
    decision = "FIELD" if field_score >= 50 else "BAT"
    confidence = round(abs(field_score - 50.0) / 50.0, 3)
    return {
        "decision": decision,
        "field_score": round(field_score, 2),
        "confidence": confidence,
        "reasons": [
            {"factor": "Venue Dew Factor", "weight": 0.35, "value": venue_dew_factor},
            {"factor": "Pitch Deterioration", "weight": 0.25, "value": pitch_deterioration},
            {"factor": "Captain Field Tendency", "weight": 0.25, "value": captain_field_tendency},
            {"factor": "Chase Success Rate", "weight": 0.15, "value": chase_success_rate},
        ],
    }

