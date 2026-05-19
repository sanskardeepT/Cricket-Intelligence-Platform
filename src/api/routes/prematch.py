"""Pre-match and toss prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import PrematchRequest, TossRequest
from src.explainer.toss_explainer import explain_toss_decision
from src.features.match_features import elo_win_probability


router = APIRouter(prefix="/prematch", tags=["prematch"])


@router.post("/winner")
def predict_prematch(request: PrematchRequest) -> dict[str, object]:
    """Predict pre-match winner using ELO, form, and venue context."""

    elo_prior = elo_win_probability(request.team_a_elo, request.team_b_elo)
    form_edge = (request.team_a_form - 50.0) / 250.0
    venue_edge = (request.venue_advantage - 50.0) / 300.0
    probability = max(0.02, min(0.98, elo_prior + form_edge + venue_edge))
    return {
        "team_a": request.team_a,
        "team_b": request.team_b,
        "venue": request.venue,
        "winner": request.team_a if probability >= 0.5 else request.team_b,
        "win_probability": round(probability, 4),
        "reasons": [
            {"factor": "ELO strength", "value": round(elo_prior, 4)},
            {"factor": "Recent form edge", "value": round(form_edge, 4)},
            {"factor": "Venue advantage", "value": round(venue_edge, 4)},
        ],
    }


@router.post("/toss")
def predict_toss(request: TossRequest) -> dict[str, object]:
    """Predict toss decision with blueprint weights."""

    return explain_toss_decision(
        venue_dew_factor=request.venue_dew_factor,
        pitch_deterioration=request.pitch_deterioration,
        captain_field_tendency=request.captain_field_tendency,
        chase_success_rate=request.chase_success_rate,
    )

