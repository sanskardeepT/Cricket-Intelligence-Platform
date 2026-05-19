"""Live match prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import LivePredictionRequest
from src.explainer.ball_explainer import predict_ball_outcome
from src.explainer.shap_explainer import fallback_reasons, net_explanation
from src.features.match_features import MatchState, build_match_features
from src.features.pressure_index import PressureInputs, pressure_index
from src.models.ensemble import heuristic_live_prediction
from src.models.monte_carlo import SimulationState, simulate_chase


router = APIRouter(prefix="/live", tags=["live"])


def build_live_payload(request: LivePredictionRequest) -> dict[str, object]:
    """Build the complete live prediction response."""

    state = MatchState(
        batting_team=request.batting_team,
        bowling_team=request.bowling_team,
        venue=request.venue,
        score=request.score,
        wickets=request.wickets,
        overs=request.overs,
        target=request.target,
        batting_elo=request.batting_elo,
        bowling_elo=request.bowling_elo,
    )
    features = build_match_features(state)
    pressure = pressure_index(
        PressureInputs(
            runs_needed=max(request.target - request.score, 0),
            balls_left=int(features["balls_left"]),
            wickets_lost=request.wickets,
        )
    )
    mc = simulate_chase(
        SimulationState(
            score=request.score,
            wickets=request.wickets,
            target=request.target,
            balls_left=int(features["balls_left"]),
            pressure=pressure,
            batter_settle_score=request.batter_settle_score,
            bowler_fatigue=request.bowler_fatigue,
        ),
        simulations=2500,
    )
    prediction = heuristic_live_prediction(
        elo_prior=float(features["elo_prior"]),
        monte_carlo_probability=float(mc["win_probability"]),
        pressure=pressure,
        required_run_rate=float(features["required_run_rate"]),
        current_run_rate=float(features["current_run_rate"]),
    )
    explanation_features = {
        "elo_prior": float(features["elo_prior"]),
        "current_run_rate": float(features["current_run_rate"]),
        "required_run_rate": float(features["required_run_rate"]),
        "pressure": pressure,
        "balls_left": float(features["balls_left"]),
    }
    reasons = fallback_reasons(explanation_features, prediction.probability)
    ball = predict_ball_outcome(pressure, request.batter_settle_score, request.bowler_fatigue)
    return {
        "match_state": features,
        "pressure_index": pressure,
        "prediction": prediction.__dict__,
        "monte_carlo": mc,
        "ball_prediction": ball,
        "explanation": net_explanation(prediction.probability, reasons),
    }


@router.post("/predict")
def predict_live(request: LivePredictionRequest) -> dict[str, object]:
    """Return live win probability, next-ball forecast, and explanations."""

    return build_live_payload(request)


@router.get("/demo")
def demo_live() -> dict[str, object]:
    """Return a realistic demo payload for the frontend."""

    return build_live_payload(LivePredictionRequest())

