"""Live match prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import LivePredictionRequest
from src.explainer.ball_explainer import predict_ball_outcome
from src.explainer.shap_explainer import fallback_reasons, net_explanation
from src.features.match_features import MatchState, build_match_features
from src.features.pressure_index import PressureInputs, pressure_index
from src.models.ensemble import heuristic_live_prediction
from src.models.ball_outcome import predict_next_ball_with_artifact
from src.models.inference import predict_with_artifact
from src.models.monte_carlo import SimulationState, simulate_chase
from src.db.database import PredictionLog, log_prediction


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
    heuristic_prediction = heuristic_live_prediction(
        elo_prior=float(features["elo_prior"]),
        monte_carlo_probability=float(mc["win_probability"]),
        pressure=pressure,
        required_run_rate=float(features["required_run_rate"]),
        current_run_rate=float(features["current_run_rate"]),
    )
    artifact_prediction = predict_with_artifact(request)
    if artifact_prediction:
        model_votes = {
            "trained_artifact": artifact_prediction.probability,
            "heuristic": heuristic_prediction.probability,
            "monte_carlo": round(float(mc["win_probability"]), 4),
        }
        prediction = {
            "probability": artifact_prediction.probability,
            "confidence": artifact_prediction.confidence,
            "label": artifact_prediction.label,
            "model_name": artifact_prediction.model_name,
            "source": "trained_artifact",
            "model_votes": model_votes,
        }
    else:
        prediction = {
            **heuristic_prediction.__dict__,
            "model_name": "demo-heuristic-v1",
            "source": "heuristic_fallback",
        }
    explanation_features = {
        "elo_prior": float(features["elo_prior"]),
        "current_run_rate": float(features["current_run_rate"]),
        "required_run_rate": float(features["required_run_rate"]),
        "pressure": pressure,
        "balls_left": float(features["balls_left"]),
    }
    reasons = fallback_reasons(explanation_features, float(prediction["probability"]))
    ball = predict_next_ball_with_artifact(request) or predict_ball_outcome(
        pressure, request.batter_settle_score, request.bowler_fatigue
    )
    return {
        "match_state": features,
        "pressure_index": pressure,
        "prediction": prediction,
        "monte_carlo": mc,
        "ball_prediction": ball,
        "explanation": net_explanation(float(prediction["probability"]), reasons),
    }


@router.post("/predict")
def predict_live(request: LivePredictionRequest) -> dict[str, object]:
    """Return live win probability, next-ball forecast, and explanations."""

    payload = build_live_payload(request)
    prediction = payload["prediction"]
    pred_id = log_prediction(
        PredictionLog(
            pred_type="live_win_probability",
            predicted_value=str(prediction["label"]),
            confidence=float(prediction["confidence"]),
            probability=float(prediction["probability"]),
            match_id=None,
            explanation=payload["explanation"],
            feature_snapshot=payload["match_state"],
        )
    )
    payload["prediction_id"] = pred_id
    return payload


@router.get("/demo")
def demo_live() -> dict[str, object]:
    """Return a realistic demo payload for the frontend."""

    return build_live_payload(LivePredictionRequest())
