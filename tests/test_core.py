from src.api.routes.live import build_live_payload
from src.api.schemas import LivePredictionRequest
from src.features.match_features import balls_bowled_from_overs, elo_win_probability
from src.features.pressure_index import PressureInputs, pressure_index
from src.models.monte_carlo import SimulationState, simulate_chase


def test_cricket_over_conversion():
    assert balls_bowled_from_overs(15.2) == 92


def test_pressure_index_range():
    score = pressure_index(PressureInputs(runs_needed=50, balls_left=28, wickets_lost=4))
    assert 0 <= score <= 100


def test_elo_probability_bounds():
    probability = elo_win_probability(1545, 1510)
    assert 0.5 < probability < 1.0


def test_monte_carlo_payload():
    result = simulate_chase(SimulationState(score=128, wickets=4, target=178, balls_left=28), simulations=200)
    assert 0 <= result["win_probability"] <= 1


def test_live_payload_shape():
    payload = build_live_payload(LivePredictionRequest())
    assert "prediction" in payload
    assert "ball_prediction" in payload
    assert "explanation" in payload
