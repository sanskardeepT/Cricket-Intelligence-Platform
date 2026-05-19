from src.api.routes.live import build_live_payload
from src.api.schemas import LivePredictionRequest
from src.features.match_features import balls_bowled_from_overs, elo_win_probability
from src.features.pressure_index import PressureInputs, pressure_index
from src.models.monte_carlo import SimulationState, simulate_chase
from src.db.database import database_health, initialize_schema
from src.data.ingest import build_delivery_rows, build_match_rows, summarize_source
import pandas as pd


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


def test_database_skips_without_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert initialize_schema() is False
    assert database_health()["status"] == "not_configured"


def test_ingestion_summary_from_csv(tmp_path):
    csv_path = tmp_path / "deliveries.csv"
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "innings": 1,
                "over": 0,
                "ball": 1,
                "batting_team": "MI",
                "bowling_team": "CSK",
                "batter": "R Sharma",
                "bowler": "D Chahar",
                "runs": 4,
                "extras": 0,
                "is_wicket": 0,
                "venue": "Wankhede Stadium",
            }
        ]
    ).to_csv(csv_path, index=False)

    summary = summarize_source(csv_path)

    assert summary.matches == 1
    assert summary.deliveries == 1
    assert summary.players == 2
    assert summary.venues == 1


def test_ingestion_row_builders():
    frame = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "innings": 1,
                "over": 0,
                "ball": 1,
                "batting_team": "MI",
                "bowling_team": "CSK",
                "batter": "R Sharma",
                "bowler": "D Chahar",
                "runs": 1,
                "extras": 0,
                "is_wicket": 0,
            }
        ]
    )

    assert build_match_rows(frame)[0]["team1"] == "CSK"
    assert build_delivery_rows(frame)[0]["batter"] == "R Sharma"
