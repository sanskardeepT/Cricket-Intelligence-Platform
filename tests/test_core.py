from src.api.routes.live import build_live_payload
from src.api.schemas import LivePredictionRequest
from src.features.match_features import balls_bowled_from_overs, elo_win_probability
from src.features.pressure_index import PressureInputs, pressure_index
from src.models.monte_carlo import SimulationState, simulate_chase
from src.db.database import database_health, initialize_schema
from src.data.ingest import build_delivery_rows, build_match_rows, summarize_source
from src.features.matrix import build_feature_frame, write_feature_matrices
from src.models.training import train_baselines
from src.models.inference import load_artifact, predict_with_artifact
from src.data.preprocessor import _normalize_columns
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


def test_preprocessor_derives_over_from_cricsheet_ball_notation():
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "ball": 0.1, "batting_team": "MI", "bowling_team": "CSK", "runs": 0},
            {"match_id": "m1", "ball": 0.6, "batting_team": "MI", "bowling_team": "CSK", "runs": 1},
            {"match_id": "m1", "ball": 15.2, "batting_team": "MI", "bowling_team": "CSK", "runs": 4},
        ]
    )

    normalized = _normalize_columns(frame)

    assert normalized["over"].tolist() == [0, 0, 15]
    assert normalized["ball"].tolist() == [1, 6, 2]


def test_preprocessor_preserves_extra_delivery_notation():
    frame = pd.DataFrame(
        [{"match_id": "m1", "ball": 2.8, "batting_team": "MI", "bowling_team": "CSK", "runs": 1}]
    )

    normalized = _normalize_columns(frame)

    assert normalized["over"].tolist() == [2]
    assert normalized["ball"].tolist() == [8]


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


def test_feature_matrix_builder(tmp_path):
    rows = []
    for innings in [1, 2]:
        for ball in range(1, 7):
            rows.append(
                {
                    "match_id": "m1",
                    "innings": innings,
                    "over": 0,
                    "ball": ball,
                    "batting_team": "MI" if innings == 1 else "CSK",
                    "bowling_team": "CSK" if innings == 1 else "MI",
                    "batter": "Batter",
                    "bowler": "Bowler",
                    "runs": 2 if innings == 1 else 1,
                    "extras": 0,
                    "is_wicket": 0,
                    "venue": "Wankhede Stadium",
                }
            )
    frame = pd.DataFrame(rows)

    features, labels = build_feature_frame(frame)

    assert len(features) == 12
    assert "pressure_index" in features.columns
    assert set(labels.unique()).issubset({0, 1})

    csv_path = tmp_path / "deliveries.csv"
    frame.to_csv(csv_path, index=False)
    summary = write_feature_matrices(csv_path, tmp_path / "features", test_fraction=0.25)

    assert summary.train_rows == 9
    assert (tmp_path / "features" / "X_train.csv").exists()
    assert (tmp_path / "features" / "y_test.csv").exists()


def test_baseline_training_pipeline(tmp_path):
    rows = []
    for match in range(8):
        for innings in [1, 2]:
            for ball in range(1, 7):
                chasing_wins = match % 2 == 0
                first_runs = 1 if chasing_wins else 2
                second_runs = 2 if chasing_wins else 1
                rows.append(
                    {
                        "match_id": f"m{match}",
                        "innings": innings,
                        "over": 0,
                        "ball": ball,
                        "batting_team": "MI" if innings == 1 else "CSK",
                        "bowling_team": "CSK" if innings == 1 else "MI",
                        "batter": "Batter",
                        "bowler": "Bowler",
                        "runs": first_runs if innings == 1 else second_runs,
                        "extras": 0,
                        "is_wicket": 0,
                        "venue": "Wankhede Stadium",
                    }
                )
    csv_path = tmp_path / "deliveries.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_feature_matrices(csv_path, tmp_path / "features", test_fraction=0.25)

    summary = train_baselines(tmp_path / "features", tmp_path / "artifacts", folds=3)

    assert summary.best_model in {"logistic_regression", "random_forest"}
    assert summary.features > 0
    assert (tmp_path / "artifacts" / "win_probability_baseline.joblib").exists()
    assert (tmp_path / "artifacts" / "training_metrics.json").exists()


def test_artifact_inference_pipeline(tmp_path):
    rows = []
    for match in range(8):
        for innings in [1, 2]:
            for ball in range(1, 7):
                rows.append(
                    {
                        "match_id": f"m{match}",
                        "innings": innings,
                        "over": 0,
                        "ball": ball,
                        "batting_team": "MI" if innings == 1 else "CSK",
                        "bowling_team": "CSK" if innings == 1 else "MI",
                        "batter": "Batter",
                        "bowler": "Bowler",
                        "runs": 1 if innings == 1 else 2,
                        "extras": 0,
                        "is_wicket": 0,
                        "venue": "Wankhede Stadium",
                    }
                )
    csv_path = tmp_path / "deliveries.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_feature_matrices(csv_path, tmp_path / "features", test_fraction=0.25)
    summary = train_baselines(tmp_path / "features", tmp_path / "artifacts", folds=3)

    load_artifact.cache_clear()
    prediction = predict_with_artifact(LivePredictionRequest(), summary.artifact_path)

    assert prediction is not None
    assert 0 <= prediction.probability <= 1
