# Cricket Intelligence Platform

Full-stack cricket prediction system for IPL, T20, and ODI match intelligence. It predicts live win probability, toss decision, next-ball outcome, and explains every result with scientific reasons.

## What Is Built

- Data ingestion for Cricsheet IPL/T20/ODI zips.
- Cleaning pipeline for Kaggle/Cricsheet deliveries.
- Feature engineering for run rates, ELO, pressure index, player form, H2H, and venue DNA.
- Model layer for XGBoost, LightGBM, LSTM, Random Forest, Monte Carlo, and Logistic Regression stacking.
- Explanation layer for SHAP, toss logic, and ball-by-ball reasons.
- FastAPI backend with `/health`, `/prematch/winner`, `/prematch/toss`, `/live/demo`, `/live/predict`, and `/ws/live`.
- React dashboard with win probability chart, next-ball predictor, reason cards, and model vote panel.
- Docker Compose for API, PostgreSQL, and Redis.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

For full training with all heavyweight model libraries:

```powershell
pip install -r requirements-ml.txt
```

## API Smoke Checks

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/live/demo
```

## Database Setup

The API runs without PostgreSQL for demo mode. For real prediction logging and training data storage, start PostgreSQL through Docker Compose:

```powershell
docker compose up -d postgres redis
$env:DATABASE_URL="postgresql://cricket:cricket@localhost:5432/cricket"
python scripts/init_db.py
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

The schema creates `matches`, `deliveries`, `players`, `player_match_stats`, `venues`, `live_match_state`, and `predictions`. Live prediction requests write to `predictions` when `DATABASE_URL` is configured.

## Real Data Flow

```python
from src.data.download import download_cricsheet_dataset
from src.data.preprocessor import load_cricsheet_zip, save_processed_deliveries

zip_path = download_cricsheet_dataset("ipl")
deliveries = load_cricsheet_zip(zip_path)
save_processed_deliveries(deliveries, "data/processed/ipl_deliveries.csv")
```

Validate and load data into PostgreSQL:

```powershell
python scripts/ingest_data.py data/processed/ipl_deliveries.csv --dry-run
$env:DATABASE_URL="postgresql://cricket:cricket@localhost:5432/cricket"
python scripts/ingest_data.py data/processed/ipl_deliveries.csv --format IPL
```

Build model feature matrices with chronological train/test ordering:

```powershell
python scripts/build_features.py data/processed/ipl_deliveries.csv --output-dir data/features
```

This writes `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and `metadata.csv`.

Train baseline win-probability models:

```powershell
python scripts/train_baseline.py --feature-dir data/features --artifact-dir artifacts/models
```

The trainer uses `TimeSeriesSplit`, evaluates Logistic Regression, Random Forest, native Gradient Boosting, and optional XGBoost/LightGBM when installed, then saves the best local artifact to `artifacts/models/`.

When `artifacts/models/win_probability_baseline.joblib` exists, `/live/predict` automatically uses it for live inference. Set `MODEL_ARTIFACT_PATH` to point at a different artifact.

Track real-world prediction accuracy:

```powershell
curl http://127.0.0.1:8000/accuracy/summary
curl http://127.0.0.1:8000/accuracy/recent
curl -X POST http://127.0.0.1:8000/accuracy/predictions/<prediction_id>/actual -H "Content-Type: application/json" -d "{\"actual_value\":\"win\"}"
```

Latest local IPL baseline run:

- Source: Cricsheet `ipl_csv2.zip`
- Processed deliveries: 293,308 rows across 1,233 matches
- Feature matrix: 19 columns, 234,646 train rows, 58,662 test rows
- Best baseline: Logistic Regression
- Test accuracy: 68.59%
- Test ROC-AUC: 0.7686
- Test log loss: 0.5676
- Compared models: Logistic Regression, Random Forest, HistGradientBoosting

## Blueprint Status

The repo now contains the complete folder map from the PDF and a working vertical slice across Week 1-12 components. For a serious accuracy claim, train on full historical data with TimeSeriesSplit and log predictions against real results. The app is functional in demo mode immediately; real accuracy depends on downloaded datasets and trained model artifacts.
