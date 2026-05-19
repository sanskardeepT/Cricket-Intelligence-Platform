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

## Real Data Flow

```python
from src.data.download import download_cricsheet_dataset
from src.data.preprocessor import load_cricsheet_zip, save_processed_deliveries

zip_path = download_cricsheet_dataset("ipl")
deliveries = load_cricsheet_zip(zip_path)
save_processed_deliveries(deliveries, "data/processed/ipl_deliveries.csv")
```

## Blueprint Status

The repo now contains the complete folder map from the PDF and a working vertical slice across Week 1-12 components. For a serious accuracy claim, train on full historical data with TimeSeriesSplit and log predictions against real results. The app is functional in demo mode immediately; real accuracy depends on downloaded datasets and trained model artifacts.
