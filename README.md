# 🏏 Cricket Intelligence Platform

An advanced AI-powered cricket analytics and prediction platform built for IPL, T20, and ODI matches.
This project combines machine learning, real-time match intelligence, statistical modeling, and explainable AI to predict outcomes with data-backed reasoning instead of random probability guesses.

The platform can analyze live match situations, predict win probability, estimate next-ball outcomes, evaluate toss decisions, and explain *why* a prediction was generated using contextual cricket intelligence.

---

## 🚀 Why This Project Exists

Most cricket prediction systems only display percentages without explaining the actual match context behind them.

This platform was built to solve that problem by combining:

* Historical cricket datasets
* Live match momentum analysis
* Venue behavior patterns
* Team pressure situations
* Batter and bowler form
* Explainable AI reasoning

The goal is to create a realistic cricket intelligence engine that behaves more like an analyst than a basic prediction model.

---

# ✨ Core Features

### 📊 Live Win Probability Prediction

Predicts the winning chances of both teams during live matches using:

* Current score
* Required run rate
* Wickets remaining
* Momentum shifts
* Venue scoring trends
* Team strength comparison

---

### 🎯 Next Ball Outcome Prediction

Predicts the probability of:

* Dot Ball
* Single
* Double
* Boundary
* Six
* Wicket
* Extras / Other outcomes

using ball-by-ball match context and player behavior patterns.

---

### 🪙 Toss Decision Intelligence

Analyzes whether teams should:

* Bat first
* Bowl first

based on:

* Venue history
* Dew factor
* Chasing advantage
* Team composition
* Historical success rates

---

### 🧠 Explainable AI Layer

Instead of only showing predictions, the system explains:

* Why win probability changed
* Which factors influenced predictions
* Match pressure conditions
* Batter vs bowler matchup impact
* Momentum shifts

using SHAP explainability and contextual reasoning.

---

### 📈 Advanced Feature Engineering

The platform generates intelligent cricket features including:

* Dynamic Run Rate
* Required Run Rate
* Pressure Index
* Venue DNA
* Team ELO Ratings
* Batter Form Index
* Bowler Economy Trends
* Head-to-Head Statistics
* Phase-wise Scoring Patterns

---

# 🛠️ Tech Stack

## Backend

* FastAPI
* PostgreSQL
* Redis
* WebSockets
* Python

## Frontend

* React
* Vite
* Recharts

## Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* LSTM
* Monte Carlo Simulation
* Logistic Regression
* Random Forest

## DevOps

* Docker
* Docker Compose

---

# 📂 Project Architecture

```bash
cricket-intelligence-platform/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   └── live_engine/
│
├── scripts/
├── data/
├── artifacts/
├── frontend/
└── docker/
```

---

# ⚡ Quick Start

## 1️⃣ Clone Repository

```bash
git clone <your-repository-url>
cd cricket-intelligence-platform
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment (Windows)

```bash
.\.venv\Scripts\Activate.ps1
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

For full ML model training support:

```bash
pip install -r requirements-ml.txt
```

---

## 4️⃣ Run Backend Server

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 5️⃣ Run Frontend

Open another terminal:

```bash
npm install
npm run dev
```

Frontend usually runs on:

```bash
http://127.0.0.1:5173
```

---

# 🔍 API Endpoints

| Endpoint           | Description                 |
| ------------------ | --------------------------- |
| `/health`          | API health check            |
| `/prematch/winner` | Pre-match winner prediction |
| `/prematch/toss`   | Toss decision intelligence  |
| `/live/demo`       | Demo live prediction        |
| `/live/predict`    | Real-time match prediction  |
| `/ws/live`         | WebSocket live updates      |

---

# 🧪 API Smoke Test

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/live/demo
```

---

# 🗄️ Database Setup

The platform supports running in lightweight demo mode without PostgreSQL.

For full-scale prediction logging and training pipelines:

```bash
docker compose up -d postgres redis
```

Set database URL:

```bash
$env:DATABASE_URL="postgresql://cricket:cricket@localhost:5432/cricket"
```

Initialize schema:

```bash
python scripts/init_db.py
```

Start API:

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

---

# 📥 Real Match Data Pipeline

Download and process Cricsheet datasets:

```python
from src.data.download import download_cricsheet_dataset
from src.data.preprocessor import load_cricsheet_zip, save_processed_deliveries

zip_path = download_cricsheet_dataset("ipl")
deliveries = load_cricsheet_zip(zip_path)

save_processed_deliveries(
    deliveries,
    "data/processed/ipl_deliveries.csv"
)
```

---

# 📦 Data Ingestion

Validate dataset:

```bash
python scripts/ingest_data.py data/processed/ipl_deliveries.csv --dry-run
```

Load into PostgreSQL:

```bash
python scripts/ingest_data.py data/processed/ipl_deliveries.csv --format IPL
```

---

# 🧬 Feature Generation

Generate training datasets:

```bash
python scripts/build_features.py data/processed/ipl_deliveries.csv --output-dir data/features
```

Generated outputs include:

* X_train.csv
* X_test.csv
* y_train.csv
* y_test.csv
* metadata.csv

---

# 🤖 Model Training

Train baseline models:

```bash
python scripts/train_baseline.py --feature-dir data/features --artifact-dir artifacts/models
```

The training pipeline evaluates:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* XGBoost
* LightGBM

using chronological train-test splitting for realistic cricket forecasting.

---

# 📊 Latest IPL Model Performance

## 🏆 Win Probability Model

| Metric     | Result              |
| ---------- | ------------------- |
| Dataset    | Cricsheet IPL       |
| Matches    | 1,233               |
| Deliveries | 293,308             |
| Best Model | Logistic Regression |
| Accuracy   | 68.26%              |
| ROC-AUC    | 0.7673              |
| Log Loss   | 0.5714              |

### Compared Models

* Logistic Regression
* Random Forest
* HistGradientBoosting
* XGBoost
* LightGBM

---

## 🎯 Next Ball Outcome Model

| Metric     | Result               |
| ---------- | -------------------- |
| Best Model | HistGradientBoosting |
| Accuracy   | 43.76%               |
| Log Loss   | 1.4879               |

Predicted classes:

* Dot Ball
* Single
* Two Runs
* Four
* Six
* Wicket
* Other Runs

---

## 🪙 Toss Decision Model

| Metric     | Result               |
| ---------- | -------------------- |
| Best Model | HistGradientBoosting |
| Accuracy   | 76.92%               |
| Log Loss   | 0.7069               |

---

## 🧑‍🏏 Player Runs Prediction

| Metric     | Result               |
| ---------- | -------------------- |
| Best Model | HistGradientBoosting |
| MAE        | 16.325               |
| RMSE       | 22.142               |

---

# 📡 Live Prediction Tracking

Track prediction accuracy over time:

```bash
curl http://127.0.0.1:8000/accuracy/summary
curl http://127.0.0.1:8000/accuracy/recent
```

Update actual outcomes:

```bash
curl -X POST http://127.0.0.1:8000/accuracy/predictions/<prediction_id>/actual \
-H "Content-Type: application/json" \
-d "{\"actual_value\":\"win\"}"
```

---

# 🧠 Future Roadmap

Planned improvements:

* Real-time IPL streaming integration
* Player fatigue modeling
* Fantasy cricket recommendation engine
* AI commentary generation
* Graph Neural Networks for player relationships
* Multi-league support (BBL, PSL, CPL)
* Mobile application
* Reinforcement learning-based strategy engine

---

# 📌 Vision

The long-term vision is to build a complete AI-powered cricket intelligence ecosystem capable of:

* Real-time analytics
* Match simulations
* Tactical recommendations
* Fantasy insights
* Predictive commentary
* Deep cricket intelligence for fans, analysts, and teams

---

# 📄 License

This project is built for educational, research, and analytical purposes.

---

# ⭐ Support

If you found this project useful:

* Star the repository
* Fork the project
* Contribute improvements
* Share feedback

Cricket + AI is just getting started 🚀
