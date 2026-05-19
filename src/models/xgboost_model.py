"""XGBoost win predictor with TimeSeriesSplit validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class BacktestResult:
    """Time-series backtest metrics."""

    accuracy: float
    roc_auc: float
    folds: int


class XGBoostWinModel:
    """Trainable XGBoost model for match win probability."""

    def __init__(self) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed; run pip install -r requirements.txt") from exc
        self.model = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        self.feature_columns: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "XGBoostWinModel":
        """Fit the model on numeric features."""

        if features.empty:
            raise ValueError("features cannot be empty")
        self.feature_columns = list(features.columns)
        self.model.fit(features, target)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""

        if not self.feature_columns:
            raise RuntimeError("model must be fitted before prediction")
        return self.model.predict_proba(features[self.feature_columns])

    def backtest(self, features: pd.DataFrame, target: pd.Series, splits: int = 5) -> BacktestResult:
        """Evaluate with TimeSeriesSplit, never random future leakage."""

        if len(features) <= splits:
            raise ValueError("not enough rows for requested TimeSeriesSplit")
        tscv = TimeSeriesSplit(n_splits=splits)
        accuracies: list[float] = []
        aucs: list[float] = []
        for train_idx, test_idx in tscv.split(features):
            model = type(self)()
            model.fit(features.iloc[train_idx], target.iloc[train_idx])
            probs = model.predict_proba(features.iloc[test_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            accuracies.append(accuracy_score(target.iloc[test_idx], preds))
            if len(set(target.iloc[test_idx])) > 1:
                aucs.append(roc_auc_score(target.iloc[test_idx], probs))
        return BacktestResult(
            accuracy=round(float(np.mean(accuracies)), 4),
            roc_auc=round(float(np.mean(aucs)), 4) if aucs else 0.0,
            folds=splits,
        )

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, target)
        return target

