"""Random Forest player runs predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


class PlayerRunsForest:
    """Predict player runs with a robust tree ensemble."""

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_columns: list[str] = []

    def fit(self, features: pd.DataFrame, target_runs: pd.Series) -> "PlayerRunsForest":
        """Fit player runs regression model."""

        if features.empty:
            raise ValueError("features cannot be empty")
        self.feature_columns = list(features.columns)
        self.model.fit(features, target_runs)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict expected runs."""

        if not self.feature_columns:
            raise RuntimeError("model must be fitted before prediction")
        return self.model.predict(features[self.feature_columns])

    def backtest_mae(self, features: pd.DataFrame, target_runs: pd.Series, splits: int = 5) -> float:
        """Return TimeSeriesSplit mean absolute error."""

        tscv = TimeSeriesSplit(n_splits=splits)
        errors: list[float] = []
        for train_idx, test_idx in tscv.split(features):
            model = PlayerRunsForest().fit(features.iloc[train_idx], target_runs.iloc[train_idx])
            preds = model.predict(features.iloc[test_idx])
            errors.append(mean_absolute_error(target_runs.iloc[test_idx], preds))
        return round(float(np.mean(errors)), 3)

