"""LightGBM win predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LightGBMWinModel:
    """Fast gradient boosting model used as an ensemble member."""

    def __init__(self) -> None:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed; run pip install -r requirements.txt") from exc
        self.model = LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.feature_columns: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMWinModel":
        """Fit the LightGBM classifier."""

        if features.empty:
            raise ValueError("features cannot be empty")
        self.feature_columns = list(features.columns)
        self.model.fit(features, target)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities."""

        if not self.feature_columns:
            raise RuntimeError("model must be fitted before prediction")
        return self.model.predict_proba(features[self.feature_columns])

