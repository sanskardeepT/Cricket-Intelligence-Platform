"""Sequential live win model interface.

TensorFlow is intentionally imported lazily so the rest of the platform can run
in lightweight environments while still providing a production train path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LSTMConfig:
    """Configuration for the 50-ball sequence model."""

    sequence_length: int = 50
    n_features: int = 12
    lstm_units: int = 64


def build_lstm_model(config: LSTMConfig):
    """Build a Keras model with input shape (batch, 50_balls, n_features)."""

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("tensorflow is required for LSTM training") from exc
    inputs = keras.Input(shape=(config.sequence_length, config.n_features))
    x = keras.layers.Masking()(inputs)
    x = keras.layers.LSTM(config.lstm_units, dropout=0.15, recurrent_dropout=0.05)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def validate_sequence_batch(batch: np.ndarray, config: LSTMConfig) -> None:
    """Validate LSTM input shape before training/inference."""

    if batch.ndim != 3:
        raise ValueError("batch must be 3D: (batch, 50_balls, n_features)")
    if batch.shape[1] != config.sequence_length or batch.shape[2] != config.n_features:
        raise ValueError(
            f"expected (*, {config.sequence_length}, {config.n_features}), got {batch.shape}"
        )

