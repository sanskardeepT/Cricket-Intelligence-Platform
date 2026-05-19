"""PostgreSQL persistence for match state and prediction accuracy tracking."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class PredictionLog:
    """Prediction record stored for real-world accuracy tracking."""

    pred_type: str
    predicted_value: str
    confidence: float
    probability: float | None
    match_id: str | None
    explanation: dict[str, Any]
    feature_snapshot: dict[str, Any]
    model_version: str = "demo-heuristic-v1"


def database_url() -> str | None:
    """Return configured PostgreSQL URL, if present."""

    return os.getenv("DATABASE_URL")


def connect():
    """Create a psycopg connection or raise a clear runtime error."""

    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for PostgreSQL; install requirements.txt") from exc
    return psycopg.connect(url)


def initialize_schema() -> bool:
    """Create all platform tables when DATABASE_URL is configured.

    Returns False when no database URL exists, allowing local demo mode to run
    without PostgreSQL.
    """

    if not database_url():
        return False
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(schema)
        conn.commit()
    return True


def log_prediction(record: PredictionLog) -> str | None:
    """Persist a prediction and return its UUID; skip cleanly without DB."""

    if not database_url():
        return None
    pred_id = str(uuid4())
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    pred_id, match_id, pred_type, predicted_value, confidence,
                    probability, model_version, explanation, feature_snapshot
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    pred_id,
                    record.match_id,
                    record.pred_type,
                    record.predicted_value,
                    record.confidence,
                    record.probability,
                    record.model_version,
                    json.dumps(record.explanation),
                    json.dumps(record.feature_snapshot),
                ),
            )
        conn.commit()
    return pred_id


def database_health() -> dict[str, str | bool]:
    """Return database connection status without crashing the API."""

    if not database_url():
        return {"configured": False, "status": "not_configured"}
    try:
        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return {"configured": True, "status": "ok"}
    except Exception as exc:  # pragma: no cover - depends on external service
        return {"configured": True, "status": f"error: {exc}"}
