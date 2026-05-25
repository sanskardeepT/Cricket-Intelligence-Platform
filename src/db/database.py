"""PostgreSQL persistence for match state and prediction accuracy tracking."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
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
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError as exc:
        raise RuntimeError(f"database is not reachable at {host}:{port}") from exc
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for PostgreSQL; install requirements.txt") from exc
    return psycopg.connect(url, connect_timeout=5)


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


def update_prediction_actual(prediction_id: str, actual_value: str) -> dict[str, Any] | None:
    """Attach an actual outcome to a prediction and mark correctness."""

    if not database_url():
        return None
    try:
        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE predictions
                    SET actual_value = %s,
                        is_correct = (lower(predicted_value) = lower(%s))
                    WHERE pred_id = %s
                    RETURNING pred_id::text, pred_type, predicted_value, actual_value, is_correct
                    """,
                    (actual_value, actual_value, prediction_id),
                )
                row = cursor.fetchone()
            conn.commit()
    except Exception:
        return None
    if row is None:
        return None
    return {
        "pred_id": row[0],
        "pred_type": row[1],
        "predicted_value": row[2],
        "actual_value": row[3],
        "is_correct": row[4],
    }


def prediction_accuracy_summary() -> dict[str, Any]:
    """Return aggregate prediction accuracy grouped by prediction type."""

    if not database_url():
        return {"configured": False, "summary": []}
    try:
        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        pred_type,
                        count(*) AS total_predictions,
                        count(actual_value) AS resolved_predictions,
                        coalesce(avg(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END)
                            FILTER (WHERE actual_value IS NOT NULL), 0) AS accuracy,
                        avg(confidence) AS avg_confidence,
                        avg(probability) AS avg_probability
                    FROM predictions
                    GROUP BY pred_type
                    ORDER BY total_predictions DESC
                    """
                )
                rows = cursor.fetchall()
    except Exception as exc:
        return {"configured": True, "status": f"error: {exc}", "summary": []}
    return {
        "configured": True,
        "summary": [
            {
                "pred_type": row[0],
                "total_predictions": int(row[1]),
                "resolved_predictions": int(row[2]),
                "accuracy": round(float(row[3]), 4),
                "avg_confidence": round(float(row[4] or 0), 4),
                "avg_probability": round(float(row[5] or 0), 4),
            }
            for row in rows
        ],
    }


def recent_predictions(limit: int = 20) -> dict[str, Any]:
    """Return recent prediction records for auditing."""

    if limit <= 0 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if not database_url():
        return {"configured": False, "predictions": []}
    try:
        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        pred_id::text, match_id, pred_type, predicted_value, confidence,
                        probability, model_version, actual_value, is_correct, created_at
                    FROM predictions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        return {"configured": True, "status": f"error: {exc}", "predictions": []}
    return {
        "configured": True,
        "predictions": [
            {
                "pred_id": row[0],
                "match_id": row[1],
                "pred_type": row[2],
                "predicted_value": row[3],
                "confidence": float(row[4]),
                "probability": float(row[5]) if row[5] is not None else None,
                "model_version": row[6],
                "actual_value": row[7],
                "is_correct": row[8],
                "created_at": row[9].isoformat(),
            }
            for row in rows
        ],
    }
