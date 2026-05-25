"""Prediction accuracy tracking endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import PredictionActualRequest
from src.db.database import prediction_accuracy_summary, recent_predictions, update_prediction_actual


router = APIRouter(prefix="/accuracy", tags=["accuracy"])


@router.get("/summary")
def accuracy_summary() -> dict[str, object]:
    """Return aggregate accuracy metrics from logged predictions."""

    return prediction_accuracy_summary()


@router.get("/recent")
def accuracy_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, object]:
    """Return recent predictions for audit/debug views."""

    return recent_predictions(limit)


@router.post("/predictions/{prediction_id}/actual")
def set_prediction_actual(prediction_id: str, request: PredictionActualRequest) -> dict[str, object]:
    """Set actual result for one prediction and compute correctness."""

    updated = update_prediction_actual(prediction_id, request.actual_value)
    if updated is None:
        raise HTTPException(status_code=404, detail="prediction not found or database not configured")
    return updated
