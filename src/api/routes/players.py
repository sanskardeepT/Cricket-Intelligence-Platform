"""Player prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import PlayerRunsRequest
from src.models.player_runs import predict_player_runs


router = APIRouter(prefix="/players", tags=["players"])


@router.post("/runs")
def predict_runs(request: PlayerRunsRequest) -> dict[str, object]:
    """Predict expected batter runs for an upcoming innings."""

    return predict_player_runs(request)
