"""WebSocket stream for live demo updates."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.routes.live import build_live_payload
from src.api.schemas import LivePredictionRequest


router = APIRouter(tags=["websocket"])


@router.websocket("/ws/live")
async def live_websocket(websocket: WebSocket) -> None:
    """Stream evolving live predictions every two seconds."""

    await websocket.accept()
    request = LivePredictionRequest()
    try:
        for tick in range(8):
            score = request.score + tick * 3
            balls_bowled = 15 * 6 + 2 + tick
            overs = float(f"{balls_bowled // 6}.{balls_bowled % 6}")
            wickets = min(9, request.wickets + (1 if tick == 5 else 0))
            payload = build_live_payload(request.model_copy(update={"score": score, "overs": overs, "wickets": wickets}))
            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return
