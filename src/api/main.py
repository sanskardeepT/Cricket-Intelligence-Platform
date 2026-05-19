"""FastAPI entry point for the Cricket Intelligence Platform."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.live import router as live_router
from src.api.routes.prematch import router as prematch_router
from src.api.websocket import router as websocket_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""

    app = FastAPI(
        title="Cricket Intelligence Platform",
        description="IPL/T20/ODI win probability, toss, ball prediction, and scientific explanations.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(prematch_router)
    app.include_router(live_router)
    app.include_router(websocket_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "cricket-intelligence-platform"}

    return app


app = create_app()

