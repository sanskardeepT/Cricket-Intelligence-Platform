"""FastAPI entry point for the Cricket Intelligence Platform."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.accuracy import router as accuracy_router
from src.api.routes.live import router as live_router
from src.api.routes.prematch import router as prematch_router
from src.api.websocket import router as websocket_router
from src.db.database import database_health, initialize_schema


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
    app.include_router(accuracy_router)
    app.include_router(prematch_router)
    app.include_router(live_router)
    app.include_router(websocket_router)

    @app.on_event("startup")
    def startup() -> None:
        try:
            initialize_schema()
        except Exception as exc:  # pragma: no cover - depends on external DB availability
            app.state.database_startup_error = str(exc)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "cricket-intelligence-platform",
            "database": database_health(),
        }

    return app


app = create_app()
