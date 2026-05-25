"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LivePredictionRequest(BaseModel):
    """Live prediction request from frontend or client."""

    batting_team: str = Field(default="MI")
    bowling_team: str = Field(default="CSK")
    venue: str = Field(default="Wankhede Stadium")
    score: int = Field(default=128, ge=0)
    wickets: int = Field(default=4, ge=0, le=10)
    overs: float = Field(default=15.2, ge=0)
    target: int = Field(default=178, ge=1)
    batting_elo: float = Field(default=1545)
    bowling_elo: float = Field(default=1510)
    batter_settle_score: float = Field(default=72, ge=0, le=100)
    bowler_fatigue: float = Field(default=61, ge=0, le=100)


class TossRequest(BaseModel):
    """Toss predictor request."""

    venue_dew_factor: float = Field(default=78, ge=0, le=100)
    pitch_deterioration: float = Field(default=38, ge=0, le=100)
    captain_field_tendency: float = Field(default=67, ge=0, le=100)
    chase_success_rate: float = Field(default=64, ge=0, le=100)


class PrematchRequest(BaseModel):
    """Pre-match team strength request."""

    team_a: str = "MI"
    team_b: str = "CSK"
    venue: str = "Wankhede Stadium"
    team_a_elo: float = 1545
    team_b_elo: float = 1510
    team_a_form: float = Field(default=62, ge=0, le=100)
    venue_advantage: float = Field(default=58, ge=0, le=100)


class PredictionActualRequest(BaseModel):
    """Resolve a logged prediction with the real outcome."""

    actual_value: str = Field(min_length=1, max_length=100)
