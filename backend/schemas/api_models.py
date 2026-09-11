from typing import Any, Literal

from pydantic import BaseModel


class Availability(BaseModel):
    status: Literal["available", "partial", "unavailable", "pending", "blocked"]
    message: str | None = None
    provenance: dict[str, Any] | None = None


class PlayerState(BaseModel):
    player_id: str
    team: Literal["home", "away"]
    x: float | None
    y: float | None
    confidence: float
    visible: bool


class GameState(BaseModel):
    match_id: str
    frame: int
    timestamp: float
    fps: float
    players: list[PlayerState]
    provenance: dict[str, Any]


class Match(BaseModel):
    match_id: str
    label: str
    fps: float
    frame_count: int
    events_available: bool
    tracking_available: bool


class ApiEnvelope(BaseModel):
    availability: Availability
    data: Any = None