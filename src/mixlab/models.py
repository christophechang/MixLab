from __future__ import annotations

from pydantic import BaseModel, Field


class Track(BaseModel):
    track_id: str
    artist: str
    title: str
    bpm: float
    camelot_key: str
    genre: str
    energy: int | None = None
    label: str = ""
    play_count: int = 0
    tags: list[str] = Field(default_factory=list)
    year: int | None = None
    album: str = ""
    remixer: str = ""
    mix: list[str] = Field(default_factory=list)
    enrichment_confidence: str = ""  # "high" | "medium" | "low" | ""


class PlayedTrack(BaseModel):
    artist: str
    title: str


class MixConcept(BaseModel):
    title: str
    mood: str
    track_ids: list[str]
