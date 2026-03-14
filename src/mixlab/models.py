from __future__ import annotations

from pydantic import BaseModel


class Track(BaseModel):
    track_id: str
    artist: str
    title: str
    bpm: float
    camelot_key: str
    genre: str


class PlayedTrack(BaseModel):
    artist: str
    title: str


class MixConcept(BaseModel):
    title: str
    mood: str
    track_ids: list[str]
