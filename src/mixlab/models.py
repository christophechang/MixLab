from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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


SeedTier = Literal["anchor", "supporting", "optional"]
SetRole = Literal["opener", "builder", "pivot", "peak", "cleanser", "closer", "utility", "unknown"]
EnergyShape = Literal["single_arc", "double_peak", "plateau", "flat", "unclear"]
RiskTolerance = Literal["low", "medium", "high"]


@dataclass
class AdjacencyFragment:
    track_ids: list[str]  # exactly 2 track IDs, in original order
    confidence: float  # 0.0–1.0
    reason: str  # e.g., "camelot_compatible + bpm_close"


@dataclass
class SeedAnalysis:
    track_id: str
    tier: SeedTier
    inferred_role: SetRole
    drop_cost: float  # 0.0 = never drop, 1.0 = freely droppable


@dataclass
class IntentBrief:
    overall_vibe: str
    energy_shape: EnergyShape
    risk_tolerance: RiskTolerance
    is_coherent_set: bool
    seed_analyses: list[SeedAnalysis]  # one per seed track
    missing_roles: list[SetRole]
    strong_adjacencies: list[AdjacencyFragment]
    bpm_range: tuple[float, float]
    # Derived convenience sets — populated by __post_init__
    anchor_ids: frozenset[str] = field(default_factory=frozenset)
    supporting_ids: frozenset[str] = field(default_factory=frozenset)
    optional_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.anchor_ids = frozenset(s.track_id for s in self.seed_analyses if s.tier == "anchor")
        self.supporting_ids = frozenset(s.track_id for s in self.seed_analyses if s.tier == "supporting")
        self.optional_ids = frozenset(s.track_id for s in self.seed_analyses if s.tier == "optional")


@dataclass
class CompletionVariant:
    strategy: Literal["conservative", "bold"]
    concept: MixConcept
    anchor_retention_rate: float  # retained_anchors / total_anchors
    role_coverage: float  # filled_missing_roles / total_missing_roles

    @property
    def score(self) -> float:
        return self.anchor_retention_rate * 0.65 + self.role_coverage * 0.35
