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


class Transition(BaseModel):
    from_id: str
    to_id: str
    is_risky: bool = False
    risk_type: str = ""  # "chapter_pivot" | "peak_impact" | "deliberate_reset"
                         # | "closer_move" | "cut_only" | "low_tonal_risk" | ""


class MixConcept(BaseModel):
    title: str
    mood: str
    track_ids: list[str]
    transitions: list[Transition] = Field(default_factory=list)


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
class DJPracticalityScore:
    bpm_smoothness: float     # 0.0–1.0
    harmonic_ratio: float     # 0.0–1.0
    risk_justified: float     # 0.0–1.0
    fragment_preserved: float # 0.0–1.0

    @property
    def overall(self) -> float:
        return (
            self.bpm_smoothness       * 0.30
            + self.harmonic_ratio     * 0.30
            + self.risk_justified     * 0.25
            + self.fragment_preserved * 0.15
        )


@dataclass
class CompletionVariant:
    strategy: Literal["practical", "balanced", "adventurous"]
    concept: MixConcept
    anchor_retention_rate: float  # retained_anchors / total_anchors
    practicality_score: DJPracticalityScore

    @property
    def score(self) -> float:
        return self.practicality_score.overall
