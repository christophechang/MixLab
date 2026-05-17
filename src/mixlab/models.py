from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

TrackMode = Literal["unplayed", "all", "played"]


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
    enrichment_confidence: Literal["high", "medium", "low", ""] = ""


class PlayedTrack(BaseModel):
    artist: str
    title: str


class Transition(BaseModel):
    from_id: str
    to_id: str
    is_risky: bool = False
    risk_type: str = ""  # "chapter_pivot" | "peak_impact" | "deliberate_reset"
    # | "closer_move" | "cut_only" | "low_tonal_risk" | ""


# Structured energy/mood arc descriptor emitted by Stage 2 curation.
# Structural arcs: plateau, wave, progressive-build, build-and-drop, double-peak,
# sustained-pressure, front-loaded.
# Directional/mood arcs: dark-to-light, light-to-dark, narrative, abstract-journey.
ArcType = Literal[
    "plateau",
    "wave",
    "progressive-build",
    "build-and-drop",
    "double-peak",
    "sustained-pressure",
    "front-loaded",
    "dark-to-light",
    "light-to-dark",
    "narrative",
    "abstract-journey",
]


class MixConcept(BaseModel):
    title: str
    mood: str
    track_ids: list[str]
    transitions: list[Transition] = Field(default_factory=list)
    name_reason: str = ""
    arc_type: ArcType | None = None


@dataclass
class BpmPools:
    core: list[Track]  # |bpm - median| <= 6
    bridge: list[Track]  # 6 < |bpm - median| <= 12
    wildcard: list[Track]  # |bpm - median| > 12


@dataclass
class CanvasRoleCandidates:
    opener: list[str]
    groove_locker: list[str]
    builder: list[str]
    pivot: list[str]
    peak: list[str]
    closer: list[str]


@dataclass
class ContrastAssets:
    vocal_moments: list[str]
    texture_changes: list[str]
    darker_turns: list[str]
    brighter_lifts: list[str]
    lower_pressure_resets: list[str]


@dataclass
class CanvasScore:
    technical_viability: float = 0.0
    role_coverage: float = 0.0
    anchor_strength: float = 0.0
    contrast_potential: float = 0.0
    distinctiveness: float = 1.0
    novelty: float = 1.0
    weakness_penalty: float = 0.0  # subtracted from the weighted sum (max 0.20)
    floor_multiplier: float = 1.0  # 0.5 when core_n < 8, 1.0 otherwise
    overall: float = 0.0


@dataclass
class MixCanvas:
    canvas_id: str
    genre: str
    bpm_range: tuple[float, float]
    dominant_bpm: float
    dominant_camelot: str
    core_track_ids: list[str]
    bridge_track_ids: list[str]
    wildcard_track_ids: list[str]
    roles: CanvasRoleCandidates
    contrast: ContrastAssets
    risk_notes: list[str]
    score: CanvasScore
    source_concept: MixConcept


SeedTier = Literal["anchor", "supporting", "optional"]
SetRole = Literal[
    "opener",
    "world_setter",
    "groove_locker",
    "early_hook",
    "builder",
    "connector",
    "pivot",
    "pressure",
    "lift",
    "vocal_moment",
    "texture_change",
    "cleanser",
    "risk",
    "weapon",
    "peak",
    "post_peak",
    "resolution",
    "closer",
    "utility",
    "unknown",
]
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
    bpm_smoothness: float  # 0.0–1.0
    harmonic_ratio: float  # 0.0–1.0
    risk_justified: float  # 0.0–1.0
    fragment_preserved: float  # 0.0–1.0

    @property
    def overall(self) -> float:
        return (
            self.bpm_smoothness * 0.30
            + self.harmonic_ratio * 0.30
            + self.risk_justified * 0.25
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
