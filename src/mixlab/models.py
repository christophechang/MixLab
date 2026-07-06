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
    duration_secs: int | None = None
    date_added: str = ""
    rating: int | None = None


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


CritiqueVerdict = Literal["solid", "needs_attention", "weak"]


class Critique(BaseModel):
    """Stage 2 self-critique result (#22, opt-in via --deep).

    Generated in a separate Anthropic call between curation and the prose report.
    The output is never applied automatically — it surfaces in the report so the
    user can decide whether to accept the concept as shipped or revise.
    """

    verdict: CritiqueVerdict
    single_weakest_moment: str = ""
    structural_issues: list[str] = Field(default_factory=list)
    suggested_substitution: str | None = None


class MixConcept(BaseModel):
    title: str
    mood: str
    track_ids: list[str]
    transitions: list[Transition] = Field(default_factory=list)
    name_reason: str = ""
    arc_type: ArcType | None = None
    critique: Critique | None = None


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


@dataclass(frozen=True)
class CanvasScoreWeights:
    """Per-mode weights for the canvas score components (#24).

    All weights must sum to 1.0. Validated by ``__post_init__``. The eight fields
    line up with the eight :class:`CanvasScore` scoring components — penalty terms
    (weakness_penalty, floor_multiplier) are separate and not weighted.
    """

    technical_viability: float
    role_coverage: float
    anchor_strength: float
    contrast_potential: float
    distinctiveness: float
    era_coherence: float
    label_coherence: float
    novelty: float

    def __post_init__(self) -> None:
        total = (
            self.technical_viability
            + self.role_coverage
            + self.anchor_strength
            + self.contrast_potential
            + self.distinctiveness
            + self.era_coherence
            + self.label_coherence
            + self.novelty
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"CanvasScoreWeights must sum to 1.0; got {total:.6f}")


@dataclass
class CanvasScore:
    technical_viability: float = 0.0
    role_coverage: float = 0.0
    anchor_strength: float = 0.0
    contrast_potential: float = 0.0
    distinctiveness: float = 1.0
    novelty: float = 1.0
    # Era/label coherence (#20). 0.0 = no signal (missing data or scattered) — never a penalty,
    # only a bonus when the canvas has a tight era window or a dominant label.
    era_coherence: float = 0.0
    label_coherence: float = 0.0
    weakness_penalty: float = 0.0  # subtracted from the weighted sum (max 0.20)
    floor_multiplier: float = 1.0  # 0.5 when core_n < 8, 1.0 otherwise
    overall: float = 0.0


ConceptAnchorType = Literal["peak", "identity", "structural-exception"]


@dataclass(frozen=True)
class ConceptAnchorCandidate:
    """A bridge/wildcard track flagged as a structurally exceptional pick (#10).

    The category indicates the dominant reason the system flagged the track —
    Stage 2 still chooses whether and how to use it, but the tag lets the model
    skip rederiving the signal from per-track metadata.
    """

    track_id: str
    anchor_type: ConceptAnchorType


@dataclass(frozen=True)
class ConceptShape:
    """Deterministic shape fingerprint of a concept for shape-novelty scoring (#7).

    Fields are deliberately reliable-or-omitted: ``energy_path`` is excluded from
    similarity comparison when empty (Option A — unknown arc is not a match).
    """

    bpm_band: str  # e.g. "170-180" (10-BPM bucket). "" when unknown.
    camelot_zone: str  # dominant Camelot key. "" when unknown.
    energy_path: str  # arc_type from history. "" when unknown — excluded from comparison.
    has_opener: bool
    has_closer: bool
    has_peak: bool


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
    # Genre-mode anchor candidates from `score_anchors` (#19). Top-scoring core-pool
    # tracks Stage 2 is nudged to prefer as identity-defining picks. Signal only —
    # not enforced. Empty when no track clears the threshold (or for legacy canvases).
    core_anchor_ids: list[str] = field(default_factory=list)
    # Era and label canvas dimensions (#20). ``era_window`` is (min_year, max_year)
    # over core tracks with year data; None when coverage is too patchy. ``dominant_label``
    # is the most-frequent label name when its share clears the threshold; None otherwise.
    # ``label_share`` is the share of that label across labelled core tracks (0.0 when no
    # dominant label).
    era_window: tuple[int, int] | None = None
    dominant_label: str | None = None
    label_share: float = 0.0
    # Bridge/wildcard tracks flagged as structurally exceptional (#10). Surfaces in
    # the Stage 2 canvas header so the model can deliberately use them in structural
    # roles. Empty when no off-core track crosses the multi-signal threshold.
    concept_anchor_candidates: list[ConceptAnchorCandidate] = field(default_factory=list)
    # Concept-direction metadata (#53). When ``brief`` is non-empty this canvas was
    # materialised from a cross-strata creative direction (mood journey, era dialogue,
    # etc.) rather than a BPM-stratum shortlist. ``brief`` is the DIRECTION BRIEF text
    # rendered ahead of the canvas block in the Stage 2 prompt; ``direction_type`` names
    # the direction family; ``thread_artist`` scopes the artist-thread validator
    # suppression (empty for every other direction type and for classic canvases).
    brief: str = ""
    direction_type: str = ""
    thread_artist: str = ""


SeedTier = Literal["anchor", "supporting", "optional"]
# Stage 2 set-role vocabulary. Sharpened from 19 → 10 roles in v0.11.0 (#23) to
# remove semantic overlap (world-setter/early-hook subsumed by hook+opener;
# groove-locker/builder/connector merged into groove; pressure/lift consolidated;
# weapon merged into peak; post-peak/cleanser merged into resolution; risk merged
# into pivot; utility removed). Old role strings emitted by upstream LLM responses
# are coerced to "unknown" by the parser.
SetRole = Literal[
    "opener",
    "groove",
    "hook",
    "pivot",
    "lift",
    "vocal_moment",
    "texture_change",
    "peak",
    "resolution",
    "closer",
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
