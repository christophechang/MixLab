# Playlist Completion Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace conservative/bold variant scoring with a deterministic DJ Practicality Score, add `transitions` annotation to Stage 2 output, improve chapter detection, and update the report format to a compact per-track layout.

**Architecture:** Data models change first (`Transition`, `DJPracticalityScore`, updated `CompletionVariant`) since everything downstream depends on them. Pure algorithmic helpers (`camelot_distance`, `_compute_practicality_score`) are built TDD before the orchestration code that calls them. LLM prompt string constants change last — they have no unit tests.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, mypy --strict, Ruff. All commands run via `.venv/bin/python -m <tool>`.

---

## File Map

| File | Role |
|---|---|
| `src/mixlab/models.py` | Add `Transition`, `DJPracticalityScore`; update `MixConcept`, `CompletionVariant` |
| `src/mixlab/clustering.py` | Add `camelot_distance()` |
| `src/mixlab/playlist_mode.py` | Update `compute_deterministic_intent` for chapter detection |
| `src/mixlab/llm.py` | Add `_pair_consecutive`, `_compute_practicality_score`, `_passes_floor`; update `_score_variant`, `_select_best_variant`, selection flow, rejected summary, `_parse_curated_concepts`, `_rewrite_playlist_report`, prompt constants |
| `tests/test_clustering.py` | New tests for `camelot_distance` |
| `tests/test_playlist_mode.py` | New test for chapter detection |
| `tests/test_intent.py` | Update existing `CompletionVariant` constructions; new tests for scoring, selection, pre-filter |
| `tests/test_llm.py` | New tests for transitions parsing, report rewriting |

---

## Task 1: Data Models

**Files:**
- Modify: `src/mixlab/models.py`

- [ ] **Step 1: Replace the full contents of `models.py`**

```python
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
```

- [ ] **Step 2: Run type-checker and linter to confirm models are valid**

```bash
.venv/bin/python -m mypy src/mixlab/models.py
.venv/bin/python -m ruff check src/mixlab/models.py
```

Expected: no errors.

- [ ] **Step 3: Run tests to identify what is now broken**

```bash
.venv/bin/python -m pytest tests/test_intent.py -v 2>&1 | head -40
```

Expected: failures in `test_select_best_variant_*` because `CompletionVariant` no longer takes `role_coverage`.

- [ ] **Step 4: Commit the model changes**

```bash
git add src/mixlab/models.py
git commit -m "feat(models): add Transition, DJPracticalityScore; update CompletionVariant"
```

---

## Task 2: Fix Broken Tests in test_intent.py

**Files:**
- Modify: `tests/test_intent.py`

The two `_select_best_variant` tests use the old `CompletionVariant` constructor with `role_coverage`. They need to be updated to use `practicality_score: DJPracticalityScore` instead, and the test logic needs to change because the new `score` is `practicality_score.overall`.

- [ ] **Step 1: Update the import and helper at the top of test_intent.py**

The file currently imports from `mixlab.models`:
```python
from mixlab.models import CompletionVariant, IntentBrief, MixConcept, SeedAnalysis, Track
```

Add `DJPracticalityScore`:
```python
from mixlab.models import CompletionVariant, DJPracticalityScore, IntentBrief, MixConcept, SeedAnalysis, Track
```

- [ ] **Step 2: Replace the two broken test functions**

Replace `test_select_best_variant_prefers_higher_anchor_retention` and `test_select_best_variant_single_variant_returned_as_is` with:

```python
def _make_practicality(overall: float) -> DJPracticalityScore:
    # Distribute `overall` across weights: 0.30 + 0.30 + 0.25 + 0.15 = 1.0
    # Simplest: set all components equal so overall == each component.
    return DJPracticalityScore(
        bpm_smoothness=overall,
        harmonic_ratio=overall,
        risk_justified=overall,
        fragment_preserved=overall,
    )


def test_select_best_variant_prefers_higher_practicality_score() -> None:
    concept_a = MixConcept(title="A", mood="practical", track_ids=["1", "2", "3"])
    concept_b = MixConcept(title="B", mood="balanced", track_ids=["1", "4", "5"])
    v_a = CompletionVariant(
        strategy="practical", concept=concept_a,
        anchor_retention_rate=1.0, practicality_score=_make_practicality(0.9),
    )
    v_b = CompletionVariant(
        strategy="balanced", concept=concept_b,
        anchor_retention_rate=0.5, practicality_score=_make_practicality(0.6),
    )
    best = _select_best_variant([v_a, v_b])
    assert best.strategy == "practical"


def test_select_best_variant_single_variant_returned_as_is() -> None:
    concept = MixConcept(title="T", mood="practical", track_ids=["1"])
    v = CompletionVariant(
        strategy="practical", concept=concept,
        anchor_retention_rate=1.0, practicality_score=_make_practicality(0.8),
    )
    assert _select_best_variant([v]) is v
```

- [ ] **Step 3: Run the tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_intent.py -v
```

Expected: all pass.

- [ ] **Step 4: Run full suite to confirm nothing else broke**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all pass (any existing `CompletionVariant` uses elsewhere are in `llm.py` internals not tested directly yet).

- [ ] **Step 5: Commit**

```bash
git add tests/test_intent.py
git commit -m "test(intent): update CompletionVariant fixtures for DJPracticalityScore"
```

---

## Task 3: camelot_distance in clustering.py

**Files:**
- Modify: `src/mixlab/clustering.py`
- Modify: `tests/test_clustering.py`

`_CAMELOT_RE` already exists in `clustering.py` at line 12: `re.compile(r"^(\d{1,2})([AB])$", re.IGNORECASE)`. The new function uses it directly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_clustering.py`:

```python
from mixlab.clustering import camelot_distance


def test_camelot_distance_identical_keys_returns_zero() -> None:
    assert camelot_distance("8A", "8A") == 0


def test_camelot_distance_adjacent_same_ring_returns_one() -> None:
    assert camelot_distance("8A", "9A") == 1


def test_camelot_distance_adjacent_wraps_twelve_to_one() -> None:
    assert camelot_distance("12A", "1A") == 1


def test_camelot_distance_same_number_opposite_ring_returns_one() -> None:
    assert camelot_distance("8A", "8B") == 1


def test_camelot_distance_cross_ring_two_steps_apart() -> None:
    # 8A to 9B: ring_dist=1, cross-ring adds 1 → 2
    assert camelot_distance("8A", "9B") == 2


def test_camelot_distance_unparseable_returns_999() -> None:
    assert camelot_distance("X", "8A") == 999
    assert camelot_distance("8A", "") == 999


def test_camelot_distance_large_gap_same_ring() -> None:
    # 1A to 7A: min(6, 12-6) = 6
    assert camelot_distance("1A", "7A") == 6


def test_camelot_distance_is_symmetric() -> None:
    assert camelot_distance("3B", "9A") == camelot_distance("9A", "3B")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_clustering.py -k "camelot_distance" -v
```

Expected: `ImportError` or `AttributeError` — `camelot_distance` does not exist yet.

- [ ] **Step 3: Add camelot_distance to clustering.py**

Insert after the existing `_camelot_compatible` function (around line 40):

```python
def camelot_distance(key_a: str, key_b: str) -> int:
    """Return minimum Camelot wheel steps between two keys (0 = identical, 1 = adjacent).

    Adjacent = ±1 same ring (wraps 12↔1), or same number opposite ring.
    Returns 999 if either key is unparseable.
    """
    ma = _CAMELOT_RE.match(key_a)
    mb = _CAMELOT_RE.match(key_b)
    if not ma or not mb:
        return 999
    num_a, mode_a = int(ma.group(1)), ma.group(2).upper()
    num_b, mode_b = int(mb.group(1)), mb.group(2).upper()
    if num_a == num_b and mode_a == mode_b:
        return 0
    if num_a == num_b:  # same number, opposite ring
        return 1
    ring_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if mode_a == mode_b:
        return ring_dist
    # Cross-ring: minimum path is same-ring distance + one ring crossing
    return ring_dist + 1
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_clustering.py -k "camelot_distance" -v
```

Expected: all 8 pass.

- [ ] **Step 5: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/clustering.py
.venv/bin/python -m ruff check src/mixlab/clustering.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/mixlab/clustering.py tests/test_clustering.py
git commit -m "feat(clustering): add camelot_distance"
```

---

## Task 4: Chapter Detection in playlist_mode.py

**Files:**
- Modify: `src/mixlab/playlist_mode.py`
- Modify: `tests/test_playlist_mode.py`

`compute_deterministic_intent` currently hard-codes `is_coherent_set=True` at line 201. The fix is one line: call `cluster_seed_zones` with `min_zone_tracks=1` and set `is_coherent_set = len(zones) <= 1`. The default `_ZONE_MIN_TRACKS = 3` would silently absorb small chapters; `min_zone_tracks=1` prevents any merging.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_playlist_mode.py`:

```python
def test_compute_deterministic_intent_chapter_split_sets_incoherent() -> None:
    """Seeds split by a 20 BPM gap should yield is_coherent_set=False."""
    tracks_by_id = {
        "a1": _make_track("a1", bpm=120.0),
        "a2": _make_track("a2", bpm=121.0),
        "a3": _make_track("a3", bpm=122.0),
        "b1": _make_track("b1", bpm=142.0),
        "b2": _make_track("b2", bpm=143.0),
    }
    brief = compute_deterministic_intent(["a1", "a2", "a3", "b1", "b2"], tracks_by_id)
    assert brief.is_coherent_set is False


def test_compute_deterministic_intent_tight_bpm_range_is_coherent() -> None:
    """Seeds within 10 BPM should yield is_coherent_set=True."""
    tracks_by_id = {
        str(i): _make_track(str(i), bpm=120.0 + i) for i in range(5)
    }
    brief = compute_deterministic_intent([str(i) for i in range(5)], tracks_by_id)
    assert brief.is_coherent_set is True


def test_compute_deterministic_intent_single_track_chapter_not_absorbed() -> None:
    """A single-track zone separated by >=12 BPM must produce 2 zones → incoherent."""
    tracks_by_id = {
        "lone": _make_track("lone", bpm=110.0),
        "c1": _make_track("c1", bpm=125.0),
        "c2": _make_track("c2", bpm=126.0),
        "c3": _make_track("c3", bpm=127.0),
    }
    brief = compute_deterministic_intent(["lone", "c1", "c2", "c3"], tracks_by_id)
    assert brief.is_coherent_set is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_playlist_mode.py -k "chapter" -v
```

Expected: first test fails (is_coherent_set is True when it should be False). Third test likely also fails.

- [ ] **Step 3: Update compute_deterministic_intent**

In `src/mixlab/playlist_mode.py`, find the line `is_coherent_set=True,` inside `compute_deterministic_intent` (around line 201) and replace the entire `return IntentBrief(...)` call's `is_coherent_set` value. First, add the zone calculation just before the `return`:

Replace the block that starts around line 197:
```python
    return IntentBrief(
        overall_vibe="Analysing...",
        energy_shape=energy_shape,
        risk_tolerance=risk_tolerance,
        is_coherent_set=True,
        seed_analyses=analyses,
```

with:

```python
    zones = cluster_seed_zones(tracks, min_zone_tracks=1)
    is_coherent_set = len(zones) <= 1

    return IntentBrief(
        overall_vibe="Analysing...",
        energy_shape=energy_shape,
        risk_tolerance=risk_tolerance,
        is_coherent_set=is_coherent_set,
        seed_analyses=analyses,
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_playlist_mode.py -k "chapter or coherent" -v
```

Expected: all 3 new tests pass.

- [ ] **Step 5: Run full playlist_mode suite to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/test_playlist_mode.py -v
```

Expected: all pass.

- [ ] **Step 6: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/playlist_mode.py
.venv/bin/python -m ruff check src/mixlab/playlist_mode.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/mixlab/playlist_mode.py tests/test_playlist_mode.py
git commit -m "feat(playlist_mode): detect chapters via cluster_seed_zones in compute_deterministic_intent"
```

---

## Task 5: _pair_consecutive and _compute_practicality_score in llm.py

**Files:**
- Modify: `src/mixlab/llm.py`
- Modify: `tests/test_intent.py`

`_compute_practicality_score` needs `statistics` (not currently imported in `llm.py`) and `camelot_distance` (from `clustering.py`). Both imports must be added. `Transition` and `DJPracticalityScore` must also be added to the `from mixlab.models import ...` block.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_intent.py`:

```python
from mixlab.llm import (
    _compute_practicality_score,  # noqa: PLC2701
    _minimum_playlist_seed_retention,
    _pair_consecutive,  # noqa: PLC2701
    _parse_intent_brief,
    _select_best_variant,
)
from mixlab.models import (
    AdjacencyFragment,
    CompletionVariant,
    DJPracticalityScore,
    IntentBrief,
    MixConcept,
    SeedAnalysis,
    Track,
    Transition,
)
```

(Replace the existing import block at the top of test_intent.py.)

Then add these test functions:

```python
def _make_track(track_id: str, *, bpm: float = 124.0, camelot_key: str = "8A", energy: int | None = None) -> Track:
    return Track(
        track_id=track_id,
        artist=f"Artist {track_id}",
        title=f"Title {track_id}",
        bpm=bpm,
        camelot_key=camelot_key,
        genre="House",
        energy=energy,
    )


def test_pair_consecutive_adjacent_returns_true() -> None:
    assert _pair_consecutive("a", "b", ["a", "b", "c"]) is True


def test_pair_consecutive_not_adjacent_returns_false() -> None:
    assert _pair_consecutive("a", "c", ["a", "b", "c"]) is False


def test_pair_consecutive_missing_id_returns_false() -> None:
    assert _pair_consecutive("x", "b", ["a", "b", "c"]) is False


def test_compute_practicality_score_perfect_bpm_smoothness() -> None:
    """All same BPM → stdev of deltas is 0 → bpm_smoothness=1.0."""
    concept = MixConcept(
        title="T", mood="practical",
        track_ids=["1", "2", "3", "4"],
    )
    tracks_by_id = {str(i): _make_track(str(i), bpm=124.0, camelot_key="8A") for i in range(1, 5)}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.bpm_smoothness == 1.0


def test_compute_practicality_score_perfect_harmonic_ratio() -> None:
    """All adjacent Camelot keys (distance ≤ 1) → harmonic_ratio=1.0."""
    concept = MixConcept(
        title="T", mood="practical",
        track_ids=["1", "2", "3"],
    )
    tracks_by_id = {
        "1": _make_track("1", camelot_key="8A"),
        "2": _make_track("2", camelot_key="9A"),
        "3": _make_track("3", camelot_key="9B"),
    }
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.harmonic_ratio == 1.0


def test_compute_practicality_score_zero_harmonic_ratio() -> None:
    """Keys far apart → harmonic_ratio=0.0."""
    concept = MixConcept(
        title="T", mood="practical",
        track_ids=["1", "2"],
    )
    tracks_by_id = {
        "1": _make_track("1", camelot_key="1A"),
        "2": _make_track("2", camelot_key="7B"),  # distance > 1
    }
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.harmonic_ratio == 0.0


def test_compute_practicality_score_no_transitions_gives_full_risk_score() -> None:
    """No Transition annotations → no annotated risks → risk_justified=1.0."""
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2"])
    tracks_by_id = {"1": _make_track("1"), "2": _make_track("2")}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.risk_justified == 1.0


def test_compute_practicality_score_cut_only_penalises_risk_score() -> None:
    """All transitions cut_only → risk_justified=0.0."""
    concept = MixConcept(
        title="T", mood="practical",
        track_ids=["1", "2", "3"],
        transitions=[
            Transition(from_id="1", to_id="2", is_risky=True, risk_type="cut_only"),
            Transition(from_id="2", to_id="3", is_risky=True, risk_type="cut_only"),
        ],
    )
    tracks_by_id = {str(i): _make_track(str(i)) for i in range(1, 4)}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.risk_justified == 0.0


def test_compute_practicality_score_named_risk_type_not_penalised() -> None:
    """chapter_pivot is a justified risk → risk_justified=1.0 (0 unjustified of 1 risky)."""
    concept = MixConcept(
        title="T", mood="practical",
        track_ids=["1", "2"],
        transitions=[
            Transition(from_id="1", to_id="2", is_risky=True, risk_type="chapter_pivot"),
        ],
    )
    tracks_by_id = {"1": _make_track("1"), "2": _make_track("2")}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.risk_justified == 1.0


def test_compute_practicality_score_fragment_preserved_with_adjacency() -> None:
    """Strong adjacency pair preserved in sequence → fragment_preserved=1.0."""
    frag = AdjacencyFragment(track_ids=["1", "2"], confidence=0.9, reason="bpm_close")
    brief = IntentBrief(
        overall_vibe="Test", energy_shape="unclear", risk_tolerance="medium",
        is_coherent_set=True, seed_analyses=[], missing_roles=[],
        strong_adjacencies=[frag], bpm_range=(120.0, 125.0),
    )
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2", "3"])
    tracks_by_id = {str(i): _make_track(str(i)) for i in range(1, 4)}
    score = _compute_practicality_score(concept, tracks_by_id, brief)
    assert score.fragment_preserved == 1.0


def test_compute_practicality_score_fragment_broken_reduces_score() -> None:
    """Strong adjacency pair broken (reversed) → fragment_preserved=0.0."""
    frag = AdjacencyFragment(track_ids=["1", "2"], confidence=0.9, reason="bpm_close")
    brief = IntentBrief(
        overall_vibe="Test", energy_shape="unclear", risk_tolerance="medium",
        is_coherent_set=True, seed_analyses=[], missing_roles=[],
        strong_adjacencies=[frag], bpm_range=(120.0, 125.0),
    )
    concept = MixConcept(title="T", mood="practical", track_ids=["2", "1", "3"])
    tracks_by_id = {str(i): _make_track(str(i)) for i in range(1, 4)}
    score = _compute_practicality_score(concept, tracks_by_id, brief)
    assert score.fragment_preserved == 0.0


def test_compute_practicality_score_overall_is_weighted_sum() -> None:
    """overall = bpm_smoothness*0.30 + harmonic_ratio*0.30 + risk_justified*0.25 + fragment_preserved*0.15."""
    s = DJPracticalityScore(bpm_smoothness=1.0, harmonic_ratio=0.5, risk_justified=0.8, fragment_preserved=0.6)
    expected = 1.0 * 0.30 + 0.5 * 0.30 + 0.8 * 0.25 + 0.6 * 0.15
    assert abs(s.overall - expected) < 1e-9
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_intent.py -k "pair_consecutive or practicality" -v 2>&1 | head -20
```

Expected: `ImportError` — functions not defined yet.

- [ ] **Step 3: Add imports to llm.py**

At the top of `src/mixlab/llm.py`, add `import statistics` to the stdlib block (after `import re`):

```python
import statistics
```

Extend the `from mixlab.models import (...)` block to include the new types:

```python
from mixlab.models import (
    CompletionVariant,
    DJPracticalityScore,
    IntentBrief,
    MixConcept,
    RiskTolerance,
    SeedAnalysis,
    SeedTier,
    SetRole,
    Track,
    Transition,
)
```

Add this import after the `mixlab.models` import:

```python
from mixlab.clustering import camelot_distance
```

- [ ] **Step 4: Add _pair_consecutive and _compute_practicality_score to llm.py**

Insert both functions directly before the existing `_score_variant` function (around line 871):

```python
def _pair_consecutive(a: str, b: str, ids: list[str]) -> bool:
    """Return True if b immediately follows a in ids."""
    for i in range(len(ids) - 1):
        if ids[i] == a and ids[i + 1] == b:
            return True
    return False


def _compute_practicality_score(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
    intent_brief: IntentBrief | None,
) -> DJPracticalityScore:
    """Compute a DJ Practicality Score deterministically from a concept's track sequence."""
    track_sequence = [tracks_by_id[tid] for tid in concept.track_ids if tid in tracks_by_id]
    n = len(track_sequence)

    # bpm_smoothness: how even the consecutive BPM steps are
    if n < 3:
        bpm_smoothness = 1.0
    else:
        bpm_deltas = [abs(track_sequence[i + 1].bpm - track_sequence[i].bpm) for i in range(n - 1)]
        std = statistics.stdev(bpm_deltas)
        bpm_smoothness = max(0.0, 1.0 - std / 10.0)

    # harmonic_ratio: fraction of consecutive pairs that are Camelot-compatible (distance ≤ 1)
    if n < 2:
        harmonic_ratio = 1.0
    else:
        total_pairs = n - 1
        compatible = sum(
            1 for i in range(total_pairs)
            if camelot_distance(track_sequence[i].camelot_key, track_sequence[i + 1].camelot_key) <= 1
        )
        harmonic_ratio = compatible / total_pairs

    # risk_justified: penalise transitions annotated is_risky=True with risk_type "cut_only" or ""
    risky = [t for t in concept.transitions if t.is_risky]
    unjustified = [t for t in risky if t.risk_type in ("cut_only", "")]
    if not risky:
        risk_justified = 1.0
    else:
        risk_justified = max(0.0, 1.0 - len(unjustified) / len(risky))

    # fragment_preserved: fraction of strong adjacency pairs preserved in sequence
    if intent_brief is None or not intent_brief.strong_adjacencies:
        fragment_preserved = 1.0
    else:
        concept_ids = list(concept.track_ids)
        preserved = sum(
            1
            for frag in intent_brief.strong_adjacencies
            if _pair_consecutive(frag.track_ids[0], frag.track_ids[1], concept_ids)
        )
        fragment_preserved = preserved / len(intent_brief.strong_adjacencies)

    return DJPracticalityScore(
        bpm_smoothness=bpm_smoothness,
        harmonic_ratio=harmonic_ratio,
        risk_justified=risk_justified,
        fragment_preserved=fragment_preserved,
    )
```

- [ ] **Step 5: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_intent.py -k "pair_consecutive or practicality" -v
```

Expected: all pass.

- [ ] **Step 6: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/llm.py
.venv/bin/python -m ruff check src/mixlab/llm.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/mixlab/llm.py tests/test_intent.py
git commit -m "feat(llm): add _pair_consecutive and _compute_practicality_score"
```

---

## Task 6: Update _score_variant, _select_best_variant, _passes_floor, and Selection Flow

**Files:**
- Modify: `src/mixlab/llm.py`
- Modify: `tests/test_intent.py`

This task replaces the current selection logic. `_score_variant` drops `role_coverage` and calls `_compute_practicality_score`. `_select_best_variant` uses a strategy priority tiebreak. `_passes_floor` is a new module-level function. The selection flow in the main `run_playlist_mode` function changes from select-then-check to pre-filter-then-select.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_intent.py`:

```python
from mixlab.llm import (
    _compute_practicality_score,  # noqa: PLC2701
    _minimum_playlist_seed_retention,
    _pair_consecutive,  # noqa: PLC2701
    _passes_floor,  # noqa: PLC2701
    _parse_intent_brief,
    _score_variant,  # noqa: PLC2701
    _select_best_variant,
)
```

(Update the existing import line to include `_passes_floor` and `_score_variant`.)

Add these tests:

```python
def test_select_best_variant_tiebreak_prefers_practical() -> None:
    """When practicality scores are equal, practical > balanced > adventurous."""
    concepts = [
        MixConcept(title="P", mood="practical", track_ids=["1"]),
        MixConcept(title="B", mood="balanced", track_ids=["2"]),
        MixConcept(title="A", mood="adventurous", track_ids=["3"]),
    ]
    variants = [
        CompletionVariant(strategy="adventurous", concept=concepts[2],
                          anchor_retention_rate=1.0, practicality_score=_make_practicality(0.7)),
        CompletionVariant(strategy="balanced", concept=concepts[1],
                          anchor_retention_rate=1.0, practicality_score=_make_practicality(0.7)),
        CompletionVariant(strategy="practical", concept=concepts[0],
                          anchor_retention_rate=1.0, practicality_score=_make_practicality(0.7)),
    ]
    best = _select_best_variant(variants)
    assert best.strategy == "practical"


def test_passes_floor_with_intent_brief_per_tier() -> None:
    """Variant retaining all anchors and enough supporting passes."""
    brief = _make_brief_with_tiers(anchor_ids=["a1", "a2", "a3", "a4"], supporting_ids=["s1", "s2"])
    # floor: ceil(4*0.75)=3 anchors, ceil(2*0.40)=1 supporting
    concept_pass = MixConcept(title="P", mood="practical", track_ids=["a1", "a2", "a3", "s1", "x1"])
    concept_fail = MixConcept(title="F", mood="practical", track_ids=["a1", "s1", "s2", "x1", "x2"])
    v_pass = CompletionVariant(strategy="practical", concept=concept_pass,
                               anchor_retention_rate=0.75, practicality_score=_make_practicality(0.8))
    v_fail = CompletionVariant(strategy="practical", concept=concept_fail,
                               anchor_retention_rate=0.25, practicality_score=_make_practicality(0.9))
    seed_ids = ["a1", "a2", "a3", "a4", "s1", "s2"]
    min_seeds = _minimum_playlist_seed_retention(len(seed_ids), brief)
    assert _passes_floor(v_pass, brief, seed_ids, min_seeds) is True
    assert _passes_floor(v_fail, brief, seed_ids, min_seeds) is False


def test_passes_floor_optional_seeds_do_not_satisfy_anchor_requirement() -> None:
    """A variant with many optional seeds but too few anchors must fail the floor."""
    brief = _make_brief_with_tiers(anchor_ids=["a1", "a2", "a3", "a4"], supporting_ids=[])
    # floor: ceil(4*0.75)=3 anchors
    # concept keeps only 2 anchors but 10 optional seeds — total retained > floor sum, but per-tier fails
    opt_ids = [f"o{i}" for i in range(10)]
    concept = MixConcept(title="T", mood="practical", track_ids=["a1", "a2"] + opt_ids)
    v = CompletionVariant(strategy="practical", concept=concept,
                          anchor_retention_rate=0.5, practicality_score=_make_practicality(0.9))
    seed_ids = ["a1", "a2", "a3", "a4"]
    min_seeds = _minimum_playlist_seed_retention(len(seed_ids), brief)
    assert _passes_floor(v, brief, seed_ids, min_seeds) is False


def test_score_variant_returns_completion_variant_with_practicality_score() -> None:
    from mixlab.llm import _score_variant  # noqa: PLC2701
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2", "3"])
    tracks_by_id = {str(i): _make_track(str(i), bpm=124.0, camelot_key="8A") for i in range(1, 4)}
    variant = _score_variant(concept, ["1", "2"], None, tracks_by_id)
    assert variant.strategy == "practical"
    assert isinstance(variant.practicality_score, DJPracticalityScore)
    assert 0.0 <= variant.score <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_intent.py -k "tiebreak or passes_floor or score_variant" -v 2>&1 | head -30
```

Expected: failures/errors because `_passes_floor` doesn't exist, `_score_variant` still uses old signature, `_select_best_variant` has no tiebreak.

- [ ] **Step 3: Replace _score_variant in llm.py**

Find `_score_variant` (around line 871) and replace the entire function:

```python
def _score_variant(
    concept: MixConcept,
    seed_track_ids: list[str],
    intent_brief: IntentBrief | None,
    tracks_by_id: dict[str, Track],
) -> CompletionVariant:
    """Compute a CompletionVariant with anchor_retention_rate and practicality_score."""
    concept_id_set = set(concept.track_ids)
    strategy_raw = concept.mood.lower()
    strategy: Literal["practical", "balanced", "adventurous"] = (
        "practical" if strategy_raw == "practical"
        else "balanced" if strategy_raw == "balanced"
        else "adventurous" if strategy_raw == "adventurous"
        else "practical"
    )

    if intent_brief is not None and intent_brief.anchor_ids:
        retained_anchors = sum(1 for aid in intent_brief.anchor_ids if aid in concept_id_set)
        anchor_retention_rate = retained_anchors / len(intent_brief.anchor_ids)
    else:
        retained_seeds = sum(1 for sid in seed_track_ids if sid in concept_id_set)
        anchor_retention_rate = retained_seeds / max(len(seed_track_ids), 1)

    practicality_score = _compute_practicality_score(concept, tracks_by_id, intent_brief)

    return CompletionVariant(
        strategy=strategy,
        concept=concept,
        anchor_retention_rate=anchor_retention_rate,
        practicality_score=practicality_score,
    )
```

- [ ] **Step 4: Replace _select_best_variant in llm.py**

Find `_select_best_variant` (around line 912) and replace:

```python
_STRATEGY_PRIORITY: dict[str, int] = {"practical": 0, "balanced": 1, "adventurous": 2}


def _select_best_variant(variants: list[CompletionVariant]) -> CompletionVariant:
    """Return highest-scoring variant; ties broken by practical > balanced > adventurous."""
    return max(
        variants,
        key=lambda v: (v.score, -_STRATEGY_PRIORITY.get(v.strategy, 99)),
    )
```

- [ ] **Step 5: Add _passes_floor to llm.py**

Insert immediately after `_select_best_variant`:

```python
def _passes_floor(
    variant: CompletionVariant,
    intent_brief: IntentBrief | None,
    playlist_seed_track_ids: list[str],
    minimum_seed_tracks: int,
) -> bool:
    """Return True if variant meets the per-tier retention floor."""
    concept_ids = set(variant.concept.track_ids)
    if intent_brief is not None and intent_brief.anchor_ids:
        anchor_floor = math.ceil(len(intent_brief.anchor_ids) * 0.75)
        supporting_floor = math.ceil(len(intent_brief.supporting_ids) * 0.40)
        return (
            sum(1 for aid in intent_brief.anchor_ids if aid in concept_ids) >= anchor_floor
            and sum(1 for sid in intent_brief.supporting_ids if sid in concept_ids) >= supporting_floor
        )
    retained = sum(1 for tid in playlist_seed_track_ids if tid in concept_ids)
    return retained >= minimum_seed_tracks
```

- [ ] **Step 6: Replace the selection flow in the main playlist orchestration**

Find the block starting at `# Score and select best variant from up to 2 returned concepts` (around line 1167). Replace everything from that comment up to and including the existing `# Summarise rejected variant in report` block (around line 1208):

```python
        # Score all returned concepts
        variants = [_score_variant(c, playlist_seed_track_ids, intent_brief, tracks_by_id) for c in curated]

        # Pre-filter: per-tier retention floor before selection
        passing = [v for v in variants if _passes_floor(v, intent_brief, playlist_seed_track_ids, minimum_seed_tracks)]
        candidates = passing if passing else variants

        best = _select_best_variant(candidates)
        concept = best.concept

        # Retry only if no variant passed the floor
        if not passing:
            retained_ids, dropped_ids, _ = _playlist_retention_stats(concept, playlist_seed_track_ids)
            dropped_labels = _format_playlist_track_labels(dropped_ids, tracks_by_id, limit=len(dropped_ids))
            retry_prompt = (
                f"Your previous attempt retained only {len(retained_ids)} seed tracks "
                f"(minimum required: {minimum_seed_tracks}).\n"
                f"Dropped seeds: {dropped_labels}.\n"
                "Retry with one concept only. Include as many dropped seeds as possible. "
                "For each one still excluded, give one sentence of musical justification.\n\n"
            ) + prompt
            raw, stage2_model_display = await _call_stage2_raw(
                retry_prompt, stage2_system, stage2_key, use_minimax, stage2_model_display
            )
            curated, report = _parse_curated_concepts(raw, valid_ids)
            if curated:
                concept = curated[0]
                retained_ids, _, _ = _playlist_retention_stats(concept, playlist_seed_track_ids)
            else:
                retained_ids = []
            if len(retained_ids) < minimum_seed_tracks:
                raise RuntimeError(
                    "Stage 2 playlist curation retained too few seed tracks after retry: "
                    f"{len(retained_ids)} retained, minimum acceptable {minimum_seed_tracks}."
                )
            variants = []  # suppress rejected_summary on retry path

        # Summarise rejected variants in report
        rejected_summary = ""
        if variants:
            rejected = [v for v in variants if v.concept is not concept]
            if rejected:
                parts = [
                    f"{v.strategy} (practicality: {v.practicality_score.overall:.2f}, "
                    f"anchor retention: {v.anchor_retention_rate:.0%}) — not selected"
                    for v in rejected
                ]
                rejected_summary = "\nAlternative strategies considered: " + "; ".join(parts) + "."
```

- [ ] **Step 7: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_intent.py -v
```

Expected: all pass.

- [ ] **Step 8: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/llm.py
.venv/bin/python -m ruff check src/mixlab/llm.py
```

Expected: no errors.

- [ ] **Step 9: Run full test suite**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/mixlab/llm.py tests/test_intent.py
git commit -m "feat(llm): replace role_coverage scoring with DJPracticalityScore; add _passes_floor pre-filter"
```

---

## Task 7: Transitions Parsing in _parse_curated_concepts

**Files:**
- Modify: `src/mixlab/llm.py`
- Modify: `tests/test_llm.py`

`_parse_curated_concepts` currently builds `MixConcept(title=..., mood=..., track_ids=...)`. It needs to also parse `transitions` from each item dict. IDs are NOT validated at parse time — unmatched entries are simply never looked up during scoring.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm.py`:

```python
def test_parse_curated_concepts_parses_transitions() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps([{
        "title": "T", "mood": "practical",
        "track_ids": ["1", "2", "3", "4"],
        "transitions": [
            {"from_id": "1", "to_id": "2", "is_risky": False, "risk_type": ""},
            {"from_id": "2", "to_id": "3", "is_risky": True, "risk_type": "chapter_pivot"},
        ],
        "report": "x",
    }])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert len(concepts[0].transitions) == 2
    assert concepts[0].transitions[1].is_risky is True
    assert concepts[0].transitions[1].risk_type == "chapter_pivot"


def test_parse_curated_concepts_missing_transitions_key_yields_empty_list() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps([{"title": "T", "mood": "m", "track_ids": ["1", "2", "3", "4"], "report": "x"}])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert concepts[0].transitions == []


def test_parse_curated_concepts_unmatched_transition_ids_stored_as_is() -> None:
    """Transition IDs not in track_ids are stored without filtering — ignored at scoring time."""
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps([{
        "title": "T", "mood": "m",
        "track_ids": ["1", "2", "3", "4"],
        "transitions": [
            {"from_id": "99", "to_id": "100", "is_risky": True, "risk_type": "cut_only"},
        ],
        "report": "x",
    }])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    # stored verbatim — scorer ignores them when looking up consecutive pairs
    assert len(concepts[0].transitions) == 1
    assert concepts[0].transitions[0].from_id == "99"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "transitions" -v
```

Expected: first test fails — `concepts[0].transitions` is empty list (not parsed yet).

- [ ] **Step 3: Update _parse_curated_concepts in llm.py**

Find the `curated.append(MixConcept(...))` call inside `_parse_curated_concepts` (around line 835) and replace:

```python
        raw_transitions = item.get("transitions", [])
        transitions: list[Transition] = []
        for tr in (raw_transitions if isinstance(raw_transitions, list) else []):
            if isinstance(tr, dict):
                transitions.append(
                    Transition(
                        from_id=str(tr.get("from_id", "")),
                        to_id=str(tr.get("to_id", "")),
                        is_risky=bool(tr.get("is_risky", False)),
                        risk_type=str(tr.get("risk_type", "")),
                    )
                )
        curated.append(
            MixConcept(
                title=str(item.get("title", "")),
                mood=str(item.get("mood", "")),
                track_ids=track_ids,
                transitions=transitions,
            )
        )
```

(This replaces the current single-line `curated.append(MixConcept(...))` call.)

- [ ] **Step 4: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "transitions" -v
```

Expected: all 3 pass.

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all pass.

- [ ] **Step 6: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/llm.py
.venv/bin/python -m ruff check src/mixlab/llm.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/mixlab/llm.py tests/test_llm.py
git commit -m "feat(llm): parse transitions from Stage 2 JSON in _parse_curated_concepts"
```

---

## Task 8: _rewrite_playlist_report Marker Update

**Files:**
- Modify: `src/mixlab/llm.py`
- Modify: `tests/test_llm.py`

The marker string `"Track order (Camelot / BPM):"` must change to `"Track order:"` to match the new compact report format the LLM will produce.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm.py`:

```python
def test_rewrite_playlist_report_injects_summary_at_track_order_marker() -> None:
    """With the new compact header 'Track order:', summary is injected before the track list."""
    from mixlab.llm import _rewrite_playlist_report

    concept = MixConcept(title="Set", mood="practical", track_ids=["1", "2", "5", "6"])
    tracks_by_id = {
        str(i): Track(
            track_id=str(i), artist=f"Artist {i}", title=f"Title {i}",
            bpm=120.0, camelot_key="8A", genre="House",
        )
        for i in range(1, 7)
    }
    report = (
        "CONCEPT: Set\n\nA driving set.\n\n"
        "Track order:\n"
        "1. Artist 1 — Title 1 [8A · 120.0] | Role: opener | Why: sets tone | Risk: none\n"
        "2. Artist 2 — Title 2 [8A · 120.0] | Role: builder | Why: builds | Risk: none"
    )
    rewritten = _rewrite_playlist_report(report, "Monday", concept, ["1", "2", "3", "4"], tracks_by_id)

    assert "Seed tracks retained: 2" in rewritten
    assert "Seed tracks dropped: 2." in rewritten
    # Summary must appear BEFORE the track list, not appended at end
    summary_pos = rewritten.index("Seed tracks retained")
    track_list_pos = rewritten.index("Track order:")
    assert summary_pos > track_list_pos  # summary is injected after the "Track order:" marker line
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "track_order_marker" -v
```

Expected: fails — marker not found, summary appended at end instead of injected.

- [ ] **Step 3: Update the marker in _rewrite_playlist_report**

In `src/mixlab/llm.py`, find (around line 950):

```python
    marker = "Track order (Camelot / BPM):"
```

Replace with:

```python
    marker = "Track order:"
```

- [ ] **Step 4: Run the new test**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "track_order_marker" -v
```

Expected: passes.

- [ ] **Step 5: Confirm the existing rewrite test still passes**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "rewrite_playlist" -v
```

Expected: both the old test (`test_rewrite_playlist_report_overwrites_incorrect_counts`) and the new one pass. Note: the old test uses the old marker string `"Track order (Camelot / BPM):"` in its report fixture — update it to `"Track order:"` so it continues to exercise the injection path.

Find in `tests/test_llm.py` the test `test_rewrite_playlist_report_overwrites_incorrect_counts` and update its report fixture:

```python
    report = (
        "CONCEPT: Set\n\nSome thesis.\n\n"
        "Track order:\n"
        "Artist 1 — Title 1 [8A · 120.0]"
    )
```

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all pass.

- [ ] **Step 7: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/llm.py
.venv/bin/python -m ruff check src/mixlab/llm.py
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/mixlab/llm.py tests/test_llm.py
git commit -m "feat(llm): update _rewrite_playlist_report marker to 'Track order:'"
```

---

## Task 9: Stage 2 Prompt Changes

**Files:**
- Modify: `src/mixlab/llm.py`

These are string-constant changes only. No new tests — the LLM prompt content is not unit-tested. Three changes: (1) JSON schema gains `transitions`, (2) report format becomes compact, (3) `_STAGE2_SYSTEM_PLAYLIST` switches from conservative/bold to practical/balanced/adventurous, and (4) the runtime prompt builder strings at lines 1123 and 1132 are updated.

- [ ] **Step 1: Update the JSON schema in _STAGE2_SYSTEM**

Find in `_STAGE2_SYSTEM` (around line 653):

```python
Your output must be a JSON array where each element has exactly this schema:
{
  "title": "...",
  "mood": "...",
  "track_ids": ["id1", "id2", ...],
  "report": "..."
}
```

Replace with:

```python
Your output must be a JSON array where each element has exactly this schema:
{
  "title": "...",
  "mood": "...",
  "track_ids": ["id1", "id2", ...],
  "transitions": [
    {"from_id": "id1", "to_id": "id2", "is_risky": false, "risk_type": ""},
    {"from_id": "id2", "to_id": "id3", "is_risky": true,  "risk_type": "chapter_pivot"}
  ],
  "report": "..."
}

transitions: one entry per consecutive pair in track_ids (len(track_ids) - 1 entries).
is_risky: true if the move is a notable harmonic or energy risk.
risk_type: one of "chapter_pivot" | "peak_impact" | "deliberate_reset" | "closer_move" \
           | "cut_only" | "low_tonal_risk" | "" (empty string when is_risky=false).
"cut_only" means: risky, with no mechanism that earns it — just a hard cut.
```

- [ ] **Step 2: Update the report format in _STAGE2_SYSTEM**

Find the block starting with `The "report" value must be a single string` (around line 663) through the `Assumptions:` bullet section (around line 685). Replace it with:

```python
The "report" value must be a single string (with \\n for line breaks) in this exact format:

CONCEPT: [title]

[1–2 sentences: thesis — what this set asks of the room.]

Track order:
[For each track in play order, one line:]
N. Artist — Title [Key · BPM] | Role: [role] | Why: [one short phrase] | Risk: [one short phrase or "none"]

Assumptions: [only if material — [unverified] tracks, vocal clash, tight blend window. One line each. Omit section if nothing material.]

Role options: opener, builder, pivot, peak, cleanser, closer, utility.
Risk: describe the transition risk into this track (not out of it). "none" if clean.
Why: why this track at this moment — one phrase, no full sentences needed.
```

Note: The entire multi-paragraph block (Arc, per-track narrative lines, Standout transitions, Assumptions bullets) is replaced by the compact single-line-per-track format above. Remove the old block including the lines starting with `Arc:`, `[One line per track...]`, `Standout transitions or calculated risks:`, and the `Assumptions:` bullet section.

- [ ] **Step 3: Update _STAGE2_SYSTEM_PLAYLIST second .replace() for three variants**

Find the second `.replace()` call in the `_STAGE2_SYSTEM_PLAYLIST` definition (around line 710). The `new_str` argument currently starts with `"""Produce EXACTLY TWO concepts...`. Replace that entire `new_str` argument with:

```python
    """Produce EXACTLY THREE concepts from the same pool using these strategies:

1. "practical" (mood = "practical"): maximise harmonic continuity. Prefer adjacent Camelot keys \
(distance ≤ 1). BPM moves ≤ 2 BPM per step where possible. Preserve all strong seed adjacency \
pairs. Protect anchors. Avoid unearned key jumps.

2. "balanced" (mood = "balanced"): one major key jump (distance 2–3) or single BPM arc allowed. \
Anchors protected. Optional seeds may be swapped when a library track clearly serves the arc \
better. Adjacency pairs are hints, not constraints.

3. "adventurous" (mood = "adventurous"): prioritise set narrative and role completeness. Chapter \
pivots and peak impacts are permitted when musically justified — name the mechanism. Anchors \
protected; adjacency pairs may be broken with one-sentence reason. Optional and supporting seeds \
replaceable if a library track serves the arc materially better.

Label each concept's mood field with exactly "practical", "balanced", or "adventurous".
All three must meet the anchor protection rules and the seed retention floor.
If the pool is too thin to produce even one strong set: [{"diagnostic": "..."}].""",
```

- [ ] **Step 4: Update the runtime prompt builder strings**

Find (around line 1122):

```python
        prompt = (
            f"{intent_section}"
            f"Curate two completion variants from the following {n} BPM zone shortlists. "
```

Replace with:

```python
        prompt = (
            f"{intent_section}"
            f"Curate three completion variants from the following {n} BPM zone shortlists. "
```

Find (around line 1132):

```python
            "Produce EXACTLY TWO concepts as described in your instructions (conservative + bold).\n\n"
```

Replace with:

```python
            "Produce EXACTLY THREE concepts as described in your instructions (practical / balanced / adventurous).\n\n"
```

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all pass (prompt strings are not directly tested; structure tests remain green).

- [ ] **Step 6: Type-check and lint**

```bash
.venv/bin/python -m mypy src/mixlab/llm.py
.venv/bin/python -m ruff check src/mixlab/llm.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/mixlab/llm.py
git commit -m "feat(llm): update Stage 2 prompts — transitions schema, compact report, three variants"
```

---

## Task 10: Final Verification

- [ ] **Step 1: Run the full test suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all pass, no skips.

- [ ] **Step 2: Full type-check across all changed modules**

```bash
.venv/bin/python -m mypy src/mixlab/models.py src/mixlab/clustering.py src/mixlab/playlist_mode.py src/mixlab/llm.py
```

Expected: no errors.

- [ ] **Step 3: Linter across all changed modules**

```bash
.venv/bin/python -m ruff check src/mixlab/ tests/
.venv/bin/python -m ruff format --check src/mixlab/ tests/
```

Expected: no errors.

- [ ] **Step 4: Commit if anything was auto-fixed by formatter**

```bash
.venv/bin/python -m ruff format src/mixlab/ tests/
git add -p
git commit -m "style: ruff format after playlist hardening"
```

Only commit if there are actual changes.
