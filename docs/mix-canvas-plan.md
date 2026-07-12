# Plan: Mix Canvas — Diversity-Aware Stage 1 Selection

## Context

MixLab's current `select_shortlists_for_stage2()` (`llm.py:654`) picks up to 6 shortlists by random-sampling the top-12 by pool size. This means Stage 2 Sonnet repeatedly receives large but technically similar pools with no role candidates, no contrast metadata, no risk notes, and no novelty signal. Repeated runs for the same genre hand Stage 2 nearly identical inputs.

Goal: replace random pool-size selection with deterministic, diversity-aware **Mix Canvas** selection. No new premium LLM calls. Stage 2 Sonnet keeps its current creative role.

---

## 1. Current Pipeline (Grounded in Code)

```
parse XML → filter denylist/range → fetch played tracks → filter unplayed
→ build_custom_genre_pool() OR partition_outliers() + resolve_genre_clusters()
→ filter_by_bpm()           ← hard ±6 BPM median gate (clustering.py:9, 148-153)
→ sort_by_camelot()
→ select_stage1_window()    ← random contiguous window (llm.py:641-651)
→ stage1_concepts()         ← free LLM cascade per chunk (llm.py:622-638)
→ select_shortlists_for_stage2()  ← random top-12→6 (llm.py:654-663)
→ stage2_curate_and_report()      ← single Anthropic call (llm.py:1397)
→ export XML → Discord
```

---

## 2. Key Files and Functions That Change

| File | Function | Change |
|------|----------|--------|
| `src/mixlab/models.py` | — | Add `MixCanvas` dataclass |
| `src/mixlab/clustering.py` | `filter_by_bpm()` | Augment: return three pools instead of hard-filtering |
| `src/mixlab/clustering.py` | new `build_mix_canvas()` | Build canvas from MixConcept + track pool |
| `src/mixlab/clustering.py` | new `score_canvas()` | Deterministic scoring |
| `src/mixlab/clustering.py` | new `select_canvases()` | Replace `select_shortlists_for_stage2` |
| `src/mixlab/history.py` | new module | Read/write `.mixlab/concept-history.json` |
| `src/mixlab/llm.py` | `select_shortlists_for_stage2()` | Remove; route through `select_canvases()` |
| `src/mixlab/llm.py` | `stage2_curate_and_report()` | Accept `list[MixCanvas]`; update prompt |
| `src/mixlab/llm.py` | new `validate_stage2_output()` | Warn-only post-Stage-2 validation |
| `src/mixlab/__main__.py` | `run()` | Wire canvas flow; append to history after Stage 2 |

`select_shortlists_for_playlist_stage2()` (`llm.py:666`) is out of scope — playlist mode has different seed-rank logic.

---

## 3. New Data Models (`src/mixlab/models.py`)

```python
@dataclass
class BpmPools:
    """Three-tier BPM partition of a track pool."""
    core: list[Track]      # |bpm - median| <= 6
    bridge: list[Track]    # 6 < |bpm - median| <= 12
    wildcard: list[Track]  # |bpm - median| > 12

@dataclass
class CanvasRoleCandidates:
    opener: list[str]        # track_ids
    groove_locker: list[str]
    builder: list[str]
    pivot: list[str]
    peak: list[str]
    closer: list[str]

@dataclass
class ContrastAssets:
    vocal_moments: list[str]     # track_ids
    texture_changes: list[str]
    darker_turns: list[str]
    brighter_lifts: list[str]
    lower_pressure_resets: list[str]

@dataclass
class CanvasScore:
    technical_viability: float   # 0–1
    role_coverage: float         # 0–1, fraction of 6 roles with ≥1 candidate
    anchor_strength: float       # 0–1, opener+closer candidate quality
    contrast_potential: float    # 0–1
    distinctiveness: float       # 0–1 vs other canvases in batch
    novelty: float               # 0–1 vs concept history
    overall: float               # weighted sum

@dataclass
class MixCanvas:
    canvas_id: str               # "{genre}_{bpm_center}_{camelot_zone}"
    genre: str
    bpm_range: tuple[float, float]
    dominant_bpm: float          # pool median
    dominant_camelot: str        # most common key in core pool
    core_track_ids: list[str]    # from BpmPools.core
    bridge_track_ids: list[str]
    wildcard_track_ids: list[str]
    roles: CanvasRoleCandidates
    contrast: ContrastAssets
    risk_notes: list[str]        # strings, e.g. "weak closer pool"
    score: CanvasScore
    source_concept: MixConcept   # original Stage 1 concept
```

---

## 4. Proposed Pipeline Changes

### 4a. Three-Pool BPM Partitioning (`clustering.py`)

Replace `filter_by_bpm()` call in `__main__.py` with a new function that keeps all tracks but labels them:

```python
def partition_bpm_pools(tracks: list[Track]) -> BpmPools:
    median = statistics.median(t.bpm for t in tracks)
    core, bridge, wildcard = [], [], []
    for t in tracks:
        delta = abs(t.bpm - median)
        if delta <= 6.0:
            core.append(t)
        elif delta <= 12.0:
            bridge.append(t)
        else:
            wildcard.append(t)
    return BpmPools(core=core, bridge=bridge, wildcard=wildcard)
```

`filter_by_bpm()` stays for backwards compat in custom genre hard-range filtering. `partition_bpm_pools()` replaces it in the standard genre pipeline in `__main__.py`.

### 4b. Canvas Building (`clustering.py`)

`build_mix_canvas(concept: MixConcept, tracks_by_id: dict[str, Track]) -> MixCanvas`

Split into three focused sub-functions to keep each small and independently testable:

```python
def _infer_roles(tracks: list[Track], pools: BpmPools) -> CanvasRoleCandidates: ...
def _detect_contrast(tracks: list[Track]) -> ContrastAssets: ...
def _generate_risk_notes(tracks: list[Track], pools: BpmPools, roles: CanvasRoleCandidates) -> list[str]: ...

def build_mix_canvas(concept: MixConcept, tracks_by_id: dict[str, Track]) -> MixCanvas:
    tracks = [tracks_by_id[tid] for tid in concept.track_ids if tid in tracks_by_id]
    pools = partition_bpm_pools(tracks)
    roles = _infer_roles(tracks, pools)
    contrast = _detect_contrast(tracks)
    risk_notes = _generate_risk_notes(tracks, pools, roles)
    return MixCanvas(
        canvas_id=_canvas_id(concept, tracks),
        source_concept=concept,
        ...
    )
```

`build_mix_canvas()` itself stays short — it delegates and assembles. Scores start at zero; `score_canvas()` fills them in separately.

### 4c. Diversity-Aware Canvas Selection (`clustering.py`)

`select_canvases(canvases: list[MixCanvas], history: ConceptHistory, n: int = 6) -> list[MixCanvas]`

1. Score all canvases via `score_canvas()`.
2. Sort by `overall` score descending.
3. When picking each canvas, apply overlap penalty to remaining candidates (tracks shared with already-picked canvases reduce their distinctiveness score).
4. Return top `n`.

This replaces `select_shortlists_for_stage2()`.

---

## 5. Deterministic Role Inference and Scoring

### Role Candidates (no LLM)

Infer from `Track` fields available: `bpm`, `camelot_key`, `energy` (0–8 or None), `label`, `artist`.

| Role | Heuristic |
|------|-----------|
| opener | Low-mid energy (≤3/8 if available), or first/last in Camelot sort chain, or in bridge pool |
| groove_locker | Core pool, energy 3–5/8, BPM within ±2 of pool median |
| builder | Core pool, energy 4–6/8, ascending energy vs groove_locker |
| pivot | Track in a Camelot key ≥3 steps from pool dominant; bridge pool preferred |
| peak | Core pool, highest energy candidates (6–8/8), or highest BPM in core |
| closer | Lower energy (≤4/8), harmonically stable (same or adjacent to opener's key) |

When `energy` is None for all tracks: fall back to BPM-position proxies only.

### Contrast Assets (deterministic)

| Asset | Detection |
|-------|-----------|
| vocal_moments | `"feat."` or `"ft."` in artist/title, or track tags contain "vocal" |
| texture_changes | Camelot key ≥3 steps from pool dominant key |
| darker_turns | Energy < pool energy median (if energy available) |
| brighter_lifts | Energy > pool energy median by ≥2 |
| lower_pressure_resets | Bridge/wildcard + lower energy than core median |

### Risk Notes (deterministic)

- `"weak closer pool"`: fewer than 2 closer candidates
- `"weak opener pool"`: fewer than 2 opener candidates
- `"excessive BPM spread"`: core pool BPM range > 10
- `"too-similar midsection"`: >60% of core pool within ±1 Camelot step of dominant
- `"over-repeated artist"`: any artist appears ≥3 times in core pool
- `"over-repeated label"`: any label appears ≥4 times in core pool
- `"all high energy"`: >75% of tracks have energy ≥6

### Canvas Scoring Weights

```
technical_viability = core_pool_size / 20  (capped at 1.0)
role_coverage       = roles_with_candidates / 6
anchor_strength     = (len(opener_candidates) + len(closer_candidates)) / 8  (capped at 1.0)
contrast_potential  = min(1.0, total_contrast_assets / 4)
distinctiveness     = 1.0 - (overlap_fraction_with_already_picked)
novelty             = 1.0 - max_similarity_to_history  (from ConceptHistory)

overall = (
    technical_viability * 0.25 +
    role_coverage       * 0.25 +
    anchor_strength     * 0.15 +
    contrast_potential  * 0.15 +
    distinctiveness     * 0.10 +
    novelty             * 0.10
)
```

`overlap_fraction` = `|shared_track_ids| / |canvas_track_ids|` against all already-selected canvases.

---

## 6. Concept History (`src/mixlab/history.py`)

New module. No external deps — stdlib `json` only.

**File path:** `.mixlab/concept-history.json` (relative to cwd). Created on first run; silently skipped if missing or corrupt.

**Schema:**

```json
{
  "runs": [
    {
      "run_id": "uuid4",
      "created_at": "2026-05-06T...",
      "mode": "standard",
      "genre": "drum_and_bass",
      "selected_canvas_ids": ["dnb_172.0_4A"],
      "dominant_bpm_clusters": [172.0],
      "dominant_camelot_keys": ["4A", "3A"],
      "core_track_ids": ["T001"],
      "anchor_track_ids": ["T003"],
      "opener_candidates": ["T001"],
      "closer_candidates": ["T012"],
      "concept_title": "Midnight Descent",
      "concept_track_ids": ["T003", "T007"],
      "energy_path": "single_arc",
      "mood": "dark, hypnotic",
      "rating": null
    }
  ]
}
```

**API:**

```python
def load_history(path: Path) -> ConceptHistory: ...
def append_run(history: ConceptHistory, entry: HistoryEntry, path: Path) -> None: ...
def similarity_to_history(canvas: MixCanvas, history: ConceptHistory, recency_window: int = 10) -> float: ...
```

`similarity_to_history()`: compare canvas track IDs against last `recency_window` entries. Jaccard similarity on track IDs. Use max across window as penalty. Older entries decay (multiply similarity by 0.8^age).

**Safety:** wrap all file I/O in try/except; log warning and return empty history on any error.

### How repeat runs become more diverse

Two sources of diversity compound:

**Source 1 — Stage 1 window is still random.** `select_stage1_window()` picks a random contiguous slice of the Camelot-sorted core pool. Different windows → Stage 1 free LLM generates different MixConcept pools. This is already true today but has no memory.

**Source 2 — Novelty penalty against history.** After each run, the selected canvases' track IDs are stored. On the next run, any canvas whose track IDs heavily overlap with a recent run gets a novelty score near 0.0, dragging its `overall` score down even if it has great role coverage. Canvases built from different Camelot zones or different BPM cluster centers have naturally low overlap → higher novelty → preferred.

**The decay means good concepts can return.** With `0.8^age` decay, a concept from 5 runs ago has ~33% of its original penalty. After 10 runs, it's ~11%. Popular BPM centers are not permanently blocked — they just need time off.

**What novelty scoring does NOT solve:** if the collection has only one meaningful BPM cluster in a genre, all canvases will have high overlap and novelty scores will be uniformly low. Canvas selection then falls back to role coverage and anchor strength — the best-structured pool still wins.

**Worst case:** no history file → all canvases get novelty = 1.0 → selection behaves identically to scoring without novelty, no regression.

---

## 7. Stage 2 Prompt Changes (`llm.py`)

`stage2_curate_and_report()` currently accepts `list[MixConcept]`. Change to accept `list[MixCanvas]`.

### Compact canvas format (token budget)

Canvas metadata uses track aliases only (T001 format). Full track metadata is already in the existing track listing — canvas is an index, not a duplicate. Keeps each canvas block to ~100–150 tokens, six canvases under +800 total tokens (~10–15% prompt increase).

```
[Canvas dnb_172.0_4A | novelty:0.82]
Core: T003 T007 T011 T014 T018 T022
Bridge: T031 T036 | Wildcard: T044
Opener: T003 T036 | Groove-locker: T007 T011 | Builder: T014 T018
Peak: T018 T022 | Pivot: T036 | Closer: T022 T044
Vocal: T014 | Texture: T036 | Darker: T011 T044 | Brighter: T018
Risks: weak closer pool, over-repeated label
```

Stage 2 system prompt additions:
- Core tracks: use freely
- Bridge tracks: must assign role that justifies BPM deviation (opener, pivot, reset, closer)
- Wildcard tracks: must state explicit creative reason; only if concept-defining
- If a canvas has risk notes, Stage 2 may still use it but must address noted weaknesses
- Not every mix needs every role
- Harmonic/BPM compatibility: helper, not boss
- Distinguish known data from inferred assumptions in transition notes

---

## 8. Post-Stage-2 Validation (`llm.py`)

New function: `validate_stage2_output(concepts: list[MixConcept], canvases: list[MixCanvas], tracks_by_id: dict[str, Track], played_ids: set[str], denylist_ids: set[str], all_tracks: bool) -> list[str]`

Returns list of warning strings (v1 warns only, does not reject).

Checks:
1. All selected track IDs exist in `tracks_by_id`
2. No denylist track IDs in any concept
3. No played track IDs (skipped if `all_tracks=True`)
4. Track count within genre target range (`config.TRACK_COUNT_TARGETS`)
5. First and last track in play order are plausible opener/closer
6. Bridge/wildcard tracks in a concept have a non-`unknown` role assigned in `Transition`
7. Artist appears ≥3 times: emit warning
8. BPM jump >15 between consecutive tracks: emit warning
9. Camelot jump >4 between consecutive tracks: emit warning
10. Energy declared as `single_arc` but final tracks have higher energy than openers: emit warning (if energy data exists)

Warnings appended to Discord report as a "⚠ Validation Notes" section.

---

## 9. Wiring in `__main__.py`

```python
# After stage1_concepts() returns:
pools = partition_bpm_pools(cluster)            # new
history = load_history(Path(".mixlab/concept-history.json"))  # new
canvases = [build_mix_canvas(c, tracks_by_id) for c in concepts]  # new
selected = select_canvases(canvases, history)   # replaces select_shortlists_for_stage2

# After stage2_curate_and_report():
warnings = validate_stage2_output(...)          # new
entry = HistoryEntry.from_run(selected, final_concepts, genre, mode)  # new
append_run(history, entry, Path(".mixlab/concept-history.json"))  # new
```

---

## 10. Migration / Backwards Compatibility

- `filter_by_bpm()` stays in `clustering.py` — still used by custom genre hard-range filter in `build_custom_genre_pool()`.
- `select_shortlists_for_stage2()` deprecated but not deleted in phase 1 — mark with comment.
- `select_shortlists_for_playlist_stage2()` unchanged (playlist mode out of scope).
- Stage 2 prompt is additive — canvas metadata sections added before existing track listing.
- `MixConcept` model unchanged. `MixCanvas.source_concept` holds the original concept for fallback.
- If canvas building produces zero canvases (edge case), fall back to old `select_shortlists_for_stage2()` behaviour with a logged warning.

---

## 11. Test Plan

### New tests in `tests/test_clustering.py`

- `test_partition_bpm_pools_splits_correctly`: tracks at ±4, ±8, ±15 from median → correct pools
- `test_partition_bpm_pools_empty_input`: empty list → empty BpmPools
- `test_build_mix_canvas_role_inference_with_energy`: high-energy tracks land in peak candidates
- `test_build_mix_canvas_role_inference_no_energy`: no energy field → falls back to BPM proxies
- `test_build_mix_canvas_contrast_assets_vocal`: track with "feat." in artist → vocal_moments
- `test_build_mix_canvas_risk_notes_weak_closer`: fewer than 2 closer candidates → risk note present
- `test_score_canvas_weights`: verify weighted sum matches expected formula
- `test_select_canvases_prefers_higher_score`: high-scoring canvas always selected
- `test_select_canvases_overlap_penalty`: two canvases sharing 80% tracks → second penalised
- `test_select_canvases_novelty_penalty`: canvas matching last run → lower novelty score

### New `tests/test_history.py`

- `test_load_history_missing_file`: returns empty history, no exception
- `test_load_history_corrupt_json`: returns empty history, no exception
- `test_append_run_creates_file`: first append creates `.mixlab/` and file
- `test_append_run_truncates_at_max`: history capped at 50 entries
- `test_similarity_to_history_identical`: returns 1.0 for exact track match
- `test_similarity_to_history_disjoint`: returns 0.0 for no shared tracks
- `test_similarity_to_history_decay`: older entries score lower than recent

### Updates to `tests/test_llm.py`

- Update `select_shortlists_for_stage2` tests → add canvas-selection equivalent
- Add `test_validate_stage2_output_missing_track_id`
- Add `test_validate_stage2_output_denylist_track`
- Add `test_validate_stage2_output_bpm_jump_warning`
- Stage 2 prompt test: verify canvas metadata sections present in constructed prompt

### Updates to `tests/test_clustering.py`

- Extend existing BPM filter tests to cover the three-pool boundary cases

---

## 12. Impact by Usage Mode

### Mode 1: Unplayed tracks (standard, primary target)

Full Mix Canvas pipeline applies.

- `partition_bpm_pools()` replaces `filter_by_bpm()` in standard genre path
- `build_mix_canvas()` wraps each Stage 1 concept
- `select_canvases()` replaces `select_shortlists_for_stage2()`
- History written after each successful run

Fallback: if canvas building produces zero canvases, revert to `select_shortlists_for_stage2()` with a logged warning.

### Mode 2: All tracks (`--all-tracks`)

Same pipeline as Mode 1, larger pool. Stage 1 still receives only core tracks, chunked at 40–60. Bridge/wildcard never sent to Stage 1 — stored as canvas metadata only. Larger core pool → more free LLM chunks, not more Anthropic calls. Canvas selection still caps at `n=6`. BPM median is more stable with larger pools.

Validation: played-track check skipped when `all_tracks=True`.

### Mode 3: Playlist completion (`run_playlist_mode()`)

**Unchanged. Out of scope.** Seed anchors define the concept; canvas diversity scoring would fight user intent. `select_shortlists_for_playlist_stage2()` stays as-is.

---

## 13. Risks and Trade-Offs

| Risk | Mitigation |
|------|------------|
| Role inference without energy data is coarse | Document limitation; MixedInKey energy often present |
| Canvas scoring weights are arbitrary | Reasonable defaults; tune without API changes |
| Concept history grows unbounded | Cap at 50 entries; prune oldest |
| History file absent or corrupt on first run | Graceful fallback: empty history, full novelty score |
| Stage 2 prompt grows with canvas metadata | Compact alias-only format; +10–15% tokens, not +40% |
| Playlist mode untouched | Explicitly out of scope for phase 1 |
| `build_mix_canvas()` becomes large function | Split into `_infer_roles`, `_detect_contrast`, `_generate_risk_notes` |

---

## 14. Implementation Phases

**Phase 1 — Foundation (models + BPM pools + history)**
- `models.py`: add `BpmPools`, `CanvasRoleCandidates`, `ContrastAssets`, `CanvasScore`, `MixCanvas`
- `clustering.py`: add `partition_bpm_pools()`
- `history.py`: new module with `load_history`, `append_run`, `similarity_to_history`
- Tests: `test_history.py`, `test_partition_bpm_pools_*`

**Phase 2 — Canvas builder and scoring**
- `clustering.py`: add `_infer_roles()`, `_detect_contrast()`, `_generate_risk_notes()`, `build_mix_canvas()`, `score_canvas()`, `select_canvases()`
- Tests: canvas build + select tests in `test_clustering.py`

**Phase 3 — Stage 2 integration**
- `llm.py`: update `stage2_curate_and_report()` signature and prompt
- `llm.py`: add `validate_stage2_output()`
- Tests: updated Stage 2 prompt tests + validation tests in `test_llm.py`

**Phase 4 — Wire up and history persistence**
- `__main__.py`: replace `select_shortlists_for_stage2`, add history read/write, add validation
- Full pipeline smoke test

---

## 15. Acceptance Criteria

- [ ] No new required Anthropic/Sonnet calls
- [ ] `select_shortlists_for_stage2()` fully replaced by `select_canvases()` in standard genre mode
- [ ] Canvas selection is deterministic given the same input (no `random.sample`)
- [ ] Repeated runs for same genre produce increasing novelty scores (penalised by history)
- [ ] `.mixlab/concept-history.json` written after each successful Stage 2 run
- [ ] Three BPM pools correctly partitioned with ±6/±12 boundaries
- [ ] Stage 2 prompt includes core/bridge/wildcard labels and role candidates in compact alias format
- [ ] Post-Stage-2 validation warns on denylist tracks, BPM jumps, artist repetition
- [ ] All new functions have corresponding unit tests
- [ ] `mypy --strict`, `ruff check`, and `pytest` all pass
- [ ] Playlist mode and custom genre mode unaffected
