# Design: Mix Canvas Scoring Diagnostics (`--debug` flag)

**Issue:** #1 — Add Mix Canvas scoring diagnostics (--debug flag)
**Date:** 2026-05-07
**Status:** Approved for implementation (v3 — post-review)

---

## Problem

Canvas selection is opaque. When a run produces unexpected output there is no way to know why a canvas was picked, what scores drove the decision, or what history penalty was applied. Repeated runs for the same genre can feel identical with no explanation.

---

## Goal

Add an optional `--debug` flag. When active:
- Shows per-canvas score breakdowns on stderr
- Shows why each canvas was selected or rejected (provable from data, not free-text guesses)
- Writes full structured data to `.mixlab/last-debug.json`
- Does not change stdout, Discord output, or which canvases are selected

---

## Decisions made

### Output destination: stderr + `.mixlab/last-debug.json`

Stderr gives live feedback during the run. The JSON file persists across runs — useful for diffing two runs and for Issue #8 (novelty debug output) which will extend it.

### Architecture: structured return value, not in-function I/O

`select_canvases()` returns `(list[MixCanvas], CanvasSelectionDebug | None)` — `None` when `debug=False`. Functions stay pure; all I/O (stderr print, JSON write) happens in `__main__.py`. This keeps clustering.py testable without output capture.

### `score_canvas()` signature: unchanged

Debug data is collected by `select_canvases()` from the scores it already computes during the selection loop. No change to `score_canvas()`.

### `llm.py`: no changes in this issue

Stage 2 already has `MIXLAB_DEBUG_STAGE2` env var. Debug threading into `stage2_curate_and_report()` is deferred to Issue #8 which specifically covers novelty debug output.

### `CanvasSelectionDebug` in `models.py`

New dataclass. Serialisable via `dataclasses.asdict`. Zero overhead when `debug=False` (not constructed).

### Honest rejection scores: debug-only final rescore for rejected only

**Review finding (v2):** The selection loop scores remaining canvases before each pick and stops when `n` is reached. A rejected canvas is never scored against the last-picked canvas, so its `distinctiveness` / `overlap_penalty` would be understated.

Fix: after the main selection loop completes, if `debug=True`, rescore all **rejected** canvases once more against the full `picked_core_ids`. Selected canvases keep the score from the round they were picked — rescoring them against `picked_core_ids` (which includes their own tracks) would incorrectly penalise them for self-overlap via `score_canvas()` line 456. Since rejected canvases are never in `picked_core_ids`, rescoring them is safe and honest. The rescore does not affect which canvases were selected — it is debug-only.

### `score` field semantics by status

- **selected** entry: `score` = score computed in the round the canvas was picked. `overlap_penalty` reflects overlap with the canvases already selected before it.
- **rejected** entry: `score` = score recomputed after selection completes against full `picked_core_ids`. `overlap_penalty` is honest against the complete selected set.

### Per-round rank tracking replaces free-text rejection reason

**Review finding (v2):** "score below selected threshold at all rounds" is not verifiable from a single final score. Instead, track `best_round_rank` (best rank achieved in any round, 1 = best) and `rounds_evaluated` (how many rounds the canvas was scored before being eliminated). The rejection reason can then be computed as a fact: "ranked {best_round_rank}/{total} best across {rounds_evaluated} round(s)".

### No-competition path (`len(canvases) <= n`): explicit field values

**Review finding (v3):** When all canvases fit without competitive selection, `best_round_rank` and `rounds_evaluated` must still be defined integers. Values: `selection_round=0` (signals no competition), `rounds_evaluated=1` (scored once for the initial sort), `best_round_rank=rank in the initial sort order (1-based)`.

---

## New types (models.py)

```python
@dataclass
class CanvasEntryDebug:
    canvas_id: str
    status: Literal["selected", "rejected"]
    # Round this canvas was selected (1-based).
    # None if rejected. 0 if all canvases fit with no competitive selection.
    selection_round: int | None
    core_count: int
    bridge_count: int
    wildcard_count: int
    risk_notes: list[str]
    # selected entries: score from the round this canvas was picked (no self-overlap issue).
    # rejected entries: score recomputed debug-only after selection against full picked_core_ids.
    score: CanvasScore
    novelty_penalty: float  # 1.0 - score.novelty
    overlap_penalty: float  # 1.0 - score.distinctiveness; for rejected = vs full selected set
    best_round_rank: int  # best rank achieved across all rounds (1 = ranked first)
    best_round_candidate_count: int  # candidate pool size in the round best_round_rank was achieved
    rounds_evaluated: int  # number of rounds this canvas was scored in
    # Human-readable, derived from data: e.g. "best rank 3/7 across 4 rounds"
    # Denominator is best_round_candidate_count (pool shrinks each round).
    rejection_reason: str  # "" if selected


@dataclass
class CanvasSelectionDebug:
    total_candidates: int
    selected_count: int
    entries: list[CanvasEntryDebug]
```

---

## Function signature changes (clustering.py)

```python
# Before
def select_canvases(
    canvases: list[MixCanvas],
    history: ConceptHistory,
    n: int = 6,
) -> list[MixCanvas]: ...


# After
def select_canvases(
    canvases: list[MixCanvas],
    history: ConceptHistory,
    n: int = 6,
    *,
    debug: bool = False,
) -> tuple[list[MixCanvas], CanvasSelectionDebug | None]: ...
```

**API break:** `select_canvases()` currently returns `list[MixCanvas]`. The return type change to a tuple breaks all callers and all tests that call it directly. `__main__.py` is the only production caller — it must unpack: `selected, debug_data = select_canvases(...)`. Tests in `test_clustering.py` that call `select_canvases()` directly must also be updated to unpack the tuple. This is manageable but must be treated as a breaking change, not a purely additive one.

---

## __main__.py changes

1. Add `--debug` argparse flag (store_true, default False)
2. Pass `debug=args.debug` into `run()` and down to `select_canvases()`
3. Unpack: `selected_canvases, debug_data = select_canvases(..., debug=debug)`
4. After selection, if `debug_data is not None`:
   - Print human-readable summary to stderr
   - Write full JSON to `.mixlab/last-debug.json` (create dir if missing)

---

## Stderr output format

All lines prefixed `[DEBUG]` — visually distinct from normal pipeline output.

```
[DEBUG] Canvas selection: 8 candidates → 6 selected

[DEBUG] SELECTED #1 (round 1): rolldark_172.0_9A
[DEBUG]   score=0.82  tv=0.85  rc=0.83  as=0.75  cp=0.50  d=1.00  n=0.90
[DEBUG]   pool: core=17  bridge=3  wildcard=1
[DEBUG]   risks: none
[DEBUG]   novelty_penalty=0.10  overlap_penalty=0.00  (best rank 1/8 in round 1)

[DEBUG] SELECTED #2 (round 2): deeproll_170.5_6B
[DEBUG]   score=0.78  tv=0.80  rc=0.83  as=0.75  cp=0.75  d=0.88  n=0.85
[DEBUG]   pool: core=16  bridge=2  wildcard=0
[DEBUG]   risks: over-repeated artist
[DEBUG]   novelty_penalty=0.15  overlap_penalty=0.12  (best rank 1/7 in round 2)

[DEBUG] REJECTED: lofiroll_168.0_3A  [score vs full selected set]
[DEBUG]   score=0.61  tv=0.60  rc=0.67  as=0.50  cp=0.25  d=0.72  n=0.95
[DEBUG]   pool: core=12  bridge=4  wildcard=2
[DEBUG]   risks: weak opener pool, too-similar midsection
[DEBUG]   novelty_penalty=0.05  overlap_penalty=0.28
[DEBUG]   reason: best rank 7/8 across 3 rounds

[DEBUG] Debug data written to .mixlab/last-debug.json
```

---

## JSON file format (`.mixlab/last-debug.json`)

```json
{
  "total_candidates": 8,
  "selected_count": 6,
  "entries": [
    {
      "canvas_id": "rolldark_172.0_9A",
      "status": "selected",
      "selection_round": 1,
      "core_count": 17,
      "bridge_count": 3,
      "wildcard_count": 1,
      "risk_notes": [],
      "score": {
        "technical_viability": 0.85,
        "role_coverage": 0.83,
        "anchor_strength": 0.75,
        "contrast_potential": 0.50,
        "distinctiveness": 1.0,
        "novelty": 0.90,
        "overall": 0.82
      },
      "novelty_penalty": 0.10,
      "overlap_penalty": 0.00,
      "best_round_rank": 1,
      "rounds_evaluated": 1,
      "rejection_reason": ""
    }
  ]
}
```

---

## Files changing

| File | Change |
|------|--------|
| `src/mixlab/models.py` | Add `CanvasEntryDebug`, `CanvasSelectionDebug` dataclasses |
| `src/mixlab/clustering.py` | Update `select_canvases()` signature and return type; collect `best_round_rank` + `rounds_evaluated` per canvas during selection loop; debug-only final rescore of rejected canvases against full `picked_core_ids` |
| `src/mixlab/__main__.py` | Add `--debug` flag; unpack new return value; print stderr summary; write JSON file |
| `tests/test_clustering.py` | Update all `select_canvases()` call sites to unpack tuple |

`llm.py` — no changes in this issue.

---

## Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_select_canvases_debug_returns_none_when_disabled` | `test_clustering.py` | `debug=False` → second element is `None` |
| `test_select_canvases_debug_structure` | `test_clustering.py` | `debug=True` → `CanvasSelectionDebug` with correct total/selected counts, all canvas IDs present, selected status correct |
| `test_select_canvases_debug_same_selection_as_normal` | `test_clustering.py` | Selection order identical with and without `debug=True` |
| `test_select_canvases_debug_rejected_rescored` | `test_clustering.py` | Rejected canvas `overlap_penalty` reflects full selected set (not partial); `rounds_evaluated` matches how many rounds the canvas was scored |
| `test_select_canvases_debug_best_round_rank` | `test_clustering.py` | `best_round_rank` is the minimum rank achieved across all rounds a canvas was evaluated |
| `test_main_debug_flag_writes_json` (integration) | `test_main.py` | `--debug` writes `.mixlab/last-debug.json`; valid JSON; expected shape |
| `test_main_no_debug_no_json` (integration) | `test_main.py` | No `--debug` → `.mixlab/last-debug.json` not created in `tmp_path` isolated cwd; test must not rely on absence of a pre-existing file |

---

## What is explicitly not changing

- `score_canvas()` signature and return type
- `stage2_curate_and_report()` — no changes
- Discord output — no debug data sent
- stdout — unchanged
- Which canvases are selected — identical to non-debug run
- Scoring weights or logic — separate issue (#9)

---

## Acceptance criteria (from issue)

- [x] Running with `--debug` shows per-canvas score breakdown
- [x] Normal run output (stdout + Discord) unchanged
- [x] No debug output sent to Discord
- [x] Debug output structured enough to understand selection reasoning at a glance
- [x] (Added) `.mixlab/last-debug.json` written on debug run; not written on normal run
- [x] (Added) Rejected canvas scores reflect full selected set — honest overlap penalty; selected scores use round-of-selection score (no self-overlap)
- [x] (Added) Rejection reason derivable from `best_round_rank` + `rounds_evaluated` — not free-text
- [x] (Added) No-competition path defines `rounds_evaluated=1`, `best_round_rank=rank in initial sort`
- [x] (Added) `test_main_no_debug_no_json` uses `tmp_path` isolated cwd

---

## Risk

Low to medium. Flag defaults to off — no production behaviour changes. However `select_canvases()` return type changes from `list[MixCanvas]` to `tuple[list[MixCanvas], CanvasSelectionDebug | None]`. This breaks existing callers and tests. All call sites must be updated. The debug-only rescore of rejected canvases calls `score_canvas()` extra times — negligible cost (pure Python, no I/O), but must be gated on `debug=True` to preserve zero-overhead normal runs.
