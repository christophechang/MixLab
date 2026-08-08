# Conjunction Mining + Direction Scoring Replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the saturated direction feasibility formula with a discriminating two-pass scorer, then add a deterministic conjunction miner that surfaces up to 3 "found" directions per pool.

**Architecture:** Stage 1 (Tasks 1–3) restructures `directions.py`: one shared candidate-enumeration function, gates split from score, per-builder identity renormalisation, two-pass score (rank/dedupe, then distinctiveness over the final field). Stage 2 (Tasks 4–8) adds `mining.py` (pure predicate extraction → cross-kind pair scan → lift gate → subsumption → shortlist) and wires mined candidates into the shared field. Spec: `docs/superpowers/specs/2026-08-08-conjunction-mining-design.md`.

**Tech Stack:** Python 3.12, pytest, stdlib only (no new dependencies).

## Global Constraints

- Branch: `claude/conjunction-mining` off `develop`. Conventional commits. **No `Co-Authored-By` trailers.**
- Determinism everywhere: no `random` in mining, every sort totally ordered with tie-breaks `(-lift, title, sorted(track_ids))`; same pool + same seed → byte-identical output.
- No LLM calls, no I/O in `mining.py`.
- Payload shape unchanged: `library_map._direction_entry` keys stay exactly `{direction_type, title, mood, brief, feasibility, track_ids}`.
- Titles and moods NEVER contain BPM numbers, key codes, or equipment terms (v1.13.2 rule). Brief *evidence* lines may cite BPM/key values (genre_traverse precedent). Mood strings ASCII-only (`x`, not `×`) — `_canvas_id` truncates mood to 8 chars.
- `feasibility` stays 0–1, rounded 4 decimals.
- Test suite: `pytest tests/ -q` must pass at the end of every task.
- Model routing (orchestrator note, not for implementers): T1/T2/T4/T6/T8 → Sonnet 5; T3/T5/T7 → Opus 5. Fable reviews every task diff.

---

### Task 1: Factor the shared candidate enumeration (pure refactor)

**Files:**
- Modify: `src/mixlab/directions.py` (lines 639–697: `enumerate_directions`, `generate_directions`)
- Test: `tests/test_directions.py` (existing tests are the harness — no new tests)

**Interfaces:**
- Produces: `_named_candidates(pool: list[Track], *, seed: int) -> list[Direction]` — runs all `_BUILDERS`, returns unsorted survivors. Both public functions call it. Task 3 extends this call site; Task 7 adds mined candidates beside it.

- [ ] **Step 1: Run the existing suite to establish green baseline**

Run: `pytest tests/test_directions.py -q`
Expected: PASS (record the count).

- [ ] **Step 2: Extract the duplicated builder loop**

In `directions.py`, both `enumerate_directions` (line 647) and `generate_directions` (line 673) contain:

```python
candidates = [direction for builder in _BUILDERS if (direction := builder(pool, seed=seed)) is not None]
```

Replace both with a call to one new module-level function directly above `enumerate_directions`:

```python
def _named_candidates(pool: list[Track], *, seed: int) -> list[Direction]:
    """Every surviving named-builder candidate, unsorted. Single enumeration
    point shared by the map path and the run path (previously duplicated)."""
    return [direction for builder in _BUILDERS if (direction := builder(pool, seed=seed)) is not None]
```

No other logic changes: `enumerate_directions` still sorts by `(-feasibility, direction_type)`; `generate_directions` still prints its diagnostic and rotates.

- [ ] **Step 3: Verify no behaviour change**

Run: `pytest tests/test_directions.py -q`
Expected: PASS, same count as Step 1.

- [ ] **Step 4: Commit**

```bash
git add src/mixlab/directions.py
git commit -m "refactor(directions): single shared builder enumeration"
```

---

### Task 2: Freshness metric + per-builder identity renormalisation

**Files:**
- Modify: `src/mixlab/directions.py` (`_build_mood_journey`, `_build_label_spotlight`, `_build_fresh_crate`, new helpers near `_balance`)
- Test: `tests/test_directions.py` (append)

**Interfaces:**
- Produces:
  - `_freshness(chosen: list[Track], pool: list[Track]) -> float` — median `date_added` count-percentile of `chosen` within `pool`; empty `date_added` sorts oldest; single-track pool → 0.5.
  - `_log_lift(ratio: float) -> float` — `min(log2(max(ratio, 1.0)) / 3.0, 1.0)`.
  - Builder signature change: `_build_label_spotlight(pool, *, seed, collection: list[Track] | None = None)`; `_named_candidates` gains and threads the same keyword (default `None`).
- Consumes: `_named_candidates` from Task 1.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_directions.py`:

```python
from mixlab.directions import _freshness, _log_lift, _balance


def _t(track_id: str, *, date_added: str = "", label: str = "", year: int | None = None,
       tags: list[str] | None = None, energy: int | None = None, bpm: float = 174.0,
       key: str = "8A") -> Track:
    return Track(track_id=track_id, artist=f"A{track_id}", title=f"T{track_id}",
                 bpm=bpm, camelot_key=key, genre="Drum & Bass", label=label,
                 year=year, tags=tags or [], energy=energy, date_added=date_added)


class TestFreshness:
    def test_newest_half_scores_above_oldest_half(self):
        pool = [_t(f"o{i}", date_added=f"2020-01-{i+1:02d}") for i in range(10)] + \
               [_t(f"n{i}", date_added=f"2026-06-{i+1:02d}") for i in range(10)]
        newest = [t for t in pool if t.track_id.startswith("n")]
        oldest = [t for t in pool if t.track_id.startswith("o")]
        assert _freshness(newest, pool) > 0.7
        assert _freshness(oldest, pool) < 0.3

    def test_missing_date_added_sorts_oldest(self):
        pool = [_t("u1"), _t("u2")] + [_t(f"d{i}", date_added=f"2026-01-{i+1:02d}") for i in range(8)]
        assert _freshness([pool[0], pool[1]], pool) < 0.2

    def test_all_unplayed_pool_does_not_saturate(self):
        # freshness is date-rank based, so an "all-unplayed" pool still spreads
        pool = [_t(f"x{i}", date_added=f"20{20+i//5}-01-01") for i in range(20)]
        vals = {_freshness([t], pool) for t in pool}
        assert len(vals) > 1


class TestLogLift:
    def test_anchor_points(self):
        assert _log_lift(1.0) == 0.0
        assert abs(_log_lift(2.0) - 1 / 3) < 1e-9
        assert abs(_log_lift(8.0) - 1.0) < 1e-9
        assert _log_lift(10.4) == 1.0     # live max saturates at the cap, not below it
        assert _log_lift(0.5) == 0.0      # sub-chance clamps to 0, never negative


class TestIdentityRenormalisation:
    def test_mood_journey_balance_uses_untruncated_pole_counts(self):
        # 40 dark vs 8 euphoric: old code truncated both to [:10] first → balance 0.8.
        # New: _balance(40, 8) = 0.2.
        pool = ([_t(f"d{i}", tags=["dark"], year=2020) for i in range(40)] +
                [_t(f"e{i}", tags=["euphoric"], year=2020) for i in range(8)] +
                [_t(f"b{i}", year=2020) for i in range(10)])
        d = _build_mood_journey(pool, seed=1)
        assert d is not None
        # signal lands in feasibility via the Task 3 scorer; here assert the builder's
        # stored signal directly (Task 2 exposes it — see Step 2)
        assert abs(d.identity - 0.2) < 1e-9

    def test_label_spotlight_collection_lift(self):
        pool = [_t(f"l{i}", label="Metalheadz") for i in range(10)] + \
               [_t(f"p{i}") for i in range(10)]
        collection = pool + [_t(f"c{i}") for i in range(80)]
        d = _build_label_spotlight(pool, seed=1, collection=collection)
        assert d is not None
        # share_in_pool 0.5, share_in_collection 0.1 → ratio 5 → log2(5)/3 ≈ 0.774
        assert abs(d.identity - math.log2(5) / 3) < 1e-6   # add `import math` at file top

    def test_label_spotlight_without_collection_falls_back_to_share(self):
        pool = [_t(f"l{i}", label="Metalheadz") for i in range(10)] + \
               [_t(f"p{i}") for i in range(10)]
        d = _build_label_spotlight(pool, seed=1)
        assert d is not None
        assert abs(d.identity - 0.5) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_directions.py -k "Freshness or LogLift or IdentityRenorm" -q`
Expected: FAIL — `ImportError: cannot import name '_freshness'`.

- [ ] **Step 3: Implement**

In `directions.py`:

(a) Add `identity: float = 0.0` and `freshness: float = 0.0` fields to the `Direction` dataclass (after `thread_artist`). They are internal scoring carriers; `library_map._direction_entry` selects its keys explicitly, so the payload is unaffected.

(b) New helpers after `_balance`:

```python
import math


def _freshness(chosen: list[Track], pool: list[Track]) -> float:
    """Median date_added count-percentile of ``chosen`` within ``pool``.

    Rank-based (ISO strings sort lexicographically — no date parsing), so it
    cannot saturate under an all-unplayed pool. Empty date_added sorts oldest.
    """
    if len(pool) < 2:
        return 0.5
    ordered = sorted(pool, key=lambda t: (t.date_added, t.track_id))
    pct = {t.track_id: i / (len(ordered) - 1) for i, t in enumerate(ordered)}
    return statistics.median(pct[t.track_id] for t in chosen)


def _log_lift(ratio: float) -> float:
    """Common identity scale for concentration ratios: 1x→0, 2x→1/3, 8x+→1."""
    return min(math.log2(max(ratio, 1.0)) / 3.0, 1.0)
```

(c) `_finalise` gains `freshness` computation and stores both carriers (still computes the OLD feasibility formula — Task 3 replaces it; this keeps existing tests green mid-flight):

```python
# in _finalise's signature: add   pool: list[Track],
# callers pass their pool positionally-by-keyword: pool=pool
    return Direction(
        ...,
        feasibility=round(feasibility, 4),
        thread_artist=thread_artist,
        identity=signal_clamped,
        freshness=round(_freshness(pool_capped, pool), 4),
    )
```

(where `pool_capped` is the existing local `pool` variable inside `_finalise` — rename it `chosen_capped` to avoid shadowing the new parameter).

(d) `_build_mood_journey`: change `signal = _balance(len(start_ranked), len(end_ranked))` → `signal = _balance(len(start_tracks), len(end_tracks))`.

(e) `_build_label_spotlight(pool, *, seed, collection=None)`: replace `share = label_counts[label] / len(pool)` with:

```python
    share_pool = label_counts[label] / len(pool) if pool else 0.0
    if collection:
        share_coll = sum(1 for t in collection if t.label == label) / len(collection)
        signal = _log_lift(share_pool / share_coll) if share_coll else 0.0
    else:
        signal = share_pool
```

(f) `_build_fresh_crate`: replace `signal = min(1.0, len(newest_slice) / 15.0)` with `signal = len(dated) / len(pool)` (recency-read coverage — spreads instead of pinning at 1.0).

(g) `_named_candidates(pool, *, seed, collection=None)` threads `collection` to `_build_label_spotlight` only.

- [ ] **Step 4: Run new tests + full suite**

Run: `pytest tests/test_directions.py -q`
Expected: new tests PASS. Existing tests asserting exact feasibility values may fail on `fresh_crate`/`mood_journey`/`label_spotlight` rows (signal semantics changed): update those expected values by recomputing `0.4·pool_fill + 0.3·ratio + 0.3·new_signal` — do NOT weaken assertions to ranges.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/directions.py tests/test_directions.py
git commit -m "feat(directions): freshness metric and per-builder identity renormalisation"
```

---

### Task 3: Two-pass scorer (gates split from score)

**Files:**
- Modify: `src/mixlab/directions.py` (`_finalise`, new `_score_field`, `enumerate_directions`, `generate_directions`), `src/mixlab/library_map.py` (one line: thread `scoped` as `collection`)
- Test: `tests/test_directions.py` (append)

**Interfaces:**
- Produces: `_score_field(candidates: list[Direction]) -> list[Direction]` — pass 1 rank `(0.25·freshness + 0.45·identity)/0.70`, dedupe at shipped-set Jaccard > 0.6 walking in rank order, then pass 2 `feasibility = 0.25·freshness + 0.45·identity + 0.30·distinctiveness` via `dataclasses.replace`, distinctiveness measured over the surviving field only; lone survivor → distinctiveness 0.5. Returns survivors sorted `(-feasibility, direction_type)`.
- Consumes: `Direction.identity` / `Direction.freshness` carriers from Task 2.
- `library_map._pool_entry(pool, *, seed, collection)` passes `collection` through to `enumerate_directions` (payload shape untouched).

- [ ] **Step 1: Write failing tests**

```python
import dataclasses
from mixlab.directions import _score_field


def _cand(dtype: str, ids: list[str], *, identity: float, freshness: float) -> Direction:
    return Direction(direction_type=dtype, title=dtype, mood="m", track_ids=ids,
                     brief="b", feasibility=0.0, identity=identity, freshness=freshness)


class TestScoreField:
    def test_spread_where_old_formula_saturated(self):
        # Three full-size, path-feasible candidates that all scored ~1.0 before
        a = _cand("fresh_crate", [f"a{i}" for i in range(25)], identity=1.0, freshness=0.97)
        b = _cand("label_spotlight", [f"b{i}" for i in range(25)], identity=0.3, freshness=0.4)
        c = _cand("mood_journey", [f"c{i}" for i in range(25)], identity=0.6, freshness=0.5)
        scores = {d.direction_type: d.feasibility for d in _score_field([a, b, c])}
        assert max(scores.values()) - min(scores.values()) > 0.15

    def test_dedupe_drops_lower_ranked_clone(self):
        shared = [f"s{i}" for i in range(20)]
        hi = _cand("era_dialogue", shared + ["h1", "h2"], identity=0.9, freshness=0.5)
        lo = _cand("found_1", shared + ["l1", "l2"], identity=0.2, freshness=0.5)
        out = _score_field([hi, lo])
        assert [d.direction_type for d in out] == ["era_dialogue"]

    def test_lone_candidate_distinctiveness_is_half(self):
        a = _cand("artist_thread", [f"a{i}" for i in range(25)], identity=0.4, freshness=0.4)
        [scored] = _score_field([a])
        # 0.25*0.4 + 0.45*0.4 + 0.30*0.5 = 0.43
        assert abs(scored.feasibility - 0.43) < 1e-9

    def test_deterministic_under_input_order(self):
        a = _cand("x1", [f"a{i}" for i in range(25)], identity=0.5, freshness=0.5)
        b = _cand("x2", [f"b{i}" for i in range(25)], identity=0.5, freshness=0.4)
        assert _score_field([a, b]) == _score_field([b, a])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_directions.py -k ScoreField -q`
Expected: FAIL — `_score_field` not defined.

- [ ] **Step 3: Implement**

```python
_W_FRESHNESS = 0.25
_W_IDENTITY = 0.45
_W_DISTINCT = 0.30
_DEDUPE_JACCARD = 0.6


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _score_field(candidates: list[Direction]) -> list[Direction]:
    """Two-pass, acyclic scoring of a pool's combined candidate field.

    Pass 1 ranks on the distinctiveness-free score and dedupes clones (Jaccard
    > 0.6) in rank order. Pass 2 measures distinctiveness against the surviving
    field only — the rows the operator actually sees — and computes the final
    feasibility. Deterministic: total sort keys, order-independent output.
    """
    rank_key = lambda d: (-(_W_FRESHNESS * d.freshness + _W_IDENTITY * d.identity) / 0.70,
                          d.direction_type, d.title, tuple(sorted(d.track_ids)))
    kept: list[Direction] = []
    for cand in sorted(candidates, key=rank_key):
        if all(_jaccard(cand.track_ids, k.track_ids) <= _DEDUPE_JACCARD for k in kept):
            kept.append(cand)
    out: list[Direction] = []
    for cand in kept:
        others = [k for k in kept if k is not cand]
        distinct = 1.0 - max((_jaccard(cand.track_ids, k.track_ids) for k in others), default=0.5)
        if not others:
            distinct = 0.5
        feasibility = _W_FRESHNESS * cand.freshness + _W_IDENTITY * cand.identity + _W_DISTINCT * distinct
        out.append(dataclasses.replace(cand, feasibility=round(feasibility, 4)))
    return sorted(out, key=lambda d: (-d.feasibility, d.direction_type))
```

Then:
- `_finalise`: delete the `0.4/0.3/0.3` formula; gates stay (`MIN_DIRECTION_POOL`, `_path_feasible`); set `feasibility=0.0` (the field scorer owns it).
- `enumerate_directions(pool, *, seed, collection=None)` becomes: `return _score_field(_named_candidates(pool, seed=seed, collection=collection))`.
- `generate_directions`: call `_score_field(_named_candidates(...))`, keep diagnostic print and rotation on the scored list (rotation change comes in Task 7).
- `library_map._pool_entry(pool, *, seed, collection)`; `build_map_payload` passes `collection=scoped`. `_direction_entry` untouched.

- [ ] **Step 4: Run full suite, update expected feasibility values**

Run: `pytest tests/ -q`
Expected: `_score_field` tests PASS; existing exact-value assertions on feasibility need recomputation under the new formula (compute by hand from each fixture's identity/freshness/distinctiveness — do not loosen to ranges). `library_map` golden outputs regenerate.

- [ ] **Step 5: Validate spread on the live collection**

Run:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from mixlab.reader import parse_collection
from mixlab.library_map import build_map_payload
tracks = parse_collection(Path("import/rekordbox.xml"))
payload = build_map_payload(tracks, mode="all", seed=0, played=[])
scores = sorted(round(d["feasibility"], 3)
                for e in payload["pools"].values() for d in e["directions"])
print(f"n={len(scores)} min={scores[0]} max={scores[-1]} spread={scores[-1]-scores[0]:.3f}")
assert scores[-1] - scores[0] > 0.2, "scorer failed to spread"
EOF
```
Expected: prints n≈45–50, spread > 0.2. (Old formula: 44/49 rows in [0.70, 1.00].)

- [ ] **Step 6: Commit**

```bash
git add src/mixlab/directions.py src/mixlab/library_map.py tests/
git commit -m "feat(directions): two-pass scorer replaces saturated feasibility formula"
```

---

### Task 4: `mining.py` — predicate extraction

**Files:**
- Create: `src/mixlab/mining.py`
- Test: Create `tests/test_mining.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class Predicate:
    kind: str            # "label"|"era"|"tag"|"remixer"|"energy"|"bpm_regime"|"key_hood"
    value: str           # display value: "Hospital Records", "2015-2019", "high", "174", "8A"
    namable: bool
    track_ids: frozenset[str]

def extract_predicates(pool: list[Track]) -> list[Predicate]
```

Sorted `(kind, value)`. Gates: ≥8 tracks (remixer ≥5). Coverage cap: predicates covering > 70 % of the pool are dropped. Era = calendar-aligned 5-year windows (`year - year % 5`, display `"2015-2019"`); `year None/≤0` joins none. Tags lowercased. `bpm_regime` from `clustering`-style density peaks via `directions._tempo_regimes`; value = peak label `f"{round(peak)}"` — implementers: `_tempo_regimes` returns regimes, use each regime's min-BPM-sorted index and median rounded BPM as value. `key_hood`: for each camelot key present, the set of tracks whose key is `camelot_compatible` with it; value = the key code.
- Consumes: `Track` (`models.py`), `_tempo_regimes` and `camelot_compatible` (import from `mixlab.directions` / `mixlab.clustering`).

- [ ] **Step 1: Write failing tests**

```python
from mixlab.mining import Predicate, extract_predicates
from mixlab.models import Track


def _t(track_id, *, label="", year=None, tags=None, energy=None, remixer="",
       bpm=174.0, key="8A", date_added=""):
    return Track(track_id=track_id, artist=f"A{track_id}", title=f"T{track_id}",
                 bpm=bpm, camelot_key=key, genre="Drum & Bass", label=label,
                 year=year, tags=tags or [], energy=energy, remixer=remixer,
                 date_added=date_added)


class TestExtractPredicates:
    def test_label_gate_eight(self):
        pool = [_t(f"a{i}", label="Metalheadz") for i in range(8)] + \
               [_t(f"b{i}", label="Dispatch") for i in range(7)] + \
               [_t(f"c{i}") for i in range(10)]
        kinds = {(p.kind, p.value) for p in extract_predicates(pool)}
        assert ("label", "Metalheadz") in kinds
        assert ("label", "Dispatch") not in kinds        # 7 < 8

    def test_remixer_gate_five(self):
        pool = [_t(f"a{i}", remixer="Calibre") for i in range(5)] + \
               [_t(f"b{i}") for i in range(15)]
        assert ("remixer", "Calibre") in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_seventy_percent(self):
        # 96% of pool at high energy → predicate excluded (would ride along with anything)
        pool = [_t(f"a{i}", energy=8) for i in range(24)] + [_t("b0", energy=3)]
        assert ("energy", "high") not in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_era_calendar_aligned(self):
        pool = [_t(f"a{i}", year=2016 + (i % 4)) for i in range(10)] + [_t(f"b{i}") for i in range(5)]
        preds = {(p.kind, p.value) for p in extract_predicates(pool)}
        assert ("era", "2015-2019") in preds
        assert not any(k == "era" and v not in {"2015-2019"} for k, v in preds)

    def test_year_none_or_zero_joins_no_era(self):
        pool = [_t(f"a{i}", year=0) for i in range(10)] + [_t(f"b{i}") for i in range(10)]
        assert not any(p.kind == "era" for p in extract_predicates(pool))

    def test_mechanical_predicates_not_namable(self):
        pool = [_t(f"a{i}", bpm=174.0, key="8A") for i in range(10)] + \
               [_t(f"b{i}", bpm=120.0, key="3B") for i in range(10)]
        for p in extract_predicates(pool):
            if p.kind in ("bpm_regime", "key_hood"):
                assert p.namable is False

    def test_deterministic_order(self):
        pool = [_t(f"a{i}", label="Hospital Records", year=2020, tags=["Liquid"]) for i in range(10)] + \
               [_t(f"b{i}") for i in range(5)]
        assert extract_predicates(pool) == extract_predicates(list(reversed(pool)))

    def test_tags_lowercased(self):
        pool = [_t(f"a{i}", tags=["Liquid"]) for i in range(8)] + [_t(f"b{i}") for i in range(8)]
        assert ("tag", "liquid") in {(p.kind, p.value) for p in extract_predicates(pool)}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_mining.py -q`
Expected: FAIL — `ModuleNotFoundError: mixlab.mining`.

- [ ] **Step 3: Implement `src/mixlab/mining.py`**

```python
"""Conjunction mining — deterministic 'found' directions (spec 2026-08-08).

Pure and deterministic: no I/O, no RNG. Extracts atomic predicates from a
genre pool, scans cross-kind pairs, gates on support/lift, prunes subsumed
pairs, and shortlists by lift for scoring in directions._score_field.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mixlab.clustering import camelot_compatible
from mixlab.models import Track

_GATE_DEFAULT = 8
_GATE_REMIXER = 5
_COVERAGE_CAP = 0.70


@dataclass(frozen=True)
class Predicate:
    kind: str
    value: str
    namable: bool
    track_ids: frozenset[str]


def _gate(kind: str) -> int:
    return _GATE_REMIXER if kind == "remixer" else _GATE_DEFAULT


def extract_predicates(pool: list[Track]) -> list[Predicate]:
    """All qualifying atomic predicates over ``pool``, sorted (kind, value)."""
    if not pool:
        return []
    groups: dict[tuple[str, str, bool], set[str]] = {}

    def _add(kind: str, value: str, namable: bool, track: Track) -> None:
        groups.setdefault((kind, value, namable), set()).add(track.track_id)

    for t in pool:
        if t.label:
            _add("label", t.label, True, t)
        if t.year is not None and t.year > 0:
            lo = t.year - t.year % 5
            _add("era", f"{lo}-{lo + 4}", True, t)
        for tag in t.tags:
            _add("tag", tag.lower(), True, t)
        if t.remixer:
            _add("remixer", t.remixer, True, t)
        if t.energy is not None:
            band = "low" if t.energy <= 4 else "high" if t.energy >= 6 else "mid"
            _add("energy", band, True, t)

    # Mechanical kinds — minable, never namable.
    from mixlab.directions import _tempo_regimes  # local import avoids cycle at module load

    usable = sorted((t for t in pool if t.bpm > 0), key=lambda t: (t.bpm, t.track_id))
    for regime in _tempo_regimes(usable):
        peak = round(statistics.median(t.bpm for t in regime))
        for t in regime:
            _add("bpm_regime", str(peak), False, t)
    for key in sorted({t.camelot_key for t in pool if t.camelot_key}):
        hood = [t for t in pool if camelot_compatible(key, t.camelot_key)]
        for t in hood:
            _add("key_hood", key, False, t)

    cap = _COVERAGE_CAP * len(pool)
    out = [
        Predicate(kind=k, value=v, namable=n, track_ids=frozenset(ids))
        for (k, v, n), ids in groups.items()
        if len(ids) >= _gate(k) and len(ids) <= cap
    ]
    return sorted(out, key=lambda p: (p.kind, p.value))
```

(`import statistics` at top. If `_tempo_regimes` import from `directions` creates a real cycle at runtime, move `_tempo_regimes` into `clustering.py` and import from there in both places — mechanical refactor, keep the function byte-identical.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_mining.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/mining.py tests/test_mining.py
git commit -m "feat(mining): predicate extraction with gates and coverage cap"
```

---

### Task 5: `mining.py` — pair scan, lift gate, subsumption, shortlist

**Files:**
- Modify: `src/mixlab/mining.py`
- Test: `tests/test_mining.py` (append)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class MinedPair:
    a: Predicate
    b: Predicate                      # (a.kind, a.value) < (b.kind, b.value)
    support: int                      # |A∩B|
    lift: float                       # support * N / (|A|*|B|)
    member_ids: tuple[str, ...]       # intersection, sorted

def scan_pairs(predicates: list[Predicate], pool_size: int) -> list[MinedPair]
```

Gates: cross-kind only; support ≥ 15; ≥ 1 namable side; lift ≥ 1.3. Subsumption: drop pair if Jaccard(member set, either parent set) > 0.9. Shortlist: top 12 by `(-lift, title-equivalent tuple (a.value, b.value), member_ids)`.
- Consumes: `Predicate` from Task 4.

- [ ] **Step 1: Write failing tests**

```python
from mixlab.mining import MinedPair, scan_pairs


def _pred(kind, value, ids, namable=True):
    return Predicate(kind=kind, value=value, namable=namable, track_ids=frozenset(ids))


class TestScanPairs:
    def test_lift_math(self):
        # N=100, |A|=20, |B|=20, |A∩B|=15 → lift = 15*100/400 = 3.75
        a = _pred("label", "Hospital Records", [f"x{i}" for i in range(15)] + [f"a{i}" for i in range(5)])
        b = _pred("era", "2015-2019", [f"x{i}" for i in range(15)] + [f"b{i}" for i in range(5)])
        [pair] = scan_pairs([a, b], pool_size=100)
        assert pair.support == 15
        assert abs(pair.lift - 3.75) < 1e-9

    def test_chance_pair_dies(self):
        # lift exactly 1.0 < 1.3 → gone (the first draft shipped these)
        a = _pred("era", "2000-2004", [f"x{i}" for i in range(20)])
        b = _pred("key_hood", "8A", [f"x{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=20) == []

    def test_support_floor_fifteen(self):
        a = _pred("label", "L", [f"x{i}" for i in range(14)] + ["a0"])
        b = _pred("era", "2020-2024", [f"x{i}" for i in range(14)] + ["b0"])
        assert scan_pairs([a, b], pool_size=200) == []   # support 14

    def test_same_kind_pairs_skipped(self):
        a = _pred("key_hood", "8A", [f"x{i}" for i in range(20)], namable=False)
        b = _pred("key_hood", "9A", [f"x{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=200) == []

    def test_both_mechanical_skipped(self):
        a = _pred("bpm_regime", "174", [f"x{i}" for i in range(20)], namable=False)
        b = _pred("key_hood", "8A", [f"x{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=200) == []   # no namable side

    def test_subsumption_pair_identical_to_parent_dies(self):
        ids = [f"x{i}" for i in range(20)]
        a = _pred("label", "L", ids)
        b = _pred("tag", "liquid", ids + [f"y{i}" for i in range(2)])
        # pair members == a's set exactly → Jaccard 1.0 vs parent a → dropped
        assert scan_pairs([a, b], pool_size=200) == []

    def test_shortlist_caps_at_twelve_by_lift(self):
        preds = []
        base = [f"s{i}" for i in range(15)]
        for i in range(15):
            preds.append(_pred("label", f"L{i:02d}", base + [f"l{i}{j}" for j in range(5 + i)]))
            preds.append(_pred("tag", f"t{i:02d}", base + [f"g{i}{j}" for j in range(5 + i)]))
        out = scan_pairs(preds, pool_size=3000)
        assert len(out) <= 12
        lifts = [p.lift for p in out]
        assert lifts == sorted(lifts, reverse=True)

    def test_deterministic(self):
        a = _pred("label", "L", [f"x{i}" for i in range(16)])
        b = _pred("era", "2020-2024", [f"x{i}" for i in range(16)])
        assert scan_pairs([a, b], 100) == scan_pairs([b, a], 100)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_mining.py -k ScanPairs -q`
Expected: FAIL — `scan_pairs` not defined.

- [ ] **Step 3: Implement**

```python
MIN_SUPPORT = 15          # == directions.MIN_DIRECTION_POOL; import there, assert equal in tests
MIN_LIFT = 1.3
_SUBSUME_JACCARD = 0.9
SHORTLIST = 12


@dataclass(frozen=True)
class MinedPair:
    a: Predicate
    b: Predicate
    support: int
    lift: float
    member_ids: tuple[str, ...]


def _jaccard_sets(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def scan_pairs(predicates: list[Predicate], pool_size: int) -> list[MinedPair]:
    """Gated, pruned, shortlisted cross-kind conjunctions. Deterministic."""
    if pool_size == 0:
        return []
    ordered = sorted(predicates, key=lambda p: (p.kind, p.value))
    pairs: list[MinedPair] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.kind == b.kind or not (a.namable or b.namable):
                continue
            members = a.track_ids & b.track_ids
            if len(members) < MIN_SUPPORT:
                continue
            lift = len(members) * pool_size / (len(a.track_ids) * len(b.track_ids))
            if lift < MIN_LIFT:
                continue
            if (_jaccard_sets(frozenset(members), a.track_ids) > _SUBSUME_JACCARD
                    or _jaccard_sets(frozenset(members), b.track_ids) > _SUBSUME_JACCARD):
                continue
            pairs.append(MinedPair(a=a, b=b, support=len(members), lift=lift,
                                   member_ids=tuple(sorted(members))))
    pairs.sort(key=lambda p: (-p.lift, p.a.value, p.b.value, p.member_ids))
    return pairs[:SHORTLIST]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_mining.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/mining.py tests/test_mining.py
git commit -m "feat(mining): cross-kind pair scan with lift gate, subsumption, shortlist"
```

---

### Task 6: `mining.py` — materialise mined Directions (shipped set, title, mood, brief)

**Files:**
- Modify: `src/mixlab/mining.py`
- Test: `tests/test_mining.py` (append)

**Interfaces:**
- Produces: `mine_pool(pool: list[Track]) -> list[Direction]` — full pipeline for one pool: extract → scan → materialise. Each returned `Direction` has `direction_type="found"` (Task 7 renames to `found_N`), shipped `track_ids = _centrality_rank(members)[:MAX_DIRECTION_POOL]`, passes `_path_feasible(shipped) ratio ≥ 0.8` (else dropped), `identity=_log_lift(lift)`, `freshness=_freshness(shipped, pool)`, `feasibility=0.0`, title/mood/brief per below.
- Title: both namable → `f"Found: {a.value} × {b.value}"`; one namable → `f"Found: {namable.value}"`. NEVER a mechanical value.
- Mood: ASCII kind pair with display nouns `{"bpm_regime": "tempo", "key_hood": "harmonic"}`, e.g. `"label x era"`, `"tag x harmonic"`.
- Brief (both-namable):
  `FOUND SET. Play the corner of the crate where {a.value} meets {b.value} as a scene, not a coincidence — treat the conjunction as the thesis and let its shared sound carry the set. Evidence: {support} tracks sit in this overlap, {lift:.1f}x denser than chance{sel}. Anchors: {anchors}.`
  where `sel = f", of which the {len(shipped)} most central are selected" if support > len(shipped) else ""` and `anchors = "; ".join(f"{t.artist} — {t.title}" for t in shipped[:3])`.
- Brief (single-namable): `FOUND SET. Play the corner of the crate where {namable.value} clusters in one {noun} pocket (around {mech.value}{" BPM" if mech.kind == "bpm_regime" else ""}) as a scene, not a coincidence — …` (same evidence/anchors tail; `noun` = "tempo" or "harmonic").
- Consumes: `scan_pairs` (Task 5), `_centrality_rank`/`_path_feasible`/`_freshness`/`_log_lift`/`Direction`/`MAX_DIRECTION_POOL` (from `mixlab.directions`; import inside `mine_pool` to avoid the module-load cycle — `directions` will import `mining` in Task 7).

- [ ] **Step 1: Write failing tests**

```python
from mixlab.mining import mine_pool


def _conj_pool():
    """20-track Hospital×liquid overlap inside an 80-track pool, mixed enough
    that gates pass and lift is well above 1.3."""
    pool = []
    for i in range(20):
        pool.append(_t(f"hl{i}", label="Hospital Records", tags=["Liquid"], year=2016 + i % 4,
                       bpm=172.0 + (i % 5), date_added=f"2026-0{1 + i % 6}-10"))
    for i in range(20):
        pool.append(_t(f"h{i}", label="Hospital Records", year=2005 + i % 5, bpm=174.0,
                       date_added=f"2021-0{1 + i % 6}-10"))
    for i in range(20):
        pool.append(_t(f"l{i}", tags=["Liquid"], year=2010 + i % 5, bpm=173.0,
                       date_added=f"2022-0{1 + i % 6}-10"))
    for i in range(20):
        pool.append(_t(f"p{i}", label="Shogun Audio", tags=["Deep"], year=2018, bpm=174.0,
                       date_added=f"2023-0{1 + i % 6}-10"))
    return pool


class TestMinePool:
    def test_finds_the_planted_conjunction(self):
        found = mine_pool(_conj_pool())
        assert found, "expected at least one mined direction"
        top = found[0]
        assert top.direction_type == "found"
        assert top.title == "Found: Hospital Records × liquid"
        assert 15 <= len(top.track_ids) <= 25
        assert top.identity > 0.0
        assert top.feasibility == 0.0          # field scorer owns the final score

    def test_title_never_contains_mechanical_values(self):
        for d in mine_pool(_conj_pool()):
            for banned in ("BPM", "8A", "174"):
                assert banned not in d.title
                assert banned not in d.mood

    def test_mood_is_ascii_kind_pair(self):
        found = mine_pool(_conj_pool())
        assert found[0].mood == "label x tag"
        assert found[0].mood.isascii()

    def test_brief_leads_with_imperative_not_statistic(self):
        found = mine_pool(_conj_pool())
        assert found[0].brief.startswith("FOUND SET. Play")
        assert "denser than chance" in found[0].brief

    def test_brief_cites_selection_when_support_exceeds_cap(self):
        # widen overlap to 40 so support > 25
        pool = _conj_pool() + [
            _t(f"hx{i}", label="Hospital Records", tags=["Liquid"], year=2020, bpm=174.0,
               date_added=f"2026-0{1 + i % 6}-15") for i in range(20)]
        found = mine_pool(pool)
        top = next(d for d in found if d.title == "Found: Hospital Records × liquid")
        assert "most central are selected" in top.brief
        assert len(top.track_ids) == 25

    def test_empty_and_sparse_pools_mine_nothing(self):
        assert mine_pool([]) == []
        assert mine_pool([_t(f"s{i}") for i in range(10)]) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_mining.py -k MinePool -q`
Expected: FAIL — `mine_pool` not defined.

- [ ] **Step 3: Implement**

```python
_MECH_NOUN = {"bpm_regime": "tempo", "key_hood": "harmonic"}


def _kind_noun(kind: str) -> str:
    return _MECH_NOUN.get(kind, kind)


def mine_pool(pool: list[Track]) -> list[Direction]:
    """Extract → scan → materialise. feasibility stays 0.0 for the field scorer."""
    from mixlab.directions import (MAX_DIRECTION_POOL, Direction, _freshness,
                                   _log_lift, _path_feasible, _rank)
    # _rank = directions.py:79, the empty-tolerant _centrality_rank wrapper

    by_id = {t.track_id: t for t in pool}
    out: list[Direction] = []
    for pair in scan_pairs(extract_predicates(pool), len(pool)):
        members = [by_id[i] for i in pair.member_ids]
        shipped = _rank(members)[:MAX_DIRECTION_POOL]
        ok, _ratio = _path_feasible(shipped)
        if not ok:
            continue
        namables = [p for p in (pair.a, pair.b) if p.namable]
        mech = [p for p in (pair.a, pair.b) if not p.namable]
        if len(namables) == 2:
            title = f"Found: {namables[0].value} × {namables[1].value}"
            where = (f"where {namables[0].value} meets {namables[1].value}")
        else:
            unit = " BPM" if mech[0].kind == "bpm_regime" else ""
            title = f"Found: {namables[0].value}"
            where = (f"where {namables[0].value} clusters in one "
                     f"{_kind_noun(mech[0].kind)} pocket (around {mech[0].value}{unit})")
        sel = (f", of which the {len(shipped)} most central are selected"
               if pair.support > len(shipped) else "")
        anchors = "; ".join(f"{t.artist} — {t.title}" for t in shipped[:3])
        brief = (
            f"FOUND SET. Play the corner of the crate {where} as a scene, not a "
            f"coincidence — treat the conjunction as the thesis and let its shared "
            f"sound carry the set. Evidence: {pair.support} tracks sit in this "
            f"overlap, {pair.lift:.1f}x denser than chance{sel}. Anchors: {anchors}."
        )
        mood = f"{_kind_noun(pair.a.kind)} x {_kind_noun(pair.b.kind)}"
        out.append(Direction(
            direction_type="found",
            title=title,
            mood=mood,
            track_ids=[t.track_id for t in shipped],
            brief=brief,
            feasibility=0.0,
            identity=round(_log_lift(pair.lift), 4),
            freshness=round(_freshness(shipped, pool), 4),
        ))
    return out
```

(Implementer note: do not call `_centrality_rank` directly on possibly-empty lists — `_rank` is the tolerant wrapper.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_mining.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/mining.py tests/test_mining.py
git commit -m "feat(mining): materialise found directions with evidence briefs"
```

---

### Task 7: Wire mining into the shared field (`found_N`, caps, run path)

**Files:**
- Modify: `src/mixlab/directions.py` (`enumerate_directions`, `generate_directions`, new `_combined_field`)
- Test: `tests/test_directions.py` (append)

**Interfaces:**
- Produces: `_combined_field(pool, *, seed, collection=None) -> list[Direction]` — `_named_candidates` + `mining.mine_pool`, through `_score_field`, then: cap mined survivors at 3 per pool, enforce title uniqueness among mined rows (keep higher-ranked), rename `found` → `found_1`/`found_2`/`found_3` in final rank order (via `dataclasses.replace`). `enumerate_directions` returns it sorted `(-feasibility, direction_type)`.
- `generate_directions`: mined rows occupy at most 1 of `max_directions` — `picked = mined[:1] + rotation_over_named[:max_directions - len(mined[:1])]`. Diagnostic line becomes `f"Directions proposed: {named_types} ({len(named)}/{len(_BUILDERS)} builders, {len(mined)} found)"`.
- Consumes: `mine_pool` (Task 6), `_score_field` (Task 3), `_named_candidates` (Task 1).

- [ ] **Step 1: Write failing tests**

```python
from mixlab.directions import enumerate_directions, generate_directions


def _minable_pool():
    """Pool where at least one named builder AND the miner both fire."""
    pool = _conj_pool()          # from test_mining fixtures — move to a shared conftest helper
    # give fresh_crate its material: dated, rated anchors
    return pool


class TestCombinedField:
    def test_mined_rows_have_distinct_found_n_types(self):
        out = enumerate_directions(_minable_pool(), seed=0)
        found_types = [d.direction_type for d in out if d.direction_type.startswith("found")]
        assert found_types == [f"found_{i + 1}" for i in range(len(found_types))]
        assert len(found_types) <= 3

    def test_mined_rows_scored_not_zero(self):
        out = enumerate_directions(_minable_pool(), seed=0)
        for d in out:
            if d.direction_type.startswith("found"):
                assert d.feasibility > 0.0

    def test_no_duplicate_titles_among_mined(self):
        out = enumerate_directions(_minable_pool(), seed=0)
        titles = [d.title for d in out if d.direction_type.startswith("found")]
        assert len(titles) == len(set(titles))

    def test_same_seed_byte_identical(self):
        a = enumerate_directions(_minable_pool(), seed=7)
        b = enumerate_directions(_minable_pool(), seed=7)
        assert a == b


class TestRunPathMinedCap:
    def test_at_most_one_found_canvas_per_run(self):
        pool = _minable_pool()
        tracks_by_id = {t.track_id: t for t in pool}
        canvases = generate_directions(pool, tracks_by_id, seed=3, max_directions=3)
        found = [c for c in canvases if c.direction_type.startswith("found")]
        assert len(found) <= 1

    def test_diagnostic_line_reports_found_separately(self, capsys):
        pool = _minable_pool()
        generate_directions(pool, {t.track_id: t for t in pool}, seed=3)
        out = capsys.readouterr().out
        assert "found)" in out and "builders" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_directions.py -k "CombinedField or RunPathMinedCap" -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
from mixlab import mining

_MAX_MINED_PER_POOL = 3


def _combined_field(pool: list[Track], *, seed: int, collection: list[Track] | None = None) -> list[Direction]:
    """Named + mined candidates through the shared two-pass scorer, with the
    mined cap, title uniqueness, and found_N rename applied post-score."""
    scored = _score_field(_named_candidates(pool, seed=seed, collection=collection)
                          + mining.mine_pool(pool))
    out: list[Direction] = []
    mined_kept = 0
    seen_titles: set[str] = set()
    for d in scored:                      # already (-feasibility, direction_type) sorted
        if d.direction_type == "found":
            if mined_kept >= _MAX_MINED_PER_POOL or d.title in seen_titles:
                continue
            mined_kept += 1
            seen_titles.add(d.title)
            d = dataclasses.replace(d, direction_type=f"found_{mined_kept}")
        out.append(d)
    return out


def enumerate_directions(pool: list[Track], *, seed: int,
                         collection: list[Track] | None = None) -> list[Direction]:
    return sorted(_combined_field(pool, seed=seed, collection=collection),
                  key=lambda d: (-d.feasibility, d.direction_type))
```

`generate_directions` gains a `collection: list[Track] | None = None` keyword (existing callers unchanged), and its body changes:

```python
    field = _combined_field(pool, seed=seed, collection=collection)
    named = [d for d in field if not d.direction_type.startswith("found")]
    mined = [d for d in field if d.direction_type.startswith("found")]
    proposed = ", ".join(sorted(d.direction_type for d in named)) if named else "none"
    print(f"Directions proposed: {proposed} ({len(named)}/{len(_BUILDERS)} builders, {len(mined)} found)")
    if not field:
        return []
    named.sort(key=lambda d: (-d.feasibility, d.direction_type))
    picked = mined[:1]
    if named:
        rng = random.Random(seed)
        offset = rng.randrange(len(named))
        ordered = [named[(offset + i) % len(named)] for i in range(len(named))]
        picked += ordered[: max_directions - len(picked)]
```

(rest of the materialisation loop unchanged).

- [ ] **Step 4: Run full suite**

Run: `pytest tests/ -q`
Expected: PASS. Existing `generate_directions` seed-expectation tests will shift (candidate counts changed — spec §1 records this as accepted); update expected picks by running the fixture and reading the new deterministic output ONCE, then hard-coding it.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/directions.py tests/test_directions.py
git commit -m "feat(directions): mined candidates join the field with found_N types and run-path cap"
```

---

### Task 8: Integration — payload fixture, golden determinism, live validation

**Files:**
- Modify: `tests/test_library_map.py` (append)
- Test: same

**Interfaces:**
- Consumes: everything above; `build_map_payload` (unchanged shape).

- [ ] **Step 1: Write failing integration tests**

```python
class TestMapPayloadWithMining:
    def test_found_rows_ship_with_valid_shape(self):
        payload = build_map_payload(_fixture_collection(), mode="all", seed=0, played=[])
        found = [d for e in payload["pools"].values() for d in e["directions"]
                 if d["direction_type"].startswith("found_")]
        assert found, "fixture collection must mine at least one found row"
        for d in found:
            assert set(d) == {"direction_type", "title", "mood", "brief",
                              "feasibility", "track_ids"}
            assert d["title"].startswith("Found: ")
            assert 0.0 < d["feasibility"] <= 1.0
            assert 15 <= len(d["track_ids"]) <= 25
            assert d["mood"].isascii()

    def test_payload_byte_identical_across_runs(self):
        col = _fixture_collection()
        a = render_map_json(build_map_payload(col, mode="all", seed=0, played=[]))
        b = render_map_json(build_map_payload(col, mode="all", seed=0, played=[]))
        assert a == b
```

(`_fixture_collection()`: reuse the module's existing collection fixture if one exists; otherwise build ~80 tracks from the Task 6 `_conj_pool` pattern spread across two genres. Keep it in this test file.)

- [ ] **Step 2: Run to verify failure, then pass**

Run: `pytest tests/test_library_map.py -q`
Expected: first FAIL if fixture lacks minable structure (fix the fixture), then PASS.

- [ ] **Step 3: Live-collection validation (manual gate, not CI)**

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from collections import Counter
from mixlab.reader import parse_collection
from mixlab.library_map import build_map_payload
tracks = parse_collection(Path("import/rekordbox.xml"))
payload = build_map_payload(tracks, mode="all", seed=0, played=[])
rows = [(k, d["direction_type"], d["title"], d["feasibility"])
        for k, e in payload["pools"].items() for d in e["directions"]]
found = [r for r in rows if r[1].startswith("found_")]
scores = sorted(r[3] for r in rows)
print(f"rows={len(rows)} found={len(found)} spread={scores[-1]-scores[0]:.3f}")
for r in found: print("  ", r)
tautology = [r for r in found if "2025" in r[2] and "fresh" in r[2].lower()]
assert not tautology, "era x fresh tautology resurfaced"
EOF
```
Expected: found rows present in the dense pools, none titled from mechanical values, no era×fresh rows (predicate removed), spread > 0.2. Paste the output into the PR description.

- [ ] **Step 4: Full suite + commit**

Run: `pytest tests/ -q`
Expected: PASS.

```bash
git add tests/test_library_map.py
git commit -m "test(map): integration coverage for mined found rows"
```

- [ ] **Step 5: Open PR**

```bash
git push -u origin claude/conjunction-mining
gh pr create --base develop --title "feat: conjunction mining + direction scoring replacement" \
  --body "Implements docs/superpowers/specs/2026-08-08-conjunction-mining-design.md (see spec §10 for adversarial-review disposition). Stage 1: two-pass scorer. Stage 2: conjunction miner. Live-collection validation output below."
```
