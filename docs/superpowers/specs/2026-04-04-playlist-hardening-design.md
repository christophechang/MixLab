# Playlist Completion Hardening — Design Spec

**Date:** 2026-04-04
**Branch:** feat/intent-aware-playlist
**Status:** Approved — ready for implementation planning

---

## Background

Phase 1 (implemented) added intent extraction (Stage 0), anchor/supporting/optional seed classification, harmonic + role-aware candidate expansion, and a two-variant Stage 2 output (conservative/bold). Winner selection uses `anchor_retention * 0.65 + role_coverage * 0.35`.

This hardening pass upgrades winner selection to a **DJ Practicality Score** computed deterministically in Python from the returned track sequences, adds a `transitions` annotation array to the Stage 2 JSON output, replaces conservative/bold with **practical/balanced/adventurous**, improves chapter detection, and compacts the report format for all modes.

---

## Goals

1. **Winner selection driven by sequence quality**, not narrative labels — winner is the variant whose tracklist is most playable (smooth BPM contour, high harmonic ratio, justified risks, fragment preservation).
2. **Explicit transition classification** — LLM annotates each transition with `is_risky` and `risk_type`; Python evaluates whether the risk is acceptable.
3. **Three genuinely distinct variants**: practical / balanced / adventurous with concrete structural rules.
4. **Compact report format** — all modes; per-track Role / Why / Risk in one line, no narrative paragraphs.
5. **Chapter detection** — deterministic, based on `cluster_seed_zones` count, feeds `is_coherent_set`.

---

## Architecture

One Anthropic Stage 2 call (unchanged). Stage 2 output gains a `transitions` array per concept. Python post-processes the full response:

```
Stage 2 JSON
  └─ per concept:
       track_ids          → sequence to score
       transitions[]      → LLM-annotated risk classifications
       report             → compact format (all modes)

Python scoring:
  camelot_distance()     → bpm_delta + camelot_jump per pair (deterministic)
  _compute_practicality_score()  → DJPracticalityScore from sequence + LLM annotations
  _select_best_variant()         → max(practicality_score.overall), tiebreak by strategy priority
```

---

## Data Models

### New / changed in `models.py`

**`Transition`** — Pydantic model, represents LLM annotation for one consecutive pair:

```python
class Transition(BaseModel):
    from_id: str
    to_id: str
    is_risky: bool = False
    risk_type: str = ""   # "chapter_pivot" | "peak_impact" | "deliberate_reset"
                          # | "closer_move" | "cut_only" | "low_tonal_risk" | ""
```

**`DJPracticalityScore`** — dataclass, computed deterministically in Python:

```python
@dataclass
class DJPracticalityScore:
    bpm_smoothness: float     # 0.0–1.0
    harmonic_ratio: float     # 0.0–1.0
    risk_justified: float     # 0.0–1.0
    fragment_preserved: float # 0.0–1.0

    @property
    def overall(self) -> float:
        return (
            self.bpm_smoothness    * 0.30
            + self.harmonic_ratio  * 0.30
            + self.risk_justified  * 0.25
            + self.fragment_preserved * 0.15
        )
```

**`MixConcept`** — add optional `transitions` field (Pydantic, default empty list):

```python
transitions: list[Transition] = Field(default_factory=list)
```

**`CompletionVariant`** — update strategy type; replace `role_coverage` with `practicality_score`:

```python
@dataclass
class CompletionVariant:
    strategy: Literal["practical", "balanced", "adventurous"]
    concept: MixConcept
    anchor_retention_rate: float
    practicality_score: DJPracticalityScore

    @property
    def score(self) -> float:
        return self.practicality_score.overall
```

---

## New Function: `camelot_distance` in `clustering.py`

Computes the minimum number of steps on the Camelot wheel between two keys.

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
    if num_a == num_b:             # same number, opposite ring
        return 1
    ring_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if mode_a == mode_b:
        return ring_dist
    # Cross-ring: minimum path is same-ring distance + one ring crossing
    return ring_dist + 1
```

---

## Updated Chapter Detection in `playlist_mode.py`

`compute_deterministic_intent` currently always sets `is_coherent_set=True`. Update it:

```python
from mixlab.playlist_mode import cluster_seed_zones  # already in same file

zones = cluster_seed_zones(tracks)
is_coherent_set = len(zones) <= 1
```

This reuses the existing zone clustering logic — if the seed set splits into 2+ zones (gap ≥ 12 BPM), the set is treated as chapters.

---

## DJ Practicality Score: `_compute_practicality_score` in `llm.py`

```python
def _compute_practicality_score(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
    intent_brief: IntentBrief | None,
) -> DJPracticalityScore:
```

### bpm_smoothness (0.30 weight)

Measures how even consecutive BPM steps are.

```
bpm_deltas = [abs(track[i+1].bpm - track[i].bpm) for i in range(n-1)]
if len(bpm_deltas) < 2:
    bpm_smoothness = 1.0
else:
    std = statistics.stdev(bpm_deltas)
    bpm_smoothness = max(0.0, 1.0 - std / 10.0)   # 10 BPM std = 0.0
```

### harmonic_ratio (0.30 weight)

Fraction of consecutive pairs that are Camelot-compatible (distance ≤ 1).

```
harmonic_ratio = compatible_pairs / max(total_pairs, 1)
```

Uses `camelot_distance(a.camelot_key, b.camelot_key) <= 1`.

### risk_justified (0.25 weight)

Penalizes unjustified risks. A risk is unjustified if `is_risky=True` and `risk_type` is `"cut_only"` or `""`.

```
risky = [t for t in transitions if t.is_risky]
unjustified = [t for t in risky if t.risk_type in ("cut_only", "")]
if not risky:
    risk_justified = 1.0   # no risks = full score
else:
    risk_justified = max(0.0, 1.0 - len(unjustified) / len(risky))
```

For pairs without a matching `Transition` annotation, assume `is_risky=False` (no penalty).

### fragment_preserved (0.15 weight)

Fraction of strong adjacency pairs from `IntentBrief.strong_adjacencies` that are preserved in the returned sequence.

```
if intent_brief is None or not intent_brief.strong_adjacencies:
    fragment_preserved = 1.0
else:
    concept_ids = list(concept.track_ids)
    preserved = sum(
        1 for frag in intent_brief.strong_adjacencies
        if _pair_consecutive(frag.track_ids[0], frag.track_ids[1], concept_ids)
    )
    fragment_preserved = preserved / len(intent_brief.strong_adjacencies)
```

`_pair_consecutive(a, b, ids)` returns True if `b` immediately follows `a` in `ids`.

---

## Updated `_score_variant`, `_select_best_variant`, and Rejected Summary

`_score_variant` signature update:

```python
def _score_variant(
    concept: MixConcept,
    seed_track_ids: list[str],
    intent_brief: IntentBrief | None,
    tracks_by_id: dict[str, Track],
) -> CompletionVariant:
```

Changes:
1. Parse strategy from `concept.mood.lower()`: `"practical" | "balanced" | "adventurous"` (fallback `"practical"`)
2. Compute `anchor_retention_rate` as before (unchanged)
3. Call `_compute_practicality_score(concept, tracks_by_id, intent_brief)` → `practicality_score`
4. Return `CompletionVariant(strategy, concept, anchor_retention_rate, practicality_score)`

`_select_best_variant` tiebreak:

```python
_STRATEGY_PRIORITY = {"practical": 0, "balanced": 1, "adventurous": 2}

def _select_best_variant(variants: list[CompletionVariant]) -> CompletionVariant:
    return max(
        variants,
        key=lambda v: (v.score, -_STRATEGY_PRIORITY.get(v.strategy, 99)),
    )
```

Higher score wins; ties broken by `practical > balanced > adventurous`.

### Rejected summary update

With 3 variants there can be up to 2 rejected. Replace the current single-`other` summary with a loop. Also remove `other.role_coverage` (no longer on `CompletionVariant`):

```python
rejected_summary = ""
if len(variants) > 1:
    rejected = [v for v in variants if v.concept is not concept]
    if rejected:
        parts = [
            f"{v.strategy} (practicality: {v.practicality_score.overall:.2f}, "
            f"anchor retention: {v.anchor_retention_rate:.0%}) — not selected"
            for v in rejected
        ]
        rejected_summary = "\nAlternative strategies considered: " + "; ".join(parts) + "."
```

---

## Stage 2 Prompt Changes

### Three variants replacing conservative/bold

Update `_STAGE2_SYSTEM_PLAYLIST` variant instructions:

```
Produce EXACTLY THREE concepts from the same pool using these strategies:

1. "practical" (mood = "practical"): maximise harmonic continuity. Prefer adjacent Camelot keys
   (distance ≤ 1). BPM moves ≤ 2 BPM per step where possible. Preserve all strong seed adjacency
   pairs. Protect anchors. Avoid unearned key jumps.

2. "balanced" (mood = "balanced"): one major key jump (distance 2–3) or single BPM arc allowed.
   Anchors protected. Optional seeds may be swapped when a library track clearly serves the arc
   better. Adjacency pairs are hints, not constraints.

3. "adventurous" (mood = "adventurous"): prioritise set narrative and role completeness. Chapter
   pivots and peak impacts are permitted when musically justified — name the mechanism. Anchors
   protected; adjacency pairs may be broken with one-sentence reason. Optional and supporting seeds
   replaceable if a library track serves the arc materially better.

Label each concept's mood field with exactly "practical", "balanced", or "adventurous".
All three must meet the anchor protection rules and the seed retention floor.
If the pool is too thin to produce even one strong set: [{"diagnostic": "..."}].
```

### `transitions` array added to JSON schema

Update `_STAGE2_SYSTEM` (both modes) JSON schema:

```
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
risk_type: one of "chapter_pivot" | "peak_impact" | "deliberate_reset" | "closer_move"
           | "cut_only" | "low_tonal_risk" | "" (empty string when is_risky=false).
"cut_only" means: risky, with no mechanism that earns it — just a hard cut.
```

### Compact report format (all modes)

Replace the current multi-section report spec in `_STAGE2_SYSTEM` with:

```
The "report" value must be a single string (with \n for line breaks) in this exact format:

CONCEPT: [title]

[1–2 sentences: thesis — what this set asks of the room.]

Track order:
[For each track in play order, one line:]
N. Artist — Title [Key · BPM] | Role: [role] | Why: [one short phrase] | Risk: [one short phrase or "none"]

[If playlist completion: seeds retained / dropped / added line — Python adds this; omit from your output.]

Assumptions: [only if material — [unverified] tracks, vocal clash, tight blend window. One line each. Omit section if nothing material.]

Role options: opener, builder, pivot, peak, cleanser, closer, utility.
Risk: describe the transition risk into this track (not out of it). "none" if clean.
Why: why this track at this moment — one phrase, no full sentences needed.
```

---

## Parsing Changes in `_parse_curated_concepts`

After parsing each item dict, also parse `transitions`:

```python
raw_transitions = item.get("transitions", [])
transitions = []
for tr in (raw_transitions if isinstance(raw_transitions, list) else []):
    if isinstance(tr, dict):
        transitions.append(Transition(
            from_id=str(tr.get("from_id", "")),
            to_id=str(tr.get("to_id", "")),
            is_risky=bool(tr.get("is_risky", False)),
            risk_type=str(tr.get("risk_type", "")),
        ))
concept = MixConcept(
    title=..., mood=..., track_ids=track_ids, transitions=transitions
)
```

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/mixlab/models.py` | Modify | Add `Transition` (Pydantic); add `DJPracticalityScore` (dataclass); add `transitions` to `MixConcept`; update `CompletionVariant` (strategy type, replace `role_coverage` with `practicality_score`) |
| `src/mixlab/clustering.py` | Modify | Add public `camelot_distance(key_a, key_b) -> int` |
| `src/mixlab/playlist_mode.py` | Modify | Update `compute_deterministic_intent` to detect chapters via `cluster_seed_zones` |
| `src/mixlab/llm.py` | Modify | Update JSON schema (add `transitions`); update report format spec; update `_STAGE2_SYSTEM_PLAYLIST` (3 variants); update `_parse_curated_concepts` (parse transitions); add `_compute_practicality_score` + `_pair_consecutive`; update `_score_variant` + `_select_best_variant`; update rejected variant summary for 3 variants |
| `tests/test_clustering.py` | Modify | Add tests for `camelot_distance` |
| `tests/test_playlist_mode.py` | Modify | Add test for chapter detection in `compute_deterministic_intent` |
| `tests/test_intent.py` | Modify | Update existing `CompletionVariant` constructions (remove `role_coverage` kwarg, add `practicality_score`); add tests for `_compute_practicality_score`, `_select_best_variant` (3 variants, tiebreak) |

---

## Risks and Trade-offs

| Risk | Mitigation |
|---|---|
| LLM returns 0 transitions or mismatched IDs | `_compute_practicality_score` falls back to deterministic BPM/Camelot pair analysis for any missing annotations |
| 3 variants instead of 2 increases Stage 2 token usage | Transitions array is ~400–600 extra tokens; compact report saves ~300–500 per concept; net is roughly flat |
| `camelot_distance` with cross-ring keys | Tested explicitly; formula handles A↔B ring crossing correctly |
| `practical > balanced > adventurous` tiebreak feels arbitrary | Practical is the conservative fallback — if scores are tied, safer is better; this is intentional |
| `cut_only` risk_type may be over-applied by LLM | Python only penalizes `cut_only` (and empty) risk_type — other named risk types (chapter_pivot, peak_impact, etc.) are treated as justified by default |
