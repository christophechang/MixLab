# Mix Canvas Architecture

This document describes the core data and control flow behind MixLab's mix concept generation. It is the canonical reference for a contributor — or future-self after six months — who needs to understand the pipeline without reading every source file.

It is not marketing copy. It explains what each layer is responsible for, what it deliberately does *not* try to do, and why the decisions were made the way they were. When the source disagrees with this document, the source is right; please fix the document.

---

## 1. Why canvases exist

MixLab uses two LLM stages with different purposes:

- **Stage 1** is technical pre-screening. Given a pool of tracks, it groups them into shortlists of 15–25 tracks that are plausibly compatible by BPM and key. It is a clustering problem with creative flavour. The system uses a free-provider cascade (Groq → Gemini → Mistral) because the work is high-volume and uncreative.
- **Stage 2** is creative curation and narrative design. Given pre-screened pools, it selects 8–14 tracks, orders them, assigns roles per track, picks an energy arc, names the concept, and writes a prose report. The system uses Claude Sonnet 4.6 because the work demands real DJ judgment.

If Stage 2 received Stage 1's raw shortlists, the creative model would also have to figure out: which tracks are openers, which can serve as peaks, which are structurally weak, what risks the pool has, and how distinctive any particular concept would be relative to other recent runs. That is too much. The Stage 2 prompt would inflate, and the model's attention would split.

The **Mix Canvas** is the deterministic layer between Stage 1 and Stage 2. It takes a Stage 1 shortlist and post-processes it into a structured menu of materials:

- BPM tiers: which tracks are core, which are bridge candidates, which are wildcards.
- Role pools: which tracks could plausibly open, build, peak, close.
- Contrast assets: which tracks supply vocal moments, texture changes, darker or brighter turns.
- Risk notes: pre-flagged structural weaknesses (weak opener pool, over-repeated artist, etc.).
- A scored summary used to pick the most viable canvases for the limited Stage 2 budget.

This means Stage 2 reasons about *materials*, not raw tracks. Its prompt can stay focused on sequencing, narrative, and risk — the things only a creative model can do.

---

## 2. Pipeline overview

```
parse_collection (XML)
    │
    ▼
filters: denylist, BPM/year ranges, played/unplayed
    │
    ▼
genre clustering (or custom-genre cross-genre pool)
    │
    ▼
partition_bpm_pools  →  core / bridge / wildcard
    │
    ▼
Stage 1 LLM   (cascade: Groq → Gemini → Mistral)
    │  returns 3–5 MixConcept shortlists per cluster
    ▼
build_mix_canvas
    │  attaches roles, contrast assets, risk notes
    ▼
score_canvas + select_canvases  (deterministic, diversity-aware)
    │  picks top 6 canvases for Stage 2
    ▼
Stage 2 LLM   (Claude Sonnet 4.6, two passes)
    │  pass 1: selection JSON
    │  pass 2: prose report per concept
    ▼
validate_stage2_output  (warn-only)
    │
    ▼
history.append_run  +  Discord delivery
```

Playlist mode is a separate but parallel pipeline: it uses a Stage 0 intent extraction step over a seed playlist, then zone-based shortlisting and a three-variant Stage 2 (practical / balanced / adventurous). The canvas system is genre-mode-specific.

---

## 3. BPM pool semantics (core / bridge / wildcard)

Defined in [`clustering.py`](../../src/mixlab/clustering.py) `partition_bpm_pools`.

After a track pool is selected for a single genre cluster, the median BPM is computed. Tracks are then split into three tiers by their absolute distance from that median:

| Tier     | Threshold              | Constant         | What it represents                                             |
| -------- | ---------------------- | ---------------- | -------------------------------------------------------------- |
| Core     | `|BPM − median| ≤ 6`   | `_BPM_SPREAD`    | Tightly coherent — usable freely.                              |
| Bridge   | `6 < |BPM − median| ≤ 12` | `_BRIDGE_SPREAD` | Off-cohort but plausible for openers, pivots, resets, closers. |
| Wildcard | `|BPM − median| > 12`  | implicit         | Risky; only used when concept-defining.                        |

**Why three tiers and not just a hard ±6 BPM filter?**

The old single-tier filter discarded musically interesting tracks that happened to sit outside the cohort. A track at the BPM edge can be the ideal opener (deliberate BPM drop-in) or closer (BPM step-down that reads as resolution). Putting them in bridge/wildcard preserves their availability without polluting the core.

**Stage 2 rules**: core tracks may be used freely; bridge and wildcard picks require a named structural justification ("opener with deliberate BPM drop-in", "pivot that earns a key reset", etc.). This is enforced verbally in the prompt and instrumented by the validator (bridge/wildcard transition without `risk_type` → warning).

---

## 4. Role candidate inference

Defined in [`clustering.py`](../../src/mixlab/clustering.py) `_infer_roles`.

Each Mix Canvas carries six **role candidate pools** computed deterministically from the track pool. These are not assignments — they are hints derived from energy score, BPM proximity to the cluster median, and Camelot distance from the dominant key.

| Role            | Heuristic                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `opener`        | energy ≤ 3/10, or bridge-tier track when energy is missing                                          |
| `groove_locker` | core tier, energy 3–5/10 (MIK lounge/smooth-groove band), BPM within ±2 of median                                                  |
| `builder`       | core tier, energy 4–6/10                                                                            |
| `peak`          | core tier, energy ≥ 7/10 (MIK: upbeat/club-peak territory)                                                                            |
| `pivot`         | Camelot distance ≥ 3 from the dominant key (regardless of tier)                                   |
| `closer`        | energy ≤ 4/10                                                                                       |

A track may appear in multiple pools. When energy data is missing, a BPM-proxy fallback fires (higher-BPM-than-median → peak candidate; lower → closer candidate). The Stage 2 prompt is told explicitly that role candidates are **hints, not assignments** — it may override them when its DJ instinct disagrees.

---

## 5. Contrast assets

Defined in [`clustering.py`](../../src/mixlab/clustering.py) `_detect_contrast`.

Beyond roles, every canvas carries four contrast-asset pools:

- `vocal_moments`: tracks whose artist or title contains a vocal marker (`feat.`, `ft.`, `vocal`).
- `texture_changes`: tracks at Camelot distance ≥ 3 from the dominant key.
- `darker_turns`: tracks with energy below the canvas median.
- `brighter_lifts`: tracks with energy above the canvas median + 1.

These give Stage 2 a vocabulary for designing the journey: where the vocal moments land, where the texture shifts, where the darker passages live. Each is a list of track IDs surfaced in the canvas header.

---

## 6. Risk notes

Defined in [`clustering.py`](../../src/mixlab/clustering.py) `_generate_risk_notes`.

Pre-flagged structural weaknesses surfaced in the canvas header so Stage 2 can address them in track selection or energy arc design. The current notes are:

- `weak opener pool` — fewer than 2 opener-candidate tracks
- `weak closer pool` — fewer than 2 closer-candidate tracks
- `excessive BPM spread` — core BPMs span more than 10 BPM
- `too-similar midsection` — more than 60% of core keys near the dominant Camelot
- `over-repeated artist` — any artist appears ≥3 times in core
- `over-repeated label` — any label appears ≥4 times in core
- `all high energy` — more than 75% of tracks at energy ≥6/10 (danceable or hotter)

Since v0.10, the `weakness_penalty` term in `score_canvas` subtracts 0.04 per risk note from the weighted score (capped at 0.20). A canvas with multiple flags is mechanically deprioritised, not just diagnostically flagged. Stage 2 is also permitted to skip a canvas entirely when its risk notes describe structural problems that cannot be overcome with track selection.

---

## 7. Canvas scoring

Defined in [`clustering.py`](../../src/mixlab/clustering.py) `score_canvas`.

Each canvas receives eight scoring components plus two penalty terms, weighted and combined into a single `overall` value used to rank canvases. Current weights (tuned in v0.10 via #9 and v0.11 via #20):

```
overall = (
    technical_viability * 0.10 +
    role_coverage       * 0.25 +
    anchor_strength     * 0.15 +
    contrast_potential  * 0.15 +
    distinctiveness     * 0.15 +
    era_coherence       * 0.05 +
    label_coherence     * 0.05 +
    novelty             * 0.10
) − weakness_penalty
overall *= floor_multiplier
overall = max(0.0, overall)
```

| Component             | Formula                                                                                       | Meaning                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `technical_viability` | `min(1.0, log(core_n + 1) / log(16))` (saturates at 15 tracks)                                | Pool size with diminishing returns. A 30-track pool no longer beats a 15-track one.|
| `role_coverage`       | non-empty role pools / 6                                                                       | How many of the six roles have at least one candidate.                              |
| `anchor_strength`     | `0.5 × bool(opener) + 0.5 × bool(closer)`                                                      | Presence-based, not volume. Rewards having both opener AND closer.                  |
| `contrast_potential`  | `min(1.0, contrast_assets / 4)`                                                                | Count of non-empty contrast pools (vocal/texture/darker/brighter).                  |
| `distinctiveness`     | `1 − overlap_with_already_picked / core_n`                                                     | Penalty when a canvas shares core tracks with canvases already selected this run.   |
| `novelty`             | `1 − similarity_to_history` (Jaccard with decay; see §8)                                       | Penalty when a canvas overlaps with recent prior runs.                              |
| `era_coherence`       | 1.0 when core span ≤ 3 years, linear decay to 0.0 at span ≥ 18 years (needs ≥ 60% year cover)   | Bonus when a canvas is era-defined; never a penalty when year data is missing.       |
| `label_coherence`     | ramp from 0.0 at 40 % share to 1.0 at 70 %+ share (needs ≥ 5 tracks on the dominant label)      | Bonus when a single label dominates the core pool; never a penalty when no label dominates. |
| `weakness_penalty`    | `min(0.20, len(risk_notes) × 0.04)`                                                            | Subtracted from the weighted sum.                                                   |
| `floor_multiplier`    | `0.5` when `core_n < 8`, `1.0` otherwise                                                       | Halves the score for canvases below the minimum core size.                          |

**Why these weights?**

The original weights (v0.6, `tech_via 0.25 / role 0.25 / anchor 0.15 / contrast 0.15 / dist 0.10 / novelty 0.10`) had two issues identified in v0.8 strategic review:

1. The first pick of the greedy selector was unpenalised by overlap, so subsequent picks were reweighted but the dominant winner had already cleared. Distinctiveness at 0.10 wasn't enough to flatten the top-6.
2. `technical_viability` was linear and capped at 20 tracks — a 30-track generic pool beat a 15-track sharper one on this dimension alone.

v0.10 raised distinctiveness from 0.10 to 0.15 (taken from technical_viability), made technical_viability logarithmic with saturation at 15, and added the floor multiplier so canvases below 8 core tracks no longer compete on equal footing. The weakness_penalty wires the existing risk-note diagnostics into the actual selection consequence.

---

## 8. Novelty and concept history

Defined in [`history.py`](../../src/mixlab/history.py).

Each successful run writes a record to `.mixlab/concept-history.json`. The history stores per-run data needed for two purposes:

1. **Novelty scoring** — `similarity_to_history` computes Jaccard similarity of canvas core track IDs vs the last 10 runs, with `0.8^age` decay (so an identical canvas from yesterday penalises more than one from a week ago). The `novelty` component of the canvas score is `1 − this similarity`.
2. **Diagnostic and feedback signals** — since v0.10, the history also stores `canvas_score_breakdown`, `selected_canvas_risk_notes`, `bpm_band`, `role_pattern`, and the `energy_path` (Stage 2's `arc_type`). These power downstream features like concept-shape novelty (#7) and the recent-concepts injection (#13 — see §10).

Schema is forward-compatible: `load_history` filters incoming JSON to known dataclass fields, so old files load with defaults and future unknown fields are silently dropped. `bpm_band` is coerced back to a tuple on load (JSON serialises tuples as lists).

History caps at 50 entries; oldest are dropped first. The recency window for novelty scoring is 10 (constant `_RECENCY_WINDOW`), not the full history.

---

## 9. Canvas selection

Defined in [`clustering.py`](../../src/mixlab/clustering.py) `select_canvases`.

Selection is greedy and diversity-aware:

1. Score every candidate canvas with `picked_ids = ∅`.
2. Pick the canvas with the highest `overall` score.
3. Add its core track IDs to `picked_ids`.
4. Re-score the remaining candidates (their `distinctiveness` term changes because `picked_ids` is bigger).
5. Repeat until `n` canvases (default 6) are picked or no candidates remain.

This is a sub-modular maximisation: each iteration the top candidate is the canvas that scores highest given what has already been picked. The result is a top-6 that is not just "the strongest 6 individually" but "the strongest 6 collectively, given they each penalise the next pick for overlap."

`--debug` (or `MIXLAB_DEBUG_SCORE=1`) emits per-canvas score breakdown to stderr, including every component, the weakness_penalty, the floor_multiplier, the overlap penalty against already-picked canvases, and the novelty source (which prior run contributed the largest similarity).

---

## 10. Stage 2 responsibility boundary

Defined across [`llm.py`](../../src/mixlab/llm.py) (`_STAGE2_SYSTEM`, `_STAGE2_CANVAS_RULES`, and `stage2_curate_and_report`).

### What Stage 2 decides

- Which 8–14 tracks to select from the canvas pool (rejecting tracks that weaken the journey).
- The play order.
- The role per track (Stage 2's vocabulary is 19 roles; the canvas only carries 6 role candidate pools as hints).
- The energy path: one of seven explicit shapes (Slow Climb, Wave, Plateau With Detail, Double Peak, Front-Loaded Hook, Dark to Light, Light to Dark).
- The structured `arc_type` field (11 enum values; this is the machine-readable counterpart to the prose energy-path label, added in v0.9 for downstream consumers like history and validation).
- The concept's title (oblique, evocative — the prompt explicitly forbids generic `[Adjective][Noun]` patterns) and `name_reason`.
- Transition risk annotations (`is_risky`, `risk_type` per consecutive pair).
- The prose mix report (generated in a separate, second Stage 2 call to keep selection JSON under 8K tokens).
- Whether to skip a canvas whose risk notes describe unfixable structural problems.

### What is pre-decided before Stage 2

- The pool of candidate tracks (Stage 1 + canvas building).
- The canvas's core / bridge / wildcard tiering.
- Role candidate hints (Stage 2 may override).
- Contrast asset pools (vocal / texture / darker / brighter).
- Risk notes flagged on the canvas.
- The validation criteria the run will be judged against (see §11).

### What Stage 2 sees

The Stage 2 prompt receives, for each canvas:

- The canvas header: ID, novelty score, role and contrast pools, risk notes, BPM tiers.
- The candidate tracks with metadata (artist, title, BPM, key, genre, optional year, label, remixer, mix tags, energy score, `[unverified]` flag).
- The `[bridge]` / `[wildcard]` annotation on off-core tracks.
- The `unplayed` marker on tracks that have not been played live (when `--mode all` has catalog data; see #14).
- A `RECENT CONCEPTS` block listing the last 5 concepts (since #13) so the model can deliberately diverge from prior runs.

Stage 2 does *not* see: play count, denylist status, BPM corrections, history Jaccard similarity values, or canvas score breakdown. Those are signals the model would mishandle.

---

## 11. Validation philosophy

Defined in [`llm.py`](../../src/mixlab/llm.py) `validate_stage2_output`.

Validation is **warn-only**. No check ever aborts a run. The reasoning:

- The pipeline does real LLM work; a single false-positive warning that aborts costs the full Stage 2 budget.
- Most validation findings are advisory ("you might want to check this"), not fatal ("this output is broken"). A DJ reading the report can use the warning to triage.
- True correctness failures (track ID not in library, denylisted track, etc.) are still surfaced — they just don't kill the run.

Warnings fall into tiers:

- **Strong tier** (always fires): track ID missing, denylist hit, played-track leakage in `--mode unplayed`, opener absent in first 1–2 positions, closer absent in last 1–2 positions, bridge/wildcard used without a justified transition, artist appearing ≥ 3 times, BPM jumps > 15 between consecutive tracks, Camelot jumps > 4.
- **Soft tier** (softened by genre and concept `arc_type`): no peak in sequence, no wind-down before closer, three-or-more consecutive same-role-family tracks, all tracks in the same energy band.

Soft-tier softening:

- DnB-family genres (`drum_and_bass`, `jungle`, `170`) tolerate sustained pressure and dark closer statements.
- UK-bass-family genres (`uk_bass`, `uk_garage`, `breakbeat`, `140`) tolerate rhythmic pivots and role-family runs.
- `electronica` tolerates wide structural variation.
- `arc_type=plateau` or `sustained-pressure` suppresses missing-peak / no-wind-down / all-high-energy warnings.
- `arc_type=plateau` also suppresses role-family-run warnings.

Two distinctiveness checks added in v0.10 (review L3 + L4):

- **Cross-concept track overlap >50%** — pairwise warning naming both concepts. Catches the case where Stage 2 returns near-duplicate concepts from the strongest canvas.
- **Generic-name regex** — concept titles matching `[Adjective][Noun]` patterns (`Warm Gravity`, `Orbital Descent`) trigger a distinctiveness warning. Does not auto-retry — surfaces for user review.

All warnings appear under "⚠ Validation Notes" in the Discord report and never abort.

---

## 12. Operating modes

| Mode         | Pool                                          | Stage 2 `unplayed` marker                                                                |
| ------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `unplayed`   | Tracks not in play history (default)          | Every track (pool is by definition unplayed)                                              |
| `played`     | Tracks in play history                        | None (all played)                                                                         |
| `all`        | Full collection, no play-history filter       | Real played/unplayed split when `CATALOG_API_URL` is set; fallback "every track unplayed" when absent (with stderr warning) |

The Stage 2 prompt's "prefer unplayed in ties" rule becomes useful only when the marker conveys real information — that is, only in `--mode all` with catalog data. The other two modes either have a homogeneous pool (so the marker is uninformative) or filter the pool entirely.

Playlist mode bypasses canvases. It runs Stage 0 intent extraction over the seed playlist, builds BPM-zone shortlists, and asks Stage 2 for three explicit variants (practical / balanced / adventurous) which are scored post-hoc with `DJPracticalityScore` to pick a winner. The seed-retention floor (75% of anchors, 40% of supporting) is enforced after Stage 2 returns; a retry path fires when the floor is missed.

---

## 13. Where to look

| Question                                              | File                                                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| How is the BPM pool partitioned?                      | [`clustering.py`](../../src/mixlab/clustering.py) `partition_bpm_pools`                                       |
| How are roles inferred?                               | [`clustering.py`](../../src/mixlab/clustering.py) `_infer_roles`                                              |
| How are risk notes generated?                         | [`clustering.py`](../../src/mixlab/clustering.py) `_generate_risk_notes`                                      |
| How is a canvas scored?                               | [`clustering.py`](../../src/mixlab/clustering.py) `score_canvas` (with constants `_WEIGHT_*` at top of file)  |
| How are canvases selected?                            | [`clustering.py`](../../src/mixlab/clustering.py) `select_canvases`                                           |
| How is novelty computed?                              | [`history.py`](../../src/mixlab/history.py) `similarity_to_history`                                            |
| What does history store?                              | [`history.py`](../../src/mixlab/history.py) `HistoryEntry`                                                     |
| What is the Stage 2 prompt?                           | [`llm.py`](../../src/mixlab/llm.py) `_STAGE2_SYSTEM`, `_STAGE2_CANVAS_RULES`                                  |
| What validation checks run?                           | [`llm.py`](../../src/mixlab/llm.py) `validate_stage2_output`, `_structural_warnings`                          |
| How is the canvas formatted for the LLM?              | [`llm.py`](../../src/mixlab/llm.py) `_format_canvas_section`                                                  |
| Where is the "Bold moves:" line generated?            | [`llm.py`](../../src/mixlab/llm.py) `_format_bold_moves`, `_append_bold_moves_to_report`                      |
| What is the recent-concepts block?                    | [`history.py`](../../src/mixlab/history.py) `format_recent_concepts`                                          |
| How does the CLI dispatch by mode?                    | [`__main__.py`](../../src/mixlab/__main__.py) `run` (genre) and `run_playlist_mode` (playlist)                |
