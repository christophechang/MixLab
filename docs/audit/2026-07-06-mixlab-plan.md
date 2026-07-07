# MixLab Improvement Plan — 2026-07-06

Companion to `2026-07-06-mixlab-audit.md` (finding IDs referenced as B*/L*/D*/O*). Goal: **genuinely better, more creative playlists**. The plan favours a small number of structural ideas over many shallow tweaks, ordered so foundations land before the creative layers built on them.

Effort scale: **S** ≤ ½ day · **M** 1–2 days · **L** 3+ days.
**Live-run flag**: work whose *validation* requires real LLM/network runs at home. Code + unit tests against stubs always ship regardless.

---

## Workstream A — Foundations (fix what everything else stands on)

### P1. Repair Stage 1: honest pool sizes + seeded exploration
**Findings:** B1, B2, L8 · **Effort:** M–L · **Deps:** none · **Live run:** no (fully unit-testable)

Two coupled fixes to `partition_pool` / `_resize_shortlists`:

1. **Enforce the size contract.** After the existing resize passes, any shortlist still above `MAX_SHORTLIST` is reduced to a *window* of `MAX_SHORTLIST` tracks — but chosen by **seeded stratified sampling** (keep anchors/centrality spine deterministically, sample the remainder), not silent 160-track tolerance. Excess tracks are recorded per-shortlist as an `overflow` reserve so nothing is invisible-discarded (surfaced in `--debug` and pipeline summary).
2. **Make `seed` real.** `--stage1-seed` (default: derived from date, printed at run start so any run is reproducible) drives the sampling above. Same seed → identical output (preserves the determinism property the spec cares about); new day → new corner of a large pool (restores the exploration the README promises, deliberately instead of accidentally).

Also: `select_canvases` applies novelty/score ordering *even when* candidates ≤ cap, and forwards `n` best rather than everything, once ≥ 4 candidates exist — so novelty has consequence again (L8).

**Acceptance:** 800-track pool → 3–5 shortlists, all ≤ 25 tracks; same seed reproduces; different seeds differ; overflow reported; all existing partition_pool tests pass or are consciously updated.

---

### P2. Duration + richer ingestion (TotalTime, DateAdded, Rating)
**Findings:** D1, D3, L10 · **Effort:** S · **Deps:** none · **Live run:** no

Parse `TotalTime`, `DateAdded`, `Rating` into `Track`. Wire duration through:
- Per-track duration in Stage 2 prompt lines (`4:32`), cumulative set length in reports.
- `--mix-length` works in **genre mode** too, computed from real durations, replacing the minutes/4 heuristic in playlist mode.
- Shortfall/target logic can speak minutes, not just track counts (helps B8's fix in P5).

**Acceptance:** durations parsed from fixture XML; reports show per-track + total runtime; genre `--mix-length 60` produces a duration-targeted instruction; graceful when TotalTime absent.

---

### P3. Stage 2 resilience + configurability
**Findings:** L12, L3(partial), B9 · **Effort:** S–M · **Deps:** none · **Live run:** yes (config verified locally; live behaviour at home)

- Retry with exponential backoff on 429/5xx/timeout for all Anthropic calls (bounded, e.g. 3 attempts); check `stop_reason` and warn on truncation; scale report `max_tokens` with track count (B9).
- `MIXLAB_STAGE2_MODEL`, `MIXLAB_STAGE2_TEMPERATURE` env overrides (document in `.env.example`); default selection temperature raised 0.3 → 0.5 behind the env var so it's trivially revertible at home.
- Scope Stage 2 ID validation to the union of offered canvas pools, not the whole library (B7) — reject-and-warn on out-of-pool IDs at parse time.

**Acceptance:** respx tests for retry/truncation paths; env overrides honoured; out-of-pool IDs dropped with a warning line.

---

### P4. Quick-fix batch (mechanical, high confidence)
**Findings:** B3, B4, B5, B10, O1, O2 · **Effort:** S · **Deps:** none · **Live run:** no

- `_has_vocal_token` → word-boundary regex (`\bfeat\.?\b|\bft\.?\b|\bvocal(s)?\b|w/`).
- `run_export_unplayed`: check `CATALOG_API_URL` before parsing XML; make the test hermetic.
- Playlist unplayed bonus: make effective weight match the documented 0.15 (drop the double multiplication).
- Remove dead `build_playlist_pool` / `filter_by_bpm`; sync `pyproject` version; drop unused `anthropic` dep; fix `mixlab.skill.json`.
- README: rewrite the stale "random window" section to describe deterministic Stage 1 + P1 seeding; correct "fuzzy-matched" wording.

**Acceptance:** ruff/mypy/pytest green on clean clone (fixes O4).

---

## Workstream B — Creative core (the point of this plan)

### P5. Concept Directions: briefs instead of tempo strata
**Findings:** L1, L5, B8 · **Effort:** L (flagship) · **Deps:** P1 · **Live run:** yes (prompt quality tuning)

The single biggest creativity unlock. Insert a deterministic **direction generator** between canvas building and Stage 2. A *direction* is a creative brief with its own candidate pool assembled **across** BPM strata under technical guardrails, e.g.:

- **Mood journey** — pick a start/end mood pair from the library's own tag vocabulary (`dark → euphoric`, `brooding → carnival`); pool = tracks tagged along the gradient, BPM-feasible path required.
- **Era dialogue** — old-vs-new conversation when the pool's year data splits (reuses `_era_split` signals).
- **Label / scene spotlight** — dominant-label or label-cluster pool (reuses `_compute_dominant_label`).
- **Artist thread** — one artist/remixer appearing 2–3 times as deliberate throughline (currently *warned against* unconditionally; the direction legitimises it).
- **Energy-shape-first** — choose the arc (wave, double-peak…) *first*, then assemble a pool whose energy inventory can realise it (uses L6 machinery from P6).
- **Fresh crate** — DateAdded-recent spotlight (needs P2).

Each run: directions are enumerated from what the library actually supports, scored for feasibility (pool size, arc feasibility, harmonic path existence), and 2–3 are **seed-rotated** in alongside 2–3 classic canvases. Stage 2 prompt gets a per-direction brief (thesis seed, why these tracks, what to honour). Genre-specific track-count/duration targets ride in the brief (fixes B8). Mood tags finally do generative work (L5).

This deliberately reuses the existing canvas plumbing: a direction *is* a canvas with a brief attached — role pools, risk notes, scoring all still apply.

**Acceptance:** deterministic given seed; directions only proposed when feasible (unit-tested per direction type on synthetic libraries); prompt renders brief blocks; classic canvases still flow; live A/B at home decides default mix of directions vs classics.

---

### P6. Transition graph + arc verification (deterministic sequencing intelligence)
**Findings:** L6, L7 · **Effort:** M–L · **Deps:** none (P2 helps) · **Live run:** partially (prompt effect)

A small pure module (`transitions.py`) that scores track-pair mixability:

- **Tempo**: compatible if within pitch window (±6%) at 1:1, **or at 2:1 / 1:2 (half/double-time), or 3:4** — the relationship is named (`halftime`, `double`, `3:4 shuffle`). This finally teaches the system that 86 ↔ 172 is a lock, not a 15-BPM-rule violation ×6.
- **Harmonic**: existing `camelot_distance`, plus energy-boost moves (+2 semitone = +1 Camelot wrap) tagged as `energy_lift`.
- **Energy/mood**: energy delta and tag-overlap contrast.

Used three ways:
1. **Prompt**: each canvas/direction header gains a `Strong transitions:` line (top-N edges with mechanism names) — real sequencing material instead of noisy role hints (L4: role hints demoted to make room).
2. **Verification**: after Stage 2, trace declared `arc_type` against the actual energy sequence and transition edges; mismatches become precise warnings ("declared wave, energy is monotonic 4→7").
3. **Scoring**: `DJPracticalityScore.bpm_smoothness` and validator BPM-jump checks become tempo-relationship-aware (a clean halftime drop stops being penalised).

**Acceptance:** pure-function unit tests over the tempo/harmonic matrix; validator/practicality updated; canvas header renders edges; arc-trace warnings fire on synthetic mismatches.

---

### P7. Close the loop: one bounded self-revision pass
**Findings:** L13, (uses P6) · **Effort:** M · **Deps:** P6 (better signals), P3 (retries) · **Live run:** yes

When a concept accumulates ≥ N hard warnings (out-of-pool ID, unjustified jumps, arc mismatch) — or `--deep` returns `weak`/`needs_attention` with a `suggested_substitution` — run **one** targeted revision call: original concept + the specific findings + the canvas pool, asking for a minimal repair (swap/reorder, not regenerate). Strictly bounded: one pass, per-concept, opt-out via flag. Revised concept re-validated; report notes what changed ("Revised: swapped track 5, arc now traces wave").

Makes `--deep` finally *do* something and gives the warn-only philosophy a consequence path without abandoning it.

**Acceptance:** stubbed-LLM tests for trigger/no-trigger/repair-parse paths; revision never loops; report annotation present.

---

### P8. Playlist mode: steerable intent + fair variant selection
**Findings:** L2, L9 · **Effort:** M · **Deps:** none · **Live run:** yes (light)

1. **`--intent` in playlist mode**: merged into the Stage 0 brief as explicit user override (user text > inferred vibe on conflict; conflicts surfaced in the report).
2. **Fix winner selection**: score variants by *fitness for the brief*, not raw caution. Blend: `fit = practicality × w_p + adventure_dividend × w_a`, where weights come from `risk_tolerance` (Stage 0/`--intent`): `low` ≈ today's behaviour, `high` makes justified-risk density and role-completeness count *for* a variant. Ties no longer auto-break toward practical when tolerance is high. All three variants still reported; the WINNER label just stops being structurally rigged (L2).

**Acceptance:** unit tests: high-tolerance brief → adventurous can win on same inputs where today it can't; low-tolerance unchanged; `--intent` text visible in playlist prompt and report context.

---

### P9. Adventurous mode (`--risk high`) — reuses issue #42
**Findings:** L2, L3, #42 · **Effort:** M · **Deps:** P1, P3 (temp config), P5 helps · **Live run:** yes

Genre-mode risk knob (`--risk low|medium|high`, default medium ≈ today):
- `high`: canvas weights shift to contrast/wildcard promotion (new weight table), concept-anchor bridge/wildcard candidates explicitly offered as *featured* picks, selection temperature bumped, validator jump thresholds relaxed with mechanisms required, report flags `[ADVENTUROUS]`.
- `low`: tighter thresholds, warmer defaults — useful for radio/background briefs.

Naturally composes with P5 directions (a "mood journey at risk high" is a genuinely different artifact from today's output).

**Acceptance:** weight tables sum-validated; flag plumbed through both prompt and validator; default behaviour byte-stable at `medium`.

---

## Workstream C — Memory and follow-through

### P10. History that remembers everything + feedback verdicts
**Findings:** B6, L8, #12 discovery doc · **Effort:** M · **Deps:** P1 · **Live run:** no

- Record **all** concepts per run (schema: `concepts: [...]` list; forward-compatible loader already tolerates new fields). RECENT CONCEPTS block lists recent concepts (not runs), capped and deduped; shape-novelty compares against every stored concept.
- Minimal feedback capture from the #12 discovery doc: `mixlab feedback --last` interactive (or `--concept "title" --verdict played|rejected`) writing `feedback` onto history entries; `played` amplifies novelty penalty (×1.5), `rejected` mutes that entry's penalty (×0.25). One-line formula, no ML, exactly as the discovery doc scoped.

**Acceptance:** old history files load; new files round-trip; feedback verdicts change novelty deterministically in tests; CLI command covered.

---

### P11. Remove deprecated Stage 1 LLM path — existing issue #46 as written
**Effort:** S · **Deps:** none (but after P1 so the soak-fallback question is moot) · **Live run:** no

Mechanical deletion per #46's checklist. Keeps cascade for Stage 0.

---

## Explicitly deferred (and why)

- **#41 cue/grid workability** — high value but pure output-side; P2 takes the cheap slice (durations). Revisit after P5/P6 land; P6's transition graph is where cue data would eventually plug in.
- **#40 library map, #45 HTML report** — output/UX, not generation quality. Keep in backlog; P5's feasibility enumerator produces the data #40 would visualise.
- **#44 cross-genre bridging as specced** — superseded in spirit by P5 (cross-strata pools) + P6 (named bridge mechanisms). The remaining delta (explicit two-canvas bridge concepts in genre mode) should be re-scoped after P5 ships rather than built on today's one-canvas architecture.
- **Embeddings/audio analysis** — out of scope; the tag vocabulary + LLM knowledge is enough signal for this library's scale, and it keeps the system explainable.

## Suggested execution order

```
P4 quick fixes → P1 Stage 1 repair → P2 durations → P3 resilience/config → P11 cleanup
→ P6 transition graph → P10 history depth → P5 concept directions
→ P8 playlist intent/fairness → P7 revision loop → P9 adventurous mode
```

Rationale: everything up to P11 is deterministic, sandbox-verifiable, and de-risks the rest; P6 before P5 because directions want the feasibility signals; P7/P9 last because they tune behaviour that P3–P6 define. Items flagged **live run** ship code + stub tests now and get a validation pass at home (each carries a `needs-live-run` label on its ticket).
