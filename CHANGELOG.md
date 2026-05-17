# Changelog

## v0.10.0

- **Canvas scoring weights tuned.** Five-change scoring pass landed via #9. `technical_viability` is now logarithmic with saturation at 15 core tracks (was linear, capping at 20) — a 30-track generic pool no longer outscores a 15-track sharper one. New `floor_multiplier` halves the overall score for canvases with fewer than 8 core tracks, so weak pools survive only when every other dimension is strong. `anchor_strength` is now presence-based (`0.5 × bool(opener) + 0.5 × bool(closer)`) instead of volume-based — having both opener and closer beats having ten opener candidates and zero closers. New `weakness_penalty` subtracts up to 0.20 from the weighted sum based on the canvas's risk-note count, wiring the existing risk-note diagnostics into selection consequence. `distinctiveness` weight raised from 0.10 to 0.15 (taken from `technical_viability`, now 0.20) so the greedy selector's later picks penalise overlap more strongly.
- **Richer concept history schema.** `HistoryEntry` extended with `canvas_score_breakdown` (mean per scoring dimension across selected canvases), `selected_canvas_risk_notes` (flat list of every risk note across canvases), `bpm_band` (min/max BPM tuple), and `role_pattern` (concept's tracks classified against the matched canvas's role pools, in play order). `load_history` now filters incoming JSON to known dataclass fields — old history files load cleanly with defaults, future files with extra unknown fields load without crashing. `bpm_band` is coerced from JSON list back to tuple on load.
- **`energy_path` always-empty bug fixed.** `HistoryEntry.from_run()` previously hardcoded `energy_path=""` regardless of input. Now reads from `concepts[0].arc_type` (the field added in v0.9.0). Concept-shape novelty scoring (planned in #7) was blocked by this regression; the field now carries real arc data on every run.
- **Stage 2 sees its own recent concepts.** A `RECENT CONCEPTS (avoid retreading these):` block is prepended to the Stage 2 user prompt listing the last 5 concepts (title, date, genre, `arc_type`, mood) in reverse chronological order, followed by an instruction telling the model to diverge deliberately. Empty history → block omitted entirely (no orphan header). Playlist mode also loads history now and gets the same block.
- **DJ-structural validation checks.** `validate_stage2_output` extended with seven new warn-only checks: missing opener in first 1–2 positions, missing closer in last 1–2 positions, no peak-role track in sequence, no wind-down in final 3 tracks, three-or-more consecutive same-role-family tracks, all tracks high-energy (≥6/8), all tracks low-energy (≤3/8). Strong-tier checks (opener/closer) always fire; soft-tier checks are softened by genre family (DnB tolerates sustained pressure; UK-bass family tolerates rhythmic pivots; electronica tolerates wide variation) and by `concept.arc_type` (plateau/sustained-pressure suppresses missing-peak / no-wind-down / all-high-energy). Structural checks are skipped entirely when the canvas has no role data, preventing false positives on legacy fixtures.
- **Cross-concept distinctiveness validation.** Pairwise track-set overlap >50% between any two concepts fires a warning naming both concepts. Backstops the Stage 2 prompt's qualitative "knowledgeable listener test" with a deterministic check.
- **Generic-name regex warning.** Concept titles matching `[Adjective][Noun]` patterns (`Warm Gravity`, `Slow Descent`, `Deep Pulse`, `Orbital Descent`) trigger a distinctiveness warning. Does not auto-retry — surfaces for user review. Oblique titles (`Late Latitude`, `Fever`, `Interzone`, `Red Light`, `The Slow Hours`) are unaffected.
- **Known-good test fixture and edge-case validation coverage.** New `_known_good_setup()` builds a 10-track concept that satisfies every check; asserting zero warnings on it is the no-false-positive guard. Edge cases (empty concept list, zero-track concept, single-track concept, empty cross-concept overlap) verified to not crash. Parametrized matrix tests cover the generic-name regex across both forbidden and acceptable titles, and the high-energy warning across (genre, `arc_type`) combinations.
- **Mix Canvas architecture documentation.** New `docs/architecture/mix-canvas.md` covers the full pipeline: why canvases exist, BPM pool semantics, role inference, contrast assets, risk notes, scoring weights with provenance (v0.6 → v0.10), novelty and history, canvas selection, Stage 2 responsibility boundary, validation philosophy, operating modes, and a where-to-look index. Module docstrings added to `clustering.py` and `llm.py` pointing readers to the architecture doc.

---

## v0.9.0

- **Stage 2 canvas rules tightened.** Role candidates shown in the canvas header are now explicitly framed as hints derived from technical analysis rather than fixed assignments — Stage 2 can override them when its DJ instinct disagrees, with a brief reason in the report. Bridge and wildcard tracks now require a specific structural role as justification ("opener with deliberate BPM drop-in", "pivot that earns a key reset") rather than a generic "interesting track" rationale. Stage 2 may skip a canvas entirely when its risk notes describe structural problems that cannot be overcome with track selection — every canvas need not produce a concept.
- **`arc_type` structured output field.** Every Stage 2 concept now carries a structured `arc_type` value alongside the prose energy-path description. Enum: `plateau`, `wave`, `progressive-build`, `build-and-drop`, `double-peak`, `sustained-pressure`, `front-loaded`, `dark-to-light`, `light-to-dark`, `narrative`, `abstract-journey`. The parser tolerates case variants and underscore/hyphen drift; missing or invalid values fall back to `None` so downstream consumers can ship independently.
- **`--mode all` honours real played/unplayed status.** When `CATALOG_API_URL` is set, `--mode all` now fetches the play history and passes the real unplayed track IDs to Stage 2. Previously every track was marked `unplayed` in the prompt, making the tiebreaker rule a no-op. Stage 2 can now favour unplayed tracks in ties when composing concepts from the full collection. Falls back gracefully with a one-line stderr warning when `CATALOG_API_URL` is absent or the fetch fails.
- **Bold moves annotation in concept reports.** Every concept report (genre mode) now ends with a `Bold moves:` line summarising how many bridge and wildcard tracks made the final selection. When the Stage 2 transition specifies a `risk_type`, a bullet per bold pick names the mechanism (chapter pivot, peak impact, deliberate reset, etc.). Concepts drawn entirely from core tracks read `Bold moves: none`. Provides measurement of how often the three-tier BPM partitioning actually surfaces in the final output.

---

## v0.8.0

- **`--mode` flag replaces `--all-tracks`.** Track selection is now controlled by `--mode {unplayed,all,played}` (default: `unplayed`). `unplayed` preserves the previous default behaviour. `all` replicates the old `--all-tracks` flag. `played` is new: restricts the candidate pool to tracks that have appeared in your play history — battle-tested and SoundCloud-proven. Works in both genre mode and playlist mode.
- **`filter_played()` added to matcher.** Symmetric counterpart to `filter_unplayed`. Returns tracks whose artist/title normalised key matches any entry in the played history.
- **`TrackMode` type added to models.** `Literal["unplayed", "all", "played"]` exported from `models.py` for use across the CLI and future extensions.
- **Availability table and pipeline summary reflect active mode.** The header `Available tracks (unplayed / in collection)` now reads `played` when `--mode played` is active. Pipeline summary `Track pool:` line follows the same label.

---

## v0.7.0

- **Canvas scoring diagnostics (`--debug`).** Pass `--debug` (or set `MIXLAB_DEBUG_SCORE=1`) to emit per-canvas score breakdowns to stderr. Output includes all six scoring components (`technical_viability`, `role_coverage`, `anchor_strength`, `contrast_potential`, `distinctiveness`, `novelty`, `overall`), overlap penalty with count of shared core tracks, novelty penalty with the closest history match (run date, genre, decayed Jaccard), core/bridge/wildcard track counts, and risk notes. Also shows the full candidate list before selection and the final pick order. Normal stdout and Discord output are unchanged.

---

## v0.6.0

- **Mix Canvas selection.** Stage 1 concepts are now wrapped into structured Mix Canvases before Stage 2 receives them. Each canvas carries deterministic role candidates (opener, groove-locker, builder, pivot, peak, closer), contrast assets (vocal moments, texture changes, darker/brighter turns), and risk notes (weak opener/closer pool, BPM spread, artist or label over-repetition). Up to 6 canvases are selected for Stage 2 using a weighted scoring model — technical viability, role coverage, anchor strength, contrast potential, cross-canvas distinctiveness, and novelty — replacing the previous random sampling from the top-12 by pool size. Selection is deterministic given the same input.
- **Three-pool BPM partitioning.** Tracks are no longer hard-filtered at ±6 BPM. Instead, each cluster is partitioned into core (±6 BPM from median), bridge (±12 BPM), and wildcard (>12 BPM). Core tracks go to Stage 1; bridge and wildcard tracks are retained as canvas metadata and are available to Stage 2 for structural roles — opener, pivot, reset, closer — where BPM deviation is intentional.
- **Concept history and novelty scoring.** Each successful run writes a record to `.mixlab/concept-history.json`. On the next run, canvases whose track IDs overlap heavily with recent selections receive a novelty penalty (Jaccard similarity over a 10-run recency window, decaying at 0.8^age per run). This compounds with Stage 1's random window sampling to push repeated runs toward different corners of the collection over time.
- **Post-Stage-2 validation.** A new warn-only validation pass runs after Stage 2 and before delivery. It checks for track IDs not in the library, denylist or played-track violations, Camelot jumps greater than 4, BPM jumps greater than 15 between consecutive tracks, and artist repeats of 3 or more. Warnings appear under **⚠ Validation Notes** in the Discord report and never abort a run.

---

## v0.5.0

- **Section-based mix thinking.** Stage 2 now explicitly reasons in five sections — Invitation, Groove Lock, Development, Peak/Payoff, Resolution — assigning every track to a section before deciding play order.
- **Explicit energy path selection.** Stage 2 must choose a named energy shape before sequencing: Slow Climb, Wave, Plateau With Detail, Double Peak, Front-Loaded Hook, Dark to Light, or Light to Dark. The chosen shape must be visible in the track ordering.
- **Expanded track role vocabulary.** Stage 2 now has 19 roles to choose from (up from 7): opener, world-setter, groove-locker, early-hook, builder, connector, pivot, pressure, lift, vocal-moment, texture-change, cleanser, risk, weapon, peak, post-peak, resolution, closer, utility. Every track must carry at least one role; multiple roles are allowed.
- **Richer mix reports.** Reports now include: energy path label, section breakdown with track numbers, opener rationale, closer rationale, and excluded tracks with reasons.

---

## v0.4.1

- **Playlist mode duplicate variants fixed.** The selection pass was generating one concept per shortlist instead of three total. Now explicitly tells the model to treat all shortlists as a single combined candidate pool.
- **Playlist mode JSON parse robustness.** When the model added reasoning prose before the JSON (containing `[seed]` patterns), the bracket extractor grabbed the wrong array start and crashed. Selection system prompt now instructs the model to output the opening `[` immediately with no preamble; parser now searches for `[{` as the array-of-objects start to skip false positives in prose.
- **Playlist mode track target raised to 14.** Cap raised from 12 to 14. The seed-retention instruction was rewritten to prioritise arc quality over seed count ("every track must earn its place in the arc") — the minimum seed floor is still enforced deterministically by the Python layer, not the LLM.
- **Strategy dedup guard.** If the model returns duplicate strategy concepts despite instructions, the highest-scoring one per strategy is kept and a diagnostic is logged to stderr.

---

## v0.4.0

- **Stage 2 two-pass split.** Selection and report generation are now separate LLM calls. Pass 1 asks Claude Sonnet to pick and order tracks, returning compact JSON (≤8K tokens, well within the API timeout). Pass 2 fires one Anthropic call per curated concept in parallel to generate the prose mix report. Reports arrive faster and more reliably — the selection call no longer races the API timeout while also writing prose.
- **Parallel report generation.** In non-playlist mode, all concept reports are generated concurrently via `asyncio.gather`. In playlist mode, reports for all variants (practical, balanced, adventurous) are generated in parallel before the winner is selected and the report is finalised.
- **MiniMax removed.** Stage 1 cascade is now Groq → Gemini → Mistral. Stage 2 is Anthropic-only. The `--stage2-provider` flag and `MINIMAX_API_KEY` / `STAGE2_PROVIDER` env vars are removed.

---

## v0.3.1

- **Export unplayed collection.** `--export-unplayed` compares your full Rekordbox collection against your play history and exports every unplayed track as a dated Rekordbox-compatible XML file, ready to import and browse in Rekordbox. Posts the XML attachment to Discord. No LLM calls.

---

## v0.3.0

- **Stage 1 track ID aliasing.** Stage 1 prompts now use short positional aliases (`T001`, `T002`, …) instead of raw track IDs. Hallucinated IDs are structurally impossible — the model can only return aliases that were handed to it — and the real IDs are remapped after parsing.
- **DO NOT RECOMMEND playlist exclusion.** Tracks in a Rekordbox playlist named `DO NOT RECOMMEND` are silently excluded from every run. The crate snapshot shows how many were excluded, and a warning fires if the playlist is missing from the XML.
- **BPM and year range filters.** Four new CLI flags — `--min-bpm`, `--max-bpm`, `--min-year`, `--max-year` — narrow the candidate pool before Stage 1. In playlist mode, filters apply only to library additions and never touch seed tracks. Active filters appear in the Discord crate snapshot label.
- **Catalogue name deduplication.** Stage 2 now receives a list of existing mix names from the catalogue and avoids repeating words, tropes, or phrasing from them. Each concept also carries a `name_reason` field — a one-sentence justification tying the name to the set's thesis rather than individual tracks.

---

## v0.2.0

- **Playlist completion mode.** Pass `--playlist "My Playlist"` to use an existing Rekordbox playlist as the seed. MixLab infers the set's intent, clusters seed tracks into natural BPM zones, and completes the set rather than replacing it.
- **Stage 0 intent analysis.** Before shortlisting, MixLab runs an intent-analysis pass over the seed playlist — extracting the overall vibe, energy shape, and anchor tracks using the same free-provider cascade as Stage 1 (with a deterministic fallback).
- **Three completion variants.** Stage 2 generates `practical`, `balanced`, and `adventurous` variants and auto-selects the strongest based on a DJ practicality score. The report names the rejected alternatives and explains why.
- **Anchor-aware seed retention.** Anchor tracks (the tracks that define the set's identity) are always kept. The retention floor is enforced: 75% of anchors and 40% of supporting tracks must survive into the final concept.
- **Transition analysis.** Each track-to-track move is scored and flagged — risky transitions (chapter pivots, deliberate resets) are named and justified in the report.
