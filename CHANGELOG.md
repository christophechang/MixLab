# Changelog

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
