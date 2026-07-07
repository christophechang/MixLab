# MixLab Audit — 2026-07-06

Full-repo audit at v1.2.0 (`388f01e`). Every source file, test file, and doc was read end-to-end; the claims below that could be verified empirically were verified with reproducible scripts against the installed package (evidence inline). Finding IDs (`B*` bugs, `L*` structural limiters, `D*` data-model gaps, `O*` ops/docs) are referenced by the companion plan (`2026-07-06-mixlab-plan.md`) and by GitHub tickets.

---

## 1. What MixLab is, in one paragraph

A Python 3.12 CLI that ingests a Rekordbox XML collection, optionally subtracts played tracks (via a personal catalog API), clusters the remainder by genre/BPM/key, and asks Claude Sonnet to curate named, sequenced DJ mix concepts with prose reports — delivered to Discord and exportable back to Rekordbox as playlists. Two workflows: **genre mode** (generate 3–6 concepts from a genre pool) and **playlist mode** (complete/extend a seed playlist with three strategy variants and an auto-selected winner).

---

## 2. Architecture

### 2.1 Module map

| Module | Lines | Responsibility |
|---|---|---|
| `__main__.py` | 1118 | CLI, pipeline orchestration for both modes, filters, availability table |
| `reader.py` | 160 | Rekordbox XML → `Track` models; playlist parsing; DnB BPM doubling |
| `matcher.py` | 64 | Played-track exclusion via normalised artist+title key equality |
| `client.py` | 62 | Catalog API (played tracks, mix names) |
| `clustering.py` | 1660 | Genre grouping, BPM pools, **Mix Canvas** build/score/select, deterministic Stage 1 (`partition_pool`) |
| `playlist_mode.py` | 454 | Seed-zone clustering, candidate scoring, deterministic intent signals |
| `llm.py` | 2754 | Stage 0/1/2 prompts + HTTP, parsing, validation, practicality scoring, intent keyword parser |
| `history.py` | 379 | Concept history persistence, track+shape novelty scoring, RECENT CONCEPTS block |
| `playlist_exporter.py` | 226 | Merged Rekordbox XML export |
| `discord_client.py` | 250 | Report formatting, chunked Discord delivery |
| `config.py` / `models.py` / `cache.py` | ~440 | Genre maps, Pydantic/dataclass models, genre-count cache |

Tests: 9,150 lines across 13 files, 648 tests, respx-mocked HTTP throughout. Discipline is genuinely good.

### 2.2 Genre-mode pipeline

```
parse_collection → denylist → BPM correction → BPM/year range filters
  → played/unplayed/all pool (catalog API)
  → genre scoping (GENRE_MAP or CUSTOM_GENRES merge + optional hard BPM window)
  → partition_bpm_pools (core ±6 / bridge ±12 / wildcard) [standard genres only]
  → Stage 1: partition_pool() — deterministic BPM-histogram → Camelot-BFS → era-split → resize
  → build_mix_canvas per shortlist (roles, contrast, risk notes, anchors, era/label)
  → select_canvases — greedy, overlap- and novelty-aware, mode-specific weights, cap 6
  → Stage 2 pass 1 (Sonnet, temp 0.3): selection JSON — 3–6 concepts, one canvas each
  → [--deep] critique pass → Stage 2 pass 2 (parallel): prose report per concept
  → validate_stage2_output (warn-only) → history append → Discord + XML export
```

### 2.3 Playlist-mode pipeline

```
seed playlist resolve → Stage 0 intent brief (free-LLM cascade + deterministic
energy-shape/adjacency/missing-roles) → cluster_seed_zones (BPM gaps >12)
  → per-zone shortlists: seeds + ≤20 scored library candidates (±8 BPM)
  → Stage 2: EXACTLY three variants (practical/balanced/adventurous)
  → deterministic scoring (DJPracticalityScore) → seed-retention floor → winner
  → report rewrite with retention stats → Discord + XML
```

Playlist mode bypasses canvases, canvas scoring, and (mostly) novelty — it loads history only for the RECENT CONCEPTS prompt block.

### 2.4 LLM topology

- **Stage 0** (playlist only): free cascade Groq → Gemini → Mistral → OpenRouter×2, graceful deterministic fallback.
- **Stage 1**: since v0.13 fully deterministic (`partition_pool`); the LLM path survives behind `MIXLAB_STAGE1_LLM=1` (deprecated, ticketed for removal in #46).
- **Stage 2**: Anthropic only, `claude-sonnet-4-6` hardcoded (`llm.py:1905`, `1976`, `2075`, and `_try_anthropic`), two-pass: selection (max_tokens 32768, temp 0.3, timeout 600 s) then one report call per concept in parallel (max_tokens 2048, temp 0.7). No retry, no fallback, no model/temperature configuration.

---

## 3. Data model (Rekordbox XML in)

`reader.py` parses per `<TRACK>`: `TrackID, Artist, Name, AverageBpm, Tonality (Camelot), Genre, Comments (MIK energy 1–8 + /* mood tags */), Label, PlayCount, Year, Album, Remixer, Mix, Colour (→ enrichment confidence)`. Tracks missing BPM or key are excluded with a warning; SoundCloud-location tracks silently dropped. DnB tracks below 100 BPM are doubled.

**Not parsed** (present in the XML per `docs/exploration/h3-cue-grid-prep.md`):

- **D1 — `TotalTime` (duration).** Nothing in the system knows how long any track is. `--mix-length 60` therefore means "≈ minutes/4 tracks" (`llm.py:2550`), a prompt-level heuristic. Real set-length planning is impossible without this one attribute.
- **D2 — `<TEMPO>` beat grid and `<POSITION_MARK>` cues/loops.** Already investigated (h3 doc, issue #41): intro/outro lengths and mix-point data exist in the file and would materially change transition reasoning.
- **D3 — `DateAdded`, `Rating`.** "Recently added" is a natural creative axis (fresh-crate spotlight) and rating is an explicit user preference signal; both are free.
- **D5 — matching is exact, not fuzzy.** `matcher.normalise()` (unicode/dash/feat./version-suffix stripping) then **set equality**. README calls it "fuzzy-matched" — it isn't; one typo in the catalog and a played track leaks into the unplayed pool. The v0.12.3 changelog shows this class of miss has already bitten (17 tracks).

Mood tags (`/* dark / driving */`) are parsed into `Track.tags` but are used **only** as prose passthrough in prompts — they never inform clustering, scoring, contrast detection, or novelty (see L5).

---

## 4. The AI/prompt flow, examined closely

### 4.1 Stage 2 selection prompt

`_STAGE2_SYSTEM` (llm.py:1057–1208) is a single ~1,900-word system prompt carrying: selection rules, opener/closer specs, thesis requirement, 10-role vocabulary, seven energy paths, five sections, peak-budget rule, vocal/bass/blend-window awareness, bold-move mechanics, metadata legend, distinctiveness test, output schema, naming rules. Genre mode appends `_STAGE2_CANVAS_RULES` (11 more rules) and a mode fragment; used mix names are spliced in via a **sentinel-string replace with an `assert`** (llm.py:2574–2587) — same pattern builds the playlist variant at import time. This works but is brittle and hard to reason about; the prompt has accreted rule-by-rule across 13 releases and rules now compete for attention (momentum vs. bold moves; hints vs. instincts; "8–12 tracks" vs. genre targets — see B8).

### 4.2 What Stage 2 receives

Per canvas: header (novelty score, core/bridge/wildcard ID lists, role-candidate pools, anchors, era window, dominant label, concept-anchor tags, contrast pools, risk notes) + full track listing with metadata. Plus a RECENT CONCEPTS block (last 5 runs) and, in genre mode, the verbatim `--intent` text with heuristically parsed signals.

Notable: role-candidate hints come from `_infer_roles` (energy-threshold heuristics). The project's own v0.12 release notes (#27) record that these hints "disagreed with Stage 2's textual role picks on virtually every real-run concept" — disagreement so complete that the validator checks based on them were deleted. The hints are still injected into every prompt (**L4**).

### 4.3 Parsing and safety

Selection JSON parsing is robust (fence stripping, `[{` anchoring, control-char repair, truncated-array object recovery). But `_parse_curated_concepts` validates returned IDs against **the entire library** (`valid_ids = set(tracks_by_id.keys())`, llm.py:2594), not against the canvases actually offered. A hallucinated-but-real track ID passes silently; the only backstop is the warn-only wildcard check, which covers only tracks that happen to be in some canvas's wildcard list (**B7**). The Stage 1 alias trick (T001…) that made hallucination structurally impossible was never applied to Stage 2.

### 4.4 Validation and self-correction

`validate_stage2_output` is warn-only by design (documented philosophy, sensible). But nothing ever **acts** on the warnings: no revision pass, no retry, no scoring consequence. Same for `--deep` critiques — verdicts of "weak" with named structural issues and even a `suggested_substitution` are printed and discarded (**L13**). The only closed loop in the system is the playlist-mode seed-retention floor retry.

### 4.5 History and novelty

Each run appends a `HistoryEntry`; canvas novelty = 0.65·track-Jaccard + 0.35·shape-similarity, decayed 0.8^age over a 10-run window. Two problems:

- **B6 — history records only `concepts[0]`.** `HistoryEntry.from_run` (history.py:115–129) takes title/mood/track_ids/arc from the *first* concept only. A run producing 5 concepts contributes one to RECENT CONCEPTS and one to shape novelty; the other four are invisible. The "avoid retreading" block therefore under-reports what was actually generated by ~4×.
- **L8 — novelty lost most of its power when Stage 1 went deterministic.** Novelty is a *canvas-selection* penalty. With `partition_pool` producing identical canvases every run and `select_canvases` returning **all** candidates whenever ≤ 6 exist (clustering.py:1046–1055 — score order is computed, then everything is forwarded anyway), the penalty changes nothing for typical standard-genre runs. The mechanisms designed for a stochastic Stage 1 (novelty, distinctiveness re-scoring) now mostly re-rank a fixed set that gets sent regardless. The only remaining diversity levers are the RECENT CONCEPTS prompt block and selection-pass temperature 0.3.

---

## 5. Verified bugs (with evidence)

### B1 — `partition_pool` sizing contract collapses on large pools

Spec (`docs/architecture/deterministic-stage1.md`) promises 3–5 shortlists of 15–25 tracks, tolerating "≤ 4 over" in narrow cases. Reality, on an 800-track pool shaped like a real `4x4` run (Gaussian BPM mixture 125/132, random keys/energies/years):

```
pool=800 → 5 shortlists, sizes [160, 160, 160, 160, 160], 100% of pool → Stage 2
```

Root cause: the constraint set is infeasible above ~125 tracks (5 pools × 25 max), and `_resize_shortlists` resolves the conflict in the wrong direction — Attempt-4 trims push remainders into neighbours, the pool-level merge (which runs *after* size enforcement, clustering.py:1566–1594) re-merges everything to ≤ 5 pools with no size cap, and the 3-iteration loop gives up. Consequences: the "shortlist" premise is void for custom genres (Stage 2 receives the entire pool as ~20k tokens of track lines), canvas semantics degrade (a 160-track "canvas" has meaningless dominant-BPM/key and diluted role pools), and cost scales with library size. The 4x4 README section explaining the old 120-track random window is now fiction (see O1).

### B2 — Zero run-to-run variety; `seed` is a no-op

Same script, run twice, `seed=None` vs `seed=123`: **byte-identical output**. `partition_pool` ignores `seed` (`_ = seed  # reserved`, clustering.py:1609). Combined with deterministic canvas selection, a given library+genre+mode produces the same canvases with the same candidates every single run. The README's promise that repeated runs "explore different corners of the collection" (README §custom genres, and the `--help` epilog at `__main__.py:882–884`) has been false since v0.13. Only Stage 2 sampling temperature separates run N from run N+1.

### B3 — Vocal detection false-positives on any word containing "ft"

`_has_vocal_token` (clustering.py:257–259) does substring matching over `{"feat.", "ft.", "feat", "ft", "vocal", "vocals", "w/"}`. Verified:

```
'Afterlife' → vocal=True   'Left Field' → vocal=True   'Soft Focus' → vocal=True
'The Loft'  → vocal=True   'Swift Motion' → vocal=True
```

Every such title lands in `contrast.vocal_moments`, inflating `contrast_potential` scoring and feeding Stage 2 wrong "Vocal:" hints — directly corrupting the vocal-clash reasoning the prompt asks for.

### B4 — `run_export_unplayed` fails before its own error message on clean checkouts

It calls `parse_collection(_XML_PATH)` (`__main__.py:433`) before checking `CATALOG_API_URL`, so without `import/rekordbox.xml` the run dies with `FileNotFoundError` instead of the intended clean exit. Consequence: `test_run_export_unplayed_exits_without_catalog_url` **fails on a fresh clone** (verified: 1 failed / 647 passed) and only passes on machines where the gitignored XML happens to exist. Env checks should precede file parsing, and the test should not depend on repo-external state.

### B5 — Playlist-mode unplayed bonus is 0.045, not the documented 0.15

`_score_candidate` (playlist_mode.py:377–379): `unplayed_bonus = 0.3`, then `... + unplayed_bonus * 0.15` → effective max contribution 0.045 against a documented weight of 0.15. The unplayed bias in zone shortlists is ~3× weaker than designed (and than the docstring claims).

### B6 — History records only the first concept per run

See §4.5. Also affects shape-novelty: `energy_path`/`role_pattern` come from `concepts[0]` only.

### B7 — Stage 2 track-ID validation scoped to library, not offered canvases

See §4.3. A plausible-ID hallucination (real track, wrong pool — e.g. a played or out-of-genre track) enters the concept and at best triggers a soft warning.

### B8 — Track-count targets contradict the prompt

Prompt: "SELECT the best 8–12 tracks" (llm.py:1063). Config: Jungle target (12, 16), DnB (10, 14), UKG (10, 13) (`config.py:57–66`); the validator warns below min. A jungle run that obeys the prompt's 8–12 instruction is structurally guaranteed warnings; a 14-track DnB set the config *wants* violates the prompt's cap. Genre-specific targets never reach the prompt.

### B9 — Report pass can truncate long reports

`_call_stage2_report_single` caps `max_tokens=2048` (llm.py:1976). A 22-track `--mix-length 90` completion report (per-track lines + sections + excluded list) plausibly exceeds this; truncation is silent (no stop-reason check anywhere in `_call_anthropic_http`).

### B10 — Dead code and stale metadata

- `build_playlist_pool` (playlist_mode.py:241) has no callers outside tests (superseded by `build_zone_shortlists`); `filter_by_bpm` (clustering.py:183) likewise.
- `pyproject.toml` `version = "0.12.3"` while CHANGELOG/tags are at v1.2.0.
- `anthropic` is a declared dependency but never imported (raw httpx is used).
- `mixlab.skill.json` still documents the removed `--stage2-provider minimax` flag.

---

## 6. Structural limiters on playlist quality and creativity

These are the things that cap how good the output can get, independent of bugs.

### L1 — Concepts are tempo strata, not ideas

The generative axis of the whole system is BPM. `partition_pool` clusters by BPM histogram, then Camelot connectivity, then (rarely) era. Canvases inherit those strata; Stage 2 is instructed "each concept must draw only from tracks within a single canvas" (llm.py:2536). So the concept space is literally: *one concept per tempo/key band*. Titles produced by real runs confirm it ("Deep 122 BPM / 4A–7A Pool" → concepts distinguished mostly by band). Everything a DJ would recognise as a *concept* — a mood journey, a label retrospective, an era dialogue, one artist threaded through a set, a dark-to-euphoric arc that crosses 10 BPM — is either accidental within a band or structurally impossible across bands. Issue #44 (cross-genre bridging) gestures at this but frames it as a genre problem; it is really a **one-pool-per-concept** problem.

### L2 — Playlist-mode winner selection is anti-adventurous by construction

The three variants are scored by `DJPracticalityScore` = 0.30·bpm_smoothness + 0.30·harmonic_ratio + 0.25·risk_justified + 0.15·fragment_preserved, and `_select_best_variant` takes the max with ties broken practical > balanced > adventurous (llm.py:1802–1810). Every component is maximised by *not taking risks*: the metric measures exactly the properties the adventurous variant is instructed to trade away. The adventurous variant can effectively never win unless the practical one is broken. The system generates creativity and then deterministically discards it. `IntentBrief.risk_tolerance` is computed (Stage 0) and never consulted here.

### L3 — Creativity throttled at the point it happens

All creative decisions (track choice, order, naming, arc, risk-taking) happen in the selection pass at **temperature 0.3**; the report pass (which merely narrates decisions already made) runs at 0.7. That allocation is backwards for variety. Neither value, nor the model, is configurable (L12/B-adjacent: `claude-sonnet-4-6` hardcoded in four places).

### L4 — Known-noisy role hints still injected

See §4.2. Either the heuristics should improve (use tags, position-in-pool, remixer) or the hints should be dropped/demoted — shipping hints the model demonstrably ignores wastes prompt budget and dilutes the signals that matter.

### L5 — The richest human signal (mood tags) is inert

The user hand-curates a 27-word mood vocabulary (`dark`, `driving`, `dreamy`, `raga`…). Tags appear as trailing prose per track line and in `_infer_shortlist_mood` titles — and nowhere else. No mood-coherence clustering, no mood-contrast assets (`darker_turns` uses energy only), no mood axis in shape-novelty, no mood-led concept briefs. Meanwhile the system computes elaborate proxies (label share, era windows, energy medians) for character information the user has *already written down*.

### L6 — No deterministic arc design or verification

Energy arcs are chosen and asserted by the LLM ("the chosen shape must be visible in the track sequence") and checked only by three soft warn-only heuristics with broad genre suppressions. Energy values per track exist; nothing traces the declared `arc_type` against the actual energy sequence, at selection time or after. A "wave" concept that is actually monotonic ships without comment in house/techno/DnB (all suppressed families).

### L7 — No tempo-relationship knowledge

Compatibility is absolute-BPM proximity everywhere (±6/±12 pools, ±8 zone gaps, 15-BPM jump warnings). Half/double-time relationships (85↔170, 70↔140) — a foundational DnB/hip-hop/UKG mixing technique, and the *point* of several genres in this library — do not exist anywhere in the model. A 172 DnB track and an 86 downtempo track are maximally incompatible to every scorer, while a real DJ hears them as locked. Same for the ±6% pitch-fader window. This forecloses whole families of transitions and makes genuinely creative cross-tempo concepts unreachable.

### L8 — Diversity machinery outlived the randomness it was built for

See §4.5. Novelty, distinctiveness, and mode-aware weights all act on canvas *selection* — a stage that, post-v0.13, has almost nothing left to select. When ≤ 6 canvases exist, all are forwarded regardless of score.

### L9 — No user steering in playlist mode

`--intent` is genre-mode only and hard-rejected in playlist mode. Stage 0 infers intent *from the seed*, which is useful but different: "complete this warm-up playlist, but make the back half peak harder than the seeds suggest" cannot be expressed at all.

### L10 — No set-length control in genre mode; no duration awareness anywhere

`--mix-length` exists only in playlist mode and works in track-count heuristics because D1 (TotalTime unparsed). Genre-mode concepts are pinned to 8–12 tracks (B8) regardless of purpose.

### L12 — Single-provider Stage 2 with zero resilience

One 429/529/overload on the selection call → `RuntimeError` → entire run lost after Stage 0/1 work is done. No retry, no backoff, no `stop_reason` check, no model fallback. For a tool whose runs cost minutes and real tokens, this is the sharpest operational edge.

### L13 — All evaluation is advisory; nothing loops

Validation warnings, critiques, practicality scores: computed, printed, never fed back (§4.4). The architecture already has every ingredient for one bounded self-revision pass; it just never takes it.

---

## 7. Strengths worth preserving

- **The canvas idea itself** — deterministic material-preparation between clustering and curation is the right shape; the problem is what the canvases *are* (tempo strata), not that they exist.
- **Two-pass Stage 2** (selection JSON then parallel prose) — good latency/reliability trade.
- **Stage 1 alias mapping** (T001…) — structurally eliminates hallucination; should be extended to Stage 2, not retired.
- **Warn-only validation philosophy** — right default; it needs a consequence path, not a mode change.
- **Forward-compatible history schema**, provider cascade with per-run state, honest stderr diagnostics, `--debug` scoring transparency.
- **Test discipline** — 648 tests, respx everywhere, adversarial edge-case coverage; and docs (`mix-canvas.md`) that actually match the code.

## 8. Docs/ops drift (O*)

- **O1** — README: "random 120-track window" section (§custom genres) and `--help` epilog describe pre-v0.13 behaviour; "fuzzy-matched" overstates matcher; Stage 2 model line will drift the moment the model is configurable. `pyproject` version stale (B10).
- **O2** — `mixlab.skill.json` references removed flags (`--stage2-provider minimax`).
- **O3** — `docs/notes/intent-usage.md` gate counters stopped updating pre-v1.0; harmless but stale.
- **O4** — Baseline check on clean clone: ruff ✅, mypy --strict ✅, pytest **1 failed** (B4) / 647 passed.

## 9. Summary of findings

| ID | Severity | One-liner |
|---|---|---|
| B1 | High | Stage 1 sizing collapses on large pools (5×160-track "shortlists", 100% of pool → Stage 2) |
| B2 | High | Zero run-to-run variety; `--stage1-seed` is a no-op; README promises otherwise |
| B3 | Medium | Vocal detection substring bug — "Afterlife" is a vocal moment |
| B4 | Medium | `--export-unplayed` env-check ordering; test suite fails on clean clone |
| B5 | Low | Playlist unplayed bonus 0.045 effective vs 0.15 documented |
| B6 | Medium | History/novelty sees only the first concept of each run |
| B7 | Medium | Stage 2 IDs validated against whole library, not offered pools |
| B8 | Medium | Prompt's "8–12 tracks" contradicts per-genre targets (jungle 12–16) |
| B9 | Low | Report pass max_tokens 2048 can silently truncate long sets |
| B10 | Low | Dead code, stale version/deps/skill file |
| L1 | High | Concepts = BPM strata; one-pool-per-concept caps creativity |
| L2 | High | Playlist winner metric structurally eliminates the adventurous variant |
| L3 | Medium | Creative pass at temp 0.3; model/temp hardcoded |
| L4 | Medium | Known-noisy role hints still injected |
| L5 | High | Mood tags parsed but inert in generation |
| L6 | Medium | No arc design/verification against energy data |
| L7 | High | No half/double-time or pitch-window tempo knowledge |
| L8 | Medium | Novelty/diversity machinery no-ops post-deterministic Stage 1 |
| L9 | Medium | No `--intent` in playlist mode |
| L10 | Medium | No genre-mode length control; no duration data (D1) |
| L12 | Medium | No Stage 2 retry/fallback; hard model pin |
| L13 | Medium | Warnings/critiques/scores never feed back into output |
| D1–D5 | — | TotalTime, cues/grid, DateAdded/Rating unparsed; matcher exact-only |
| O1–O4 | — | README/skill/docs drift; failing baseline test |
