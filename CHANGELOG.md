# Changelog

## Unreleased

## v1.18.0 — 2026-08-09

- **New `--track-pool` flag restricts a genre run to an operator-chosen block of
  tracks.** Takes a JSON id list (plus an optional label) from mixlab-web's "Run this
  block" — machine-generated, so ids are authoritative: a conflicting `--mode` is
  overridden to `all` rather than rejected, `--min-bpm`/`--max-bpm`/`--min-year`/
  `--max-year` are ignored outright (a stderr note names which were passed) rather than
  running before id resolution and silently eating block tracks, and the restriction
  lands before Stage 1 so downstream clustering, directions, and the report all see
  only the resolved block. Unknown ids are dropped silently; fewer than 15 resolved
  (the Stage 1 partition floor) is fatal — the block is stale against the current
  collection and needs re-creating. A second floor catches a block that spans genres:
  if the block intersected with `--genre` still has fewer than 15 tracks, the run
  fails fast naming the genre and count rather than shipping a thin shortlist. The
  label-lift baseline for directions stays the whole pre-restriction, mode-scoped pool
  (all genres) even for a block run, so a block confined to one genre doesn't collapse
  a label's identity signal to zero. Combining with `--direction-spec` (a different,
  incompatible scoping mechanism) is also fatal. Ignored (warn-and-drop) in playlist
  mode.

## v1.17.0 — 2026-08-08

- **Pinned runs are now a study of the direction, not a menu around it.** Operator call:
  when intent comes from the library map, quality beats quantity. A pinned run now ships
  the pinned canvas plus **one** classic canvas (was five), and Stage 2 is asked for two
  or three **contrasting readings** of the pinned direction — same spine and brief every
  time (key tracks enforced per reading), different companions, arc, and sequencing —
  borrowing playlist mode's practical/balanced/adventurous framing, plus at most one
  contrast concept from the classic canvas. Non-pinned runs keep their exact prompts and
  canvas mix.

## v1.16.0 — 2026-08-08

- **Pinned directions now enforce their defining tracks.** First pinned production run
  shipped an artist thread whose thesis said "three Dusky tracks" while the tracklist
  held two — Stage 2 curated away a pillar and kept the language. Every direction
  builder now declares its defining subsets as `key_groups` (the spine of an artist
  thread — all of it; both poles of a mood journey; both sides of an era dialogue; a
  label's catalogue slice; an energy arc's troughs and crests; a found set's
  conjunction members; fresh_crate and genre_traverse deliberately declare none), the
  map payload and `--direction-spec` round-trip them, and a single generic validator
  turns a shortfall on a **pinned** canvas into a hard finding that alone qualifies
  the concept for the existing self-revision repair pass — the missing tracks are
  named so the repair can put them back or rewrite the thesis honestly. Requirements
  are clamped to what actually ships (and to what still resolves after collection
  drift), so a group can never be unsatisfiable by construction. Enumerated daily-run
  direction canvases carry `key_groups` too but are not yet enforced — the gate is
  pinned-only so existing runs keep their exact behaviour and cost.

## v1.15.0 — 2026-08-08

- **`--direction-spec` — "Run this direction" actually runs the direction.** The web app's
  library-map "Run this direction" used to hand over only the genre and the brief-as-intent;
  the run then re-enumerated directions from a different seed over a different pool, so the
  clicked direction (its tracks included) almost never materialised — an artist-thread pick
  could ship with none of the thread's tracks. The map's direction entry now round-trips as a
  JSON spec (`directionSpec` on the run manifest → `--direction-spec` in the worker argv):
  the pipeline materialises those exact tracks, brief, and type as a **pinned canvas**, the
  Stage 2 prompt marks it and mandates a concept from it (the skip-a-weak-canvas allowance
  explicitly does not apply), and a deterministic post-check logs a warning if Stage 2
  somehow skipped it anyway. The pinned canvas replaces the enumerated-directions slot
  (5 classic canvases + the pin); `--directions` off/only are superseded with a stderr note,
  and playlist mode ignores the flag like the other genre-only levers. Map payload direction
  entries now also carry `thread_artist` so artist-thread pins keep their spine for the
  repeat-suppression validator; specs from older payloads derive it from the title. A spec
  whose pinned tracks no longer resolve (stale analysis after collection drift) fails the
  run with a message telling the operator to re-run the analysis, rather than silently
  shipping a run without the chosen tracks. All existing paths are byte-identical when the
  flag is absent.

## v1.14.0 — 2026-08-08

- **Conjunction mining: directions the vocabulary has no name for (#53 follow-up).** Each
  genre pool is now scanned for dense two-predicate conjunctions (label × era, tag × tempo
  pocket, remixer × key neighbourhood, …) and the ones materially denser than chance —
  lift ≥ 1.3 over ≥ 15 shared tracks, subsumed pairs pruned — ship as
  `found_1`/`found_2`/`found_3` directions with their own briefs. BPM and key predicates
  can be mined but never named, so no title or mood carries DJ mechanics (the v1.13.2
  rule). The library map shows up to three per pool; a run materialises at most one, so a
  discovery adds to the day's variety instead of taking it over.
- **Direction feasibility means something again.** The old score was pool fill, BPM-path
  viability and a type-specific signal — the first two are 1.0 for every candidate that
  gets built, so 44 of 49 live rows landed in [0.70, 1.00] and the ranking carried almost
  no information. Pool size and path viability are now pass/fail gates, and feasibility is
  `0.25·freshness + 0.45·identity + 0.30·distinctiveness`: how recently the shipped tracks
  were added (a rank percentile, so an all-unplayed pool still spreads), how strong the
  defining idea is (per-builder renormalisation — pole balance on untruncated counts,
  collection-relative label lift, mined lift on a log scale, fresh_crate's recency
  concentration), and how little the shipped tracks overlap the pool's other surviving
  directions. Named and mined candidates go through one scorer and one dedupe, so a found
  set and a label spotlight compete on equal terms. Payload shape is unchanged;
  `feasibility` keeps its name and 0–1 range, so mixlab-web and the API need no changes.

## v1.13.2 — 2026-08-07

- **Concept titles stop naming the craft.** Operator feedback: titles built from DJ
  mechanics ("The 139 Dip", "Fader Down at Six") are unwanted — nobody cares about the
  BPM or the fader. The number/BPM naming lens is gone, the booth-object lens now points
  at the dancefloor rather than the equipment, both Stage 2 prompts carry an explicit
  ban (no BPM numbers, no keys/Camelot codes, no equipment or mixing moves), and a
  deterministic warn-only guard flags any mechanics-built title that slips through.

## v1.13.1 — 2026-08-07

- **Worker log lines carry timestamps.** Every line the worker itself writes (online banner,
  sync reports, map-job status, transport and unexpected-error reports) is now prefixed
  `[YYYY-MM-DD HH:MM:SS]`. During the 2026-08-07 network outage the untimestamped error log
  made it impossible to tell live failures from historical noise without file-mtime forensics;
  outage windows are now readable straight from the log.

## v1.13.0 — 2026-08-06

- **Worker now claims and runs library-map jobs (#40, Milestone B).** `mixlab --worker` polls
  `/api/mixlab/maps/claim` once the run queue is empty, downloads the job's collection to its
  own isolated path, executes `mixlab --map --mode unplayed` as a subprocess, and pushes the
  raw JSON stdout back via `/api/mixlab/maps/{uploadId}/result` — closing the loop between the
  API's maps endpoints and the web app's constellation overlay. Run jobs keep priority (the map
  queue is only checked when the run queue is empty); a non-zero exit or any unexpected error
  after the claim best-effort reports failure via `/api/mixlab/maps/{uploadId}/fail` so the job
  doesn't hang the API-side lease.

## v1.12.0 — 2026-08-06

- **`mixlab --map` — library-map payload for the web app (#40, Milestone B).** Emits a
  deterministic JSON map of concept-direction candidates per engine pool (all standard +
  custom pools, exhaustive builder enumeration with feasibility scores, no LLM calls),
  honouring `--mode` and the do-not-recommend playlist. Consumed by mixlab-web's
  constellation overlay; contract lives in mixlab-web's library-map design spec §7.

## v1.11.2 — 2026-07-16

- **Playlist-mode runs now get a real `conceptId`.** `run_playlist_mode` never stamped `concept_id` — only genre mode's `run()` did — so every playlist-mode `summary.json` since v1.11.1 shipped `"conceptId": ""`. The web app's Save feedback button submitted that empty id as a URL path segment (`.../concepts//feedback`), which 404'd before ever reaching the concept-lookup logic. `run_playlist_mode` now mints a `concept_id` immediately after Stage 2 returns, matching genre mode.

## v1.11.1 — 2026-07-15

- **Playlist-completion runs no longer get misreported as failed.** The worker requires a `Run summary: <path>` line in the pipeline's stdout to mark any run complete, but `run_playlist_mode` never printed one — only genre-mode runs built the `summary.json` artifact. Every successful playlist-mode run through MixLab Anywhere was therefore reported to the API (and shown in the web app) as failed, even though the report, export, and Discord delivery all completed. `run_playlist_mode` now builds and prints the same `Run summary:` artifact genre mode does, reusing `build_run_summary` unchanged since playlist mode already produces the same `MixConcept` data.

## v1.11.0 — 2026-07-15

Worker support for playlist-completion mode and BPM/year filters driven from MixLab Anywhere. The CLI already accepted these flags; this release lets a queued run manifest actually carry them through to `./mixlab`.

- **`build_argv` allow-lists five new run flags.** The worker's strict manifest→argv mapper now maps `playlist` → `--playlist`, `minBpm`/`maxBpm` → `--min-bpm`/`--max-bpm`, and `minYear`/`maxYear` → `--min-year`/`--max-year`. Previously any of these keys in a run manifest was rejected outright as an unknown flag, so playlist-seeded and BPM/year-filtered runs could only be started from the command line, never from the web app.
- **New `"float"` flag kind.** BPM is a float on the CLI, but the mapper only understood `str`/`enum`/`int`/`bool`. The new kind accepts an int or a float, rejects booleans (which are ints in Python and would otherwise smuggle through as `1`/`0`), and formats with `str()` — so `--min-bpm 128.5` survives the round trip intact.
- **Seed retention is now covered by a regression test.** `--playlist` runs apply BPM/year range filters only to the *library* tracks that get added around the seed; the seed playlist's own tracks are always kept even when they fall outside the range. That invariant was previously untested, and the web app's filter copy now promises it, so a test pins it in place.
- **Housekeeping.** The `CLAUDE.md` release runbook is condensed to defer to the shared user-level `deploy-release` skill rather than restating the whole procedure, and `.gitignore` now covers `.claude/settings.local.json`.

## v1.10.2 — 2026-07-12

Repository and release-process housekeeping — no runtime behaviour change.

- **Deploy release process is now comprehensive end-to-end.** The `CLAUDE.md` runbook gained the previously-missing `develop → main` merge, moved tagging and GitHub-release creation onto `main` (matching where tags actually live), and added a "return to develop" step. It now also restarts the MixLab Anywhere worker after the production pull — the long-running `mixlab --worker` daemon loads its harness code once and does not hot-reload, so a deploy that skipped the restart left it running stale code — plus a post-restart health check (launchd status + fresh "Worker online" banner + clean stderr).
- **Docs consolidated under `docs/`.** Planning docs and the `mixlab.skill.json` manifest (now carrying the MiniMax Stage 2 command patterns) moved out of the repo root and the ignored `plans/` directory into `docs/`.

## v1.10.1 — 2026-07-11

- **Energy-arc sparkline in the HTML report now matches the web app.** The report used to auto-scale each concept's energy arc to its own min/max, so a steady set living in a narrow band (e.g. all 5–7) was stretched into a full-height canyon that read as wild swings. The y-domain is now a min-span window — the concept's energy range plus one level of padding, floored to 4 levels and centred on the data — with faint dashed guides at the MIK band edges (6 danceable, 8 high-intensity) that keep the absolute level legible under the re-centred window (`html_report.py`). The sparkline also adopts the web component's visual language: per-track dots, genuine line breaks at unrated tracks (no interpolation), and true set-position spacing. Shared domain logic is mirrored 1:1 with `mixlab-web`'s `energyArcDomain` so the report and the app draw the same shape.

## v1.10.0 — 2026-07-11

- **Energy handling aligned with Mixed In Key's official 1–10 scale.** The reader always parsed MIK's 1–10 energy values, but every prompt, warning, and doc described a 1–8 scale — so the LLM read energy 7 as near-maximum when MIK considers it merely "upbeat", and treated 8–10 as out of range. All Stage 0/1/2 prompt surfaces now show `energy:N/10` with the official band meanings (1–2 very chill/atmospheric, 3–5 lounge/smooth groove, 6–7 danceable/upbeat, 8–10 high intensity), and `config.py` gains a canonical `MIK_ENERGY_BANDS` / `energy_band_label()` / shared prompt-guidance definition.
- **Stage 2 curation now follows MIK's mixing rule of thumb.** Craft rules instruct the model to move at most one energy level between consecutive tracks by default (5→5 steady, 5→6 lift) and to treat any 2+ level jump as a deliberate, named reset/contrast moment; the deep-critique pass flags unowned 2+ jumps.
- **Transition scoring recalibrated to the same rule.** `transitions._energy_component` now scores ±1 as ideal (1.0), a 2-level move as a deliberate lift (0.75), 3 as a reset (0.5) and 4+ as a slam (0.3) — previously anything ≥3 collapsed to a flat 0.5.
- **Two energy-band bugs fixed.** The anchor-scoring energy signal gave genuine high-intensity (8–10) tracks a score of 0.0 — a leftover from reading the scale as 1–8 — and now scores them highest; the peak role threshold moved from ≥6 to ≥7, since 6 only "starts to feel danceable" on the official scale (kept in lockstep with concept-anchor peak tagging).
- **Structural warnings re-tuned.** A level-5 close no longer trips the "no wind-down" warning (5 is MIK's lounge/smooth-groove band — a legitimate landing); the all-high-energy warning is reworded to "danceable-or-hotter (≥6/10)". The exported HTML report's energy cells now carry the official band as hover text.

## v1.9.2 — 2026-07-09

- **Worker no longer shares the local collection file.** The MixLab Anywhere worker now downloads each run's collection to `.mixlab/worker-collection.xml` (was `import/rekordbox.xml`) and points the pipeline at it via a new `MIXLAB_COLLECTION_PATH` env var, so a remote run can never overwrite the `import/rekordbox.xml` a local run or scheduled automation uses. Normal CLI usage is unchanged (the env var is unset). Configurable via `MIXLAB_WORKER_XML_PATH`.

## v1.9.1 — 2026-07-09

- **Remote worker now uploads the Rekordbox playlist export.** The worker runs the pipeline with `--export`, so every remote run writes the merged `rekordbox_export.xml` and passes it to the API's `complete` endpoint — the SPA's "Download export.xml" button now works instead of reporting "no export for this run." The export is optional (a run with no concepts produces none), so completion still succeeds when there's nothing to write. Past runs stay export-less; the change applies to new runs after the worker pulls this release.

## v1.9.0 — 2026-07-09

MixLab Anywhere: the engine can now run as a pull-based remote worker for the Changsta API, so runs can be triggered, archived, and fed back from mixlab.changsta.com. Strictly additive — every existing CLI path is unchanged.

- **`--worker` / `--worker-once`** (`worker.py`): crash-safe poll loop that claims queued runs from the API, downloads the collection, executes the normal pipeline as a subprocess with a hard timeout, and uploads the HTML report + run summary. Transport failures leave the claim to lease expiry (the API requeues); pipeline failures are reported with a bounded log tail. SIGTERM finishes the current cycle before exiting.
- **`remote.py`**: typed client for the MixLab Anywhere API — claim, collection download (streamed gunzip, atomic write), complete (multipart), fail, ETag-guarded history get/put, pending-feedback fetch/ack.
- **`sync.py`**: history + feedback sync around each run. Web-recorded verdicts/ratings flow through the existing `apply_feedback_verdict` machinery, so novelty multipliers and the name-avoid list react exactly as if `--feedback` had been used locally. Conflicts merge via ETag retry. Applied feedback is summarised in the worker log (`sync: applied N verdict(s), …`).
- **Docs & ops**: README worker section, `docs/ops/worker-launchd.md` (Mac mini launchd runbook), `docs/ops/anywhere-smoke.md` (eight-step end-to-end smoke checklist), flags-guide entries, `.env.example` worker variables.
- The unit suite grew from 1101 to 1167 tests.

## v1.8.4 — 2026-07-08

The second (and decisive) half of making genre_traverse fire — v1.8.3 shipped only the chain-start fix; the verification log proved it insufficient on its own.

- **Regime detection now uses tempo-density peaks.** Gap-based regime splitting needed a >12 BPM hole between sorted neighbours, and a full collection's BPMs form a near-continuum with no holes — the builder saw one giant regime and bailed before the chain logic ever ran. Regimes are now density peaks: a smoothed 1-BPM histogram, peaks picked by mass with ≥20 BPM separation, regime = tracks within ±8 BPM of the peak. In-between material simply doesn't join a chapter. Regression test uses exactly the failing shape: a 77–180 continuum plus heavy house/DnB concentrations.
- **Direction observability.** The run log now prints `Directions proposed: … (N/7 builders)` so a silently non-firing builder is visible immediately — this took four production runs to notice.

---

## v1.8.3 — 2026-07-08

- **Fix: the genre_traverse chain was anchored to the lowest tempo regime.** The chain-builder started at the lowest-BPM regime and skipped everything unreachable from it, so a low block with no ratio partner could starve the whole direction even with plentiful 126↔168 bridges above. The builder now tries every regime as the chain start and keeps the best chain — most chapters, ties broken by total material (avoiding chains that die at the 15-track pool floor), then earliest start. (Necessary but not sufficient — see v1.8.4.)

---

## v1.8.2 — 2026-07-08

- **Regime-crossing awareness for every concept.** The first badge-equipped run (v1.8.1) revealed that other direction types (fresh_crate, artist_thread) build traverse-shaped sets over cross-genre pools too — the traverse-scoped hard finding couldn't see them. Non-traverse concepts now draw a warn-only "regime crossing without a ratio bridge … plan a cut or a reorder" note for incompatible crossings — fired only when the BPM-jump warning doesn't already cover the pair (justified-risk or sub-threshold), so no pair ever draws two warnings, and never a hard finding.

---

## v1.8.1 — 2026-07-08

Four tuning fixes from the first two production traverse runs.

- **Direction badges on report cards.** Each concept card now shows which direction produced it (dashed badge next to the arc: `genre_traverse`, `artist_thread`, …) — a three-regime concept with no bridges appeared in a live report and its origin couldn't be identified. Classic-canvas concepts show no badge.
- **Unbridged regime crossings are a hard finding for traverse concepts.** A traverse-brief concept shipped spanning 77–174 BPM with raw jumps across regimes. Crossings over 12 BPM whose tempo relation is incompatible now warn ("unbridged regime crossing … traverse hops must be ratio bridges"), count as hard findings (triggering self-revision), and are never suppressed by risk annotations.
- **Thread-artist cap 3 → 5 with a clearer message.** An artist-thread concept ran its spine artist on 11 of 14 tracks; the generic repeat warning read as noise. The spine artist is now quiet up to 5 tracks and over-cap draws "thread artist 'X' appears N times — thread cap is 5, trim the spine".
- **Traverse track limits (8, 16).** Journey concepts legitimately need more than the default 12 tracks — 3+ per chapter across up to 4 regimes plus bridges.

---

## v1.8.0 — 2026-07-07

- **Genre Traverse (#82).** A seventh concept direction: cross-genre journey sets that travel between tempo regimes (house → UKG → jungle/DnB) via pitch-locked ratio bridges. The builder splits the pool into BPM regimes (sorted-neighbour gap > 12), chains regimes reachable through verified halftime/double-time/3:4/4:3 bridge pairs (≥2 per hop, Camelot-compatible preferred, unreachable regimes skipped), seed-flips climb vs descend, selects bridge endpoints plus centrality fill per chapter, and writes a DIRECTION BRIEF naming each hop's bridge pairs with their mechanisms. Fires only when the material genuinely supports it — single-regime pools (every standard-genre run) are unaffected. New `traverse` custom genre label pools the whole collection to feed it: `./mixlab --genre traverse`.
- **Flags guide.** `docs/flags-guide.md` — a tutorial covering every lever by use case (gig prep, playlist completion, cross-genre journeys, the cue-prep loop, feedback verdicts, reproduction/debugging, exports), with a quick-reference table and the flag-composition rules. Linked from the README.

---

## v1.7.5 — 2026-07-07

- **Title re-roll (#75).** Prompt pressure alone cannot stop exact title repeats — the v1.7.4 live run still reproduced two forbidden titles verbatim ('Ladbroke Spine', 'Rej & The Room') for the same canvases. Exact collisions (casefolded match against the avoid list) now trigger one bounded rename-only LLM call for just the colliding concepts, before the report pass so prose uses the final titles. Replacements that fail, collide with the avoid list, or duplicate another title are rejected — the original stays and the echo warning stands. Accepted renames appear in run notes ("**Renamed**: 'old' → 'new'").

---

## v1.7.4 — 2026-07-07

Two fixes from the first v1.7.3 live run — a strong run (zero exact name repeats against 27 forbidden names, lens character clearly visible) that exposed two subtler defects.

- **History-echo warnings survived only on revision-free runs.** `revise_concepts`' final revalidation didn't receive the name-avoid list, so any run that revised a concept silently dropped every history-echo warning ('Heist at 4AM' vs 'Heist Recordings' went unflagged). The final revalidation now carries `used_mix_names`.
- **Lens sampler could contradict the anchor-lift cap.** Three of the twelve naming lenses instruct drawing words from the pool's own titles/artists; sampling two or more forced the model to violate the one-anchor-lift-per-response rule (live run produced four anchor-lifted names). The sampler now picks at most one pool-drawing lens per run, and the lens block restates the cap.

---

## v1.7.3 — 2026-07-07

Two fixes from the first post-v1.7.2 live run, where four of six concept names were exact repeats of earlier same-day runs — the detection guard flagged every one, but the prompt-side avoidance was skated past.

- **Forbidden-names block.** The avoid list moves from a mid-sentence aside to a hard `FORBIDDEN NAMES` block at the end of the Stage 2 system prompt, with an explicit per-title self-check ("if it appears here, or shares a distinctive word with an entry here, DISCARD it").
- **Within-day lens rotation.** The naming-lens seed now advances with every recorded run (`date seed + history length`), so same-day runs get different lenses instead of identical naming pressure. Still fully reproducible from `--stage1-seed` plus history state.

---

## v1.7.2 — 2026-07-07

- **Recent concept titles join the name-avoid list (#75).** Live finding from the first Name Studio run: 'Heist Recordings' was generated twice in one day and 'Rej & The Room' echoed the earlier 'Rej' — the avoid list only carried catalogue mix names, so same-day repeats were invisible to both the prompt and the name-family guard. `recent_concept_titles` (last 10 runs, deduped, newest first) now merges into the avoid list for the Stage 2 prompt and the history-echo validation check.

---

## v1.7.1 — 2026-07-07

- **Fix: `--prep` now applies the DO NOT RECOMMEND denylist.** The first production run reported 414 house tracks where the concept pipeline sees 370 — `--prep` was counting (and could rank) denylisted tracks, and cueing tracks that are never recommended is exactly the wasted prep time the feature exists to avoid. The filter now runs before ranking, matching every other mode.

---

## v1.7.0 — 2026-07-07

- **Cue-Prep Assistant (epic #72).** `mixlab --prep` ranks every track with missing or partial cue data by expected payoff — how often concept history programs it, how harmonically central it is in its genre bucket, unplayed status, and gap severity — so cue-prep time in Rekordbox goes where booth sheets and blend scoring gain the most. Fully offline: no LLM, no API, no Discord. `--top N` and `--genre <standard label>` scope the table. Born from the Booth Sheet's scout notes (v1.6.0).
- **Named tracks in validation warnings.** Bridge/wildcard warnings now say which track ("wildcard track Artist — Title (ID …) used without a justified transition") instead of a bare Rekordbox ID.
- **Name Studio (#75).** Mix names get technique instead of vocabulary. Live finding: the Stage 2 prompt's own example names were leaking into output ("Fever" + "Late Latitude" exemplars produced "Fever Latitude" and "Fever Chart" on the same day). The naming section now marks all examples as ALREADY TAKEN, bans the generic [Adjective][Noun] pattern outright, and enforces per-run family rules (no shared salient words between titles, one "[Word] & [Word]" max, one anchor-lift from a concept's own track list max — twisted, not quoted). Twelve naming lenses (lyric fragments cut mid-phrase, portmanteaus, misheard titles, phone-notes-at-4am, dubplate scrawl…) rotate three per run on the Stage 1 date seed. A deterministic warn-only family guard surfaces history echoes and within-run repeats in validation notes without ever triggering revision. Name quality needs live-run verification.

---

## v1.6.0 — 2026-07-07

- **Booth Sheet (epic #67, mock approved by owner).** Each concept card in the HTML report now carries a per-transition execution plan: clock position of the outgoing mix-out cue, pitch-fader percentage (tempo-relation aware), key move, bars available, a plan line in booth language with fallback and scout notes, colour-coded by blend headroom (relaxed/tight/hard). Fully deterministic — computed from cue points and beat grids already in the Rekordbox XML, zero LLM calls (#68 core module, #69 rendering). Scout notes double as a cue-prep to-do list.

---

## v1.5.3 — 2026-07-07

- **Fix: v1.5.2's revised-concept prose regeneration never fired in production.** The genre-mode report always carries trailing sections after the concept prose (shortfall warnings, "Main brain"), so v1.5.2's exact-count guard never matched and every accepted revision silently fell back to the pre-revision disclaimer. The guard now only requires the concept sections to be present (they are always first), and the splice replaces by index. Tests mirror the real report shape so the trailer case is pinned.

---

## v1.5.2 — 2026-07-07

- **Revised concepts get regenerated prose.** When the self-revision pass (#55) accepts a repair, the concept's report section is now regenerated so the card's prose describes the shipped track list (previously the HTML card showed the revised table next to pre-revision prose). One extra report call per accepted revision; a regeneration failure keeps the repair and falls back to the old pre-revision disclaimer. Found via live run: a revised concept's prose still narrated a track that revision had swapped out.

---

## v1.5.1 — 2026-07-07

Three fixes from the first real production run of the v1.5.0 HTML report (house crate, live feedback).

- **Arc-consistent prose.** The Stage 2 report pass now receives each concept's declared `arc_type` and must match its "Energy path:" label to it (explicit mapping for all nine arc values) — no more `wave` badge next to "Slow Climb" prose.
- **`intro:0b` read as deliberate.** Owner-confirmed: cues at 0:00 are real mix-in points. Stage 2 prompts now carry a mix-point token legend stating that `intro:0b` means the track mixes in from bar one by design — never "cold drop / nothing to blend over" — and that a missing token means no cue data, not no intro.
- **HTML run notes de-duplicated.** The textual "⚠ Validation Notes" block embedded in the Discord report text is stripped from the HTML report's Run notes, which already renders warnings in a dedicated section.

---

## v1.5.0 — 2026-07-07

- **Standalone HTML report (#45).** Every run now writes a self-contained HTML artifact to `output/reports/` (`MIXLAB_REPORT_DIR` override) and attaches it to the Discord message alongside the XML. One file, zero external requests, dark-mode aware, mobile-readable. Concept cards carry: energy sparklines, full track tables (key/BPM/duration/energy/intro-outro bars) with click-to-copy titles for Rekordbox search, per-transition mechanism and blend labels computed live from the transition graph with score-band colouring, the concept's prose report section, runtime/practicality, and the run's validation notes and crate snapshot. Discord text output unchanged; attachment MIME now inferred per file (was hardcoded `application/xml`).

---

## v1.4.0 — 2026-07-07

**The Mix Engine** (epic #58, sub-issues #41/#59/#60/#61): cue-aware blends and optimal sequencing. The beat grids and cue points already in the Rekordbox XML now inform transitions and play order — turning transition intelligence from prompt hints into deterministic guarantees.

- **Cue/grid parsing → MixPoints (#41).** `<TEMPO>` anchors and `<POSITION_MARK>` cues are parsed per track; the owner's cue conventions are encoded directly: first cue by position = mix-in, last cue = mix-out (only with ≥2 cues landing in the back half — a single-cue track is partially prepped), loops recorded as zones, cueless tracks neutral. Intro/outro lengths derived in **bars** via piecewise integration across multi-anchor beat grids.
- **Blend-aware transition edges (#59).** Edges gain a headroom score from outgoing outro bars vs incoming intro bars, with a loop-zone bonus, labelled in booth language (`29 bars out / 32 in — tight`, `cut or manual loop likely` — short outros are never called unmixable, since they get looped live). Composite edge weights rebalance only when both sides carry data; cueless scoring is byte-identical to v1.3.0.
- **Beam-search sequencer (#60).** New pure `sequencer.py`: `optimal_order` finds the best play order for any track set (deterministic beam search over the transition-edge matrix, pinned opener/closer support); `score_order` judges any existing order on the identical scale (edge quality + arc fit + endpoint fit); `suggest_swaps` proposes bounded interior-only improvements with human-readable reasons.
- **Pipeline integration (#61).** Warn-only `blend risk` validation findings (feeding the self-revision trigger); practicality gains a `blend_feasibility` component when at least half the pairs carry data; after Stage 2 the sequencer checks every concept's order and suggests improvements clearing a 10% score bar — suggestion-only by default, `--resequence` applies them (rebuilding transition annotations); Stage 2 prompts carry per-track `intro:16b/outro:32b` tokens.

Live spot-check for at home: confirm derived intro/outro bars match booth reality on a few well-known tracks (`docs/exploration/h3-cue-grid-prep.md` has the schema background).

---

## v1.3.0 — 2026-07-07

Creativity wave from the 2026-07-06 audit (`docs/audit/2026-07-06-mixlab-audit.md` + plan). Eleven issues (#42, #46–#55) shipped; live-validated on the real collection (house + 4x4 smoke runs).

- **Concept Directions (`--directions mixed|off|only`, #53).** A second, cross-strata generative axis alongside classic BPM-stratum canvases: deterministic creative briefs — mood journeys, era dialogues, label spotlights, artist threads, energy-shape-first designs, fresh-crate showcases — proposed only when the library supports them, feasibility-scored, BPM-path-checked, and seed-rotated per run. Each becomes a Mix Canvas with a DIRECTION BRIEF that Stage 2 must honour. Default `mixed` blends 2–3 directions with classic canvases.
- **Stage 1 size contract enforced with seeded windowing (#48).** Oversized shortlists (an 800-track `4x4` pool previously produced five 160-track "shortlists") are now capped at 25 tracks via a 15-track most-central spine plus seed-rotated sample fill. Overflow is reported per-shortlist and in the pipeline summary, never silently dropped. The run seed defaults to today's date, is printed at run start, and reproduces exactly via `--stage1-seed`. `select_canvases` now always runs the greedy overlap-aware loop (`n_effective = min(n, max(3, ceil(0.75·candidates)))`), so novelty/distinctiveness scoring has consequence even with few candidates.
- **Transition graph + arc verification (#51).** New pure `transitions.py`: tempo compatibility within a ±6% pitch window at 1:1, 2:1, 1:2, 3:4 and 4:3 ratios — a locked 172→86 halftime blend is finally understood as a lock, not an 86-BPM violation. Canvas headers gain `Strong transitions:` lines with named mechanisms (`halftime lock 172→86`, `energy lift 8A→9A`); noisy groove-locker/builder/peak/pivot role hints demoted from the prompt. Declared `arc_type` is traced against the actual energy sequence (warn-only `arc mismatch` findings); BPM-jump validation and practicality scoring are tempo-relationship-aware.
- **Bounded self-revision (`--no-revise` to opt out, #55).** Concepts with ≥2 hard validation findings — or, under `--deep`, a weak critique or a concrete suggested substitution — get one minimal-repair Stage 2 call (swap/reorder/drop from the same canvas pool; title and thesis preserved). Accepted only when it strictly reduces hard findings; hard one-pass cap. Reports carry a **Revised** annotation and Validation Notes reflect post-revision state.
- **Risk knob (`--risk low|medium|high`, #42).** `high` shifts canvas scoring toward contrast/novelty, offers flagged bridge/wildcard concept-anchors as featured picks, tags concepts `[ADVENTUROUS]`, and relaxes jump thresholds for annotated-risky transitions (20 BPM / 5 Camelot); `low` briefs restraint and tightens thresholds (10/3); `medium` is byte-identical to prior behaviour. Genre mode only; composes with `--directions`.
- **Playlist mode: `--intent` + fair variant selection (#54).** `--intent` now works in playlist mode as an explicit override of the Stage 0 brief (conflicts noted in Assumptions); risk keywords in the intent ("surprise me", "safe") override the inferred `risk_tolerance`. Winner selection is tolerance-aware — `fit = practicality·w_p + adventure_dividend·w_a` — so the adventurous variant can genuinely win at high tolerance instead of being structurally eliminated by the practicality-only metric. `low` tolerance reproduces the old selection exactly.
- **History depth + feedback loop (#52).** History records every concept per run (previously only the first — RECENT CONCEPTS and shape-novelty under-reported ~4×). New `mixlab --feedback` records `played`/`played_modified`/`rejected`/`unused` verdicts per concept; `played` amplifies that entry's novelty penalty ×1.5, all-`rejected` mutes it ×0.25. Old history files load unchanged.
- **Duration-aware set planning (#49).** `TotalTime`, `DateAdded`, and `Rating` parsed from the Rekordbox XML. `--mix-length` now works in genre mode too, with targets derived from real mean track duration (minutes/4 heuristic kept as fallback); prompts carry per-track `m:ss` durations and reports a `Runtime: ~NNm` footer.
- **Stage 2 resilience + config (#50).** Retries with backoff on 429/5xx/timeouts (retry-after honoured); `stop_reason` truncation detection; report `max_tokens` scales with track count; `MIXLAB_STAGE2_MODEL` and `MIXLAB_STAGE2_TEMPERATURE` env overrides (selection-pass default temperature 0.3 → 0.5). Returned track IDs validated against the offered canvas pools instead of the whole library — plausible-ID hallucinations are rejected at parse time.
- **Per-genre track targets in the prompt.** Canvas headers carry `Target: min–max tracks` from `TRACK_COUNT_TARGETS`, replacing the hardcoded "8–12 tracks" instruction that guaranteed shortfall warnings for jungle/DnB/UKG.
- **Deprecated LLM Stage 1 path removed (#46).** `MIXLAB_STAGE1_LLM`, `stage1_concepts`, `select_stage1_window`, and the Stage 1 prompt variants deleted; the free-provider cascade remains for Stage 0 playlist intent extraction.
- **Quick fixes (#47).** Vocal-token detection uses word boundaries ("Afterlife"/"Left Field" no longer flagged as vocal moments, cleaning `contrast_potential` scoring); `--export-unplayed` checks `CATALOG_API_URL` before parsing XML (test suite now green on a clean clone); playlist unplayed-candidate bonus corrected to the documented 0.15 effective weight (was 0.045); dead `build_playlist_pool`/`filter_by_bpm` removed; unused `anthropic` dependency dropped; stale README/skill-file content (pre-v0.13 random-window docs, removed minimax flag) corrected.
- **Polish from live smoke runs.** Stage 2 `mood` field capped to a short phrase (was returning full thesis paragraphs at temperature 0.5); stdout line-buffered under pipes so teed logs interleave correctly with stderr diagnostics.

Flagged for continued live observation: direction-brief wording, revision repair quality, temperature 0.5, and `--risk high` output character (revert levers: `MIXLAB_STAGE2_TEMPERATURE`, `--directions off`, `--no-revise`, `--risk medium`).

---

## v1.2.0 — 2026-06-09

- **`--locked` flag for playlist mode.** Prevents Stage 2 from adding tracks outside the seed playlist — it can only remove or reorder. Useful when you have already curated a pool (e.g. 31 tracks for a 1-hour set) and want MixLab to trim it without pulling from the library. Shown in report context as "locked pool" and surfaced on Discord. Ignored outside `--playlist` mode with a stderr warning.
- **`--mix-length` shown in Discord report.** Report context line now includes the set length when `--mix-length` is used, e.g. `(Uk Bass, unplayed tracks, 60min set)`.

---

## v1.1.0 — 2026-06-09

- **`--mix-length` flag for playlist mode.** Pass `--mix-length <minutes>` with `--playlist` to scale the number of tracks Stage 2 selects. Formula: `max(10, round(minutes / 4))` — e.g. 60 min targets ~15 tracks, 90 min targets ~22. Without the flag, playlist mode defaults to 10–14 tracks as before. Standard genre mode is entirely unaffected. The target is injected as a soft instruction in the Stage 2 user prompt; arc quality still takes priority over hitting the number.

---

## v1.0.0 — 2026-06-03

First stable release. No new features — version bump marks the pipeline as production-complete: deterministic Stage 1 (`partition_pool`), two-pass Stage 2 (Anthropic-only), Mix Canvas selection, post-run validation, concept history and novelty scoring, playlist mode, and all operating modes (`unplayed`/`played`/`all`) are stable. See v0.13.0 and prior entries for the full feature history.

---

## v0.13.0 — 2026-06-03

- **Deterministic Stage 1 via `partition_pool`.** Stage 1 shortlisting is now a pure-Python algorithm — same pool and same seed always produce identical output. Helpers `_median_bpm`, `_min_track_id`, `_infer_shortlist_mood`, `_find_bpm_peaks`, `_camelot_components`, `_era_split`, and `_resize_shortlists` underpin the public `partition_pool()` entry-point. All three Stage 1 call sites in `__main__.py` are wired to `partition_pool`; the original LLM path is retained behind `MIXLAB_STAGE1_LLM=1` (documented in `.env.example`) for soak-period comparison. `--stage1-seed` CLI flag added for reproducible runs. Old `stage1_concepts` / `select_stage1_window` / `_STAGE1_SYSTEM*` are deprecated.
- **Fix: Camelot sector split for tight-BPM pools.** When BPM-peak detection merges peaks that should be distinct clusters, `partition_pool` now splits the resulting pool by Camelot sector, yielding musically coherent sub-pools rather than one over-sized merged cluster. Prevents tight-BPM genres (e.g. techno at 130–132 BPM) from collapsing into a single undifferentiated shortlist.
- **`--intent` signal parsing.** `_parse_user_intent()` added to `llm.py`: heuristic keyword extraction of register, mood, occasion, arc-hint, era, and audience signals from the `--intent` string. Extracted signals are injected as a `Parsed signals:` line in the `USER INTENT` block so Stage 2 can key on structured cues even when the intent is written in free prose. Present only when at least one signal fires; absent when no signals match.
- **`--intent` prompt placement and meta-instruction strengthened.** `recent_concepts` block now precedes `genre_intent_block` in both genre-mode prompt branches so intent reads as the dominant override after seeing prior history. Meta-instruction rewritten from passive suggestion to primary curatorial lens: Stage 2 must serve the stated direction and surface conflicts in Assumptions when the pool makes full compliance impossible. `--intent` help text updated with examples, word-count guidance, and axis coverage.
- **`_warn_intent()` validation.** Extracted as a testable helper; warns on empty, <5-word, or >100-word intents before the run proceeds. Catches accidental flag misuse (e.g. passing a filename as intent) without aborting the run.
- **Intent/mood parsing hardening.** Series of targeted fixes to `_parse_user_intent`: position-aware mood suppression (re-scans all register-key spans, handles multiple occurrences), mood prefix collision resolved, whitespace normalisation, dict ordering stabilised, missing aliases added (`podcast`, etc.), negation caveat strengthened. `_kw_match` consistency fixes from adversarial review.

---

## v0.12.4

- **Pool-relative opener/closer fallback in `_infer_roles`.** When absolute energy thresholds yield no opener or closer candidates (e.g. D&B pools where MIK tags everything 6–7), the minimum-energy track(s) in the pool are promoted as relative opener/closer candidates. Avoids spurious "weak opener pool" / "weak closer pool" risk notes on high-energy but internally-varied pools. Risk-note threshold also tightened: notes fire only when the candidate list is completely empty, not merely `< 2`.
- **Intent surfaced in report context.** `_format_report_context` now appends the `--intent` string to the context block passed to Stage 2, so the report prompt sees the user's free-text creative direction alongside mode and export details.
- **Discord delivery progress prints.** Two `print()` lines added to `DiscordClient.post`: one before chunked posting (logs resolved channel ID) and one after (logs message + attachment counts). Aids smoke-test visibility.

---

## v0.12.3

- **Fix played-track matching: strip version suffixes and bare `feat` before key comparison.** `normalise()` now removes parenthesised/bracketed version tokens (`Original Mix`, `Extended Mix`, `Club Mix`, `VIP`, `Dub`, `Vocal Mix`, `Instrumental`, `Remaster`, etc.) from both catalog and Rekordbox titles before building the match key. Also handles bare `Original Mix` without brackets (e.g. Rekordbox title `Freaker Original Mix`) and `feat` without trailing period inside parens (`feat Slay` vs `feat. Slay`). Root cause: SoundCloud tracklist descriptions routinely omit mix version suffixes, so `filter_unplayed` was treating 17 in-collection played tracks as unplayed — including Dam Swindle *That's Right (Original Mix)* appearing in the unplayed pool despite being used in *Slow Burn*.

---

## v0.12.2

- **Stage 2 prose/JSON risk-annotation alignment (#29).** The Stage 2 report pass previously received only the concept title, mood, thesis, and track listing — never the `transitions` array from the selection pass. The report writer guessed prose `Risk:` content from the track sequence alone, often producing rich risk descriptions while the structured `Transition.is_risky` stayed `False`. v0.12.1's validator suppression depends on the structured annotation, so the gap meant warnings still fired on chapter pivots the LLM had described as risky. Two changes ship together: (1) `_call_stage2_report_single` now appends a `Transition annotations from selection` block to the report prompt listing each transition's `is_risky` and `risk_type`; (2) `_STAGE2_REPORT_SYSTEM` gains a `CONSISTENCY` rule requiring prose `Risk:` lines to mirror those fields. Smoke runs show the report now uses the `risk_type` enum vocabulary verbatim in prose (`chapter pivot`, `peak impact`, `deliberate reset`, `closer move`), closing the gap.

---

## v0.12.1

- **Validator: suppress BPM/Camelot jump warnings on justified-risk transitions (#28).** `validate_stage2_output` previously fired `BPM jump >15` and `Camelot jump >4` warnings regardless of whether the transition was annotated as a deliberate risk. Mirroring the bridge/wildcard role-check pattern already in place, both warnings are now suppressed when the corresponding `Transition` has `is_risky=True` AND a non-empty `risk_type` (`chapter_pivot`, `peak_impact`, `deliberate_reset`, `closer_move`, `low_tonal_risk`). Unannotated jumps still warn, and `is_risky=True` with empty `risk_type` still warns (unjustified risk). Thresholds (15 BPM / 4 Camelot) unchanged — per-mode tuning deferred until an adventurous mode knob lands. Warning becomes "unjustified threshold breach" signal rather than "any threshold breach".

---

## v0.12.0

- **Sharper Stage 2 role vocabulary (#23).** The Stage 2 set-role list trimmed 19 → 10: `opener`, `groove`, `hook`, `pivot`, `lift`, `vocal-moment`, `texture-change`, `peak`, `resolution`, `closer`. Removes semantic overlap (early-hook ≈ world-setter ≈ opener; groove-locker ≈ builder ≈ connector; weapon ≈ peak; post-peak ≈ cleanser ≈ resolution; risk ≈ pivot; utility removed). Stage 2 selection + report prompts updated, parser coerces old strings to `unknown`, playlist_mode missing-role checks switched (`builder` → `groove`, `cleanser` → `resolution`). `CanvasRoleCandidates` field names kept stable (internal canvas pools).
- **Mode-aware canvas scoring weights (#24).** `score_canvas` now accepts a `CanvasScoreWeights` table; `select_canvases(mode=...)` selects from `CANVAS_SCORE_WEIGHTS_BY_MODE`. Each mode shifts 5–10pp around the v0.11 baseline: `unplayed` boosts novelty (0.10 → 0.15) at the cost of anchor strength; `played` boosts anchor strength (0.15 → 0.25) at the cost of distinctiveness and novelty; `all` boosts distinctiveness (0.15 → 0.20). `CanvasScoreWeights` enforces sum-to-1.0 on construction. `DEFAULT_CANVAS_WEIGHTS` kept as a no-mode fallback for legacy callers.
- **Opt-in Stage 2 critique loop via `--deep` (#22).** A self-critique pass runs between curation and the prose report when `--deep` is set. Each concept gets reviewed by Sonnet as a peer DJ would — opener spec, closer finality, energy-path-vs-sequence, risky-transition justifications, thesis defensibility. The output is a `Critique` (verdict + single_weakest_moment + structural_issues + suggested_substitution) attached to `MixConcept` and surfaced inline as a `CRITIQUE (DEEP MODE)` block. Never auto-applied. Tolerant JSON parser falls back to a needs_attention critique on malformed responses or HTTP failures rather than aborting the run. Genre-mode only; playlist mode emits a stderr note and skips. Doubles Stage 2 cost when enabled; default behaviour unchanged.
- **Validator role-pool checks dropped + house/techno softening (#27).** The canvas-pool-based `_classify_track_roles` heuristic disagreed with Stage 2's textual role picks on virtually every real-run concept (observed in v0.10 hip-hop and v0.11 house smoke tests). The three warnings that depended on it — `no opener-role in first 2 positions`, `no closer-role in last 2 positions`, `N consecutive <role> tracks` — are removed. `_classify_track_roles` deleted. House, techno, and the 4x4 custom genre now belong to a new `_SUSTAINED_GROOVE_GENRES` softening family covering `no peak in sequence`, `no wind-down`, and `all tracks high-energy` soft-tier checks, matching the existing DnB-family treatment. Energy-band, BPM-jump, Camelot-jump, and bridge/wildcard-without-justification checks stay strong-tier.
- **Concept feedback loop discovery (#12).** [`docs/exploration/concept-feedback-loop.md`](docs/exploration/concept-feedback-loop.md) sketches the data model, capture channels (CLI primary, Discord reactions secondary, manual JSON edit escape hatch), scoring integration via FEEDBACK_MULTIPLIER + score_anchors boost, decay strategy, risks, and a decision signal for when to revisit. No implementation — per the ticket's explicit scope, this is a design document gated on real-run signal that the existing novelty system is no longer doing enough.
- **README synced** with the Phase 2 + Phase 3 shipped work: 10-role vocabulary, era/label canvas dimensions, core anchors, concept-anchor candidates, mode-aware scoring weights, combined track + shape novelty, `arc_type` / `Bold moves` / `Practicality` report lines, full validator warning list (strong-tier vs soft-tier), and a new `--debug` section documenting the diagnostics output.

---

## v0.11.0

- **`--intent` CLI flag for genre mode (#16).** Free-text creative direction is injected verbatim into the Stage 2 genre-mode user prompt under a USER INTENT header. Pure passthrough — no parsing, no LLM extraction. Example: `mixlab --genre house --intent "warmup set, outdoor afternoon, low pressure, melodic"`. Ignored (with a stderr note) in playlist mode, which has its own Stage 0 intent-extraction pass. Blank or whitespace intent is treated as no intent.
- **Mode-specific Stage 2 prompt fragments (#18).** Each of the three operating modes now gets distinct creative direction appended to the Stage 2 system prompt: `MODE: UNPLAYED` framing nudges concepts toward discovery-worthy debuts; `MODE: PLAYED` invites bolder Camelot jumps and chapter pivots because familiarity is an asset; `MODE: ALL` asks the model to interleave played and unplayed material and to note the lean in the thesis. Playlist mode unaffected.
- **Concept-shape novelty scoring (#7).** `similarity_to_history` now combines track-overlap Jaccard with a deterministic concept-shape similarity over reliably-populated fields: BPM band (10-BPM bucket), dominant Camelot zone, role-presence flags, and `energy_path` (excluded when empty — unknown ≠ same). Weights are 65% track + 35% shape, conservative enough that a strong canvas can still win against a structurally similar history entry. New `similarity_breakdown_to_history` surfaces both components for `--debug`. No embeddings, no extra LLM calls.
- **Anchor track detection for genre mode (#19).** `score_anchors()` combines provenance distinctiveness (remixer, enrichment confidence, label), library rarity (label/artist counts across the full collection), pool centrality (BPM median, dominant Camelot), energy positioning, and recency at 30/25/20/15/10 weights. Tracks in the top 20% of the canvas core AND scoring ≥ 0.55 become `core_anchor_ids` — typically 2–5 anchors per 15–25-track canvas, 0 on uniformly weak pools. Stage 2 sees an `Anchors:` line in the canvas header and a corresponding rule in `_STAGE2_CANVAS_RULES` nudging anchor inclusion as preference, not requirement. Signal only — no change to canvas scoring or selection.
- **DJ practicality score for genre-mode concepts (#21).** Every genre-mode concept now gets a one-line Practicality summary in its report: `bpm_smoothness`, `harmonic_ratio`, `risk_justified`, `overall`. In genre mode `fragment_preserved` is always 1.0 (no seed adjacencies), which slightly inflates `overall` vs playlist-mode scores — decorative for now, not used for ranking. Playlist mode unchanged: it already surfaces practicality via the WINNER labelling.
- **Era and label as canvas dimensions (#20).** `build_mix_canvas` now computes `era_window` (min/max year over core tracks when ≥ 60% of the pool has year data) and `dominant_label` + `label_share` (the most-frequent label when its share ≥ 40% and count ≥ 5). `era_coherence` ramps from 1.0 at span ≤ 3 years to 0.0 at span ≥ 18 years; `label_coherence` ramps from 0.0 at the share threshold up to 1.0 at ~70% share. Both surface in the Stage 2 canvas header. Canvas score weights rebalanced (sum = 1.0): `technical_viability` 0.20 → 0.10, with 5pp each going to `era_coherence` and `label_coherence`. Missing year/label data yields 0.0 coherence — bonus only, never a penalty.
- **Bridge/wildcard concept-anchor candidates (#10).** Multi-signal scoring (role-fit, within-canvas + collection-level label/artist rarity, harmonic contrast vs the dominant key, energy-role suitability) identifies bridge/wildcard tracks above a composite threshold and tags them by category: `peak` (high energy wildcard), `identity` (distinctive label/artist), `structural-exception` (opener/closer/pivot fit despite BPM outlier). Energy is demoted from primary filter to one signal among several. Stage 2 sees a `Concept anchors:` line in the canvas header and a corresponding rule in `_STAGE2_CANVAS_RULES` raising the bar for unflagged wildcard use. `validate_stage2_output` now warns when a wildcard appears in a concept without being on the anchor candidate list.

---

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
