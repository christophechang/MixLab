# Conjunction Mining + Direction Scoring Replacement — Design

**Date:** 2026-08-08
**Scope:** MixLab only. Payload contract (`library_map.py` output shape) unchanged;
mixlab-web and the API need no changes.
**Origin:** Review of the shipped library-map direction rail (mixlab-web
`docs/superpowers/specs/2026-08-06-library-map-design.md` §7). Two findings drive this
design: (a) the feasibility formula saturates — `pool_fill` and `path_ratio` pin at
1.0, so scores collapse to `0.7 + 0.3·signal` and the rail's ranking carries no
quality information; (b) the seven named builders instantiate pre-decided ideas —
none *discovers* structure the way `genre_traverse` does, which is why its output is
the only one that reads as creative.

## Decision summary

| Fork | Decision |
| --- | --- |
| Scope | Conjunction mining **and** the scoring replacement, one change — mined rows ranked by lift cannot slot honestly into a saturated ranking |
| Mining mechanism | Bounded conjunction scan: 1- and 2-predicate conjunctions, lift-ranked (no FP-Growth, no targeted subgroup discovery) |
| BPM/key predicates | Minable, never namable — titles/moods built only from human-legible predicates (v1.13.2 no-DJ-mechanics rule) |
| Cap | Top 3 mined rows per pool, deduped against named candidates |
| Contract | Mined rows ship as `direction_type: "found"` in the existing payload shape; `feasibility` keeps its name and 0–1 range with new semantics |

## 1. Placement

- **New module `src/mixlab/mining.py`** — predicate extraction, conjunction scan,
  lift, pruning, mined-brief templating. Pure and deterministic; no I/O.
- **`directions.py`** — gains the shared scorer (replacing `_finalise`'s formula) and
  calls the miner from `enumerate_directions`. `generate_directions` (the live run
  path) enumerates the same combined candidate field, so a mined direction clicked
  as "Run this direction" behaves like any named one.
- **`library_map.py`** — untouched. `_direction_entry` already serialises whatever
  `enumerate_directions` returns.
- **Web** — zero changes. `direction_type: "found"` renders through the existing
  `replace(/_/g, " ")` row label; the feasibility bar renders spread instead of a
  wall of 1.00.

## 2. Predicate vocabulary

Extracted per pool. Each predicate = `(kind, value, track_id set, namable flag)`.
A predicate exists only if its track set meets its gate.

| Kind | Value | Namable | Gate |
| --- | --- | --- | --- |
| `label` | exact label string | yes | ≥ 8 tracks |
| `era` | aligned 5-year window (e.g. 2015–2019) | yes | ≥ 8 |
| `tag` | exact tag, lowercased | yes | ≥ 8 |
| `remixer` | exact remixer string | yes | ≥ 5 |
| `fresh` | `date_added` in newest 20% of pool | yes | ≥ 8 |
| `energy` | band: low ≤ 4 / mid / high ≥ 6 | yes | ≥ 8 |
| `bpm_regime` | membership of a `_tempo_regimes` density peak | **no** | ≥ 8 |
| `key_hood` | camelot key + its compatible neighbours | **no** | ≥ 8 |

Notes:

- Era windows are calendar-aligned (…2010–2014, 2015–2019…) so window identity is
  stable as the collection grows. Tracks with `year is None` or `≤ 0` join no era
  predicate.
- `fresh` uses count-percentile over ISO `date_added` strings sorted
  lexicographically — no date parsing, matching `_build_fresh_crate`'s precedent.
  Tracks with empty `date_added` join no fresh predicate.
- `key_hood` reuses `camelot_compatible` for the neighbour set.
- Tag values are matched case-insensitively, mirroring `_has_tag`.

## 3. Mining pipeline

Per pool, in order:

1. **Enumerate predicates** (§2).
2. **Candidate conjunctions:** all singletons + all cross-kind pairs. Same-kind
   pairs are skipped (label×label is empty; era×era is empty; tag×tag is legal in
   principle but produces unwieldy names — excluded for v1).
3. **Gate each candidate:** intersect track sets; require support ≥ 15
   (`MIN_DIRECTION_POOL`), at least one namable predicate, and path feasibility
   (`_path_feasible` ratio ≥ 0.8 — retained as a *gate*, §4).
4. **Lift:**
   - pair `A×B`: `|A∩B| / (|A|·|B| / |pool|)`
   - singleton `A`: concentration vs base rate — `(|A|/|pool|)` has no independence
     baseline, so singletons are scored by identity strength alone (label share,
     tag share, …) and only survive when no surviving pair contains them
     (pairs are the discovery payoff; a singleton is a fallback, not a peer).
5. **Subsumption pruning:** if a pair's track set is ≈ a parent singleton's
   (Jaccard > 0.9), keep the pair only if its lift ≥ 1.2× the parent's identity
   score; else keep the parent.
6. **Score** with the shared scorer (§4), then **dedupe**: walk candidates in rank
   order (mined and named together); drop any candidate whose track-set Jaccard
   vs an already-kept candidate exceeds 0.6. Cap mined survivors at **3 per pool**.
7. **Title/mood from namable predicates only:**
   - title: `Found: Hospital Records × 2015–2019` (namable predicate display
     names joined with `×`, singleton drops the join)
   - mood: predicate kinds, e.g. `label × era conjunction`
   - `direction_type: "found"`, `thread_artist: ""`
   - A candidate whose namable side is only `fresh`/`energy` names from those
     ("Found: fresh arrivals × high energy") — legal, human-legible.
   - Deterministic throughout: no seed participation anywhere in mining. Ties in
     every sort break on `(lift desc, title asc)`; track sets iterate in
     `track_id` order.

Cost: ~100 predicates/pool → ≤ ~5k pairs → milliseconds at 2k tracks. No new
dependencies.

## 4. Shared scorer (replaces the `_finalise` formula)

Applies to **all** candidates, named and mined.

**Gates** (pass/fail, no longer scored): support ≥ 15 after capping;
`_path_feasible` ratio ≥ 0.8.

**Score:**

```
score = 0.25·freshness + 0.25·mixability + 0.20·identity + 0.30·distinctiveness
```

- **freshness** — median `date_added` count-percentile of the chosen set within
  the pool (0 = oldest crates, 1 = newest). Rank-based, so it cannot saturate the
  way unplayed-share does under the default `--mode unplayed` map run (where every
  track is unplayed and the share pins at 1.0 — the same trap as `pool_fill`).
- **mixability** — mean over the chosen set of each track's best
  `score_transition` to any other member, normalised to 0–1 by the scorer's
  maximum. O(n²) at n ≤ 25. Replaces the tautological BPM-sorted-adjacency check
  as a live signal (that check survives only as a gate).
- **identity** — strength of the defining idea.
  Mined pair: `min(lift / 3, 1)`. Mined singleton: its concentration score
  (share of pool carrying the predicate), clamped 0–1 — singletons have no
  independence baseline, so share stands in for lift.
  Named: the builder's existing signal, renormalised per builder where saturating
  (`fresh_crate`'s `min(newest_slice/15, 1)` → count-percentile form). Builders'
  signal semantics are otherwise preserved.
- **distinctiveness** — `1 − max(Jaccard vs every other candidate in the pool)`,
  computed in a second pass after the pool's full candidate field (named + mined)
  exists. Two-pass keeps it order-independent → deterministic. Structurally kills
  the cloned-rows problem: two near-identical candidates *both* score low.

`feasibility` keeps its field name and 0–1 range — the payload contract and the
web renderer are untouched. Round to 4 decimals as today.

## 5. Mined briefs

`genre_traverse` standard: template carries rhetoric, computed facts carry content.
Interpolated per candidate:

> Found structure: **{n} tracks** where **{predicate A}** meets **{predicate B}**
> — **{lift:.1f}×** denser than chance. {unplayed_line} Anchors:
> {2–3 top-centrality tracks, artist — title}. Treat the conjunction as the
> thesis: this corner of the crate has a coherent sound {kinds} explain together;
> play it as a scene, not a coincidence.

- `{unplayed_line}` ("11 of 14 unplayed.") appears only when a non-empty played
  list reaches the miner **and** the pool mixes played/unplayed. Today that is
  never: `--mode unplayed` pools are all-unplayed, and `--mode all` skips the
  catalog fetch entirely (`_run_map_cli`). The line is a cheap conditional that
  lights up if mode=all ever fetches the catalog — not a reason to change the
  CLI now.
- Anchor tracks come from `_centrality_rank` over the chosen set.
- Evidence lines may cite BPM (traverse precedent); title and mood never do.

## 6. Error handling

Sparse metadata degrades silently, never raises:

- No labels / years / tags in a pool → those predicates simply don't exist →
  fewer or zero mined rows. Zero mined rows is normal output, not an error.
- Empty `date_added` → track sorts as oldest (percentile 0) for freshness and
  joins no `fresh` predicate.
- A pool below 15 tracks mines nothing (no predicate can gate in).

## 7. Testing

- **Miner units:** predicate extraction per kind incl. gates and missing-metadata
  behaviour; pair lift math against hand-computed fixtures; subsumption pruning;
  cross-candidate dedupe order; cap; naming rule (a candidate whose namable side
  is empty is skipped — assert no mechanical-only titles ever emitted).
- **Determinism:** same pool mined twice → byte-identical `Direction` list;
  golden snapshot of `enumerate_directions` on a fixture pool.
- **Scorer:** spread assertion — a synthetic pool where the old formula yields
  all ≈ 1.00 must yield materially distinct new scores; freshness non-saturation
  under an all-unplayed pool; distinctiveness punishes a planted clone pair.
- **Integration:** `build_map_payload` on a fixture collection contains `found`
  rows with valid shape; existing direction tests updated for new score values
  (shapes unchanged).

## 8. Explicitly not doing

- No web or API changes; no payload shape change.
- No FP-Growth / ≥3-predicate conjunctions; no tag×tag pairs (v1).
- No LLM anywhere in this path.
- No fact-chips UI, no merit-rank rewrite of within-builder `rng.choice`, no
  `artist_thread` gate change — separate follow-ups, deliberately out of scope.
- No seed participation in mining (named builders keep theirs).
