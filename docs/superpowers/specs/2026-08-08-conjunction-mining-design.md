# Conjunction Mining + Direction Scoring Replacement — Design

**Date:** 2026-08-08 (hardened same day after adversarial review — see §10)
**Scope:** MixLab only. Payload shape (`library_map.py` output) unchanged; mixlab-web
and the API need no changes.
**Origin:** Review of the shipped library-map direction rail (mixlab-web
`docs/superpowers/specs/2026-08-06-library-map-design.md` §7). Two findings drive this
design: (a) the feasibility formula mostly saturates — `pool_fill` and `path_ratio`
pin at 1.0 for filled candidates, so scores collapse toward `0.7 + 0.3·signal`
(measured on the live collection: 44 of 49 rows in [0.70, 1.00]) and the ranking
carries almost no quality information; (b) the seven named builders instantiate
pre-decided ideas — none *discovers* structure the way `genre_traverse` does, which
is why its output is the only one that reads as creative.

## Decision summary

| Fork | Decision |
| --- | --- |
| Scope | Conjunction mining **and** the scoring replacement, one spec — mined rows ranked by lift cannot slot honestly into a saturated ranking. (Implementation may sequence the scorer first and validate it on the live rows before the miner lands.) |
| Mining mechanism | Bounded conjunction scan: cross-kind 2-predicate pairs only, lift-ranked. No singletons, no FP-Growth, no targeted subgroup discovery |
| BPM/key predicates | Minable, never namable — titles built only from human-legible predicates (v1.13.2 no-DJ-mechanics rule) |
| Cap | Top 3 mined rows per pool, `direction_type` = `found_1`/`found_2`/`found_3` by final rank |
| Contract | Payload shape unchanged; `feasibility` keeps its name and 0–1 range with new semantics. Distinct `found_N` types keep the web's `pool:direction_type` identity key collision-free with zero web changes |

## 1. Placement

- **New module `src/mixlab/mining.py`** — predicate extraction, pair scan, lift,
  pruning, mined-brief templating. Pure; no I/O.
- **`directions.py`** — the shared scorer (replacing `_finalise`'s formula) and a
  single shared candidate-enumeration function that **both** `enumerate_directions`
  and `generate_directions` call (today they duplicate the builder loop at two call
  sites — that duplication is removed, not extended).
- **Run path (`generate_directions`):** mined candidates occupy **at most 1** of
  `max_directions` slots — the top mined row (if any) plus seed-rotation over the
  named candidates for the rest. Mining must add to the day's variety, not take it
  over (mined rows score similarly by construction and would otherwise be picked as
  a contiguous block). Known behaviour change: candidate-count changes shift the
  existing seed rotation's picks; acceptable, noted here deliberately.
- **`library_map.py`** — untouched.
- **Web** — zero changes. `found_1`…`found_3` render as "found 1"… via the existing
  `replace(/_/g, " ")` row label; the identity key `pool:direction_type` stays
  unique.
- **Diagnostic line:** `generate_directions`' "Directions proposed:" print becomes
  `… (5/7 builders, 2 found)` — mined count reported separately so the
  `N/len(_BUILDERS)` denominator stays truthful.

## 2. Predicate vocabulary

Extracted per pool. Each predicate = `(kind, value, track_id set, namable flag)`.
A predicate exists only if its track set meets its gate. A predicate covering
**> 70 % of the pool** is excluded from pairing — it would ride along with any
partner and mine nothing (measured: `energy=high` covers 96 % of drum & bass).

| Kind | Value | Namable | Gate |
| --- | --- | --- | --- |
| `label` | exact label string | yes | ≥ 8 tracks |
| `era` | aligned 5-year window (e.g. 2015–2019) | yes | ≥ 8 |
| `tag` | exact tag, lowercased | yes | ≥ 8 |
| `remixer` | exact remixer string | yes | ≥ 5 |
| `energy` | band: low ≤ 4 / mid / high ≥ 6 | yes | ≥ 8 |
| `bpm_regime` | membership of a `_tempo_regimes` density peak | **no** | ≥ 8 |
| `key_hood` | camelot key + its compatible neighbours | **no** | ≥ 8 |

Not in the vocabulary:

- **No `fresh` predicate.** `era × fresh` was the highest-lift pair in most live
  pools ("2025 releases were added recently" — a tautology, not a discovery), it
  double-counts with the scorer's freshness term, and `fresh_crate` already owns
  the idea.
- **No singletons.** A single predicate has no independence baseline; share-based
  identity maximally rewards the blandest predicate (the 96 % case above). Pairs
  only.

Notes: era windows calendar-aligned so identity is stable as the collection grows;
`year None/≤0` joins no era predicate. `key_hood` reuses `camelot_compatible`. Tags
matched case-insensitively (mirrors `_has_tag`). Mechanical kinds get display nouns
for prose only — `bpm_regime` → "tempo pocket", `key_hood` → "harmonic pocket" —
values (BPM numbers, key codes) may appear in brief evidence lines (the
`genre_traverse` precedent) but never in titles.

## 3. Mining pipeline

Per pool, in order. Deterministic throughout: every sort has a total tie-break
`(lift desc, title asc, track-set-hash asc)`; track sets iterate in `track_id`
order; no RNG.

1. **Enumerate predicates** (§2, with the 70 % coverage cap).
2. **Candidate pairs:** all cross-kind pairs. Same-kind pairs are skipped —
   most are structurally empty (label×label, era×era), `key_hood×key_hood`
   overlaps heavily by construction and must be guarded anyway, and `tag×tag`
   names badly. One rule covers all: cross-kind only.
3. **Gate each pair:** intersect track sets. Require support ≥ 15
   (`MIN_DIRECTION_POOL`), **≥ 1 namable predicate**, and **lift ≥ 1.3** —
   a pair at chance is not a discovery (the previous draft shipped four
   exactly-chance rows on the repo's own test fixture).
   `lift = |A∩B| / (|A|·|B| / |pool|)`.
4. **Subsumption pruning:** drop the pair if its track set's Jaccard vs
   *either* parent predicate's set exceeds 0.9 — a pair that adds no members
   over one parent adds no information (no score comparison; the previous
   draft compared a lift against a share, which is dimensionally meaningless).
5. **Shortlist:** top **12** pairs by the tie-broken lift order advance.
   (Prune-then-score: the expensive per-candidate work happens ≤ 12 times per
   pool, not once per gated pair — the live collection produces 1,423 gated
   pairs across 15 pools.)
6. **Shipped set:** each surviving pair's track set is
   `_centrality_rank(intersection)[:MAX_DIRECTION_POOL]`. **Every** downstream
   quantity — the `_path_feasible ≥ 0.8` gate, freshness, dedupe Jaccard,
   distinctiveness — operates on this shipped set, never the raw support set
   (live pairs reach support 236; the payload ships ≤ 25 `track_ids` and the
   wheel draws only those).
7. **Score, dedupe, cap** (§4): scored candidates (named + mined together) are
   walked in rank order; drop any whose shipped-set Jaccard vs an already-kept
   candidate exceeds 0.6. Cap mined survivors at **3 per pool**. **Title
   uniqueness:** if two would-ship mined rows render identical titles (both
   namable-half pairs of the same namable predicate — live example: one tag ×
   two different key hoods), keep only the higher-ranked.
8. **Materialise:** `direction_type` = `found_1`/`found_2`/`found_3` in final
   rank order. Title from namable predicates only:
   `Found: Hospital Records × 2015–2019`; a single-namable pair titles from its
   namable half alone (`Found: liquid`). Mood: ASCII kind pair for prose/id use —
   `label x era`, `tag x harmonic`. ASCII is load-bearing (`_canvas_id` truncates
   mood to 8 chars and `×` is non-ASCII — use `x`); kind-distinctness within those
   8 chars is **not** required — see §10. `thread_artist: ""`.

## 4. Shared scorer (replaces the `_finalise` formula)

Applies to **all** candidates, named and mined.

**Gates** (pass/fail, not scored): support ≥ 15 (shipped set);
`_path_feasible` ratio ≥ 0.8 on the shipped set. (`_path_feasible` survives as a
gate only. Mixability was evaluated as a scored term — mean best
`score_transition().score` per member — and measured **non-discriminative**: sd
≈ 0.01 across random 25-track subsets of the same pool; a near-constant term is
the same defect this design exists to remove.)

**Two passes, acyclic** (the previous draft's score→dedupe→distinctiveness
ordering was circular):

- **Pass 1 — rank & dedupe.** `rank = (0.25·freshness + 0.45·identity) / 0.70`.
  Walk in rank order, dedupe at shipped-set Jaccard > 0.6, cap mined at 3/pool.
- **Pass 2 — final score over the shipped field.**
  `feasibility = 0.25·freshness + 0.45·identity + 0.30·distinctiveness`, where
  distinctiveness is measured against the **post-dedupe, post-cap** field — the
  rows the operator actually sees. `Direction` is frozen; pass 2 uses
  `dataclasses.replace`.

Terms:

- **freshness** — median `date_added` count-percentile of the shipped set within
  the pool (0 = oldest, 1 = newest). Rank-based: cannot saturate under the
  default `--mode unplayed` run the way unplayed-share would. Empty
  `date_added` sorts oldest.
- **identity** — strength of the defining idea, on a common log scale.
  - Mined: `min(log2(max(lift, 1)) / 3, 1)` — lift 1.3 → 0.13, 2 → 0.33,
    4 → 0.67, 8+ → 1.0. (Raw `lift/3` saturates: live top lift is 10.4.)
  - Named — per-builder renormalisation, **all seven specified**:

    | Builder | Today | Becomes |
    | --- | --- | --- |
    | mood_journey | `_balance` of poles **after** `[:10]` truncation (pins at 1.0 whenever both poles ≥ 10) | `_balance` of the untruncated pole counts |
    | era_dialogue | `_balance(len(old), len(new))` | unchanged (uncapped already) |
    | label_spotlight | label share of pool (0.048 on a live 418-track pool — incomparably small) | collection-lift: `min(log2(max(share_in_pool / share_in_collection, 1)) / 3, 1)` — same scale as mined identity |
    | artist_thread | mean transition tightness | unchanged |
    | energy_shape_first | `_balance(len(low), len(high))` | unchanged |
    | fresh_crate | `min(newest_slice / 15, 1)` (pins at 1.0 on any real pool) | recency **concentration** of the shipped set: `max(0, 2·(freshness − 0.5))` — 0 at or below the pool's median age, → 1 as the set concentrates at the newest end |
    | genre_traverse | bridge-count + size-balance blend | unchanged |

- **distinctiveness** — `1 − max(shipped-set Jaccard vs every other row in the
  pool's final field)`. A pool whose final field is a **single row** gets 0.5
  (unknown, not "maximally distinct" — three live pools produce exactly one
  candidate, and a free 1.0 would let the thinnest pools top the aggregate rail).

**Honesty notes, recorded deliberately:**

- `feasibility` is strictly comparable **within** a pool; the web's aggregate
  "all" view flattens pools into one sorted list (existing behaviour, no worse
  than today's wall of 1.00). Cross-pool calibration is a possible follow-up,
  not v1.
- **Mined rows are seed-dependent** through the shared field: which named
  candidates exist depends on the seed (builders `rng.choice` their label /
  pole / artist), and dedupe + distinctiveness run against them. The
  determinism guarantee is: same pool + same seed → byte-identical output.
  (A previous draft claimed seed-invariant mining; that claim was false and is
  withdrawn.)
- Freshness (0.25) structurally favours recent material — measured live:
  `fresh_crate` freshness ≈ 0.97 vs 0.22–0.65 for other builders. Accepted
  bias: surfacing new material is this product's point. The double-count that
  *was* a bug (`fresh` predicate + freshness term) is gone with the predicate.

`feasibility` keeps its field name and 0–1 range; round to 4 decimals as today.

## 5. Mined briefs

Imperative first, evidence second — the brief is consumed as a **thesis** by
Stage 2 (`llm.py` renders `DIRECTION BRIEF:` and instructs the model to honour
it) and as a trigger-form **intent** by the web's "Run this direction". A brief
that leads with a statistic produces a report thesis about statistics. Template:

> FOUND SET. Play the corner of the crate where **{A}** meets **{B}** as a
> scene, not a coincidence — treat the conjunction as the thesis and let its
> shared sound carry the set. Evidence: {support} tracks sit in this overlap,
> {lift:.1f}× denser than chance{, of which the {shipped} most central are
> selected | omitted when support ≤ 25}. Anchors: {2–3 top-centrality tracks,
> artist — title}.

- Single-namable variant: "…where **{A}** clusters in one {tempo|harmonic}
  pocket…" — evidence lines may cite the pocket's BPM/key values
  (`genre_traverse` precedent); titles and moods never do.
- Anchors sit behind the imperative, never in the lead.
- No unplayed line: under `--mode unplayed` it is vacuous, under `--mode all`
  the CLI never fetches the catalog, and threading `played` into the miner
  would touch `library_map.py` — out of scope.

## 6. Error handling

Sparse metadata degrades silently, never raises: a kind with no qualifying
values simply contributes no predicates; a pool below 15 tracks mines nothing;
zero mined rows is normal output. Empty `date_added` → percentile 0.

## 7. Testing

- **Miner units:** predicate extraction per kind (gates, 70 % coverage cap,
  missing metadata); lift against hand-computed fixtures; the lift ≥ 1.3 gate
  (fixture: the previous draft's four exactly-chance pairs must all die);
  subsumption (a pair Jaccard ≈ 1.0 vs its parent dies); shortlist order;
  title uniqueness; naming rule (no mechanical values in any title/mood).
- **Determinism:** same pool + same seed twice → byte-identical combined
  candidate list; golden snapshot of the shared enumeration on a fixture pool.
  (Not "same pool mined twice" — that asserts purity of a pure function and
  cannot fail.)
- **Scorer units (function-level, not output-level):** spread — a synthetic
  pool where the old formula yields all ≈ 1.00 must yield materially distinct
  scores; freshness non-saturation on an all-unplayed pool; mood_journey
  identity on untruncated counts (poles 40/8 → 0.2, not 1.0); label_spotlight
  collection-lift scale; lone-row distinctiveness = 0.5; clone punishment
  tested by calling the pass-2 scorer directly on a planted near-duplicate
  field (dedupe removes true clones before output, so it is not observable
  end-to-end).
- **Run path:** mined cap of 1 slot in `generate_directions`; diagnostic line
  format; `found_N` types distinct and rank-ordered.
- **Integration:** `build_map_payload` fixture run contains valid `found_N`
  rows; existing direction tests updated for new score values (shapes
  unchanged).

## 8. Explicitly not doing

- No web or API changes; no payload shape change.
- No singletons; no `fresh` predicate; no same-kind pairs; no ≥3-predicate
  conjunctions; no FP-Growth.
- No LLM anywhere in this path.
- No mixability term in the score (measured non-discriminative; gate only).
- No cross-pool score calibration (v1 limitation, recorded in §4).
- No fact-chips UI, no merit-rank rewrite of within-builder `rng.choice`, no
  `artist_thread` gate change — separate follow-ups.

## 9. Sequencing for implementation

Two independently landable stages, in order:

1. **Scorer replacement** (`directions.py` only): shared enumeration factored,
   gates split from score, per-builder identity renormalisation, two-pass
   dedupe/distinctiveness. Validate against the live collection's 49 rows —
   scores must spread.
2. **Miner** (`mining.py` + wiring): lands on top of an already-honest ranking.

## 10. Adversarial review disposition

Independent review (Opus, fresh context, measured against the live collection)
found 4 blocking / 10 major / 9 minor issues in the first draft. All blocking
and major findings are incorporated above: web identity-key collision →
`found_N` types (B1); lift floor + like-for-like subsumption (B2); shipped-set
rule (B3); acyclic two-pass scoring (B4); mixability dropped as measured-constant
(M1); singletons dropped (M2); lone-row distinctiveness + log-scale lift (M3);
all-seven identity renormalisation table (M4); `fresh` predicate dropped as
tautology-generator (M5); prune-then-score (M6); seed-invariance claim withdrawn
(M7); shared enumeration + 1-slot run-path cap (M8); title uniqueness +
display-noun rule (M9); imperative-first briefs (M10). Minor findings folded into
§2–§7 text.

**Final whole-branch review (post-implementation, same reviewer protocol).** Three
corrections landed against the implemented branch and are reflected in the text
above:

- **`fresh_crate` identity was still saturating.** The implementation read
  `len(dated) / len(pool)`, which is 1.0 on any pool Rekordbox produced (it stamps
  `DateAdded` on ~every track) — the §4 table's "count-percentile form (spread
  restored)" had been read as a dated-share. The row now states the formula:
  `max(0, 2·(freshness − 0.5))`, recency *concentration* of the shipped set.
- **Distinctiveness was measured pre-cap.** §4 always said post-dedupe *post-cap*;
  the implementation scored the post-dedupe field and capped afterwards, so a mined
  row that never ships could reorder — or, at the run path's `mined[:1]`, replace —
  the one that does. The scorer is now split into `_rank_and_dedupe` (pass 1) and
  `_score_final` (pass 2 over exactly the field handed to it), with the mined cap
  and title-uniqueness rules applied in between.
- **§3.8's "first 8 chars kind-distinct" clause is withdrawn**, not implemented.
  With seven kind nouns the 8-char prefix collides for most pairs (`tempo x era`
  and `tempo x label` both truncate to `tempo x `), and no cheap reordering fixes
  it. The clause was also not load-bearing: only one mined row ships per run, so
  two mined moods never contend for a `_canvas_id`. ASCII-ness, which *is*
  load-bearing, stays.
