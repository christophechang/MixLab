# Deterministic Stage 1: Design Spec

**Issue:** #17  
**Status:** Spec — awaiting 10-run gate on `--intent` before implementation  
**Prerequisite:** `--intent` flag (#16) merged and used in ≥ 10 real genre runs

---

## 1. Problem

Stage 1 currently calls an LLM (free-cascade: Groq → Gemini → Mistral) to group a
genre-scoped track pool into 3–5 shortlists of 15–25 tracks. The prompt says "purely
technical pre-screening — not creative curation", but then asks for a title, a mood
line, and distinct BPM/key-character groupings.

The work the LLM is actually doing is **clustering**. Every dimension it uses —
BPM proximity, Camelot adjacency, era, tag character — is a numeric or categorical
comparison available deterministically. The LLM adds:

- **Non-determinism**: same pool, different partitions across runs. This undermines
  `--debug` diagnostics and the concept-novelty comparison in Stage 2 (which compares
  against prior runs).
- **Latency**: 5–15 seconds per genre call.
- **Cost**: consumes the cascade budget that Stage 0 (intent extraction, genuinely
  natural-language) needs.
- **Under-use of metadata**: tags, label, year, energy band, enrichment confidence —
  all available, all underused.

The fix: replace Stage 1 with `partition_pool()` in `clustering.py` — a deterministic
function that produces equivalent or better shortlists and runs in milliseconds.

---

## 2. Scope

**In scope:**
- New `partition_pool()` function (and private helpers) in `clustering.py`
- Replace all three Stage 1 LLM call sites in `__main__.py` (see §5)
- Remove `_STAGE1_SYSTEM`, `_STAGE1_SYSTEM_CUSTOM`, `_STAGE1_SYSTEM_PLAYLIST` from
  `llm.py` (after 30-day production soak — see §8)
- Update tests: replace LLM-mocked Stage 1 tests with deterministic-output tests
- Optional `--stage1-seed INT` CLI flag for reproducible tie-breaking

**Out of scope:**
- Stage 2 changes
- Canvas scoring changes
- Replacing the cascade for Stage 0 (intent extraction needs natural-language reasoning)
- Anchor candidate detection — that's per-canvas; this is pre-canvas clustering

---

## 3. Function signature

```python
def partition_pool(
    tracks: list[Track],
    *,
    seed: int | None = None,
) -> list[MixConcept]:
    """Partition a genre-scoped track pool into shortlists for Stage 2.

    Normal output: 3–5 shortlists of 15–25 tracks each.
    Small-pool exceptions (Step 1 early exits):
      - len(tracks) < 5 (ABSOLUTE_MIN): returns [].
      - len(tracks) < 15 (MIN_SHORTLIST): returns one shortlist with all tracks.
      - pools of 15–29 tracks where no detectable peaks exist (n_groups=1 fallback):
        one shortlist of up to 29 tracks — at most 4 over MAX_SHORTLIST. Accepted.
        This ONLY applies to 15–29 track pools (n_groups = min(5, len//15) = 1).
        Note: for len < 15, min(5, len//15) = 0 — but the '< MIN_SHORTLIST' early
        exit above fires first, so n_groups=0 is never reached in the fallback. The
        fallback can only receive n_groups ∈ {1, 2, 3, 4, 5}.
        Larger no-peak pools (30+) compute n_groups >= 2, produce multiple equal-
        sized BPM-ordered groups, and proceed through Steps 2–4 normally — they do
        NOT produce a single oversize shortlist. The "up to 29" exception is strictly
        limited to the n_groups=1 case, which is strictly limited to 15–29 track pools.
        Note: the flat-BPM guard (max-min < 4) is a DIFFERENT code path — it
        proceeds to Step 2 (Camelot sub-clustering) and then Step 4 sizing enforcement.
        A flat-BPM pool of 26+ tracks with a single Camelot component will be trimmed
        to 25 by Step 4 Attempt 3 (not kept whole). Only the n_groups=1 fallback
        bypasses Step 4 and may return up to 29 tracks intact.


    Deterministic replacement for the Stage 1 LLM call. Same pool + same seed =
    same output across runs. Most tie-breaks use track_id (lexicographic). The pool-
    level merge uses list index as a last resort, which is also deterministic because
    the shortlist list order is fully determined by BFS traversal order and snapshot
    sorting (both spec-mandated). Seed value has no effect in v1.

    Preconditions:
        - Tracks with bpm <= 0 should be excluded by the caller — they are treated as
          the lowest-BPM tracks and will distort cluster centroids.
        - An empty tracks list is valid input: the `len < ABSOLUTE_MIN` early exit
          returns [] immediately without reaching any helper that requires non-empty.
        - All track_ids in the pool must be unique. Duplicate track_ids produce
          non-deterministic min_track_id tie-breaks and BFS ordering. The caller
          (genre pool construction in __main__.py) must ensure uniqueness.

    Note: this function accepts only tracks and seed. The genre label and
    cascade_state passed to the old stage1_concepts() are not needed — genre
    is already implicit in the pool, and cascade_state is LLM infrastructure.

    Note on seed: in v1 the output is fully deterministic — track_id is the final
    tie-breaker in every ordering where other criteria tie. (Not ALL tie-breaks use
    track_id: Step 1 peak-merge uses lower-BPM for survivor selection and lowest
    combined BPM for merge-pair distance ties; Step 3 uses smaller gap index; pool-
    level merge uses index position as last resort.)
    Every seed value (including None) produces identical output. The parameter is
    reserved for a future BPM-jitter extension. Callers may always pass seed=None.

    Args:
        tracks: Genre-scoped, mode-filtered pool. Must be non-empty.
        seed: Reserved for future use. Has no effect in v1.

    Returns:
        List of MixConcept (title + mood + track_ids). Returns [] when:
          - pool has fewer than ABSOLUTE_MIN=5 tracks (Step 1 early exit), OR
          - Step 4 drops every shortlist below ABSOLUTE_MIN (extremely fragmented
            pool; rare in practice — callers should treat [] as "no usable partition").
    """
```

Return type reuses `MixConcept` (title, mood, track_ids) so the output is
drop-in compatible with the Stage 1 LLM path — no downstream changes required.

**Dropped parameters:** the old `stage1_concepts(pool, genre, cascade_state, ...)` call
takes a `genre` string and `cascade_state` object. Neither is needed by `partition_pool()`:
genre is implicit in the pool composition, and `cascade_state` is LLM infrastructure
that disappears with the LLM path. The feature flag wrapper (§8) passes them only
when falling back to `stage1_concepts()`.

---

## 4. Algorithm

### Shared constants

```python
MIN_SHORTLIST          = 15
MAX_SHORTLIST          = 25
ABSOLUTE_MIN           = 5   # shortlists below this are dropped rather than passed to Stage 2
MIN_POOL_COUNT         = 3
MAX_POOL_COUNT         = 5
MIN_CAMELOT_COMPONENT  = 8   # components smaller than this are merged in Step 2.4
```

**`min_track_id(shortlist)`** is used as a sort key throughout Step 4. It means:
`min(t.track_id for t in shortlist)` — lexicographic minimum string, NOT numeric.
Example: `min_track_id(["10", "9"]) == "10"` (string "10" < "9" lexicographically).

### Helper: `_median_bpm(shortlist)`

```python
def _median_bpm(shortlist: list[Track]) -> float:
    return statistics.median(t.bpm for t in shortlist)
```

Used throughout Steps 1 and 4. Callers must ensure the shortlist is non-empty and
all tracks have bpm > 0 (see §3 precondition).

### Step 1 — BPM clustering (primary partition)

**Early exits (checked before histogram construction):**

```
- If len(tracks) < ABSOLUTE_MIN (< 5): return [].
- If len(tracks) < MIN_SHORTLIST (< 15): return [all tracks as one MixConcept].
  Track order in the MixConcept: sort ascending by BPM, then by track_id
  (lexicographic) for ties — same order as Step 1.1 would produce.
  Skip Steps 2–4. (No clustering needed — the pool is too small to split.)
- If all tracks have |bpm_i - bpm_j| < 4 for all pairs (equivalently: max_bpm -
  min_bpm < 4; implement as O(n) via min/max): treat all tracks as a single BPM
  cluster and proceed directly to Step 2 with that one cluster.
  This bypasses histogram construction, smoothing, peak detection, AND the
  uniform-spread fallback entirely.
  Boundary note: a span of EXACTLY 4.0 (max_bpm - min_bpm == 4.0) does NOT satisfy
  the strict < 4 guard and falls through to histogram construction. With e.g.
  min=120.0, max=124.0 the alignment may produce only 2 buckets ([120,123) and
  [123,126)), which triggers the "< 3 buckets → skip smoothing and peak detection,
  proceed directly to uniform-spread fallback" rule below. This is expected and
  acceptable — a 4-BPM span is effectively flat.
```

**Histogram construction:**

```
1. Sort tracks ascending by BPM, then by track_id (lexicographic) for ties.
   (Equal-BPM tracks are common; without a secondary key their order is input-order
   dependent, breaking same-pool determinism.)
2. Bin tracks into 3-BPM-wide histogram buckets using left-inclusive, right-exclusive
   intervals [lo, lo+3). The first bucket starts at floor(min_bpm / 3) * 3.
   Example: min_bpm=122.4 → first bucket is [120, 123).
   A track with bpm at an exact boundary falls in the NEW bucket:
   bpm=123.0 falls in [123, 126), not [120, 123).
   Create ALL consecutive buckets from the first start up to and including the bucket
   that contains max_bpm, even if intermediate buckets are empty (count=0). Empty
   interior buckets ARE included — they preserve the spatial structure of the BPM
   distribution.
   Total bucket count = (bucket_index(max_bpm) - bucket_index(min_bpm) + 1), where
   bucket_index(b) = (floor(b / 3) * 3 - first_bucket_start) // 3.
   Note: because bpm=123.0 falls in bucket [123,126) (the left-inclusive rule from
   above), a pool spanning [120.0, 127.0] (span=7, >=4, not caught by flat-BPM guard)
   with min_bpm=120.0 produces 3 buckets: [120,123), [123,126), [126,129).
   A narrower example — min=120.0, max=123.0 — has span=3 < 4 and is caught by the
   flat-BPM guard (line 149) BEFORE histogram construction; that guard bypasses the
   histogram entirely, so no 2-bucket histogram is actually constructed for it.
3. If number of buckets (including empty ones) < 3: skip smoothing AND peak detection
   entirely; proceed directly to the uniform-spread fallback.
   Otherwise, smooth the histogram with a 3-bucket moving average.
   Let N = index of the last bucket = (total bucket count - 1), 0-based, counting all
   buckets including empty ones. Example: 5 buckets → N=4; buckets are indexed 0..4.
   - Interior bucket i (0 < i < N): smoothed[i] = (raw[i-1] + raw[i] + raw[i+1]) / 3
   - Left edge (i=0): smoothed[0] = (raw[0] + raw[1]) / 2
   - Right edge (i=N): smoothed[N] = (raw[N-1] + raw[N]) / 2
   The divisor is the number of buckets in the window (2 for edges, 3 for interior),
   NOT the number of buckets with non-zero raw count. Empty (count=0) buckets are
   fully included in the window and counted in the denominator.
```

**Peak detection:**

```
4. Find local maxima using a plateau-aware algorithm:
   a. Decompose the smoothed histogram into maximal runs of consecutive equal values.
      A run of length 1 is a single-bucket (non-plateau) run.
   b. For each run:
      If the run starts at i=0 OR ends at i=N: no peak (edge run). Skip to next run.
      Otherwise (interior run only):
        L = smoothed value of the bucket immediately to the left of the run.
        R = smoothed value of the bucket immediately to the right of the run.
        If run_value > L AND run_value > R: the leftmost bucket of the run is a
          peak. (Single-bucket runs with this property are standard strict peaks.)
        Else: no peak for this run.

Note: this unified algorithm replaces the "strict rule + plateau exception" framing.
It produces identical results to the strict rule for single-bucket runs, and correctly
identifies plateau peaks for multi-bucket runs — both as a single pass, no separate
plateau detection step required.
Note on monotone histograms: a strictly-increasing or strictly-decreasing 3-bucket
histogram (e.g., smoothed = [3, 5, 7]) has its sole interior bucket not exceeding its
right neighbour (R > run_value), so no peak is found → uniform-spread fallback. This
is intentional: a monotone BPM distribution has no meaningful cluster peak.
Note on edge-smoothing inflation: a 3-bucket histogram with a narrow unimodal peak in
the middle bucket can still fail peak detection. Example: raw = [0, 10, 0] →
smoothed = [5.0, 3.33, 5.0]. The interior bucket (3.33) is below both smoothed edges
(5.0), so no peak → fallback. This is a known limitation of the 3-bucket moving average:
edge buckets average with the interior, inflating edge smoothed values above the interior
for unimodal distributions that sit entirely in the interior bucket. Such pools fall
back to the uniform-spread path, which is acceptable for a single tight cluster.

5. Complete ALL peak merges before beginning track assignment (Step 6).
   Merge peaks that are closer than 8 BPM (strict: distance < 8.0) — on each iteration:
   - Find the pair with the smallest |centre_a - centre_b|.
   - Merge: keep the peak with the higher raw track count ("heavier").
     "Raw track count" for a peak = the SUM of raw counts across ALL buckets in the
     peak's plateau run (or, for a single-bucket peak, just that bucket's count). This
     prevents the leftmost-bucket rule from creating zero-count plateau peaks that
     always lose the heavier comparison. After a merge, the surviving peak's raw count
     is its ORIGINAL plateau-sum count (not recomputed to include the absorbed tracks).
     Note: if both peaks are single-bucket peaks on empty buckets (count=0), both sums
     are 0 — this is a tie resolved by the rule below.
   - Tie (equal raw count, including both=0): keep the lower-BPM peak.
   - Distance tie (two pairs are equidistant): choose the pair with the lowest
     combined peak BPM (sum of the two centres).
   - Combined-BPM tie (two pairs share equal distance AND equal centre-sum): choose
     the pair whose lower-BPM peak has the lower bucket index (i.e. lower centre).
   - Repeat until no two peaks have distance < 8.0.

Note on merge vs assignment thresholds: 'distance < 8 BPM' for merging and
'|bpm - centre| ≤ 7.0' for assignment in Step 6 are intentionally different in form.
After merging, two adjacent peaks are at least 8.0 BPM apart; their ±7.0 assignment
windows may overlap by up to 6 BPM (e.g., peaks at 120 and 128 share the window
121–127). The nearest-peak tie-break in Step 6 (lower-index peak wins) resolves all
such overlaps — no track can be assigned to two peaks.
Implement both thresholds exactly as written (< 8.0 for merging, ≤ 7.0 for assignment).
```

**Peak centre** = midpoint of the 3-BPM bucket: `bucket_start + 1.5`.
After a merge, the surviving peak retains its original centre (not recomputed).
Note on wide plateau bias: a plateau run spanning multiple buckets uses the LEFTMOST
bucket's midpoint as the peak centre. Tracks in the high-BPM end of a wide plateau
(e.g., 4+ buckets = 12+ BPM wide) may fall outside the leftmost-centre ±7 window and
become outliers even though they sit squarely within the plateau. This is accepted
behaviour — outliers are attached in Step 7 via nearest-cluster assignment.
Tracks that fall in the eliminated peak's ±7 window but outside the surviving peak's
±7 window will have no candidate peak when Step 6 runs — they are assigned as
outliers by Step 6's "no candidate peak within ±7 BPM" rule and attached in Step 7.
(Since Steps 6–7 run AFTER all merges complete, no track is ever "re-classified";
it simply finds no candidate peak during assignment.)

**Track assignment:**

```
6. Assign each track to a peak:
   - Candidate peaks: those with |track.bpm - peak.centre| <= 7.0 (inclusive).
   - Assign to the nearest candidate (smallest distance).
   - Ties (equidistant from two peaks): assign to the peak with the lower index in the
     sorted ascending-BPM peak list.
   - Tracks with no candidate peak within ±7 BPM become "outliers".
7. Attach each outlier unconditionally to the nearest cluster by absolute BPM distance
   (no cap). There is no "hold as merge candidate" step.
   Ties (equidistant from two clusters): attach to the cluster with the lower index
   in the sorted ascending-BPM peak list (same list used in Step 6).
```

Typically 2–6 raw BPM clusters before sizing adjustments. A unimodal pool may yield
exactly 1 cluster (one dominant peak after merging); the uniform-spread fallback also
produces exactly n_groups clusters. The 2-6 range is a description of the typical
case, not a hard guarantee.

**Uniform-spread fallback** (triggered when: no peaks found, or histogram has < 3 buckets):

```
n_groups = min(MAX_POOL_COUNT, len(tracks) // MIN_SHORTLIST)
If n_groups < 2:
  Skip outer algorithm Steps 2, 3, and 4 (Camelot sub-clustering, era split, and
  sizing enforcement). Proceed directly to outer Step 5 (mood/title inference) and
  outer Step 6 (MixConcept assembly) using all tracks as a single group (in the
  ascending-BPM order from Step 1.1), then return.
  Note: "outer Steps 5–6" refers to the top-level §4 algorithm steps, NOT to
  intra-Step-1 sub-items 5 (peak-merge) and 6 (track-assignment), which are
  inapplicable when no peaks exist.
  Note: pools of 15–29 tracks yield n_groups = 1. The returned shortlist may have
  up to 29 tracks — 4 over MAX_SHORTLIST. This is accepted whenever no meaningful
  split exists (any pool that reaches this fallback with n_groups=1, not only
  flat-BPM pools — see docstring for the distinction).
  Note on asymmetry with the peaks path: a 26–29 track cluster that arrives via the
  peaks path (i.e. it was produced by Step 1 peak-windowing or by the n_groups≥2
  uniform spread) DOES pass through Step 4 sizing enforcement, where it may be split
  or trimmed to fit [15, 25]. The n_groups=1 fallback intentionally bypasses Step 4
  because, with only one group, there is no partner to split into and no surplus to
  discard meaningfully; the minor overrun (≤4 tracks) is the accepted trade-off for
  simplicity.
Otherwise: split into n_groups equal-sized groups using CONTIGUOUS slices of the
  ascending-BPM sort from Step 1.1 (reuse it — do not re-sort; do NOT round-robin).
  Group sizes: if len(tracks) % n_groups != 0, the first (len(tracks) % n_groups)
  groups each receive one extra track. E.g. 31 tracks / 2 groups = [16, 15]
  (tracks[0:16] and tracks[16:31]).
  Proceed to Step 2 with these groups.
```

**Note on `camelot_compatible` adjacency:** the existing `camelot_compatible()` function
connects only same-key, ±1-same-mode, or same-number-opposite-mode pairs (maximum
Camelot distance of 1). Tracks 2+ steps apart are not adjacent. In Step 2, keys that
are musically close but 2 steps apart on the wheel will form separate components and
be merged back via the < 8-track merge rule. This is intentional — the coarser
adjacency is conservative and correct. Do not broaden the definition.

### Step 2 — Camelot sub-clusters (within each BPM cluster)

```
1. Build an adjacency graph: edge between tracks A and B iff
   camelot_compatible(A.camelot_key, B.camelot_key) is True.
   (camelot_compatible returns False for any unparseable key, making that track a
   Camelot singleton. Tracks with unknown keys form their own singleton components
   and are merged into the largest component in Step 2.4.)
   **Key normalisation**: before ALL Camelot comparisons and Counter operations
   throughout the algorithm (Steps 2, 4 Attempt 3 centrality, and Step 5 mood
   inference), normalise every camelot_key to UPPERCASE (e.g. `key.upper()`).
   _CAMELOT_RE uses IGNORECASE, so '8a' and '8A' are both parseable; without
   normalisation they count as distinct keys, splitting frequency counts and
   producing non-deterministic dominant_key selection.
2. Find connected components via BFS. For determinism:
   - Sort all tracks ascending by track_id (lexicographic string order, NOT numeric)
     before building the adjacency graph.
   - BFS queue: start from the unvisited track with the lexicographically smallest
     track_id; expand neighbours in ascending lexicographic track_id order.
   - This fixes the internal track ordering within each component.
3. If ALL tracks in the BPM cluster have no/unknown Camelot key (empty string or
   doesn't match _CAMELOT_RE): treat the entire cluster as a single component — keep
   as-is. This guard prevents the BFS from producing N singletons.
   "All unknown" means every track in the cluster fails _CAMELOT_RE.match().
4. For each component with < MIN_CAMELOT_COMPONENT (< 8) tracks: merge it into the
   current-largest component in the same BPM cluster. The source component being
   merged is NOT a candidate for "current-largest" — only other components are.
   Remove the source component from the component list immediately after merging.
   Guard: if the component is the ONLY component in its BPM cluster (no other
   component exists), keep it as-is and skip.
   Note: within partition_pool's own Step 2, this guard is unreachable for pools with
   >= 15 tracks (the < MIN_SHORTLIST early exit fires first), so a lone BPM cluster
   of fewer than 8 tracks cannot be constructed via normal Step 2 execution. However,
   Step 4 Attempt 1 re-runs BFS on an oversized shortlist — that BFS re-run is NOT
   subject to the < 15 early exit and CAN produce a lone component of any size.
   Step 4 undersizing handles under-MIN_SHORTLIST components.
   - The set of merge SOURCES (components with < 8 tracks) is determined once at
     the start of Step 2.4. Merges only increase component sizes, so no new sub-8
     source can appear mid-step. The candidate set monotonically shrinks.
   - Apply merges in size-ascending order (smallest component first).
   - After each individual merge, recompute component sizes before selecting the next
     merge source (to ensure the current-LARGEST TARGET is up to date). The source
     candidates themselves are fixed (those with < 8 tracks at step start).
   - Tie in size (two source components both have equal size): process the one
     whose minimum track_id is lexicographically smaller first.
   - Tie in "largest" target (two components share the max size after a recompute):
     target the one whose minimum track_id is lexicographically smaller.

Note on thresholds: Step 2 uses MIN_CAMELOT_COMPONENT=8 as the threshold for
merging small Camelot components during initial clustering. Step 4 Attempt 1 uses
MIN_SHORTLIST=15 as the threshold for whether a split produces shortlists usable
for Stage 2. These are different concerns: 8 tracks is the minimum for a
musically coherent sub-cluster; 15 tracks is the minimum for a useful shortlist.
5. After small-component merges, if 2+ components remain and each has ≥ 8 tracks:
   treat each surviving component as a candidate sub-partition.
```

### Step 3 — Era split (optional)

Applied independently to each candidate shortlist *after* Camelot sub-clustering.

```
1. Collect known-year tracks: t.year is not None and t.year > 0. Call this set KY.
   "Unknown-year tracks" = all tracks NOT in KY (i.e. year is None OR year <= 0).
2. If len(KY) / len(shortlist) < 0.60: skip era split for this cluster.
   len(shortlist) is the total count of all tracks in the shortlist (including
   unknown-year tracks, which are not in KY). Tracks with bpm <= 0 are excluded by
   the §3 precondition and will never reach this step.
3. Sort KY ascending by year, then by track_id (lexicographic) for tracks with
   equal year. Call the sorted list ky (length K).
   If K < 2: skip era split (no consecutive pair exists to compute a gap).
4. Find gap_idx: the index i (0 ≤ i ≤ K-2) that maximises ky[i+1].year - ky[i].year.
   gap_idx is the index of the last track BEFORE the gap.
   Tie (two equal gap sizes): choose the smaller i.
   gap_start = ky[gap_idx].year
   gap_size  = ky[gap_idx + 1].year - gap_start
5. Count era_old_known = number of tracks in KY with year <= gap_start.
   Count era_new_known = number of tracks in KY with year >  gap_start.
   (These counts use only KY — unknown-year tracks are not counted here.)
6. If gap_size >= 8 AND era_old_known >= per_side_min AND era_new_known >= per_side_min:
   where per_side_min is a parameter (default 8 for base Step 3; 15 for Step 4 Attempt 2).
   IMPORTANT: per_side_min is checked against era_old_known and era_new_known — counts
   of KNOWN-year tracks only (from KY). Unknown-year tracks assigned to era_old in
   Step 3.6 do NOT count toward the era_old_known >= per_side_min threshold. A
   shortlist with many unknown-year tracks (assigned to era_old unconditionally) can
   still fail the era_old_known guard if fewer than per_side_min tracks have a
   confirmed known year <= gap_start.
   Note: the default >=8 per-side threshold is intentionally LOWER than MIN_SHORTLIST=15.
   A 20-track shortlist split into two 10-track halves (each with 10 known-year
   tracks) passes Step 3 but produces two sub-MIN_SHORTLIST shortlists. Step 4
   under-sizing then merges them back (or grows them). This is the intended
   design — Step 3 splits optimistically; Step 4 enforces the size contract.
   Note on --min-year interaction: the caller's --min-year CLI flag (applied upstream
   before Stage 1) means every track in the pool has a year >= min_year, making
   year known for all tracks. If --min-year is set, the 60% coverage gate effectively
   always passes. Implementers should be aware that era-split will trigger more often
   in --min-year runs than in unfiltered runs.
   - era_old: ALL tracks in the shortlist EXCEPT those with year>0 AND year>gap_start.
     Equivalently: era_old = (known-year tracks with year > 0 AND year <= gap_start)
                           ∪ (unknown-year tracks with year=None OR year<=0).
     Unknown-year tracks are ALWAYS added to era_old — no conditional logic, no
     centroid computation. era_old holds known-old tracks AND all unknowns.
   - era_new: ONLY tracks with year is not None and year > 0 and year > gap_start.
     Unknown-year tracks never appear in era_new.
   Note: the era_old membership definition above supersedes the sub-bullet below —
   do NOT implement era_old as just 'year>0 AND year<=gap_start'; you must include
   the unknown-year tracks too.
     IMPORTANT: this is an unconditional rule — do not compute centroids or apply any
     conditional logic. era_old always receives unknown-year tracks.
7. Otherwise: no era split.
```

### Step 4 — Sizing enforcement

After BPM + Camelot + era passes, the candidate shortlists are resized to meet
the 15–25 track target. All comparisons use current shortlist contents.

**Sizing pass order:** execute as a loop of at most 3 iterations, where one iteration
= one oversized pass followed by one undersized pass. "Current pass" means the
half-iteration currently executing (oversized OR undersized), not the whole iteration.
Within each pass, snapshot the shortlist list sorted at the start of that pass:
  - Oversized pass: sort by (length descending, min_track_id ascending).
  - Undersized pass: sort by (length ascending, min_track_id ascending).
Iterate through the snapshot in order. During iteration:
  - Shortlists created or split during the CURRENT HALF-PASS (oversized OR undersized)
    are NOT in that half-pass's snapshot and are NOT processed in it. "Current pass"
    in all rules below means the current half-pass, not the whole iteration.
  - A shortlist created by the oversized half-pass IS captured in the snapshot taken
    at the START of the following undersized half-pass (the snapshot is fresh each half).
  - Shortlists already consumed (merged into another) during the current half-pass are
    skipped when their snapshot slot is reached — check that the shortlist still
    exists in the live list before processing it.
Exit the loop early (before 3 iterations) after completing BOTH passes of an
iteration if NEITHER the oversized pass NOR the undersized pass of that iteration
made any change. Checking is per full iteration (both halves), not per individual pass.
Note on last-iteration oversized output: the undersized pass has overflow checks
in both Phase 1 and Phase 2 (neither can produce > MAX_SHORTLIST). The real sources
of > MAX_SHORTLIST results are:
  1. Pool-level merge (after per-shortlist passes, documented separately in §9).
  2. Attempt 3 remainder-attach (in the oversized pass): trimming a 30-track shortlist
     to 25 and attaching the 5-track remainder to a 24-track neighbour yields a
     29-track neighbour (> MAX_SHORTLIST). The modified neighbour is not re-processed
     in the current pass (it was already handled or was not in the oversized snapshot).
     If this is the last iteration, that oversized neighbour is returned as-is.
The §9 parenthetical covers both.
**A change** is defined as: any shortlist being merged (under-sizing), split into two
(Attempt 1 or 2), trimmed with or without remainder attached (Attempt 3), or dropped
(len < ABSOLUTE_MIN). Simply evaluating a shortlist and leaving it untouched does NOT
count as a change.

**"Original index" and live-list placement** throughout Step 4:
Track each shortlist by object identity (reference), not by numeric index. When the
spec says "place at the lower of the two original indices," this means:
  1. Identify which of the two shortlists had the lower index in the snapshot at the
     START of the current pass.
  2. Find that shortlist's CURRENT position in the live list (by reference lookup).
  3. Replace it in-place with the result; remove the other shortlist from the live list.
This handles the case where prior operations in the same pass have shifted positions:
after a split inserts two entries at index i, a later shortlist's current live-list
position may differ from its snapshot index — using reference lookup is correct.

**Under-sizing (len < MIN_SHORTLIST):**

```
Search the current undersized-pass snapshot (excluding the shortlist being processed
AND excluding any shortlists already consumed/merged during the current undersized pass)
for the other shortlist with the smallest abs(_median_bpm(this) - _median_bpm(other)).
Do NOT include shortlists added to the live list during the current undersized pass
itself — only snapshot shortlists that still exist in the live list are eligible.
Note: shortlists created by the preceding oversized pass ARE in the undersized-pass
snapshot (the spec states they are eligible); they have defined snapshot indices.
(Tie in BPM distance: choose the shortlist whose minimum track_id is lexicographically
smaller.)

If no other shortlist exists (only one shortlist in the pool):
  - Keep it as-is regardless of size (cannot satisfy MIN_POOL_COUNT=3 anyway).
  - Note: a lone shortlist with len < ABSOLUTE_MIN cannot be reached here in
    practice — Step 1 filters pools < ABSOLUTE_MIN before any clustering begins,
    and Steps 2–3 cannot reduce a shortlist below 8 tracks. This branch is a
    defensive guard only.

If a merge partner exists, apply in two phases:

Phase 1 — first attempt (closest partner only):
  - If merging the closest partner would exceed MAX_SHORTLIST:
    Do NOT merge in Phase 1. Enter Phase 2 directly, using the PRE-Phase-1 median
    of the undersized shortlist as the reference BPM for Phase 2 distance comparisons.
    (Skipping Phase 2 here would leave the shortlist undersized even when a farther,
    smaller partner fits within MAX_SHORTLIST — an unnecessary undersized result.)
    If len(undersized) < ABSOLUTE_MIN, drop it and skip Phase 2. This is a defensive
    guard; in practice Step 1 early-exits pools < ABSOLUTE_MIN before clustering.
  - Otherwise: merge with the closest partner. Place the merged shortlist at the
    lower of the two original indices; remove the higher-index shortlist from the
    live list.

Phase 2 — retry loop (entered when: (a) Phase 1 merged but result is still
  < MIN_SHORTLIST, OR (b) Phase 1's closest partner would overflow and was skipped):
  Reference BPM for Phase 2 distance comparisons:
    - Case (a) post-Phase-1-merge: use the merged shortlist's current _median_bpm().
    - Case (b) Phase 1 overflow (no merge): use the original shortlist's _median_bpm()
      (the same value that was the pre-Phase-1 median — the shortlist is unchanged).
  Compute partner ordering ONCE at Phase 2 entry: ascending _median_bpm() distance
  from the reference BPM. Do NOT recompute after each merge — the ordering is fixed
  at Phase 2 entry regardless of merges that occur within Phase 2.
  In case (b) the closest-distance partner in Phase 2's ordering is the same partner
  Phase 1 already rejected (it overflows). Skip it immediately (overflow check:
  current_len + partner_len > MAX_SHORTLIST) and proceed to the next.
  Tie in distance: choose the partner with the lexicographically smaller min_track_id.
  "Remaining partners" = snapshot partners not yet processed in Phase 2 AND still
  present in the live list (not consumed during the current undersized pass).
  For each remaining partner in the fixed order:
    - Overflow check: use the CURRENT length of the accumulating merged shortlist
      (which grows after each successful Phase 2 merge), NOT the original length.
      Skip if current_len + partner_len > MAX_SHORTLIST.
    - Otherwise: merge. Place result at the lower of the two UNDERSIZED PASS
      snapshot-time indices (resolve by reference lookup as defined above);
      remove the other shortlist. If result >= MIN_SHORTLIST: STOP (success).
      Partners created by the preceding oversized pass are in the undersized snapshot
      and have defined snapshot indices — treat them identically to other partners.
  End loop when: (a) result >= MIN_SHORTLIST, (b) no un-skipped partners remain,
  or (c) all remaining partners would overflow.
  After the loop: keep the shortlist if len >= ABSOLUTE_MIN, otherwise drop it.

Note: shortlists dropped (< ABSOLUTE_MIN) during per-shortlist passes are not
resurrected by the pool-level MIN_POOL_COUNT guard later.
```

**Over-sizing (len > MAX_SHORTLIST):**

```
Attempt 1: split by Camelot component. Run BFS on the oversized shortlist's tracks.
  This attempt succeeds only when BFS yields EXACTLY 2 components and both have
  >= MIN_SHORTLIST tracks. Do NOT re-apply the Step 2.4 small-component merge rule
  here — use raw BFS components only. With 1 or 3+ BFS components, or with either
  component below MIN_SHORTLIST, skip to Attempt 2.
  Note: shortlists formed in Step 2 by merging multiple small components typically
  produce 3+ raw BFS components, causing Attempt 1 to fail. This is expected and
  intentional — Attempts 2 and 3 serve as fallbacks for such shortlists.
  Note: oversized shortlists of 26–29 tracks can NEVER satisfy 'two components each
  >= MIN_SHORTLIST=15' (would need >= 30 tracks total). Attempt 1 is unreachable for
  that size range — Attempt 2 or 3 always handles them.
  On success: replace the oversized shortlist with the two components in ascending
  _median_bpm() order. Resolved by reference lookup:
    - First component (lower-BPM): replaces the oversized shortlist at its current
      live-list position.
    - Second component (higher-BPM): inserted immediately after the first (at the
      position directly following the first component in the live list).
  Tie (both components have equal _median_bpm()): place the component with the
  lexicographically smaller min_track_id first.
Attempt 2: re-apply the full Step 3 era-split logic (including the 60% coverage
  gate) to the oversized shortlist. The coverage gate is evaluated on the oversized
  shortlist's current length — a larger shortlist may fail the 60% threshold even if
  its original sub-components would have passed it individually. This is intentional.
  Attempt 2 succeeds only when ALL of the following hold:
    (a) Step 3's logic produces a split (none of the skip conditions in Steps 3.2–3.6
        fired — coverage gate passes, K >= 2, gap_size >= 8, both sides >=
        per_side_min; for Attempt 2, per_side_min = 15, NOT the default 8 used in
        base Step 3 — the step is re-invoked with per_side_min=15 here);
    (b) len(era_old) >= MIN_SHORTLIST; AND
    (c) len(era_new) >= MIN_SHORTLIST.
  NOTE on population differences: conditions (a), (b), and (c) check DIFFERENT
  populations. Condition (a) gates on era_old_known and era_new_known (KY-only
  counts, i.e. only tracks with confirmed known year). Conditions (b) and (c) gate
  on the full era_old and era_new lists, which for era_old also includes unknown-year
  tracks (unconditionally assigned to era_old in Step 3.6). Since era_new NEVER
  receives unknown-year tracks, era_new_known == len(era_new), so conditions (a.new)
  and (c) are equivalent checks — but era_old_known can be much smaller than
  len(era_old). A shortlist with 14 known-year-old tracks and 20 unknown-year tracks
  in era_old has era_old_known=14 (fails condition a, per_side_min=15) even though
  len(era_old)=34 (would pass condition b). Condition (a) fires first and causes
  Attempt 2 to fail — condition (b) is never evaluated for that shortlist.
  If Step 3 produces no split (any skip condition fires), OR if either half is below
  MIN_SHORTLIST, treat Attempt 2 as failed and proceed to Attempt 3.
  Note: because all unknown-year tracks are assigned to era_old (Step 3.6), era_new
  is drawn entirely from KY. A shortlist that passes the 60% coverage gate can still
  have era_new < MIN_SHORTLIST if fewer than 15 known-year tracks fall after gap_start.
  This structural bias means Attempt 2 fails more often than the coverage gate alone
  suggests — do not assume Attempt 2 succeeds just because the gate passes.
  On success: replace the oversized shortlist with era_old first (at original index,
  resolved by reference lookup), era_new second (at reference position + 1).
Attempt 3: rank by centrality ascending, keep the first MAX_SHORTLIST tracks as the
  trimmed shortlist. Replace the oversized shortlist with the trimmed shortlist at
  its current live-list position (resolved by reference lookup).
  Attach the remainder (tracks beyond rank MAX_SHORTLIST) to the nearest other
  shortlist by _median_bpm() distance measured from the TRIMMED shortlist (not from
  the remainder group).
  Tie in distance: attach to the shortlist whose minimum track_id is
  lexicographically smaller.
  If no other shortlist exists, discard the remainder. This is accepted behavior —
  a single large coherent cluster (e.g. flat-BPM, single Camelot key) cannot be split
  further; returning the most central MAX_SHORTLIST tracks is the best available output.
  Emit a diagnostic when tracks are discarded:
  `print(f"partition_pool: Attempt 3 discarded {len(remainder)} tracks (no split target).", file=sys.stderr)`
  Note: appending the remainder to an existing shortlist may push that shortlist over
  MAX_SHORTLIST. This is accepted — do NOT trigger a recursive sizing pass. Stage 2
  tolerates slight oversize, same as the pool-level merge note below.

  Centrality computation:
    # Normalise to uppercase before counting (see Step 2.1 key normalisation note).
    parsed_keys = [t.camelot_key.upper() for t in shortlist
                   if t.camelot_key and _CAMELOT_RE.match(t.camelot_key)]
    # Deterministic dominant_key: sort by (-count, key_str) to break count ties.
    # Counter.most_common() is non-deterministic for equal counts (heap insertion order).
    dominant_key = (
        min(parsed_keys, key=lambda k: (-Counter(parsed_keys)[k], k))
        if parsed_keys else None
    )
    bpm_vals = [t.bpm for t in shortlist]
    bpm_range = max(bpm_vals) - min(bpm_vals)
    if bpm_range < 1e-6:
        bpm_range = 1.0   # all tracks have effectively identical BPM
    For each track t:
      bpm_norm = abs(t.bpm - _median_bpm(shortlist)) / bpm_range
      # bpm_norm ∈ [0, 1.0]: max deviation from median is at most bpm_range
      # (since all tracks lie in [min_bpm, max_bpm]), so bpm_norm ≤ 1.0.
      if dominant_key is None:
        camelot_norm = 1.0   # no parseable keys in cluster — max peripheral score
      else:
        d = camelot_distance(t.camelot_key.upper(), dominant_key)
        # camelot_distance returns 999 for any unparseable key (including when
        # dominant_key is parseable but t.camelot_key is not). Cap at 1.0.
        # dominant_key is already upper() from the parsed_keys comprehension above.
        camelot_norm = 1.0 if d >= 999 else d / 7.0
        # 7.0 is the maximum finite camelot_distance (e.g. '1A' to '7B' = 7).
        # camelot_norm ∈ [0, 1.0]. For same-ring keys, max finite d=6 → norm=6/7≈0.857.
        # For cross-ring keys, max finite d=7 → norm=1.0 (same as unparseable).
        # The 999 guard must be kept — do not remove it as "dead code".
      centrality = bpm_norm + camelot_norm
      # centrality ∈ [0, 2.0]. Do not cap or normalise — sort uses raw value.
    Sort key: 2-tuple (centrality, track_id) ascending, where track_id is compared
    lexicographically (string comparison, not numeric). Using track_id as the explicit
    secondary key (not relying on input-list order + sort stability) ensures the
    result is deterministic regardless of how tracks were ordered in the shortlist.
```

**Pool-level adjustments (after all per-shortlist passes):**

```
If total count < MIN_POOL_COUNT: keep all remaining shortlists (even if undersized).
If total count > MAX_POOL_COUNT:
  Repeat:
    Find the pair of shortlists with the smallest abs(_median_bpm(A) - _median_bpm(B)).
    Tie-break chain (apply in order until resolved):
      1. Smallest min(min_track_id(A), min_track_id(B)) — lexicographically.
         This is the lex-minimum of the two shortlists' own min_track_ids, i.e. the
         globally-smallest track_id that appears in EITHER shortlist of the pair.
         If two pairs share the same globally-smallest shortlist (both contain the
         shortlist with the overall minimum track_id), this step does NOT resolve
         the tie — fall through to step 2.
      2. Smallest sum of the pair's 0-based indices in the current shortlist list.
      3. Lowest index of the lower-index member of the pair (i.e. min of the two
         shortlists' indices in the current shortlist list). "First shortlist in the
         pair" means the member with the lower 0-based index — not the shortlist that
         was discovered first or listed first in a BPM sort.
    Merge that pair. Insert the merged shortlist at the lower of the two original
    indices; remove the higher-index shortlist. (This preserves stable ordering.)
  Until total count <= MAX_POOL_COUNT.
  Note: the pool-level merge may produce a shortlist exceeding MAX_SHORTLIST.
  Do NOT re-run per-shortlist sizing on the result — Stage 2 tolerates slight oversize.
```

### Step 5 — Mood/title inference

Each shortlist gets a deterministic title and mood derived from cluster
characteristics. These are labels for Stage 2, not prose — Stage 2 ignores them
creatively but uses them as shortlist identifiers.

Precondition: `tracks` must be non-empty. `partition_pool` must never call this
function with an empty list (the ABSOLUTE_MIN guard in Step 1 and Step 4 ensures this).

```python
def _infer_shortlist_mood(tracks: list[Track]) -> tuple[str, str]:
    """Returns (title, mood) for a shortlist. tracks must be non-empty."""
    from collections import Counter

    bpm_vals = [t.bpm for t in tracks]
    bpm_lo, bpm_hi = round(min(bpm_vals)), round(max(bpm_vals))

    # Dominant Camelot key — use full key string (e.g. "8A", "5B"), not just number.
    # Normalise to uppercase before counting (IGNORECASE regex means '8a' == '8A').
    parsed_keys = [
        t.camelot_key.upper() for t in tracks
        if t.camelot_key and _CAMELOT_RE.match(t.camelot_key)
    ]
    # Deterministic dominant_key: sort by (-count, key_str). Counter.most_common() is
    # non-deterministic for equal counts.
    dominant_key = (
        min(parsed_keys, key=lambda k: (-Counter(parsed_keys)[k], k))
        if parsed_keys else "?"
    )

    # Era window
    years = [t.year for t in tracks if t.year is not None and t.year > 0]
    era = f"{min(years)}–{max(years)}" if years else ""

    # Dominant tags (top 2 by frequency across tracks)
    all_tags: list[str] = [tag for t in tracks for tag in t.tags]
    tag_counts = Counter(all_tags)
    # Sort by (-count, tag_str) to break ties deterministically (Counter.most_common
    # uses heap order which is insertion-order for equal counts — non-deterministic).
    top_tags = ", ".join(
        tag_str
        for tag_str, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    ) if tag_counts else ""

    parts = [f"{bpm_lo}–{bpm_hi} BPM", dominant_key]
    if era:
        parts.append(era)
    if top_tags:
        parts.append(top_tags)

    title = " / ".join(parts)
    mood = top_tags or f"{bpm_lo}–{bpm_hi} BPM"
    return title, mood
```

### Step 6 — MixConcept assembly (final output)

After sizing enforcement, `partition_pool` holds a list of track groups (each a
`list[Track]`). This step converts them to the return type:

```python
results: list[MixConcept] = []
for group in final_groups:
    title, mood = _infer_shortlist_mood(group)
    results.append(MixConcept(
        title=title,
        mood=mood,
        track_ids=[t.track_id for t in group],
    ))
return results
```

The pool < MIN_SHORTLIST early-exit path must also construct one MixConcept object
using _infer_shortlist_mood(tracks) before returning — not skip Step 6.
The pool < ABSOLUTE_MIN early-exit returns [] directly; no MixConcept is constructed.

---

## 5. Call site changes in `__main__.py`

There are **three** Stage 1 call sites in `__main__.py`. All three must be replaced.
All three live inside `run()`, not `run_playlist_mode` — playlist mode does not call
`stage1_concepts` and requires no changes for this feature.
The old signature `stage1_concepts(pool, genre, cascade_state, ...)` drops `genre` and
`cascade_state` in the new path.

**Update the existing `from mixlab.clustering import (...)` block in `__main__.py`**
(lines ~15–25) to include `partition_pool` and `ABSOLUTE_MIN`. Do NOT add a separate
import line for the same module — Ruff flags duplicate-module imports (E401/E402).
Merge them into the existing parenthesised block:

```python
from mixlab.clustering import (
    ...,          # existing names unchanged
    partition_pool,
    ABSOLUTE_MIN,
)
```

Confirm `import os` and `import sys` are present in `__main__.py` — the new code
uses `os.environ.get(...)` at all three call sites and `sys.stderr` in the call site B
empty-pool diagnostic. These are almost certainly already imported; if not, add them.

Note: `ABSOLUTE_MIN` must be defined as a **public** (no leading underscore) module-level
constant in `clustering.py`. If existing constants in that file use a leading underscore
(private convention), ensure `ABSOLUTE_MIN` (and `MIN_SHORTLIST`, `MAX_SHORTLIST`, etc.)
are declared without one so callers can import them.

**`run()` takes individual keyword parameters, NOT an `args` namespace.** Add
`stage1_seed: int | None = None` to `run()`'s signature, and pass
`stage1_seed=args.stage1_seed` at the `asyncio.run(run(...))` call in `main()`.

```python
# run() signature — append stage1_seed AFTER the existing final parameter (debug):
async def run(
    genre: str | None,
    export_dir: Path | None,
    mode: TrackMode = "unplayed",
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    intent: str | None = None,
    deep: bool = False,
    debug: bool = False,
    stage1_seed: int | None = None,   # ← new — must come last (keyword-only safe)
) -> None: ...

# asyncio.run(run(...)) call in main() — add the argument:
asyncio.run(
    run(
        args.genre,
        export_dir,
        args.mode,
        ...,
        stage1_seed=args.stage1_seed,   # ← new
    )
)
```

**Add the `--stage1-seed` argument to the argparse setup:**

```python
parser.add_argument(
    "--stage1-seed", type=int, default=None, dest="stage1_seed",
    help="Seed for deterministic Stage 1 tie-breaking. Default: None (stable sort).",
)
```

### Call site A — custom-genre pool (line ~579–584, inside `if is_custom:`)

The current code calls `select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)`
before Stage 1. That window is a **LLM cost-control measure only** — do not call it on
the deterministic path. Pass the full `bpm_sorted_pool` to `partition_pool()`.

**Precondition check (MUST enforce):** before calling `partition_pool()` at all three
call sites, filter out tracks with `bpm <= 0`. Zero/negative-BPM tracks distort
centroids and histogram construction. If the pool-building code does not already
exclude them, add:
```python
bpm_sorted_pool = [t for t in bpm_sorted_pool if t.bpm > 0]  # call site A
sorted_tracks   = [t for t in sorted_tracks   if t.bpm > 0]  # call site B
# Call site C: apply inside the call site C code block, BEFORE the >= ABSOLUTE_MIN guard
# (shown in the call site C code snippet below).
```
The filter at A and B must be applied before the LLM-path window-trim check too,
so both paths operate on the same cleaned pool.

```python
# Before (lines ~578–584; include the sort line at ~578 as it is context for the filter
# placement change — the After block replaces the filter but keeps the sort unchanged)
bpm_sorted_pool = sorted(pool, key=lambda t: t.bpm)   # line ~578 — UNCHANGED; shown for context
stage1_pool = select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)
if len(stage1_pool) < len(pool):
    print(f"  Selected {len(stage1_pool)}-track window from pool for Stage 1 (randomised per run).")
cfg = CUSTOM_GENRES[genre]
custom_genre_sub_genres = cfg["genres"]
all_shortlists.extend(await stage1_concepts(stage1_pool, genre, cascade_state, custom=True))

# After — replaces the entire Before block shown above. Line 578 is kept verbatim
# (the sort), and the filter (next line) is inserted immediately after it. Do NOT
# keep the original line 578 AND add the After block — that would double-sort.
bpm_sorted_pool = sorted(pool, key=lambda t: t.bpm)   # KEPT from Before (line 578 unchanged)
bpm_sorted_pool = [t for t in bpm_sorted_pool if t.bpm > 0]  # INSERTED: precondition filter
cfg = CUSTOM_GENRES[genre]
custom_genre_sub_genres = cfg["genres"]
# IMPORTANT: capture os.environ.get("MIXLAB_STAGE1_LLM") ONCE into a local bool BEFORE
# any branch that assigns stage1_pool. A second os.environ lookup in the bookkeeping
# block (below) may evaluate differently in tests (monkeypatched env), leaving
# stage1_pool unbound if the first if-branch was not entered. Use:
#   use_llm = bool(os.environ.get("MIXLAB_STAGE1_LLM"))
# and replace both os.environ.get() calls below with `use_llm`. Shown as two separate
# checks here for readability only.
if os.environ.get("MIXLAB_STAGE1_LLM"):
    stage1_pool = select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)
    if len(stage1_pool) < len(bpm_sorted_pool):  # compare against filtered pool
        # Note: behavior change from Before — the Before code compared len(stage1_pool)
        # < len(pool) (unfiltered). The After code compares against the bpm>0-filtered
        # pool. A pool with N tracks of bpm<=0 may not print this message even if the
        # window trimmed bpm-filtered tracks (e.g., filter reduces 130→119, window
        # capacity=120 returns all 119, so 119 < 119 is False → no message). Acceptable.
        print(f"  Selected {len(stage1_pool)}-track window from pool for Stage 1 (randomised per run).")
    shortlists = await stage1_concepts(stage1_pool, genre, cascade_state, custom=True)
    stage1_pool_size = len(stage1_pool)   # capture before leaving this branch
else:
    shortlists = partition_pool(bpm_sorted_pool, seed=stage1_seed)
    stage1_pool_size = len(bpm_sorted_pool)  # deterministic: full bpm>0-filtered pool
if not shortlists:
    print(f"Stage 1: pool too small to partition — skipping {genre} (custom).", file=sys.stderr)
all_shortlists.extend(shortlists)
# Bookkeeping — keep these lines immediately after the stage1 block.
# genre_unplayed_track_ids_source and genre_cluster_counts use the UNFILTERED `pool`
# on both paths (intentional: these track "available" unplayed tracks for XML export,
# not the subset passed to Stage 1). Do NOT switch them to bpm_sorted_pool.
genre_unplayed_track_ids_source = pool
genre_outliers: list[Track] = []
outliers: list[Track] = []
genre_cluster_counts = {genre: len(pool)}
# bpm_filtered_counts tracks what was actually passed to Stage 1 (captured above):
bpm_filtered_counts = {genre: stage1_pool_size}
# Note: no early-exit at call site A — it is not inside a loop.
# Call site B (below) uses `continue` because it IS inside a for-loop.
```

### Call site B — standard genre pool (line ~611, inside `for genre_label, cluster_tracks in clusters.items():`)

The existing code passes `sorted_tracks = sort_by_camelot(pools.core)` — the core BPM
sub-pool, Camelot-sorted. Pass this same variable (not `cluster_tracks`) to
`partition_pool()`. The pre-filtering to the core BPM pool is intentional; bridge and
wildcard tracks are excluded here just as they were from the LLM path.

**Existing small-pool guard:** the current code has an early-continue guard before the
Stage 1 call (approximately: `if len(pools.core) < MIN_SHORTLIST_TRACKS: continue`).
`MIN_SHORTLIST_TRACKS` (=8) is imported from `llm.py`. With the deterministic path,
`partition_pool` handles small pools gracefully (returns [] for < ABSOLUTE_MIN, one
shortlist for < MIN_SHORTLIST), so this guard is **no longer needed on the deterministic
path**. Remove it and rely on the `if not shortlists: continue` check shown below.

**Behavior change:** genre clusters with 5–7 core tracks (previously skipped by the
MIN_SHORTLIST_TRACKS=8 guard) now produce one shortlist instead of being skipped.
This is intentional — `partition_pool` returns a usable single shortlist for these
pools. The feature-flag LLM path also drops the MIN_SHORTLIST_TRACKS guard in the
After code below, so it too will now call stage1_concepts for 5–7 track pools. If you
need to preserve the old LLM skip behavior, re-add the guard inside the
`if os.environ.get("MIXLAB_STAGE1_LLM"):` branch.

**Unused import:** after removing the `MIN_SHORTLIST_TRACKS` guard at call site B, the
only remaining use of `MIN_SHORTLIST_TRACKS` in `__main__.py` (line ~604) is gone.
Remove ONLY the `MIN_SHORTLIST_TRACKS` name from the parenthesized `from mixlab.llm
import (...)` block at the top of `__main__.py` (line ~31). Do NOT remove the entire
block or any other name from it — `MAX_STAGE1_POOL_CUSTOM` and `select_stage1_window`
are still imported from that same block and are still used by call site A's LLM path.
Removing them would cause NameError when `MIXLAB_STAGE1_LLM=1`.
Leaving `MIN_SHORTLIST_TRACKS` in causes an `F401` unused-import error that blocks the
build per Ruff policy. Add this removal to the §6 file-change table.

```python
# Before (wider context — show all lines that change or that MUST be kept)
pools = partition_bpm_pools(cluster_tracks)
bpm_filtered_counts[genre_label] = len(pools.core)     # ← REPLACED (moved into if/else in After)
if len(pools.core) < MIN_SHORTLIST_TRACKS:             # ← REMOVE this entire block
    print(
        f"Stage 1: skipping {genre_label} — {len(pools.core)} tracks in core BPM pool "
        f"(minimum {MIN_SHORTLIST_TRACKS})"
    )
    continue
sorted_tracks = sort_by_camelot(pools.core)
all_shortlists.extend(await stage1_concepts(sorted_tracks, genre_label, cascade_state))  # ← REPLACE

# After (replace only the guard+stage1 call block; bpm_filtered_counts now records
# the bpm>0-filtered size on BOTH paths because the filter is applied before the
# if/else — stage1_concepts and partition_pool both receive the same filtered tracks)
pools = partition_bpm_pools(cluster_tracks)
sorted_tracks = sort_by_camelot(pools.core)
sorted_tracks = [t for t in sorted_tracks if t.bpm > 0]  # precondition filter
bpm_filtered_counts[genre_label] = len(sorted_tracks)    # both paths: what Stage 1 receives
# IMPORTANT: capture the env flag ONCE before branching to avoid a second os.environ.get()
# call — in tests, monkeypatching os.environ between the filter line and the branch would
# leave stage1_pool unbound if the first branch is not entered.
if os.environ.get("MIXLAB_STAGE1_LLM"):
    shortlists = await stage1_concepts(sorted_tracks, genre_label, cascade_state)
else:
    shortlists = partition_pool(sorted_tracks, seed=stage1_seed)
if not shortlists:
    print(f"Stage 1: pool too small to partition — skipping {genre_label}.", file=sys.stderr)
    continue  # valid: this is inside the for-genre_label loop
all_shortlists.extend(shortlists)
```

### Call site C — outlier / "Misc" pool (line ~616–617, inside `if len(genre_outliers) >= 4:`)

The existing guard `len(genre_outliers) >= 4` must be updated for the deterministic
path to `>= ABSOLUTE_MIN` (>=5) because `partition_pool` returns `[]` for pools
smaller than ABSOLUTE_MIN=5. A pool of exactly 4 outliers would silently produce no
shortlist on the deterministic path (whereas the LLM path accepted it).

The LLM path must preserve the original `>= 4` threshold to fully restore pre-refactor
behaviour when `MIXLAB_STAGE1_LLM=1`. Using a single unified guard of `>= ABSOLUTE_MIN`
would silently change the LLM path too, violating the §9 "restores LLM path without code
change" criterion.

```python
# Before
if len(genre_outliers) >= 4:
    all_shortlists.extend(await stage1_concepts(genre_outliers, "Misc", cascade_state))

# After
if os.environ.get("MIXLAB_STAGE1_LLM"):
    # LLM path: preserve original behaviour exactly (no bpm filter, original threshold)
    if len(genre_outliers) >= 4:
        shortlists = await stage1_concepts(genre_outliers, "Misc", cascade_state)
        all_shortlists.extend(shortlists)
else:
    # Deterministic path: filter bpm<=0, use ABSOLUTE_MIN threshold.
    # Use a LOCAL variable (bpm_filtered_outliers) — do NOT reassign genre_outliers,
    # which is read later (line ~733) for XML export. Mutating it here would silently
    # drop bpm<=0 tracks from the export even though the LLM path does not.
    bpm_filtered_outliers = [t for t in genre_outliers if t.bpm > 0]
    if len(bpm_filtered_outliers) >= ABSOLUTE_MIN:
        shortlists = partition_pool(bpm_filtered_outliers, seed=stage1_seed)
        all_shortlists.extend(shortlists)
# Note: no early-exit at call site C — it is not inside a loop.
```

---

## 6. Files to change

| File | Change |
|------|--------|
| `src/mixlab/clustering.py` | Add `partition_pool()`, `_find_bpm_peaks()`, `_camelot_components()`, `_era_split()`, `_resize_shortlists()`, `_median_bpm()`, `_infer_shortlist_mood()`; add public module-level constants `MIN_SHORTLIST`, `MAX_SHORTLIST`, `ABSOLUTE_MIN`, `MIN_POOL_COUNT`, `MAX_POOL_COUNT`, `MIN_CAMELOT_COMPONENT` (no leading underscore — `ABSOLUTE_MIN` is imported by `__main__.py`). **Note:** `MIN_SHORTLIST = 15` is distinct from `MIN_SHORTLIST_TRACKS = 8` in `llm.py` — the latter is the LLM Stage 1 output filter threshold, unrelated to partition_pool sizing. Also add `from collections import Counter` and `import statistics` at module scope if not already present (used by centrality and _infer_shortlist_mood). **Helper signatures:** `_era_split(shortlist: list[Track], *, per_side_min: int = 8) -> tuple[list[Track], list[Track]] \| None` — returns `(era_old, era_new)` when a split is found, or `None` when any skip condition fires (Step 3.2–3.6). `_infer_shortlist_mood(tracks: list[Track]) -> tuple[str, str]` — returns `(title, mood)`. `_median_bpm(shortlist: list[Track]) -> float`. `_camelot_components(tracks: list[Track]) -> list[list[Track]]` — returns list of components (each a list of tracks). `_resize_shortlists(shortlists: list[list[Track]]) -> list[list[Track]]` — applies Steps 4 and pool-level adjustments in-place and returns the final list. `_find_bpm_peaks(tracks: list[Track]) -> list[list[Track]] \| None` — returns list of clusters (one per peak + outliers attached), or `None` when no peaks found (triggers uniform-spread fallback in caller). |
| `src/mixlab/__main__.py` | Replace all three Stage 1 LLM call sites; add `--stage1-seed` flag; add feature flag env var; add `stage1_seed: int \| None = None` to `run()` signature; remove unused `MIN_SHORTLIST_TRACKS` import (becomes F401 after call-site B guard removal) |
| `src/mixlab/llm.py` | Mark `_STAGE1_SYSTEM*` prompts and `stage1_concepts()` as deprecated (remove after 30-day soak) |
| `.env.example` | Add `MIXLAB_STAGE1_LLM=` (empty = deterministic path; set to `1` to restore LLM path during soak period) |
| `tests/test_clustering.py` | New tests for all `partition_pool` helpers and end-to-end partitioning |
| `tests/test_llm.py` | Remove or migrate Stage 1 LLM tests |
| `tests/test_main.py` | Update all three Stage 1 call-site tests |

---

## 7. Test strategy

### Unit tests (`tests/test_clustering.py`)

```
# Step 1 — BPM peak detection
test_find_bpm_peaks_single_tight_cluster_returns_one_peak
test_find_bpm_peaks_two_well_separated_groups_returns_two_peaks
test_find_bpm_peaks_merges_peaks_closer_than_8_bpm
test_find_bpm_peaks_merge_picks_globally_closest_pair_first
test_find_bpm_peaks_merge_tie_keeps_lower_bpm_peak
test_find_bpm_peaks_plateau_interior_uses_leftmost_bucket
test_find_bpm_peaks_plateau_at_last_bucket_not_returned_as_peak
test_find_bpm_peaks_plateau_at_first_bucket_not_returned_as_peak
test_find_bpm_peaks_edge_buckets_never_returned_as_peaks
test_find_bpm_peaks_boundary_bpm_assigned_to_new_bucket
test_find_bpm_peaks_track_exactly_7_bpm_from_centre_is_within_window
test_find_bpm_peaks_track_beyond_7_bpm_becomes_outlier
test_find_bpm_peaks_equidistant_track_assigned_to_lower_index_peak
test_find_bpm_peaks_outliers_have_no_candidate_peak_within_7_bpm
test_partition_pool_no_peaks_uniform_spread_returns_equal_bpm_ordered_groups
test_partition_pool_no_peaks_small_pool_uniform_spread_equal_sized_groups
test_partition_pool_single_bucket_histogram_triggers_fallback
test_partition_pool_two_bucket_histogram_triggers_fallback
test_find_bpm_peaks_three_bucket_all_equal_histogram_triggers_fallback
# Note: single- and two-bucket fallbacks are tested at partition_pool level (the
# _find_bpm_peaks helper is not directly involved — the < 3 bucket check short-
# circuits before peak detection runs). Three-bucket tests that do reach _find_bpm_peaks
# can keep the helper prefix.
# Note: the following test_partition_pool_* tests cover Step-1 fallback behaviour at
# the partition_pool level (not the _find_bpm_peaks helper, which returns peak objects):
test_partition_pool_below_absolute_min_returns_empty
test_partition_pool_small_but_above_absolute_min_returns_single_shortlist
test_partition_pool_n_groups_one_fallback_no_peaks_returns_single_shortlist_up_to_29_tracks
test_partition_pool_n_groups_one_fallback_bypasses_step4_peaks_path_same_size_does_not
# Fixture note for both n_groups=1 fallback tests above: tracks must have span >= 4.0
# (to bypass the flat-BPM guard — which fires ONLY when max_bpm - min_bpm < 4.0) AND
# a histogram distribution that produces NO detectable peaks after smoothing. A pool of
# tracks all sharing exactly the same BPM has span=0 and triggers the flat-BPM guard,
# not the n_groups=1 fallback. Suitable fixture: 15–29 tracks with varied BPMs spanning
# 4+ BPM but arranged so the smoothed histogram is monotone or produces only edge peaks
# (e.g. all BPMs increasing linearly across the range).

# Step 2 — Camelot components
test_camelot_components_connected_keys_form_single_component
test_camelot_components_unrelated_keys_form_separate_components
test_camelot_components_all_unknown_keys_kept_as_single_component
test_camelot_components_some_unknown_keys_become_singletons_then_merged
test_camelot_components_three_components_small_one_merged_into_largest
test_camelot_components_merge_recomputes_largest_after_each_step
test_camelot_components_merge_tie_uses_min_track_id
test_camelot_components_merge_tie_min_track_id_is_lexicographic_not_numeric

# Step 3 — Era split
test_era_split_applies_when_gap_large_and_both_sides_sufficient
test_era_split_applies_at_exact_8_year_gap_boundary
test_era_split_skipped_at_7_year_gap_boundary
test_era_split_skipped_when_gap_below_threshold
test_era_split_skipped_when_one_side_too_small
test_era_split_skipped_when_too_few_known_years
test_era_split_unknown_year_tracks_assigned_to_era_old_unconditionally
test_era_split_year_zero_tracks_treated_as_unknown_year
test_era_split_unknown_year_tracks_include_zero_and_none_years
test_era_split_gap_idx_points_to_last_track_before_gap
test_era_split_counts_only_positive_year_tracks_for_side_guard
test_era_split_attempt2_fails_when_era_old_known_below_per_side_min_despite_unknowns_inflating_total
# ^ This test is specifically for the Attempt 2 trap: a shortlist where len(era_old)>=15
# (because unknown-year tracks are included) but era_old_known<per_side_min=15 — condition
# (a) fails (Step 3 produces no split) even though condition (b) would pass. An
# implementation that checks len(era_old)>=15 instead of era_old_known>=per_side_min would
# pass the other tests but fail this one.

# Step 4 — Sizing enforcement
test_resize_shortlists_merges_undersized_into_nearest_bpm_neighbour
test_resize_shortlists_remerges_when_merged_result_still_undersized
test_resize_shortlists_keeps_when_all_merges_exhausted_if_above_absolute_min
test_resize_shortlists_drops_when_all_merges_exhausted_if_below_absolute_min
test_resize_shortlists_keeps_both_when_merge_would_exceed_max
test_resize_shortlists_drops_undersized_below_absolute_min_after_failed_merge
test_resize_shortlists_lone_undersized_above_absolute_min_kept
test_resize_shortlists_lone_undersized_below_absolute_min_kept_as_is
test_resize_shortlists_splits_oversized_by_camelot_component
test_resize_shortlists_centrality_ranks_ascending_keeps_central_tracks
test_resize_shortlists_centrality_normalises_bpm_and_camelot_independently
test_resize_shortlists_centrality_bpm_range_near_zero_uses_fallback_1
test_resize_shortlists_centrality_dominant_key_none_all_camelot_norm_1
test_resize_shortlists_centrality_unparseable_key_capped_at_1_not_999_over_7
test_resize_shortlists_centrality_tie_broken_by_track_id
test_resize_shortlists_single_oversized_no_overflow_target_trims_in_place
test_resize_shortlists_pool_level_merge_loops_until_max_pool_count
test_resize_shortlists_pool_level_merge_inserts_at_lower_index
test_resize_shortlists_pool_level_merge_may_exceed_max_shortlist
test_resize_shortlists_dropped_shortlists_not_resurrected_by_min_pool_count

# Step 5 — Mood/title inference
test_infer_shortlist_mood_uses_full_camelot_key_string_not_number
test_infer_shortlist_mood_mode_b_key_not_coerced_to_a
test_infer_shortlist_mood_filters_unparseable_camelot_keys
test_infer_shortlist_mood_includes_era_when_years_present
test_infer_shortlist_mood_no_tags_falls_back_to_bpm_range

# End-to-end partition_pool
test_partition_pool_returns_three_to_five_shortlists_on_typical_input
test_partition_pool_typical_input_shortlists_within_15_to_25_tracks
test_partition_pool_same_seed_same_output_reproducibility
test_partition_pool_tiny_pool_returns_single_shortlist
test_partition_pool_below_absolute_min_returns_empty
test_partition_pool_all_same_bpm_15to25_tracks_returns_camelot_subgroups_or_single_intact
test_partition_pool_all_same_bpm_26plus_tracks_trimmed_to_max_shortlist_by_step4
test_partition_pool_outliers_attached_to_nearest_cluster
test_partition_pool_no_peaks_uses_equal_bpm_split_before_step2
test_partition_pool_custom_genre_pool_respects_sub_genre_coherence

# Feature flag and call-site integration (tests/test_main.py)
test_main_stage1_llm_flag_routes_to_stage1_concepts_at_call_site_a
test_main_stage1_llm_flag_routes_to_stage1_concepts_at_call_site_b
test_main_stage1_llm_flag_routes_to_stage1_concepts_at_call_site_c
test_main_stage1_deterministic_path_calls_partition_pool_at_all_three_sites
test_main_stage1_seed_flag_parsed_and_passed_to_run
```

### Regression (manual, on real library snapshot)

Before shipping:
1. Run Stage 1 LLM path on a real genre pool, capture shortlists.
2. Run `partition_pool()` on the same pool, capture shortlists.
3. Compare: track-count per shortlist, BPM spread, Camelot coherence.
4. Run Stage 2 on both sets of shortlists and compare concept quality.
5. Document findings in `docs/notes/` before merging.

---

## 8. Feature flag and rollout

Add env var `MIXLAB_STAGE1_LLM=1` to restore the LLM path during the soak period.
The wrapper pattern is shown per call site in §5.

Document `MIXLAB_STAGE1_LLM` in `.env.example`.

**Removal timeline:**
- Ship deterministic Stage 1 behind flag OFF by default.
- After 30 days of production use with no regressions, remove the flag and
  delete `_STAGE1_SYSTEM*` prompts + `stage1_concepts()` for LLM Stage 1.
- The cascade infrastructure stays (Stage 0 uses it).

---

## 9. Acceptance criteria

- [ ] `partition_pool()` produces 3–5 shortlists of 15–25 tracks on a real library snapshot
      (except documented edge cases: pool < ABSOLUTE_MIN returns []; pool < MIN_SHORTLIST
      returns exactly one shortlist; unimodal or thin pool may return 1–2 shortlists; pool-
      level merge may exceed 25; Phase 2 or last-iteration undersized merge may exceed 25;
      Phase 1 overflow enters Phase 2 (never skips it), so a below-15 result only occurs
      when Phase 2 ALSO exhausts or overflows all available partners (correct behavior, not
      a bug); n_groups=1 fallback may return one shortlist up to 29; Attempt 3 remainder-
      attach may push the target shortlist above 25)
- [ ] Same pool + same seed → identical output across two runs
- [ ] Stage 2 concept quality on manual review of 5 snapshots: equal or better than LLM Stage 1
- [ ] No regression in `--mode all`, `--mode unplayed`, `--mode played`
- [ ] Custom-genre pools handled correctly (sub-genre coherence respected)
- [ ] End-to-end latency drops by ≥ 5 seconds per genre call
- [ ] `_STAGE1_SYSTEM*` prompts and `stage1_concepts()` marked deprecated in `llm.py`
- [ ] `MIXLAB_STAGE1_LLM=1` restores LLM path for all three call sites with minimal
      behavioural differences: call site A prints the window-trim message under
      slightly different conditions (bpm-filtered pool vs. original); call site C
      now applies the bpm>0 filter ONLY on the deterministic branch (LLM path is
      unfiltered, matching original). These are intentional and acceptable trade-offs.
- [ ] All new tests pass; Stage 1 LLM tests removed or migrated
- [ ] Outlier tracks always attached to nearest cluster (no unresolved "merge candidates"),
      except when the entire genre pool yields only ONE shortlist after sizing — in that
      case Attempt 3 remainder tracks are discarded with a stderr diagnostic (unavoidable;
      no other shortlist exists to attach to)
- [ ] `select_stage1_window` not called on the deterministic path at call site A

---

## 10. Open questions

1. **Seed exposure**: should `--stage1-seed` be a CLI flag, or always derived from
   current date (e.g., `seed = int(datetime.date.today().strftime("%Y%m%d"))`)?
   Date-derived seed means "one partition per day" — deterministic within a session
   but varying across days without user action. Preferred: explicit flag, default None
   (stable sort-based tie-breaking).

2. **Custom-genre pools**: the current Stage 1 LLM is aware of sub-genre labels.
   `partition_pool()` will use BPM + Camelot only for clustering. If sub-genre
   coherence matters more than BPM proximity for a given custom genre, a future
   extension can add a `sub_genre_weight` parameter. Out of scope for v1.
