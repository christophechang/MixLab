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
    """Partition a genre-scoped track pool into 3–5 shortlists of 15–25 tracks.

    Deterministic replacement for the Stage 1 LLM call. Same pool + same seed =
    same output across runs. With seed=None, tie-breaking is stable (sorted track_ids)
    but pools with equal-density BPM clusters may vary ordering.

    Preconditions:
        - tracks must be non-empty.
        - Tracks with bpm <= 0 should be excluded by the caller — they are treated as
          the lowest-BPM tracks and will distort cluster centroids.

    Note: this function accepts only tracks and seed. The genre label and
    cascade_state passed to the old stage1_concepts() are not needed — genre
    is already implicit in the pool, and cascade_state is LLM infrastructure.

    Note on seed: in v1 all tie-breaking is deterministic via sorted track_id, so
    every seed value (including None) produces identical output. The parameter is
    reserved for a future BPM-jitter extension. Callers may always pass seed=None.

    Args:
        tracks: Genre-scoped, mode-filtered pool. Must be non-empty.
        seed: Reserved for future use. Has no effect in v1.

    Returns:
        List of MixConcept (title + mood + track_ids). Returns [] if pool is
        too small to form any valid shortlist (fewer than ABSOLUTE_MIN=5 tracks).
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
MIN_SHORTLIST  = 15
MAX_SHORTLIST  = 25
ABSOLUTE_MIN   = 5   # shortlists below this are dropped rather than passed to Stage 2
MIN_POOL_COUNT = 3
MAX_POOL_COUNT = 5
```

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
  Skip Steps 2–4. (No clustering needed — the pool is too small to split.)
- If all tracks have |bpm_i - bpm_j| < 4 for all pairs: treat all tracks as a single
  BPM cluster and proceed directly to Step 2 with that one cluster.
```

**Histogram construction:**

```
1. Sort tracks ascending by BPM.
2. Bin tracks into 3-BPM-wide histogram buckets using left-inclusive, right-exclusive
   intervals [lo, lo+3). The first bucket starts at floor(min_bpm / 3) * 3.
   Example: min_bpm=122.4 → first bucket is [120, 123).
   A track with bpm at an exact boundary falls in the NEW bucket:
   bpm=123.0 falls in [123, 126), not [120, 123).
3. Smooth the histogram with a 3-bucket moving average. Let N = (number of buckets - 1).
   - If number of buckets < 3: skip peak detection, proceed directly to the
     uniform-spread fallback (no interior buckets can be peaks).
   - Interior bucket i (0 < i < N): smoothed[i] = (raw[i-1] + raw[i] + raw[i+1]) / 3
   - Left edge (i=0): smoothed[0] = (raw[0] + raw[1]) / 2
   - Right edge (i=N): smoothed[N] = (raw[N-1] + raw[N]) / 2
   Divide by the count of available buckets in the window (2 or 3), not always 3.
```

**Peak detection:**

```
4. Find local maxima (peaks) in the smoothed histogram:
   - A bucket at index i is a peak iff:
       smoothed[i] > smoothed[i-1]  AND  smoothed[i] > smoothed[i+1]
   - Edge buckets (i=0 and i=N) are NEVER peaks.
   - Plateaus (consecutive equal smoothed counts) — only the leftmost bucket of
     the run is the peak candidate:
       * Plateau starts at i=0 (left-edge plateau): treat as non-peak.
       * Plateau ends at i=N (right-edge plateau): treat as non-peak.
       * Interior plateau: compare leftmost bucket against the bucket immediately
         to its left and the bucket immediately after the run ends.
         Peak iff both comparisons are strictly greater.
5. Complete ALL peak merges before beginning track assignment (Step 6).
   Merge peaks that are closer than 8 BPM (strict: distance < 8.0) — on each iteration:
   - Find the pair with the smallest |centre_a - centre_b|.
   - Merge: keep the peak with the higher raw track count in its bucket ("heavier").
   - Tie (equal raw count): keep the lower-BPM peak.
   - Distance tie (two pairs are equidistant): choose the pair with the lowest
     combined peak BPM (sum of the two centres).
   - Repeat until no two peaks have distance < 8.0.

Note on threshold equivalence: 'distance < 8 BPM' for merging and '|bpm - centre| ≤ 7.0'
for assignment in Step 6 are numerically equivalent (7 < 8 and 7 ≤ 7). Both are
intentional; the phrasing differs only to match natural language ("closer than 8" vs
"within ±7"). Implement both as their written form.
```

**Peak centre** = midpoint of the 3-BPM bucket: `bucket_start + 1.5`.
After a merge, the surviving peak retains its original centre (not recomputed).
Tracks that fell in the eliminated peak's ±7 window but outside the surviving peak's
±7 window become outliers and are attached unconditionally in Step 7.

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
   Ties (equidistant from two clusters): attach to the cluster with the lower index.
```

Target: 2–6 raw BPM clusters before sizing adjustments.

**Uniform-spread fallback** (triggered when: no peaks found, or histogram has < 3 buckets):

```
n_groups = min(MAX_POOL_COUNT, len(tracks) // MIN_SHORTLIST)
If n_groups < 2:
  Skip Steps 2–4. Call _infer_shortlist_mood on all tracks, assemble one MixConcept,
  and return a single-element list. (Do NOT interpret "Steps 5–6" as the intra-Step-1
  numbered sub-items 5 and 6 — those are peak-merge and track-assignment, which
  require peaks to exist.)
  Note: pools of 15–29 tracks yield n_groups = 1. The returned shortlist may have
  up to 29 tracks — 4 over MAX_SHORTLIST. This is accepted for flat-BPM pools where
  no meaningful split exists.
Otherwise: split into n_groups equal-sized groups by sorted BPM rank.
  Group sizes: if len(tracks) % n_groups != 0, the first (len(tracks) % n_groups)
  groups each receive one extra track. E.g. 31 tracks / 2 groups = [16, 15].
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
2. Find connected components via BFS.
3. If ALL tracks in the BPM cluster have no/unknown Camelot key (empty string or
   doesn't match _CAMELOT_RE): treat the entire cluster as a single component — keep
   as-is. This guard prevents the BFS from producing N singletons.
   "All unknown" means every track in the cluster fails _CAMELOT_RE.match().
4. For each component with < 8 tracks: merge it into the current-largest component
   in the same BPM cluster. The source component being merged is NOT a candidate
   for "current-largest" — only other components are.
   - Apply merges in size-ascending order (smallest component first).
   - After each individual merge, recompute component sizes before the next merge.
   - Tie in size (two components to merge both have equal size): process the one
     whose minimum track_id is lexicographically smaller first.
   - Tie in "largest" target (two components share the max size after a recompute):
     target the one whose minimum track_id is lexicographically smaller.
5. After small-component merges, if 2+ components remain and each has ≥ 8 tracks:
   treat each surviving component as a candidate sub-partition.
```

### Step 3 — Era split (optional)

Applied independently to each candidate shortlist *after* Camelot sub-clustering.

```
1. Collect known-year tracks: t.year is not None and t.year > 0. Call this set KY.
   "Unknown-year tracks" = all tracks NOT in KY (i.e. year is None OR year <= 0).
2. If len(KY) / len(shortlist) < 0.60: skip era split for this cluster.
3. Sort KY ascending by year. Call the sorted list ky (length K).
   If K < 2: skip era split (no consecutive pair exists to compute a gap).
4. Find gap_idx: the index i (0 ≤ i ≤ K-2) that maximises ky[i+1].year - ky[i].year.
   gap_idx is the index of the last track BEFORE the gap.
   Tie (two equal gap sizes): choose the smaller i.
   gap_start = ky[gap_idx].year
   gap_size  = ky[gap_idx + 1].year - gap_start
5. Count era_old_known = number of tracks in KY with year <= gap_start.
   Count era_new_known = number of tracks in KY with year >  gap_start.
   (These counts use only KY — unknown-year tracks are not counted here.)
6. If gap_size >= 8 AND era_old_known >= 8 AND era_new_known >= 8:
   - era_old: tracks with year > 0 and year <= gap_start
   - era_new: tracks with year > 0 and year >  gap_start
   - Unknown-year tracks (year=None OR year<=0): assign by comparing era centroids.
     centroid_old = mean(t.year for t in era_old)   # always >= 8 known-year tracks
     centroid_new = mean(t.year for t in era_new)   # always >= 8 known-year tracks
     For each unknown-year track: assign to era_old if centroid_old <= centroid_new,
     else assign to era_new. (Tie: assign to era_old.)
     IMPORTANT: the centroid comparison does NOT use the unknown-year track's own year
     value — all unknown-year tracks in a shortlist are assigned to the same side,
     determined solely by comparing centroid_old vs centroid_new.
7. Otherwise: no era split.
```

### Step 4 — Sizing enforcement

After BPM + Camelot + era passes, the candidate shortlists are resized to meet
the 15–25 track target. All comparisons use current shortlist contents.

**Sizing pass order:** make one oversized pass (descending size order), then one
undersized pass (ascending size order). Repeat both passes until no shortlist needs
resizing, up to a maximum of 3 full iterations. In practice, two passes suffice.

**Under-sizing (len < MIN_SHORTLIST):**

```
Find the other shortlist with the smallest abs(_median_bpm(this) - _median_bpm(other)).
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
    Keep both as-is. If len(undersized) < ABSOLUTE_MIN, drop it. STOP — do not
    enter Phase 2. (The closest partner overflows; no further attempts are made.)
  - Otherwise: merge with the closest partner.

Phase 2 — retry loop (only entered if Phase 1 merged but result still < MIN_SHORTLIST):
  Use the post-Phase-1 merged shortlist's _median_bpm() as the reference for all
  Phase 2 distance comparisons.
  For each remaining partner in ascending _median_bpm() distance order:
    - If this partner would cause overflow (merged len > MAX_SHORTLIST): skip it.
    - Otherwise: merge. If result >= MIN_SHORTLIST: STOP (success).
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
  On success: replace the oversized shortlist with the two components in ascending
  _median_bpm() order (lower-BPM component first, at the original index).
Attempt 2: re-apply the full Step 3 era-split logic (including the 60% coverage
  gate) to the oversized shortlist. Attempt 2 succeeds only when the era conditions
  pass AND len(era_old) >= MIN_SHORTLIST AND len(era_new) >= MIN_SHORTLIST. If either
  half would be below MIN_SHORTLIST, treat Attempt 2 as failed and proceed to Attempt 3.
  On success: replace the oversized shortlist with era_old first (at original index),
  era_new second (at original index + 1).
Attempt 3: rank by centrality ascending, keep the first MAX_SHORTLIST, attach
  the remainder to the nearest other shortlist by _median_bpm() distance measured
  from the TRIMMED shortlist (not from the remainder group).
  Tie in distance: attach to the shortlist whose minimum track_id is
  lexicographically smaller.
  If no other shortlist exists, discard the remainder.

  Centrality computation:
    parsed_keys = [t.camelot_key for t in shortlist
                   if t.camelot_key and _CAMELOT_RE.match(t.camelot_key)]
    dominant_key = Counter(parsed_keys).most_common(1)[0][0] if parsed_keys else None
    bpm_vals = [t.bpm for t in shortlist]
    bpm_range = max(bpm_vals) - min(bpm_vals)
    if bpm_range < 1e-6:
        bpm_range = 1.0   # all tracks have effectively identical BPM
    For each track t:
      bpm_norm = abs(t.bpm - _median_bpm(shortlist)) / bpm_range
      if dominant_key is None:
        camelot_norm = 1.0   # no parseable keys in cluster — max peripheral score
      else:
        d = camelot_distance(t.camelot_key, dominant_key)
        # camelot_distance returns 999 for any unparseable key (including when
        # dominant_key is parseable but t.camelot_key is not). Cap at 1.0.
        camelot_norm = 1.0 if d >= 999 else d / 7.0
        # 7.0 is the maximum finite camelot_distance (e.g. '1A' to '7B' = 7).
        # The 999 guard must be kept — do not remove it as "dead code".
      centrality = bpm_norm + camelot_norm
      # centrality ∈ [0, 2.0]. Do not cap or normalise — sort uses raw value.
    Ties in centrality score: break by ascending track_id (lexicographic, stable).
```

**Pool-level adjustments (after all per-shortlist passes):**

```
If total count < MIN_POOL_COUNT: keep all remaining shortlists (even if undersized).
If total count > MAX_POOL_COUNT:
  Repeat:
    Find the pair of shortlists with the smallest abs(_median_bpm(A) - _median_bpm(B)).
    Tie-break chain (apply in order until resolved):
      1. Smallest min(min_track_id(A), min_track_id(B)) — lexicographically.
      2. Smallest sum of the pair's 0-based indices in the current shortlist list.
      3. Lowest index of the first shortlist in the pair.
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
    # Filter to tracks with a parseable key to preserve the mode letter (A vs B).
    parsed_keys = [
        t.camelot_key for t in tracks
        if t.camelot_key and _CAMELOT_RE.match(t.camelot_key)
    ]
    dominant_key = Counter(parsed_keys).most_common(1)[0][0] if parsed_keys else "?"

    # Era window
    years = [t.year for t in tracks if t.year and t.year > 0]
    era = f"{min(years)}–{max(years)}" if years else ""

    # Dominant tags (top 2 by frequency across tracks)
    all_tags: list[str] = [tag for t in tracks for tag in t.tags]
    tag_counts = Counter(all_tags)
    # Use tag_str (not t) to avoid shadowing the Track loop variable above.
    top_tags = ", ".join(tag_str for tag_str, _ in tag_counts.most_common(2)) if tag_counts else ""

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

The two early-exit paths (pool < ABSOLUTE_MIN, pool < MIN_SHORTLIST) must also
construct MixConcept objects in the same way before returning.

---

## 5. Call site changes in `__main__.py`

There are **three** Stage 1 call sites in `__main__.py`. All three must be replaced.
The old signature `stage1_concepts(pool, genre, cascade_state, ...)` drops `genre` and
`cascade_state` in the new path.

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

```python
# Before
stage1_pool = select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)
if len(stage1_pool) < len(pool):
    print(f"  Selected {len(stage1_pool)}-track window from pool for Stage 1 (randomised per run).")
cfg = CUSTOM_GENRES[genre]
custom_genre_sub_genres = cfg["genres"]
all_shortlists.extend(await stage1_concepts(stage1_pool, genre, cascade_state, custom=True))

# After
cfg = CUSTOM_GENRES[genre]
custom_genre_sub_genres = cfg["genres"]
if os.environ.get("MIXLAB_STAGE1_LLM"):
    stage1_pool = select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)
    if len(stage1_pool) < len(bpm_sorted_pool):
        print(f"  Selected {len(stage1_pool)}-track window from pool for Stage 1 (randomised per run).")
    shortlists = await stage1_concepts(stage1_pool, genre, cascade_state, custom=True)
else:
    shortlists = partition_pool(bpm_sorted_pool, seed=args.stage1_seed)
all_shortlists.extend(shortlists)
# Note: no early-exit at call site A — it is not inside a loop.
# Call site B (below) uses `continue` because it IS inside a for-loop.
```

### Call site B — standard genre pool (line ~611, inside `for genre_label, cluster_tracks in clusters.items():`)

The existing code passes `sorted_tracks = sort_by_camelot(pools.core)` — the core BPM
sub-pool, Camelot-sorted. Pass this same variable (not `cluster_tracks`) to
`partition_pool()`. The pre-filtering to the core BPM pool is intentional; bridge and
wildcard tracks are excluded here just as they were from the LLM path.

```python
# Before
all_shortlists.extend(await stage1_concepts(sorted_tracks, genre_label, cascade_state))

# After
if os.environ.get("MIXLAB_STAGE1_LLM"):
    shortlists = await stage1_concepts(sorted_tracks, genre_label, cascade_state)
else:
    shortlists = partition_pool(sorted_tracks, seed=args.stage1_seed)
if not shortlists:
    print(f"Stage 1: pool too small to partition — skipping {genre_label}.", file=sys.stderr)
    continue  # valid: this is inside the for-genre_label loop
all_shortlists.extend(shortlists)
```

### Call site C — outlier / "Misc" pool (line ~616–617, inside `if len(genre_outliers) >= 4:`)

```python
# Before
if len(genre_outliers) >= 4:
    all_shortlists.extend(await stage1_concepts(genre_outliers, "Misc", cascade_state))

# After
if len(genre_outliers) >= 4:
    if os.environ.get("MIXLAB_STAGE1_LLM"):
        shortlists = await stage1_concepts(genre_outliers, "Misc", cascade_state)
    else:
        shortlists = partition_pool(genre_outliers, seed=args.stage1_seed)
    all_shortlists.extend(shortlists)
# Note: no early-exit at call site C — it is not inside a loop.
```

---

## 6. Files to change

| File | Change |
|------|--------|
| `src/mixlab/clustering.py` | Add `partition_pool()`, `_find_bpm_peaks()`, `_camelot_components()`, `_era_split()`, `_resize_shortlists()`, `_median_bpm()`, `_infer_shortlist_mood()` |
| `src/mixlab/__main__.py` | Replace all three Stage 1 LLM call sites; add `--stage1-seed` flag; add feature flag env var |
| `src/mixlab/llm.py` | Mark `_STAGE1_SYSTEM*` prompts and `stage1_concepts()` as deprecated (remove after 30-day soak) |
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
test_find_bpm_peaks_outliers_attached_to_nearest_cluster
test_find_bpm_peaks_uniform_spread_falls_back_to_quantile_splits
test_find_bpm_peaks_quantile_fallback_small_pool_returns_single_group
test_find_bpm_peaks_quantile_fallback_single_shortlist_has_populated_title_mood
test_find_bpm_peaks_single_bucket_histogram_uses_fallback_not_peak_detection
test_find_bpm_peaks_two_bucket_histogram_uses_fallback
test_find_bpm_peaks_pool_below_absolute_min_returns_empty
test_find_bpm_peaks_pool_small_but_above_absolute_min_returns_single_shortlist

# Step 2 — Camelot components
test_camelot_components_connected_keys_form_single_component
test_camelot_components_unrelated_keys_form_separate_components
test_camelot_components_all_unknown_keys_kept_as_single_component
test_camelot_components_some_unknown_keys_become_singletons_then_merged
test_camelot_components_three_components_small_one_merged_into_largest
test_camelot_components_merge_recomputes_largest_after_each_step
test_camelot_components_merge_tie_uses_min_track_id

# Step 3 — Era split
test_era_split_applies_when_gap_large_and_both_sides_sufficient
test_era_split_applies_at_exact_8_year_gap_boundary
test_era_split_skipped_at_7_year_gap_boundary
test_era_split_skipped_when_gap_below_threshold
test_era_split_skipped_when_one_side_too_small
test_era_split_skipped_when_too_few_known_years
test_era_split_unknown_year_tracks_routed_by_centroid
test_era_split_year_zero_tracks_treated_as_unknown_year
test_era_split_centroid_tie_assigns_to_era_old
test_era_split_gap_idx_points_to_last_track_before_gap
test_era_split_counts_only_positive_year_tracks_for_side_guard

# Step 4 — Sizing enforcement
test_resize_shortlists_merges_undersized_into_nearest_bpm_neighbour
test_resize_shortlists_remerges_when_merged_result_still_undersized
test_resize_shortlists_keeps_when_all_merges_exhausted_if_above_absolute_min
test_resize_shortlists_drops_when_all_merges_exhausted_if_below_absolute_min
test_resize_shortlists_keeps_both_when_merge_would_exceed_max
test_resize_shortlists_drops_undersized_below_absolute_min_after_failed_merge
test_resize_shortlists_lone_undersized_above_absolute_min_kept
test_resize_shortlists_lone_undersized_below_absolute_min_returns_empty
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
test_partition_pool_each_shortlist_has_15_to_25_tracks
test_partition_pool_same_seed_same_output_reproducibility
test_partition_pool_tiny_pool_returns_single_shortlist
test_partition_pool_below_absolute_min_returns_empty
test_partition_pool_all_same_bpm_returns_camelot_subgroups_or_single
test_partition_pool_outliers_attached_to_nearest_cluster
test_partition_pool_uniform_spread_falls_back_to_quantile_splits
test_partition_pool_custom_genre_pool_respects_sub_genre_coherence
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
- [ ] Same pool + same seed → identical output across two runs
- [ ] Stage 2 concept quality on manual review of 5 snapshots: equal or better than LLM Stage 1
- [ ] No regression in `--mode all`, `--mode unplayed`, `--mode played`
- [ ] Custom-genre pools handled correctly (sub-genre coherence respected)
- [ ] End-to-end latency drops by ≥ 5 seconds per genre call
- [ ] `_STAGE1_SYSTEM*` prompts and `stage1_concepts()` marked deprecated in `llm.py`
- [ ] `MIXLAB_STAGE1_LLM=1` restores LLM path without code change (all three call sites)
- [ ] All new tests pass; Stage 1 LLM tests removed or migrated
- [ ] Outlier tracks always attached to nearest cluster (no unresolved "merge candidates")
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
