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
    same output across runs. With seed=None, tie-breaking is stable (sorted IDs)
    but pools with equal-density BPM clusters may vary ordering.

    Note: this function accepts only tracks and seed. The genre label and
    cascade_state passed to the old stage1_concepts() are not needed — genre
    is already implicit in the pool, and cascade_state is LLM infrastructure.

    Args:
        tracks: Genre-scoped, mode-filtered pool. Must contain ≥ 3 tracks.
        seed: Optional seed for tie-breaking in merge/split decisions.

    Returns:
        List of MixConcept (title + mood + track_ids). Returns [] if pool is
        too small to form any valid shortlist.
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

### Step 1 — BPM clustering (primary partition)

```
1. Sort tracks by BPM.
2. Bin tracks into 3-BPM-wide histogram buckets.
3. Smooth the histogram with a 3-bucket moving average (window = [prev, curr, next];
   edge buckets use only the available neighbours — no wrap-around).
4. Find local maxima (peaks) in the smoothed histogram:
   - A bucket at index i is a peak iff its smoothed count strictly exceeds both
     immediate neighbours: smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1].
   - Edge buckets (i=0, i=last) are never peaks.
   - Plateaus (consecutive buckets with equal smoothed count): treat the leftmost
     bucket of the run as the peak candidate; compare its value against the bucket
     to its left and the bucket immediately after the run ends.
5. Merge any two peaks closer than 8 BPM (keep the heavier one; ties by lower BPM).
6. Assign each track to its nearest qualifying peak:
   - Only consider peaks whose centre is within ±7 BPM of the track's BPM.
   - If multiple peaks qualify, assign to the nearest (smallest |track.bpm - peak.centre|).
   - Ties (equidistant from two peaks): assign to the peak with the lower index in the
     sorted ascending-BPM peak list.
   - Tracks with no qualifying peak become "outliers".
7. Attach each outlier unconditionally to the nearest cluster by absolute BPM distance
   (no ±7 cap, no "hold as merge candidate" logic).
```

Target: 2–6 raw BPM clusters before sizing adjustments.

**Edge cases:**
- Pool has ≤ 5 tracks: return as one shortlist directly (no clustering needed).
- All tracks within 4 BPM: one cluster, proceed to Camelot sub-clustering.
- Pool is uniformly spread (no peaks found after smoothing): fall back to equal-sized
  quantile splits into `min(MAX_POOL_COUNT, len(tracks) // MIN_SHORTLIST)` groups.

### Step 2 — Camelot sub-clusters (within each BPM cluster)

```
1. Build an adjacency graph: edge between tracks A and B iff
   camelot_compatible(A.camelot_key, B.camelot_key) is True
   (same key, adjacent on wheel, or relative mode pair).
2. Find connected components via BFS.
3. If a BPM cluster has only one Camelot component (or all tracks have
   no/unknown key): keep as-is.
4. If a BPM cluster has 2+ components AND both have ≥ 8 tracks:
   treat each component as a candidate sub-partition.
5. If a component is < 8 tracks: merge back into the largest component
   of the same BPM cluster.
```

### Step 3 — Era split (optional)

Applied independently to each candidate shortlist *after* Camelot sub-clustering.

```
1. Collect tracks with a known year (year is not None and year > 0).
2. If < 60% of tracks have a known year: skip era split for this cluster.
3. Sort known-year tracks ascending.
4. Find the largest year gap between consecutive tracks.
5. If the largest gap ≥ 8 years AND both sides have ≥ 8 tracks:
   split into era_old (year ≤ gap_start) and era_new (year > gap_start).
   Tracks without a known year go to the side whose centroid year is closer.
6. Otherwise: no era split.
```

### Step 4 — Sizing enforcement

After BPM + Camelot + era passes, the candidate shortlists are resized to meet
the 15–25 track target.

```
MIN_SHORTLIST  = 15
MAX_SHORTLIST  = 25
ABSOLUTE_MIN   = 5   # below this a shortlist is dropped rather than passed to Stage 2
MIN_POOL_COUNT = 3
MAX_POOL_COUNT = 5

For each candidate shortlist:
  If len < MIN_SHORTLIST:
    - Attempt merge with the nearest-BPM neighbour shortlist.
    - "Nearest" = abs(median_bpm(A) - median_bpm(B)) is smallest.
    - If merging would exceed MAX_SHORTLIST, keep both and accept the under-sized one.
    - If after the merge attempt the shortlist still has < ABSOLUTE_MIN tracks,
      drop it (Stage 2 cannot meaningfully operate on fewer than 5 tracks).
  If len > MAX_SHORTLIST:
    - Split by Camelot component if one exists that produces two groups each ≥ MIN_SHORTLIST.
    - Otherwise split by era if an 8-year gap exists and both sides ≥ MIN_SHORTLIST.
    - Otherwise: rank tracks by centrality score ascending (lower score = closer to
      cluster centre = keep), take the first MAX_SHORTLIST, and attach the remainder
      to the nearest other shortlist.
      Centrality score components:
        bpm_range    = max(bpm) - min(bpm) across the cluster, or 1.0 if all identical
        bpm_norm     = |t.bpm - median_bpm| / bpm_range            (range 0–0.5 typical)
        camelot_norm = camelot_distance(t.camelot_key, dominant_key) / 6.0
                       (range 0–1.0; max Camelot wheel distance = 6)
        Tracks with no parseable Camelot key: camelot_norm = 1.0 (worst score)
        centrality   = bpm_norm + camelot_norm

After all merges/splits:
  If total count < MIN_POOL_COUNT: keep all remaining shortlists (even if undersized).
  If total count > MAX_POOL_COUNT: merge the two most similar (by BPM centroid proximity).
```

### Step 5 — Mood/title inference

Each shortlist gets a deterministic title and mood derived from cluster
characteristics. These are labels for Stage 2, not prose — Stage 2 ignores them
creatively but uses them as shortlist identifiers.

```python
def _infer_shortlist_mood(tracks: list[Track]) -> tuple[str, str]:
    """Returns (title, mood) for a shortlist."""
    from collections import Counter

    bpm_vals = [t.bpm for t in tracks]
    bpm_lo, bpm_hi = round(min(bpm_vals)), round(max(bpm_vals))

    # Dominant Camelot key — use full key string (e.g. "8A", "5B"), not just number.
    # Filter to tracks with a parseable key to avoid polluting the mode with garbage.
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
    top_tags = ", ".join(t for t, _ in tag_counts.most_common(2)) if tag_counts else ""

    parts = [f"{bpm_lo}–{bpm_hi} BPM", dominant_key]
    if era:
        parts.append(era)
    if top_tags:
        parts.append(top_tags)

    title = " / ".join(parts)
    mood = top_tags or f"{bpm_lo}–{bpm_hi} BPM"
    return title, mood
```

---

## 5. Call site changes in `__main__.py`

There are **three** Stage 1 call sites in `__main__.py`. All three must be replaced.
The old signature `stage1_concepts(pool, genre, cascade_state, ...)` drops `genre` and
`cascade_state` in the new path; the feature flag wrapper handles the fallback.

### Call site A — custom-genre pool (line ~584)

```python
# Before
shortlists = await stage1_concepts(stage1_pool, genre, cascade_state, custom=True)

# After
if os.environ.get("MIXLAB_STAGE1_LLM"):
    shortlists = await stage1_concepts(stage1_pool, genre, cascade_state, custom=True)
else:
    shortlists = partition_pool(stage1_pool, seed=args.stage1_seed)
if not shortlists:
    print("Stage 1: pool too small to partition — skipping.", file=sys.stderr)
    continue
```

### Call site B — standard genre pool (line ~611)

```python
# Before
shortlists = await stage1_concepts(sorted_tracks, genre_label, cascade_state)

# After
if os.environ.get("MIXLAB_STAGE1_LLM"):
    shortlists = await stage1_concepts(sorted_tracks, genre_label, cascade_state)
else:
    shortlists = partition_pool(sorted_tracks, seed=args.stage1_seed)
if not shortlists:
    print("Stage 1: pool too small to partition — skipping.", file=sys.stderr)
    continue
```

### Call site C — outlier / "Misc" pool (line ~617)

```python
# Before
shortlists = await stage1_concepts(genre_outliers, "Misc", cascade_state)

# After
if os.environ.get("MIXLAB_STAGE1_LLM"):
    shortlists = await stage1_concepts(genre_outliers, "Misc", cascade_state)
else:
    shortlists = partition_pool(genre_outliers, seed=args.stage1_seed)
if not shortlists:
    print("Stage 1: pool too small to partition — skipping.", file=sys.stderr)
    continue
```

---

## 6. Files to change

| File | Change |
|------|--------|
| `src/mixlab/clustering.py` | Add `partition_pool()`, `_find_bpm_peaks()`, `_camelot_components()`, `_era_split()`, `_resize_shortlists()`, `_infer_shortlist_mood()` |
| `src/mixlab/__main__.py` | Replace all three Stage 1 LLM call sites; add `--stage1-seed` flag; add feature flag env var |
| `src/mixlab/llm.py` | Mark `_STAGE1_SYSTEM*` prompts and `stage1_concepts()` as deprecated (remove after 30-day soak) |
| `tests/test_clustering.py` | New tests for all `partition_pool` helpers and end-to-end partitioning |
| `tests/test_llm.py` | Remove or migrate Stage 1 LLM tests |
| `tests/test_main.py` | Update all three Stage 1 call-site tests |

---

## 7. Test strategy

### Unit tests (`tests/test_clustering.py`)

```
test_find_bpm_peaks_single_tight_cluster_returns_one_peak
test_find_bpm_peaks_two_well_separated_groups_returns_two_peaks
test_find_bpm_peaks_merges_peaks_closer_than_8_bpm
test_find_bpm_peaks_plateau_uses_leftmost_bucket_as_candidate
test_find_bpm_peaks_edge_buckets_never_returned_as_peaks
test_find_bpm_peaks_equidistant_track_assigned_to_lower_index_peak
test_camelot_components_connected_keys_form_single_component
test_camelot_components_unrelated_keys_form_separate_components
test_era_split_applies_when_gap_large_and_both_sides_sufficient
test_era_split_skipped_when_gap_below_threshold
test_era_split_skipped_when_one_side_too_small
test_era_split_skipped_when_too_few_known_years
test_resize_shortlists_merges_undersized_into_nearest_bpm_neighbour
test_resize_shortlists_drops_shortlist_below_absolute_min
test_resize_shortlists_splits_oversized_by_camelot_component
test_resize_shortlists_centrality_ranks_ascending_keeps_central_tracks
test_resize_shortlists_centrality_normalises_bpm_and_camelot_independently
test_infer_shortlist_mood_uses_full_camelot_key_string_not_number
test_infer_shortlist_mood_mode_b_key_not_coerced_to_a
test_infer_shortlist_mood_filters_unparseable_camelot_keys
test_infer_shortlist_mood_includes_era_when_years_present
test_partition_pool_returns_three_to_five_shortlists_on_typical_input
test_partition_pool_each_shortlist_has_15_to_25_tracks
test_partition_pool_same_seed_same_output_reproducibility
test_partition_pool_tiny_pool_returns_single_shortlist
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
