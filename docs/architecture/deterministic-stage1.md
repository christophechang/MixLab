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
- Replace Stage 1 LLM call site in `__main__.py`
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

---

## 4. Algorithm

### Step 1 — BPM clustering (primary partition)

```
1. Sort tracks by BPM.
2. Bin tracks into 3-BPM-wide histogram buckets.
3. Smooth the histogram with a 3-bucket moving average.
4. Find local maxima (peaks) in the smoothed histogram.
5. Merge any two peaks closer than 8 BPM (keep the heavier one).
6. Assign each track to the nearest peak whose center is within ±7 BPM.
   Tracks outside ±7 BPM of any peak become "outliers".
7. Attach outliers to the nearest cluster by absolute BPM distance
   (no ±7 cap) if the cluster has < 12 tracks, otherwise hold as a
   potential merge candidate.
```

Target: 2–6 raw BPM clusters before sizing adjustments.

**Edge cases:**
- Pool has ≤ 5 tracks: return as one shortlist directly (no clustering needed).
- All tracks within 4 BPM: one cluster, proceed to Camelot sub-clustering.
- Pool is uniformly spread (no peaks): fall back to equal-sized quantile splits.

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
MIN_SHORTLIST = 15
MAX_SHORTLIST = 25
MIN_POOL_COUNT = 3
MAX_POOL_COUNT = 5

For each candidate shortlist:
  If len < MIN_SHORTLIST:
    - Merge with the nearest-BPM neighbour shortlist.
    - "Nearest" = abs(median_bpm(A) - median_bpm(B)) is smallest.
    - If merging would exceed MAX_SHORTLIST, keep both and accept
      the under-sized one (it's better than overshooting the max).
  If len > MAX_SHORTLIST:
    - Split by Camelot component if one exists that cleanly splits.
    - Otherwise split by era if an 8-year gap exists.
    - Otherwise: rank tracks by centrality (BPM distance to cluster median +
      Camelot distance to dominant key), take the top MAX_SHORTLIST,
      and attach the remainder to the nearest other shortlist.

After all merges/splits:
  If total count < MIN_POOL_COUNT: keep all (even undersized).
  If total count > MAX_POOL_COUNT: merge the two most similar (by BPM centroid).
```

### Step 5 — Mood/title inference

Each shortlist gets a deterministic title and mood derived from cluster
characteristics. These are labels for Stage 2, not prose — Stage 2 ignores them
creatively but uses them as shortlist identifiers.

```python
def _infer_shortlist_mood(tracks: list[Track]) -> tuple[str, str]:
    """Returns (title, mood) for a shortlist."""
    bpm_vals = [t.bpm for t in tracks]
    bpm_lo, bpm_hi = round(min(bpm_vals)), round(max(bpm_vals))

    # Dominant Camelot zone (most common key number)
    key_nums = [_camelot_number(t.camelot_key) for t in tracks if t.camelot_key]
    dominant_key = f"{statistics.mode(key_nums)}A" if key_nums else "?"

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

## 5. Call site change in `__main__.py`

Current (genre mode):

```python
# Stage 1 — LLM call
shortlists = await stage1_concepts(pool, cascade_state, ...)
```

Replacement:

```python
# Stage 1 — deterministic partitioner
from mixlab.clustering import partition_pool
shortlists = partition_pool(pool, seed=args.stage1_seed)
if not shortlists:
    print("Stage 1: pool too small to partition — skipping genre.", file=sys.stderr)
    return
```

The feature flag (see §8) wraps this so the LLM path can be restored without
a code change during the soak period.

---

## 6. Files to change

| File | Change |
|------|--------|
| `src/mixlab/clustering.py` | Add `partition_pool()`, `_find_bpm_peaks()`, `_camelot_components()`, `_era_split()`, `_resize_shortlists()`, `_infer_shortlist_mood()` |
| `src/mixlab/__main__.py` | Replace Stage 1 LLM call; add `--stage1-seed` flag; add feature flag env var |
| `src/mixlab/llm.py` | Mark `_STAGE1_SYSTEM*` prompts and `stage1_concepts()` as deprecated (remove after 30-day soak) |
| `tests/test_clustering.py` | New tests for all `partition_pool` helpers and end-to-end partitioning |
| `tests/test_llm.py` | Remove or migrate Stage 1 LLM tests |
| `tests/test_main.py` | Update Stage 1 call-site tests |

---

## 7. Test strategy

### Unit tests (`tests/test_clustering.py`)

```
test_find_bpm_peaks_single_tight_cluster_returns_one_peak
test_find_bpm_peaks_two_well_separated_groups_returns_two_peaks
test_find_bpm_peaks_merges_peaks_closer_than_8_bpm
test_camelot_components_connected_keys_form_single_component
test_camelot_components_unrelated_keys_form_separate_components
test_era_split_applies_when_gap_large_and_both_sides_sufficient
test_era_split_skipped_when_gap_below_threshold
test_era_split_skipped_when_one_side_too_small
test_era_split_skipped_when_too_few_known_years
test_resize_shortlists_merges_undersized_into_nearest_bpm_neighbour
test_resize_shortlists_splits_oversized_by_camelot_component
test_infer_shortlist_mood_uses_bpm_and_dominant_key
test_infer_shortlist_mood_includes_era_when_years_present
test_partition_pool_returns_three_to_five_shortlists_on_typical_input
test_partition_pool_each_shortlist_has_15_to_25_tracks
test_partition_pool_same_seed_same_output_reproducibility
test_partition_pool_tiny_pool_returns_single_shortlist
test_partition_pool_all_same_bpm_returns_camelot_subgroups_or_single
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

Add env var `MIXLAB_STAGE1_LLM=1` to restore the LLM path during the soak period:

```python
# __main__.py — Stage 1 call site
if os.environ.get("MIXLAB_STAGE1_LLM"):
    shortlists = await stage1_concepts(pool, cascade_state, ...)
else:
    shortlists = partition_pool(pool, seed=args.stage1_seed)
```

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
- [ ] `MIXLAB_STAGE1_LLM=1` restores LLM path without code change
- [ ] All new tests pass; Stage 1 LLM tests removed or migrated

---

## 10. Open questions

1. **Outlier handling**: tracks that fall outside all BPM clusters — attach to nearest
   cluster always, or surface as a separate "miscellaneous" shortlist? Current proposal:
   attach to nearest. Revisit if outlier count is consistently > 20% of pool.

2. **Seed exposure**: should `--stage1-seed` be a CLI flag, or always derived from
   current date (e.g., `seed = int(datetime.date.today().strftime("%Y%m%d"))`)?
   Date-derived seed means "one partition per day" — deterministic within a session
   but varying across days without user action. Preferred: explicit flag, default None
   (stable sort-based tie-breaking).

3. **Custom-genre pools**: the current Stage 1 LLM is aware of sub-genre labels.
   `partition_pool()` will use BPM + Camelot only for clustering. If sub-genre
   coherence matters more than BPM proximity for a given custom genre, a future
   extension can add a `sub_genre_weight` parameter. Out of scope for v1.
