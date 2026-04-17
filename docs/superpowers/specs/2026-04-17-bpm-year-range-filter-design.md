# BPM and Year Range Filter — Design Spec

**Date:** 2026-04-17

## Overview

Add optional `--min-bpm`, `--max-bpm`, `--min-year`, and `--max-year` CLI flags to MixLab. When specified, tracks outside the requested range are dropped before pipeline processing. Behaviour differs between run modes:

- **Standard run / `--all-tracks`:** filter applied to the full collection right after ingestion, so the availability table, genre clusters, and everything downstream all see the filtered set.
- **Playlist mode (`--playlist`):** filter applied **only to the candidate library pool** — the tracks that may be added around the seed playlist. Seed tracks from the named playlist are never filtered; the source playlist is always resolved from the full parsed + denylisted + BPM-corrected collection.

## CLI Interface

```
mixlab --genre house --min-bpm 125 --max-bpm 130
mixlab --playlist "Monday Night" --min-year 2020
mixlab --genre 170 --min-bpm 168 --max-bpm 175 --min-year 2019 --max-year 2024
mixlab                               # no filters — behaviour unchanged
```

All four flags are optional and independent. Any combination is valid.

## Filter Behaviour

- **BPM:** inclusive on both ends. A track with `bpm=135` passes `--min-bpm 135 --max-bpm 140`.
- **Year:** inclusive on both ends. Tracks with `year=None` are excluded when any year filter is active.
- **No-op when omitted:** if neither min nor max is specified for an axis, no filtering is applied and no log line is emitted for that axis.
- Filters are applied **after** `apply_bpm_corrections` so half-time DnB BPMs are corrected before the range check.

## Implementation

### New helper in `__main__.py`

```python
def _apply_range_filters(
    tracks: list[Track],
    *,
    min_bpm: float | None,
    max_bpm: float | None,
    min_year: int | None,
    max_year: int | None,
) -> list[Track]:
```

Returns the filtered list and logs to stderr when a filter is active (one line per active axis):

```
BPM filter [135–140]: excluded 47 track(s), 312 remain.
Year filter [2019–2024]: excluded 12 track(s), 300 remain.
```

### Call sites (two locations in `__main__.py`)

1. **`run()`** — called once, after `apply_bpm_corrections`, before the unplayed-track fetch. The availability-table-only path (no genre) is the early-return branch inside `run()`, so it naturally sees the filtered collection without a separate call site.

2. **`run_playlist_mode()`** — called on `library_tracks` only, **after** `seed_tracks` are resolved and removed from the candidate pool. Seed resolution still uses the full (unfiltered) `tracks` list so the source playlist is never truncated.

### Argument parsing in `main()`

```
--min-bpm FLOAT   Minimum BPM (inclusive). No filter if omitted.
--max-bpm FLOAT   Maximum BPM (inclusive). No filter if omitted.
--min-year INT    Minimum release year (inclusive). No filter if omitted.
--max-year INT    Maximum release year (inclusive). Tracks with no year are excluded.
```

Values are passed down into `run()` and `run_playlist_mode()` as additional keyword arguments.

## Error Handling

- If `--min-bpm > --max-bpm` or `--min-year > --max-year`, print an error to stderr and exit with code 1 before touching the XML.

## Testing

### Helper unit tests
- `test_apply_range_filters_bpm_inclusive` — edges included, out-of-range excluded.
- `test_apply_range_filters_bpm_no_op_when_omitted` — all tracks pass when neither bound given.
- `test_apply_range_filters_year_excludes_none_year` — `year=None` tracks excluded when year filter active.
- `test_apply_range_filters_year_no_op_when_omitted` — `year=None` tracks pass when no year filter.
- `test_apply_range_filters_logs_to_stderr` — stderr output matches expected format.

### Validation tests
- `test_main_rejects_inverted_bpm_range` — `--min-bpm 140 --max-bpm 130` exits 1.
- `test_main_rejects_inverted_year_range` — same for year.

### Playlist-mode integration test
- `test_playlist_mode_range_filter_preserves_seeds_excludes_library` — seed tracks outside the BPM/year range are kept; library tracks outside the range are excluded before `build_zone_shortlists`.

## Out of Scope

- No changes to `reader.py`, `models.py`, or any downstream pipeline module.
- No changes to Discord output format or XML export.
- No persistent filter config — flags must be passed on every invocation.
