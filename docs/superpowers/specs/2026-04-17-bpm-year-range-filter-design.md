# BPM and Year Range Filter — Design Spec

**Date:** 2026-04-17

## Overview

Add optional `--min-bpm`, `--max-bpm`, `--min-year`, and `--max-year` CLI flags to MixLab. When specified, tracks outside the requested range are dropped immediately after XML ingestion and before any pipeline processing. All run modes (standard genre run, `--all-tracks`, `--playlist`) are affected uniformly.

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

Single function, called once per run path right after the existing ingestion + denylist + BPM-correction sequence. Returns the filtered list and logs to stderr when a filter is active.

Log format (one line per active axis):
```
BPM filter [135–140]: excluded 47 track(s), 312 remain.
Year filter [2019–2024]: excluded 12 track(s), 300 remain.
```

### Call sites (three locations in `__main__.py`)

1. `run()` — after `apply_bpm_corrections`, before the unplayed-track fetch.
2. `run_playlist_mode()` — same position.
3. The availability-table-only path (no genre specified) — same position, so the table reflects the filtered collection.

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

- `test_apply_range_filters_bpm_inclusive` — edges included, out-of-range excluded.
- `test_apply_range_filters_bpm_no_op_when_omitted` — all tracks pass when neither bound given.
- `test_apply_range_filters_year_excludes_none_year` — `year=None` tracks excluded when year filter active.
- `test_apply_range_filters_year_no_op_when_omitted` — `year=None` tracks pass when no year filter.
- `test_apply_range_filters_logs_to_stderr` — stderr output matches expected format.
- `test_main_rejects_inverted_bpm_range` — `--min-bpm 140 --max-bpm 130` exits 1.
- `test_main_rejects_inverted_year_range` — same for year.

## Out of Scope

- No changes to `reader.py`, `models.py`, or any downstream pipeline module.
- No changes to Discord output format or XML export.
- No persistent filter config — flags must be passed on every invocation.
