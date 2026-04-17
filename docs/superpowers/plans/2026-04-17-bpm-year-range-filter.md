# BPM and Year Range Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--min-bpm`, `--max-bpm`, `--min-year`, and `--max-year` CLI flags that filter the track collection post-ingestion; in playlist mode the filter applies only to the candidate library, not to seed tracks.

**Architecture:** A single `_apply_range_filters()` helper is added to `__main__.py`. `run()` calls it on the full collection right after BPM correction. `run_playlist_mode()` calls it on `library_tracks` only, after seeds are resolved. `main()` wires the four new argparse flags and validates that min ≤ max before touching the XML.

**Tech Stack:** Python 3.12, Pydantic (`Track` model), argparse, pytest, `capsys` for stderr assertions.

---

## File Map

| File | Change |
|------|--------|
| `src/mixlab/__main__.py` | Add `_apply_range_filters()`, update `run()` and `run_playlist_mode()` signatures, call filter at correct sites, add 4 argparse args + validation to `main()` |
| `tests/test_main.py` | Add 8 new tests covering the helper, validation, and playlist-mode integration |

---

### Task 1: `_apply_range_filters()` helper — tests first

**Files:**
- Modify: `tests/test_main.py`
- Modify: `src/mixlab/__main__.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_main.py`:

```python
from mixlab.__main__ import _apply_range_filters
from mixlab.models import Track


def _make_track(
    track_id: str,
    bpm: float,
    year: int | None = 2020,
) -> Track:
    return Track(
        track_id=track_id,
        artist="Artist",
        title="Title",
        bpm=bpm,
        camelot_key="8A",
        genre="House",
        year=year,
    )


def test_apply_range_filters_bpm_inclusive(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0), _make_track("2", 135.0), _make_track("3", 140.0), _make_track("4", 145.0)]
    result = _apply_range_filters(tracks, min_bpm=135.0, max_bpm=140.0, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["2", "3"]
    assert "BPM filter [135.0–140.0]" in capsys.readouterr().err


def test_apply_range_filters_bpm_no_op_when_omitted(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 120.0), _make_track("2", 200.0)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["1", "2"]
    assert "BPM filter" not in capsys.readouterr().err


def test_apply_range_filters_year_excludes_none_year(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=2021), _make_track("2", 130.0, year=None)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=2020, max_year=None)
    assert [t.track_id for t in result] == ["1"]
    assert "Year filter" in capsys.readouterr().err


def test_apply_range_filters_year_no_op_when_omitted(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=None), _make_track("2", 130.0, year=2022)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["1", "2"]
    assert "Year filter" not in capsys.readouterr().err


def test_apply_range_filters_logs_format_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=2018), _make_track("2", 135.0, year=2022)]
    _apply_range_filters(tracks, min_bpm=135.0, max_bpm=140.0, min_year=2020, max_year=None)
    err = capsys.readouterr().err
    assert "BPM filter [135.0–140.0]: excluded 1 track(s), 1 remain." in err
    assert "Year filter [2020–]: excluded 0 track(s), 1 remain." in err
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_main.py::test_apply_range_filters_bpm_inclusive -v
```

Expected: `ImportError` or `FAILED` — `_apply_range_filters` does not exist yet.

- [ ] **Step 3: Add `_apply_range_filters` to `__main__.py`**

Insert after the `_apply_do_not_recommend_filter` function (after line 127, before `_format_pipeline_counts`):

```python
def _apply_range_filters(
    tracks: list[Track],
    *,
    min_bpm: float | None,
    max_bpm: float | None,
    min_year: int | None,
    max_year: int | None,
) -> list[Track]:
    result = tracks
    if min_bpm is not None or max_bpm is not None:
        before = len(result)
        lo: float = min_bpm if min_bpm is not None else float("-inf")
        hi: float = max_bpm if max_bpm is not None else float("inf")
        result = [t for t in result if lo <= t.bpm <= hi]
        lo_str = str(min_bpm) if min_bpm is not None else ""
        hi_str = str(max_bpm) if max_bpm is not None else ""
        print(
            f"BPM filter [{lo_str}–{hi_str}]: excluded {before - len(result)} track(s), {len(result)} remain.",
            file=sys.stderr,
        )
    if min_year is not None or max_year is not None:
        before = len(result)
        result = [
            t
            for t in result
            if t.year is not None
            and (min_year is None or t.year >= min_year)
            and (max_year is None or t.year <= max_year)
        ]
        lo_year = str(min_year) if min_year is not None else ""
        hi_year = str(max_year) if max_year is not None else ""
        print(
            f"Year filter [{lo_year}–{hi_year}]: excluded {before - len(result)} track(s), {len(result)} remain.",
            file=sys.stderr,
        )
    return result
```

- [ ] **Step 4: Run all new helper tests**

```bash
.venv/bin/python -m pytest tests/test_main.py -k "apply_range_filters" -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/__main__.py tests/test_main.py
git commit -m "feat(pipeline): add _apply_range_filters helper with BPM and year filtering"
```

---

### Task 2: Wire filter into `run()` (standard + `--all-tracks` mode)

**Files:**
- Modify: `src/mixlab/__main__.py`

- [ ] **Step 1: Update `run()` signature**

Change the signature of `run()` (currently at line 340) from:

```python
async def run(
    genre: str | None,
    duration: int | None,
    export_dir: Path | None,
    stage2_provider: str | None = None,
    all_tracks: bool = False,
) -> None:  # noqa: ARG001 — duration reserved
```

To:

```python
async def run(
    genre: str | None,
    duration: int | None,
    export_dir: Path | None,
    stage2_provider: str | None = None,
    all_tracks: bool = False,
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
) -> None:  # noqa: ARG001 — duration reserved
```

- [ ] **Step 2: Insert the filter call in `run()` body**

After the three ingestion lines (currently lines 348–350):

```python
    tracks = parse_collection(_XML_PATH)
    tracks, denylist_excluded = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)
```

Add immediately after:

```python
    tracks = _apply_range_filters(tracks, min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year)
```

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -v
```

Expected: all existing tests PASS (the new parameters default to `None` so existing call sites are unaffected).

- [ ] **Step 4: Commit**

```bash
git add src/mixlab/__main__.py
git commit -m "feat(pipeline): apply range filters in run() after BPM correction"
```

---

### Task 3: Wire filter into `run_playlist_mode()` (library tracks only)

**Files:**
- Modify: `src/mixlab/__main__.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write the playlist-mode integration test first**

Append to `tests/test_main.py`. This test calls `_apply_range_filters` directly on a simulated library list (mirroring what `run_playlist_mode` will do), and verifies seeds are untouched:

```python
def test_range_filter_preserves_out_of_range_seeds_excludes_library() -> None:
    """Seeds outside BPM range must survive; library tracks outside range must be dropped."""
    seed_tracks = [_make_track("seed1", 120.0, year=2015)]  # would fail a 130–140 BPM filter
    all_tracks = seed_tracks + [
        _make_track("lib1", 135.0, year=2022),  # in range
        _make_track("lib2", 150.0, year=2022),  # out of BPM range
    ]
    seed_ids = frozenset(t.track_id for t in seed_tracks)
    library_tracks = [t for t in all_tracks if t.track_id not in seed_ids]

    filtered_library = _apply_range_filters(library_tracks, min_bpm=130.0, max_bpm=140.0, min_year=None, max_year=None)

    # Seed untouched (it was never passed to the filter).
    assert [t.track_id for t in seed_tracks] == ["seed1"]
    # Only in-range library track remains.
    assert [t.track_id for t in filtered_library] == ["lib1"]
```

- [ ] **Step 2: Run test to confirm it passes already (it tests the helper, not the wiring)**

```bash
.venv/bin/python -m pytest tests/test_main.py::test_range_filter_preserves_out_of_range_seeds_excludes_library -v
```

Expected: PASSED (the helper itself is correct; this test proves the intended call pattern).

- [ ] **Step 3: Update `run_playlist_mode()` signature**

Change the signature (currently at line 212):

```python
async def run_playlist_mode(
    playlist_name: str,
    genre: str | None,
    export_dir: Path | None,
    stage2_provider: str | None,
    all_tracks: bool,
) -> None:
```

To:

```python
async def run_playlist_mode(
    playlist_name: str,
    genre: str | None,
    export_dir: Path | None,
    stage2_provider: str | None,
    all_tracks: bool,
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
) -> None:
```

- [ ] **Step 4: Insert the filter call after `library_tracks` is built**

Find the line that reads (currently line 275):

```python
    library_tracks = [t for t in library_source if t.track_id not in seed_ids]
```

Insert immediately after it:

```python
    library_tracks = _apply_range_filters(
        library_tracks, min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year
    )
```

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mixlab/__main__.py tests/test_main.py
git commit -m "feat(pipeline): apply range filter to library tracks only in playlist mode"
```

---

### Task 4: Add CLI args and inverted-range validation to `main()`

**Files:**
- Modify: `src/mixlab/__main__.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write the validation tests**

Append to `tests/test_main.py`:

```python
def test_main_rejects_inverted_bpm_range(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.__main__ import _validate_range_args

    with pytest.raises(SystemExit) as exc_info:
        _validate_range_args(min_bpm=140.0, max_bpm=130.0, min_year=None, max_year=None)
    assert exc_info.value.code == 1
    assert "--min-bpm" in capsys.readouterr().err


def test_main_rejects_inverted_year_range(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.__main__ import _validate_range_args

    with pytest.raises(SystemExit) as exc_info:
        _validate_range_args(min_bpm=None, max_bpm=None, min_year=2024, max_year=2020)
    assert exc_info.value.code == 1
    assert "--min-year" in capsys.readouterr().err


def test_validate_range_args_passes_when_valid() -> None:
    from mixlab.__main__ import _validate_range_args

    # Should not raise.
    _validate_range_args(min_bpm=130.0, max_bpm=140.0, min_year=2019, max_year=2024)
    _validate_range_args(min_bpm=None, max_bpm=None, min_year=None, max_year=None)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_main.py -k "validate_range" -v
```

Expected: `ImportError` — `_validate_range_args` does not exist yet.

- [ ] **Step 3: Add `_validate_range_args` helper to `__main__.py`**

Insert immediately before `main()`:

```python
def _validate_range_args(
    *,
    min_bpm: float | None,
    max_bpm: float | None,
    min_year: int | None,
    max_year: int | None,
) -> None:
    if min_bpm is not None and max_bpm is not None and min_bpm > max_bpm:
        print(f"ERROR: --min-bpm ({min_bpm}) must not exceed --max-bpm ({max_bpm}).", file=sys.stderr)
        sys.exit(1)
    if min_year is not None and max_year is not None and min_year > max_year:
        print(f"ERROR: --min-year ({min_year}) must not exceed --max-year ({max_year}).", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run validation tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_main.py -k "validate_range" -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Add the four argparse arguments in `main()`**

After the existing `--all-tracks` argument block (which ends around line 620), add:

```python
    parser.add_argument(
        "--min-bpm",
        type=float,
        default=None,
        metavar="BPM",
        help="Minimum BPM (inclusive). Tracks below this value are excluded after ingestion.",
    )
    parser.add_argument(
        "--max-bpm",
        type=float,
        default=None,
        metavar="BPM",
        help="Maximum BPM (inclusive). Tracks above this value are excluded after ingestion.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Minimum release year (inclusive). Tracks with no year are excluded when this is set.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Maximum release year (inclusive). Tracks with no year are excluded when this is set.",
    )
```

- [ ] **Step 6: Wire validation and pass args through in `main()`**

After `args = parser.parse_args()`, before the `if args.genres:` check, add:

```python
    _validate_range_args(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        min_year=args.min_year,
        max_year=args.max_year,
    )
```

Update the `asyncio.run(run_playlist_mode(...))` call:

```python
    if args.playlist:
        asyncio.run(
            run_playlist_mode(
                args.playlist,
                args.genre,
                export_dir,
                args.stage2_provider,
                args.all_tracks,
                min_bpm=args.min_bpm,
                max_bpm=args.max_bpm,
                min_year=args.min_year,
                max_year=args.max_year,
            )
        )
        return
```

Update the `asyncio.run(run(...))` call:

```python
    asyncio.run(
        run(
            args.genre,
            args.duration,
            export_dir,
            args.stage2_provider,
            args.all_tracks,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            min_year=args.min_year,
            max_year=args.max_year,
        )
    )
```

- [ ] **Step 7: Run the full test suite + linting**

```bash
.venv/bin/python -m ruff format . && .venv/bin/python -m ruff check . && .venv/bin/python -m mypy . && .venv/bin/python -m pytest -v
```

Expected: all checks pass, all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mixlab/__main__.py tests/test_main.py
git commit -m "feat(cli): add --min-bpm, --max-bpm, --min-year, --max-year flags"
```
