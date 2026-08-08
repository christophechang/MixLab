# Library Map Subcommand (`mixlab --map`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `mixlab --map` emits a deterministic JSON payload of concept-direction candidates per engine pool — the engine half of mixlab-web's constellation overlay (issue #40, Milestone B).

**Architecture:** A new public `enumerate_directions()` in `directions.py` runs the existing #53 builders exhaustively (no seed rotation, no cap, no materialisation, no printing). A new `library_map.py` module assembles pools exactly as the run pipeline does (`group_by_genre` + `build_custom_genre_pool`, played-mode filtering via `matcher`) and serialises the payload. `__main__.py` gains a `--map` early-dispatch mirroring the `--worker` pattern. Contract consumer: mixlab-web `docs/superpowers/specs/2026-08-06-library-map-design.md` §7.

**Tech Stack:** Python 3.12, Pydantic models, argparse (flat flags), pytest.

## Global Constraints

- `mypy --strict` clean; full annotations; `from __future__ import annotations` in every file; built-in generics; no `Any`.
- Ruff format + check clean; no suppressions without rule code and reason.
- No LLM calls anywhere in the map path. Deterministic: same XML + same mode + same seed + same catalog → byte-identical JSON.
- When `--map` is active, **stdout carries the JSON payload and nothing else** (the API worker will capture stdout); any diagnostics go to stderr.
- Reuse existing functions (`group_by_genre`, `build_custom_genre_pool`, `filter_unplayed`, `filter_played`, `fetch_played_tracks`, `parse_collection`, `apply_bpm_corrections`, the do-not-recommend filter) — never duplicate pool or matching logic.
- Tests mirror src layout, named `test_<what>_<condition>_<expected>`, no external service calls (HTTP via `respx` or direct function monkeypatching with `pytest-mock`).
- Commands (from repo root): `.venv/bin/python -m ruff format .`, `.venv/bin/python -m ruff check .`, `.venv/bin/python -m mypy .`, `.venv/bin/python -m pytest`.
- Conventional commits, no Co-Authored-By trailer.

---

### Task 1: `enumerate_directions` in directions.py

**Files:**
- Modify: `src/mixlab/directions.py` (add one function after `generate_directions`)
- Test: `tests/test_directions.py` (append)

**Interfaces:**
- Produces: `enumerate_directions(pool: list[Track], *, seed: int) -> list[Direction]` — every surviving builder candidate, sorted by `(-feasibility, direction_type)`; no rotation, no `max_directions` cap, no `build_mix_canvas`, no stdout printing.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_directions.py`, reusing the file's existing pool fixtures (read the file first; it already builds pools rich enough for `generate_directions` tests — use the same fixture(s) the `generate_directions` tests use):

```python
def test_enumerate_directions_returns_sorted_candidates(rich_pool: list[Track]) -> None:
    # rich_pool: substitute the existing fixture name used by generate_directions tests.
    result = enumerate_directions(rich_pool, seed=0)
    assert result, "builders should propose at least one direction for the rich fixture pool"
    feasibilities = [d.feasibility for d in result]
    assert feasibilities == sorted(feasibilities, reverse=True) or [
        (-d.feasibility, d.direction_type) for d in result
    ] == sorted((-d.feasibility, d.direction_type) for d in result)
    assert all(isinstance(d, Direction) for d in result)


def test_enumerate_directions_same_seed_identical_output(rich_pool: list[Track]) -> None:
    first = enumerate_directions(rich_pool, seed=7)
    second = enumerate_directions(rich_pool, seed=7)
    assert [(d.direction_type, d.title, d.track_ids) for d in first] == [
        (d.direction_type, d.title, d.track_ids) for d in second
    ]


def test_enumerate_directions_empty_pool_returns_empty(capsys: pytest.CaptureFixture[str]) -> None:
    assert enumerate_directions([], seed=0) == []
    assert capsys.readouterr().out == ""  # never prints, unlike generate_directions
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_directions.py -k enumerate -q` → FAIL (name not defined).

- [ ] **Step 3: Implement** — in `src/mixlab/directions.py`, directly above `generate_directions`:

```python
def enumerate_directions(pool: list[Track], *, seed: int) -> list[Direction]:
    """Every surviving Direction candidate over ``pool``, exhaustively.

    The library-map path (#40): unlike :func:`generate_directions` there is no
    seed-derived rotation, no ``max_directions`` cap, no materialisation into
    MixCanvas, and no run-log printing — the caller wants the full candidate
    field with feasibility scores, deterministically ordered.
    """
    candidates = [
        direction for builder in _BUILDERS if (direction := builder(pool, seed=seed)) is not None
    ]
    candidates.sort(key=lambda d: (-d.feasibility, d.direction_type))
    return candidates
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/test_directions.py -q` → all pass (pre-existing included).

- [ ] **Step 5: Commit** — `git add src/mixlab/directions.py tests/test_directions.py && git commit -m "feat(directions): add exhaustive enumerate_directions for library map"`

---

### Task 2: `library_map.py` payload builder

**Files:**
- Create: `src/mixlab/library_map.py`
- Test: `tests/test_library_map.py`

**Interfaces:**
- Consumes: `enumerate_directions` (Task 1); `group_by_genre`, `build_custom_genre_pool` (clustering.py); `GENRE_MAP`, `CUSTOM_GENRES` (config.py); `filter_played`, `filter_unplayed` (matcher.py); `Track`, `PlayedTrack`, `Direction` (models.py).
- Produces: `MapMode = Literal["all", "unplayed", "played"]`; `build_map_payload(tracks: list[Track], *, mode: MapMode, seed: int, played: list[PlayedTrack]) -> dict[str, object]`; `render_map_json(payload: dict[str, object]) -> str` (2-space indent, trailing newline).

Payload shape (the §7 contract plus an envelope):

```json
{
  "version": 1,
  "mode": "unplayed",
  "seed": 0,
  "collection_tracks": 1997,
  "catalog_tracks": 1381,
  "pools": {
    "house": {
      "track_count": 220,
      "directions": [
        {"direction_type": "label_spotlight", "title": "…", "mood": "…",
         "brief": "…", "feasibility": 0.87, "track_ids": ["123", "456"]}
      ]
    }
  }
}
```

Pool key order: `GENRE_MAP` insertion order, then `CUSTOM_GENRES` insertion order — every key present even when its pool is empty (`track_count: 0, directions: []`), so the web app never guesses at absent keys.

- [ ] **Step 1: Write the failing tests** — `tests/test_library_map.py`. Build a small synthetic `Track` list directly (mirror the constructor usage in `tests/test_clustering.py` — read it first for required fields), covering two standard pools plus one custom-pool BPM inclusion/exclusion pair:

```python
from __future__ import annotations

import json

from mixlab.config import CUSTOM_GENRES, GENRE_MAP
from mixlab.library_map import build_map_payload, render_map_json
from mixlab.models import PlayedTrack


def test_build_map_payload_all_mode_covers_every_pool_key(map_tracks) -> None:
    payload = build_map_payload(map_tracks, mode="all", seed=0, played=[])
    assert payload["version"] == 1
    assert payload["mode"] == "all"
    pools = payload["pools"]
    assert list(pools) == [*GENRE_MAP, *CUSTOM_GENRES]
    for entry in pools.values():
        assert set(entry) == {"track_count", "directions"}


def test_build_map_payload_unplayed_mode_filters_catalog_matches(map_tracks) -> None:
    # Pick one track from the fixture; a PlayedTrack with its artist/title drops it.
    victim = map_tracks[0]
    played = [PlayedTrack(artist=victim.artist, title=victim.title)]
    all_counts = build_map_payload(map_tracks, mode="all", seed=0, played=played)
    unplayed = build_map_payload(map_tracks, mode="unplayed", seed=0, played=played)
    assert unplayed["catalog_tracks"] == 1
    total = lambda p: sum(e["track_count"] for k, e in p["pools"].items() if k in GENRE_MAP)  # noqa: E731
    assert total(unplayed) == total(all_counts) - 1


def test_render_map_json_deterministic_and_parseable(map_tracks) -> None:
    payload = build_map_payload(map_tracks, mode="all", seed=3, played=[])
    first = render_map_json(payload)
    second = render_map_json(build_map_payload(map_tracks, mode="all", seed=3, played=[]))
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["seed"] == 3
```

Define `map_tracks` as a module-level pytest fixture in this file (a dozen tracks across House/Techno tags from `GENRE_MAP`, BPMs straddling the `170` custom range so the custom-pool BPM filter is exercised).

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_library_map.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** — `src/mixlab/library_map.py`:

```python
"""Library-map payload (#40 / mixlab-web Milestone B).

Deterministic, LLM-free enumeration of concept-direction candidates per engine
pool, serialised for the web app's constellation overlay. Contract:
mixlab-web docs/superpowers/specs/2026-08-06-library-map-design.md §7 —
change the shape there first, then here.
"""

from __future__ import annotations

import json
from typing import Literal

from mixlab.clustering import build_custom_genre_pool, group_by_genre
from mixlab.config import CUSTOM_GENRES, GENRE_MAP
from mixlab.directions import enumerate_directions
from mixlab.matcher import filter_played, filter_unplayed
from mixlab.models import Direction, PlayedTrack, Track

MapMode = Literal["all", "unplayed", "played"]


def _direction_entry(direction: Direction) -> dict[str, object]:
    return {
        "direction_type": direction.direction_type,
        "title": direction.title,
        "mood": direction.mood,
        "brief": direction.brief,
        "feasibility": direction.feasibility,
        "track_ids": list(direction.track_ids),
    }


def _pool_entry(pool: list[Track], *, seed: int) -> dict[str, object]:
    return {
        "track_count": len(pool),
        "directions": [_direction_entry(d) for d in enumerate_directions(pool, seed=seed)],
    }


def build_map_payload(
    tracks: list[Track],
    *,
    mode: MapMode,
    seed: int,
    played: list[PlayedTrack],
) -> dict[str, object]:
    """Assemble the full map payload over ``tracks`` (post do-not-recommend)."""
    if mode == "unplayed":
        scoped = filter_unplayed(tracks, played)
    elif mode == "played":
        scoped = filter_played(tracks, played)
    else:
        scoped = tracks

    by_genre = group_by_genre(scoped, GENRE_MAP)
    pools: dict[str, dict[str, object]] = {}
    for key in GENRE_MAP:
        pools[key] = _pool_entry(by_genre.get(key, []), seed=seed)
    for key in CUSTOM_GENRES:
        custom_pool = build_custom_genre_pool(key, scoped, CUSTOM_GENRES, GENRE_MAP)
        pools[key] = _pool_entry(custom_pool, seed=seed)

    return {
        "version": 1,
        "mode": mode,
        "seed": seed,
        "collection_tracks": len(tracks),
        "catalog_tracks": len(played),
        "pools": pools,
    }


def render_map_json(payload: dict[str, object]) -> str:
    """Stable serialisation: 2-space indent, insertion order, trailing newline."""
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/test_library_map.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add src/mixlab/library_map.py tests/test_library_map.py && git commit -m "feat: build deterministic library-map payload per engine pool"`

---

### Task 3: CLI `--map` dispatch

**Files:**
- Modify: `src/mixlab/__main__.py` (three additions: two argparse flags, one early dispatch, one helper)
- Test: `tests/test_library_map.py` (append CLI-helper tests)

**Interfaces:**
- Consumes: `build_map_payload`, `render_map_json` (Task 2); existing `_XML_PATH`, `parse_collection`, `apply_bpm_corrections`, `_apply_do_not_recommend_filter`, `fetch_played_tracks`, existing `--mode` flag (choices unplayed/all/played, default unplayed).
- Produces: `mixlab --map [--mode …] [--seed N] [--out PATH]`. New helper `_run_map_cli(mode: str, seed: int, out: Path | None) -> int`.

Behavior:
- Read the collection exactly as the run pipeline's preamble does: `parse_collection(_XML_PATH)` → `apply_bpm_corrections(...)` if and only if the run paths apply it at this stage (mirror the call sequence around `__main__.py:452-456` and `:712` — read those regions first and copy the order) → `_apply_do_not_recommend_filter(tracks, _XML_PATH)`.
- `mode != "all"` → fetch the catalog the same way the availability path does (see the `fetch_played_tracks` call sites around `__main__.py:712-722`; wrap in `asyncio.run` if that's what the sync call sites do — copy the existing pattern including api key/base URL sourcing). `mode == "all"` → `played=[]`, no network.
- Emit `render_map_json(payload)` to stdout via a bare `print(..., end="")`; with `--out PATH`, also write the file and print nothing else to stdout except the JSON (the JSON always goes to stdout — `--out` is additive). Any status lines go to `sys.stderr`.
- Dispatch early, immediately after the `args.worker` dispatch block: `if args.map: raise SystemExit(_run_map_cli(args.mode, args.seed, args.out))`.

Argparse additions (place next to the existing mode/worker flags, matching the file's style):

```python
    parser.add_argument(
        "--map",
        action="store_true",
        help="Emit a deterministic JSON library map (direction candidates per pool) and exit. No LLM calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for --map direction enumeration (default 0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="With --map: also write the JSON payload to this path.",
    )
```

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_library_map.py` — test the helper, not a subprocess. Monkeypatch the collection/catalog loaders so no file or network is touched:

```python
def test_run_map_cli_all_mode_prints_json_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], map_tracks
) -> None:
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    exit_code = cli._run_map_cli("all", 0, None)
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "all"
    assert "pools" in payload


def test_run_map_cli_out_flag_writes_identical_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, map_tracks
) -> None:
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    target = tmp_path / "map.json"
    assert cli._run_map_cli("all", 0, target) == 0
    assert target.read_text() == capsys.readouterr().out
```

Adjust the monkeypatch targets to whatever `_run_map_cli` actually calls (if `apply_bpm_corrections` sits in the sequence, patch it to identity as well). If `--mode unplayed` requires API config env vars, the helper must fail with a clear stderr message and non-zero exit when they're absent — add a test for that case using `monkeypatch.delenv`.

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_library_map.py -q` → new tests FAIL (`_run_map_cli` missing).

- [ ] **Step 3: Implement** `_run_map_cli` in `__main__.py` + the argparse flags + the dispatch line, honoring the Behavior block above verbatim.

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/test_library_map.py tests/test_directions.py -q` → PASS. Then a live smoke run against the real collection if `import/rekordbox.xml` (or `MIXLAB_COLLECTION_PATH`) exists: `./mixlab --map --mode all | .venv/bin/python -m json.tool | head -20` — record the output head in the report; skip gracefully (and say so) if no collection file is present.

- [ ] **Step 5: Commit** — `git add src/mixlab/__main__.py tests/test_library_map.py && git commit -m "feat(cli): add --map flag emitting the library-map JSON"`

---

### Task 4: Full gate + changelog

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Changelog.** New top section following the file's existing style (this repo has no Unreleased section — releases add their own dated heading; put the entry under a `## Unreleased` heading for the next release to rename):

```markdown
## Unreleased

- **`mixlab --map` — library-map payload for the web app (#40, Milestone B).** Emits a
  deterministic JSON map of concept-direction candidates per engine pool (all standard +
  custom pools, exhaustive builder enumeration with feasibility scores, no LLM calls),
  honouring `--mode` and the do-not-recommend playlist. Consumed by mixlab-web's
  constellation overlay; contract lives in mixlab-web's library-map design spec §7.
```

- [ ] **Step 2: Full gate.**

Run: `.venv/bin/python -m ruff format . && .venv/bin/python -m ruff check . && .venv/bin/python -m mypy . && .venv/bin/python -m pytest`
Expected: all clean/green. Fix anything that isn't before committing.

- [ ] **Step 3: Commit** — `git add -A && git commit -m "chore: changelog for mixlab --map"`

---

## Out of scope

API worker job + endpoints (soundcloud-ai-mix-recommender-api) and the web constellation overlay are the next milestones; this plan delivers only the engine half. The `--map` flag intentionally ships unused by the worker until the API job lands.
