# Playlist Completion Mode — Implementation Plan

## Context

MixLab is a Python CLI app (`src/mixlab/`) that reads a Rekordbox XML export, filters played tracks,
generates LLM-backed mix concepts via a Stage 1 (pre-screening) → Stage 2 (curation + report) pipeline,
exports Rekordbox playlists, and optionally posts to Discord.

This plan adds a new `--playlist` mode that seeds concept generation from an existing Rekordbox
playlist, expands the candidate pool across the full library, and produces exactly one coherent
concept as output.

---

## Architecture Summary

### What exists today

| Concern | Location | Notes |
|---|---|---|
| Collection parsing | `src/mixlab/reader.py:parse_collection` | Reads `<COLLECTION>` only |
| Playlist membership | **Not parsed** | `<PLAYLISTS>` subtree exists in XML but is unread |
| Play history | `src/mixlab/matcher.py:filter_unplayed` + `Track.play_count` | Two signals: external API and Rekordbox play count |
| Candidate pool | `src/mixlab/__main__.py` inline, via `clustering.py` | Genre-scoped |
| Stage 1 (pre-screen) | `src/mixlab/llm.py:stage1_concepts` | Technical grouping by BPM + key |
| Stage 2 (curation) | `src/mixlab/llm.py:stage2_curate_and_report` | Produces 3–6 concepts |
| Export | `src/mixlab/playlist_exporter.py` | Accepts `list[MixConcept]` — 1-element list is fine |

### What is reused unchanged

- `parse_collection` / `apply_bpm_corrections`
- `filter_unplayed` / `is_played` from `matcher.py`
- `sort_by_camelot` from `clustering.py`
- `select_stage1_window` from `llm.py`
- All of `playlist_exporter.py`
- `send_report` from `discord_client.py`

### New surface area

| What | Where |
|---|---|
| `parse_playlists(xml_path)` | Add to `reader.py` |
| `src/mixlab/playlist_mode.py` | New module: resolve playlist, build candidate pool |
| `_STAGE1_SYSTEM_PLAYLIST` constant | Add to `llm.py` |
| `[seed]` annotation in `_tracks_to_text` **and** Stage 2 rendering loop | Modify `llm.py` |
| `seed_ids` param in `stage2_curate_and_report` | Modify `llm.py` |
| Playlist-mode prompt prefix in Stage 2 | Modify `llm.py:stage2_curate_and_report` |
| `select_shortlists_for_playlist_stage2` | Add to `llm.py` |
| `run_playlist_mode()` coroutine | Add to `__main__.py` |
| `--playlist` CLI argument | Add to `__main__.py` |

---

## Step-by-Step Implementation

### Step 1 — Add `parse_playlists` to `reader.py`

**Purpose:** Read playlist membership from the Rekordbox XML `<PLAYLISTS>` subtree.

**Rekordbox XML structure:**
```xml
<PLAYLISTS>
  <NODE Type="0" Name="ROOT" Count="N">
    <NODE Type="0" Name="Folder" Count="M">
      <NODE Type="1" Name="My Playlist" KeyType="0" Entries="K">
        <TRACK Key="123"/>
        <TRACK Key="456"/>
      </NODE>
    </NODE>
  </NODE>
</PLAYLISTS>
```
`Type="0"` nodes are folders. `Type="1"` nodes are playlists. `TRACK Key` is the TrackID.

**New function signature:**
```python
def parse_playlists(xml_path: Path) -> dict[str, list[str]]:
    """Return {full_path: [track_id, ...]} from the PLAYLISTS subtree.

    Keys are slash-separated folder paths, e.g. "Folder/SubFolder/My Playlist".
    Top-level playlists use just their name as the key.
    Recurses into folder nodes (Type="0"). Leaf nodes (Type="1") become entries.
    Returns an empty dict if no PLAYLISTS node is present.
    """
```

**Keys are full folder paths, not bare names.** This avoids silent collisions when Rekordbox
contains playlists with the same name in different folders.

Example: a playlist "Warm Up" inside folder "Sets" is keyed as `"Sets/Warm Up"`. A top-level
playlist "Warm Up" is keyed as `"Warm Up"`. These are distinct entries.

**Implementation notes:**
- Parse with `etree.parse` (same pattern as `parse_collection`)
- Walk recursively, threading a path prefix: if node has `Type="0"`, recurse with
  `prefix + node_name + "/"` (skip the synthetic ROOT node at depth 0); if `Type="1"`, collect
  `TRACK Key` values and store under `prefix + node_name`
- Values are strings (same as `Track.track_id`)
- Do not filter for valid BPM/key here — done downstream via `tracks_by_id` lookup
- The ROOT node itself (the top-level `Type="0"` with Name="ROOT") is the container, not a
  folder name to include in paths — skip it when building the prefix

**Tests to add in `tests/test_reader.py`:**
- `test_parse_playlists_returns_full_path_keys` — nested playlist keyed as "Folder/Playlist"
- `test_parse_playlists_top_level_playlist_keyed_by_name_only` — no folder prefix
- `test_parse_playlists_returns_empty_when_no_playlists_node` — XML with no `<PLAYLISTS>`
- `test_parse_playlists_duplicate_names_in_different_folders_are_distinct_keys`

---

### Step 2 — Add `src/mixlab/playlist_mode.py` (new module)

**Purpose:** Resolve playlist name, validate, build candidate pool.

#### 2a — Playlist resolver

```python
def resolve_playlist(
    name: str,
    playlists: dict[str, list[str]],
) -> list[str]:
    """Return track IDs for the named playlist.

    Lookup order:
    1. Exact match against full-path keys (case-insensitive).
    2. Name-only suffix match: check whether the name equals the final
       component of any key (case-insensitive).

    If step 2 finds exactly one match, return it.
    If step 2 finds multiple matches, raise ValueError listing all
    ambiguous full paths so the user can pass the full path instead.
    If no match at all, raise ValueError with difflib close-match
    suggestions.
    """
```

- Exact match: `name.lower() in {k.lower(): v for k, v in playlists.items()}`
- Suffix match: `k.lower().endswith("/" + name.lower()) or k.lower() == name.lower()`
- Multiple suffix matches → raise with message:
  ```
  Ambiguous playlist name "Warm Up". Found in multiple folders:
    Sets/Warm Up
    Archive/Warm Up
  Pass the full path to disambiguate.
  ```
- No match → `difflib.get_close_matches(name.lower(), [k.lower() for k in playlists], n=5, cutoff=0.6)` → include in error message
- Do not call `sys.exit` here — let `__main__.py` handle that

#### 2b — Candidate pool construction

```python
_SEED_WEIGHT: float = 2.0
_UNPLAYED_WEIGHT: float = 1.5
_MAX_PLAYLIST_POOL: int = 120
_MIN_PLAYLIST_TRACKS: int = 4

def build_playlist_pool(
    seed_track_ids: list[str],
    all_tracks: list[Track],
    tracks_by_id: dict[str, Track],
    unplayed_ids: set[str] | None,
    all_tracks_flag: bool = False,
    bpm_expansion: float = 15.0,
) -> list[Track]:
    """Return a scored, sorted, capped candidate pool for playlist mode.

    Pool = seed tracks ∪ BPM-range-compatible library tracks.
    Seed tracks are always included. Library tracks are capped so the
    total does not exceed _MAX_PLAYLIST_POOL.

    unplayed_ids: set of track IDs known to be unplayed (derived from
        filter_unplayed when catalog API is available, or None to fall
        back to play_count == 0).
    all_tracks_flag: when True, unplayed weighting is disabled entirely.
    """
```

**Unplayed signal rules:**

| Condition | Unplayed signal used for scoring |
|---|---|
| `all_tracks_flag=True` | Disabled — `_UNPLAYED_WEIGHT` not applied |
| `all_tracks_flag=False`, `unplayed_ids` is a `set` | `t.track_id in unplayed_ids` |
| `all_tracks_flag=False`, `unplayed_ids is None` | `t.play_count == 0` (Rekordbox's own count) |

This matches existing pipeline semantics: catalog API result takes precedence; Rekordbox
`play_count` is the fallback; `--all-tracks` disables history entirely.

**Logic:**
1. `seed_tracks = [tracks_by_id[tid] for tid in seed_track_ids if tid in tracks_by_id]`
2. If `len(seed_tracks) < _MIN_PLAYLIST_TRACKS`: raise `ValueError` with explanatory message
3. `bpm_min = min(t.bpm for t in seed_tracks) - bpm_expansion`
   `bpm_max = max(t.bpm for t in seed_tracks) + bpm_expansion`
4. `seed_ids = {t.track_id for t in seed_tracks}`
5. `library_tracks = [t for t in all_tracks if t.track_id not in seed_ids and bpm_min <= t.bpm <= bpm_max]`
6. Scoring function:
   ```python
   def _is_unplayed(t: Track) -> bool:
       if unplayed_ids is not None:
           return t.track_id in unplayed_ids
       return t.play_count == 0

   def _score(t: Track) -> float:
       score = 1.0
       if t.track_id in seed_ids:
           score *= _SEED_WEIGHT
       if not all_tracks_flag and _is_unplayed(t):
           score *= _UNPLAYED_WEIGHT
       return score
   ```
7. Sort library tracks by `_score` descending, cap at `_MAX_PLAYLIST_POOL - len(seed_tracks)`
8. Return `seed_tracks + capped_library_tracks`

**Tests to add in `tests/test_playlist_mode.py`:**
- `test_resolve_playlist_exact_match`
- `test_resolve_playlist_case_insensitive`
- `test_resolve_playlist_suffix_match_unambiguous`
- `test_resolve_playlist_suffix_match_ambiguous_raises_with_paths`
- `test_resolve_playlist_not_found_raises_with_suggestions`
- `test_resolve_playlist_no_suggestions_when_nothing_close`
- `test_build_playlist_pool_includes_all_seed_tracks`
- `test_build_playlist_pool_expands_with_bpm_range`
- `test_build_playlist_pool_too_few_seed_tracks_raises`
- `test_build_playlist_pool_seed_tracks_score_highest`
- `test_build_playlist_pool_unplayed_ids_ranked_above_played`
- `test_build_playlist_pool_play_count_fallback_when_unplayed_ids_none`
- `test_build_playlist_pool_played_tracks_remain_in_pool`
- `test_build_playlist_pool_all_tracks_flag_disables_unplayed_bonus`
- `test_build_playlist_pool_capped_at_max_with_seed_guaranteed`
- `test_build_playlist_pool_deduplicates_seed_from_library`

---

### Step 3 — Stage 1 prompt variant and `[seed]` annotation in `llm.py`

#### 3a — Annotate seed tracks in `_tracks_to_text`

Add an optional `seed_ids: frozenset[str] | None = None` parameter:

```python
def _tracks_to_text(tracks: list[Track], seed_ids: frozenset[str] | None = None) -> str:
```

Append ` | [seed]` to the line when `seed_ids is not None and t.track_id in seed_ids`.

#### 3b — New `_STAGE1_SYSTEM_PLAYLIST` constant

```python
_STAGE1_SYSTEM_PLAYLIST = """\
You are a music data analyst pre-screening a DJ's track collection to build candidate shortlists \
for a playlist completion concept.

Tracks marked [seed] come from an existing playlist and represent the intended musical direction. \
Treat them as strong candidates — but group by BPM and harmonic compatibility above all else. \
A seed track that is an outlier (more than 8 BPM from the group median, or in a harmonically \
unrelated key) should still be excluded from any group where it does not fit.

For each shortlist:
- Group tracks that are plausibly technically compatible: similar BPM (±6 BPM within the pool) \
and harmonically related keys (adjacent or nearby Camelot positions).
- Each shortlist should contain 15–25 candidate tracks.
- Generate 1–3 distinct shortlists. If the material only supports one coherent group, produce one.
- Do NOT make final ordering decisions. Simply group technically compatible tracks.

Give each shortlist a rough descriptive title and a one-line sonic mood.

Some tracks include supplementary metadata: `energy:N/8` is a Mixed in Key score. [seed] marks \
tracks from the source playlist.

Respond ONLY with a JSON array:
[{"title": "...", "mood": "...", "track_ids": ["id1", "id2", ...]}]\
"""
```

#### 3c — Thread `seed_ids` through `stage1_concepts`

Add `seed_ids: frozenset[str] | None = None` to `stage1_concepts` and `_call_stage1_once`.
Pass through to `_tracks_to_text`. Pass `system=_STAGE1_SYSTEM_PLAYLIST` when `seed_ids` is set.

**Tests to add in `tests/test_llm.py`:**
- `test_tracks_to_text_emits_seed_annotation` — track in seed_ids gets `[seed]` suffix
- `test_tracks_to_text_no_seed_annotation_without_seed_ids` — no regression on existing behaviour

---

### Step 4 — Stage 2 playlist mode in `llm.py`

Stage 2 has its own inline track rendering loop (lines 592–623) that is separate from
`_tracks_to_text`. **The `[seed]` annotation must be added there too.** This is the critical
path: if seed_ids are not threaded into Stage 2 rendering, the model never sees which tracks
are from the source playlist.

#### 4a — Add `seed_ids` and `unplayed_ids` to Stage 2 rendering

Add `seed_ids: frozenset[str] | None = None`, `playlist_name: str | None = None`, and
`unplayed_ids: set[str] | None = None` to `stage2_curate_and_report`:

```python
async def stage2_curate_and_report(
    shortlists: list[MixConcept],
    tracks_by_id: dict[str, Track],
    stage2_provider: str | None = None,
    custom_genre_label: str | None = None,
    custom_genre_sub_genres: list[str] | None = None,
    playlist_name: str | None = None,
    seed_ids: frozenset[str] | None = None,
    unplayed_ids: set[str] | None = None,
) -> tuple[list[MixConcept], str]:
```

In the shortlist rendering loop, replace the current `t.play_count == 0` check with the same
precedence rule used in `build_playlist_pool`:

```python
# [seed] annotation
if seed_ids is not None and tid in seed_ids:
    extras.append("[seed]")

# unplayed annotation — catalog API result takes precedence over Rekordbox play_count
if unplayed_ids is not None:
    if tid in unplayed_ids:
        extras.append("unplayed")
else:
    if t.play_count == 0:
        extras.append("unplayed")
```

When `unplayed_ids=None` (non-playlist mode or no API configured), the fallback to
`t.play_count == 0` is identical to current behaviour — this change is backwards-compatible.

The prompt tells the model to prefer tracks "marked unplayed". Both annotations must reflect the
same source of truth, or the prompt instruction is misleading.

#### 4b — Playlist mode prompt prefix

When `playlist_name` is set, replace the standard prompt preamble with:

```
Curate and narrate a SINGLE mix concept from the following {n} candidate shortlists.
This is a playlist completion run seeded from the Rekordbox playlist "{playlist_name}".

Tracks marked [seed] come from that playlist — treat them as strong hints about the intended
direction, but do not force them into the concept if they weaken it. You may remove seed tracks
that undermine coherence, tempo flow, or harmonic fit. You may include non-seed tracks from the
shortlists when they improve the concept.

When two otherwise equally suitable tracks compete for a slot, prefer the one marked unplayed.
Played tracks are fully eligible when they are the stronger musical choice.

Produce EXACTLY ONE concept. Not 2, not 3 — one.

In the report, after the concept thesis paragraph, include a short section:
Source playlist: {playlist_name}
State how many seed tracks were retained, how many were dropped, and how many library tracks
were added. For any notable drop or addition, give one sentence of musical reasoning.
```

The standard prompt preamble (currently `"Curate and narrate a mix report from the following..."`)
is only written when `playlist_name is None`.

`_parse_curated_concepts` is unchanged — it already handles a 1-element array correctly.

#### 4c — Shortlist selection for playlist mode

**Do not use `select_shortlists_for_stage2` in playlist mode.** That function samples randomly
from the top pools by size, which is appropriate for exploratory genre runs but works against
deterministic playlist completion.

Add a new function:

```python
def select_shortlists_for_playlist_stage2(
    shortlists: list[MixConcept],
    seed_ids: frozenset[str],
) -> list[MixConcept]:
    """Return shortlists sorted by seed track count descending.

    All shortlists are passed (Stage 1 is instructed to produce 1–3, well
    within the _STAGE2_CAP). Capped at _STAGE2_CAP as a safety net in case
    Stage 1 ignores the instruction and over-produces.
    """
    ranked = sorted(
        shortlists,
        key=lambda s: sum(1 for tid in s.track_ids if tid in seed_ids),
        reverse=True,
    )
    return ranked[:_STAGE2_CAP]
```

Call this in `run_playlist_mode` instead of `select_shortlists_for_stage2`.

**Tests to add in `tests/test_llm.py`:**
- `test_stage2_rendering_includes_seed_annotation_for_seed_tracks`
- `test_stage2_rendering_no_seed_annotation_without_seed_ids`
- `test_stage2_rendering_unplayed_uses_unplayed_ids_when_provided` — track with `play_count=1` but in `unplayed_ids` is marked `unplayed`; track with `play_count=0` but absent from `unplayed_ids` is not
- `test_stage2_rendering_unplayed_falls_back_to_play_count_when_unplayed_ids_none` — `unplayed_ids=None` → existing `play_count == 0` behaviour
- `test_stage2_playlist_mode_prompt_contains_single_concept_instruction`
- `test_stage2_playlist_mode_prompt_contains_playlist_name`
- `test_stage2_playlist_mode_prompt_contains_seed_instruction`
- `test_select_shortlists_for_playlist_stage2_ranks_by_seed_count`
- `test_select_shortlists_for_playlist_stage2_caps_at_stage2_cap`

---

### Step 5 — `run_playlist_mode` in `__main__.py`

```python
async def run_playlist_mode(
    playlist_name: str,
    export_dir: Path | None,
    stage2_provider: str | None,
    all_tracks: bool,
) -> None:
```

**Flow:**

```
1.  parse_collection(_XML_PATH) → tracks
2.  apply_bpm_corrections(tracks) → tracks
3.  Fetch played tracks — same semantics as run():
    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    if all_tracks:
        unplayed_ids: set[str] | None = None  ← no weighting
    elif catalog_url:
        played = await fetch_played_tracks(api_key, catalog_url)
        unplayed = filter_unplayed(tracks, played)
        unplayed_ids = {t.track_id for t in unplayed}
    else:
        unplayed_ids = None  ← pool builder falls back to play_count == 0

4.  parse_playlists(_XML_PATH) → playlists
5.  raw_seed_ids = resolve_playlist(playlist_name, playlists)
    ← raises ValueError on not-found or ambiguous match
6.  tracks_by_id = _build_tracks_by_id(tracks)
7.  pool = build_playlist_pool(
        raw_seed_ids, tracks, tracks_by_id, unplayed_ids, all_tracks
    )
    ← raises ValueError if fewer than _MIN_PLAYLIST_TRACKS seed tracks survive
8.  seed_ids = frozenset(tid for tid in raw_seed_ids if tid in tracks_by_id)
9.  sorted_pool = sort_by_camelot(pool)
10. cascade_state = make_cascade_state()
11. shortlists = await stage1_concepts(sorted_pool, playlist_name, cascade_state, seed_ids=seed_ids)
12. shortlists = [s for s in shortlists if any(tid in tracks_by_id for tid in s.track_ids)]
13. shortlists = select_shortlists_for_playlist_stage2(shortlists, seed_ids)
    ← deterministic, ranked by seed coverage
14. if not shortlists:
        print("No shortlists generated.", file=sys.stderr); sys.exit(1)
15. all_concepts, report = await stage2_curate_and_report(
        shortlists, tracks_by_id, stage2_provider,
        playlist_name=playlist_name, seed_ids=seed_ids,
        unplayed_ids=unplayed_ids,
    )
16. if not all_concepts:
        print(report, file=sys.stderr); sys.exit(1)
17. concept = all_concepts[0]
18. elapsed = ...
19. report += f"\n\n---\n\nPlaylist completion: \"{playlist_name}\""
20. report += f"\n⏱ Generated in {elapsed_str}"
21. print(report)
22. raw_tracks_xml = parse_raw_tracks(_XML_PATH)
23. today = datetime.date.today().isoformat()
24. folder_name = f"Mix Lab - {playlist_name} - {today}"
25. merged_bytes = generate_merged_xml_bytes([concept], raw_tracks_xml, folder_name, None)
26. xml_attachments = [("rekordbox_export.xml", merged_bytes)] if merged_bytes else []
27. if export_dir:
        export_merged_xml([concept], raw_tracks_xml, export_dir / "rekordbox_export.xml", folder_name)
28. await send_report(report, [concept], [], tracks_by_id,
        counts={}, attachments=xml_attachments, show_unplayed=False)
```

**Error handling:**
- `ValueError` from `resolve_playlist` → print to stderr, `sys.exit(1)`
- `ValueError` from `build_playlist_pool` (too few tracks) → print to stderr, `sys.exit(1)`
- Existing error paths from Stage 1/2 are unchanged

**Tests to add in `tests/test_playlist_mode.py`:**
- `test_run_playlist_mode_happy_path` — coroutine with respx mocks, verifies 1 concept returned
- `test_run_playlist_mode_playlist_not_found_exits`
- `test_run_playlist_mode_ambiguous_playlist_name_exits`
- `test_run_playlist_mode_too_few_tracks_exits`
- `test_run_playlist_mode_no_catalog_api_uses_play_count_fallback`
- `test_run_playlist_mode_all_tracks_flag_neutral_weighting`

---

### Step 6 — CLI argument in `__main__.py`

Move `--genre` into a mutually exclusive group with `--playlist`:

```python
mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument(
    "--genre",
    type=str,
    default=None,
    metavar="LABEL",
    help="Genre to target. ...",
)
mode_group.add_argument(
    "--playlist",
    type=str,
    default=None,
    metavar="NAME",
    help=(
        "Rekordbox playlist name (or folder/name path) to use as seed for a single "
        "playlist-completion concept."
    ),
)
```

In `main()`, route before the existing genre path:

```python
if args.playlist:
    asyncio.run(run_playlist_mode(args.playlist, export_dir, args.stage2_provider, args.all_tracks))
    return
```

Update the epilog with playlist examples:

```
  mixlab --playlist "Monday Night"       complete a playlist concept from seed
  mixlab --playlist "Sets/Monday Night"  use full folder path if name is ambiguous
```

---

## Environment Variables

No new environment variables. Playlist mode uses all existing env vars unchanged:
- `CHANGSTA_API_KEY` / `CATALOG_API_URL` — optional; used for unplayed weighting
- `ANTHROPIC_API_KEY` — Stage 2
- `GROQ_API_KEY`, `GEMINI_API_KEY`, etc. — Stage 1 cascade
- `DISCORD_BOT_TOKEN` etc. — delivery

---

## File Change Summary

| File | Change type | Notes |
|---|---|---|
| `src/mixlab/reader.py` | Add function | `parse_playlists` — full-path keyed dict |
| `src/mixlab/playlist_mode.py` | New file | Resolver + pool builder |
| `src/mixlab/llm.py` | Modify | `_STAGE1_SYSTEM_PLAYLIST`, `seed_ids` + `playlist_name` params on `stage2_curate_and_report`, `[seed]` in Stage 2 rendering loop, `select_shortlists_for_playlist_stage2` |
| `src/mixlab/__main__.py` | Modify | `run_playlist_mode`, `--playlist` arg, routing |
| `tests/test_reader.py` | Extend | Playlist parsing tests |
| `tests/test_llm.py` | Extend | Seed annotation in both rendering paths + prompt content + shortlist selection |
| `tests/test_playlist_mode.py` | New file | Resolver, pool, and integration tests |

No other files need to change. Existing tests must pass without modification.

---

## Quality Gates

Before marking any step complete:
```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest
```

All four commands must pass clean.
