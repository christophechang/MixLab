# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 11 issues identified in the April 2026 code review: 5 Important and 6 Minor.

**Architecture:** Fixes are isolated to individual modules; no new dependencies, no new files except test additions. Tasks ordered so that later tasks can rely on symbols renamed in earlier tasks.

**Tech Stack:** Python 3.12, Pydantic, httpx, pytest, respx, mypy --strict, ruff.

---

## Files Modified

- `src/mixlab/clustering.py` — rename `_camelot_compatible` → `camelot_compatible`; fix `sort_by_camelot` O(n²)
- `src/mixlab/llm.py` — rename `_MIN_SHORTLIST_TRACKS` / `_MAX_STAGE1_POOL_CUSTOM` to public; remove duplicate `_shortfall_warning`; fix prompt fragility; add temperature param
- `src/mixlab/discord_client.py` — remove duplicate `_shortfall_warning`; add rate-limit escape hatch
- `src/mixlab/models.py` — `enrichment_confidence` → `Literal[...]`
- `src/mixlab/matcher.py` — O(n+m) set-based matching
- `src/mixlab/playlist_mode.py` — update import after rename
- `src/mixlab/__main__.py` — update imports after rename
- `src/mixlab/config.py` — add shared `shortfall_warning` function
- `src/mixlab/reporter.py` — delete
- `.env.example` — remove `OPENROUTER_API_KEY`
- `tests/test_intent.py` — add `stage0_intent_brief` integration test
- `tests/test_discord_client.py` — add rate-limit guard test
- `tests/test_matcher.py` — verify O(n+m) path
- `tests/test_reporter.py` — delete

---

## Task 1: Rename `_camelot_compatible` to public in `clustering.py`

**Files:**
- Modify: `src/mixlab/clustering.py:20`
- Modify: `src/mixlab/playlist_mode.py:7`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Verify existing test imports pass**

```bash
cd /path/to/MixLab && .venv/bin/python -m pytest tests/test_clustering.py -v
```
Expected: all pass.

- [ ] **Step 2: Rename `_camelot_compatible` to `camelot_compatible` in `clustering.py`**

In `src/mixlab/clustering.py`, change line 20:
```python
def camelot_compatible(a: str, b: str) -> bool:
```
Also update the call at line 80 inside `sort_by_camelot`:
```python
compatible = [t for t in remaining if camelot_compatible(last_key, t.camelot_key)]
```

- [ ] **Step 3: Update `playlist_mode.py` import**

In `src/mixlab/playlist_mode.py`, change line 7:
```python
from mixlab.clustering import (
    camelot_compatible,
    build_custom_genre_pool,
)
```
Remove the `# noqa: PLC2701` comment.

Also update all calls to `_camelot_compatible` inside `playlist_mode.py` to use `camelot_compatible`.

- [ ] **Step 4: Run tests and mypy**

```bash
.venv/bin/python -m pytest tests/test_clustering.py tests/test_playlist_mode.py -v
.venv/bin/python -m mypy src/mixlab/clustering.py src/mixlab/playlist_mode.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/clustering.py src/mixlab/playlist_mode.py
git commit -m "refactor(clustering): promote _camelot_compatible to public API"
```

---

## Task 2: Rename private constants in `llm.py` to public

**Files:**
- Modify: `src/mixlab/llm.py:34-35`
- Modify: `src/mixlab/__main__.py:27-28`

- [ ] **Step 1: Rename constants in `llm.py`**

In `src/mixlab/llm.py`, change lines 34–35:
```python
MAX_STAGE1_POOL_CUSTOM = 120  # random window size for custom pools (2 chunks × 60 = 2 API calls)
MIN_SHORTLIST_TRACKS = 8  # Stage 1: minimum candidates per pool
```
Update all internal references in `llm.py` that used `_MIN_SHORTLIST_TRACKS` or `_MAX_STAGE1_POOL_CUSTOM`:
```bash
grep -n "_MIN_SHORTLIST_TRACKS\|_MAX_STAGE1_POOL_CUSTOM" src/mixlab/llm.py
```
Replace each occurrence with the new unprefixed names.

- [ ] **Step 2: Update `__main__.py` imports**

In `src/mixlab/__main__.py`, change lines 27–28:
```python
from mixlab.llm import (
    MIN_SHORTLIST_TRACKS,
    make_cascade_state,
    select_shortlists_for_stage2,
    select_stage1_window,
    stage0_intent_brief,
    stage1_concepts,
    stage2_curate_and_report,
)
```
Also remove the inline `from mixlab.llm import _MAX_STAGE1_POOL_CUSTOM` at line ~500:
```python
        from mixlab.llm import MAX_STAGE1_POOL_CUSTOM
```
Wait — check: the `_MAX_STAGE1_POOL_CUSTOM` is imported via a local `from mixlab.llm import _MAX_STAGE1_POOL_CUSTOM` inside the `if is_custom:` block. Change it to use the top-level import instead:
```python
        pool = build_custom_genre_pool(genre, unplayed, CUSTOM_GENRES, GENRE_MAP)
        if not pool:
            ...
        stage1_pool = select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)
```
Add `MAX_STAGE1_POOL_CUSTOM` to the top-level import from `mixlab.llm`, and delete the inline import inside the `if is_custom:` block.

And replace the reference `_MIN_SHORTLIST_TRACKS` at line 535 with `MIN_SHORTLIST_TRACKS`.

- [ ] **Step 3: Run tests and mypy**

```bash
.venv/bin/python -m pytest tests/test_llm.py tests/test_main.py -v
.venv/bin/python -m mypy src/mixlab/llm.py src/mixlab/__main__.py
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/mixlab/llm.py src/mixlab/__main__.py
git commit -m "refactor(llm): promote _MIN_SHORTLIST_TRACKS and _MAX_STAGE1_POOL_CUSTOM to public"
```

---

## Task 3: Extract shared `shortfall_warning` from `discord_client.py` and `llm.py`

Both `discord_client.py:19` and `llm.py:845` define an identical `_shortfall_warning` function. Extract to `config.py` (which already owns `TRACK_COUNT_TARGETS`).

**Files:**
- Modify: `src/mixlab/config.py` — add `shortfall_warning`
- Modify: `src/mixlab/discord_client.py` — remove local def, import from config
- Modify: `src/mixlab/llm.py` — remove local def, import from config
- Test: `tests/test_discord_client.py`

- [ ] **Step 1: Write a test for shortfall warning to anchor the behavior**

In `tests/test_discord_client.py`, add:
```python
from mixlab.config import shortfall_warning
from mixlab.models import MixConcept


def test_shortfall_warning_returns_none_when_shortfall_small() -> None:
    concept = MixConcept(title="T", mood="m", track_ids=[str(i) for i in range(20)])
    assert shortfall_warning(concept, "house") is None


def test_shortfall_warning_returns_message_when_shortfall_large() -> None:
    # house minimum from TRACK_COUNT_TARGETS — check config.py for actual value
    concept = MixConcept(title="T", mood="m", track_ids=["1", "2"])
    result = shortfall_warning(concept, "house")
    assert result is not None
    assert "2 tracks found" in result
    assert "more to fill" in result
```

Run to confirm failure:
```bash
.venv/bin/python -m pytest tests/test_discord_client.py::test_shortfall_warning_returns_none_when_shortfall_small -v
```
Expected: ImportError or AttributeError — `shortfall_warning` not yet in `config`.

- [ ] **Step 2: Add `shortfall_warning` to `config.py`**

In `src/mixlab/config.py`, add at the bottom (after `TRACK_COUNT_TARGETS` is defined):
```python
from __future__ import annotations  # already present

# Add this import if not present at top:
# from mixlab.models import MixConcept  ← avoid circular import; use TYPE_CHECKING

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mixlab.models import MixConcept

_SHORTFALL_THRESHOLD = 4


def shortfall_warning(concept: "MixConcept", genre: str) -> str | None:
    min_count, _ = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])
    n = len(concept.track_ids)
    shortfall = min_count - n
    if shortfall > _SHORTFALL_THRESHOLD:
        return f"⚠️ {n} tracks found — needs {shortfall} more to fill a set. Crate dig to complete."
    return None
```

**Important:** `config.py` must not import from `models.py` at runtime to avoid circular imports. Use `TYPE_CHECKING` guard as shown.

- [ ] **Step 3: Run new tests**

```bash
.venv/bin/python -m pytest tests/test_discord_client.py::test_shortfall_warning_returns_none_when_shortfall_small tests/test_discord_client.py::test_shortfall_warning_returns_message_when_shortfall_large -v
```
Expected: PASS.

- [ ] **Step 4: Remove `_shortfall_warning` from `discord_client.py`, import from `config`**

In `src/mixlab/discord_client.py`:
- Delete lines 17–25 (`_SHORTFALL_THRESHOLD` constant + `_shortfall_warning` function)
- Add `shortfall_warning` to the import from `mixlab.config`:
  ```python
  from mixlab.config import TRACK_COUNT_TARGETS, shortfall_warning
  ```
- Change the call at line 90 from `_shortfall_warning(concept, genre)` to `shortfall_warning(concept, genre)`

- [ ] **Step 5: Remove `_shortfall_warning` from `llm.py`, import from `config`**

In `src/mixlab/llm.py`:
- Delete lines 841–851 (`_SHORTFALL_THRESHOLD` constant + `_shortfall_warning` function)
- Add `shortfall_warning` to the import from `mixlab.config`:
  ```python
  from mixlab.config import TRACK_COUNT_TARGETS, shortfall_warning
  ```
- Change the call at line ~1565 from `_shortfall_warning(concept, genre)` to `shortfall_warning(concept, genre)`

- [ ] **Step 6: Run full test suite and mypy**

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m mypy src/mixlab/config.py src/mixlab/discord_client.py src/mixlab/llm.py
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/mixlab/config.py src/mixlab/discord_client.py src/mixlab/llm.py tests/test_discord_client.py
git commit -m "refactor: extract shortfall_warning to config.py, remove duplicate"
```

---

## Task 4: Add rate-limit escape hatch to Discord client

`_post_raw` and `_post_with_files` loop indefinitely on HTTP 429. Add a `_MAX_RATE_LIMIT_RETRIES = 5` guard.

**Files:**
- Modify: `src/mixlab/discord_client.py`
- Test: `tests/test_discord_client.py`

- [ ] **Step 1: Write failing test**

In `tests/test_discord_client.py`, add:
```python
import pytest
import respx
import httpx
from mixlab.discord_client import DiscordClient


@pytest.mark.anyio
async def test_post_raw_raises_after_max_retries_on_429() -> None:
    client = DiscordClient(bot_token="tok", guild_id="g", channel_id="c123", channel_name="mix-lab")
    with respx.mock:
        respx.post("https://discord.com/api/v10/channels/c123/messages").mock(
            return_value=httpx.Response(429, json={"retry_after": 0.01})
        )
        async with httpx.AsyncClient(timeout=5) as http:
            with pytest.raises(RuntimeError, match="rate limit"):
                await client._post_raw(http, "c123", "hello")
```

Run to confirm failure:
```bash
.venv/bin/python -m pytest tests/test_discord_client.py::test_post_raw_raises_after_max_retries_on_429 -v
```
Expected: FAIL — no `RuntimeError` raised yet.

- [ ] **Step 2: Add `_MAX_RATE_LIMIT_RETRIES` constant and guard to `discord_client.py`**

In `src/mixlab/discord_client.py`, add at module level (after existing constants):
```python
_MAX_RATE_LIMIT_RETRIES = 5
```

Change `_post_raw`:
```python
async def _post_raw(self, client: httpx.AsyncClient, channel_id: str, content: str) -> None:
    url = f"{_DISCORD_API}/channels/{channel_id}/messages"
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        resp = await client.post(url, headers=self._headers, json={"content": content})
        if resp.status_code == 429:
            retry_after = float(resp.json().get("retry_after", 1.0))
            await asyncio.sleep(retry_after + _RATE_LIMIT_BUFFER)
            continue
        resp.raise_for_status()
        return
    raise RuntimeError(f"Discord rate limit: channel {channel_id} still 429 after {_MAX_RATE_LIMIT_RETRIES} retries")
```

Change `_post_with_files` identically:
```python
async def _post_with_files(
    self,
    client: httpx.AsyncClient,
    channel_id: str,
    content: str,
    attachments: list[tuple[str, bytes]],
) -> None:
    url = f"{_DISCORD_API}/channels/{channel_id}/messages"
    headers = {k: v for k, v in self._headers.items() if k.lower() != "content-type"}
    files = {f"files[{i}]": (name, data, "application/xml") for i, (name, data) in enumerate(attachments)}
    data = {"payload_json": json.dumps({"content": content})}
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        resp = await client.post(url, headers=headers, data=data, files=files)
        if resp.status_code == 429:
            retry_after = float(resp.json().get("retry_after", 1.0))
            await asyncio.sleep(retry_after + _RATE_LIMIT_BUFFER)
            continue
        resp.raise_for_status()
        return
    raise RuntimeError(f"Discord rate limit: channel {channel_id} still 429 after {_MAX_RATE_LIMIT_RETRIES} retries")
```

- [ ] **Step 3: Run new test**

```bash
.venv/bin/python -m pytest tests/test_discord_client.py::test_post_raw_raises_after_max_retries_on_429 -v
```
Expected: PASS.

Note: The `post()` method already has a broad `except Exception` that catches this `RuntimeError` and returns `False`, so Discord delivery failure is non-fatal to the pipeline. No changes needed there.

- [ ] **Step 4: Run full discord tests**

```bash
.venv/bin/python -m pytest tests/test_discord_client.py -v
.venv/bin/python -m mypy src/mixlab/discord_client.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/discord_client.py tests/test_discord_client.py
git commit -m "fix(discord): add max-retry guard to 429 rate-limit loop"
```

---

## Task 5: Fix O(n*m) matching in `matcher.py`

**Files:**
- Modify: `src/mixlab/matcher.py`
- Test: `tests/test_matcher.py`

- [ ] **Step 1: Read current tests**

```bash
.venv/bin/python -m pytest tests/test_matcher.py -v
```
Confirm all pass. Note what's tested — `is_played` and `filter_unplayed`.

- [ ] **Step 2: Refactor `filter_unplayed` to pre-build set**

Replace the content of `src/mixlab/matcher.py` with:
```python
from __future__ import annotations

import re
import unicodedata

from mixlab.models import PlayedTrack, Track

_FEAT_RE = re.compile(
    r"\s*[\(\[]\s*(?:feat\.|ft\.|featuring)[^\)\]]*[\)\]]"
    r"|\s+(?:feat\.|ft\.|featuring)\s+\S+(?:\s+\S+)*",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s-]")
_DASH_RE = re.compile(r"[‐-―−﹘﹣－]")


def normalise(text: str) -> str:
    text = text.lower()
    text = _DASH_RE.sub("-", text)
    text = unicodedata.normalize("NFKD", text)
    text = _FEAT_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    text = " ".join(text.split())
    return text


def _played_key(p: PlayedTrack) -> str:
    return normalise(p.artist) + " " + normalise(p.title)


def is_played(track: Track, played: list[PlayedTrack]) -> bool:
    needle = normalise(track.artist) + " " + normalise(track.title)
    return any(_played_key(p) == needle for p in played)


def filter_unplayed(tracks: list[Track], played: list[PlayedTrack]) -> list[Track]:
    played_set = {_played_key(p) for p in played}
    return [t for t in tracks if normalise(t.artist) + " " + normalise(t.title) not in played_set]
```

The `is_played` function signature is unchanged (public API, still works for single-track checks). Only `filter_unplayed` gets the O(n+m) path, which is the hot path called on the full catalog.

- [ ] **Step 3: Run matcher tests**

```bash
.venv/bin/python -m pytest tests/test_matcher.py -v
.venv/bin/python -m mypy src/mixlab/matcher.py
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/mixlab/matcher.py
git commit -m "perf(matcher): pre-build played set in filter_unplayed for O(n+m) matching"
```

---

## Task 6: Fix `enrichment_confidence` type annotation in `models.py`

**Files:**
- Modify: `src/mixlab/models.py:24`

- [ ] **Step 1: Change `enrichment_confidence` to Literal**

In `src/mixlab/models.py`, replace line 24:
```python
enrichment_confidence: Literal["high", "medium", "low", ""] = ""
```

The `Literal` import is already present in the file.

- [ ] **Step 2: Run mypy and tests**

```bash
.venv/bin/python -m mypy src/mixlab/models.py
.venv/bin/python -m pytest -v -x
```
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/mixlab/models.py
git commit -m "fix(models): use Literal type for enrichment_confidence"
```

---

## Task 7: Fix `sort_by_camelot` O(n²) in `clustering.py`

`remaining.remove(next_track)` is O(n) per step. Replace with index-based removal.

**Files:**
- Modify: `src/mixlab/clustering.py:70-85`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Run existing sort test**

```bash
.venv/bin/python -m pytest tests/test_clustering.py -k "camelot" -v
```
Expected: PASS — these become our regression guard.

- [ ] **Step 2: Replace `sort_by_camelot` implementation**

In `src/mixlab/clustering.py`, replace the `sort_by_camelot` function body:
```python
def sort_by_camelot(tracks: list[Track]) -> list[Track]:
    if not tracks:
        return []

    remaining = list(tracks)
    sorted_tracks: list[Track] = [remaining.pop(0)]

    while remaining:
        last_key = sorted_tracks[-1].camelot_key
        compatible_idx = [i for i, t in enumerate(remaining) if camelot_compatible(last_key, t.camelot_key)]
        if compatible_idx:
            best_idx = min(compatible_idx, key=lambda i: remaining[i].bpm)
        else:
            best_idx = min(range(len(remaining)), key=lambda i: remaining[i].bpm)
        sorted_tracks.append(remaining.pop(best_idx))

    return sorted_tracks
```

`list.pop(index)` is O(n) in the worst case too, but it avoids the linear scan that `.remove()` adds on top of the `min()` scan — the structure is now a single pass per step instead of two. For the input sizes used (≤180 tracks), this is the correct practical fix.

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/test_clustering.py -v
.venv/bin/python -m mypy src/mixlab/clustering.py
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/mixlab/clustering.py
git commit -m "perf(clustering): remove redundant linear scan in sort_by_camelot"
```

---

## Task 8: Fix Stage 2 prompt fragility in `llm.py`

`_STAGE2_SYSTEM_PLAYLIST` is constructed via chained `.replace()` on `_STAGE2_SYSTEM`. Add runtime assertions so that any future change to the base string that breaks the replacement fails loudly at import time.

**Files:**
- Modify: `src/mixlab/llm.py:748-791`

- [ ] **Step 1: Extract the two variable sections as named constants before the replacement**

In `src/mixlab/llm.py`, immediately before the `_STAGE2_SYSTEM_PLAYLIST = ...` block (~line 748), add:

```python
# Variable sections that differ between standard and playlist Stage 2 prompts.
# These constants exist so that the .replace() calls below have a named reference
# to assert against — if either constant drifts from _STAGE2_SYSTEM, the assert
# fires at import time rather than silently producing a wrong prompt.
_STAGE2_SELECT_STANDARD = """For each shortlist or sub-pool you carve from it:
- SELECT the best 8–12 tracks from the pool for a coherent DJ set. Exclude tracks that weaken the journey. \
Weakness is practical: a track whose intro gives no workable mix point, a vocal that starts on bar one with \
no room to bring it in, a bass-heavy record dropped after another with no frequency relief, a big moment \
used so early it makes everything after feel like a comedown."""

_STAGE2_PRODUCE_STANDARD = """Produce as many concepts as the pool genuinely supports — between 3 and 6. If the pool only yields 4 \
strong concepts, produce 4. Do not pad with weak concepts to hit a number. If the pool is too thin to \
produce even 3 distinct, coherent sets, return a single-element array with a diagnostic object instead \
of concept objects: [{"diagnostic": "...explanation of why the pool is insufficient..."}] — this is \
useful information, not a failure."""
```

Then change the `_STAGE2_SYSTEM_PLAYLIST` assignment to use these constants:
```python
_tmp = _STAGE2_SYSTEM.replace(
    _STAGE2_SELECT_STANDARD,
    """For playlist-completion runs:
- Prefer seed tracks over library tracks when both serve the arc equally — but arc quality and narrative \
tightness take priority over seed count. Cut a seed that weakens the concept; keep a library track that \
strengthens it. The Python layer enforces the minimum seed floor independently.
- SELECT a coherent final tracklist from the pool. Target 10–14 tracks. Do not exceed 14. A great concept \
is defined as much by what you cut as what you keep — every track must earn its place in the arc. Exclude \
tracks that weaken the journey. Weakness is practical: a track whose intro gives no workable mix point, a \
vocal that starts on bar one with no room to bring it in, a bass-heavy record dropped after another with no \
frequency relief, a big moment used so early it makes everything after feel like a comedown.""",
)
assert _tmp != _STAGE2_SYSTEM, (
    "Stage 2 playlist prompt: select section not found in base — _STAGE2_SELECT_STANDARD drifted from _STAGE2_SYSTEM"
)
_STAGE2_SYSTEM_PLAYLIST = _tmp.replace(
    _STAGE2_PRODUCE_STANDARD,
    """Produce EXACTLY THREE concepts from the same pool using these strategies:

1. "practical" (mood = "practical"): maximise harmonic continuity. Prefer adjacent Camelot keys \
(distance ≤ 1). BPM moves ≤ 2 BPM per step where possible. Preserve all strong seed adjacency \
pairs. Protect anchors. Avoid unearned key jumps.

2. "balanced" (mood = "balanced"): one major key jump (distance 2–3) or single BPM arc allowed. \
Anchors protected. Optional seeds may be swapped when a library track clearly serves the arc \
better. Adjacency pairs are hints, not constraints.

3. "adventurous" (mood = "adventurous"): prioritise set narrative and role completeness. Chapter \
pivots and peak impacts are permitted when musically justified — name the mechanism. Anchors \
protected; adjacency pairs may be broken with one-sentence reason. Optional and supporting seeds \
replaceable if a library track serves the arc materially better.

Label each concept's mood field with exactly "practical", "balanced", or "adventurous".
When multiple credible openers exist, do not default to the same obviously ambient opener in every concept. \
Practical may open with the cleanest restrained groove, balanced may open warmer or more melodic, and \
adventurous may open stranger or more spacious. Reuse the same opener across all three only if it is clearly \
superior for blendability and intent.
All three must meet the anchor protection rules and the seed retention floor.
If the pool is too thin to produce even one strong set: [{"diagnostic": "..."}].""",
)
assert _STAGE2_SYSTEM_PLAYLIST != _tmp, (
    "Stage 2 playlist prompt: produce section not found in base — _STAGE2_PRODUCE_STANDARD drifted from _STAGE2_SYSTEM"
)
```

Also remove the old `_STAGE2_SYSTEM_PLAYLIST = _STAGE2_SYSTEM.replace(...).replace(...)` block that this replaces.

- [ ] **Step 2: Run tests and mypy**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v
.venv/bin/python -m mypy src/mixlab/llm.py
```
Expected: all pass. The assert fires at import time, so any test that imports `llm` would catch a broken prompt.

- [ ] **Step 3: Commit**

```bash
git add src/mixlab/llm.py
git commit -m "fix(llm): assert Stage 2 playlist prompt replacements succeed at import time"
```

---

## Task 9: Lower temperature for Stage 2 selection pass

`_call_anthropic_http` hardcodes `temperature=0.7`. The selection pass benefits from lower temperature (0.3) for deterministic variant labels. Add a `temperature` parameter with a default of 0.7 for backward-compat; callers that need lower can pass it explicitly.

**Files:**
- Modify: `src/mixlab/llm.py:402-426`

- [ ] **Step 1: Add `temperature` parameter to `_call_anthropic_http`**

```python
async def _call_anthropic_http(
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 2048,
    timeout: int = 90,
    temperature: float = 0.7,
) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "output-128k-2025-02-19",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["content"][0]["text"])
```

- [ ] **Step 2: Pass `temperature=0.3` in `_call_stage2_raw`**

`_call_stage2_raw` is the selection pass. Change it:
```python
async def _call_stage2_raw(
    prompt: str,
    stage2_system: str,
    stage2_key: str,
    max_tokens: int = 32768,
) -> str:
    try:
        return await _call_anthropic_http(
            stage2_key,
            "claude-sonnet-4-6",
            stage2_system,
            prompt,
            max_tokens=max_tokens,
            timeout=600,
            temperature=0.3,
        )
    except Exception as exc:
        raise RuntimeError(f"Stage 2 curation failed: {exc}") from exc
```

Report generation (`_call_stage2_report_single`) keeps the default `temperature=0.7`.

- [ ] **Step 3: Run tests and mypy**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v
.venv/bin/python -m mypy src/mixlab/llm.py
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/mixlab/llm.py
git commit -m "fix(llm): lower Stage 2 selection pass temperature to 0.3 for variant label stability"
```

---

## Task 10: Delete `reporter.py` dead code

**Files:**
- Delete: `src/mixlab/reporter.py`
- Delete: `tests/test_reporter.py`

- [ ] **Step 1: Verify nothing imports `reporter`**

```bash
grep -r "from mixlab.reporter\|import reporter" src/ tests/
```
Expected: no output (nothing imports it).

- [ ] **Step 2: Delete both files**

```bash
rm src/mixlab/reporter.py tests/test_reporter.py
```

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest -v
```
Expected: all pass, test count reduced by however many were in `test_reporter.py`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove reporter.py dead code stub and its test file"
```

---

## Task 11: Remove dead `OPENROUTER_API_KEY` from `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Verify no code reads `OPENROUTER_API_KEY`**

```bash
grep -r "OPENROUTER_API_KEY" src/ tests/
```
Expected: no output.

- [ ] **Step 2: Remove line from `.env.example`**

Delete the `OPENROUTER_API_KEY=` line.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: remove unused OPENROUTER_API_KEY from .env.example"
```

---

## Task 12: Add `stage0_intent_brief` integration test

The orchestrating function `stage0_intent_brief` has no integration test. Cover: (a) successful LLM path with merge, (b) short-seed early-return path, (c) all-providers-fail fallback.

**Files:**
- Modify: `tests/test_intent.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_intent.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from mixlab.llm import stage0_intent_brief, make_cascade_state
from mixlab.models import Track


def _make_track(track_id: str, *, bpm: float = 124.0, camelot_key: str = "8A") -> Track:
    return Track(
        track_id=track_id,
        artist=f"Artist {track_id}",
        title=f"Title {track_id}",
        bpm=bpm,
        camelot_key=camelot_key,
        genre="House",
    )


_VALID_LLM_RESPONSE = """{
  "overall_vibe": "Deep hypnotic groove",
  "is_coherent_set": true,
  "risk_tolerance": "medium",
  "missing_roles": ["closer"],
  "seed_analyses": [
    {"track_id": "1", "tier": "anchor", "inferred_role": "peak"},
    {"track_id": "2", "tier": "supporting", "inferred_role": "builder"},
    {"track_id": "3", "tier": "optional", "inferred_role": "opener"},
    {"track_id": "4", "tier": "anchor", "inferred_role": "pivot"},
    {"track_id": "5", "tier": "supporting", "inferred_role": "builder"},
    {"track_id": "6", "tier": "optional", "inferred_role": "cleanser"}
  ]
}"""


@pytest.mark.anyio
async def test_stage0_returns_deterministic_brief_for_short_seed() -> None:
    """Seeds ≤5 tracks skip LLM and return deterministic brief."""
    seed_tracks = [_make_track(str(i)) for i in range(1, 4)]
    seed_ids = [t.track_id for t in seed_tracks]
    state = make_cascade_state()
    with patch("mixlab.llm._try_groq") as mock_groq:
        brief = await stage0_intent_brief(seed_tracks, seed_ids, state, bpm_range=(124.0, 124.0))
    mock_groq.assert_not_called()
    # Deterministic brief populates all seeds as supporting by default
    assert all(tid in brief.supporting_ids or tid in brief.anchor_ids for tid in seed_ids)


@pytest.mark.anyio
async def test_stage0_merges_llm_tiers_with_deterministic_shape() -> None:
    """LLM response tiers are used; deterministic energy_shape and strong_adjacencies override LLM values."""
    seed_tracks = [_make_track(str(i), bpm=124.0 + i * 0.5) for i in range(1, 7)]
    seed_ids = [t.track_id for t in seed_tracks]
    state = make_cascade_state()
    with patch("mixlab.llm._try_groq", new_callable=AsyncMock, return_value=_VALID_LLM_RESPONSE):
        brief = await stage0_intent_brief(seed_tracks, seed_ids, state, bpm_range=(124.0, 126.5))
    # LLM tiers applied
    assert "1" in brief.anchor_ids
    assert "2" in brief.supporting_ids
    assert brief.overall_vibe == "Deep hypnotic groove"
    assert brief.risk_tolerance == "medium"
    # Deterministic fields override LLM
    assert brief.energy_shape in ("single_arc", "double_peak", "plateau", "flat", "unclear")
    # bpm_range comes from deterministic pass
    assert brief.bpm_range[0] <= brief.bpm_range[1]


@pytest.mark.anyio
async def test_stage0_falls_back_to_deterministic_when_all_providers_fail() -> None:
    """If every cascade provider fails or returns None, return deterministic brief without raising."""
    seed_tracks = [_make_track(str(i)) for i in range(1, 7)]
    seed_ids = [t.track_id for t in seed_tracks]
    state = make_cascade_state()
    with (
        patch("mixlab.llm._try_groq", new_callable=AsyncMock, return_value=None),
        patch("mixlab.llm._try_gemini", new_callable=AsyncMock, return_value=None),
        patch("mixlab.llm._try_mistral", new_callable=AsyncMock, return_value=None),
    ):
        brief = await stage0_intent_brief(seed_tracks, seed_ids, state, bpm_range=(124.0, 124.0))
    # Should not raise — returns deterministic fallback
    assert brief is not None
    assert all(tid in brief.supporting_ids or tid in brief.anchor_ids for tid in seed_ids)
```

Run to confirm failure (function exists but tests need to work with anyio):
```bash
.venv/bin/python -m pytest tests/test_intent.py::test_stage0_returns_deterministic_brief_for_short_seed -v
```
Expected: PASS already for the short-seed test (that path is simple). Others may fail — fix `anyio` marker if needed.

Check pytest config:
```bash
grep -n "anyio\|asyncio" pyproject.toml
```
If `anyio_mode` is not configured, add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
asyncio_mode = "auto"
```
or use `pytest.mark.asyncio` instead of `pytest.mark.anyio`.

- [ ] **Step 2: Verify all three tests pass**

```bash
.venv/bin/python -m pytest tests/test_intent.py -v
```
Expected: all pass.

- [ ] **Step 3: Run full suite**

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m mypy src/
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_intent.py
git commit -m "test(llm): add stage0_intent_brief integration tests for all code paths"
```

---

## Final Verification

- [ ] **Run full suite**

```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest -v
```
Expected: no ruff errors, no mypy errors, all tests pass.
