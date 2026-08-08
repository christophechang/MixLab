# Stage 2 Two-Pass Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Stage 2 from one giant LLM call (selection + reports combined) into two passes: selection-only JSON first, then parallel prose report generation per concept.

**Architecture:** Pass 1 asks Anthropic to select and order tracks and return compact JSON (~8K tokens output max). Pass 2 fires one Anthropic call per curated concept in parallel to generate prose reports (~2K tokens each). All scoring, variant selection, and playlist retention logic runs between the two passes on the compact JSON. Also fixes track count cap (≤12) and removes temporary debug logging added during diagnosis.

**Tech Stack:** Python 3.12, httpx, asyncio.gather, respx (tests)

---

### Task 1: Fix playlist mode track count cap

**Files:**
- Modify: `src/mixlab/llm.py:775`

- [ ] **Step 1: Write the failing test**

In `tests/test_llm.py`, add after the existing `_parse_curated_concepts` tests:

```python
def test_stage2_playlist_system_caps_tracks_at_twelve() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_PLAYLIST

    assert "12–18 tracks" not in _STAGE2_SYSTEM_PLAYLIST
    assert "10–12 tracks" in _STAGE2_SYSTEM_PLAYLIST
```

- [ ] **Step 2: Run to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_llm.py::test_stage2_playlist_system_caps_tracks_at_twelve -v
```

Expected: FAIL — `assert "10–12 tracks" in _STAGE2_SYSTEM_PLAYLIST`

- [ ] **Step 3: Apply the fix**

In `src/mixlab/llm.py`, in the `_STAGE2_SYSTEM_PLAYLIST` definition (the `.replace()` block), change the track count line:

```python
# OLD (line ~775):
"- SELECT a coherent final tracklist from the pool. Prefer roughly 12–18 tracks when the material supports it, and allow a longer list when needed to preserve a strong seed-led arc. Exclude only tracks that genuinely weaken the journey. Weakness is practical: a track whose intro gives no workable mix point, a vocal that starts on bar one with no room to bring it in, a bass-heavy record dropped after another with no frequency relief, a big moment used so early it makes everything after feel like a comedown.",

# NEW:
"- SELECT a coherent final tracklist from the pool. Prefer 10–12 tracks. Do not exceed 12 tracks unless dropping a 13th would break an anchor adjacency pair — in that case, include it and note the exception. Exclude tracks that genuinely weaken the journey. Weakness is practical: a track whose intro gives no workable mix point, a vocal that starts on bar one with no room to bring it in, a bass-heavy record dropped after another with no frequency relief, a big moment used so early it makes everything after feel like a comedown.",
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
.venv/bin/python -m pytest tests/test_llm.py::test_stage2_playlist_system_caps_tracks_at_twelve -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/llm.py tests/test_llm.py
git commit -m "fix(llm): cap playlist mode track count at 12"
```

---

### Task 2: Add selection system prompts and report system prompt

**Files:**
- Modify: `src/mixlab/llm.py` — add after `_STAGE2_SYSTEM_PLAYLIST` block (before `_SHORTFALL_THRESHOLD`)

- [ ] **Step 1: Write the failing tests**

In `tests/test_llm.py`:

```python
def test_make_selection_system_removes_report_schema_field() -> None:
    from mixlab.llm import _make_selection_system, _STAGE2_SYSTEM

    result = _make_selection_system(_STAGE2_SYSTEM)
    assert '"report":' not in result
    assert '"track_ids":' in result
    assert '"transitions":' in result
    assert "Respond ONLY with the JSON array." in result


def test_make_selection_system_removes_report_format_instructions() -> None:
    from mixlab.llm import _make_selection_system, _STAGE2_SYSTEM

    result = _make_selection_system(_STAGE2_SYSTEM)
    assert "The \"report\" value must be a single string" not in result
    assert "Role options: opener" not in result  # part of the report-format section


def test_make_selection_system_preserves_curation_instructions() -> None:
    from mixlab.llm import _make_selection_system, _STAGE2_SYSTEM

    result = _make_selection_system(_STAGE2_SYSTEM)
    # Key curation guidance must survive
    assert "peak weapons" in result
    assert "name_reason" in result
    assert "chapter_pivot" in result


def test_selection_system_playlist_variant_has_practical_balanced_adventurous() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_PLAYLIST_SELECTION

    assert '"practical"' in _STAGE2_SYSTEM_PLAYLIST_SELECTION
    assert '"balanced"' in _STAGE2_SYSTEM_PLAYLIST_SELECTION
    assert '"adventurous"' in _STAGE2_SYSTEM_PLAYLIST_SELECTION
    assert '"report":' not in _STAGE2_SYSTEM_PLAYLIST_SELECTION
```

- [ ] **Step 2: Run to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "selection_system" -v
```

Expected: all FAIL with ImportError on `_make_selection_system`

- [ ] **Step 3: Add `_make_selection_system`, derived constants, and `_STAGE2_REPORT_SYSTEM`**

In `src/mixlab/llm.py`, immediately after the `_STAGE2_SYSTEM_PLAYLIST` block (before `_SHORTFALL_THRESHOLD = 4`), add:

```python
def _make_selection_system(base: str) -> str:
    """Strip report schema field and format instructions from a Stage 2 system prompt.

    The report is generated in a separate pass so the selection call stays under 8K
    output tokens and completes reliably within the API timeout.
    """
    base = base.replace('\n  "report": "..."', "")
    marker = 'The "report" value must be a single string'
    idx = base.find(marker)
    if idx != -1:
        base = base[:idx].rstrip() + "\n\nRespond ONLY with the JSON array."
    return base


_STAGE2_SYSTEM_SELECTION: str = _make_selection_system(_STAGE2_SYSTEM)
_STAGE2_SYSTEM_PLAYLIST_SELECTION: str = _make_selection_system(_STAGE2_SYSTEM_PLAYLIST)

_STAGE2_REPORT_SYSTEM = """\
You are a world-class DJ and mix curator with deep real-world club experience. The track selection \
and play order for this concept have already been decided. Your job is to write the mix report only.

Write the report in this exact format:

CONCEPT: [concept_title]

[1–2 sentences: thesis — what this set asks of the room.]

Track order:
[For each track in play order, one line:]
N. Artist — Title [Key · BPM] | Role: [role] | Why: [one short phrase] | Risk: [one short phrase or "none"]

Assumptions: [only if material — [unverified] tracks, vocal clash, tight blend window. One line each. \
Omit section if nothing material.]

Role options: opener, builder, pivot, peak, cleanser, closer, utility.
Risk: describe the transition risk into this track (not out of it). "none" if clean.
Why: why this track at this moment — one phrase, no full sentences needed.

Be opinionated, musical, and honest. Peer-to-peer, no marketing language, no filler.

Return ONLY the report text. No JSON, no markdown fences, no preamble.\
"""
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "selection_system" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/llm.py tests/test_llm.py
git commit -m "feat(llm): add selection-only and report system prompts for two-pass Stage 2"
```

---

### Task 3: Add `_call_stage2_report_single` and `_call_stage2_reports`

**Files:**
- Modify: `src/mixlab/llm.py` — add `import asyncio`, add two new functions after `_call_stage2_raw`

- [ ] **Step 1: Write the failing tests**

In `tests/test_llm.py`, add a helper and two tests:

```python
def _selection_payload() -> str:
    """Stage 2 selection-only response — no report field."""
    return json.dumps(
        [
            {
                "title": "Dark Rollers",
                "name_reason": "Relentless drive from open to close.",
                "mood": "heavy and relentless",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [
                    {"from_id": "1", "to_id": "2", "is_risky": False, "risk_type": ""},
                    {"from_id": "2", "to_id": "3", "is_risky": False, "risk_type": ""},
                    {"from_id": "3", "to_id": "4", "is_risky": False, "risk_type": ""},
                ],
            }
        ]
    )


_REPORT_TEXT = "CONCEPT: Dark Rollers\n\nA relentless journey.\n\nTrack order:\n1. Artist 1 — Title 1 [8A · 174.0] | Role: opener | Why: sets dark tone | Risk: none"


@respx.mock
async def test_call_stage2_reports_returns_one_report_per_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import MixConcept, _call_stage2_reports

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    concepts = [
        MixConcept(title="Set A", mood="dark", track_ids=["1", "2", "3", "4"]),
        MixConcept(title="Set B", mood="light", track_ids=["1", "2", "3", "4"]),
    ]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json={"content": [{"text": "CONCEPT: Set A\n\nReport A."}], "stop_reason": "end_turn"}),
            Response(200, json={"content": [{"text": "CONCEPT: Set B\n\nReport B."}], "stop_reason": "end_turn"}),
        ]
    )

    reports = await _call_stage2_reports(concepts, tracks_by_id, None, None, "test-key")

    assert len(reports) == 2
    assert "Report A" in reports[0]
    assert "Report B" in reports[1]


@respx.mock
async def test_call_stage2_reports_fires_parallel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import MixConcept, _call_stage2_reports

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    call_count = 0

    def count_calls(request: object) -> Response:
        nonlocal call_count
        call_count += 1
        return Response(200, json={"content": [{"text": f"Report {call_count}"}], "stop_reason": "end_turn"})

    concepts = [MixConcept(title=f"Set {i}", mood="dark", track_ids=["1", "2", "3", "4"]) for i in range(3)]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A", title=f"T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    respx.post(_ANTHROPIC_URL).mock(side_effect=count_calls)

    reports = await _call_stage2_reports(concepts, tracks_by_id, None, None, "test-key")

    assert len(reports) == 3
    assert call_count == 3
```

- [ ] **Step 2: Run to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "call_stage2_reports" -v
```

Expected: FAIL with ImportError on `_call_stage2_reports`

- [ ] **Step 3: Add `import asyncio` and the two new functions**

At the top of `src/mixlab/llm.py`, add `import asyncio` to the stdlib imports block (alphabetically between `import json` and `import math`... actually insert it before `import json`):

```python
import asyncio
```

Then, in `src/mixlab/llm.py`, add these two functions immediately after `_call_stage2_raw` (before `stage2_curate_and_report`):

```python
async def _call_stage2_report_single(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
    seed_ids: frozenset[str] | None,
    unplayed_ids: set[str] | None,
    stage2_key: str,
) -> str:
    """Generate a prose mix report for one curated concept via Anthropic."""
    track_lines: list[str] = []
    for i, tid in enumerate(concept.track_ids, 1):
        t = tracks_by_id.get(tid)
        if t is None:
            continue
        extras: list[str] = []
        if t.year is not None:
            extras.append(str(t.year))
        if t.label:
            extras.append(t.label)
        if t.remixer:
            extras.append(f"remix by {t.remixer}")
        if t.mix:
            extras.append(f"mix:{', '.join(t.mix)}")
        if t.energy is not None:
            extras.append(f"energy:{t.energy}/8")
        if seed_ids is not None and tid in seed_ids:
            extras.append("[seed]")
        if unplayed_ids is not None:
            if tid in unplayed_ids:
                extras.append("unplayed")
        elif t.play_count == 0:
            extras.append("unplayed")
        if t.tags:
            extras.append(", ".join(t.tags))
        if t.enrichment_confidence == "low":
            extras.append("[unverified]")
        extra_str = " | " + " | ".join(extras) if extras else ""
        track_lines.append(
            f"{i}. ID:{tid} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key} | {t.genre}{extra_str}"
        )

    prompt = (
        f"Concept title: {concept.title}\n"
        f"Strategy/mood: {concept.mood}\n"
        f"Thesis: {concept.name_reason}\n\n"
        f"Tracks in play order:\n" + "\n".join(track_lines)
    )

    try:
        return await _call_anthropic_http(
            stage2_key, "claude-sonnet-4-6", _STAGE2_REPORT_SYSTEM, prompt, max_tokens=2048, timeout=120
        )
    except Exception as exc:
        raise RuntimeError(f"Stage 2 report generation failed: {exc}") from exc


async def _call_stage2_reports(
    concepts: list[MixConcept],
    tracks_by_id: dict[str, Track],
    seed_ids: frozenset[str] | None,
    unplayed_ids: set[str] | None,
    stage2_key: str,
) -> list[str]:
    """Generate prose reports for all concepts in parallel."""
    return list(
        await asyncio.gather(
            *[
                _call_stage2_report_single(c, tracks_by_id, seed_ids, unplayed_ids, stage2_key)
                for c in concepts
            ]
        )
    )
```

- [ ] **Step 4: Run tests + full suite**

```bash
.venv/bin/python -m pytest tests/test_llm.py -k "call_stage2_reports" -v
.venv/bin/python -m pytest -q
```

Expected: new tests PASS, full suite PASS

- [ ] **Step 5: Commit**

```bash
git add src/mixlab/llm.py tests/test_llm.py
git commit -m "feat(llm): add parallel report generation for Stage 2 two-pass split"
```

---

### Task 4: Parameterise `_call_stage2_raw` and remove debug logging

**Files:**
- Modify: `src/mixlab/llm.py:1163–1215` (`_call_stage2_raw`)

No new tests needed — `test_stage2_returns_curated_concepts_and_report` exercises this path (updated in Task 6).

- [ ] **Step 1: Rewrite `_call_stage2_raw` — Anthropic-only, add `max_tokens` param, remove debug logging**

Replace the entire `_call_stage2_raw` function:

```python
async def _call_stage2_raw(
    prompt: str,
    stage2_system: str,
    stage2_key: str,
    max_tokens: int = 8192,
) -> str:
    """Make the Stage 2 selection HTTP call via Anthropic."""
    try:
        return await _call_anthropic_http(
            stage2_key, "claude-sonnet-4-6", stage2_system, prompt, max_tokens=max_tokens, timeout=600
        )
    except Exception as exc:
        raise RuntimeError(f"Stage 2 curation failed: {exc}") from exc
```

Return type simplified from `tuple[str, str]` to `str` — model display is always `"Claude Sonnet 4.6"`, set as a constant in `stage2_curate_and_report`.

Also remove the debug `print` block added to `_call_anthropic_http` during diagnosis — revert that function to:

```python
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["content"][0]["text"])
```

- [ ] **Step 2: Run full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/mixlab/llm.py
git commit -m "refactor(llm): parameterise _call_stage2_raw max_tokens, remove debug logging"
```

---

### Task 5: Refactor `stage2_curate_and_report` to two-pass

**Files:**
- Modify: `src/mixlab/llm.py` — `stage2_curate_and_report` function (lines ~1218–1491)

This is the main wiring task. The scoring/variant logic is unchanged — only the system prompt selection and the report generation placement change.

- [ ] **Step 1: Identify the four change points before editing**

Open `src/mixlab/llm.py` and locate:
1. Line where `stage2_system` is assigned (`_STAGE2_SYSTEM_PLAYLIST if playlist_name is not None else _STAGE2_SYSTEM`)
2. The `_call_stage2_raw` call at the bottom of prompt-building
3. The `_parse_curated_concepts` call and `curated, report =` assignment
4. The playlist post-processing block that builds `section_reports` and `report_by_title`
5. The retry block's `_call_stage2_raw` call and `_parse_curated_concepts` call
6. The two `if variants:` / `else:` blocks that build `ordered_reports`

- [ ] **Step 2: Change the system prompt assignment**

```python
# OLD:
stage2_system = _STAGE2_SYSTEM_PLAYLIST if playlist_name is not None else _STAGE2_SYSTEM

# NEW:
stage2_system = _STAGE2_SYSTEM_PLAYLIST_SELECTION if playlist_name is not None else _STAGE2_SYSTEM_SELECTION
```

- [ ] **Step 3: Update `used_mix_names` injection**

The injection string-replaces a phrase that still exists in the selection prompts, so no change needed — verify the target string `"Give each concept a compelling creative name — not the pool name from Stage 1. \\"` is present in `_STAGE2_SYSTEM_SELECTION` by running:

```python
from mixlab.llm import _STAGE2_SYSTEM_SELECTION
assert "Give each concept a compelling creative name" in _STAGE2_SYSTEM_SELECTION
```

- [ ] **Step 4: Strip report instructions from the user prompt**

The user prompt is built in `stage2_curate_and_report`. The current prompt text asks the model to produce reports alongside curation; with the split, the selection pass must not request prose reports.

Non-playlist prompt (line ~1350):
```python
# OLD:
f"Curate and narrate a mix report from the following {n} candidate shortlists. "
f"Produce between 3 and 6 distinct concepts total. ..."

# NEW:
f"Curate a set of mix concepts from the following {n} candidate shortlists. "
f"Produce between 3 and 6 distinct concepts total. ..."
```

Playlist prompt — remove the entire report-format injection block (line ~1344):
```python
# DELETE this line (it instructs the model to write prose inside the JSON):
f"In each concept's report, after the thesis paragraph, include:\nSource playlist: {playlist_name}\n"
"State: seeds retained / dropped / added. For any notable drop or addition, one sentence of reasoning.\n\n"
```
The source-playlist context and seed accounting will move to the report-generation prompt (`_STAGE2_REPORT_SYSTEM`) in a later step if needed, or be computed deterministically by `_rewrite_playlist_report`.

- [ ] **Step 5: Update the main `_call_stage2_raw` call**

```python
# OLD:
raw, stage2_model_display = await _call_stage2_raw(
    prompt, stage2_system, stage2_key, use_minimax, stage2_model_display
)

# NEW:
raw = await _call_stage2_raw(prompt, stage2_system, stage2_key, max_tokens=8192)
```

- [ ] **Step 6: Update `_parse_curated_concepts` call to discard the empty report**

```python
# OLD:
curated, report = _parse_curated_concepts(raw, valid_ids)

# NEW:
curated, _ = _parse_curated_concepts(raw, valid_ids)
```

- [ ] **Step 7: Add non-playlist report generation**

Locate the `if playlist_name is not None and seed_ids is not None and curated:` block. Immediately before it (after the `curated, _ =` line), add a non-playlist branch:

```python
if playlist_name is None:
    reports = await _call_stage2_reports(curated, tracks_by_id, seed_ids, unplayed_ids, stage2_key)
    report = "\n\n---\n\n".join(reports)
```

- [ ] **Step 8: Update the playlist retry prompt and `_call_stage2_raw` call**

Inside the `if not passing:` block, strip the prose instruction from `retry_prompt` and update the `_call_stage2_raw` / `_parse_curated_concepts` calls:

```python
# OLD:
retry_prompt = (
    f"Your previous attempt retained only {len(retained_ids)} seed tracks "
    f"(minimum required: {minimum_seed_tracks}).\n"
    f"Dropped seeds: {dropped_labels}.\n"
    "Retry with one concept only. Include as many dropped seeds as possible. "
    "For each one still excluded, give one sentence of musical justification.\n\n"
) + prompt
raw, stage2_model_display = await _call_stage2_raw(
    retry_prompt, stage2_system, stage2_key, use_minimax, stage2_model_display
)
curated, report = _parse_curated_concepts(raw, valid_ids)

# NEW:
retry_prompt = (
    f"Your previous attempt retained only {len(retained_ids)} seed tracks "
    f"(minimum required: {minimum_seed_tracks}).\n"
    f"Dropped seeds: {dropped_labels}.\n"
    "Retry with one concept only. Include as many dropped seeds as possible.\n\n"
) + prompt
raw = await _call_stage2_raw(retry_prompt, stage2_system, stage2_key, max_tokens=8192)
curated, _ = _parse_curated_concepts(raw, valid_ids)
```

The removed sentence ("For each one still excluded, give one sentence of musical justification.") is a prose instruction inconsistent with selection-only JSON output — the retry call no longer returns a `report` field.

- [ ] **Step 9: Replace the `if variants:` / `else:` report-assembly blocks**

Find the existing block that starts with `if variants:` (builds `ordered_reports`, `ordered_concepts`) and the `else:` that handles the retry path. Replace the entire block with:

```python
        if variants:
            ordered_variants = (
                [best]
                + sorted(
                    [v for v in variants if v.concept is not concept],
                    key=lambda v: _STRATEGY_PRIORITY.get(v.strategy, 99),
                )
            )

            rejected_summary = ""
            rejected = ordered_variants[1:]
            if rejected:
                parts = [
                    f"{v.strategy} (practicality: {v.practicality_score.overall:.2f}, "
                    f"anchor retention: {v.anchor_retention_rate:.0%}) — not selected"
                    for v in rejected
                ]
                rejected_summary = "\nAlternative strategies considered: " + "; ".join(parts) + "."

            reports = await _call_stage2_reports(
                [v.concept for v in ordered_variants],
                tracks_by_id,
                seed_ids,
                unplayed_ids,
                stage2_key,
            )

            ordered_reports: list[str] = []
            ordered_concepts: list[MixConcept] = []
            for variant, base_report in zip(ordered_variants, reports, strict=True):
                is_winner = variant.concept is concept
                if is_winner:
                    base_report = _rewrite_playlist_report(
                        base_report,
                        playlist_name,
                        variant.concept,
                        playlist_seed_track_ids,
                        tracks_by_id,
                        rejected_summary,
                    )
                ordered_reports.append(
                    _label_playlist_report_section(base_report, variant.strategy, is_winner=is_winner)
                )
                ordered_concepts.append(
                    _retitle_playlist_concept(variant.concept, variant.strategy, is_winner=is_winner)
                )
            report = "\n\n---\n\n".join(ordered_reports)
            curated = ordered_concepts
        else:
            rejected_summary = ""
            retry_reports = await _call_stage2_reports(
                [concept], tracks_by_id, seed_ids, unplayed_ids, stage2_key
            )
            base_report = retry_reports[0] if retry_reports else ""
            report = _label_playlist_report_section(
                _rewrite_playlist_report(
                    base_report, playlist_name, concept, playlist_seed_track_ids, tracks_by_id, rejected_summary
                ),
                concept.mood.lower(),
                is_winner=True,
            )
            curated = [_retitle_playlist_concept(concept, concept.mood.lower(), is_winner=True)]
```

Note: the old code had `ordered_variants` and `rejected_summary` built before this block in separate steps. With the new structure, both are computed inside `if variants:`. Remove the old `ordered_variants =` and `rejected_summary =` assignments that preceded the old `if variants:` block.

- [ ] **Step 10: Remove the now-dead `section_reports` / `report_by_title` code**

Delete these lines from the start of the `if playlist_name is not None ...` block (they are no longer used):

```python
section_reports = report.split("\n\n---\n\n") if report else []
report_by_title = (
    {concept.title: section for concept, section in zip(curated, section_reports, strict=False)}
    if len(section_reports) == len(curated)
    else {}
)
```

- [ ] **Step 11: Run type check and full suite**

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest -q
```

Expected: ruff clean, mypy passes, existing tests may fail (they mock 1 Anthropic call; now 2 are needed — fix in Task 6)

- [ ] **Step 12: Commit once tests pass after Task 6**


Hold commit until Task 6 is done.

---

### Task 6: Update tests for two-pass flow

**Files:**
- Modify: `tests/test_llm.py`

The existing `stage2_curate_and_report` tests each mock a single Anthropic call returning combined JSON. Now the function makes two calls: first for selection JSON, second for the prose report. Update each test accordingly.

- [ ] **Step 1: Update `_curated_payload` to selection-only (no `report` field)**

```python
def _curated_payload() -> str:
    """Stage 2 selection-only response — no report field."""
    return json.dumps(
        [
            {
                "title": "Dark Rollers",
                "name_reason": "Relentless drive from open to close.",
                "mood": "heavy and relentless",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [
                    {"from_id": "1", "to_id": "2", "is_risky": False, "risk_type": ""},
                    {"from_id": "2", "to_id": "3", "is_risky": False, "risk_type": ""},
                    {"from_id": "3", "to_id": "4", "is_risky": False, "risk_type": ""},
                ],
            }
        ]
    )
```

- [ ] **Step 2: Add `_report_text` helper and update `_anthropic_response`**

Add at module level:

```python
_REPORT_TEXT = "CONCEPT: Dark Rollers\n\nA relentless journey.\n\nTrack order:\n1. Artist 1 — Title 1 [8A · 174.0] | Role: opener | Why: sets dark tone | Risk: none"


def _anthropic_response(content: str) -> dict[str, object]:
    return {"content": [{"text": content}], "stop_reason": "end_turn"}
```

- [ ] **Step 3: Update `test_stage2_returns_curated_concepts_and_report`**

```python
@respx.mock
async def test_stage2_returns_curated_concepts_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),  # selection
            Response(200, json=_anthropic_response(_REPORT_TEXT)),         # report
        ]
    )

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(
            track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass"
        )
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(shortlists, tracks_by_id)

    assert len(concepts) == 1
    assert concepts[0].title == "Dark Rollers"
    assert concepts[0].track_ids == ["1", "2", "3", "4"]
    assert "CONCEPT: Dark Rollers" in report
    assert "Claude Sonnet 4.6" in report
```

- [ ] **Step 4: Update `test_stage2_strips_hallucinated_ids`**

```python
@respx.mock
async def test_stage2_strips_hallucinated_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "T",
                "name_reason": "n/a",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "999"],
                "transitions": [],
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(payload)),
            Response(200, json=_anthropic_response("CONCEPT: T\n\nBrief.")),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="m", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    concepts, _ = await stage2_curate_and_report(shortlists, tracks_by_id)
    assert "999" not in concepts[0].track_ids
```

- [ ] **Step 5: Delete `test_stage2_falls_back_to_minimax_on_anthropic_failure`**

MiniMax removed — delete this test entirely. Do not update it.

- [ ] **Step 6: Delete `test_stage2_uses_minimax_when_stage2_provider_set`**

MiniMax removed — delete this test entirely. Do not update it.

- [ ] **Step 7: Update `test_stage2_playlist_mode_report_rewrites_seed_counts_deterministically`**

Remove the embedded `"report"` from the selection payload and mock two Anthropic calls: one for selection, one for the report (which `_rewrite_playlist_report` will then rewrite the seed counts on).

```python
@respx.mock
async def test_stage2_playlist_mode_report_rewrites_seed_counts_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Set",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "5", "6", "7", "8"],
                "transitions": [],
            }
        ]
    )
    prose_report = (
        "CONCEPT: Set\n\nThesis.\n\n"
        "Source playlist: Monday Night\n"
        "Seed tracks retained: 1.\n"
        "Seed tracks dropped: 0.\n"
        "Library tracks added: 7.\n\n"
        "Track order:\nArtist 1 — Title 1 [8A · 120.0] | Role: opener | Why: opens | Risk: none"
    )
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(payload)),        # selection
            Response(200, json=_anthropic_response(prose_report)),   # report
        ]
    )

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=[str(i) for i in range(1, 9)])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=120.0 + i, camelot_key="8A", genre="House")
        for i in range(1, 9)
    }

    _concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1", "2", "3", "4", "9", "10"}),
        seed_track_ids=["1", "2", "3", "4", "9", "10"],
    )

    assert "Seed tracks retained: 4" in report
    assert "Seed tracks dropped: 2." in report
    assert "Library tracks added: 4." in report
```

- [ ] **Step 8: Update `test_stage2_playlist_mode_raises_when_retention_below_minimum`**

Remove the embedded `"report"` from the payload — error is raised before the report call, so the single mock is sufficient.

```python
@respx.mock
async def test_stage2_playlist_mode_raises_when_retention_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Set",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "11", "12", "13", "14", "15", "16", "17", "18"],
                "transitions": [],
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(payload)))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=[str(i) for i in range(1, 19)])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=120.0 + i, camelot_key="8A", genre="House")
        for i in range(1, 19)
    }

    with pytest.raises(RuntimeError, match="retained too few seed tracks"):
        await stage2_curate_and_report(
            shortlists,
            tracks_by_id,
            playlist_name="Monday Night",
            seed_ids=frozenset({str(i) for i in range(1, 11)}),
            seed_track_ids=[str(i) for i in range(1, 11)],
        )
```

- [ ] **Step 9: Update `test_stage2_playlist_mode_returns_winner_first_with_labeled_titles`**

Remove embedded `"report"` fields from the selection payload. After selection, `_call_stage2_reports` fires 3 parallel calls (one per `ordered_variant`). Mock 1 selection + 3 report responses (4 total). The report order matches `ordered_variants`: winner first (Practical), then Balanced, then Adventurous.

```python
@respx.mock
async def test_stage2_playlist_mode_returns_winner_first_with_labeled_titles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Balanced Set",
                "mood": "balanced",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [],
            },
            {
                "title": "Practical Set",
                "mood": "practical",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [],
            },
            {
                "title": "Adventurous Set",
                "mood": "adventurous",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [],
            },
        ]
    )

    def make_report(title: str) -> str:
        return f"CONCEPT: {title}\n\nThesis.\n\nTrack order:\n1. A1 — T1 [8A · 121.0] | Role: opener | Why: opens | Risk: none"

    # ordered_variants = winner (Practical) + sorted remainder → Practical, Balanced, Adventurous
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(payload)),                                   # selection
            Response(200, json=_anthropic_response(make_report("Practical Set"))),              # report: winner
            Response(200, json=_anthropic_response(make_report("Balanced Set"))),               # report: balanced
            Response(200, json=_anthropic_response(make_report("Adventurous Set"))),            # report: adventurous
        ]
    )

    def fake_score_variant(
        concept: MixConcept,
        seed_track_ids: list[str],
        intent_brief: object,
        tracks_by_id: dict[str, Track],
    ) -> CompletionVariant:
        del seed_track_ids, intent_brief, tracks_by_id
        score_by_strategy = {
            "practical": 0.9,
            "balanced": 0.7,
            "adventurous": 0.6,
        }
        score = score_by_strategy[concept.mood]
        return CompletionVariant(
            strategy=concept.mood,  # type: ignore[arg-type]
            concept=concept,
            anchor_retention_rate=1.0,
            practicality_score=DJPracticalityScore(score, score, score, score),
        )

    monkeypatch.setattr("mixlab.llm._score_variant", fake_score_variant)

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=120.0 + i, camelot_key="8A", genre="House")
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1", "2", "3", "4"}),
        seed_track_ids=["1", "2", "3", "4"],
    )

    assert [concept.title for concept in concepts] == [
        "WINNER - PRACTICAL - Practical Set",
        "BALANCED - Balanced Set",
        "ADVENTUROUS - Adventurous Set",
    ]
    assert report.startswith("VARIANT: WINNER - PRACTICAL")
    assert "VARIANT: BALANCED" in report
    assert "VARIANT: ADVENTUROUS" in report
    assert report.index("VARIANT: WINNER - PRACTICAL") < report.index("VARIANT: BALANCED")
    assert report.index("VARIANT: BALANCED") < report.index("VARIANT: ADVENTUROUS")
    assert (
        "Alternative strategies considered: balanced (practicality: 0.70, anchor retention: 100%) — not selected; "
        "adventurous (practicality: 0.60, anchor retention: 100%) — not selected."
    ) in report
    assert "Alternative strategies considered: practical" not in report
```

- [ ] **Step 10: Run full suite**

```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest -q
```

Expected: all checks pass, all tests pass.

- [ ] **Step 11: Commit Tasks 5 + 6 together**

```bash
git add src/mixlab/llm.py tests/test_llm.py
git commit -m "refactor(llm): split Stage 2 into selection pass + parallel report generation"
```

---

### Task 7: Remove MiniMax completely

**Context:** MiniMax subscription cancelled. Stage 2 code (Tasks 3–4) is already Anthropic-only. This task cleans up what remains: Stage 1 cascade, `stage2_curate_and_report` provider setup, CLI flag, env vars, and existing MiniMax tests.

**Files:**
- Modify: `src/mixlab/llm.py`
- Modify: `src/mixlab/__main__.py`
- Modify: `.env.example`
- Modify: `tests/test_llm.py`

---

- [ ] **Step 1: Remove Stage 1 MiniMax code from `llm.py`**

Delete the two constants:
```python
_MINIMAX_STAGE1_TIMEOUT = 360
_MINIMAX_STAGE2_MAX_TOKENS = 40000
```

Delete the `_try_minimax` function entirely.

Update `_CASCADE` and the preceding comment:
```python
# OLD:
# Stage 1 free providers only — OpenRouter and Anthropic are Stage 2 / paid.
# MiniMax is last: throttled to 50 TPS on entry plan, making it too slow to lead.
_CASCADE: list[_Stage1Provider] = [_try_groq, _try_gemini, _try_mistral, _try_minimax]

# NEW:
# Stage 1 free providers only — Anthropic is Stage 2 / paid.
_CASCADE: list[_Stage1Provider] = [_try_groq, _try_gemini, _try_mistral]
```

Update the `_strip_thinking` comment to remove the MiniMax mention (function still needed for Gemini).

- [ ] **Step 2: Simplify `stage2_curate_and_report` provider setup**

Remove `stage2_provider` param and all MiniMax provider logic:

```python
# OLD:
async def stage2_curate_and_report(
    shortlists: list[MixConcept],
    tracks_by_id: dict[str, Track],
    stage2_provider: str | None = None,
    ...
) -> tuple[list[MixConcept], str]:
    provider = (stage2_provider or os.environ.get("STAGE2_PROVIDER", "anthropic")).lower()
    use_minimax = provider == "minimax"
    if use_minimax:
        stage2_key = os.environ.get("MINIMAX_API_KEY")
        if not stage2_key:
            raise RuntimeError("STAGE2_PROVIDER=minimax but MINIMAX_API_KEY is not set.")
        stage2_model_display = "MiniMax M2.7"
    else:
        stage2_key = os.environ.get("ANTHROPIC_API_KEY")
        if not stage2_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — Stage 2 curation requires Anthropic.")
        stage2_model_display = "Claude Sonnet 4.6"

# NEW:
async def stage2_curate_and_report(
    shortlists: list[MixConcept],
    tracks_by_id: dict[str, Track],
    ...
) -> tuple[list[MixConcept], str]:
    stage2_key = os.environ.get("ANTHROPIC_API_KEY")
    if not stage2_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — Stage 2 curation requires Anthropic.")
    stage2_model_display = "Claude Sonnet 4.6"
```

- [ ] **Step 3: Remove `--stage2-provider` from `__main__.py` and all downstream plumbing**

`stage2_provider` is threaded through three functions in addition to the CLI parser. Remove all of it:

**a) CLI parser** — delete the `parser.add_argument("--stage2-provider", ...)` block.

**b) `_format_run_label` (≈ line 63)** — remove `stage2_provider: str | None` from the signature, and remove the `if stage2_provider is not None: details.append(...)` block.

**c) `run_playlist` (≈ line 252)** — remove `stage2_provider: str | None` from the signature; remove it from the `stage2_curate_and_report(...)` keyword args (Task 2 already removes that param from the callee); remove `stage2_provider=stage2_provider` from the `_format_run_label(...)` call.

**d) `run` (≈ line 444)** — same as (c): remove `stage2_provider: str | None = None` from the signature, drop it from both `stage2_curate_and_report(...)` and `_format_run_label(...)` calls.

**e) CLI dispatch (≈ lines 833–851)** — remove `args.stage2_provider` from the `run_playlist(...)` and `run(...)` call sites.

After all removals, `stage2_provider` should not appear anywhere in `__main__.py`.

- [ ] **Step 4: Update `.env.example`**

Remove `MINIMAX_API_KEY=`, `STAGE2_PROVIDER=`, and the Stage 2 provider comment line.

- [ ] **Step 5: Update tests**

**Delete these three existing tests** (Task 6 Steps 5-6 already mark them for deletion):
- `test_stage2_falls_back_to_minimax_on_anthropic_failure`
- `test_stage2_uses_minimax_when_stage2_provider_set`
- `test_stage2_raises_if_minimax_key_missing_when_provider_set`

**Remove `_MINIMAX_URL`** constant at the top of `test_llm.py`.

**Remove all `monkeypatch.delenv("MINIMAX_API_KEY", ...)` calls** from tests that have them.

**`test_stage2_raises_if_key_missing`**: remove `monkeypatch.delenv("STAGE2_PROVIDER", raising=False)` — provider selection no longer exists.

**`test_stage1_raises_if_all_providers_missing`**: remove `"MINIMAX_API_KEY"` from the for-loop.

**Stage 1 consecutive-failures test**: remove `monkeypatch.setenv("MINIMAX_API_KEY", "key")` and `respx.post(_MINIMAX_URL).mock(...)`.

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/mixlab/llm.py src/mixlab/__main__.py .env.example tests/test_llm.py
git commit -m "chore(llm): remove MiniMax provider from Stage 1 and Stage 2"
```

---

## Self-Review

**Spec coverage:**
- ✅ Two-pass split: selection JSON first, reports in parallel after
- ✅ Track count cap: playlist mode capped at 12
- ✅ Debug logging removed
- ✅ MiniMax removed entirely (Tasks 3, 4, 7) — Stage 1 and Stage 2 Anthropic-only from the start, no churn
- ✅ Playlist retention scoring/variant selection runs between passes (unchanged logic)
- ✅ Retry path uses selection-only call and generates single report at end
- ✅ `asyncio.gather` for parallel report calls
- ✅ `rejected_summary` initialised to `""` in `else:` (retry) branch — no `UnboundLocalError`
- ✅ Model attribution footer unchanged: `Main brain: Claude Sonnet 4.6` — single provider, no dual-attribution needed
- ✅ Selection user prompt stripped of report instructions (non-playlist "narrate" and playlist seed-accounting lines removed in Task 5 Step 4)
- ✅ `_call_stage2_report_single` includes `t.mix` metadata — matches track context provided in selection prompt
- ✅ Playlist-mode tests updated for two-call flow: seed-count rewrite, retention-failure, winner-ordering

**Placeholder scan:** None found.

**Type consistency:**
- `_call_stage2_reports` returns `list[str]` — consumed as `reports` in both playlist and non-playlist paths ✅
- `_call_stage2_raw` returns `str` (Anthropic-only, no model-display tuple) — all callers updated ✅
- `_call_stage2_raw` `max_tokens` defaults to 8192 — all callers either pass explicit value or accept default ✅
- `ordered_variants` list built inside `if variants:` block — `zip(ordered_variants, reports, strict=True)` requires equal lengths; both derived from same `curated` list ✅
- `rejected_summary` always defined before `_rewrite_playlist_report` call — both `if variants:` and `else:` branches set it ✅

**Known limitation:** `test_call_stage2_reports_fires_parallel_calls` verifies call count (3) but not concurrency — a serial implementation would also pass. Verifying true parallelism requires timing assertions, which add flakiness; call-count is the practical proxy.
