from __future__ import annotations

import json
import os
import random
import re
import sys
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from mixlab.config import TRACK_COUNT_TARGETS
from mixlab.models import MixConcept, Track

_MAX_TRACKS_PER_CALL = 40
_MAX_TRACKS_PER_CALL_CUSTOM = 60  # larger chunks for custom multi-genre pools
_MAX_STAGE1_POOL_CUSTOM = 120  # random window size for custom pools (2 chunks × 60 = 2 API calls)
_MIN_SHORTLIST_TRACKS = 8  # Stage 1: minimum candidates per pool
_MIN_CONCEPT_TRACKS = 4  # Stage 2: minimum tracks in a final curated set
_STAGE2_CAP = 6  # max shortlists sent to Stage 2
_STAGE2_CANDIDATE_POOL = 12  # top N by size to sample from (ensures variety across runs)
_STAGE1_TIMEOUT = 120  # seconds — default for openai-compat providers
_MINIMAX_STAGE1_TIMEOUT = 360  # MiniMax is a thinking model; needs extra time

# Strip inline thinking blocks emitted by reasoning models (e.g. MiniMax M2.7).
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    stripped = _THINK_RE.sub("", text).strip()
    # If stripping removed everything, the model wrapped its entire output in thinking tags.
    # Fall back to the original so _parse_concepts can attempt to extract JSON from it.
    return stripped or text


# ---------------------------------------------------------------------------
# Stage 1 — candidate shortlisting (provider cascade)
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = """\
You are a music data analyst pre-screening a DJ's track collection to build candidate shortlists for mix concepts.

Your task is purely technical pre-screening — not creative curation.

For each shortlist you create:
- Group tracks that are plausibly technically compatible: similar BPM (±6 BPM within the pool) and harmonically \
related keys (adjacent or nearby Camelot positions).
- Each shortlist should contain 15–25 candidate tracks that a DJ could plausibly draw from for one mix concept.
- Generate 3–5 distinct shortlists with different BPM centres or key characters so they serve different mood directions. \
If the material only supports 1 or 2 distinct shortlists, produce what the material supports — do not pad. \
If no coherent shortlist of 8+ tracks can be formed, return an empty array [].
- Do NOT make final ordering decisions. Do NOT decide openers or closers. Simply group technically compatible tracks.
- Exclude obvious outliers: tracks more than 8 BPM from the group median, or in keys with no harmonic relationship \
to the rest of the pool.

Give each shortlist a rough descriptive title (e.g. "Deep 122 BPM / 4A–7A Pool") and a one-line sonic mood.

Some tracks include supplementary metadata: `energy:N/8` is a Mixed in Key automated score (0=lowest, 8=highest) and can help signal intensity. Treat it as a useful hint when present — not all tracks will have it, and its absence says nothing about the track's quality or suitability. When Year is present, you may form era-coherent groupings (e.g. a 1994–1997 pool alongside a 2018–present pool) as an alternative dimension to BPM-centre variation — but only when the material clearly separates into eras.

Respond ONLY with a JSON array matching this schema:
[{"title": "...", "mood": "...", "track_ids": ["id1", "id2", ...]}]\
"""

# Custom genre variant: larger shortlists (20-25) to give Stage 2 more material to discard.
_STAGE1_SYSTEM_CUSTOM = """\
You are a music data analyst pre-screening a DJ's track collection to build candidate shortlists for mix concepts.

Your task is purely technical pre-screening — not creative curation.

This pool spans multiple related sub-genres. Tracks from different sub-genres may appear together.

For each shortlist you create:
- Group tracks that are plausibly technically compatible: similar BPM (±6 BPM within the pool) and harmonically \
related keys (adjacent or nearby Camelot positions).
- Each shortlist should contain 20–25 candidate tracks that a DJ could plausibly draw from for one mix concept.
- Generate 3–5 distinct shortlists with different BPM centres or key characters so they serve different mood directions. \
If the material only supports 1 or 2 distinct shortlists, produce what the material supports — do not pad. \
If no coherent shortlist of 8+ tracks can be formed, return an empty array [].
- Do NOT make final ordering decisions. Do NOT decide openers or closers. Simply group technically compatible tracks.
- Exclude obvious outliers: tracks more than 8 BPM from the group median, or in keys with no harmonic relationship \
to the rest of the pool.

Give each shortlist a rough descriptive title (e.g. "Deep 122 BPM / 4A–7A Pool") and a one-line sonic mood.

Some tracks include supplementary metadata: `energy:N/8` is a Mixed in Key automated score (0=lowest, 8=highest) and can help signal intensity. Treat it as a useful hint when present — not all tracks will have it, and its absence says nothing about the track's quality or suitability. When Year is present, you may form era-coherent groupings as an alternative dimension to BPM-centre variation — but only when the material clearly separates into eras.

Respond ONLY with a JSON array matching this schema:
[{"title": "...", "mood": "...", "track_ids": ["id1", "id2", ...]}]\
"""


def _tracks_to_text(tracks: list[Track]) -> str:
    lines = []
    for t in tracks:
        line = f"ID:{t.track_id} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key}"
        if t.year is not None:
            line += f" | {t.year}"
        if t.energy is not None:
            line += f" | energy:{t.energy}/8"
        lines.append(line)
    return "\n".join(lines)


def _parse_concepts(raw: str) -> list[MixConcept]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()
    # Extract the JSON array by finding the outermost [...] bounds.
    # This handles both leading prose and trailing prose from any model.
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        raw = raw[first_bracket : last_bracket + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: escape literal newlines/carriage returns inside JSON strings
        # (some models emit unescaped control characters in string values).
        data = json.loads(_fix_json_strings(raw))
    return [MixConcept(**item) for item in data]


async def _call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    path: str = "/v1/chat/completions",
    timeout: int = _STAGE1_TIMEOUT,
    system: str = _STAGE1_SYSTEM,
    max_tokens: int = 4096,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{base_url.rstrip('/')}{path}", headers=headers, json=payload)
        resp.raise_for_status()
        return _strip_thinking(str(resp.json()["choices"][0]["message"]["content"]))


async def _call_anthropic_http(
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 2048,
    timeout: int = 90,
) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["content"][0]["text"])


async def _try_minimax(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        return None
    return await _call_openai_compat(
        "https://api.minimax.io/v1",
        key,
        "MiniMax-M2.7",
        prompt,
        path="/text/chatcompletion_v2",
        timeout=_MINIMAX_STAGE1_TIMEOUT,
        system=system,
    )


async def _try_groq(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return await _call_openai_compat(
        "https://api.groq.com/openai", key, "llama-3.3-70b-versatile", prompt, system=system
    )


async def _try_gemini(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    return await _call_openai_compat(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        key,
        "gemini-2.5-flash",
        prompt,
        path="/chat/completions",
        system=system,
    )


async def _try_mistral(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        return None
    return await _call_openai_compat("https://api.mistral.ai", key, "mistral-small-latest", prompt, system=system)


async def _try_anthropic(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return await _call_anthropic_http(key, "claude-sonnet-4-6", system, prompt)


# Stage 1 free providers only — OpenRouter and Anthropic are Stage 2 / paid.
# MiniMax is last: throttled to 50 TPS on entry plan, making it too slow to lead.
_Stage1Provider = Callable[..., Coroutine[Any, Any, str | None]]
_CASCADE: list[_Stage1Provider] = [_try_groq, _try_gemini, _try_mistral, _try_minimax]


@dataclass
class CascadeState:
    """Shared mutable state threaded through all Stage 1 calls in a single run."""

    index: int = 0
    consecutive_failures: int = field(default=0)


def make_cascade_state() -> CascadeState:
    return CascadeState()


async def _call_stage1_once(
    tracks: list[Track], genre: str, state: CascadeState, custom: bool = False
) -> list[MixConcept]:
    valid_ids = {t.track_id for t in tracks}
    prompt = f"Genre: {genre}\n\nTracks:\n{_tracks_to_text(tracks)}"
    system = _STAGE1_SYSTEM_CUSTOM if custom else _STAGE1_SYSTEM

    for _ in range(len(_CASCADE)):
        provider = _CASCADE[state.index]
        try:
            result = await provider(prompt, system=system)
            if result is None:  # provider not configured — skip silently, no failure counted
                state.index = (state.index + 1) % len(_CASCADE)
                continue
            if not result.strip():  # provider returned empty content — treat as a failure
                raise ValueError(f"Provider {provider.__name__} returned empty content.")
            concepts = _parse_concepts(result)
            cleaned = [
                MixConcept(title=c.title, mood=c.mood, track_ids=[tid for tid in c.track_ids if tid in valid_ids])
                for c in concepts
            ]
            state.consecutive_failures = 0
            return [c for c in cleaned if len(c.track_ids) >= _MIN_SHORTLIST_TRACKS]
        except Exception as exc:  # noqa: BLE001 — cascade
            print(f"Provider {provider.__name__} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            state.consecutive_failures += 1
            if state.consecutive_failures >= len(_CASCADE):
                raise RuntimeError(
                    f"All Stage 1 providers failed — {state.consecutive_failures} consecutive failures "
                    f"with no successful call."
                ) from exc
            state.index = (state.index + 1) % len(_CASCADE)

    raise RuntimeError(f"All Stage 1 providers exhausted for genre '{genre}'.")


async def stage1_concepts(
    cluster: list[Track], genre: str, state: CascadeState, custom: bool = False
) -> list[MixConcept]:
    chunk_size = _MAX_TRACKS_PER_CALL_CUSTOM if custom else _MAX_TRACKS_PER_CALL
    if len(cluster) <= chunk_size:
        return await _call_stage1_once(cluster, genre, state, custom=custom)

    concepts: list[MixConcept] = []
    for i in range(0, len(cluster), chunk_size):
        concepts.extend(await _call_stage1_once(cluster[i : i + chunk_size], genre, state, custom=custom))
    return concepts


def select_stage1_window(tracks: list[Track], max_count: int) -> list[Track]:
    """Pick a random contiguous window of up to max_count tracks from a Camelot-sorted pool.

    Each run starts at a different position in the sorted list, giving variety across runs
    while keeping Stage 1 input small enough to avoid LLM rate limits.
    The window is a coherent BPM/key slice because the input is Camelot-sorted.
    """
    if len(tracks) <= max_count:
        return tracks
    start = random.randint(0, len(tracks) - max_count)
    return tracks[start : start + max_count]


def select_shortlists_for_stage2(shortlists: list[MixConcept]) -> list[MixConcept]:
    """Select up to _STAGE2_CAP shortlists for Stage 2, sampling randomly from the top candidates by pool size.

    Always picks from the _STAGE2_CANDIDATE_POOL largest shortlists so every run has a chance of variety
    while still favouring well-stocked pools.
    """
    if len(shortlists) <= _STAGE2_CAP:
        return shortlists
    candidates = sorted(shortlists, key=lambda s: len(s.track_ids), reverse=True)[:_STAGE2_CANDIDATE_POOL]
    return random.sample(candidates, min(_STAGE2_CAP, len(candidates)))


# ---------------------------------------------------------------------------
# Stage 2 — creative curation + report (Anthropic, single call, no fallback)
# ---------------------------------------------------------------------------

_STAGE2_SYSTEM = """\
Act as a world-class DJ and mix curator with deep real-world club experience. You will receive candidate \
shortlists of tracks that have been pre-screened for technical compatibility. Your job is to curate and \
narrate — select the best tracks from each shortlist, decide the play order, and write the full mix report.

For each shortlist or sub-pool you carve from it:
- SELECT the best 8–12 tracks from the pool for a coherent DJ set. Exclude tracks that weaken the journey. \
Weakness is practical: a track whose intro gives no workable mix point, a vocal that starts on bar one with \
no room to bring it in, a bass-heavy record dropped after another with no frequency relief, a big moment \
used so early it makes everything after feel like a comedown.
- ORDER them as the intended play sequence: opener first, closer last.
- The opener plays to a room that isn't committed yet. It must work as ambient architecture — rewarding \
attention without requiring it. It should not telegraph where the set is going. A track that demands \
engagement in its first 32 bars is the wrong opener regardless of its quality.
- The closer must signal its own finality before it arrives. The room should feel the set ending. The \
default closer resolves — it has weight, sufficient outro length to mix out of cleanly, and leaves the \
room with a feeling rather than a question. A track whose energy rises continuously into its final bars \
can close a set only if its authority is strong enough to signal finality without resolution. If you are \
not certain it carries that weight, it is not the closer.
- Each concept should have a thesis — not just a mood, but an intention. What does this set ask of the \
room? The creative brief must answer this.
- Assign each track a role: opener, builder, pivot, peak weapon, palette cleanser, or closer. Not every \
role needs to be present in every set — but every track should have one.
- Design an intentional energy curve. This need not be a single arc — consider double peaks, \
plateau-and-release structures, or a false resolution before the final push. The shape should feel \
inevitable in retrospect, not predictable in real time.
- A set can sustain two or three genuine peak moments at most. Everything else is architecture that makes \
those moments land. Do not load the tracklist with peak weapons — they cancel each other out and produce \
a set with no dynamic range.
- Be aware of vocal density, percussion character, and production era across consecutive tracks. Avoid \
creating a blend window where two active vocals are audible simultaneously — if the incoming vocal starts \
early and the outgoing vocal hasn't cleared, you need an instrumental bridge or a different track order. \
When arrangement data is unavailable, flag this risk in Assumptions.
- Be aware of bass weight across consecutive tracks. A sequence of high-intensity bassline tracks without \
textural relief will fatigue a room physically, not just emotionally.
- Be aware of transition-window usability. A perfect harmonic and energy match is worthless if the tracks \
don't actually blend — a four-bar outro into a one-bar intro is not a transition, it's a cut. Where \
arrangement data is unavailable, use knowledge of the artist's production style to assess blend headroom \
and flag any transition where mix execution is likely to be tight or forced. Name the risk in Assumptions.
- When choosing between a track that sustains momentum and a track that is more interesting on paper, \
prefer momentum. Novelty that breaks the groove is a mistake regardless of how well it reads.
- Allow bold moves — larger key jumps, tempo pivots — when they serve the narrative. For any Camelot jump \
of 3+ positions, name the specific mechanism that makes it survivable — BPM lock, rhythmic momentum, a \
slow intro that buys the room time to adjust, or an emotional peak that earns the disruption. The \
placement of harmonic risk matters as much as the risk itself — a large key jump works best when the \
floor is already committed and moving, mid-to-late set at or approaching peak energy.
- Do NOT optimise only for BPM and key. Optimise for flow, tension, release, memorability, and emotional \
payoff.
- Some tracks carry enrichment metadata. These are context clues — use them to deepen the narrative, not \
to sort or constrain selection. Unenriched tracks are first-class; absence of any field says nothing about \
quality. When fields are missing, reason from BPM, key, genre, and artist knowledge as normal.
  - `energy:N/8` — Mixed in Key score (0=lowest, 8=highest). Use to build the energy arc.
  - `unplayed` — never played live. Available for debut.
  - Year — production era. Useful when articulating era dialogue or coherence. Either is a valid concept \
shape; neither is required.
  - Label — scene DNA. Tells you why something sounds the way it does. Not a selection criterion — great \
concepts span many labels.
  - `remix by [Remixer]` — remixer's style may define the track more than the original artist's. Name them \
when it changes how the track functions.
  - `mix:[styles]` — BPM-filtered cross-genre tags. Use when a pivot moment calls for it; ignore when it \
adds nothing to the concept.
  - Tags — comma-separated genre/mood descriptors (e.g. `Breakbeat, Acid, Dark, Driving`).
  - `[unverified]` — low-confidence match. Treat label, year, and album as indicative. Flag in Assumptions \
only if those fields are directly driving a curatorial decision.
- Before finalising, verify that each concept is genuinely distinct. The test is not mechanical — it is \
this: could a knowledgeable listener hear thirty seconds of any track from one concept and know it does \
not belong in another? If two concepts share more than two tracks, or if they would feel like the same \
set to someone on the dancefloor regardless of what the metadata says, they are not distinct enough — \
collapse or redesign.

Produce as many concepts as the pool genuinely supports — between 3 and 6. If the pool only yields 4 \
strong concepts, produce 4. Do not pad with weak concepts to hit a number. If the pool is too thin to \
produce even 3 distinct, coherent sets, return a single-element array with a diagnostic object instead \
of concept objects: [{"diagnostic": "...explanation of why the pool is insufficient..."}] — this is \
useful information, not a failure.

Your output must be a JSON array where each element has exactly this schema:
{
  "title": "...",
  "mood": "...",
  "track_ids": ["id1", "id2", ...],
  "report": "..."
}

Give each concept a compelling creative name — not the pool name from Stage 1.
The track_ids must be the final selected tracks in play order.
The "report" value must be a single string (with \\n for line breaks) in this exact format:

CONCEPT: [title]

[1–2 sentences: the set's thesis — what it asks of the room. Not mood alone; intention.]

Track order (Camelot / BPM):
[Artist — Title [Key · BPM] for each track in play order, one track per line]

Arc: [One sentence describing the overall energy shape.]

[One line per track in play order. No blank lines between them. \
Format: Artist — Title (role) — one sentence on why this track at this moment. \
If the move to the next track is non-obvious (Camelot jump 3+ positions or BPM shift >5 BPM), \
append the mechanism in the same line after a semicolon. Nothing else.]

Standout transitions or calculated risks: [Only for risks not already covered in the track lines \
above. One sentence per item. If nothing qualifies, omit this section entirely — do not write it.]

Assumptions:
- [One bullet per flag. Lead with the risk. Cover only genuine uncertainty: [unverified] tracks, \
vocal clash risks, transition-window concerns. Omit section if nothing material.]

First decide whether the set would still make sense without any written justification. Only then write \
the report.

Be opinionated, musical, and honest. Peer-to-peer, no marketing language, no filler. When choosing \
between sounding clever and being right, be right.

Respond ONLY with the JSON array.\
"""

_SHORTFALL_THRESHOLD = 4


def _shortfall_warning(concept: MixConcept, genre: str) -> str | None:
    min_count, _ = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])
    n = len(concept.track_ids)
    shortfall = min_count - n
    if shortfall > _SHORTFALL_THRESHOLD:
        return f"⚠️ {n} tracks found — needs {shortfall} more to fill a set. Crate dig to complete."
    return None


def _fix_json_strings(raw: str) -> str:
    """Escape literal newlines and carriage returns inside JSON string values."""
    result: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        else:
            result.append(ch)
    return "".join(result)


def _parse_curated_concepts(raw: str, valid_ids: set[str]) -> tuple[list[MixConcept], str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()
    try:
        data: list[dict[str, object]] = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(_fix_json_strings(raw))

    # Diagnostic: model signals the pool is too thin to produce 3+ distinct concepts.
    if len(data) == 1 and "diagnostic" in data[0] and "title" not in data[0]:
        return [], f"POOL DIAGNOSTIC\n{data[0]['diagnostic']}"

    curated: list[MixConcept] = []
    report_parts: list[str] = []

    for item in data:
        raw_ids = item.get("track_ids")
        track_ids = [str(tid) for tid in (raw_ids if isinstance(raw_ids, list) else []) if str(tid) in valid_ids]
        if len(track_ids) < _MIN_CONCEPT_TRACKS:
            continue
        curated.append(
            MixConcept(title=str(item.get("title", "")), mood=str(item.get("mood", "")), track_ids=track_ids)
        )
        report_parts.append(str(item.get("report", "")))

    return curated, "\n\n---\n\n".join(report_parts)


async def stage2_curate_and_report(
    shortlists: list[MixConcept],
    tracks_by_id: dict[str, Track],
    stage2_provider: str | None = None,
    custom_genre_label: str | None = None,
    custom_genre_sub_genres: list[str] | None = None,
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

    sections: list[str] = []
    for shortlist in shortlists:
        track_lines = []
        for tid in shortlist.track_ids:
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
            if t.play_count == 0:
                extras.append("unplayed")
            if t.tags:
                extras.append(", ".join(t.tags))
            if t.enrichment_confidence == "low":
                extras.append("[unverified]")
            extra_str = " | " + " | ".join(extras) if extras else ""
            track_lines.append(
                f"  ID:{tid} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key} | {t.genre}{extra_str}"
            )
        if track_lines:
            sections.append(
                f"Shortlist: {shortlist.title}\nCharacter: {shortlist.mood}\nCandidates:\n" + "\n".join(track_lines)
            )

    n = len(sections)
    if n == 0:
        raise RuntimeError("Stage 2 received no shortlists with resolvable tracks — nothing to curate.")

    prompt = (
        f"Curate and narrate a mix report from the following {n} candidate shortlists. "
        f"Produce between 3 and 6 distinct concepts total. A rich shortlist may yield more than one concept; "
        f"a thin shortlist may yield none — but each concept must draw only from tracks within a single shortlist.\n\n"
        + "\n\n".join(sections)
    )

    if custom_genre_label and custom_genre_sub_genres:
        sub_genre_str = ", ".join(custom_genre_sub_genres)
        prompt += (
            f"\n\nThis is a multi-genre custom pool: '{custom_genre_label}' ({sub_genre_str}). "
            f"When sequencing across sub-genre boundaries, you must justify the move — "
            f"name the specific mechanism that makes the transition work: BPM alignment, rhythmic character, "
            f"harmonic relationship, or the energy state of the room that earns the shift. "
            f"Do not avoid cross-genre moves — they are the point of this pool. But every such move must be defensible."
        )

    raw: str
    if use_minimax:
        try:
            raw = await _call_openai_compat(
                "https://api.minimax.io/v1",
                stage2_key,
                "MiniMax-M2.7",
                prompt,
                path="/text/chatcompletion_v2",
                timeout=300,
                system=_STAGE2_SYSTEM,
            )
        except Exception as exc:
            raise RuntimeError(f"Stage 2 curation failed: {exc}") from exc
    else:
        try:
            raw = await _call_anthropic_http(
                stage2_key, "claude-sonnet-4-6", _STAGE2_SYSTEM, prompt, max_tokens=32000, timeout=600
            )
        except Exception as anthropic_exc:
            print(
                f"Stage 2 Anthropic failed ({type(anthropic_exc).__name__}: {anthropic_exc}), trying MiniMax M2.7 fallback...",
                file=sys.stderr,
            )
            minimax_key = os.environ.get("MINIMAX_API_KEY")
            if not minimax_key:
                raise RuntimeError(f"Stage 2 curation failed: {anthropic_exc}") from anthropic_exc
            try:
                raw = await _call_openai_compat(
                    "https://api.minimax.io/v1",
                    minimax_key,
                    "MiniMax-M2.7",
                    prompt,
                    path="/text/chatcompletion_v2",
                    timeout=300,
                    system=_STAGE2_SYSTEM,
                )
            except Exception as minimax_exc:
                raise RuntimeError(
                    f"Stage 2 curation failed (Anthropic and MiniMax both failed): {minimax_exc}"
                ) from minimax_exc
            stage2_model_display = "MiniMax M2.7 (Anthropic fallback)"

    valid_ids = set(tracks_by_id.keys())
    curated, report = _parse_curated_concepts(raw, valid_ids)

    warnings: list[str] = []
    for concept in curated:
        if not concept.track_ids:
            continue
        track = tracks_by_id.get(concept.track_ids[0])
        genre = track.genre if track else "_default"
        warning = _shortfall_warning(concept, genre)
        if warning:
            warnings.append(f"⚠️ {concept.title}: {warning}")

    if warnings:
        report += "\n\n---\n\nSHORTFALL WARNINGS\n" + "\n".join(warnings)

    report += f"\n\n---\n\nMain brain: {stage2_model_display}"

    return curated, report
