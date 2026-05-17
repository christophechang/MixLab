from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, cast, get_args

import httpx

from mixlab.clustering import camelot_distance
from mixlab.config import TRACK_COUNT_TARGETS, shortfall_warning
from mixlab.history import ConceptHistory, format_recent_concepts
from mixlab.models import (
    ArcType,
    CompletionVariant,
    DJPracticalityScore,
    IntentBrief,
    MixCanvas,
    MixConcept,
    RiskTolerance,
    SeedAnalysis,
    SeedTier,
    SetRole,
    Track,
    Transition,
)

_ARC_TYPE_VALUES: frozenset[str] = frozenset(get_args(ArcType))

_MAX_TRACKS_PER_CALL = 40
_MAX_TRACKS_PER_CALL_CUSTOM = 60  # larger chunks for custom multi-genre pools
MAX_STAGE1_POOL_CUSTOM = 120  # random window size for custom pools (2 chunks × 60 = 2 API calls)
MIN_SHORTLIST_TRACKS = 8  # Stage 1: minimum candidates per pool
_MIN_CONCEPT_TRACKS = 4  # Stage 2: minimum tracks in a final curated set
_STAGE2_CAP = 6  # max shortlists sent to Stage 2
_STAGE2_CANDIDATE_POOL = 12  # top N by size to sample from (ensures variety across runs)
_STAGE1_TIMEOUT = 120  # seconds — default for openai-compat providers
_THINKING_MODEL_MAX_TOKENS = 16000  # thinking models spend tokens on reasoning before output

# Strip inline thinking blocks emitted by reasoning models (e.g. Gemini 2.5 Flash).
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


_STAGE1_SYSTEM_PLAYLIST = """\
You are a music data analyst pre-screening a DJ's track collection to build candidate shortlists for a playlist completion concept.

Tracks marked [seed] come from an existing playlist and represent the intended musical direction. Treat them as strong candidates — but group by BPM and harmonic compatibility above all else. A seed track that is an outlier (more than 8 BPM from the group median, or in a harmonically unrelated key) should still be excluded from any group where it does not fit.

For each shortlist:
- Group tracks that are plausibly technically compatible: similar BPM (±6 BPM within the pool) and harmonically related keys (adjacent or nearby Camelot positions).
- Each shortlist should contain 15–25 candidate tracks.
- Generate 1–3 distinct shortlists. If the material only supports one coherent group, produce one.
- Do NOT make final ordering decisions. Simply group technically compatible tracks.

Give each shortlist a rough descriptive title and a one-line sonic mood.

Some tracks include supplementary metadata: `energy:N/8` is a Mixed in Key score. [seed] marks tracks from the source playlist.

Respond ONLY with a JSON array:
[{"title": "...", "mood": "...", "track_ids": ["id1", "id2", ...]}]\
"""


# ---------------------------------------------------------------------------
# Stage 0 — intent extraction (free provider cascade, playlist mode only)
# ---------------------------------------------------------------------------

_STAGE0_SYSTEM = """\
You are a DJ set analyst. You will receive seed tracks from a DJ's draft playlist in original order.

Classify each track and describe the set's intent.

Tier definitions:
- anchor: 2–4 tracks that define the set's identity. The DJ would never swap these out.
- supporting: tracks that serve the arc; keep by default, replaceable with musical reason.
- optional: filler or candidates — lowest priority to retain.

Inferred role options: opener, world_setter, groove_locker, early_hook, builder, connector, pivot, pressure, lift, vocal_moment, texture_change, cleanser, risk, weapon, peak, post_peak, resolution, closer, utility, unknown

Use energy:N/8 when present. When absent, reason from BPM, Camelot key, and list position:
- Opener candidate: first 1–2 positions, energy 1–3/8 or lowest BPM relative to pool
- Peak candidate: energy 7–8/8 or highest BPM
- Closer candidate: last 1–2 positions

missing_roles: list which of [opener, builder, peak, cleanser, closer] appear absent.

Risk tolerance — BPM spread and Camelot key spread across all seeds:
- low: BPM spread < 10, mostly adjacent keys
- medium: BPM spread 10–20, 1–2 key jumps
- high: BPM spread > 20 or large Camelot jumps throughout

is_coherent_set: true if one set idea; false if distinct chapters.

Return ONLY a JSON object (no prose, no markdown fence):
{
  "overall_vibe": "one sentence about what this set is trying to do",
  "is_coherent_set": true,
  "risk_tolerance": "medium",
  "missing_roles": ["opener"],
  "seed_analyses": [
    {"track_id": "123", "tier": "anchor", "inferred_role": "peak"}
  ]
}\
"""


def _make_alias_map(tracks: list[Track]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (alias_to_id, id_to_alias) for a Stage 1 track list.

    Aliases are T001, T002, … — short sequential labels that LLMs reproduce
    exactly, replacing opaque Rekordbox IDs that models hallucinate.
    """
    alias_to_id: dict[str, str] = {}
    id_to_alias: dict[str, str] = {}
    for i, track in enumerate(tracks, start=1):
        alias = f"T{i:03d}"
        alias_to_id[alias] = track.track_id
        id_to_alias[track.track_id] = alias
    return alias_to_id, id_to_alias


def _tracks_to_text(
    tracks: list[Track],
    seed_ids: frozenset[str] | None = None,
    id_to_alias: dict[str, str] | None = None,
) -> str:
    lines = []
    for t in tracks:
        display_id = id_to_alias[t.track_id] if id_to_alias is not None else t.track_id
        line = f"ID:{display_id} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key}"
        if t.year is not None:
            line += f" | {t.year}"
        if t.energy is not None:
            line += f" | energy:{t.energy}/8"
        if seed_ids is not None and t.track_id in seed_ids:
            line += " | [seed]"
        lines.append(line)
    return "\n".join(lines)


def _parse_concepts(raw: str) -> list[MixConcept]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()
    raw_for_recovery = raw  # preserve before bracket extraction for truncation fallback
    # Extract the JSON array by finding the outermost [...] bounds.
    # This handles both leading prose and trailing prose from any model.
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        raw = raw[first_bracket : last_bracket + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # Fallback: escape literal newlines/carriage returns inside JSON strings
            # (some models emit unescaped control characters in string values).
            data = json.loads(_fix_json_strings(raw))
        except json.JSONDecodeError:
            # Last resort: thinking model exhausted token budget mid-response, leaving
            # the outer array unclosed. rfind("]") above finds a stray inner "]" and
            # clips the string before any complete object's "}", so we must use the
            # pre-extraction string here to recover whatever complete objects exist.
            data = _extract_complete_objects(raw_for_recovery)
    return [MixConcept(**item) for item in data]


def _parse_intent_brief(
    raw: str,
    seed_tracks: list[Track],
    bpm_range: tuple[float, float],
) -> IntentBrief:
    """Parse Stage 0 LLM output into an IntentBrief.

    Falls back gracefully: unknown tier → supporting, missing tracks → supporting.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    # Extract outermost {...}
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        raw = raw[first : last + 1]

    try:
        data: dict[str, object] = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    valid_tiers = {"anchor", "supporting", "optional"}
    valid_roles = {
        "opener",
        "world_setter",
        "groove_locker",
        "early_hook",
        "builder",
        "connector",
        "pivot",
        "pressure",
        "lift",
        "vocal_moment",
        "texture_change",
        "cleanser",
        "risk",
        "weapon",
        "peak",
        "post_peak",
        "resolution",
        "closer",
        "utility",
        "unknown",
    }
    valid_risk = {"low", "medium", "high"}

    analyses_raw = data.get("seed_analyses", [])
    analysed_ids: dict[str, SeedAnalysis] = {}
    for item in analyses_raw if isinstance(analyses_raw, list) else []:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("track_id", ""))
        tier_raw = str(item.get("tier", "supporting"))
        role_raw = str(item.get("inferred_role", "unknown"))
        # cast: mypy cannot narrow str → Literal after set membership test
        tier = cast("SeedTier", tier_raw if tier_raw in valid_tiers else "supporting")
        role = cast("SetRole", role_raw if role_raw in valid_roles else "unknown")
        drop_cost = {"anchor": 0.1, "supporting": 0.5, "optional": 0.9}.get(tier, 0.5)
        analysed_ids[tid] = SeedAnalysis(
            track_id=tid,
            tier=tier,
            inferred_role=role,
            drop_cost=drop_cost,
        )

    # Ensure every seed has an analysis (fill missing with 'supporting')
    analyses: list[SeedAnalysis] = []
    for t in seed_tracks:
        if t.track_id in analysed_ids:
            analyses.append(analysed_ids[t.track_id])
        else:
            analyses.append(
                SeedAnalysis(
                    track_id=t.track_id,
                    tier="supporting",
                    inferred_role="unknown",
                    drop_cost=0.5,
                )
            )

    missing_roles_raw = data.get("missing_roles", [])
    missing_roles_strs = [
        cast("SetRole", r)
        for r in (missing_roles_raw if isinstance(missing_roles_raw, list) else [])
        if isinstance(r, str) and r in valid_roles
    ]

    risk_raw = str(data.get("risk_tolerance", "medium"))
    risk_tolerance = cast("RiskTolerance", risk_raw if risk_raw in valid_risk else "medium")

    is_coherent = bool(data.get("is_coherent_set", True))
    overall_vibe = str(data.get("overall_vibe", "Intent unclear."))

    return IntentBrief(
        overall_vibe=overall_vibe,
        energy_shape="unclear",  # Stage 0 LLM doesn't compute this; deterministic step does
        risk_tolerance=risk_tolerance,
        is_coherent_set=is_coherent,
        seed_analyses=analyses,
        missing_roles=missing_roles_strs,
        strong_adjacencies=[],  # populated later by deterministic step
        bpm_range=bpm_range,
    )


async def stage0_intent_brief(
    seed_tracks: list[Track],
    seed_track_ids: list[str],
    state: CascadeState,
    bpm_range: tuple[float, float],
) -> IntentBrief:
    """Run Stage 0 intent extraction using the free provider cascade.

    Falls back to a deterministic-only IntentBrief if the LLM call fails or
    the seed set is too small (<= 5 tracks) for meaningful classification.

    The returned brief always has energy_shape and strong_adjacencies populated
    by the deterministic pass regardless of LLM success.
    """
    # Avoid circular import — playlist_mode imports from llm
    from mixlab.playlist_mode import compute_deterministic_intent

    tracks_by_id = {t.track_id: t for t in seed_tracks}
    det_brief = compute_deterministic_intent(seed_track_ids, tracks_by_id)

    if len(seed_tracks) <= 5:
        # Too few tracks for LLM classification to be reliable
        return det_brief

    prompt = f"Seed playlist ({len(seed_tracks)} tracks in original order):\n" + _tracks_to_text(
        seed_tracks, seed_ids=frozenset(seed_track_ids)
    )

    for _ in range(len(_CASCADE)):
        provider = _CASCADE[state.index]
        try:
            result = await provider(prompt, system=_STAGE0_SYSTEM)
            if result is None:
                state.index = (state.index + 1) % len(_CASCADE)
                continue
            if not result.strip():
                raise ValueError(f"Provider {provider.__name__} returned empty content for Stage 0.")
            llm_brief = _parse_intent_brief(result, seed_tracks, bpm_range)
            # Merge: use LLM tiers but deterministic shape + adjacencies
            llm_brief.energy_shape = det_brief.energy_shape
            llm_brief.strong_adjacencies = det_brief.strong_adjacencies
            llm_brief.bpm_range = det_brief.bpm_range
            state.consecutive_failures = 0
            return llm_brief
        except Exception as exc:  # noqa: BLE001 — cascade
            print(f"Stage 0 provider {provider.__name__} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            state.consecutive_failures += 1
            state.index = (state.index + 1) % len(_CASCADE)

    # All providers failed — fall back to deterministic
    print("Stage 0 LLM failed — using deterministic intent brief.", file=sys.stderr)
    state.consecutive_failures = 0  # reset so later Stage 1 calls get a fresh start
    return det_brief


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
        max_tokens=_THINKING_MODEL_MAX_TOKENS,
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


_OPENROUTER_BASE = "https://openrouter.ai/api"
_OPENROUTER_REFERER = "https://openclaw.local"


async def _call_openrouter(model: str, prompt: str, system: str) -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    headers_extra = {"HTTP-Referer": _OPENROUTER_REFERER}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **headers_extra,
    }
    async with httpx.AsyncClient(timeout=_STAGE1_TIMEOUT) as client:
        resp = await client.post(f"{_OPENROUTER_BASE}/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return _strip_thinking(str(resp.json()["choices"][0]["message"]["content"]))


async def _try_openrouter_free(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    return await _call_openrouter("openrouter/free", prompt, system)


async def _try_openrouter_mistral_small(prompt: str, system: str = _STAGE1_SYSTEM) -> str | None:
    return await _call_openrouter("mistralai/mistral-small", prompt, system)


# Stage 1 free providers only — Anthropic is Stage 2 / paid.
_Stage1Provider = Callable[..., Coroutine[Any, Any, str | None]]
_CASCADE: list[_Stage1Provider] = [
    _try_groq,
    _try_gemini,
    _try_mistral,
    _try_openrouter_free,
    _try_openrouter_mistral_small,
]


@dataclass
class CascadeState:
    """Shared mutable state threaded through all Stage 1 calls in a single run."""

    index: int = 0
    consecutive_failures: int = field(default=0)


def make_cascade_state() -> CascadeState:
    return CascadeState()


def _print_stage1_provider_summary(
    provider_name: str,
    genre: str,
    input_track_count: int,
    parsed: list[MixConcept],
    cleaned: list[MixConcept],
    kept: list[MixConcept],
) -> None:
    print(
        f"Stage 1 provider: {provider_name} | genre={genre} | input={input_track_count} tracks | "
        f"parsed={len(parsed)} | cleaned={len(cleaned)} | kept={len(kept)}"
    )
    for raw_concept, cleaned_concept in zip(parsed, cleaned, strict=False):
        status = "kept" if len(cleaned_concept.track_ids) >= MIN_SHORTLIST_TRACKS else "dropped (<8)"
        print(
            f"  - {cleaned_concept.title} | raw={len(raw_concept.track_ids)} | "
            f"kept={len(cleaned_concept.track_ids)} | {status}"
        )
    if not cleaned:
        print("  - no concepts returned")


def _print_stage1_provider_attempt(provider_name: str, genre: str, input_track_count: int) -> None:
    print(f"Stage 1 trying provider: {provider_name} | genre={genre} | input={input_track_count} tracks")


async def _call_stage1_once(
    tracks: list[Track],
    genre: str,
    state: CascadeState,
    custom: bool = False,
    seed_ids: frozenset[str] | None = None,
) -> list[MixConcept]:
    alias_to_id, id_to_alias = _make_alias_map(tracks)
    prompt = f"Genre: {genre}\n\nTracks:\n{_tracks_to_text(tracks, seed_ids=seed_ids, id_to_alias=id_to_alias)}"
    system = _STAGE1_SYSTEM_PLAYLIST if seed_ids is not None else _STAGE1_SYSTEM_CUSTOM if custom else _STAGE1_SYSTEM

    for _ in range(len(_CASCADE)):
        provider = _CASCADE[state.index]
        try:
            _print_stage1_provider_attempt(provider.__name__, genre, len(tracks))
            result = await provider(prompt, system=system)
            if result is None:  # provider not configured — skip silently, no failure counted
                state.index = (state.index + 1) % len(_CASCADE)
                continue
            if not result.strip():  # provider returned empty content — treat as a failure
                raise ValueError(f"Provider {provider.__name__} returned empty content.")
            concepts = _parse_concepts(result)
            # Map aliases back to real IDs. Aliases not in the map are hallucinated — drop them.
            cleaned = [
                MixConcept(
                    title=c.title,
                    mood=c.mood,
                    track_ids=[alias_to_id[a] for a in c.track_ids if a in alias_to_id],
                )
                for c in concepts
            ]
            kept = [c for c in cleaned if len(c.track_ids) >= MIN_SHORTLIST_TRACKS]
            _print_stage1_provider_summary(provider.__name__, genre, len(tracks), concepts, cleaned, kept)
            state.consecutive_failures = 0
            return kept
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
    cluster: list[Track],
    genre: str,
    state: CascadeState,
    custom: bool = False,
    seed_ids: frozenset[str] | None = None,
) -> list[MixConcept]:
    chunk_size = _MAX_TRACKS_PER_CALL_CUSTOM if custom else _MAX_TRACKS_PER_CALL
    if len(cluster) <= chunk_size:
        return await _call_stage1_once(cluster, genre, state, custom=custom, seed_ids=seed_ids)

    concepts: list[MixConcept] = []
    for i in range(0, len(cluster), chunk_size):
        concepts.extend(
            await _call_stage1_once(cluster[i : i + chunk_size], genre, state, custom=custom, seed_ids=seed_ids)
        )
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


def _format_canvas_section(canvas: MixCanvas, tracks_by_id: dict[str, Track]) -> str:
    """Build the compact canvas header block for the Stage 2 prompt."""

    def ids_block(ids: list[str]) -> str:
        return " ".join(f"ID:{i}" for i in ids) if ids else "none"

    r = canvas.roles
    c = canvas.contrast
    lines = [
        f"[Canvas {canvas.canvas_id} | novelty:{canvas.score.novelty:.2f}]",
        f"Core: {ids_block(canvas.core_track_ids)}",
    ]
    bridge_str = ids_block(canvas.bridge_track_ids)
    wild_str = ids_block(canvas.wildcard_track_ids)
    if canvas.bridge_track_ids or canvas.wildcard_track_ids:
        lines.append(f"Bridge: {bridge_str} | Wildcard: {wild_str}")

    role_parts = []
    if r.opener:
        role_parts.append(f"Opener: {ids_block(r.opener)}")
    if r.groove_locker:
        role_parts.append(f"Groove-locker: {ids_block(r.groove_locker)}")
    if r.builder:
        role_parts.append(f"Builder: {ids_block(r.builder)}")
    if r.peak:
        role_parts.append(f"Peak: {ids_block(r.peak)}")
    if r.pivot:
        role_parts.append(f"Pivot: {ids_block(r.pivot)}")
    if r.closer:
        role_parts.append(f"Closer: {ids_block(r.closer)}")
    if role_parts:
        lines.append(" | ".join(role_parts))

    contrast_parts = []
    if c.vocal_moments:
        contrast_parts.append(f"Vocal: {ids_block(c.vocal_moments)}")
    if c.texture_changes:
        contrast_parts.append(f"Texture: {ids_block(c.texture_changes)}")
    if c.darker_turns:
        contrast_parts.append(f"Darker: {ids_block(c.darker_turns)}")
    if c.brighter_lifts:
        contrast_parts.append(f"Brighter: {ids_block(c.brighter_lifts)}")
    if contrast_parts:
        lines.append(" | ".join(contrast_parts))

    if canvas.risk_notes:
        lines.append(f"Risks: {', '.join(canvas.risk_notes)}")

    # Full track listing after header
    track_lines: list[str] = []
    for tid in canvas.source_concept.track_ids:
        t = tracks_by_id.get(tid)
        if t is None:
            continue
        pool_label = ""
        if tid in canvas.bridge_track_ids:
            pool_label = " [bridge]"
        elif tid in canvas.wildcard_track_ids:
            pool_label = " [wildcard]"
        extras: list[str] = []
        if t.year is not None:
            extras.append(str(t.year))
        if t.label:
            extras.append(t.label)
        if t.remixer:
            extras.append(f"remix by {t.remixer}")
        if t.energy is not None:
            extras.append(f"energy:{t.energy}/8")
        if t.tags:
            extras.append(", ".join(t.tags))
        if t.enrichment_confidence == "low":
            extras.append("[unverified]")
        extra_str = " | " + " | ".join(extras) if extras else ""
        track_lines.append(
            f"  ID:{tid} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key} | {t.genre}{pool_label}{extra_str}"
        )

    header = "\n".join(lines)
    candidates = "Candidates:\n" + "\n".join(track_lines) if track_lines else ""
    return f"{header}\n{candidates}"


_STAGE2_CANVAS_RULES = """\
\nCANVAS USAGE RULES (for this run):\n\
- Core tracks: use freely.\n\
- Bridge tracks [bridge]: require role-aware justification — suitable for opener, pivot, reset, or closer where BPM deviation is intentional.\n\
- Wildcard tracks [wildcard]: require explicit creative reason; use only if concept-defining.\n\
- Role candidates shown in the canvas header (Opener:, Builder:, Peak:, etc.) are derived from technical analysis — energy score, BPM proximity, Camelot distance. They are hints, not assignments. Override them freely when your DJ instinct says a track serves a different role better, and briefly say why in the report.\n\
- For bridge or wildcard tracks, "creative reason" means a specific structural role (e.g. "opener with deliberate BPM drop-in", "pivot that earns a key reset", "closer whose energy arc signals finality") — not "interesting track" or "good contrast".\n\
- If a canvas shows risk notes, address the noted weakness in your selection or energy path.\n\
- If a canvas's risk notes describe structural problems you cannot overcome with track selection (e.g. weak closer pool with no resolution candidate, all-high-energy with no viable dynamic arc), you may skip that canvas rather than force a weak concept. You are not obligated to produce a concept from every canvas.\n\
- Not every mix needs every role.\n\
- Harmonic and BPM compatibility are helpers, not constraints.\n\
- For transitions involving bridge or wildcard tracks, state the specific mechanism that makes it survivable.\
"""


def select_shortlists_for_stage2(shortlists: list[MixConcept]) -> list[MixConcept]:
    """Select up to _STAGE2_CAP shortlists for Stage 2, sampling randomly from the top candidates by pool size.

    Always picks from the _STAGE2_CANDIDATE_POOL largest shortlists so every run has a chance of variety
    while still favouring well-stocked pools.
    """
    if len(shortlists) <= _STAGE2_CAP:
        return shortlists
    candidates = sorted(shortlists, key=lambda s: len(s.track_ids), reverse=True)[:_STAGE2_CANDIDATE_POOL]
    return random.sample(candidates, min(_STAGE2_CAP, len(candidates)))


def select_shortlists_for_playlist_stage2(
    shortlists: list[MixConcept],
    seed_ids: frozenset[str],
) -> list[MixConcept]:
    ranked = sorted(
        shortlists,
        key=lambda shortlist: sum(1 for track_id in shortlist.track_ids if track_id in seed_ids),
        reverse=True,
    )
    return ranked[:_STAGE2_CAP]


def validate_stage2_output(
    concepts: list[MixConcept],
    canvases: list[MixCanvas],
    tracks_by_id: dict[str, Track],
    played_ids: set[str],
    denylist_ids: set[str],
    allow_played: bool = False,
    genre: str = "_default",
) -> list[str]:
    """Warn-only post-Stage-2 validation. Returns a list of warning strings (never raises)."""
    from collections import Counter

    warnings: list[str] = []
    bridge_ids: set[str] = {tid for c in canvases for tid in c.bridge_track_ids}
    wildcard_ids: set[str] = {tid for c in canvases for tid in c.wildcard_track_ids}

    min_count, max_count = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])

    for concept in concepts:
        label = f"[{concept.title}]"
        for tid in concept.track_ids:
            if tid not in tracks_by_id:
                warnings.append(f"{label} track ID {tid} not found in library")
            if tid in denylist_ids:
                warnings.append(f"{label} track ID {tid} is denylisted")
            if not allow_played and tid in played_ids:
                warnings.append(f"{label} track ID {tid} has been played")

        n = len(concept.track_ids)
        if n < min_count:
            warnings.append(f"{label} only {n} tracks — minimum is {min_count} for this genre")
        if n > max_count:
            warnings.append(f"{label} {n} tracks — maximum is {max_count} for this genre")

        seq = [tracks_by_id[tid] for tid in concept.track_ids if tid in tracks_by_id]
        artist_counts = Counter(t.artist for t in seq)
        for artist, count in artist_counts.items():
            if count >= 3:
                warnings.append(f"{label} artist '{artist}' appears {count} times")

        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            bpm_jump = abs(a.bpm - b.bpm)
            if bpm_jump > 15:
                warnings.append(
                    f"{label} BPM jump {bpm_jump:.1f} between {a.artist} — {a.title} and {b.artist} — {b.title}"
                )
            cam_dist = camelot_distance(a.camelot_key, b.camelot_key)
            if cam_dist > 4:
                warnings.append(f"{label} Camelot jump {cam_dist} between {a.camelot_key} and {b.camelot_key}")

        # Bridge/wildcard tracks used without is_risky flag
        transition_map = {(t.from_id, t.to_id): t for t in concept.transitions}
        for i in range(len(seq) - 1):
            to_id = seq[i + 1].track_id
            if to_id in bridge_ids or to_id in wildcard_ids:
                tr = transition_map.get((seq[i].track_id, to_id))
                pool = "bridge" if to_id in bridge_ids else "wildcard"
                if tr is None or (not tr.is_risky and tr.risk_type == ""):
                    warnings.append(f"{label} {pool} track ID {to_id} used without a justified transition")

    return warnings


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
- The opener plays to a room that isn't committed yet. It can work as ambient architecture or as a restrained, \
low-slung groove — the point is headroom, not drift. Reward attention without requiring it, and avoid \
telegraphing the set too early. A track that demands full engagement in its first 32 bars is the wrong \
opener regardless of its quality.
- The closer must signal its own finality before it arrives. The room should feel the set ending. The \
default closer resolves — it has weight, sufficient outro length to mix out of cleanly, and leaves the \
room with a feeling rather than a question. A track whose energy rises continuously into its final bars \
can close a set only if its authority is strong enough to signal finality without resolution. If you are \
not certain it carries that weight, it is not the closer.
- Each concept should have a thesis — not just a mood, but an intention. What does this set ask of the \
room? The creative brief must answer this.
- Assign each track a role. Choose from: opener, world-setter, groove-locker, early-hook, builder, \
connector, pivot, pressure, lift, vocal-moment, texture-change, cleanser, risk, weapon, peak, \
post-peak, resolution, closer, utility. A track may carry more than one role. Every track must have \
at least one — no roleless inclusions.
- Before finalising the order, choose an explicit energy path for the concept: Slow Climb (controlled \
build throughout), Wave (builds, releases, builds again — often the strongest default), Plateau With \
Detail (holds a groove level, creates interest through texture and contrast), Double Peak (first peak \
grabs attention, second stronger payoff), Front-Loaded Hook (immediate engagement then settles into the \
journey), Dark to Light (starts moody or tense, opens emotionally later), Light to Dark (starts \
accessible, grows heavier or stranger). The chosen shape must be visible in the track sequence.
- Think in sections. A coherent mix divides into five: Invitation (open the world, hook the listener, \
avoid full peak — opener, world-setter, early-hook roles), Groove Lock (settle the rhythm, build trust \
— groove-locker, builder, connector roles), Development (add contrast, increase tension or pressure — \
pivot, pressure, lift, vocal-moment, texture-change roles), Peak/Payoff (strongest moment, not \
necessarily the loudest — weapon, peak roles), Resolution (stabilise and close with intent — post-peak, \
resolution, closer roles). Assign every track to a section before deciding the final order.
- Design an intentional energy curve that follows the chosen energy path. This need not be a single arc \
— consider double peaks, plateau-and-release structures, or a false resolution before the final push. \
The shape should feel inevitable in retrospect, not predictable in real time.
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
  "name_reason": "...",
  "mood": "...",
  "track_ids": ["id1", "id2", ...],
  "transitions": [
    {"from_id": "id1", "to_id": "id2", "is_risky": false, "risk_type": ""},
    {"from_id": "id2", "to_id": "id3", "is_risky": true,  "risk_type": "chapter_pivot"}
  ],
  "arc_type": "...",
  "report": "..."
}

transitions: one entry per consecutive pair in track_ids (len(track_ids) - 1 entries).
is_risky: true if the move is a notable harmonic or energy risk.
risk_type: one of "chapter_pivot" | "peak_impact" | "deliberate_reset" | "closer_move" \
           | "cut_only" | "low_tonal_risk" | "" (empty string when is_risky=false).
"cut_only" means: risky, with no mechanism that earns it — just a hard cut.
arc_type: one of "plateau" | "wave" | "progressive-build" | "build-and-drop" | "double-peak" \
          | "sustained-pressure" | "front-loaded" | "dark-to-light" | "light-to-dark" \
          | "narrative" | "abstract-journey". Pick the value that best describes this \
concept's overall arc. Structural arcs (plateau, wave, progressive-build, build-and-drop, \
double-peak, sustained-pressure, front-loaded) describe the energy shape; directional arcs \
(dark-to-light, light-to-dark) describe mood travel; narrative describes a chapter-based \
story; abstract-journey describes a non-linear, impressionistic set. The chosen arc_type \
should be visible in the track sequence and consistent with the energy path you describe \
in the report.

Give each concept a short, evocative name (2–4 words max) — not the pool name from Stage 1. \
Avoid generic [Adjective][Noun] patterns (e.g. "Warm Gravity", "Committed Floor", "Orbital Descent" are bad). \
Good names are oblique, specific, or surprising — they suggest a place, a feeling, a moment, or a cultural \
reference rather than describing the music. Think how a DJ would title a mix: "Late Latitude", "Fever", \
"Interzone", "Red Light", "The Slow Hours". The name should make someone curious, not nod in recognition. \
Add a "name_reason" field: one short sentence (max 15 words) explaining WHY this specific title was \
chosen — what literal or metaphorical quality in the track list earns the name. This must reference \
the title explicitly and connect it to something audible (BPM arc, key centre, a specific track, \
a mood shift). Do not repeat the mood description. Example: if title is "Slow Burn", write \
"Ten tracks of locked 122–127 BPM tension with no release until the final two." Must be verifiable \
from the track list; do not invent.
The track_ids must be the final selected tracks in play order.
The "report" value must be a single string (with \\n for line breaks) in this exact format:

CONCEPT: [title]

[1–2 sentences: thesis — what this set asks of the room.]

Track order:
[For each track in play order, one line:]
N. Artist — Title [Key · BPM] | Role: [role] | Why: [one short phrase] | Risk: [one short phrase or "none"]

Assumptions: [only if material — [unverified] tracks, vocal clash, tight blend window. One line each. Omit section if nothing material.]

Role options: opener, world-setter, groove-locker, early-hook, builder, connector, pivot, pressure, lift, \
vocal-moment, texture-change, cleanser, risk, weapon, peak, post-peak, resolution, closer, utility.
Risk: describe the transition risk into this track (not out of it). "none" if clean.
Why: why this track at this moment — one phrase, no full sentences needed.

Be opinionated, musical, and honest. Peer-to-peer, no marketing language, no filler. When choosing \
between sounding clever and being right, be right.

Respond ONLY with the JSON array.\
"""

# Named targets for the two sections that differ between standard and playlist prompts.
# Defined as constants so that the .replace() calls below can be asserted against them —
# if either drifts from _STAGE2_SYSTEM, the assert fires at import time instead of
# silently producing a prompt with the old text.
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
    "Stage 2 playlist prompt: select section not found — _STAGE2_SELECT_STANDARD drifted from _STAGE2_SYSTEM"
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
assert _tmp != _STAGE2_SYSTEM_PLAYLIST, (
    "Stage 2 playlist prompt: produce section not found — _STAGE2_PRODUCE_STANDARD drifted from _STAGE2_SYSTEM"
)


def _make_selection_system(base: str) -> str:
    """Strip report schema field and format instructions from a Stage 2 system prompt.

    The report is generated in a separate pass so the selection call stays under 8K
    output tokens and completes reliably within the API timeout.
    """
    base = base.replace('\n  "report": "..."', "")
    marker = 'The "report" value must be a single string'
    idx = base.find(marker)
    if idx != -1:
        base = (
            base[:idx].rstrip() + "\n\nRespond ONLY with the JSON array. "
            "Output the opening [ immediately — no analysis, preamble, or explanation before it."
        )
    return base


_STAGE2_SYSTEM_SELECTION: str = _make_selection_system(_STAGE2_SYSTEM)
_STAGE2_SYSTEM_PLAYLIST_SELECTION: str = _make_selection_system(_STAGE2_SYSTEM_PLAYLIST)

_STAGE2_REPORT_SYSTEM = """\
You are a world-class DJ and mix curator with deep real-world club experience. The track selection \
and play order for this concept have already been decided. Your job is to write the mix report only.

Write the report in this exact format:

CONCEPT: [concept_title]

Energy path: [one of: Slow Climb | Wave | Plateau With Detail | Double Peak | Front-Loaded Hook | Dark to Light | Light to Dark]

[1–2 sentences: thesis — what this set asks of the room.]

Sections:
Invitation: tracks [N–N]
Groove Lock: tracks [N–N]
Development: tracks [N–N]
Peak/Payoff: tracks [N]
Resolution: tracks [N–N]

Track order:
[For each track in play order, one line:]
N. Artist — Title [Key · BPM] | Role: [role] | Why: [one short phrase] | Risk: [one short phrase or "none"]

Opener: [one sentence — why this track opens this mix, using the opener's specific qualities]
Closer: [one sentence — why this track closes this mix and what aftertaste it leaves]

Excluded: [Artist — Title — reason for cutting]; [repeat per cut track. Omit this section entirely if nothing was cut from the candidate pool.]

Assumptions: [only if material — [unverified] tracks, vocal clash, tight blend window. One line each. \
Omit section if nothing material.]

Role options: opener, world-setter, groove-locker, early-hook, builder, connector, pivot, pressure, \
lift, vocal-moment, texture-change, cleanser, risk, weapon, peak, post-peak, resolution, closer, utility.
Risk: describe the transition risk into this track (not out of it). "none" if clean.
Why: why this track at this moment — one phrase, no full sentences needed.

Be opinionated, musical, and honest. Peer-to-peer, no marketing language, no filler.

Return ONLY the report text. No JSON, no markdown fences, no preamble.\
"""

_PLAYLIST_MINIMUM_SEED_RATIO = 0.6


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


def _extract_complete_objects(raw: str) -> list[dict[str, object]]:
    """Extract complete JSON objects from a truncated JSON array.

    Used when a thinking model exhausts its token budget mid-output, leaving the outer
    array unclosed. Returns however many well-formed objects were found before the
    truncation point; raises ValueError if none, so the cascade can fall through.
    """
    obj_array_match = re.search(r"\[\s*\{", raw)
    start = obj_array_match.start() if obj_array_match else raw.find("[")
    if start == -1:
        raise ValueError("No JSON array start found in truncated response")
    objects: list[dict[str, object]] = []
    depth = 0
    obj_start: int | None = None
    in_string = False
    escaped = False
    for i, ch in enumerate(raw[start + 1 :], start + 1):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_str = raw[obj_start : i + 1]
                try:
                    obj = json.loads(obj_str)
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(_fix_json_strings(obj_str))
                    except json.JSONDecodeError:
                        obj_start = None
                        continue
                if isinstance(obj, dict):
                    objects.append(obj)
                obj_start = None
    if not objects:
        raise ValueError("No complete JSON objects could be extracted from truncated response")
    return objects


def _parse_curated_concepts(raw: str, valid_ids: set[str]) -> tuple[list[MixConcept], str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()
    if not raw:
        raise ValueError("Stage 2 returned empty content — Anthropic API may have truncated the response.")
    raw_for_recovery = raw
    # Prefer [{ as array-of-objects start to skip prose containing [word] patterns.
    obj_array_match = re.search(r"\[\s*\{", raw)
    first_bracket = obj_array_match.start() if obj_array_match else raw.find("[")
    last_bracket = raw.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        raw = raw[first_bracket : last_bracket + 1]
    try:
        data: list[dict[str, object]] = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(_fix_json_strings(raw))
        except json.JSONDecodeError:
            data = _extract_complete_objects(raw_for_recovery)

    # Diagnostic: model signals the pool is too thin to produce 3+ distinct concepts.
    if len(data) == 1 and "diagnostic" in data[0] and "title" not in data[0]:
        return [], f"POOL DIAGNOSTIC\n{data[0]['diagnostic']}"

    curated: list[MixConcept] = []
    report_parts: list[str] = []

    def _normalise_id(raw: object) -> str:
        s = str(raw)
        return s[3:] if s.startswith("ID:") else s

    for item in data:
        raw_ids = item.get("track_ids")
        track_ids = [
            nid for tid in (raw_ids if isinstance(raw_ids, list) else []) if (nid := _normalise_id(tid)) in valid_ids
        ]
        if len(track_ids) < _MIN_CONCEPT_TRACKS:
            continue
        raw_transitions = item.get("transitions", [])
        transitions: list[Transition] = []
        for tr in raw_transitions if isinstance(raw_transitions, list) else []:
            if isinstance(tr, dict):
                transitions.append(
                    Transition(
                        from_id=_normalise_id(tr.get("from_id", "")),
                        to_id=_normalise_id(tr.get("to_id", "")),
                        is_risky=bool(tr.get("is_risky", False)),
                        risk_type=str(tr.get("risk_type", "")),
                    )
                )
        name_reason = str(item.get("name_reason", ""))
        arc_type = _extract_arc_type(item.get("arc_type"))
        curated.append(
            MixConcept(
                title=str(item.get("title", "")),
                mood=str(item.get("mood", "")),
                track_ids=track_ids,
                transitions=transitions,
                name_reason=name_reason,
                arc_type=arc_type,
            )
        )
        report_parts.append(str(item.get("report", "")))

    return curated, "\n\n---\n\n".join(report_parts)


def _extract_arc_type(raw: object) -> ArcType | None:
    """Defensively extract an arc_type value from Stage 2 JSON output.

    Returns None when the field is missing or carries an unknown value, so
    downstream consumers can fall back gracefully. Accepts mild formatting
    drift (case, underscores instead of hyphens).
    """
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower().replace("_", "-")
    if normalized in _ARC_TYPE_VALUES:
        return cast(ArcType, normalized)
    return None


# Risk-type enum values from Transition.risk_type mapped to human-readable mechanism labels.
# Empty string and unknown values yield no mechanism (bullet line is skipped).
_RISK_TYPE_LABELS: dict[str, str] = {
    "chapter_pivot": "chapter pivot",
    "peak_impact": "peak impact",
    "deliberate_reset": "deliberate reset",
    "closer_move": "closer move",
    "cut_only": "cut only",
    "low_tonal_risk": "low tonal risk",
}


def _match_canvas_for_concept(concept: MixConcept, canvases: list[MixCanvas]) -> MixCanvas | None:
    """Pick the canvas whose full pool covers the most of the concept's track IDs.

    Returns None when no canvas shares any tracks with the concept.
    """
    if not canvases:
        return None
    concept_ids = set(concept.track_ids)
    best: MixCanvas | None = None
    best_overlap = 0
    for canvas in canvases:
        pool = set(canvas.core_track_ids) | set(canvas.bridge_track_ids) | set(canvas.wildcard_track_ids)
        overlap = len(concept_ids & pool)
        if overlap > best_overlap:
            best_overlap = overlap
            best = canvas
    return best


def _format_bold_moves(
    concept: MixConcept,
    canvas: MixCanvas,
    tracks_by_id: dict[str, Track],
) -> str:
    """Build the 'Bold moves:' annotation listing bridge/wildcard tracks in a concept.

    Counts non-core picks and emits per-track bullet lines when an incoming transition
    names a mechanism (Transition.risk_type). Tracks whose incoming transition has no
    risk_type yield only the count, no bullet.
    """
    bridge_ids = set(canvas.bridge_track_ids)
    wildcard_ids = set(canvas.wildcard_track_ids)
    incoming: dict[str, Transition] = {t.to_id: t for t in concept.transitions}

    bridge_hits: list[tuple[str, str]] = []  # (track_id, mechanism_label)
    wildcard_hits: list[tuple[str, str]] = []
    for tid in concept.track_ids:
        if tid in bridge_ids:
            mech = _RISK_TYPE_LABELS.get(incoming.get(tid, Transition(from_id="", to_id=tid)).risk_type, "")
            bridge_hits.append((tid, mech))
        elif tid in wildcard_ids:
            mech = _RISK_TYPE_LABELS.get(incoming.get(tid, Transition(from_id="", to_id=tid)).risk_type, "")
            wildcard_hits.append((tid, mech))

    if not bridge_hits and not wildcard_hits:
        return "Bold moves: none"

    parts: list[str] = []
    if bridge_hits:
        parts.append(f"{len(bridge_hits)} bridge")
    if wildcard_hits:
        parts.append(f"{len(wildcard_hits)} wildcard")
    lines = [f"Bold moves: {', '.join(parts)}"]
    for tid, mech in bridge_hits:
        if not mech:
            continue
        t = tracks_by_id.get(tid)
        if t is None:
            continue
        lines.append(f"- {t.artist} — {t.title} (bridge): {mech}")
    for tid, mech in wildcard_hits:
        if not mech:
            continue
        t = tracks_by_id.get(tid)
        if t is None:
            continue
        lines.append(f"- {t.artist} — {t.title} (wildcard): {mech}")
    return "\n".join(lines)


def _append_bold_moves_to_report(
    report: str,
    concept: MixConcept,
    canvases: list[MixCanvas] | None,
    tracks_by_id: dict[str, Track],
) -> str:
    """Append a 'Bold moves:' annotation block to a concept report.

    Skips when no canvases are available (e.g. playlist mode) — there is no bridge/wildcard
    classification without a canvas.
    """
    if not canvases:
        return report
    canvas = _match_canvas_for_concept(concept, canvases)
    if canvas is None:
        return report
    annotation = _format_bold_moves(concept, canvas, tracks_by_id)
    return f"{report.rstrip()}\n\n{annotation}"


def _playlist_retention_stats(
    concept: MixConcept,
    seed_track_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    seed_id_set = set(seed_track_ids)
    retained_ids = [track_id for track_id in concept.track_ids if track_id in seed_id_set]
    dropped_ids = [track_id for track_id in seed_track_ids if track_id not in retained_ids]
    added_ids = [track_id for track_id in concept.track_ids if track_id not in seed_id_set]
    return retained_ids, dropped_ids, added_ids


def _minimum_playlist_seed_retention(seed_count: int, intent_brief: IntentBrief | None = None) -> int:
    """Return the minimum number of seed tracks Stage 2 must retain.

    With an IntentBrief: floor is per-tier (75% of anchors + 40% of supporting).
    Without: falls back to the flat 60% floor.
    """
    if seed_count <= 0:
        return 0
    if intent_brief is None or not intent_brief.seed_analyses:
        return math.ceil(seed_count * _PLAYLIST_MINIMUM_SEED_RATIO)
    anchor_count = len(intent_brief.anchor_ids)
    supporting_count = len(intent_brief.supporting_ids)
    anchor_floor = math.ceil(anchor_count * 0.75)
    supporting_floor = math.ceil(supporting_count * 0.40)
    return anchor_floor + supporting_floor


def _pair_consecutive(a: str, b: str, ids: list[str]) -> bool:
    """Return True if b immediately follows a in ids."""
    return any(ids[i] == a and ids[i + 1] == b for i in range(len(ids) - 1))


def _compute_practicality_score(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
    intent_brief: IntentBrief | None,
) -> DJPracticalityScore:
    """Compute a DJ Practicality Score deterministically from a concept's track sequence."""
    track_sequence = [tracks_by_id[tid] for tid in concept.track_ids if tid in tracks_by_id]
    n = len(track_sequence)

    # bpm_smoothness: how even the consecutive BPM steps are
    if n < 3:
        bpm_smoothness = 1.0
    else:
        bpm_deltas = [abs(track_sequence[i + 1].bpm - track_sequence[i].bpm) for i in range(n - 1)]
        std = statistics.stdev(bpm_deltas)
        bpm_smoothness = max(0.0, 1.0 - std / 10.0)

    # harmonic_ratio: fraction of consecutive pairs that are Camelot-compatible (distance ≤ 1)
    if n < 2:
        harmonic_ratio = 1.0
    else:
        total_pairs = n - 1
        compatible = sum(
            1
            for i in range(total_pairs)
            if camelot_distance(track_sequence[i].camelot_key, track_sequence[i + 1].camelot_key) <= 1
        )
        harmonic_ratio = compatible / total_pairs

    # risk_justified: penalise transitions annotated is_risky=True with risk_type "cut_only" or ""
    risky = [t for t in concept.transitions if t.is_risky]
    unjustified = [t for t in risky if t.risk_type in ("cut_only", "")]
    risk_justified = 1.0 if not risky else max(0.0, 1.0 - len(unjustified) / len(risky))

    # fragment_preserved: fraction of strong adjacency pairs preserved in sequence
    if intent_brief is None or not intent_brief.strong_adjacencies:
        fragment_preserved = 1.0
    else:
        concept_ids = list(concept.track_ids)
        preserved = sum(
            1
            for frag in intent_brief.strong_adjacencies
            if _pair_consecutive(frag.track_ids[0], frag.track_ids[1], concept_ids)
        )
        fragment_preserved = preserved / len(intent_brief.strong_adjacencies)

    return DJPracticalityScore(
        bpm_smoothness=bpm_smoothness,
        harmonic_ratio=harmonic_ratio,
        risk_justified=risk_justified,
        fragment_preserved=fragment_preserved,
    )


def _score_variant(
    concept: MixConcept,
    seed_track_ids: list[str],
    intent_brief: IntentBrief | None,
    tracks_by_id: dict[str, Track],
) -> CompletionVariant:
    """Compute a CompletionVariant with anchor_retention_rate and practicality_score."""
    concept_id_set = set(concept.track_ids)
    strategy_raw = concept.mood.lower()
    strategy: Literal["practical", "balanced", "adventurous"] = (
        "practical"
        if strategy_raw == "practical"
        else "balanced"
        if strategy_raw == "balanced"
        else "adventurous"
        if strategy_raw == "adventurous"
        else "practical"
    )

    if intent_brief is not None and intent_brief.anchor_ids:
        retained_anchors = sum(1 for aid in intent_brief.anchor_ids if aid in concept_id_set)
        anchor_retention_rate = retained_anchors / len(intent_brief.anchor_ids)
    else:
        retained_seeds = sum(1 for sid in seed_track_ids if sid in concept_id_set)
        anchor_retention_rate = retained_seeds / max(len(seed_track_ids), 1)

    practicality_score = _compute_practicality_score(concept, tracks_by_id, intent_brief)

    return CompletionVariant(
        strategy=strategy,
        concept=concept,
        anchor_retention_rate=anchor_retention_rate,
        practicality_score=practicality_score,
    )


_STRATEGY_PRIORITY: dict[str, int] = {"practical": 0, "balanced": 1, "adventurous": 2}


def _select_best_variant(variants: list[CompletionVariant]) -> CompletionVariant:
    """Return highest-scoring variant; ties broken by practical > balanced > adventurous."""
    return max(
        variants,
        key=lambda v: (v.score, -_STRATEGY_PRIORITY.get(v.strategy, 99)),
    )


def _passes_floor(
    variant: CompletionVariant,
    intent_brief: IntentBrief | None,
    playlist_seed_track_ids: list[str],
    minimum_seed_tracks: int,
) -> bool:
    """Return True if variant meets the per-tier retention floor."""
    concept_ids = set(variant.concept.track_ids)
    if intent_brief is not None and intent_brief.anchor_ids:
        anchor_floor = math.ceil(len(intent_brief.anchor_ids) * 0.75)
        supporting_floor = math.ceil(len(intent_brief.supporting_ids) * 0.40)
        return (
            sum(1 for aid in intent_brief.anchor_ids if aid in concept_ids) >= anchor_floor
            and sum(1 for sid in intent_brief.supporting_ids if sid in concept_ids) >= supporting_floor
        )
    retained = sum(1 for tid in playlist_seed_track_ids if tid in concept_ids)
    return retained >= minimum_seed_tracks


def _format_playlist_track_labels(track_ids: list[str], tracks_by_id: dict[str, Track], limit: int = 8) -> str:
    labels: list[str] = []
    for track_id in track_ids[:limit]:
        track = tracks_by_id.get(track_id)
        if track is None:
            labels.append(track_id)
        else:
            labels.append(f"{track.artist} — {track.title}")
    if len(track_ids) > limit:
        labels.append(f"+ {len(track_ids) - limit} more")
    return ", ".join(labels)


def _rewrite_playlist_report(
    report: str,
    playlist_name: str,
    concept: MixConcept,
    seed_track_ids: list[str],
    tracks_by_id: dict[str, Track],
    rejected_summary: str = "",
) -> str:
    retained_ids, dropped_ids, added_ids = _playlist_retention_stats(concept, seed_track_ids)
    retained_suffix = ""
    if retained_ids:
        retained_suffix = f" ({_format_playlist_track_labels(retained_ids, tracks_by_id)})"

    summary = (
        f"Source playlist: {playlist_name}\n"
        f"Seed tracks retained: {len(retained_ids)}{retained_suffix}.\n"
        f"Seed tracks dropped: {len(dropped_ids)}.\n"
        f"Library tracks added: {len(added_ids)}." + rejected_summary
    )

    marker = "Track order:"
    if marker not in report:
        return report.rstrip() + f"\n\n{summary}"

    prefix, suffix = report.split(marker, 1)
    source_index = prefix.find("\nSource playlist:")
    if source_index != -1:
        prefix = prefix[:source_index]

    return prefix.rstrip() + f"\n\n{summary}\n\n{marker}{suffix}"


def _playlist_variant_label(strategy: str, *, is_winner: bool) -> str:
    base = strategy.upper()
    return f"WINNER - {base}" if is_winner else base


def _label_playlist_report_section(report: str, strategy: str, *, is_winner: bool) -> str:
    label = _playlist_variant_label(strategy, is_winner=is_winner)
    report = report.strip()
    if not report:
        return f"VARIANT: {label}"
    return f"VARIANT: {label}\n\n{report}"


def _retitle_playlist_concept(concept: MixConcept, strategy: str, *, is_winner: bool) -> MixConcept:
    label = _playlist_variant_label(strategy, is_winner=is_winner)
    return concept.model_copy(update={"title": f"{label} - {concept.title}"})


async def _call_stage2_raw(
    prompt: str,
    stage2_system: str,
    stage2_key: str,
    max_tokens: int = 32768,
) -> str:
    """Make the Stage 2 selection HTTP call via Anthropic."""
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
            *[_call_stage2_report_single(c, tracks_by_id, seed_ids, unplayed_ids, stage2_key) for c in concepts]
        )
    )


async def stage2_curate_and_report(
    shortlists: list[MixConcept],
    tracks_by_id: dict[str, Track],
    custom_genre_label: str | None = None,
    custom_genre_sub_genres: list[str] | None = None,
    playlist_name: str | None = None,
    seed_ids: frozenset[str] | None = None,
    seed_track_ids: list[str] | None = None,
    unplayed_ids: set[str] | None = None,
    intent_brief: IntentBrief | None = None,
    used_mix_names: list[str] | None = None,
    canvases: list[MixCanvas] | None = None,
    concept_history: ConceptHistory | None = None,
    debug: bool = False,
) -> tuple[list[MixConcept], str]:
    stage2_key = os.environ.get("ANTHROPIC_API_KEY")
    if not stage2_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — Stage 2 curation requires Anthropic.")
    stage2_model_display = "Claude Sonnet 4.6"

    sections: list[str] = []
    if canvases is not None:
        # Canvas-aware path: build sections from MixCanvas objects
        for canvas in canvases:
            section = _format_canvas_section(canvas, tracks_by_id)
            if section.strip():
                sections.append(section)
    else:
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
                    f"  ID:{tid} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key} | {t.genre}{extra_str}"
                )
            if track_lines:
                sections.append(
                    f"Shortlist: {shortlist.title}\nCharacter: {shortlist.mood}\nCandidates:\n" + "\n".join(track_lines)
                )

    n = len(sections)
    if n == 0:
        raise RuntimeError("Stage 2 received no shortlists with resolvable tracks — nothing to curate.")

    # Recent-concepts block — shown to Stage 2 so it can deliberately diverge from
    # prior runs. Skipped on first-ever run (empty history) or when caller did not
    # provide a history snapshot. See history.format_recent_concepts.
    recent_concepts_block = ""
    if concept_history is not None:
        block = format_recent_concepts(concept_history)
        if block:
            recent_concepts_block = block + "\n\n"

    if playlist_name is not None:
        playlist_seed_track_ids = seed_track_ids or sorted(seed_ids or [])
        minimum_seed_tracks = _minimum_playlist_seed_retention(len(playlist_seed_track_ids), intent_brief)
        # Build intent brief section for injection into prompt
        intent_section = ""
        if intent_brief is not None:
            anchor_labels = (
                ", ".join(
                    f"{tracks_by_id[tid].artist} \u2014 {tracks_by_id[tid].title}"
                    for tid in sorted(intent_brief.anchor_ids)
                    if tid in tracks_by_id
                )
                or "none identified"
            )
            adjacency_hints = (
                "; ".join(
                    f"{tracks_by_id[f.track_ids[0]].artist} \u2014 {tracks_by_id[f.track_ids[0]].title}"
                    f" \u2192 "
                    f"{tracks_by_id[f.track_ids[1]].artist} \u2014 {tracks_by_id[f.track_ids[1]].title}"
                    f" (confidence: {f.confidence:.1f})"
                    for f in intent_brief.strong_adjacencies
                    if len(f.track_ids) >= 2 and f.track_ids[0] in tracks_by_id and f.track_ids[1] in tracks_by_id
                )
                or "none detected"
            )
            missing_roles_str = ", ".join(str(r) for r in intent_brief.missing_roles) or "none identified"
            intent_section = (
                f"DJ INTENT BRIEF\n"
                f"Vibe: {intent_brief.overall_vibe}\n"
                f"Energy shape: {intent_brief.energy_shape}\n"
                f"Risk tolerance: {intent_brief.risk_tolerance}\n"
                f"Anchor tracks (protect these): {anchor_labels}\n"
                f"Missing roles to fill: {missing_roles_str}\n"
                f"Intentional adjacencies (preserve if possible): {adjacency_hints}\n"
                f"Coherent set: {'yes' if intent_brief.is_coherent_set else 'no \u2014 treat as chapters'}\n\n"
            )
        prompt = (
            f"{recent_concepts_block}"
            f"{intent_section}"
            f"Curate three completion variants from the following {n} BPM zone shortlists. "
            f'This is a playlist completion run seeded from the Rekordbox playlist "{playlist_name}".\n\n'
            "Each shortlist below represents one natural BPM zone from the source playlist. "
            "Tracks marked [seed] are from the original playlist; other tracks are library additions.\n\n"
            "Your task:\n"
            "1. Identify the dominant zone(s) — where most seed tracks live.\n"
            "2. You may drop outlier zones if they break set coherence.\n"
            f"3. Retain at least {minimum_seed_tracks} seed tracks total (anchors protected as per brief).\n"
            "4. When two otherwise equally suitable tracks compete, prefer the one marked unplayed.\n\n"
            "Produce EXACTLY THREE concepts (one practical, one balanced, one adventurous) drawing from the combined "
            "candidate pool below. All shortlists together form one pool — do not produce one concept per shortlist.\n\n"
            + "\n\n".join(sections)
        )
    else:
        if canvases is not None:
            prompt = (
                f"{recent_concepts_block}"
                f"Curate a set of mix concepts from the following {n} candidate canvases. "
                f"Produce between 3 and 6 distinct concepts total. Each concept must draw only from tracks within a single canvas.\n\n"
                + "\n\n".join(sections)
            )
        else:
            prompt = (
                f"{recent_concepts_block}"
                f"Curate a set of mix concepts from the following {n} candidate shortlists. "
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

    stage2_system = _STAGE2_SYSTEM_PLAYLIST_SELECTION if playlist_name is not None else _STAGE2_SYSTEM_SELECTION
    if canvases is not None and playlist_name is None:
        stage2_system = stage2_system + _STAGE2_CANVAS_RULES
    _name_dedup_sentinel = 'The name should make someone curious, not nod in recognition. Add a "name_reason" field'
    if used_mix_names:
        assert _name_dedup_sentinel in stage2_system, (
            "Stage 2 name-dedup injection: sentinel not found — naming instruction drifted from expected text"
        )
        names_str = ", ".join(used_mix_names)
        stage2_system = stage2_system.replace(
            _name_dedup_sentinel,
            "The name should make someone curious, not nod in recognition. "
            f"Do not reuse or closely echo any of these existing mix names from the DJ's catalogue: {names_str}. "
            "Avoid borrowing any word, phrase, or trope from those names — even as a prefix, suffix, or modifier "
            f"(e.g. if '{used_mix_names[0]}' is in the list, '{used_mix_names[0]} Vol. 2' and any variation is forbidden). "
            'Add a "name_reason" field',
        )

    raw = await _call_stage2_raw(prompt, stage2_system, stage2_key, max_tokens=32768)

    if os.environ.get("MIXLAB_DEBUG_STAGE2"):
        print(f"[DEBUG] Stage 2 raw response (first 2000 chars):\n{raw[:2000]}", file=sys.stderr)

    valid_ids = set(tracks_by_id.keys())
    curated, _ = _parse_curated_concepts(raw, valid_ids)
    print(
        f"Stage 2 selection pass: {len(curated)} concept(s) returned (moods: {[c.mood for c in curated]})",
        file=sys.stderr,
    )
    report = ""

    if playlist_name is None:
        reports = await _call_stage2_reports(curated, tracks_by_id, seed_ids, unplayed_ids, stage2_key)
        annotated_reports = [
            _append_bold_moves_to_report(r, c, canvases, tracks_by_id) for r, c in zip(reports, curated, strict=True)
        ]
        report = "\n\n---\n\n".join(annotated_reports)

    if playlist_name is not None and seed_ids is not None and curated:
        playlist_seed_track_ids = seed_track_ids or sorted(seed_ids)
        minimum_seed_tracks = _minimum_playlist_seed_retention(len(playlist_seed_track_ids), intent_brief)

        # Score all returned concepts
        variants = [_score_variant(c, playlist_seed_track_ids, intent_brief, tracks_by_id) for c in curated]

        # Deduplicate: keep only the best-scoring variant per strategy.
        # The LLM sometimes returns multiple concepts with the same mood label.
        by_strategy: dict[str, CompletionVariant] = {}
        for v in variants:
            if v.strategy not in by_strategy or v.score > by_strategy[v.strategy].score:
                by_strategy[v.strategy] = v
        if len(by_strategy) < len(variants):
            print(
                f"Stage 2 dedup: {len(variants)} → {len(by_strategy)} variant(s) "
                f"(LLM returned duplicate strategy concepts)",
                file=sys.stderr,
            )
        variants = list(by_strategy.values())

        # Pre-filter: per-tier retention floor before selection
        passing = [v for v in variants if _passes_floor(v, intent_brief, playlist_seed_track_ids, minimum_seed_tracks)]
        candidates = passing if passing else variants

        best = _select_best_variant(candidates)
        concept = best.concept

        # Retry only if no variant passed the floor
        if not passing:
            retained_ids, dropped_ids, _ = _playlist_retention_stats(concept, playlist_seed_track_ids)
            dropped_labels = _format_playlist_track_labels(dropped_ids, tracks_by_id, limit=len(dropped_ids))
            retry_prompt = (
                f"Your previous attempt retained only {len(retained_ids)} seed tracks "
                f"(minimum required: {minimum_seed_tracks}).\n"
                f"Dropped seeds: {dropped_labels}.\n"
                "Retry with one concept only. Include as many dropped seeds as possible.\n\n"
            ) + prompt
            raw = await _call_stage2_raw(retry_prompt, stage2_system, stage2_key, max_tokens=32768)
            curated, _ = _parse_curated_concepts(raw, valid_ids)
            if curated:
                concept = curated[0]
                retained_ids, _, _ = _playlist_retention_stats(concept, playlist_seed_track_ids)
            else:
                retained_ids = []
            if len(retained_ids) < minimum_seed_tracks:
                raise RuntimeError(
                    "Stage 2 playlist curation retained too few seed tracks after retry: "
                    f"{len(retained_ids)} retained, minimum acceptable {minimum_seed_tracks}."
                )
            variants = []  # suppress rejected_summary on retry path

        if variants:
            ordered_variants = [best] + sorted(
                [v for v in variants if v.concept is not concept],
                key=lambda v: _STRATEGY_PRIORITY.get(v.strategy, 99),
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
            retry_reports = await _call_stage2_reports([concept], tracks_by_id, seed_ids, unplayed_ids, stage2_key)
            base_report = retry_reports[0] if retry_reports else ""
            report = _label_playlist_report_section(
                _rewrite_playlist_report(
                    base_report, playlist_name, concept, playlist_seed_track_ids, tracks_by_id, rejected_summary
                ),
                concept.mood.lower(),
                is_winner=True,
            )
            curated = [_retitle_playlist_concept(concept, concept.mood.lower(), is_winner=True)]

    warnings: list[str] = []
    for concept in curated:
        if not concept.track_ids:
            continue
        track = tracks_by_id.get(concept.track_ids[0])
        genre = track.genre if track else "_default"
        warning = shortfall_warning(concept, genre)
        if warning:
            warnings.append(f"⚠️ {concept.title}: {warning}")

    if warnings:
        report += "\n\n---\n\nSHORTFALL WARNINGS\n" + "\n".join(warnings)

    report += f"\n\n---\n\nMain brain: {stage2_model_display}"

    return curated, report
