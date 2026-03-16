from __future__ import annotations

import json
import os
import re
import sys

import httpx

from mixlab.config import TRACK_COUNT_TARGETS
from mixlab.models import MixConcept, Track

_MAX_TRACKS_PER_CALL = 40
_MIN_SHORTLIST_TRACKS = 8  # Stage 1: minimum candidates per pool
_MIN_CONCEPT_TRACKS = 4  # Stage 2: minimum tracks in a final curated set

# Strip inline thinking blocks emitted by reasoning models (e.g. MiniMax M2.5).
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


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
- Generate 2–3 distinct shortlists with different BPM centres or key characters so they serve different mood directions.
- Do NOT make final ordering decisions. Do NOT decide openers or closers. Simply group technically compatible tracks.
- Exclude obvious outliers: tracks more than 8 BPM from the group median, or in keys with no harmonic relationship \
to the rest of the pool.

Give each shortlist a rough descriptive title (e.g. "Deep 122 BPM / 4A–7A Pool") and a one-line sonic mood.

Respond ONLY with a JSON array matching this schema:
[{"title": "...", "mood": "...", "track_ids": ["id1", "id2", ...]}]\
"""


def _tracks_to_text(tracks: list[Track]) -> str:
    lines = [f"ID:{t.track_id} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key}" for t in tracks]
    return "\n".join(lines)


def _parse_concepts(raw: str) -> list[MixConcept]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    data = json.loads(raw.strip())
    return [MixConcept(**item) for item in data]


async def _call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    path: str = "/v1/chat/completions",
    timeout: int = 60,
    system: str = _STAGE1_SYSTEM,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.7,
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


async def _try_minimax(prompt: str) -> str | None:
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        return None
    return await _call_openai_compat("https://api.minimaxi.chat", key, "MiniMax-M2.1", prompt, timeout=120)


async def _try_groq(prompt: str) -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return await _call_openai_compat("https://api.groq.com/openai", key, "llama-3.3-70b-versatile", prompt)


async def _try_gemini(prompt: str) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    return await _call_openai_compat(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        key,
        "gemini-2.5-flash",
        prompt,
        path="/chat/completions",
    )


async def _try_openrouter(prompt: str) -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    return await _call_openai_compat("https://openrouter.ai/api", key, "deepseek/deepseek-chat", prompt)


async def _try_anthropic(prompt: str) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return await _call_anthropic_http(key, "claude-sonnet-4-6", _STAGE1_SYSTEM, prompt)


_CASCADE = [_try_minimax, _try_groq, _try_gemini, _try_openrouter, _try_anthropic]


async def _call_stage1_once(tracks: list[Track], genre: str) -> list[MixConcept]:
    valid_ids = {t.track_id for t in tracks}
    prompt = f"Genre: {genre}\n\nTracks:\n{_tracks_to_text(tracks)}"
    for provider in _CASCADE:
        try:
            result = await provider(prompt)
            if result is not None:
                concepts = _parse_concepts(result)
                cleaned = [
                    MixConcept(title=c.title, mood=c.mood, track_ids=[tid for tid in c.track_ids if tid in valid_ids])
                    for c in concepts
                ]
                return [c for c in cleaned if len(c.track_ids) >= _MIN_SHORTLIST_TRACKS]
        except Exception as exc:  # noqa: BLE001 — cascade, swallow and try next
            print(f"Provider {provider.__name__} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
    raise RuntimeError(f"All LLM providers failed for genre '{genre}'.")


async def stage1_concepts(cluster: list[Track], genre: str) -> list[MixConcept]:
    if len(cluster) <= _MAX_TRACKS_PER_CALL:
        return await _call_stage1_once(cluster, genre)

    concepts: list[MixConcept] = []
    for i in range(0, len(cluster), _MAX_TRACKS_PER_CALL):
        chunk = cluster[i : i + _MAX_TRACKS_PER_CALL]
        concepts.extend(await _call_stage1_once(chunk, genre))
    return concepts


# ---------------------------------------------------------------------------
# Stage 2 — creative curation + report (Anthropic, single call, no fallback)
# ---------------------------------------------------------------------------

_STAGE2_SYSTEM = """\
Act as a world-class DJ and mix curator. You will receive candidate shortlists of tracks that have been \
pre-screened for technical compatibility. Your job is to curate and narrate — select the best tracks from \
each shortlist, decide the play order, and write the full mix report.

For each shortlist provided:
- SELECT the best 8–12 tracks from the pool for a coherent DJ set. Exclude tracks that weaken the journey.
- ORDER them as the intended play sequence: opener first, closer last.
- The opener must set the tone, create intrigue, and leave room to build. Do not open with the most energetic track.
- The closer must provide resolution and feel like a final statement.
- Design an intentional energy curve: tension, build, peak, release, landing.
- Allow bold moves — larger key jumps, tempo pivots — when they serve the narrative.
- Do NOT optimise only for BPM and key. Optimise for flow, tension, release, memorability, and emotional payoff.

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

[2–3 sentence creative brief]

Track order (Camelot / BPM):
[Artist — Title [Key · BPM] for each track in play order]

Arc: [energy shape, emotional journey, and structural logic across the full set — name specific moments]

Opener: [why this track works first — tone, identity, room to build]

Closer: [why this track works last — resolution, emotional weight, final statement]

Standout transitions or calculated risks: [1–3 specific moves worth naming — be precise about which tracks]

Assumptions: [what was inferred from metadata where audio analysis was not possible]

Be opinionated, musical, and honest. Peer-to-peer, no marketing language, no filler.

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


def _parse_curated_concepts(raw: str, valid_ids: set[str]) -> tuple[list[MixConcept], str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    data: list[dict[str, object]] = json.loads(raw.strip())

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
) -> tuple[list[MixConcept], str]:
    provider = (stage2_provider or os.environ.get("STAGE2_PROVIDER", "anthropic")).lower()
    use_minimax = provider == "minimax"

    if use_minimax:
        stage2_key = os.environ.get("MINIMAX_API_KEY")
        if not stage2_key:
            raise RuntimeError("STAGE2_PROVIDER=minimax but MINIMAX_API_KEY is not set.")
        stage2_model_display = "MiniMax M2.5"
    else:
        stage2_key = os.environ.get("ANTHROPIC_API_KEY")
        if not stage2_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — Stage 2 curation requires Anthropic.")
        stage2_model_display = "Claude Sonnet 4.6"

    sections: list[str] = []
    for shortlist in shortlists:
        track_lines = [
            f"  ID:{tid} | {t.artist} — {t.title} | {t.bpm} BPM | {t.camelot_key}"
            for tid in shortlist.track_ids
            if (t := tracks_by_id.get(tid)) is not None
        ]
        if track_lines:
            sections.append(
                f"Shortlist: {shortlist.title}\nCharacter: {shortlist.mood}\nCandidates:\n" + "\n".join(track_lines)
            )

    n = len(sections)
    prompt = (
        f"Curate and narrate a mix report from the following {n} candidate shortlists. "
        f"Your JSON array MUST contain exactly {n} objects — one per shortlist, in the order given. "
        f"Do not merge, skip, or consolidate shortlists.\n\n" + "\n\n".join(sections)
    )

    try:
        if use_minimax:
            raw = await _call_openai_compat(
                "https://api.minimaxi.chat",
                stage2_key,
                "MiniMax-M2.5",
                prompt,
                timeout=300,
                system=_STAGE2_SYSTEM,
            )
        else:
            raw = await _call_anthropic_http(
                stage2_key, "claude-sonnet-4-6", _STAGE2_SYSTEM, prompt, max_tokens=8192, timeout=300
            )
    except Exception as exc:
        raise RuntimeError(f"Stage 2 curation failed: {exc}") from exc

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
