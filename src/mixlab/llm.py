from __future__ import annotations

import json
import os
import re
import sys

import httpx

from mixlab.config import TRACK_COUNT_TARGETS
from mixlab.models import MixConcept, Track

_MAX_TRACKS_PER_CALL = 40
_MIN_CONCEPT_TRACKS = 4

# Strip inline thinking blocks emitted by reasoning models (e.g. MiniMax M2.5).
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Stage 1 — mix concept generation (provider cascade)
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = (
    "You are an expert DJ and music curator. Given a list of tracks from a DJ's collection, "
    "generate 2–3 distinct mix concepts. Each concept should have a creative title, a one-line "
    "mood descriptor, and a curated selection of track IDs from the provided list. "
    "Keep each concept's BPM range tight — no more than ±6 BPM from the concept's median tempo — "
    "so every track can be mixed without excessive pitch shifting. "
    "Aim for a coherent Camelot key journey within each concept — prefer adjacent moves (±1, same number opposite mode). "
    "Occasional 2-step jumps are fine as deliberate energy shifts, but avoid scattering tracks across unrelated keys. "
    "Respond ONLY with a JSON array matching this schema: "
    '[{"title": "...", "mood": "...", "track_ids": ["id1", "id2", ...]}]'
)


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
    base_url: str, api_key: str, model: str, prompt: str, path: str = "/v1/chat/completions", timeout: int = 60
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": _STAGE1_SYSTEM}, {"role": "user", "content": prompt}],
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
                # Strip hallucinated IDs, then drop concepts below minimum viable length.
                cleaned = [
                    MixConcept(title=c.title, mood=c.mood, track_ids=[tid for tid in c.track_ids if tid in valid_ids])
                    for c in concepts
                ]
                return [c for c in cleaned if len(c.track_ids) >= _MIN_CONCEPT_TRACKS]
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
# Stage 2 — report generation (Anthropic direct HTTP, no fallback)
# ---------------------------------------------------------------------------

_STAGE2_SYSTEM = (
    "You are an experienced DJ writing a mix planning report for a fellow DJ. "
    "Write with authority and specificity — peer-to-peer, no marketing language, no filler. "
    "For each mix concept: provide a 2–3 sentence creative brief, then list the tracks in "
    "Camelot key order with their BPM. Use plain text with minimal formatting."
)

_SHORTFALL_THRESHOLD = 4


def _shortfall_warning(concept: MixConcept, genre: str) -> str | None:
    min_count, _ = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])
    n = len(concept.track_ids)
    shortfall = min_count - n
    if shortfall > _SHORTFALL_THRESHOLD:
        return f"⚠️ {n} tracks found — needs {shortfall} more to fill a set. Crate dig to complete."
    return None


async def stage2_report(concepts: list[MixConcept], tracks_by_id: dict[str, Track]) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — Stage 2 report generation requires Anthropic.")

    sections: list[str] = []
    for concept in concepts:
        track_lines = []
        for tid in concept.track_ids:
            if tid in tracks_by_id:
                t = tracks_by_id[tid]
                track_lines.append(f"  {t.artist} — {t.title} [{t.camelot_key} · {t.bpm} BPM]")
        sections.append(f"Concept: {concept.title}\nMood: {concept.mood}\nTracks:\n" + "\n".join(track_lines))

    prompt = "Write a mix planning report for the following concepts:\n\n" + "\n\n".join(sections)

    try:
        report = await _call_anthropic_http(
            key, "claude-sonnet-4-6", _STAGE2_SYSTEM, prompt, max_tokens=8192, timeout=300
        )
    except Exception as exc:
        raise RuntimeError(f"Stage 2 report generation failed: {exc}") from exc

    warnings: list[str] = []
    for concept in concepts:
        if not concept.track_ids:
            continue
        track = tracks_by_id.get(concept.track_ids[0])
        genre = track.genre if track else "_default"
        warning = _shortfall_warning(concept, genre)
        if warning:
            warnings.append(f"⚠️ {concept.title}: {warning}")

    if warnings:
        report += "\n\n---\n\nSHORTFALL WARNINGS\n" + "\n".join(warnings)
    return report
