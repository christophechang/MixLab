from __future__ import annotations

import asyncio
import json
import os

import httpx

from mixlab.config import TRACK_COUNT_TARGETS
from mixlab.models import MixConcept, Track

_DISCORD_API = "https://discord.com/api/v10"
_CHUNK_SIZE = 2000
_CHUNK_SLEEP = 0.5
_RATE_LIMIT_BUFFER = 0.5
_SHORTFALL_THRESHOLD = 4


def _shortfall_warning(concept: MixConcept, genre: str) -> str | None:
    min_count, _ = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])
    n = len(concept.track_ids)
    shortfall = min_count - n
    if shortfall > _SHORTFALL_THRESHOLD:
        return f"⚠️ {n} tracks found — needs {shortfall} more to fill a set. Crate dig to complete."
    return None


def format_report(
    report: str,
    concepts: list[MixConcept],
    tracks_by_id: dict[str, Track],
    outliers: list[Track],
    skipped: int = 0,
    counts: dict[str, tuple[int, int]] | None = None,
    show_unplayed: bool = True,
) -> str:
    lines: list[str] = []

    if counts:
        if show_unplayed:
            lines.append("**Crate Snapshot (unplayed / in collection)**")
            lines.append("```")
            for label, (total, available) in counts.items():
                pct = available / total if total else 0
                filled = round(pct * 20)
                bar = "█" * filled + "░" * (20 - filled)
                lines.append(f"  {label:<20} {available:>4} / {total:<4}  {bar}  {pct:.0%}")
        else:
            lines.append("**Full Catalogue Snapshot**")
            lines.append("```")
            max_total = max(t for t, _ in counts.values()) if counts else 1
            bar_width = 20
            for label, (total, _) in counts.items():
                filled = round((total / max_total) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                lines.append(f"  {label:<20} {total:<4}  {bar}")
        lines.append("```\n")

    if skipped:
        lines.append(f"ℹ️ {skipped} track(s) excluded during parsing.\n")

    for concept in concepts:
        first_track = next((tracks_by_id[tid] for tid in concept.track_ids if tid in tracks_by_id), None)
        genre = first_track.genre if first_track else "_default"

        lines.append(f"🎵 **{genre} · {concept.title}**")
        lines.append(f"_{concept.mood}_")

        if not report:
            lines.append("")
            for idx, tid in enumerate(concept.track_ids, start=1):
                if tid in tracks_by_id:
                    t = tracks_by_id[tid]
                    lines.append(f"{idx}. {t.artist} — {t.title} [{t.camelot_key} · {t.bpm} BPM]")

            warning = _shortfall_warning(concept, genre)
            if warning:
                lines.append(f"\n{warning}")

        lines.append("")

    if outliers:
        lines.append("🔍 **Outliers — Uncharted Territory**")
        for idx, t in enumerate(outliers, start=1):
            lines.append(f"{idx}. {t.artist} — {t.title} ({t.genre}) [{t.camelot_key} · {t.bpm} BPM]")

    if report:
        lines.append("")
        lines.append(report)

    return "\n".join(lines)


class DiscordClient:
    def __init__(self, bot_token: str, guild_id: str, channel_id: str, channel_name: str):
        self._token = bot_token
        self._guild_id = guild_id
        self._channel_id = channel_id
        self._channel_name = channel_name
        self._channel_cache: dict[str, str] = {}
        self._headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= _CHUNK_SIZE:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= _CHUNK_SIZE:
                chunks.append(text)
                break
            split = text.rfind("\n", 0, _CHUNK_SIZE)
            if split == -1:
                split = _CHUNK_SIZE
            chunks.append(text[:split])
            text = text[split:].lstrip("\n")
        return chunks

    async def _resolve_channel(self, client: httpx.AsyncClient) -> str:
        # If a channel ID was provided directly, use it without a guild lookup.
        if self._channel_id:
            return self._channel_id

        if self._channel_name in self._channel_cache:
            return self._channel_cache[self._channel_name]

        resp = await client.get(f"{_DISCORD_API}/guilds/{self._guild_id}/channels", headers=self._headers)
        resp.raise_for_status()
        for ch in resp.json():
            self._channel_cache[str(ch["name"])] = str(ch["id"])

        if self._channel_name not in self._channel_cache:
            raise ValueError(f"Channel '{self._channel_name}' not found in guild {self._guild_id}")

        return self._channel_cache[self._channel_name]

    async def _post_raw(self, client: httpx.AsyncClient, channel_id: str, content: str) -> None:
        url = f"{_DISCORD_API}/channels/{channel_id}/messages"
        while True:
            resp = await client.post(url, headers=self._headers, json={"content": content})
            if resp.status_code == 429:
                retry_after = float(resp.json().get("retry_after", 1.0))
                await asyncio.sleep(retry_after + _RATE_LIMIT_BUFFER)
                continue
            resp.raise_for_status()
            return

    async def _post_with_files(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
        content: str,
        attachments: list[tuple[str, bytes]],
    ) -> None:
        url = f"{_DISCORD_API}/channels/{channel_id}/messages"
        # Multipart upload — httpx sets Content-Type with boundary automatically
        headers = {k: v for k, v in self._headers.items() if k.lower() != "content-type"}
        files = {f"files[{i}]": (name, data, "application/xml") for i, (name, data) in enumerate(attachments)}
        data = {"payload_json": json.dumps({"content": content})}
        while True:
            resp = await client.post(url, headers=headers, data=data, files=files)
            if resp.status_code == 429:
                retry_after = float(resp.json().get("retry_after", 1.0))
                await asyncio.sleep(retry_after + _RATE_LIMIT_BUFFER)
                continue
            resp.raise_for_status()
            return

    async def post(self, message: str, attachments: list[tuple[str, bytes]] | None = None) -> bool:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                channel_id = await self._resolve_channel(client)
                chunks = self._chunk_text(message)
                for i, chunk in enumerate(chunks):
                    await self._post_raw(client, channel_id, chunk)
                    if i < len(chunks) - 1:
                        await asyncio.sleep(_CHUNK_SLEEP)
                # One message per concept so each file is clearly labelled.
                for filename, data in attachments or []:
                    await asyncio.sleep(_CHUNK_SLEEP)
                    label = filename.removesuffix(".xml").replace("_", " ")
                    await self._post_with_files(client, channel_id, f"🎵 **{label}**", [(filename, data)])
            return True
        except Exception as exc:  # noqa: BLE001 — never crash the pipeline
            print(f"Discord delivery failed: {exc}")
            return False


def make_discord_client() -> DiscordClient | None:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("WARNING: DISCORD_BOT_TOKEN not set — Discord delivery disabled.")
        return None
    return DiscordClient(
        bot_token=token,
        guild_id=os.environ.get("DISCORD_GUILD_ID", ""),
        channel_id=os.environ.get("MIXLAB_DISCORD_CHANNEL_ID", ""),
        channel_name=os.environ.get("MIXLAB_DISCORD_CHANNEL", "mix-lab"),
    )


async def send_report(
    report: str,
    concepts: list[MixConcept],
    outliers: list[Track],
    tracks_by_id: dict[str, Track] | None = None,
    skipped: int = 0,
    counts: dict[str, tuple[int, int]] | None = None,
    attachments: list[tuple[str, bytes]] | None = None,
    show_unplayed: bool = True,
) -> bool:
    client = make_discord_client()
    if client is None:
        return False

    if tracks_by_id is None:
        tracks_by_id = {}

    full_text = format_report(report, concepts, tracks_by_id, outliers, skipped, counts, show_unplayed)
    return await client.post(full_text, attachments)
