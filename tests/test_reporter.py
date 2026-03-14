from __future__ import annotations

import pytest
import respx
from httpx import Response

from mixlab.discord_client import DiscordClient, format_report, send_report
from mixlab.models import MixConcept, Track

_GUILD_ID = "123456789"
_CHANNEL_NAME = "mix-lab"
_CHANNEL_ID = "1482325879257301124"
_CHANNELS_URL = f"https://discord.com/api/v10/guilds/{_GUILD_ID}/channels"
_MESSAGES_URL = f"https://discord.com/api/v10/channels/{_CHANNEL_ID}/messages"

_CHANNELS_RESPONSE = [{"id": _CHANNEL_ID, "name": _CHANNEL_NAME}]


def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_GUILD_ID", _GUILD_ID)
    monkeypatch.setenv("MIXLAB_DISCORD_CHANNEL", _CHANNEL_NAME)
    monkeypatch.setenv("MIXLAB_DISCORD_CHANNEL_ID", _CHANNEL_ID)


def _make_client() -> DiscordClient:
    return DiscordClient(
        bot_token="test-token",
        guild_id=_GUILD_ID,
        channel_id=_CHANNEL_ID,
        channel_name=_CHANNEL_NAME,
    )


def _make_concepts() -> list[MixConcept]:
    return [MixConcept(title="Dark Rollers", mood="heavy", track_ids=["1"])]


def _make_tracks_by_id() -> dict[str, Track]:
    return {
        "1": Track(track_id="1", artist="Calibre", title="All Good", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
    }


async def test_send_report_returns_false_if_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    result = await send_report("report", _make_concepts(), [], _make_tracks_by_id())
    assert result is False


@respx.mock
async def test_send_report_chunks_long_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_env(monkeypatch)
    post_route = respx.post(_MESSAGES_URL).mock(return_value=Response(200, json={"id": "1"}))

    long_report = "x" * 3000
    concepts = [MixConcept(title=f"Concept {i}", mood="dark", track_ids=[str(i)]) for i in range(5)]
    tracks = {
        str(i): Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(5)
    }

    result = await send_report(long_report, concepts, [], tracks)
    assert result is True
    assert post_route.call_count >= 1


@respx.mock
async def test_send_report_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_env(monkeypatch)
    post_route = respx.post(_MESSAGES_URL).mock(
        side_effect=[
            Response(429, json={"retry_after": 0.0}),
            Response(200, json={"id": "1"}),
        ]
    )

    result = await send_report("hello", _make_concepts(), [], _make_tracks_by_id())
    assert result is True
    assert post_route.call_count == 2


@respx.mock
async def test_send_report_returns_false_on_discord_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_env(monkeypatch)
    respx.post(_MESSAGES_URL).mock(return_value=Response(500, text="Server Error"))

    result = await send_report("hello", _make_concepts(), [], _make_tracks_by_id())
    assert result is False


def test_crate_snapshot_appears_first_when_counts_provided() -> None:
    counts = {"house": (100, 40), "drum_and_bass": (80, 30)}
    text = format_report("", _make_concepts(), _make_tracks_by_id(), [], counts=counts)
    snapshot_pos = text.find("Crate Snapshot")
    concept_pos = text.find("Dark Rollers")
    assert snapshot_pos != -1
    assert snapshot_pos < concept_pos, "Crate Snapshot must appear before concepts"
    assert "house" in text
    assert "40 / 100" in text


def test_no_snapshot_when_counts_not_provided() -> None:
    text = format_report("", _make_concepts(), _make_tracks_by_id(), [])
    assert "Crate Snapshot" not in text


def test_outliers_appear_after_concepts() -> None:
    outlier = Track(
        track_id="99", artist="Unknown", title="Mystery Track", bpm=140.0, camelot_key="5A", genre="Ambient"
    )
    text = format_report("", _make_concepts(), _make_tracks_by_id(), [outlier])
    concept_pos = text.find("Dark Rollers")
    outlier_pos = text.find("Outliers")
    assert concept_pos < outlier_pos, "Outlier section must appear after all concepts"


def test_track_listings_shown_when_no_report() -> None:
    text = format_report("", _make_concepts(), _make_tracks_by_id(), [])
    assert "Calibre" in text


def test_track_listings_hidden_when_report_present() -> None:
    text = format_report("⏱ Generated in 42s", _make_concepts(), _make_tracks_by_id(), [])
    assert "Calibre" not in text


def test_report_text_included_in_output() -> None:
    text = format_report("⏱ Generated in 42s", _make_concepts(), _make_tracks_by_id(), [])
    assert "⏱ Generated in 42s" in text


def test_report_text_appears_after_concepts() -> None:
    text = format_report("⏱ Generated in 42s", _make_concepts(), _make_tracks_by_id(), [])
    concept_pos = text.find("Dark Rollers")
    timing_pos = text.find("⏱ Generated in 42s")
    assert concept_pos < timing_pos, "Report text must appear after concepts"


def test_chunk_text_splits_at_newline_boundary() -> None:
    client = _make_client()
    long_text = "line\n" * 500  # 2500 chars, should split at newline
    chunks = client._chunk_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 2000
