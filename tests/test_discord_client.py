from __future__ import annotations

import httpx
import pytest
import respx

from mixlab.config import shortfall_warning
from mixlab.discord_client import DiscordClient, format_report
from mixlab.models import MixConcept, Track


def _make_track(i: int, genre: str = "Drum & Bass") -> Track:
    return Track(track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=174.0, camelot_key="8A", genre=genre)


_COUNTS: dict[str, tuple[int, int]] = {
    "drum_and_bass": (200, 80),
    "house": (150, 50),
}


def test_format_report_snapshot_unplayed_mode() -> None:
    result = format_report("", [], {}, [], counts=_COUNTS, show_unplayed=True)
    assert "**Crate Snapshot (unplayed / in collection)**" in result
    assert "80 / 200" in result
    assert "50 / 150" in result
    assert "Full Catalogue" not in result


def test_format_report_snapshot_full_catalogue_mode() -> None:
    result = format_report("", [], {}, [], counts=_COUNTS, show_unplayed=False)
    assert "**Full Catalogue Snapshot**" in result


def test_format_report_snapshot_full_catalogue_with_filter_desc() -> None:
    result = format_report("", [], {}, [], counts=_COUNTS, show_unplayed=False, filter_desc="BPM 126–130, year ≥ 2001")
    assert "**Full Catalogue Snapshot (BPM 126–130, year ≥ 2001)**" in result
    assert "200" in result
    assert "150" in result
    # Should NOT show ratio format
    assert "/ 200" not in result
    assert "unplayed / in collection" not in result
    # Largest genre (200) gets a full bar; smaller (150) gets a proportionally shorter one
    lines = [line for line in result.splitlines() if "drum_and_bass" in line or "house" in line]
    dnb_bar = lines[0].split("200")[1]
    house_bar = lines[1].split("150")[1]
    assert dnb_bar.count("█") > house_bar.count("█")


def test_format_report_places_context_above_snapshot() -> None:
    result = format_report(
        "",
        [],
        {},
        [],
        counts=_COUNTS,
        show_unplayed=True,
        report_context="Report context: 140 (custom genre, unplayed tracks)",
    )
    lines = result.splitlines()
    assert lines[0] == "Report context: 140 (custom genre, unplayed tracks)"
    assert lines[2] == "**Crate Snapshot (unplayed / in collection)**"


def test_format_report_no_counts_omits_snapshot() -> None:
    result = format_report("", [], {}, [])
    assert "Snapshot" not in result
    assert "```" not in result


def test_format_report_includes_report_text() -> None:
    result = format_report("CONCEPT: Test\n\nReport body.", [], {}, [])
    assert "CONCEPT: Test" in result


def test_format_report_includes_outliers() -> None:
    outlier = _make_track(1, genre="Reggae")
    result = format_report("", [], {}, [outlier])
    assert "Outliers" in result
    assert "Artist 1" in result


def test_format_report_includes_concept_headers() -> None:
    concept = MixConcept(title="Dark Rollers", mood="heavy", track_ids=["1"])
    tracks_by_id = {"1": _make_track(1)}
    result = format_report("", [concept], tracks_by_id, [])
    assert "Dark Rollers" in result


def test_format_report_shows_name_reason_with_label() -> None:
    concept = MixConcept(title="Dark Rollers", mood="heavy", track_ids=["1"], name_reason="Relentless drive.")
    tracks_by_id = {"1": _make_track(1)}
    result = format_report("", [concept], tracks_by_id, [])
    assert "**Name:**" in result
    assert "Relentless drive." in result


def test_format_report_omits_name_reason_when_empty() -> None:
    concept = MixConcept(title="Dark Rollers", mood="heavy", track_ids=["1"], name_reason="")
    tracks_by_id = {"1": _make_track(1)}
    result = format_report("", [concept], tracks_by_id, [])
    assert "**Name:**" not in result


def test_format_report_excluded_count_appears_inside_snapshot_block() -> None:
    result = format_report("", [], {}, [], counts=_COUNTS, show_unplayed=True, excluded_count=42)
    # Must mention the count
    assert "42 track(s) excluded (DO NOT RECOMMEND)" in result
    # Must appear before the closing ``` of the code block
    closing_tick = result.rindex("```")
    excluded_pos = result.index("42 track(s) excluded")
    assert excluded_pos < closing_tick


def test_format_report_excluded_count_zero_omits_line() -> None:
    result = format_report("", [], {}, [], counts=_COUNTS, show_unplayed=True, excluded_count=0)
    assert "excluded" not in result


def test_shortfall_warning_returns_none_when_shortfall_small() -> None:
    concept = MixConcept(title="T", mood="m", track_ids=[str(i) for i in range(20)])
    assert shortfall_warning(concept, "house") is None


def test_shortfall_warning_returns_message_when_shortfall_large() -> None:
    concept = MixConcept(title="T", mood="m", track_ids=["1", "2"])
    result = shortfall_warning(concept, "House")
    assert result is not None
    assert "2 tracks found" in result
    assert "more to fill" in result


async def test_post_raw_raises_after_max_retries_on_429() -> None:
    client = DiscordClient(
        bot_token="tok", guild_id="g", channel_id="c123", channel_name="mix-lab"
    )
    with respx.mock:
        respx.post("https://discord.com/api/v10/channels/c123/messages").mock(
            return_value=httpx.Response(429, json={"retry_after": 0.001})
        )
        async with httpx.AsyncClient(timeout=5) as http:
            with pytest.raises(RuntimeError, match="rate limit"):
                await client._post_raw(http, "c123", "hello")
