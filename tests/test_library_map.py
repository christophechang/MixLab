from __future__ import annotations

import json
from typing import Callable, cast

import pytest

from mixlab.config import CUSTOM_GENRES, GENRE_MAP
from mixlab.library_map import build_map_payload, render_map_json
from mixlab.models import PlayedTrack, Track


def _track(
    *,
    track_id: str,
    artist: str,
    title: str,
    bpm: float,
    camelot_key: str,
    genre: str,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=camelot_key,
        genre=genre,
    )


@pytest.fixture
def map_tracks() -> list[Track]:
    """A dozen tracks across House/Techno GENRE_MAP tags, plus a Drum & Bass
    inclusion/exclusion pair straddling the "170" custom pool's (165.0, 175.0)
    BPM range (see CUSTOM_GENRES in config.py) — "170" only admits
    drum_and_bass/jungle genres, so the pair must carry that genre for the
    filter to actually be exercised.
    """
    house = [
        _track(
            track_id=f"h{i}",
            artist=f"House Artist {i}",
            title=f"House Track {i}",
            bpm=122.0 + i,
            camelot_key="8B",
            genre="House",
        )
        for i in range(5)
    ]
    techno = [
        _track(
            track_id=f"t{i}",
            artist=f"Techno Artist {i}",
            title=f"Techno Track {i}",
            bpm=130.0 + i,
            camelot_key="9A",
            genre="Techno",
        )
        for i in range(5)
    ]
    dnb_in_range = _track(
        track_id="dnb_in",
        artist="DnB Artist In",
        title="DnB Track In",
        bpm=170.0,
        camelot_key="4A",
        genre="Drum & Bass",
    )
    dnb_out_of_range = _track(
        track_id="dnb_out",
        artist="DnB Artist Out",
        title="DnB Track Out",
        bpm=185.0,
        camelot_key="4A",
        genre="Drum & Bass",
    )
    return [*house, *techno, dnb_in_range, dnb_out_of_range]


def test_build_map_payload_all_mode_covers_every_pool_key(map_tracks: list[Track]) -> None:
    payload = build_map_payload(map_tracks, mode="all", seed=0, played=[])
    assert payload["version"] == 1
    assert payload["mode"] == "all"
    pools = cast("dict[str, dict[str, object]]", payload["pools"])
    assert list(pools) == [*GENRE_MAP, *CUSTOM_GENRES]
    for entry in pools.values():
        assert set(entry) == {"track_count", "directions"}


def test_build_map_payload_unplayed_mode_filters_catalog_matches(map_tracks: list[Track]) -> None:
    # Pick one track from the fixture; a PlayedTrack with its artist/title drops it.
    victim = map_tracks[0]
    played = [PlayedTrack(artist=victim.artist, title=victim.title)]
    all_counts = build_map_payload(map_tracks, mode="all", seed=0, played=played)
    unplayed = build_map_payload(map_tracks, mode="unplayed", seed=0, played=played)
    assert unplayed["catalog_tracks"] == 1
    total: Callable[[dict[str, object]], int] = lambda p: sum(  # noqa: E731
        cast(int, e["track_count"])
        for k, e in cast("dict[str, dict[str, object]]", p["pools"]).items()
        if k in GENRE_MAP
    )
    assert total(unplayed) == total(all_counts) - 1


def test_build_map_payload_custom_pool_170_applies_bpm_filter(map_tracks: list[Track]) -> None:
    # Fixture carries exactly one Drum & Bass track inside the "170" custom pool's
    # (165.0, 175.0) BPM range (dnb_in, bpm=170.0) and one outside it (dnb_out,
    # bpm=185.0) — a reversed or off-by-one boundary would silently pass the other
    # two tests, since neither asserts on custom-pool contents.
    payload = build_map_payload(map_tracks, mode="all", seed=0, played=[])
    pools = cast("dict[str, dict[str, object]]", payload["pools"])
    pool_170 = pools["170"]
    assert pool_170["track_count"] == 1
    directions = cast("list[dict[str, object]]", pool_170["directions"])
    for direction in directions:
        track_ids = cast("list[str]", direction["track_ids"])
        assert "dnb_out" not in track_ids


def test_render_map_json_deterministic_and_parseable(map_tracks: list[Track]) -> None:
    payload = build_map_payload(map_tracks, mode="all", seed=3, played=[])
    first = render_map_json(payload)
    second = render_map_json(build_map_payload(map_tracks, mode="all", seed=3, played=[]))
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["seed"] == 3
