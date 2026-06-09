from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from mixlab.models import IntentBrief, MixConcept, Track
from mixlab.playlist_mode import (
    _score_candidate,  # noqa: PLC2701
    build_playlist_pool,
    build_zone_shortlists,
    cluster_seed_zones,
    compute_deterministic_intent,
    detect_adjacency_fragments,
    detect_energy_shape,
    filter_tracks_for_playlist_genre,
    identify_missing_roles,
    resolve_playlist,
)


def _make_track(
    track_id: str,
    *,
    bpm: float = 124.0,
    play_count: int = 0,
    artist: str | None = None,
    title: str | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist or f"Artist {track_id}",
        title=title or f"Title {track_id}",
        bpm=bpm,
        camelot_key="8A",
        genre="House",
        play_count=play_count,
    )


def test_resolve_playlist_exact_match() -> None:
    playlists = {"Sets/Warm Up": ["1", "2"]}
    assert resolve_playlist("Sets/Warm Up", playlists) == ["1", "2"]


def test_resolve_playlist_case_insensitive() -> None:
    playlists = {"Sets/Warm Up": ["1", "2"]}
    assert resolve_playlist("sets/warm up", playlists) == ["1", "2"]


def test_resolve_playlist_suffix_match_unambiguous() -> None:
    playlists = {"Sets/Warm Up": ["1", "2"], "Archive/Closer": ["3"]}
    assert resolve_playlist("Warm Up", playlists) == ["1", "2"]


def test_resolve_playlist_suffix_match_ambiguous_raises_with_paths() -> None:
    playlists = {"Sets/Warm Up": ["1"], "Archive/Warm Up": ["2"]}
    with pytest.raises(ValueError, match="Ambiguous playlist name"):
        resolve_playlist("Warm Up", playlists)


def test_resolve_playlist_not_found_raises_with_suggestions() -> None:
    playlists = {"Sets/Warm Up": ["1"]}
    with pytest.raises(ValueError, match="Did you mean"):
        resolve_playlist("Warm", playlists)


def test_resolve_playlist_no_suggestions_when_nothing_close() -> None:
    playlists = {"Sets/Warm Up": ["1"]}
    with pytest.raises(ValueError, match='Playlist "Techno Tools" not found\\.'):
        resolve_playlist("Techno Tools", playlists)


def test_build_playlist_pool_includes_all_seed_tracks() -> None:
    tracks = [
        _make_track("1", bpm=122.0),
        _make_track("2", bpm=124.0),
        _make_track("3", bpm=126.0),
        _make_track("4", bpm=128.0),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, {"1", "2", "3", "4"})
    assert [track.track_id for track in pool[:4]] == ["1", "2", "3", "4"]


def test_build_playlist_pool_expands_with_bpm_range() -> None:
    tracks = [
        _make_track("1", bpm=120.0),
        _make_track("2", bpm=122.0),
        _make_track("3", bpm=124.0),
        _make_track("4", bpm=126.0),
        _make_track("5", bpm=139.0),
        _make_track("6", bpm=145.5),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, {"1", "2", "3", "4", "5"})
    assert "5" in [track.track_id for track in pool]
    assert "6" not in [track.track_id for track in pool]


def test_build_playlist_pool_too_few_seed_tracks_raises() -> None:
    tracks = [_make_track("1"), _make_track("2"), _make_track("3")]
    tracks_by_id = {track.track_id: track for track in tracks}
    with pytest.raises(ValueError, match="requires at least 4 valid seed tracks"):
        build_playlist_pool(["1", "2", "3"], tracks, tracks_by_id, {"1", "2", "3"})


def test_build_playlist_pool_unplayed_ids_ranked_above_played() -> None:
    tracks = [
        _make_track("1", bpm=122.0),
        _make_track("2", bpm=124.0),
        _make_track("3", bpm=126.0),
        _make_track("4", bpm=128.0),
        _make_track("5", bpm=123.0, play_count=1),
        _make_track("6", bpm=123.5, play_count=1),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, {"1", "2", "3", "4", "6"})
    ids = [track.track_id for track in pool]
    assert ids.index("6") < ids.index("5")


def test_build_playlist_pool_play_count_fallback_when_unplayed_ids_none() -> None:
    tracks = [
        _make_track("1", bpm=122.0),
        _make_track("2", bpm=124.0),
        _make_track("3", bpm=126.0),
        _make_track("4", bpm=128.0),
        _make_track("5", bpm=123.0, play_count=1),
        _make_track("6", bpm=123.5, play_count=0),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, None)
    ids = [track.track_id for track in pool]
    assert ids.index("6") < ids.index("5")


def test_build_playlist_pool_played_tracks_remain_in_pool() -> None:
    tracks = [
        _make_track("1", bpm=122.0),
        _make_track("2", bpm=124.0),
        _make_track("3", bpm=126.0),
        _make_track("4", bpm=128.0),
        _make_track("5", bpm=123.0, play_count=3),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, {"1", "2", "3", "4"})
    assert "5" in [track.track_id for track in pool]


def test_build_playlist_pool_all_tracks_flag_disables_unplayed_bonus() -> None:
    tracks = [
        _make_track("1", bpm=122.0),
        _make_track("2", bpm=124.0),
        _make_track("3", bpm=126.0),
        _make_track("4", bpm=128.0),
        _make_track("5", bpm=123.0, play_count=1, artist="Z Artist"),
        _make_track("6", bpm=129.8, play_count=0, artist="A Artist"),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(
        ["1", "2", "3", "4"], tracks, tracks_by_id, {"1", "2", "3", "4", "6"}, all_tracks_flag=True
    )
    ids = [track.track_id for track in pool]
    assert ids.index("5") < ids.index("6")


def test_build_playlist_pool_capped_at_max_with_seed_guaranteed() -> None:
    seed_tracks = [_make_track(str(index), bpm=120.0 + index) for index in range(1, 5)]
    extras = [_make_track(str(index), bpm=123.0) for index in range(5, 140)]
    tracks = seed_tracks + extras
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, None)
    assert len(pool) == 120
    assert [track.track_id for track in pool[:4]] == ["1", "2", "3", "4"]


def test_build_playlist_pool_deduplicates_seed_from_library() -> None:
    tracks = [
        _make_track("1", bpm=122.0),
        _make_track("2", bpm=124.0),
        _make_track("3", bpm=126.0),
        _make_track("4", bpm=128.0),
    ]
    tracks_by_id = {track.track_id: track for track in tracks}
    pool = build_playlist_pool(["1", "2", "3", "4"], tracks, tracks_by_id, None)
    assert [track.track_id for track in pool].count("1") == 1


# ---------------------------------------------------------------------------
# cluster_seed_zones
# ---------------------------------------------------------------------------


def test_cluster_seed_zones_single_zone_when_bpm_close() -> None:
    tracks = [_make_track(str(i), bpm=120.0 + i) for i in range(5)]
    zones = cluster_seed_zones(tracks)
    assert len(zones) == 1
    assert len(zones[0]) == 5


def test_cluster_seed_zones_splits_at_gap() -> None:
    low = [_make_track(str(i), bpm=90.0 + i) for i in range(4)]
    high = [_make_track(str(i + 10), bpm=120.0 + i) for i in range(4)]
    zones = cluster_seed_zones(low + high)
    assert len(zones) == 2
    assert all(t.bpm < 110 for t in zones[0])
    assert all(t.bpm >= 120 for t in zones[1])


def test_cluster_seed_zones_merges_undersized_zone_into_nearest() -> None:
    # Zone A: 88-90 BPM (2 tracks — below _ZONE_MIN_TRACKS=3), Zone B: 120-122 BPM (4 tracks)
    zone_a = [_make_track(str(i), bpm=88.0 + i) for i in range(2)]
    zone_b = [_make_track(str(i + 10), bpm=120.0 + i) for i in range(4)]
    zones = cluster_seed_zones(zone_a + zone_b)
    assert len(zones) == 1
    assert len(zones[0]) == 6


def test_cluster_seed_zones_empty_returns_empty() -> None:
    assert cluster_seed_zones([]) == []


def test_cluster_seed_zones_sorted_by_bpm_ascending() -> None:
    tracks = [_make_track("a", bpm=130.0), _make_track("b", bpm=90.0), _make_track("c", bpm=95.0)]
    zones = cluster_seed_zones(tracks)
    assert zones[0][0].bpm == 90.0


# ---------------------------------------------------------------------------
# build_zone_shortlists
# ---------------------------------------------------------------------------


def test_build_zone_shortlists_produces_one_shortlist_per_zone() -> None:
    low = [_make_track(str(i), bpm=90.0 + i) for i in range(4)]
    high = [_make_track(str(i + 10), bpm=125.0 + i) for i in range(4)]
    shortlists = build_zone_shortlists(low + high, [], None)
    assert len(shortlists) == 2


def test_build_zone_shortlists_includes_all_seed_tracks() -> None:
    seeds = [_make_track(str(i), bpm=120.0 + i) for i in range(5)]
    shortlists = build_zone_shortlists(seeds, [], None)
    assert len(shortlists) == 1
    assert set(shortlists[0].track_ids) == {t.track_id for t in seeds}


def test_build_zone_shortlists_excludes_seeds_from_library() -> None:
    seeds = [_make_track(str(i), bpm=120.0 + i) for i in range(5)]
    # library_tracks contains same track as a seed — should not be double-counted
    library = seeds[:2] + [_make_track("99", bpm=122.0)]
    shortlists = build_zone_shortlists(seeds, library, None)
    ids = shortlists[0].track_ids
    assert ids.count("0") == 1
    assert "99" in ids


def test_build_zone_shortlists_caps_library_per_zone() -> None:
    seeds = [_make_track(str(i), bpm=120.0 + i) for i in range(5)]
    library = [_make_track(str(i + 100), bpm=122.0) for i in range(50)]
    shortlists = build_zone_shortlists(seeds, library, None)
    n_lib = len(shortlists[0].track_ids) - 5  # total minus seeds
    assert n_lib <= 20


def test_build_zone_shortlists_raises_too_few_seeds() -> None:
    seeds = [_make_track(str(i), bpm=120.0 + i) for i in range(3)]
    with pytest.raises(ValueError, match="requires at least 4"):
        build_zone_shortlists(seeds, [], None)


def test_build_zone_shortlists_title_contains_bpm_and_seed_count() -> None:
    seeds = [_make_track(str(i), bpm=120.0 + i) for i in range(5)]
    shortlists = build_zone_shortlists(seeds, [], None)
    assert "seed tracks" in shortlists[0].title
    assert "5" in shortlists[0].title


def test_filter_tracks_for_playlist_genre_standard_label() -> None:
    tracks = [
        _make_track("1").model_copy(update={"genre": "Electronica"}),
        _make_track("2").model_copy(update={"genre": "Electronic"}),
        _make_track("3").model_copy(update={"genre": "House"}),
    ]
    filtered = filter_tracks_for_playlist_genre(
        tracks,
        "electronica",
        {"electronica": ["Electronica", "Electronic"]},
        {},
    )
    assert [track.track_id for track in filtered] == ["1", "2"]


def test_filter_tracks_for_playlist_genre_direct_tag() -> None:
    tracks = [
        _make_track("1").model_copy(update={"genre": "Deep House"}),
        _make_track("2").model_copy(update={"genre": "Electronica"}),
    ]
    filtered = filter_tracks_for_playlist_genre(tracks, "Deep House", {"electronica": ["Electronica"]}, {})
    assert [track.track_id for track in filtered] == ["1"]


def test_filter_tracks_for_playlist_genre_invalid_raises() -> None:
    tracks = [_make_track("1")]
    with pytest.raises(ValueError, match="No tracks found for genre filter"):
        filter_tracks_for_playlist_genre(tracks, "techno", {"electronica": ["Electronica"]}, {})


_PLAYLIST_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="4">
        <TRACK TrackID="1" Name="One" Artist="Artist 1" AverageBpm="122.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/one.mp3"/>
        <TRACK TrackID="2" Name="Two" Artist="Artist 2" AverageBpm="123.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/two.mp3"/>
        <TRACK TrackID="3" Name="Three" Artist="Artist 3" AverageBpm="124.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/three.mp3"/>
        <TRACK TrackID="4" Name="Four" Artist="Artist 4" AverageBpm="125.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/four.mp3"/>
      </COLLECTION>
      <PLAYLISTS>
        <NODE Type="0" Name="ROOT" Count="1">
          <NODE Type="1" Name="Monday Night" KeyType="0" Entries="4">
            <TRACK Key="1"/>
            <TRACK Key="2"/>
            <TRACK Key="3"/>
            <TRACK Key="4"/>
          </NODE>
        </NODE>
      </PLAYLISTS>
    </DJ_PLAYLISTS>
""")


def test_run_playlist_mode_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import mixlab.__main__ as main_mod

    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PLAYLIST_XML)
    monkeypatch.setattr(main_mod, "_XML_PATH", xml_path)

    async def fake_stage2(
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
        canvases: object = None,
        concept_history: object = None,
        debug: bool = False,
        mix_length: int | None = None,
    ) -> tuple[list[MixConcept], str]:
        assert playlist_name == "Monday Night"
        assert seed_ids == frozenset({"1", "2", "3", "4"})
        assert seed_track_ids == ["1", "2", "3", "4"]
        assert unplayed_ids is None
        # shortlists come from build_zone_shortlists — at least one zone
        assert len(shortlists) >= 1
        del used_mix_names
        return ([MixConcept(title="Completed Set", mood="warm", track_ids=["1", "2", "3", "4"])], "Playlist report")

    async def fake_send_report(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(main_mod, "stage2_curate_and_report", fake_stage2)
    monkeypatch.setattr(main_mod, "send_report", fake_send_report)

    asyncio.run(main_mod.run_playlist_mode("Monday Night", None, None, "unplayed"))
    captured = capsys.readouterr()
    assert "Playlist report" in captured.out


_PLAYLIST_XML_WITH_EXTRA = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="5">
        <TRACK TrackID="1" Name="One" Artist="Artist 1" AverageBpm="122.00" Tonality="8A" Genre="Electronica"
               Location="file://localhost/Users/dj/Music/one.mp3"/>
        <TRACK TrackID="2" Name="Two" Artist="Artist 2" AverageBpm="123.00" Tonality="8A" Genre="Electronica"
               Location="file://localhost/Users/dj/Music/two.mp3"/>
        <TRACK TrackID="3" Name="Three" Artist="Artist 3" AverageBpm="124.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/three.mp3"/>
        <TRACK TrackID="4" Name="Four" Artist="Artist 4" AverageBpm="125.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/four.mp3"/>
        <TRACK TrackID="5" Name="Five" Artist="Artist 5" AverageBpm="124.00" Tonality="8A" Genre="Electronica"
               Location="file://localhost/Users/dj/Music/five.mp3"/>
      </COLLECTION>
      <PLAYLISTS>
        <NODE Type="0" Name="ROOT" Count="1">
          <NODE Type="1" Name="Monday Night" KeyType="0" Entries="4">
            <TRACK Key="1"/>
            <TRACK Key="2"/>
            <TRACK Key="3"/>
            <TRACK Key="4"/>
          </NODE>
        </NODE>
      </PLAYLISTS>
    </DJ_PLAYLISTS>
""")


def test_run_playlist_mode_with_genre_filter_limits_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import mixlab.__main__ as main_mod

    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PLAYLIST_XML_WITH_EXTRA)
    monkeypatch.setattr(main_mod, "_XML_PATH", xml_path)

    seen_library_genres: list[str] = []

    def fake_build_zone_shortlists(
        seed_tracks: list[Track],
        library_tracks: list[Track],
        unplayed_ids: set[str] | None,
        all_tracks_flag: bool = False,
        intent_brief: IntentBrief | None = None,
    ) -> list[MixConcept]:
        del seed_tracks, unplayed_ids, all_tracks_flag, intent_brief
        seen_library_genres.extend(sorted({t.genre for t in library_tracks}))
        return [
            MixConcept(
                title="Zone: 122–125 BPM (4 seed tracks)",
                mood="4 seed tracks, 0 library additions",
                track_ids=["1", "2", "3", "4"],
            )
        ]

    async def fake_stage2(
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
        canvases: object = None,
        concept_history: object = None,
        debug: bool = False,
        mix_length: int | None = None,
    ) -> tuple[list[MixConcept], str]:
        del shortlists, tracks_by_id, custom_genre_label, custom_genre_sub_genres
        del playlist_name, seed_ids, seed_track_ids, unplayed_ids, intent_brief, used_mix_names, canvases, debug
        return ([MixConcept(title="Completed Set", mood="warm", track_ids=["1", "2", "3", "4"])], "Playlist report")

    async def fake_send_report(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(main_mod, "build_zone_shortlists", fake_build_zone_shortlists)
    monkeypatch.setattr(main_mod, "stage2_curate_and_report", fake_stage2)
    monkeypatch.setattr(main_mod, "send_report", fake_send_report)

    asyncio.run(main_mod.run_playlist_mode("Monday Night", "electronica", None, "unplayed"))
    assert seen_library_genres == ["Electronica"]


def test_run_playlist_mode_exports_all_ranked_variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import mixlab.__main__ as main_mod

    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PLAYLIST_XML)
    monkeypatch.setattr(main_mod, "_XML_PATH", xml_path)

    exported_titles: list[str] = []
    generated_titles: list[str] = []

    async def fake_stage2(
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
        canvases: object = None,
        concept_history: object = None,
        debug: bool = False,
        mix_length: int | None = None,
    ) -> tuple[list[MixConcept], str]:
        del shortlists, tracks_by_id, custom_genre_label, custom_genre_sub_genres
        del playlist_name, seed_ids, seed_track_ids, unplayed_ids, intent_brief, used_mix_names, canvases, debug
        return (
            [
                MixConcept(title="WINNER - PRACTICAL - Gravity Well", mood="practical", track_ids=["1", "2", "3", "4"]),
                MixConcept(title="BALANCED - The Long Exhale", mood="balanced", track_ids=["1", "2", "3", "4"]),
                MixConcept(title="ADVENTUROUS - Signal & Noise", mood="adventurous", track_ids=["1", "2", "3", "4"]),
            ],
            "Playlist report",
        )

    def fake_generate_merged_xml_bytes(
        concepts: list[MixConcept],
        raw_tracks_xml: dict[str, object],
        folder_name: str = "Generated Playlists",
        unplayed_ids: list[str] | None = None,
    ) -> bytes:
        del raw_tracks_xml, folder_name, unplayed_ids
        generated_titles.extend(concept.title for concept in concepts)
        return b"<xml/>"

    def fake_export_merged_xml(
        concepts: list[MixConcept],
        raw_tracks_xml: dict[str, object],
        output_path: Path,
        folder_name: str = "Generated Playlists",
        unplayed_ids: list[str] | None = None,
    ) -> Path:
        del raw_tracks_xml, folder_name, unplayed_ids
        exported_titles.extend(concept.title for concept in concepts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"<xml/>")
        return output_path

    async def fake_send_report(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(main_mod, "stage2_curate_and_report", fake_stage2)
    monkeypatch.setattr(main_mod, "generate_merged_xml_bytes", fake_generate_merged_xml_bytes)
    monkeypatch.setattr(main_mod, "export_merged_xml", fake_export_merged_xml)
    monkeypatch.setattr(main_mod, "send_report", fake_send_report)

    asyncio.run(main_mod.run_playlist_mode("Monday Night", None, tmp_path, "unplayed"))

    expected_titles = [
        "WINNER - PRACTICAL - Gravity Well",
        "BALANCED - The Long Exhale",
        "ADVENTUROUS - Signal & Noise",
    ]
    assert generated_titles == expected_titles
    assert exported_titles == expected_titles


def test_run_playlist_mode_playlist_not_found_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import mixlab.__main__ as main_mod

    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PLAYLIST_XML)
    monkeypatch.setattr(main_mod, "_XML_PATH", xml_path)

    with pytest.raises(SystemExit, match="1"):
        asyncio.run(main_mod.run_playlist_mode("Missing", None, None, "unplayed"))


# ---------------------------------------------------------------------------
# Intent analysis helpers
# ---------------------------------------------------------------------------


def _make_track_full(
    track_id: str,
    *,
    bpm: float = 124.0,
    camelot_key: str = "8A",
    energy: int | None = None,
    play_count: int = 0,
) -> Track:
    return Track(
        track_id=track_id,
        artist=f"Artist {track_id}",
        title=f"Title {track_id}",
        bpm=bpm,
        camelot_key=camelot_key,
        genre="House",
        energy=energy,
        play_count=play_count,
    )


def test_detect_adjacency_fragments_compatible_pair_returns_high_confidence() -> None:
    t1 = _make_track_full("1", bpm=124.0, camelot_key="8A")
    t2 = _make_track_full("2", bpm=126.0, camelot_key="9A")  # adjacent Camelot, close BPM
    tracks_by_id = {"1": t1, "2": t2}
    frags = detect_adjacency_fragments(["1", "2"], tracks_by_id)
    assert len(frags) == 1
    assert frags[0].confidence >= 0.8
    assert "camelot" in frags[0].reason


def test_detect_adjacency_fragments_incompatible_pair_omitted() -> None:
    t1 = _make_track_full("1", bpm=124.0, camelot_key="1A")
    t2 = _make_track_full("2", bpm=145.0, camelot_key="7B")  # far BPM, incompatible key
    tracks_by_id = {"1": t1, "2": t2}
    frags = detect_adjacency_fragments(["1", "2"], tracks_by_id)
    assert len(frags) == 0


def test_detect_adjacency_fragments_only_camelot_compatible_returns_medium_confidence() -> None:
    t1 = _make_track_full("1", bpm=124.0, camelot_key="8A")
    t2 = _make_track_full("2", bpm=140.0, camelot_key="9A")  # compatible key, BPM too far
    tracks_by_id = {"1": t1, "2": t2}
    frags = detect_adjacency_fragments(["1", "2"], tracks_by_id)
    assert len(frags) == 1
    assert frags[0].confidence == 0.6
    assert frags[0].reason == "camelot_compatible"


def test_detect_energy_shape_rising_returns_single_arc() -> None:
    ids = ["1", "2", "3", "4", "5", "6"]
    tracks_by_id = {
        "1": _make_track_full("1", energy=1),
        "2": _make_track_full("2", energy=2),
        "3": _make_track_full("3", energy=4),
        "4": _make_track_full("4", energy=6),
        "5": _make_track_full("5", energy=7),
        "6": _make_track_full("6", energy=8),
    }
    shape = detect_energy_shape(ids, tracks_by_id)
    assert shape == "single_arc"


def test_detect_energy_shape_returns_unclear_when_sparse() -> None:
    ids = ["1", "2", "3"]
    tracks_by_id = {
        "1": _make_track_full("1", energy=None),
        "2": _make_track_full("2", energy=5),
        "3": _make_track_full("3", energy=None),
    }
    assert detect_energy_shape(ids, tracks_by_id) == "unclear"


def test_identify_missing_roles_detects_no_peak_when_all_low_energy() -> None:
    ids = [str(i) for i in range(1, 7)]
    tracks_by_id = {str(i): _make_track_full(str(i), energy=3) for i in range(1, 7)}
    missing = identify_missing_roles(ids, tracks_by_id)
    assert "peak" in missing


def test_identify_missing_roles_returns_empty_when_too_few_seeds() -> None:
    ids = ["1", "2", "3"]
    tracks_by_id = {str(i): _make_track_full(str(i), energy=3) for i in range(1, 4)}
    assert identify_missing_roles(ids, tracks_by_id) == []


def test_compute_deterministic_intent_returns_intent_brief() -> None:
    ids = ["1", "2", "3", "4", "5"]
    tracks_by_id = {
        "1": _make_track_full("1", bpm=120.0, camelot_key="8A", energy=2),
        "2": _make_track_full("2", bpm=122.0, camelot_key="9A", energy=4),
        "3": _make_track_full("3", bpm=124.0, camelot_key="8A", energy=6),
        "4": _make_track_full("4", bpm=126.0, camelot_key="8A", energy=7),
        "5": _make_track_full("5", bpm=124.0, camelot_key="9A", energy=5),
    }
    brief = compute_deterministic_intent(ids, tracks_by_id)
    assert isinstance(brief, IntentBrief)
    assert brief.bpm_range == (120.0, 126.0)
    assert len(brief.seed_analyses) == 5
    assert all(s.tier == "supporting" for s in brief.seed_analyses)


# ---------------------------------------------------------------------------
# _score_candidate
# ---------------------------------------------------------------------------


def test_score_candidate_prefers_harmonically_compatible_track() -> None:
    zone_seeds = [_make_track_full("s1", bpm=124.0, camelot_key="8A")]
    compatible = _make_track_full("c1", bpm=125.0, camelot_key="9A")  # compatible key
    incompatible = _make_track_full("c2", bpm=125.0, camelot_key="3B")  # incompatible key
    score_compat = _score_candidate(compatible, zone_seeds, 124.0, missing_roles=[], is_unplayed=False)
    score_incompat = _score_candidate(incompatible, zone_seeds, 124.0, missing_roles=[], is_unplayed=False)
    assert score_compat > score_incompat


def test_score_candidate_gives_role_bonus_for_peak_filler() -> None:
    zone_seeds = [_make_track_full("s1", bpm=124.0, camelot_key="8A")]
    peak_candidate = _make_track_full("c1", bpm=124.0, camelot_key="8A", energy=8)
    non_peak = _make_track_full("c2", bpm=124.0, camelot_key="8A", energy=3)
    score_peak = _score_candidate(peak_candidate, zone_seeds, 124.0, missing_roles=["peak"], is_unplayed=False)
    score_non = _score_candidate(non_peak, zone_seeds, 124.0, missing_roles=["peak"], is_unplayed=False)
    assert score_peak > score_non


def test_score_candidate_gives_unplayed_bonus() -> None:
    zone_seeds = [_make_track_full("s1", bpm=124.0, camelot_key="8A")]
    track = _make_track_full("c1", bpm=124.0, camelot_key="8A")
    score_unplayed = _score_candidate(track, zone_seeds, 124.0, missing_roles=[], is_unplayed=True)
    score_played = _score_candidate(track, zone_seeds, 124.0, missing_roles=[], is_unplayed=False)
    assert score_unplayed > score_played


def test_score_candidate_opener_bonus_prefers_same_genre_and_zone_fit() -> None:
    zone_seeds = [
        _make_track_full("s1", bpm=124.0, camelot_key="8A"),
        _make_track_full("s2", bpm=125.0, camelot_key="9A"),
    ]
    same_genre_opener = _make_track_full("c1", bpm=124.5, camelot_key="8A", energy=3)
    off_profile_opener = Track(
        track_id="c2",
        artist="Artist c2",
        title="Title c2",
        bpm=124.5,
        camelot_key="8A",
        genre="Electronica",
        energy=3,
    )

    score_same = _score_candidate(same_genre_opener, zone_seeds, 124.5, missing_roles=["opener"], is_unplayed=False)
    score_off = _score_candidate(off_profile_opener, zone_seeds, 124.5, missing_roles=["opener"], is_unplayed=False)
    assert score_same > score_off


def test_build_zone_shortlists_ranks_harmonically_compatible_library_tracks_first() -> None:
    """With IntentBrief, harmonically compatible library tracks rank above incompatible ones."""
    seed_tracks = [
        _make_track_full("s1", bpm=124.0, camelot_key="8A"),
        _make_track_full("s2", bpm=125.0, camelot_key="9A"),
        _make_track_full("s3", bpm=126.0, camelot_key="8A"),
        _make_track_full("s4", bpm=123.0, camelot_key="7A"),
    ]
    compatible_lib = _make_track_full("lib1", bpm=124.5, camelot_key="8A")
    incompatible_lib = _make_track_full("lib2", bpm=124.5, camelot_key="3B")

    shortlists = build_zone_shortlists(
        seed_tracks,
        [compatible_lib, incompatible_lib],
        unplayed_ids=None,
        all_tracks_flag=False,
    )
    assert len(shortlists) >= 1
    ids_in_order = shortlists[0].track_ids
    assert "lib1" in ids_in_order and "lib2" in ids_in_order
    assert ids_in_order.index("lib1") < ids_in_order.index("lib2")


def test_compute_deterministic_intent_chapter_split_sets_incoherent() -> None:
    """Seeds split by a 20 BPM gap should yield is_coherent_set=False."""
    tracks_by_id = {
        "a1": _make_track("a1", bpm=120.0),
        "a2": _make_track("a2", bpm=121.0),
        "a3": _make_track("a3", bpm=122.0),
        "b1": _make_track("b1", bpm=142.0),
        "b2": _make_track("b2", bpm=143.0),
    }
    brief = compute_deterministic_intent(["a1", "a2", "a3", "b1", "b2"], tracks_by_id)
    assert brief.is_coherent_set is False


def test_compute_deterministic_intent_tight_bpm_range_is_coherent() -> None:
    """Seeds within 10 BPM should yield is_coherent_set=True."""
    tracks_by_id = {str(i): _make_track(str(i), bpm=120.0 + i) for i in range(5)}
    brief = compute_deterministic_intent([str(i) for i in range(5)], tracks_by_id)
    assert brief.is_coherent_set is True


def test_compute_deterministic_intent_single_track_chapter_not_absorbed() -> None:
    """A single-track zone separated by >=12 BPM must produce 2 zones → incoherent."""
    tracks_by_id = {
        "lone": _make_track("lone", bpm=110.0),
        "c1": _make_track("c1", bpm=125.0),
        "c2": _make_track("c2", bpm=126.0),
        "c3": _make_track("c3", bpm=127.0),
    }
    brief = compute_deterministic_intent(["lone", "c1", "c2", "c3"], tracks_by_id)
    assert brief.is_coherent_set is False
