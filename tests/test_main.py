from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mixlab.__main__ import (
    _apply_do_not_recommend_filter,
    _apply_range_filters,
    _build_filter_desc,
    _format_pipeline_counts,
    _format_report_context,
    _print_availability,
    _print_pipeline_summary,
    _report_stage1_window,
    _validate_range_args,
    _warn_intent,
    main,
    run,
    run_export_unplayed,
)
from mixlab.models import PlayedTrack, Track
from mixlab.playlist_exporter import build_merged_xml, parse_raw_tracks
from mixlab.reader import parse_collection


def test_format_report_context_standard_genre_unplayed() -> None:
    result = _format_report_context(
        genre="electronica",
        playlist_name=None,
        mode="unplayed",
        export_dir=None,
    )
    assert result == "Report context: Electronica (unplayed tracks)"


def test_format_report_context_custom_genre_all_tracks() -> None:
    result = _format_report_context(
        genre="140",
        playlist_name=None,
        mode="all",
        export_dir=None,
    )
    assert result == "Report context: 140 (custom genre, All Tracks)"


def test_format_report_context_playlist_includes_active_options() -> None:
    result = _format_report_context(
        genre="electronica",
        playlist_name="Monday Night",
        mode="all",
        export_dir=Path("output/playlists"),
    )
    assert result == "Report context: Monday Night playlist (Electronica, All Tracks, export enabled)"


def test_format_report_context_played_mode() -> None:
    result = _format_report_context(
        genre="house",
        playlist_name=None,
        mode="played",
        export_dir=None,
    )
    assert result == "Report context: House (played tracks)"


_DO_NOT_RECOMMEND_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="4">
        <TRACK TrackID="1" Name="Allowed" Artist="Artist A" AverageBpm="124.00" Tonality="8A" Genre="House"/>
        <TRACK TrackID="2" Name="Blocked" Artist="Artist B" AverageBpm="125.00" Tonality="9A" Genre="House"/>
        <TRACK TrackID="3" Name="Blocked Nested" Artist="Artist C" AverageBpm="126.00" Tonality="10A" Genre="Techno"/>
        <TRACK TrackID="4" Name="Also Allowed" Artist="Artist D" AverageBpm="127.00" Tonality="11A" Genre="Techno"/>
      </COLLECTION>
      <PLAYLISTS>
        <NODE Type="0" Name="ROOT" Count="2">
          <NODE Type="1" Name="DO NOT RECOMMEND" KeyType="0" Entries="1">
            <TRACK Key="2"/>
            <TRACK Key="999"/>
          </NODE>
          <NODE Type="0" Name="Utility" Count="1">
            <NODE Type="1" Name="Do Not Recommend" KeyType="0" Entries="1">
              <TRACK Key="3"/>
            </NODE>
          </NODE>
        </NODE>
      </PLAYLISTS>
    </DJ_PLAYLISTS>
""")

_NO_DENYLIST_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="2">
        <TRACK TrackID="1" Name="Allowed" Artist="Artist A" AverageBpm="124.00" Tonality="8A" Genre="House"/>
        <TRACK TrackID="2" Name="Also Allowed" Artist="Artist B" AverageBpm="125.00" Tonality="9A" Genre="Techno"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")

# PLAYLISTS section present but with a differently-named playlist — no DO NOT RECOMMEND.
_WRONG_NAME_DENYLIST_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="2">
        <TRACK TrackID="1" Name="Allowed" Artist="Artist A" AverageBpm="124.00" Tonality="8A" Genre="House"/>
        <TRACK TrackID="2" Name="Blocked" Artist="Artist B" AverageBpm="125.00" Tonality="9A" Genre="House"/>
      </COLLECTION>
      <PLAYLISTS>
        <NODE Type="0" Name="ROOT" Count="1">
          <NODE Type="1" Name="My Favourites" KeyType="0" Entries="1">
            <TRACK Key="2"/>
          </NODE>
        </NODE>
      </PLAYLISTS>
    </DJ_PLAYLISTS>
""")


def _write_xml(tmp_path: Path, content: str) -> Path:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(content)
    return xml_path


def test_apply_do_not_recommend_filter_excludes_matching_playlist_tracks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xml_path = _write_xml(tmp_path, _DO_NOT_RECOMMEND_XML)
    tracks = parse_collection(xml_path)

    filtered_tracks, excluded_count = _apply_do_not_recommend_filter(tracks, xml_path)

    assert [track.track_id for track in filtered_tracks] == ["1", "4"]
    assert excluded_count == 2
    captured = capsys.readouterr()
    assert "DO NOT RECOMMEND filter: 3 IDs in playlist, excluded 2 track(s) from collection." in captured.err


def test_apply_do_not_recommend_filter_warns_when_no_playlists_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xml_path = _write_xml(tmp_path, _NO_DENYLIST_XML)
    tracks = parse_collection(xml_path)

    filtered_tracks, excluded_count = _apply_do_not_recommend_filter(tracks, xml_path)

    assert [track.track_id for track in filtered_tracks] == ["1", "2"]
    assert excluded_count == 0
    captured = capsys.readouterr()
    assert 'WARNING: "DO NOT RECOMMEND" playlist not found in XML' in captured.err


def test_apply_do_not_recommend_filter_warns_when_playlist_absent_from_playlists_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tracks must not be silently passed through when PLAYLISTS exists but DO NOT RECOMMEND is absent."""
    xml_path = _write_xml(tmp_path, _WRONG_NAME_DENYLIST_XML)
    tracks = parse_collection(xml_path)

    filtered_tracks, excluded_count = _apply_do_not_recommend_filter(tracks, xml_path)

    # Both tracks returned unchanged — no filter applied, but warning is emitted.
    assert [track.track_id for track in filtered_tracks] == ["1", "2"]
    assert excluded_count == 0
    captured = capsys.readouterr()
    assert 'WARNING: "DO NOT RECOMMEND" playlist not found in XML' in captured.err


def test_do_not_recommend_tracks_absent_from_all_unplayed_tunes_playlist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tracks filtered by the DO NOT RECOMMEND denylist must not appear in 'All Unplayed Tunes'."""
    xml_path = _write_xml(tmp_path, _DO_NOT_RECOMMEND_XML)

    tracks = parse_collection(xml_path)
    filtered_tracks, _ = _apply_do_not_recommend_filter(tracks, xml_path)
    unplayed_ids = [t.track_id for t in filtered_tracks]

    raw_tracks_xml = parse_raw_tracks(xml_path)
    root = build_merged_xml([], raw_tracks_xml, unplayed_ids=unplayed_ids)

    assert root is not None
    unplayed_node = root.find(".//PLAYLISTS//NODE[@Name='All Unplayed Tunes']")
    assert unplayed_node is not None
    keys = {el.get("Key") for el in unplayed_node.findall("TRACK")}

    # TrackIDs 2 and 3 are in DO NOT RECOMMEND playlists — must not appear
    assert "2" not in keys
    assert "3" not in keys
    # TrackIDs 1 and 4 are not denylisted — must appear
    assert keys == {"1", "4"}


def test_format_pipeline_counts_returns_compact_list() -> None:
    assert _format_pipeline_counts({"Electronica": 84, "Downtempo": 17}) == "Electronica=84, Downtempo=17"


# ---------------------------------------------------------------------------
# _apply_range_filters
# ---------------------------------------------------------------------------


def _make_track(
    track_id: str,
    bpm: float,
    year: int | None = 2020,
) -> Track:
    return Track(
        track_id=track_id,
        artist="Artist",
        title="Title",
        bpm=bpm,
        camelot_key="8A",
        genre="House",
        year=year,
    )


def test_apply_range_filters_bpm_inclusive(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0), _make_track("2", 135.0), _make_track("3", 140.0), _make_track("4", 145.0)]
    result = _apply_range_filters(tracks, min_bpm=135.0, max_bpm=140.0, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["2", "3"]
    assert "BPM filter [135–140]" in capsys.readouterr().err


def test_apply_range_filters_bpm_no_op_when_omitted(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 120.0), _make_track("2", 200.0)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["1", "2"]
    assert "BPM filter" not in capsys.readouterr().err


def test_apply_range_filters_bpm_max_only(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 120.0), _make_track("2", 130.0), _make_track("3", 140.0)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=130.0, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["1", "2"]
    assert "BPM filter [–130]" in capsys.readouterr().err


def test_apply_range_filters_year_excludes_none_year(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=2021), _make_track("2", 130.0, year=None)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=2020, max_year=None)
    assert [t.track_id for t in result] == ["1"]
    assert "Year filter" in capsys.readouterr().err


def test_apply_range_filters_year_excludes_zero_year(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=2021), _make_track("2", 130.0, year=0)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=2020, max_year=None)
    assert [t.track_id for t in result] == ["1"]
    assert "Year filter" in capsys.readouterr().err


def test_apply_range_filters_year_no_op_when_omitted(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=None), _make_track("2", 130.0, year=0), _make_track("3", 130.0, year=2022)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=None, max_year=None)
    assert [t.track_id for t in result] == ["1", "2", "3"]
    assert "Year filter" not in capsys.readouterr().err


def test_apply_range_filters_year_max_only(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=2018), _make_track("2", 130.0, year=2022)]
    result = _apply_range_filters(tracks, min_bpm=None, max_bpm=None, min_year=None, max_year=2019)
    assert [t.track_id for t in result] == ["1"]
    assert "Year filter [–2019]" in capsys.readouterr().err


def test_apply_range_filters_logs_format_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    tracks = [_make_track("1", 130.0, year=2018), _make_track("2", 135.0, year=2022)]
    _apply_range_filters(tracks, min_bpm=135.0, max_bpm=140.0, min_year=2020, max_year=None)
    err = capsys.readouterr().err
    assert "BPM filter [135–140]: excluded 1 track(s), 1 remain." in err
    assert "Year filter [2020–]: excluded 0 track(s), 1 remain." in err


def test_range_filter_preserves_out_of_range_seeds_excludes_library() -> None:
    seed_tracks = [_make_track("seed1", 120.0, year=2015)]
    all_tracks = seed_tracks + [
        _make_track("lib1", 135.0, year=2022),
        _make_track("lib2", 150.0, year=2022),
    ]
    seed_ids = frozenset(t.track_id for t in seed_tracks)
    library_tracks = [t for t in all_tracks if t.track_id not in seed_ids]

    filtered_library = _apply_range_filters(library_tracks, min_bpm=130.0, max_bpm=140.0, min_year=None, max_year=None)

    assert [t.track_id for t in seed_tracks] == ["seed1"]
    assert [t.track_id for t in filtered_library] == ["lib1"]


def test_format_pipeline_counts_returns_none_for_empty_dict() -> None:
    assert _format_pipeline_counts({}) == "none"


def test_print_availability_shows_excluded_count_when_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.models import Track

    track = Track(track_id="1", artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House")
    _print_availability([track], [track], show_unplayed=True, excluded_count=3)

    captured = capsys.readouterr()
    assert "3 track(s) excluded (DO NOT RECOMMEND)" in captured.out


def test_print_availability_omits_excluded_line_when_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.models import Track

    track = Track(track_id="1", artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House")
    _print_availability([track], [track], show_unplayed=True, excluded_count=0)

    captured = capsys.readouterr()
    assert "excluded" not in captured.out


def test_print_pipeline_summary_outputs_compact_block(capsys: pytest.CaptureFixture[str]) -> None:
    _print_pipeline_summary(
        collection_count=2328,
        unplayed_count=101,
        used_catalog_api=True,
        genre_cluster_counts={"Electronica": 84, "Downtempo": 17},
        bpm_filtered_counts={"Electronica": 41, "Downtempo": 9},
        same_genre_outlier_count=0,
        stage1_shortlist_count=2,
        stage2_shortlist_count=2,
    )

    captured = capsys.readouterr()
    assert captured.out == (
        "Pipeline summary:\n"
        "- Track pool: 101 unplayed / 2328 in scoped collection\n"
        "- Genre scope before BPM filter: Electronica=84, Downtempo=17\n"
        "- Genre scope after BPM filter: Electronica=41, Downtempo=9\n"
        "- Same-genre outliers considered: 0\n"
        "- Stage 1 shortlists: 2\n"
        "- Stage 2 shortlists sent: 2\n\n"
    )


def test_print_pipeline_summary_includes_overflow_line_when_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    # #48: the overflow line appears only when windowing dropped tracks (default 0 omits it,
    # keeping the compact-block test above unchanged).
    _print_pipeline_summary(
        collection_count=2328,
        unplayed_count=101,
        used_catalog_api=True,
        genre_cluster_counts={"House": 800},
        bpm_filtered_counts={"House": 800},
        same_genre_outlier_count=0,
        stage1_shortlist_count=5,
        stage2_shortlist_count=5,
        stage1_overflow=675,
    )
    out = capsys.readouterr().out
    assert "- Stage 1 overflow (tracks beyond windows): 675\n" in out
    # Ordered between the shortlist count and the stage-2 line.
    assert out.index("Stage 1 shortlists") < out.index("Stage 1 overflow") < out.index("Stage 2 shortlists sent")


def test_report_stage1_window_notes_only_capped_shortlists_and_totals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # #48: one stderr note per windowed shortlist; zero-overflow shortlists are silent.
    total = _report_stage1_window([0, 12, 0, 3])
    assert total == 15
    err = capsys.readouterr().err
    assert "shortlist 1 capped at 25 tracks (12 overflow, rotated by seed)" in err
    assert "shortlist 3 capped at 25 tracks (3 overflow, rotated by seed)" in err
    assert "shortlist 0" not in err
    assert "shortlist 2" not in err


# ---------------------------------------------------------------------------
# _validate_range_args
# ---------------------------------------------------------------------------


def test_main_rejects_inverted_bpm_range(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _validate_range_args(min_bpm=140.0, max_bpm=130.0, min_year=None, max_year=None)
    assert exc_info.value.code == 1
    assert "--min-bpm" in capsys.readouterr().err


def test_main_rejects_inverted_year_range(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _validate_range_args(min_bpm=None, max_bpm=None, min_year=2024, max_year=2020)
    assert exc_info.value.code == 1
    assert "--min-year" in capsys.readouterr().err


def test_validate_range_args_passes_when_valid() -> None:
    _validate_range_args(min_bpm=130.0, max_bpm=140.0, min_year=2019, max_year=2024)
    _validate_range_args(min_bpm=None, max_bpm=None, min_year=None, max_year=None)


# ---------------------------------------------------------------------------
# _build_filter_desc
# ---------------------------------------------------------------------------


def test_build_filter_desc_both_bpm_and_year() -> None:
    assert _build_filter_desc(min_bpm=126.0, max_bpm=130.0, min_year=2001, max_year=None) == "BPM 126–130, year ≥ 2001"


def test_build_filter_desc_bpm_only_range() -> None:
    assert _build_filter_desc(min_bpm=126.0, max_bpm=130.0, min_year=None, max_year=None) == "BPM 126–130"


def test_build_filter_desc_bpm_min_only() -> None:
    assert _build_filter_desc(min_bpm=126.0, max_bpm=None, min_year=None, max_year=None) == "BPM ≥ 126"


def test_build_filter_desc_bpm_max_only() -> None:
    assert _build_filter_desc(min_bpm=None, max_bpm=130.0, min_year=None, max_year=None) == "BPM ≤ 130"


def test_build_filter_desc_year_range() -> None:
    assert _build_filter_desc(min_bpm=None, max_bpm=None, min_year=2001, max_year=2019) == "year 2001–2019"


def test_build_filter_desc_none_when_no_filters() -> None:
    assert _build_filter_desc(min_bpm=None, max_bpm=None, min_year=None, max_year=None) is None


# ---------------------------------------------------------------------------
# run_export_unplayed
# ---------------------------------------------------------------------------

_EXPORT_UNPLAYED_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="3">
        <TRACK TrackID="1" Name="Unplayed Track" Artist="Artist A" AverageBpm="124.00" Tonality="8A" Genre="House"
               Location="file://localhost/music/a.mp3"/>
        <TRACK TrackID="2" Name="Played Track" Artist="Artist B" AverageBpm="125.00" Tonality="9A" Genre="House"
               Location="file://localhost/music/b.mp3"/>
        <TRACK TrackID="3" Name="Also Unplayed" Artist="Artist C" AverageBpm="126.00" Tonality="10A" Genre="Techno"
               Location="file://localhost/music/c.mp3"/>
      </COLLECTION>
      <PLAYLISTS>
        <NODE Type="0" Name="ROOT" Count="1">
          <NODE Type="1" Name="DO NOT RECOMMEND" KeyType="0" Entries="0"/>
        </NODE>
      </PLAYLISTS>
    </DJ_PLAYLISTS>
""")


async def test_run_export_unplayed_exits_without_catalog_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        await run_export_unplayed()
    assert exc_info.value.code == 1
    assert "CATALOG_API_URL" in capsys.readouterr().err


async def test_run_export_unplayed_exports_only_unplayed_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_EXPORT_UNPLAYED_XML)

    played = [PlayedTrack(artist="Artist B", title="Played Track")]

    monkeypatch.setenv("CATALOG_API_URL", "http://fake")
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)

    with (
        patch("mixlab.__main__.fetch_played_tracks", new=AsyncMock(return_value=played)),
        patch("mixlab.__main__.export_merged_xml", return_value=tmp_path / "out.xml") as mock_export,
        patch("mixlab.__main__.generate_merged_xml_bytes", return_value=b"<xml/>"),
        patch("mixlab.__main__.send_report", new=AsyncMock()),
    ):
        await run_export_unplayed()

    _, _, _, folder_name, exported_ids = mock_export.call_args.args
    assert exported_ids is not None
    assert "2" not in exported_ids
    assert "1" in exported_ids
    assert "3" in exported_ids
    out = capsys.readouterr().out
    assert "2 / 3" in out


# ---------------------------------------------------------------------------
# run() --mode all played/unplayed signal handling
# ---------------------------------------------------------------------------

# Minimal collection used for --mode all dispatch tests. genre=None makes run() exit
# after the availability table, so we exercise the mode-dispatch logic without
# triggering Stage 1 / Stage 2 LLM calls.
_MODE_ALL_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="3">
        <TRACK TrackID="1" Name="A" Artist="Artist A" AverageBpm="124.00" Tonality="8A" Genre="House"/>
        <TRACK TrackID="2" Name="B" Artist="Artist B" AverageBpm="125.00" Tonality="9A" Genre="House"/>
        <TRACK TrackID="3" Name="C" Artist="Artist C" AverageBpm="126.00" Tonality="10A" Genre="House"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")


async def test_run_mode_all_warns_when_catalog_url_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--mode all without CATALOG_API_URL emits a one-line stderr warning and continues."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_MODE_ALL_XML)
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    # save_genre_cache writes to cwd-relative path; chdir to tmp so the project's real cache is untouched.
    monkeypatch.chdir(tmp_path)

    await run(genre=None, export_dir=None, mode="all")

    captured = capsys.readouterr()
    assert "CATALOG_API_URL not set" in captured.err
    assert "played/unplayed signal disabled" in captured.err


async def test_run_mode_all_fetches_played_when_catalog_url_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--mode all with CATALOG_API_URL set fetches played tracks and reports played/unplayed split."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_MODE_ALL_XML)
    monkeypatch.setenv("CATALOG_API_URL", "http://fake")
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    played = [PlayedTrack(artist="Artist B", title="B")]
    with (
        patch("mixlab.__main__.fetch_played_tracks", new=AsyncMock(return_value=played)),
        patch("mixlab.__main__.fetch_mix_names", new=AsyncMock(return_value=[])),
    ):
        await run(genre=None, export_dir=None, mode="all")

    captured = capsys.readouterr()
    # Status message includes the played/unplayed counts (1 of 3 played, 2 unplayed).
    assert "1/3 played" in captured.out
    assert "2 unplayed" in captured.out
    assert "favour unplayed in ties" in captured.out


async def test_run_mode_all_handles_catalog_fetch_failure_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--mode all warns and continues when the catalog API call raises."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_MODE_ALL_XML)
    monkeypatch.setenv("CATALOG_API_URL", "http://fake")
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    fetch_played = AsyncMock(side_effect=RuntimeError("network down"))
    with (
        patch("mixlab.__main__.fetch_played_tracks", fetch_played),
        patch("mixlab.__main__.fetch_mix_names", new=AsyncMock(return_value=[])),
    ):
        # Should NOT raise — graceful fallback per --mode all design.
        await run(genre=None, export_dir=None, mode="all")

    captured = capsys.readouterr()
    assert "catalog fetch failed" in captured.err
    assert "played/unplayed signal disabled" in captured.err


# ---------------------------------------------------------------------------
# _warn_intent
# ---------------------------------------------------------------------------


def test_warn_intent_none_emits_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent(None)
    assert capsys.readouterr().err == ""


def test_warn_intent_empty_string_warns(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent("   ")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "empty" in err


def test_warn_intent_short_intent_warns(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent("dark")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "short" in err
    assert "1 word" in err


def test_warn_intent_four_words_warns(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent("dark and very cool")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "4 words" in err


def test_warn_intent_five_words_no_warning(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent("dark hypnotic late night opener")
    assert capsys.readouterr().err == ""


def test_warn_intent_exactly_50_words_no_warning(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent(" ".join(["word"] * 50))
    assert capsys.readouterr().err == ""


def test_warn_intent_51_words_warns(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_intent(" ".join(["word"] * 51))
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "51" in err


# ---------------------------------------------------------------------------
# _format_report_context — mix_length in genre mode (issue #49)
# ---------------------------------------------------------------------------


def test_format_report_context_genre_mode_shows_mix_length() -> None:
    result = _format_report_context(
        genre="house",
        playlist_name=None,
        mode="unplayed",
        export_dir=None,
        mix_length=60,
    )
    assert result == "Report context: House (unplayed tracks, 60min set)"


# ---------------------------------------------------------------------------
# main() — --mix-length accepted in genre mode (issue #49)
# ---------------------------------------------------------------------------


def test_main_accepts_mix_length_in_genre_mode_without_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--mix-length used to warn and be ignored outside --playlist; it now applies in genre mode too."""
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--mix-length", "60"])
    run_mock = AsyncMock(return_value=None)

    # Real load_dotenv() would pollute os.environ (e.g. CATALOG_API_URL) for the rest of
    # the test session, so stub it out — irrelevant to what this test verifies.
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()

    err = capsys.readouterr().err
    assert "--mix-length is only used in playlist mode" not in err
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["mix_length"] == 60
