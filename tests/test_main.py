from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mixlab.__main__ import (
    _annotate_direction_types,
    _apply_do_not_recommend_filter,
    _apply_range_filters,
    _build_filter_desc,
    _format_pipeline_counts,
    _format_report_context,
    _print_availability,
    _print_pipeline_summary,
    _report_stage1_window,
    _resolve_collection_path,
    _validate_range_args,
    _warn_intent,
    main,
    run,
    run_export_unplayed,
    run_feedback,
    run_prep,
)
from mixlab.history import ConceptHistory, ConceptRecord, HistoryEntry, append_run, load_history
from mixlab.models import (
    CanvasRoleCandidates,
    CanvasScore,
    ContrastAssets,
    MixCanvas,
    MixConcept,
    PlayedTrack,
    Track,
)
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


def test_resolve_collection_path_defaults_to_local_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIXLAB_COLLECTION_PATH", raising=False)
    assert _resolve_collection_path() == Path("import/rekordbox.xml")


def test_resolve_collection_path_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_COLLECTION_PATH", "/srv/runs/r1/collection.xml")
    assert _resolve_collection_path() == Path("/srv/runs/r1/collection.xml")


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
# run() --track-pool restriction ("Run this block", mixlab-web)
# ---------------------------------------------------------------------------

# These tests use _house_collection_xml (defined further below, under the standalone
# HTML report tests) for a 20-track collection (TrackID "1".."20") — big enough to
# clear MIN_SHORTLIST after restriction. genre=None makes run() exit after the
# availability table (step 4), same short-circuit used by the --mode all dispatch
# tests above, so the restriction and its fatal checks (which all sit before step 4)
# are exercised without Stage 1/2.


async def test_run_track_pool_forces_mode_all_with_stderr_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--track-pool ids are authoritative — a conflicting --mode is overridden, not fatal."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml(20))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    pool_ids = [str(i) for i in range(1, 17)]  # 16 of 20 — clears MIN_SHORTLIST
    track_pool_raw = json.dumps({"track_ids": pool_ids, "label": "Monday block"})

    await run(genre=None, export_dir=None, mode="unplayed", track_pool_raw=track_pool_raw)

    err = capsys.readouterr().err
    assert "--track-pool: ids are authoritative — forcing --mode all (was unplayed)" in err


async def test_run_track_pool_restricts_unplayed_before_availability_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The resolved pool replaces `unplayed` before _print_availability (step 3) runs."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml(20))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    pool_ids = [str(i) for i in range(1, 17)]  # 16 of 20
    track_pool_raw = json.dumps({"track_ids": pool_ids, "label": "Monday block"})

    availability_mock = Mock(return_value=({}, 0, {}))
    with patch("mixlab.__main__._print_availability", availability_mock):
        await run(genre=None, export_dir=None, mode="all", track_pool_raw=track_pool_raw)

    assert availability_mock.call_args is not None
    passed_unplayed = availability_mock.call_args.args[1]
    assert {t.track_id for t in passed_unplayed} == set(pool_ids)
    assert availability_mock.call_args.kwargs["pool_label"] == 'block "Monday block"'

    out = capsys.readouterr().out
    assert '--track-pool "Monday block": 16 of 16 ids resolved.' in out


async def test_run_track_pool_unknown_ids_silently_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ids that don't resolve against the collection are dropped, not errored — only the
    resolution count (printed once) reflects them."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml(20))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    known_ids = [str(i) for i in range(1, 17)]  # 16 known
    unknown_ids = ["unknown-a", "unknown-b"]  # 2 unknown
    track_pool_raw = json.dumps({"track_ids": known_ids + unknown_ids, "label": "Monday block"})

    await run(genre=None, export_dir=None, mode="all", track_pool_raw=track_pool_raw)

    out = capsys.readouterr().out
    assert '--track-pool "Monday block": 16 of 18 ids resolved.' in out
    # Printed exactly once.
    assert out.count("ids resolved.") == 1


async def test_run_track_pool_below_min_shortlist_exits_with_stale_block_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fewer than MIN_SHORTLIST resolved ids is fatal — the block is stale, not just thin."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml(20))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    pool_ids = [str(i) for i in range(1, 11)]  # 10 of 20 — below MIN_SHORTLIST (15)
    track_pool_raw = json.dumps({"track_ids": pool_ids, "label": "Thin block"})

    with pytest.raises(SystemExit) as exc_info:
        await run(genre=None, export_dir=None, mode="all", track_pool_raw=track_pool_raw)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "track pool resolves to 10 tracks (< 15)" in err
    assert "the block is stale against this collection" in err


async def test_run_track_pool_and_direction_spec_together_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--track-pool and --direction-spec are two different scoping mechanisms — combining
    them is a fatal operator error, not a silent precedence rule."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml(20))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    track_pool_raw = json.dumps({"track_ids": [str(i) for i in range(1, 17)]})

    with pytest.raises(SystemExit) as exc_info:
        await run(
            genre=None,
            export_dir=None,
            mode="all",
            direction_spec='{"direction_type": "artist_thread"}',
            track_pool_raw=track_pool_raw,
        )

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "--track-pool and --direction-spec cannot be combined" in err


async def test_run_track_pool_invalid_json_exits_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml(20))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        await run(genre=None, export_dir=None, mode="all", track_pool_raw="{not json")

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "--track-pool is not valid JSON" in err


# ---------------------------------------------------------------------------
# main() — --track-pool flag parsing
# ---------------------------------------------------------------------------


def test_main_track_pool_forwarded_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_json = '{"track_ids": ["1", "2"], "label": "Monday block"}'
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--track-pool", pool_json])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["track_pool_raw"] == pool_json


def test_main_track_pool_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["track_pool_raw"] is None


def test_main_track_pool_ignored_in_playlist_mode_with_note(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--playlist", "Monday", "--track-pool", '{"track_ids": ["1"]}'])
    playlist_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run_playlist_mode", playlist_mock):
        main()
    err = capsys.readouterr().err
    assert "--track-pool ignored in playlist mode" in err
    assert playlist_mock.await_args is not None


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


# ---------------------------------------------------------------------------
# run_feedback / mixlab --feedback (#52)
# ---------------------------------------------------------------------------


def _minimal_entry(run_id: str, genre: str = "house") -> HistoryEntry:
    return HistoryEntry(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        mode="standard",
        genre=genre,
        selected_canvas_ids=[],
        dominant_bpm_clusters=[],
        dominant_camelot_keys=[],
        core_track_ids=[],
        anchor_track_ids=[],
        opener_candidates=[],
        closer_candidates=[],
        concept_title="",
        concept_track_ids=[],
        energy_path="",
        mood="",
    )


def test_run_feedback_no_flags_lists_most_recent_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    history_path = tmp_path / "history.json"
    entry = _minimal_entry("r1")
    entry.concepts = [
        ConceptRecord(concept_id="c1", title="First", mood="dark", track_ids=["T001"], arc_type=""),
        ConceptRecord(
            concept_id="c2", title="Second", mood="light", track_ids=["T002"], arc_type="", feedback="played"
        ),
    ]
    append_run(ConceptHistory(), entry, history_path)

    exit_code = run_feedback(None, None, "", history_path)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "First" in out
    assert "(none)" in out
    assert "Second" in out
    assert "played" in out


def test_run_feedback_no_history_file_lists_nothing_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = run_feedback(None, None, "", tmp_path / "no-such-file.json")
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "No concept history found" in out


def test_run_feedback_records_verdict_and_persists_to_file(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _minimal_entry("r1")
    entry.concepts = [
        ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T001"], arc_type="")
    ]
    append_run(ConceptHistory(), entry, history_path)

    exit_code = run_feedback("midnight run", "played", "great opener", history_path)

    assert exit_code == 0
    reloaded = load_history(history_path)
    record = reloaded.runs[0].concepts[0]
    assert record.feedback == "played"
    assert record.feedback_notes == "great opener"
    assert record.feedback_recorded_at != ""


def test_run_feedback_matches_concept_id_prefix(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _minimal_entry("r1")
    entry.concepts = [
        ConceptRecord(concept_id="abcdef12-3456", title="Midnight Run", mood="dark", track_ids=["T001"], arc_type="")
    ]
    append_run(ConceptHistory(), entry, history_path)

    exit_code = run_feedback("abcdef12", "rejected", "", history_path)

    assert exit_code == 0
    reloaded = load_history(history_path)
    assert reloaded.runs[0].concepts[0].feedback == "rejected"


def test_run_feedback_uses_most_recent_run_when_title_repeats(tmp_path: Path) -> None:
    """Same title generated in two runs — the most recent run's concept is updated."""
    from mixlab.history import save_history

    history_path = tmp_path / "history.json"
    older = _minimal_entry("r1")
    older.created_at = "2026-01-01T00:00:00+00:00"
    older.concepts = [ConceptRecord(concept_id="old", title="Repeat", mood="dark", track_ids=["T001"], arc_type="")]
    newer = _minimal_entry("r2")
    newer.created_at = "2026-02-01T00:00:00+00:00"
    newer.concepts = [ConceptRecord(concept_id="new", title="Repeat", mood="light", track_ids=["T002"], arc_type="")]
    save_history(ConceptHistory(runs=[older, newer]), history_path)

    exit_code = run_feedback("repeat", "played", "", history_path)

    assert exit_code == 0
    reloaded = load_history(history_path)
    assert reloaded.runs[0].concepts[0].feedback == ""
    assert reloaded.runs[1].concepts[0].feedback == "played"


def test_run_feedback_unknown_title_exits_one_with_suggestions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history_path = tmp_path / "history.json"
    entry = _minimal_entry("r1")
    entry.concepts = [
        ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T001"], arc_type="")
    ]
    append_run(ConceptHistory(), entry, history_path)

    exit_code = run_feedback("Midnigth Run", "played", "", history_path)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "No concept found" in err
    assert "Midnight Run" in err


def test_run_feedback_verdict_without_concept_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_feedback(None, "played", "", tmp_path / "history.json")
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "--concept" in err


def test_run_feedback_concept_without_verdict_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_feedback("Some Title", None, "", tmp_path / "history.json")
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "--verdict" in err


def test_main_feedback_flag_lists_concepts_via_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history_path = tmp_path / "history.json"
    entry = _minimal_entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="First", mood="dark", track_ids=["T001"], arc_type="")]
    append_run(ConceptHistory(), entry, history_path)

    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", history_path)
    monkeypatch.setattr("sys.argv", ["mixlab", "--feedback"])
    with patch("mixlab.__main__.load_dotenv"), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "First" in capsys.readouterr().out


def test_main_feedback_flag_records_verdict_via_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.json"
    entry = _minimal_entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="First", mood="dark", track_ids=["T001"], arc_type="")]
    append_run(ConceptHistory(), entry, history_path)

    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", history_path)
    monkeypatch.setattr(
        "sys.argv",
        ["mixlab", "--feedback", "--concept", "First", "--verdict", "played", "--notes", "great"],
    )
    with patch("mixlab.__main__.load_dotenv"), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    reloaded = load_history(history_path)
    assert reloaded.runs[0].concepts[0].feedback == "played"
    assert reloaded.runs[0].concepts[0].feedback_notes == "great"


# ---------------------------------------------------------------------------
# main() — --directions flag parsing (#53)
# ---------------------------------------------------------------------------


def test_main_directions_defaults_to_mixed_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["directions"] == "mixed"


def test_main_no_revise_defaults_false_and_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["no_revise"] is False


def test_main_no_revise_flag_sets_kwarg_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--no-revise"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["no_revise"] is True


def test_main_directions_off_accepted_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--directions", "off"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["directions"] == "off"


def test_main_directions_only_accepted_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--directions", "only"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["directions"] == "only"


def test_main_directions_ignored_in_playlist_mode_with_note(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--playlist", "Monday", "--directions", "only"])
    playlist_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run_playlist_mode", playlist_mock):
        main()
    err = capsys.readouterr().err
    assert "--directions ignored in playlist mode" in err
    assert playlist_mock.await_args is not None


# ---------------------------------------------------------------------------
# main() — --risk flag parsing (#42)
# ---------------------------------------------------------------------------


def test_main_risk_defaults_to_medium_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["risk"] == "medium"


def test_main_risk_high_accepted_and_threaded_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--risk", "high"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["risk"] == "high"


def test_main_risk_low_accepted_and_threaded_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--risk", "low"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["risk"] == "low"


def test_main_risk_high_ignored_in_playlist_mode_with_note(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--playlist", "Monday", "--risk", "high"])
    playlist_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run_playlist_mode", playlist_mock):
        main()
    err = capsys.readouterr().err
    assert "--risk ignored in playlist mode" in err
    assert playlist_mock.await_args is not None


def test_main_risk_medium_no_note_in_playlist_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default risk (medium) must not trigger the playlist-mode ignore note."""
    monkeypatch.setattr("sys.argv", ["mixlab", "--playlist", "Monday"])
    playlist_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run_playlist_mode", playlist_mock):
        main()
    err = capsys.readouterr().err
    assert "--risk ignored in playlist mode" not in err


# ---------------------------------------------------------------------------
# main() — --resequence flag parsing (#61)
# ---------------------------------------------------------------------------


def test_main_resequence_defaults_false_and_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["resequence"] is False


def test_main_resequence_flag_sets_kwarg_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--resequence"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["resequence"] is True


# ---------------------------------------------------------------------------
# main() — --intent works in playlist mode too (#54)
# ---------------------------------------------------------------------------


def test_main_playlist_mode_intent_no_longer_ignored(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--playlist combined with --intent must no longer print an 'ignored' warning,
    and the intent text must be threaded through to run_playlist_mode."""
    monkeypatch.setattr(
        "sys.argv",
        ["mixlab", "--playlist", "Monday", "--intent", "dark hypnotic late night warmup"],
    )
    playlist_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run_playlist_mode", playlist_mock):
        main()
    err = capsys.readouterr().err
    assert "--intent ignored in playlist mode" not in err
    assert playlist_mock.await_args is not None
    assert playlist_mock.await_args.kwargs["intent"] == "dark hypnotic late night warmup"


# ---------------------------------------------------------------------------
# run() — standalone HTML report attachment (#45)
# ---------------------------------------------------------------------------


def _house_collection_xml(n: int = 18) -> str:
    keys = ["8A", "9A", "10A", "11A", "12A", "1A", "2A", "3A", "4A", "5A", "6A", "7A"]
    rows = "\n".join(
        f'    <TRACK TrackID="{i}" Name="Track {i}" Artist="Artist {i}" '
        f'AverageBpm="124.00" Tonality="{keys[i % len(keys)]}" Genre="House" '
        f'TotalTime="300" Rating="0"/>'
        for i in range(1, n + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<DJ_PLAYLISTS Version="1.0.0">\n'
        f'  <COLLECTION Entries="{n}">\n{rows}\n  </COLLECTION>\n'
        "</DJ_PLAYLISTS>\n"
    )


async def test_run_genre_mode_attaches_html_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full genre run writes the HTML report to MIXLAB_REPORT_DIR and attaches it to Discord."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml())
    report_dir = tmp_path / "reports"

    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setenv("MIXLAB_REPORT_DIR", str(report_dir))
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.chdir(tmp_path)

    concepts = [MixConcept(title="Deep Cuts", mood="warm", track_ids=["1", "2", "3", "4"])]
    stage2 = AsyncMock(return_value=(concepts, "concept prose\n\n---\n\nmain-brain note"))
    send_report_mock = AsyncMock()

    with (
        patch("mixlab.__main__.stage2_curate_and_report", stage2),
        patch("mixlab.__main__.validate_stage2_output", return_value=[]),
        patch("mixlab.__main__.send_report", send_report_mock),
    ):
        await run(genre="house", export_dir=None, directions="off")

    assert send_report_mock.await_args is not None
    attachments = send_report_mock.await_args.kwargs["attachments"]
    html_names = [name for name, _ in attachments]
    assert any(name.endswith(".html") for name in html_names)
    # The HTML file was written to the overridden report dir.
    written = list(report_dir.glob("*.html"))
    assert len(written) == 1
    assert written[0].name.startswith("mixlab-house-")


async def test_run_genre_mode_scores_directions_against_the_scoped_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_directions gets the whole mode-scoped pool as `collection`, not just
    the genre-scoped direction pool. label_spotlight measures a label's concentration
    against that baseline; without it the run path and the library map (#40), which
    passes its own scoped collection, would score the same label differently."""
    xml_path = tmp_path / "rekordbox.xml"
    # 18 house tracks (the direction pool) plus 4 techno tracks that are in the
    # collection but out of scope for a --genre house run.
    lines = _house_collection_xml(18).split("\n")
    techno_rows = "\n".join(
        f'    <TRACK TrackID="{100 + i}" Name="T{i}" Artist="A{i}" AverageBpm="140.00" '
        f'Tonality="8A" Genre="Techno" TotalTime="300" Rating="0"/>'
        for i in range(4)
    )
    closing = lines.index("  </COLLECTION>")
    xml_path.write_text("\n".join(lines[:closing] + [techno_rows] + lines[closing:]))

    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setenv("MIXLAB_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.chdir(tmp_path)

    concepts = [MixConcept(title="Deep Cuts", mood="warm", track_ids=["1", "2", "3", "4"])]
    stage2 = AsyncMock(return_value=(concepts, "concept prose\n\n---\n\nmain-brain note"))
    directions_mock = Mock(return_value=[])

    with (
        patch("mixlab.__main__.generate_directions", directions_mock),
        patch("mixlab.__main__.stage2_curate_and_report", stage2),
        patch("mixlab.__main__.validate_stage2_output", return_value=[]),
        patch("mixlab.__main__.send_report", AsyncMock()),
    ):
        await run(genre="house", export_dir=None, directions="mixed")

    assert directions_mock.call_args is not None
    collection = directions_mock.call_args.kwargs["collection"]
    direction_pool = directions_mock.call_args.args[0]
    assert {t.genre for t in collection} == {"House", "Techno"}  # whole scope, not just the pool
    assert len(collection) > len(direction_pool)


# ---------------------------------------------------------------------------
# run() — summary.json run artifact + conceptId unification (#89 / M1)
# ---------------------------------------------------------------------------


async def test_run_genre_mode_writes_summary_json_next_to_html_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full genre run writes summary.json next to report.html (same stem), and the
    concept carries a non-empty conceptId stamped post-Stage-2 (#89)."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_house_collection_xml())
    report_dir = tmp_path / "reports"

    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    monkeypatch.setenv("MIXLAB_REPORT_DIR", str(report_dir))
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.chdir(tmp_path)

    concepts = [MixConcept(title="Deep Cuts", mood="warm", track_ids=["1", "2", "3", "4"])]
    stage2 = AsyncMock(return_value=(concepts, "concept prose\n\n---\n\nmain-brain note"))
    send_report_mock = AsyncMock()

    with (
        patch("mixlab.__main__.stage2_curate_and_report", stage2),
        patch("mixlab.__main__.validate_stage2_output", return_value=[]),
        patch("mixlab.__main__.send_report", send_report_mock),
    ):
        await run(genre="house", export_dir=None, directions="off")

    written_html = list(report_dir.glob("*.html"))
    written_json = list(report_dir.glob("*.json"))
    assert len(written_html) == 1
    assert len(written_json) == 1
    assert written_json[0].stem == written_html[0].stem

    summary = json.loads(written_json[0].read_text())
    assert summary["schemaVersion"] == 1
    assert summary["concepts"][0]["title"] == "Deep Cuts"
    assert summary["concepts"][0]["conceptId"]  # stamped non-empty by run() (#89)
    assert summary["flags"]["genre"] == "house"
    assert summary["flags"]["directions"] == "off"


# ---------------------------------------------------------------------------
# run_prep / mixlab --prep (#74) — Cue-Prep Assistant
# ---------------------------------------------------------------------------

# House bucket: track 1 fully cued (excluded), 2 and 4 uncued, 3 half-cued (no mix-out)
# -> 3 of 4 gapped. Techno bucket: 5 and 6 both uncued -> 2 of 2 gapped. House has the
# higher gap count so its footer line sorts first.
_PREP_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="6">
        <TRACK TrackID="1" Name="Full Cue" Artist="Artist A" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300">
          <POSITION_MARK Name="" Type="0" Start="20.0" Num="0"/>
          <POSITION_MARK Name="" Type="0" Start="250.0" Num="1"/>
        </TRACK>
        <TRACK TrackID="2" Name="No Cue" Artist="Artist B" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300"/>
        <TRACK TrackID="3" Name="Half Cue" Artist="Artist C" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300">
          <POSITION_MARK Name="" Type="0" Start="20.0" Num="0"/>
        </TRACK>
        <TRACK TrackID="4" Name="No Cue Too" Artist="Artist D" AverageBpm="125.00" Tonality="9A"
               Genre="House" TotalTime="300"/>
        <TRACK TrackID="5" Name="Techno No Cue" Artist="Artist E" AverageBpm="128.00" Tonality="8A"
               Genre="Techno" TotalTime="300"/>
        <TRACK TrackID="6" Name="Techno No Cue Too" Artist="Artist F" AverageBpm="128.00" Tonality="8A"
               Genre="Techno" TotalTime="300"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")

# Every track fully cued (two marks, second past the midpoint) -> nothing to prep.
_PREP_ALL_CUED_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="2">
        <TRACK TrackID="1" Name="Full Cue A" Artist="Artist A" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300">
          <POSITION_MARK Name="" Type="0" Start="20.0" Num="0"/>
          <POSITION_MARK Name="" Type="0" Start="250.0" Num="1"/>
        </TRACK>
        <TRACK TrackID="2" Name="Full Cue B" Artist="Artist B" AverageBpm="128.00" Tonality="9A"
               Genre="Techno" TotalTime="300">
          <POSITION_MARK Name="" Type="0" Start="20.0" Num="0"/>
          <POSITION_MARK Name="" Type="0" Start="260.0" Num="1"/>
        </TRACK>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")


# _PREP_XML plus a DO NOT RECOMMEND playlist denylisting track 2 ("No Cue").
_PREP_DENYLIST_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="4">
        <TRACK TrackID="1" Name="Full Cue" Artist="Artist A" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300">
          <POSITION_MARK Name="" Type="0" Start="20.0" Num="0"/>
          <POSITION_MARK Name="" Type="0" Start="250.0" Num="1"/>
        </TRACK>
        <TRACK TrackID="2" Name="No Cue" Artist="Artist B" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300"/>
        <TRACK TrackID="3" Name="Half Cue" Artist="Artist C" AverageBpm="124.00" Tonality="8A"
               Genre="House" TotalTime="300">
          <POSITION_MARK Name="" Type="0" Start="20.0" Num="0"/>
        </TRACK>
        <TRACK TrackID="4" Name="No Cue Too" Artist="Artist D" AverageBpm="125.00" Tonality="9A"
               Genre="House" TotalTime="300"/>
      </COLLECTION>
      <PLAYLISTS>
        <NODE Type="0" Name="ROOT" Count="1">
          <NODE Type="1" Name="DO NOT RECOMMEND" KeyType="0" Entries="1">
            <TRACK Key="2"/>
          </NODE>
        </NODE>
      </PLAYLISTS>
    </DJ_PLAYLISTS>
""")


def test_run_prep_denylisted_track_excluded_from_ranking_and_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--prep must not rank tracks in DO NOT RECOMMEND — cueing them is wasted time.

    Live finding: --prep reported 414 house tracks where the concept pipeline
    (which applies the denylist) sees 370.
    """
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PREP_DENYLIST_XML)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")

    exit_code = run_prep(None, 20)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Artist B — No Cue" not in out  # denylisted
    assert "Artist D — No Cue Too" in out  # still ranked
    assert "  house: 2 of 3 tracks lack cue data" in out  # denominator excludes the denylisted track


def test_run_prep_prints_table_footer_and_hint_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PREP_XML)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")

    exit_code = run_prep(None, 20)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Artist B — No Cue" in out
    assert "Artist C — Half Cue" in out
    assert "Artist D — No Cue Too" in out
    assert "Artist E — Techno No Cue" in out
    assert "Full Cue" not in out  # fully-cued track excluded from the table
    assert "  house: 3 of 4 tracks lack cue data" in out
    assert "  techno: 2 of 2 tracks lack cue data" in out
    assert out.index("house: 3 of 4") < out.index("techno: 2 of 2")
    assert "Cue up the top entries in Rekordbox, re-export, and booth sheets gain clock times." in out


def test_run_prep_genre_house_scopes_to_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PREP_XML)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")

    exit_code = run_prep("house", 20)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Artist B — No Cue" in out
    assert "Techno" not in out
    assert "  house: 3 of 4 tracks lack cue data" in out
    assert "techno:" not in out


def test_run_prep_genre_4x4_errors_with_exit_one(capsys: pytest.CaptureFixture[str]) -> None:
    """--prep only accepts standard GENRE_MAP labels — custom pools (170/140/4x4) are rejected."""
    exit_code = run_prep("4x4", 20)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "--prep accepts standard genre labels only" in err
    assert "4x4" in err


def _count_prep_rows(out: str) -> int:
    """Count printed table rows: lines whose rank column starts with a digit.

    Excludes the "Cue-Prep Assistant — top N target(s):" header line (starts with
    "Cue-Prep", not a digit) even though it also contains an em dash.
    """
    return sum(1 for line in out.splitlines() if line.strip()[:1].isdigit() and " — " in line)


def test_run_prep_top_1_limits_to_one_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PREP_XML)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")

    exit_code = run_prep(None, 1)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert _count_prep_rows(out) == 1


def test_run_prep_empty_gap_library_prints_congratulatory_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PREP_ALL_CUED_XML)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")

    exit_code = run_prep(None, 20)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "nothing to prep" in out.lower()
    assert "Cue-Prep Assistant" not in out


def test_main_prep_flag_runs_via_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--prep and --top wire through main()'s argv parsing to run_prep (#74)."""
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_text(_PREP_XML)
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.setattr("sys.argv", ["mixlab", "--prep", "--top", "1"])

    with patch("mixlab.__main__.load_dotenv"), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert _count_prep_rows(out) == 1


# ---------------------------------------------------------------------------
# _annotate_direction_types (#84)
# ---------------------------------------------------------------------------


def _concept(track_ids: list[str], **kwargs: object) -> MixConcept:
    return MixConcept(title="Test Concept", mood="tense", track_ids=track_ids, **kwargs)  # type: ignore[arg-type]


def _canvas(
    canvas_id: str,
    core_ids: list[str],
    concept: MixConcept,
    *,
    direction_type: str = "",
) -> MixCanvas:
    return MixCanvas(
        canvas_id=canvas_id,
        genre="Drum & Bass",
        bpm_range=(160.0, 180.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[]),
        contrast=ContrastAssets(
            vocal_moments=[],
            texture_changes=[],
            darker_turns=[],
            brighter_lifts=[],
            lower_pressure_resets=[],
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=concept,
        direction_type=direction_type,
    )


def test_annotate_direction_types_matching_direction_canvas_stamps_concept() -> None:
    concept = _concept(["1", "2", "3"])
    canvas = _canvas("c1", ["1", "2", "3"], concept, direction_type="genre_traverse")

    _annotate_direction_types([concept], [canvas])

    assert concept.direction_type == "genre_traverse"


def test_annotate_direction_types_matching_classic_canvas_stays_empty() -> None:
    concept = _concept(["1", "2", "3"])
    canvas = _canvas("c1", ["1", "2", "3"], concept, direction_type="")

    _annotate_direction_types([concept], [canvas])

    assert concept.direction_type == ""


def test_annotate_direction_types_no_matching_canvas_stays_empty() -> None:
    concept = _concept(["1", "2", "3"])
    other_concept = _concept(["9"])
    canvas = _canvas("c1", ["9"], other_concept, direction_type="mood_journey")

    _annotate_direction_types([concept], [canvas])

    assert concept.direction_type == ""


# ---------------------------------------------------------------------------
# main() — --direction-spec flag parsing
# ---------------------------------------------------------------------------


def test_main_direction_spec_forwarded_in_genre_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = '{"direction_type": "artist_thread", "title": "Artist thread: Dusky", "brief": "b", "track_ids": ["1"]}'
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house", "--direction-spec", spec])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["direction_spec"] == spec


def test_main_direction_spec_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--genre", "house"])
    run_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run", run_mock):
        main()
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["direction_spec"] is None


def test_main_direction_spec_ignored_in_playlist_mode_with_note(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["mixlab", "--playlist", "Monday", "--direction-spec", "{}"])
    playlist_mock = AsyncMock(return_value=None)
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.run_playlist_mode", playlist_mock):
        main()
    err = capsys.readouterr().err
    assert "--direction-spec ignored in playlist mode" in err
    assert playlist_mock.await_args is not None
