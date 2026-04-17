from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mixlab.__main__ import (
    _apply_do_not_recommend_filter,
    _format_pipeline_counts,
    _format_report_context,
    _print_availability,
    _print_pipeline_summary,
)
from mixlab.playlist_exporter import build_merged_xml, parse_raw_tracks
from mixlab.reader import parse_collection


def test_format_report_context_standard_genre_unplayed() -> None:
    result = _format_report_context(
        genre="electronica",
        playlist_name=None,
        all_tracks=False,
        stage2_provider=None,
        export_dir=None,
    )
    assert result == "Report context: Electronica (unplayed tracks)"


def test_format_report_context_custom_genre_all_tracks() -> None:
    result = _format_report_context(
        genre="140",
        playlist_name=None,
        all_tracks=True,
        stage2_provider=None,
        export_dir=None,
    )
    assert result == "Report context: 140 (custom genre, All Tracks)"


def test_format_report_context_playlist_includes_active_options() -> None:
    result = _format_report_context(
        genre="electronica",
        playlist_name="Monday Night",
        all_tracks=True,
        stage2_provider="minimax",
        export_dir=Path("output/playlists"),
    )
    assert result == "Report context: Monday Night playlist (Electronica, All Tracks, stage 2: minimax, export enabled)"


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
