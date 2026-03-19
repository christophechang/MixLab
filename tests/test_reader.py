from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mixlab.models import Track
from mixlab.reader import apply_bpm_corrections, parse_collection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="3">
        <TRACK TrackID="1" Name="Alpha" Artist="Artist A" AverageBpm="174.00" Tonality="8A" Genre="Drum &amp; Bass"/>
        <TRACK TrackID="2" Name="Beta"  Artist="Artist B" AverageBpm="132.00" Tonality="9B" Genre="UK Garage"/>
        <TRACK TrackID="3" Name="Gamma" Artist="Artist C" AverageBpm="140.00" Tonality="10A" Genre="Techno"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")

_MISSING_BPM_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="2">
        <TRACK TrackID="1" Name="Good"    Artist="Artist A" AverageBpm="174.00" Tonality="8A" Genre="Drum &amp; Bass"/>
        <TRACK TrackID="2" Name="NoBpm"   Artist="Artist B" AverageBpm=""        Tonality="9B" Genre="Techno"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")

_MISSING_KEY_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="2">
        <TRACK TrackID="1" Name="Good"   Artist="Artist A" AverageBpm="174.00" Tonality="8A" Genre="Drum &amp; Bass"/>
        <TRACK TrackID="2" Name="NoKey"  Artist="Artist B" AverageBpm="140.00" Tonality=""   Genre="Techno"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")

_SOUNDCLOUD_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="3">
        <TRACK TrackID="1" Name="Local"  Artist="Artist A" AverageBpm="174.00" Tonality="8A" Genre="House"
               Location="file://localhost/Users/dj/Music/track.mp3"/>
        <TRACK TrackID="2" Name="Cloud"  Artist="Artist B" AverageBpm="126.00" Tonality="9A" Genre="House"
               Location="file://localhostsoundcloud:tracks:12345"/>
        <TRACK TrackID="3" Name="Local2" Artist="Artist C" AverageBpm="125.00" Tonality="7A" Genre="Techno"
               Location="file://localhost/Users/dj/Music/track2.mp3"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")


def _write_xml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "rekordbox.xml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Phase 1 — parse_collection
# ---------------------------------------------------------------------------


def test_parse_collection_returns_valid_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _VALID_XML)
    tracks = parse_collection(xml_path)
    assert len(tracks) == 3
    assert tracks[0].title == "Alpha"
    assert tracks[1].title == "Beta"
    assert tracks[2].title == "Gamma"


def test_parse_collection_excludes_tracks_missing_bpm(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _MISSING_BPM_XML)
    tracks = parse_collection(xml_path)
    assert len(tracks) == 1
    assert tracks[0].title == "Good"


def test_parse_collection_excludes_tracks_missing_key(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _MISSING_KEY_XML)
    tracks = parse_collection(xml_path)
    assert len(tracks) == 1
    assert tracks[0].title == "Good"


def test_parse_collection_excludes_soundcloud_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SOUNDCLOUD_XML)
    tracks = parse_collection(xml_path)
    assert len(tracks) == 2
    assert all(t.title != "Cloud" for t in tracks)


def test_parse_collection_raises_if_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_collection(tmp_path / "nonexistent.xml")


# ---------------------------------------------------------------------------
# Phase 2 — apply_bpm_corrections
# ---------------------------------------------------------------------------


def _make_track(
    *,
    track_id: str = "1",
    artist: str = "Artist",
    title: str = "Title",
    bpm: float = 87.0,
    camelot_key: str = "8A",
    genre: str = "Drum & Bass",
    energy: int | None = None,
    colour: str | None = None,
    label: str = "",
    play_count: int = 0,
    tags: list[str] | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=camelot_key,
        genre=genre,
        energy=energy,
        colour=colour,
        label=label,
        play_count=play_count,
        tags=tags or [],
    )


def test_bpm_correction_doubles_dnb_below_100() -> None:
    track = _make_track(bpm=87.0, genre="Drum & Bass")
    result = apply_bpm_corrections([track])
    assert result[0].bpm == 174.0


def test_bpm_correction_does_not_modify_dnb_above_100() -> None:
    track = _make_track(bpm=174.0, genre="Drum & Bass")
    result = apply_bpm_corrections([track])
    assert result[0].bpm == 174.0


def test_bpm_correction_does_not_modify_slow_liquid_dnb() -> None:
    # 130 BPM is valid autonomic/halfstep DnB — should not be doubled.
    track = _make_track(bpm=130.0, genre="Drum & Bass")
    result = apply_bpm_corrections([track])
    assert result[0].bpm == 130.0


def test_bpm_correction_ignores_non_dnb_genres() -> None:
    track = _make_track(bpm=70.0, genre="House")
    result = apply_bpm_corrections([track])
    assert result[0].bpm == 70.0


def test_bpm_correction_logs_correction(capsys: pytest.CaptureFixture[str]) -> None:
    track = _make_track(artist="Calibre", title="All Good", bpm=87.0, genre="DnB")
    apply_bpm_corrections([track])
    captured = capsys.readouterr()
    assert "BPM corrected" in captured.err
    assert "Calibre" in captured.err
    assert "All Good" in captured.err


# ---------------------------------------------------------------------------
# Phase 3 — enriched metadata extraction
# ---------------------------------------------------------------------------

_ENRICHED_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="4">
        <TRACK TrackID="10" Name="Rated High" Artist="Artist A" AverageBpm="136.00" Tonality="8A"
               Genre="Breakbeat" Rating="255" Colour="0xFF0000" Label="Running Back" PlayCount="0"
               Comments="8A - Energy 7 /* Breakbeat / Dark / Aggressive */"
               Location="file://localhost/Users/dj/Music/a.mp3"/>
        <TRACK TrackID="11" Name="Rated Mid" Artist="Artist B" AverageBpm="125.00" Tonality="4A"
               Genre="House" Rating="153" Colour="0xFFA500" Label="" PlayCount="3"
               Comments="4A - Energy 5 /* House / In the Groove */"
               Location="file://localhost/Users/dj/Music/b.mp3"/>
        <TRACK TrackID="12" Name="No Rating" Artist="Artist C" AverageBpm="125.00" Tonality="5A"
               Genre="House" Rating="0" PlayCount="0"
               Comments=""
               Location="file://localhost/Users/dj/Music/c.mp3"/>
        <TRACK TrackID="13" Name="Unknown Colour" Artist="Artist D" AverageBpm="130.00" Tonality="9A"
               Genre="Techno" Colour="0x123456" PlayCount="1"
               Comments="9A - Energy 6 /* Techno / Driving */"
               Location="file://localhost/Users/dj/Music/d.mp3"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")


def test_parse_collection_extracts_mik_energy(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["10"].energy == 7
    assert by_id["11"].energy == 5
    assert by_id["13"].energy == 6


def test_parse_collection_energy_is_none_when_comments_empty(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["12"].energy is None


def test_parse_collection_extracts_colour(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["10"].colour == "Red"
    assert by_id["11"].colour == "Orange"


def test_parse_collection_colour_is_none_for_unknown_hex(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["13"].colour is None


def test_parse_collection_extracts_label_and_playcount(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["10"].label == "Running Back"
    assert by_id["10"].play_count == 0
    assert by_id["11"].play_count == 3


def test_parse_collection_extracts_tags(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["10"].tags == ["Breakbeat", "Dark", "Aggressive"]
    assert by_id["11"].tags == ["House", "In the Groove"]


def test_parse_collection_tags_empty_when_no_block(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _ENRICHED_XML)
    tracks = parse_collection(xml_path)
    by_id = {t.track_id: t for t in tracks}
    assert by_id["12"].tags == []


# ---------------------------------------------------------------------------
# _parse_tags / _parse_mik_energy unit tests
# ---------------------------------------------------------------------------


def test_parse_tags_extracts_flat_list() -> None:
    from mixlab.reader import _parse_tags

    result = _parse_tags("8A - Energy 7 /* Breakbeat / Dark / Breaks & UK Bass */")
    assert result == ["Breakbeat", "Dark", "Breaks & UK Bass"]


def test_parse_tags_returns_empty_for_no_block() -> None:
    from mixlab.reader import _parse_tags

    assert _parse_tags("4A - Energy 6") == []
    assert _parse_tags("") == []
    assert _parse_tags("For Promotional Use Only!") == []


def test_parse_tags_single_tag() -> None:
    from mixlab.reader import _parse_tags

    assert _parse_tags("/* Breakbeat */") == ["Breakbeat"]


def test_parse_mik_energy_extracts_value() -> None:
    from mixlab.reader import _parse_mik_energy

    assert _parse_mik_energy("9A - Energy 6 /* Breakbeat / Dark */") == 6
    assert _parse_mik_energy("4A - Energy 7") == 7


def test_parse_mik_energy_returns_none_when_absent() -> None:
    from mixlab.reader import _parse_mik_energy

    assert _parse_mik_energy("") is None
    assert _parse_mik_energy("For Promotional Use Only!") is None
    assert _parse_mik_energy("/* Breakbeat / Dark */") is None
