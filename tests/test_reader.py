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
) -> Track:
    return Track(track_id=track_id, artist=artist, title=title, bpm=bpm, camelot_key=camelot_key, genre=genre)


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
