from __future__ import annotations

from mixlab.discord_client import format_report
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
