from __future__ import annotations

import pytest

from mixlab.matcher import filter_unplayed, is_played, normalise
from mixlab.models import PlayedTrack, Track


def _track(artist: str, title: str) -> Track:
    return Track(track_id="1", artist=artist, title=title, bpm=174.0, camelot_key="8A", genre="Drum & Bass")


def _played(artist: str, title: str) -> PlayedTrack:
    return PlayedTrack(artist=artist, title=title)


def test_normalise_lowercases() -> None:
    assert normalise("Calibre") == "calibre"


def test_normalise_strips_punctuation() -> None:
    assert "," not in normalise("Hello, World!")
    assert "!" not in normalise("Hello, World!")


def test_normalise_collapses_unicode_dashes() -> None:
    result = normalise("A\u2013B")  # en-dash
    assert result == "a-b"


@pytest.mark.parametrize("feat_token", ["feat.", "ft.", "featuring"])
def test_normalise_removes_feat_variants(feat_token: str) -> None:
    result = normalise(f"Artist {feat_token} Guest")
    assert feat_token.rstrip(".") not in result
    assert "guest" not in result


def test_is_played_returns_true_for_matched_track() -> None:
    track = _track("Calibre", "All Good")
    played = [_played("Calibre", "All Good")]
    assert is_played(track, played) is True


def test_is_played_returns_false_for_unmatched_track() -> None:
    track = _track("Calibre", "All Good")
    played = [_played("Noisia", "Stigma")]
    assert is_played(track, played) is False


def test_filter_unplayed_excludes_played_tracks() -> None:
    tracks = [_track("Calibre", "All Good"), _track("Noisia", "Stigma")]
    played = [_played("Calibre", "All Good")]
    result = filter_unplayed(tracks, played)
    assert len(result) == 1
    assert result[0].artist == "Noisia"
