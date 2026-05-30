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


def test_normalise_feat_in_parens_preserves_remix() -> None:
    # Rekordbox stores "Sweet Time (feat. Izo FitzRoy) [Yuksek Remix]"
    # Catalog may store "Sweet Time (Yuksek Remix)" — both must normalise identically.
    xml_title = normalise("Kraak & Smaak Sweet Time (feat. Izo FitzRoy) [Yuksek Remix]")
    catalog_title = normalise("Kraak & Smaak Sweet Time (Yuksek Remix)")
    assert xml_title == catalog_title


def test_is_played_matches_feat_in_parens_with_different_bracket_styles() -> None:
    # Rekordbox: feat. in title parens; catalog: feat. stripped from title
    track = _track("Kraak & Smaak", "Sweet Time (feat. Izo FitzRoy) [Yuksek Remix]")
    played = [_played("Kraak & Smaak", "Sweet Time (Yuksek Remix)")]
    assert is_played(track, played) is True


def test_is_played_matches_feat_in_artist_field() -> None:
    # Real catalog format: artist="Kraak & Smaak feat. Izo FitzRoy", title="Sweet Time (Yuksek Remix)"
    # Rekordbox format:    artist="Kraak & Smaak",                   title="Sweet Time (feat. Izo FitzRoy) [Yuksek Remix]"
    track = _track("Kraak & Smaak", "Sweet Time (feat. Izo FitzRoy) [Yuksek Remix]")
    played = [_played("Kraak & Smaak feat. Izo FitzRoy", "Sweet Time (Yuksek Remix)")]
    assert is_played(track, played) is True


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


@pytest.mark.parametrize(
    "rb_title,catalog_title",
    [
        ("That's Right (Original Mix)", "That's Right"),
        ("Don't Let Me Go (Original Mix)", "Don't Let Me Go"),
        ("Cloud (Extended Mix)", "Cloud"),
        ("Smile (Original Mix)", "Smile"),
        ("Freaker Original Mix", "Freaker"),  # bare suffix, no brackets
        ("I want more (original_mix)", "I Want More"),  # underscore variant
    ],
)
def test_normalise_strips_version_suffixes(rb_title: str, catalog_title: str) -> None:
    assert normalise(rb_title) == normalise(catalog_title)


def test_is_played_matches_original_mix_suffix_difference() -> None:
    # Catalog stores "That's Right"; Rekordbox has "That's Right (Original Mix)"
    track = _track("Dam Swindle", "That's Right (Original Mix)")
    played = [_played("Dam Swindle", "That's Right")]
    assert is_played(track, played) is True


def test_normalise_removes_feat_without_period_in_parens() -> None:
    # Rekordbox: "(feat. Slay)"; catalog: "(feat Slay)" — both should strip
    assert normalise("Let It Slide (feat. Slay)") == normalise("Let It Slide (feat Slay)")
