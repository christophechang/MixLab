from __future__ import annotations

from pathlib import Path

from mixlab.__main__ import _format_report_context


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
    assert (
        result
        == "Report context: Monday Night playlist (Electronica, All Tracks, stage 2: minimax, export enabled)"
    )
