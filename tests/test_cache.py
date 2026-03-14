from __future__ import annotations

import json
from pathlib import Path

import pytest

from mixlab.cache import load_genre_cache, save_genre_cache


def test_save_genre_cache_writes_valid_json(tmp_path: Path) -> None:
    cache_file = tmp_path / "genres.json"
    counts = {"drum_and_bass": (120, 45), "house": (80, 30)}
    outlier_genres = {"UK Bass Music": (12, 8)}
    save_genre_cache(counts, 8, outlier_genres, path=cache_file)
    data = json.loads(cache_file.read_text())
    assert data["counts"]["drum_and_bass"] == {"total": 120, "unplayed": 45}
    assert data["counts"]["house"] == {"total": 80, "unplayed": 30}
    assert data["outliers"] == 8
    assert data["outlier_genres"]["UK Bass Music"] == {"total": 12, "unplayed": 8}
    assert "saved_at" in data


def test_save_genre_cache_empty_counts(tmp_path: Path) -> None:
    cache_file = tmp_path / "genres.json"
    save_genre_cache({}, 0, {}, path=cache_file)
    data = json.loads(cache_file.read_text())
    assert data["counts"] == {}
    assert data["outliers"] == 0
    assert data["outlier_genres"] == {}


def test_load_genre_cache_returns_saved_data(tmp_path: Path) -> None:
    cache_file = tmp_path / "genres.json"
    counts = {"techno": (60, 20)}
    save_genre_cache(counts, 5, {"Ambient": (5, 5)}, path=cache_file)
    result = load_genre_cache(path=cache_file)
    assert result is not None
    assert result["counts"]["techno"] == {"total": 60, "unplayed": 20}
    assert result["outliers"] == 5
    assert result["outlier_genres"]["Ambient"] == {"total": 5, "unplayed": 5}


def test_load_genre_cache_missing_file_returns_none(tmp_path: Path) -> None:
    result = load_genre_cache(path=tmp_path / "nonexistent.json")
    assert result is None


def test_load_genre_cache_corrupt_json_returns_none(tmp_path: Path) -> None:
    cache_file = tmp_path / "genres.json"
    cache_file.write_text("{ not valid json }")
    result = load_genre_cache(path=cache_file)
    assert result is None


@pytest.mark.parametrize(
    ("counts", "outliers", "outlier_genres"),
    [
        ({"drum_and_bass": (100, 50)}, 0, {}),
        ({"house": (10, 5), "techno": (20, 15)}, 3, {"Ambient": (3, 3)}),
        ({}, 99, {"Unknown": (99, 99)}),
    ],
)
def test_save_load_roundtrip(
    tmp_path: Path, counts: dict[str, tuple[int, int]], outliers: int, outlier_genres: dict[str, tuple[int, int]]
) -> None:
    cache_file = tmp_path / "genres.json"
    save_genre_cache(counts, outliers, outlier_genres, path=cache_file)
    result = load_genre_cache(path=cache_file)
    assert result is not None
    assert result["outliers"] == outliers
    for label, (total, unplayed) in counts.items():
        assert result["counts"][label] == {"total": total, "unplayed": unplayed}
    for tag, (total, unplayed) in outlier_genres.items():
        assert result["outlier_genres"][tag] == {"total": total, "unplayed": unplayed}
