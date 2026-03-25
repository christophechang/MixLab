from __future__ import annotations

from mixlab.clustering import (
    build_custom_genre_pool,
    count_outlier_genres,
    filter_by_bpm_range,
    group_by_genre,
    partition_outliers,
    sort_by_camelot,
)
from mixlab.config import CustomGenre
from mixlab.models import Track

_GENRE_MAP = {
    "drum_and_bass": ["Drum & Bass", "DnB"],
    "techno": ["Techno"],
}


def _track(
    *,
    track_id: str = "1",
    artist: str = "A",
    title: str = "T",
    bpm: float = 174.0,
    camelot_key: str = "8A",
    genre: str = "Drum & Bass",
) -> Track:
    return Track(track_id=track_id, artist=artist, title=title, bpm=bpm, camelot_key=camelot_key, genre=genre)


def test_group_by_genre_buckets_correctly() -> None:
    tracks = [
        _track(track_id="1", genre="Drum & Bass"),
        _track(track_id="2", genre="Drum & Bass"),
        _track(track_id="3", genre="Techno"),
    ]
    result = group_by_genre(tracks, _GENRE_MAP)
    assert len(result["Drum & Bass"]) == 2
    assert len(result["Techno"]) == 1


def test_sort_by_camelot_orders_compatible_keys() -> None:
    tracks = [
        _track(track_id="1", camelot_key="9A", bpm=174.0),
        _track(track_id="2", camelot_key="8A", bpm=170.0),  # compatible with 9A (adjacent)
        _track(track_id="3", camelot_key="8B", bpm=172.0),  # compatible with 8A (same number, opposite mode)
    ]
    result = sort_by_camelot(tracks)
    # Should form a chain: 9A → 8A → 8B (all compatible transitions)
    assert result[0].camelot_key == "9A"
    assert result[1].camelot_key == "8A"
    assert result[2].camelot_key == "8B"


def test_partition_outliers_separates_unmapped_genres() -> None:
    tracks = [
        _track(track_id="1", genre="Drum & Bass"),
        _track(track_id="2", genre="Ambient"),  # not in GENRE_MAP
        _track(track_id="3", genre="Techno"),
    ]
    clusters, outliers = partition_outliers(tracks, _GENRE_MAP)
    assert len(outliers) == 1
    assert outliers[0].genre == "Ambient"
    assert "Drum & Bass" in clusters
    assert "Techno" in clusters


def test_count_available_by_genre_returns_correct_counts() -> None:
    from mixlab.clustering import count_available_by_genre

    all_tracks = [
        _track(track_id="1", genre="Drum & Bass"),
        _track(track_id="2", genre="Drum & Bass"),
        _track(track_id="3", genre="Techno"),
    ]
    unplayed = [all_tracks[0], all_tracks[2]]  # track 2 (DnB) was played
    counts = count_available_by_genre(all_tracks, unplayed, _GENRE_MAP)
    assert counts["drum_and_bass"] == (2, 1)
    assert counts["techno"] == (1, 1)


def test_resolve_genre_clusters_matches_api_label() -> None:
    from mixlab.clustering import resolve_genre_clusters

    clusters = {
        "Drum & Bass": [_track(track_id="1", genre="Drum & Bass")],
        "Techno": [_track(track_id="2", genre="Techno")],
    }
    result = resolve_genre_clusters("drum_and_bass", clusters, _GENRE_MAP)
    assert "Drum & Bass" in result
    assert "Techno" not in result


def test_resolve_genre_clusters_matches_rekordbox_name() -> None:
    from mixlab.clustering import resolve_genre_clusters

    clusters = {
        "Drum & Bass": [_track(track_id="1", genre="Drum & Bass")],
        "Techno": [_track(track_id="2", genre="Techno")],
    }
    result = resolve_genre_clusters("Drum & Bass", clusters, _GENRE_MAP)
    assert "Drum & Bass" in result
    assert "Techno" not in result


def test_count_outlier_genres_returns_unmapped_tags() -> None:
    all_tracks = [
        _track(track_id="1", genre="Drum & Bass"),
        _track(track_id="2", genre="UK Bass Music"),  # not in GENRE_MAP
        _track(track_id="3", genre="UK Bass Music"),
        _track(track_id="4", genre="Ambient"),  # not in GENRE_MAP
    ]
    unplayed = [all_tracks[1], all_tracks[3]]  # one UK Bass Music unplayed, one Ambient unplayed
    result = count_outlier_genres(all_tracks, unplayed, _GENRE_MAP)
    assert "UK Bass Music" in result
    assert result["UK Bass Music"] == (2, 1)
    assert "Ambient" in result
    assert result["Ambient"] == (1, 1)
    assert "Drum & Bass" not in result


def test_count_outlier_genres_empty_when_all_mapped() -> None:
    all_tracks = [_track(track_id="1", genre="Drum & Bass"), _track(track_id="2", genre="Techno")]
    result = count_outlier_genres(all_tracks, all_tracks, _GENRE_MAP)
    assert result == {}


def test_count_outlier_genres_excludes_ignored() -> None:
    all_tracks = [
        _track(track_id="1", genre="Rock"),
        _track(track_id="2", genre="UK Bass Music"),
    ]
    result = count_outlier_genres(all_tracks, all_tracks, _GENRE_MAP, ignored=frozenset({"Rock"}))
    assert "Rock" not in result
    assert "UK Bass Music" in result


def test_count_outlier_genres_sorted_by_unplayed_desc() -> None:
    all_tracks = [
        _track(track_id="1", genre="Ambient"),
        _track(track_id="2", genre="UK Bass Music"),
        _track(track_id="3", genre="UK Bass Music"),
    ]
    unplayed = all_tracks  # all unplayed
    result = count_outlier_genres(all_tracks, unplayed, _GENRE_MAP)
    labels = list(result.keys())
    assert labels[0] == "UK Bass Music"  # 2 unplayed before 1


def test_outliers_below_threshold_skips_llm() -> None:
    # Outliers < 4 should not be processed by LLM — this test verifies the
    # partition count; LLM invocation is tested in test_llm.py.
    tracks = [_track(track_id=str(i), genre="Ambient") for i in range(3)]
    _, outliers = partition_outliers(tracks, _GENRE_MAP)
    assert len(outliers) == 3
    assert len(outliers) < 4


# ---------------------------------------------------------------------------
# filter_by_bpm_range
# ---------------------------------------------------------------------------


def test_filter_by_bpm_range_keeps_tracks_within_bounds() -> None:
    tracks = [
        _track(track_id="1", bpm=165.0),
        _track(track_id="2", bpm=170.0),
        _track(track_id="3", bpm=175.0),
        _track(track_id="4", bpm=164.9),
        _track(track_id="5", bpm=175.1),
    ]
    result = filter_by_bpm_range(tracks, 165.0, 175.0)
    ids = {t.track_id for t in result}
    assert ids == {"1", "2", "3"}


def test_filter_by_bpm_range_inclusive_boundaries() -> None:
    tracks = [
        _track(track_id="1", bpm=130.0),
        _track(track_id="2", bpm=140.0),
    ]
    result = filter_by_bpm_range(tracks, 130.0, 140.0)
    assert len(result) == 2


def test_filter_by_bpm_range_empty_input() -> None:
    assert filter_by_bpm_range([], 130.0, 140.0) == []


# ---------------------------------------------------------------------------
# build_custom_genre_pool
# ---------------------------------------------------------------------------

_CUSTOM_GENRE_MAP: dict[str, list[str]] = {
    "drum_and_bass": ["Drum & Bass", "DnB"],
    "jungle": ["Jungle", "Ragga Jungle"],
    "house": ["House", "Deep House"],
    "techno": ["Techno"],
}

_CUSTOM_GENRES: dict[str, CustomGenre] = {
    "170": CustomGenre(genres=["drum_and_bass", "jungle"], bpm_range=(165.0, 175.0)),
    "4x4": CustomGenre(genres=["house", "techno"], bpm_range=None),
}


def test_build_custom_genre_pool_merges_sub_genres() -> None:
    tracks = [
        _track(track_id="1", genre="Drum & Bass", bpm=170.0),
        _track(track_id="2", genre="Jungle", bpm=168.0),
        _track(track_id="3", genre="House", bpm=125.0),
        _track(track_id="4", genre="Techno", bpm=132.0),
    ]
    pool = build_custom_genre_pool("170", tracks, _CUSTOM_GENRES, _CUSTOM_GENRE_MAP)
    ids = {t.track_id for t in pool}
    assert ids == {"1", "2"}


def test_build_custom_genre_pool_applies_bpm_range_filter() -> None:
    tracks = [
        _track(track_id="1", genre="Drum & Bass", bpm=170.0),
        _track(track_id="2", genre="Drum & Bass", bpm=160.0),  # outside 165–175
        _track(track_id="3", genre="Jungle", bpm=176.0),  # outside 165–175
    ]
    pool = build_custom_genre_pool("170", tracks, _CUSTOM_GENRES, _CUSTOM_GENRE_MAP)
    ids = {t.track_id for t in pool}
    assert ids == {"1"}


def test_build_custom_genre_pool_no_bpm_filter_when_none() -> None:
    tracks = [
        _track(track_id="1", genre="House", bpm=120.0),
        _track(track_id="2", genre="Techno", bpm=138.0),
    ]
    pool = build_custom_genre_pool("4x4", tracks, _CUSTOM_GENRES, _CUSTOM_GENRE_MAP)
    assert len(pool) == 2


def test_build_custom_genre_pool_returns_empty_for_no_matches() -> None:
    tracks = [_track(track_id="1", genre="Ambient", bpm=90.0)]
    pool = build_custom_genre_pool("170", tracks, _CUSTOM_GENRES, _CUSTOM_GENRE_MAP)
    assert pool == []
