from __future__ import annotations

from typing import Literal

import pytest

from mixlab.clustering import (
    build_custom_genre_pool,
    build_mix_canvas,
    camelot_distance,
    count_outlier_genres,
    filter_by_bpm_range,
    group_by_genre,
    partition_bpm_pools,
    partition_outliers,
    score_anchors,
    score_canvas,
    select_canvases,
    sort_by_camelot,
)
from mixlab.config import CustomGenre
from mixlab.history import ConceptHistory
from mixlab.models import MixConcept, Track

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
    energy: int | None = None,
    label: str = "",
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=camelot_key,
        genre=genre,
        energy=energy,
        label=label,
    )


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


# ---------------------------------------------------------------------------
# camelot_distance
# ---------------------------------------------------------------------------


def test_camelot_distance_identical_keys_returns_zero() -> None:
    assert camelot_distance("8A", "8A") == 0


def test_camelot_distance_adjacent_same_ring_returns_one() -> None:
    assert camelot_distance("8A", "9A") == 1


def test_camelot_distance_adjacent_wraps_twelve_to_one() -> None:
    assert camelot_distance("12A", "1A") == 1


def test_camelot_distance_same_number_opposite_ring_returns_one() -> None:
    assert camelot_distance("8A", "8B") == 1


def test_camelot_distance_cross_ring_two_steps_apart() -> None:
    # 8A to 9B: ring_dist=1, cross-ring adds 1 → 2
    assert camelot_distance("8A", "9B") == 2


def test_camelot_distance_unparseable_returns_999() -> None:
    assert camelot_distance("X", "8A") == 999
    assert camelot_distance("8A", "") == 999


def test_camelot_distance_large_gap_same_ring() -> None:
    # 1A to 7A: min(6, 12-6) = 6
    assert camelot_distance("1A", "7A") == 6


def test_camelot_distance_is_symmetric() -> None:
    assert camelot_distance("3B", "9A") == camelot_distance("9A", "3B")


# ---------------------------------------------------------------------------
# partition_bpm_pools
# ---------------------------------------------------------------------------


def test_partition_bpm_pools_splits_correctly() -> None:
    # Median = 172; core ±6 = [166, 178]; bridge 6–12 = (178, 184]; wildcard >12
    tracks = [
        _track(track_id="core1", bpm=172.0),  # delta=0 → core
        _track(track_id="core2", bpm=178.0),  # delta=6 → core (boundary)
        _track(track_id="core3", bpm=166.0),  # delta=6 → core (boundary)
        _track(track_id="bridge1", bpm=179.0),  # delta=7 → bridge
        _track(track_id="bridge2", bpm=161.0),  # delta=11 → bridge
        _track(track_id="wild1", bpm=185.0),  # delta=13 → wildcard
        _track(track_id="wild2", bpm=158.0),  # delta=14 → wildcard
    ]
    pools = partition_bpm_pools(tracks)
    core_ids = {t.track_id for t in pools.core}
    bridge_ids = {t.track_id for t in pools.bridge}
    wild_ids = {t.track_id for t in pools.wildcard}

    assert core_ids == {"core1", "core2", "core3"}
    assert bridge_ids == {"bridge1", "bridge2"}
    assert wild_ids == {"wild1", "wild2"}


def test_partition_bpm_pools_empty_input() -> None:
    pools = partition_bpm_pools([])
    assert pools.core == []
    assert pools.bridge == []
    assert pools.wildcard == []


def test_partition_bpm_pools_all_core_when_tight_cluster() -> None:
    tracks = [_track(track_id=str(i), bpm=170.0 + i) for i in range(5)]  # 170–174, median=172
    pools = partition_bpm_pools(tracks)
    assert len(pools.core) == 5
    assert pools.bridge == []
    assert pools.wildcard == []


def test_partition_bpm_pools_preserves_all_tracks() -> None:
    tracks = [_track(track_id=str(i), bpm=float(100 + i * 10)) for i in range(10)]
    pools = partition_bpm_pools(tracks)
    total = len(pools.core) + len(pools.bridge) + len(pools.wildcard)
    assert total == len(tracks)


# ---------------------------------------------------------------------------
# build_mix_canvas
# ---------------------------------------------------------------------------


def _concept(track_ids: list[str]) -> MixConcept:
    return MixConcept(title="Test", mood="dark", track_ids=track_ids)


def _tracks_by_id(*tracks: Track) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


def test_build_mix_canvas_role_inference_with_energy() -> None:
    high_e = _track(track_id="peak", bpm=172.0, energy=7)
    low_e = _track(track_id="opener", bpm=172.0, energy=2)
    mid_e = _track(track_id="groove", bpm=172.0, energy=4)
    tracks = [high_e, low_e, mid_e]
    concept = _concept([t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*tracks))
    assert "peak" in canvas.roles.peak
    assert "opener" in canvas.roles.opener
    assert "groove" in canvas.roles.groove_locker


def test_build_mix_canvas_role_inference_no_energy() -> None:
    # No energy field; BPM proxy: bridge pool tracks become opener candidates.
    # Use 5 core tracks to anchor median=172, then one clear bridge track.
    core_tracks = [_track(track_id=f"c{i}", bpm=172.0) for i in range(5)]
    bridge = _track(track_id="br", bpm=181.0)  # delta=9 from median=172 → bridge
    all_tracks = [*core_tracks, bridge]
    concept = _concept([t.track_id for t in all_tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*all_tracks))
    assert "br" in canvas.bridge_track_ids
    assert "br" in canvas.roles.opener


def test_build_mix_canvas_contrast_assets_vocal() -> None:
    vocal_t = _track(track_id="voc", artist="Artist feat. Singer")
    plain_t = _track(track_id="plain", artist="Artist")
    concept = _concept(["voc", "plain"])
    canvas = build_mix_canvas(concept, _tracks_by_id(vocal_t, plain_t))
    assert "voc" in canvas.contrast.vocal_moments
    assert "plain" not in canvas.contrast.vocal_moments


def test_build_mix_canvas_risk_notes_weak_closer() -> None:
    # All high energy → no closer candidates → risk note
    tracks = [_track(track_id=str(i), bpm=172.0, energy=7) for i in range(5)]
    concept = _concept([t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*tracks))
    assert "weak closer pool" in canvas.risk_notes


def test_build_mix_canvas_risk_notes_over_repeated_artist() -> None:
    tracks = [_track(track_id=str(i), bpm=172.0, artist="Same Artist") for i in range(3)]
    concept = _concept([t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*tracks))
    assert "over-repeated artist" in canvas.risk_notes


def test_build_mix_canvas_pools_correctly_split() -> None:
    # 5 core tracks anchor median=172; bridge delta=8; wildcard delta=19
    core_tracks = [_track(track_id=f"core{i}", bpm=172.0) for i in range(5)]
    bridge_t = _track(track_id="bridge", bpm=180.0)  # delta=8 from 172 → bridge
    wild_t = _track(track_id="wild", bpm=191.0)  # delta=19 from 172 → wildcard
    all_tracks = [*core_tracks, bridge_t, wild_t]
    concept = _concept([t.track_id for t in all_tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*all_tracks))
    assert all(f"core{i}" in canvas.core_track_ids for i in range(5))
    assert "bridge" in canvas.bridge_track_ids
    assert "wild" in canvas.wildcard_track_ids


def test_build_mix_canvas_empty_concept() -> None:
    concept = _concept([])
    canvas = build_mix_canvas(concept, {})
    assert canvas.core_track_ids == []
    assert canvas.bridge_track_ids == []
    assert canvas.wildcard_track_ids == []


# ---------------------------------------------------------------------------
# score_canvas / select_canvases
# ---------------------------------------------------------------------------


def _rich_canvas(canvas_id: str, core_ids: list[str]) -> object:
    from mixlab.models import CanvasRoleCandidates, CanvasScore, ContrastAssets, MixCanvas

    n = len(core_ids)
    half = n // 2
    roles = CanvasRoleCandidates(
        opener=core_ids[:1],
        groove_locker=core_ids[:2],
        builder=core_ids[1:3],
        pivot=core_ids[half : half + 1],
        peak=core_ids[-2:],
        closer=core_ids[-1:],
    )
    contrast = ContrastAssets(
        vocal_moments=core_ids[:1],
        texture_changes=core_ids[1:2],
        darker_turns=core_ids[2:3],
        brighter_lifts=core_ids[3:4] if n > 3 else [],
        lower_pressure_resets=[],
    )
    return MixCanvas(
        canvas_id=canvas_id,
        genre="Drum & Bass",
        bpm_range=(168.0, 176.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=roles,
        contrast=contrast,
        risk_notes=[],
        score=CanvasScore(),
        source_concept=MixConcept(title="T", mood="dark", track_ids=core_ids),
    )


def test_score_canvas_weights_sum_to_one() -> None:
    from mixlab.models import MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    # Weighted sum minus weakness_penalty, multiplied by floor_multiplier. Weights total 1.0.
    weighted = (
        s.technical_viability * 0.10
        + s.role_coverage * 0.25
        + s.anchor_strength * 0.15
        + s.contrast_potential * 0.15
        + s.distinctiveness * 0.15
        + s.era_coherence * 0.05
        + s.label_coherence * 0.05
        + s.novelty * 0.10
    )
    expected = max(0.0, (weighted - s.weakness_penalty) * s.floor_multiplier)
    assert s.overall == pytest.approx(expected, abs=1e-9)


def test_score_canvas_full_core_pool_maxes_technical_viability() -> None:
    from mixlab.models import MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    # Logarithmic curve saturates at 15 core tracks; 20 still hits the cap of 1.0.
    assert s.technical_viability == pytest.approx(1.0)


def test_score_canvas_technical_viability_saturates_at_15_tracks() -> None:
    """Logarithmic curve: 15 tracks → 1.0, more tracks does not exceed 1.0."""
    from mixlab.models import MixCanvas

    canvas_15 = _rich_canvas("c15", [f"T{i:03d}" for i in range(15)])
    canvas_30 = _rich_canvas("c30", [f"T{i:03d}" for i in range(30)])
    assert isinstance(canvas_15, MixCanvas)
    assert isinstance(canvas_30, MixCanvas)
    s15 = score_canvas(canvas_15, ConceptHistory(), frozenset())
    s30 = score_canvas(canvas_30, ConceptHistory(), frozenset())
    assert s15.technical_viability == pytest.approx(1.0)
    assert s30.technical_viability == pytest.approx(1.0)


def test_score_canvas_technical_viability_grows_logarithmically_below_15() -> None:
    """Diminishing returns: 8-track canvas scores between 0.7 and 0.9, not the linear 0.4."""
    from mixlab.models import MixCanvas

    canvas = _rich_canvas("c8", [f"T{i:03d}" for i in range(8)])
    assert isinstance(canvas, MixCanvas)
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    # log(9)/log(16) ≈ 0.792
    assert 0.7 < s.technical_viability < 0.9


def test_score_canvas_weak_pool_floor_multiplier_halves_overall() -> None:
    """Canvases with fewer than 8 core tracks have their overall halved."""
    from mixlab.models import MixCanvas

    canvas_7 = _rich_canvas("c7", [f"T{i:03d}" for i in range(7)])
    canvas_8 = _rich_canvas("c8", [f"T{i:03d}" for i in range(8)])
    assert isinstance(canvas_7, MixCanvas)
    assert isinstance(canvas_8, MixCanvas)
    s7 = score_canvas(canvas_7, ConceptHistory(), frozenset())
    s8 = score_canvas(canvas_8, ConceptHistory(), frozenset())
    assert s7.floor_multiplier == 0.5
    assert s8.floor_multiplier == 1.0


def test_score_canvas_anchor_strength_presence_based() -> None:
    """Anchor strength = 0.5 per role (opener + closer), regardless of pool size."""
    from mixlab.models import CanvasRoleCandidates, MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    # Replace roles to test presence variants.
    canvas.roles = CanvasRoleCandidates(
        opener=["T000"], groove_locker=[], builder=[], pivot=[], peak=[], closer=["T019"]
    )
    s_both = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s_both.anchor_strength == pytest.approx(1.0)

    canvas.roles = CanvasRoleCandidates(opener=["T000"], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])
    s_opener_only = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s_opener_only.anchor_strength == pytest.approx(0.5)

    canvas.roles = CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])
    s_neither = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s_neither.anchor_strength == pytest.approx(0.0)


def test_score_canvas_anchor_strength_does_not_reward_volume() -> None:
    """One opener candidate scores the same on anchor_strength as ten opener candidates."""
    from mixlab.models import CanvasRoleCandidates, MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    canvas.roles = CanvasRoleCandidates(
        opener=["T000"], groove_locker=[], builder=[], pivot=[], peak=[], closer=["T019"]
    )
    s_one_each = score_canvas(canvas, ConceptHistory(), frozenset())
    canvas.roles = CanvasRoleCandidates(
        opener=[f"T{i:03d}" for i in range(10)],
        groove_locker=[],
        builder=[],
        pivot=[],
        peak=[],
        closer=[f"T{i:03d}" for i in range(10, 20)],
    )
    s_many_each = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s_one_each.anchor_strength == s_many_each.anchor_strength


def test_score_canvas_weakness_penalty_subtracts_per_risk_note() -> None:
    """Each risk note shaves 0.04 off the weighted sum, up to a cap of 0.20."""
    from mixlab.models import MixCanvas

    base = _rich_canvas("clean", [f"T{i:03d}" for i in range(20)])
    flagged = _rich_canvas("flagged", [f"T{i:03d}" for i in range(20)])
    assert isinstance(base, MixCanvas)
    assert isinstance(flagged, MixCanvas)
    flagged.risk_notes = ["weak opener pool", "excessive BPM spread", "over-repeated artist"]
    s_clean = score_canvas(base, ConceptHistory(), frozenset())
    s_flagged = score_canvas(flagged, ConceptHistory(), frozenset())
    assert s_flagged.weakness_penalty == pytest.approx(0.12, abs=1e-9)  # 3 * 0.04
    assert s_flagged.overall < s_clean.overall


def test_score_canvas_weakness_penalty_caps_at_0_20() -> None:
    """Six or more risk notes do not exceed the 0.20 weakness penalty cap."""
    from mixlab.models import MixCanvas

    canvas = _rich_canvas("noisy", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    canvas.risk_notes = [f"note {i}" for i in range(10)]
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.weakness_penalty == pytest.approx(0.20, abs=1e-9)


def test_score_canvas_small_distinctive_canvas_beats_large_generic() -> None:
    """A 15-track canvas with all roles filled and no overlap beats a 30-track canvas with role gaps and overlap."""
    from mixlab.models import CanvasRoleCandidates, ContrastAssets, MixCanvas

    # Canvas A: 15 distinctive tracks, all roles filled, contrast assets present
    ids_a = [f"A{i:03d}" for i in range(15)]
    canvas_a = _rich_canvas("A", ids_a)

    # Canvas B: 30 generic tracks but only opener role, no contrast, overlaps with picked set
    ids_b = [f"B{i:03d}" for i in range(30)]
    canvas_b = _rich_canvas("B", ids_b)
    assert isinstance(canvas_a, MixCanvas)
    assert isinstance(canvas_b, MixCanvas)
    canvas_b.roles = CanvasRoleCandidates(opener=["B000"], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])
    canvas_b.contrast = ContrastAssets(
        vocal_moments=[], texture_changes=[], darker_turns=[], brighter_lifts=[], lower_pressure_resets=[]
    )

    # Already picked some tracks from canvas_b — penalises its distinctiveness.
    picked = frozenset(ids_b[:15])
    s_a = score_canvas(canvas_a, ConceptHistory(), picked)
    s_b = score_canvas(canvas_b, ConceptHistory(), picked)
    assert s_a.overall > s_b.overall


def test_score_canvas_overall_never_negative() -> None:
    """Even an extremely weak canvas (small + many risk notes) stays at or above zero."""
    from mixlab.models import CanvasRoleCandidates, ContrastAssets, MixCanvas

    canvas = _rich_canvas("bad", ["T1", "T2"])  # below floor
    assert isinstance(canvas, MixCanvas)
    canvas.risk_notes = [f"note {i}" for i in range(20)]
    canvas.roles = CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])
    canvas.contrast = ContrastAssets(
        vocal_moments=[], texture_changes=[], darker_turns=[], brighter_lifts=[], lower_pressure_resets=[]
    )
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.overall >= 0.0


def test_select_canvases_prefers_higher_score() -> None:
    from mixlab.models import MixCanvas

    good = _rich_canvas("good", [f"T{i:03d}" for i in range(20)])
    poor = _rich_canvas("poor", ["X001"])
    assert isinstance(good, MixCanvas)
    assert isinstance(poor, MixCanvas)
    selected = select_canvases([poor, good], ConceptHistory(), n=1)
    assert selected[0].canvas_id == "good"


def test_select_canvases_overlap_penalty() -> None:
    from mixlab.models import MixCanvas

    ids_a = [f"T{i:03d}" for i in range(20)]
    ids_b = ids_a[:]  # 100% overlap with a
    ids_c = [f"X{i:03d}" for i in range(20)]  # no overlap

    canvas_a = _rich_canvas("A", ids_a)
    canvas_b = _rich_canvas("B", ids_b)
    canvas_c = _rich_canvas("C", ids_c)
    assert isinstance(canvas_a, MixCanvas)
    assert isinstance(canvas_b, MixCanvas)
    assert isinstance(canvas_c, MixCanvas)
    selected = select_canvases([canvas_a, canvas_b, canvas_c], ConceptHistory(), n=2)
    selected_ids = {c.canvas_id for c in selected}
    # C should beat B because C has no overlap with A
    assert "A" in selected_ids
    assert "C" in selected_ids


def test_select_canvases_novelty_penalty() -> None:
    from mixlab.history import HistoryEntry
    from mixlab.models import MixCanvas

    core_ids = [f"T{i:03d}" for i in range(20)]
    canvas = _rich_canvas("c1", core_ids)
    assert isinstance(canvas, MixCanvas)

    entry = HistoryEntry(
        run_id="r1",
        created_at="2026-01-01T00:00:00+00:00",
        mode="standard",
        genre="drum_and_bass",
        selected_canvas_ids=["c1"],
        dominant_bpm_clusters=[172.0],
        dominant_camelot_keys=["4A"],
        core_track_ids=core_ids,  # identical
        anchor_track_ids=[core_ids[0]],
        opener_candidates=[core_ids[0]],
        closer_candidates=[core_ids[-1]],
        concept_title="Old",
        concept_track_ids=core_ids,
        energy_path="single_arc",
        mood="dark",
        rating=None,
    )
    history = ConceptHistory(runs=[entry])
    s = score_canvas(canvas, history, frozenset())
    assert s.novelty < 1.0  # penalised by history


def test_select_canvases_returns_all_when_fewer_than_n() -> None:
    from mixlab.models import MixCanvas

    canvases = [_rich_canvas(f"c{i}", [f"T{i}{j:02d}" for j in range(5)]) for i in range(3)]
    assert all(isinstance(c, MixCanvas) for c in canvases)
    selected = select_canvases(canvases, ConceptHistory(), n=6)  # type: ignore[arg-type]
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# debug output
# ---------------------------------------------------------------------------


def test_score_canvas_debug_emits_score_fields_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.models import MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    score_canvas(canvas, ConceptHistory(), frozenset(), debug=True)
    captured = capsys.readouterr()
    assert "technical_viability" in captured.err
    assert "role_coverage" in captured.err
    assert "anchor_strength" in captured.err
    assert "novelty" in captured.err
    assert "overall" in captured.err
    assert "overlap_penalty" in captured.err
    assert "novelty_penalty" in captured.err
    assert "risk_notes" in captured.err
    assert "weakness_penalty" in captured.err
    assert "floor_multiplier" in captured.err


def test_score_canvas_no_debug_no_stderr_output(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.models import MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    score_canvas(canvas, ConceptHistory(), frozenset())
    captured = capsys.readouterr()
    assert captured.err == ""


def test_select_canvases_debug_emits_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.models import MixCanvas

    canvases = [_rich_canvas(f"c{i}", [f"T{i}{j:02d}" for j in range(10)]) for i in range(3)]
    assert all(isinstance(c, MixCanvas) for c in canvases)
    select_canvases(canvases, ConceptHistory(), n=2, debug=True)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "DEBUG" in captured.err
    assert "select_canvases" in captured.err
    assert "pick #" in captured.err


def test_select_canvases_debug_does_not_change_selection() -> None:
    from mixlab.models import MixCanvas

    canvases_a = [_rich_canvas(f"c{i}", [f"T{i}{j:02d}" for j in range(10)]) for i in range(4)]
    canvases_b = [_rich_canvas(f"c{i}", [f"T{i}{j:02d}" for j in range(10)]) for i in range(4)]
    assert all(isinstance(c, MixCanvas) for c in canvases_a + canvases_b)
    selected_no_debug = [c.canvas_id for c in select_canvases(canvases_a, ConceptHistory(), n=3)]  # type: ignore[arg-type]
    selected_debug = [c.canvas_id for c in select_canvases(canvases_b, ConceptHistory(), n=3, debug=True)]  # type: ignore[arg-type]
    assert selected_no_debug == selected_debug


# ---------------------------------------------------------------------------
# Anchor detection (#19)
# ---------------------------------------------------------------------------


def _anchor_track(
    track_id: str,
    *,
    bpm: float = 172.0,
    camelot: str = "4A",
    label: str = "",
    artist: str = "Artist",
    remixer: str = "",
    enrichment: Literal["high", "medium", "low", ""] = "",
    energy: int | None = None,
    year: int | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=f"Title {track_id}",
        bpm=bpm,
        camelot_key=camelot,
        genre="Drum & Bass",
        label=label,
        remixer=remixer,
        enrichment_confidence=enrichment,
        energy=energy,
        year=year,
    )


def test_score_anchors_returns_value_per_track_in_unit_range() -> None:
    pool = [_anchor_track(f"T{i}", bpm=170.0 + i) for i in range(5)]
    scores = score_anchors(pool, {t.track_id: t for t in pool})
    assert set(scores.keys()) == {t.track_id for t in pool}
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_score_anchors_provenance_boosts_score() -> None:
    bare = _anchor_track("T1")
    rich = _anchor_track(
        "T2",
        remixer="Famous Remixer",
        enrichment="high",
        label="Rare Label",
    )
    pool = [bare, rich]
    scores = score_anchors(pool, {t.track_id: t for t in pool})
    assert scores["T2"] > scores["T1"]


def test_score_anchors_rarity_rewards_unseen_label() -> None:
    rare_label = _anchor_track("T_rare", label="OneOff")
    common_label = _anchor_track("T_common", label="Bigname")
    # Build a fake collection where Bigname appears 10 times.
    collection: dict[str, Track] = {"T_rare": rare_label, "T_common": common_label}
    for i in range(10):
        collection[f"filler_{i}"] = _anchor_track(f"filler_{i}", label="Bigname")
    scores = score_anchors([rare_label, common_label], collection)
    assert scores["T_rare"] > scores["T_common"]


def test_score_anchors_empty_pool_returns_empty_dict() -> None:
    assert score_anchors([], {}) == {}


def test_build_mix_canvas_populates_core_anchor_ids() -> None:
    from mixlab.models import MixCanvas

    # Mix a strongly-anchor track with weak filler tracks. Anchor: rare label, high
    # enrichment, remixer set, year recent, energy in peak band. Filler: nothing.
    tracks = [
        _anchor_track(
            "anchor",
            remixer="X",
            enrichment="high",
            label="UniqueLabel",
            energy=6,
            year=2025,
        ),
    ]
    for i in range(11):
        tracks.append(_anchor_track(f"f{i:02d}", bpm=171.0 + (i * 0.1)))
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    # core pool size ~12; top 20% ≈ 2–3 anchors with floor 0.55.
    assert canvas.core_anchor_ids, "expected at least one anchor"
    assert "anchor" in canvas.core_anchor_ids


def test_build_mix_canvas_no_anchors_when_pool_uniformly_weak() -> None:
    from mixlab.models import MixCanvas

    # All tracks identical & bare — none should clear the anchor threshold.
    tracks = [_anchor_track(f"flat{i:02d}", bpm=172.0 + (i * 0.05), artist=f"A{i}", label="") for i in range(12)]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.core_anchor_ids == []


# ---------------------------------------------------------------------------
# Era and label canvas dimensions (#20)
# ---------------------------------------------------------------------------


def test_build_mix_canvas_tight_era_window_sets_high_coherence() -> None:
    from mixlab.models import MixCanvas

    tracks = [_anchor_track(f"t{i:02d}", bpm=172.0 + (i * 0.05), year=1996 + (i % 3)) for i in range(12)]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.era_window == (1996, 1998)
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.era_coherence == pytest.approx(1.0)


def test_build_mix_canvas_wide_era_span_zero_coherence() -> None:
    from mixlab.models import MixCanvas

    # Span 1992–2024 = 32 years → well past the _ERA_SPAN_ZERO cap.
    tracks = [_anchor_track(f"t{i:02d}", bpm=172.0 + (i * 0.05), year=1992 + (i * 3)) for i in range(11)]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.era_window is not None
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.era_coherence == pytest.approx(0.0)


def test_build_mix_canvas_patchy_year_data_yields_no_era_window() -> None:
    from mixlab.models import MixCanvas

    # Only 2/12 tracks have year — below the 60% coverage floor.
    tracks = [_anchor_track(f"t{i:02d}", bpm=172.0 + (i * 0.05), year=None) for i in range(10)]
    tracks.append(_anchor_track("d1", bpm=172.5, year=2020))
    tracks.append(_anchor_track("d2", bpm=172.6, year=2021))
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.era_window is None
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.era_coherence == pytest.approx(0.0)


def test_build_mix_canvas_dominant_label_above_share_threshold() -> None:
    from mixlab.models import MixCanvas

    # 6 of 11 (~55%) labelled tracks are on Warp — clears 40% share + 5-count floor.
    tracks = [_anchor_track(f"w{i:02d}", bpm=172.0 + (i * 0.05), label="Warp") for i in range(6)]
    tracks += [_anchor_track(f"o{i:02d}", bpm=172.0 + ((i + 6) * 0.05), label=f"Other{i}") for i in range(5)]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.dominant_label == "Warp"
    assert canvas.label_share == pytest.approx(6 / 11, abs=1e-3)
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.label_coherence > 0.0


# ---------------------------------------------------------------------------
# Concept-anchor candidates (#10)
# ---------------------------------------------------------------------------


def _wildcard_track(
    track_id: str,
    *,
    bpm: float = 195.0,
    camelot: str = "12A",
    label: str = "",
    artist: str = "Artist",
    energy: int | None = None,
) -> Track:
    """Build a track far enough from 172 BPM to land in the wildcard tier."""
    return Track(
        track_id=track_id,
        artist=artist,
        title=f"Wild {track_id}",
        bpm=bpm,
        camelot_key=camelot,
        genre="Drum & Bass",
        label=label,
        energy=energy,
    )


def test_concept_anchor_identity_anchor_from_distinctive_label() -> None:
    from mixlab.models import MixCanvas

    # 12 core tracks on common label + 1 wildcard on a unique label → identity anchor.
    core = [_anchor_track(f"c{i:02d}", bpm=172.0 + (i * 0.05), label="Common") for i in range(12)]
    wild = _wildcard_track("wild1", bpm=200.0, camelot="9B", label="Rare Imprint", artist="Solo Artist")
    tracks = core + [wild]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    ids = {a.track_id: a.anchor_type for a in canvas.concept_anchor_candidates}
    assert "wild1" in ids
    assert ids["wild1"] == "identity"


def test_concept_anchor_peak_anchor_from_high_energy_wildcard() -> None:
    from mixlab.models import MixCanvas

    core = [_anchor_track(f"c{i:02d}", bpm=172.0 + (i * 0.05), label="Common") for i in range(12)]
    wild = _wildcard_track("wildPeak", bpm=200.0, camelot="4A", label="Common", energy=7)
    tracks = core + [wild]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    ids = {a.track_id: a.anchor_type for a in canvas.concept_anchor_candidates}
    assert "wildPeak" in ids
    assert ids["wildPeak"] == "peak"


def test_concept_anchor_empty_when_no_exceptional_tracks() -> None:
    from mixlab.models import MixCanvas

    # All-core canvas with no bridge/wildcard tracks → no anchor candidates.
    tracks = [_anchor_track(f"c{i:02d}", bpm=172.0 + (i * 0.05), label="Common") for i in range(12)]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.concept_anchor_candidates == []


def test_concept_anchor_canvas_label_rarity_is_primary() -> None:
    """Within-canvas rarity drives identity flag even when the label is common in the collection."""
    from mixlab.models import MixCanvas

    # Wildcard label appears once in canvas. Same label appears many times in the wider
    # collection — within-canvas rarity should still be the dominant identity signal.
    core = [_anchor_track(f"c{i:02d}", bpm=172.0 + (i * 0.05), label="HouseLabel") for i in range(12)]
    wild = _wildcard_track("wild_loner", bpm=200.0, camelot="9B", label="SoloIn", artist="SoloArtist")
    canvas_tracks = core + [wild]
    # Wider collection adds many tracks on "SoloIn" so collection-level rarity is low.
    collection = {t.track_id: t for t in canvas_tracks}
    for i in range(40):
        collection[f"coll_{i}"] = _anchor_track(f"coll_{i}", label="SoloIn", artist="SoloArtist")
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in canvas_tracks])
    canvas = build_mix_canvas(concept, collection)
    assert isinstance(canvas, MixCanvas)
    ids = {a.track_id: a.anchor_type for a in canvas.concept_anchor_candidates}
    assert ids.get("wild_loner") == "identity"


# ---------------------------------------------------------------------------
# Mode-aware canvas weights (#24)
# ---------------------------------------------------------------------------


def test_canvas_weights_per_mode_sum_to_one() -> None:
    """Every mode-specific weight table must sum exactly to 1.0."""
    from mixlab.clustering import CANVAS_SCORE_WEIGHTS_BY_MODE, DEFAULT_CANVAS_WEIGHTS

    for mode, w in CANVAS_SCORE_WEIGHTS_BY_MODE.items():
        total = (
            w.technical_viability
            + w.role_coverage
            + w.anchor_strength
            + w.contrast_potential
            + w.distinctiveness
            + w.era_coherence
            + w.label_coherence
            + w.novelty
        )
        assert abs(total - 1.0) < 1e-9, f"mode {mode!r} weights sum to {total}, not 1.0"
    default_total = (
        DEFAULT_CANVAS_WEIGHTS.technical_viability
        + DEFAULT_CANVAS_WEIGHTS.role_coverage
        + DEFAULT_CANVAS_WEIGHTS.anchor_strength
        + DEFAULT_CANVAS_WEIGHTS.contrast_potential
        + DEFAULT_CANVAS_WEIGHTS.distinctiveness
        + DEFAULT_CANVAS_WEIGHTS.era_coherence
        + DEFAULT_CANVAS_WEIGHTS.label_coherence
        + DEFAULT_CANVAS_WEIGHTS.novelty
    )
    assert abs(default_total - 1.0) < 1e-9


def test_canvas_weights_invalid_sum_raises() -> None:
    """CanvasScoreWeights enforces the sum-to-1.0 invariant on construction."""
    from mixlab.models import CanvasScoreWeights

    with pytest.raises(ValueError, match="must sum to 1.0"):
        CanvasScoreWeights(
            technical_viability=0.5,
            role_coverage=0.5,
            anchor_strength=0.0,
            contrast_potential=0.0,
            distinctiveness=0.0,
            era_coherence=0.0,
            label_coherence=0.0,
            novelty=0.5,  # total = 1.5
        )


def test_score_canvas_played_mode_rewards_strong_anchors() -> None:
    """Played mode weights anchor_strength heavier — the gap between anchor-rich and
    anchor-poor canvases must widen under played vs unplayed.
    """
    from mixlab.clustering import CANVAS_SCORE_WEIGHTS_BY_MODE
    from mixlab.models import CanvasRoleCandidates, MixCanvas

    canvas_with_anchors = _rich_canvas("with", [f"T{i:03d}" for i in range(20)])
    canvas_no_anchors = _rich_canvas("none", [f"X{i:03d}" for i in range(20)])
    assert isinstance(canvas_with_anchors, MixCanvas)
    assert isinstance(canvas_no_anchors, MixCanvas)
    # Strip opener/closer pools from the second canvas so its anchor_strength → 0.
    canvas_no_anchors.roles = CanvasRoleCandidates(
        opener=[],
        groove_locker=canvas_no_anchors.roles.groove_locker,
        builder=canvas_no_anchors.roles.builder,
        pivot=canvas_no_anchors.roles.pivot,
        peak=canvas_no_anchors.roles.peak,
        closer=[],
    )
    unplayed = CANVAS_SCORE_WEIGHTS_BY_MODE["unplayed"]
    played = CANVAS_SCORE_WEIGHTS_BY_MODE["played"]
    s_with_unplayed = score_canvas(canvas_with_anchors, ConceptHistory(), frozenset(), weights=unplayed)
    s_no_unplayed = score_canvas(canvas_no_anchors, ConceptHistory(), frozenset(), weights=unplayed)
    s_with_played = score_canvas(canvas_with_anchors, ConceptHistory(), frozenset(), weights=played)
    s_no_played = score_canvas(canvas_no_anchors, ConceptHistory(), frozenset(), weights=played)
    gap_unplayed = s_with_unplayed.overall - s_no_unplayed.overall
    gap_played = s_with_played.overall - s_no_played.overall
    assert gap_played > gap_unplayed


def test_score_canvas_mode_affects_overall_when_components_differ() -> None:
    """Unplayed vs played differ on a canvas whose components are not all 1.0 — strip the
    opener pool so anchor_strength=0.5, leaving the mode weights room to disagree.
    """
    from mixlab.clustering import CANVAS_SCORE_WEIGHTS_BY_MODE
    from mixlab.models import CanvasRoleCandidates, MixCanvas

    canvas = _rich_canvas("c1", [f"T{i:03d}" for i in range(20)])
    assert isinstance(canvas, MixCanvas)
    canvas.roles = CanvasRoleCandidates(
        opener=[],
        groove_locker=canvas.roles.groove_locker,
        builder=canvas.roles.builder,
        pivot=canvas.roles.pivot,
        peak=canvas.roles.peak,
        closer=canvas.roles.closer,
    )
    s_unplayed = score_canvas(canvas, ConceptHistory(), frozenset(), weights=CANVAS_SCORE_WEIGHTS_BY_MODE["unplayed"])
    s_played = score_canvas(canvas, ConceptHistory(), frozenset(), weights=CANVAS_SCORE_WEIGHTS_BY_MODE["played"])
    assert s_unplayed.overall != s_played.overall


def test_select_canvases_threads_mode_to_score_canvas() -> None:
    """select_canvases with mode='played' produces different scores than mode='unplayed'."""
    from mixlab.models import CanvasRoleCandidates, MixCanvas

    def _strip_opener(canvases: list[MixCanvas]) -> None:
        # Force a non-trivial scoring surface so the mode weights disagree.
        for c in canvases:
            c.roles = CanvasRoleCandidates(
                opener=[],
                groove_locker=c.roles.groove_locker,
                builder=c.roles.builder,
                pivot=c.roles.pivot,
                peak=c.roles.peak,
                closer=c.roles.closer,
            )

    canvases_a = [_rich_canvas(f"c{i}", [f"T{i}{j:02d}" for j in range(10)]) for i in range(4)]
    canvases_b = [_rich_canvas(f"c{i}", [f"T{i}{j:02d}" for j in range(10)]) for i in range(4)]
    assert all(isinstance(c, MixCanvas) for c in canvases_a + canvases_b)
    typed_a: list[MixCanvas] = [c for c in canvases_a if isinstance(c, MixCanvas)]
    typed_b: list[MixCanvas] = [c for c in canvases_b if isinstance(c, MixCanvas)]
    _strip_opener(typed_a)
    _strip_opener(typed_b)
    picked_unplayed = select_canvases(typed_a, ConceptHistory(), n=4, mode="unplayed")
    picked_played = select_canvases(typed_b, ConceptHistory(), n=4, mode="played")
    scores_unplayed = [c.score.overall for c in picked_unplayed]
    scores_played = [c.score.overall for c in picked_played]
    assert scores_unplayed != scores_played


def test_build_mix_canvas_no_dominant_label_when_scattered() -> None:
    from mixlab.models import MixCanvas

    tracks = [_anchor_track(f"t{i:02d}", bpm=172.0 + (i * 0.05), label=f"Label{i}") for i in range(12)]
    concept = MixConcept(title="P", mood="dark", track_ids=[t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, {t.track_id: t for t in tracks})
    assert isinstance(canvas, MixCanvas)
    assert canvas.dominant_label is None
    s = score_canvas(canvas, ConceptHistory(), frozenset())
    assert s.label_coherence == pytest.approx(0.0)
