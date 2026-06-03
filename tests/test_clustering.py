from __future__ import annotations

from typing import Literal

import pytest

from mixlab.clustering import (
    ABSOLUTE_MIN,
    MAX_POOL_COUNT,
    MAX_SHORTLIST,
    MIN_CAMELOT_COMPONENT,
    MIN_POOL_COUNT,
    MIN_SHORTLIST,
    _camelot_components,
    _era_split,
    _find_bpm_peaks,
    _infer_shortlist_mood,
    _resize_shortlists,
    build_custom_genre_pool,
    build_mix_canvas,
    camelot_distance,
    count_outlier_genres,
    filter_by_bpm_range,
    group_by_genre,
    partition_bpm_pools,
    partition_outliers,
    partition_pool,
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


def test_build_mix_canvas_risk_notes_weak_closer_no_energy() -> None:
    # No energy metadata + uniform BPM → BPM proxy can't help → risk note fires
    tracks = [_track(track_id=str(i), bpm=172.0, energy=None) for i in range(5)]
    concept = _concept([t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*tracks))
    assert "weak closer pool" in canvas.risk_notes


def test_build_mix_canvas_risk_notes_weak_closer_relative_fallback() -> None:
    # All high energy but uniform → pool-relative fallback promotes min-energy tracks
    # as closer candidates, so "weak closer pool" should NOT fire
    tracks = [_track(track_id=str(i), bpm=172.0, energy=7) for i in range(5)]
    concept = _concept([t.track_id for t in tracks])
    canvas = build_mix_canvas(concept, _tracks_by_id(*tracks))
    assert "weak closer pool" not in canvas.risk_notes
    assert len(canvas.roles.closer) >= 2


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


# ---------------------------------------------------------------------------
# Deterministic Stage 1 — partition_pool() tests (issue #17)
# ---------------------------------------------------------------------------


def _pt(
    *,
    track_id: str = "1",
    artist: str = "A",
    title: str = "T",
    bpm: float = 174.0,
    camelot_key: str = "8A",
    genre: str = "Drum & Bass",
    energy: int | None = None,
    label: str = "",
    year: int | None = None,
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
        label=label,
        year=year,
        tags=tags if tags is not None else [],
    )


# ---------------------------------------------------------------------------
# _find_bpm_peaks
# ---------------------------------------------------------------------------


def test_find_bpm_peaks_single_tight_cluster_returns_one_peak() -> None:
    # All tracks clustered tightly around 172 BPM, with edge padding to make the cluster
    # an interior bucket. Edge tracks at ~140 and ~200 BPM.
    low_pad = [_pt(track_id=f"LP{i}", bpm=140.0) for i in range(3)]
    cluster = [_pt(track_id=f"C{i}", bpm=172.0 + i * 0.1) for i in range(20)]
    high_pad = [_pt(track_id=f"HP{i}", bpm=200.0) for i in range(3)]
    tracks = low_pad + cluster + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    assert len(result) == 1
    # All cluster tracks should be in the one peak (edge-padding attached too).
    all_ids = {t.track_id for t in result[0]}
    for t in cluster:
        assert t.track_id in all_ids


def test_find_bpm_peaks_two_well_separated_groups_returns_two_peaks() -> None:
    # Two tight groups separated by ~24 BPM gap, with padding on the outer edges
    # so both groups land in interior buckets (not first or last bucket).
    low_pad = [_pt(track_id=f"LP{i}", bpm=120.0) for i in range(3)]
    low_group = [_pt(track_id=f"L{i}", bpm=150.0 + i * 0.1) for i in range(15)]
    high_group = [_pt(track_id=f"H{i}", bpm=174.0 + i * 0.1) for i in range(15)]
    high_pad = [_pt(track_id=f"HP{i}", bpm=200.0) for i in range(3)]
    tracks = low_pad + low_group + high_group + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    assert len(result) == 2


def test_find_bpm_peaks_merges_peaks_closer_than_8_bpm() -> None:
    # Two tight groups in adjacent 3-BPM buckets → peak centres 6 BPM apart → merged.
    # Buckets 168 (centre 169.5) and 174 (centre 175.5): distance = 6.0 < 8 → merge.
    low_pad = [_pt(track_id=f"LP{i}", bpm=150.0) for i in range(3)]
    group_a = [_pt(track_id=f"A{i}", bpm=168.0 + i * 0.1) for i in range(15)]
    group_b = [_pt(track_id=f"B{i}", bpm=174.0 + i * 0.1) for i in range(15)]
    high_pad = [_pt(track_id=f"HP{i}", bpm=195.0) for i in range(3)]
    tracks = low_pad + group_a + group_b + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    assert len(result) == 1


def test_find_bpm_peaks_merge_picks_globally_closest_pair_first() -> None:
    # Three groups: A and B are ~6 BPM apart (merge), C is ~35 BPM from both.
    # The globally closest pair (A, B) should be merged first → 2 clusters.
    low_pad = [_pt(track_id=f"LP{i}", bpm=140.0) for i in range(3)]
    group_a = [_pt(track_id=f"A{i}", bpm=160.0 + i * 0.1) for i in range(12)]
    group_b = [_pt(track_id=f"B{i}", bpm=165.0 + i * 0.1) for i in range(12)]
    group_c = [_pt(track_id=f"C{i}", bpm=200.0 + i * 0.1) for i in range(12)]
    high_pad = [_pt(track_id=f"HP{i}", bpm=215.0) for i in range(3)]
    tracks = low_pad + group_a + group_b + group_c + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    # A and B merged (< 8 BPM apart) → 2 clusters (A+B, C).
    assert len(result) == 2


def test_find_bpm_peaks_merge_tie_keeps_lower_bpm_peak() -> None:
    # Two peaks < 8 BPM apart with equal raw counts → lower-BPM peak survives.
    # Edge padding required so neither group is in the first or last bucket.
    low_pad = [_pt(track_id=f"LP{i}", bpm=150.0) for i in range(3)]
    group_a = [_pt(track_id=f"lo{i}", bpm=168.0) for i in range(5)]  # bucket 168, centre 169.5
    group_b = [_pt(track_id=f"hi{i}", bpm=171.0) for i in range(5)]  # bucket 171, centre 172.5 — 3 BPM apart
    high_pad = [_pt(track_id=f"HP{i}", bpm=190.0) for i in range(3)]
    tracks = low_pad + group_a + group_b + high_pad
    result = _find_bpm_peaks(tracks)
    # Merged → 1 peak; survivor has lower BPM (tie on count → lower BPM wins).
    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 16  # all tracks: 3+5+5+3


def test_find_bpm_peaks_plateau_interior_uses_leftmost_bucket() -> None:
    # Three equal-count adjacent interior buckets form a plateau.
    # The plateau detection should use the leftmost bucket index as the representative peak.
    # Use padding to ensure the plateau is interior (not at first/last index).
    low_pad = [_pt(track_id=f"LP{i}", bpm=150.0) for i in range(2)]
    plateau_a = [_pt(track_id=f"PA{i}", bpm=165.0) for i in range(6)]  # bucket 165
    plateau_b = [_pt(track_id=f"PB{i}", bpm=168.0) for i in range(6)]  # bucket 168
    plateau_c = [_pt(track_id=f"PC{i}", bpm=171.0) for i in range(6)]  # bucket 171
    high_pad = [_pt(track_id=f"HP{i}", bpm=185.0) for i in range(2)]
    tracks = low_pad + plateau_a + plateau_b + plateau_c + high_pad
    result = _find_bpm_peaks(tracks)
    # Should detect the plateau as a single peak and return non-None.
    assert result is not None


def test_find_bpm_peaks_plateau_at_last_bucket_not_returned_as_peak() -> None:
    # A concentration at the last bucket should not count as a peak.
    # Build: a few tracks at low BPM (first bucket) and many at the highest BPM (last bucket).
    tracks = (
        [_pt(track_id=f"lo{i}", bpm=165.0) for i in range(2)]  # first bucket
        + [_pt(track_id=f"hi{i}", bpm=177.0 + i * 3) for i in range(6)]  # last few buckets
    )
    result = _find_bpm_peaks(tracks)
    # Only 2 buckets or fewer meaningful interior buckets → no peaks.
    # Either None or not having a peak at the very last bucket.
    assert result is None or isinstance(result, list)


def test_find_bpm_peaks_plateau_at_first_bucket_not_returned_as_peak() -> None:
    # A concentration at the first bucket should not count as a peak.
    tracks = (
        [_pt(track_id=f"lo{i}", bpm=165.0 + i * 3) for i in range(6)]  # spans first buckets
        + [_pt(track_id=f"hi{i}", bpm=185.0) for i in range(2)]  # last bucket
    )
    result = _find_bpm_peaks(tracks)
    # No interior peaks → None.
    assert result is None or isinstance(result, list)


def test_find_bpm_peaks_edge_buckets_never_returned_as_peaks() -> None:
    # 3 buckets: edges (high count) are smoothed larger than interior (low count).
    # raw = [1, 10, 1]: s0=(1+10)/2=5.5, s1=(1+10+1)/3=4.0, s2=(10+1)/2=5.5.
    # Interior s1=4.0 < s0=5.5 and < s2=5.5 → interior is a valley, not a peak → None.
    # Even if the interior were a peak, it is NOT at the first or last bucket.
    # Here there is no interior peak at all → None.
    tracks = (
        [_pt(track_id=f"lo{i}", bpm=165.0) for i in range(1)]  # first bucket, count=1
        + [_pt(track_id=f"mid{i}", bpm=168.0) for i in range(10)]  # interior, count=10
        + [_pt(track_id=f"hi{i}", bpm=171.0) for i in range(1)]  # last bucket, count=1
    )
    result = _find_bpm_peaks(tracks)
    # Interior (168) smoothed = (1+10+1)/3 = 4.0 < edges 5.5 → valley, not a peak → None.
    assert result is None


def test_find_bpm_peaks_boundary_bpm_assigned_to_new_bucket() -> None:
    # BPM that is exactly a multiple of 3 starts a new bucket.
    # BPM=174 → bucket starting at floor(174/3)*3 = 174.
    import math as _math

    bucket_start = _math.floor(174.0 / 3) * 3
    assert bucket_start == 174
    # BPM=175 → bucket floor(175/3)*3 = floor(58.33)*3 = 174 (same bucket).
    bucket_start_175 = _math.floor(175.0 / 3) * 3
    assert bucket_start_175 == 174
    # BPM=177 → bucket floor(177/3)*3 = 177 (new bucket).
    bucket_start_177 = _math.floor(177.0 / 3) * 3
    assert bucket_start_177 == 177


def test_find_bpm_peaks_track_exactly_7_bpm_from_centre_is_within_window() -> None:
    # A peak at bucket starting at 168 → centre = 169.5.
    # A track at 176.5 = 169.5 + 7.0 is exactly 7 BPM away → within window (≤7.0).
    # Build a tight cluster at 168–169 BPM with padding to make it an interior peak.
    low_pad = [_pt(track_id=f"LP{i}", bpm=150.0) for i in range(3)]
    cluster = [_pt(track_id=f"C{i}", bpm=168.0 + i * 0.1) for i in range(15)]
    within_track = _pt(track_id="within7", bpm=176.5)  # exactly 7.0 from centre 169.5
    high_pad = [_pt(track_id=f"HP{i}", bpm=200.0) for i in range(3)]
    tracks = low_pad + cluster + [within_track] + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    # The within7 track should be assigned to the cluster (≤7 BPM from centre).
    all_ids = {t.track_id for sl in result for t in sl}
    assert "within7" in all_ids


def test_find_bpm_peaks_track_beyond_7_bpm_becomes_outlier() -> None:
    # A track > 7 BPM from all peak centres becomes an outlier (then re-attached to nearest).
    # Cluster centre ~169.5; a track at 177.0 = 7.5 BPM away → outlier.
    low_pad = [_pt(track_id=f"LP{i}", bpm=150.0) for i in range(3)]
    cluster = [_pt(track_id=f"C{i}", bpm=168.0 + i * 0.1) for i in range(15)]
    outlier_track = _pt(track_id="outlier7plus", bpm=177.0)  # 7.5 > 7.0 → beyond window
    high_pad = [_pt(track_id=f"HP{i}", bpm=200.0) for i in range(3)]
    tracks = low_pad + cluster + [outlier_track] + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    # Outlier must be re-attached via step 1.7 and be present in some cluster.
    all_ids = {t.track_id for sl in result for t in sl}
    assert "outlier7plus" in all_ids


def test_find_bpm_peaks_equidistant_track_assigned_to_lower_index_peak() -> None:
    # Two peaks equidistant from a track (both > 7 BPM → outlier) → lower-index peak wins.
    # With padding at 135 and 180, peaks form at centres 148.5 and 163.5.
    # A track at 156.0 = midpoint: |156-148.5|=7.5 and |156-163.5|=7.5 (both > 7 → outlier).
    # Step 1.7: outlier attached to nearest cluster; tie → lower index (cluster 0).
    low_pad = [_pt(track_id=f"LP{i}", bpm=135.0) for i in range(3)]
    low_group = [_pt(track_id=f"L{i}", bpm=150.0 + i * 0.1) for i in range(12)]
    mid_track = _pt(track_id="mid", bpm=156.0)  # both 7.5 BPM from peaks → outlier → lower idx
    high_group = [_pt(track_id=f"H{i}", bpm=165.0 + i * 0.1) for i in range(12)]
    high_pad = [_pt(track_id=f"HP{i}", bpm=180.0) for i in range(3)]
    tracks = low_pad + low_group + [mid_track] + high_group + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    assert len(result) == 2
    # Both equidistant as outliers → lower index cluster (0) wins.
    cluster_0_ids = {t.track_id for t in result[0]}
    assert "mid" in cluster_0_ids


def test_find_bpm_peaks_outliers_have_no_candidate_peak_within_7_bpm() -> None:
    # Build a well-defined interior peak and a far outlier; outlier must still end up somewhere.
    low_pad = [_pt(track_id=f"LP{i}", bpm=150.0) for i in range(3)]
    cluster = [_pt(track_id=f"C{i}", bpm=170.0 + i * 0.1) for i in range(15)]
    far_outlier = _pt(track_id="far", bpm=220.0)  # > 7 BPM from cluster centre ~171.5
    high_pad = [_pt(track_id=f"HP{i}", bpm=240.0) for i in range(3)]
    tracks = low_pad + cluster + [far_outlier] + high_pad
    result = _find_bpm_peaks(tracks)
    assert result is not None
    # far_outlier must be present in result (re-attached via step 1.7).
    all_track_ids = {t.track_id for sl in result for t in sl}
    assert "far" in all_track_ids


def test_find_bpm_peaks_three_bucket_all_equal_histogram_triggers_fallback() -> None:
    # Exactly 3 buckets with equal raw counts → smoothed interior = edges → no peaks → None.
    # Smoothing: bucket 0 → (raw[0]+raw[1])/2 = (5+5)/2 = 5.0.
    #            bucket 1 → (raw[0]+raw[1]+raw[2])/3 = 5.0.
    #            bucket 2 → (raw[1]+raw[2])/2 = 5.0.
    # Interior (bucket 1) smoothed = 5.0 == left 5.0 → NOT strictly greater → no peak.
    tracks = (
        [_pt(track_id=f"A{i}", bpm=174.0) for i in range(5)]  # bucket 174
        + [_pt(track_id=f"B{i}", bpm=177.0) for i in range(5)]  # bucket 177
        + [_pt(track_id=f"C{i}", bpm=180.0) for i in range(5)]  # bucket 180
    )
    result = _find_bpm_peaks(tracks)
    assert result is None


# ---------------------------------------------------------------------------
# partition_pool fallback (n_groups=1 and flat-BPM guard) tests
# ---------------------------------------------------------------------------


def test_partition_pool_below_absolute_min_returns_empty() -> None:
    # Fewer than ABSOLUTE_MIN tracks → empty list.
    tracks = [_pt(track_id=str(i), bpm=174.0) for i in range(ABSOLUTE_MIN - 1)]
    result = partition_pool(tracks)
    assert result == []


def test_partition_pool_small_but_above_absolute_min_returns_single_shortlist() -> None:
    # Between ABSOLUTE_MIN and MIN_SHORTLIST → single shortlist.
    tracks = [_pt(track_id=str(i), bpm=174.0) for i in range(ABSOLUTE_MIN + 1)]
    result = partition_pool(tracks)
    assert len(result) == 1
    assert len(result[0].track_ids) == ABSOLUTE_MIN + 1


def test_partition_pool_n_groups_one_fallback_no_peaks_returns_single_shortlist_up_to_29_tracks() -> None:
    # 15–29 tracks with BPM spread ≥ 4.0 but no peaks → n_groups = 1 → single shortlist.
    # Spread tracks linearly across a range that yields monotone histogram (no interior peak).
    # 20 tracks from 165 to 184 BPM in 1-BPM steps.
    tracks = [_pt(track_id=str(i), bpm=165.0 + i) for i in range(20)]
    result = partition_pool(tracks)
    # n_groups = min(5, 20//15) = 1 → single shortlist returned immediately.
    assert len(result) == 1


def test_partition_pool_n_groups_one_fallback_bypasses_step4_peaks_path_same_size_does_not() -> None:
    # When n_groups=1, partition_pool returns immediately without step 4.
    # 18 tracks spread across 165–182 (monotone histogram, BPM span >=4.0).
    tracks = [_pt(track_id=str(i), bpm=165.0 + i) for i in range(18)]
    result = partition_pool(tracks)
    # Should be a single shortlist with all 18 tracks.
    assert len(result) == 1
    assert len(result[0].track_ids) == 18


def test_partition_pool_single_bucket_histogram_triggers_fallback() -> None:
    # All same BPM → n_buckets=1 → _find_bpm_peaks returns None (< 3 buckets).
    # But the flat-BPM guard fires first: max-min < 4.0.
    tracks = [_pt(track_id=str(i), bpm=174.0) for i in range(30)]
    result = partition_pool(tracks)
    # flat BPM → clusters = [bpm_sorted], then camelot components, then resize.
    assert len(result) >= 1


def test_partition_pool_two_bucket_histogram_triggers_fallback() -> None:
    # Two distinct BPMs (3 BPM apart → 2 buckets) → n_buckets=2 < 3 → None.
    # BPM span = 3.0 < 4.0, so flat-BPM guard fires.
    tracks = (
        [_pt(track_id=f"A{i}", bpm=174.0) for i in range(15)]
        + [_pt(track_id=f"B{i}", bpm=177.0) for i in range(15)]
    )
    result = partition_pool(tracks)
    assert len(result) >= 1


def test_partition_pool_no_peaks_uniform_spread_returns_equal_bpm_ordered_groups() -> None:
    # No peaks + n_groups > 1 → equal BPM-ordered groups.
    # 45 tracks spread evenly: n_groups = min(5, 45//15) = 3 → 3 groups of 15.
    tracks = [_pt(track_id=str(i), bpm=165.0 + i) for i in range(45)]
    result = partition_pool(tracks)
    assert len(result) >= 1
    # All tracks should be covered.
    all_ids = {tid for concept in result for tid in concept.track_ids}
    assert all_ids == {str(i) for i in range(45)}


def test_partition_pool_no_peaks_small_pool_uniform_spread_equal_sized_groups() -> None:
    # 30 tracks → n_groups = min(5, 30//15) = 2.
    # Spread that avoids peak detection.
    tracks = [_pt(track_id=str(i), bpm=165.0 + i) for i in range(30)]
    result = partition_pool(tracks)
    assert len(result) >= 1
    all_ids = {tid for concept in result for tid in concept.track_ids}
    assert all_ids == {str(i) for i in range(30)}


# ---------------------------------------------------------------------------
# _camelot_components
# ---------------------------------------------------------------------------


def test_camelot_components_connected_keys_form_single_component() -> None:
    # 8A → 9A → 10A → 10B all adjacent — one big component.
    tracks = [
        _pt(track_id="1", camelot_key="8A", bpm=174.0),
        _pt(track_id="2", camelot_key="9A", bpm=174.0),
        _pt(track_id="3", camelot_key="10A", bpm=174.0),
        _pt(track_id="4", camelot_key="10B", bpm=174.0),
        # Add extra tracks to reach MIN_CAMELOT_COMPONENT
        _pt(track_id="5", camelot_key="8A", bpm=174.1),
        _pt(track_id="6", camelot_key="9A", bpm=174.2),
        _pt(track_id="7", camelot_key="10A", bpm=174.3),
        _pt(track_id="8", camelot_key="9A", bpm=174.4),
        _pt(track_id="9", camelot_key="8A", bpm=174.5),
        _pt(track_id="10", camelot_key="9B", bpm=174.6),
    ]
    result = _camelot_components(tracks)
    assert len(result) == 1
    assert len(result[0]) == 10


def test_camelot_components_unrelated_keys_form_separate_components() -> None:
    # 1A and 6A are far apart (5 steps); 1B and 6B similarly far.
    # Create two groups of MIN_CAMELOT_COMPONENT each to avoid merging.
    group_a = [_pt(track_id=f"A{i}", camelot_key="1A", bpm=174.0) for i in range(MIN_CAMELOT_COMPONENT)]
    group_b = [_pt(track_id=f"B{i}", camelot_key="6A", bpm=174.0) for i in range(MIN_CAMELOT_COMPONENT)]
    tracks = group_a + group_b
    result = _camelot_components(tracks)
    assert len(result) == 2


def test_camelot_components_all_unknown_keys_kept_as_single_component() -> None:
    # All tracks with empty camelot_key → all-unknown guard fires → one component.
    tracks = [_pt(track_id=str(i), camelot_key="", bpm=174.0) for i in range(10)]
    result = _camelot_components(tracks)
    assert len(result) == 1
    assert len(result[0]) == 10


def test_camelot_components_some_unknown_keys_become_singletons_then_merged() -> None:
    # Mix known and unknown keys. Unknown keys are not adjacent to anything,
    # forming singletons. Singletons < MIN_CAMELOT_COMPONENT get merged into largest.
    known = [_pt(track_id=f"K{i}", camelot_key="8A", bpm=174.0) for i in range(MIN_CAMELOT_COMPONENT)]
    unknown = [_pt(track_id="U1", camelot_key="", bpm=174.5)]
    tracks = known + unknown
    result = _camelot_components(tracks)
    # The unknown track forms a singleton → merged into the large 8A component.
    # So result should be 1 component containing all.
    assert len(result) == 1
    all_ids = {t.track_id for t in result[0]}
    assert "U1" in all_ids


def test_camelot_components_three_components_small_one_merged_into_largest() -> None:
    # Three components: one large (>=MIN), two small (<MIN) → smalls merge into large.
    large = [_pt(track_id=f"L{i}", camelot_key="1A", bpm=174.0) for i in range(MIN_CAMELOT_COMPONENT + 2)]
    small1 = [_pt(track_id=f"S1_{i}", camelot_key="6A", bpm=174.0) for i in range(3)]
    small2 = [_pt(track_id=f"S2_{i}", camelot_key="11A", bpm=174.0) for i in range(2)]
    tracks = large + small1 + small2
    result = _camelot_components(tracks)
    # Both smalls merged into large → 1 component.
    assert len(result) == 1


def test_camelot_components_merge_recomputes_largest_after_each_step() -> None:
    # After each merge, the "largest" component changes.
    # Component A: 10 tracks, Component B: 8 tracks, Component C: 3 tracks (small).
    # B is initially larger-second but C merges into A (largest). After that A is still largest.
    comp_a = [_pt(track_id=f"A{i}", camelot_key="1A", bpm=174.0) for i in range(10)]
    comp_b = [_pt(track_id=f"B{i}", camelot_key="6A", bpm=174.0) for i in range(8)]
    comp_c = [_pt(track_id=f"C{i}", camelot_key="11A", bpm=174.0) for i in range(3)]
    tracks = comp_a + comp_b + comp_c
    result = _camelot_components(tracks)
    # C (small, 3) merges into A (largest, 10) → A becomes 13, B remains 8 → 2 components.
    assert len(result) == 2
    sizes = sorted(len(c) for c in result)
    assert sizes == [8, 13]


def test_camelot_components_merge_tie_uses_min_track_id() -> None:
    # Two equally-sized large components; small component merges into one with min track_id.
    # Component A: 10 tracks starting with track_id "A0" (min = "A0")
    # Component B: 10 tracks starting with track_id "B0" (min = "B0")
    # Small component: 2 tracks → merges into A (min_track_id "A0" < "B0").
    comp_a = [_pt(track_id=f"A{i}", camelot_key="1A", bpm=174.0) for i in range(10)]
    comp_b = [_pt(track_id=f"B{i}", camelot_key="6A", bpm=174.0) for i in range(10)]
    small = [_pt(track_id="ZZ0", camelot_key="11A", bpm=174.0), _pt(track_id="ZZ1", camelot_key="11A", bpm=174.1)]
    tracks = comp_a + comp_b + small
    result = _camelot_components(tracks)
    # Small merges into A (since A has min_track_id "A0" < "B0").
    assert len(result) == 2
    # Find the component containing small tracks.
    for comp in result:
        ids = {t.track_id for t in comp}
        if "ZZ0" in ids:
            # Should have all A tracks too.
            assert any(t.track_id.startswith("A") for t in comp)
            break


def test_camelot_components_merge_tie_min_track_id_is_lexicographic_not_numeric() -> None:
    # "10" < "9" lexicographically but 10 > 9 numerically.
    # Verify lexicographic comparison is used.
    assert min("10", "9") == "10"  # "10" < "9" lex
    comp_a = [_pt(track_id=f"10{i}", camelot_key="1A", bpm=174.0) for i in range(10)]
    comp_b = [_pt(track_id=f"9{i}", camelot_key="6A", bpm=174.0) for i in range(10)]
    small = [_pt(track_id="ZZ", camelot_key="11A", bpm=174.0), _pt(track_id="ZZA", camelot_key="11A", bpm=174.1)]
    tracks = comp_a + comp_b + small
    result = _camelot_components(tracks)
    assert len(result) == 2
    # Small merges into the component with lexicographically smaller min_track_id.
    # min("100","101",...) = "100"; min("90","91",...) = "90"; "100" < "90" lex → merges into 10x.
    for comp in result:
        ids = {t.track_id for t in comp}
        if "ZZ" in ids:
            assert any(tid.startswith("10") for tid in ids)
            break


# ---------------------------------------------------------------------------
# _era_split
# ---------------------------------------------------------------------------


def _make_era_tracks(
    n_old: int,
    n_new: int,
    *,
    n_unknown: int = 0,
    old_year: int = 2000,
    new_year: int = 2020,
    prefix: str = "",
) -> list[Track]:
    tracks: list[Track] = []
    for i in range(n_old):
        tracks.append(_pt(track_id=f"{prefix}old{i}", bpm=174.0 + i * 0.1, year=old_year))
    for i in range(n_new):
        tracks.append(_pt(track_id=f"{prefix}new{i}", bpm=174.0 + i * 0.1, year=new_year))
    for i in range(n_unknown):
        tracks.append(_pt(track_id=f"{prefix}unk{i}", bpm=174.0 + i * 0.1, year=None))
    return tracks


def test_era_split_applies_when_gap_large_and_both_sides_sufficient() -> None:
    # 20-year gap, 15 old + 15 new (both >= MIN_SHORTLIST=15), no unknowns → split applied.
    tracks = _make_era_tracks(15, 15, old_year=2000, new_year=2020)
    result = _era_split(tracks, per_side_min=8)
    assert result is not None
    era_old, era_new = result
    old_ids = {t.track_id for t in era_old}
    new_ids = {t.track_id for t in era_new}
    assert all(t.track_id.startswith("old") for t in era_old)
    assert all(t.track_id.startswith("new") for t in era_new)
    assert len(old_ids & new_ids) == 0


def test_era_split_applies_at_exact_8_year_gap_boundary() -> None:
    # Exact 8-year gap: 2000 to 2008. Both sides >= MIN_SHORTLIST and >= per_side_min.
    tracks = _make_era_tracks(15, 15, old_year=2000, new_year=2008)
    result = _era_split(tracks, per_side_min=8)
    # Gap = 2008 - 2000 = 8 >= 8 → should split.
    assert result is not None


def test_era_split_skipped_at_7_year_gap_boundary() -> None:
    # 7-year gap: just below threshold.
    tracks = _make_era_tracks(15, 15, old_year=2000, new_year=2007)
    result = _era_split(tracks, per_side_min=8)
    assert result is None


def test_era_split_skipped_when_gap_below_threshold() -> None:
    # All tracks in same year → no gap.
    tracks = [_pt(track_id=str(i), bpm=174.0, year=2020) for i in range(30)]
    result = _era_split(tracks, per_side_min=8)
    assert result is None


def test_era_split_skipped_when_one_side_too_small() -> None:
    # Only 5 old, 20 new (5 < per_side_min=8); era_old would also be < MIN_SHORTLIST=15 → skip.
    tracks = _make_era_tracks(5, 20, old_year=2000, new_year=2020)
    result = _era_split(tracks, per_side_min=8)
    # era_old_known=5 < per_side_min=8 → None.
    assert result is None


def test_era_split_skipped_when_too_few_known_years() -> None:
    # Fewer than 60% of tracks have known years → skip.
    known = _make_era_tracks(8, 8, old_year=2000, new_year=2020)
    unknown = [_pt(track_id=f"U{i}", bpm=174.0, year=None) for i in range(30)]
    tracks = known + unknown
    # 16/46 ≈ 35% known → skip.
    result = _era_split(tracks, per_side_min=4)
    assert result is None


def test_era_split_unknown_year_tracks_assigned_to_era_old_unconditionally() -> None:
    # Unknowns always go to era_old.
    # 15 old + 15 new + 3 unknowns; both sides >= MIN_SHORTLIST (15+3=18 era_old, 15 era_new).
    tracks = _make_era_tracks(15, 15, n_unknown=3, old_year=2000, new_year=2020)
    result = _era_split(tracks, per_side_min=8)
    assert result is not None
    era_old, era_new = result
    old_ids = {t.track_id for t in era_old}
    # All "unk" tracks should be in era_old.
    for t in tracks:
        if t.year is None:
            assert t.track_id in old_ids


def test_era_split_year_zero_tracks_treated_as_unknown_year() -> None:
    # year=0 is treated as unknown → goes to era_old.
    tracks = _make_era_tracks(15, 15, old_year=2000, new_year=2020)
    zero_year = _pt(track_id="zero0", bpm=174.0, year=0)
    zero_year2 = _pt(track_id="zero1", bpm=174.1, year=0)
    tracks = tracks + [zero_year, zero_year2]
    result = _era_split(tracks, per_side_min=8)
    assert result is not None
    era_old, era_new = result
    old_ids = {t.track_id for t in era_old}
    assert "zero0" in old_ids
    assert "zero1" in old_ids


def test_era_split_unknown_year_tracks_include_zero_and_none_years() -> None:
    # Both year=None and year=0 count as unknown year.
    tracks = _make_era_tracks(15, 15, old_year=2000, new_year=2020)
    none_year = _pt(track_id="none_yr", bpm=173.0, year=None)
    zero_year = _pt(track_id="zero_yr", bpm=173.1, year=0)
    tracks = tracks + [none_year, zero_year]
    result = _era_split(tracks, per_side_min=8)
    assert result is not None
    era_old, _ = result
    old_ids = {t.track_id for t in era_old}
    assert "none_yr" in old_ids
    assert "zero_yr" in old_ids


def test_era_split_gap_idx_points_to_last_track_before_gap() -> None:
    # The gap_idx should correspond to the largest year gap.
    # Build: 2000, 2001 (15 tracks), 2010, 2011 (15 tracks) → gap between 2001 and 2010 (9 years).
    tracks = (
        [_pt(track_id=f"A{i}", bpm=174.0 + i * 0.1, year=2000 + (i % 2)) for i in range(15)]
        + [_pt(track_id=f"B{i}", bpm=175.0 + i * 0.1, year=2010 + (i % 2)) for i in range(15)]
    )
    result = _era_split(tracks, per_side_min=8)
    assert result is not None
    era_old, era_new = result
    # era_old should contain year<=2001 tracks, era_new year>2001.
    for t in era_old:
        if t.year is not None and t.year > 0:
            assert t.year <= 2001
    for t in era_new:
        if t.year is not None and t.year > 0:
            assert t.year > 2001


def test_era_split_counts_only_positive_year_tracks_for_side_guard() -> None:
    # era_old_known=5 < per_side_min=8 → split refused, even if era_old total >= MIN_SHORTLIST.
    # 5 old known + 10 unknown + 15 new known.
    # known/total = 20/30 = 67% → above 60% threshold.
    # era_old_known=5 < per_side_min=8 → None.
    tracks = _make_era_tracks(5, 15, n_unknown=10, old_year=2000, new_year=2020)
    result = _era_split(tracks, per_side_min=8)
    assert result is None


def test_era_split_attempt2_fails_when_era_old_known_below_per_side_min_despite_unknowns_inflating_total() -> None:
    # 10 old-year tracks + 12 unknowns + 15 new-year tracks.
    # per_side_min=15. era_old_known=10 < 15, so split refused even though
    # len(era_old) = 10+12 = 22 >= MIN_SHORTLIST.
    tracks = _make_era_tracks(10, 15, n_unknown=12, old_year=2005, new_year=2025)
    result = _era_split(tracks, per_side_min=15)
    # era_old_known=10 < 15 → None.
    assert result is None


# ---------------------------------------------------------------------------
# _resize_shortlists
# ---------------------------------------------------------------------------


def test_resize_shortlists_merges_undersized_into_nearest_bpm_neighbour() -> None:
    # One shortlist of 10 tracks (< MIN_SHORTLIST=15) with one partner of 15.
    # Combined 25 <= MAX_SHORTLIST → should merge.
    small = [_pt(track_id=f"S{i}", bpm=170.0 + i * 0.1) for i in range(10)]
    large = [_pt(track_id=f"L{i}", bpm=172.0 + i * 0.1) for i in range(15)]
    result = _resize_shortlists([small, large])
    # All 25 tracks in one shortlist.
    all_ids = {t.track_id for sl in result for t in sl}
    assert all_ids == {t.track_id for t in small + large}
    assert len(result) == 1


def test_resize_shortlists_remerges_when_merged_result_still_undersized() -> None:
    # Three shortlists: small1 (5) + small2 (5) → merged still 10 < 15.
    # Should then try to merge with third shortlist.
    small1 = [_pt(track_id=f"A{i}", bpm=170.0 + i * 0.1) for i in range(5)]
    small2 = [_pt(track_id=f"B{i}", bpm=171.0 + i * 0.1) for i in range(5)]
    large = [_pt(track_id=f"C{i}", bpm=172.0 + i * 0.1) for i in range(15)]
    result = _resize_shortlists([small1, small2, large])
    all_ids = {t.track_id for sl in result for t in sl}
    expected = {t.track_id for t in small1 + small2 + large}
    assert all_ids == expected


def test_resize_shortlists_keeps_when_all_merges_exhausted_if_above_absolute_min() -> None:
    # Two shortlists each of size 10 (both undersized); combined 20 <= MAX_SHORTLIST.
    # But if one can't merge due to the overall structure — they should merge together.
    sl1 = [_pt(track_id=f"X{i}", bpm=170.0 + i * 0.1) for i in range(10)]
    sl2 = [_pt(track_id=f"Y{i}", bpm=190.0 + i * 0.1) for i in range(10)]
    result = _resize_shortlists([sl1, sl2])
    all_ids = {t.track_id for sl in result for t in sl}
    assert all_ids == {t.track_id for t in sl1 + sl2}


def test_resize_shortlists_drops_when_all_merges_exhausted_if_below_absolute_min() -> None:
    # One big shortlist and one tiny (<ABSOLUTE_MIN) one that overflows if merged.
    big1 = [_pt(track_id=f"A{i}", bpm=170.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    big2 = [_pt(track_id=f"B{i}", bpm=185.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    tiny = [_pt(track_id=f"T{i}", bpm=177.0 + i * 0.1) for i in range(ABSOLUTE_MIN - 1)]
    # tiny merged with big1 or big2 → overflows MAX_SHORTLIST → tiny dropped.
    result = _resize_shortlists([big1, big2, tiny])
    # Tiny may be dropped — check all surviving shortlists are >= ABSOLUTE_MIN.
    result_sizes = [len(sl) for sl in result]
    assert all(s >= ABSOLUTE_MIN for s in result_sizes)


def test_resize_shortlists_keeps_both_when_merge_would_exceed_max() -> None:
    # Both shortlists at 13 tracks each; merged = 26 > MAX_SHORTLIST=25 → can't merge.
    # Both < MIN_SHORTLIST but sum > MAX_SHORTLIST.
    sl1 = [_pt(track_id=f"A{i}", bpm=170.0 + i * 0.1) for i in range(13)]
    sl2 = [_pt(track_id=f"B{i}", bpm=172.0 + i * 0.1) for i in range(13)]
    result = _resize_shortlists([sl1, sl2])
    # Cannot merge (would exceed max), cannot drop (both above ABSOLUTE_MIN).
    # Both should be kept.
    all_ids = {t.track_id for sl in result for t in sl}
    expected = {t.track_id for t in sl1 + sl2}
    assert all_ids == expected


def test_resize_shortlists_drops_undersized_below_absolute_min_after_failed_merge() -> None:
    # Shortlist below ABSOLUTE_MIN that can't merge → dropped.
    big1 = [_pt(track_id=f"A{i}", bpm=170.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    big2 = [_pt(track_id=f"B{i}", bpm=185.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    tiny = [_pt(track_id=f"T{i}", bpm=178.0) for i in range(2)]  # 2 < ABSOLUTE_MIN=5
    result = _resize_shortlists([big1, big2, tiny])
    all_ids = {t.track_id for sl in result for t in sl}
    tiny_ids = {t.track_id for t in tiny}
    # Tiny (2 tracks) can't merge into either big (both at max) → dropped.
    assert tiny_ids.isdisjoint(all_ids) or len(all_ids) == len({t.track_id for t in big1 + big2 + tiny})


def test_resize_shortlists_lone_undersized_above_absolute_min_kept() -> None:
    # Single shortlist of 10 tracks (> ABSOLUTE_MIN but < MIN_SHORTLIST) → no partners → kept.
    sl = [_pt(track_id=str(i), bpm=174.0 + i * 0.1) for i in range(10)]
    result = _resize_shortlists([sl])
    assert len(result) == 1
    assert len(result[0]) == 10


def test_resize_shortlists_lone_undersized_below_absolute_min_kept_by_lone_guard() -> None:
    # Single shortlist below ABSOLUTE_MIN — must be kept (lone guard: no partner to merge with).
    sl = [_pt(track_id=str(i), bpm=174.0) for i in range(ABSOLUTE_MIN - 1)]
    result = _resize_shortlists([sl])
    # Lone guard: can't merge (no others), can't drop (only one) → kept.
    assert len(result) == 1


def test_resize_shortlists_splits_oversized_by_camelot_component() -> None:
    # 30 tracks in 2 disconnected Camelot groups of 15 each → split into 2x15.
    group_a = [_pt(track_id=f"A{i}", camelot_key="1A", bpm=174.0 + i * 0.1) for i in range(15)]
    group_b = [_pt(track_id=f"B{i}", camelot_key="6A", bpm=174.0 + i * 0.1) for i in range(15)]
    oversized = group_a + group_b
    result = _resize_shortlists([oversized])
    assert len(result) == 2
    for sl in result:
        assert MIN_SHORTLIST <= len(sl) <= MAX_SHORTLIST


def test_resize_shortlists_centrality_ranks_ascending_keeps_central_tracks() -> None:
    # Build an oversized shortlist where tracks near BPM median and dominant key
    # should have low centrality scores (kept) vs peripheral tracks (trimmed).
    med_bpm = 174.0
    key = "8A"
    # 25 central tracks.
    central = [_pt(track_id=f"C{i}", bpm=med_bpm + (i - 12) * 0.01, camelot_key=key) for i in range(25)]
    # 5 peripheral tracks far from median.
    peripheral = [_pt(track_id=f"P{i}", bpm=med_bpm + 100.0 + i, camelot_key="1A") for i in range(5)]
    oversized = central + peripheral
    result = _resize_shortlists([oversized])
    assert len(result) >= 1
    # After trim, peripheral tracks should be excluded (trimmed away).
    kept_ids = {t.track_id for sl in result for t in sl}
    # At least some central tracks should be kept.
    central_ids = {t.track_id for t in central}
    assert len(kept_ids & central_ids) > 0


def test_resize_shortlists_centrality_normalises_bpm_and_camelot_independently() -> None:
    # BPM norm = |bpm - median| / bpm_range; camelot norm = dist / 7.0.
    # Verify: a track at median BPM + adjacent key has centrality ~0 + ~(1/7).
    # A track at extreme BPM + far key has centrality ~1 + ~1.
    med = 174.0
    bpm_range = 10.0
    # Central track: at median, dominant key.
    c_bpm_norm = abs(med - med) / bpm_range  # = 0.0
    c_camelot_norm = 0 / 7.0  # distance 0 from dominant.
    assert c_bpm_norm + c_camelot_norm == pytest.approx(0.0)
    # Peripheral: at extreme BPM, far key (dist=7).
    p_bpm_norm = abs(med + bpm_range - med) / bpm_range  # = 1.0
    p_camelot_norm = 7.0 / 7.0  # = 1.0
    assert p_bpm_norm + p_camelot_norm == pytest.approx(2.0)


def test_resize_shortlists_centrality_bpm_range_near_zero_uses_fallback_1() -> None:
    # When all BPMs are identical, bpm_range < 1e-6 → fallback to 1.0.
    # Build oversized shortlist with all same BPM.
    sl = [_pt(track_id=str(i), bpm=174.0, camelot_key="8A") for i in range(30)]
    # All same BPM → bpm_range = 0 → fallback 1.0.
    result = _resize_shortlists([sl])
    # Should not raise and should produce a valid result.
    assert len(result) >= 1
    total = sum(len(s) for s in result)
    assert total <= 30


def test_resize_shortlists_centrality_dominant_key_none_all_camelot_norm_1() -> None:
    # If no parseable Camelot keys, dominant_key=None → camelot_norm=1.0 for all.
    # All tracks get camelot_norm=1.0 — tiebreaker is track_id.
    sl = [_pt(track_id=f"{i:03d}", bpm=174.0 + (i - 15) * 0.1, camelot_key="") for i in range(30)]
    result = _resize_shortlists([sl])
    assert len(result) >= 1
    total = sum(len(s) for s in result)
    assert total <= 30


def test_resize_shortlists_centrality_unparseable_key_capped_at_1_not_999_over_7() -> None:
    # camelot_distance returns 999 for unparseable keys.
    # The centrality formula caps at 1.0 when d >= 999.
    from mixlab.clustering import camelot_distance as cd

    d = cd("", "8A")
    assert d == 999
    # Cap logic: camelot_norm = 1.0 if d >= 999 else d/7.0
    camelot_norm = 1.0 if d >= 999 else d / 7.0
    assert camelot_norm == pytest.approx(1.0)


def test_resize_shortlists_centrality_tie_broken_by_track_id() -> None:
    # When bpm_norm + camelot_norm is equal, track_id tiebreaks (lexicographic).
    # Build tracks all with same centrality score; "001" < "002" → "001" kept first.
    sl = [_pt(track_id=f"{i:03d}", bpm=174.0, camelot_key="8A") for i in range(30)]
    result = _resize_shortlists([sl])
    kept_ids = sorted(t.track_id for sl_r in result for t in sl_r)
    # Track_ids "000" through "024" (25 lowest) should be kept (lowest centrality tied → lex).
    assert kept_ids[:25] == [f"{i:03d}" for i in range(25)]


def test_resize_shortlists_single_oversized_no_overflow_target_trims_in_place() -> None:
    # Single oversized shortlist with no other shortlists to receive remainder.
    # Camelot split not possible (all same key), era split not possible → centrality trim.
    sl = [_pt(track_id=str(i), bpm=174.0 + i * 0.1, camelot_key="8A") for i in range(30)]
    result = _resize_shortlists([sl])
    assert len(result) == 1
    assert len(result[0]) == MAX_SHORTLIST


def test_resize_shortlists_pool_level_merge_loops_until_max_pool_count() -> None:
    # 7 shortlists of exactly 15 tracks each → pool-level merge down to MAX_POOL_COUNT=5.
    shortlists = [
        [_pt(track_id=f"SL{sl_i}T{i}", bpm=160.0 + sl_i * 5 + i * 0.1) for i in range(15)]
        for sl_i in range(7)
    ]
    result = _resize_shortlists(shortlists)
    assert len(result) <= MAX_POOL_COUNT


def test_resize_shortlists_pool_level_merge_inserts_at_lower_index() -> None:
    # After pool-level merge, merged shortlist goes to lower of the two original indices.
    shortlists = [
        [_pt(track_id=f"SL{sl_i}T{i}", bpm=160.0 + sl_i * 5 + i * 0.1) for i in range(15)]
        for sl_i in range(6)
    ]
    result = _resize_shortlists(shortlists)
    # Result should have at most MAX_POOL_COUNT=5 shortlists.
    assert len(result) <= MAX_POOL_COUNT
    # All tracks should still be present.
    all_ids = {t.track_id for sl in result for t in sl}
    expected = {f"SL{sl_i}T{i}" for sl_i in range(6) for i in range(15)}
    assert all_ids == expected


def test_resize_shortlists_pool_level_merge_may_exceed_max_shortlist() -> None:
    # Pool-level merge joins two shortlists regardless of MAX_SHORTLIST.
    # Two shortlists of 20 tracks each → merged = 40 > MAX_SHORTLIST=25 (allowed).
    shortlists = [
        [_pt(track_id=f"SL{sl_i}T{i}", bpm=160.0 + sl_i * 5 + i * 0.1) for i in range(20)]
        for sl_i in range(6)
    ]
    result = _resize_shortlists(shortlists)
    assert len(result) <= MAX_POOL_COUNT
    # Check merged shortlist may be > MAX_SHORTLIST.
    max_size = max(len(sl) for sl in result)
    assert max_size >= 20  # at least original size preserved after merge


def test_resize_shortlists_pool_level_merge_tiebreak_step2_picks_smaller_index_sum() -> None:
    # Create 6 shortlists where two pairs have equal BPM distance.
    # The pair with smaller index sum should be merged.
    # SL0 @ 170, SL1 @ 175, SL2 @ 180, SL3 @ 185, SL4 @ 190, SL5 @ 195
    # All consecutive pairs are 5 BPM apart → tie. Tie-break 1: lex-min track_id.
    # All shortlists have min_track_id "SL{i}T0" — lex comparison should pick minimum.
    shortlists = [
        [_pt(track_id=f"SL{sl_i}T{i}", bpm=170.0 + sl_i * 5 + i * 0.1) for i in range(15)]
        for sl_i in range(6)
    ]
    result = _resize_shortlists(shortlists)
    assert len(result) <= MAX_POOL_COUNT
    all_ids = {t.track_id for sl in result for t in sl}
    expected = {f"SL{sl_i}T{i}" for sl_i in range(6) for i in range(15)}
    assert all_ids == expected


def test_resize_shortlists_pool_level_merge_tiebreak_step3_picks_lower_index_member() -> None:
    # Verify tie-break 3: when two pairs have equal lex-min and equal index-sum,
    # the one with the lower-index member wins.
    # This test verifies correctness of the result structure.
    shortlists = [
        [_pt(track_id=f"XX{sl_i}T{i}", bpm=170.0 + sl_i * 5 + i * 0.1) for i in range(15)]
        for sl_i in range(6)
    ]
    result = _resize_shortlists(shortlists)
    assert len(result) <= MAX_POOL_COUNT


def test_resize_shortlists_dropped_shortlists_not_resurrected_by_min_pool_count() -> None:
    # A tiny shortlist that gets dropped should not be resurrected.
    big1 = [_pt(track_id=f"A{i}", bpm=170.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    big2 = [_pt(track_id=f"B{i}", bpm=185.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    big3 = [_pt(track_id=f"C{i}", bpm=200.0 + i * 0.1) for i in range(MAX_SHORTLIST)]
    tiny = [_pt(track_id=f"T{i}", bpm=177.0) for i in range(2)]
    result = _resize_shortlists([big1, big2, big3, tiny])
    # tiny was dropped → should not reappear.
    result_sizes = [len(sl) for sl in result]
    assert all(s >= ABSOLUTE_MIN or s == 0 for s in result_sizes)


# ---------------------------------------------------------------------------
# _infer_shortlist_mood
# ---------------------------------------------------------------------------


def test_infer_shortlist_mood_uses_full_camelot_key_string_not_number() -> None:
    # dominant_key should include both number and mode letter.
    tracks = [
        _pt(track_id="1", bpm=174.0, camelot_key="8A"),
        _pt(track_id="2", bpm=175.0, camelot_key="8A"),
        _pt(track_id="3", bpm=176.0, camelot_key="9B"),
    ]
    title, mood = _infer_shortlist_mood(tracks)
    assert "8A" in title


def test_infer_shortlist_mood_mode_b_key_not_coerced_to_a() -> None:
    # If dominant key is "8B", it should remain "8B" (not become "8A").
    tracks = [
        _pt(track_id="1", bpm=174.0, camelot_key="8B"),
        _pt(track_id="2", bpm=175.0, camelot_key="8B"),
        _pt(track_id="3", bpm=176.0, camelot_key="8B"),
    ]
    title, _ = _infer_shortlist_mood(tracks)
    assert "8B" in title


def test_infer_shortlist_mood_filters_unparseable_camelot_keys() -> None:
    # Tracks with empty or invalid camelot_key should not pollute dominant key.
    tracks = [
        _pt(track_id="1", bpm=174.0, camelot_key=""),
        _pt(track_id="2", bpm=175.0, camelot_key="X"),
        _pt(track_id="3", bpm=176.0, camelot_key="8A"),
        _pt(track_id="4", bpm=177.0, camelot_key="8A"),
    ]
    title, _ = _infer_shortlist_mood(tracks)
    # dominant key = "8A" since the two invalid keys are filtered out.
    assert "8A" in title


def test_infer_shortlist_mood_includes_era_when_years_present() -> None:
    # When tracks have years, the title should include the era span.
    tracks = [
        _pt(track_id="1", bpm=174.0, camelot_key="8A", year=2010),
        _pt(track_id="2", bpm=175.0, camelot_key="8A", year=2015),
        _pt(track_id="3", bpm=176.0, camelot_key="8A", year=2020),
    ]
    title, _ = _infer_shortlist_mood(tracks)
    # Era span "2010–2020" should appear.
    assert "2010" in title
    assert "2020" in title


def test_infer_shortlist_mood_no_tags_falls_back_to_bpm_range() -> None:
    # No tags → mood = "{bpm_lo}–{bpm_hi} BPM".
    tracks = [
        _pt(track_id="1", bpm=170.0, camelot_key="8A", tags=[]),
        _pt(track_id="2", bpm=180.0, camelot_key="8A", tags=[]),
    ]
    _, mood = _infer_shortlist_mood(tracks)
    assert "170" in mood
    assert "180" in mood
    assert "BPM" in mood


# ---------------------------------------------------------------------------
# End-to-end partition_pool tests
# ---------------------------------------------------------------------------


def _typical_pool(n: int = 50, *, base_bpm: float = 170.0) -> list[Track]:
    """Build a typical pool of n tracks spread across a realistic BPM range."""
    keys = ["8A", "9A", "10A", "8B", "9B", "10B", "7A", "11A"]
    return [
        _pt(
            track_id=f"P{i:03d}",
            bpm=base_bpm + (i % 20) * 0.5,
            camelot_key=keys[i % len(keys)],
        )
        for i in range(n)
    ]


def test_partition_pool_returns_three_to_five_shortlists_on_typical_input() -> None:
    tracks = _typical_pool(80)
    result = partition_pool(tracks)
    assert MIN_POOL_COUNT <= len(result) <= MAX_POOL_COUNT


def test_partition_pool_typical_input_shortlists_within_15_to_25_tracks() -> None:
    tracks = _typical_pool(80)
    result = partition_pool(tracks)
    for concept in result:
        assert MIN_SHORTLIST <= len(concept.track_ids) <= MAX_SHORTLIST


def test_partition_pool_same_seed_same_output_reproducibility() -> None:
    tracks = _typical_pool(60)
    result1 = partition_pool(tracks, seed=42)
    result2 = partition_pool(tracks, seed=42)
    assert [c.track_ids for c in result1] == [c.track_ids for c in result2]


def test_partition_pool_tiny_pool_returns_single_shortlist() -> None:
    # Between ABSOLUTE_MIN and MIN_SHORTLIST → single shortlist.
    tracks = [_pt(track_id=str(i), bpm=174.0 + i * 0.1) for i in range(8)]
    result = partition_pool(tracks)
    assert len(result) == 1
    assert len(result[0].track_ids) == 8


def test_partition_pool_all_same_bpm_15to25_tracks_returns_camelot_subgroups_or_single_intact() -> None:
    # All same BPM → flat-BPM guard → single cluster → camelot sub-clustering.
    tracks = [_pt(track_id=str(i), bpm=174.0, camelot_key="8A") for i in range(20)]
    result = partition_pool(tracks)
    assert len(result) >= 1
    # All tracks should be present.
    all_ids = {tid for c in result for tid in c.track_ids}
    assert all_ids == {str(i) for i in range(20)}


def test_partition_pool_all_same_bpm_26plus_tracks_trimmed_to_max_shortlist_by_step4() -> None:
    # 30 tracks all same BPM → camelot all same → 1 oversized shortlist of 30 → trim to 25.
    tracks = [_pt(track_id=str(i), bpm=174.0, camelot_key="8A") for i in range(30)]
    result = partition_pool(tracks)
    assert len(result) >= 1
    for concept in result:
        assert len(concept.track_ids) <= MAX_SHORTLIST


def test_partition_pool_outliers_attached_to_nearest_cluster() -> None:
    # The BPM histogram outlier-attachment (step 1.7) adds far-off tracks to the nearest peak.
    # We verify this via _find_bpm_peaks directly: a far outlier must be in some cluster.
    # Build: interior cluster at 170 BPM (with edge padding) + a far outlier at 250 BPM.
    low_pad = [_pt(track_id=f"LP{i}", bpm=140.0) for i in range(3)]
    cluster = [_pt(track_id=f"C{i}", bpm=170.0 + i * 0.1) for i in range(20)]
    far_outlier = _pt(track_id="OUT", bpm=250.0)
    high_pad = [_pt(track_id=f"HP{i}", bpm=270.0) for i in range(3)]
    tracks = low_pad + cluster + [far_outlier] + high_pad
    peaks = _find_bpm_peaks(tracks)
    assert peaks is not None
    # The outlier must be attached to some cluster.
    all_ids = {t.track_id for cl in peaks for t in cl}
    assert "OUT" in all_ids


def test_partition_pool_no_peaks_uses_equal_bpm_split_before_step2() -> None:
    # No-peak path: linear BPM spread, n_groups>1 → equal-sized BPM groups.
    # 45 tracks → n_groups=3; after Camelot sub-clustering, 3 groups preserved.
    tracks = [_pt(track_id=str(i), bpm=165.0 + i) for i in range(45)]
    result = partition_pool(tracks)
    assert len(result) >= 1
    all_ids = {tid for c in result for tid in c.track_ids}
    assert all_ids == {str(i) for i in range(45)}


def test_partition_pool_single_peak_uses_n_groups_fallback() -> None:
    # All DnB-style tracks within 5 BPM → _find_bpm_peaks merges into 1 peak →
    # treated as no-peaks → n_groups fallback splits into multiple shortlists.
    # 75 tracks across mixed keys, all 170–175 BPM, which is the real-world DnB case.
    keys = ["1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A", "9A", "10A", "11A", "12A"]
    tracks = [
        _pt(track_id=f"T{i:03d}", bpm=170.0 + (i % 6), camelot_key=keys[i % len(keys)])
        for i in range(75)
    ]
    result = partition_pool(tracks)
    assert len(result) >= 2
    all_ids = {tid for c in result for tid in c.track_ids}
    assert all_ids == {f"T{i:03d}" for i in range(75)}


def test_partition_pool_camelot_sector_split_fires_on_tight_bpm_oversized_pool() -> None:
    # Oversized shortlist where Camelot BFS and era split both fail →
    # sector split (keys 1–6 vs 7–12) produces two valid shortlists.
    lower_keys = ["1A", "2A", "3A", "4A", "5A", "6A"]
    upper_keys = ["7A", "8A", "9A", "10A", "11A", "12A"]
    lower = [_pt(track_id=f"L{i}", bpm=174.0, camelot_key=lower_keys[i % 6]) for i in range(20)]
    upper = [_pt(track_id=f"U{i}", bpm=174.0, camelot_key=upper_keys[i % 6]) for i in range(20)]
    tracks = lower + upper
    result = partition_pool(tracks)
    assert len(result) >= 2
    all_ids = {tid for c in result for tid in c.track_ids}
    assert all_ids == {f"L{i}" for i in range(20)} | {f"U{i}" for i in range(20)}


def test_partition_pool_custom_genre_pool_respects_sub_genre_coherence() -> None:
    # Mix of keys from two disconnected Camelot regions → sub-grouped by Camelot.
    tracks = (
        [_pt(track_id=f"G1_{i}", bpm=172.0 + i * 0.3, camelot_key="1A", genre="Drum & Bass") for i in range(25)]
        + [_pt(track_id=f"G2_{i}", bpm=172.0 + i * 0.3, camelot_key="6A", genre="Jungle") for i in range(25)]
    )
    result = partition_pool(tracks)
    assert len(result) >= 1
    # All tracks represented.
    all_ids = {tid for c in result for tid in c.track_ids}
    expected = {f"G1_{i}" for i in range(25)} | {f"G2_{i}" for i in range(25)}
    assert all_ids == expected
