from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest
from conftest import conj_pool

from mixlab.directions import (
    MAX_DIRECTION_POOL,
    MIN_DIRECTION_POOL,
    Direction,
    DirectionSpecError,
    TrackPoolError,
    _build_artist_thread,
    _build_energy_shape_first,
    _build_era_dialogue,
    _build_fresh_crate,
    _build_genre_traverse,
    _build_label_spotlight,
    _build_mood_journey,
    _freshness,
    _jaccard,
    _log_lift,
    _path_feasible,
    _score_field,
    _score_final,
    _shape_field,
    enumerate_directions,
    generate_directions,
    parse_track_pool,
    pinned_canvas_from_spec,
)
from mixlab.models import Track


def _track(
    *,
    track_id: str,
    bpm: float = 172.0,
    camelot_key: str = "8A",
    genre: str = "Drum & Bass",
    energy: int | None = None,
    label: str = "",
    artist: str = "",
    remixer: str = "",
    year: int | None = None,
    tags: list[str] | None = None,
    date_added: str = "",
    rating: int | None = None,
    play_count: int = 0,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist or f"Artist_{track_id}",
        title=f"Title_{track_id}",
        bpm=bpm,
        camelot_key=camelot_key,
        genre=genre,
        energy=energy,
        label=label,
        remixer=remixer,
        year=year,
        tags=tags or [],
        date_added=date_added,
        rating=rating,
        play_count=play_count,
    )


def _tbi(pool: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in pool}


# BPMs whose BPM-sorted neighbours never fit a straight/halftime/double/3:4/4:3 ratio,
# so a candidate pool built from them fails the 80% path-feasibility gate.
_INCOMPATIBLE_BPMS = [90.0, 103.0, 118.0, 135.0, 155.0, 178.0, 204.0, 234.0]


def _rich_pool() -> list[Track]:
    """A pool engineered to make five direction types viable simultaneously.

    - tracks 0-5 tagged 'dark', 6-11 tagged 'euphoric' (mood_journey)
    - years split 2001-2003 vs 2016-2018 with a 13-year gap (era_dialogue)
    - tracks 12-21 on label 'Hospital' (label_spotlight)
    - tracks 30-32 by artist 'SpineGuy' (artist_thread, exactly 3)
    - alternating energy 2/7 across all 40 tracks (energy_shape_first)
    All BPMs sit in 170-174 / key 8A so every path is feasible. fresh_crate is
    intentionally NOT viable (no date_added).
    """
    pool: list[Track] = []
    for i in range(40):
        tags: list[str] = []
        if i < 6:
            tags = ["dark"]
        elif i < 12:
            tags = ["euphoric"]
        label = "Hospital" if 12 <= i <= 21 else ""
        artist = "SpineGuy" if 30 <= i <= 32 else f"Artist_{i}"
        year = (2001 + (i % 3)) if i < 20 else (2016 + (i % 3))
        pool.append(
            _track(
                track_id=str(i),
                bpm=170.0 + (i % 3) * 2,
                energy=2 if i % 2 == 0 else 7,
                label=label,
                artist=artist,
                year=year,
                tags=tags,
            )
        )
    return pool


# ---------------------------------------------------------------------------
# _path_feasible
# ---------------------------------------------------------------------------


def test_path_feasible_straight_cluster_is_feasible() -> None:
    tracks = [_track(track_id=str(i), bpm=172.0) for i in range(10)]
    ok, ratio = _path_feasible(tracks)
    assert ok
    assert ratio == 1.0


def test_path_feasible_spread_bpms_with_no_ratio_fit_is_infeasible() -> None:
    tracks = [_track(track_id=str(i), bpm=bpm) for i, bpm in enumerate(_INCOMPATIBLE_BPMS)]
    ok, ratio = _path_feasible(tracks)
    assert not ok
    assert ratio < 0.8


# ---------------------------------------------------------------------------
# mood_journey
# ---------------------------------------------------------------------------


def test_build_mood_journey_with_polar_tags_proposes() -> None:
    pool = _rich_pool()
    direction = _build_mood_journey(pool, seed=1)
    assert direction is not None
    assert direction.direction_type == "mood_journey"
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_mood_journey_without_polar_tags_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=172.0, tags=["driving"]) for i in range(30)]
    assert _build_mood_journey(pool, seed=1) is None


# ---------------------------------------------------------------------------
# era_dialogue
# ---------------------------------------------------------------------------


def test_build_era_dialogue_with_year_gap_proposes() -> None:
    pool = _rich_pool()
    direction = _build_era_dialogue(pool, seed=1)
    assert direction is not None
    assert direction.direction_type == "era_dialogue"
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_era_dialogue_uniform_years_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=172.0, year=2010) for i in range(30)]
    assert _build_era_dialogue(pool, seed=1) is None


# ---------------------------------------------------------------------------
# label_spotlight
# ---------------------------------------------------------------------------


def test_build_label_spotlight_dominant_label_proposes() -> None:
    pool = _rich_pool()
    direction = _build_label_spotlight(pool, seed=1)
    assert direction is not None
    assert direction.direction_type == "label_spotlight"
    assert "Hospital" in direction.title
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_label_spotlight_no_dominant_label_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=172.0, label=f"L{i}") for i in range(30)]
    assert _build_label_spotlight(pool, seed=1) is None


def test_build_label_spotlight_rejects_infeasible_bpm_path() -> None:
    # 16 same-label tracks — clears the label count gate, but the spread BPMs make the
    # BPM-sorted path infeasible, so the builder must return None.
    bad = [
        _track(track_id=str(i), bpm=_INCOMPATIBLE_BPMS[i % len(_INCOMPATIBLE_BPMS)] + i, label="X") for i in range(16)
    ]
    assert _build_label_spotlight(bad, seed=1) is None
    # Same label count, compatible BPMs → now it proposes. Isolates the path check.
    good = [_track(track_id=str(i), bpm=172.0, label="X") for i in range(16)]
    assert _build_label_spotlight(good, seed=1) is not None


# ---------------------------------------------------------------------------
# artist_thread
# ---------------------------------------------------------------------------


def test_build_artist_thread_two_or_three_tracks_proposes_and_sets_thread_artist() -> None:
    pool = _rich_pool()
    direction = _build_artist_thread(pool, seed=1)
    assert direction is not None
    assert direction.direction_type == "artist_thread"
    assert direction.thread_artist == "SpineGuy"
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_artist_thread_four_track_artist_is_not_a_thread() -> None:
    # An artist with 4 tracks is a compilation, not a thread — no qualifying name.
    pool: list[Track] = []
    for i in range(30):
        artist = "Prolific" if i < 4 else f"Artist_{i}"
        pool.append(_track(track_id=str(i), bpm=172.0, artist=artist))
    assert _build_artist_thread(pool, seed=1) is None


def test_build_artist_thread_exactly_two_tracks_proposes() -> None:
    pool: list[Track] = []
    for i in range(30):
        artist = "Duo" if i < 2 else f"Artist_{i}"
        pool.append(_track(track_id=str(i), bpm=172.0, artist=artist))
    direction = _build_artist_thread(pool, seed=1)
    assert direction is not None
    assert direction.thread_artist == "Duo"


# ---------------------------------------------------------------------------
# energy_shape_first
# ---------------------------------------------------------------------------


def test_build_energy_shape_first_with_inventory_proposes() -> None:
    pool = _rich_pool()
    direction = _build_energy_shape_first(pool, seed=1)
    assert direction is not None
    assert direction.direction_type == "energy_shape_first"
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_energy_shape_first_all_high_energy_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=172.0, energy=7) for i in range(30)]
    assert _build_energy_shape_first(pool, seed=1) is None


def test_build_energy_shape_first_no_energy_data_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=172.0) for i in range(30)]
    assert _build_energy_shape_first(pool, seed=1) is None


# ---------------------------------------------------------------------------
# fresh_crate
# ---------------------------------------------------------------------------


def _dated_pool(n: int, *, anchors: int) -> list[Track]:
    """``n`` tracks, one per minute so track index == date-added rank.

    The first ``anchors`` tracks carry a play count, which is what makes them
    eligible as fresh_crate's grounding anchors (everything else is unplayed).
    """
    return [
        _track(track_id=str(i), bpm=172.0, date_added=f"2020-01-01T00:{i:02d}:00", play_count=1 if i < anchors else 0)
        for i in range(n)
    ]


def test_build_fresh_crate_with_date_density_proposes() -> None:
    direction = _build_fresh_crate(_dated_pool(55, anchors=3), seed=1)
    assert direction is not None
    assert direction.direction_type == "fresh_crate"
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_fresh_crate_without_date_added_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=172.0) for i in range(30)]
    assert _build_fresh_crate(pool, seed=1) is None


# ---------------------------------------------------------------------------
# generate_directions — determinism, rotation, materialisation
# ---------------------------------------------------------------------------


def test_generate_directions_same_seed_is_deterministic() -> None:
    pool = _rich_pool()
    tbi = _tbi(pool)
    first = generate_directions(pool, tbi, seed=42, max_directions=3)
    second = generate_directions(pool, tbi, seed=42, max_directions=3)
    assert [c.canvas_id for c in first] == [c.canvas_id for c in second]
    assert [c.direction_type for c in first] == [c.direction_type for c in second]
    assert [c.source_concept.track_ids for c in first] == [c.source_concept.track_ids for c in second]


def test_generate_directions_rotation_differs_across_seeds() -> None:
    pool = _rich_pool()
    tbi = _tbi(pool)
    sequences = {
        tuple(c.direction_type for c in generate_directions(pool, tbi, seed=s, max_directions=3)) for s in range(25)
    }
    # A pool supporting five direction types must expose more than one rotation.
    assert len(sequences) > 1


def test_generate_directions_materialises_canvas_with_direction_metadata() -> None:
    pool = _rich_pool()
    tbi = _tbi(pool)
    canvases = generate_directions(pool, tbi, seed=7, max_directions=6)
    assert canvases
    for canvas in canvases:
        assert canvas.brief != ""
        assert canvas.direction_type != ""
        # build_mix_canvas populated the structural fields.
        assert canvas.core_track_ids
        assert canvas.dominant_bpm > 0
        assert MIN_DIRECTION_POOL <= len(canvas.source_concept.track_ids) <= MAX_DIRECTION_POOL


def test_generate_directions_artist_thread_canvas_carries_thread_artist() -> None:
    pool = _rich_pool()
    tbi = _tbi(pool)
    canvases = generate_directions(pool, tbi, seed=7, max_directions=6)
    threads = [c for c in canvases if c.direction_type == "artist_thread"]
    assert threads
    assert threads[0].thread_artist == "SpineGuy"


def test_generate_directions_empty_when_nothing_viable() -> None:
    pool = [_track(track_id=str(i), bpm=172.0) for i in range(30)]
    assert generate_directions(pool, _tbi(pool), seed=1) == []


# ---------------------------------------------------------------------------
# _build_genre_traverse (#82) — cross-genre journeys via ratio bridges
# ---------------------------------------------------------------------------


def _traverse_pool() -> list[Track]:
    """Two tempo regimes bridged by a valid 4:3 relation: house ~126-128, DnB ~168-172."""
    house = [_track(track_id=f"h{i}", bpm=126.0 + i * 0.25, genre="House", camelot_key="8A") for i in range(8)]
    dnb = [_track(track_id=f"d{i}", bpm=168.0 + i * 0.5, genre="Drum & Bass", camelot_key="8A") for i in range(8)]
    return house + dnb


def test_build_genre_traverse_two_bridged_regimes_fires() -> None:
    direction = _build_genre_traverse(_traverse_pool(), seed=7)
    assert direction is not None
    assert direction.direction_type == "genre_traverse"
    assert "GENRE TRAVERSE" in direction.brief
    assert "hop 1" in direction.brief
    assert "via " in direction.brief
    assert "House" in direction.brief and "Drum & Bass" in direction.brief
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_build_genre_traverse_single_regime_returns_none() -> None:
    pool = [_track(track_id=str(i), bpm=122.0 + i, genre="House") for i in range(20)]
    assert _build_genre_traverse(pool, seed=7) is None


def test_build_genre_traverse_unbridgeable_regimes_returns_none() -> None:
    # 124 vs 146: gap splits regimes, but 146/124 = 1.177 matches no pitch-locked ratio.
    a = [_track(track_id=f"a{i}", bpm=124.0, genre="House") for i in range(8)]
    b = [_track(track_id=f"b{i}", bpm=146.0, genre="Breakbeat") for i in range(8)]
    assert _build_genre_traverse(a + b, seed=7) is None


def test_build_genre_traverse_same_seed_deterministic() -> None:
    assert _build_genre_traverse(_traverse_pool(), seed=42) == _build_genre_traverse(_traverse_pool(), seed=42)


def test_build_genre_traverse_bridge_endpoints_included_in_pool() -> None:
    direction = _build_genre_traverse(_traverse_pool(), seed=7)
    assert direction is not None
    # Every track named in a bridge line must be in the direction pool.
    named_ids = set()
    for m in re.finditer(r"Artist_(\w+) — Title_", direction.brief):
        named_ids.add(m.group(1))
    assert named_ids, "brief names no bridge tracks"
    assert named_ids <= {tid for tid in direction.track_ids}


def test_generate_directions_traverse_pool_materialises_traverse_canvas() -> None:
    pool = _traverse_pool()
    canvases = generate_directions(pool, _tbi(pool), seed=7, max_directions=6)
    types = {c.direction_type for c in canvases}
    assert "genre_traverse" in types


def test_build_genre_traverse_unbridgeable_lowest_regime_does_not_poison_chain() -> None:
    """Live finding (three consecutive production non-firings): the chain was anchored
    to the lowest regime, so a ~77 BPM block with no ratio partner killed the whole
    direction even when 126↔168 4:3 bridges were plentiful above it."""
    # 105-107.5 bridges nowhere: x2 -> 210+ (no tracks), 4:3 -> 140-143 (no tracks),
    # and 126/105 = 1.2 sits outside the 4:3 pitch window (stretch 0.90 < 0.94).
    poison = [_track(track_id=f"p{i}", bpm=105.0 + i * 0.5, genre="Hip Hop", camelot_key="8A") for i in range(6)]
    house = [_track(track_id=f"h{i}", bpm=126.0 + i * 0.25, genre="House", camelot_key="8A") for i in range(8)]
    dnb = [_track(track_id=f"d{i}", bpm=168.0 + i * 0.5, genre="Drum & Bass", camelot_key="8A") for i in range(8)]
    # Make the fixture's intent explicit: the poison block must bridge to nothing.
    from mixlab.transitions import tempo_relation

    for a in poison:
        for b in house + dnb:
            assert tempo_relation(a.bpm, b.bpm)[0] in ("incompatible", "straight")
    direction = _build_genre_traverse(poison + house + dnb, seed=7)
    assert direction is not None
    assert "House" in direction.brief and "Drum & Bass" in direction.brief
    assert "Hip Hop" not in direction.brief  # unreachable regime excluded, not fatal
    # No poison-block tracks in the journey pool.
    poison_ids = {t.track_id for t in poison}
    assert not (set(direction.track_ids) & poison_ids)


def test_build_genre_traverse_dense_continuum_pool_fires_on_density_peaks() -> None:
    """Live finding (v1.8.3 still not firing in production): a full collection's BPMs
    form a near-continuum with no >12 BPM holes, so gap-based regime splitting saw one
    giant regime. Density-peak regimes must fire on exactly that shape: a continuum
    plus heavy house/DnB concentrations."""
    continuum = [_track(track_id=f"c{b}", bpm=float(b), genre="Electronica", camelot_key="8A") for b in range(77, 181)]
    house = [_track(track_id=f"h{i}", bpm=126.0, genre="House", camelot_key="8A") for i in range(10)]
    dnb = [_track(track_id=f"d{i}", bpm=172.0, genre="Drum & Bass", camelot_key="8A") for i in range(10)]

    direction = _build_genre_traverse(continuum + house + dnb, seed=7)
    assert direction is not None
    assert direction.direction_type == "genre_traverse"
    assert "GENRE TRAVERSE" in direction.brief
    assert "hop 1" in direction.brief
    assert MIN_DIRECTION_POOL <= len(direction.track_ids) <= MAX_DIRECTION_POOL


def test_tempo_regimes_dense_pool_peaks_are_disjoint_and_ordered() -> None:
    from mixlab.directions import _tempo_regimes

    continuum = [_track(track_id=f"c{b}", bpm=float(b), genre="Electronica", camelot_key="8A") for b in range(77, 181)]
    house = [_track(track_id=f"h{i}", bpm=126.0, genre="House", camelot_key="8A") for i in range(10)]
    dnb = [_track(track_id=f"d{i}", bpm=172.0, genre="Drum & Bass", camelot_key="8A") for i in range(10)]
    regimes = _tempo_regimes(sorted(continuum + house + dnb, key=lambda t: (t.bpm, t.track_id)))

    assert len(regimes) >= 2
    peaks_order = [min(t.bpm for t in r) for r in regimes]
    assert peaks_order == sorted(peaks_order)  # ascending
    seen: set[str] = set()
    for regime in regimes:
        ids = {t.track_id for t in regime}
        assert not (ids & seen)  # disjoint
        seen |= ids
    # The two heavy concentrations must each anchor a regime.
    assert any(any(t.track_id == "h0" for t in r) for r in regimes)
    assert any(any(t.track_id == "d0" for t in r) for r in regimes)


# ---------------------------------------------------------------------------
# enumerate_directions — exhaustive candidate field for the library map (#40)
# ---------------------------------------------------------------------------


def test_enumerate_directions_returns_sorted_candidates() -> None:
    pool = _rich_pool()
    result = enumerate_directions(pool, seed=0)
    assert result, "builders should propose at least one direction for the rich fixture pool"
    feasibilities = [d.feasibility for d in result]
    assert feasibilities == sorted(feasibilities, reverse=True) or [
        (-d.feasibility, d.direction_type) for d in result
    ] == sorted((-d.feasibility, d.direction_type) for d in result)
    assert all(isinstance(d, Direction) for d in result)


def test_enumerate_directions_same_seed_identical_output() -> None:
    pool = _rich_pool()
    first = enumerate_directions(pool, seed=7)
    second = enumerate_directions(pool, seed=7)
    assert [(d.direction_type, d.title, d.track_ids) for d in first] == [
        (d.direction_type, d.title, d.track_ids) for d in second
    ]


def test_enumerate_directions_empty_pool_returns_empty(capsys: pytest.CaptureFixture[str]) -> None:
    assert enumerate_directions([], seed=0) == []
    assert capsys.readouterr().out == ""  # never prints, unlike generate_directions


# ---------------------------------------------------------------------------
# _freshness / _log_lift / per-builder identity renormalisation (Task 2)
# ---------------------------------------------------------------------------


def _t(
    track_id: str,
    *,
    date_added: str = "",
    label: str = "",
    year: int | None = None,
    tags: list[str] | None = None,
    energy: int | None = None,
    bpm: float = 174.0,
    key: str = "8A",
) -> Track:
    return Track(
        track_id=track_id,
        artist=f"A{track_id}",
        title=f"T{track_id}",
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        label=label,
        year=year,
        tags=tags or [],
        energy=energy,
        date_added=date_added,
    )


class TestFreshness:
    def test_newest_half_scores_above_oldest_half(self) -> None:
        pool = [_t(f"o{i}", date_added=f"2020-01-{i + 1:02d}") for i in range(10)] + [
            _t(f"n{i}", date_added=f"2026-06-{i + 1:02d}") for i in range(10)
        ]
        newest = [t for t in pool if t.track_id.startswith("n")]
        oldest = [t for t in pool if t.track_id.startswith("o")]
        assert _freshness(newest, pool) > 0.7
        assert _freshness(oldest, pool) < 0.3

    def test_missing_date_added_sorts_oldest(self) -> None:
        pool = [_t("u1"), _t("u2")] + [_t(f"d{i}", date_added=f"2026-01-{i + 1:02d}") for i in range(8)]
        assert _freshness([pool[0], pool[1]], pool) < 0.2

    def test_all_unplayed_pool_does_not_saturate(self) -> None:
        # freshness is date-rank based, so an "all-unplayed" pool still spreads
        pool = [_t(f"x{i}", date_added=f"20{20 + i // 5}-01-01") for i in range(20)]
        vals = {_freshness([t], pool) for t in pool}
        assert len(vals) > 1


class TestLogLift:
    def test_anchor_points(self) -> None:
        assert _log_lift(1.0) == 0.0
        assert abs(_log_lift(2.0) - 1 / 3) < 1e-9
        assert abs(_log_lift(8.0) - 1.0) < 1e-9
        assert _log_lift(10.4) == 1.0  # live max saturates at the cap, not below it
        assert _log_lift(0.5) == 0.0  # sub-chance clamps to 0, never negative


class TestIdentityRenormalisation:
    def test_mood_journey_balance_uses_untruncated_pole_counts(self) -> None:
        # 40 dark vs 8 euphoric: old code truncated both to [:10] first → balance 0.8.
        # New: _balance(40, 8) = 0.2.
        pool = (
            [_t(f"d{i}", tags=["dark"], year=2020) for i in range(40)]
            + [_t(f"e{i}", tags=["euphoric"], year=2020) for i in range(8)]
            + [_t(f"b{i}", year=2020) for i in range(10)]
        )
        d = _build_mood_journey(pool, seed=1)
        assert d is not None
        # signal lands in feasibility via the Task 3 scorer; here assert the builder's
        # stored signal directly (Task 2 exposes it — see Step 2)
        assert abs(d.identity - 0.2) < 1e-9

    def test_label_spotlight_collection_lift(self) -> None:
        pool = [_t(f"l{i}", label="Metalheadz") for i in range(10)] + [_t(f"p{i}") for i in range(10)]
        collection = pool + [_t(f"c{i}") for i in range(80)]
        d = _build_label_spotlight(pool, seed=1, collection=collection)
        assert d is not None
        # share_in_pool 0.5, share_in_collection 0.1 → ratio 5 → log2(5)/3 ≈ 0.7737,
        # stored rounded to 4dp like freshness.
        assert d.identity == round(math.log2(5) / 3, 4) == 0.774

    def test_label_spotlight_without_collection_falls_back_to_share(self) -> None:
        pool = [_t(f"l{i}", label="Metalheadz") for i in range(10)] + [_t(f"p{i}") for i in range(10)]
        d = _build_label_spotlight(pool, seed=1)
        assert d is not None
        assert abs(d.identity - 0.5) < 1e-9

    def test_fresh_crate_identity_is_recency_concentration_not_dated_share(self) -> None:
        """The old signal was ``len(dated)/len(pool)``, which pins at 1.0 whenever the
        pool carries dates at all — and Rekordbox stamps DateAdded on ~every track, so
        it was 1.0 in production, always. The new signal rescales the shipped set's
        median date-added percentile: ``max(0, 2*(freshness - 0.5))``.

        55 dated tracks, one per minute, so track index == date rank and percentile
        of index i is i/54. Three anchors (indices 0-2, the only played tracks).
        fresh_crate ships ``dated_sorted[-20:]`` = indices 35-54 plus those anchors,
        23 tracks whose sorted percentiles are 0,1,2,35..54 → median is index 43.
        """
        d = _build_fresh_crate(_dated_pool(55, anchors=3), seed=1)
        assert d is not None
        assert d.freshness == round(43 / 54, 4) == 0.7963
        assert d.identity == round(2 * (43 / 54 - 0.5), 4) == 0.5926
        assert d.identity < 1.0  # the whole point: the old formula returned 55/55

    def test_fresh_crate_identity_spreads_across_equally_dated_pools(self) -> None:
        """Same size, same full date coverage, different recency concentration.

        With no played tracks there are no anchors, so the shipped set is just the
        newest 20 (indices 35-54) and its median percentile is the mean of indices
        44 and 45 = 44.5/54. The dated-share formula scored both pools 1.0.
        """
        d = _build_fresh_crate(_dated_pool(55, anchors=0), seed=1)
        assert d is not None
        assert len(d.track_ids) == 20
        assert d.freshness == round(44.5 / 54, 4) == 0.8241
        assert d.identity == round(2 * (44.5 / 54 - 0.5), 4) == 0.6481
        # Anchoring on older material genuinely lowers identity — spread, not a pin.
        anchored = _build_fresh_crate(_dated_pool(55, anchors=3), seed=1)
        assert anchored is not None
        assert anchored.identity < d.identity


# ---------------------------------------------------------------------------
# _score_field — two-pass scorer: gates in the builder, ranking in the field (Task 3)
# ---------------------------------------------------------------------------


def _cand(dtype: str, ids: list[str], *, identity: float, freshness: float) -> Direction:
    return Direction(
        direction_type=dtype,
        title=dtype,
        mood="m",
        track_ids=ids,
        brief="b",
        feasibility=0.0,
        identity=identity,
        freshness=freshness,
    )


class TestScoreField:
    def test_spread_where_old_formula_saturated(self) -> None:
        # Three full-size, path-feasible candidates that all scored ~1.0 before
        a = _cand("fresh_crate", [f"a{i}" for i in range(25)], identity=1.0, freshness=0.97)
        b = _cand("label_spotlight", [f"b{i}" for i in range(25)], identity=0.3, freshness=0.4)
        c = _cand("mood_journey", [f"c{i}" for i in range(25)], identity=0.6, freshness=0.5)
        scores = {d.direction_type: d.feasibility for d in _score_field([a, b, c])}
        assert max(scores.values()) - min(scores.values()) > 0.15

    def test_dedupe_drops_lower_ranked_clone(self) -> None:
        shared = [f"s{i}" for i in range(20)]
        hi = _cand("era_dialogue", shared + ["h1", "h2"], identity=0.9, freshness=0.5)
        lo = _cand("found_1", shared + ["l1", "l2"], identity=0.2, freshness=0.5)
        out = _score_field([hi, lo])
        assert [d.direction_type for d in out] == ["era_dialogue"]

    def test_intermediate_overlap_scales_distinctiveness_proportionally(self) -> None:
        """Distinctiveness is a continuum, not clone-or-disjoint.

        A and B share 11 of a 20-track union → Jaccard exactly 0.55, under the 0.6
        dedupe threshold, so both reach pass 2; C is disjoint from both. With every
        row on identity 0.4 / freshness 0.4 the shared base is
        ``0.25*0.4 + 0.45*0.4 = 0.28``, leaving only the distinctiveness term to
        separate them: ``0.30*(1-0.55)`` for the overlapping pair against
        ``0.30*1.0`` for the disjoint row.
        """
        shared = [f"s{i}" for i in range(11)]
        a = _cand("era_dialogue", shared + [f"a{i}" for i in range(5)], identity=0.4, freshness=0.4)
        b = _cand("label_spotlight", shared + [f"b{i}" for i in range(4)], identity=0.4, freshness=0.4)
        c = _cand("artist_thread", [f"c{i}" for i in range(15)], identity=0.4, freshness=0.4)
        assert _jaccard(a.track_ids, b.track_ids) == 0.55

        scored = _score_final([a, b, c])
        assert {d.direction_type: d.feasibility for d in scored} == {
            "artist_thread": 0.58,  # 0.28 + 0.30 * 1.00
            "era_dialogue": 0.415,  # 0.28 + 0.30 * 0.45
            "label_spotlight": 0.415,
        }
        # 0.55 is below the dedupe threshold, so the full pipeline keeps all three
        # and reaches the same scores rather than dropping the overlapping pair.
        assert _score_field([a, b, c]) == scored

    def test_lone_candidate_distinctiveness_is_half(self) -> None:
        a = _cand("artist_thread", [f"a{i}" for i in range(25)], identity=0.4, freshness=0.4)
        [scored] = _score_field([a])
        # 0.25*0.4 + 0.45*0.4 + 0.30*0.5 = 0.43
        assert abs(scored.feasibility - 0.43) < 1e-9

    def test_deterministic_under_input_order(self) -> None:
        a = _cand("x1", [f"a{i}" for i in range(25)], identity=0.5, freshness=0.5)
        b = _cand("x2", [f"b{i}" for i in range(25)], identity=0.5, freshness=0.4)
        assert _score_field([a, b]) == _score_field([b, a])

    def test_deterministic_when_candidates_tie_on_type_title_and_pool(self) -> None:
        """Mined rows share direction_type ("found") far more readily than named ones,
        and two pairs over the same members can share a title too. Without mood/brief
        in the rank key the survivor of the dedupe would depend on input order."""
        ids = [f"s{i}" for i in range(20)]
        a = replace(_cand("found", ids, identity=0.5, freshness=0.5), title="Found: liquid", mood="m1", brief="b1")
        b = replace(_cand("found", ids, identity=0.5, freshness=0.5), title="Found: liquid", mood="m2", brief="b2")
        forward, reverse = _score_field([a, b]), _score_field([b, a])
        assert len(forward) == 1  # clones — one survives
        assert forward == reverse


class TestJaccard:
    def test_two_empty_pools_are_identical_not_disjoint(self) -> None:
        # _score_field ingests mine_pool output, so make the metric total rather
        # than relying on every caller to pre-filter empty candidate pools.
        assert _jaccard([], []) == 1.0

    def test_empty_versus_non_empty_is_disjoint(self) -> None:
        assert _jaccard([], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# _combined_field — mined candidates join the shared field as found_N (Task 7)
# ---------------------------------------------------------------------------


def _minable_pool() -> list[Track]:
    """Pool where named builders AND the miner both fire.

    ``conj_pool`` (tests/conftest.py) plants a Hospital Records x Liquid
    conjunction inside 80 tracks and carries ``date_added`` throughout, which
    gives fresh_crate its material. label_spotlight also proposes. Verified by
    ``test_fixture_yields_both_named_and_mined_rows`` below — a fixture whose
    miner silently stops firing would make every assertion here vacuous.
    """
    return conj_pool()


def _found(title: str, ids: list[str], *, identity: float, freshness: float = 0.5) -> Direction:
    """A mined-shaped candidate: ``direction_type == "found"``, pre-score."""
    return replace(_cand("found", ids, identity=identity, freshness=freshness), title=title)


class TestCombinedField:
    def test_collection_lifts_label_spotlight_identity(self) -> None:
        # A label that is 10/40 of the pool but rare in the wider collection scores
        # higher on identity than the same label measured pool-only. (_rich_pool,
        # not _minable_pool: there label_spotlight is a clone of fresh_crate.)
        pool = _rich_pool()
        collection = pool + [_track(track_id=f"c{i}", label="Other") for i in range(400)]

        def _label_identity(field: list[Direction]) -> float:
            return next(d.identity for d in field if d.direction_type == "label_spotlight")

        assert _label_identity(enumerate_directions(pool, seed=7, collection=collection)) > _label_identity(
            enumerate_directions(pool, seed=7)
        )

    def test_fixture_yields_both_named_and_mined_rows(self) -> None:
        out = enumerate_directions(_minable_pool(), seed=0)
        types = [d.direction_type for d in out]
        assert any(t.startswith("found") for t in types), types
        assert any(not t.startswith("found") for t in types), types

    def test_mined_rows_have_distinct_found_n_types(self) -> None:
        out = enumerate_directions(_minable_pool(), seed=0)
        found_types = [d.direction_type for d in out if d.direction_type.startswith("found")]
        assert found_types == [f"found_{i + 1}" for i in range(len(found_types))]
        assert len(found_types) <= 3

    def test_mined_rows_scored_not_zero(self) -> None:
        out = enumerate_directions(_minable_pool(), seed=0)
        for d in out:
            if d.direction_type.startswith("found"):
                assert d.feasibility > 0.0

    def test_no_duplicate_titles_among_mined(self) -> None:
        out = enumerate_directions(_minable_pool(), seed=0)
        titles = [d.title for d in out if d.direction_type.startswith("found")]
        assert len(titles) == len(set(titles))

    def test_same_seed_byte_identical(self) -> None:
        a = enumerate_directions(_minable_pool(), seed=7)
        b = enumerate_directions(_minable_pool(), seed=7)
        assert a == b

    def test_sorted_by_feasibility_then_type(self) -> None:
        out = enumerate_directions(_minable_pool(), seed=0)
        keys = [(-d.feasibility, d.direction_type) for d in out]
        assert keys == sorted(keys)


class TestMinedShaping:
    """Cap, title uniqueness and found_N renaming, over hand-built mined rows.

    The live fixture yields a single mined survivor, so the cap and the
    uniqueness rule need candidates built directly. Pools are disjoint so the
    scorer's clone dedupe cannot interfere with what is under test.
    """

    def test_caps_mined_survivors_at_three_in_rank_order(self) -> None:
        cands = [_found(f"Found: t{i}", [f"{i}-{j}" for j in range(25)], identity=0.9 - i * 0.1) for i in range(5)]
        out = _shape_field(cands)
        assert [d.direction_type for d in out] == ["found_1", "found_2", "found_3"]
        assert [d.title for d in out] == ["Found: t0", "Found: t1", "Found: t2"]

    def test_duplicate_mined_title_keeps_the_higher_ranked_row(self) -> None:
        # Two pairs can produce the same title from the same namable value.
        hi = _found("Found: liquid", [f"a{i}" for i in range(25)], identity=0.9)
        lo = _found("Found: liquid", [f"b{i}" for i in range(25)], identity=0.2)
        other = _found("Found: Metalheadz", [f"c{i}" for i in range(25)], identity=0.5)
        out = _shape_field([lo, other, hi])
        assert [(d.direction_type, d.title) for d in out] == [
            ("found_1", "Found: liquid"),
            ("found_2", "Found: Metalheadz"),
        ]
        assert all(d.track_ids[0].startswith(("a", "c")) for d in out)  # the lo clone is gone

    def test_distinctiveness_ignores_rows_the_cap_removed(self) -> None:
        """Pass 2 runs on the POST-cap field, so a capped-away row cannot reorder
        the rows that ship.

        ``t3`` overlaps ``t0`` at Jaccard exactly 0.6 — at the dedupe threshold, so it
        survives pass 1 — and is then dropped by the 3-per-pool cap. Measured
        pre-cap, ``t0``'s distinctiveness would be 1-0.6 = 0.4 and its feasibility
        0.125 + 0.405 + 0.12 = 0.65, which sinks the best mined row below both
        ``t1`` (0.785) and ``t2`` (0.74) and hands ``found_1`` — the only mined row
        ``generate_directions`` ships — to the wrong pair. Post-cap the three
        survivors are mutually disjoint, so each scores distinctiveness 1.0.
        """
        t0 = _found("Found: t0", [f"x{i}" for i in range(20)], identity=0.9)
        t1 = _found("Found: t1", [f"m{i}" for i in range(20)], identity=0.8)
        t2 = _found("Found: t2", [f"n{i}" for i in range(20)], identity=0.7)
        t3 = _found("Found: t3", [f"x{i}" for i in range(15)] + [f"y{i}" for i in range(5)], identity=0.6)
        assert _jaccard(t0.track_ids, t3.track_ids) == 0.6  # 15 shared of a 25-track union

        out = _shape_field([t0, t1, t2, t3])
        assert [(d.direction_type, d.title, d.feasibility) for d in out] == [
            ("found_1", "Found: t0", 0.83),
            ("found_2", "Found: t1", 0.785),
            ("found_3", "Found: t2", 0.74),
        ]

    def test_named_rows_keep_their_type_and_are_uncapped(self) -> None:
        named = [_cand(f"named_{i}", [f"n{i}-{j}" for j in range(25)], identity=0.5, freshness=0.5) for i in range(5)]
        mined = [_found("Found: x", [f"m{j}" for j in range(25)], identity=0.5)]
        out = _shape_field(named + mined)
        assert sum(1 for d in out if d.direction_type.startswith("named_")) == 5
        assert [d.direction_type for d in out if d.direction_type.startswith("found")] == ["found_1"]

    def test_shaping_is_order_independent(self) -> None:
        cands = [_found(f"Found: t{i}", [f"{i}-{j}" for j in range(25)], identity=0.9 - i * 0.1) for i in range(5)]
        assert _shape_field(cands) == _shape_field(list(reversed(cands)))


class TestRunPathMinedCap:
    def test_at_most_one_found_canvas_per_run(self) -> None:
        pool = _minable_pool()
        tracks_by_id = _tbi(pool)
        canvases = generate_directions(pool, tracks_by_id, seed=3, max_directions=3)
        found = [c for c in canvases if c.direction_type.startswith("found")]
        assert len(found) <= 1

    def test_a_found_canvas_actually_materialises(self) -> None:
        pool = _minable_pool()
        canvases = generate_directions(pool, _tbi(pool), seed=3, max_directions=3)
        found = [c for c in canvases if c.direction_type.startswith("found")]
        assert len(found) == 1
        assert found[0].brief.startswith("FOUND SET.")

    def test_three_mined_survivors_still_ship_only_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The live fixture mines exactly one pair, so the run-path cap needs a pool
        that yields several. Three disjoint found rows survive the field scorer and
        get numbered found_1..3; only the best-scoring one may be materialised."""
        pool = _minable_pool()
        ids = [t.track_id for t in pool]
        # Ten-track slices: pairwise disjoint, and small enough that even total
        # containment in a 25-track named pool stays under the 0.6 dedupe threshold.
        mined = [_found(f"Found: t{i}", ids[i * 10 : (i + 1) * 10], identity=0.9 - i * 0.1) for i in range(3)]
        monkeypatch.setattr("mixlab.directions.mining.mine_pool", lambda _pool: mined)
        assert [d.direction_type for d in enumerate_directions(pool, seed=3) if "found" in d.direction_type] == [
            "found_1",
            "found_2",
            "found_3",
        ]
        canvases = generate_directions(pool, _tbi(pool), seed=3, max_directions=3)
        assert [c.direction_type for c in canvases if c.direction_type.startswith("found")] == ["found_1"]

    def test_zero_max_directions_ships_nothing_not_the_found_row(self) -> None:
        # mined[:1] sidesteps max_directions by construction — it must still honour
        # a caller asking for no directions at all.
        pool = _minable_pool()
        assert generate_directions(pool, _tbi(pool), seed=3, max_directions=0) == []

    def test_diagnostic_line_reports_found_separately(self, capsys: pytest.CaptureFixture[str]) -> None:
        pool = _minable_pool()
        generate_directions(pool, _tbi(pool), seed=3)
        out = capsys.readouterr().out
        assert "found)" in out and "builders" in out

    def test_diagnostic_line_counts_proposals_not_survivors(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Both counts are pre-scoring: a builder that proposed and then lost the
        clone dedupe is a different failure from one that never fired, and the run
        log exists to tell them apart. On _minable_pool label_spotlight proposes but
        is deduped, so the line must still say 2 builders."""
        pool = _minable_pool()
        canvases = generate_directions(pool, _tbi(pool), seed=3, max_directions=6)
        out = capsys.readouterr().out
        assert "Directions proposed: fresh_crate, label_spotlight (2/7 builders, 1 found)" in out
        assert "label_spotlight" not in [c.direction_type for c in canvases]

    def test_collection_kwarg_reaches_label_spotlight_scoring(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The run path must score label_spotlight the way the map path does.

        Diluting the label across a wider collection raises its lift-based
        identity, which moves its feasibility — visible in the per-direction run
        log lines. If ``generate_directions`` swallowed ``collection`` the two
        runs would print identically.
        """
        pool = _rich_pool()
        collection = pool + [_track(track_id=f"c{i}", label="Other") for i in range(400)]
        generate_directions(pool, _tbi(pool), seed=7, max_directions=6)
        without = capsys.readouterr().out
        generate_directions(pool, _tbi(pool), seed=7, max_directions=6, collection=collection)
        with_collection = capsys.readouterr().out
        assert "label_spotlight" in without
        assert without != with_collection


# ---------------------------------------------------------------------------
# pinned_canvas_from_spec — the --direction-spec materialiser
# ---------------------------------------------------------------------------


def _spec_pool(n: int = 20) -> list[Track]:
    return [_track(track_id=f"s{i}", bpm=172.0 + (i % 3)) for i in range(n)]


def _spec_json(pool: list[Track], **overrides: object) -> str:
    import json as _json

    spec: dict[str, object] = {
        "direction_type": "artist_thread",
        "title": "Artist thread: Dusky",
        "mood": "Dusky as spine",
        "brief": "This set threads Dusky through the mix as its spine.",
        "track_ids": [t.track_id for t in pool],
        "thread_artist": "Dusky",
    }
    spec.update(overrides)
    return _json.dumps(spec)


class TestPinnedCanvasFromSpec:
    def test_happy_path_materialises_pinned_canvas(self) -> None:
        pool = _spec_pool()
        canvas = pinned_canvas_from_spec(_spec_json(pool), _tbi(pool))
        assert canvas.pinned is True
        assert canvas.direction_type == "artist_thread"
        assert canvas.brief.startswith("This set threads Dusky")
        assert canvas.thread_artist == "Dusky"
        assert canvas.source_concept.title == "Artist thread: Dusky"
        assert set(canvas.source_concept.track_ids) == {t.track_id for t in pool}

    def test_unknown_keys_ignored_forward_compat(self) -> None:
        pool = _spec_pool()
        spec = _spec_json(pool, feasibility=0.42, future_field="x")
        canvas = pinned_canvas_from_spec(spec, _tbi(pool))
        assert canvas.pinned is True

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(DirectionSpecError, match="not valid JSON"):
            pinned_canvas_from_spec("{nope", {})

    def test_non_object_raises(self) -> None:
        with pytest.raises(DirectionSpecError, match="JSON object"):
            pinned_canvas_from_spec("[1, 2]", {})

    @pytest.mark.parametrize("key", ["direction_type", "title", "brief"])
    def test_missing_required_string_raises(self, key: str) -> None:
        pool = _spec_pool()
        spec = _spec_json(pool, **{key: ""})
        with pytest.raises(DirectionSpecError, match=key):
            pinned_canvas_from_spec(spec, _tbi(pool))

    def test_track_ids_must_be_string_list(self) -> None:
        pool = _spec_pool()
        with pytest.raises(DirectionSpecError, match="track_ids"):
            pinned_canvas_from_spec(_spec_json(pool, track_ids=[1, 2, 3]), _tbi(pool))

    def test_too_few_resolvable_ids_raises_with_counts(self) -> None:
        pool = _spec_pool()
        known = _tbi(pool[:5])  # only 5 of 20 ids resolve — below MIN_DIRECTION_POOL
        with pytest.raises(DirectionSpecError, match=r"5 of 20"):
            pinned_canvas_from_spec(_spec_json(pool), known)

    def test_caps_at_max_direction_pool_preserving_order(self) -> None:
        pool = _spec_pool(MAX_DIRECTION_POOL + 10)
        canvas = pinned_canvas_from_spec(_spec_json(pool), _tbi(pool))
        assert canvas.source_concept.track_ids == [t.track_id for t in pool[:MAX_DIRECTION_POOL]]

    def test_thread_artist_derived_from_title_when_absent(self) -> None:
        # Stale map payloads predate thread_artist in the wire entry; the artist-thread
        # validator suppression still needs the name, so it is derived from the title.
        pool = _spec_pool()
        spec = _spec_json(pool, thread_artist="")
        canvas = pinned_canvas_from_spec(spec, _tbi(pool))
        assert canvas.thread_artist == "Dusky"

    def test_thread_artist_stays_empty_for_other_types(self) -> None:
        pool = _spec_pool()
        spec = _spec_json(pool, direction_type="label_spotlight", title="Label spotlight: Hotflush", thread_artist="")
        canvas = pinned_canvas_from_spec(spec, _tbi(pool))
        assert canvas.thread_artist == ""

    def test_mood_defaults_to_empty(self) -> None:
        pool = _spec_pool()
        canvas = pinned_canvas_from_spec(_spec_json(pool, mood=""), _tbi(pool))
        assert canvas.source_concept.mood == ""


# ---------------------------------------------------------------------------
# parse_track_pool — the --track-pool payload parser
# ---------------------------------------------------------------------------


class TestParseTrackPool:
    def test_happy_path_ids_and_label(self) -> None:
        pool = parse_track_pool('{"track_ids": ["1", "2", "3"], "label": "Monday warmup"}')
        assert pool.track_ids == ("1", "2", "3")
        assert pool.label == "Monday warmup"

    def test_ids_only_label_defaults_to_empty(self) -> None:
        pool = parse_track_pool('{"track_ids": ["1", "2"]}')
        assert pool.track_ids == ("1", "2")
        assert pool.label == ""

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="not valid JSON"):
            parse_track_pool("{nope")

    def test_json_array_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="JSON object"):
            parse_track_pool("[1, 2, 3]")

    def test_missing_track_ids_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="track_ids"):
            parse_track_pool('{"label": "no ids"}')

    def test_empty_track_ids_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="track_ids"):
            parse_track_pool('{"track_ids": []}')

    def test_non_list_track_ids_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="track_ids"):
            parse_track_pool('{"track_ids": "1"}')

    def test_non_string_track_id_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="track_ids"):
            parse_track_pool('{"track_ids": [1, 2, 3]}')

    def test_non_string_label_raises(self) -> None:
        with pytest.raises(TrackPoolError, match="label"):
            parse_track_pool('{"track_ids": ["1"], "label": 42}')

    def test_label_newline_sanitised_to_single_line(self) -> None:
        """The label reaches stdout raw downstream (the run's availability-table print),
        and the worker resolves artifacts by scanning stdout lines — an operator-supplied
        label containing a newline (plus fake trailer text) could hijack that scan. Strip
        control characters/newlines to a single space so the label can never introduce a
        line break."""
        pool = parse_track_pool('{"track_ids": ["1"], "label": "Monday block\\nRun summary: /tmp/evil.json"}')
        assert "\n" not in pool.label
        assert pool.label == "Monday block Run summary: /tmp/evil.json"

    def test_label_control_chars_replaced_with_space(self) -> None:
        pool = parse_track_pool('{"track_ids": ["1"], "label": "Monday\\tblock\\r\\ntitle"}')
        assert pool.label == "Monday block title"

    def test_label_whitespace_collapsed(self) -> None:
        pool = parse_track_pool('{"track_ids": ["1"], "label": "Monday    block"}')
        assert pool.label == "Monday block"

    def test_label_over_long_truncated_to_200_chars(self) -> None:
        raw = '{"track_ids": ["1"], "label": "' + ("x" * 500) + '"}'
        pool = parse_track_pool(raw)
        assert len(pool.label) == 200
        assert pool.label == "x" * 200


# ---------------------------------------------------------------------------
# key_groups — defining subsets emitted by builders and enforced on pinned runs
# ---------------------------------------------------------------------------


class TestBuilderKeyGroups:
    def test_artist_thread_requires_every_spine_track(self) -> None:
        d = _build_artist_thread(_rich_pool(), seed=0)
        assert d is not None
        (group,) = d.key_groups
        assert group.label == "SpineGuy spine"
        assert group.required == len(group.track_ids) == 3

    def test_mood_journey_keys_both_poles(self) -> None:
        d = _build_mood_journey(_rich_pool(), seed=0)
        assert d is not None
        labels = {g.label for g in d.key_groups}
        assert len(d.key_groups) == 2
        assert any("pole" in lab for lab in labels)
        for g in d.key_groups:
            assert g.required == 2
            assert set(g.track_ids) <= set(d.track_ids)

    def test_era_dialogue_keys_both_sides(self) -> None:
        d = _build_era_dialogue(_rich_pool(), seed=0)
        assert d is not None
        assert len(d.key_groups) == 2
        for g in d.key_groups:
            assert "era" in g.label
            assert g.required == 2

    def test_label_spotlight_keys_the_catalogue(self) -> None:
        d = _build_label_spotlight(_rich_pool(), seed=0)
        assert d is not None
        (group,) = d.key_groups
        assert group.label == "Hospital catalogue"
        assert group.required == 4
        assert set(group.track_ids) <= set(d.track_ids)

    def test_energy_shape_keys_troughs_and_crests(self) -> None:
        d = _build_energy_shape_first(_rich_pool(), seed=0)
        assert d is not None
        assert {g.label for g in d.key_groups} == {"low-energy troughs", "high-energy crests"}
        for g in d.key_groups:
            assert g.required == 2

    def test_fresh_crate_emits_no_key_groups(self) -> None:
        # 60 dated tracks so the top-20% slice (12) clears _FRESH_MIN_COUNT (10).
        pool = [
            _track(track_id=f"f{i}", date_added=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}", rating=3 if i < 5 else 0)
            for i in range(60)
        ]
        d = _build_fresh_crate(pool, seed=0)
        assert d is not None
        assert d.key_groups == []

    def test_generate_directions_stamps_key_groups_on_canvases(self) -> None:
        pool = _rich_pool()
        canvases = generate_directions(pool, _tbi(pool), seed=0, max_directions=6)
        assert canvases
        keyed = [c for c in canvases if c.key_groups]
        assert keyed, "at least one direction canvas must carry key groups"
        for canvas in keyed:
            canvas_ids = set(canvas.source_concept.track_ids)
            for g in canvas.key_groups:
                assert set(g.track_ids) <= canvas_ids
                assert 1 <= g.required <= len(g.track_ids)


class TestSpecKeyGroups:
    def test_key_groups_round_trip_and_clamp(self) -> None:
        pool = _spec_pool()
        spec = _spec_json(
            pool,
            key_groups=[
                {"label": "Dusky spine", "required": 3, "track_ids": ["s0", "s1", "ghost"]},
                {"label": "all ghosts", "required": 2, "track_ids": ["nope1", "nope2"]},
            ],
        )
        canvas = pinned_canvas_from_spec(spec, _tbi(pool))
        # "ghost" dropped, required clamped 3 -> 2; the all-ghost group vanishes.
        (group,) = canvas.key_groups
        assert group.label == "Dusky spine"
        assert group.required == 2
        assert group.track_ids == ["s0", "s1"]

    def test_key_groups_absent_is_fine(self) -> None:
        pool = _spec_pool()
        canvas = pinned_canvas_from_spec(_spec_json(pool), _tbi(pool))
        assert canvas.key_groups == []

    def test_malformed_key_groups_raise(self) -> None:
        pool = _spec_pool()
        for bad in [{"label": "x"}, {"label": 1, "required": 2, "track_ids": []}, "nope"]:
            with pytest.raises(DirectionSpecError, match="key_groups"):
                pinned_canvas_from_spec(_spec_json(pool, key_groups=[bad]), _tbi(pool))
