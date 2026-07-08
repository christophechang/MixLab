from __future__ import annotations

import re

from mixlab.directions import (
    MAX_DIRECTION_POOL,
    MIN_DIRECTION_POOL,
    _build_artist_thread,
    _build_energy_shape_first,
    _build_era_dialogue,
    _build_fresh_crate,
    _build_genre_traverse,
    _build_label_spotlight,
    _build_mood_journey,
    _path_feasible,
    generate_directions,
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


def test_build_fresh_crate_with_date_density_proposes() -> None:
    pool = [
        _track(track_id=str(i), bpm=172.0, date_added=f"2020-01-01T00:{i:02d}:00", play_count=1 if i < 3 else 0)
        for i in range(55)
    ]
    direction = _build_fresh_crate(pool, seed=1)
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
