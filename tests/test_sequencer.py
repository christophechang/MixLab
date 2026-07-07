from __future__ import annotations

import pytest

from mixlab.models import ArcType, MixPoints, Track
from mixlab.sequencer import (
    Swap,
    optimal_order,
    score_order,
    suggest_swaps,
)
from mixlab.sequencer import (
    _arc_fit as arc_fit,
)
from mixlab.sequencer import (
    _endpoint_fit as endpoint_fit,
)


def _track(
    track_id: str,
    *,
    bpm: float = 128.0,
    key: str = "8A",
    energy: int | None = None,
    tags: list[str] | None = None,
    mix_points: MixPoints | None = None,
    duration_secs: int | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        artist=f"Artist_{track_id}",
        title=f"Title_{track_id}",
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        energy=energy,
        tags=tags or [],
        mix_points=mix_points,
        duration_secs=duration_secs,
    )


def _by_id(tracks: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


# Camelot chain 1A..12A: consecutive keys are wheel-adjacent (distance 1), pairs two or
# more apart clash. A clean substrate for testing that the solver recovers a chain.
def _chain_track(idx: int) -> Track:
    return _track(f"t{idx}", bpm=170.0 + idx, key=f"{idx + 1}A")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_optimal_order_twice_is_identical() -> None:
    tracks = [_chain_track(i) for i in range(10)]
    a = optimal_order(tracks)
    b = optimal_order(tracks)
    assert a == b
    assert a.order == b.order
    assert a.total_score == b.total_score


# ---------------------------------------------------------------------------
# planted-chain recovery
# ---------------------------------------------------------------------------


def test_optimal_order_recovers_planted_harmonic_chain() -> None:
    planted = [_chain_track(i) for i in range(8)]
    tracks_by_id = _by_id(planted)
    planted_ids = [t.track_id for t in planted]

    plan = optimal_order(planted)
    planted_plan = score_order(planted_ids, tracks_by_id)

    # Solver is a maximiser, so it can only match or beat the planted chain (or its
    # reverse, which scores identically).
    assert plan.total_score >= planted_plan.total_score - 1e-9
    assert plan.order in (planted_ids, list(reversed(planted_ids)))

    # A scramble that breaks the chain must score strictly worse than the chain.
    scrambled = list(planted_ids)
    scrambled[3], scrambled[6] = scrambled[6], scrambled[3]
    assert score_order(scrambled, tracks_by_id).total_score < planted_plan.total_score


# ---------------------------------------------------------------------------
# pinned opener / closer
# ---------------------------------------------------------------------------


def test_optimal_order_respects_pinned_opener_and_closer() -> None:
    tracks = [_chain_track(i) for i in range(8)]
    plan = optimal_order(tracks, opener_id="t5", closer_id="t2")
    assert plan.order[0] == "t5"
    assert plan.order[-1] == "t2"
    assert set(plan.order) == {t.track_id for t in tracks}
    assert len(plan.order) == len(tracks)


def test_optimal_order_pinned_opener_only() -> None:
    tracks = [_chain_track(i) for i in range(6)]
    plan = optimal_order(tracks, opener_id="t4")
    assert plan.order[0] == "t4"


# ---------------------------------------------------------------------------
# ValueError cases
# ---------------------------------------------------------------------------


def test_optimal_order_too_few_tracks_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        optimal_order([_chain_track(0)])


def test_optimal_order_too_many_tracks_raises() -> None:
    with pytest.raises(ValueError, match="at most 40"):
        optimal_order([_chain_track(i) for i in range(41)])


def test_optimal_order_unknown_opener_raises() -> None:
    with pytest.raises(ValueError, match="opener_id"):
        optimal_order([_chain_track(i) for i in range(4)], opener_id="nope")


def test_optimal_order_unknown_closer_raises() -> None:
    with pytest.raises(ValueError, match="closer_id"):
        optimal_order([_chain_track(i) for i in range(4)], closer_id="nope")


# ---------------------------------------------------------------------------
# arc_fit — one case per branch
# ---------------------------------------------------------------------------


def test_arc_fit_none_or_short_is_neutral() -> None:
    assert arc_fit([3, 5, 7, 8], None) == pytest.approx(0.7)
    assert arc_fit([3, 5, 7], "wave") == pytest.approx(0.7)  # fewer than 4


def test_arc_fit_progressive_build_rising_and_falling() -> None:
    # delta = 6 - 3 = 3 → (3 + 0.5) / 1.5 = 2.33 → clamped 1.0
    assert arc_fit([3, 3, 6, 6, 6, 6], "progressive-build") == pytest.approx(1.0)
    # delta = 3 - 6 = -3 → (-3 + 0.5) / 1.5 < 0 → clamped 0.0
    assert arc_fit([6, 5, 4, 3], "progressive-build") == pytest.approx(0.0)


def test_arc_fit_build_and_drop_branches() -> None:
    assert arc_fit([3, 5, 8, 4], "build-and-drop") == pytest.approx(1.0)  # rise + drop
    assert arc_fit([3, 5, 8, 7], "build-and-drop") == pytest.approx(0.5)  # rise, no drop
    assert arc_fit([8, 7, 6, 5], "build-and-drop") == pytest.approx(0.3)  # no rise


def test_arc_fit_wave_monotonic_vs_not() -> None:
    assert arc_fit([3, 6, 4, 7], "wave") == pytest.approx(1.0)  # rise and fall
    assert arc_fit([3, 4, 5, 6], "double-peak") == pytest.approx(0.3)  # monotonic


def test_arc_fit_plateau_spread_scale() -> None:
    assert arc_fit([5, 5, 6, 7], "plateau") == pytest.approx(1.0)  # spread 2
    assert arc_fit([5, 6, 7, 8], "sustained-pressure") == pytest.approx(0.7333, abs=1e-4)  # spread 3
    assert arc_fit([3, 5, 7, 9], "plateau") == pytest.approx(0.2)  # spread 6 >= 5


def test_arc_fit_front_loaded() -> None:
    assert arc_fit([7, 6, 4, 3], "front-loaded") == pytest.approx(1.0)
    assert arc_fit([3, 4, 6, 7], "front-loaded") == pytest.approx(0.4)


def test_arc_fit_directional_arcs_are_neutral() -> None:
    for arc in ("dark-to-light", "light-to-dark", "narrative", "abstract-journey"):
        assert arc_fit([8, 1, 8, 1], arc) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# endpoint_fit branches
# ---------------------------------------------------------------------------


def test_endpoint_fit_soft_open_and_close() -> None:
    tracks = [_track("a", energy=3), _track("b", energy=6), _track("c", energy=5)]
    # base 0.5 + first<=4 (0.25) + last<=5 (0.25) = 1.0
    assert endpoint_fit(tracks) == pytest.approx(1.0)


def test_endpoint_fit_hot_open_and_close() -> None:
    tracks = [_track("a", energy=9), _track("b", energy=6), _track("c", energy=9)]
    # base 0.5 + neither endpoint qualifies
    assert endpoint_fit(tracks) == pytest.approx(0.5)


def test_endpoint_fit_unknown_endpoints_hedge() -> None:
    tracks = [_track("a", energy=None), _track("b", energy=6), _track("c", energy=None)]
    # base 0.5 + 0.1 (unknown first) + 0.1 (unknown last) = 0.7
    assert endpoint_fit(tracks) == pytest.approx(0.7)


def test_endpoint_fit_empty_is_base() -> None:
    assert endpoint_fit([]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# suggest_swaps
# ---------------------------------------------------------------------------


def _swap_case_tracks() -> tuple[list[str], dict[str, Track]]:
    # Two tempo families: positions 0..5 sit at 170 BPM, 6..9 at 205 BPM (an
    # incompatible ratio). The ideal order keeps each family contiguous. The planted
    # order swaps the tracks that belong at positions 3 and 6, dropping a 205 track into
    # the slow block and a 170 track into the fast block — three transitions collapse to
    # score 0 until a single 3<->6 swap restores them. Same key everywhere so tempo, not
    # harmony, drives the gain.
    bpms = [170.0] * 6 + [205.0] * 4
    tracks = [_track(f"k{i}", bpm=bpms[i], key="8A") for i in range(10)]
    tracks_by_id = _by_id(tracks)
    planted = [f"k{i}" for i in range(10)]
    planted[3], planted[6] = planted[6], planted[3]
    return planted, tracks_by_id


def test_suggest_swaps_finds_the_improving_swap() -> None:
    order, tracks_by_id = _swap_case_tracks()
    swaps = suggest_swaps(order, tracks_by_id, max_swaps=2)
    assert len(swaps) >= 1
    first = swaps[0]
    assert (first.pos_a, first.pos_b) == (3, 6)
    assert first.gain > 0.0
    assert first.reason  # non-empty one-liner


def test_suggest_swaps_never_touches_endpoints() -> None:
    order, tracks_by_id = _swap_case_tracks()
    swaps = suggest_swaps(order, tracks_by_id, max_swaps=3)
    n = len(order)
    for s in swaps:
        assert 0 < s.pos_a < s.pos_b < n - 1


def test_suggest_swaps_respects_max_swaps() -> None:
    order, tracks_by_id = _swap_case_tracks()
    swaps = suggest_swaps(order, tracks_by_id, max_swaps=1)
    assert len(swaps) <= 1


def test_suggest_swaps_no_improvement_returns_empty() -> None:
    # Already an ideal chain — no interior swap improves it.
    ideal = [f"c{i}" for i in range(10)]
    tracks_by_id = _by_id([_track(f"c{i}", bpm=170.0, key=f"{i + 1}A") for i in range(10)])
    assert suggest_swaps(ideal, tracks_by_id) == []


def test_suggest_swaps_applies_swap_result_beats_baseline() -> None:
    order, tracks_by_id = _swap_case_tracks()
    baseline = score_order(order, tracks_by_id).total_score
    swaps = suggest_swaps(order, tracks_by_id)
    applied = list(order)
    for s in swaps:
        applied[s.pos_a], applied[s.pos_b] = applied[s.pos_b], applied[s.pos_a]
    assert score_order(applied, tracks_by_id).total_score > baseline


# ---------------------------------------------------------------------------
# mixed / degenerate data
# ---------------------------------------------------------------------------


def test_score_order_with_mix_points_flows_through() -> None:
    tracks = [
        _track("a", bpm=172.0, key="8A", mix_points=MixPoints(mix_in_secs=0.0, outro_bars=32.0)),
        _track("b", bpm=86.0, key="8A", mix_points=MixPoints(mix_in_secs=0.0, intro_bars=16.0)),
        _track("c", bpm=86.0, key="9A"),
    ]
    plan = score_order([t.track_id for t in tracks], _by_id(tracks))
    assert plan.order == ["a", "b", "c"]
    assert len(plan.edge_scores) == 2
    assert 0.0 <= plan.total_score <= 1.0


def test_score_order_energyless_tracks_edge_only() -> None:
    tracks = [_track("a", key="8A"), _track("b", key="9A"), _track("c", key="10A")]
    plan = score_order([t.track_id for t in tracks], _by_id(tracks))
    # No energy anywhere → arc neutral 0.7, endpoints unknown → 0.7.
    assert plan.arc_fit == pytest.approx(0.7)
    assert plan.endpoint_fit == pytest.approx(0.7)


def test_score_order_skips_unknown_ids() -> None:
    tracks = [_track("a", key="8A"), _track("b", key="9A")]
    plan = score_order(["a", "ghost", "b"], _by_id(tracks))
    assert plan.order == ["a", "b"]


def test_score_order_single_known_id_has_zero_edge_component() -> None:
    tracks = [_track("a", key="8A"), _track("b", key="9A")]
    plan = score_order(["a", "ghost"], _by_id(tracks))
    assert plan.order == ["a"]
    assert plan.edge_scores == []
    # edge_mean 0.0 → total = 0 * 0.7 + arc 0.7 * 0.2 + endpoint * 0.1
    assert plan.total_score == pytest.approx(round(0.7 * 0.2 + plan.endpoint_fit * 0.1, 4))


def test_score_order_empty_does_not_crash() -> None:
    plan = score_order([], {})
    assert plan.order == []
    assert plan.edge_scores == []
    assert plan.total_score == pytest.approx(round(0.7 * 0.2 + 0.5 * 0.1, 4))


def test_notes_flag_incompatible_and_key_jumps() -> None:
    tracks = [
        _track("a", bpm=128.0, key="1A"),
        _track("b", bpm=150.0, key="7A"),  # incompatible tempo + big key jump
        _track("c", bpm=150.0, key="8A"),
    ]
    plan = score_order([t.track_id for t in tracks], _by_id(tracks))
    assert any(note.startswith("0->1") and "incompatible" in note for note in plan.notes)


def test_swap_type_is_frozen() -> None:
    s = Swap(pos_a=1, pos_b=2, gain=0.1, reason="x")
    with pytest.raises(AttributeError):
        s.gain = 0.2  # type: ignore[misc]


def test_arc_type_annotation_used() -> None:
    # Ensures arc_type parameter threads through to arc_fit in a full plan.
    tracks = [_track(f"e{i}", key=f"{i + 1}A", energy=e) for i, e in enumerate([3, 4, 6, 7])]
    ids = [t.track_id for t in tracks]
    tbi = _by_id(tracks)
    built: ArcType = "progressive-build"
    plan = score_order(ids, tbi, built)
    # progressive build rising → arc_fit should be high (well above neutral 0.7)
    assert plan.arc_fit > 0.7
