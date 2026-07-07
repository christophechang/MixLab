from __future__ import annotations

import pytest

from mixlab.models import MixConcept, MixPoints, Track
from mixlab.transitions import (
    TransitionEdge,
    blend_headroom,
    relation_label,
    score_transition,
    strong_edges,
    tempo_relation,
    trace_arc,
)


def _track(
    track_id: str,
    *,
    bpm: float = 128.0,
    key: str = "8A",
    energy: int | None = None,
    tags: list[str] | None = None,
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
    )


def _track_with_mix_points(
    track_id: str,
    *,
    bpm: float = 128.0,
    key: str = "8A",
    outro_bars: float | None = None,
    intro_bars: float | None = None,
    loops: list[tuple[float, float]] | None = None,
    duration_secs: int | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        artist=f"Artist_{track_id}",
        title=f"Title_{track_id}",
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        duration_secs=duration_secs,
        mix_points=MixPoints(
            mix_in_secs=0.0,
            outro_bars=outro_bars,
            intro_bars=intro_bars,
            loops=loops or [],
        ),
    )


# ---------------------------------------------------------------------------
# tempo_relation
# ---------------------------------------------------------------------------


def test_tempo_relation_172_to_86_is_halftime() -> None:
    rel, stretch = tempo_relation(172.0, 86.0)
    assert rel == "halftime"
    assert stretch == pytest.approx(0.0)


def test_tempo_relation_86_to_172_is_double() -> None:
    rel, stretch = tempo_relation(86.0, 172.0)
    assert rel == "double"
    assert stretch == pytest.approx(0.0)


def test_tempo_relation_128_to_96_is_three_four() -> None:
    rel, stretch = tempo_relation(128.0, 96.0)
    assert rel == "three_four"
    assert stretch == pytest.approx(0.0)


def test_tempo_relation_96_to_128_is_four_three() -> None:
    rel, stretch = tempo_relation(96.0, 128.0)
    assert rel == "four_three"
    assert stretch == pytest.approx(0.0)


def test_tempo_relation_128_to_128_is_straight_zero_stretch() -> None:
    rel, stretch = tempo_relation(128.0, 128.0)
    assert rel == "straight"
    assert stretch == pytest.approx(0.0)


def test_tempo_relation_124_to_128_is_straight_small_stretch() -> None:
    rel, stretch = tempo_relation(124.0, 128.0)
    assert rel == "straight"
    assert 0.0 < stretch < 0.06


def test_tempo_relation_128_to_150_is_incompatible() -> None:
    # 150 sits outside every ratio window from 128 (straight, 3:4, 4:3, half, double).
    rel, _ = tempo_relation(128.0, 150.0)
    assert rel == "incompatible"


def test_tempo_relation_zero_bpm_is_incompatible() -> None:
    rel, stretch = tempo_relation(0.0, 120.0)
    assert rel == "incompatible"
    assert stretch == pytest.approx(1.0)


def test_tempo_relation_exactly_six_percent_stretch_passes() -> None:
    # 106 is exactly 6% above 100 — on the boundary, straight still wins.
    rel, stretch = tempo_relation(100.0, 106.0)
    assert rel == "straight"
    assert stretch == pytest.approx(0.06)


def test_tempo_relation_six_point_five_percent_stretch_fails() -> None:
    # 106.5 is 6.5% above 100 and fits no other ratio → incompatible.
    rel, _ = tempo_relation(100.0, 106.5)
    assert rel == "incompatible"


# ---------------------------------------------------------------------------
# score_transition
# ---------------------------------------------------------------------------


def test_score_transition_detects_energy_lift_same_ring_plus_one() -> None:
    edge = score_transition(_track("a", key="8A"), _track("b", key="9A"))
    assert edge.is_energy_lift is True


def test_score_transition_detects_energy_lift_wraps_12a_to_1a() -> None:
    edge = score_transition(_track("a", key="12A"), _track("b", key="1A"))
    assert edge.is_energy_lift is True


def test_score_transition_no_energy_lift_when_ring_differs() -> None:
    edge = score_transition(_track("a", key="8A"), _track("b", key="9B"))
    assert edge.is_energy_lift is False


def test_score_transition_unparseable_key_gives_neutral_harmonic() -> None:
    # Unparseable key → camelot_distance 999 → neutral-ish 0.3 harmonic component.
    edge = score_transition(_track("a", key="???"), _track("b", key="8A"))
    assert edge.camelot_distance == 999
    # straight tempo (1.0*0.4) + harmonic (0.3*0.3) + energy (0.7*0.15) + tags (0.7*0.15)
    expected = 1.0 * 0.4 + 0.3 * 0.3 + 0.7 * 0.15 + 0.7 * 0.15
    assert edge.score == pytest.approx(round(expected, 4))


def test_score_transition_incompatible_scores_zero() -> None:
    edge = score_transition(_track("a", bpm=128.0), _track("b", bpm=150.0))
    assert edge.tempo_relation == "incompatible"
    assert edge.score == 0.0


def test_score_transition_energy_delta_none_when_either_missing() -> None:
    edge = score_transition(_track("a", energy=5), _track("b", energy=None))
    assert edge.energy_delta is None


def test_score_transition_tag_overlap_jaccard() -> None:
    edge = score_transition(
        _track("a", tags=["Dark", "Rolling"]),
        _track("b", tags=["dark", "Liquid"]),
    )
    # {dark, rolling} ∩ {dark, liquid} = 1 ; union = 3
    assert edge.tag_overlap == pytest.approx(round(1 / 3, 4))


def test_score_transition_without_mix_points_matches_legacy_weights_regression() -> None:
    # Regression guard (#59): tracks without mix_points must be byte-identical to the
    # pre-blend-headroom formula (tempo 0.4 / harmonic 0.3 / energy 0.15 / tags 0.15).
    edge = score_transition(_track("a", key="???"), _track("b", key="8A"))
    expected = 1.0 * 0.4 + 0.3 * 0.3 + 0.7 * 0.15 + 0.7 * 0.15
    assert edge.score == pytest.approx(round(expected, 4))
    assert edge.blend_headroom is None
    assert edge.blend_label == ""


def test_score_transition_with_mix_points_uses_rebalanced_blend_weights() -> None:
    # a: halftime 172→86 same key, no tags/energy. outro 32 bars / intro 16 bars →
    # ratio 2.0 → headroom 1.0.
    a = _track_with_mix_points("a", bpm=172.0, key="8A", outro_bars=32.0)
    b = _track_with_mix_points("b", bpm=86.0, key="8A", intro_bars=16.0)
    edge = score_transition(a, b)

    assert edge.blend_headroom == pytest.approx(1.0)
    assert edge.blend_label == "32 bars out / 16 in"

    # tempo: halftime, zero stretch → 1.0 * 0.9 = 0.9 (non-straight ratio penalty)
    # harmonic: same key → 1.0
    # energy: both None → 0.7
    # tags: both empty → 0.7
    expected = 0.9 * 0.35 + 1.0 * 0.25 + 1.0 * 0.20 + 0.7 * 0.10 + 0.7 * 0.10
    assert edge.score == pytest.approx(round(expected, 4))


def test_score_transition_incompatible_still_zero_with_mix_points() -> None:
    a = _track_with_mix_points("a", bpm=128.0, outro_bars=32.0)
    b = _track_with_mix_points("b", bpm=150.0, intro_bars=16.0)
    edge = score_transition(a, b)
    assert edge.tempo_relation == "incompatible"
    assert edge.score == 0.0


# ---------------------------------------------------------------------------
# blend_headroom
# ---------------------------------------------------------------------------


def test_blend_headroom_comfortable_ratio_scores_one() -> None:
    a = _track_with_mix_points("a", outro_bars=48.0)
    b = _track_with_mix_points("b", intro_bars=16.0)
    score, label = blend_headroom(a, b)
    assert score == pytest.approx(1.0)
    assert label == "48 bars out / 16 in"


def test_blend_headroom_exact_one_point_five_boundary_scores_one() -> None:
    a = _track_with_mix_points("a", outro_bars=24.0)
    b = _track_with_mix_points("b", intro_bars=16.0)
    score, label = blend_headroom(a, b)
    assert score == pytest.approx(1.0)
    assert label == "24 bars out / 16 in"


def test_blend_headroom_ratio_one_point_two_linear_value() -> None:
    a = _track_with_mix_points("a", outro_bars=24.0)
    b = _track_with_mix_points("b", intro_bars=20.0)
    score, label = blend_headroom(a, b)
    # ratio 1.2 → 0.7 + (1.2 - 1.0) / 0.5 * 0.3 = 0.82
    assert score == pytest.approx(0.82)
    assert label == "24 bars out / 20 in"


def test_blend_headroom_ratio_zero_point_seven_is_tight() -> None:
    a = _track_with_mix_points("a", outro_bars=14.0)
    b = _track_with_mix_points("b", intro_bars=20.0)
    score, label = blend_headroom(a, b)
    # ratio 0.7 → 0.3 + (0.7 - 0.5) / 0.5 * 0.4 = 0.46
    assert score == pytest.approx(0.46)
    assert label == "14 bars out / 20 in — tight"


def test_blend_headroom_ratio_zero_point_three_is_lowest_tier() -> None:
    a = _track_with_mix_points("a", outro_bars=6.0)
    b = _track_with_mix_points("b", intro_bars=20.0)
    score, label = blend_headroom(a, b)
    # ratio 0.3 < 0.5 → flat 0.15 tier
    assert score == pytest.approx(0.15)
    assert label == "6 bars out / 20 in — cut or manual loop likely"


def test_blend_headroom_none_when_either_track_lacks_mix_points() -> None:
    a = _track("a")
    b = _track_with_mix_points("b", intro_bars=16.0)
    assert blend_headroom(a, b) == (None, "")
    assert blend_headroom(b, a) == (None, "")


def test_blend_headroom_none_when_outro_bars_missing() -> None:
    a = _track_with_mix_points("a", outro_bars=None)
    b = _track_with_mix_points("b", intro_bars=16.0)
    assert blend_headroom(a, b) == (None, "")


def test_blend_headroom_none_when_intro_bars_missing() -> None:
    a = _track_with_mix_points("a", outro_bars=32.0)
    b = _track_with_mix_points("b", intro_bars=None)
    assert blend_headroom(a, b) == (None, "")


def test_blend_headroom_loop_bonus_fires_when_loop_in_final_third_and_duration_known() -> None:
    # Base ratio 0.3 → 0.15 tier; loop in final third + known duration → +0.15 → 0.30,
    # which crosses the tier boundary into "tight".
    a = _track_with_mix_points("a", outro_bars=6.0, loops=[(250.0, 260.0)], duration_secs=300)
    b = _track_with_mix_points("b", intro_bars=20.0)
    score, label = blend_headroom(a, b)
    assert score == pytest.approx(0.30)
    assert label == "6 bars out / 20 in — tight (loop zone available)"


def test_blend_headroom_loop_bonus_skipped_when_loop_not_in_final_third() -> None:
    a = _track_with_mix_points("a", outro_bars=6.0, loops=[(10.0, 20.0)], duration_secs=300)
    b = _track_with_mix_points("b", intro_bars=20.0)
    score, label = blend_headroom(a, b)
    assert score == pytest.approx(0.15)
    assert "loop zone available" not in label


def test_blend_headroom_loop_bonus_skipped_when_duration_unknown() -> None:
    a = _track_with_mix_points("a", outro_bars=6.0, loops=[(250.0, 260.0)], duration_secs=None)
    b = _track_with_mix_points("b", intro_bars=20.0)
    score, label = blend_headroom(a, b)
    assert score == pytest.approx(0.15)
    assert "loop zone available" not in label


def test_blend_headroom_loop_bonus_capped_at_one() -> None:
    a = _track_with_mix_points("a", outro_bars=48.0, loops=[(250.0, 260.0)], duration_secs=300)
    b = _track_with_mix_points("b", intro_bars=16.0)
    score, _label = blend_headroom(a, b)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# strong_edges
# ---------------------------------------------------------------------------


def test_strong_edges_deterministic_with_mixed_cued_and_cueless_tracks() -> None:
    tracks = [
        _track_with_mix_points("a", bpm=172.0, key="8A", outro_bars=32.0),
        _track("b", bpm=86.0, key="8A"),
        _track_with_mix_points("c", bpm=128.0, key="9A", intro_bars=16.0),
    ]
    assert strong_edges(tracks) == strong_edges(tracks)


def test_strong_edges_is_deterministic() -> None:
    tracks = [
        _track("a", bpm=172.0, key="8A"),
        _track("b", bpm=86.0, key="8A"),
        _track("c", bpm=128.0, key="9A"),
    ]
    assert strong_edges(tracks) == strong_edges(tracks)


def test_strong_edges_dedupes_symmetric_pairs() -> None:
    tracks = [_track("a", bpm=172.0, key="8A"), _track("b", bpm=86.0, key="8A")]
    edges = strong_edges(tracks)
    pairs = [frozenset({e.from_id, e.to_id}) for e in edges]
    assert len(pairs) == len(set(pairs))
    assert len(edges) == 1
    # Tie in score → lexicographically lowest (from_id, to_id) direction is kept.
    assert (edges[0].from_id, edges[0].to_id) == ("a", "b")


def test_strong_edges_suppresses_boring_straight_edges() -> None:
    # Same BPM, same key, no energy movement → nothing non-obvious to surface.
    tracks = [_track("a", bpm=128.0, key="8A"), _track("b", bpm=128.0, key="8A")]
    assert strong_edges(tracks) == []


def test_strong_edges_respects_top_n() -> None:
    tracks = [
        _track("a", bpm=172.0, key="8A"),
        _track("b", bpm=86.0, key="8A"),
        _track("c", bpm=128.0, key="9A"),
        _track("d", bpm=96.0, key="10A"),
    ]
    edges = strong_edges(tracks, top_n=1)
    assert len(edges) == 1


def test_strong_edges_respects_min_score() -> None:
    tracks = [
        _track("a", bpm=172.0, key="8A"),
        _track("b", bpm=86.0, key="8A"),
    ]
    assert strong_edges(tracks, min_score=0.99) == []


# ---------------------------------------------------------------------------
# relation_label
# ---------------------------------------------------------------------------


def test_relation_label_halftime_lock() -> None:
    a, b = _track("a", bpm=172.0, key="8A"), _track("b", bpm=86.0, key="8A")
    edge = score_transition(a, b)
    assert relation_label(edge, a, b) == "halftime lock 172→86"


def test_relation_label_three_four_shuffle() -> None:
    a, b = _track("a", bpm=128.0, key="8A"), _track("b", bpm=96.0, key="3A")
    edge = score_transition(a, b)
    assert relation_label(edge, a, b) == "3:4 shuffle 128→96"


def test_relation_label_energy_lift() -> None:
    a, b = _track("a", bpm=128.0, key="8A"), _track("b", bpm=128.0, key="9A")
    edge = score_transition(a, b)
    assert relation_label(edge, a, b) == "energy lift 8A→9A"


def test_relation_label_straight() -> None:
    a, b = _track("a", bpm=128.0, key="8A"), _track("b", bpm=128.0, key="3A")
    edge = score_transition(a, b)
    assert relation_label(edge, a, b) == "straight"


# ---------------------------------------------------------------------------
# trace_arc
# ---------------------------------------------------------------------------


def _concept_with(energies: list[int | None], arc: str | None) -> tuple[MixConcept, dict[str, Track]]:
    ids = [str(i) for i in range(len(energies))]
    tracks_by_id = {tid: _track(tid, energy=e) for tid, e in zip(ids, energies, strict=True)}
    concept = MixConcept(title="T", mood="dark", track_ids=ids, arc_type=arc)  # type: ignore[arg-type]
    return concept, tracks_by_id


def test_trace_arc_none_arc_returns_none() -> None:
    concept, tracks = _concept_with([3, 5, 6, 7], None)
    assert trace_arc(concept, tracks) is None


def test_trace_arc_fewer_than_four_energies_returns_none() -> None:
    concept, tracks = _concept_with([3, 4, 5], "wave")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_progressive_build_falling_flags() -> None:
    concept, tracks = _concept_with([7, 6, 5, 4], "progressive-build")
    msg = trace_arc(concept, tracks)
    assert msg is not None
    assert "progressive-build" in msg
    assert "falls" in msg


def test_trace_arc_progressive_build_rising_matches() -> None:
    concept, tracks = _concept_with([3, 5, 6, 8], "progressive-build")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_build_and_drop_no_drop_flags() -> None:
    concept, tracks = _concept_with([3, 5, 6, 8], "build-and-drop")
    msg = trace_arc(concept, tracks)
    assert msg is not None
    assert "never drops" in msg


def test_trace_arc_build_and_drop_with_drop_matches() -> None:
    concept, tracks = _concept_with([3, 5, 8, 5], "build-and-drop")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_wave_monotonic_rising_flags() -> None:
    concept, tracks = _concept_with([3, 4, 6, 7], "wave")
    msg = trace_arc(concept, tracks)
    assert msg is not None
    assert "monotonic rising 3→7" in msg


def test_trace_arc_wave_non_monotonic_matches() -> None:
    concept, tracks = _concept_with([3, 6, 4, 7], "wave")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_plateau_wide_spread_flags() -> None:
    concept, tracks = _concept_with([3, 5, 7, 8], "plateau")
    msg = trace_arc(concept, tracks)
    assert msg is not None
    assert "spans 3→8" in msg


def test_trace_arc_plateau_tight_spread_matches() -> None:
    concept, tracks = _concept_with([5, 5, 6, 5], "plateau")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_front_loaded_starts_low_flags() -> None:
    concept, tracks = _concept_with([3, 4, 6, 7], "front-loaded")
    msg = trace_arc(concept, tracks)
    assert msg is not None
    assert "front-loaded" in msg


def test_trace_arc_front_loaded_starts_high_matches() -> None:
    concept, tracks = _concept_with([7, 6, 4, 3], "front-loaded")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_never_flagged_arc_returns_none_on_monotonic() -> None:
    # dark-to-light has no reliable energy signature — never flagged even when weird.
    concept, tracks = _concept_with([3, 4, 5, 6], "dark-to-light")
    assert trace_arc(concept, tracks) is None


def test_trace_arc_narrative_never_flagged() -> None:
    concept, tracks = _concept_with([8, 1, 8, 1], "narrative")
    assert trace_arc(concept, tracks) is None


def test_transition_edge_is_frozen() -> None:
    edge = TransitionEdge(
        from_id="a",
        to_id="b",
        tempo_relation="straight",
        tempo_stretch=0.0,
        camelot_distance=0,
        is_energy_lift=False,
        energy_delta=None,
        tag_overlap=0.0,
        score=1.0,
    )
    with pytest.raises(AttributeError):
        edge.score = 0.0  # type: ignore[misc]
