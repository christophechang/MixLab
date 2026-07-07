from __future__ import annotations

import pytest

from mixlab.history import ConceptHistory, ConceptRecord, HistoryEntry
from mixlab.models import MixPoints, Track
from mixlab.prep import rank_prep_targets
from mixlab.transitions import score_transition


def _track(
    track_id: str,
    *,
    artist: str = "Artist",
    title: str = "Title",
    bpm: float = 172.0,
    key: str = "4A",
    genre: str = "Drum & Bass",
    play_count: int = 0,
    mix_points: MixPoints | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=key,
        genre=genre,
        play_count=play_count,
        mix_points=mix_points,
    )


def _record(concept_id: str, track_ids: list[str]) -> ConceptRecord:
    return ConceptRecord(
        concept_id=concept_id,
        title="Test",
        mood="dark",
        track_ids=track_ids,
        arc_type="",
    )


def _entry(run_id: str, concepts: list[ConceptRecord], genre: str = "drum_and_bass") -> HistoryEntry:
    return HistoryEntry(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        mode="standard",
        genre=genre,
        selected_canvas_ids=[],
        dominant_bpm_clusters=[],
        dominant_camelot_keys=[],
        core_track_ids=[],
        anchor_track_ids=[],
        opener_candidates=[],
        closer_candidates=[],
        concept_title="",
        concept_track_ids=[],
        energy_path="",
        mood="",
        concepts=concepts,
    )


# ---------------------------------------------------------------------------
# gap classification
# ---------------------------------------------------------------------------


def test_rank_prep_targets_gap_full_data_excluded_but_counted_in_totals() -> None:
    uncued = _track("T_uncued", mix_points=None)
    partial = _track("T_partial", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=None))
    full = _track("T_full", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))

    items, totals = rank_prep_targets([uncued, partial, full], ConceptHistory())

    ids = {item.track_id for item in items}
    assert ids == {"T_uncued", "T_partial"}
    gap_by_id = {item.track_id: item.gap for item in items}
    assert gap_by_id["T_uncued"] == "uncued"
    assert gap_by_id["T_partial"] == "no-mix-out"
    assert totals["drum_and_bass"] == (2, 3)


# ---------------------------------------------------------------------------
# demand
# ---------------------------------------------------------------------------


def test_rank_prep_targets_demand_counts_concepts_across_multiple_runs() -> None:
    track = _track("T1", mix_points=None)
    other = _track("T2", mix_points=None)

    history = ConceptHistory(
        runs=[
            _entry("r1", [_record("c1", ["T1"]), _record("c2", ["T1", "T2"])]),
            _entry("r2", [_record("c3", ["T2"])]),
            _entry("r3", [_record("c4", ["T1"])]),
        ]
    )

    items, _totals = rank_prep_targets([track, other], history)
    demand_by_id = {item.track_id: item.demand for item in items}
    assert demand_by_id["T1"] == 3
    assert demand_by_id["T2"] == 2


# ---------------------------------------------------------------------------
# centrality
# ---------------------------------------------------------------------------


def test_rank_prep_targets_centrality_two_of_three_strong_edges() -> None:
    track_x = _track("X1", bpm=172.0, key="4A", mix_points=None)
    strong_a = _track("A1", bpm=172.0, key="4A", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))
    strong_b = _track("B1", bpm=172.0, key="4A", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))
    weak_c = _track("C1", bpm=95.0, key="9B", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))

    # Verify the edge-score assumptions directly before trusting the derived centrality.
    assert score_transition(track_x, strong_a).score >= 0.7
    assert score_transition(track_x, strong_b).score >= 0.7
    assert score_transition(track_x, weak_c).score < 0.7

    items, _totals = rank_prep_targets([track_x, strong_a, strong_b, weak_c], ConceptHistory())

    assert len(items) == 1
    assert items[0].track_id == "X1"
    assert items[0].centrality == pytest.approx(0.667)


def test_rank_prep_targets_peer_cap_sampling_is_deterministic_and_uses_sampled_subset() -> None:
    track_x = _track("X0", bpm=172.0, key="4A", mix_points=None)
    # Sorted peer ids A..F; peer_cap=3 samples indices [0, 2, 5] -> A, C, F (see
    # rank_prep_targets/_sample_peers docstring for the sampling formula).
    strong_ids = {"A", "C", "F"}
    peers = [
        _track(
            pid,
            bpm=172.0 if pid in strong_ids else 95.0,
            key="4A" if pid in strong_ids else "9B",
            mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0),
        )
        for pid in ["A", "B", "C", "D", "E", "F"]
    ]
    tracks = [track_x, *peers]

    items_capped_first, _ = rank_prep_targets(tracks, ConceptHistory(), peer_cap=3)
    items_capped_second, _ = rank_prep_targets(tracks, ConceptHistory(), peer_cap=3)
    assert items_capped_first == items_capped_second
    assert items_capped_first[0].centrality == pytest.approx(1.0)

    items_uncapped, _ = rank_prep_targets(tracks, ConceptHistory(), peer_cap=600)
    assert items_uncapped[0].centrality == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# score formula
# ---------------------------------------------------------------------------


def test_rank_prep_targets_score_formula_and_gap_weight() -> None:
    uncued_unplayed = _track("U1", genre="Vaporwave", play_count=0, mix_points=None)
    no_mix_out_played = _track(
        "U2",
        genre="Vaporwave",
        play_count=4,
        mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=None),
    )

    history = ConceptHistory(runs=[_entry("r1", [_record("c1", ["U1"]), _record("c2", ["U1"])])])

    items, _totals = rank_prep_targets([uncued_unplayed, no_mix_out_played], history)
    by_id = {item.track_id: item for item in items}

    # weight 1.0 * (1 + 2*2 + 1.5*0 + 0.5) = 5.5
    assert by_id["U1"].score == pytest.approx(5.5)
    # weight 0.7 * (1 + 0 + 0 + 0) = 0.7
    assert by_id["U2"].score == pytest.approx(0.7)


def test_rank_prep_targets_tiebreak_orders_by_track_id_ascending() -> None:
    t2 = _track("T2", genre="Vaporwave", mix_points=None)
    t1 = _track("T1", genre="Vaporwave", mix_points=None)

    items, _totals = rank_prep_targets([t2, t1], ConceptHistory())
    assert [item.track_id for item in items] == ["T1", "T2"]
    assert items[0].score == items[1].score


# ---------------------------------------------------------------------------
# reason fragments
# ---------------------------------------------------------------------------


def test_rank_prep_targets_reason_singular_demand_fragment() -> None:
    track = _track("R1", genre="Vaporwave", play_count=5, mix_points=None)
    history = ConceptHistory(runs=[_entry("r1", [_record("c1", ["R1"])])])
    items, _totals = rank_prep_targets([track], history)
    assert items[0].reason == "in 1 planned concept · fully uncued"


def test_rank_prep_targets_reason_plural_demand_fragment() -> None:
    track = _track("R2", genre="Vaporwave", play_count=5, mix_points=None)
    history = ConceptHistory(runs=[_entry("r1", [_record("c1", ["R2"]), _record("c2", ["R2"])])])
    items, _totals = rank_prep_targets([track], history)
    assert items[0].reason == "in 2 planned concepts · fully uncued"


def test_rank_prep_targets_reason_no_demand_omits_concept_fragment() -> None:
    track = _track("R3", genre="Vaporwave", play_count=5, mix_points=None)
    items, _totals = rank_prep_targets([track], ConceptHistory())
    assert items[0].reason == "fully uncued"


def test_rank_prep_targets_reason_unplayed_fragment() -> None:
    track = _track("R4", genre="Vaporwave", play_count=0, mix_points=None)
    items, _totals = rank_prep_targets([track], ConceptHistory())
    assert items[0].reason == "unplayed · fully uncued"


def test_rank_prep_targets_reason_no_mix_out_suffix() -> None:
    track = _track(
        "R5",
        genre="Vaporwave",
        play_count=5,
        mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=None),
    )
    items, _totals = rank_prep_targets([track], ConceptHistory())
    assert items[0].reason == "missing mix-out cue"


def test_rank_prep_targets_reason_all_fragments_joined_in_order() -> None:
    track_x = _track("Y1", bpm=172.0, key="4A", play_count=0, mix_points=None)
    strong_a = _track("Y2", bpm=172.0, key="4A", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))
    strong_b = _track("Y3", bpm=172.0, key="4A", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))
    history = ConceptHistory(runs=[_entry("r1", [_record("c1", ["Y1"]), _record("c2", ["Y1"]), _record("c3", ["Y1"])])])

    items, _totals = rank_prep_targets([track_x, strong_a, strong_b], history)
    by_id = {item.track_id: item for item in items}
    assert by_id["Y1"].centrality >= 0.5
    assert by_id["Y1"].reason == "in 3 planned concepts · harmonically central · unplayed · fully uncued"


# ---------------------------------------------------------------------------
# bucket scoping
# ---------------------------------------------------------------------------


def test_rank_prep_targets_bucket_scoping_filters_items_and_totals() -> None:
    dnb = _track("D1", genre="Drum & Bass", mix_points=None)
    dnb_full = _track("D2", genre="Drum & Bass", mix_points=MixPoints(mix_in_secs=0.0, mix_out_secs=90.0))
    techno = _track("T1", genre="Techno", mix_points=None)

    items, totals = rank_prep_targets([dnb, dnb_full, techno], ConceptHistory(), bucket="techno")

    assert [item.track_id for item in items] == ["T1"]
    assert totals == {"techno": (1, 1)}


# ---------------------------------------------------------------------------
# top slicing
# ---------------------------------------------------------------------------


def test_rank_prep_targets_top_slicing_limits_result_count() -> None:
    tracks = [_track(f"S{i}", genre="Vaporwave", mix_points=None, play_count=0 if i == 0 else 5) for i in range(5)]
    # S0 is unplayed -> strictly higher score than the (tied) rest.
    items, _totals = rank_prep_targets(tracks, ConceptHistory(), top=2)
    assert len(items) == 2
    assert items[0].track_id == "S0"


def test_rank_prep_targets_top_zero_returns_empty_but_totals_populated() -> None:
    track = _track("Z1", genre="Drum & Bass", mix_points=None)
    items, totals = rank_prep_targets([track], ConceptHistory(), top=0)
    assert items == []
    assert totals == {"drum_and_bass": (1, 1)}


# ---------------------------------------------------------------------------
# unmapped genre
# ---------------------------------------------------------------------------


def test_rank_prep_targets_unmapped_genre_ranked_with_zero_centrality() -> None:
    track = _track("N1", genre="Vaporwave", mix_points=None)
    items, totals = rank_prep_targets([track], ConceptHistory())
    assert len(items) == 1
    assert items[0].bucket is None
    assert items[0].centrality == 0.0
    assert totals == {}
