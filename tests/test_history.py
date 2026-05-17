from __future__ import annotations

import json
from pathlib import Path

import pytest

from mixlab.history import (
    ConceptHistory,
    HistoryEntry,
    append_run,
    load_history,
    similarity_to_history,
)
from mixlab.models import (
    CanvasRoleCandidates,
    CanvasScore,
    ContrastAssets,
    MixCanvas,
    MixConcept,
)


def _entry(run_id: str, core_ids: list[str], genre: str = "drum_and_bass") -> HistoryEntry:
    return HistoryEntry(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        mode="standard",
        genre=genre,
        selected_canvas_ids=["canvas_1"],
        dominant_bpm_clusters=[172.0],
        dominant_camelot_keys=["4A"],
        core_track_ids=core_ids,
        anchor_track_ids=core_ids[:1],
        opener_candidates=core_ids[:1],
        closer_candidates=core_ids[-1:],
        concept_title="Test",
        concept_track_ids=core_ids,
        energy_path="single_arc",
        mood="dark",
        rating=None,
    )


def _canvas(core_ids: list[str]) -> MixCanvas:
    return MixCanvas(
        canvas_id="dnb_172.0_4A",
        genre="drum_and_bass",
        bpm_range=(166.0, 178.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[]),
        contrast=ContrastAssets(
            vocal_moments=[],
            texture_changes=[],
            darker_turns=[],
            brighter_lifts=[],
            lower_pressure_resets=[],
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=MixConcept(title="Test", mood="dark", track_ids=core_ids),
    )


def test_load_history_missing_file(tmp_path: Path) -> None:
    history = load_history(tmp_path / "no-such-file.json")
    assert history.runs == []


def test_load_history_corrupt_json(tmp_path: Path) -> None:
    p = tmp_path / "history.json"
    p.write_text("not json{{")
    history = load_history(p)
    assert history.runs == []


def test_load_history_valid(tmp_path: Path) -> None:
    p = tmp_path / "history.json"
    entry = _entry("r1", ["T001", "T002"])
    append_run(ConceptHistory(), entry, p)
    history = load_history(p)
    assert len(history.runs) == 1
    assert history.runs[0].run_id == "r1"
    assert history.runs[0].core_track_ids == ["T001", "T002"]


def test_append_run_creates_file(tmp_path: Path) -> None:
    p = tmp_path / ".mixlab" / "concept-history.json"
    assert not p.exists()
    history = ConceptHistory()
    append_run(history, _entry("r1", ["T001"]), p)
    assert p.exists()
    data = json.loads(p.read_text())
    assert len(data["runs"]) == 1


def test_append_run_truncates_at_max(tmp_path: Path) -> None:
    from mixlab.history import _MAX_HISTORY

    p = tmp_path / "history.json"
    history = ConceptHistory()
    for i in range(_MAX_HISTORY + 10):
        append_run(history, _entry(f"r{i}", [f"T{i:03d}"]), p)
    reloaded = load_history(p)
    assert len(reloaded.runs) == _MAX_HISTORY


def test_similarity_to_history_identical() -> None:
    canvas = _canvas(["T001", "T002", "T003"])
    history = ConceptHistory(runs=[_entry("r1", ["T001", "T002", "T003"])])
    sim = similarity_to_history(canvas, history)
    assert sim == pytest.approx(1.0)


def test_similarity_to_history_disjoint() -> None:
    canvas = _canvas(["T001", "T002"])
    history = ConceptHistory(runs=[_entry("r1", ["T010", "T011"])])
    sim = similarity_to_history(canvas, history)
    assert sim == pytest.approx(0.0)


def test_similarity_to_history_empty_history() -> None:
    canvas = _canvas(["T001"])
    sim = similarity_to_history(canvas, ConceptHistory())
    assert sim == pytest.approx(0.0)


def test_similarity_to_history_decay() -> None:
    """Older entries contribute less similarity than recent ones."""
    from mixlab.history import _DECAY

    canvas = _canvas(["T001", "T002", "T003"])
    old_entry = _entry("r_old", ["T001", "T002", "T003"])
    recent_entry = _entry("r_recent", ["X001", "X002", "X003"])  # disjoint
    # old_entry is age=1 (second from end in reversed order), recent is age=0 but disjoint
    history = ConceptHistory(runs=[old_entry, recent_entry])
    sim = similarity_to_history(canvas, history)
    # old_entry jaccard=1.0 decayed by _DECAY^1, recent=0.0
    assert sim == pytest.approx(1.0 * _DECAY)


def test_similarity_to_history_partial_overlap() -> None:
    canvas = _canvas(["T001", "T002", "T003", "T004"])
    history = ConceptHistory(runs=[_entry("r1", ["T001", "T002", "X001", "X002"])])
    sim = similarity_to_history(canvas, history)
    # intersection=2, union=6, jaccard=1/3, age=0 → no decay
    assert sim == pytest.approx(2 / 6)


def test_history_entry_from_run() -> None:
    concept = MixConcept(title="Midnight", mood="dark", track_ids=["T001", "T002"])
    canvas = _canvas(["T001", "T002"])
    canvas.roles.opener = ["T001"]
    canvas.roles.closer = ["T002"]
    entry = HistoryEntry.from_run([canvas], [concept], genre="drum_and_bass", mode="standard")
    assert entry.genre == "drum_and_bass"
    assert entry.mode == "standard"
    assert entry.concept_title == "Midnight"
    assert "T001" in entry.core_track_ids
    assert "T001" in entry.opener_candidates
    assert "T002" in entry.closer_candidates


# ---------------------------------------------------------------------------
# v0.10 schema additions — canvas score breakdown, risk notes, bpm band,
# role pattern, energy_path fix (#6)
# ---------------------------------------------------------------------------


def test_from_run_populates_energy_path_from_arc_type() -> None:
    """Regression fix: energy_path was always '' before #6. Now reads from concept.arc_type."""
    concept = MixConcept(title="Wave Set", mood="dark", track_ids=["T001", "T002"], arc_type="wave")
    canvas = _canvas(["T001", "T002"])
    entry = HistoryEntry.from_run([canvas], [concept], genre="house", mode="standard")
    assert entry.energy_path == "wave"


def test_from_run_energy_path_empty_when_arc_type_none() -> None:
    """Graceful fallback: when Stage 2 emits no arc_type (e.g. parser dropped it as invalid)."""
    concept = MixConcept(title="No Arc", mood="dark", track_ids=["T001", "T002"], arc_type=None)
    canvas = _canvas(["T001", "T002"])
    entry = HistoryEntry.from_run([canvas], [concept], genre="house", mode="standard")
    assert entry.energy_path == ""


def test_from_run_populates_canvas_score_breakdown() -> None:
    """Each scoring dimension from CanvasScore is captured as a numeric value."""
    canvas = _canvas(["T001", "T002"])
    canvas.score = CanvasScore(
        technical_viability=0.85,
        role_coverage=0.60,
        anchor_strength=0.5,
        contrast_potential=0.4,
        distinctiveness=0.9,
        novelty=0.7,
        weakness_penalty=0.04,
        overall=0.65,
    )
    concept = MixConcept(title="T", mood="m", track_ids=["T001", "T002"])
    entry = HistoryEntry.from_run([canvas], [concept], genre="house", mode="standard")
    assert entry.canvas_score_breakdown["technical_viability"] == pytest.approx(0.85)
    assert entry.canvas_score_breakdown["role_coverage"] == pytest.approx(0.60)
    assert entry.canvas_score_breakdown["overall"] == pytest.approx(0.65)
    assert entry.canvas_score_breakdown["weakness_penalty"] == pytest.approx(0.04)


def test_from_run_aggregates_score_across_canvases_as_mean() -> None:
    canvas_a = _canvas(["T001"])
    canvas_a.score = CanvasScore(overall=0.4)
    canvas_b = _canvas(["T002"])
    canvas_b.score = CanvasScore(overall=0.8)
    concept = MixConcept(title="T", mood="m", track_ids=["T001"])
    entry = HistoryEntry.from_run([canvas_a, canvas_b], [concept], genre="house", mode="standard")
    assert entry.canvas_score_breakdown["overall"] == pytest.approx(0.6)


def test_from_run_collects_risk_notes_across_canvases() -> None:
    canvas_a = _canvas(["T001", "T002"])
    canvas_a.risk_notes = ["weak opener pool", "over-repeated artist"]
    canvas_b = _canvas(["T003", "T004"])
    canvas_b.risk_notes = ["all high energy"]
    concept = MixConcept(title="T", mood="m", track_ids=["T001", "T003"])
    entry = HistoryEntry.from_run([canvas_a, canvas_b], [concept], genre="house", mode="standard")
    assert entry.selected_canvas_risk_notes == ["weak opener pool", "over-repeated artist", "all high energy"]


def test_from_run_computes_bpm_band_across_canvases() -> None:
    canvas_a = _canvas(["T001"])
    canvas_a.bpm_range = (120.0, 124.0)
    canvas_b = _canvas(["T002"])
    canvas_b.bpm_range = (126.0, 130.0)
    concept = MixConcept(title="T", mood="m", track_ids=["T001", "T002"])
    entry = HistoryEntry.from_run([canvas_a, canvas_b], [concept], genre="house", mode="standard")
    assert entry.bpm_band == (120.0, 130.0)


def test_from_run_populates_role_pattern_from_canvas_role_pools() -> None:
    canvas = _canvas(["T001", "T002", "T003", "T004"])
    canvas.roles.opener = ["T001"]
    canvas.roles.builder = ["T002"]
    canvas.roles.peak = ["T003"]
    canvas.roles.closer = ["T004"]
    concept = MixConcept(title="T", mood="m", track_ids=["T001", "T002", "T003", "T004"])
    entry = HistoryEntry.from_run([canvas], [concept], genre="house", mode="standard")
    assert entry.role_pattern == ["opener", "builder", "peak", "closer"]


def test_from_run_role_pattern_uses_unknown_for_unclassified_tracks() -> None:
    canvas = _canvas(["T001", "T002"])
    # No role assignments — tracks fall through to "unknown".
    concept = MixConcept(title="T", mood="m", track_ids=["T001", "T002"])
    entry = HistoryEntry.from_run([canvas], [concept], genre="house", mode="standard")
    assert entry.role_pattern == ["unknown", "unknown"]


def test_load_history_handles_old_format_missing_new_fields(tmp_path: Path) -> None:
    """Pre-v0.10 history files lack canvas_score_breakdown/bpm_band/etc. — must load with defaults."""
    p = tmp_path / "history.json"
    old_entry = {
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "mode": "standard",
        "genre": "house",
        "selected_canvas_ids": ["c1"],
        "dominant_bpm_clusters": [124.0],
        "dominant_camelot_keys": ["8A"],
        "core_track_ids": ["T001", "T002"],
        "anchor_track_ids": ["T001"],
        "opener_candidates": ["T001"],
        "closer_candidates": ["T002"],
        "concept_title": "Old",
        "concept_track_ids": ["T001", "T002"],
        "energy_path": "",
        "mood": "dark",
        "rating": None,
    }
    p.write_text(json.dumps({"runs": [old_entry]}))
    history = load_history(p)
    assert len(history.runs) == 1
    assert history.runs[0].canvas_score_breakdown == {}
    assert history.runs[0].selected_canvas_risk_notes == []
    assert history.runs[0].bpm_band == (0.0, 0.0)
    assert history.runs[0].role_pattern == []


def test_load_history_tolerates_extra_unknown_fields(tmp_path: Path) -> None:
    """Future-format fields must not crash the loader — they are simply ignored."""
    p = tmp_path / "history.json"
    future_entry = {
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "mode": "standard",
        "genre": "house",
        "selected_canvas_ids": [],
        "dominant_bpm_clusters": [],
        "dominant_camelot_keys": [],
        "core_track_ids": [],
        "anchor_track_ids": [],
        "opener_candidates": [],
        "closer_candidates": [],
        "concept_title": "",
        "concept_track_ids": [],
        "energy_path": "",
        "mood": "",
        "future_field_X": "ignored",
        "future_field_Y": [1, 2, 3],
    }
    p.write_text(json.dumps({"runs": [future_entry]}))
    history = load_history(p)
    assert len(history.runs) == 1
    assert not hasattr(history.runs[0], "future_field_X")


def test_load_history_coerces_bpm_band_list_back_to_tuple(tmp_path: Path) -> None:
    """JSON serialises tuples as lists. Loader must coerce bpm_band back to tuple."""
    p = tmp_path / "history.json"
    entry_payload = {
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "mode": "standard",
        "genre": "house",
        "selected_canvas_ids": [],
        "dominant_bpm_clusters": [],
        "dominant_camelot_keys": [],
        "core_track_ids": [],
        "anchor_track_ids": [],
        "opener_candidates": [],
        "closer_candidates": [],
        "concept_title": "",
        "concept_track_ids": [],
        "energy_path": "",
        "mood": "",
        "bpm_band": [120.0, 130.0],  # serialised as list
    }
    p.write_text(json.dumps({"runs": [entry_payload]}))
    history = load_history(p)
    assert history.runs[0].bpm_band == (120.0, 130.0)
    assert isinstance(history.runs[0].bpm_band, tuple)


def test_append_run_round_trip_preserves_new_fields(tmp_path: Path) -> None:
    """A run written with v0.10 fields must be loadable with all fields preserved."""
    p = tmp_path / "history.json"
    canvas = _canvas(["T001", "T002"])
    canvas.score = CanvasScore(technical_viability=0.9, overall=0.75)
    canvas.risk_notes = ["weak opener pool"]
    canvas.bpm_range = (122.0, 126.0)
    canvas.roles.opener = ["T001"]
    canvas.roles.closer = ["T002"]
    concept = MixConcept(title="Round Trip", mood="dark", track_ids=["T001", "T002"], arc_type="wave")
    entry = HistoryEntry.from_run([canvas], [concept], genre="house", mode="standard")
    append_run(ConceptHistory(), entry, p)
    reloaded = load_history(p)
    assert reloaded.runs[0].energy_path == "wave"
    assert reloaded.runs[0].canvas_score_breakdown["overall"] == pytest.approx(0.75)
    assert reloaded.runs[0].selected_canvas_risk_notes == ["weak opener pool"]
    assert reloaded.runs[0].bpm_band == (122.0, 126.0)
    assert reloaded.runs[0].role_pattern == ["opener", "closer"]
