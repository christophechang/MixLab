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
