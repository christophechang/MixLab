from __future__ import annotations

import json
from pathlib import Path

import pytest

from mixlab.history import (
    _NOVELTY_SHAPE_WEIGHT,
    _NOVELTY_TRACK_WEIGHT,
    ConceptHistory,
    ConceptRecord,
    HistoryEntry,
    append_run,
    concept_shape_from_canvas,
    concept_shape_from_entry,
    concept_shape_from_record,
    concept_shape_similarity,
    load_history,
    save_history,
    similarity_breakdown_to_history,
    similarity_to_history,
)
from mixlab.models import (
    CanvasRoleCandidates,
    CanvasScore,
    ConceptShape,
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
    # Roles populated so the shape fingerprint matches `_entry` (has_opener/closer True).
    # This keeps the existing similarity_to_history tests focused on track-overlap
    # behaviour against a fixed matching shape; shape-specific tests below vary it.
    return MixCanvas(
        canvas_id="dnb_172.0_4A",
        genre="drum_and_bass",
        bpm_range=(166.0, 178.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(
            opener=core_ids[:1],
            groove_locker=[],
            builder=[],
            pivot=[],
            peak=[],
            closer=core_ids[-1:],
        ),
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


def test_similarity_to_history_disjoint_tracks_same_shape() -> None:
    """Disjoint tracks but matching concept shape — shape contributes 0.35 (#7)."""
    canvas = _canvas(["T001", "T002"])
    history = ConceptHistory(runs=[_entry("r1", ["T010", "T011"])])
    sim = similarity_to_history(canvas, history)
    # track_sim=0.0, shape_sim=1.0, age=0 → combined = 0.35
    assert sim == pytest.approx(_NOVELTY_SHAPE_WEIGHT)


def test_similarity_to_history_empty_history() -> None:
    canvas = _canvas(["T001"])
    sim = similarity_to_history(canvas, ConceptHistory())
    assert sim == pytest.approx(0.0)


def test_similarity_to_history_decay() -> None:
    """Older entries contribute less similarity than recent ones."""
    from mixlab.history import _DECAY

    canvas = _canvas(["T001", "T002", "T003"])
    old_entry = _entry("r_old", ["T001", "T002", "T003"])
    recent_entry = _entry("r_recent", ["X001", "X002", "X003"])  # disjoint tracks but same shape
    # old_entry is age=1 (second from end in reversed order), recent is age=0
    history = ConceptHistory(runs=[old_entry, recent_entry])
    sim = similarity_to_history(canvas, history)
    # old_entry: combined=1.0, decayed by _DECAY^1 = 0.8
    # recent_entry: combined=0.35 (shape only), age=0 = 0.35
    # max picks the older identical match.
    assert sim == pytest.approx(1.0 * _DECAY)


def test_similarity_to_history_partial_overlap_same_shape() -> None:
    canvas = _canvas(["T001", "T002", "T003", "T004"])
    history = ConceptHistory(runs=[_entry("r1", ["T001", "T002", "X001", "X002"])])
    sim = similarity_to_history(canvas, history)
    # track jaccard=2/6=1/3, shape=1.0 → combined = 0.65*(1/3) + 0.35
    expected = _NOVELTY_TRACK_WEIGHT * (2 / 6) + _NOVELTY_SHAPE_WEIGHT * 1.0
    assert sim == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Concept-shape novelty (#7)
# ---------------------------------------------------------------------------


def _shape(
    bpm: str = "170-180",
    cam: str = "4A",
    arc: str = "",
    opener: bool = True,
    closer: bool = True,
    peak: bool = False,
) -> ConceptShape:
    return ConceptShape(
        bpm_band=bpm,
        camelot_zone=cam,
        energy_path=arc,
        has_opener=opener,
        has_closer=closer,
        has_peak=peak,
    )


def test_concept_shape_similarity_identical() -> None:
    s = _shape()
    assert concept_shape_similarity(s, s) == pytest.approx(1.0)


def test_concept_shape_similarity_fully_different() -> None:
    a = _shape(bpm="120-130", cam="8A", opener=True, closer=True, peak=True)
    b = _shape(bpm="170-180", cam="4A", opener=False, closer=False, peak=False)
    # 5 comparable fields (energy_path excluded both ""), all mismatch.
    assert concept_shape_similarity(a, b) == pytest.approx(0.0)


def test_concept_shape_similarity_partial() -> None:
    a = _shape(bpm="170-180", cam="4A", opener=True, closer=True, peak=False)
    b = _shape(bpm="170-180", cam="8A", opener=False, closer=True, peak=False)
    # 5 fields compared: bpm match, camelot mismatch, has_opener mismatch,
    # has_closer match, has_peak match → 3/5.
    assert concept_shape_similarity(a, b) == pytest.approx(3 / 5)


def test_concept_shape_similarity_excludes_empty_energy_path() -> None:
    """Empty energy_path is excluded — unknown is not a match (Option A)."""
    a = _shape(arc="")
    b = _shape(arc="")
    # 5 fields compared (energy_path excluded both sides). All other fields match → 1.0.
    assert concept_shape_similarity(a, b) == pytest.approx(1.0)


def test_concept_shape_similarity_compares_energy_path_when_both_populated() -> None:
    a = _shape(arc="wave")
    b = _shape(arc="single_arc")
    # 6 fields compared, energy_path mismatch, rest match → 5/6.
    assert concept_shape_similarity(a, b) == pytest.approx(5 / 6)


def test_concept_shape_from_canvas_uses_bpm_bucket() -> None:
    canvas = _canvas(["T001", "T002"])
    shape = concept_shape_from_canvas(canvas)
    assert shape.bpm_band == "170-180"
    assert shape.camelot_zone == "4A"
    # Canvas-side shape never has energy_path (Stage 2 has not run).
    assert shape.energy_path == ""
    assert shape.has_opener is True
    assert shape.has_closer is True
    assert shape.has_peak is False


def test_concept_shape_from_entry_pulls_arc_type() -> None:
    entry = _entry("r1", ["T001", "T002"])
    shape = concept_shape_from_entry(entry)
    assert shape.energy_path == "single_arc"
    assert shape.bpm_band == "170-180"


def test_similarity_breakdown_decomposes_track_and_shape() -> None:
    """Debug breakdown reports both components and which run drove the score."""
    canvas = _canvas(["T001", "T002", "T003"])
    history = ConceptHistory(runs=[_entry("r1", ["T001", "T002", "T003"])])
    breakdown = similarity_breakdown_to_history(canvas, history)
    assert breakdown.combined == pytest.approx(1.0)
    assert breakdown.track_similarity == pytest.approx(1.0)
    assert breakdown.shape_similarity == pytest.approx(1.0)
    assert breakdown.age_of_top_match == 0


def test_similarity_breakdown_isolates_shape_only_match() -> None:
    """Disjoint tracks but matching shape — combined is shape-only."""
    canvas = _canvas(["T001"])
    history = ConceptHistory(runs=[_entry("r1", ["X999"])])
    breakdown = similarity_breakdown_to_history(canvas, history)
    assert breakdown.track_similarity == pytest.approx(0.0)
    assert breakdown.shape_similarity == pytest.approx(1.0)
    assert breakdown.combined == pytest.approx(_NOVELTY_SHAPE_WEIGHT)


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
    # No role assignments — tracks fall through to "unknown". Build a canvas with
    # empty role pools explicitly (the default `_canvas` populates opener/closer for
    # shape-matching in similarity tests).
    canvas = _canvas(["T001", "T002"])
    canvas.roles.opener = []
    canvas.roles.closer = []
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


# ---------------------------------------------------------------------------
# format_recent_concepts — Stage 2 prompt injection helper (#13)
# ---------------------------------------------------------------------------


def test_format_recent_concepts_returns_none_for_empty_history() -> None:
    from mixlab.history import format_recent_concepts

    assert format_recent_concepts(ConceptHistory()) is None


def test_format_recent_concepts_lists_titles_in_reverse_chronological_order() -> None:
    from mixlab.history import format_recent_concepts

    e1 = _entry("r1", ["T001"], genre="house")
    e1.concept_title = "Older"
    e2 = _entry("r2", ["T002"], genre="house")
    e2.concept_title = "Newer"
    block = format_recent_concepts(ConceptHistory(runs=[e1, e2]))
    assert block is not None
    # Newer appears before Older — reverse chronological.
    assert block.index("Newer") < block.index("Older")


def test_format_recent_concepts_includes_arc_and_mood() -> None:
    from mixlab.history import format_recent_concepts

    entry = _entry("r1", ["T001"], genre="house")
    entry.concept_title = "Late Latitude"
    entry.energy_path = "wave"
    entry.mood = "dark"
    block = format_recent_concepts(ConceptHistory(runs=[entry]))
    assert block is not None
    assert "Late Latitude" in block
    assert "arc: wave" in block
    assert "mood: dark" in block


def test_format_recent_concepts_respects_limit() -> None:
    from mixlab.history import format_recent_concepts

    entries = []
    for i in range(10):
        e = _entry(f"r{i}", [f"T{i:03d}"], genre="house")
        e.concept_title = f"Concept{i}"
        entries.append(e)
    block = format_recent_concepts(ConceptHistory(runs=entries), limit=3)
    assert block is not None
    # Only the three most recent (Concept9, Concept8, Concept7) should appear.
    assert "Concept9" in block
    assert "Concept8" in block
    assert "Concept7" in block
    assert "Concept6" not in block
    assert "Concept0" not in block


def test_format_recent_concepts_includes_divergence_instruction() -> None:
    """The block ends with a paragraph telling the model to diverge from prior runs."""
    from mixlab.history import format_recent_concepts

    block = format_recent_concepts(ConceptHistory(runs=[_entry("r1", ["T001"])]))
    assert block is not None
    assert "Diverge deliberately" in block


# ---------------------------------------------------------------------------
# ConceptRecord — full per-run concept list (#52)
# ---------------------------------------------------------------------------


def _record(
    concept_id: str,
    title: str,
    *,
    track_ids: list[str] | None = None,
    mood: str = "dark",
    arc_type: str = "",
    role_pattern: list[str] | None = None,
    feedback: str = "",
) -> ConceptRecord:
    return ConceptRecord(
        concept_id=concept_id,
        title=title,
        mood=mood,
        track_ids=track_ids if track_ids is not None else [],
        arc_type=arc_type,
        role_pattern=role_pattern if role_pattern is not None else [],
        feedback=feedback,
    )


def test_append_run_round_trip_preserves_multiple_concept_records(tmp_path: Path) -> None:
    """A run with 3 ConceptRecords survives a save/load round-trip with all fields intact."""
    p = tmp_path / "history.json"
    entry = _entry("r1", ["T001", "T002"])
    entry.concepts = [
        _record("c1", "First", track_ids=["T001"], arc_type="wave", role_pattern=["opener"]),
        _record(
            "c2",
            "Second",
            track_ids=["T002"],
            mood="light",
            arc_type="plateau",
            role_pattern=["closer"],
            feedback="played",
        ),
        _record("c3", "Third", track_ids=["T001", "T002"]),
    ]
    append_run(ConceptHistory(), entry, p)
    reloaded = load_history(p)

    assert len(reloaded.runs) == 1
    assert len(reloaded.runs[0].concepts) == 3
    assert [c.title for c in reloaded.runs[0].concepts] == ["First", "Second", "Third"]
    assert reloaded.runs[0].concepts[1].feedback == "played"
    assert reloaded.runs[0].concepts[1].mood == "light"
    assert reloaded.runs[0].concepts[0].role_pattern == ["opener"]


def test_load_history_old_format_without_concepts_key_loads_empty_list(tmp_path: Path) -> None:
    """Pre-#52 history files have no 'concepts' key at all — must load with concepts=[]."""
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
    assert history.runs[0].concepts == []
    # Legacy fields remain intact.
    assert history.runs[0].concept_title == "Old"
    assert history.runs[0].concept_track_ids == ["T001", "T002"]


def test_load_history_coerces_concept_dicts_missing_optional_fields(tmp_path: Path) -> None:
    """A stored concept dict missing feedback fields loads with their defaults."""
    p = tmp_path / "history.json"
    entry = _entry("r1", ["T001"])
    entry.concepts = [_record("c1", "First", track_ids=["T001"])]
    p.write_text(json.dumps({"runs": [_history_entry_dict(entry)]}))
    history = load_history(p)
    assert history.runs[0].concepts[0].feedback == ""
    assert history.runs[0].concepts[0].feedback_notes == ""


def _history_entry_dict(entry: HistoryEntry) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(entry)


def test_save_history_persists_without_appending(tmp_path: Path) -> None:
    """save_history writes the file as-is — no append/truncate semantics."""
    p = tmp_path / "history.json"
    history = ConceptHistory(runs=[_entry("r1", ["T001"]), _entry("r2", ["T002"])])
    save_history(history, p)
    reloaded = load_history(p)
    assert [r.run_id for r in reloaded.runs] == ["r1", "r2"]


def test_save_history_persists_mutated_feedback(tmp_path: Path) -> None:
    """Round-trips a feedback edit made in-place on a loaded history (the --feedback flow)."""
    p = tmp_path / "history.json"
    entry = _entry("r1", ["T001"])
    entry.concepts = [_record("c1", "First", track_ids=["T001"])]
    save_history(ConceptHistory(runs=[entry]), p)

    reloaded = load_history(p)
    reloaded.runs[0].concepts[0].feedback = "played"
    reloaded.runs[0].concepts[0].feedback_notes = "great opener"
    save_history(reloaded, p)

    reloaded_again = load_history(p)
    assert reloaded_again.runs[0].concepts[0].feedback == "played"
    assert reloaded_again.runs[0].concepts[0].feedback_notes == "great opener"


def test_concept_shape_from_record_uses_entry_dominant_lists_and_record_arc_role() -> None:
    entry = _entry("r1", ["T001", "T002"])
    record = _record("c1", "T", track_ids=["T001"], arc_type="build-and-drop", role_pattern=["peak"])
    shape = concept_shape_from_record(record, entry)
    assert shape.bpm_band == "170-180"
    assert shape.camelot_zone == "4A"
    assert shape.energy_path == "build-and-drop"
    assert shape.has_peak is True


# ---------------------------------------------------------------------------
# format_recent_concepts — lists every concept a run produced (#52)
# ---------------------------------------------------------------------------


def test_format_recent_concepts_lists_all_concepts_in_a_multi_concept_run() -> None:
    from mixlab.history import format_recent_concepts

    entry = _entry("r1", ["T001"], genre="house")
    entry.concepts = [
        _record("c1", "First", track_ids=["T001"], arc_type="wave"),
        _record("c2", "Second", track_ids=["T002"], arc_type="plateau"),
        _record("c3", "Third", track_ids=["T003"]),
    ]
    block = format_recent_concepts(ConceptHistory(runs=[entry]))
    assert block is not None
    assert "First" in block
    assert "Second" in block
    assert "Third" in block


def test_format_recent_concepts_dedupes_by_lowercased_title_keeping_most_recent() -> None:
    from mixlab.history import format_recent_concepts

    e1 = _entry("r1", ["T001"], genre="house")
    e1.concepts = [_record("c1", "Midnight Run", track_ids=["T001"], mood="dark")]
    e2 = _entry("r2", ["T002"], genre="house")
    e2.concepts = [_record("c2", "midnight run", track_ids=["T002"], mood="light")]
    block = format_recent_concepts(ConceptHistory(runs=[e1, e2]))
    assert block is not None
    # Only one line for the title overall — the more recent run (e2, "light" mood) wins.
    assert block.lower().count("midnight run") == 1
    assert "mood: light" in block
    assert "mood: dark" not in block


def test_format_recent_concepts_respects_limit_across_multiple_concepts_per_run() -> None:
    from mixlab.history import format_recent_concepts

    entry = _entry("r1", ["T001"], genre="house")
    entry.concepts = [_record(f"c{i}", f"Concept{i}", track_ids=[f"T{i:03d}"]) for i in range(10)]
    block = format_recent_concepts(ConceptHistory(runs=[entry]), limit=3)
    assert block is not None
    assert "Concept0" in block
    assert "Concept1" in block
    assert "Concept2" in block
    assert "Concept3" not in block


def test_format_recent_concepts_falls_back_to_legacy_fields_when_concepts_empty() -> None:
    from mixlab.history import format_recent_concepts

    entry = _entry("r1", ["T001"], genre="house")
    entry.concept_title = "Legacy Title"
    entry.energy_path = "wave"
    entry.mood = "dark"
    assert entry.concepts == []
    block = format_recent_concepts(ConceptHistory(runs=[entry]))
    assert block is not None
    assert "Legacy Title" in block
    assert "arc: wave" in block


def test_recent_concepts_limit_constant_is_eight() -> None:
    from mixlab.history import _RECENT_CONCEPTS_LIMIT

    assert _RECENT_CONCEPTS_LIMIT == 8


# ---------------------------------------------------------------------------
# Shape-novelty over every stored concept, not just the first (#52)
# ---------------------------------------------------------------------------


def test_similarity_breakdown_shape_uses_max_across_concepts_in_a_run() -> None:
    """A second concept whose shape matches the candidate canvas raises shape_sim above
    what a single non-matching first concept alone would produce."""
    canvas = _canvas(["T001", "T002"])
    canvas.roles.peak = ["T001"]  # canvas_shape.has_peak = True

    entry_single = _entry("r1", ["X001", "X002"])  # disjoint tracks -> track_sim = 0.0
    entry_single.concepts = [_record("c1", "No Peak", track_ids=["X001"], role_pattern=[])]
    breakdown_single = similarity_breakdown_to_history(canvas, ConceptHistory(runs=[entry_single]))

    entry_multi = _entry("r1", ["X001", "X002"])
    entry_multi.concepts = [
        _record("c1", "No Peak", track_ids=["X001"], role_pattern=[]),
        _record("c2", "Has Peak", track_ids=["X002"], role_pattern=["peak"]),
    ]
    breakdown_multi = similarity_breakdown_to_history(canvas, ConceptHistory(runs=[entry_multi]))

    assert breakdown_multi.shape_similarity > breakdown_single.shape_similarity
    assert breakdown_multi.combined > breakdown_single.combined


def test_similarity_breakdown_falls_back_to_entry_shape_when_concepts_empty() -> None:
    """No concept records at all (legacy entry) -> behaviour unchanged from pre-#52."""
    canvas = _canvas(["T001", "T002", "T003"])
    entry = _entry("r1", ["T001", "T002", "T003"])
    assert entry.concepts == []
    breakdown = similarity_breakdown_to_history(canvas, ConceptHistory(runs=[entry]))
    assert breakdown.combined == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feedback multiplier (#12 discovery doc / #52)
# ---------------------------------------------------------------------------


def test_feedback_multiplier_played_scales_up() -> None:
    from mixlab.history import _feedback_multiplier

    entry = _entry("r1", ["T001"])
    entry.concepts = [_record("c1", "T", track_ids=["T001"], feedback="played")]
    assert _feedback_multiplier(entry) == pytest.approx(1.5)


def test_feedback_multiplier_played_modified_scales_up() -> None:
    from mixlab.history import _feedback_multiplier

    entry = _entry("r1", ["T001"])
    entry.concepts = [_record("c1", "T", track_ids=["T001"], feedback="played_modified")]
    assert _feedback_multiplier(entry) == pytest.approx(1.5)


def test_feedback_multiplier_all_rejected_scales_down() -> None:
    from mixlab.history import _feedback_multiplier

    entry = _entry("r1", ["T001"])
    entry.concepts = [
        _record("c1", "A", track_ids=["T001"], feedback="rejected"),
        _record("c2", "B", track_ids=["T002"], feedback="rejected"),
    ]
    assert _feedback_multiplier(entry) == pytest.approx(0.25)


def test_feedback_multiplier_rejected_plus_unset_still_scales_down() -> None:
    """A rejected concept alongside an as-yet-unrated one is still all-rejected among
    the non-empty verdicts — 0.25x, not diluted back to neutral."""
    from mixlab.history import _feedback_multiplier

    entry = _entry("r1", ["T001"])
    entry.concepts = [
        _record("c1", "A", track_ids=["T001"], feedback="rejected"),
        _record("c2", "B", track_ids=["T002"], feedback=""),
    ]
    assert _feedback_multiplier(entry) == pytest.approx(0.25)


def test_feedback_multiplier_mixed_rejected_and_unused_is_neutral() -> None:
    """A genuinely mixed bag (rejected + unused, neither played) falls through to 1.0 —
    only an all-rejected set of non-empty verdicts triggers the 0.25x discount."""
    from mixlab.history import _feedback_multiplier

    entry = _entry("r1", ["T001"])
    entry.concepts = [
        _record("c1", "A", track_ids=["T001"], feedback="rejected"),
        _record("c2", "B", track_ids=["T002"], feedback="unused"),
    ]
    assert _feedback_multiplier(entry) == pytest.approx(1.0)


def test_feedback_multiplier_no_concepts_is_neutral() -> None:
    from mixlab.history import _feedback_multiplier

    entry = _entry("r1", ["T001"])
    assert _feedback_multiplier(entry) == pytest.approx(1.0)


def test_similarity_breakdown_clamps_played_multiplier_to_one() -> None:
    """Perfect track+shape match with a 'played' concept: 1.0 * 1.5 must clamp to 1.0."""
    canvas = _canvas(["T001", "T002", "T003"])
    entry = _entry("r1", ["T001", "T002", "T003"])
    entry.concepts = [
        _record(
            "c1",
            "T",
            track_ids=["T001", "T002", "T003"],
            arc_type="single_arc",
            role_pattern=["opener", "closer"],
            feedback="played",
        )
    ]
    breakdown = similarity_breakdown_to_history(canvas, ConceptHistory(runs=[entry]))
    assert breakdown.feedback_multiplier == pytest.approx(1.5)
    assert breakdown.combined == pytest.approx(1.0)


def test_similarity_breakdown_scales_shape_only_match_by_rejected_multiplier() -> None:
    """Shape-only match (0.35 combined) with an all-rejected concept scales to 0.25x."""
    canvas = _canvas(["T001"])
    entry = _entry("r1", ["X999"])  # disjoint tracks -> track_sim = 0.0, shape matches by default
    entry.concepts = [_record("c1", "T", track_ids=["X999"], feedback="rejected")]
    breakdown = similarity_breakdown_to_history(canvas, ConceptHistory(runs=[entry]))
    assert breakdown.feedback_multiplier == pytest.approx(0.25)
    assert breakdown.combined == pytest.approx(_NOVELTY_SHAPE_WEIGHT * 0.25)


# ---------------------------------------------------------------------------
# recent_concept_titles (#75) — name-avoid list from history
# ---------------------------------------------------------------------------


def test_recent_concept_titles_newest_first_across_runs() -> None:
    from mixlab.history import recent_concept_titles

    older = _entry("r1", ["1"])
    older.concepts = [_record("c1", "Milk & Rust"), _record("c2", "Orbital Debt")]
    newer = _entry("r2", ["2"])
    newer.concepts = [_record("c3", "Heist Recordings"), _record("c4", "Rej & The Room")]

    titles = recent_concept_titles(ConceptHistory(runs=[older, newer]))
    assert titles == ["Heist Recordings", "Rej & The Room", "Milk & Rust", "Orbital Debt"]


def test_recent_concept_titles_dedupes_case_insensitively() -> None:
    from mixlab.history import recent_concept_titles

    a = _entry("r1", ["1"])
    a.concepts = [_record("c1", "heist recordings")]
    b = _entry("r2", ["2"])
    b.concepts = [_record("c2", "Heist Recordings"), _record("c3", "Fresh Name")]

    titles = recent_concept_titles(ConceptHistory(runs=[a, b]))
    assert titles == ["Heist Recordings", "Fresh Name"]


def test_recent_concept_titles_legacy_entry_falls_back_to_concept_title() -> None:
    from mixlab.history import recent_concept_titles

    legacy = _entry("r1", ["1"])  # no .concepts list — pre-#52 shape
    legacy.concept_title = "Old Single Title"

    assert recent_concept_titles(ConceptHistory(runs=[legacy])) == ["Old Single Title"]


def test_recent_concept_titles_limit_runs_skips_older_entries() -> None:
    from mixlab.history import recent_concept_titles

    entries = []
    for i in range(4):
        e = _entry(f"r{i}", [str(i)])
        e.concepts = [_record(f"c{i}", f"Title {i}")]
        entries.append(e)

    titles = recent_concept_titles(ConceptHistory(runs=entries), limit_runs=2)
    assert titles == ["Title 3", "Title 2"]


def test_recent_concept_titles_empty_history_returns_empty() -> None:
    from mixlab.history import recent_concept_titles

    assert recent_concept_titles(ConceptHistory()) == []
