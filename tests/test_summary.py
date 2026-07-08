from __future__ import annotations

import json
from typing import Any

from mixlab.booth_sheet import build_booth_sheet
from mixlab.html_report import render_html_report
from mixlab.models import MixConcept, MixPoints, Track, Transition
from mixlab.summary import build_run_summary


def _track(
    tid: str,
    *,
    artist: str = "Artist",
    title: str = "Title",
    bpm: float = 174.0,
    key: str = "8A",
    energy: int | None = 5,
    duration: int | None = 300,
    intro: float | None = None,
    outro: float | None = None,
    mix_out: float | None = None,
) -> Track:
    mix_points = None
    if intro is not None or outro is not None or mix_out is not None:
        mix_points = MixPoints(mix_in_secs=8.0, mix_out_secs=mix_out, intro_bars=intro, outro_bars=outro)
    return Track(
        track_id=tid,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        energy=energy,
        duration_secs=duration,
        mix_points=mix_points,
    )


def _concept(
    track_ids: list[str],
    *,
    title: str = "Dark Rollers",
    mood: str = "heavy and hypnotic",
    **kwargs: object,
) -> MixConcept:
    return MixConcept(title=title, mood=mood, track_ids=track_ids, **kwargs)  # type: ignore[arg-type]


def _tracks_by_id(tracks: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


def _summary(
    concepts: list[MixConcept],
    tracks_by_id: dict[str, Track],
    **overrides: Any,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "flags": {"genre": "house"},
        "seed": 20260708,
        "validation_warnings": [],
        "run_notes": "",
        "generated_at": "2026-07-08",
    }
    kwargs.update(overrides)
    return build_run_summary(concepts, tracks_by_id, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Golden shape / contract
# ---------------------------------------------------------------------------


def test_build_run_summary_top_level_keys_present_and_schema_version_1() -> None:
    tracks = [_track(str(i)) for i in range(3)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    summary = _summary([concept], _tracks_by_id(tracks))
    assert summary["schemaVersion"] == 1
    for key in ("generatedAt", "seed", "flags", "validationWarnings", "runNotes", "concepts"):
        assert key in summary


def test_build_run_summary_concept_keys_all_present() -> None:
    tracks = [_track(str(i)) for i in range(3)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1", arc_type="wave", direction_type="mood_journey")
    summary = _summary([concept], _tracks_by_id(tracks))
    payload = summary["concepts"][0]  # type: ignore[index]
    for key in (
        "conceptId",
        "title",
        "directionType",
        "arcType",
        "mood",
        "thesis",
        "practicality",
        "tracks",
        "transitions",
        "boothSheet",
        "conceptWarnings",
    ):
        assert key in payload
    assert payload["conceptId"] == "c1"
    assert payload["arcType"] == "wave"
    assert payload["directionType"] == "mood_journey"


def test_build_run_summary_arc_type_null_when_unset() -> None:
    tracks = [_track(str(i)) for i in range(2)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    summary = _summary([concept], _tracks_by_id(tracks))
    assert summary["concepts"][0]["arcType"] is None  # type: ignore[index]


def test_build_run_summary_keys_are_camel_case() -> None:
    tracks = [_track(str(i)) for i in range(3)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    summary = _summary([concept], _tracks_by_id(tracks))
    assert all("_" not in key for key in summary)
    payload = summary["concepts"][0]  # type: ignore[index]
    assert all("_" not in key for key in payload)
    assert all("_" not in key for key in payload["practicality"])
    for track_row in payload["tracks"]:
        assert all("_" not in key for key in track_row)
    for transition_row in payload["transitions"]:
        assert all("_" not in key for key in transition_row)
    for booth_row in payload["boothSheet"]:
        assert all("_" not in key for key in booth_row)


def test_build_run_summary_json_round_trip() -> None:
    tracks = [_track(str(i), intro=8, outro=16) for i in range(4)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    summary = _summary([concept], _tracks_by_id(tracks))
    reloaded = json.loads(json.dumps(summary))
    assert reloaded == summary


def test_build_run_summary_deterministic_for_identical_inputs() -> None:
    tracks = [_track(str(i), intro=8, outro=16) for i in range(4)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    tbi = _tracks_by_id(tracks)
    a = _summary([concept], tbi)
    b = _summary([concept], tbi)
    assert a == b


# ---------------------------------------------------------------------------
# Transitions parity with html_report
# ---------------------------------------------------------------------------


def test_build_run_summary_transitions_mechanisms_appear_in_html_report() -> None:
    tracks = [_track(str(i), bpm=172.0 if i < 2 else 86.0, key="8A") for i in range(4)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    tbi = _tracks_by_id(tracks)
    summary = _summary([concept], tbi)
    html = render_html_report(
        [concept], tbi, report_text="", report_context="", validation_warnings=[], generated_at="2026-07-08"
    )
    transitions_payload = summary["concepts"][0]["transitions"]  # type: ignore[index]
    assert len(transitions_payload) == 3
    for row in transitions_payload:
        assert row["mechanism"] in html
    assert any(row["mechanism"].startswith("halftime lock") for row in transitions_payload)


def test_build_run_summary_transition_risk_type_from_annotation() -> None:
    tracks = [_track(str(i)) for i in range(2)]
    concept = _concept(
        [t.track_id for t in tracks],
        concept_id="c1",
        transitions=[Transition(from_id="0", to_id="1", is_risky=True, risk_type="peak_impact")],
    )
    summary = _summary([concept], _tracks_by_id(tracks))
    assert summary["concepts"][0]["transitions"][0]["riskType"] == "peak_impact"  # type: ignore[index]


def test_build_run_summary_transition_risk_type_null_when_unannotated() -> None:
    tracks = [_track(str(i)) for i in range(2)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    summary = _summary([concept], _tracks_by_id(tracks))
    assert summary["concepts"][0]["transitions"][0]["riskType"] is None  # type: ignore[index]


def test_build_run_summary_transition_blend_label_null_when_no_blend_data() -> None:
    tracks = [_track(str(i)) for i in range(2)]  # no mix_points -> no blend_label
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    summary = _summary([concept], _tracks_by_id(tracks))
    assert summary["concepts"][0]["transitions"][0]["blendLabel"] is None  # type: ignore[index]


# ---------------------------------------------------------------------------
# Booth sheet parity
# ---------------------------------------------------------------------------


def test_build_run_summary_booth_sheet_matches_build_booth_sheet_output() -> None:
    tracks = [_track(str(i), intro=8, outro=16, mix_out=250) for i in range(3)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    tbi = _tracks_by_id(tracks)
    summary = _summary([concept], tbi)
    steps = build_booth_sheet(concept, tbi)
    booth_payload = summary["concepts"][0]["boothSheet"]  # type: ignore[index]
    assert len(booth_payload) == len(steps) == 2
    for row, step in zip(booth_payload, steps, strict=True):
        assert row["index"] == step.index
        assert row["fromId"] == step.from_id
        assert row["toId"] == step.to_id
        assert row["fromLabel"] == step.from_label
        assert row["toLabel"] == step.to_label
        assert row["openAtSecs"] == step.open_at_secs
        assert row["outroBars"] == step.outro_bars
        assert row["introBars"] == step.intro_bars
        assert row["pitchPct"] == step.pitch_pct
        assert row["tier"] == step.tier
        assert row["chips"] == [[text, is_warn] for text, is_warn in step.chips]
        assert row["plan"] == step.plan
        assert row["fallback"] == step.fallback
        assert row["scout"] == step.scout


# ---------------------------------------------------------------------------
# Concept warnings scoping
# ---------------------------------------------------------------------------


def test_build_run_summary_concept_warnings_scoped_by_title_prefix() -> None:
    tracks = [_track(str(i)) for i in range(2)]
    concept_a = _concept([t.track_id for t in tracks], title="Alpha", concept_id="a")
    concept_b = _concept([t.track_id for t in tracks], title="Beta", concept_id="b")
    warnings = ["[Alpha] BPM jump 20.0 between A and B", "[Beta] artist 'X' appears 3 times"]
    summary = _summary([concept_a, concept_b], _tracks_by_id(tracks), validation_warnings=warnings)
    assert summary["concepts"][0]["conceptWarnings"] == ["[Alpha] BPM jump 20.0 between A and B"]  # type: ignore[index]
    assert summary["concepts"][1]["conceptWarnings"] == ["[Beta] artist 'X' appears 3 times"]  # type: ignore[index]


def test_build_run_summary_validation_warnings_carried_at_top_level() -> None:
    tracks = [_track(str(i)) for i in range(2)]
    concept = _concept([t.track_id for t in tracks], concept_id="c1")
    warnings = ["[Dark Rollers] some warning"]
    summary = _summary([concept], _tracks_by_id(tracks), validation_warnings=warnings)
    assert summary["validationWarnings"] == warnings


# ---------------------------------------------------------------------------
# Missing tracks skipped consistently with html_report
# ---------------------------------------------------------------------------


def test_build_run_summary_missing_tracks_skipped_like_html_report() -> None:
    tracks = [_track(str(i)) for i in [1, 2]]
    concept = _concept(["1", "999", "2"], concept_id="c1")  # "999" is not resolvable
    tbi = _tracks_by_id(tracks)
    summary = _summary([concept], tbi)
    track_ids_in_payload = [row["id"] for row in summary["concepts"][0]["tracks"]]  # type: ignore[index]
    assert track_ids_in_payload == ["1", "2"]
    # Two resolved tracks -> exactly one bridged transition, matching html_report.
    assert len(summary["concepts"][0]["transitions"]) == 1  # type: ignore[index]
