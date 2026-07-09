"""Structured run-summary artifact (#89 / M1, docs/architecture/mixlab-anywhere.md §5.2).

Pure, deterministic serialisation of a completed genre-mode run into the
``summary.json`` contract consumed by the .NET API and the TypeScript SPA (see the
architecture doc for the full external contract). This module intentionally adds no
new scoring, transition, or booth-sheet logic of its own — it reuses the exact
computations the HTML report and booth sheet already use (``_compute_practicality_score``,
``mixlab.transitions``, ``mixlab.booth_sheet.build_booth_sheet``) so the two artifacts
describe the same run identically.

Every top-level and per-concept key is camelCase — this is an external contract, not
an internal data structure.
"""

from __future__ import annotations

from mixlab.booth_sheet import build_booth_sheet
from mixlab.llm import _compute_practicality_score  # noqa: PLC2701 — reused for identical practicality math (#89)
from mixlab.models import MixConcept, Track
from mixlab.transitions import relation_label, score_transition

SCHEMA_VERSION = 1


def _resolved_tracks(concept: MixConcept, tracks_by_id: dict[str, Track]) -> list[Track]:
    """Concept track_ids resolved to ``Track`` objects, dropping any missing ids.

    Mirrors ``html_report._resolved_tracks`` so ``tracks``/``transitions`` below
    describe the same sequence the HTML report renders for this concept.
    """
    return [t for tid in concept.track_ids if (t := tracks_by_id.get(tid)) is not None]


def _track_payload(track: Track) -> dict[str, object]:
    return {
        "id": track.track_id,
        "artist": track.artist,
        "title": track.title,
        "bpm": track.bpm,
        "camelotKey": track.camelot_key,
        "energy": track.energy,
        "durationSecs": track.duration_secs,
    }


def _transition_payload(concept: MixConcept, tracks: list[Track]) -> list[dict[str, object]]:
    """One entry per consecutive resolvable pair, matching ``html_report._render_transitions``."""
    annotations = {(tr.from_id, tr.to_id): tr for tr in concept.transitions}
    rows: list[dict[str, object]] = []
    for i in range(len(tracks) - 1):
        a, b = tracks[i], tracks[i + 1]
        edge = score_transition(a, b)
        mechanism = relation_label(edge, a, b)
        annotation = annotations.get((a.track_id, b.track_id))
        risk_type = annotation.risk_type if annotation is not None and annotation.risk_type else None
        rows.append(
            {
                "mechanism": mechanism,
                "score": edge.score,
                "blendLabel": edge.blend_label or None,
                "riskType": risk_type,
            }
        )
    return rows


def _practicality_payload(concept: MixConcept, tracks_by_id: dict[str, Track]) -> dict[str, object]:
    """DJ Practicality Score, computed identically to the genre-mode report (intent_brief=None)."""
    score = _compute_practicality_score(concept, tracks_by_id, None)
    return {
        "bpmSmoothness": score.bpm_smoothness,
        "harmonicRatio": score.harmonic_ratio,
        "riskJustified": score.risk_justified,
        "blendFeasibility": score.blend_feasibility,
        "hasBlendData": score.has_blend_data,
        "overall": score.overall,
    }


def _booth_sheet_payload(concept: MixConcept, tracks_by_id: dict[str, Track]) -> list[dict[str, object]]:
    steps = build_booth_sheet(concept, tracks_by_id)
    return [
        {
            "index": step.index,
            "fromId": step.from_id,
            "toId": step.to_id,
            "fromLabel": step.from_label,
            "toLabel": step.to_label,
            "openAtSecs": step.open_at_secs,
            "outroBars": step.outro_bars,
            "introBars": step.intro_bars,
            "pitchPct": step.pitch_pct,
            "tier": step.tier,
            "chips": [[text, is_warn] for text, is_warn in step.chips],
            "plan": step.plan,
            "fallback": step.fallback,
            "scout": step.scout,
        }
        for step in steps
    ]


def _concept_warnings(concept: MixConcept, validation_warnings: list[str]) -> list[str]:
    """Validation warnings scoped to this concept via the ``[title]`` prefix (see llm.validate_stage2_output)."""
    prefix = f"[{concept.title}]"
    return [w for w in validation_warnings if w.startswith(prefix)]


def _concept_payload(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
    validation_warnings: list[str],
) -> dict[str, object]:
    tracks = _resolved_tracks(concept, tracks_by_id)
    return {
        "conceptId": concept.concept_id,
        "title": concept.title,
        "directionType": concept.direction_type,
        "arcType": concept.arc_type,
        "mood": concept.mood,
        "thesis": concept.name_reason,
        "practicality": _practicality_payload(concept, tracks_by_id),
        "tracks": [_track_payload(t) for t in tracks],
        "transitions": _transition_payload(concept, tracks),
        "boothSheet": _booth_sheet_payload(concept, tracks_by_id),
        "conceptWarnings": _concept_warnings(concept, validation_warnings),
    }


def build_run_summary(
    concepts: list[MixConcept],
    tracks_by_id: dict[str, Track],
    *,
    flags: dict[str, object],
    seed: int,
    validation_warnings: list[str],
    run_notes: str,
    generated_at: str,
) -> dict[str, object]:
    """Build the JSON-serialisable ``summary.json`` payload for a completed run (#89).

    Deterministic: identical inputs produce an identical dict (the only caller-supplied
    time-varying value is ``generated_at``, matching the ``render_html_report`` contract).
    """
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": generated_at,
        "seed": seed,
        "flags": flags,
        "validationWarnings": list(validation_warnings),
        "runNotes": run_notes,
        "concepts": [_concept_payload(c, tracks_by_id, validation_warnings) for c in concepts],
    }
