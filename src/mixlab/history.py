from __future__ import annotations

import json
import logging
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mixlab.models import MixCanvas, MixConcept

logger = logging.getLogger(__name__)

_MAX_HISTORY = 50
_RECENCY_WINDOW = 10
_DECAY = 0.8


def _match_concept_to_canvas(concept: MixConcept, canvases: list[MixCanvas]) -> MixCanvas | None:
    """Return the canvas whose full pool covers the most of a concept's tracks."""
    if not canvases:
        return None
    concept_ids = set(concept.track_ids)
    best: MixCanvas | None = None
    best_overlap = 0
    for canvas in canvases:
        pool = set(canvas.core_track_ids) | set(canvas.bridge_track_ids) | set(canvas.wildcard_track_ids)
        overlap = len(concept_ids & pool)
        if overlap > best_overlap:
            best_overlap = overlap
            best = canvas
    return best


# Lookup priority: most distinctive roles first so a track in both opener and
# groove_locker pools gets tagged as opener (the more informative classification).
_ROLE_LOOKUP_ORDER: tuple[str, ...] = ("opener", "closer", "peak", "builder", "groove_locker", "pivot")


def _role_for_track(track_id: str, canvas: MixCanvas) -> str:
    role_pools: dict[str, list[str]] = {
        "opener": canvas.roles.opener,
        "closer": canvas.roles.closer,
        "peak": canvas.roles.peak,
        "builder": canvas.roles.builder,
        "groove_locker": canvas.roles.groove_locker,
        "pivot": canvas.roles.pivot,
    }
    for role in _ROLE_LOOKUP_ORDER:
        if track_id in role_pools[role]:
            return role
    return "unknown"


@dataclass
class HistoryEntry:
    run_id: str
    created_at: str
    mode: str
    genre: str
    selected_canvas_ids: list[str]
    dominant_bpm_clusters: list[float]
    dominant_camelot_keys: list[str]
    core_track_ids: list[str]
    anchor_track_ids: list[str]
    opener_candidates: list[str]
    closer_candidates: list[str]
    concept_title: str
    concept_track_ids: list[str]
    energy_path: str
    mood: str
    rating: float | None = None
    # v0.10 additions — populated on new runs, defaults preserve backward compat with old files.
    canvas_score_breakdown: dict[str, float] = field(default_factory=dict)
    selected_canvas_risk_notes: list[str] = field(default_factory=list)
    bpm_band: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    role_pattern: list[str] = field(default_factory=list)

    @classmethod
    def from_run(
        cls,
        canvases: list[MixCanvas],
        concepts: list[MixConcept],
        genre: str,
        mode: str,
    ) -> HistoryEntry:
        core_ids: list[str] = []
        opener_ids: list[str] = []
        closer_ids: list[str] = []
        canvas_ids: list[str] = []
        bpm_clusters: list[float] = []
        camelot_keys: list[str] = []

        for canvas in canvases:
            canvas_ids.append(canvas.canvas_id)
            core_ids.extend(canvas.core_track_ids)
            opener_ids.extend(canvas.roles.opener)
            closer_ids.extend(canvas.roles.closer)
            bpm_clusters.append(canvas.dominant_bpm)
            camelot_keys.append(canvas.dominant_camelot)

        concept_track_ids: list[str] = []
        concept_title = ""
        mood = ""
        energy_path = ""
        role_pattern: list[str] = []
        if concepts:
            primary = concepts[0]
            concept_track_ids = list(primary.track_ids)
            concept_title = primary.title
            mood = primary.mood
            # Fix v0.9 bug: energy_path was always written as "". Populate from the
            # arc_type Stage 2 emits since v0.9 (defaults to None if absent, which we
            # render as "" for backward compatibility with the field type).
            energy_path = primary.arc_type or ""
            # role_pattern: classify each concept track against the matched canvas's
            # role candidate pools. Stage 2 does not emit per-track roles structurally;
            # this is the closest reproducible signal available without prompt changes.
            primary_canvas = _match_concept_to_canvas(primary, canvases)
            if primary_canvas is not None:
                role_pattern = [_role_for_track(tid, primary_canvas) for tid in primary.track_ids]

        canvas_score_breakdown: dict[str, float] = {}
        selected_canvas_risk_notes: list[str] = []
        bpm_band: tuple[float, float] = (0.0, 0.0)
        if canvases:
            canvas_score_breakdown = {
                "technical_viability": statistics.mean(c.score.technical_viability for c in canvases),
                "role_coverage": statistics.mean(c.score.role_coverage for c in canvases),
                "anchor_strength": statistics.mean(c.score.anchor_strength for c in canvases),
                "contrast_potential": statistics.mean(c.score.contrast_potential for c in canvases),
                "distinctiveness": statistics.mean(c.score.distinctiveness for c in canvases),
                "novelty": statistics.mean(c.score.novelty for c in canvases),
                "weakness_penalty": statistics.mean(c.score.weakness_penalty for c in canvases),
                "overall": statistics.mean(c.score.overall for c in canvases),
            }
            # Flatten risk notes across canvases. Order preserved; duplicates kept so the
            # distribution of complaints is visible (over-repeated artist appears in many
            # canvases is itself a signal).
            selected_canvas_risk_notes = [note for c in canvases for note in c.risk_notes]
            bpm_mins = [c.bpm_range[0] for c in canvases if c.bpm_range[0] > 0]
            bpm_maxs = [c.bpm_range[1] for c in canvases if c.bpm_range[1] > 0]
            if bpm_mins and bpm_maxs:
                bpm_band = (min(bpm_mins), max(bpm_maxs))

        return cls(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            mode=mode,
            genre=genre,
            selected_canvas_ids=canvas_ids,
            dominant_bpm_clusters=list(dict.fromkeys(bpm_clusters)),
            dominant_camelot_keys=list(dict.fromkeys(camelot_keys)),
            core_track_ids=list(dict.fromkeys(core_ids)),
            anchor_track_ids=list(dict.fromkeys(opener_ids[:2])),
            opener_candidates=list(dict.fromkeys(opener_ids)),
            closer_candidates=list(dict.fromkeys(closer_ids)),
            concept_title=concept_title,
            concept_track_ids=concept_track_ids,
            energy_path=energy_path,
            mood=mood,
            canvas_score_breakdown=canvas_score_breakdown,
            selected_canvas_risk_notes=selected_canvas_risk_notes,
            bpm_band=bpm_band,
            role_pattern=role_pattern,
        )


@dataclass
class ConceptHistory:
    runs: list[HistoryEntry] = field(default_factory=list)


def load_history(path: Path) -> ConceptHistory:
    try:
        if not path.exists():
            return ConceptHistory()
        raw = json.loads(path.read_text())
        # Filter incoming JSON to known dataclass fields so old history files (missing
        # v0.10 fields like canvas_score_breakdown) and any future stray fields both
        # load cleanly. Missing fields get the dataclass default.
        valid_fields = set(HistoryEntry.__dataclass_fields__)
        entries: list[HistoryEntry] = []
        for r in raw.get("runs", []):
            if not isinstance(r, dict):
                continue
            payload: dict[str, Any] = {k: v for k, v in r.items() if k in valid_fields}
            # JSON serialises tuples as lists; bpm_band must be coerced back so equality
            # comparisons and downstream consumers (concept-shape novelty in #7) see the
            # right type.
            bpm_band_raw = payload.get("bpm_band")
            if isinstance(bpm_band_raw, list) and len(bpm_band_raw) == 2:
                payload["bpm_band"] = (float(bpm_band_raw[0]), float(bpm_band_raw[1]))
            entries.append(HistoryEntry(**payload))
        return ConceptHistory(runs=entries)
    except Exception:
        logger.warning("concept-history.json unreadable; starting fresh")
        return ConceptHistory()


def append_run(history: ConceptHistory, entry: HistoryEntry, path: Path) -> None:
    try:
        history.runs.append(entry)
        if len(history.runs) > _MAX_HISTORY:
            history.runs = history.runs[-_MAX_HISTORY:]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": [asdict(r) for r in history.runs]}
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        logger.warning("Failed to write concept-history.json")


def similarity_to_history(
    canvas: MixCanvas,
    history: ConceptHistory,
    recency_window: int = _RECENCY_WINDOW,
) -> float:
    """Jaccard similarity of canvas core track IDs vs recent history entries, with age decay."""
    if not history.runs:
        return 0.0
    canvas_ids = frozenset(canvas.core_track_ids)
    if not canvas_ids:
        return 0.0
    recent = history.runs[-recency_window:]
    max_sim = 0.0
    for age, entry in enumerate(reversed(recent)):
        hist_ids = frozenset(entry.core_track_ids)
        if not hist_ids:
            continue
        union = canvas_ids | hist_ids
        intersection = canvas_ids & hist_ids
        jaccard = len(intersection) / len(union)
        decayed = jaccard * (_DECAY**age)
        if decayed > max_sim:
            max_sim = decayed
    return max_sim
