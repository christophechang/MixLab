from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mixlab.models import MixCanvas, MixConcept

logger = logging.getLogger(__name__)

_MAX_HISTORY = 50
_RECENCY_WINDOW = 10
_DECAY = 0.8


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
        if concepts:
            primary = concepts[0]
            concept_track_ids = list(primary.track_ids)
            concept_title = primary.title
            mood = primary.mood

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
            energy_path="",
            mood=mood,
        )


@dataclass
class ConceptHistory:
    runs: list[HistoryEntry] = field(default_factory=list)


def load_history(path: Path) -> ConceptHistory:
    try:
        if not path.exists():
            return ConceptHistory()
        raw = json.loads(path.read_text())
        entries = [HistoryEntry(**r) for r in raw.get("runs", [])]
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
