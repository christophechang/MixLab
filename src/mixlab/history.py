from __future__ import annotations

import json
import logging
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mixlab.models import ConceptShape, MixCanvas, MixConcept

logger = logging.getLogger(__name__)

_MAX_HISTORY = 50
_RECENCY_WINDOW = 10
_DECAY = 0.8

# Combined-novelty weights (#7). Track-overlap stays primary; concept-shape is the
# secondary signal that catches "same shape, different tracks" repetition.
_NOVELTY_TRACK_WEIGHT = 0.65
_NOVELTY_SHAPE_WEIGHT = 0.35
_BPM_BUCKET = 10.0  # 10-BPM band buckets for shape comparison


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


_RECENT_CONCEPTS_LIMIT = 5


def format_recent_concepts(history: ConceptHistory, limit: int = _RECENT_CONCEPTS_LIMIT) -> str | None:
    """Build a compact 'recent concepts' block for Stage 2 prompt injection.

    Returns None when history is empty so the caller can omit the section entirely
    (rather than rendering an empty header). The block lists the most recent N
    concepts in reverse chronological order with title, date, genre, arc_type, and
    mood — enough character for the model to deliberately diverge from prior runs.
    """
    if not history.runs:
        return None
    recent = history.runs[-limit:]
    lines: list[str] = ["RECENT CONCEPTS (avoid retreading these):"]
    for entry in reversed(recent):
        date_part = entry.created_at[:10] if entry.created_at else "unknown"
        title = entry.concept_title or "(untitled)"
        arc = entry.energy_path or "unspecified"
        mood = entry.mood or "unspecified"
        lines.append(f'- "{title}" ({date_part}, {entry.genre}, arc: {arc}, mood: {mood})')
    lines.append("")
    lines.append(
        "These are concepts you generated previously. Diverge deliberately: pick titles, "
        "theses, and energy arcs that contrast with the list above. Do not echo their "
        "character. If a recent concept used a wave arc, prefer a different shape today. "
        "If recent titles were abstract and place-based, today's might lean object-based "
        "or moment-based. Repeating an arc or character only counts when the pool genuinely "
        "supports nothing else."
    )
    return "\n".join(lines)


def _bpm_band_label(bpm: float) -> str:
    """Bucket a BPM into a 10-BPM band label like '170-180'. Returns '' when bpm <= 0."""
    if bpm <= 0:
        return ""
    lo = int(bpm // _BPM_BUCKET) * int(_BPM_BUCKET)
    return f"{lo}-{lo + int(_BPM_BUCKET)}"


def concept_shape_from_canvas(canvas: MixCanvas) -> ConceptShape:
    """Extract the deterministic concept-shape fingerprint from a candidate canvas."""
    return ConceptShape(
        bpm_band=_bpm_band_label(canvas.dominant_bpm),
        camelot_zone=canvas.dominant_camelot or "",
        # Canvas-side shape has no arc_type yet (Stage 2 hasn't run). Always "" so the
        # field is excluded from comparison until Stage 2 result is in history.
        energy_path="",
        has_opener=bool(canvas.roles.opener),
        has_closer=bool(canvas.roles.closer),
        has_peak=bool(canvas.roles.peak),
    )


def concept_shape_from_entry(entry: HistoryEntry) -> ConceptShape:
    """Extract the concept-shape fingerprint from a stored history entry."""
    dominant_bpm = entry.dominant_bpm_clusters[0] if entry.dominant_bpm_clusters else 0.0
    camelot = entry.dominant_camelot_keys[0] if entry.dominant_camelot_keys else ""
    role_pattern = entry.role_pattern
    return ConceptShape(
        bpm_band=_bpm_band_label(dominant_bpm),
        camelot_zone=camelot,
        energy_path=entry.energy_path or "",
        has_opener="opener" in role_pattern or bool(entry.opener_candidates),
        has_closer="closer" in role_pattern or bool(entry.closer_candidates),
        has_peak="peak" in role_pattern,
    )


def concept_shape_similarity(a: ConceptShape, b: ConceptShape) -> float:
    """Similarity over reliably-populated shape fields.

    Empty ``energy_path`` is excluded from comparison (Option A — unknown ≠ same).
    Other fields always participate; equal bools count as a match. Returns 0.0 when
    no field can be meaningfully compared.
    """
    matches = 0
    total = 0
    if a.bpm_band and b.bpm_band:
        matches += int(a.bpm_band == b.bpm_band)
        total += 1
    if a.camelot_zone and b.camelot_zone:
        matches += int(a.camelot_zone == b.camelot_zone)
        total += 1
    # energy_path: only compared when both sides populated (Option A).
    if a.energy_path and b.energy_path:
        matches += int(a.energy_path == b.energy_path)
        total += 1
    # Role-presence flags are always defined — compare unconditionally.
    matches += int(a.has_opener == b.has_opener)
    matches += int(a.has_closer == b.has_closer)
    matches += int(a.has_peak == b.has_peak)
    total += 3
    if total == 0:
        return 0.0
    return matches / total


@dataclass(frozen=True)
class NoveltyBreakdown:
    """Diagnostic breakdown of the similarity-to-history penalty (#7 / --debug)."""

    track_similarity: float
    shape_similarity: float
    combined: float
    age_of_top_match: int  # 0 = most recent run; -1 when no comparable history


def similarity_breakdown_to_history(
    canvas: MixCanvas,
    history: ConceptHistory,
    recency_window: int = _RECENCY_WINDOW,
) -> NoveltyBreakdown:
    """Combined track + shape similarity vs recent history, with age decay.

    Combines a track-overlap Jaccard with a deterministic concept-shape similarity
    (BPM band, dominant Camelot zone, role pattern, optional energy_path). The two
    components are weighted (0.65 / 0.35) and decayed jointly per recency age before
    taking the max across the recent window. Track-overlap remains primary; the shape
    component catches "same shape, different tracks" repetition.
    """
    empty = NoveltyBreakdown(track_similarity=0.0, shape_similarity=0.0, combined=0.0, age_of_top_match=-1)
    if not history.runs:
        return empty
    canvas_ids = frozenset(canvas.core_track_ids)
    canvas_shape = concept_shape_from_canvas(canvas)
    recent = history.runs[-recency_window:]
    best = empty
    for age, entry in enumerate(reversed(recent)):
        hist_ids = frozenset(entry.core_track_ids)
        union = canvas_ids | hist_ids
        track_sim = len(canvas_ids & hist_ids) / len(union) if union else 0.0
        shape_sim = concept_shape_similarity(canvas_shape, concept_shape_from_entry(entry))
        combined = _NOVELTY_TRACK_WEIGHT * track_sim + _NOVELTY_SHAPE_WEIGHT * shape_sim
        decayed = combined * (_DECAY**age)
        if decayed > best.combined:
            best = NoveltyBreakdown(
                track_similarity=track_sim * (_DECAY**age),
                shape_similarity=shape_sim * (_DECAY**age),
                combined=decayed,
                age_of_top_match=age,
            )
    return best


def similarity_to_history(
    canvas: MixCanvas,
    history: ConceptHistory,
    recency_window: int = _RECENCY_WINDOW,
) -> float:
    """Combined track + shape similarity vs recent history, with age decay (#7).

    Thin wrapper over :func:`similarity_breakdown_to_history` that returns just the
    combined value for use in :func:`score_canvas`.
    """
    return similarity_breakdown_to_history(canvas, history, recency_window).combined
