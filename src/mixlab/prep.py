"""Cue-Prep Assistant scoring engine (#72/#73) — pure, no I/O, no LLM.

Ranks tracks that are missing (or only partially have) mix-point cue data by how
much cueing them would pay off: how often they show up in previously planned
concepts (``demand``), how well-connected they are to their genre-bucket peers
(``centrality``), and whether they are unplayed. Consumers (CLI/report layers)
own presentation; this module only computes the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mixlab.config import GENRE_MAP
from mixlab.history import ConceptHistory
from mixlab.models import Track
from mixlab.transitions import score_transition

GapType = Literal["uncued", "no-mix-out"]

# Uncued tracks (no mix_points at all) pay off more to cue than tracks that are
# merely missing a mix-out point.
_GAP_WEIGHTS: dict[GapType, float] = {"uncued": 1.0, "no-mix-out": 0.7}

# Transition score at/above which a peer edge counts as "strong" for centrality.
_STRONG_EDGE_THRESHOLD = 0.7

# Centrality at/above which the reason string calls a track "harmonically central".
_CENTRAL_REASON_THRESHOLD = 0.5


@dataclass(frozen=True)
class PrepItem:
    track_id: str
    label: str  # "Artist — Title"
    genre: str  # raw Track.genre tag
    bucket: str | None  # GENRE_MAP bucket key, None if unmapped
    bpm: float
    camelot_key: str
    gap: GapType
    demand: int
    centrality: float  # 0..1, rounded to 3 decimals
    unplayed: bool
    score: float  # rounded to 2 decimals
    reason: str


def _bucket_for_genre(genre: str) -> str | None:
    """Map a raw Track.genre tag to its GENRE_MAP bucket key.

    Mirrors the exact (case-sensitive) membership test used by
    ``clustering.group_by_genre`` / ``clustering.count_available_by_genre`` — a
    track belongs to the first bucket whose genre list contains its genre tag
    verbatim. Replicated here (rather than imported) because those helpers
    return track buckets keyed differently (by raw genre tag, not API label) and
    pulling them in would be a poor fit for this module's needs.
    """
    for bucket_key, rb_genres in GENRE_MAP.items():
        if genre in rb_genres:
            return bucket_key
    return None


def _gap_for_track(track: Track) -> GapType | None:
    """Classify a track's cue gap; None means it already has full mix-point data."""
    if track.mix_points is None:
        return "uncued"
    if track.mix_points.mix_out_secs is None:
        return "no-mix-out"
    return None


def _demand(track_id: str, history: ConceptHistory) -> int:
    """Count concept records (across every history run) that include this track."""
    count = 0
    for entry in history.runs:
        for concept in entry.concepts:
            if track_id in concept.track_ids:
                count += 1
    return count


def _sample_peers(peers_sorted: list[Track], peer_cap: int) -> list[Track]:
    """Deterministically down-sample a peer pool to exactly ``peer_cap`` entries.

    Evenly-spaced indices over the sorted pool (stable, reproducible across calls
    and across environments since it involves no set/dict iteration order).
    """
    n = len(peers_sorted)
    if n <= peer_cap:
        return peers_sorted
    if peer_cap <= 1:
        return peers_sorted[:peer_cap]
    return [peers_sorted[round(i * (n - 1) / (peer_cap - 1))] for i in range(peer_cap)]


def _centrality(
    track: Track,
    bucket: str | None,
    tracks_by_bucket: dict[str, list[Track]],
    peer_cap: int,
) -> float:
    """Fraction of (possibly sampled) same-bucket peers with a strong transition edge."""
    if bucket is None:
        return 0.0
    peers = [t for t in tracks_by_bucket.get(bucket, []) if t.track_id != track.track_id]
    if len(peers) < 2:
        return 0.0
    peers_sorted = sorted(peers, key=lambda t: t.track_id)
    sample = _sample_peers(peers_sorted, peer_cap)
    if not sample:
        return 0.0
    strong = sum(1 for peer in sample if score_transition(track, peer).score >= _STRONG_EDGE_THRESHOLD)
    return round(strong / len(sample), 3)


def _reason(demand: int, centrality: float, unplayed: bool, gap: GapType) -> str:
    fragments: list[str] = []
    if demand > 0:
        noun = "concept" if demand == 1 else "concepts"
        fragments.append(f"in {demand} planned {noun}")
    if centrality >= _CENTRAL_REASON_THRESHOLD:
        fragments.append("harmonically central")
    if unplayed:
        fragments.append("unplayed")
    fragments.append("fully uncued" if gap == "uncued" else "missing mix-out cue")
    return " · ".join(fragments)


def rank_prep_targets(
    tracks: list[Track],
    history: ConceptHistory,
    *,
    bucket: str | None = None,
    top: int = 20,
    peer_cap: int = 600,
) -> tuple[list[PrepItem], dict[str, tuple[int, int]]]:
    """Rank tracks with a cue gap by expected prep payoff.

    Returns ``(ranked_items, totals)`` where ``totals`` maps GENRE_MAP bucket key
    to ``(tracks_with_a_cue_gap, total_tracks_in_bucket)``. When ``bucket`` is
    given, both the ranking and the totals are scoped to that bucket only. Peer
    pools for centrality are always built from the full ``tracks`` input — the
    ``bucket`` scope only filters which tracks are emitted/counted, not which
    tracks count as peers of an in-scope track.
    """
    tracks_by_bucket: dict[str, list[Track]] = {}
    for t in tracks:
        b = _bucket_for_genre(t.genre)
        if b is not None:
            tracks_by_bucket.setdefault(b, []).append(t)

    totals: dict[str, tuple[int, int]] = {}
    for b, bucket_tracks in tracks_by_bucket.items():
        if bucket is not None and b != bucket:
            continue
        gap_count = sum(1 for t in bucket_tracks if _gap_for_track(t) is not None)
        totals[b] = (gap_count, len(bucket_tracks))

    items: list[PrepItem] = []
    for t in tracks:
        b = _bucket_for_genre(t.genre)
        if bucket is not None and b != bucket:
            continue
        gap = _gap_for_track(t)
        if gap is None:
            continue

        demand = _demand(t.track_id, history)
        centrality = _centrality(t, b, tracks_by_bucket, peer_cap)
        unplayed = t.play_count == 0
        weight = _GAP_WEIGHTS[gap]
        score = round(weight * (1.0 + 2.0 * demand + 1.5 * centrality + (0.5 if unplayed else 0.0)), 2)
        reason = _reason(demand, centrality, unplayed, gap)

        items.append(
            PrepItem(
                track_id=t.track_id,
                label=f"{t.artist} — {t.title}",
                genre=t.genre,
                bucket=b,
                bpm=t.bpm,
                camelot_key=t.camelot_key,
                gap=gap,
                demand=demand,
                centrality=centrality,
                unplayed=unplayed,
                score=score,
                reason=reason,
            )
        )

    items.sort(key=lambda item: (-item.score, item.track_id))
    ranked = items[:top] if top > 0 else []
    return ranked, totals
