"""Library-map payload (#40 / mixlab-web Milestone B).

Deterministic, LLM-free enumeration of concept-direction candidates per engine
pool, serialised for the web app's constellation overlay. Contract:
mixlab-web docs/superpowers/specs/2026-08-06-library-map-design.md §7 —
change the shape there first, then here.

Pool-semantics note: the run pipeline's direction pool is the flattened genre
scope plus that genre's resolved same-genre outliers (see ``direction_pool`` in
``__main__.py``), whereas this module's standard-genre pools contain only
tracks whose Rekordbox tag is already a mapped ``GENRE_MAP`` member. The map is
therefore intentionally conservative — it can under-count relative to a live
run — because outlier resolution is a run-time concern (it needs the LLM-backed
outlier classifier) that has no place in a static, deterministic map payload.
"""

from __future__ import annotations

import json
from typing import Literal

from mixlab.clustering import build_custom_genre_pool, group_by_genre
from mixlab.config import CUSTOM_GENRES, GENRE_MAP
from mixlab.directions import Direction, enumerate_directions
from mixlab.matcher import filter_played, filter_unplayed
from mixlab.models import PlayedTrack, Track

MapMode = Literal["all", "unplayed", "played"]


def _direction_entry(direction: Direction) -> dict[str, object]:
    return {
        "direction_type": direction.direction_type,
        "title": direction.title,
        "mood": direction.mood,
        "brief": direction.brief,
        "feasibility": direction.feasibility,
        "track_ids": list(direction.track_ids),
    }


def _pool_entry(pool: list[Track], *, seed: int) -> dict[str, object]:
    return {
        "track_count": len(pool),
        "directions": [_direction_entry(d) for d in enumerate_directions(pool, seed=seed)],
    }


def build_map_payload(
    tracks: list[Track],
    *,
    mode: MapMode,
    seed: int,
    played: list[PlayedTrack],
) -> dict[str, object]:
    """Assemble the full map payload over ``tracks`` (post do-not-recommend)."""
    if mode == "unplayed":
        scoped = filter_unplayed(tracks, played)
    elif mode == "played":
        scoped = filter_played(tracks, played)
    else:
        scoped = tracks

    # group_by_genre buckets by the literal Rekordbox genre tag on each track
    # (e.g. "Deep House"), not by the GENRE_MAP API key ("house") — merge every
    # tag belonging to a key's tag list to get that pool's tracks.
    by_genre = group_by_genre(scoped, GENRE_MAP)
    pools: dict[str, dict[str, object]] = {}
    for key, rb_tags in GENRE_MAP.items():
        pool: list[Track] = []
        for tag in rb_tags:
            pool.extend(by_genre.get(tag, []))
        pools[key] = _pool_entry(pool, seed=seed)
    for key in CUSTOM_GENRES:
        custom_pool = build_custom_genre_pool(key, scoped, CUSTOM_GENRES, GENRE_MAP)
        pools[key] = _pool_entry(custom_pool, seed=seed)

    return {
        "version": 1,
        "mode": mode,
        "seed": seed,
        "collection_tracks": len(tracks),
        "catalog_tracks": len(played),
        "pools": pools,
    }


def render_map_json(payload: dict[str, object]) -> str:
    """Stable serialisation: 2-space indent, insertion order, trailing newline."""
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
