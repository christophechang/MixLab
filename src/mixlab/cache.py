from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

CACHE_PATH = Path(".mixlab_genres.json")


class _GenreEntry(TypedDict):
    total: int
    unplayed: int


class GenreCache(TypedDict):
    saved_at: str
    counts: dict[str, _GenreEntry]
    outliers: int
    outlier_genres: dict[str, _GenreEntry]


def save_genre_cache(
    counts: dict[str, tuple[int, int]],
    outlier_count: int,
    outlier_genres: dict[str, tuple[int, int]],
    path: Path = CACHE_PATH,
) -> None:
    cache: GenreCache = {
        "saved_at": datetime.now(UTC).isoformat(),
        "counts": {label: {"total": t, "unplayed": u} for label, (t, u) in counts.items()},
        "outliers": outlier_count,
        "outlier_genres": {tag: {"total": t, "unplayed": u} for tag, (t, u) in outlier_genres.items()},
    }
    path.write_text(json.dumps(cache, indent=2))


def load_genre_cache(path: Path = CACHE_PATH) -> GenreCache | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cast(GenreCache, data)
