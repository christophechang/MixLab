from __future__ import annotations

from mixlab.models import MixConcept, Track


def assemble_report(
    concepts: list[MixConcept],
    tracks_by_id: dict[str, Track],
    outliers: list[Track],
) -> str:
    raise NotImplementedError
