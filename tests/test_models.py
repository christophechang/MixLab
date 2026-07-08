from __future__ import annotations

from mixlab.models import MixConcept


def test_mix_concept_concept_id_default_empty_string() -> None:
    concept = MixConcept(title="Test", mood="dark", track_ids=["1", "2"])
    assert concept.concept_id == ""


def test_mix_concept_concept_id_settable_after_construction() -> None:
    concept = MixConcept(title="Test", mood="dark", track_ids=["1", "2"])
    concept.concept_id = "11111111-1111-1111-1111-111111111111"
    assert concept.concept_id == "11111111-1111-1111-1111-111111111111"
