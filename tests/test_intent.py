from __future__ import annotations

from mixlab.llm import _minimum_playlist_seed_retention, _parse_intent_brief, _select_best_variant
from mixlab.models import CompletionVariant, IntentBrief, MixConcept, SeedAnalysis, Track


def _make_track(track_id: str, *, bpm: float = 124.0, camelot_key: str = "8A", energy: int | None = None) -> Track:
    return Track(
        track_id=track_id,
        artist=f"Artist {track_id}",
        title=f"Title {track_id}",
        bpm=bpm,
        camelot_key=camelot_key,
        genre="House",
        energy=energy,
    )


def test_parse_intent_brief_valid_json_populates_anchor_ids() -> None:
    raw = """{
      "overall_vibe": "A deep hypnotic journey",
      "is_coherent_set": true,
      "risk_tolerance": "medium",
      "missing_roles": ["opener"],
      "seed_analyses": [
        {"track_id": "1", "tier": "anchor", "inferred_role": "peak"},
        {"track_id": "2", "tier": "supporting", "inferred_role": "builder"}
      ]
    }"""
    seed_tracks = [_make_track("1"), _make_track("2")]
    brief = _parse_intent_brief(raw, seed_tracks, bpm_range=(122.0, 126.0))
    assert brief.overall_vibe == "A deep hypnotic journey"
    assert brief.risk_tolerance == "medium"
    assert "1" in brief.anchor_ids
    assert "2" in brief.supporting_ids
    assert "opener" in brief.missing_roles
    assert brief.is_coherent_set is True


def test_parse_intent_brief_invalid_tier_defaults_to_supporting() -> None:
    raw = """{
      "overall_vibe": "Unknown",
      "is_coherent_set": false,
      "risk_tolerance": "low",
      "missing_roles": [],
      "seed_analyses": [
        {"track_id": "1", "tier": "INVALID", "inferred_role": "unknown"}
      ]
    }"""
    seed_tracks = [_make_track("1")]
    brief = _parse_intent_brief(raw, seed_tracks, bpm_range=(124.0, 124.0))
    assert "1" in brief.supporting_ids


def test_parse_intent_brief_missing_track_in_analyses_defaults_to_supporting() -> None:
    """If LLM omits a seed from seed_analyses, it defaults to 'supporting'."""
    raw = """{
      "overall_vibe": "Test",
      "is_coherent_set": true,
      "risk_tolerance": "high",
      "missing_roles": [],
      "seed_analyses": []
    }"""
    seed_tracks = [_make_track("1"), _make_track("2")]
    brief = _parse_intent_brief(raw, seed_tracks, bpm_range=(124.0, 124.0))
    assert "1" in brief.supporting_ids
    assert "2" in brief.supporting_ids


def test_parse_intent_brief_strips_markdown_fence() -> None:
    raw = '```json\n{"overall_vibe": "Test", "is_coherent_set": true, "risk_tolerance": "low", "missing_roles": [], "seed_analyses": []}\n```'
    brief = _parse_intent_brief(raw, [], bpm_range=(0.0, 0.0))
    assert brief.overall_vibe == "Test"


def test_parse_intent_brief_invalid_json_returns_fallback() -> None:
    """Invalid JSON should not raise — returns a safe fallback brief."""
    raw = "this is not json at all"
    seed_tracks = [_make_track("1")]
    brief = _parse_intent_brief(raw, seed_tracks, bpm_range=(124.0, 124.0))
    assert "1" in brief.supporting_ids  # fallback: seed treated as supporting


def _make_brief_with_tiers(anchor_ids: list[str], supporting_ids: list[str]) -> IntentBrief:
    analyses = [
        SeedAnalysis(track_id=tid, tier="anchor", inferred_role="peak", drop_cost=0.1) for tid in anchor_ids
    ] + [
        SeedAnalysis(track_id=tid, tier="supporting", inferred_role="builder", drop_cost=0.5) for tid in supporting_ids
    ]
    return IntentBrief(
        overall_vibe="Test",
        energy_shape="unclear",
        risk_tolerance="medium",
        is_coherent_set=True,
        seed_analyses=analyses,
        missing_roles=[],
        strong_adjacencies=[],
        bpm_range=(120.0, 130.0),
    )


def test_minimum_retention_uses_anchor_floor_with_brief() -> None:
    brief = _make_brief_with_tiers(anchor_ids=["1", "2", "3", "4"], supporting_ids=["5", "6"])
    # 75% of 4 anchors = 3, 40% of 2 supporting = 1 → floor = 4
    floor = _minimum_playlist_seed_retention(6, brief)
    assert floor == 4


def test_minimum_retention_falls_back_to_sixty_percent_without_brief() -> None:
    floor = _minimum_playlist_seed_retention(10, None)
    assert floor == 6  # 60% of 10


def test_select_best_variant_prefers_higher_anchor_retention() -> None:
    concept_cons = MixConcept(title="C", mood="conservative", track_ids=["1", "2", "3"])
    concept_bold = MixConcept(title="B", mood="bold", track_ids=["1", "4", "5"])
    v_cons = CompletionVariant(
        strategy="conservative", concept=concept_cons, anchor_retention_rate=1.0, role_coverage=0.5
    )
    v_bold = CompletionVariant(strategy="bold", concept=concept_bold, anchor_retention_rate=0.5, role_coverage=0.8)
    best = _select_best_variant([v_cons, v_bold])
    assert best.strategy == "conservative"


def test_select_best_variant_single_variant_returned_as_is() -> None:
    concept = MixConcept(title="T", mood="conservative", track_ids=["1"])
    v = CompletionVariant(strategy="conservative", concept=concept, anchor_retention_rate=1.0, role_coverage=0.0)
    assert _select_best_variant([v]) is v
