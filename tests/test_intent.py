from __future__ import annotations

from mixlab.llm import _parse_intent_brief
from mixlab.models import Track


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
