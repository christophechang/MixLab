from __future__ import annotations

from unittest.mock import AsyncMock, patch

from mixlab.llm import (
    _compute_practicality_score,  # noqa: PLC2701
    _minimum_playlist_seed_retention,
    _pair_consecutive,  # noqa: PLC2701
    _parse_intent_brief,
    _passes_floor,  # noqa: PLC2701
    _score_variant,  # noqa: PLC2701
    _select_best_variant,
    make_cascade_state,
    stage0_intent_brief,
)
from mixlab.models import (
    AdjacencyFragment,
    CompletionVariant,
    DJPracticalityScore,
    IntentBrief,
    MixConcept,
    SeedAnalysis,
    Track,
    Transition,
)


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
    ] + [SeedAnalysis(track_id=tid, tier="supporting", inferred_role="groove", drop_cost=0.5) for tid in supporting_ids]
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


def _make_practicality(overall: float) -> DJPracticalityScore:
    # Set all components equal so that DJPracticalityScore.overall == overall
    # (weights sum to 1.0, so equal components → overall == each component)
    return DJPracticalityScore(
        bpm_smoothness=overall,
        harmonic_ratio=overall,
        risk_justified=overall,
        fragment_preserved=overall,
    )


def test_select_best_variant_prefers_higher_practicality_score() -> None:
    concept_a = MixConcept(title="A", mood="practical", track_ids=["1", "2", "3"])
    concept_b = MixConcept(title="B", mood="balanced", track_ids=["1", "4", "5"])
    v_a = CompletionVariant(
        strategy="practical",
        concept=concept_a,
        anchor_retention_rate=1.0,
        practicality_score=_make_practicality(0.9),
    )
    v_b = CompletionVariant(
        strategy="balanced",
        concept=concept_b,
        anchor_retention_rate=0.5,
        practicality_score=_make_practicality(0.6),
    )
    best = _select_best_variant([v_a, v_b])
    assert best.strategy == "practical"


def test_select_best_variant_single_variant_returned_as_is() -> None:
    concept = MixConcept(title="T", mood="practical", track_ids=["1"])
    v = CompletionVariant(
        strategy="practical",
        concept=concept,
        anchor_retention_rate=1.0,
        practicality_score=_make_practicality(0.8),
    )
    assert _select_best_variant([v]) is v


def test_pair_consecutive_adjacent_returns_true() -> None:
    assert _pair_consecutive("a", "b", ["a", "b", "c"]) is True


def test_pair_consecutive_not_adjacent_returns_false() -> None:
    assert _pair_consecutive("a", "c", ["a", "b", "c"]) is False


def test_pair_consecutive_missing_id_returns_false() -> None:
    assert _pair_consecutive("x", "b", ["a", "b", "c"]) is False


def test_compute_practicality_score_perfect_bpm_smoothness() -> None:
    """All same BPM → stdev of deltas is 0 → bpm_smoothness=1.0."""
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2", "3", "4"])
    tracks_by_id = {str(i): _make_track(str(i), bpm=124.0, camelot_key="8A") for i in range(1, 5)}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.bpm_smoothness == 1.0


def test_compute_practicality_score_perfect_harmonic_ratio() -> None:
    """All adjacent Camelot keys (distance ≤ 1) → harmonic_ratio=1.0."""
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2", "3"])
    tracks_by_id = {
        "1": _make_track("1", camelot_key="8A"),
        "2": _make_track("2", camelot_key="9A"),
        "3": _make_track("3", camelot_key="9B"),
    }
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.harmonic_ratio == 1.0


def test_compute_practicality_score_zero_harmonic_ratio() -> None:
    """Keys far apart → harmonic_ratio=0.0."""
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2"])
    tracks_by_id = {
        "1": _make_track("1", camelot_key="1A"),
        "2": _make_track("2", camelot_key="7B"),
    }
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.harmonic_ratio == 0.0


def test_compute_practicality_score_no_transitions_gives_full_risk_score() -> None:
    """No Transition annotations → no annotated risks → risk_justified=1.0."""
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2"])
    tracks_by_id = {"1": _make_track("1"), "2": _make_track("2")}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.risk_justified == 1.0


def test_compute_practicality_score_cut_only_penalises_risk_score() -> None:
    """All transitions cut_only → risk_justified=0.0."""
    concept = MixConcept(
        title="T",
        mood="practical",
        track_ids=["1", "2", "3"],
        transitions=[
            Transition(from_id="1", to_id="2", is_risky=True, risk_type="cut_only"),
            Transition(from_id="2", to_id="3", is_risky=True, risk_type="cut_only"),
        ],
    )
    tracks_by_id = {str(i): _make_track(str(i)) for i in range(1, 4)}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.risk_justified == 0.0


def test_compute_practicality_score_named_risk_type_not_penalised() -> None:
    """chapter_pivot is a justified risk → risk_justified=1.0."""
    concept = MixConcept(
        title="T",
        mood="practical",
        track_ids=["1", "2"],
        transitions=[
            Transition(from_id="1", to_id="2", is_risky=True, risk_type="chapter_pivot"),
        ],
    )
    tracks_by_id = {"1": _make_track("1"), "2": _make_track("2")}
    score = _compute_practicality_score(concept, tracks_by_id, None)
    assert score.risk_justified == 1.0


def test_compute_practicality_score_fragment_preserved_with_adjacency() -> None:
    """Strong adjacency pair preserved in sequence → fragment_preserved=1.0."""
    frag = AdjacencyFragment(track_ids=["1", "2"], confidence=0.9, reason="bpm_close")
    brief = IntentBrief(
        overall_vibe="Test",
        energy_shape="unclear",
        risk_tolerance="medium",
        is_coherent_set=True,
        seed_analyses=[],
        missing_roles=[],
        strong_adjacencies=[frag],
        bpm_range=(120.0, 125.0),
    )
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2", "3"])
    tracks_by_id = {str(i): _make_track(str(i)) for i in range(1, 4)}
    score = _compute_practicality_score(concept, tracks_by_id, brief)
    assert score.fragment_preserved == 1.0


def test_compute_practicality_score_fragment_broken_reduces_score() -> None:
    """Strong adjacency pair broken (reversed) → fragment_preserved=0.0."""
    frag = AdjacencyFragment(track_ids=["1", "2"], confidence=0.9, reason="bpm_close")
    brief = IntentBrief(
        overall_vibe="Test",
        energy_shape="unclear",
        risk_tolerance="medium",
        is_coherent_set=True,
        seed_analyses=[],
        missing_roles=[],
        strong_adjacencies=[frag],
        bpm_range=(120.0, 125.0),
    )
    concept = MixConcept(title="T", mood="practical", track_ids=["2", "1", "3"])
    tracks_by_id = {str(i): _make_track(str(i)) for i in range(1, 4)}
    score = _compute_practicality_score(concept, tracks_by_id, brief)
    assert score.fragment_preserved == 0.0


def test_compute_practicality_score_overall_is_weighted_sum() -> None:
    """overall = bpm_smoothness*0.30 + harmonic_ratio*0.30 + risk_justified*0.25 + fragment_preserved*0.15."""
    s = DJPracticalityScore(bpm_smoothness=1.0, harmonic_ratio=0.5, risk_justified=0.8, fragment_preserved=0.6)
    expected = 1.0 * 0.30 + 0.5 * 0.30 + 0.8 * 0.25 + 0.6 * 0.15
    assert abs(s.overall - expected) < 1e-9


def test_select_best_variant_tiebreak_prefers_practical() -> None:
    """When practicality scores are equal, practical > balanced > adventurous."""
    concepts = [
        MixConcept(title="P", mood="practical", track_ids=["1"]),
        MixConcept(title="B", mood="balanced", track_ids=["2"]),
        MixConcept(title="A", mood="adventurous", track_ids=["3"]),
    ]
    variants = [
        CompletionVariant(
            strategy="adventurous",
            concept=concepts[2],
            anchor_retention_rate=1.0,
            practicality_score=_make_practicality(0.7),
        ),
        CompletionVariant(
            strategy="balanced",
            concept=concepts[1],
            anchor_retention_rate=1.0,
            practicality_score=_make_practicality(0.7),
        ),
        CompletionVariant(
            strategy="practical",
            concept=concepts[0],
            anchor_retention_rate=1.0,
            practicality_score=_make_practicality(0.7),
        ),
    ]
    best = _select_best_variant(variants)
    assert best.strategy == "practical"


def test_passes_floor_with_intent_brief_per_tier() -> None:
    """Variant retaining all anchors and enough supporting passes."""
    brief = _make_brief_with_tiers(anchor_ids=["a1", "a2", "a3", "a4"], supporting_ids=["s1", "s2"])
    # floor: ceil(4*0.75)=3 anchors, ceil(2*0.40)=1 supporting
    concept_pass = MixConcept(title="P", mood="practical", track_ids=["a1", "a2", "a3", "s1", "x1"])
    concept_fail = MixConcept(title="F", mood="practical", track_ids=["a1", "s1", "s2", "x1", "x2"])
    v_pass = CompletionVariant(
        strategy="practical",
        concept=concept_pass,
        anchor_retention_rate=0.75,
        practicality_score=_make_practicality(0.8),
    )
    v_fail = CompletionVariant(
        strategy="practical",
        concept=concept_fail,
        anchor_retention_rate=0.25,
        practicality_score=_make_practicality(0.9),
    )
    seed_ids = ["a1", "a2", "a3", "a4", "s1", "s2"]
    min_seeds = _minimum_playlist_seed_retention(len(seed_ids), brief)
    assert _passes_floor(v_pass, brief, seed_ids, min_seeds) is True
    assert _passes_floor(v_fail, brief, seed_ids, min_seeds) is False


def test_passes_floor_optional_seeds_do_not_satisfy_anchor_requirement() -> None:
    """A variant with many optional seeds but too few anchors must fail the floor."""
    brief = _make_brief_with_tiers(anchor_ids=["a1", "a2", "a3", "a4"], supporting_ids=[])
    # floor: ceil(4*0.75)=3 anchors
    # concept keeps only 2 anchors but 10 optional seeds — total retained > floor sum, but per-tier fails
    opt_ids = [f"o{i}" for i in range(10)]
    concept = MixConcept(title="T", mood="practical", track_ids=["a1", "a2"] + opt_ids)
    v = CompletionVariant(
        strategy="practical", concept=concept, anchor_retention_rate=0.5, practicality_score=_make_practicality(0.9)
    )
    seed_ids = ["a1", "a2", "a3", "a4"]
    min_seeds = _minimum_playlist_seed_retention(len(seed_ids), brief)
    assert _passes_floor(v, brief, seed_ids, min_seeds) is False


def test_score_variant_returns_completion_variant_with_practicality_score() -> None:
    concept = MixConcept(title="T", mood="practical", track_ids=["1", "2", "3"])
    tracks_by_id = {str(i): _make_track(str(i), bpm=124.0, camelot_key="8A") for i in range(1, 4)}
    variant = _score_variant(concept, ["1", "2"], None, tracks_by_id)
    assert variant.strategy == "practical"
    assert isinstance(variant.practicality_score, DJPracticalityScore)
    assert 0.0 <= variant.score <= 1.0


# ---------------------------------------------------------------------------
# stage0_intent_brief integration tests
# ---------------------------------------------------------------------------

_VALID_INTENT_RESPONSE = """{
  "overall_vibe": "Deep hypnotic groove",
  "is_coherent_set": true,
  "risk_tolerance": "medium",
  "missing_roles": ["closer"],
  "seed_analyses": [
    {"track_id": "1", "tier": "anchor", "inferred_role": "peak"},
    {"track_id": "2", "tier": "supporting", "inferred_role": "groove"},
    {"track_id": "3", "tier": "optional", "inferred_role": "opener"},
    {"track_id": "4", "tier": "anchor", "inferred_role": "pivot"},
    {"track_id": "5", "tier": "supporting", "inferred_role": "groove"},
    {"track_id": "6", "tier": "optional", "inferred_role": "resolution"}
  ]
}"""


async def test_stage0_returns_deterministic_brief_for_short_seed() -> None:
    """Seeds <=5 tracks skip LLM and return deterministic brief without calling any provider."""
    seed_tracks = [_make_track(str(i)) for i in range(1, 4)]
    seed_ids = [t.track_id for t in seed_tracks]
    state = make_cascade_state()
    # _CASCADE is built at module load time; replace it with a tracked mock to assert no call.
    sentinel = AsyncMock(return_value=None)
    sentinel.__name__ = "mock_provider"
    with patch("mixlab.llm._CASCADE", [sentinel]):
        brief = await stage0_intent_brief(seed_tracks, seed_ids, state, bpm_range=(124.0, 124.0))
    sentinel.assert_not_called()
    assert all(tid in brief.supporting_ids or tid in brief.anchor_ids for tid in seed_ids)


async def test_stage0_merges_llm_tiers_with_deterministic_shape() -> None:
    """LLM tiers are used; deterministic energy_shape, bpm_range, and adjacencies override."""
    seed_tracks = [_make_track(str(i), bpm=124.0 + i * 0.5) for i in range(1, 7)]
    seed_ids = [t.track_id for t in seed_tracks]
    state = make_cascade_state()
    # Patch _CASCADE so the first provider returns the desired response.
    llm_provider = AsyncMock(return_value=_VALID_INTENT_RESPONSE)
    llm_provider.__name__ = "mock_llm_provider"
    with patch("mixlab.llm._CASCADE", [llm_provider]):
        brief = await stage0_intent_brief(seed_tracks, seed_ids, state, bpm_range=(124.0, 126.5))
    assert "1" in brief.anchor_ids
    assert "2" in brief.supporting_ids
    assert brief.overall_vibe == "Deep hypnotic groove"
    assert brief.risk_tolerance == "medium"
    assert brief.energy_shape in ("single_arc", "double_peak", "plateau", "flat", "unclear")
    assert brief.bpm_range[0] <= brief.bpm_range[1]


async def test_stage0_falls_back_to_deterministic_when_all_providers_fail() -> None:
    """All cascade providers return None → deterministic fallback, no exception raised."""
    seed_tracks = [_make_track(str(i)) for i in range(1, 7)]
    seed_ids = [t.track_id for t in seed_tracks]
    state = make_cascade_state()
    # Three mock providers all returning None simulates every provider being unavailable.
    none_providers = [AsyncMock(return_value=None) for _ in range(3)]
    for p in none_providers:
        p.__name__ = "mock_none_provider"
    with patch("mixlab.llm._CASCADE", none_providers):
        brief = await stage0_intent_brief(seed_tracks, seed_ids, state, bpm_range=(124.0, 124.0))
    assert brief is not None
    assert all(tid in brief.supporting_ids or tid in brief.anchor_ids for tid in seed_ids)
