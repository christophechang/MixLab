from __future__ import annotations

from conftest import conj_pool as _conj_pool
from conftest import mech_conj_pool as _mech_conj_pool
from conftest import mining_track as _t

from mixlab import directions
from mixlab.mining import (  # noqa: F401 — asserts public interface
    MIN_SUPPORT,
    MinedPair,
    Predicate,
    extract_predicates,
    mine_pool,
    scan_pairs,
)


class TestExtractPredicates:
    def test_label_gate_eight(self) -> None:
        pool = (
            [_t(f"a{i}", label="Metalheadz") for i in range(8)]
            + [_t(f"b{i}", label="Dispatch") for i in range(7)]
            + [_t(f"c{i}") for i in range(10)]
        )
        kinds = {(p.kind, p.value) for p in extract_predicates(pool)}
        assert ("label", "Metalheadz") in kinds
        assert ("label", "Dispatch") not in kinds  # 7 < 8

    def test_remixer_gate_five(self) -> None:
        pool = [_t(f"a{i}", remixer="Calibre") for i in range(5)] + [_t(f"b{i}") for i in range(15)]
        assert ("remixer", "Calibre") in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_seventy_percent(self) -> None:
        # 96% of pool at high energy → predicate excluded (would ride along with anything)
        pool = [_t(f"a{i}", energy=8) for i in range(24)] + [_t("b0", energy=3)]
        assert ("energy", "high") not in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_boundary_exact_seventy_percent_kept(self) -> None:
        # 63/90 == exactly 70% — must be KEPT (cap drops only strictly >70%).
        # Regression for float rounding: 0.70 * 90 == 62.99999999999999 in IEEE-754
        # double precision, which would falsely exclude the 63rd track under a
        # float comparison (`63 <= 62.99999999999999` is False).
        pool = [_t(f"a{i}", tags=["boundary"]) for i in range(63)] + [_t(f"b{i}") for i in range(27)]
        assert ("tag", "boundary") in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_boundary_just_over_seventy_percent_dropped(self) -> None:
        # 64/90 > 70% — must be DROPPED.
        pool = [_t(f"a{i}", tags=["boundary"]) for i in range(64)] + [_t(f"b{i}") for i in range(26)]
        assert ("tag", "boundary") not in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_era_calendar_aligned(self) -> None:
        pool = [_t(f"a{i}", year=2016 + (i % 4)) for i in range(10)] + [_t(f"b{i}") for i in range(5)]
        preds = {(p.kind, p.value) for p in extract_predicates(pool)}
        assert ("era", "2015-2019") in preds
        assert not any(k == "era" and v not in {"2015-2019"} for k, v in preds)

    def test_year_none_or_zero_joins_no_era(self) -> None:
        pool = [_t(f"a{i}", year=0) for i in range(10)] + [_t(f"b{i}") for i in range(10)]
        assert not any(p.kind == "era" for p in extract_predicates(pool))

    def test_mechanical_predicates_not_namable(self) -> None:
        pool = [_t(f"a{i}", bpm=174.0, key="8A") for i in range(10)] + [
            _t(f"b{i}", bpm=120.0, key="3B") for i in range(10)
        ]
        for p in extract_predicates(pool):
            if p.kind in ("bpm_regime", "key_hood"):
                assert p.namable is False

    def test_deterministic_order(self) -> None:
        pool = [_t(f"a{i}", label="Hospital Records", year=2020, tags=["Liquid"]) for i in range(10)] + [
            _t(f"b{i}") for i in range(5)
        ]
        assert extract_predicates(pool) == extract_predicates(list(reversed(pool)))

    def test_tags_lowercased(self) -> None:
        pool = [_t(f"a{i}", tags=["Liquid"]) for i in range(8)] + [_t(f"b{i}") for i in range(8)]
        assert ("tag", "liquid") in {(p.kind, p.value) for p in extract_predicates(pool)}


def _pred(kind: str, value: str, ids: list[str], namable: bool = True) -> Predicate:
    return Predicate(kind=kind, value=value, namable=namable, track_ids=frozenset(ids))


_SHARED_15 = tuple(f"x{i}" for i in range(15))


def _unsubsumed(kind: str, value: str, private_prefix: str, namable: bool = True) -> Predicate:
    """A predicate over 15 shared members plus 5 private ones.

    Two of these intersect in exactly the 15 shared ids, giving
    Jaccard(intersection, parent) == 15/20 == 0.75 — well under the 0.9
    subsumption threshold, and support 15 / lift >= 1.3 at any sane pool size.
    Negative tests build from this so the pair can only die at the gate under
    test, never at subsumption.
    """
    ids = list(_SHARED_15) + [f"{private_prefix}{i}" for i in range(5)]
    return _pred(kind, value, ids, namable=namable)


class TestScanPairs:
    def test_support_floor_matches_direction_pool(self) -> None:
        assert MIN_SUPPORT == directions.MIN_DIRECTION_POOL

    def test_lift_math(self) -> None:
        # N=100, |A|=20, |B|=20, |A∩B|=15 → lift = 15*100/400 = 3.75
        a = _pred("label", "Hospital Records", [f"x{i}" for i in range(15)] + [f"a{i}" for i in range(5)])
        b = _pred("era", "2015-2019", [f"x{i}" for i in range(15)] + [f"b{i}" for i in range(5)])
        [pair] = scan_pairs([a, b], pool_size=100)
        assert pair.support == 15
        assert abs(pair.lift - 3.75) < 1e-9

    def test_chance_pair_dies(self) -> None:
        # lift exactly 1.0 < 1.3 → gone (the first draft shipped these).
        # |A| = |B| = 40, |A∩B| = 20, N = 80 → lift 1.0. Support clears 15 and
        # Jaccard is 0.5, so the lift gate is the only thing that can drop this.
        shared = [f"x{i}" for i in range(20)]
        a = _pred("era", "2000-2004", shared + [f"a{i}" for i in range(20)])
        b = _pred("key_hood", "8A", shared + [f"b{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=80) == []

    def test_support_floor_fifteen(self) -> None:
        # support 14, one short. Jaccard 14/19 == 0.74 and lift 7.76, so only the
        # support floor can drop it — lower MIN_SUPPORT to 14 and this fails.
        shared = [f"x{i}" for i in range(14)]
        a = _pred("label", "L", shared + [f"a{i}" for i in range(5)])
        b = _pred("era", "2020-2024", shared + [f"b{i}" for i in range(5)])
        assert scan_pairs([a, b], pool_size=200) == []

    def test_same_kind_pairs_skipped(self) -> None:
        # Both namable and both unsubsumed, so nothing but the cross-kind rule
        # stands between this pair and the shortlist.
        a = _unsubsumed("era", "2000-2004", "a")
        b = _unsubsumed("era", "2005-2009", "b")
        assert scan_pairs([a, b], pool_size=200) == []

    def test_both_mechanical_skipped(self) -> None:
        # Cross-kind, support 15, lift 7.5, Jaccard 0.75 — every other gate is
        # satisfied, so deleting the namable gate makes this fail.
        a = _unsubsumed("bpm_regime", "174", "a", namable=False)
        b = _unsubsumed("key_hood", "8A", "b", namable=False)
        assert scan_pairs([a, b], pool_size=200) == []

    def test_subsumption_pair_identical_to_parent_dies(self) -> None:
        ids = [f"x{i}" for i in range(20)]
        a = _pred("label", "L", ids)
        b = _pred("tag", "liquid", ids + [f"y{i}" for i in range(2)])
        # pair members == a's set exactly → Jaccard 1.0 vs parent a → dropped
        assert scan_pairs([a, b], pool_size=200) == []

    def test_shortlist_caps_at_twelve_by_lift(self) -> None:
        preds = []
        base = [f"s{i}" for i in range(15)]
        for i in range(15):
            preds.append(_pred("label", f"L{i:02d}", base + [f"l{i}-{j}" for j in range(5 + i)]))
            preds.append(_pred("tag", f"t{i:02d}", base + [f"g{i}-{j}" for j in range(5 + i)]))
        out = scan_pairs(preds, pool_size=3000)
        assert len(out) == 12
        lifts = [p.lift for p in out]
        assert lifts == sorted(lifts, reverse=True)

    def test_deterministic(self) -> None:
        a = _unsubsumed("label", "L", "a")
        b = _unsubsumed("era", "2020-2024", "b")
        forward, reverse = scan_pairs([a, b], 100), scan_pairs([b, a], 100)
        assert forward != []  # the fixture survives — this is not [] == []
        assert forward == reverse

    def test_pair_sides_ordered_and_members_sorted(self) -> None:
        a = _pred("tag", "liquid", [f"x{i}" for i in range(15)] + [f"a{i}" for i in range(6)])
        b = _pred("era", "2015-2019", [f"x{i}" for i in range(15)] + [f"b{i}" for i in range(6)])
        [pair] = scan_pairs([a, b], pool_size=100)
        assert (pair.a.kind, pair.a.value) < (pair.b.kind, pair.b.value)
        assert pair.a.kind == "era"
        assert pair.member_ids == tuple(sorted(f"x{i}" for i in range(15)))
        assert isinstance(pair, MinedPair)

    def test_non_positive_pool_size_returns_empty(self) -> None:
        a = _unsubsumed("label", "L", "a")
        b = _unsubsumed("era", "2020-2024", "b")
        assert scan_pairs([a, b], pool_size=200) != []  # survives at a real pool size
        assert scan_pairs([a, b], pool_size=0) == []
        assert scan_pairs([a, b], pool_size=-1) == []


class TestMinePool:
    def test_finds_the_planted_conjunction(self) -> None:
        found = mine_pool(_conj_pool())
        assert found, "expected at least one mined direction"
        top = found[0]
        assert top.direction_type == "found"
        assert top.title == "Found: Hospital Records × liquid"
        assert 15 <= len(top.track_ids) <= 25
        assert top.identity > 0.0
        assert top.feasibility == 0.0  # field scorer owns the final score

    def test_title_never_contains_mechanical_values(self) -> None:
        for d in mine_pool(_conj_pool()):
            for banned in ("BPM", "8A", "174"):
                assert banned not in d.title
                assert banned not in d.mood

    def test_mood_is_ascii_kind_pair(self) -> None:
        found = mine_pool(_conj_pool())
        assert found[0].mood == "label x tag"
        assert found[0].mood.isascii()

    def test_brief_leads_with_imperative_not_statistic(self) -> None:
        found = mine_pool(_conj_pool())
        assert found[0].brief.startswith("FOUND SET. Play")
        assert "denser than chance" in found[0].brief

    def test_brief_cites_selection_when_support_exceeds_cap(self) -> None:
        # widen overlap to 40 so support > 25
        pool = _conj_pool() + [
            _t(
                f"hx{i}",
                label="Hospital Records",
                tags=["Liquid"],
                year=2020,
                bpm=174.0,
                date_added=f"2026-0{1 + i % 6}-15",
            )
            for i in range(20)
        ]
        found = mine_pool(pool)
        top = next(d for d in found if d.title == "Found: Hospital Records × liquid")
        assert "most central are selected" in top.brief
        assert len(top.track_ids) == 25

    def test_empty_and_sparse_pools_mine_nothing(self) -> None:
        assert mine_pool([]) == []
        assert mine_pool([_t(f"s{i}") for i in range(10)]) == []

    def test_single_namable_title_has_no_mechanical_value_or_multiplication_sign(self) -> None:
        found = mine_pool(_mech_conj_pool())
        assert found, "expected the planted tag x bpm_regime conjunction to fire"
        top = found[0]
        assert top.title == "Found: liquid"
        assert "×" not in top.title
        for banned in ("BPM", "8A", "130"):
            assert banned not in top.title
            assert banned not in top.mood

    def test_single_namable_mood_uses_display_noun_and_is_ascii(self) -> None:
        found = mine_pool(_mech_conj_pool())
        assert found[0].mood == "tempo x tag"
        assert found[0].mood.isascii()

    def test_single_namable_brief_cites_mechanical_value_with_bpm_suffix(self) -> None:
        found = mine_pool(_mech_conj_pool())
        assert "clusters in one tempo pocket (around 130 BPM)" in found[0].brief
