from mixlab import directions
from mixlab.mining import (  # noqa: F401 — asserts public interface
    MIN_SUPPORT,
    MinedPair,
    Predicate,
    extract_predicates,
    scan_pairs,
)
from mixlab.models import Track


def _t(track_id, *, label="", year=None, tags=None, energy=None, remixer="", bpm=174.0, key="8A", date_added=""):
    return Track(
        track_id=track_id,
        artist=f"A{track_id}",
        title=f"T{track_id}",
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        label=label,
        year=year,
        tags=tags or [],
        energy=energy,
        remixer=remixer,
        date_added=date_added,
    )


class TestExtractPredicates:
    def test_label_gate_eight(self):
        pool = (
            [_t(f"a{i}", label="Metalheadz") for i in range(8)]
            + [_t(f"b{i}", label="Dispatch") for i in range(7)]
            + [_t(f"c{i}") for i in range(10)]
        )
        kinds = {(p.kind, p.value) for p in extract_predicates(pool)}
        assert ("label", "Metalheadz") in kinds
        assert ("label", "Dispatch") not in kinds  # 7 < 8

    def test_remixer_gate_five(self):
        pool = [_t(f"a{i}", remixer="Calibre") for i in range(5)] + [_t(f"b{i}") for i in range(15)]
        assert ("remixer", "Calibre") in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_seventy_percent(self):
        # 96% of pool at high energy → predicate excluded (would ride along with anything)
        pool = [_t(f"a{i}", energy=8) for i in range(24)] + [_t("b0", energy=3)]
        assert ("energy", "high") not in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_boundary_exact_seventy_percent_kept(self):
        # 63/90 == exactly 70% — must be KEPT (cap drops only strictly >70%).
        # Regression for float rounding: 0.70 * 90 == 62.99999999999999 in IEEE-754
        # double precision, which would falsely exclude the 63rd track under a
        # float comparison (`63 <= 62.99999999999999` is False).
        pool = [_t(f"a{i}", tags=["boundary"]) for i in range(63)] + [_t(f"b{i}") for i in range(27)]
        assert ("tag", "boundary") in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_coverage_cap_boundary_just_over_seventy_percent_dropped(self):
        # 64/90 > 70% — must be DROPPED.
        pool = [_t(f"a{i}", tags=["boundary"]) for i in range(64)] + [_t(f"b{i}") for i in range(26)]
        assert ("tag", "boundary") not in {(p.kind, p.value) for p in extract_predicates(pool)}

    def test_era_calendar_aligned(self):
        pool = [_t(f"a{i}", year=2016 + (i % 4)) for i in range(10)] + [_t(f"b{i}") for i in range(5)]
        preds = {(p.kind, p.value) for p in extract_predicates(pool)}
        assert ("era", "2015-2019") in preds
        assert not any(k == "era" and v not in {"2015-2019"} for k, v in preds)

    def test_year_none_or_zero_joins_no_era(self):
        pool = [_t(f"a{i}", year=0) for i in range(10)] + [_t(f"b{i}") for i in range(10)]
        assert not any(p.kind == "era" for p in extract_predicates(pool))

    def test_mechanical_predicates_not_namable(self):
        pool = [_t(f"a{i}", bpm=174.0, key="8A") for i in range(10)] + [
            _t(f"b{i}", bpm=120.0, key="3B") for i in range(10)
        ]
        for p in extract_predicates(pool):
            if p.kind in ("bpm_regime", "key_hood"):
                assert p.namable is False

    def test_deterministic_order(self):
        pool = [_t(f"a{i}", label="Hospital Records", year=2020, tags=["Liquid"]) for i in range(10)] + [
            _t(f"b{i}") for i in range(5)
        ]
        assert extract_predicates(pool) == extract_predicates(list(reversed(pool)))

    def test_tags_lowercased(self):
        pool = [_t(f"a{i}", tags=["Liquid"]) for i in range(8)] + [_t(f"b{i}") for i in range(8)]
        assert ("tag", "liquid") in {(p.kind, p.value) for p in extract_predicates(pool)}


def _pred(kind, value, ids, namable=True):
    return Predicate(kind=kind, value=value, namable=namable, track_ids=frozenset(ids))


class TestScanPairs:
    def test_support_floor_matches_direction_pool(self):
        assert MIN_SUPPORT == directions.MIN_DIRECTION_POOL

    def test_lift_math(self):
        # N=100, |A|=20, |B|=20, |A∩B|=15 → lift = 15*100/400 = 3.75
        a = _pred("label", "Hospital Records", [f"x{i}" for i in range(15)] + [f"a{i}" for i in range(5)])
        b = _pred("era", "2015-2019", [f"x{i}" for i in range(15)] + [f"b{i}" for i in range(5)])
        [pair] = scan_pairs([a, b], pool_size=100)
        assert pair.support == 15
        assert abs(pair.lift - 3.75) < 1e-9

    def test_chance_pair_dies(self):
        # lift exactly 1.0 < 1.3 → gone (the first draft shipped these)
        a = _pred("era", "2000-2004", [f"x{i}" for i in range(20)])
        b = _pred("key_hood", "8A", [f"x{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=20) == []

    def test_support_floor_fifteen(self):
        a = _pred("label", "L", [f"x{i}" for i in range(14)] + ["a0"])
        b = _pred("era", "2020-2024", [f"x{i}" for i in range(14)] + ["b0"])
        assert scan_pairs([a, b], pool_size=200) == []  # support 14

    def test_same_kind_pairs_skipped(self):
        a = _pred("key_hood", "8A", [f"x{i}" for i in range(20)], namable=False)
        b = _pred("key_hood", "9A", [f"x{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=200) == []

    def test_both_mechanical_skipped(self):
        a = _pred("bpm_regime", "174", [f"x{i}" for i in range(20)], namable=False)
        b = _pred("key_hood", "8A", [f"x{i}" for i in range(20)], namable=False)
        assert scan_pairs([a, b], pool_size=200) == []  # no namable side

    def test_subsumption_pair_identical_to_parent_dies(self):
        ids = [f"x{i}" for i in range(20)]
        a = _pred("label", "L", ids)
        b = _pred("tag", "liquid", ids + [f"y{i}" for i in range(2)])
        # pair members == a's set exactly → Jaccard 1.0 vs parent a → dropped
        assert scan_pairs([a, b], pool_size=200) == []

    def test_shortlist_caps_at_twelve_by_lift(self):
        preds = []
        base = [f"s{i}" for i in range(15)]
        for i in range(15):
            preds.append(_pred("label", f"L{i:02d}", base + [f"l{i}-{j}" for j in range(5 + i)]))
            preds.append(_pred("tag", f"t{i:02d}", base + [f"g{i}-{j}" for j in range(5 + i)]))
        out = scan_pairs(preds, pool_size=3000)
        assert len(out) == 12
        lifts = [p.lift for p in out]
        assert lifts == sorted(lifts, reverse=True)

    def test_deterministic(self):
        a = _pred("label", "L", [f"x{i}" for i in range(16)])
        b = _pred("era", "2020-2024", [f"x{i}" for i in range(16)])
        assert scan_pairs([a, b], 100) == scan_pairs([b, a], 100)

    def test_pair_sides_ordered_and_members_sorted(self):
        a = _pred("tag", "liquid", [f"x{i}" for i in range(15)] + [f"a{i}" for i in range(6)])
        b = _pred("era", "2015-2019", [f"x{i}" for i in range(15)] + [f"b{i}" for i in range(6)])
        [pair] = scan_pairs([a, b], pool_size=100)
        assert (pair.a.kind, pair.a.value) < (pair.b.kind, pair.b.value)
        assert pair.a.kind == "era"
        assert pair.member_ids == tuple(sorted(f"x{i}" for i in range(15)))
        assert isinstance(pair, MinedPair)

    def test_empty_pool_size_returns_empty(self):
        a = _pred("label", "L", [f"x{i}" for i in range(16)])
        b = _pred("era", "2020-2024", [f"x{i}" for i in range(16)] + [f"b{i}" for i in range(20)])
        assert scan_pairs([a, b], pool_size=0) == []
