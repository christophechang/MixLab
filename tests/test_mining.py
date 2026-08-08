from mixlab.mining import Predicate, extract_predicates  # noqa: F401 — asserts public interface
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
