from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, cast

import pytest
from conftest import conj_pool

from mixlab.config import CUSTOM_GENRES, GENRE_MAP
from mixlab.library_map import build_map_payload, render_map_json
from mixlab.models import PlayedTrack, Track


def _track(
    *,
    track_id: str,
    artist: str,
    title: str,
    bpm: float,
    camelot_key: str,
    genre: str,
) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=camelot_key,
        genre=genre,
    )


@pytest.fixture
def map_tracks() -> list[Track]:
    """A dozen tracks across House/Techno GENRE_MAP tags, plus a Drum & Bass
    inclusion/exclusion pair straddling the "170" custom pool's (165.0, 175.0)
    BPM range (see CUSTOM_GENRES in config.py) — "170" only admits
    drum_and_bass/jungle genres, so the pair must carry that genre for the
    filter to actually be exercised.
    """
    house = [
        _track(
            track_id=f"h{i}",
            artist=f"House Artist {i}",
            title=f"House Track {i}",
            bpm=122.0 + i,
            camelot_key="8B",
            genre="House",
        )
        for i in range(5)
    ]
    techno = [
        _track(
            track_id=f"t{i}",
            artist=f"Techno Artist {i}",
            title=f"Techno Track {i}",
            bpm=130.0 + i,
            camelot_key="9A",
            genre="Techno",
        )
        for i in range(5)
    ]
    dnb_in_range = _track(
        track_id="dnb_in",
        artist="DnB Artist In",
        title="DnB Track In",
        bpm=170.0,
        camelot_key="4A",
        genre="Drum & Bass",
    )
    dnb_out_of_range = _track(
        track_id="dnb_out",
        artist="DnB Artist Out",
        title="DnB Track Out",
        bpm=185.0,
        camelot_key="4A",
        genre="Drum & Bass",
    )
    return [*house, *techno, dnb_in_range, dnb_out_of_range]


def test_build_map_payload_all_mode_covers_every_pool_key(map_tracks: list[Track]) -> None:
    payload = build_map_payload(map_tracks, mode="all", seed=0, played=[])
    assert payload["version"] == 1
    assert payload["mode"] == "all"
    pools = cast("dict[str, dict[str, object]]", payload["pools"])
    assert list(pools) == [*GENRE_MAP, *CUSTOM_GENRES]
    for entry in pools.values():
        assert set(entry) == {"track_count", "directions"}


def test_build_map_payload_unplayed_mode_filters_catalog_matches(map_tracks: list[Track]) -> None:
    # Pick one track from the fixture; a PlayedTrack with its artist/title drops it.
    victim = map_tracks[0]
    played = [PlayedTrack(artist=victim.artist, title=victim.title)]
    all_counts = build_map_payload(map_tracks, mode="all", seed=0, played=played)
    unplayed = build_map_payload(map_tracks, mode="unplayed", seed=0, played=played)
    assert unplayed["catalog_tracks"] == 1
    total: Callable[[dict[str, object]], int] = lambda p: sum(  # noqa: E731
        cast(int, e["track_count"])
        for k, e in cast("dict[str, dict[str, object]]", p["pools"]).items()
        if k in GENRE_MAP
    )
    assert total(unplayed) == total(all_counts) - 1


def test_build_map_payload_custom_pool_170_applies_bpm_filter(map_tracks: list[Track]) -> None:
    # Fixture carries exactly one Drum & Bass track inside the "170" custom pool's
    # (165.0, 175.0) BPM range (dnb_in, bpm=170.0) and one outside it (dnb_out,
    # bpm=185.0) — a reversed or off-by-one boundary would silently pass the other
    # two tests, since neither asserts on custom-pool contents.
    payload = build_map_payload(map_tracks, mode="all", seed=0, played=[])
    pools = cast("dict[str, dict[str, object]]", payload["pools"])
    pool_170 = pools["170"]
    assert pool_170["track_count"] == 1
    directions = cast("list[dict[str, object]]", pool_170["directions"])
    for direction in directions:
        track_ids = cast("list[str]", direction["track_ids"])
        assert "dnb_out" not in track_ids


def test_render_map_json_deterministic_and_parseable(map_tracks: list[Track]) -> None:
    payload = build_map_payload(map_tracks, mode="all", seed=3, played=[])
    first = render_map_json(payload)
    second = render_map_json(build_map_payload(map_tracks, mode="all", seed=3, played=[]))
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["seed"] == 3


def test_run_map_cli_all_mode_prints_json_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], map_tracks: list[Track]
) -> None:
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "apply_bpm_corrections", lambda tracks: tracks)
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    exit_code = cli._run_map_cli("all", 0, None)
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "all"
    assert "pools" in payload


def test_run_map_cli_out_flag_writes_identical_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, map_tracks: list[Track]
) -> None:
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "apply_bpm_corrections", lambda tracks: tracks)
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    target = tmp_path / "map.json"
    assert cli._run_map_cli("all", 0, target) == 0
    assert target.read_text() == capsys.readouterr().out


def test_run_map_cli_unplayed_mode_without_catalog_url_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], map_tracks: list[Track]
) -> None:
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "apply_bpm_corrections", lambda tracks: tracks)
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    monkeypatch.delenv("CATALOG_API_URL", raising=False)
    exit_code = cli._run_map_cli("unplayed", 0, None)
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "CATALOG_API_URL" in captured.err


def test_run_map_cli_unplayed_mode_emits_pure_json_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], map_tracks: list[Track]
) -> None:
    """Regression for the --map contract: stdout must be *only* the rendered JSON, even
    in the default (unplayed) mode, which fetches played tracks from the catalog API.
    ``fetch_played_tracks`` used to print a progress line to stdout ahead of the
    payload — the worker capturing stdout for JSON.parse would fail on that line.
    """
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "apply_bpm_corrections", lambda tracks: tracks)
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    monkeypatch.setenv("CATALOG_API_URL", "https://api.changsta.com")
    monkeypatch.setenv("CHANGSTA_API_KEY", "test-key")

    async def _fake_fetch_played_tracks(api_key: str, base_url: str) -> list[PlayedTrack]:
        return []

    monkeypatch.setattr(cli, "fetch_played_tracks", _fake_fetch_played_tracks)

    exit_code = cli._run_map_cli("unplayed", 0, None)
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)  # raises if anything precedes/follows the JSON
    assert payload["mode"] == "unplayed"


def test_run_map_cli_out_write_failure_reports_stderr_keeps_stdout_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, map_tracks: list[Track]
) -> None:
    """F2: a failing ``--out`` write must not raise an uncaught traceback after the JSON
    payload has already gone to stdout — it should report clearly on stderr and exit 1,
    leaving stdout as pure, already-emitted JSON.
    """
    import mixlab.__main__ as cli

    monkeypatch.setattr(cli, "parse_collection", lambda _path: list(map_tracks))
    monkeypatch.setattr(cli, "apply_bpm_corrections", lambda tracks: tracks)
    monkeypatch.setattr(cli, "_apply_do_not_recommend_filter", lambda tracks, _p: (tracks, 0))
    target = tmp_path / "does-not-exist" / "map.json"

    exit_code = cli._run_map_cli("all", 0, target)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(target) in captured.err
    payload = json.loads(captured.out)
    assert payload["mode"] == "all"


class TestMapPayloadWithMining:
    """Integration coverage for mined ``found_N`` rows reaching the map payload (Task 8).

    Reuses tests/conftest.py's ``conj_pool`` (Task 6) rather than assembling a new
    fixture: its 80 tracks plant a Hospital Records x Liquid conjunction that
    provably fires (support=20, lift=2.56 — see conftest for why the brief's
    original 20/20/20/20 split under-lifted), and every track carries the
    default genre "Drum & Bass", which lands the whole pool in
    ``build_map_payload``'s "drum_and_bass" GENRE_MAP pool — so the miner runs on
    exactly the pool it was tuned against.
    """

    def test_found_rows_ship_with_valid_shape(self) -> None:
        payload = build_map_payload(conj_pool(), mode="all", seed=0, played=[])
        pools = cast("dict[str, dict[str, object]]", payload["pools"])
        found = [
            d
            for e in pools.values()
            for d in cast("list[dict[str, object]]", e["directions"])
            if cast(str, d["direction_type"]).startswith("found_")
        ]
        assert found, "fixture collection must mine at least one found row"
        for d in found:
            assert set(d) == {"direction_type", "title", "mood", "brief", "feasibility", "track_ids", "thread_artist"}
            assert cast(str, d["title"]).startswith("Found: ")
            assert 0.0 < cast(float, d["feasibility"]) <= 1.0
            assert 15 <= len(cast("list[str]", d["track_ids"])) <= 25
            assert cast(str, d["mood"]).isascii()

    def test_payload_byte_identical_across_runs(self) -> None:
        a = render_map_json(build_map_payload(conj_pool(), mode="all", seed=0, played=[]))
        b = render_map_json(build_map_payload(conj_pool(), mode="all", seed=0, played=[]))
        assert a == b


def test_direction_entries_carry_thread_artist() -> None:
    """Every direction entry ships thread_artist (empty for non-thread types) so the
    web app can round-trip it into a --direction-spec without re-deriving it."""
    payload = build_map_payload(conj_pool(), mode="all", seed=0, played=[])
    pools = cast("dict[str, dict[str, object]]", payload["pools"])
    entries = [
        d for e in pools.values() for d in cast("list[dict[str, object]]", e["directions"])
    ]
    assert entries
    for d in entries:
        assert "thread_artist" in d
        if d["direction_type"] == "artist_thread":
            assert d["thread_artist"], "artist_thread entries must name the spine artist"
        assert isinstance(d["thread_artist"], str)
