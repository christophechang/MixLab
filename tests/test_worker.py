from __future__ import annotations

import gzip
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import respx
from httpx import Response

import mixlab.worker as worker_module
from mixlab.__main__ import main
from mixlab.remote import MixLabRemote, RemoteConfig
from mixlab.sync import SyncReport
from mixlab.worker import (
    _MAP_XML_PATH,
    WorkerConfig,
    WorkerFlagError,
    _resolve_optional_export,
    _run_claimed_map,
    build_argv,
    process_one,
    run_worker,
)

_BASE = "https://mixlab-api.example.com"
_SECRET = "test-secret"

_CLAIM_URL = f"{_BASE}/api/mixlab/worker/claim"
_MAP_CLAIM_URL = f"{_BASE}/api/mixlab/maps/claim"


def _remote() -> MixLabRemote:
    return MixLabRemote(RemoteConfig(base_url=_BASE, api_secret=_SECRET))


def _config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(
        xml_path=tmp_path / "import" / "rekordbox.xml",
        run_timeout=1200.0,
        repo_root=tmp_path,
        worker_id="test-worker",
    )


def test_worker_config_from_env_defaults_collection_off_the_local_import_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default download target is isolated under .mixlab/, NOT import/rekordbox.xml, so a
    # remote run never clobbers the collection a local run/automation uses.
    monkeypatch.delenv("MIXLAB_WORKER_XML_PATH", raising=False)
    config = WorkerConfig.from_env(repo_root=tmp_path)
    assert config.xml_path == tmp_path / ".mixlab" / "worker-collection.xml"
    assert config.xml_path != tmp_path / "import" / "rekordbox.xml"


def _manifest_dict(
    *,
    run_id: str = "r1",
    flags: dict[str, object] | None = None,
    upload_id: str = "u1",
) -> dict[str, object]:
    return {
        "runId": run_id,
        "createdAt": "2026-07-08T00:00:00Z",
        "status": "running",
        "flags": flags if flags is not None else {},
        "uploadId": upload_id,
        "claimedAt": "2026-07-08T00:00:01Z",
        "workerId": "test-worker",
        "concepts": [],
    }


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


def _completed_bytes(
    *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
) -> subprocess.CompletedProcess[bytes]:
    # Map jobs run without text=True — stdout/stderr come back as bytes (the payload
    # contract requires the raw stdout bytes, untouched, to reach complete_map).
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


def _write_artifacts(repo_root: Path) -> str:
    """Create the report + summary artifacts and return the pipeline stdout that names them."""
    reports = repo_root / "output" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "run.html").write_text("<html></html>", encoding="utf-8")
    (reports / "run.json").write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    return "Some progress\nHTML report: output/reports/run.html\nRun summary: output/reports/run.json\nDone\n"


# ===========================================================================
# build_argv — table-driven
# ===========================================================================


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({}, []),
        ({"genre": "house"}, ["--genre", "house"]),
        ({"genre": "170"}, ["--genre", "170"]),  # custom label
        ({"genre": "My Custom Vibe"}, ["--genre", "My Custom Vibe"]),  # arbitrary label
        ({"mode": "all"}, ["--mode", "all"]),
        ({"risk": "high"}, ["--risk", "high"]),
        ({"directions": "only"}, ["--directions", "only"]),
        ({"intent": "warmup set"}, ["--intent", "warmup set"]),
        ({"mixLength": 60}, ["--mix-length", "60"]),
        ({"stage1Seed": 20260708}, ["--stage1-seed", "20260708"]),
        ({"resequence": True}, ["--resequence"]),
        ({"resequence": False}, []),
        ({"deep": True}, ["--deep"]),
        ({"deep": False}, []),
        ({"genre": None, "mode": None}, []),  # None omitted
        ({"intent": None}, []),
        ({"mixLength": None}, []),
        (
            # full manifest — asserts fixed canonical ordering regardless of input order
            {
                "deep": True,
                "resequence": True,
                "stage1Seed": 7,
                "mixLength": 90,
                "intent": "late night",
                "directions": "mixed",
                "risk": "low",
                "mode": "played",
                "genre": "techno",
            },
            [
                "--genre",
                "techno",
                "--mode",
                "played",
                "--risk",
                "low",
                "--directions",
                "mixed",
                "--intent",
                "late night",
                "--mix-length",
                "90",
                "--stage1-seed",
                "7",
                "--resequence",
                "--deep",
            ],
        ),
    ],
)
def test_build_argv_valid_flags_maps_to_expected_argv(flags: dict[str, object], expected: list[str]) -> None:
    assert build_argv(flags) == expected


def test_build_argv_input_order_does_not_change_output_order() -> None:
    forward = build_argv({"genre": "house", "mode": "all", "deep": True})
    reversed_in = build_argv({"deep": True, "mode": "all", "genre": "house"})
    assert forward == reversed_in == ["--genre", "house", "--mode", "all", "--deep"]


def test_build_argv_unknown_key_raises_naming_key() -> None:
    with pytest.raises(WorkerFlagError, match="bogus"):
        build_argv({"bogus": "x"})


def test_build_argv_bool_as_int_for_mixlength_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="mixLength"):
        build_argv({"mixLength": True})


def test_build_argv_non_int_mixlength_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="mixLength"):
        build_argv({"mixLength": "60"})


def test_build_argv_float_mixlength_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="mixLength"):
        build_argv({"mixLength": 60.0})


def test_build_argv_bad_enum_mode_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="mode"):
        build_argv({"mode": "sideways"})


def test_build_argv_bad_enum_risk_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="risk"):
        build_argv({"risk": "reckless"})


def test_build_argv_non_string_genre_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="genre"):
        build_argv({"genre": 5})


def test_build_argv_empty_genre_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="genre"):
        build_argv({"genre": ""})


def test_build_argv_non_bool_resequence_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="resequence"):
        build_argv({"resequence": 1})


def test_build_argv_maps_playlist() -> None:
    assert build_argv({"playlist": "DJ CRATES 2025/Eclectic"}) == ["--playlist", "DJ CRATES 2025/Eclectic"]


def test_build_argv_empty_playlist_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="playlist"):
        build_argv({"playlist": ""})


def test_build_argv_maps_float_bpm() -> None:
    assert build_argv({"minBpm": 128.5}) == ["--min-bpm", "128.5"]


def test_build_argv_maps_int_bpm_as_number() -> None:
    assert build_argv({"maxBpm": 130}) == ["--max-bpm", "130"]


def test_build_argv_maps_year_ints() -> None:
    assert build_argv({"minYear": 2000, "maxYear": 2020}) == ["--min-year", "2000", "--max-year", "2020"]


def test_build_argv_bool_as_bpm_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="minBpm"):
        build_argv({"minBpm": True})


def test_build_argv_string_bpm_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="minBpm"):
        build_argv({"minBpm": "128"})


def test_build_argv_float_year_rejected() -> None:
    with pytest.raises(WorkerFlagError, match="minYear"):
        build_argv({"minYear": 2000.5})


# ===========================================================================
# process_one — full cycle + failure taxonomy
# ===========================================================================


@respx.mock
def test_process_one_happy_cycle_completes_and_syncs(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    manifest = _manifest_dict(flags={"genre": "house", "mode": "all", "resequence": True})

    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=manifest))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(
        return_value=Response(200, content=gzip.compress(b"<COLLECTION/>"))
    )
    complete_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/complete").mock(return_value=Response(200))

    stdout = _write_artifacts(tmp_path)
    history_path = tmp_path / ".mixlab" / "concept-history.json"

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()) as sync_down_mock,
        patch("mixlab.worker.sync_up") as sync_up_mock,
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)) as run_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    # sync_down ran with the repo-root history path
    sync_down_mock.assert_called_once_with(remote, history_path)
    # collection downloaded + gunzipped to xml_path
    assert config.xml_path.read_bytes() == b"<COLLECTION/>"
    # subprocess invoked with the manifest flags followed by the worker's own --export flag
    run_argv = run_mock.call_args.args[0]
    assert run_argv[:3] == [run_argv[0], "-m", "mixlab"]
    assert run_argv[3:] == [
        "--genre",
        "house",
        "--mode",
        "all",
        "--resequence",
        "--export",
        str(tmp_path / "output" / "worker-export"),
    ]
    # the export dir is created up-front so the pipeline writes into an existing directory
    assert (tmp_path / "output" / "worker-export").is_dir()
    assert run_mock.call_args.kwargs["cwd"] == tmp_path
    assert run_mock.call_args.kwargs["timeout"] == 1200.0
    # the pipeline is pointed at the worker's isolated collection via the environment
    assert run_mock.call_args.kwargs["env"]["MIXLAB_COLLECTION_PATH"] == str(config.xml_path)
    # complete posted, then sync_up ran
    assert complete_route.called
    sync_up_mock.assert_called_once_with(remote, history_path)


def test_resolve_optional_export_absent_line_returns_none(tmp_path: Path) -> None:
    assert _resolve_optional_export("HTML report: x\nRun summary: y\n", tmp_path) is None


def test_resolve_optional_export_present_but_missing_file_returns_none(tmp_path: Path) -> None:
    # The line names a file the pipeline claims it wrote, but it isn't on disk — treat as
    # absent rather than failing an otherwise-good run.
    assert _resolve_optional_export("Exported: output/worker-export/gone.xml\n", tmp_path) is None


def test_resolve_optional_export_present_file_resolves_against_repo_root(tmp_path: Path) -> None:
    export = tmp_path / "output" / "worker-export" / "rekordbox_export.xml"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text("<DJ_PLAYLISTS/>", encoding="utf-8")
    assert _resolve_optional_export("Exported: output/worker-export/rekordbox_export.xml\n", tmp_path) == export


@respx.mock
def test_process_one_uploads_export_when_pipeline_writes_one(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(
        return_value=Response(200, content=gzip.compress(b"<COLLECTION/>"))
    )

    # Report + summary, plus an exported playlist XML that actually exists on disk.
    stdout = _write_artifacts(tmp_path)
    export_dir = tmp_path / "output" / "worker-export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_file = export_dir / "rekordbox_export.xml"
    export_file.write_text("<DJ_PLAYLISTS/>", encoding="utf-8")
    stdout += f"Exported: {export_file}\n"

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch("mixlab.worker.sync_up"),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
        patch.object(remote, "complete") as complete_mock,
    ):
        assert process_one(remote, config) is True

    complete_mock.assert_called_once()
    assert complete_mock.call_args.kwargs["export_path"] == export_file


@respx.mock
def test_process_one_completes_without_export_when_pipeline_writes_none(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(
        return_value=Response(200, content=gzip.compress(b"<COLLECTION/>"))
    )

    # No "Exported:" line — a run that produced no playlist. Completion still succeeds.
    stdout = _write_artifacts(tmp_path)

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch("mixlab.worker.sync_up"),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
        patch.object(remote, "complete") as complete_mock,
    ):
        assert process_one(remote, config) is True

    complete_mock.assert_called_once()
    assert complete_mock.call_args.kwargs["export_path"] is None


@respx.mock
def test_process_one_pipeline_nonzero_fails_with_truncated_log_tail(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    oversized = _completed(returncode=1, stdout="x" * 5000, stderr="y" * 5000)

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch.object(remote, "download_collection", return_value=config.xml_path),
        patch("mixlab.worker.subprocess.run", return_value=oversized),
    ):
        result = process_one(remote, config)

    assert result is True
    assert fail_route.called
    body = json.loads(fail_route.calls.last.request.content)
    assert body["error"] == "pipeline exited 1"
    # log tail is combined stderr+stdout, truncated to the last 4000 chars
    assert len(body["logTail"]) == 4000
    assert body["logTail"] == "x" * 4000


@respx.mock
def test_process_one_timeout_fails_mentioning_timeout(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    timeout_exc = subprocess.TimeoutExpired(cmd="mixlab", timeout=1200.0, output="partial-out", stderr="partial-err")

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch.object(remote, "download_collection", return_value=config.xml_path),
        patch("mixlab.worker.subprocess.run", side_effect=timeout_exc),
    ):
        result = process_one(remote, config)

    assert result is True
    body = json.loads(fail_route.calls.last.request.content)
    assert "timed out" in body["error"]
    assert "1200" in body["error"]
    assert "partial-out" in body["logTail"]
    assert "partial-err" in body["logTail"]


@respx.mock
def test_process_one_unknown_flag_fails_without_running_subprocess(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"nope": "x"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch.object(remote, "download_collection", return_value=config.xml_path),
        patch("mixlab.worker.subprocess.run") as run_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    run_mock.assert_not_called()
    body = json.loads(fail_route.calls.last.request.content)
    assert "nope" in body["error"]


@respx.mock
def test_process_one_sync_report_with_activity_printed_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(
        return_value=Response(200, content=gzip.compress(b"<COLLECTION/>"))
    )
    respx.post(f"{_BASE}/api/mixlab/runs/r1/complete").mock(return_value=Response(200))
    stdout = _write_artifacts(tmp_path)

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport(applied=1, skipped=0, acked=1)),
        patch("mixlab.worker.sync_up"),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
    ):
        assert process_one(remote, config) is True

    # The smoke runbook (docs/ops/anywhere-smoke.md step 7) greps the launchd stdout
    # log for exactly this line.
    assert "sync: applied 1 verdict(s), skipped 0, acked 1" in capsys.readouterr().out


@respx.mock
def test_process_one_sync_report_without_activity_stays_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(
        return_value=Response(200, content=gzip.compress(b"<COLLECTION/>"))
    )
    respx.post(f"{_BASE}/api/mixlab/runs/r1/complete").mock(return_value=Response(200))
    stdout = _write_artifacts(tmp_path)

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch("mixlab.worker.sync_up"),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
    ):
        assert process_one(remote, config) is True

    assert "sync:" not in capsys.readouterr().out


@respx.mock
def test_process_one_empty_queue_returns_false_and_does_nothing(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    # Both queues empty: run claim 204, then the map claim 204 too — the genuine
    # both-empty path (a respx.mock run leaves the map-claim route unmocked, an
    # unmocked request raises, and that error would be swallowed by process_one's
    # generic exception handler too, masking this as a false pass for the wrong reason).
    respx.post(_CLAIM_URL).mock(return_value=Response(204))
    respx.post(_MAP_CLAIM_URL).mock(return_value=Response(204))

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()) as sync_down_mock,
        patch("mixlab.worker.subprocess.run") as run_mock,
    ):
        result = process_one(remote, config)

    assert result is False
    sync_down_mock.assert_not_called()
    run_mock.assert_not_called()


@respx.mock
def test_process_one_claim_transport_error_returns_false_without_fail(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(500, text="down"))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    with patch("mixlab.worker.sync_down", return_value=SyncReport()) as sync_down_mock:
        result = process_one(remote, config)

    assert result is False
    sync_down_mock.assert_not_called()
    assert fail_route.call_count == 0


@respx.mock
def test_process_one_download_transport_error_returns_false_without_fail(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(return_value=Response(503, text="unavailable"))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch("mixlab.worker.subprocess.run") as run_mock,
    ):
        result = process_one(remote, config)

    assert result is False
    assert fail_route.call_count == 0
    run_mock.assert_not_called()


@respx.mock
def test_process_one_download_oserror_fails_and_returns_true(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch.object(remote, "download_collection", side_effect=OSError("disk full")),
        patch("mixlab.worker.subprocess.run") as run_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    assert fail_route.called
    body = json.loads(fail_route.calls.last.request.content)
    assert "download_collection failed" in body["error"]
    run_mock.assert_not_called()


@respx.mock
def test_process_one_missing_summary_line_fails_descriptively(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    # HTML report line present, but no "Run summary:" line
    stdout = "HTML report: output/reports/run.html\nDone\n"

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch.object(remote, "download_collection", return_value=config.xml_path),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
    ):
        result = process_one(remote, config)

    assert result is True
    body = json.loads(fail_route.calls.last.request.content)
    assert "Run summary" in body["error"]


@respx.mock
def test_process_one_sync_up_failure_after_complete_still_true_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    complete_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/complete").mock(return_value=Response(200))
    stdout = _write_artifacts(tmp_path)

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch("mixlab.worker.sync_up", side_effect=RuntimeError("history push boom")),
        patch.object(remote, "download_collection", return_value=config.xml_path),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
    ):
        result = process_one(remote, config)

    assert result is True
    assert complete_route.called
    assert "sync_up after complete failed" in capsys.readouterr().err


@respx.mock
def test_process_one_sync_down_runtime_error_best_effort_fails_and_returns_true(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))

    with (
        patch("mixlab.worker.sync_down", side_effect=RuntimeError("unexpected boom")),
        patch("mixlab.worker.subprocess.run") as run_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    assert fail_route.called
    body = json.loads(fail_route.calls.last.request.content)
    assert "sync_down failed" in body["error"]
    run_mock.assert_not_called()


@respx.mock
def test_process_one_unexpected_error_in_body_best_effort_fails_loop_safe(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    fail_route = respx.post(f"{_BASE}/api/mixlab/runs/r1/fail").mock(return_value=Response(200))
    respx.post(f"{_BASE}/api/mixlab/runs/r1/complete").mock(return_value=Response(200))
    stdout = _write_artifacts(tmp_path)

    # complete() succeeds at the network layer, but sync_up isn't reached because we make
    # the post-parse complete call blow up unexpectedly via a patched remote.complete.
    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch.object(remote, "download_collection", return_value=config.xml_path),
        patch.object(remote, "complete", side_effect=RuntimeError("kaboom")),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
    ):
        result = process_one(remote, config)

    assert result is True  # a job was claimed, so the run is consumed
    assert fail_route.called  # generic handler attempted a best-effort fail


# ===========================================================================
# process_one / _run_claimed_map — map-job cycle
# ===========================================================================


@respx.mock
def test_process_one_prefers_run_queue_over_maps(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(200, json=_manifest_dict(flags={"genre": "house"})))
    respx.get(f"{_BASE}/api/mixlab/uploads/u1").mock(
        return_value=Response(200, content=gzip.compress(b"<COLLECTION/>"))
    )
    respx.post(f"{_BASE}/api/mixlab/runs/r1/complete").mock(return_value=Response(200))
    stdout = _write_artifacts(tmp_path)

    with (
        patch("mixlab.worker.sync_down", return_value=SyncReport()),
        patch("mixlab.worker.sync_up"),
        patch("mixlab.worker.subprocess.run", return_value=_completed(stdout=stdout)),
        patch.object(remote, "claim_map") as claim_map_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    claim_map_mock.assert_not_called()


@respx.mock
def test_process_one_claims_map_when_run_queue_empty(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(204))
    respx.post(_MAP_CLAIM_URL).mock(return_value=Response(200, json={"uploadId": "up-1"}))

    map_xml_path = config.repo_root / _MAP_XML_PATH
    stdout_bytes = b'{"cells": []}'

    with (
        patch.object(remote, "download_collection", return_value=map_xml_path) as download_mock,
        patch("mixlab.worker.subprocess.run", return_value=_completed_bytes(stdout=stdout_bytes)) as run_mock,
        patch.object(remote, "complete_map") as complete_map_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    download_mock.assert_called_once_with("up-1", map_xml_path)
    run_argv = run_mock.call_args.args[0]
    assert run_argv[:3] == [run_argv[0], "-m", "mixlab"]
    assert run_argv[3:] == ["--map", "--mode", "unplayed"]
    assert run_mock.call_args.kwargs["env"]["MIXLAB_COLLECTION_PATH"] == str(map_xml_path)
    assert run_mock.call_args.kwargs["cwd"] == config.repo_root
    assert run_mock.call_args.kwargs["timeout"] == config.run_timeout
    # stdout must stay raw bytes for complete_map's payload contract — text=True (or an
    # encoding kwarg) would decode it to str and silently break that contract.
    assert "text" not in run_mock.call_args.kwargs
    assert "encoding" not in run_mock.call_args.kwargs
    complete_map_mock.assert_called_once_with("up-1", stdout_bytes)


def test_run_claimed_map_nonzero_exit_fails_job_with_bounded_tail(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    map_xml_path = config.repo_root / _MAP_XML_PATH

    oversized = _completed_bytes(returncode=1, stdout=b"", stderr=b"e" * 5000)

    with (
        patch.object(remote, "download_collection", return_value=map_xml_path),
        patch("mixlab.worker.subprocess.run", return_value=oversized),
        patch.object(remote, "fail_map") as fail_map_mock,
        patch.object(remote, "complete_map") as complete_map_mock,
    ):
        result = _run_claimed_map(remote, config, "up-1")

    assert result is True
    complete_map_mock.assert_not_called()
    fail_map_mock.assert_called_once()
    call_kwargs = fail_map_mock.call_args.kwargs
    assert fail_map_mock.call_args.args[0] == "up-1"
    assert "mixlab --map exited 1" in call_kwargs["error"]
    assert "e" * 4000 in call_kwargs["error"]
    # the whole error string carries only a bounded stderr tail, not the full 5000 chars
    assert "e" * 5000 not in call_kwargs["error"]


def test_run_claimed_map_timeout_fails_job_mentioning_timeout_and_tail(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    map_xml_path = config.repo_root / _MAP_XML_PATH

    timeout_exc = subprocess.TimeoutExpired(
        cmd="mixlab", timeout=config.run_timeout, output=b"partial-out", stderr=b"partial-err"
    )

    with (
        patch.object(remote, "download_collection", return_value=map_xml_path),
        patch("mixlab.worker.subprocess.run", side_effect=timeout_exc),
        patch.object(remote, "fail_map") as fail_map_mock,
        patch.object(remote, "complete_map") as complete_map_mock,
    ):
        result = _run_claimed_map(remote, config, "up-1")

    assert result is True
    complete_map_mock.assert_not_called()
    fail_map_mock.assert_called_once()
    call_kwargs = fail_map_mock.call_args.kwargs
    assert fail_map_mock.call_args.args[0] == "up-1"
    assert "timed out" in call_kwargs["error"]
    assert f"{config.run_timeout:g}" in call_kwargs["error"]
    assert "partial-out" in call_kwargs["error"]
    assert "partial-err" in call_kwargs["error"]


@respx.mock
def test_process_one_map_claim_unexpected_error_best_effort_fails_map(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(return_value=Response(204))
    respx.post(_MAP_CLAIM_URL).mock(return_value=Response(200, json={"uploadId": "up-1"}))

    with (
        patch.object(remote, "download_collection", side_effect=RuntimeError("disk full")),
        patch("mixlab.worker.subprocess.run") as run_mock,
        patch.object(remote, "fail_map") as fail_map_mock,
    ):
        result = process_one(remote, config)

    assert result is True
    run_mock.assert_not_called()
    fail_map_mock.assert_called_once()
    assert fail_map_mock.call_args.args[0] == "up-1"


# ===========================================================================
# run_worker — once + loop
# ===========================================================================


def test_run_worker_once_runs_single_cycle_and_returns_zero(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    with patch("mixlab.worker.process_one", return_value=False) as process_mock:
        rc = run_worker(remote, config, poll_seconds=30.0, once=True)
    assert rc == 0
    process_mock.assert_called_once_with(remote, config)


def test_run_worker_loop_sleeps_on_empty_then_exits_cleanly(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    # Empty poll -> the loop reaches sleep; the fake sleep raises KeyboardInterrupt to
    # exit deterministically without any real waiting.
    with (
        patch("mixlab.worker.process_one", return_value=False) as process_mock,
        patch("mixlab.worker.time.sleep", side_effect=KeyboardInterrupt) as sleep_mock,
    ):
        rc = run_worker(remote, config, poll_seconds=5.0, once=False)
    assert rc == 0
    process_mock.assert_called_once_with(remote, config)
    sleep_mock.assert_called_once_with(5.0)


def test_run_worker_loop_busy_cycles_do_not_sleep(tmp_path: Path) -> None:
    remote = _remote()
    config = _config(tmp_path)
    # Two claimed cycles (True) then a KeyboardInterrupt to stop; a claimed cycle never sleeps.
    with (
        patch("mixlab.worker.process_one", side_effect=[True, True, KeyboardInterrupt]) as process_mock,
        patch("mixlab.worker.time.sleep") as sleep_mock,
    ):
        rc = run_worker(remote, config, poll_seconds=5.0, once=False)
    assert rc == 0
    assert process_mock.call_count == 3
    sleep_mock.assert_not_called()


# ===========================================================================
# CLI wiring — mixlab --worker / --worker-once
# ===========================================================================


def test_main_worker_once_without_config_exits_one_with_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MIXLAB_API_URL", raising=False)
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    monkeypatch.setattr("sys.argv", ["mixlab", "--worker-once"])

    with patch("mixlab.__main__.load_dotenv"), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "MIXLAB_API_URL" in capsys.readouterr().err


def test_main_worker_once_with_config_invokes_run_worker_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_URL", "https://example.com")
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    monkeypatch.setattr("sys.argv", ["mixlab", "--worker-once"])

    run_worker_mock = MagicMock(return_value=0)
    with (
        patch("mixlab.__main__.load_dotenv"),
        patch("mixlab.__main__.run_worker", run_worker_mock),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 0
    assert run_worker_mock.call_args.kwargs["once"] is True


def test_main_worker_loop_invokes_run_worker_with_once_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_URL", "https://example.com")
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    monkeypatch.setattr("sys.argv", ["mixlab", "--worker"])

    run_worker_mock = MagicMock(return_value=0)
    with (
        patch("mixlab.__main__.load_dotenv"),
        patch("mixlab.__main__.run_worker", run_worker_mock),
        pytest.raises(SystemExit),
    ):
        main()

    assert run_worker_mock.call_args.kwargs["once"] is False


def test_main_worker_once_wins_when_both_flags_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_URL", "https://example.com")
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    monkeypatch.setattr("sys.argv", ["mixlab", "--worker", "--worker-once"])

    run_worker_mock = MagicMock(return_value=0)
    with (
        patch("mixlab.__main__.load_dotenv"),
        patch("mixlab.__main__.run_worker", run_worker_mock),
        pytest.raises(SystemExit),
    ):
        main()

    assert run_worker_mock.call_args.kwargs["once"] is True


def test_main_worker_poll_seconds_env_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_URL", "https://example.com")
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    monkeypatch.setenv("MIXLAB_WORKER_POLL_SECONDS", "12")
    monkeypatch.setattr("sys.argv", ["mixlab", "--worker"])

    run_worker_mock = MagicMock(return_value=0)
    with (
        patch("mixlab.__main__.load_dotenv"),
        patch("mixlab.__main__.run_worker", run_worker_mock),
        pytest.raises(SystemExit),
    ):
        main()

    assert run_worker_mock.call_args.kwargs["poll_seconds"] == 12.0


# ===========================================================================
# _log — timestamped worker log lines
# ===========================================================================

_TS_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


def test_log_prefixes_stdout_lines_with_timestamp(capsys: pytest.CaptureFixture[str]) -> None:
    worker_module._log("hello from the worker")
    out = capsys.readouterr().out
    assert "hello from the worker" in out
    assert _TS_PREFIX_RE.match(out), f"no timestamp prefix on stdout line: {out!r}"


def test_log_err_prefixes_stderr_lines_with_timestamp(capsys: pytest.CaptureFixture[str]) -> None:
    worker_module._log("something broke", err=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _TS_PREFIX_RE.match(captured.err), f"no timestamp prefix on stderr line: {captured.err!r}"


@respx.mock
def test_process_one_error_line_carries_timestamp(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    remote = _remote()
    config = _config(tmp_path)
    respx.post(_CLAIM_URL).mock(side_effect=RuntimeError("api exploded"))

    assert process_one(remote, config) is False
    err = capsys.readouterr().err
    assert "unexpected error in process_one" in err
    assert _TS_PREFIX_RE.match(err), f"no timestamp prefix on error line: {err!r}"
