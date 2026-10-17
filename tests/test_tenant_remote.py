from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import parse_qs

import pytest
from tenant_api_stub import (
    CLAIM_ID,
    CLIENT_SECRET,
    CREDENTIAL_ENV,
    RUN_ID,
    RUN_TOKEN,
    WORKER_TOKEN,
    Answer,
    FakeClock,
    Harness,
    build_harness,
    busy,
    prepare_env,
    scenario_with_feedback,
)

from mixlab import tenant_remote
from mixlab.tenant_remote import (
    Deadline,
    TenantConfig,
    TenantConfigError,
    TransportError,
    diagnostics_ref,
)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    prepare_env(monkeypatch, fake)
    return fake


@pytest.fixture
def make_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> Iterator[Callable[..., Harness]]:
    with ExitStack() as stack:

        def make(**kwargs: object) -> Harness:
            return build_harness(tmp_path, monkeypatch, stack, clock, **kwargs)  # type: ignore[arg-type]

        yield make


def test_claim_sends_protocol_version_and_worker_bearer(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    assert h.run() == 0

    (token,) = h.api.calls("POST token")
    assert token.headers["content-type"] == "application/x-www-form-urlencoded"
    assert parse_qs(token.body.decode()) == {
        "client_id": [CREDENTIAL_ENV["MIXLAB_WORKER_CLIENT_ID"]],
        "client_secret": [CLIENT_SECRET],
        "scope": [CREDENTIAL_ENV["MIXLAB_WORKER_SCOPE"]],
        "grant_type": ["client_credentials"],
    }
    (claim,) = h.api.calls("POST claim")
    assert claim.json() == {"protocolVersion": 2}
    assert claim.headers["authorization"] == f"Bearer {WORKER_TOKEN}"
    assert claim.headers["content-type"] == "application/json"
    assert "x-mixlab-claim" not in claim.headers


def test_busy_repeated_once_then_transport_error(make_harness: Callable[..., Harness], clock: FakeClock) -> None:
    h = make_harness()
    claim = h.remote.claim(RUN_ID, WORKER_TOKEN, Deadline(60))

    h.scenario.queue("GET history", busy(), Answer(204))
    assert h.remote.get_history(claim, Deadline(180)) == (None, None)
    assert clock.sleeps == [1.0]

    h.scenario.queue("GET history", busy(), busy())
    with pytest.raises(TransportError, match="busy twice"):
        h.remote.get_history(claim, Deadline(180))
    assert len(h.api.calls("GET history")) == 4
    assert clock.sleeps == [1.0, 1.0]

    # On the claim too: a busy claim is repeated once, then a 200 holds.
    h.scenario.queue("POST claim", busy())
    assert h.remote.claim(RUN_ID, WORKER_TOKEN, Deadline(60)).claim_id == CLAIM_ID


def test_every_call_after_claim_carries_run_token_and_claim_header(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(scenario=scenario_with_feedback())

    assert h.run() == 0

    routes = [r.route for r in h.api.requests]
    assert "PUT history" in routes
    assert "POST feedback/ack" in routes
    after_claim = [r for r in h.api.requests if r.path.startswith("/api/worker/") and r.route != "POST claim"]
    assert len(after_claim) == 9
    for request in after_claim:
        assert request.headers["authorization"] == f"Bearer {RUN_TOKEN}", request.route
        assert request.headers["x-mixlab-claim"] == CLAIM_ID, request.route


def test_deadline_timeout_is_the_cap_or_the_time_left(clock: FakeClock) -> None:
    deadline = Deadline(180)
    assert deadline.timeout() == 30.0
    clock.advance(170)
    assert deadline.timeout() == 10.0
    assert deadline.shortened(60).expired()
    clock.advance(10)
    assert deadline.expired()
    assert deadline.timeout() == 0.0


def test_diagnostics_ref_read_from_the_url_path() -> None:
    url = "https://acct.blob.core.windows.net/runs/tenants/t1/runs/r1/attempts/c%2D1/diagnostics.txt?sv=x&sig=y"
    assert diagnostics_ref(url, "r1") == "attempts/c-1/diagnostics.txt"
    assert diagnostics_ref(url, "other") is None
    assert diagnostics_ref("", "r1") is None


def test_tenant_config_from_env_names_the_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "MIXLAB_API_URL": "https://api.example.test/",
        "MIXLAB_PROTOCOL_VERSION": "2",
        "MIXLAB_WORKER_CLIENT_ID": "id",
        "MIXLAB_WORKER_CLIENT_SECRET": CLIENT_SECRET,
        "MIXLAB_WORKER_TOKEN_ENDPOINT": "https://login.example.test/token",
        "MIXLAB_WORKER_SCOPE": "api://x/.default",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = TenantConfig.from_env()
    assert config.api_url == "https://api.example.test"
    assert config.protocol_version == 2
    assert config.timeout == tenant_remote.REQUEST_TIMEOUT_SECS
    assert CLIENT_SECRET not in repr(config)

    monkeypatch.setenv("MIXLAB_PROTOCOL_VERSION", "two")
    with pytest.raises(TenantConfigError, match="MIXLAB_PROTOCOL_VERSION"):
        TenantConfig.from_env()
    monkeypatch.setenv("MIXLAB_PROTOCOL_VERSION", "2")
    monkeypatch.delenv("MIXLAB_WORKER_CLIENT_SECRET")
    with pytest.raises(TenantConfigError, match="MIXLAB_WORKER_CLIENT_SECRET is not set") as excinfo:
        TenantConfig.from_env()
    assert CLIENT_SECRET not in str(excinfo.value)
