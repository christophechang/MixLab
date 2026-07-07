from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_report_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route HTML report output to a temp dir for every test (#45).

    run()/run_playlist_mode() write an HTML report unconditionally; without this
    guard any end-to-end test would leave artifacts in the repo's output/reports/.
    """
    monkeypatch.setenv("MIXLAB_REPORT_DIR", str(tmp_path / "reports"))
