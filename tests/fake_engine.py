"""Stands in for ``python -m mixlab`` in the tenant worker's tests (M2b spec §4.9).

The worker runs ``[sys.executable, "-m", "mixlab", *argv]``; the tests point ``sys.executable``
at a wrapper that runs this file instead. It writes ``summary.json`` and ``report.html`` under
``MIXLAB_REPORT_DIR`` and, when ``--export`` is passed, the export under that directory, printing
the engine's artifact lines, then exits with ``FAKE_ENGINE_EXIT`` (default 0). Its other knobs:

- ``FAKE_ENGINE_OMIT``: comma-separated ``summary``, ``report``, ``export``: neither file nor line;
- ``FAKE_ENGINE_RECORD``: a path to write ``{"argv", "cwd", "env"}`` to;
- ``FAKE_ENGINE_STDOUT`` / ``FAKE_ENGINE_STDERR``: text to print first;
- ``FAKE_ENGINE_DUMP``: text for a ``truncated-1.txt`` under ``MIXLAB_DIAGNOSTICS_DIR``;
- ``FAKE_ENGINE_REPORT_BYTES``: the report's size;
- ``FAKE_ENGINE_SLEEP``: seconds to sleep before writing anything.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    args = argv[2:] if argv[:2] == ["-m", "mixlab"] else argv
    record = os.environ.get("FAKE_ENGINE_RECORD")
    if record:
        Path(record).write_text(json.dumps({"argv": args, "cwd": os.getcwd(), "env": dict(os.environ)}))
    sleep = os.environ.get("FAKE_ENGINE_SLEEP")
    if sleep:
        time.sleep(float(sleep))
    sys.stdout.write(os.environ.get("FAKE_ENGINE_STDOUT", ""))
    sys.stderr.write(os.environ.get("FAKE_ENGINE_STDERR", ""))
    dump = os.environ.get("FAKE_ENGINE_DUMP")
    if dump:
        diagnostics = Path(os.environ["MIXLAB_DIAGNOSTICS_DIR"])
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / "truncated-1.txt").write_text(dump, encoding="utf-8")

    omit = set(os.environ.get("FAKE_ENGINE_OMIT", "").split(","))
    reports = Path(os.environ["MIXLAB_REPORT_DIR"])
    reports.mkdir(parents=True, exist_ok=True)
    if "report" not in omit:
        report = reports / "techno-2026-10-17.html"
        size = int(os.environ.get("FAKE_ENGINE_REPORT_BYTES", "0"))
        report.write_text("<html>report</html>" + "x" * max(0, size - 19), encoding="utf-8")
        print(f"HTML report: {report}")
    if "summary" not in omit:
        summary = reports / "techno-2026-10-17.json"
        summary.write_text(json.dumps({"schemaVersion": 2, "concepts": []}), encoding="utf-8")
        print(f"Run summary: {summary}")
    if "--export" in args and "export" not in omit:
        export_dir = Path(args[args.index("--export") + 1])
        export = export_dir / "rekordbox_export.xml"
        export.write_text('<?xml version="1.0"?><DJ_PLAYLISTS/>', encoding="utf-8")
        print(f"Exported: {export}")
    return int(os.environ.get("FAKE_ENGINE_EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
