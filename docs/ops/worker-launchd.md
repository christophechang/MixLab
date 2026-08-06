# MixLab Anywhere worker runbook (macOS launchd)

This document covers running MixLab as a persistent remote worker on a Mac mini using launchd to keep it alive across reboots and crashes.

## Prerequisites

- Repository cloned to a local path (e.g. `/Users/christophechang/OpenClaw/Automations/MixLab`)
- Python virtual environment built via `./setup.sh`
- `.env` file populated with `MIXLAB_API_URL` and `MIXLAB_API_SECRET`
- `.env` also populated with `CATALOG_API_URL` (and `CHANGSTA_API_KEY` if the catalog
  requires it) — a run degrades gracefully without it, but a claimed map job
  (`mixlab --map --mode unplayed`) hard-fails without it (see Worker behavior notes below)

## Create the launchd agent

Create a file at `~/Library/LaunchAgents/com.changsta.mixlab-worker.plist` with the following contents. Replace `/Users/christophechang/OpenClaw/Automations/MixLab` with your actual repo path and fill in placeholder values for the API URL and secret:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>com.changsta.mixlab-worker</string>
	<key>ProgramArguments</key>
	<array>
		<string>/Users/christophechang/OpenClaw/Automations/MixLab/.venv/bin/python</string>
		<string>-m</string>
		<string>mixlab</string>
		<string>--worker</string>
	</array>
	<key>WorkingDirectory</key>
	<string>/Users/christophechang/OpenClaw/Automations/MixLab</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>MIXLAB_API_URL</key>
		<string>YOUR-API-URL-HERE</string>
		<key>MIXLAB_API_SECRET</key>
		<string>YOUR-SECRET-HERE</string>
		<key>CATALOG_API_URL</key>
		<string>YOUR-CATALOG-API-URL-HERE</string>
		<key>PYTHONPATH</key>
		<string>/Users/christophechang/OpenClaw/Automations/MixLab/src</string>
	</dict>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>StandardOutPath</key>
	<string>/Users/christophechang/Library/Logs/mixlab-worker/stdout.log</string>
	<key>StandardErrorPath</key>
	<string>/Users/christophechang/Library/Logs/mixlab-worker/stderr.log</string>
</dict>
</plist>
```

## Create the log directory

The launchd daemon does not create the log directory automatically. Create it once before installing:

```bash
mkdir -p ~/Library/Logs/mixlab-worker
```

## Install and start

Load the agent into launchd:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist
```

Launchd will start the worker immediately (because `RunAtLoad=true`) and keep it running (because `KeepAlive=true`).

## Verify the worker is running

Check the launchd status:

```bash
launchctl print gui/$(id -u)/com.changsta.mixlab-worker
```

Tail the log files to see startup and poll activity:

```bash
tail -f ~/Library/Logs/mixlab-worker/stdout.log
tail -f ~/Library/Logs/mixlab-worker/stderr.log
```

The first log line from the worker should be `Worker online — polling ...` showing the API URL and poll interval.

## Stop or uninstall

Stop the worker and unload it from launchd:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist
```

The worker finishes its current poll cycle before exiting (clean shutdown via SIGTERM).

## Restart after edits

If you edit the plist or `.env` file, uninstall and reinstall:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist
```

## Worker behavior notes

- **SIGTERM graceful shutdown**: When you run `launchctl bootout`, launchd sends SIGTERM to the worker. It finishes the current claim cycle before exiting, so mid-run claims are never abandoned.
- **API unreachable**: If the API is down, the worker's claim call fails with a transport error. It does **not** attempt to fail the run (that would fail too); it leaves the claim to lease expiry. The run is requeued when the API comes back and the claim expires.
- **Pipeline crashes**: The worker runs `./mixlab ...` as a subprocess, so a crash in the pipeline is isolated — the loop survives and reports the failure via the fail endpoint with a log tail.
- **Map jobs need `CATALOG_API_URL`**: A claimed map job always runs `mixlab --map --mode unplayed`, which hard-fails (exit 1, reported via `fail_map`) if `CATALOG_API_URL` is unset — unlike a run, which merely degrades to the full collection without a played-track filter.
- **Empty polls are silent**: When the queue is empty, the worker sleeps and prints nothing (designed to avoid flooding launchd logs with heartbeat lines).

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| Worker exits immediately | Missing or invalid `MIXLAB_API_URL` or `MIXLAB_API_SECRET` in plist or `.env` | Run the worker once manually to see the error on stderr; `./mixlab --worker-once` will print env var issues to stderr |
| Runs fail with "invalid run flags" | API sent a flag that's outside the worker's allow-list | Update the API's flag validation; the worker defensively rejects unknown flags before they reach the pipeline |
| Pipeline times out frequently | `MIXLAB_WORKER_RUN_TIMEOUT` is too low for your pipeline | Increase the timeout value (default 1200 s = 20 min); adjust both the plist EnvironmentVariables and `.env` |
| Log files grow without bound | stdout/stderr are not being rotated | Use a log rotation tool like `newsyslog` or `logrotate`; alternatively, periodically truncate: `> ~/Library/Logs/mixlab-worker/stdout.log` |
