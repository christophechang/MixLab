# MixLab — Claude Code Prompt

You are running MixLab, an AI-powered DJ crate assistant, on the Mac Mini.

## How to run

Working directory: `/Users/christophechang/OpenClaw/Automations/MixLab`
Python binary: `venv/bin/python`

### Show available genres (no LLM cost)
```
cd /Users/christophechang/OpenClaw/Automations/MixLab && venv/bin/python -m mixlab
```

### Generate mix concepts for a genre
```
cd /Users/christophechang/OpenClaw/Automations/MixLab && venv/bin/python -m mixlab --genre <genre>
```

### Optional: target set duration in minutes
```
cd /Users/christophechang/OpenClaw/Automations/MixLab && venv/bin/python -m mixlab --genre <genre> --duration <minutes>
```

## Valid genres
`drum_and_bass`, `house`, `techno`, `uk_garage`, `uk_bass`, `jungle`, `breakbeat`, `electronica`, `hip_hop`, `progressive`, `disco`

## Notes
- The pipeline posts results directly to Discord — no further action needed after the command completes.
- Rekordbox XML must be present at `import/rekordbox.xml` inside the project directory (updated manually by Christophe before running).
- All API keys are in `.env` — do not print or log them.
- If the command fails with a missing XML error, tell the user to update `import/rekordbox.xml` first.
