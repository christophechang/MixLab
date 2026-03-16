You have access to MixLab, an AI-powered DJ crate assistant that analyses a Rekordbox collection and generates mix concepts. It is installed on this machine at `/Users/christophechang/OpenClaw/Automations/MixLab/`.

## How to run MixLab

When asked to run MixLab (e.g. "run mixlab for house", "mixlab drum and bass", "do a mixlab for techno"):

1. Map the requested genre to one of the valid genre keys below
2. Run the command:

```
cd /Users/christophechang/OpenClaw/Automations/MixLab && venv/bin/python -m mixlab --genre <genre>
```

MixLab posts the results directly to Discord — no further action needed after the command completes.

## Valid genres

| What the user might say | Genre key to use |
|---|---|
| drum and bass, dnb, d&b | `drum_and_bass` |
| house, deep house | `house` |
| techno | `techno` |
| uk garage, ukg, garage | `uk_garage` |
| uk bass | `uk_bass` |
| jungle | `jungle` |
| breakbeat, breaks | `breakbeat` |
| electronica, downtempo | `electronica` |
| hip hop, hip-hop | `hip_hop` |
| progressive | `progressive` |
| disco | `disco` |

## Show available genres (no LLM cost)

If asked what genres are available or how many tracks are in the collection:

```
cd /Users/christophechang/OpenClaw/Automations/MixLab && venv/bin/python -m mixlab
```

## Notes

- The Rekordbox XML at `import/rekordbox.xml` is updated manually by Christophe — if it's missing or stale, let him know
- If the command fails with `CATALOG_API_URL is not set`, the `.env` file needs updating
- Results go to Discord automatically — do not try to relay the output yourself
