# Intent-Usage Smoke Log

Running notes file tracking real `mixlab` runs. Two purposes:

1. **#17 gate.** `mixlab --intent` must be used in ≥10 real runs before replacing Stage 1 LLM with deterministic clustering. Each entry below with `--intent` set is one data point toward that gate.
2. **Prompt-alignment evidence.** Tracks observations about Stage 2 output quality — especially whether narrative-risk-language (prose "Risk: ...") correlates with structured `Transition.is_risky` annotations. When ≥3 runs show systematic prose/JSON divergence, open prompt-alignment ticket.

Schema per entry:

- **Date** — ISO date (absolute, not relative)
- **Command** — full CLI invocation
- **Intent** — verbatim `--intent` text, or `(none)`
- **Mode / Genre** — `--mode` value and `--genre` value
- **Concepts** — titles produced
- **Validation notes** — warnings emitted, classified as legitimate vs spurious
- **Observations** — anything non-obvious worth tracking

---

## 2026-05-17 — v0.12.1 release smoke

- **Command:** `./mixlab --genre house`
- **Intent:** (none) — no `--intent` flag used; default `unplayed` mode, `house` genre
- **Mode / Genre:** unplayed / house
- **Concepts:**
  - Pneumatic Gospel (energy path: Wave; 10 tracks)
  - Nocturama (energy path: Wave; 8 tracks)
- **Validation notes:**
  - `[Pneumatic Gospel] Camelot jump 5 between 10A and 5A` — fires on track 3→4 (Dusky LF10 → Jex Opolis Pneumatic). Stage 2 prose for track 4 reads "Risk: 10A→5A is a hard key jump". The corresponding `Transition` in JSON had `is_risky=False` — so the new v0.12.1 suppression branch did not activate. Validator correctly flagged unannotated jump.
  - `[Nocturama] bridge track ID 141150488 used without a justified transition` — bridge/wildcard role check (separate code path from v0.12.1 fix).
- **Observations:**
  - **Prose / JSON divergence (1 of N).** Stage 2 narrative routinely flags transitions as risky in prose while leaving `Transition.is_risky=False` in JSON. v0.12.1 suppression depends on structured annotation, not prose. If this pattern repeats across 2-3 more runs, open prompt-alignment ticket to tighten Stage 2 instructions around the `is_risky` + `risk_type` JSON fields.
  - **No `--intent` used.** Counter for #17 gate unchanged. Still ~2-3 real `--intent` runs total.
  - Runtime ~1m 55s. No errors. No `--deep` critique. Default model (Sonnet 4.6).

---

<!-- Append new entries above this line. Keep most-recent first. -->
