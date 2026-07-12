Create a Claude Code skill named `mixlab-intent` that helps craft a strong `--intent` string for the MixLab CLI.

Context for the skill author:

`--intent` is a free-text flag on `mixlab` (genre mode only — playlist mode ignores it and uses its own Stage 0 extraction). It gets passed verbatim into the Stage 2 LLM prompt as the "primary curatorial lens" for the run — every concept must serve it, not just technically fit the pool.

Constraints the skill should help the user satisfy:
- Word count sweet spot: 5–50 words. Below 5 triggers a "too short, add purpose/mood/stylistic scope" warning; above 50 triggers a "too long, keep it focused" warning. See `_warn_intent` in src/mixlab/__main__.py (around line 824).
- A lightweight heuristic parser (`_parse_user_intent` in src/mixlab/llm.py, around line 2301) extracts structured signals from the text: a "register" (warmup / peak / closing / late-night — see `_REGISTER_MAP` around line 2170) and up to a few "mood" words (dark, euphoric, melancholic, hypnotic, playful, driving, deep, etc. — see `_MOOD_KEYWORDS` around line 2197). These parsed signals get echoed back to the LLM alongside the quoted intent. The parser doesn't handle negation well (e.g. "not dark" can still extract mood=dark), so the skill should nudge users toward affirmative phrasing rather than negations.
- The intent is a thesis, not a tag list — it should describe purpose, mood/register, and stylistic scope/occasion in a way a curator could act on (e.g. "warmup set for an outdoor afternoon — melodic, sun-drenched, gradually building energy" rather than "house, melodic, afternoon, chill, good vibes").

Build the skill so that when invoked it:
1. Asks the user 3–4 short questions covering: occasion/register (when/where this set plays), mood/emotional quality, energy arc (flat / building / peak-and-release / wind-down), and any stylistic or scope constraints (genre boundaries, eras, things to avoid — phrased affirmatively, not as negations).
2. Drafts a single `--intent` string from the answers that lands in the 5–50 word range, reads as a coherent thesis (not a keyword dump), and naturally surfaces register/mood vocabulary the user already gravitates toward (no need to force-match the exact `_REGISTER_MAP`/`_MOOD_KEYWORDS` lists — those are illustrative of the kind of words that land well, not a fixed vocabulary).
3. Shows the user the draft plus the exact `mixlab --genre <genre> --intent "..."` invocation, and offers to tighten/expand if the word count is off or the thesis feels vague.

Keep the skill itself concise — it's a conversational helper, not a rigid checklist. Follow this project's conventions for skill structure (frontmatter with name/description, etc.) per the `superpowers:writing-skills` skill if available.
