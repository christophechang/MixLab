# Run-contract fixtures

Source repository: `soltech-consulting/mixlab-multi-tenant`, commit `c186cc8` (v0.4.0; the files are
unchanged at `3c376f7`).

The vendored `src/mixlab/run_contract/` files come from the same repository's `tools/run-contract/`:
`flags.json`, `genres.json` and `keys.json` at `c186cc8`, and `summary.schema.json` re-vendored at
`5fad4c2` (soltech-consulting/mixlab-multi-tenant#28, v0.5.0), where `llmCall.model`'s description
now says it is the model that answered.

| File | What it is | Origin |
| --- | --- | --- |
| `s6-standard.xml` | 44 rows, 40 admitted: every admission rule, the deny-list, the doubling, the exact-12 techno pool, playlists | `tests/fixtures/rekordbox/` |
| `s6-classical.xml` | 24 classical spellings, one per Camelot code, plus `8m` and `Hm` | `tests/fixtures/rekordbox/` |
| `s6-mixed-keys.xml` | three Camelot and three classical keys | `tests/fixtures/rekordbox/` |
| `s6-doctype.xml` | `s6-no-playlists.xml` with a DOCTYPE and an unused internal entity; the API refuses it | `tests/fixtures/rekordbox/` |
| `s6-bom.xml` | `s6-no-playlists.xml` (12 House rows) behind a UTF-8 byte-order mark | `tests/fixtures/rekordbox/` |
| `bpm_grammar.txt` | The `bpmGrammar` regular expression, one line | `tools/run-contract/preflight.json`, JSON-decoded |

The XML files are copied byte for byte; the descriptions are that repository's `tests/fixtures/rekordbox/README.md`. All of them are synthetic: they hold no DJ's data.

The packaged contract files `src/mixlab/run_contract/*.json` (`flags.json`, `summary.schema.json`,
`genres.json`, `keys.json`) come from `tools/run-contract/` at the same commit, byte for byte.
