# Adding an engine or a version

No placeholder folders. An engine gets a folder when it has its first mod or archived patch.

An engine needs one row in `engines.json`:
- `base_identity`: how the base is named and verified. Implemented kinds: `git_commit` (HEAD of the checkout +
  clean touched files) and `pip_version_wheel_hash` (METADATA version, RECORD sha256, touched files' RECORD hashes,
  wheel sha256 when kept). A row with another kind fails the check until `tools/check.py` implements it.
- `applier`: how a patch is applied and reverted. Implemented: `git_apply`, `patch_p1`.
- `switch_kinds`, `switch_presence` (per kind, the text that proves a switch is in the added lines, e.g. `"{name}"`),
  `switch_literal_regex` and `env_read_regex` (switch-shaped and env-read literals; undeclared ones fail; null if the
  engine has none). The three keys are required.
- `aliases`: other names of the engine, for the lint that keeps other engines out of rows and patches.
- `off_is_stock`, `tested_tree`: how "switch off = stock" is verified and where measured trees live.

A version needs `engines/<engine>/<version>/mods.json` (schema `arkey_ie_mods.mods.v1`), `budget_rule` and `patches/`.
One row per mod: `name, patch, engine, base, switch {kind, name}, check, reads, marker, files, lines, budget,
depends_on, output {label, evidence {run, numbers, match | gate}}, gain {value, unit, vs, conditions, run, basis},
exercised_by, limits, rule_exceptions [{rule, why}], static`.
- `marker`: a literal the patch adds that proves it is in a build.
- `switch: null` needs `switch_null_why`. A missing check, a missing switch or a switch on by default each need a
  `rule_exceptions` entry (`check`, `switch`, `default_off`).
- `identical` needs `evidence.match {what, equal, of}` with equal == of; `quality-gated` needs `evidence.gate`.
- A changed line count or budget needs `budget_raised [{from, to, why, date}]`.
- A file rewritten to remove a model name, another engine or a private detail records `scrubbed_from {sha256, what}`.

Optional lists: `parked` (rows like mods, not in the series), `archived` (decommissioned or rejected: name, kind,
switch, written_on, why, evidence, date, files, lines, sha256; files in `archive/`; also in `archive-ledger.json`),
`tools` (path, sha256, needs, what), `tested_trees` [{id, path, what, contains, apply}] or `tested_trees_none {why}`,
`verified` (entries that record a measured build), `also_applies_to` [{version, how}] (checked by applying).

Paths: `tested_trees.path` and evidence runs are references `<root>/rest`; `local.json` maps roots and base
checkouts to this machine. Run `tools/check.py --write` after a change: it regenerates index.json and
archive/README.md and appends new archived patches to the ledger; the hook fails while they are stale.

The rules do not change per engine: one switch per mod, off = stock, a check mode, bit-identical or a labelled
output with evidence, a line budget, gains against that engine's stock build only. Keyed on engine + version +
kernel or graph shape, never on a model name.

`tools/check.py` and the hook stay engine-blind. A new engine adds rows, and at most one applier or base class.
