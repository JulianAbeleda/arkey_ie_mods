# Details

Everything beyond applying the mods and reading the benchmarks: per-mod limits, check modes, the repo's rules,
its layout and the check. The [README](README.md) has how to apply and the numbers.

## llama.cpp v1.0.0

### Limits that bite

- 03 aborts on context shift, mid-sequence edits, full slot save/restore and the prompt cache. One sequence only.
- 04 has no check mode. It changes what the draft proposes, never what the target computes.
- 07 covers one-query calls only (the draft passes, q8_0 KV).
- 08 covers decode shapes only (under 8 query tokens) and needs 03 on. Prefill keeps the stock copy.
- 09 is bit-identical only for an identity kernel. Each installed kernel carries its own evidence.
- 06 missed its own 0.22 ms line (0.19 measured). Kept: positive in 9 of 9 pair-buckets past 32k, 82 lines
  (the nine values are in its row's limits).
- 09 loads the cubin/PTX named by an opt-in manifest and runs it on the GPU: point it only at kernels you trust.
  The manifest schema `boltbeam.kernel_manifest.v1` is the format BoltBeam writes. A manifest loads only on a build
  whose ggml commit matches its `engine.builds` (build with the commit flags under Apply and build).
- 10 covers prefill batches over 32 tokens through MMQ mul_mat_id with the expert cache on (`--moe-cache-mib`). Cache misses
  are still uploaded one by one, as stock.
- 11 pins 3.7 GB of host RAM (`--ctx-checkpoints` + 1 buffers of the state size) at the first checkpoint and holds it until
  exit. That first checkpoint takes about 2.3 s once. 8 MiB of VRAM (measured with 10 at -ub 512).
- 13 covers MTP draft contexts only and is measured at N=1 (about 4 MiB). N>1 costs about 60 MiB more and ran out of
  memory at 128k. Slot buffers are not in memory_breakdown. No check mode. Off when `GGML_CUDA_GRAPH_OPT` is set.

### Check modes

Set the check switch beside the mod's switch. Rows marked G also need `GGML_CUDA_DISABLE_GRAPHS=1`.

| mod | check switch | |
|---|---|---|
| 01 | `LLAMA_KPOOL_CHECK=1` | |
| 02 | `LLAMA_SPARSE_DIRECT_CHECK=N` (every Nth call) | G |
| 03 | `LLAMA_IDX_POOLED_CHECK=1` | |
| 04 | none | |
| 05 | `LLAMA_KPOOL_INPUT_CHECK=1` | |
| 06 | `LLAMA_SEQ_RM_CHECK=1` | |
| 07 | `LLAMA_FATTN_VEC_SKIP_CHECK=N` | G |
| 08 | `LLAMA_IDX_INPLACE_CHECK=1` | G |
| 09 | `GGML_CUDA_EXT_KERNELS_CHECK=1` | G |
| 10 | `LLAMA_MOE_INPLACE_CHECK=1` | G |
| 11 | `LLAMA_CKPT_PINNED_CHECK=1` | |
| 12 | `LLAMA_TOPK_FUSED_CHECK=1` | G |
| 13 | none | |

A check run is slow. It is for correctness, not timing.

### Engine flags in the measured setup (stock flags, not patches)

- `-ncmoe 16 --moe-cache-mib N`: the stock expert cache. Changes reply text. Depends on request history.
  Distribution gate inside the noise floor.
- `-ub 1024`: prefill chunk. Changes rounding. Passed the distribution gate and MMLU-Pro and LiveCodeBench
  non-inferiority against stock, with the cache and 01-13 on.
- `-lm none`: host experts read into RAM at load. Next-token distributions identical, 4,608 of 4,608 positions.
- `-ctkd q8_0 -ctvd q8_0`: draft KV at q8_0. Changes the draft only. A net VRAM loss on stock (+110 MiB);
  a gain only with 02 on.

### Parked

Parallel mask compaction (`LLAMA_FATTN_COMPACT_PAR`). Identical output. 0.10 ms/step for 119 lines,
a third of the derived gain. `parked/fattn-compact-par.patch` applies after the series.
Rejected patches (KV host old, staged gather a2, sparse-direct half 2, graph recapture) are in `archive/`.

## Rules (the same for every engine)

1. Lean. Each patch has a line budget from its folder's `budget_rule` (measured size, or measured rounded up).
   Over budget fails the check; a changed budget or line count needs a `budget_raised` entry.
2. One switch per mod. Unset = stock.
3. A check mode: runs the stock path beside the mod, aborts on a difference.
4. Output bit-identical to stock, or labelled (`quality-gated`, `untested`) with the evidence.
5. Gains are stated against the same engine's stock path. An engine-vs-engine comparison lives only in a
   benchmark folder's `vs-<engine>` file, which alone may name that engine.

## Evidence references

Evidence references in the rows (`boltbeam-runs/...`, BoltBeam git shas, BoltBeam `docs/evidence/...`, BoltBeam
`tests/...`) are private measurement records, not included in this repo; the numbers are summarized in each row.

## Layout

```
engines.json                      one row per engine: base identity, applier, switch kinds, off = stock, tested tree
engines/README.md                 what adding an engine or a version requires
engines/<engine>/<version>/
    mods.json                     one row per mod; parked, archived, tools, engine flags, tested trees
    patches/NN-name.patch         the series, in order
    parked/                       correct but weak, or a record kept beside the series; may come back
    archive/                      decommissioned or rejected patches; never applied (archive/README.md is generated)
    tools/                        helper scripts a mod needs (not patches)
index.json                        generated: every mod, archived patch and tool, for consumers
archive-ledger.json               append-only: every patch ever archived (id, sha256, switch); archived never comes back
benchmarks/<engine>-<version>-<run>/  a benchmark record: README.md (method, results, reproduce) + .json summaries
local.example.json                copy to local.json (gitignored): this machine's base checkouts and run roots
tools/check.py                    the one check; standard library only; names no engine; --write regenerates the exports
.githooks/pre-commit              runs it on the staged content
```

Setup after a clone:

```
git config core.hooksPath .githooks          # the hook checks what you stage
cp local.example.json local.json               # then set this machine's paths (or ARKEY_IE_MODS_LOCAL=/path/to/file)
```

## Archive

`engines/<engine>/<version>/archive/` keeps every decommissioned or rejected patch, so nothing that was patched is lost.
Each has a row in that version's mods.json `archived` (why, measured reason, evidence path, date, lines, sha256)
and an entry in `archive-ledger.json`, which only grows. `archive/README.md` is generated from those rows. The check
verifies each sha256; a series or parked patch with an archived sha256 or switch fails; archived patches are never in
the series, parked or index.json's active list (index.json lists them under `archived`).
Parked (may come back) stays in `parked/`.

## Kernels

An engine kernel is an artifact here too, never in a tool that produced it: `engines/<engine>/<version>/kernels/<name>/`
holds its source, its manifest entry for the install hook mod (llama.cpp 1.0.0: 09 ext-kernels), built binaries if
kept, and a `kernels` row in mods.json: hook (the mod whose switch turns it on and whose check mode checks it), files
with sha256, source lines and budget (`source`: one file, or a list when one manifest installs several kernels; lines
are summed), an output label with evidence (identical: a counted match), gain vs stock. The check verifies each file,
the budget and the label; index.json lists them under `kernels`. A kernel's manifest loads only on a build whose ggml
commit (`GGML_COMMIT` in build/ggml/src/ggml-version.h) matches its `engine.builds`; on any other build, a tree
without .git configured without the commit flags included, the hook runs the engine's own kernels.

## index.json

Generated by `python3 tools/check.py --write`; the hook fails when it is stale. `versions`: per engine version, the
base, applier and `verified` (tree: byte-equal to a measured tree, or none, with why). `mods`: one entry per mod with
a stable `id` (engine/version/name), state (series / parked), order, applier, switch, check, `marker` (a literal the
patch adds: proves it is in a build), `knobs`, output label, patch path, sha256, lines, budget, files, depends_on,
rule exceptions. Then `archived` and `tools`. Gains, limits and evidence live in mods.json at the same commit.
Consumers read it at a pinned commit of this repo.

## The check

`python3 tools/check.py` (one folder: `python3 tools/check.py engines/llama.cpp/1.0.0`; `--strict`: any SKIP fails)

It verifies, per engine folder:
- the base is the recorded one (git commit, or pip version + RECORD hash) and the touched files are clean;
- every patch applies in order; the series reverts to the base bytes;
- every patch applies on the base plus only the mods it says it depends on;
- added and removed lines equal the row and fit the budget;
- the switch, check and marker are in the patch; no undeclared switch-shaped or env-read literal; no archived switch;
- rows: known keys, an output label with its evidence, a rule exception for each rule a mod breaks, no model name,
  no other engine, no private detail (rows, patch headers and added lines);
- base + patches is byte-equal to the trees that were built and measured (what each holds is below);
- archived patches match their sha256 and the ledger; every file has a row; index.json and archive/README.md are fresh.

A missing local checkout, tree or history is a named SKIP. The hook runs on the staged content. On the maintainer's
machine three SKIPs remain, each a fact `--strict` refuses to hide:
- llama.cpp 7592b61d3: no tested tree (the measured trees are gone);
- vLLM 0.31.0: the wheel sha256 was never recorded (the base is identified by version + RECORD hash);
- vLLM 0.31.0: no tested tree (the measured build, 0.31.0+rocm723, was on another machine).

Tested trees (llama.cpp 1.0.0), byte-equal on the touched files, no other file differs, no extra file:
- `mask-indexer`: 01-08 + parked fattn-compact-par + the archived kv-host-old code (its switch unset in every kept
  measurement). It is not a series-only tree: 09 is missing and two extra patches are in it.
- `ext-kernel-hook`: 01-06 + 09 + the archived kv-host-old code (cut before 07 and 08).
Series-only builds (01-09 and 01-11 on the base, nothing else) are recorded in mods.json `verified`.

It does not prove that a hunk which rewrites stock lines keeps stock behaviour with the switch off.
That is a run-time fact: the evidence is in each row, and the byte-equality check ties the patches to the measured trees.
It compiles nothing and runs nothing on a GPU.
## License

- This repository's own material (tools/, the json rows, the docs, the added lines of each patch, the tools/ scripts
  beside a version) is MIT: `LICENSE`, Copyright (c) 2026 Arkey.
- The context and removed lines of the llama.cpp patches come from llama.cpp: MIT, `LICENSES/MIT-llama.cpp.txt`.
- The vLLM patches modify vLLM files: Apache License 2.0, `LICENSES/Apache-2.0.txt`. Each vLLM patch says that it
  was modified and what changed (section 4(b)). `quark_lm_head_int4.py` is original work except one constant (`REV`,
  vLLM's AWQ reverse pack order), which carries vLLM's license.
