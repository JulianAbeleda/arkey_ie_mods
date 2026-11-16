# Arkey inference engine mods

Opt-in patches for inference engines. Engine-agnostic and model-agnostic.

- One home for every engine. A mod is keyed on engine + version + kernel or graph shape. Never on a model name.
  A row may say which architectures exercise it. That is evidence, not a key.
- Every mod is one switch. Switch off = the engine's stock path. Older mods that break a rule say so in
  `rule_exceptions`.
- Not submitted upstream to any engine.

| engine | version | mods | state |
|---|---|---|---|
| llama.cpp | v1.0.0 (b12359, b1b770d9) | 13 in the series, 1 parked, 4 archived | active |
| llama.cpp | master 7592b61d3 | 6 in the series, 1 parked, 1 archived | active |
| llama.cpp | master 5193ef1d | archive only: 1 | - |
| vLLM | 0.31.0 | 2 in the series, 1 tool | active |
| TensorRT-LLM | - | none yet | planned |

## Rules (the same for every engine)

1. Lean. Each patch has a line budget from its folder's `budget_rule` (measured size, or measured rounded up).
   Over budget fails the check; a changed budget or line count needs a `budget_raised` entry.
2. One switch per mod. Unset = stock.
3. A check mode: runs the stock path beside the mod, aborts on a difference.
4. Output bit-identical to stock, or labelled (`quality-gated`, `untested`) with the evidence.
5. Gains are stated against the same engine's stock path only. No engine-vs-engine numbers in this repo.

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
local.example.json                copy to local.json (gitignored): this machine's base checkouts and run roots
tools/check.py                    the one check; standard library only; names no engine; --write regenerates the exports
.githooks/pre-commit              runs it on the staged content
```

Setup after a clone:

```
git config core.hooksPath .githooks          # the hook checks what you stage
cp local.example.json local.json               # then set this machine's paths (or ARKEY_IE_MODS_LOCAL=/path/to/file)
```

## llama.cpp v1.0.0: long-context MoE decode and prefill

Base: tag b12359, commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`. CUDA backend.
Measured on one RTX 5090, one request at a time, a 59-turn session to 128k context, q8_0 KV, MTP draft head k=3.
"vs off" = the same build with the switch unset, which is the stock path.

| # | mod | switch | lines | gain vs stock | output |
|---|---|---|---|---|---|
| 01 | k-pool layout updated from the edit point | `LLAMA_KPOOL_INCREMENTAL=1` | +95 -32 | 88 -> 118.5 tok/s, 128k session (vs the stock build) | identical |
| 02 | sparse attention reads q8_0 K/V rows by index | `LLAMA_SPARSE_DIRECT=1` | +103 -4 | +4.3% [3.3, 5.2] session; +8% at 96-128k (vs off) | identical |
| 03 | indexer cache keeps pooled keys only | `LLAMA_IDX_POOLED=1` | +299 -19 | -424 MiB VRAM; speed unchanged | identical |
| 04 | draft head reads the target's output projection | `LLAMA_MTP_SHARE_HEAD=1` | +45 -4 | -340 MiB VRAM; about -1% decode speed | quality-gated: target bit-identical, draft changes |
| 05 | k-pool inputs live on the device | `LLAMA_KPOOL_INPUT_DEVICE=1` | +359 -45 | 0.585 ms/step [0.484, 0.686] at 96-128k (+3.1%) | identical |
| 06 | seq_rm walks an index, not the cache | `LLAMA_SEQ_RM_INDEXED=1` | +82 -1 | 0.19 ms/step [0.02, 0.36], flat with depth | identical |
| 07 | vector attention kernel skips masked rows | `LLAMA_FATTN_VEC_SKIP=1` | +102 -10 | 0.99 ms/step at 96-128k | identical |
| 08 | indexer scores pooled q8_0 keys in place | `LLAMA_IDX_INPLACE=1` | +59 -0 | about 0.30 ms/step at about 102k | identical |
| 09 | external kernel hook | `GGML_CUDA_EXT_KERNELS=/path/manifest.json` | +292 -0 | 0 by itself; installs kernels by op and shape | identical with an identity kernel |
| 10 | prefill reads cached experts in place | `LLAMA_MOE_INPLACE=1` | +147 -15 | prefill -342 ms/turn [-365, -321] at -ub 128 (-16%) | identical |
| 11 | context checkpoints in a pinned pool | `LLAMA_CKPT_PINNED=1` | +184 -0 | prefill -152 ms/turn [-168, -140] at 96-128k; on top of 10 at -ub 512: -153 [-157, -149] | identical |
| 12 | radix top-k of a few rows in one launch | `LLAMA_TOPK_FUSED=1` | +314 -1 | 0.30 ms/step [0.17, 0.43] session (+1.9%); 0.32 [0.27, 0.38] at 96-128k | identical |
| 13 | draft context keeps a graph per small ubatch size | `LLAMA_GRAPH_SLOTS=1` | +102 -0 | 0.19 ms/step [0.13, 0.26] session (+1.2%); every bucket clears | identical |

Series total: 13 patches, +2,183 -131 lines. Evidence, conditions and limits per mod: `engines/llama.cpp/1.0.0/mods.json`.

Limits that bite:
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

### Apply and build

```
cd llama.cpp                                  # the base b1b770d9: a clean checkout, or a tree without .git (git archive, tarball)
for p in /path/arkey_ie_mods/engines/llama.cpp/1.0.0/patches/*.patch; do git apply "$p"; done
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_COMMIT=b1b770d9 -DGGML_BUILD_COMMIT=b1b770d9 -DGIT_EXE=OFF
cmake --build build -j --target llama-server
grep GGML_COMMIT build/ggml/src/ggml-version.h   # must say "b1b770d9"
```

The three commit flags are what lets 09 load a kernel. The hook takes a manifest only on a build whose ggml commit
matches the manifest's `engine.builds`. A tree without .git builds as "unknown", the hook refuses every manifest and
the engine's own kernels run; the only sign is a WARN line. `-DGGML_BUILD_COMMIT` alone is not enough: the top-level
CMakeLists.txt sets it from `LLAMA_BUILD_COMMIT`, and ggml/CMakeLists.txt runs `git rev-parse`, whose empty output
outside a repo replaces it, unless `GIT_EXE` is off. In a git checkout at the base the flags give the same commit.

Verify a kernel loaded: with `GGML_CUDA_EXT_KERNELS` set, the log must say `ext_load: loaded <id>` once per manifest
entry. `ext_load: ... not used` or `entry ... dropped` means the engine's own kernels run. These lines are WARN level:
llama-server drops ggml INFO lines at its default verbosity, so a mod's switch and load lines are WARN.

Apply-order dependencies: 05 needs 01 and 03; 07 needs 02; 08 needs 03. The rest (10 and 11 included) apply alone on the base.
Revert: `git apply -R` in reverse order. With no switch set, the patched build takes the stock path.

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

- `-ncmoe 16 --moe-cache-mib 6400`: the stock expert cache. Changes reply text. Depends on request history.
  Distribution gate inside the noise floor. Benchmark verdict pending.
- `-ctkd q8_0 -ctvd q8_0`: draft KV at q8_0. Changes the draft only. A net VRAM loss on stock (+110 MiB);
  a gain only with 02 on.

### Parked

Parallel mask compaction (`LLAMA_FATTN_COMPACT_PAR`). Identical output. 0.10 ms/step for 119 lines,
a third of the derived gain. `parked/fattn-compact-par.patch` applies after the series.
Rejected patches (KV host old, staged gather a2, sparse-direct half 2, graph recapture) are in `archive/`.

## llama.cpp master 7592b61d3: MoE prefill

Base: master `7592b61d3b53ca5b39a25a03508fa1246b78c579` (b12274-2). Each also applies alone on v1.0.0.
Written before these rules: most have no check mode, two are on by default, one has no switch.
Each row lists its `rule_exceptions`. Measured on one RTX 5090 with a MoE GGUF, prefill, one request at a time.

| # | mod | switch | lines | gain vs stock | output |
|---|---|---|---|---|---|
| 01 | MoE routing sort, MMQ tile knob, pinned staging | none (default on) | +274 -6 | 10k 5,924 -> 6,455 tok/s | identical |
| 02 | host-RAM experts on a pinned copy stream | `LLAMA_WS` (default on, `0` = stock) | +474 -0 | 10k at -ncmoe 12: 2,757 -> 4,352 tok/s | identical |
| 03 | no checkpoint at a user start in the prompt head | `LLAMA_SERVER_CKPT_MIN_HEAD=N` | +9 -1 | 10k 4,551 -> 5,278 (vs off) | untested |
| 04 | keep only the tail checkpoint | `LLAMA_SERVER_CKPT_TAIL_ONLY=1` | +7 -1 | 10k 5,892 -> 6,016 (vs off) | identical (one request) |
| 05 | PLE row prefetch on a thread | `LLAMA_PLE_PREFETCH_ASYNC=1` | +9 -1 | 10k 5,105 -> 5,397 (vs off) | identical |
| 06 | MTP sidecar found in an `mtp/` subfolder | none (file lookup) | +32 -5 | none claimed | untested |

Parked: `mmid-helper-par` (01's routing sort alone, kept as its record; conflicts with 01).
04 drops the earlier checkpoint: a follow-up that diverges before the last few tokens reprocesses the whole prompt.
Verified: none. The measured trees are gone, so there is no byte-equality check (a named SKIP).

## vLLM 0.31.0

Apply in vLLM's `site-packages`: `patch -p1 --dry-run < X.patch`, then without `--dry-run`. Revert: `patch -R -p1`.
No switch: applied = on. Base identity: the version, the installed RECORD's sha256 and the touched files' RECORD hashes.

| # | mod | lines | gain vs stock | output |
|---|---|---|---|---|
| 01 | split-kv-verify: TRITON_ATTN keeps the split-KV kernel for a verify of up to 8 tokens | +4 -1 | MTP k=2 at ~3k context: 51 -> ~90 tok/s (self-reported) | untested (5e-4 vs the 2D kernel) |
| 02 | quark-lm-head: Quark quantizes an lm_head the config does not exclude | +7 -1 | 0 by itself; lets vLLM load an int4-head checkpoint | untested |

`tools/quark_lm_head_int4.py SRC DST` builds the int4-head checkpoint 02 loads (its numbers are in 02's row).
Verified: none. The runs used 0.31.0+rocm723 on 2 x R9700 (self-reported, no raw files kept). The check applies the
patches to a local CUDA 0.31.0 install, which is a different build and was not measured.

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
No tree holds exactly the series 01-09. A series-only build is recorded in mods.json `verified` when it exists.

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
