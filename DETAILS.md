# Details

Everything beyond applying the mods and reading the benchmarks: per-mod limits, check modes, the repo's rules,
its layout and the check. The [README](README.md) has how to apply and the numbers.

## llama.cpp v1.0.0: each mod

"vs off" = the same build with the switch unset, which is the stock path.

| # | lines | gain vs stock | output |
|---|---|---|---|
| 01 | +95 -32 | 88 -> 118.5 tok/s, 128k session (vs the stock build) | identical |
| 02 | +103 -4 | +4.3% [3.3, 5.2] session; +8% at 96-128k (vs off) | identical |
| 03 | +299 -19 | -424 MiB VRAM; speed unchanged | identical |
| 04 | +45 -4 | -340 MiB VRAM; about -1% decode speed | quality-gated: target bit-identical, draft changes |
| 05 | +359 -45 | 0.585 ms/step [0.484, 0.686] at 96-128k (+3.1%) | identical |
| 06 | +82 -1 | 0.19 ms/step [0.02, 0.36], flat with depth | identical |
| 07 | +102 -10 | 0.99 ms/step at 96-128k | identical |
| 08 | +59 -0 | about 0.30 ms/step at about 102k | identical |
| 09 | +292 -0 | 0 by itself; installs kernels by op and shape | identical with an identity kernel |
| 10 | +147 -15 | prefill -342 ms/turn [-365, -321] at -ub 128 (-16%) | identical |
| 11 | +184 -0 | prefill -152 ms/turn [-168, -140] at 96-128k; on top of 10 at -ub 512: -153 [-157, -149] | identical |
| 12 | +314 -1 | 0.30 ms/step [0.17, 0.43] session (+1.9%); 0.32 [0.27, 0.38] at 96-128k | identical |
| 13 | +102 -0 | 0.19 ms/step [0.13, 0.26] session (+1.2%); every bucket clears | identical |
| 14 | +109 -1 | TTFT -79.7 ms/turn [-40.5, -118.9], turns 2-59; tokenize -123 ms at 96-128k | identical |
| 15 | +115 -0 | prefill -58.2 ms/turn [-50.2, -66.1], turns 2-59; -80.8 under 32k | identical |
| 16 | +549 -6 | -319 MiB VRAM; with 17, a hot-set draft and every expert on the GPU: session -13.8 s [-26.9, -0.7] (-8.3%) | quality-gated: stage-1 gate pass |
| 17 | +29 -0 | -96 MiB VRAM; speed unchanged | identical |
| 18 | +56 -0 | none: a diagnostic, the draft's routing for a hot-set draft | identical |
| 21 | +922 -8 | at 256k with every expert on the GPU: session -88.9 s [-89.2, -88.6] (-19.8%); a cost at 128k | quality-gated: stage-1 gate pass (identical vs off) |
| 22 | +190 -9 | at 256k: warm decode +11.8 tok/s [+5.1, +18.5], turns 60-116 (+6.1%); grows with depth | identical |
| kernel | 2 kernels | -0.56 ms/step [0.51, 0.62] (3.2%) vs the hook off | identical (59/59 turns, ~142k check comparisons) |

Series total: +4,517 -218 lines (01-22). 1.0.0 is frozen after 22 (crash fixes only); new work lands on b12521 modules. Evidence and conditions per mod: `engines/llama.cpp/1.0.0/mods.json`.

## llama.cpp master 7592b61d3: each mod

One RTX 5090, a MoE GGUF, prefill, one request at a time.

| # | mod | lines | gain vs stock | output |
|---|---|---|---|---|
| 01 | MoE routing sort, MMQ tile knob, pinned staging | +274 -6 | 10k 5,924 -> 6,455 tok/s | identical |
| 02 | host-RAM experts on a pinned copy stream | +474 -0 | 10k at -ncmoe 12: 2,757 -> 4,352 tok/s | identical |
| 03 | no checkpoint at a user start in the prompt head | +9 -1 | 10k 4,551 -> 5,278 (vs off) | untested |
| 04 | keep only the tail checkpoint | +7 -1 | 10k 5,892 -> 6,016 (vs off) | identical (one request) |
| 05 | PLE row prefetch on a thread | +9 -1 | 10k 5,105 -> 5,397 (vs off) | identical |
| 06 | MTP sidecar found in an `mtp/` subfolder | +32 -5 | none claimed | untested |

## vLLM 0.31.0: each mod

| # | mod | lines | gain vs stock | output |
|---|---|---|---|---|
| 01 | split-kv-verify: TRITON_ATTN keeps the split-KV kernel for a verify of up to 8 tokens | +4 -1 | MTP k=2 at ~3k context: 51 -> ~90 tok/s (self-reported) | untested (5e-4 vs the 2D kernel) |
| 02 | quark-lm-head: Quark quantizes an lm_head the config does not exclude | +7 -1 | 0 by itself; lets vLLM load an int4-head checkpoint | untested |

Measured on 0.31.0+rocm723, 2 x R9700 (self-reported, no raw files kept).

## How to apply: llama.cpp master 7592b61d3

Base: `7592b61d3b53ca5b39a25a03508fa1246b78c579`. Apply `engines/llama.cpp/7592b61d3/patches/*.patch` in order with
`git apply`. Each also applies alone on v1.0.0. Written before the one-switch rule: 01 and 02 are on by default
(`LLAMA_WS=0` turns 02 off), 03-05 have switches (`LLAMA_SERVER_CKPT_MIN_HEAD=N`, `LLAMA_SERVER_CKPT_TAIL_ONLY=1`,
`LLAMA_PLE_PREFETCH_ASYNC=1`), 06 is a file lookup.

## How to apply: vLLM 0.31.0

In vLLM's `site-packages`: `patch -p1 --dry-run < X.patch`, then without `--dry-run`. Revert: `patch -R -p1`.
No switch: applied = on. `tools/quark_lm_head_int4.py SRC DST` builds the int4-head checkpoint 02 loads.

## llama.cpp v1.0.0: limits and checks

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
- 14 shortens the wait before prefill (template + tokenize), not prefill or decode. Stock path when the prompt holds no
  control token, special-token parsing is off or the vocab adds EOS. It keeps the last call's pieces only: with -np > 1
  interleaved chats reuse less (still exact).
- 15 works only where the per-layer token-embedding table is lazily mapped (the qwen4exp graph); every other
  architecture is a no-op. One detached background thread for the process lifetime. A read-ahead hint: the gain depends
  on the page cache and storage. No check mode.
- 16 needs -np 1 and an architecture that opts in (the qwen4exp graph); refused with the stock layout otherwise and with
  `GGML_CUDA_GRAPH_OPT=1`. Alone it frees 319 MiB and costs +0.09 ms a step; the speed comes from spending that memory
  (the resident setup in its row). A state saved with it on restores only into a server with it on.
- 17 shares one cuBLAS handle between contexts that run one at a time on one thread (llama-server's target and draft).
  Any value turns it on. No check mode.
- 18 is for a counting run, not for serving: it syncs on every draft layer. The file is written every 20,000 draft
  forwards and at a normal exit. BoltBeam's `boltbeam derive hot-draft` reads it.
- 21 has no fixed N: `boltbeam tune` derives it per machine, from the free memory at load and the growth
  `boltbeam load-probe --stress` measures. It needs 20 (graph-cap) on: without it CUDA graph executables of dead graph
  keys grow with the request rate, and a server loaded this close to full VRAM runs out of memory.
  It pages the target's sparse-attention layers only (q8_0 K/V, one sequence, one CUDA GPU); the MTP draft layer is not
  paged. N is a multiple of 4, >= 4096 and below the context, else ignored (stock); on b12521 every such N is safe, on
  1.0.0 N must also be a multiple of 8 (see kv-stream, control block, below). It pays only where the context
  cannot hold all K/V and every expert on the GPU: at 128k it costs about 8.6 tok/s and +25 ms prefill a turn.
  Prompt ubatches past N copy the cells they read from host RAM. Context shift skips the write-through (run with -c at
  the session size); a SET_ROWS kernel from 09 on a streamed layer would too. Debug builds assert on its host buffer.

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
| 14 | `LLAMA_TOKENIZE_INCR_CHECK=1` | |
| 15 | none | |
| 16 | `LLAMA_RS_ONE_COPY_CHECK=1` (2 aborts on a difference) | |
| 17 | none | |
| 18 | none | |
| 21 | `LLAMA_KV_STREAM_CHECK=1` (aborts on a differing byte or attention output) | G |
| 22 | `LLAMA_SPARSE_SPLIT_CHECK=1` (aborts on a differing index-list word) | G |

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

Nothing is parked. Parallel mask compaction (`LLAMA_FATTN_COMPACT_PAR`, one block per list, 0.10 ms/step) is archived:
sparse-split (22; b12521: kv/sparse-split) splits the same compaction over blocks.
Rejected or superseded patches (KV host old, staged gather a2, sparse-direct half 2, graph recapture, fattn-compact-par)
are in `archive/`.

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
    mods.json                     one row per mod; modules (Modules, below), parked, retired, archived, tools, engine flags, tested trees
    stack_budget.json             caps on the whole series (check 10 stack) and the Owner-approved raises
    patches/NN-name.patch         the series, in order (a folder with modules: patches/<module>.patch, one per module)
    parked/                       correct but weak, or a record kept beside the series; may come back
    retired/                      retired mods' patches: kept and linted, never applied or counted (Retired, below)
    archive/                      decommissioned or rejected patches; never applied (archive/README.md is generated)
    tools/                        helper scripts a mod needs (not patches)
index.json                        generated: every mod, archived patch and tool, for consumers
archive-ledger.json               append-only: every patch ever archived (id, sha256, switch); archived never comes back
benchmarks/<engine>-<version>-<run>/  a benchmark record: README.md (method, results, reproduce) + .json summaries
local.example.json                copy to local.json (gitignored): this machine's base checkouts and run roots
tools/check.py                    the one check; standard library only; names no engine; --write regenerates the exports
tools/test_check.py               its tests (python3 -m unittest discover -s tools -p 'test_*.py'); the hook runs them
.githooks/pre-commit              runs it on the staged content
```

Setup after a clone:

```
git config core.hooksPath .githooks          # the hook checks what you stage
cp local.example.json local.json               # then set this machine's paths (or ARKEY_IE_MODS_LOCAL=/path/to/file)
```

## Modules

A version folder may group its mods into modules (llama.cpp b12521 does). The unit to port on a base move is the patch,
so a module is one patch on one engine extension point: the engine files that upstream changes together.

- `modules`: rows `{name, title, point, patch, files, lines, budget, static}`, in apply order. `point` names the
  extension point. Every module applies alone on the base: no module needs another to apply, so the stack metrics
  `depth` and `stacked_hunks` stay at 1 and 0, and a base move ports each module once.
- A mod row is a feature of its module: `module`, one switch, its check mode, label, evidence, marker, limits. It
  holds no patch, files, lines, budget or `depends_on.apply`; `depends_on.run` still names the features it needs on.
- The switch rule is unchanged: every switch unset = stock. Building a module in does not turn its features on.
- The check counts lines, budgets and the stack metrics per module, and requires every switch-shaped literal of a
  module's patch to be declared by one of its rows. index.json keeps one entry per mod, with `module` and the module's
  patch, sha256, files and lines: a consumer applies each patch once.
- Code that needs no engine internals goes in the module's own new file: a new file cannot conflict on a base move.

llama.cpp b12521 modules: `kv` (sparse-direct, fattn-vec-skip, kv-stream, sparse-split), `indexer` (idx-pooled, kpool-input-device,
idx-inplace, topk-fused, idx-keys, topk-rowhist, kpool-mask-range), `kernels` (ext-kernels, ext-kernels-window, graph-cap, concat-rows), `experts` (moe-inplace, cublas-shared,
draft-route-count), `state` (mtp-share-head, graph-slots, rs-one-copy), `host` (ckpt-pinned, tokenize-incr,
ple-prefetch, sched-async-inputs). The switches, check modes and limits are those of the same mods on 1.0.0, above. idx-keys, topk-rowhist and kpool-mask-range exist on b12521 only (rows in its mods.json).
kernels/region-gdn-gate (b12521 only): the fourth region of `kernels/region-chain-k2560-k10240`, one generated kernel for the GDN gating
window at the verify step (two gemvs on one activation, add, softplus, mul, sigmoid: 6 launches to 1). Its window holds the engine's
reshapes, so the window hook takes an interior RESHAPE / VIEW at offset 0 of an earlier window node, and inputs a manifest entry names
alike must be one tensor. 256k, 3 pairs: decode +3.5 tok/s [+3.0, +4.0], identical text.
kernels/concat-rows (b12521 only, `GGML_CUDA_CONCAT_ROWS=1`, check `GGML_CUDA_CONCAT_ROWS_CHECK=1`): stock llama.cpp launches the
non-contiguous concat with one 256-thread block a row, so the decode step's [7, 10240] nodes start 10,240 blocks with 7 threads busy in
each. With the switch a block takes floor(256 / row length) whole rows (285 blocks there); from 129 elements a row up the stock launch
runs. 256k, 3 pairs: decode +2.6 tok/s [+2.5, +2.8], identical text (30/30 turns, 159,745 checked nodes, 0 differ). Measured in a build
that also held unreleased hook rules no manifest entry used; the build of this series alone gives identical text too (30/30 turns in
3 arms, 159,745 checked nodes, 0 differ).
state/mtp-share-head, pending row (b12521 only, part of `LLAMA_MTP_SHARE_HEAD=1`): the draft pairs the first token of a batch with the
target's last hidden row of the batch before. Stock llama.cpp zeroes that row only when the server starts, so the first token of a
new prompt is paired with the last row of the previous request. With the shared head that made a reply depend on the request before
it (one problem: 2,323 tokens on a fresh server, 2,231 after another request, first difference at character 6,052). With the switch
on, a sequence that restarts at position 0 is paired with zeros, as on a fresh server: the reply after another request then equals
the fresh-server reply byte for byte (2 problems, one a 9,455-token reply). A restart from a checkpoint already restores its own row
in stock llama.cpp and is unchanged; a restart at a later position without a checkpoint is not covered (limits in the row).
Cost at 256k, 3 pairs, identical text (57/57 turns in 6 runs): decode -0.6 tok/s [-0.8, -0.5], session wall +0.45 s [+0.32, +0.58] on 176 s,
about 0.25% and 0.04 s over the speed check's detection line: kept as a correctness fix; the cause of the cost is not known.
kv/kv-stream, control block (b12521 only, part of `LLAMA_KV_STREAM=N`): the paging state keeps three 64-bit counters after the page
table and the five slot arrays, at an offset of context/4 + 5*N/4 four-byte words. With N a multiple of 4 but not of 8 the counters sat
on a 4-byte boundary and the first call past N ended the server with a misaligned-address GPU exception (N 65,292 and 52,852 did).
The block now starts on a 16-byte boundary for every N (one offset helper used by the allocation and every access). On the fixed build
both values run 10k and more tokens past N; 128k session turns 1-30 give identical replies (30/30) with and without the fix. 1.0.0 has
the same code and is not fixed: there use a multiple of 8.

## Archive

`engines/<engine>/<version>/archive/` keeps every decommissioned or rejected patch, so nothing that was patched is lost.
Each has a row in that version's mods.json `archived` (why, measured reason, evidence path, date, lines, sha256)
and an entry in `archive-ledger.json`, which only grows. `archive/README.md` is generated from those rows. The check
verifies each sha256; a series or parked patch with an archived sha256 or switch fails; archived patches are never in
the series, parked or index.json's active list (index.json lists them under `archived`).
Parked (may come back) stays in `parked/`.

## Retired

Retired = decommissioned code that stays in the repo, whole and linted, but is no longer maintained: it does not count
toward the maintained lines. Two scopes:
- a version: engines.json `retired_versions {"<version>": {date, why}}` on the engine row (the version stays in
  `versions` and its folder stays where it is, so every path and id keeps working). Its rows are linted for format;
  nothing is applied, it has no stack measurement and keeps no `stack_budget.json` (git history keeps the old one).
- one mod: its row moves from `mods` (or `parked`) to the version's `retired` list with `retired {date, why}`, and its
  patch moves to `retired/`. The row keeps every field and is linted like a series row (lines, budget, switch, marker,
  row lint); a series or parked mod that depends on it fails.

Retired items are excluded from the stack totals, hot spots, overlapping and stacked hunks and dependency depth, are
never applied, and are not in index.json's `versions`, `mods`, `kernels` or `tools`, so a consumer (BoltBeam's
resolver reads those) never offers them. index.json names them under `retired {versions, mods}` (id, was, by, date,
why, patch, sha256, lines) so a consumer can say why an id is gone; the index schema and `requires` are unchanged.
Retired differs from archived: archived is a rejected or dead patch reduced to a ledger row that never comes back;
a retired mod keeps its full row and can return to the series, where it is measured again against the lowered caps
(so coming back needs the room: a raise if it crosses).

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
rule exceptions, `title` (one line: what it changes), `module` (its module, or null: Modules, above). Then `archived`, `tools` and `retired` (Retired, above). Gains, limits and evidence live in mods.json at the same commit.
`requires`: the oldest consumer version that reads it (`boltbeam`, from engines.json `requires`; the check fails on
one that is not a dotted version or has no why). A consumer reads the index at one commit of this repo, a pin or the
newest commit whose schema it reads and whose `requires` it meets, and records which.

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
- archived patches match their sha256 and the ledger; every file has a row; index.json and archive/README.md are fresh;
- the whole series is within its stack caps (below), and a stack health summary prints.

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
## Stack budget

Per-mod budgets keep each patch lean; they do not stop the stack from growing into many mods editing the same engine
code. `engines/<engine>/<version>/stack_budget.json` caps the whole series:

| metric | what |
|---|---|
| `total_lines` | added + removed lines over the series |
| `mods_per_file` | the most mods that patch one engine file |
| `overlap_hunks_per_file` | the most hunks in one file whose lines (context included) another mod's hunk already covers |
| `depth` | the longest `depends_on.apply` chain, in mods |
| `stacked_hunks` | hunks whose context or removed lines are another mod's added lines: mods stacked on mods |

The check applies the series' hunks to a model of each file (placed like `git apply`; with the base checkout it
reproduces the patched files exactly) to know which mod added and which hunks covered each line.

Each cap is derived from a measurement, `set {measured, mods, date}`: cap = measured + ceil(measured / series mods),
room for one more average mod. A crossing fails and names the files, functions and mods, and the refactor it points at:
when many mods patch the same code, replace them with one extension point (a hook patch) plus rows, as the ext-kernel
hook did for kernels. If the owner accepts the growth instead, append to the metric's `raised`
`{from, to, measured, why, date, approved: "Owner-approved YYYY-MM-DD"}` with to = the value measured after review
cuts, and set `cap` to it. The check refuses a cap that is neither derived nor the last raise, a raise to anything but
its measured value, a new raise or a new or changed `set` whose measured value is not today's, a cap raised against
HEAD without a new raise, and a raise removed against HEAD. The approval is a recorded string, not a signature: review
of the raise commit is what ties it to the owner. A measured 0 gives a cap of 0 (the first such crossing needs a raise).

The ratchet: freed room is not silently reused. When a metric drops below the measurement its cap rests on (`set`, or
the last raise's `measured`) because a mod was retired, archived or shrunk, and the margin of the new value is below
the cap, the check fails until the cap is lowered: re-derive `set` to today's measurement and set cap = measured +
ceil(measured / series mods). The failure prints the exact `cap` and `set` to write. When the metric has raises, the
new `set` records `after_raises` (how many raised entries came before it): those stay as append-only history and no
longer chain. Without the base checkout, hunk placement is approximate, so a drop in `overlap_hunks_per_file` or
`stacked_hunks` is a named SKIP (with the values) there and a failure only where the checkout is. Growing back over the lowered cap is a crossing like any other: a refactor, or an Owner-approved raise
with a recorded why.

Today's hot spots in llama.cpp v1.0.0, the first refactor candidates: `src/models/qwen4exp.cpp` (6 mods),
`src/llama-memory-hybrid-idx.cpp` (4 mods, 15 overlapping hunks: 01, 03, 05, 08) and `src/llama-context.cpp` (4 mods).

## License

- This repository's own material (tools/, the json rows, the docs, the added lines of each patch, the tools/ scripts
  beside a version) is MIT: `LICENSE`, Copyright (c) 2026 Arkey.
- The context and removed lines of the llama.cpp patches come from llama.cpp: MIT, `LICENSES/MIT-llama.cpp.txt`.
- The vLLM patches modify vLLM files: Apache License 2.0, `LICENSES/Apache-2.0.txt`. Each vLLM patch says that it
  was modified and what changed (section 4(b)). `quark_lm_head_int4.py` is original work except one constant (`REV`,
  vLLM's AWQ reverse pack order), which carries vLLM's license.
