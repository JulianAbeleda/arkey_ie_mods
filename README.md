# Arkey inference engine mods

Opt-in patches for inference engines. Each mod is one switch: unset, the engine runs its stock path.
Limits, check modes, repo rules and the check: [DETAILS.md](DETAILS.md). Not submitted upstream.

## Apply

### llama.cpp v1.0.0 (CUDA)

Base: tag b12359, commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`.

With [BoltBeam](https://github.com/JulianAbeleda/BoltBeam-Public) 0.3.26 or later, one command builds the patched
engine, checks each mod is on and refuses a build whose kernels did not load:

```
boltbeam screen mods
```

By hand:

```
cd llama.cpp                                  # the base b1b770d9: a clean checkout, or a tree without .git (git archive, tarball)
for p in /path/arkey_ie_mods/engines/llama.cpp/1.0.0/patches/*.patch; do git apply "$p"; done
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_COMMIT=b1b770d9 -DGGML_BUILD_COMMIT=b1b770d9 -DGIT_EXE=OFF
cmake --build build -j --target llama-server
grep GGML_COMMIT build/ggml/src/ggml-version.h   # must say "b1b770d9"
```

The three commit flags let 09 load a kernel: its manifest loads only on a build whose ggml commit matches. Without
them a tree without .git builds as "unknown" and the engine's own kernels run silently (one WARN line).

Turn them on (all 13, the measured setup):

```
LLAMA_KPOOL_INCREMENTAL=1 LLAMA_SPARSE_DIRECT=1 LLAMA_IDX_POOLED=1 LLAMA_MTP_SHARE_HEAD=1 \
LLAMA_KPOOL_INPUT_DEVICE=1 LLAMA_SEQ_RM_INDEXED=1 LLAMA_FATTN_VEC_SKIP=1 LLAMA_IDX_INPLACE=1 \
GGML_CUDA_EXT_KERNELS=/path/arkey_ie_mods/engines/llama.cpp/1.0.0/kernels/bf16-smallm-gemv-k10240/manifest.json \
LLAMA_MOE_INPLACE=1 LLAMA_CKPT_PINNED=1 LLAMA_TOPK_FUSED=1 LLAMA_GRAPH_SLOTS=1 \
build/bin/llama-server -m MODEL.gguf -md DRAFT.gguf -c 131072 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 \
  -ub 1024 -ncmoe 16 --moe-cache-mib <from boltbeam tune> -lm none -ctkd q8_0 -ctvd q8_0
```

- `--moe-cache-mib` depends on free VRAM: `boltbeam tune` derives it, the prefill chunk and the expert placement.
- `-lm none` reads host experts into RAM at load. It matters when the model files are larger than RAM.
- Verify: the log says `ext_load: loaded <id>` once per kernel, and each mod logs its switch at WARN level.
- Revert: `git apply -R` in reverse order, or just unset the switches.
- Order dependencies: 05 needs 01 and 03; 07 needs 02; 08 needs 03. The rest apply alone on the base.

| # | mod | switch |
|---|---|---|
| 01 | k-pool layout updated from the edit point | `LLAMA_KPOOL_INCREMENTAL=1` |
| 02 | sparse attention reads q8_0 K/V rows by index | `LLAMA_SPARSE_DIRECT=1` |
| 03 | indexer cache keeps pooled keys only | `LLAMA_IDX_POOLED=1` |
| 04 | draft head reads the target's output projection | `LLAMA_MTP_SHARE_HEAD=1` |
| 05 | k-pool inputs live on the device | `LLAMA_KPOOL_INPUT_DEVICE=1` |
| 06 | seq_rm walks an index, not the cache | `LLAMA_SEQ_RM_INDEXED=1` |
| 07 | vector attention kernel skips masked rows | `LLAMA_FATTN_VEC_SKIP=1` |
| 08 | indexer scores pooled q8_0 keys in place | `LLAMA_IDX_INPLACE=1` |
| 09 | external kernel hook | `GGML_CUDA_EXT_KERNELS=/path/manifest.json` |
| 10 | prefill reads cached experts in place | `LLAMA_MOE_INPLACE=1` |
| 11 | context checkpoints in a pinned pool | `LLAMA_CKPT_PINNED=1` |
| 12 | radix top-k of a few rows in one launch | `LLAMA_TOPK_FUSED=1` |
| 13 | draft context keeps a graph per small ubatch size | `LLAMA_GRAPH_SLOTS=1` |

Kernel: `kernels/bf16-smallm-gemv-k10240/` installs through 09. Two BF16 small-batch GEMV kernels written by arkey's
scheduler, not by hand. sm_120, build b1b770d9 only. Point 09 only at kernels you trust.

### llama.cpp master 7592b61d3 (MoE prefill)

Base: `7592b61d3b53ca5b39a25a03508fa1246b78c579`. Apply `engines/llama.cpp/7592b61d3/patches/*.patch` in order with
`git apply`. Each also applies alone on v1.0.0. Written before the one-switch rule: 01 and 02 are on by default
(`LLAMA_WS=0` turns 02 off), 03-05 have switches (`LLAMA_SERVER_CKPT_MIN_HEAD=N`, `LLAMA_SERVER_CKPT_TAIL_ONLY=1`,
`LLAMA_PLE_PREFETCH_ASYNC=1`), 06 is a file lookup.

### vLLM 0.31.0

In vLLM's `site-packages`: `patch -p1 --dry-run < X.patch`, then without `--dry-run`. Revert: `patch -R -p1`.
No switch: applied = on. `tools/quark_lm_head_int4.py SRC DST` builds the int4-head checkpoint 02 loads.

## Benchmarks

### llama.cpp v1.0.0: all 13 vs stock

All 13 on with the setup above, against stock v1.0.0, same 128k session, 3 paired runs:

| | stock | mods | |
|---|---|---|---|
| whole session | 485 s | 166 s | 2.92x |
| decode | 77 tok/s | 226 tok/s | 2.93x (3.39x at 96-128k) |
| prefill per turn | 2.61 s | 0.83 s | 3.14x |
| time to first token | 2.71 s | 0.94 s | 2.9x |

Output: the mods are identical to stock except 04 (draft only). The setup's flags change rounding, so it is
quality-gated: a next-token distribution gate (0 of 18 replies past the stock noise floor) and MMLU-Pro and
LiveCodeBench non-inferiority against stock (both pass).

Method, every run, the quality gates and how to reproduce:
[benchmarks/llama.cpp-1.0.0-0.3.26](benchmarks/llama.cpp-1.0.0-0.3.26/README.md).
Against another engine on the same session and machine: [vs-strata.md](benchmarks/llama.cpp-1.0.0-0.3.26/vs-strata.md).

### Test machine

- GPU: NVIDIA GeForce RTX 5090 32 GB, power cap 450 W for timed runs. PCIe 4.0 x16 (host-to-device 28.3 GB/s pinned,
  11.5 GB/s pageable, measured).
- CPU: AMD Ryzen 7 5800X (8 cores). RAM: 32 GB DDR4-2667 (2 x 16 GB; 31 GiB usable). Board: ASRock B550M-ITX/ac.
- NVIDIA driver 595.99.02, CUDA 13.2, Linux 6.8 (Ubuntu).
- Model: Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M (ISTA-DASLab; 1.9 bits per original parameter, 256 of 512 experts
  kept) + its MTP draft head (k=3), q8_0 KV.
- Workload: one request at a time, a 59-turn coding session to about 129k tokens, temperature 0.

### llama.cpp v1.0.0: each mod

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
| kernel | 2 kernels | -0.56 ms/step [0.51, 0.62] (3.2%) vs the hook off | identical (59/59 turns, ~142k check comparisons) |

Series total: +2,183 -131 lines. Evidence and conditions per mod: `engines/llama.cpp/1.0.0/mods.json`.

### llama.cpp master 7592b61d3: each mod

One RTX 5090, a MoE GGUF, prefill, one request at a time.

| # | mod | lines | gain vs stock | output |
|---|---|---|---|---|
| 01 | MoE routing sort, MMQ tile knob, pinned staging | +274 -6 | 10k 5,924 -> 6,455 tok/s | identical |
| 02 | host-RAM experts on a pinned copy stream | +474 -0 | 10k at -ncmoe 12: 2,757 -> 4,352 tok/s | identical |
| 03 | no checkpoint at a user start in the prompt head | +9 -1 | 10k 4,551 -> 5,278 (vs off) | untested |
| 04 | keep only the tail checkpoint | +7 -1 | 10k 5,892 -> 6,016 (vs off) | identical (one request) |
| 05 | PLE row prefetch on a thread | +9 -1 | 10k 5,105 -> 5,397 (vs off) | identical |
| 06 | MTP sidecar found in an `mtp/` subfolder | +32 -5 | none claimed | untested |

### vLLM 0.31.0: each mod

| # | mod | lines | gain vs stock | output |
|---|---|---|---|---|
| 01 | split-kv-verify: TRITON_ATTN keeps the split-KV kernel for a verify of up to 8 tokens | +4 -1 | MTP k=2 at ~3k context: 51 -> ~90 tok/s (self-reported) | untested (5e-4 vs the 2D kernel) |
| 02 | quark-lm-head: Quark quantizes an lm_head the config does not exclude | +7 -1 | 0 by itself; lets vLLM load an int4-head checkpoint | untested |

Measured on 0.31.0+rocm723, 2 x R9700 (self-reported, no raw files kept).

## License

This repository's own material is MIT (`LICENSE`, Copyright (c) 2026 Arkey). Patch context and removed lines carry
their engine's license: llama.cpp MIT (`LICENSES/MIT-llama.cpp.txt`), vLLM Apache 2.0 (`LICENSES/Apache-2.0.txt`);
details in [DETAILS.md](DETAILS.md#license).
