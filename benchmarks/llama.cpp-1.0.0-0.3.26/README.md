# llama.cpp v1.0.0: the 13-mod series vs stock (BoltBeam 0.3.26)

Measured 2026-11-16 and 2026-11-17. Stock llama.cpp v1.0.0 against the same engine with the series 01-13 and the
setup BoltBeam 0.3.26 derives. Machine and model: [../../README.md#test-machine](../../README.md#test-machine).
Machine-readable: `speed.json` (every run, every depth bucket, every pair) and `quality.json`.

## Method

Setups:
- **stock**: llama.cpp v1.0.0 (b1b770d9), no patches.
  `-m MODEL -md DRAFT -c 131072 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0`, every other setting stock's default.
  It drafts with the same MTP head, at most 3 tokens a step, as the mods setup does.
- **mods**: the same base + patches 01-13, every switch on, 09 loading `kernels/bf16-smallm-gemv-k10240`.
  `-c 131072 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -fa on -ngl 99 -ncmoe 16 -ub 1024 -b 2048
  --moe-cache-mib 6240 -lm none --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128 -bs
  --cache-ram 0 -np 1 -t 8 --temp 0 --no-warmup -fit off`.

Workload: one 59-turn coding session that grows to 129.3k tokens, sent one turn at a time, temperature 0. The same
session file for every run. Each turn's reply length is fixed, so both setups generate the same token counts.

Runs: 3 pairs, alternating, `S1 F1 | F2 S2 | S3 F3`, a fresh server per run. GPU power capped at 450 W and reset
before every run. Start gate before every run: GPU at or under 70 C and the board sensor under 58 C. A run that fails
a gate (temperature, an allocation failure, a mod banner missing, a kernel not loaded) is rerun, never averaged.

Intervals: for each pair, the ratio mods/stock (tok/s, tokens per step) or stock/mods (times). The gain is the mean of
the 3 pair ratios; the bracket is its 95% t interval (2 degrees of freedom). Buckets are by context depth at the
turn's start.

## Speed

| | stock | mods | gain [95%] |
|---|---|---|---|
| decode, <32k | 96.0 tok/s | 235.8 tok/s | 2.46x [2.44, 2.48] |
| decode, 32-64k | 86.0 tok/s | 234.4 tok/s | 2.73x [2.70, 2.76] |
| decode, 64-96k | 74.2 tok/s | 229.0 tok/s | 3.09x [3.05, 3.12] |
| decode, 96-128k | 61.3 tok/s | 207.9 tok/s | 3.39x [3.35, 3.43] |
| decode, session | 77.2 tok/s | 226.1 tok/s | 2.93x [2.90, 2.96] |
| decode step, session | 44.5 ms | 15.0 ms | 2.98x [2.95, 3.00] |
| prefill per turn | 2.61 s | 0.83 s | 3.14x [3.09, 3.18] |
| time to first token per turn | 2.71 s | 0.94 s | 2.90x [2.86, 2.94] |
| wall per turn | 8.23 s | 2.82 s | 2.92x [2.89, 2.94] |
| whole session | 485 s | 166 s | 2.92x [2.89, 2.94] |

Tokens per step: 3.44 stock, 3.38 mods. The gain is the step time, not acceptance.
Run-to-run spread of session decode: 0.6 tok/s stock, 0.1 tok/s mods. Repeats of one setup give identical reply
text, 59 of 59 turns.

## Quality

The mods compute the same bits as stock, except 04 (draft proposals only). The setup does not: `-ub 1024` and the
expert cache change rounding. Mods vs stock reply text is identical on 13 of 59 turns. So the setup is gated. Every
rule was fixed before its run.

1. **Distribution gate.** Every next-token distribution of 18 replies (4,608 positions), teacher-forced, against 18
   floor pairs that differ only by rounding stock already accepts (CPU vs GPU, cache size, batch size). The line is
   the widest top-1 flip of any floor reply: 5.92 nats. Fail at 2 or more replies above it. Result: 0 (widest
   5.90). Holds. `-lm none` alone: identical at 4,608 of 4,608 positions.
2. **MMLU-Pro**, 150 items, thinking on. Non-inferiority against stock, allowance = the churn of a noise arm (stock
   with `-b 8 -ub 8`, a change that must not matter). Stock 103 (68.7%), mods 104 (69.3%), noise 100. Pass.
3. **LiveCodeBench**, 140 items, same rule. Stock 69 (49.3%), mods 73 (52.1%); 21 only stock solved, 25 only mods.
   Pass.

Two caveats:
- The LiveCodeBench noise arm was not noise-like: 112 of 140 (80%), with 17 looping replies against stock's 57. Its
  allowance (45 points) is inflated. The number that means something there is the direct one: 73 vs 69, McNemar
  p = 0.66. No sign of harm, no sign of a gain.
- Replies that loop to the length cap ("cycle") count as answers under the pre-registered rule. Counted as
  truncations instead, every arm is past the 5% truncation limit (MMLU-Pro 14.0%, LiveCodeBench 41.4%) and both
  verdicts are inconclusive.

## Reproduce

Build both engines from the base (a git checkout at b1b770d9, or a tree without .git):

```
cmake -B build-stock -DGGML_CUDA=ON && cmake --build build-stock -j --target llama-server
for p in arkey_ie_mods/engines/llama.cpp/1.0.0/patches/*.patch; do git apply "$p"; done
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_COMMIT=b1b770d9 -DGGML_BUILD_COMMIT=b1b770d9 -DGIT_EXE=OFF
cmake --build build -j --target llama-server
```

Or let BoltBeam do it (04 is quality-gated, so name the label):

```
boltbeam screen mods --engine llama.cpp --apply --label quality-gated
boltbeam tune MODEL.gguf --engine llama.cpp --context 131072     # derives -ub, -ncmoe, --moe-cache-mib, -lm
```

Run the mods server:

```
LLAMA_KPOOL_INCREMENTAL=1 LLAMA_SPARSE_DIRECT=1 LLAMA_IDX_POOLED=1 LLAMA_MTP_SHARE_HEAD=1 \
LLAMA_KPOOL_INPUT_DEVICE=1 LLAMA_SEQ_RM_INDEXED=1 LLAMA_FATTN_VEC_SKIP=1 LLAMA_IDX_INPLACE=1 \
LLAMA_MOE_INPLACE=1 LLAMA_CKPT_PINNED=1 LLAMA_TOPK_FUSED=1 LLAMA_GRAPH_SLOTS=1 \
GGML_CUDA_EXT_KERNELS=arkey_ie_mods/engines/llama.cpp/1.0.0/kernels/bf16-smallm-gemv-k10240/manifest.json \
build/bin/llama-server -m MODEL.gguf -md DRAFT.gguf -c 131072 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
  -fa on -ngl 99 -ncmoe 16 -ub 1024 -b 2048 --moe-cache-mib 6240 -lm none \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128 -bs --cache-ram 0 -np 1 -t 8 --temp 0 \
  --no-warmup -fit off
```

The log must say `ext_load: loaded` once per manifest entry (2), `graph_slots: on` and `top_k_fused: on`. The stock
server takes the same `-m -md -c -ctk -ctv -ctkd -ctvd` flags and no switch. `--moe-cache-mib 6240` fits 32 GB of
VRAM at 128k on this machine; on another, take what `boltbeam tune` derives.

Not published: the 59-turn session itself and the driver that sends it. Any long multi-turn session at batch 1
exercises the same paths; the numbers above hold only for this one.
