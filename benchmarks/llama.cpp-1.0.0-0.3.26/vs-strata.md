# The mods setup vs Strata, same session (2026-11-17)

Same machine, same 59-turn session to 129k tokens, same 450 W cap and start gate as [README.md](README.md). Order
`mods1 strata1 | strata2 mods2 | mods3 strata3`, a fresh server per run. Ratios: per pair, mods over Strata for
tok/s, Strata over mods for times (>1 = mods faster); mean and 95% t interval over the 3 pairs.
Every run: `vs-strata.json`.

## Setups

- **mods**: llama.cpp v1.0.0 + 01-13, the setup in [README.md](README.md) (q8_0 KV, `-ub 1024`, `-lm none`, MTP draft
  head, at most 3 drafted tokens a step).
- **Strata**: UnbiasedStrata `ff19037`, engine 0.1.45. Its own packed form of the same IQ1_M weights and its own
  expert profile. Flags as passed:
  `--pack <its IQ1_M pack> --native <shard 1> --ple-gguf <shard 2> --expert-profile <its profile> --expert-cache auto
  --prefill auto --max-context 131072 --kv int8 --ple-io direct --spec 3 --mtp <its MTP head>
  --resident-budget-gib 14 --spec 4 --spec-min-p 0.5`. Its log reports `spec=6`, `spec_min_p=0.50`.

## Results

| | mods | Strata | mods / Strata [95%] |
|---|---|---|---|
| whole session | 166.0 / 165.9 / 166.0 s | 171.6 / 168.8 / 169.6 s | 1.024x [1.003, 1.045] |
| decode, session | 226.5 tok/s | 208.6 tok/s | 1.086x [1.078, 1.094] |
| decode step, session | 14.93 ms | 15.58 ms | 1.043x [1.034, 1.053] |
| tokens per step | 3.38 | 3.25 | |
| prefill per turn | 830 ms | 780 ms | 0.940x [0.938, 0.943] |
| time to first token per turn | 0.932 s | 0.883 s | 0.947x [0.942, 0.952] |
| tokens generated per session | 25,141 | 24,380-24,994 | |

By depth (decode tok/s; wall per turn):

| depth | decode mods / Strata | wall per turn mods / Strata |
|---|---|---|
| <32k | 1.116x [1.103, 1.130] | 1.045x [0.988, 1.101] |
| 32-64k | 1.083x [1.063, 1.103] | 1.036x [1.020, 1.051] |
| 64-96k | 1.082x [1.038, 1.126] | 1.029x [1.000, 1.058] |
| 96-128k | 1.067x [1.037, 1.097] | 0.991x [0.936, 1.046] |

Where Strata leads: prefill per turn in every depth bucket (4-9% faster), time to first token in every bucket (4-7%).
At 96-128k its decode step equals ours (1.002x [0.995, 1.009]) and the per-turn wall is a tie.

What differs besides speed:
- Strata's replies vary run to run: identical to its own first run on 24 and 18 of 59 turns. The mods setup's are
  identical 59 of 59.
- Strata generated about 3% fewer tokens per session (24.4-25.0k against 25.1k), which shortens its session. Per
  token, the mods setup's lead is larger than the session ratio.
- KV formats differ: q8_0 (mods) against int8 (Strata).
- No quality benchmark was run on Strata here. Mods and Strata replies are identical on 7 of 59 turns.
