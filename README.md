# Arkey inference engine mods

Patches that make llama.cpp faster on long chats with big mixture-of-experts models. Each one is off until you turn it
on, and off means plain llama.cpp.

On a 128k-token coding chat, all of them together make llama.cpp **2.92x faster** than stock, with the same answers.

## Install

Needs an NVIDIA GPU, CUDA and CMake. Two commands:

```
brew install JulianAbeleda/tap/boltbeam
boltbeam mods apply
```

`boltbeam mods apply` fetches this repo at the commit BoltBeam pins into BoltBeam's own folder (a clone you name in
`BOLTBEAM_IE_MODS` is used instead), says what it fetches and from where, then builds the patches into a copy of the llama.cpp
source BoltBeam runs (`--src` to name another) and checks each one is on.

### By hand

Four commands:

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b12359    # = v1.0.0
git apply /path/to/arkey_ie_mods/engines/llama.cpp/1.0.0/patches/*.patch
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_COMMIT=b1b770d9 -DGGML_BUILD_COMMIT=b1b770d9 -DGIT_EXE=OFF
cmake --build build -j --target llama-server
```

Keep the three `b1b770d9` flags: without them the bundled GPU kernels don't load.

## Run

Turn every patch on and start the server:

```
export LLAMA_KPOOL_INCREMENTAL=1 LLAMA_SPARSE_DIRECT=1 LLAMA_IDX_POOLED=1 LLAMA_MTP_SHARE_HEAD=1 \
  LLAMA_KPOOL_INPUT_DEVICE=1 LLAMA_SEQ_RM_INDEXED=1 LLAMA_FATTN_VEC_SKIP=1 LLAMA_IDX_INPLACE=1 \
  LLAMA_MOE_INPLACE=1 LLAMA_CKPT_PINNED=1 LLAMA_TOPK_FUSED=1 LLAMA_GRAPH_SLOTS=1 \
  GGML_CUDA_EXT_KERNELS=/path/to/arkey_ie_mods/engines/llama.cpp/1.0.0/kernels/bf16-smallm-gemv-k10240/manifest.json

build/bin/llama-server -m MODEL.gguf -md DRAFT.gguf -c 131072 -fa on -ngl 99 \
  -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -ub 1024 -ncmoe 16 -lm none --moe-cache-mib 6240
```

`--moe-cache-mib` is how much spare GPU memory goes to caching experts. 6240 fits a 32 GB card at 128k;
[BoltBeam](https://github.com/JulianAbeleda/BoltBeam-Public) works out the right number for yours:
`boltbeam tune MODEL.gguf --engine llama.cpp --context 131072`.

To go back to stock, unset the variables. Each patch, what it does and its limits: [DETAILS.md](DETAILS.md).

## Benchmarks

Stock llama.cpp v1.0.0 against the same version with all patches on, on the same 59-turn, 128k-token coding chat,
3 runs each:

| | stock | with mods | |
|---|---|---|---|
| whole chat | 485 s | 166 s | **2.92x faster** |
| writing speed | 77 tok/s | 226 tok/s | 2.93x |
| reading each new message | 2.61 s | 0.83 s | 3.14x |
| wait before the first word | 2.71 s | 0.94 s | 2.9x |

Same answers: the patches give identical output. The recommended flags change rounding slightly, so we also checked
quality on MMLU-Pro and LiveCodeBench: no loss against stock.

Full method, every run and how to reproduce:
[benchmarks/llama.cpp-1.0.0-0.3.26](benchmarks/llama.cpp-1.0.0-0.3.26/README.md).
Against Strata on the same machine and chat: [vs-strata.md](benchmarks/llama.cpp-1.0.0-0.3.26/vs-strata.md).

### Test machine

- GPU: NVIDIA RTX 5090 32 GB, capped at 450 W
- CPU: AMD Ryzen 7 5800X (8 cores)
- RAM: 32 GB DDR4-2667
- Board: ASRock B550M-ITX/ac, PCIe 4.0 x16
- Software: NVIDIA driver 595.99.02, CUDA 13.2, Ubuntu (Linux 6.8)
- Model: Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M (ISTA-DASLab) with its MTP draft head
- One request at a time, temperature 0

## License

MIT. Patch lines taken from an engine keep that engine's license: [DETAILS.md](DETAILS.md#license).
