# b12521 mods on HIP (work in progress, local branch)

`hip-port.diff` applies on top of the six b12521 patches (kv, indexer, kernels, experts, state, host, in that order)
and makes them build for HIP (gfx1201, ROCm, 2x R9700, 2026-11-20). Nothing here has run on the GPUs yet.

- vendors/hip.h: HIP names for cudaStreamIsCapturing, cudaStreamCaptureStatus(None), cudaMemcpyDefault
- fattn.cu, kv-stream.cu: raw `__shfl_*_sync(0xffffffff, ...)` -> ggml_cuda_shfl_xor / ggml_cuda_shfl_up (HIP needs a
  64-bit mask)
- fattn-vec.cuh: `__any_sync` -> ggml_cuda_any; with skip_masked the K threads per row are WARP_SIZE on HIP too (the
  skip decision is per K row; RDNA used 2, other HIP 4)
- ggml-cuda.cu: ext-kernels stubbed on HIP (returns 0: the stock path); the hook loads CUDA cubins through the driver
  API, a hipModule loader is not written

To fold in: each change goes into its mod's patch (the module that owns the file), budgets re-measured, then every
mod's check mode run on the R9700 before the mods' backend rows say HIP. Open: whether topk-fused and topk-rowhist are
active on HIP (their path may sit behind CUB). Design notes: the AMD box write-up (owner's docs, 2026-11-20).
