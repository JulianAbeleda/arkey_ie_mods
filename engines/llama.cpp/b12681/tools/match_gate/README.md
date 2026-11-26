# match_gate (b12681)

The MATCH step of a region kernel's life (propose -> check -> MATCH -> time -> decide), on the CPU: `match_gate.sh` runs a
tree's unmodified engine build on a do-nothing CUDA runtime (`cudastub.cpp`) with the hook recorder (`hookspy.cpp`) in
front, drives it through a recorded session's turns (`drive_session.py`), then replays the hook's own matcher over the
recorded graphs (`hook_gate.cpp`) and writes `gate.json` (arkey.match_gate.v1, stamped by `stamp.py`). Usage, inputs and
exit codes: the header of `match_gate.sh`. BoltBeam calls it through its llama.cpp match adapter
(`boltbeam/adapters/region_match_llama_cpp.py`), which finds this folder by the build's base commit in index.json.

The b12521 tool, carried to this version's base (d141011c2). It builds at run time from the given tree's own sources, so
what changed is only where that tree differs:

- `gate_build.sh`: the matcher's own folder is on the include path (this base's ext-kernels.cpp includes `ext-src1q.cuh`
  beside it). `ext_match` itself is byte-identical to b12521's.
- `dryrun.sh`: memcap queues (`--wait`) instead of refusing when the box's memory is reserved by other jobs;
  `drive_session.py` starts its 300 s health wait once the queued server has started; `match_gate.sh` runs the dry run
  as its own process group, stops it when the drive fails (a server still queued included) and waits for it to end
  before the replay reads the recording.
- The hook entry has the same signature. This base also calls it once per evaluation with i = -1 (a reset of the
  quantized-src1 records); the recorder's new-pass rule (the node index went back) starts a pass there, and the call
  takes no node.

## What a dry run proves, and what it does not

The matcher takes a window by the tensors' shapes, wiring and in-buffer offsets, so a dry run's verdict holds on the
device only where its graphs and offsets are the device's.

- **Proven on b12521 (measured), not yet on this base:** 72,680 of 72,680 tensor offsets equal on 17 graph shapes
  between a dry run and one GPU run of the same build, kernel list and switch set, turns 1-3 (`compare_spy.py`,
  BoltBeam step 6 review, 2026-11-20). No GPU recording of a b12681 build has been compared yet.
- **Not proven:** a graph the dry run never records (multi-token verify widths), deeper turns or longer contexts, other
  builds or kernel lists. Nor what happens after a window: a `src1_q8_1` entry's quantized output is taken by a later
  stock matmul (`ggml_cuda_ext_src1_q`), outside the matcher; the replay does not judge that hand-off (the hook's check
  mode does).
- **Counts a step are the dry run's:** weighted by the dry run's own pass mix (the stub's async copies return 0, so its
  draft acceptance is not the device's).
