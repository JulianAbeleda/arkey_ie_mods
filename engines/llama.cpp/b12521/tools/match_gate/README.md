# match_gate

The MATCH step of a region kernel's life (propose -> check -> MATCH -> time -> decide), on the CPU: `match_gate.sh` runs a
tree's unmodified engine build on a do-nothing CUDA runtime (`cudastub.cpp`) with the hook recorder (`hookspy.cpp`) in
front, drives it through a recorded session's turns (`drive_session.py`), then replays the hook's own matcher over the
recorded graphs (`hook_gate.cpp`) and writes `gate.json` (arkey.match_gate.v1, stamped by `stamp.py`). Usage, inputs and
exit codes: the header of `match_gate.sh`. BoltBeam calls it through its llama.cpp match adapter
(`boltbeam/adapters/region_match_llama_cpp.py`) and reads the folder it writes.

## What a dry run proves, and what it does not

The matcher takes a window by the tensors' shapes, wiring and in-buffer offsets, so a dry run's verdict holds on the
device only where its graphs and offsets are the device's.

- **Proven (measured):** 72,680 of 72,680 tensor offsets equal on 17 graph shapes between a dry run and one GPU run of
  the same build, kernel list and switch set, turns 1-3 (`compare_spy.py`, the BoltBeam step 6 review, 2026-11-20);
  earlier, 1,181 of 1,181 on 5 draft graphs (boltbeam-runs `hook-match-gate-20261120`).
- **Not proven:** a graph the dry run never records. The multi-token verify widths: that GPU run had a 3-token verify step
  graph the dry run lacked, so no verdict covers it. Nor deeper turns or longer contexts, nor other builds or kernel
  lists: there a result placed on an input's bytes can come out differently on the device.
- **Counts a step are the dry run's:** they are weighted by the dry run's own pass mix. The do-nothing runtime's async
  copies return 0, so its draft acceptance, and with it the share of draft and verify passes, is not the device's.

BoltBeam carries the same scope as the dry run's note `offsets_unproven` (its `engine_adapters.json`, the llama.cpp
`region_match.dry_run` row) on every report the tool's dry run makes.
