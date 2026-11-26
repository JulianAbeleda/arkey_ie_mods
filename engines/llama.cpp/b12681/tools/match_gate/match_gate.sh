#!/bin/bash
# match_gate.sh OUT TREE MANIFEST CTX TURNS [ENV...] -- [hook_gate options, e.g. --expect ID=N ...]
# The MATCH step of a region kernel's life (propose -> check -> MATCH -> time -> decide), one command, CPU only:
#   1 dry run: TREE's own engine build (TREE/build/bin, unmodified) on the do-nothing CUDA runtime, driven through TURNS (FIRST-LAST)
#     of a recorded session at context CTX with MANIFEST and the switches ENV; the recorder stores every graph with its addresses
#     and what the engine's matcher took at every node (spy.jsonl); the driver marks each turn's decode steps (turns.json);
#   2 hook_gate, built from TREE's own ext-kernels.cpp, replays the stored graphs, checks itself against the engine's result and
#     reports per entry: windows taken a decode step, shape spots, refusals with their byte ranges, the rule that rejected a spot.
# Machine inputs, from the environment (nothing machine-specific lives in this folder):
#   MATCH_GATE_SESSION  the recorded session whose turns drive the run: {system, turns: [{i, user, assistant, reply_tokens}]}
#   MATCH_GATE_SERVER   llama-server's flags after the binary, without -c, --host or --port (model, draft, cache types, ...)
#   MATCH_GATE_REQUEST  a JSON file: the request fields sent with every turn, the served requests' own (BoltBeam's match adapter
#                       writes session-ab's). A field that adds graph nodes must be sent, or the dry run's step graph is not the
#                       served one (ignore_eos is a logit-bias sampler stage at the graph's tail). Default: none.
#   MATCH_GATE_BUILD    where the stub runtime (per set of symbols the tree imports), the recorder and the replay are built
#                       (default ~/.cache/arkey_ie_mods/match_gate)
# OUT gets spy.jsonl, turns.json, gate.json (arkey.match_gate.v1, stamped by stamp.py), gate.txt, ldd.txt, argv.txt, build.txt, logs.
# Exit: hook_gate's (0 every --expect met and the replay agrees with the engine; 1 an --expect missed; 3 the replay disagrees);
# 8 the dry run failed; 9 a build or the stamp failed. The report, not the exit, is the verdict.
D=$(cd "$(dirname "$0")" && pwd); OUT=$1; TREE=$2; MAN=$3; CTX=$4; TURNS=$5; shift 5
ENVS=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do ENVS+=("$1"); shift; done; [ "$1" = -- ] && shift
: "${MATCH_GATE_SESSION:?set MATCH_GATE_SESSION}" "${MATCH_GATE_SERVER:?set MATCH_GATE_SERVER}"
B=${MATCH_GATE_BUILD:-${XDG_CACHE_HOME:-$HOME/.cache}/arkey_ie_mods/match_gate}; SRC=$TREE/ggml/src/ggml-cuda/ext-kernels.cpp
mkdir -p $B; SY=$(mktemp -p $B); { cat $D/syms.txt; nm -D --undefined-only $TREE/build/bin/libggml-cuda.so.1.0.0 | awk '{print $2}' | grep -E '^cu'; } | sed 's/@.*//' | sort -u > $SY
ST=$B/stubs-$(cat $SY $D/cudastub.cpp $D/build_stubs.sh | sha256sum | cut -c1-16)   # the symbols the tree's build imports: a newer tree may import more
[ -f $ST/stublib/libcuda.so.1 ] || "$D/build_stubs.sh" $ST $SY > /dev/null || exit 9; rm -f $SY
G=$B/gates/$(cat $SRC $D/hook_gate.cpp $D/hookspy.cpp $D/gate_build.sh $TREE/ggml/include/ggml.h $(dirname $SRC)/*.cuh | sha256sum | cut -c1-16)-${ST##*-}
[ -x $G/hook_gate ] || { mkdir -p $G && "$D/gate_build.sh" $TREE $G/hook_gate $ST; } || exit 9
mkdir -p $OUT; export DRY_PORT=${DRY_PORT:-$((39600 + RANDOM % 300))}
setsid "$D/dryrun.sh" $OUT $TREE/build/bin $G $ST "GGML_CUDA_EXT_KERNELS=$MAN ${ENVS[*]}" "-c $CTX $MATCH_GATE_SERVER" > $OUT/launch.log 2>&1 & LP=$!
# the dry run is its own process group: a failed drive stops it, a server still queued in memcap --wait included
sleep 1; python3 -I $D/drive_session.py $OUT "$MATCH_GATE_SESSION" $TURNS "${MATCH_GATE_REQUEST:-}" > $OUT/drive.log 2>&1 || { kill -- -$LP 2>/dev/null; echo "dry run failed: $OUT/drive.log"; tail -3 $OUT/drive.log; exit 8; }
wait $LP 2>/dev/null
grep -h "refused\|dropped\|not used" $OUT/server.log | cut -c1-200
$G/hook_gate $OUT/spy.jsonl $MAN --check --steps $OUT/turns.json --json $OUT/gate.json "$@" > $OUT/gate.txt; RC=$?
cat $OUT/gate.txt
python3 -I $D/stamp.py $OUT $SRC || exit 9
exit $RC
