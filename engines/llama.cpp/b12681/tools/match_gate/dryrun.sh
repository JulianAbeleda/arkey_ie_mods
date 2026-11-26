#!/bin/bash
# dryrun.sh OUT BIN GATE_DIR STUBS "SWITCHES" "SERVER FLAGS": start an UNMODIFIED engine build (BIN = its build/bin) on the CPU with
# the do-nothing CUDA runtime (STUBS/stublib) and the hook recorder (GATE_DIR/libhookspy.so). No GPU: the real CUDA
# libraries are never loaded (LD_LIBRARY_PATH puts the stubs first, checked with ldd below; CUDA_VISIBLE_DEVICES is empty). CUDA
# graphs are off so the backend's node loop (and the hook) runs at every step, as in the hook's check mode. The server listens
# on DRY_PORT; the driver sends the turns and stops it by the PID in OUT/server.pid. Writes OUT/ldd.txt, build.txt, argv.txt, server.log.
OUT=$1; BIN=$2; G=$3; B=$4; SW=$5; FLAGS=$6
mkdir -p $OUT; rm -f $OUT/spy.jsonl $OUT/launches.txt
export LD_LIBRARY_PATH=$B/stublib:$BIN LD_PRELOAD=$G/libhookspy.so CUDA_VISIBLE_DEVICES= HOOKSPY_OUT=$OUT/spy.jsonl HOOKSPY_NO_PREFETCH=1 STUB_COPY_MAX=${STUB_COPY_MAX:-65536} STUB_LAUNCH_REPORT=$OUT/launches.txt
ldd $BIN/libggml-cuda.so.1.0.0 | grep -i "cuda\|cublas" > $OUT/ldd.txt
grep -v "$B/stublib" $OUT/ldd.txt | grep -q . && { echo "a real CUDA library would load: see $OUT/ldd.txt"; exit 9; }
sha256sum $BIN/llama-server $BIN/libggml-cuda.so.1.0.0 $BIN/libllama.so.1.0.0 | cut -c1-80 > $OUT/build.txt
echo "$SW $BIN/llama-server $FLAGS" > $OUT/argv.txt
CAP=(); command -v memcap > /dev/null && CAP=(memcap --max ${MATCH_GATE_MEM:-12G} --wait --)
"${CAP[@]}" env GGML_CUDA_DISABLE_GRAPHS=1 bash -c "echo \$\$ > $OUT/server.pid; exec nice -n 10 env $SW $BIN/llama-server $FLAGS --host 127.0.0.1 --port ${DRY_PORT:-39517}" > $OUT/server.log 2>&1
