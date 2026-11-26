#!/bin/bash
# gate_build.sh TREE OUT STUBS: build hook_gate (OUT) against TREE's own matcher (TREE/ggml/src/ggml-cuda/ext-kernels.cpp) and its
# ggml (TREE/build/bin/libggml-base.so), and the recorder (libhookspy.so beside OUT) against TREE's ggml headers. The matcher
# source is copied with a MECHANICAL change inside ext_match only, line numbers kept: every `return {};` records its line (and
# the loop indices k / s / o), and the refusal site records the two byte ranges. Nothing else differs from the engine's file;
# the sed fails the build if a pattern is not found. STUBS holds the stub runtime (build_stubs.sh).
set -e; D=$(cd "$(dirname "$0")" && pwd); T=$1; OUT=$2; B=$3; SRC=$T/ggml/src/ggml-cuda/ext-kernels.cpp; CU=${CUDA_INCLUDE:-/usr/local/cuda/include}
A=$(grep -n '^std::vector<ggml_tensor \*> ext_match(' $SRC | cut -d: -f1); Z=$(awk -v a=$A 'NR>a && /^}/{print NR; exit}' $SRC)
[ -n "$A" ] && [ -n "$Z" ] || { echo "ext_match not found in $SRC"; exit 9; }
mkdir -p $OUT.d
RANGES=$(grep -q 'v0 = (const char \*) v->data' $SRC && echo 'a0, a1, b0, b1, v0, v0 + ggml_nbytes(v)' || echo 'a0, a1, b0, b1, b0, b1')
sed -e "$A,$Z s/return {};/return ext_gate_reject(__LINE__);/" \
    -e "$A,$Z s/^\(    for (int k = 0; k < n; ++k) {\)/\1 ext_gate_k = k; ext_gate_s = -1;/" \
    -e "$A,$Z s/^\(        for (int k = 0; k < n; ++k) {\)/\1 ext_gate_k = k; ext_gate_s = -1;/" \
    -e "$A,$Z s/^\( *for (size_t s = 0; s < [^;]*; ++s) {\)/\1 ext_gate_s = (int) s;/" \
    -e "$A,$Z s/^\(    for (int o : e.outputs) {\)/\1 ext_gate_o = o;/" \
    -e "$A,$Z s/^\( *if (!exact) {\)/\1 ext_gate_refuse(o, k, (int) s, $RANGES);/" $SRC > $OUT.d/ext-kernels.gate.cpp
for pat in 'ext_gate_reject(__LINE__)' 'ext_gate_k = k' 'ext_gate_s = (int) s' 'ext_gate_o = o' 'ext_gate_refuse('; do grep -q "$pat" $OUT.d/ext-kernels.gate.cpp || { echo "pattern not applied: $pat"; exit 8; }; done
[ $(wc -l < $SRC) = $(wc -l < $OUT.d/ext-kernels.gate.cpp) ] || { echo "line count changed"; exit 7; }
diff $SRC $OUT.d/ext-kernels.gate.cpp > $OUT.d/gate.diff || true
gcc -O1 -I$T/vendor/hash -c $T/vendor/hash/sha256/sha256.c -o $OUT.d/sha256.o
g++ -O1 -std=c++17 -w -DGATE_SRC="\"$SRC\"" $D/hook_gate.cpp -I$OUT.d -I$(dirname $SRC) -I$T/ggml/include -I$T/ggml/src -I$T/vendor -I$T/vendor/hash -I$CU \
    $OUT.d/sha256.o -L$T/build/bin -lggml-base -L$B/stublib -l:libcuda.so.1 -Wl,-rpath,$T/build/bin -Wl,-rpath,$B/stublib -o $OUT
g++ -O1 -std=c++17 -w -fPIC -shared $D/hookspy.cpp -I$T/ggml/include -I$T/ggml/src -I$T/vendor -o $(dirname $OUT)/libhookspy.so -ldl
echo "built $OUT and $(dirname $OUT)/libhookspy.so ($(grep -c '^[<>]' $OUT.d/gate.diff) changed lines in ext_match, lines $A-$Z of $SRC)"
