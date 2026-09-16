#!/usr/bin/env bash
# Builds the identical-operand replay adapter against a pinned comparator build.
#
# The adapter links the comparator's libggml-hip.so and calls its exported
# ggml_cuda_mul_mat_mmb entry point. It also includes the comparator's own
# common.cuh, because ggml creates its streams with cudaStreamNonBlocking and
# only the real backend context exposes the stream that events must be recorded
# on. Both facts mean the shim has to be compiled with exactly the flags that
# built the comparator, not with a hand-written approximation of them.
#
# So the shim's compile command is derived from the pinned build's own
# compile_commands.json: the entry for mmb.cu is taken verbatim, its object-file
# outputs are stripped, and the shim source and link flags are substituted. A
# comparator rebuilt with different defines or a different offload arch
# therefore produces a shim built the same way.
#
#   build_shim.sh <comparator-build-dir> [output.so]
#
# Example:
#   tools/replay_bridge/build_shim.sh \
#     /home/lhl/comparators-20260915/build-halobox-pr63 \
#     /tmp/replay-bridge/libmmb_replay.so

set -euo pipefail

BUILD_DIR="${1:?usage: build_shim.sh <comparator-build-dir> [output.so]}"
OUT="${2:-/tmp/replay-bridge/libmmb_replay.so}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mmb_replay_shim.cpp"

CC_JSON="$BUILD_DIR/compile_commands.json"
if [[ ! -f "$CC_JSON" ]]; then
    echo "no compile_commands.json in $BUILD_DIR" >&2
    exit 1
fi

LIB="$BUILD_DIR/bin"
for p in "$LIB/libggml-hip.so" "$LIB/libggml-base.so"; do
    [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")"

# Derive the shim compile command from the pinned build's mmb.cu entry.
CMD_FILE="$(mktemp)"
trap 'rm -f "$CMD_FILE"' EXIT
python3 - "$CC_JSON" "$SRC" "$OUT" "$LIB" >"$CMD_FILE" <<'PY'
import json, shlex, sys

cc_json, src, out, lib = sys.argv[1:5]
entries = json.load(open(cc_json))

import os

for e in entries:
    if e["file"].endswith("mmb.cu"):
        argv = shlex.split(e["command"])
        src_tu = e["file"]
        break
else:
    sys.exit("no mmb.cu entry in compile_commands.json")

kept, i = [], 0
while i < len(argv):
    a = argv[i]
    # The pinned TU itself is replaced by the shim source below.
    if os.path.abspath(a) == os.path.abspath(src_tu):
        i += 1
        continue
    # Drop compile-to-object plumbing; keep every flag that affects codegen.
    if a in ("-c", "-MD", "-MMD", "-MT", "-MF", "-MQ", "-o"):
        i += 2 if a in ("-MT", "-MF", "-MQ", "-o") else 1
        continue
    if a.startswith(("-MT", "-MF", "-MQ", "-o=")):
        i += 1
        continue
    kept.append(a)
    i += 1

# mmb.cu resolves "common.cuh" relative to its own directory, which the
# compile_commands entry does not spell out as an -I.
kept += [f"-I{os.path.dirname(os.path.abspath(src_tu))}"]
kept += ["-fPIC", "-shared", src,
         f"-L{lib}", "-lggml-hip", "-lggml-base",
         f"-Wl,-rpath,{lib}", "-o", out]
print(" ".join(shlex.quote(x) for x in kept))
PY

echo "build dir : $BUILD_DIR"
echo "libs      : $LIB"
echo "flags     : $(sed -E 's/ -[DI][^ ]*//g' "$CMD_FILE")"
echo

bash "$CMD_FILE"

echo
echo "built: $OUT"
nm -DC "$OUT" | grep -E "he_replay_mmb_(available|run|weight_bytes|min_rotate)" || {
    echo "shim does not export the expected C ABI" >&2
    exit 1
}
