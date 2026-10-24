#!/usr/bin/env bash
set -euo pipefail
PY="$HOME/miniforge3/envs/therock/bin/python"
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
RUN=/tmp/hipengine-paro-refresh-20261023
mkdir -p "$RUN"
if [[ ! -s "$RUN/hipcc-version.txt" ]]; then
  "$ROOT/bin/hipcc" --version > "$RUN/hipcc-version.txt"
fi
exec env -i HOME="$HOME" USER="${USER:-lhl}" LOGNAME="${LOGNAME:-lhl}" SHELL=/bin/bash TERM=xterm \
  PATH="$(dirname "$PY"):$ROOT/bin:$ROOT/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" HIP_INCLUDE_PATH="$ROOT/include" \
  HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" HIPENGINE_HIP_ARCH=gfx1151 \
  GPU_MAX_HW_QUEUES=2 HIPENGINE_COMPILER_VERSION_FILE="$RUN/hipcc-version.txt" \
  PYTHONPATH=/tmp/hipengine-paro-gfx1151-20261023 PYTHONUNBUFFERED=1 \
  "$@"
