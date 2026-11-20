#!/bin/bash
set -euo pipefail
ENV_PREFIX="$HOME/miniforge3/envs/therock"
PY="$ENV_PREFIX/bin/python"
SITE="$ENV_PREFIX/lib/python3.12/site-packages"
ROOT="$("$PY" -m rocm_sdk path --root)"
export PATH="$ENV_PREFIX/bin:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" HIP_INCLUDE_PATH="$ROOT/include"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$@"
