#!/usr/bin/env python3
"""Which Q8_0 dense route claims which layer under the production default?"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = "/tmp/hipengine-hipcc-version.txt"
os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL")

from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.models import resolve_model
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)
import hipengine.runtime.gguf_linear as gl


def layer_of(dispatch):
    return gl._weight_layer_index(getattr(dispatch, "weight", None))


def main() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    index = load_gguf_index(discover_gguf_files(MODEL_ROOT)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=MODEL_ROOT, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4351,
        prefill_chunk_size=1024))

    # (route, layer, variant) -> count
    seen: dict[tuple[str, int | None, str], int] = defaultdict(int)
    claims: dict[str, dict] = {}

    def wrap(name, fn, record_claim):
        def inner(dispatch, **kwargs):
            before = dispatch.key.variant
            out = fn(dispatch, **kwargs)
            key = (name, None, f"{before} -> {out.key.variant}")
            seen[key] += 1
            if record_claim:
                claims.setdefault(before, {})[name] = out.key.variant
            return out
        return inner

    gl._wmma_prefill_dispatch = wrap(
        "wmma", gl._wmma_prefill_dispatch, True)
    gl._q8_iu8_wmma_dispatch = wrap(
        "iu8", gl._q8_iu8_wmma_dispatch, False)
    gl._q8_dense_wide_dispatch = wrap(
        "wide", gl._q8_dense_wide_dispatch, False)

    # Layer attribution needs the weight; capture it from the call site by
    # wrapping _weight_layer_index is unreliable, so record variants only and
    # attribute layers through the dispatch key's layer field.
    layers: dict[tuple[str, str], set] = defaultdict(set)

    try:
        r = generator.runner
        r.reset()
        r.prefill(list(range(1000, 2024)))
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = f"{type(exc).__name__}: {exc}"
    finally:
        generator.close()

    out = {
        "prefill_ok": ok,
        "env": {k: os.environ.get(k) for k in (
            "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS",
            "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE",
            "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS",
            "HIPENGINE_QWEN4_EXP_Q8_IU8_WMM",
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL")},
        "transitions": {f"{k[0]}: {k[2]}": v for k, v in sorted(
            seen.items(), key=lambda kv: -kv[1])[:25]},
    }
    print("RESULT " + json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
