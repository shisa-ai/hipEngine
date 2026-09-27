#!/usr/bin/env python3
"""Resolve the QSA grid/head_count observation by logging the real launch args."""

from __future__ import annotations

import json
import os
import sys
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
from hipengine.kernels.registry import _KERNELS, KernelKey

HEADS_BY_SYMBOL = {
    "strict_h256_head_quad_rows_spans": 4,
    "strict_h256_head_pair_rows_spans": 2,
    "strict_h256_page256_wave_rows_spans": 1,
    "strict_h256_wave_rows_spans": 1,
}


def main() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()

    calls: list[dict] = []
    originals: dict = {}
    for variant, heads in HEADS_BY_SYMBOL.items():
        key = KernelKey("hip_gfx1100", "qsa_sparse_attention", "bf16_kv", variant)
        fn = _KERNELS.get(key)
        if fn is None:
            continue
        originals[key] = fn

        def make(fn, variant=variant, heads=heads):
            def inner(*args, **kwargs):
                qh = kwargs.get("query_heads")
                calls.append({
                    "variant": variant, "heads": heads,
                    "rows": kwargs.get("rows"),
                    "query_heads": qh,
                    "kv_heads": kwargs.get("kv_heads"),
                    "head_dim": kwargs.get("head_dim"),
                    "block_size": kwargs.get("block_size"),
                    "selected_stride": kwargs.get("selected_stride"),
                    "grid_x": (qh // heads) if qh else None,
                })
                return fn(*args, **kwargs)
            return inner
        _KERNELS[key] = make(fn)

    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    index = load_gguf_index(discover_gguf_files(MODEL_ROOT)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=MODEL_ROOT, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4351,
        prefill_chunk_size=1024))

    try:
        runner = generator.runner
        runner.reset()
        runner.prefill(list(range(4000, 4000 + 1024)))
    finally:
        for key, fn in originals.items():
            _KERNELS[key] = fn
        generator.close()

    seen: dict[str, dict] = {}
    for c in calls:
        k = json.dumps(c, sort_keys=True)
        seen.setdefault(k, {"n": 0, **c})
        seen[k]["n"] += 1
    out = {
        "env": {k: os.environ.get(k) for k in (
            "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL",
            "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR",
            "HIPENGINE_QWEN4_EXP_QSA_WAVE32")},
        "registered_qsa_variants": sorted(
            k.variant for k in _KERNELS
            if k.layer == "qsa_sparse_attention" and k.backend == "hip_gfx1100"),
        "calls": list(seen.values()),
    }
    print("RESULT " + json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
