#!/usr/bin/env python3
"""Real expert population vs the balanced model, on a code-p4096 prefill.

The wmma tile map pads every expert to a multiple of `tile_rows` (16), so the
gate/up kernel's weight reads scale with sum(ceil(count_e/16)) tiles rather than
with the true assignment count. This measures the real histogram and the
resulting amplification, layer by layer.
"""

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
TILE_ROWS = 16

import numpy as np

from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.models import resolve_model
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)
from hipengine.core.memory import copy_device_to_host, host_array_ptr


def stats(counts: np.ndarray, tile_rows: int) -> dict:
    counts = counts[counts > 0] if (counts > 0).any() else counts
    tiles = int(np.sum(np.ceil(counts / tile_rows)))
    assignments = int(counts.sum())
    balanced_tiles = int(np.ceil(assignments / len(counts) / tile_rows)) * len(counts)
    return {
        "experts_used": int(len(counts)),
        "assignments": assignments,
        "min": int(counts.min()), "median": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
        "max_over_median": round(float(counts.max() / max(np.median(counts), 1)), 2),
        "tiles_real": tiles,
        "tiles_balanced_model": balanced_tiles,
        "padded_rows_real": tiles * tile_rows,
        "padding_waste_pct": round(100 * (1 - assignments / (tiles * tile_rows)), 1),
        "amplification_vs_ideal": round(tiles * tile_rows / assignments, 3),
    }


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

    import hipengine.runtime.qwen4_exp_runner as qr

    rows = []
    original = qr.run_qwen4_exp_moe
    try:
        runner = generator.runner
        runner.reset()
        runner.prefill(list(range(4000, 4000 + 1024)))  # warm, builds scratch
        scratch = None
        for name in ("qsa_prefill_scratch", "gdn_prefill_scratch"):
            obj = getattr(runner, name, None)
            if obj is not None and getattr(obj, "moe", None) is not None:
                scratch = obj.moe
                break
        if scratch is None:
            raise RuntimeError("no MoE scratch found")
        experts = scratch.group_expert_start.nbytes // 8 - 1
        host = np.zeros(experts + 1, dtype=np.int64)
        total_host = np.zeros(1, dtype=np.int64)
        print(f"experts={experts} tile_capacity="
              f"{scratch.group_tile_expert.nbytes // 8}", file=sys.stderr)

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            sc = kwargs.get("scratch")
            if sc is not None:
                copy_device_to_host(host_array_ptr(host), sc.group_expert_start,
                                    (experts + 1) * 8)
                copy_device_to_host(host_array_ptr(total_host),
                                    sc.group_wmma_total, 8)
                c = np.diff(host).astype(np.int64)
                st = stats(c.astype(np.float64), TILE_ROWS)
                st["wmma_total_rows_engine"] = int(total_host[0])
                st["wmma_total_rows_model"] = st["tiles_real"] * TILE_ROWS
                st["counts"] = [int(v) for v in c]
                rows.append(st)
            return out

        qr.run_qwen4_exp_moe = wrapped
        runner.reset()
        runner.prefill(list(range(4000, 4000 + 1024)))
    finally:
        qr.run_qwen4_exp_moe = original
        generator.close()

    for r in rows:
        r["tile_sweep"] = {
            f"tile{tr}": int(np.sum(np.ceil(np.asarray(r["counts"]) / tr)))
            for tr in (1, 2, 4, 8, 16, 32, 64)}
    out = Path("/tmp/qwen4exp-scope/expert-hist.json")
    out.write_text(json.dumps(rows, indent=1))
    summary = [{k: v for k, v in r.items() if k != "counts"} for r in rows]
    print("RESULT " + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
