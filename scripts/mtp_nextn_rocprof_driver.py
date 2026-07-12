"""rocprofv3 profiling driver for the native hip mtp_nextn_layer (M3 closure).

Runs the native GPU NextN layer once on the F32 M3 fixture, loading the
prebuilt cached .so (require_cached + HIPENGINE_COMPILER_VERSION_FILE) so the
profiled process does not spawn hipcc/clang.  Intended to be wrapped in:

    rocprofv3 --kernel-trace python3 scripts/mtp_nextn_rocprof_driver.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json"


def main() -> int:
    import hipengine.kernels.hip_gfx1151  # noqa: F401 - registers hip aliases
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        build_mtp_nextn,
        qwen35_gguf_mtp_nextn_layer_logits_f32,
    )
    from hipengine.quant.gguf import GGMLQuantizationType

    # Force the cached build path (no JIT under the profiler).
    build_mtp_nextn(load=True, require_cached=True)

    fixture = json.loads(FIXTURE.read_text())
    inputs = fixture["inputs"]
    kwargs = dict(fixture["kwargs"])

    def Q(name: str) -> GGMLQuantizationType:
        return GGMLQuantizationType[str(inputs[name])]

    logits = qwen35_gguf_mtp_nextn_layer_logits_f32(
        np.ascontiguousarray(inputs["hidden_seed"], dtype=np.float32),
        np.ascontiguousarray(inputs["token_embedding"], dtype=np.float32),
        np.ascontiguousarray(inputs["eh_proj_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["hnorm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["enorm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["attn_norm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["wq_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["wk_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["wv_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["wo_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["q_norm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["k_norm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["attn_post_norm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["router_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["gate_qweight"], dtype=np.float32),
        np.ascontiguousarray(inputs["up_qweight"], dtype=np.float32),
        np.ascontiguousarray(inputs["down_qweight"], dtype=np.float32),
        Q("gate_qtype"), Q("up_qtype"), Q("down_qtype"),
        np.ascontiguousarray(inputs["shared_gate_logit_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["shared_gate_qweight"], dtype=np.float32),
        np.ascontiguousarray(inputs["shared_up_qweight"], dtype=np.float32),
        np.ascontiguousarray(inputs["shared_down_qweight"], dtype=np.float32),
        Q("shared_qtype"),
        np.ascontiguousarray(inputs["shared_head_norm_weight"], dtype=np.float32),
        np.ascontiguousarray(inputs["shared_head_weight"], dtype=np.float32),
        **kwargs,
    )
    print("mtp_nextn_layer logits:", np.asarray(logits).ravel().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
