"""RED/GREEN test for native hip ``mtp_nextn_ffn`` + ``mtp_nextn_moe_routing``.

M3 deliverable: real GPU NextN MoE FFN sublayer registered under
``KernelKey(backend, "mtp_nextn_ffn", "gguf_f32", "qwen35")`` and the MoE router
under ``KernelKey(backend, "mtp_nextn_moe_routing", "gguf_f32", "qwen35")``.
Without these the registry falls back to the cpu_reference numpy oracle.

Scope (M3, correctness-first): the F32 fixture uses F32 qtype everywhere, so all
quant-gemv = plain F32 matmul. The expert loop (top_k=1, 2 experts) and shared
expert run on GPU via the mtp_linear_f32 + new silu_mul/mul/row_scale kernels.
Router softmax/top-k runs on host (tiny [tokens, experts]) for correctness-first;
M6 GPU-accelerates it. K-quant (Q4_K/Q5_K/Q8_0) expert paths are M6 work.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

import hipengine.kernels.cpu_reference  # noqa: F401,E402
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402


def _f32(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def _exact_key_registered(backend: str, layer: str, variant: str = "qwen35") -> bool:
    from hipengine.kernels.registry import KernelKey, registered_keys

    target = KernelKey(backend, layer, "gguf_f32", variant)
    return target in registered_keys()


@pytest.fixture(scope="module")
def ffn_inputs() -> dict:
    fixture = json.loads(FIXTURE.read_text())
    inputs = fixture["inputs"]
    # The FFN sublayer takes the attention output as ``hidden``.  Recompute it
    # from the cpu_reference attention to isolate the FFN gate.
    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_attention_sublayer as attn,
        qwen35_gguf_mtp_eh_proj as eh,
    )
    projected = eh(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        eps=float(fixture["kwargs"].get("eps", 1e-6)),
    )
    attn_out = attn(
        projected,
        _f32(inputs["attn_norm_weight"]),
        _f32(inputs["wq_weight"]),
        _f32(inputs["wk_weight"]),
        _f32(inputs["wv_weight"]),
        _f32(inputs["wo_weight"]),
        _f32(inputs["q_norm_weight"]),
        _f32(inputs["k_norm_weight"]),
        num_heads=int(fixture["kwargs"]["num_heads"]),
        num_kv_heads=int(fixture["kwargs"]["num_kv_heads"]),
        eps=float(fixture["kwargs"].get("eps", 1e-6)),
    )
    from hipengine.quant.gguf import GGMLQuantizationType

    return {
        "hidden": np.ascontiguousarray(attn_out, dtype=np.float32),
        "attn_post_norm_weight": _f32(inputs["attn_post_norm_weight"]),
        "router_weight": _f32(inputs["router_weight"]),
        "gate_qweight": _f32(inputs["gate_qweight"]),
        "up_qweight": _f32(inputs["up_qweight"]),
        "down_qweight": _f32(inputs["down_qweight"]),
        "gate_qtype": GGMLQuantizationType[str(inputs["gate_qtype"])],
        "up_qtype": GGMLQuantizationType[str(inputs["up_qtype"])],
        "down_qtype": GGMLQuantizationType[str(inputs["down_qtype"])],
        "shared_gate_logit_weight": _f32(inputs["shared_gate_logit_weight"]),
        "shared_gate_qweight": _f32(inputs["shared_gate_qweight"]),
        "shared_up_qweight": _f32(inputs["shared_up_qweight"]),
        "shared_down_qweight": _f32(inputs["shared_down_qweight"]),
        "shared_qtype": GGMLQuantizationType[str(inputs["shared_qtype"])],
        "experts_used": int(fixture["kwargs"]["experts_used"]),
        "eps": float(fixture["kwargs"].get("eps", 1e-6)),
    }


def test_hip_moe_routing_key_registered():
    assert _exact_key_registered("hip_gfx1100", "mtp_nextn_moe_routing"), (
        "no hip_gfx1100 mtp_nextn_moe_routing kernel registered"
    )
    assert _exact_key_registered("hip_gfx1151", "mtp_nextn_moe_routing"), (
        "no hip_gfx1151 mtp_nextn_moe_routing alias registered"
    )


def test_hip_ffn_key_registered():
    assert _exact_key_registered("hip_gfx1100", "mtp_nextn_ffn"), (
        "no hip_gfx1100 mtp_nextn_ffn kernel registered"
    )
    assert _exact_key_registered("hip_gfx1151", "mtp_nextn_ffn"), (
        "no hip_gfx1151 mtp_nextn_ffn alias registered"
    )


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_ffn_matches_cpu_reference(backend, ffn_inputs):
    if not _exact_key_registered(backend, "mtp_nextn_ffn"):
        pytest.skip(f"{backend} mtp_nextn_ffn not registered yet (RED)")

    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_ffn_sublayer as cpu_oracle,
    )
    from hipengine.kernels.registry import resolve

    expected = cpu_oracle(
        ffn_inputs["hidden"],
        ffn_inputs["attn_post_norm_weight"],
        ffn_inputs["router_weight"],
        ffn_inputs["gate_qweight"],
        ffn_inputs["up_qweight"],
        ffn_inputs["down_qweight"],
        ffn_inputs["gate_qtype"],
        ffn_inputs["up_qtype"],
        ffn_inputs["down_qtype"],
        ffn_inputs["shared_gate_logit_weight"],
        ffn_inputs["shared_gate_qweight"],
        ffn_inputs["shared_up_qweight"],
        ffn_inputs["shared_down_qweight"],
        ffn_inputs["shared_qtype"],
        experts_used=ffn_inputs["experts_used"],
        eps=ffn_inputs["eps"],
    )

    kernel = resolve(backend=backend, layer="mtp_nextn_ffn", quant="gguf_f32", variant="qwen35")
    assert kernel is not cpu_oracle, (
        f"resolve({backend}) returned cpu_reference via fallback"
    )

    actual = kernel(
        ffn_inputs["hidden"],
        ffn_inputs["attn_post_norm_weight"],
        ffn_inputs["router_weight"],
        ffn_inputs["gate_qweight"],
        ffn_inputs["up_qweight"],
        ffn_inputs["down_qweight"],
        ffn_inputs["gate_qtype"],
        ffn_inputs["up_qtype"],
        ffn_inputs["down_qtype"],
        ffn_inputs["shared_gate_logit_weight"],
        ffn_inputs["shared_gate_qweight"],
        ffn_inputs["shared_up_qweight"],
        ffn_inputs["shared_down_qweight"],
        ffn_inputs["shared_qtype"],
        experts_used=ffn_inputs["experts_used"],
        eps=ffn_inputs["eps"],
    )
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs < 1e-3, (
        f"{backend} mtp_nextn_ffn max_abs={max_abs} exceeds 1e-3 vs cpu_reference"
    )
