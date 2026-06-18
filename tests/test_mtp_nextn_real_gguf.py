"""End-to-end test: load real GGUF MTP weights and run the native GPU mtp_nextn_layer.

M6: Validates the full end-to-end path with real model weights from
/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (blk.40 NextN tensors).
Gates GPU output vs cpu_reference on the same real weights.

Note: shared_head_head (output.weight) is Q6_K [248320, 2048] — Q6_K dispatch
is not yet implemented, so we use a small F32 stub for the shared head in this
test. The eh_proj + attention + FFN path is fully validated with real weights.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

GGUF_PATH = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _gguf_available() -> bool:
    return Path(GGUF_PATH).exists()


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _gguf_available(),
    reason="ROCm/HIP or GGUF model file not available",
)

import hipengine.kernels.cpu_reference  # noqa: F401,E402
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402


def _load_real_weights():
    """Load all blk.40 NextN tensors from the real GGUF file."""
    r = GGUFReader(GGUF_PATH)
    weights = {}
    for t in r.info.tensors:
        if "blk.40" in t.name:
            data = r.tensor_data(t.name)
            weights[t.name] = (data, t.ggml_type, t.shape)
    return weights, r


@pytest.fixture(scope="module")
def real_inputs():
    weights, reader = _load_real_weights()

    def get(name):
        return weights[name][0]

    def qtype(name):
        return GGMLQuantizationType(weights[name][1])

    # Model config from GGUF metadata
    hidden = 2048
    heads = 16  # qwen35moe.attention.head_count
    kv_heads = 2  # qwen35moe.attention.head_count_kv
    qk_head_dim = 256  # from attn_q_norm shape
    inter = 512  # qwen35moe.expert_feed_forward_length
    experts = 256  # qwen35moe.expert_count
    top_k = 8  # qwen35moe.expert_used_count

    # Use 1 token with random F32 input
    np.random.seed(42)
    hidden_seed = np.random.randn(1, hidden).astype(np.float32) * 0.1
    token_embed = np.random.randn(1, hidden).astype(np.float32) * 0.1

    # Shared head: output.weight is Q6_K [248320, 2048] — not yet supported.
    # Use a small F32 stub [16, 2048] for the shared head.
    vocab_stub = 16
    shared_head_w = np.random.randn(vocab_stub, hidden).astype(np.float32) * 0.01

    # The cpu_reference expects already-dequanted F32 weights for eh_proj + attention.
    # The GPU kernel accepts raw Q8_0 block bytes with qtype kwargs for dispatch.
    # So we prepare two sets of weights: dequanted F32 for cpu_reference, raw for GPU.
    from hipengine.quant.gguf import dequantize_gguf_data

    def dequant(name):
        w, qt, shape = weights[name]
        # tensor_data already returns uint8 with correct byte_shape
        return dequantize_gguf_data(w, GGMLQuantizationType(qt)).astype(np.float32)

    # Dequanted F32 for cpu_reference (eh_proj + attention only)
    cpu_eh_proj = dequant("blk.40.nextn.eh_proj.weight")
    cpu_wq = dequant("blk.40.attn_q.weight")
    cpu_wk = dequant("blk.40.attn_k.weight")
    cpu_wv = dequant("blk.40.attn_v.weight")
    cpu_wo = dequant("blk.40.attn_output.weight")
    # Router is BF16, dequant to F32
    cpu_router = np.asarray(get("blk.40.ffn_gate_inp.weight"), dtype=np.float32)

    return {
        "hidden_seed": hidden_seed,
        "token_embed": token_embed,
        "eh_proj_weight": get("blk.40.nextn.eh_proj.weight"),
        "eh_proj_qtype": qtype("blk.40.nextn.eh_proj.weight"),
        "hnorm_weight": get("blk.40.nextn.hnorm.weight"),
        "enorm_weight": get("blk.40.nextn.enorm.weight"),
        "attn_norm_weight": get("blk.40.attn_norm.weight"),
        "wq_weight": get("blk.40.attn_q.weight"),
        "wk_weight": get("blk.40.attn_k.weight"),
        "wv_weight": get("blk.40.attn_v.weight"),
        "wo_weight": get("blk.40.attn_output.weight"),
        "q_norm_weight": get("blk.40.attn_q_norm.weight"),
        "k_norm_weight": get("blk.40.attn_k_norm.weight"),
        "wq_qtype": qtype("blk.40.attn_q.weight"),
        "wk_qtype": qtype("blk.40.attn_k.weight"),
        "wv_qtype": qtype("blk.40.attn_v.weight"),
        "wo_qtype": qtype("blk.40.attn_output.weight"),
        "post_norm_weight": get("blk.40.post_attention_norm.weight"),
        "router_weight": get("blk.40.ffn_gate_inp.weight"),
        "gate_qweight": get("blk.40.ffn_gate_exps.weight"),
        "up_qweight": get("blk.40.ffn_up_exps.weight"),
        "down_qweight": get("blk.40.ffn_down_exps.weight"),
        "gate_qtype": qtype("blk.40.ffn_gate_exps.weight"),
        "up_qtype": qtype("blk.40.ffn_up_exps.weight"),
        "down_qtype": qtype("blk.40.ffn_down_exps.weight"),
        "shared_gate_logit_weight": get("blk.40.ffn_gate_inp_shexp.weight"),
        "shared_gate_qweight": get("blk.40.ffn_gate_shexp.weight"),
        "shared_up_qweight": get("blk.40.ffn_up_shexp.weight"),
        "shared_down_qweight": get("blk.40.ffn_down_shexp.weight"),
        "shared_qtype": qtype("blk.40.ffn_gate_shexp.weight"),
        "shared_head_norm_weight": get("blk.40.nextn.shared_head_norm.weight"),
        "shared_head_weight": shared_head_w,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "experts_used": top_k,
        "eps": 1e-6,
        # Dequanted F32 weights for cpu_reference (eh_proj + attention + shared expert)
        "cpu_eh_proj": cpu_eh_proj,
        "cpu_wq": cpu_wq,
        "cpu_wk": cpu_wk,
        "cpu_wv": cpu_wv,
        "cpu_wo": cpu_wo,
        "cpu_router": cpu_router,
    }


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_real_gguf_nextn_layer_matches_cpu_reference(backend, real_inputs):
    """Run the composite mtp_nextn_layer with real GGUF weights and gate vs cpu_reference."""
    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_nextn_layer_logits as cpu_oracle,
    )
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_nextn_layer_logits_f32 as gpu_kernel,
    )
    from hipengine.kernels.registry import resolve

    # Verify the GPU kernel is the native one, not cpu_reference fallback
    resolved = resolve(backend=backend, layer="mtp_nextn_layer", quant="w4_gguf",
                       variant="qwen35_dense_logits")
    assert resolved is not cpu_oracle, f"{backend} returned cpu_reference via fallback"

    # cpu_reference infers qtypes from raw block bytes; no qtype kwargs needed
    cpu_kwargs = dict(
        num_heads=real_inputs["num_heads"],
        num_kv_heads=real_inputs["num_kv_heads"],
        experts_used=real_inputs["experts_used"],
        eps=real_inputs["eps"],
    )

    # GPU kernel needs explicit qtype kwargs for dispatch
    gpu_kwargs = dict(
        num_heads=real_inputs["num_heads"],
        num_kv_heads=real_inputs["num_kv_heads"],
        experts_used=real_inputs["experts_used"],
        eh_proj_qtype=real_inputs["eh_proj_qtype"],
        wq_qtype=real_inputs["wq_qtype"], wk_qtype=real_inputs["wk_qtype"],
        wv_qtype=real_inputs["wv_qtype"], wo_qtype=real_inputs["wo_qtype"],
        eps=real_inputs["eps"],
    )

    # cpu_reference expects dequanted F32 weights for eh_proj + attention + shared expert
    cpu_args = [
        real_inputs["hidden_seed"], real_inputs["token_embed"],
        real_inputs["cpu_eh_proj"], real_inputs["hnorm_weight"],
        real_inputs["enorm_weight"], real_inputs["attn_norm_weight"],
        real_inputs["cpu_wq"], real_inputs["cpu_wk"],
        real_inputs["cpu_wv"], real_inputs["cpu_wo"],
        real_inputs["q_norm_weight"], real_inputs["k_norm_weight"],
        real_inputs["post_norm_weight"], real_inputs["cpu_router"],
        real_inputs["gate_qweight"], real_inputs["up_qweight"],
        real_inputs["down_qweight"],
        real_inputs["gate_qtype"], real_inputs["up_qtype"], real_inputs["down_qtype"],
        real_inputs["shared_gate_logit_weight"],
        real_inputs["shared_gate_qweight"], real_inputs["shared_up_qweight"],
        real_inputs["shared_down_qweight"], real_inputs["shared_qtype"],
        real_inputs["shared_head_norm_weight"], real_inputs["shared_head_weight"],
    ]

    # GPU kernel accepts raw Q8_0 block bytes with qtype kwargs
    gpu_args = [
        real_inputs["hidden_seed"], real_inputs["token_embed"],
        real_inputs["eh_proj_weight"], real_inputs["hnorm_weight"],
        real_inputs["enorm_weight"], real_inputs["attn_norm_weight"],
        real_inputs["wq_weight"], real_inputs["wk_weight"],
        real_inputs["wv_weight"], real_inputs["wo_weight"],
        real_inputs["q_norm_weight"], real_inputs["k_norm_weight"],
        real_inputs["post_norm_weight"], real_inputs["router_weight"],
        real_inputs["gate_qweight"], real_inputs["up_qweight"],
        real_inputs["down_qweight"],
        real_inputs["gate_qtype"], real_inputs["up_qtype"], real_inputs["down_qtype"],
        real_inputs["shared_gate_logit_weight"],
        real_inputs["shared_gate_qweight"], real_inputs["shared_up_qweight"],
        real_inputs["shared_down_qweight"], real_inputs["shared_qtype"],
        real_inputs["shared_head_norm_weight"], real_inputs["shared_head_weight"],
    ]

    # Run cpu_reference (the oracle) with dequanted F32 weights
    expected = cpu_oracle(*cpu_args, **cpu_kwargs)

    # Run native GPU kernel with raw Q8_0 weights
    actual = gpu_kernel(*gpu_args, **gpu_kwargs)
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    # K-quant dequant introduces small numerical differences vs cpu_reference
    assert max_abs < 1e-2, (
        f"{backend} real-GGUF NextN layer max_abs={max_abs} exceeds 1e-2 vs cpu_reference"
    )