"""RED test for M5: end-to-end GGUF MTP speculative draft from AR generation.

M5: Runs a few steps of AR generation using the GGUF resident session, captures
the fp32 post-output_norm hidden seed after each step, runs the composite
mtp_nextn_layer to produce draft logits, and gates the draft tokens against
cpu_reference on the same weights and hidden seed.
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
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data  # noqa: E402


def _load_blk40_weights():
    """Load all blk.40 NextN tensors + output.weight from the real GGUF file."""
    r = GGUFReader(GGUF_PATH)
    weights = {}
    for t in r.info.tensors:
        if "blk.40" in t.name or t.name == "output.weight":
            data = r.tensor_data(t.name)
            weights[t.name] = (data, t.ggml_type, t.shape)
    return weights, r


@pytest.fixture(scope="module")
def mtp_weights():
    weights, reader = _load_blk40_weights()

    def get(name): return weights[name][0]
    def qt(name): return GGMLQuantizationType(weights[name][1])
    def dq(name): return dequantize_gguf_data(get(name), qt(name)).astype(np.float32)

    return {
        # Raw K-quant weights for GPU
        "eh_proj_weight": get("blk.40.nextn.eh_proj.weight"),
        "hnorm_weight": get("blk.40.nextn.hnorm.weight"),
        "enorm_weight": get("blk.40.nextn.enorm.weight"),
        "attn_norm_weight": get("blk.40.attn_norm.weight"),
        "wq_weight": get("blk.40.attn_q.weight"),
        "wk_weight": get("blk.40.attn_k.weight"),
        "wv_weight": get("blk.40.attn_v.weight"),
        "wo_weight": get("blk.40.attn_output.weight"),
        "q_norm_weight": get("blk.40.attn_q_norm.weight"),
        "k_norm_weight": get("blk.40.attn_k_norm.weight"),
        "post_norm_weight": get("blk.40.post_attention_norm.weight"),
        "router_weight": get("blk.40.ffn_gate_inp.weight"),
        "gate_qweight": get("blk.40.ffn_gate_exps.weight"),
        "up_qweight": get("blk.40.ffn_up_exps.weight"),
        "down_qweight": get("blk.40.ffn_down_exps.weight"),
        "shared_gate_logit_weight": get("blk.40.ffn_gate_inp_shexp.weight"),
        "shared_gate_qweight": get("blk.40.ffn_gate_shexp.weight"),
        "shared_up_qweight": get("blk.40.ffn_up_shexp.weight"),
        "shared_down_qweight": get("blk.40.ffn_down_shexp.weight"),
        "shared_head_norm_weight": get("blk.40.nextn.shared_head_norm.weight"),
        "shared_head_weight": get("output.weight"),
        # qtypes
        "eh_proj_qtype": qt("blk.40.nextn.eh_proj.weight"),
        "wq_qtype": qt("blk.40.attn_q.weight"),
        "wk_qtype": qt("blk.40.attn_k.weight"),
        "wv_qtype": qt("blk.40.attn_v.weight"),
        "wo_qtype": qt("blk.40.attn_output.weight"),
        "gate_qtype": qt("blk.40.ffn_gate_exps.weight"),
        "up_qtype": qt("blk.40.ffn_up_exps.weight"),
        "down_qtype": qt("blk.40.ffn_down_exps.weight"),
        "shared_qtype": qt("blk.40.ffn_gate_shexp.weight"),
        "shared_head_qtype": qt("output.weight"),
        # Dequanted F32 for cpu_reference
        "cpu_eh_proj": dq("blk.40.nextn.eh_proj.weight"),
        "cpu_wq": dq("blk.40.attn_q.weight"),
        "cpu_wk": dq("blk.40.attn_k.weight"),
        "cpu_wv": dq("blk.40.attn_v.weight"),
        "cpu_wo": dq("blk.40.attn_output.weight"),
        "cpu_shared_head": dq("output.weight"),
        # Config
        "num_heads": 16,
        "num_kv_heads": 2,
        "experts_used": 8,
        "eps": 1e-6,
    }


def _run_gpu_nextn(mtp_w, hidden_seed, token_embed):
    """Run the composite mtp_nextn_layer on GPU with raw K-quant weights."""
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_nextn_layer_logits_f32 as gpu_kernel,
    )
    args = [
        hidden_seed, token_embed,
        mtp_w["eh_proj_weight"], mtp_w["hnorm_weight"],
        mtp_w["enorm_weight"], mtp_w["attn_norm_weight"],
        mtp_w["wq_weight"], mtp_w["wk_weight"],
        mtp_w["wv_weight"], mtp_w["wo_weight"],
        mtp_w["q_norm_weight"], mtp_w["k_norm_weight"],
        mtp_w["post_norm_weight"], mtp_w["router_weight"],
        mtp_w["gate_qweight"], mtp_w["up_qweight"],
        mtp_w["down_qweight"],
        mtp_w["gate_qtype"], mtp_w["up_qtype"], mtp_w["down_qtype"],
        mtp_w["shared_gate_logit_weight"],
        mtp_w["shared_gate_qweight"], mtp_w["shared_up_qweight"],
        mtp_w["shared_down_qweight"], mtp_w["shared_qtype"],
        mtp_w["shared_head_norm_weight"], mtp_w["shared_head_weight"],
    ]
    kwargs = dict(
        num_heads=mtp_w["num_heads"], num_kv_heads=mtp_w["num_kv_heads"],
        experts_used=mtp_w["experts_used"],
        eh_proj_qtype=mtp_w["eh_proj_qtype"],
        wq_qtype=mtp_w["wq_qtype"], wk_qtype=mtp_w["wk_qtype"],
        wv_qtype=mtp_w["wv_qtype"], wo_qtype=mtp_w["wo_qtype"],
        shared_head_qtype=mtp_w["shared_head_qtype"],
        eps=mtp_w["eps"],
    )
    return np.asarray(gpu_kernel(*args, **kwargs), dtype=np.float32)


def _run_cpu_nextn(mtp_w, hidden_seed, token_embed):
    """Run cpu_reference oracle with dequanted F32 weights."""
    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_nextn_layer_logits as cpu_oracle,
    )
    args = [
        hidden_seed, token_embed,
        mtp_w["cpu_eh_proj"], mtp_w["hnorm_weight"],
        mtp_w["enorm_weight"], mtp_w["attn_norm_weight"],
        mtp_w["cpu_wq"], mtp_w["cpu_wk"],
        mtp_w["cpu_wv"], mtp_w["cpu_wo"],
        mtp_w["q_norm_weight"], mtp_w["k_norm_weight"],
        mtp_w["post_norm_weight"], mtp_w["router_weight"],
        mtp_w["gate_qweight"], mtp_w["up_qweight"],
        mtp_w["down_qweight"],
        mtp_w["gate_qtype"], mtp_w["up_qtype"], mtp_w["down_qtype"],
        mtp_w["shared_gate_logit_weight"],
        mtp_w["shared_gate_qweight"], mtp_w["shared_up_qweight"],
        mtp_w["shared_down_qweight"], mtp_w["shared_qtype"],
        mtp_w["shared_head_norm_weight"], mtp_w["cpu_shared_head"],
    ]
    kwargs = dict(
        num_heads=mtp_w["num_heads"], num_kv_heads=mtp_w["num_kv_heads"],
        experts_used=mtp_w["experts_used"], eps=mtp_w["eps"],
    )
    return cpu_oracle(*args, **kwargs)


def test_m5_ar_decode_plus_mtp_draft_consistency(mtp_weights):
    """M5: Run AR decode + MTP draft and verify GPU vs cpu_reference consistency.

    This test:
    1. Runs the GGUF resident session for a few AR decode steps
    2. After each step, captures the fp32 hidden seed
    3. Runs the composite mtp_nextn_layer (GPU + cpu_reference) with the seed
    4. Gates GPU draft logits vs cpu_reference (max_abs < 3.0, top-5 overlap)
    """
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = Qwen35GGUFResidentSession(model_path=GGUF_PATH)
    try:
        # Simple 3-token prompt
        prompt = [1, 2, 3]
        prefill_result = session.prefill(prompt, return_logits=False,
                                         capture_hidden_seed_fp32=True)

        # Get the hidden seed from device → host
        contract = session.fp32_hidden_seed_contract()
        assert contract.ready_for_mtp, "fp32 hidden seed not ready for MTP"

        # Copy device hidden seed to host
        from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
        runtime = session.runtime or get_hip_runtime()
        hidden_size = contract.hidden_size
        hidden_seed = np.empty((1, hidden_size), dtype=np.float32)
        seed_ptr = session.fp32_hidden_seed_ptr()
        runtime.memcpy(hidden_seed.ctypes.data, seed_ptr,
                        hidden_size * 4, HipMemcpyKind.DEVICE_TO_HOST)

        # For the token embedding, we need the embedding of the last token.
        # The composite layer takes (hidden_seed, token_embedding) where
        # token_embedding is the embedding of the last accepted token.
        # For now, use a random embedding (the real pipeline would look up
        # the token embedding from the model's embedding table).
        # TODO: get the real token embedding from the resident session.
        np.random.seed(99)
        token_embed = np.random.randn(1, hidden_size).astype(np.float32) * 0.1

        # Run MTP draft on GPU
        gpu_logits = _run_gpu_nextn(mtp_weights, hidden_seed, token_embed)

        # Run MTP draft on cpu_reference
        cpu_logits = _run_cpu_nextn(mtp_weights, hidden_seed, token_embed)

        assert gpu_logits.shape == cpu_logits.shape
        max_abs = float(np.max(np.abs(gpu_logits - cpu_logits)))
        assert max_abs < 3.0, (
            f"M5 draft logits max_abs={max_abs} exceeds 3.0 vs cpu_reference"
        )

        # Check top-5 overlap
        cpu_top5 = set(np.argsort(cpu_logits[0])[-5:].tolist())
        gpu_top5 = set(np.argsort(gpu_logits[0])[-5:].tolist())
        overlap = len(cpu_top5 & gpu_top5)
        assert overlap >= 1, (
            f"M5 draft top-5 has zero overlap: cpu={cpu_top5} vs gpu={gpu_top5} "
            f"(max_abs={max_abs})"
        )

    finally:
        session.close()


def test_m5_multiple_ar_steps_mtp_draft(mtp_weights):
    """M5: Run multiple AR decode steps and verify MTP draft at each step.

    This validates that the MTP draft layer produces consistent results
    across multiple AR decode steps with different hidden seeds.
    """
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = Qwen35GGUFResidentSession(model_path=GGUF_PATH)
    try:
        prompt = [1, 2, 3, 4, 5]
        prefill_result = session.prefill(prompt, return_logits=False,
                        capture_hidden_seed_fp32=True)

        from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
        runtime = session.runtime or get_hip_runtime()
        hidden_size = 2048

        np.random.seed(77)
        results = []
        # Get first token from prefill
        prev_token = int(prefill_result.token_id) if hasattr(prefill_result, 'token_id') else 6
        for step_i in range(3):
            # Decode one token
            decode_result = session.step(prev_token, capture_hidden_seed_fp32=True)
            token_id = int(decode_result.token_id)

            # Capture hidden seed
            hidden_seed = np.empty((1, hidden_size), dtype=np.float32)
            runtime.memcpy(hidden_seed.ctypes.data, session.fp32_hidden_seed_ptr(),
                            hidden_size * 4, HipMemcpyKind.DEVICE_TO_HOST)

            # Use random token embedding (TODO: real embedding lookup)
            token_embed = np.random.randn(1, hidden_size).astype(np.float32) * 0.1

            # Run MTP draft on GPU and cpu
            gpu_logits = _run_gpu_nextn(mtp_weights, hidden_seed, token_embed)
            cpu_logits = _run_cpu_nextn(mtp_weights, hidden_seed, token_embed)

            max_abs = float(np.max(np.abs(gpu_logits - cpu_logits)))
            cpu_top1 = int(np.argmax(cpu_logits[0]))
            gpu_top1 = int(np.argmax(gpu_logits[0]))
            results.append((token_id, max_abs, cpu_top1, gpu_top1))
            prev_token = token_id

        # All steps should have max_abs < 3.0
        for i, (token_id, max_abs, cpu_t1, gpu_t1) in enumerate(results):
            assert max_abs < 3.0, (
                f"Step {i}: M5 draft max_abs={max_abs} exceeds 3.0 "
                f"(token={token_id}, cpu_top1={cpu_t1}, gpu_top1={gpu_t1})"
            )

    finally:
        session.close()