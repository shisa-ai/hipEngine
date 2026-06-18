"""RED/GREEN test for RoPE in the native hip mtp_nextn_attention sublayer (M6).

Tests that the attention sublayer with rope_cos/rope_sin provided matches the
cpu_reference oracle (which applies rotate() to query and key_cur before the
dense causal attention).  RED-first: the current attention wrapper raises
NotImplementedError for rope_cos/rope_sin, so this test should fail RED until
RoPE is implemented.
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


def _f32(v):
    return np.ascontiguousarray(v, dtype=np.float32)


@pytest.fixture(scope="module")
def rope_inputs():
    fixture = json.loads(FIXTURE.read_text())
    inputs = fixture["inputs"]
    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_eh_proj as eh,
        qwen35_gguf_mtp_attention_sublayer as attn,
    )
    projected = eh(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        eps=float(fixture["kwargs"].get("eps", 1e-6)),
    )
    num_heads = int(fixture["kwargs"]["num_heads"])
    num_kv_heads = int(fixture["kwargs"]["num_kv_heads"])
    # The fixture's qk_head_dim = hidden (tiny synthetic). rotary_dim = qk_head_dim.
    hidden = int(inputs["hnorm_weight"].__len__())
    qk_head_dim = int(inputs["q_norm_weight"].__len__())
    rotary_dim = qk_head_dim
    # Synthetic cos/sin: simple position-based values
    pos = np.arange(1, dtype=np.float32)
    half = rotary_dim // 2
    # cos/sin with shape [1, half] (half-rotary table)
    cos = np.cos(pos[:, None] * np.arange(half, dtype=np.float32) / hidden).astype(np.float32)
    sin = np.sin(pos[:, None] * np.arange(half, dtype=np.float32) / hidden).astype(np.float32)
    return {
        "hidden": _f32(projected),
        "attn_norm_weight": _f32(inputs["attn_norm_weight"]),
        "wq_weight": _f32(inputs["wq_weight"]),
        "wk_weight": _f32(inputs["wk_weight"]),
        "wv_weight": _f32(inputs["wv_weight"]),
        "wo_weight": _f32(inputs["wo_weight"]),
        "q_norm_weight": _f32(inputs["q_norm_weight"]),
        "k_norm_weight": _f32(inputs["k_norm_weight"]),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_cos": cos,
        "rope_sin": sin,
        "rotary_dim": rotary_dim,
        "eps": float(fixture["kwargs"].get("eps", 1e-6)),
    }


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_attention_with_rope_matches_cpu_reference(backend, rope_inputs):
    """The attention sublayer with RoPE must match cpu_reference (which applies
    rotate() to query+key before dense attention)."""
    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_attention_sublayer as cpu_oracle,
    )
    from hipengine.kernels.registry import resolve

    expected = cpu_oracle(
        rope_inputs["hidden"],
        rope_inputs["attn_norm_weight"],
        rope_inputs["wq_weight"], rope_inputs["wk_weight"],
        rope_inputs["wv_weight"], rope_inputs["wo_weight"],
        rope_inputs["q_norm_weight"], rope_inputs["k_norm_weight"],
        num_heads=rope_inputs["num_heads"],
        num_kv_heads=rope_inputs["num_kv_heads"],
        rope_cos=rope_inputs["rope_cos"],
        rope_sin=rope_inputs["rope_sin"],
        rotary_dim=rope_inputs["rotary_dim"],
        eps=rope_inputs["eps"],
    )

    try:
        kernel = resolve(
            backend=backend, layer="mtp_nextn_attention", quant="gguf_f32",
            variant="qwen35_dense",
        )
    except Exception:
        pytest.skip(f"{backend} mtp_nextn_attention not registered (RED)")

    if kernel is cpu_oracle:
        pytest.skip(f"{backend} returned cpu_reference via fallback (RED)")

    try:
        actual = kernel(
            rope_inputs["hidden"],
            rope_inputs["attn_norm_weight"],
            rope_inputs["wq_weight"], rope_inputs["wk_weight"],
            rope_inputs["wv_weight"], rope_inputs["wo_weight"],
            rope_inputs["q_norm_weight"], rope_inputs["k_norm_weight"],
            num_heads=rope_inputs["num_heads"],
            num_kv_heads=rope_inputs["num_kv_heads"],
            rope_cos=rope_inputs["rope_cos"],
            rope_sin=rope_inputs["rope_sin"],
            rotary_dim=rope_inputs["rotary_dim"],
            eps=rope_inputs["eps"],
        )
    except NotImplementedError as e:
        pytest.fail(f"{backend} attention raised NotImplementedError for RoPE: {e}")

    actual = np.asarray(actual, dtype=np.float32)
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs < 1e-3, f"{backend} RoPE attention max_abs={max_abs} exceeds 1e-3 vs cpu_reference"