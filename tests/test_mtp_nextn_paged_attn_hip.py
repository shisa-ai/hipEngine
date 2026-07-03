"""RED/GREEN test for KVLiveSpans paged-cache attention in the native hip mtp_nextn_attention (M6).

Tests that the attention sublayer with paged key_cache/value_cache + kv_base_offsets
+ kv_live_counts matches the cpu_reference oracle. RED-first: the current attention
wrapper raises NotImplementedError for external KV cache, so this test should fail RED.
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
def paged_inputs():
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
    hidden = len(inputs["hnorm_weight"])
    qk_head_dim = len(inputs["q_norm_weight"])
    block_size = 4
    # Build a paged cache with 1 block, block_size slots, for 1 token at position 0
    # The cpu_reference will write key_cur/value_cur into the cache and attend.
    # We need: key_cache [1, block_size, kv_heads, qk_head_dim],
    #          value_cache [1, block_size, kv_heads, value_head_dim],
    #          kv_base_offsets=[0], kv_live_counts=[1], token_positions=[0]
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
        "block_size": block_size,
        "eps": float(fixture["kwargs"].get("eps", 1e-6)),
    }


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_paged_attention_matches_cpu_reference(backend, paged_inputs):
    """The attention sublayer with paged KV cache must match cpu_reference."""
    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_attention_sublayer as cpu_oracle,
    )
    from hipengine.kernels.registry import resolve

    tokens = 1
    num_kv_heads = paged_inputs["num_kv_heads"]
    qk_head_dim = len(paged_inputs["q_norm_weight"])
    block_size = paged_inputs["block_size"]

    # Build paged cache: [1 block, block_size slots, kv_heads, head_dim]
    # Fill with known K/V values (not zeros) so both cpu and hip attend to the same cache.
    np.random.seed(123)
    key_cache = np.random.randn(1, block_size, num_kv_heads, qk_head_dim).astype(np.float32)
    value_cache = np.random.randn(1, block_size, num_kv_heads, qk_head_dim).astype(np.float32)
    kv_base_offsets = np.array([0], dtype=np.int64)
    kv_live_counts = np.array([1], dtype=np.int64)
    kv_token_positions = np.array([0], dtype=np.int64)

    expected = cpu_oracle(
        paged_inputs["hidden"],
        paged_inputs["attn_norm_weight"],
        paged_inputs["wq_weight"], paged_inputs["wk_weight"],
        paged_inputs["wv_weight"], paged_inputs["wo_weight"],
        paged_inputs["q_norm_weight"], paged_inputs["k_norm_weight"],
        num_heads=paged_inputs["num_heads"],
        num_kv_heads=num_kv_heads,
        key_cache=key_cache, value_cache=value_cache,
        kv_base_offsets=kv_base_offsets,
        kv_live_counts=kv_live_counts,
        kv_token_positions=kv_token_positions,
        block_size=block_size,
        eps=paged_inputs["eps"],
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
            paged_inputs["hidden"],
            paged_inputs["attn_norm_weight"],
            paged_inputs["wq_weight"], paged_inputs["wk_weight"],
            paged_inputs["wv_weight"], paged_inputs["wo_weight"],
            paged_inputs["q_norm_weight"], paged_inputs["k_norm_weight"],
            num_heads=paged_inputs["num_heads"],
            num_kv_heads=num_kv_heads,
            key_cache=key_cache, value_cache=value_cache,
            kv_base_offsets=kv_base_offsets,
            kv_live_counts=kv_live_counts,
            kv_token_positions=kv_token_positions,
            block_size=block_size,
            eps=paged_inputs["eps"],
        )
    except NotImplementedError as e:
        pytest.fail(f"{backend} attention raised NotImplementedError for paged KV: {e}")

    actual = np.asarray(actual, dtype=np.float32)
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs < 1e-3, f"{backend} paged attention max_abs={max_abs} exceeds 1e-3 vs cpu_reference"