"""B0 golden oracle for the selected-expert GGUF MoE FFN (megakernel unit).

These tests pin the CPU-reference selected-expert MoE FFN
(``gate_up -> silu*mul -> down -> routing-weighted combine``) and the full
qwen35moe FFN block (``+ Q8_0 shared expert + sigmoid gate + residual``) that
the B1 fused-FFN megakernel must reproduce within the project KL gate.

Correctness here is established three ways:
  1. an *independent* dense-dequant recomputation (a different code path than the
     oracle's per-(token, expert) ``gguf_quant_gemv`` loop),
  2. a committed golden fixture (regression pin), and
  3. a row-invariance check (rows=1 == in-batch per row), the hard requirement
     for the T1 self-consistent accuracy tier.

The per-projection quant types mirror the real Qwen3.6-35B-A3B-UD-Q4_K_S GGUF:
Q4_K gate/up/down experts and Q8_0 shared expert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import (
    gguf_moe_ffn_block,
    gguf_moe_selected_ffn,
    gguf_quant_gemv,
)
from hipengine.kernels.cpu_reference.ops import gguf_q4_k_moe_selected_ffn
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q8_0_weight

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cpu_reference" / "moe" / "moe_ffn_selected_gguf_q4_k.json"

HIDDEN = 256
FFN_LEN = 256
SHARED_FFN_LEN = 256
NUM_EXPERTS = 4
TOP_K = 2
TOKENS = 4
# The synthetic Q4_K/Q8_0 weights carry large positive magnitudes (the helper
# targets bit-exact GEMV, not realistic scale). Scale the activations so the FFN
# stays in silu's meaningful range and outputs land at O(1), making the B1
# bf16-vs-fp32 KL gate well-conditioned.
_X_SCALE = np.float32(1.0e-3)
_RESIDUAL_SCALE = np.float32(0.25)


def _stack_experts(builder, out_features: int, in_features: int, num_experts: int, seed: int) -> np.ndarray:
    base = builder(out_features, in_features)
    return np.stack([np.roll(base, shift=e + seed, axis=0) for e in range(num_experts)], axis=0)


def build_b0_fixture() -> dict:
    """Deterministic synthetic inputs/weights for the B0 MoE FFN fixture.

    Weights are regenerated from ``tests/_gguf_synthetic_weights`` so the
    committed fixture stores only the small inputs and golden outputs.
    """

    rng = np.random.default_rng(20260609)
    x = (rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)) * _X_SCALE
    residual = (rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)) * _RESIDUAL_SCALE
    selected = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    router_logits = rng.standard_normal((TOKENS, TOP_K)).astype(np.float32)
    shifted = router_logits - router_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    routing_weights = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    gate_q = _stack_experts(make_q4_k_weight, FFN_LEN, HIDDEN, NUM_EXPERTS, seed=1)
    up_q = _stack_experts(make_q4_k_weight, FFN_LEN, HIDDEN, NUM_EXPERTS, seed=5)
    down_q = _stack_experts(make_q4_k_weight, HIDDEN, FFN_LEN, NUM_EXPERTS, seed=9)
    shared_gate_q = make_q8_0_weight(SHARED_FFN_LEN, HIDDEN)
    shared_up_q = np.roll(make_q8_0_weight(SHARED_FFN_LEN, HIDDEN), shift=3, axis=0)
    shared_down_q = make_q8_0_weight(HIDDEN, SHARED_FFN_LEN)
    shared_gate_logit_w = (rng.standard_normal((HIDDEN,)).astype(np.float32)) * np.float32(0.05)

    return {
        "x": x,
        "residual": residual,
        "selected_experts": selected,
        "routing_weights": routing_weights,
        "gate_q": gate_q,
        "up_q": up_q,
        "down_q": down_q,
        "shared_gate_q": shared_gate_q,
        "shared_up_q": shared_up_q,
        "shared_down_q": shared_down_q,
        "shared_gate_logit_w": shared_gate_logit_w,
    }


def _silu(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.float32(1.0) + np.exp(-v).astype(np.float32))


def _dense_selected_ffn(f: dict) -> np.ndarray:
    """Independent dense-dequant recomputation (different code path)."""

    x = f["x"].astype(np.float32)
    sel = f["selected_experts"]
    rw = f["routing_weights"].astype(np.float32)
    # Dequantize every expert up front into dense [E, out, in] weights.
    wg = np.stack([dequantize_gguf_data(f["gate_q"][e], GGMLQuantizationType.Q4_K) for e in range(NUM_EXPERTS)])
    wu = np.stack([dequantize_gguf_data(f["up_q"][e], GGMLQuantizationType.Q4_K) for e in range(NUM_EXPERTS)])
    wd = np.stack([dequantize_gguf_data(f["down_q"][e], GGMLQuantizationType.Q4_K) for e in range(NUM_EXPERTS)])
    out = np.zeros((TOKENS, HIDDEN), dtype=np.float32)
    for t in range(TOKENS):
        acc = np.zeros((HIDDEN,), dtype=np.float32)
        for k in range(TOP_K):
            e = int(sel[t, k])
            gate = wg[e] @ x[t]
            up = wu[e] @ x[t]
            inter = _silu(gate) * up
            down = wd[e] @ inter
            acc = acc + np.float32(rw[t, k]) * down.astype(np.float32)
        out[t] = acc
    return out


def _dense_block(f: dict) -> np.ndarray:
    x = f["x"].astype(np.float32)
    selected_out = _dense_selected_ffn(f)
    sg = dequantize_gguf_data(f["shared_gate_q"], GGMLQuantizationType.Q8_0)
    su = dequantize_gguf_data(f["shared_up_q"], GGMLQuantizationType.Q8_0)
    sd = dequantize_gguf_data(f["shared_down_q"], GGMLQuantizationType.Q8_0)
    shared_inter = _silu(x @ sg.T) * (x @ su.T)
    shared_out = shared_inter @ sd.T
    gate = 1.0 / (1.0 + np.exp(-(x @ f["shared_gate_logit_w"])))
    return (f["residual"].astype(np.float32) + selected_out + gate[:, None] * shared_out).astype(np.float32)


def _oracle_selected(f: dict) -> np.ndarray:
    return gguf_moe_selected_ffn(
        f["x"], f["selected_experts"], f["routing_weights"],
        f["gate_q"], f["up_q"], f["down_q"],
        GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K,
    )


def _oracle_block(f: dict) -> np.ndarray:
    return gguf_moe_ffn_block(
        f["x"], f["residual"], f["selected_experts"], f["routing_weights"],
        f["gate_q"], f["up_q"], f["down_q"],
        GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K,
        f["shared_gate_logit_w"], f["shared_gate_q"], f["shared_up_q"], f["shared_down_q"],
        GGMLQuantizationType.Q8_0,
    )


def test_selected_ffn_matches_independent_dense_recompute():
    f = build_b0_fixture()
    oracle = _oracle_selected(f)
    dense = _dense_selected_ffn(f)
    assert oracle.shape == (TOKENS, HIDDEN)
    np.testing.assert_allclose(oracle, dense, rtol=1e-5, atol=1e-5)


def test_full_block_matches_independent_dense_recompute():
    f = build_b0_fixture()
    oracle = _oracle_block(f)
    dense = _dense_block(f)
    np.testing.assert_allclose(oracle, dense, rtol=1e-5, atol=1e-5)


def test_q4_k_specialization_matches_general():
    f = build_b0_fixture()
    spec = gguf_q4_k_moe_selected_ffn(
        f["x"], f["selected_experts"], f["routing_weights"], f["gate_q"], f["up_q"], f["down_q"]
    )
    np.testing.assert_array_equal(spec, _oracle_selected(f))


def test_registry_resolves_moe_ffn_selected():
    kernel = resolve(backend="cpu_reference", layer="moe_ffn_selected", quant="gguf_q4_k")
    f = build_b0_fixture()
    got = kernel(
        x=f["x"], selected_experts=f["selected_experts"], routing_weights=f["routing_weights"],
        gate_qweight=f["gate_q"], up_qweight=f["up_q"], down_qweight=f["down_q"],
    )
    np.testing.assert_array_equal(got, _oracle_selected(f))


def test_selected_ffn_is_row_invariant():
    """rows=1 alone == the same row inside a B-row batch (the T1 hard requirement)."""

    f = build_b0_fixture()
    batched = _oracle_selected(f)
    for t in range(TOKENS):
        single = gguf_moe_selected_ffn(
            f["x"][t : t + 1], f["selected_experts"][t : t + 1], f["routing_weights"][t : t + 1],
            f["gate_q"], f["up_q"], f["down_q"],
            GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K,
        )
        np.testing.assert_array_equal(single[0], batched[t])


def test_matches_committed_golden_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"golden fixture missing: {FIXTURE_PATH} (run scripts/gen_b0_moe_ffn_fixture.py)")
    golden = json.loads(FIXTURE_PATH.read_text())
    f = build_b0_fixture()
    expected_selected = np.asarray(golden["expected_selected"], dtype=np.float32)
    expected_block = np.asarray(golden["expected_block"], dtype=np.float32)
    got_selected = gguf_q4_k_moe_selected_ffn(
        f["x"], f["selected_experts"], f["routing_weights"], f["gate_q"], f["up_q"], f["down_q"]
    )
    np.testing.assert_allclose(got_selected, expected_selected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_oracle_block(f), expected_block, rtol=1e-6, atol=1e-6)
