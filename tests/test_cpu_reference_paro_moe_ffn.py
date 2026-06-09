"""B3 PARO B0: golden PARO selected-expert MoE FFN oracle.

Pins the cpu_reference PARO selected-FFN composition (rotate1[shared] ->
per-expert {AWQ gate/up -> silu*mul -> down-rotate -> AWQ down} -> routing
combine) that the B3 fused PARO megakernel will be gated against (KL<=0.05).
Correctness is established by an independent dense-dequant recomputation (a
different code path than the oracle's per-(token,expert) GEMV loop) plus a
row-invariance check (the T1 requirement). The AWQ pack8 + rotate1 primitives
are separately GPU-validated in test_cpu_reference_paro_primitives.py.
"""

from __future__ import annotations

import numpy as np

from hipengine.kernels.cpu_reference import (
    awq_pack8_dequant_transposed,
    paro_moe_selected_ffn,
    paro_rotate1,
)
from hipengine.kernels.cpu_reference.ops import _silu
from hipengine.kernels.registry import resolve

HIDDEN = 256
FFN_LEN = 256
GROUP_SIZE = 128
NUM_EXPERTS = 4
TOP_K = 2
TOKENS = 4
KROT = 1


def _adjacent_pairs(dim: int, krot: int) -> np.ndarray:
    half = GROUP_SIZE // 2
    pairs = np.zeros((krot, dim), np.int64)
    for r in range(krot):
        for g in range(dim // GROUP_SIZE):
            for lane in range(half):
                pairs[r, g * GROUP_SIZE + 2 * lane] = 2 * lane
                pairs[r, g * GROUP_SIZE + 2 * lane + 1] = 2 * lane + 1
    return pairs


def _awq_expert_stack(rng, out_f: int, in_f: int, num_experts: int):
    out_packed = out_f // 8
    groups = in_f // GROUP_SIZE
    qw = rng.integers(0, 2**32, size=(num_experts, out_packed, in_f), dtype=np.uint64).astype(np.uint32).view(np.int32)
    qz = rng.integers(0, 2**32, size=(num_experts, groups, out_packed), dtype=np.uint64).astype(np.uint32).view(np.int32)
    sc = rng.uniform(0.001, 0.04, size=(num_experts, groups, out_f)).astype(np.float32)
    return qw, qz, sc


def build_paro_fixture() -> dict:
    rng = np.random.default_rng(20260609)
    x = (rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)) * np.float32(0.1)
    selected = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    rl = rng.standard_normal((TOKENS, TOP_K)).astype(np.float32)
    routing = (np.exp(rl - rl.max(1, keepdims=True)) / np.exp(rl - rl.max(1, keepdims=True)).sum(1, keepdims=True)).astype(np.float32)
    gate = _awq_expert_stack(rng, FFN_LEN, HIDDEN, NUM_EXPERTS)
    up = _awq_expert_stack(rng, FFN_LEN, HIDDEN, NUM_EXPERTS)
    down = _awq_expert_stack(rng, HIDDEN, FFN_LEN, NUM_EXPERTS)
    r1 = (_adjacent_pairs(HIDDEN, KROT), rng.uniform(-1, 1, (KROT, HIDDEN // 2)).astype(np.float32),
          rng.uniform(0.5, 1.5, HIDDEN).astype(np.float32), KROT)
    dr = (_adjacent_pairs(FFN_LEN, KROT), rng.uniform(-1, 1, (KROT, FFN_LEN // 2)).astype(np.float32),
          rng.uniform(0.5, 1.5, FFN_LEN).astype(np.float32), KROT)
    return {"x": x, "selected": selected, "routing": routing, "gate": gate, "up": up, "down": down, "r1": r1, "dr": dr}


def _oracle(f: dict) -> np.ndarray:
    return paro_moe_selected_ffn(
        f["x"], f["selected"], f["routing"], f["gate"], f["up"], f["down"],
        HIDDEN, FFN_LEN, GROUP_SIZE, f["r1"], f["dr"],
    )


def _dense_recompute(f: dict) -> np.ndarray:
    """Independent path: dense-dequant AWQ + explicit matmuls + rotations."""

    x = f["x"].astype(np.float32)
    sel = f["selected"]
    routing = f["routing"]
    gqw, gqz, gsc = f["gate"]; uqw, uqz, usc = f["up"]; dqw, dqz, dsc = f["down"]
    r1p, r1t, r1s, r1k = f["r1"]; drp, drt, drs, drk = f["dr"]
    x_rot = paro_rotate1(x, r1p, r1t, r1s, GROUP_SIZE, r1k)
    Wg = np.stack([awq_pack8_dequant_transposed(gqw[e], gqz[e], gsc[e], HIDDEN, FFN_LEN, GROUP_SIZE) for e in range(NUM_EXPERTS)])
    Wu = np.stack([awq_pack8_dequant_transposed(uqw[e], uqz[e], usc[e], HIDDEN, FFN_LEN, GROUP_SIZE) for e in range(NUM_EXPERTS)])
    Wd = np.stack([awq_pack8_dequant_transposed(dqw[e], dqz[e], dsc[e], FFN_LEN, HIDDEN, GROUP_SIZE) for e in range(NUM_EXPERTS)])
    out = np.zeros((TOKENS, HIDDEN), np.float32)
    for t in range(TOKENS):
        acc = np.zeros((HIDDEN,), np.float32)
        for k in range(TOP_K):
            e = int(sel[t, k])
            gate = Wg[e] @ x_rot[t]
            up = Wu[e] @ x_rot[t]
            act = (_silu(gate) * up).astype(np.float32)[None, :]
            act_rot = paro_rotate1(act, drp, drt, drs, GROUP_SIZE, drk)[0]
            acc = acc + np.float32(routing[t, k]) * (Wd[e] @ act_rot).astype(np.float32)
        out[t] = acc
    return out


def test_paro_oracle_matches_independent_dense_recompute():
    f = build_paro_fixture()
    oracle = _oracle(f)
    dense = _dense_recompute(f)
    assert oracle.shape == (TOKENS, HIDDEN)
    np.testing.assert_allclose(oracle, dense, rtol=1e-5, atol=1e-5)


def test_paro_oracle_is_row_invariant():
    """rows=1 alone == the same row inside the batch (the T1 requirement)."""

    f = build_paro_fixture()
    batched = _oracle(f)
    for t in range(TOKENS):
        single = paro_moe_selected_ffn(
            f["x"][t : t + 1], f["selected"][t : t + 1], f["routing"][t : t + 1],
            f["gate"], f["up"], f["down"], HIDDEN, FFN_LEN, GROUP_SIZE, f["r1"], f["dr"],
        )
        np.testing.assert_allclose(single[0], batched[t], rtol=1e-5, atol=1e-5)


def test_registry_resolves_paro_moe_ffn_selected():
    kernel = resolve(backend="cpu_reference", layer="moe_ffn_selected", quant="w4_paro")
    f = build_paro_fixture()
    got = kernel(
        f["x"], f["selected"], f["routing"], f["gate"], f["up"], f["down"],
        HIDDEN, FFN_LEN, GROUP_SIZE, f["r1"], f["dr"],
    )
    np.testing.assert_array_equal(got, _oracle(f))
