"""GPU RED: EAGLE-4 device-resident aux projection (task #25c).

append_committed_device projects the target aux rows on device (f32->bf16, per-
chunk RMS-norm, concat, fc GEMM) instead of the host numpy projection + D2H/H2D.
Gates: (1) the projected feature_seq matches the numpy projection of the bf16-
rounded aux within bf16 tolerance (catches layout/offset bugs); (2) the full
append + cached draft reproduces the AR-exact numpy context reference (same gate
as the host append_committed test). Synthetic weights, HIP-guarded for CI.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")


def _config(hidden=64, heads=4, head_dim=16, vocab=40, rank=8, candidates=12):
    return {
        "hidden_size": hidden, "num_attention_heads": heads, "num_key_value_heads": heads,
        "head_dim": head_dim, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e7},
        "sliding_window": 64, "norm_output": True, "fc_norm": True, "vocab_size": vocab,
        "intermediate_size": 128,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
        "efficient_lm_head_config": {"type": "low_rank", "rank": rank, "num_candidates": candidates},
    }


def _weights(hidden=64, vocab=40, rank=8, inter=128):
    rng = np.random.default_rng(0)
    def r(*shape): return rng.standard_normal(shape).astype(np.float32) * 0.1
    return {
        "fc.weight": r(hidden, 3 * hidden),
        "fc_norm.0.weight": np.ones(hidden, np.float32),
        "fc_norm.1.weight": np.ones(hidden, np.float32),
        "fc_norm.2.weight": np.ones(hidden, np.float32),
        "embed_tokens.weight": r(vocab, hidden),
        "layers.0.input_layernorm.weight": np.ones(hidden, np.float32),
        "layers.0.hidden_norm.weight": np.ones(hidden, np.float32),
        "layers.0.self_attn.q_proj.weight": r(hidden, 2 * hidden),
        "layers.0.self_attn.k_proj.weight": r(hidden, 2 * hidden),
        "layers.0.self_attn.v_proj.weight": r(hidden, 2 * hidden),
        "layers.0.self_attn.o_proj.weight": r(hidden, hidden),
        "layers.0.post_attention_layernorm.weight": np.ones(hidden, np.float32),
        "layers.0.mlp.gate_proj.weight": r(inter, hidden),
        "layers.0.mlp.up_proj.weight": r(inter, hidden),
        "layers.0.mlp.down_proj.weight": r(hidden, inter),
        "norm.weight": np.ones(hidden, np.float32),
        "efficient_lm_head_down_proj.weight": r(rank, hidden),
        "efficient_lm_head.weight": r(vocab, rank),
        "lm_head.weight": r(vocab, hidden),
    }


def _numpy_context_draft(nd, ctx_aux, token_seq, bonus, n_draft):
    P = len(token_seq)
    feats = list(nd.project_target_features(ctx_aux))
    toks = list(token_seq[1:]) + [int(bonus)]
    base_pos = P - 1
    drafts = []
    for _ in range(n_draft):
        T = len(feats)
        out_n, out_s = nd.layer(
            nd.w["embed_tokens.weight"][np.array(toks, dtype=np.int64)],
            np.stack(feats, axis=0),
            np.arange(base_pos - T + 1, base_pos + 1, dtype=np.int64))
        feature = (out_n if nd.norm_output else out_s)[-1]
        logits, cand = nd.efficient_head(feature[None, :])
        nxt = int(cand[0, np.argmax(logits[0])])
        drafts.append(nxt); feats.append(feature); toks.append(nxt); base_pos += 1
    return drafts


def _upload_aux_chunks(rt, new_aux, H):
    """Upload the 3 aux chunks as separate contiguous FP32 device buffers."""
    from hipengine.core.memory import malloc, copy_host_array_to_device
    m = new_aux.shape[0]
    ptrs, owners = [], []
    for i in range(3):
        chunk = np.ascontiguousarray(new_aux[:, i * H:(i + 1) * H], dtype=np.float32)
        buf = malloc(m * H * 4, runtime=rt)
        copy_host_array_to_device(buf, chunk, runtime=rt)
        ptrs.append(buf.ptr); owners.append(buf)
    return ptrs, owners


def test_project_aux_device_matches_numpy():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr, free
    from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter, bf16_to_f32
    from hipengine.runtime.eagle4 import EAGLE4NativeDrafter
    from hipengine.speculative.dflash2_native import _to_bf16_bits

    cfg = _config(); w = _weights(); H = cfg["hidden_size"]
    nd = EAGLE4NumpyDrafter(cfg, w); rt = get_hip_runtime()
    rng = np.random.default_rng(11)
    m = 3
    new_aux = rng.standard_normal((m, 3 * H)).astype(np.float32)
    native = EAGLE4NativeDrafter(cfg, w, runtime=rt, max_seq=8, max_context_len=64)
    owners = []
    try:
        aux_dev, owners = _upload_aux_chunks(rt, new_aux, H)
        native._project_aux_device(aux_dev, m)
        rt.device_synchronize()
        raw = np.empty(m * H, np.uint16)
        copy_device_to_host(host_array_ptr(raw), DeviceBuffer(native.feature_seq.ptr, raw.nbytes), raw.nbytes, runtime=rt)
        got = bf16_to_f32(raw).reshape(m, H)
        # Reference: numpy projection of the bf16-rounded aux (the device path
        # rounds aux to bf16 before norm/GEMM). Tolerance covers bf16 accumulation.
        aux_bf16 = bf16_to_f32(_to_bf16_bits(new_aux))
        expected = nd.project_target_features(aux_bf16)
        scale = max(1.0, float(np.max(np.abs(expected))))
        assert np.allclose(got, expected, rtol=2e-2, atol=2e-2 * scale), \
            f"max_diff={np.max(np.abs(got - expected))} scale={scale}"
    finally:
        for b in owners:
            free(b, runtime=rt)
        native.close()


def test_append_committed_device_matches_numpy():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free
    from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter
    from hipengine.runtime.eagle4 import EAGLE4NativeDrafter

    cfg = _config(); w = _weights(); H = cfg["hidden_size"]; V = cfg["vocab_size"]
    nd = EAGLE4NumpyDrafter(cfg, w); rt = get_hip_runtime()
    P, n_draft = 5, 4
    rng = np.random.default_rng(5)
    ctx_aux = rng.standard_normal((P, 3 * H)).astype(np.float32)
    prompt_ids = [int(t) for t in rng.integers(0, V, size=P)]
    bonus = int(rng.integers(0, V))
    native = EAGLE4NativeDrafter(cfg, w, runtime=rt, max_seq=n_draft + 2, max_context_len=64)
    owners = []
    try:
        native.prefill_context(ctx_aux, prompt_ids, bonus)
        m = 3
        new_aux = rng.standard_normal((m, 3 * H)).astype(np.float32)
        next_tokens = [int(t) for t in rng.integers(0, V, size=m)]
        aux_dev, owners = _upload_aux_chunks(rt, new_aux, H)
        native.append_committed_device(aux_dev, m, next_tokens)
        got, _ = native.draft_greedy_chain_cached(n_draft)
        ext_aux = np.concatenate([ctx_aux, new_aux], axis=0)
        ext_tokens = list(prompt_ids) + [int(bonus)] + [int(t) for t in next_tokens[:-1]]
        ext_bonus = int(next_tokens[-1])
        expected = _numpy_context_draft(nd, ext_aux, ext_tokens, ext_bonus, n_draft)
        assert [int(t) for t in got] == expected, f"got {list(got)} expected {expected}"
    finally:
        for b in owners:
            free(b, runtime=rt)
        native.close()
