"""Model-level parity gate: resident MTP draft device-chain vs legacy host loop.

Sub-win B (task #3) keeps the NextN draft chain on-device across depths: each
depth's top-1 is gathered from a resident FP32 embedding table (an exact copy of
``token_embd_f32``) instead of round-tripping through the host, and the per-depth
``device_synchronize`` + top-k readback collapse to a single drain at chain end.

Because the embedding rows and gather are exact, the device chain must produce
BIT-IDENTICAL draft tokens + top-k rows to the legacy path -> acceptance is
provably unchanged. This test asserts that exact parity on the real model.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float):
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.skipif(not MODEL.exists(), reason=f"model {MODEL} not present")
def test_device_chain_matches_legacy_exactly(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, malloc

    reader = GGUFReader(MODEL)
    meta = reader.info.metadata
    weights = {}
    for tensor in reader.info.tensors:
        if "blk.40" in tensor.name or tensor.name in ("output.weight", "token_embd.weight"):
            weights[tensor.name] = (reader.tensor_data(tensor.name), tensor.ggml_type, tensor.shape)

    def dq(name):
        return dequantize_gguf_data(weights[name][0], GGMLQuantizationType(weights[name][1])).astype(np.float32)

    token_embd_f32 = dq("token_embd.weight")
    rope_dim = int(meta.get("qwen35moe.rope.dimension_count", 64))
    rope_base = float(meta.get("qwen35moe.rope.freq_base", 1e7))
    rope_cos, rope_sin = _rope_tables(max_positions=4096, rotary_dim=rope_dim, base=rope_base)

    runtime = get_hip_runtime()
    runner = Qwen35GGUFResidentMTPDraftRunner(weights, token_embd_f32, runtime=runtime)
    kv_heads, d = runner.num_kv_heads, runner.qk_head_dim
    rng = np.random.default_rng(20260629)
    hidden_seed = (rng.standard_normal((1, runner.hidden_size)).astype(np.float32) * 0.1)

    draft_n, top_k = 5, 8

    def run(device_chain: bool):
        runner._device_chain_enabled = device_chain
        nbytes = draft_n * kv_heads * d * 4
        kc, vc = malloc(nbytes, runtime=runtime), malloc(nbytes, runtime=runtime)
        try:
            return runner.propose_chain(
                hidden_seed,
                start_token=4087,
                start_position=37,
                draft_n_max=draft_n,
                top_k=top_k,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                dense_key_cache=kc,
                dense_value_cache=vc,
                dense_cache_len=0,
            )
        finally:
            free(kc, runtime=runtime)
            free(vc, runtime=runtime)

    try:
        legacy_tokens, legacy_rows, legacy_klen = run(False)
        device_tokens, device_rows, device_klen = run(True)
    finally:
        runner.close()

    assert device_tokens == legacy_tokens, (legacy_tokens, device_tokens)
    assert device_rows == legacy_rows, (legacy_rows, device_rows)
    assert device_klen == legacy_klen


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.skipif(not MODEL.exists(), reason=f"model {MODEL} not present")
def test_resident_topk40_preserves_top8_prefix(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner
    from hipengine.core.hip import get_hip_runtime

    reader = GGUFReader(MODEL)
    meta = reader.info.metadata
    weights = {}
    for tensor in reader.info.tensors:
        if "blk.40" in tensor.name or tensor.name in ("output.weight", "token_embd.weight"):
            weights[tensor.name] = (reader.tensor_data(tensor.name), tensor.ggml_type, tensor.shape)

    def dq(name):
        return dequantize_gguf_data(weights[name][0], GGMLQuantizationType(weights[name][1])).astype(np.float32)

    token_embd_f32 = dq("token_embd.weight")
    rope_dim = int(meta.get("qwen35moe.rope.dimension_count", 64))
    rope_base = float(meta.get("qwen35moe.rope.freq_base", 1e7))
    rope_cos, rope_sin = _rope_tables(max_positions=4096, rotary_dim=rope_dim, base=rope_base)

    runtime = get_hip_runtime()
    runner = Qwen35GGUFResidentMTPDraftRunner(weights, token_embd_f32, runtime=runtime)
    rng = np.random.default_rng(20260630)
    hidden_seed = (rng.standard_normal((1, runner.hidden_size)).astype(np.float32) * 0.1)

    def run(top_k: int):
        runner._device_chain_enabled = False
        return runner.propose_chain(
            hidden_seed,
            start_token=4087,
            start_position=37,
            draft_n_max=1,
            top_k=top_k,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dense_key_cache=None,
            dense_value_cache=None,
            dense_cache_len=0,
        )

    try:
        top8_tokens, top8_rows, _ = run(8)
        top40_tokens, top40_rows, _ = run(40)
    finally:
        runner.close()

    assert len(top40_rows[0]) == 40
    assert top40_tokens == top8_tokens
    assert top40_rows[0][:8] == top8_rows[0]
