"""GPU route for the D12 attention-boundary localizer's replay plumbing.

Guarded on HIP availability so a no-ROCm runner skips rather than fails. It
drives ``replay_fixed_input_layer`` -- the helper the diagnostic uses to feed one
captured common BF16 K/V input through the registered INT8 writer and consumer --
on a synthetic two-block sequence, and checks the reconstruction against the
existing quantized CPU oracle. This is the plumbing control, not the model
measurement.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.dtype import DType
# Importing the writer module registers its kernels at collection time, before
# the shared conftest captures the registry baseline; the per-test restore would
# otherwise drop them for every test after the first.
from hipengine.kernels.hip_gfx1100.attention import paged_kv_write  # noqa: F401
from hipengine.kernels.cpu_reference.gemma4_int8 import (
    gemma4_attention_decode_int8_per_token_head,
)
from hipengine.kernels.cpu_reference.ops import dequantize_kv_int8_per_token_head
from hipengine.loading.materialize import float_array_to_bf16_bits

from scripts.gemma4_d12_int8_kv_localize import (
    TOLERANCES,
    _gather_dequantized,
    bf16_bits_to_float,
    common_cpu_attention,
    compare_replay,
    replay_fixed_input_layer,
)

# layer 0: sliding (16 q / 8 kv heads, head_dim 256); layer 5: full (16 / 2, 512,
# k_eq_v). The same geometries the model's artifact declares.
_LAYERS = {
    0: {
        "layer_type": "sliding_attention",
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 256,
        "sliding_window": 1024,
        "k_eq_v": False,
    },
    5: {
        "layer_type": "full_attention",
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 512,
        "sliding_window": None,
        "k_eq_v": True,
    },
}
_ATTENTIONS = tuple(
    (
        (_LAYERS[5] if index == 5 else _LAYERS[0])["num_heads"],
        (_LAYERS[5] if index == 5 else _LAYERS[0])["num_kv_heads"],
        (_LAYERS[5] if index == 5 else _LAYERS[0])["head_dim"],
    )
    for index in range(6)
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture(scope="module")
def runtime():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _record(rows: int, entry: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    heads, kv_heads, head_dim = entry["num_heads"], entry["num_kv_heads"], entry["head_dim"]
    q = rng.uniform(-2.0, 2.0, size=(rows, heads, head_dim)).astype(np.float32)
    k = rng.uniform(-6.0, 6.0, size=(rows, kv_heads, head_dim)).astype(np.float32)
    v = rng.uniform(-6.0, 6.0, size=(rows, kv_heads, head_dim)).astype(np.float32)
    return {
        "rows": rows,
        "q_rot": float_array_to_bf16_bits(q).reshape(-1),
        "k_rot": float_array_to_bf16_bits(k).reshape(-1),
        "v": float_array_to_bf16_bits(v).reshape(-1),
    }


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 5])
def test_replay_two_block_sequence_matches_quantized_oracle(runtime, layer: int) -> None:
    entry = _LAYERS[layer]
    prefill_rows = 8
    prefill = _record(prefill_rows, entry, seed=0xD12 + layer)
    decode = _record(1, entry, seed=0xD20 + layer)

    replay = replay_fixed_input_layer(
        layer=layer,
        geometry_entry=entry,
        attentions=_ATTENTIONS,
        prefill_record=prefill,
        decode_record=decode,
        runtime=runtime,
        block_size=256,
        capacity=256,
        max_block=64,
        scale_dtype=DType.FP16,
    )

    assert replay["context_f32"] is not None and replay["context_bf16"] is not None
    assert np.isfinite(replay["context_f32"]).all()
    # The layer's BF16 context is the BF16 rounding of the FP32 consumer output.
    np.testing.assert_array_equal(
        float_array_to_bf16_bits(replay["context_f32"]), replay["context_bf16"]
    )

    # The common CPU algorithm on the quantized representation must reproduce the
    # existing oracle on the same cache.
    context = prefill_rows + 1
    keys_int8, values_int8 = _gather_dequantized(
        replay["key_cache"], replay["value_cache"], replay["k_scale"], replay["v_scale"],
        context=context, block_size=replay["block_size"],
    )
    q_f32 = bf16_bits_to_float(decode["q_rot"]).reshape(entry["num_heads"], entry["head_dim"])
    cpu_int8 = common_cpu_attention(
        q_f32, keys_int8, values_int8,
        positions=np.arange(context), query_position=prefill_rows,
        sliding_window=entry["sliding_window"], scale=1.0, num_kv_heads=entry["num_kv_heads"],
    )
    oracle = gemma4_attention_decode_int8_per_token_head(
        q_f32,
        replay["key_cache"],
        replay["value_cache"],
        replay["k_scale"],
        replay["v_scale"],
        np.arange(replay["blocks"], dtype=np.int32),
        context,
        block_size=replay["block_size"],
        scale=1.0,
        token_positions=np.arange(context),
        row_position=prefill_rows,
        sliding_window=entry["sliding_window"],
    )
    np.testing.assert_array_equal(cpu_int8, oracle)

    # The production consumer matches the common CPU algorithm within tolerance.
    prod = np.asarray(replay["context_f32"]).reshape(entry["num_heads"], entry["head_dim"])
    rel = float(np.linalg.norm(prod - cpu_int8) / np.linalg.norm(cpu_int8))
    assert rel <= TOLERANCES["int8_consumer_vs_cpu_rel_l2"]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_compare_replay_reports_all_quantities(runtime) -> None:
    entry = _LAYERS[0]
    prefill = _record(8, entry, seed=0xABC)
    decode = _record(1, entry, seed=0xABD)
    replay = replay_fixed_input_layer(
        layer=0,
        geometry_entry=entry,
        attentions=_ATTENTIONS,
        prefill_record=prefill,
        decode_record=decode,
        runtime=runtime,
        block_size=256,
        capacity=256,
        max_block=64,
        scale_dtype=DType.FP16,
    )
    # A synthetic BF16 "production" context: the BF16 rounding of the CPU
    # control on the same BF16 input, so it is within its own readback floor.
    bf16_decode = dict(decode)
    q_f32 = bf16_bits_to_float(decode["q_rot"]).reshape(entry["num_heads"], entry["head_dim"])
    kp = bf16_bits_to_float(prefill["k_rot"]).reshape(8, entry["num_kv_heads"], entry["head_dim"])
    vp = bf16_bits_to_float(prefill["v"]).reshape(8, entry["num_kv_heads"], entry["head_dim"])
    kd = bf16_bits_to_float(decode["k_rot"]).reshape(1, entry["num_kv_heads"], entry["head_dim"])
    vd = bf16_bits_to_float(decode["v"]).reshape(1, entry["num_kv_heads"], entry["head_dim"])
    keys_bf16 = np.concatenate([kp, kd])
    values_bf16 = np.concatenate([vp, vd])
    cpu_bf16 = common_cpu_attention(
        q_f32, keys_bf16, values_bf16,
        positions=np.arange(9), query_position=8,
        sliding_window=entry["sliding_window"], scale=1.0, num_kv_heads=entry["num_kv_heads"],
    )
    bf16_decode["context"] = float_array_to_bf16_bits(cpu_bf16).reshape(-1)
    report = compare_replay(
        replay, bf16_prefill=prefill, bf16_decode=bf16_decode,
        int8_decode_context_f32=replay["context_f32"],
    )
    assert (
        report["cpu_shared_function_self_check_rel_l2"]
        <= TOLERANCES["cpu_shared_function_self_check_rel_l2"]
    )
    assert report["int8_replay_vs_wholemodel_context_f32_rel_l2"] == 0.0
    assert report["replay_context_bf16_vs_f32_rounding_literal"]["raw_bytes_equal"] is True
    assert (
        report["int8_replay_vs_wholemodel_context_f32_literal"]["raw_bytes_equal"] is True
    )
    assert report["kv_reconstruction"]["slots"] == 9
    for key in (
        "quantization_cpu_bf16_vs_cpu_int8_rel_l2",
        "fixed_input_bf16_vs_int8_bf16_rel_l2",
        "bf16_consumer_vs_cpu_rel_l2",
        "int8_consumer_vs_cpu_rel_l2",
        "bf16_readback_rounding_floor_rel_l2",
        "bf16_readback_rounding_floor_max_abs",
    ):
        assert key in report
        assert np.isfinite(report[key])
    assert isinstance(report["bf16_consumer_within_readback_floor"], bool)
    # The synthetic BF16 "production" context was the CPU BF16 result, so it is
    # within its own rounding floor and matches it by literal raw bytes.
    assert report["bf16_consumer_within_readback_floor"] is True
    assert report["bf16_consumer_matches_readback_floor_exactly"] is True
    assert report["bf16_production_matches_rounded_cpu_raw_bytes"]["raw_bytes_equal"] is True
