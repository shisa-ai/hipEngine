from __future__ import annotations

from pathlib import Path

import numpy as np

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.kernels.cpu_reference import (
    dequantize_kv_int8_per_token_head,
    full_attn_prefill,
    full_attn_prefill_varlen,
    gdn_prefill_recurrent_segments,
    linear_attn_conv_prefill_segments,
    load_fixture,
    paged_attn_decode_int8_per_token_head,
    quantize_kv_int8_per_token_head,
    register_cpu_reference_kernels,
    rmsnorm,
    rotate,
    run_fixture,
    write_paged_kv_int8_per_token_head,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _round_to_bf16_float(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


FIXTURE = Path("tests/fixtures/cpu_reference/rmsnorm_basic.json")
FIXTURE_DIR = Path("tests/fixtures/cpu_reference")


def setup_function() -> None:
    clear_registry_for_tests()
    register_cpu_reference_kernels()


def test_cpu_reference_rmsnorm_matches_manual_formula() -> None:
    x = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight = np.asarray([1.0, 0.5, -1.0, 2.0], dtype=np.float32)
    expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight

    assert np.allclose(rmsnorm(x, weight), expected, atol=1e-6, rtol=1e-6)


def test_cpu_reference_rotary_split_half() -> None:
    x = np.asarray([[1.0, 2.0, 10.0, 20.0, 99.0]], dtype=np.float32)
    cos = np.asarray([0.0, 1.0], dtype=np.float32)
    sin = np.asarray([1.0, 0.0], dtype=np.float32)

    out = rotate(x, cos, sin, rotary_dim=4)

    assert np.allclose(out, np.asarray([[-10.0, 2.0, 1.0, 20.0, 99.0]], dtype=np.float32))


def test_cpu_reference_kernels_register_and_resolve() -> None:
    fn = resolve(backend="hip_gfx1100", layer="rmsnorm", quant="fp16")
    prefill = resolve(
        backend="hip_gfx1100",
        layer="full_attn_prefill",
        quant="w4_paro",
        variant="qwen35_causal_gqa_gate_fp16",
    )

    int8_decode = resolve(backend="cpu_reference", layer="paged_attn_decode", quant="int8_per_token_head")

    assert fn is rmsnorm
    assert prefill is full_attn_prefill
    assert int8_decode is paged_attn_decode_int8_per_token_head


def test_cpu_reference_int8_kv_quantizes_per_token_head_and_handles_zero_rows() -> None:
    key = np.asarray(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, -0.5]],
            [[1.0, 2.0, -1.0, -2.0], [4.0, -4.0, 2.0, -2.0]],
        ],
        dtype=np.float32,
    )
    value = key * np.asarray([[[2.0]], [[0.25]]], dtype=np.float32)

    qk, qv, k_scale, v_scale = quantize_kv_int8_per_token_head(key, value)
    key_deq, value_deq = dequantize_kv_int8_per_token_head(qk, qv, k_scale, v_scale)

    assert qk.dtype == np.int8
    assert qv.dtype == np.int8
    assert k_scale.shape == key.shape[:-1]
    assert v_scale.shape == value.shape[:-1]
    assert k_scale[0, 0] == 0.0
    assert np.count_nonzero(qk[0, 0]) == 0
    assert np.isclose(k_scale[1, 0], 2.0 / 127.0)
    assert np.isclose(k_scale[1, 1], 4.0 / 127.0)
    assert k_scale[1, 0] != k_scale[1, 1]
    assert np.max(np.abs(key_deq - key)) <= float(np.max(k_scale) / 2.0 + 1e-7)
    assert np.max(np.abs(value_deq - value)) <= float(np.max(v_scale) / 2.0 + 1e-7)


def test_cpu_reference_int8_kv_write_respects_page_boundaries() -> None:
    key_rows = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    value_rows = key_rows.copy()

    key_cache, value_cache, k_scale, v_scale = write_paged_kv_int8_per_token_head(
        key_rows,
        value_rows,
        positions=np.asarray([0, 1, 2], dtype=np.int64),
        block_table=np.asarray([1, 0], dtype=np.int32),
        block_size=2,
    )
    key_deq, value_deq = dequantize_kv_int8_per_token_head(key_cache, value_cache, k_scale, v_scale)

    assert np.allclose(key_deq[1, 0], key_rows[0])
    assert np.allclose(value_deq[1, 1], value_rows[1])
    assert np.allclose(key_deq[0, 0], key_rows[2])
    assert np.count_nonzero(key_cache[0, 1]) == 0
    assert np.count_nonzero(value_cache[0, 1]) == 0


def test_cpu_reference_int8_paged_attention_matches_dequantized_oracle() -> None:
    key_cache = np.asarray([[[[2, 2]], [[0, 0]]], [[[2, 0]], [[0, 2]]]], dtype=np.int8)
    value_cache = np.asarray([[[[10, 12]], [[0, 0]]], [[[2, 4]], [[6, 8]]]], dtype=np.int8)
    k_scale = np.asarray([[[0.5], [0.0]], [[0.5], [0.5]]], dtype=np.float32)
    v_scale = np.asarray([[[0.5], [0.0]], [[0.5], [0.5]]], dtype=np.float32)

    out = paged_attn_decode_int8_per_token_head(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        live_counts=np.asarray([3], dtype=np.int64),
        block_table=np.asarray([1, 0], dtype=np.int32),
        block_size=2,
        scale=1.0,
    )
    dense_out = paged_attn_decode_int8_per_token_head(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[[2, 0]], [[0, 2]], [[2, 2]]], dtype=np.int8),
        np.asarray([[[2, 4]], [[6, 8]], [[10, 12]]], dtype=np.int8),
        np.asarray([[0.5], [0.5], [0.5]], dtype=np.float32),
        np.asarray([[0.5], [0.5], [0.5]], dtype=np.float32),
        live_counts=np.asarray([3], dtype=np.int64),
        scale=1.0,
    )

    assert np.allclose(out, np.asarray([[3.0, 4.0]], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(dense_out, out, atol=1e-6, rtol=1e-6)


def test_cpu_reference_full_attn_prefill_causal_gqa_gate() -> None:
    query = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    gate = np.zeros_like(query, dtype=np.float16)
    key = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    value = np.asarray(
        [
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 4.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )

    out = full_attn_prefill(
        query,
        gate,
        _bf16_bits(key).reshape(1, 2, 2, 2),
        _bf16_bits(value).reshape(1, 2, 2, 2),
        np.asarray([0, 1], dtype=np.int64),
        context_counts=np.asarray([1, 2], dtype=np.int64),
        block_table=np.asarray([0], dtype=np.int32),
        block_size=2,
        scale=1.0,
        output_dtype=np.float32,
    )

    softmax_10 = np.asarray([np.exp(1.0), 1.0], dtype=np.float32) / (np.exp(1.0) + 1.0)
    attn_head0 = _round_to_bf16_float(
        np.asarray(
            [softmax_10[0] * 1.0 + softmax_10[1] * 3.0, softmax_10[0] * 2.0 + softmax_10[1] * 4.0],
            dtype=np.float32,
        )
    )
    attn_head1 = _round_to_bf16_float(
        np.asarray(
            [softmax_10[0] * 10.0 + softmax_10[1] * 30.0, softmax_10[0] * 20.0 + softmax_10[1] * 40.0],
            dtype=np.float32,
        )
    )
    expected = np.asarray(
        [
            [[0.5, 1.0], [0.5, 1.0], [5.0, 10.0], [5.0, 10.0]],
            [
                [1.0, 1.5],
                [attn_head0[0] * 0.5, attn_head0[1] * 0.5],
                [10.0, 15.0],
                [attn_head1[0] * 0.5, attn_head1[1] * 0.5],
            ],
        ],
        dtype=np.float32,
    )

    assert np.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_cpu_reference_full_attn_prefill_varlen_is_block_diagonal() -> None:
    query = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, -1.0]],
            [[-1.0, 1.0], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    gate = np.zeros_like(query, dtype=np.float16)
    key = np.zeros((2, 4, 1, 2), dtype=np.float32)
    value = np.zeros_like(key)
    key[0, :2, 0] = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    value[0, :2, 0] = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    key[1, :2, 0] = np.asarray([[0.5, 0.5], [-0.5, 0.5]], dtype=np.float32)
    value[1, :2, 0] = np.asarray([[100.0, 1000.0], [200.0, 2000.0]], dtype=np.float32)
    block_tables = np.asarray([[0], [0], [1], [1]], dtype=np.int32)

    out = full_attn_prefill_varlen(
        query,
        gate,
        _bf16_bits(key),
        _bf16_bits(value),
        positions=np.asarray([0, 1, 0, 1], dtype=np.int64),
        cu_seqlens_q=np.asarray([0, 2, 4], dtype=np.int32),
        cu_seqlens_k=np.asarray([0, 2, 4], dtype=np.int32),
        context_counts=np.asarray([1, 2, 1, 2], dtype=np.int64),
        block_tables=block_tables,
        block_size=4,
        scale=1.0,
        output_dtype=np.float32,
    )

    seg0 = full_attn_prefill(
        query[:2],
        gate[:2],
        _bf16_bits(key[:1]),
        _bf16_bits(value[:1]),
        np.asarray([0, 1], dtype=np.int64),
        context_counts=np.asarray([1, 2], dtype=np.int64),
        block_table=np.asarray([0], dtype=np.int32),
        block_size=4,
        scale=1.0,
        output_dtype=np.float32,
    )
    seg1 = full_attn_prefill(
        query[2:],
        gate[2:],
        _bf16_bits(key[1:2]),
        _bf16_bits(value[1:2]),
        np.asarray([0, 1], dtype=np.int64),
        context_counts=np.asarray([1, 2], dtype=np.int64),
        block_table=np.asarray([0], dtype=np.int32),
        block_size=4,
        scale=1.0,
        output_dtype=np.float32,
    )

    assert np.allclose(out[:2], seg0, atol=1e-5, rtol=1e-5)
    assert np.allclose(out[2:], seg1, atol=1e-5, rtol=1e-5)
    assert np.max(out[2:]) > 50.0


def test_cpu_reference_linear_attn_conv_prefill_segments_isolates_state() -> None:
    hidden = np.asarray(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6], [0.7, -0.8], [0.9, 1.0]],
        dtype=np.float32,
    )
    state = np.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]],
            [[0.5, 0.25, -0.25, -0.5], [0.75, -0.75, 1.25, -1.25]],
            [[2.0, 1.0, 0.0, -1.0], [-2.0, -1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    weight = np.asarray([[0.25, -0.5, 0.75, 1.0], [-1.0, 0.5, 0.25, -0.25]], dtype=np.float32)

    out, new_state = linear_attn_conv_prefill_segments(
        hidden,
        state,
        weight,
        np.asarray([0, 2, 5], dtype=np.int32),
        np.asarray([2, 0], dtype=np.int64),
    )

    isolated0, state0 = linear_attn_conv_prefill_segments(hidden[:2], state[[2]], weight, [0, 2], [0])
    isolated1, state1 = linear_attn_conv_prefill_segments(hidden[2:], state[[0]], weight, [0, 3], [0])
    assert np.allclose(out[:2], isolated0, atol=1e-6, rtol=1e-6)
    assert np.allclose(out[2:], isolated1, atol=1e-6, rtol=1e-6)
    assert np.allclose(new_state[2], state0[0], atol=1e-6, rtol=1e-6)
    assert np.allclose(new_state[0], state1[0], atol=1e-6, rtol=1e-6)
    assert np.allclose(new_state[1], state[1], atol=1e-6, rtol=1e-6)


def test_cpu_reference_gdn_prefill_recurrent_segments_isolates_state() -> None:
    rng = np.random.default_rng(123)
    query = rng.normal(0.0, 0.05, size=(5, 2, 4)).astype(np.float32)
    key = rng.normal(0.0, 0.05, size=(5, 2, 4)).astype(np.float32)
    value = rng.normal(0.0, 0.05, size=(5, 2, 3)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=(5, 2)).astype(np.float32)
    decay = rng.uniform(0.8, 0.99, size=(5, 2)).astype(np.float32)
    state = rng.normal(0.0, 0.02, size=(3, 2, 4, 3)).astype(np.float32)

    out, new_state = gdn_prefill_recurrent_segments(query, key, value, beta, decay, state, [0, 2, 5], [2, 0])
    isolated0, state0 = gdn_prefill_recurrent_segments(
        query[:2], key[:2], value[:2], beta[:2], decay[:2], state[[2]], [0, 2], [0]
    )
    isolated1, state1 = gdn_prefill_recurrent_segments(
        query[2:], key[2:], value[2:], beta[2:], decay[2:], state[[0]], [0, 3], [0]
    )

    assert np.allclose(out[:2], isolated0, atol=1e-6, rtol=1e-6)
    assert np.allclose(out[2:], isolated1, atol=1e-6, rtol=1e-6)
    assert np.allclose(new_state[2], state0[0], atol=1e-6, rtol=1e-6)
    assert np.allclose(new_state[0], state1[0], atol=1e-6, rtol=1e-6)
    assert np.allclose(new_state[1], state[1], atol=1e-6, rtol=1e-6)


def test_json_layer_fixture_round_trips_and_runs() -> None:
    fixture = load_fixture(FIXTURE)
    result = run_fixture(fixture)

    assert fixture.name == "rmsnorm_basic"
    assert result.passed
    assert result.max_abs <= 1e-6


def test_all_committed_cpu_reference_fixtures_pass() -> None:
    fixture_paths = sorted(FIXTURE_DIR.glob("*.json"))

    assert {path.name for path in fixture_paths} == {
        "attention_decode_masked.json",
        "full_attn_prefill_causal_gqa_gate.json",
        "kv_int8_dequant_per_token_head.json",
        "linear_basic.json",
        "paged_attn_decode_int8_per_token_head.json",
        "rmsnorm_basic.json",
        "rotate_split_half.json",
    }
    for path in fixture_paths:
        assert run_fixture(load_fixture(path)).passed, path


def test_logit_correctness_metrics_pass_and_fail() -> None:
    reference = np.asarray([[3.0, 1.0, -1.0], [0.1, 0.2, 0.3]], dtype=np.float32)
    candidate_ok = reference + np.asarray([[0.01, -0.01, 0.0], [0.0, 0.01, -0.01]])
    candidate_bad = np.asarray([[-1.0, 1.0, 3.0], [0.3, 0.2, 0.1]], dtype=np.float32)

    ok = evaluate_logits(reference, candidate_ok)
    bad = evaluate_logits(reference, candidate_bad)

    assert ok.passed
    assert ok.top1_agreement == 1.0
    assert not bad.passed
    assert bad.top1_agreement == 0.0
