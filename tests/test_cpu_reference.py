from __future__ import annotations

from pathlib import Path

import numpy as np

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.kernels.cpu_reference import (
    full_attn_prefill,
    load_fixture,
    register_cpu_reference_kernels,
    rmsnorm,
    rotate,
    run_fixture,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


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

    assert fn is rmsnorm
    assert prefill is full_attn_prefill


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
    expected = np.asarray(
        [
            [[0.5, 1.0], [0.5, 1.0], [5.0, 10.0], [5.0, 10.0]],
            [
                [1.0, 1.5],
                [
                    (softmax_10[0] * 1.0 + softmax_10[1] * 3.0) * 0.5,
                    (softmax_10[0] * 2.0 + softmax_10[1] * 4.0) * 0.5,
                ],
                [10.0, 15.0],
                [
                    (softmax_10[0] * 10.0 + softmax_10[1] * 30.0) * 0.5,
                    (softmax_10[0] * 20.0 + softmax_10[1] * 40.0) * 0.5,
                ],
            ],
        ],
        dtype=np.float32,
    )

    assert np.allclose(out, expected, atol=1e-5, rtol=1e-5)


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
        "linear_basic.json",
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
