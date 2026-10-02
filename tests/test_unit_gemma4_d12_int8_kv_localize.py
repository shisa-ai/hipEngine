"""Unit tier for the Gemma 4 D12 INT8-vs-BF16 attention-boundary localizer.

CPU-only: the module under test keeps every device import inside functions, so
these cases run without HIP. They pin the parts the GPU diagnostic relies on but
cannot cheaply re-derive there: the frozen chain loader, the BF16 bit helpers,
the common CPU attention control (checked against the existing quantized oracle),
literal array comparisons, the structural validators that must fail loudly, the
per-layer divergence classifier, the frozen tolerances, and the overall verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4_int8 import (
    dequantize_kv_int8_per_token_head,
    gemma4_attention_decode_int8_per_token_head,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from scripts.gemma4_teacher_forced_gate import chain_sha256

from scripts.gemma4_d12_int8_kv_localize import (
    FROZEN_ARTIFACT,
    TOLERANCES,
    _build_interpretation,
    bf16_bits_to_float,
    common_cpu_attention,
    compare_kv_reconstruction,
    compare_layer_boundaries,
    compare_prefill_kv_equality,
    float_to_bf16_bits,
    judge_replay,
    literal_comparison,
    load_frozen_chain,
    overall_verdict,
    relative_l2,
    validate_decode_records,
    validate_prefill_records,
    validate_replay_layers,
    wholemodel_fixed_input_verdict,
)

_EXPECTED_CHAIN_SHA = "ac020238439bfd683dc7ec9a7200440b182277f7931cad3c83ceeae5a80d9e32"

_GEOMETRY = [
    {
        "layer_type": "sliding_attention",
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 2,
        "sliding_window": 8,
        "k_eq_v": False,
    },
    {
        "layer_type": "full_attention",
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 2,
        "sliding_window": None,
        "k_eq_v": True,
    },
]
_HIDDEN = 6


def _valid_record(entry, rows: int, *, value: int = 0) -> dict:
    qw = entry["num_heads"] * entry["head_dim"]
    kw = entry["num_kv_heads"] * entry["head_dim"]
    return {
        "rows": rows,
        "hidden_in": np.full(rows * _HIDDEN, value, np.uint16),
        "q_rot": np.full(rows * qw, value, np.uint16),
        "k_rot": np.full(rows * kw, value, np.uint16),
        "v": np.full(rows * kw, value, np.uint16),
        "context": np.full(rows * qw, value, np.uint16),
    }


def _frozen_chain():
    return load_frozen_chain(FROZEN_ARTIFACT, "prose_en_short")


def test_frozen_chain_matches_committed_artifact():
    chain = _frozen_chain()
    assert chain.name == "prose_en_short"
    assert chain.prompt_tokens == 96
    assert chain.prefill == 63
    assert chain.scored_rows == 32
    assert len(chain.prompt_ids) == 96
    assert chain.chain_sha256 == _EXPECTED_CHAIN_SHA
    assert chain.attention_geometry[0]["layer_type"] == "sliding_attention"
    assert chain.attention_geometry[5]["layer_type"] == "full_attention"
    assert chain.attention_geometry[5]["k_eq_v"] is True


def test_frozen_chain_rejects_a_tampered_hash(tmp_path: Path):
    original = json.loads(Path(FROZEN_ARTIFACT).read_text())
    for case in original["workload"]["cases"]:
        if case["name"] == "prose_en_short":
            case["prompt_ids"] = list(case["prompt_ids"])
            case["prompt_ids"][0] = int(case["prompt_ids"][0]) + 1
            break
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="chain"):
        load_frozen_chain(tampered, "prose_en_short")


def test_bf16_helpers_round_trip_against_materialize():
    rng = np.random.default_rng(1234)
    values = rng.uniform(-8.0, 8.0, size=4096).astype(np.float32)
    assert np.array_equal(float_to_bf16_bits(values), float_array_to_bf16_bits(values))
    back = bf16_bits_to_float(float_to_bf16_bits(values))
    assert np.array_equal(
        back.view(np.uint32) >> np.uint32(16), float_to_bf16_bits(values).astype(np.uint32)
    )


def test_bf16_helpers_exact_for_representable_values():
    values = np.array([0.0, 1.0, -1.0, 2.0, 0.5, -0.25], dtype=np.float32)
    assert np.array_equal(bf16_bits_to_float(float_to_bf16_bits(values)), values)


def test_relative_l2_basic():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    assert relative_l2(a, a) == 0.0
    assert relative_l2(a, b) == pytest.approx(float(np.sqrt(1.0) / np.sqrt(1.0)))
    assert relative_l2(np.zeros(3), np.zeros(3)) == 0.0


def test_literal_comparison_raw_bytes():
    a = np.array([1, 2, 3], dtype=np.uint16)
    b = np.array([1, 2, 3], dtype=np.uint16)
    c = np.array([1, 2, 4], dtype=np.uint16)
    d = np.array([1, 2, 3], dtype=np.int32)
    assert literal_comparison(a, b)["raw_bytes_equal"] is True
    assert literal_comparison(a, c)["raw_bytes_equal"] is False
    mismatch = literal_comparison(a, d)
    assert mismatch["raw_bytes_equal"] is False
    assert mismatch["dtype_match"] is False


def test_validate_replay_layers_rejects_empty_duplicate_and_out_of_range():
    assert validate_replay_layers([0, 5], 30) == [0, 5]
    with pytest.raises(ValueError, match="at least one"):
        validate_replay_layers([], 30)
    with pytest.raises(ValueError, match="duplicate"):
        validate_replay_layers([0, 0], 30)
    with pytest.raises(ValueError, match="outside"):
        validate_replay_layers([30], 30)


def test_validate_decode_records_accepts_valid():
    records = {0: _valid_record(_GEOMETRY[0], 1), 1: _valid_record(_GEOMETRY[1], 1)}
    validate_decode_records(records, _GEOMETRY, hidden_size=_HIDDEN, require_context_f32=False)


def test_validate_decode_records_rejects_missing_layer():
    records = {0: _valid_record(_GEOMETRY[0], 1)}
    with pytest.raises(ValueError, match="missing layers"):
        validate_decode_records(records, _GEOMETRY, hidden_size=_HIDDEN, require_context_f32=False)


def test_validate_decode_records_rejects_wrong_shape_and_nonfinite():
    records = {0: _valid_record(_GEOMETRY[0], 1), 1: _valid_record(_GEOMETRY[1], 1)}
    records[0]["q_rot"] = np.zeros(3, dtype=np.uint16)
    with pytest.raises(ValueError, match="elements"):
        validate_decode_records(records, _GEOMETRY, hidden_size=_HIDDEN, require_context_f32=False)

    records = {0: _valid_record(_GEOMETRY[0], 1), 1: _valid_record(_GEOMETRY[1], 1)}
    records[0]["context"][0] = np.uint16(0x7F80)  # +Inf in BF16
    with pytest.raises(ValueError, match="non-finite"):
        validate_decode_records(records, _GEOMETRY, hidden_size=_HIDDEN, require_context_f32=False)


def test_validate_decode_records_requires_context_f32_when_asked():
    records = {0: _valid_record(_GEOMETRY[0], 1), 1: _valid_record(_GEOMETRY[1], 1)}
    with pytest.raises(ValueError, match="context_f32"):
        validate_decode_records(records, _GEOMETRY, hidden_size=_HIDDEN, require_context_f32=True)


def test_validate_prefill_records_rejects_extra_layer():
    records = {0: _valid_record(_GEOMETRY[0], 3), 1: _valid_record(_GEOMETRY[1], 3)}
    with pytest.raises(ValueError, match="unexpected"):
        validate_prefill_records(
            records, _GEOMETRY, hidden_size=_HIDDEN, prefill_rows=3, required_layers=[0]
        )


def _synthetic_cache(blocks=1, block_size=8, kv_heads=2, head_dim=4, seed=7, scale_dtype=np.float16):
    rng = np.random.default_rng(seed)
    key = rng.integers(-127, 128, size=(blocks, block_size, kv_heads, head_dim)).astype(np.int8)
    value = rng.integers(-127, 128, size=(blocks, block_size, kv_heads, head_dim)).astype(np.int8)
    k_scale = rng.uniform(0.01, 0.2, size=(blocks, block_size, kv_heads)).astype(scale_dtype)
    v_scale = rng.uniform(0.01, 0.2, size=(blocks, block_size, kv_heads)).astype(scale_dtype)
    return key, value, k_scale, v_scale


def test_common_cpu_attention_matches_quantized_oracle():
    key, value, k_scale, v_scale = _synthetic_cache()
    query = np.random.default_rng(9).uniform(-2, 2, size=(4, 4)).astype(np.float32)
    context_len = 6
    block_size = key.shape[1]
    table = np.arange(key.shape[0], dtype=np.int32)
    oracle = gemma4_attention_decode_int8_per_token_head(
        query,
        key,
        value,
        k_scale,
        v_scale,
        table,
        context_len,
        block_size=block_size,
        scale=1.0,
        token_positions=np.arange(context_len, dtype=np.int64),
        row_position=context_len - 1,
    )
    dk, dv = dequantize_kv_int8_per_token_head(key, value, k_scale, v_scale)
    keys = np.stack([dk[0, slot] for slot in range(context_len)])
    values = np.stack([dv[0, slot] for slot in range(context_len)])
    mine = common_cpu_attention(
        query,
        keys,
        values,
        positions=np.arange(context_len),
        query_position=context_len - 1,
        sliding_window=None,
        scale=1.0,
        num_kv_heads=key.shape[2],
    )
    np.testing.assert_array_equal(mine, oracle)


def test_common_cpu_attention_sliding_window_and_causal_exclude_slots():
    head_dim = 2
    num_heads = 1
    keys = np.zeros((4, 1, head_dim), dtype=np.float32)
    values = np.array(
        [[[100.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32
    )
    query = np.ones((num_heads, head_dim), dtype=np.float32)
    out = common_cpu_attention(
        query,
        keys,
        values,
        positions=np.arange(4),
        query_position=3,
        sliding_window=2,
        scale=1.0,
        num_kv_heads=1,
    )
    assert np.allclose(out, 0.0)

    values2 = np.array(
        [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 50.0]]], dtype=np.float32
    )
    out2 = common_cpu_attention(
        query,
        keys,
        values2,
        positions=np.array([3, 5, 6, 7]),
        query_position=3,
        sliding_window=None,
        scale=1.0,
        num_kv_heads=1,
    )
    assert np.allclose(out2, 0.0)


def test_common_cpu_attention_all_masked_returns_zeros():
    keys = np.zeros((3, 1, 2), dtype=np.float32)
    values = np.ones((3, 1, 2), dtype=np.float32)
    query = np.ones((1, 2), dtype=np.float32)
    out = common_cpu_attention(
        query,
        keys,
        values,
        positions=np.arange(3),
        query_position=-1,
        sliding_window=None,
        scale=1.0,
        num_kv_heads=1,
    )
    assert np.allclose(out, 0.0)


def test_common_cpu_attention_gqa_maps_heads_to_kv_heads():
    head_dim = 2
    keys = np.zeros((2, 2, head_dim), dtype=np.float32)
    values = np.array(
        [
            [[5.0, 5.0], [9.0, 9.0]],
            [[6.0, 6.0], [11.0, 11.0]],
        ],
        dtype=np.float32,
    )
    query = np.zeros((4, head_dim), dtype=np.float32)
    out = common_cpu_attention(
        query,
        keys,
        values,
        positions=np.array([0, 1]),
        query_position=1,
        sliding_window=None,
        scale=1.0,
        num_kv_heads=2,
    )
    np.testing.assert_allclose(out[0], [5.5, 5.5])
    np.testing.assert_allclose(out[1], [5.5, 5.5])
    np.testing.assert_allclose(out[2], [10.0, 10.0])
    np.testing.assert_allclose(out[3], [10.0, 10.0])


def _layer_record(context, hidden, q=None, k=None, v=None):
    return {
        "hidden_in": hidden,
        "q_rot": q if q is not None else hidden,
        "k_rot": k if k is not None else hidden,
        "v": v if v is not None else hidden,
        "context": context,
    }


def test_compare_layer_boundaries_finds_first_divergence():
    bf16 = {
        0: _layer_record(np.array([1, 2, 3], np.uint16), np.array([1, 1], np.uint16)),
        1: _layer_record(np.array([4, 5, 6], np.uint16), np.array([9, 9], np.uint16)),
        2: _layer_record(np.array([7, 8, 9], np.uint16), np.array([9, 9], np.uint16)),
    }
    int8 = {
        0: _layer_record(np.array([1, 2, 4], np.uint16), np.array([1, 1], np.uint16)),
        1: _layer_record(np.array([4, 5, 6], np.uint16), np.array([9, 9], np.uint16)),
        2: _layer_record(np.array([7, 8, 9], np.uint16), np.array([9, 9], np.uint16)),
    }
    verdict = compare_layer_boundaries(bf16, int8, num_layers=3)
    assert verdict["first_divergent_output_layer"] == 0
    assert verdict["first_divergent_input_layer"] is None
    assert verdict["layers"][0]["context_equal"] is False
    assert verdict["layers"][0]["hidden_input_equal"] is True
    assert verdict["layers"][0]["context_changed_elements"] == 1


def test_compare_layer_boundaries_first_divergent_input():
    bf16 = {
        0: _layer_record(np.array([1, 2], np.uint16), np.array([1, 1], np.uint16)),
        1: _layer_record(np.array([3, 4], np.uint16), np.array([5, 5], np.uint16)),
    }
    int8 = {
        0: _layer_record(np.array([1, 2], np.uint16), np.array([1, 1], np.uint16)),
        1: _layer_record(np.array([3, 4], np.uint16), np.array([5, 6], np.uint16)),
    }
    verdict = compare_layer_boundaries(bf16, int8, num_layers=2)
    assert verdict["first_divergent_output_layer"] is None
    assert verdict["first_divergent_input_layer"] == 1
    assert verdict["layers"][1]["context_equal"] is True
    assert verdict["layers"][1]["hidden_input_equal"] is False


def test_compare_layer_boundaries_missing_layer_fails_loudly():
    bf16 = {0: _layer_record(np.array([1], np.uint16), np.array([1], np.uint16))}
    int8 = {0: _layer_record(np.array([1], np.uint16), np.array([1], np.uint16))}
    with pytest.raises(ValueError, match="missing"):
        compare_layer_boundaries(bf16, int8, num_layers=2)


def test_compare_kv_reconstruction_reports_original_vs_reconstructed():
    original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reconstructed = np.array([1.0, 2.0, 3.5], dtype=np.float32)
    report = compare_kv_reconstruction(original, original, reconstructed, reconstructed)
    assert report["slots"] == 3
    assert report["key"]["exact_equal_elements"] == 2
    assert report["key"]["original_vs_reconstructed_max_abs"] == pytest.approx(0.5)
    assert report["value"]["original_vs_reconstructed_rel_l2"] > 0.0


def test_judge_replay_accepts_within_tolerance():
    report = {
        "bf16_consumer_vs_cpu_rel_l2": 1e-4,
        "bf16_consumer_vs_cpu_max_abs": 1e-4,
        "int8_consumer_vs_cpu_rel_l2": 1e-6,
        "int8_consumer_vs_cpu_max_abs": 1e-6,
    }
    verdict = judge_replay(report, TOLERANCES)
    assert verdict["bf16_consumer_faithful"] is True
    assert verdict["int8_consumer_faithful"] is True
    assert verdict["failed"] == []


def test_judge_replay_reports_frozen_bf16_tolerance_failure():
    report = {
        "bf16_consumer_vs_cpu_rel_l2": 1e-4,
        "bf16_consumer_vs_cpu_max_abs": 0.03,
        "int8_consumer_vs_cpu_rel_l2": 1e-9,
        "int8_consumer_vs_cpu_max_abs": 1e-9,
    }
    verdict = judge_replay(report, TOLERANCES)
    assert verdict["bf16_consumer_faithful"] is False
    assert verdict["int8_consumer_faithful"] is True
    assert "bf16_consumer_vs_cpu" in verdict["failed"]


def test_overall_verdict_includes_bf16_control_failure():
    def _report(bf16_faithful: bool, raw_bytes_equal: bool = True) -> dict:
        return {
            "verdict": {"passed": bf16_faithful},
            "bf16_production_matches_rounded_cpu_raw_bytes": {
                "raw_bytes_equal": raw_bytes_equal
            },
            "replay_context_bf16_vs_f32_rounding_literal": {"raw_bytes_equal": True},
        }

    good = overall_verdict(cpu_self_check_passed=True, replay_reports={"0": _report(True)})
    assert good["passed"] is True
    bad = overall_verdict(cpu_self_check_passed=True, replay_reports={"0": _report(False)})
    assert bad["passed"] is False
    assert "replay_consumer_control:layer0" in bad["failed"]
    self_check = overall_verdict(cpu_self_check_passed=False, replay_reports={"0": _report(True)})
    assert "cpu_shared_function_self_check" in self_check["failed"]
    # Raw-byte equality against the rounded CPU control is a reported proof, not
    # a control: on a full-attention layer the GPU reduction order may round to a
    # neighbour, and that must not fail the overall verdict by itself.
    neighbour = overall_verdict(
        cpu_self_check_passed=True, replay_reports={"5": _report(True, raw_bytes_equal=False)}
    )
    assert neighbour["passed"] is True


def test_overall_verdict_fails_on_broken_bf16_rounding_invariant():
    report = {
        "verdict": {"passed": True},
        "bf16_production_matches_rounded_cpu_raw_bytes": {"raw_bytes_equal": True},
        "replay_context_bf16_vs_f32_rounding_literal": {"raw_bytes_equal": False},
    }
    verdict = overall_verdict(cpu_self_check_passed=True, replay_reports={"0": report})
    assert verdict["passed"] is False
    assert "replay_bf16_is_f32_rounding:layer0" in verdict["failed"]


def test_frozen_chain_rejects_recomputed_hash_for_tampered_ids(tmp_path: Path):
    """A tampered chain with a *recomputed* supplied hash must still be refused.

    The recorded-versus-recomputed check alone would accept this; the independent
    pinned constant is what refuses it.
    """

    original = json.loads(Path(FROZEN_ARTIFACT).read_text())
    for case in original["workload"]["cases"]:
        if case["name"] == "prose_en_short":
            ids = [int(token) for token in case["prompt_ids"]]
            ids[0] += 1
            case["prompt_ids"] = ids
            case["chain_sha256"] = chain_sha256(tuple(ids))
            break
    tampered = tmp_path / "tampered-recomputed.json"
    tampered.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="frozen chain"):
        load_frozen_chain(tampered, "prose_en_short")


@pytest.mark.parametrize(
    "field,value", [("prefill", 62), ("scored_rows", 31), ("prompt_tokens", 95)]
)
def test_frozen_chain_rejects_altered_metadata(tmp_path: Path, field: str, value: int):
    """Altered prefill / scored_rows / prompt_tokens metadata must be refused."""

    original = json.loads(Path(FROZEN_ARTIFACT).read_text())
    for case in original["workload"]["cases"]:
        if case["name"] == "prose_en_short":
            case[field] = value
            break
    tampered = tmp_path / f"tampered-{field}.json"
    tampered.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="frozen chain"):
        load_frozen_chain(tampered, "prose_en_short")


def test_frozen_chain_pins_are_the_diagnostic_constants():
    from scripts.gemma4_d12_int8_kv_localize import (
        FROZEN_PREFILL,
        FROZEN_PROMPT_TOKENS,
        FROZEN_SCORED_ROWS,
        FROZEN_CHAIN_SHA256,
    )

    assert FROZEN_CHAIN_SHA256 == _EXPECTED_CHAIN_SHA
    assert FROZEN_PROMPT_TOKENS == 96
    assert FROZEN_PREFILL == 63
    assert FROZEN_SCORED_ROWS == 32


def test_compare_prefill_kv_equality_detects_differing_prefill():
    a = {"k_rot": np.array([1, 2], np.uint16), "v": np.array([3, 4], np.uint16)}
    b = {"k_rot": np.array([1, 9], np.uint16), "v": np.array([3, 4], np.uint16)}
    report = compare_prefill_kv_equality(a, b)
    assert report["k_equal"] is False
    assert report["v_equal"] is True
    same = compare_prefill_kv_equality(a, a)
    assert same == {"k_equal": True, "v_equal": True}


def test_wholemodel_fixed_input_requires_prefill_kv_equality():
    assert wholemodel_fixed_input_verdict(True, True)["fixed_input"] is True
    differing = wholemodel_fixed_input_verdict(True, False)
    assert differing["fixed_input"] is False
    assert differing["decode_row_inputs_equal"] is True
    assert differing["prefill_kv_equal"] is False
    assert wholemodel_fixed_input_verdict(False, True)["fixed_input"] is False


def _synthetic_report(
    *,
    exact_floor: bool,
    within_floor: bool,
    bf16_faithful: bool = False,
    raw_bytes_equal: bool = True,
) -> dict:
    return {
        "verdict": {
            "int8_consumer_faithful": True,
            "bf16_consumer_faithful": bf16_faithful,
        },
        "bf16_consumer_vs_cpu_max_abs": 0.03,
        "bf16_consumer_matches_readback_floor_exactly": exact_floor,
        "bf16_consumer_within_readback_floor": within_floor,
        "bf16_production_matches_rounded_cpu_raw_bytes": {
            "raw_bytes_equal": raw_bytes_equal
        },
        "int8_replay_vs_wholemodel_context_f32_literal": {"raw_bytes_equal": False},
    }


def _synthetic_localization(*, decode_row_equal: bool) -> dict:
    return {
        "layers": {
            0: {
                "hidden_input_equal": decode_row_equal,
                "q_equal": decode_row_equal,
                "k_equal": decode_row_equal,
                "v_equal": decode_row_equal,
            }
        },
        "first_divergent_output_layer": None,
        "first_divergent_input_layer": None,
    }


def test_interpretation_prohibits_fixed_input_when_prefill_kv_differs():
    """Differing prefill K/V with equal decode-row input must not claim fixed input."""

    interpretation = _build_interpretation(
        _synthetic_localization(decode_row_equal=True),
        {"0": _synthetic_report(exact_floor=True, within_floor=True)},
        replay_layers=[0],
        prefill_kv_equality={0: {"k_equal": True, "v_equal": False}},
    )
    assert interpretation["wholemodel_decode_row_inputs_equal_at_replayed_layers"]["0"] is True
    assert interpretation["wholemodel_prefill_kv_equal_at_replayed_layers"]["0"] is False
    assert interpretation["wholemodel_fixed_input_at_replayed_layers"]["0"] is False
    assert "not a fixed-input comparison" in interpretation["notes"][-1]


def test_interpretation_claims_fixed_input_only_when_both_hold():
    interpretation = _build_interpretation(
        _synthetic_localization(decode_row_equal=True),
        {"0": _synthetic_report(exact_floor=True, within_floor=True)},
        replay_layers=[0],
        prefill_kv_equality={0: {"k_equal": True, "v_equal": True}},
    )
    assert interpretation["wholemodel_fixed_input_at_replayed_layers"]["0"] is True
    assert "a fixed-input comparison" in interpretation["notes"][-1]


def test_interpretation_floor_note_is_conditional_on_measured_flags():
    exact = _build_interpretation(
        _synthetic_localization(decode_row_equal=True),
        {"0": _synthetic_report(exact_floor=True, within_floor=True)},
        replay_layers=[0],
        prefill_kv_equality={0: {"k_equal": True, "v_equal": True}},
    )
    assert "exactly equals the BF16 readback floor" in exact["notes"][-2]
    within = _build_interpretation(
        _synthetic_localization(decode_row_equal=True),
        {"0": _synthetic_report(exact_floor=False, within_floor=True)},
        replay_layers=[0],
        prefill_kv_equality={0: {"k_equal": True, "v_equal": True}},
    )
    assert "does not exceed the recorded BF16 readback floor plus 1e-6, but does not equal the floor" in within["notes"][-2]
    # The within-floor wording must not appear when the measured flag is false.
    exceeds = _build_interpretation(
        _synthetic_localization(decode_row_equal=True),
        {"0": _synthetic_report(exact_floor=False, within_floor=False)},
        replay_layers=[0],
        prefill_kv_equality={0: {"k_equal": True, "v_equal": True}},
    )
    assert "exceeds the recorded BF16 readback floor plus 1e-6" in exceeds["notes"][-2]
    assert "is within the BF16 readback floor" not in exceeds["notes"][-2]


def test_tolerances_are_frozen():
    assert TOLERANCES == {
        "cpu_shared_function_self_check_rel_l2": 1e-6,
        "int8_consumer_vs_cpu_rel_l2": 1e-4,
        "int8_consumer_vs_cpu_max_abs": 1e-4,
        "bf16_consumer_vs_cpu_rel_l2": 5e-3,
        "bf16_consumer_vs_cpu_max_abs": 5e-3,
    }
