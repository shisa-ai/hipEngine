from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import built-ins so the registry has the Q8T16/Q8 prefill keys.
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401
import hipengine.runtime.gguf_linear as gl
from hipengine.core.dtype import DType
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_RAW_GGUF,
)


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
    }

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(
                layout=layout,
                quant_key=quant_key,
                slot_path="layers.0.test",
                source=None,
            )
            self.allocations = allocations

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def _t16_pair():
    return (
        _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"),
        _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"),
    )


def _raw_q8_pair():
    return (
        _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0"),
        _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0"),
    )


def test_concat_public_rows1_and_rows2_contract(monkeypatch) -> None:
    """Characterize the committed public contract: split at 1, concat above."""

    weight_a, weight_b = _t16_pair()
    split_calls: list[tuple[tuple, dict]] = []
    concat_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: split_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: concat_calls.append((args, kwargs)),
    )

    assert gl.launch_gguf_linear_pair_concat(
        weight_a,
        weight_b,
        100,
        200,
        1,
        2048,
        512,
        stream=7,
        runtime="runtime-sentinel",
    )
    assert gl.launch_gguf_linear_pair_concat(
        weight_a,
        weight_b,
        100,
        200,
        2,
        2048,
        512,
        stream=7,
        runtime="runtime-sentinel",
        use_wmma_prefill=False,
    )

    assert split_calls == [
        (
            (100, 14, 14, 200, 200 + 512 * DType.BF16.itemsize, 1, 2048, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]
    assert concat_calls == [
        (
            (100, 14, 14, 200, 2, 2048, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_concat_plan_rows1_split_operation_operands_and_offset(monkeypatch) -> None:
    weight_a, weight_b = _t16_pair()
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    plan = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=1,
        in_features=2048,
        out_features=512,
    )
    assert plan.operation == "q8_t16_dual_split_rows1"
    assert plan.fallback_reason is None
    assert plan.threads == 0
    assert [operand.weight for operand in plan.operands] == [weight_a, weight_b]
    assert gl._execute_gguf_linear_pair_concat_plan(
        plan,
        weight_a,
        weight_b,
        100,
        200,
        stream=7,
        runtime="runtime-sentinel",
    )

    assert calls == [
        (
            (100, 14, 14, 200, 200 + 512 * DType.BF16.itemsize, 1, 2048, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_concat_plan_rows2_uses_row_strided_concat_without_offset(monkeypatch) -> None:
    weight_a, weight_b = _t16_pair()
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    plan = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=2,
        in_features=2048,
        out_features=512,
        use_wmma_prefill=False,
        threads=128,
    )
    assert plan.operation == "q8_t16_dual_concat_rows_gt1"
    assert plan.fallback_reason is None
    assert plan.threads == 128
    assert gl._execute_gguf_linear_pair_concat_plan(
        plan,
        weight_a,
        weight_b,
        100,
        200,
        stream=7,
        runtime="runtime-sentinel",
    )

    assert calls == [
        (
            (100, 14, 14, 200, 2, 2048, 512, 512),
            {"threads": 128, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_concat_plan_rows2_wmma_declines_but_rows1_still_splits(monkeypatch) -> None:
    weight_a, weight_b = _t16_pair()
    monkeypatch.setattr(
        gl,
        "_dispatch_can_use_t16_wmma_prefill",
        lambda *args, **kwargs: True,
    )

    declined = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=2,
        in_features=2048,
        out_features=512,
        use_wmma_prefill=True,
    )
    assert declined.operation is None
    assert declined.fallback_reason == "wmma_prefill_declined"
    assert not gl._execute_gguf_linear_pair_concat_plan(
        declined, weight_a, weight_b, 100, 200
    )

    split = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=1,
        in_features=2048,
        out_features=512,
        use_wmma_prefill=True,
    )
    assert split.operation == "q8_t16_dual_split_rows1"
    assert split.fallback_reason is None


def test_concat_plan_unrelated_rows_and_shapes_fall_back(monkeypatch) -> None:
    raw_a, raw_b = _raw_q8_pair()
    rows1 = gl._resolve_gguf_linear_pair_concat_plan(
        raw_a,
        raw_b,
        rows=1,
        in_features=2048,
        out_features=4096,
        use_wmma_prefill=True,
    )
    assert rows1.operation is None
    assert rows1.fallback_reason == "unsupported_concat_dispatch"
    assert not gl._execute_gguf_linear_pair_concat_plan(
        rows1, raw_a, raw_b, 100, 300
    )

    monkeypatch.setattr(
        gl,
        "_wmma_prefill_shape_supported",
        lambda quant, in_features: False,
    )
    bad_shape = gl._resolve_gguf_linear_pair_concat_plan(
        raw_a,
        raw_b,
        rows=512,
        in_features=1000,
        out_features=4096,
        use_wmma_prefill=True,
    )
    assert bad_shape.operation is None
    assert bad_shape.fallback_reason == "unsupported_concat_dispatch"

    q4_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    q4_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    unrelated = gl._resolve_gguf_linear_pair_concat_plan(
        q4_a,
        q4_b,
        rows=512,
        in_features=2048,
        out_features=4096,
        use_wmma_prefill=True,
    )
    assert unrelated.operation is None
    assert unrelated.fallback_reason == "unsupported_concat_dispatch"


def test_concat_plan_raw_q8_wmma_exact_kwargs(monkeypatch) -> None:
    weight_a, weight_b = _raw_q8_pair()
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    plan = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=512,
        in_features=2048,
        out_features=4096,
        use_wmma_prefill=True,
    )
    assert plan.operation == "q8_wmma_prefill_dual_gate_up"
    assert plan.fallback_reason is None
    assert plan.threads == 0
    assert gl._execute_gguf_linear_pair_concat_plan(
        plan,
        weight_a,
        weight_b,
        100,
        300,
        stream=7,
        runtime="runtime-sentinel",
    )

    assert calls == [
        (
            (100, 10, 10, 300, 512, 2048, 4096, 4096),
            {
                "tile_m": 16,
                "tile_n": 32,
                "stream": 7,
                "runtime": "runtime-sentinel",
            },
        )
    ]


def test_concat_plan_resolves_callable_and_pointers_at_execution(monkeypatch) -> None:
    weight_a, weight_b = _t16_pair()
    stale_calls: list[tuple[tuple, dict]] = []
    fresh_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: stale_calls.append((args, kwargs)),
    )
    plan = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=1,
        in_features=2048,
        out_features=512,
    )
    weight_a.allocations["tiles"].tensor.ptr = 777
    weight_b.allocations["tiles"].tensor.ptr = 888
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: fresh_calls.append((args, kwargs)),
    )

    assert gl._execute_gguf_linear_pair_concat_plan(
        plan,
        weight_a,
        weight_b,
        100,
        200,
        stream=7,
        runtime="runtime-sentinel",
    )
    assert stale_calls == []
    assert fresh_calls == [
        (
            (100, 777, 888, 200, 200 + 512 * DType.BF16.itemsize, 1, 2048, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_concat_plan_rejects_foreign_weight_owner(monkeypatch) -> None:
    weight_a, weight_b = _t16_pair()
    foreign_b = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16,
        quant_key="gguf_q8_0_t16_v1",
    )
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    plan = gl._resolve_gguf_linear_pair_concat_plan(
        weight_a,
        weight_b,
        rows=1,
        in_features=2048,
        out_features=512,
    )

    with pytest.raises(ValueError, match="operand 1 owner changed"):
        gl._execute_gguf_linear_pair_concat_plan(
            plan, weight_a, foreign_b, 100, 200
        )
    assert calls == []
