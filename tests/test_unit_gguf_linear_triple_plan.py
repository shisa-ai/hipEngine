"""Focused RF-10 tests for private GGUF triple launch planning."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gl
from hipengine.kernels.registry import KernelKey, register, resolve, unregister
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q8_0_T16,
)
from hipengine.runtime.gguf_linear import (
    GGUF_TRIPLE_FALLBACK_TRIPLE_OWNER_UNREGISTERED,
    GGUF_TRIPLE_FALLBACK_UNSUPPORTED_OPERANDS,
    GGUF_TRIPLE_FALLBACK_WMMA_PREFILL_DECLINED,
    GGUF_TRIPLE_OPERATION_EXACT,
    GGUF_TRIPLE_OPERATION_ROWTILE4,
    launch_gguf_linear_triple,
)

Q8_QUANT = "gguf_q8_0_t16_v1"
Q8_SLOTS = ("layers.0.attn_q", "layers.0.attn_k", "layers.0.attn_v")


def _tiles_shape(out_features: int, in_features: int) -> tuple[int, int, int]:
    return (out_features // 16, in_features // 32, 544)


def _fake_weight(
    *,
    layout: str,
    quant_key: str,
    tiles_ptr: int = 14,
    backend: str | None = None,
    slot_path: str | None = "layers.0.attn_q",
    source_shape: tuple[int, int] | None = (1024, 2048),
    tiles_shape: tuple[int, ...] | None = None,
    tiles_nbytes: int | None = None,
):
    if tiles_shape is None:
        tiles_shape = (
            _tiles_shape(*source_shape)
            if layout == LAYOUT_GGUF_Q8_0_T16 and source_shape is not None
            else (1, 1, 544)
        )
    if tiles_nbytes is None:
        tiles_nbytes = math.prod(tiles_shape)
    allocations = {
        "tiles": SimpleNamespace(
            tensor=SimpleNamespace(ptr=tiles_ptr, shape=tiles_shape),
            buffer=SimpleNamespace(nbytes=tiles_nbytes),
        ),
    }
    source = SimpleNamespace(shape=source_shape) if source_shape is not None else None
    spec = SimpleNamespace(
        layout=layout,
        quant_key=quant_key,
        slot_path=slot_path,
        source=source,
    )

    class Weight:
        def __init__(self) -> None:
            self.spec = spec
            self.allocations = allocations
            if backend is not None:
                self.backend = backend

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def _q8_weights(
    *,
    tiles_ptrs: tuple[int, int, int] = (14, 14, 14),
    backend: str | None = None,
    slots: tuple[str | None, str | None, str | None] = Q8_SLOTS,
    source_shapes: tuple[
        tuple[int, int] | None,
        tuple[int, int] | None,
        tuple[int, int] | None,
    ] = ((1024, 2048), (1024, 2048), (1024, 2048)),
    tiles_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None,
    tiles_nbytes: tuple[int, int, int] | int | None = None,
):
    if tiles_shapes is None:
        tiles_shapes = tuple(
            _tiles_shape(*shape) if shape is not None else (1, 1, 544)
            for shape in source_shapes
        )
    if tiles_nbytes is None:
        tiles_nbytes = (None, None, None)
    elif not isinstance(tiles_nbytes, tuple):
        tiles_nbytes = (tiles_nbytes, tiles_nbytes, tiles_nbytes)
    return tuple(
        _fake_weight(
            layout=LAYOUT_GGUF_Q8_0_T16,
            quant_key=Q8_QUANT,
            tiles_ptr=tiles_ptrs[index],
            backend=backend,
            slot_path=slots[index],
            source_shape=source_shapes[index],
            tiles_shape=tiles_shapes[index],
            tiles_nbytes=tiles_nbytes[index],
        )
        for index in range(3)
    )


def _capture_triple(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple, dict]]:
    calls: list[tuple[str, tuple, dict]] = []

    def capture(name: str):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))

        return fake

    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out",
        capture("exact"),
    )
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out",
        capture("rowtile4"),
    )
    return calls


def _resolve(weights, *, rows=1, in_features=2048, out_features=1024, **kwargs):
    return gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        use_wmma_prefill=False,
        rowtile_all=False,
        **kwargs,
    )


def _execute(plan, weights):
    return gl._execute_gguf_linear_triple_plan(
        plan, *weights, 100, 200, 300, 400
    )


def test_triple_plan_below_rowtile_threshold_selects_exact(monkeypatch) -> None:
    weights = _q8_weights(
        source_shapes=((1024, 2048), (512, 2048), (512, 2048)),
    )
    calls = _capture_triple(monkeypatch)

    plan = gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=1,
        in_features=2048,
        out_features=1024,
        out_features_b=512,
        out_features_c=512,
        threads=0,
        use_wmma_prefill=False,
        rowtile_all=True,
    )

    assert plan.operation == GGUF_TRIPLE_OPERATION_EXACT
    assert plan.threads == 0
    assert plan.fallback_reason is None
    assert plan.registry_generation >= 0
    assert plan.operands[0].owner_slot == "layers.0.attn_q"
    assert plan.owner_key == KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_triple_gemv_decode_bf16_bf16_out",
    )
    assert gl._execute_gguf_linear_triple_plan(
        plan,
        *weights,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        out_c_ptr=400,
        stream=7,
        runtime="runtime-sentinel",
    ) is True
    assert calls == [
        (
            "exact",
            (100, 14, 14, 14, 200, 300, 400, 1, 2048, 1024, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_triple_plan_above_rowtile_threshold_selects_rowtile4(monkeypatch) -> None:
    weights = _q8_weights(
        source_shapes=((4096, 2048), (1024, 2048), (1024, 2048)),
    )
    calls = _capture_triple(monkeypatch)

    plan = gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=3,
        in_features=2048,
        out_features=4096,
        out_features_b=1024,
        out_features_c=1024,
        threads=0,
        use_wmma_prefill=False,
        rowtile_all=True,
    )

    assert plan.operation == GGUF_TRIPLE_OPERATION_ROWTILE4
    assert plan.threads == 128
    assert plan.fallback_reason is None
    assert gl._execute_gguf_linear_triple_plan(
        plan,
        *weights,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        out_c_ptr=400,
        stream=7,
        runtime="runtime-sentinel",
    ) is True
    assert calls == [
        (
            "rowtile4",
            (100, 14, 14, 14, 200, 300, 400, 3, 2048, 4096, 1024, 1024),
            {"threads": 128, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_triple_plan_unrelated_rows_keep_exact_owner(monkeypatch) -> None:
    weights = _q8_weights(
        source_shapes=((1024, 5120), (512, 5120), (512, 5120)),
    )
    calls = _capture_triple(monkeypatch)

    plan = gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=5,
        in_features=5120,
        out_features=1024,
        out_features_b=512,
        out_features_c=512,
        threads=0,
        use_wmma_prefill=False,
        rowtile_all=True,
    )

    assert plan.operation == GGUF_TRIPLE_OPERATION_EXACT
    assert plan.threads == 0
    assert gl._execute_gguf_linear_triple_plan(
        plan,
        *weights,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        out_c_ptr=400,
    ) is True
    assert calls == [
        (
            "exact",
            (100, 14, 14, 14, 200, 300, 400, 5, 5120, 1024, 512, 512),
            {"threads": 0, "stream": 0, "runtime": None},
        )
    ]


def test_triple_plan_thread_override_disables_rowtile(monkeypatch) -> None:
    weights = _q8_weights(
        source_shapes=((4096, 2048), (1024, 2048), (1024, 2048)),
    )
    calls = _capture_triple(monkeypatch)

    plan = gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=3,
        in_features=2048,
        out_features=4096,
        out_features_b=1024,
        out_features_c=1024,
        threads=64,
        use_wmma_prefill=False,
        rowtile_all=True,
    )

    assert plan.operation == GGUF_TRIPLE_OPERATION_EXACT
    assert plan.threads == 64
    assert gl._execute_gguf_linear_triple_plan(
        plan,
        *weights,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        out_c_ptr=400,
    ) is True
    assert calls[0][0] == "exact"
    assert calls[0][2]["threads"] == 64


@pytest.mark.parametrize("rows", [1, 2])
def test_triple_plan_wmma_declines_only_above_one_row(monkeypatch, rows) -> None:
    weights = _q8_weights(
        source_shapes=((1024, 2048), (512, 2048), (512, 2048)),
    )
    calls = _capture_triple(monkeypatch)

    plan = gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=rows,
        in_features=2048,
        out_features=1024,
        out_features_b=512,
        out_features_c=512,
        use_wmma_prefill=True,
        rowtile_all=False,
    )

    if rows == 1:
        assert plan.operation == GGUF_TRIPLE_OPERATION_EXACT
        assert plan.fallback_reason is None
        assert gl._execute_gguf_linear_triple_plan(
            plan, *weights, 100, 200, 300, 400
        ) is True
        assert calls[0][0] == "exact"
    else:
        assert plan.operation is None
        assert plan.owner_key is None
        assert plan.fallback_reason == GGUF_TRIPLE_FALLBACK_WMMA_PREFILL_DECLINED
        assert gl._execute_gguf_linear_triple_plan(
            plan, *weights, 100, 200, 300, 400
        ) is False
        assert calls == []


def test_triple_plan_rejects_mixed_and_unsupported_layouts(monkeypatch) -> None:
    q8_a, q8_b, q8_c = _q8_weights()
    q4 = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
        slot_path="layers.0.attn_k",
    )
    dense = _fake_weight(
        layout=LAYOUT_DENSE_BF16,
        quant_key="bf16",
        slot_path="layers.0.attn_q",
    )
    calls = _capture_triple(monkeypatch)

    mixed = _resolve((q8_a, q4, q8_c))
    assert mixed.operation is None
    assert mixed.fallback_reason == GGUF_TRIPLE_FALLBACK_UNSUPPORTED_OPERANDS

    unsupported = _resolve((dense, dense, dense))
    assert unsupported.operation is None
    assert unsupported.fallback_reason == GGUF_TRIPLE_FALLBACK_UNSUPPORTED_OPERANDS

    assert _execute(mixed, (q8_a, q4, q8_c)) is False
    assert calls == []


def test_triple_plan_reports_unregistered_owner(monkeypatch) -> None:
    weights = _q8_weights()
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_triple_gemv_decode_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    unregister(key)
    try:
        plan = _resolve(weights)
    finally:
        register(key, original, replace=True)
    assert plan.operation is None
    assert plan.fallback_reason == GGUF_TRIPLE_FALLBACK_TRIPLE_OWNER_UNREGISTERED


def test_triple_plan_owner_isolation_rejects_foreign_operands(monkeypatch) -> None:
    weights = _q8_weights(backend="hip_gfx1100")
    _capture_triple(monkeypatch)
    plan = gl._resolve_gguf_linear_triple_plan(
        *weights,
        rows=1,
        in_features=2048,
        out_features=1024,
        use_wmma_prefill=False,
        rowtile_all=False,
    )
    assert plan.backend == "hip_gfx1100"
    assert not hasattr(plan, "profile")

    foreign = _q8_weights(backend="hip_gfx1151")
    with pytest.raises(ValueError, match="backend"):
        _execute(plan, foreign)

    q4 = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
        backend="hip_gfx1100",
        slot_path="layers.0.attn_q",
    )
    with pytest.raises(ValueError, match="operand 0"):
        _execute(plan, (q4, weights[1], weights[2]))


def test_triple_plan_rebinds_allocations_per_execution(monkeypatch) -> None:
    original = _q8_weights(tiles_ptrs=(14, 15, 16))
    calls = _capture_triple(monkeypatch)
    plan = gl._resolve_gguf_linear_triple_plan(
        *original,
        rows=1,
        in_features=2048,
        out_features=1024,
        use_wmma_prefill=False,
        rowtile_all=False,
    )

    replacement = _q8_weights(tiles_ptrs=(44, 45, 46))
    assert _execute(plan, replacement) is True
    assert _execute(plan, original) is True

    assert calls == [
        (
            "exact",
            (100, 44, 45, 46, 200, 300, 400, 1, 2048, 1024, 1024, 1024),
            {"threads": 0, "stream": 0, "runtime": None},
        ),
        (
            "exact",
            (100, 14, 15, 16, 200, 300, 400, 1, 2048, 1024, 1024, 1024),
            {"threads": 0, "stream": 0, "runtime": None},
        ),
    ]


def test_triple_plan_rejects_transposed_k_n_with_equal_tile_bytes(monkeypatch) -> None:
    weights = _q8_weights()
    calls = _capture_triple(monkeypatch)
    plan = _resolve(weights)

    bad = _q8_weights(
        source_shapes=((2048, 1024), (1024, 2048), (1024, 2048)),
    )
    with pytest.raises(ValueError, match="operand 0"):
        _execute(plan, bad)
    assert calls == []


def test_triple_plan_rejects_changed_tiles_buffer_size(monkeypatch) -> None:
    weights = _q8_weights()
    calls = _capture_triple(monkeypatch)
    plan = _resolve(weights)

    expected_nbytes = 64 * 64 * 544
    bad = _q8_weights(
        tiles_nbytes=(expected_nbytes - 1, expected_nbytes, expected_nbytes),
    )
    with pytest.raises(ValueError, match="tiles buffer"):
        _execute(plan, bad)
    assert calls == []


def test_triple_plan_rejects_foreign_owner_slot_with_same_geometry(monkeypatch) -> None:
    weights = _q8_weights()
    calls = _capture_triple(monkeypatch)
    plan = _resolve(weights)

    bad = _q8_weights(
        slots=("layers.0.attn_k", "layers.0.attn_k", "layers.0.attn_v"),
    )
    with pytest.raises(ValueError, match="owner slot"):
        _execute(plan, bad)
    assert calls == []


def test_triple_plan_rejects_registry_generation_change(monkeypatch) -> None:
    weights = _q8_weights()
    calls = _capture_triple(monkeypatch)
    plan = _resolve(weights)

    unregister(KernelKey("hip_gfx1100", "linear", "sentinel", ""))
    with pytest.raises(ValueError, match="registry generation"):
        _execute(plan, weights)
    assert calls == []


def test_triple_plan_rejects_missing_geometry_contract(monkeypatch) -> None:
    weights = _q8_weights()
    calls = _capture_triple(monkeypatch)
    plan = _resolve(weights)

    expected_nbytes = 64 * 64 * 544
    bad = _q8_weights(
        source_shapes=(None, None, None),
        tiles_shapes=((expected_nbytes,), (expected_nbytes,), (expected_nbytes,)),
        tiles_nbytes=expected_nbytes,
    )
    with pytest.raises(ValueError, match="geometry contract"):
        _execute(plan, bad)
    assert calls == []


def test_triple_plan_rejects_resolve_time_source_shape_mismatch(monkeypatch) -> None:
    _capture_triple(monkeypatch)
    weights = _q8_weights(
        source_shapes=((512, 2048), (1024, 2048), (1024, 2048)),
    )
    with pytest.raises(ValueError, match="operand 0"):
        _resolve(weights)


def test_triple_plan_requires_owner_slot(monkeypatch) -> None:
    _capture_triple(monkeypatch)
    weights = _q8_weights(slots=(None, None, None))
    with pytest.raises(ValueError, match="slot_path"):
        _resolve(weights)


def test_triple_plan_api_is_private() -> None:
    assert not hasattr(gl, "resolve_gguf_linear_triple_plan")
    assert not hasattr(gl, "execute_gguf_linear_triple_plan")
    assert not hasattr(gl, "GGUFTripleLaunchPlan")
    assert not hasattr(gl, "GGUFTripleOperand")
    assert "GGUFTripleLaunchPlan" not in gl.__all__
    assert "GGUFTripleOperand" not in gl.__all__
    assert "resolve_gguf_linear_triple_plan" not in gl.__all__
    assert "execute_gguf_linear_triple_plan" not in gl.__all__
    assert hasattr(gl, "_resolve_gguf_linear_triple_plan")
    assert hasattr(gl, "_execute_gguf_linear_triple_plan")


def test_triple_plan_is_immutable_and_pointer_free(monkeypatch) -> None:
    weights = _q8_weights()
    _capture_triple(monkeypatch)
    plan = _resolve(weights)

    assert not hasattr(plan, "tiles_ptr")
    assert not hasattr(plan.operands[0], "ptr")
    assert not hasattr(plan, "profile")
    assert "profile" not in plan.__dataclass_fields__
    assert plan.registry_generation >= 0
    with pytest.raises(FrozenInstanceError):
        plan.threads = 128  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.operands[0].layout = LAYOUT_DENSE_BF16  # type: ignore[misc]


def test_public_triple_launch_uses_the_plan_path(monkeypatch) -> None:
    weights = _q8_weights(
        source_shapes=((1024, 2048), (512, 2048), (512, 2048)),
    )
    calls = _capture_triple(monkeypatch)
    monkeypatch.delenv("HIPENGINE_GGUF_WMMA_PREFILL", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", raising=False)
    monkeypatch.setattr(gl, "_wmma_prefill_session_enabled", None)
    monkeypatch.setattr(gl, "_q8_t16_rowtile_all_session_enabled", None)

    assert launch_gguf_linear_triple(
        *weights,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        out_c_ptr=400,
        rows=1,
        in_features=2048,
        out_features=1024,
        out_features_b=512,
        out_features_c=512,
        stream=7,
        runtime="runtime-sentinel",
    ) is True
    assert calls == [
        (
            "exact",
            (100, 14, 14, 14, 200, 300, 400, 1, 2048, 1024, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]
