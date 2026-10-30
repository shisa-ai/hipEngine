"""Focused RF-10 tests for private GGUF residual launch planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gl
from hipengine.kernels.registry import KernelKey, generation, register
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_Q4_K_PACK8,
)


def _fake_weight(
    *,
    layout: str,
    quant_key: str,
    backend: str | None = None,
    slot_path: str | None = "layers.0.test",
    tiles: bool = True,
    tiles_ptr: int = 14,
    decode_tiles: bool = False,
    source_shape: tuple[int, int] | None = (1_024, 2_048),
):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
    }
    if tiles:
        allocations["tiles"] = SimpleNamespace(
            tensor=SimpleNamespace(ptr=tiles_ptr)
        )
    if decode_tiles:
        allocations["decode_tiles"] = SimpleNamespace(
            tensor=SimpleNamespace(ptr=15)
        )
    source = (
        SimpleNamespace(shape=source_shape) if source_shape is not None else None
    )
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


def _capture(calls: list[tuple[str, tuple, dict]], label: str):
    def fake(*args, **kwargs):
        calls.append((label, args, kwargs))

    return fake


def _ensure_and_register(
    primitive: KernelKey,
    fused: KernelKey,
    capture,
) -> None:
    gl._ensure_linear_kernel_registered(primitive)
    register(fused, capture, replace=True)


def _execute(plan, weight, *, stream=0, libraries=None):
    return gl._execute_gguf_linear_residual_plan(
        plan,
        weight,
        100,
        300,
        400,
        stream=stream,
        libraries=libraries,
        runtime="runtime-sentinel",
    )


def test_residual_plan_bulk_wmma_above_rows_four_binds_launcher_and_library() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    primitive = KernelKey(
        "hip_gfx1151",
        "dense_gemv",
        "bf16",
        "prefill_wmma_out",
    )
    fused = KernelKey(
        "hip_gfx1151",
        "linear+residual",
        "bf16",
        "prefill_wmma_out_bf16_residual_bf16_out",
    )
    calls: list[tuple[str, tuple, dict]] = []
    _ensure_and_register(primitive, fused, _capture(calls, "bulk"))
    dense = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="bf16")
    pack8 = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")

    assert (
        gl._resolve_gguf_linear_residual_plan(
            dense,
            rows=0,
            in_features=3_584,
            out_features=1_024,
        ).fallback_reason
        == gl._GGUF_RESIDUAL_FALLBACK_INVALID_ROWS
    )
    assert (
        gl._resolve_gguf_linear_residual_plan(
            pack8,
            rows=5,
            in_features=3_584,
            out_features=1_024,
        ).fallback_reason
        == gl._GGUF_RESIDUAL_FALLBACK_UNSUPPORTED_BULK_LAYOUT
    )

    with gl.wmma_prefill_session(True):
        below = gl._resolve_gguf_linear_residual_plan(
            dense,
            rows=4,
            in_features=3_584,
            out_features=1_024,
        )
        plan = gl._resolve_gguf_linear_residual_plan(
            dense,
            rows=512,
            in_features=3_584,
            out_features=1_024,
            backend="hip_gfx1151",
        )

    assert below.fallback_reason == gl._GGUF_RESIDUAL_FALLBACK_WMMA_PREFILL_DECLINED
    assert plan.operation == gl._GGUF_RESIDUAL_OPERATION_DENSE_WMMA_BULK
    assert plan.normal_key == primitive
    assert plan.fused_key == fused
    assert plan.owner_weight is dense
    assert plan.abi == "dense_bf16"
    assert plan.allocation_name is None
    assert plan.sidecar_variant is None
    assert plan.fallback_reason is None
    assert plan.registry_generation == generation()
    assert not hasattr(plan, "fn")
    assert not hasattr(plan, "weight_ptr")
    assert not hasattr(plan, "allocation")
    with pytest.raises(FrozenInstanceError):
        plan.rows = 2  # type: ignore[misc]

    assert _execute(
        plan,
        dense,
        stream=9,
        libraries={"bf16:prefill_wmma_out_bf16_residual_bf16_out": "fused-lib"},
    )
    assert calls == [
        (
            "bulk",
            (100, 10, 300, 400, 512, 3_584, 1_024),
            {
                "stream": 9,
                "runtime": "runtime-sentinel",
                "library": "fused-lib",
            },
        )
    ]


@pytest.mark.parametrize(
    ("libraries", "expected_library"),
    [
        (None, None),
        ({}, None),
        ({"gguf_q4_k:pack8_bf16_residual_bf16_out": "fused",
          "gguf_q4_k:pack8_bf16_bf16_out": "normal", "gguf_q4_k": "quant"}, "fused"),
        ({"gguf_q4_k:pack8_bf16_bf16_out": "normal", "gguf_q4_k": "quant"}, "normal"),
        ({"gguf_q4_k": "quant"}, "quant"),
    ],
)
def test_residual_plan_rows_one_requires_registered_decode_and_launches_pack8(
    libraries, expected_library,
) -> None:
    primitive = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_bf16_bf16_out",
    )
    fused = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q4_k",
        "pack8_bf16_residual_bf16_out",
    )
    calls: list[tuple[str, tuple, dict]] = []
    _ensure_and_register(primitive, fused, _capture(calls, "rows1"))
    weight = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")

    denied = gl._resolve_gguf_linear_residual_plan(
        weight,
        rows=1,
        in_features=3_584,
        out_features=1_024,
    )
    assert denied.operation is None
    assert (
        denied.fallback_reason
        == gl._GGUF_RESIDUAL_FALLBACK_REGISTERED_DECODE_REQUIRED
    )

    with gl.wmma_prefill_session(True):
        plan = gl._resolve_gguf_linear_residual_plan(
            weight,
            rows=1,
            in_features=3_584,
            out_features=1_024,
            registered_decode=True,
        )

    assert plan.operation == gl._GGUF_RESIDUAL_OPERATION_REGISTERED_ROWS1
    assert plan.normal_key == primitive
    assert plan.fused_key == fused
    assert plan.abi == "pack8"
    assert plan.allocation_name is None
    assert gl.launch_gguf_linear_residual(
        weight, 100, 300, 400, 1, 3_584, 1_024,
        registered_decode=True, stream=7, libraries=libraries,
        runtime="runtime-sentinel",
    )
    expected_kwargs = {"stream": 7, "runtime": "runtime-sentinel"}
    if expected_library is not None:
        expected_kwargs["library"] = expected_library
    assert calls == [
        (
            "rows1",
            (100, 11, 12, 13, 300, 400, 1, 3_584, 1_024),
            expected_kwargs,
        )
    ]


def test_residual_plan_small_rows_prefers_sidecar_allocation() -> None:
    primitive = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_bf16_out",
    )
    fused = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_residual_bf16_out",
    )
    calls: list[tuple[str, tuple, dict]] = []
    _ensure_and_register(primitive, fused, _capture(calls, "sidecar"))
    weight = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )

    with gl.native_batch_decode_session(True):
        plan = gl._resolve_gguf_linear_residual_plan(
            weight,
            rows=3,
            in_features=17_408,
            out_features=5_120,
        )

    assert plan.operation == gl._GGUF_RESIDUAL_OPERATION_SIDECAR
    assert plan.normal_key == primitive
    assert plan.fused_key == fused
    assert plan.sidecar_variant == "dense_rowtile_bf16_bf16_out"
    assert plan.allocation_name is None
    assert _execute(plan, weight, stream=8)
    assert calls == [
        (
            "sidecar",
            (100, 15, 300, 400, 3, 17_408, 5_120),
            {"stream": 8, "runtime": "runtime-sentinel"},
        )
    ]


def test_residual_plan_small_rows_uses_tiles_and_reports_missing_workspace() -> None:
    primitive = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_bf16_out",
    )
    fused = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_residual_bf16_out",
    )
    calls: list[tuple[str, tuple, dict]] = []
    _ensure_and_register(primitive, fused, _capture(calls, "small"))
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )

    with gl.native_batch_decode_session(True):
        plan = gl._resolve_gguf_linear_residual_plan(
            weight,
            rows=3,
            in_features=17_408,
            out_features=5_120,
        )

    assert plan.operation == gl._GGUF_RESIDUAL_OPERATION_REGISTERED_SMALL
    assert plan.normal_key == primitive
    assert plan.fused_key == fused
    assert plan.allocation_name == "tiles"
    assert _execute(plan, weight, stream=10)
    assert calls == [
        (
            "small",
            (100, 14, 300, 400, 3, 17_408, 5_120),
            {"stream": 10, "runtime": "runtime-sentinel"},
        )
    ]

    missing_tiles = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
        tiles=False,
    )
    with gl.native_batch_decode_session(True):
        assert (
            gl._resolve_gguf_linear_residual_plan(
                missing_tiles,
                rows=3,
                in_features=17_408,
                out_features=5_120,
            ).fallback_reason
            == gl._GGUF_RESIDUAL_FALLBACK_WORKSPACE_MISSING
        )

    no_sidecar = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
    )
    with gl.native_batch_decode_session(True):
        assert (
            gl._resolve_gguf_linear_residual_plan(
                no_sidecar,
                rows=3,
                in_features=17_408,
                out_features=5_120,
            ).fallback_reason
            == gl._GGUF_RESIDUAL_FALLBACK_UNSUPPORTED_DISPATCH_ABI
        )


def test_residual_plan_rejects_foreign_weight_with_changed_shape_and_tiles() -> None:
    primitive = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_bf16_out",
    )
    fused = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_residual_bf16_out",
    )
    calls: list[tuple[str, tuple, dict]] = []
    _ensure_and_register(primitive, fused, _capture(calls, "small"))
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
        source_shape=(5_120, 17_408),
        tiles_ptr=14,
    )
    foreign = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
        source_shape=(1_024, 17_408),
        tiles_ptr=24,
    )

    with gl.native_batch_decode_session(True):
        plan = gl._resolve_gguf_linear_residual_plan(
            weight,
            rows=3,
            in_features=17_408,
            out_features=5_120,
        )

    assert plan.operation == gl._GGUF_RESIDUAL_OPERATION_REGISTERED_SMALL
    with pytest.raises(ValueError, match="owner weight"):
        _execute(plan, foreign)
    assert calls == []


def test_residual_plan_rejects_owner_and_registry_changes() -> None:
    primitive = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_bf16_bf16_out",
    )
    fused = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q4_k",
        "pack8_bf16_residual_bf16_out",
    )
    _ensure_and_register(primitive, fused, _capture([], "rows1"))
    weight = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    plan = gl._resolve_gguf_linear_residual_plan(
        weight,
        rows=1,
        in_features=3_584,
        out_features=1_024,
        registered_decode=True,
    )

    weight.spec.slot_path = "layers.1.test"
    with pytest.raises(ValueError, match="owner slot"):
        _execute(plan, weight)

    weight.spec.slot_path = "layers.0.test"
    register(
        KernelKey("hip_gfx1100", "linear", "test_registry_change", "test"),
        lambda: None,
        replace=True,
    )
    with pytest.raises(ValueError, match="registry generation changed"):
        _execute(plan, weight)
