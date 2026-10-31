from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import built-ins so the registry has real kernels to restore after overrides.
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401
import hipengine.runtime.gguf_linear as gl
from hipengine.kernels.registry import KernelKey, register, resolve, unregister
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
)


def _fake_weight(
    *,
    layout: str,
    quant_key: str,
    backend: str | None = None,
    slot_path: str | None = "layers.0.test",
):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
    }

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(
                layout=layout,
                quant_key=quant_key,
                slot_path=slot_path,
                source=None,
            )
            self.allocations = allocations
            if backend is not None:
                self.backend = backend

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def _gfx1151_pair_key() -> KernelKey:
    return KernelKey(
        "hip_gfx1151",
        "linear_pair",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )


def test_pair_plan_registered_decode_operation_operands_and_library() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    key = _gfx1151_pair_key()
    original = resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    )
    calls: list[tuple[tuple, dict]] = []
    register(key, lambda *args, **kwargs: calls.append((args, kwargs)), replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    gl.clear_gguf_linear_dispatch_cache()
    try:
        plan = gl._resolve_gguf_linear_pair_plan(
            weight_a,
            weight_b,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            registered_decode_only=True,
        )
        assert plan.operation == "q4_pack8_dual_decode"
        assert plan.fallback_reason is None
        assert plan.pair_key == key
        assert plan.library_keys == ("gguf_q4_k",)
        assert plan.library_always is False
        assert [operand.weight for operand in plan.operands] == [weight_a, weight_b]
        assert [operand.out_features for operand in plan.operands] == [1024, 1024]
        assert gl._execute_gguf_linear_pair_plan(
            plan,
            weight_a,
            weight_b,
            100,
            200,
            300,
            stream=7,
            libraries={"gguf_q4_k": "pair-library"},
            runtime="runtime-sentinel",
        )
    finally:
        register(key, original, replace=True)
        gl.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 11, 12, 13, 11, 12, 13, 200, 300, 1, 3072, 1024),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "pair-library",
            },
        )
    ]


def test_pair_plan_rejects_foreign_weight_owner() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    key = _gfx1151_pair_key()
    original = resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    )
    calls: list[tuple[tuple, dict]] = []
    register(key, lambda *args, **kwargs: calls.append((args, kwargs)), replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    foreign_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    gl.clear_gguf_linear_dispatch_cache()
    try:
        plan = gl._resolve_gguf_linear_pair_plan(
            weight_a,
            weight_b,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            registered_decode_only=True,
        )
        with pytest.raises(ValueError, match="operand 1 owner changed"):
            gl._execute_gguf_linear_pair_plan(
                plan,
                weight_a,
                foreign_b,
                100,
                200,
                300,
            )
    finally:
        register(key, original, replace=True)
        gl.clear_gguf_linear_dispatch_cache()

    assert calls == []


def test_pair_plan_prefill_rowtile_threshold_and_library_presence(monkeypatch) -> None:
    monkeypatch.setattr(gl, "_resolve_use_q4k_rowtile", lambda _: True)
    rowtile_calls: list[tuple[tuple, dict]] = []
    base_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q4_k_pack8_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: rowtile_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        gl,
        "gguf_q4_k_pack8_dual_prefill_bf16_bf16_out",
        lambda *args, **kwargs: base_calls.append((args, kwargs)),
    )
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    gl.clear_gguf_linear_dispatch_cache()
    with gl.native_batch_decode_session(True):
        rowtile_plan = gl._resolve_gguf_linear_pair_plan(
            weight_a,
            weight_b,
            rows=2,
            in_features=1024,
            out_features=2048,
        )
        assert rowtile_plan.operation == "q4_pack8_dual_prefill"
        assert rowtile_plan.q4_prefill_mode == "rowtile"
        assert rowtile_plan.library_keys == ("gguf_q4_k",)
        assert rowtile_plan.library_always is True
        assert gl._execute_gguf_linear_pair_plan(
            rowtile_plan,
            weight_a,
            weight_b,
            100,
            200,
            300,
            stream=7,
            libraries=None,
            runtime="runtime-sentinel",
        )
        base_plan = gl._resolve_gguf_linear_pair_plan(
            weight_a,
            weight_b,
            rows=5,
            in_features=1024,
            out_features=2048,
        )
        assert base_plan.operation == "q4_pack8_dual_prefill"
        assert base_plan.q4_prefill_mode == "base"
        assert base_plan.library_keys == ()
        assert base_plan.library_always is False
        assert gl._execute_gguf_linear_pair_plan(
            base_plan,
            weight_a,
            weight_b,
            100,
            200,
            300,
            stream=7,
            libraries={"gguf_q4_k": "must-not-be-forwarded"},
            runtime="runtime-sentinel",
        )
    gl.clear_gguf_linear_dispatch_cache()

    assert rowtile_calls == [
        (
            (100, 11, 12, 13, 200, 2, 1024, 2048),
            {"stream": 7, "runtime": "runtime-sentinel", "library": None},
        ),
        (
            (100, 11, 12, 13, 300, 2, 1024, 2048),
            {"stream": 7, "runtime": "runtime-sentinel", "library": None},
        ),
    ]
    assert base_calls == [
        (
            (100, 11, 12, 13, 11, 12, 13, 200, 300, 5, 1024, 2048),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_pair_plan_q8_split_threshold_and_kwargs_absence(monkeypatch) -> None:
    for name in (
        "HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE",
        "HIPENGINE_GGUF_Q8_T16_PAIR_COL8",
        "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL",
    ):
        monkeypatch.delenv(name, raising=False)
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("exact", args, kwargs)),
    )
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("rowtile", args, kwargs)),
    )
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    gl.clear_gguf_linear_dispatch_cache()
    exact_plan = gl._resolve_gguf_linear_pair_plan(
        weight_a,
        weight_b,
        rows=1,
        in_features=2048,
        out_features=8192,
        out_features_b=4096,
        use_wmma_prefill=False,
    )
    assert exact_plan.operation == "q8_t16_dual_split"
    assert exact_plan.q8_split_mode == "exact"
    assert exact_plan.threads == 0
    assert exact_plan.library_keys == ()
    assert gl._execute_gguf_linear_pair_plan(
        exact_plan,
        weight_a,
        weight_b,
        100,
        200,
        300,
        stream=7,
        libraries={"gguf_q8_0_t16_v1": "must-not-be-forwarded"},
        runtime="runtime-sentinel",
    )

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE", "1")
    gl.clear_gguf_linear_dispatch_cache()
    rowtile_plan = gl._resolve_gguf_linear_pair_plan(
        weight_a,
        weight_b,
        rows=3,
        in_features=2048,
        out_features=8192,
        out_features_b=4096,
        use_wmma_prefill=False,
    )
    assert rowtile_plan.operation == "q8_t16_dual_split"
    assert rowtile_plan.q8_split_mode == "rowtile"
    assert rowtile_plan.threads == 128
    assert rowtile_plan.library_keys == ()
    assert gl._execute_gguf_linear_pair_plan(
        rowtile_plan,
        weight_a,
        weight_b,
        100,
        200,
        300,
        stream=7,
        libraries={"gguf_q8_0_t16_v1": "must-not-be-forwarded"},
        runtime="runtime-sentinel",
    )
    gl.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            "exact",
            (100, 14, 14, 200, 300, 1, 2048, 8192, 4096),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "rowtile",
            (100, 14, 14, 200, 300, 3, 2048, 8192, 4096),
            {"threads": 128, "stream": 7, "runtime": "runtime-sentinel"},
        ),
    ]


def test_pair_plan_resolves_callable_and_pointers_at_execution(monkeypatch) -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    stale_calls: list[tuple[tuple, dict]] = []
    fresh_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_dual_gemv_bf16_bf16_out",
        lambda *args, **kwargs: stale_calls.append((args, kwargs)),
    )
    gl.clear_gguf_linear_dispatch_cache()
    plan = gl._resolve_gguf_linear_pair_plan(
        weight_a,
        weight_b,
        rows=1,
        in_features=1024,
        out_features=2048,
    )
    assert plan.operation == "q8_raw_dual"
    weight_a.allocations["raw"].tensor.ptr = 777
    weight_b.allocations["raw"].tensor.ptr = 888
    monkeypatch.setattr(
        gl,
        "gguf_q8_0_dual_gemv_bf16_bf16_out",
        lambda *args, **kwargs: fresh_calls.append((args, kwargs)),
    )
    assert gl._execute_gguf_linear_pair_plan(
        plan,
        weight_a,
        weight_b,
        100,
        200,
        300,
        stream=7,
        runtime="runtime-sentinel",
    )
    gl.clear_gguf_linear_dispatch_cache()

    assert stale_calls == []
    assert fresh_calls == [
        (
            (100, 777, 888, 200, 300, 1, 1024, 2048),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_pair_plan_fallback_reasons_and_false_execution() -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    gl.clear_gguf_linear_dispatch_cache()
    declined = gl._resolve_gguf_linear_pair_plan(
        weight_a,
        weight_b,
        rows=4,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=True,
    )
    assert declined.operation is None
    assert declined.fallback_reason == "pair_dispatch_declined"
    assert not gl._execute_gguf_linear_pair_plan(
        declined, weight_a, weight_b, 100, 200, 300
    )

    q4_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    q4_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    gl.clear_gguf_linear_dispatch_cache()
    registered_only = gl._resolve_gguf_linear_pair_plan(
        q4_a,
        q4_b,
        rows=1,
        in_features=3072,
        out_features=1024,
        backend="hip_gfx1100",
        registered_decode_only=True,
    )
    assert registered_only.operation is None
    assert registered_only.fallback_reason == "registered_decode_only_declined"
    assert not gl._execute_gguf_linear_pair_plan(
        registered_only, q4_a, q4_b, 100, 200, 300
    )


def test_pair_plan_wmma_bulk_library_variant_precedes_quant(monkeypatch) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    wmma_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    )
    exact_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    wmma_original = resolve(
        backend=wmma_key.backend,
        layer=wmma_key.layer,
        quant=wmma_key.quant,
        variant=wmma_key.variant,
    )
    exact_original = resolve(
        backend=exact_key.backend,
        layer=exact_key.layer,
        quant=exact_key.quant,
        variant=exact_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []
    register(
        wmma_key,
        lambda *args, **kwargs: calls.append((args, kwargs)),
        replace=True,
    )
    unregister(exact_key)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    gl.clear_gguf_linear_dispatch_cache()
    try:
        plan = gl._resolve_gguf_linear_pair_plan(
            weight_a,
            weight_b,
            rows=512,
            in_features=1024,
            out_features=3584,
            backend="hip_gfx1151",
            use_wmma_prefill=True,
        )
        assert plan.operation == "q4_pack8_dual_prefill"
        assert plan.q4_prefill_mode == "wmma_bulk"
        assert plan.pair_key == wmma_key
        assert plan.library_keys == (
            "gguf_q4_k:pack8_wmma_prefill_bf16_bf16_out",
            "gguf_q4_k",
        )
        assert plan.library_always is True
        assert (
            gl._pair_plan_library(
                plan,
                {
                    "gguf_q4_k": "quant-library",
                    "gguf_q4_k:pack8_wmma_prefill_bf16_bf16_out": None,
                },
            )
            is None
        )
        assert gl._execute_gguf_linear_pair_plan(
            plan,
            weight_a,
            weight_b,
            100,
            200,
            300,
            stream=7,
            libraries={
                "gguf_q4_k": "quant-library",
                "gguf_q4_k:pack8_wmma_prefill_bf16_bf16_out": "variant-library",
            },
            runtime="runtime-sentinel",
        )
    finally:
        register(wmma_key, wmma_original, replace=True)
        register(exact_key, exact_original, replace=True)
        gl.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 11, 12, 13, 200, 512, 1024, 3584),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "variant-library",
            },
        ),
        (
            (100, 11, 12, 13, 300, 512, 1024, 3584),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "variant-library",
            },
        ),
    ]


def test_pair_plan_explicit_rowtile_policy_beats_ambient(monkeypatch) -> None:
    """An owner's explicit input wins over the other owner's ambient value."""

    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    shapes = dict(
        rows=4,
        in_features=2048,
        out_features=8192,
        out_features_b=4096,
        use_wmma_prefill=False,
    )

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", "1")
    gl.clear_gguf_linear_dispatch_cache()
    strict_plan = gl._resolve_gguf_linear_pair_plan(
        weight_a, weight_b, **shapes, q8_t16_rowtile_all=False
    )
    assert strict_plan.operation == "q8_t16_dual_split"
    assert strict_plan.q8_split_mode == "exact"

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", "0")
    gl.clear_gguf_linear_dispatch_cache()
    production_plan = gl._resolve_gguf_linear_pair_plan(
        weight_a, weight_b, **shapes, q8_t16_rowtile_all=True
    )
    assert production_plan.operation == "q8_t16_dual_split"
    assert production_plan.q8_split_mode == "rowtile"
    assert production_plan.threads == 128
    gl.clear_gguf_linear_dispatch_cache()
