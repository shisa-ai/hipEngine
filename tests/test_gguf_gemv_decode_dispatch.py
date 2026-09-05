"""Routing tests for the P9.B6 GGUF pack8 GEMV decode opt-in dispatch.

Mirrors :mod:`tests/test_gguf_linear_dispatch.py` but focused on the
``rows == 1`` decode rewrite added in P9.B6: ``pack8_gemv_*`` ->
``pack8_gemv_decode_*`` for the matching ``(quant, layer)``, controlled by
``HIPENGINE_GGUF_GEMV_DECODE`` env var, the ``gemv_decode_session(...)``
context manager, and per-call ``use_gemv_decode`` kwarg with the same
precedence as the WMMA prefill toggle.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Real kernel module imports keep the registry populated across tests.
import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.kernels.registry import _KERNELS
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_RAW_GGUF,
)
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    gemv_decode_session,
    gguf_gemv_decode_enabled,
    launch_gguf_linear,
    launch_gguf_linear_pair_concat,
    native_batch_decode_session,
    target_verifier_production_q4_rowtile_session,
    q4k_rowtile_session,
    target_verifier_rowtile_session,
    _q4_t16_dense_native_dispatch,
    set_gemv_decode_enabled,
)


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
    }

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


_Q8_DECODE_PACK8 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
_Q8_DECODE_PACK8_F32 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_f32_out")
_Q8_GEMV_DECODE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_decode_bf16_bf16_out"
)
_Q8_PREFILL = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out")
_Q8_EXACT_PREFILL_TILE8X4 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "exact_prefill_tile8x4_bf16_bf16_out"
)
_Q6_PACK8_F32 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_bf16_f32_out"
)
_Q4_SCALAR = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_bf16_f32_out")
_Q4_GEMV_DECODE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q4_k", "pack8_gemv_decode_bf16_f32_out"
)
_Q8_WMMA_DUAL_PREFILL = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_dual_gate_up_bf16_bf16_out"
)


# Q5T16 native batch decode family (base decode + true rowtile + padded WMMA
# fallback) used to verify the rows 5-8 rowtile promotion.
_Q5_T16_DECODE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q5_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"
)
_Q5_T16_ROWTILE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q5_k_t16_v1", "t16_gemv_rowtile_bf16_bf16_out"
)
_Q5_T16_WMMA = KernelKey(
    "hip_gfx1100", "linear", "gguf_q5_k_t16_v1", "t16_wmma_prefill_bf16_bf16_out"
)

# Standard and planar-qmicro Q6T16 native batch decode families.
_Q6_T16_DECODE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"
)
_Q6_T16_ROWTILE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_rowtile_bf16_bf16_out"
)
_Q6_T16_WMMA = KernelKey(
    "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_wmma_prefill_bf16_bf16_out"
)
_Q6_PLANAR_DECODE = KernelKey(
    "hip_gfx1100",
    "linear",
    "gguf_q6_k_t16_qmicro_planar_v1",
    "t16_gemv_decode_bf16_bf16_out",
)
_Q6_PLANAR_ROWTILE = KernelKey(
    "hip_gfx1100",
    "linear",
    "gguf_q6_k_t16_qmicro_planar_v1",
    "t16_gemv_rowtile_bf16_bf16_out",
)
_Q6_PLANAR_WMMA = KernelKey(
    "hip_gfx1100",
    "linear",
    "gguf_q6_k_t16_qmicro_planar_v1",
    "t16_wmma_prefill_bf16_bf16_out",
)


@pytest.fixture(autouse=True)
def _reset_gemv_decode_state(monkeypatch):
    monkeypatch.delenv("HIPENGINE_GGUF_GEMV_DECODE", raising=False)
    set_gemv_decode_enabled(None)
    yield
    set_gemv_decode_enabled(None)


def _capture_launch(
    *,
    rows: int,
    in_features: int = 1024,
    out_features: int = 2048,
    use_gemv_decode: bool | None = None,
    quant_key: str = "gguf_q8_0",
    layout: str = LAYOUT_RAW_GGUF,
    output_dtype: str = GGUF_OUTPUT_BF16,
    extra_keys: tuple[KernelKey, ...] = (),
    remove_keys: tuple[KernelKey, ...] = (),
    native_batch_decode: bool = False,
    target_verifier_rowtile: bool = False,
    libraries=None,
):
    weight = _fake_weight(layout=layout, quant_key=quant_key)
    captured: dict[str, object] = {"key": None, "args": None, "kwargs": None}
    keys = (
        _Q8_DECODE_PACK8,
        _Q8_DECODE_PACK8_F32,
        _Q8_GEMV_DECODE,
        _Q8_PREFILL,
    ) + extra_keys
    originals = {
        k: resolve(
            backend=k.backend,
            layer=k.layer,
            quant=k.quant,
            variant=k.variant,
            missing="none",
        )
        for k in keys
    }

    def make_fake(key: KernelKey):
        def fake(*args, **kwargs):
            captured["key"] = key
            captured["args"] = args
            captured["kwargs"] = kwargs

        return fake

    try:
        for k in keys:
            register(k, make_fake(k), replace=True)
        for k in remove_keys:
            # Simulate a missing kernel by clearing the registry entry. The
            # registry stores ``None`` and the resolver treats it the same
            # as an unregistered key (see ``_KERNELS.get`` short-circuit).
            register(k, None, replace=True)
        with (
            native_batch_decode_session(native_batch_decode),
            target_verifier_rowtile_session(target_verifier_rowtile),
        ):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=200,
                rows=rows,
                in_features=in_features,
                out_features=out_features,
                output_dtype=output_dtype,
                stream=7,
                libraries=libraries,
                runtime="runtime-sentinel",
                use_gemv_decode=use_gemv_decode,
            )
    finally:
        for k, fn in originals.items():
            if fn is None:
                # The key was unregistered before the test ran; leave it
                # unregistered so we don't poison the global registry with
                # a ``None`` entry that later tests would observe.
                _KERNELS.pop(k, None)
            else:
                register(k, fn, replace=True)

    return captured["key"], captured["args"], captured["kwargs"]


# ---------------------------------------------------------------------------
# Default off + opt-in precedence.
# ---------------------------------------------------------------------------


def test_native_batch_decode_q5_t16_rows_5_8_route_to_true_rowtile() -> None:
    """Q5T16 true rowtile is now certified through rows 8, so native batch
    decode rewrites rows 5-8 to the rowtile instead of padded WMMA.
    """
    for rows in (5, 6, 7, 8):
        key, _, _ = _capture_launch(
            rows=rows,
            native_batch_decode=True,
            quant_key="gguf_q5_k_t16_v1",
            layout=LAYOUT_GGUF_Q5_K_T16,
            extra_keys=(_Q5_T16_ROWTILE, _Q5_T16_WMMA),
        )
        assert key == _Q5_T16_ROWTILE


def test_native_batch_decode_q6_planar_t16_rows_5_8_route_to_true_rowtile() -> None:
    """Planar-qmicro Q6T16 true col8 rowtile is now certified through rows 8,
    so native batch decode rewrites rows 5-8 to the rowtile instead of the
    per-row fallback (5/6) or padded WMMA (7/8).
    """
    for rows in (5, 6, 7, 8):
        key, _, _ = _capture_launch(
            rows=rows,
            native_batch_decode=True,
            quant_key="gguf_q6_k_t16_qmicro_planar_v1",
            layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
            extra_keys=(_Q6_PLANAR_ROWTILE, _Q6_PLANAR_WMMA),
        )
        assert key == _Q6_PLANAR_ROWTILE


def test_production_target_verifier_q4_scope_is_shape_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.gguf_linear as gguf_linear

    original_capability = gguf_linear.backend_package_capability

    def capability(backend: str, name: str, default=None):
        if name == "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS":
            return frozenset({8})
        if name == "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES":
            return frozenset({(5_120, 12_288), (17_408, 5_120)})
        return original_capability(backend, name, default)

    monkeypatch.setattr(gguf_linear, "backend_package_capability", capability)
    dispatch = GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        ),
        "t16",
    )
    assert (
        _q4_t16_dense_native_dispatch(
            dispatch,
            rows=8,
            in_features=5_120,
            out_features=12_288,
        )
        == dispatch
    )
    with target_verifier_production_q4_rowtile_session(True):
        selected = _q4_t16_dense_native_dispatch(
            dispatch,
            rows=8,
            in_features=5_120,
            out_features=12_288,
        )
        excluded_shape = _q4_t16_dense_native_dispatch(
            dispatch,
            rows=8,
            in_features=5_120,
            out_features=1_024,
        )
        excluded_rows = _q4_t16_dense_native_dispatch(
            dispatch,
            rows=6,
            in_features=5_120,
            out_features=12_288,
        )
    assert selected.key.variant == "dense_rowtile_bf16_bf16_out"
    assert excluded_shape == dispatch
    assert excluded_rows == dispatch


def test_target_verifier_scope_routes_only_backend_admitted_q5_q6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.gguf_linear as gguf_linear

    original_capability = gguf_linear.backend_package_capability

    def capability(backend: str, name: str, default=None):
        if name == "GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT":
            return {
                "gguf_q5_k_t16_v1": frozenset({(6_144, 5_120)}),
                "gguf_q6_k_t16_v1": frozenset({(5_120, 10_240)}),
                "gguf_q6_k_t16_qmicro_planar_v1": frozenset(
                    {(5_120, 1_024), (17_408, 5_120)}
                ),
            }
        if name == "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT":
            return {
                "gguf_q5_k_t16_v1": {
                    "default": 4,
                    "shapes": {(6_144, 5_120): 8},
                },
                "gguf_q6_k_t16_v1": 8,
                "gguf_q6_k_t16_qmicro_planar_v1": 8,
            }
        return original_capability(backend, name, default)

    monkeypatch.setattr(gguf_linear, "backend_package_capability", capability)
    key, _, _ = _capture_launch(
        rows=8,
        in_features=17_408,
        out_features=5_120,
        target_verifier_rowtile=True,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        extra_keys=(_Q6_PLANAR_ROWTILE, _Q6_PLANAR_WMMA),
    )
    assert key == _Q6_PLANAR_ROWTILE

    key, _, _ = _capture_launch(
        rows=8,
        in_features=5_120,
        out_features=10_240,
        target_verifier_rowtile=True,
        quant_key="gguf_q6_k_t16_v1",
        layout=LAYOUT_GGUF_Q6_K_T16,
        extra_keys=(_Q6_T16_DECODE, _Q6_T16_ROWTILE, _Q6_T16_WMMA),
    )
    assert key == _Q6_T16_ROWTILE

    # A quant match without an admitted actual shape keeps the old owner.
    key, _, _ = _capture_launch(
        rows=8,
        in_features=1_024,
        out_features=2_048,
        target_verifier_rowtile=True,
        quant_key="gguf_q6_k_t16_v1",
        layout=LAYOUT_GGUF_Q6_K_T16,
        extra_keys=(_Q6_T16_DECODE, _Q6_T16_ROWTILE, _Q6_T16_WMMA),
    )
    assert key == _Q6_T16_DECODE

    key, _, _ = _capture_launch(
        rows=8,
        in_features=6_144,
        out_features=5_120,
        target_verifier_rowtile=True,
        quant_key="gguf_q5_k_t16_v1",
        layout=LAYOUT_GGUF_Q5_K_T16,
        extra_keys=(_Q5_T16_DECODE, _Q5_T16_ROWTILE, _Q5_T16_WMMA),
    )
    assert key == _Q5_T16_ROWTILE

    # The target-verifier scope must not act like the broad native-batch scope:
    # unmeasured Q5 shapes retain their old owner.
    key, _, _ = _capture_launch(
        rows=8,
        in_features=2_048,
        out_features=1_024,
        target_verifier_rowtile=True,
        quant_key="gguf_q5_k_t16_v1",
        layout=LAYOUT_GGUF_Q5_K_T16,
        extra_keys=(_Q5_T16_DECODE, _Q5_T16_ROWTILE, _Q5_T16_WMMA),
    )
    assert key == _Q5_T16_DECODE


def test_target_verifier_chunks_only_admitted_q6_r12_r16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.gguf_linear as gguf_linear

    original_capability = gguf_linear.backend_package_capability

    def capability(backend: str, name: str, default=None):
        if name == "GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT":
            return {"gguf_q6_k_t16_v1": frozenset({(5_120, 10_240)})}
        if name == "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT":
            return {"gguf_q6_k_t16_v1": 8}
        if name == "GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT":
            return {"gguf_q6_k_t16_v1": frozenset({9, 12, 16})}
        return original_capability(backend, name, default)

    monkeypatch.setattr(gguf_linear, "backend_package_capability", capability)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16,
        quant_key="gguf_q6_k_t16_v1",
    )
    keys = (_Q6_T16_DECODE, _Q6_T16_ROWTILE, _Q6_T16_WMMA)
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )
        for key in keys
    }
    calls: list[tuple[KernelKey, tuple[object, ...]]] = []

    def capture(key: KernelKey):
        def fake(*args, **_kwargs):
            calls.append((key, args))

        return fake

    try:
        for key in keys:
            register(key, capture(key), replace=True)
        with target_verifier_rowtile_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=500,
                out_ptr=600,
                rows=12,
                in_features=5_120,
                out_features=10_240,
                backend="hip_gfx1100",
                stream=7,
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
            with target_verifier_production_q4_rowtile_session(True):
                launch_gguf_linear(
                    weight,
                    x_ptr=100,
                    out_ptr=200,
                    rows=12,
                    in_features=5_120,
                    out_features=10_240,
                    backend="hip_gfx1100",
                    stream=7,
                    runtime="runtime-sentinel",
                    use_wmma_prefill=False,
                )
                launch_gguf_linear(
                    weight,
                    x_ptr=700,
                    out_ptr=800,
                    rows=9,
                    in_features=5_120,
                    out_features=10_240,
                    backend="hip_gfx1100",
                    stream=7,
                    runtime="runtime-sentinel",
                    use_wmma_prefill=False,
                )
                launch_gguf_linear(
                    weight,
                    x_ptr=300,
                    out_ptr=400,
                    rows=16,
                    in_features=5_120,
                    out_features=10_240,
                    backend="hip_gfx1100",
                    stream=7,
                    runtime="runtime-sentinel",
                    use_wmma_prefill=False,
                )
    finally:
        for key, fn in originals.items():
            if fn is None:
                _KERNELS.pop(key, None)
            else:
                register(key, fn, replace=True)
        gguf_linear.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (_Q6_T16_DECODE, (500, 14, 600, 12, 5_120, 10_240)),
        (_Q6_T16_ROWTILE, (100, 14, 200, 8, 5_120, 10_240)),
        (
            _Q6_T16_ROWTILE,
            (
                100 + 8 * 5_120 * 2,
                14,
                200 + 8 * 10_240 * 2,
                4,
                5_120,
                10_240,
            ),
        ),
        (_Q6_T16_ROWTILE, (700, 14, 800, 7, 5_120, 10_240)),
        (
            _Q6_T16_ROWTILE,
            (
                700 + 7 * 5_120 * 2,
                14,
                800 + 7 * 10_240 * 2,
                2,
                5_120,
                10_240,
            ),
        ),
        (_Q6_T16_ROWTILE, (300, 14, 400, 8, 5_120, 10_240)),
        (
            _Q6_T16_ROWTILE,
            (
                300 + 8 * 5_120 * 2,
                14,
                400 + 8 * 10_240 * 2,
                8,
                5_120,
                10_240,
            ),
        ),
    ]


def test_native_batch_decode_routes_c2_c8_q6_head_to_exact_pack8_and_restores() -> None:
    for rows in (2, 4, 8):
        key, _, _ = _capture_launch(
            rows=rows,
            native_batch_decode=True,
            quant_key="gguf_q6_k",
            output_dtype=GGUF_OUTPUT_F32,
            extra_keys=(_Q6_PACK8_F32,),
        )
        assert key == _Q6_PACK8_F32
    key, _, _ = _capture_launch(rows=2)
    assert key == _Q8_DECODE_PACK8


def test_native_batch_decode_keeps_c1_and_bulk_rows_outside_bucket() -> None:
    key, _, _ = _capture_launch(rows=1, native_batch_decode=True)
    assert key == _Q8_DECODE_PACK8
    key, _, _ = _capture_launch(rows=2, native_batch_decode=True)
    assert key == _Q8_DECODE_PACK8
    key, _, _ = _capture_launch(
        rows=9,
        native_batch_decode=True,
        extra_keys=(_Q8_EXACT_PREFILL_TILE8X4,),
    )
    assert key == _Q8_EXACT_PREFILL_TILE8X4



def test_p9_c1_pair_concat_routes_q8_dual_wmma_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    """P9.C1 dispatch pin: Q8_0 shared gate+up prefill uses dual concat WMMA."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    captured: dict[str, object] = {}

    def fake_dual(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    original = resolve(
        backend=_Q8_WMMA_DUAL_PREFILL.backend,
        layer=_Q8_WMMA_DUAL_PREFILL.layer,
        quant=_Q8_WMMA_DUAL_PREFILL.quant,
        variant=_Q8_WMMA_DUAL_PREFILL.variant,
        missing="none",
    )
    try:
        register(_Q8_WMMA_DUAL_PREFILL, fake_dual, replace=True)
        monkeypatch.setattr(
            "hipengine.runtime.gguf_linear.gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out",
            fake_dual,
        )
        assert launch_gguf_linear_pair_concat(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=300,
            rows=512,
            in_features=2048,
            out_features=4096,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        ) is True
    finally:
        if original is None:
            _KERNELS.pop(_Q8_WMMA_DUAL_PREFILL, None)
        else:
            register(_Q8_WMMA_DUAL_PREFILL, original, replace=True)

    assert captured["args"] == (100, 10, 10, 300, 512, 2048, 4096, 4096)
    assert captured["kwargs"] == {
        "tile_m": 16,
        "tile_n": 32,
        "stream": 7,
        "runtime": "runtime-sentinel",
    }


def test_gemv_decode_off_by_default_routes_legacy_pack8_gemv() -> None:
    """Without any opt-in, rows==1 Q8_0 stays on the legacy pack8_gemv decoder."""

    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8


def test_gemv_decode_kwarg_opts_in_q8_0() -> None:
    """Per-call ``use_gemv_decode=True`` rewrites to the P9.B3 GEMV decoder."""

    key, args, kwargs = _capture_launch(rows=1, use_gemv_decode=True)
    assert key == _Q8_GEMV_DECODE
    # Raw ABI: (x, qweight, out, rows, in_f, out_f)
    assert args == (100, 10, 200, 1, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_gemv_decode_env_var_opts_in(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", "1")
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_GEMV_DECODE


def test_gemv_decode_env_var_falsy_keeps_legacy(monkeypatch) -> None:
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", value)
        key, _, _ = _capture_launch(rows=1)
        assert key == _Q8_DECODE_PACK8, f"env value {value!r} should keep legacy decoder"


def test_gemv_decode_env_var_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "On"):
        monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", value)
        key, _, _ = _capture_launch(rows=1)
        assert key == _Q8_GEMV_DECODE, f"env value {value!r} should enable GEMV decode"


def test_gemv_decode_kwarg_overrides_session() -> None:
    """Per-call ``use_gemv_decode=False`` wins over an enabled session."""

    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=1, use_gemv_decode=False)
    assert key == _Q8_DECODE_PACK8


def test_gemv_decode_session_toggle_persists_until_cleared() -> None:
    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_GEMV_DECODE
    set_gemv_decode_enabled(False)
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8
    set_gemv_decode_enabled(None)
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8  # env default is off


def test_gemv_decode_session_context_manager_restores_previous() -> None:
    set_gemv_decode_enabled(False)
    with gemv_decode_session(True):
        key, _, _ = _capture_launch(rows=1)
        assert key == _Q8_GEMV_DECODE
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8


def test_gguf_gemv_decode_enabled_resolver_precedence(monkeypatch) -> None:
    """The resolver mirrors :func:`gguf_wmma_prefill_enabled` precedence."""

    monkeypatch.delenv("HIPENGINE_GGUF_GEMV_DECODE", raising=False)
    set_gemv_decode_enabled(None)
    assert gguf_gemv_decode_enabled() is False
    assert gguf_gemv_decode_enabled(True) is True
    set_gemv_decode_enabled(True)
    assert gguf_gemv_decode_enabled() is True
    assert gguf_gemv_decode_enabled(False) is False
    set_gemv_decode_enabled(None)
    monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", "1")
    assert gguf_gemv_decode_enabled() is True


# ---------------------------------------------------------------------------
# Prefill path unaffected; fallback on missing key.
# ---------------------------------------------------------------------------


def test_gemv_decode_prefill_path_unaffected_by_opt_in() -> None:
    """rows>1 never gets the GEMV-decode rewrite, regardless of opt-in state.

    The raw Q4_K/Q8_0 row-tile rewrite is a separate (default-on) small-B axis;
    disable it here so this test isolates the GEMV-decode opt-in behaviour.
    """

    set_gemv_decode_enabled(True)
    with q4k_rowtile_session(False):
        key, _, _ = _capture_launch(rows=4)
    assert key == _Q8_DECODE_PACK8


def test_gemv_decode_fallback_when_registry_key_missing() -> None:
    """If the P9.B3 kernel is not registered, the rewrite silently falls back.

    The default-off behaviour must be preserved when a runtime is built without
    the new GEMV decode kernels (e.g. partial build trees or older caches).
    """

    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=1, remove_keys=(_Q8_GEMV_DECODE,))
    # Even though opt-in is on, the rewrite returns the original dispatch
    # because the rewritten registry key is missing.
    assert key == _Q8_DECODE_PACK8


# ---------------------------------------------------------------------------
# Q5_K / Q6_K dense decode opt-in. Q5_K uses the exact scale-hoisted
# Laguna decode specialization; Q6_K keeps the P9.B4b specialization.
# ---------------------------------------------------------------------------


def _q_decode_pack8(quant: str) -> KernelKey:
    return KernelKey("hip_gfx1100", "linear", quant, "pack8_gemv_bf16_bf16_out")


def _q_gemv_decode(quant: str) -> KernelKey:
    return KernelKey("hip_gfx1100", "linear", quant, "pack8_gemv_decode_bf16_bf16_out")


def test_raw_q4_k_only_jumps_to_pack8_when_decode_is_enabled() -> None:
    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        output_dtype=GGUF_OUTPUT_F32,
        use_gemv_decode=False,
        extra_keys=(_Q4_SCALAR, _Q4_GEMV_DECODE),
    )
    assert key == _Q4_SCALAR

    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        output_dtype=GGUF_OUTPUT_F32,
        use_gemv_decode=True,
        extra_keys=(_Q4_SCALAR, _Q4_GEMV_DECODE),
    )
    assert key == _Q4_GEMV_DECODE


def test_gemv_decode_q6_k_opt_in_rewrites() -> None:
    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q6_k",
        layout=LAYOUT_RAW_GGUF,
        use_gemv_decode=True,
        extra_keys=(_q_decode_pack8("gguf_q6_k"), _q_gemv_decode("gguf_q6_k")),
    )
    assert key == _q_gemv_decode("gguf_q6_k")


def test_gemv_decode_q5_k_opt_in_rewrites() -> None:
    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q5_k",
        layout=LAYOUT_RAW_GGUF,
        use_gemv_decode=True,
        extra_keys=(_q_decode_pack8("gguf_q5_k"), _q_gemv_decode("gguf_q5_k")),
    )
    assert key == _q_gemv_decode("gguf_q5_k")


def test_variant_specific_library_overrides_quant_default() -> None:
    decode_library = object()
    key, _, kwargs = _capture_launch(
        rows=1,
        quant_key="gguf_q6_k",
        layout=LAYOUT_RAW_GGUF,
        use_gemv_decode=True,
        libraries={
            "gguf_q6_k": object(),
            "gguf_q6_k:pack8_gemv_decode_bf16_bf16_out": decode_library,
        },
        extra_keys=(_q_decode_pack8("gguf_q6_k"), _q_gemv_decode("gguf_q6_k")),
    )
    assert key == _q_gemv_decode("gguf_q6_k")
    assert kwargs["library"] is decode_library
