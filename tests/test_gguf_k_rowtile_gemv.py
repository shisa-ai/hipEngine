"""Correctness for the small-B weight-amortized GGUF raw-K row-tile GEMV.

`gguf_k_prefill_out_rowtile_kernel<...,qtype,ROW_TILE>` is the verifier small-B
(rows 2..8) replacement for the per-row `gguf_k_prefill_out_kernel` for the raw
K-quants Q8_0 (qtype=8), Q5_K (5), Q6_K (6). Q8_0 is the qwen35moe dense
projection quant (attn_qkv/gate, ssm_out), which dominates the target verifier.

Gate: bit-exact vs the per-row kernel + within tolerance of a CPU dequant oracle,
for Q8_0/Q5_K/Q6_K bf16->bf16 and bf16->f32, across rows 2..8 and several shapes.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    register_gguf_k_gemv_kernels,
    gguf_q5_k_gemv_bf16_bf16_out,
    gguf_q5_k_gemv_bf16_f32_out,
    gguf_q5_k_gemv_rowbatch4_bf16_bf16_out,
    gguf_q5_k_gemv_rowbatch4_bf16_f32_out,
    gguf_q5_k_gemv_rowbatch8_bf16_bf16_out,
    gguf_q5_k_gemv_rowbatch8_bf16_f32_out,
    gguf_q5_k_gemv_rowbatch16_bf16_bf16_out,
    gguf_q5_k_gemv_rowbatch16_bf16_f32_out,
    gguf_q5_k_gemv_rowbatch32_bf16_bf16_out,
    gguf_q5_k_gemv_rowbatch32_bf16_f32_out,
    gguf_q5_k_gemv_coltile2_rowbatch16_bf16_bf16_out,
    gguf_q5_k_gemv_coltile2_rowbatch16_bf16_f32_out,
    gguf_q5_k_gemv_coltile4_rowbatch8_bf16_bf16_out,
    gguf_q5_k_gemv_coltile4_rowbatch8_bf16_f32_out,
    gguf_q5_k_gemv_rowtile_bf16_bf16_out,
    gguf_q6_k_gemv_bf16_bf16_out,
    gguf_q6_k_gemv_bf16_f32_out,
    gguf_q6_k_gemv_rowbatch4_bf16_bf16_out,
    gguf_q6_k_gemv_rowbatch4_bf16_f32_out,
    gguf_q6_k_gemv_rowbatch8_bf16_bf16_out,
    gguf_q6_k_gemv_rowbatch8_bf16_f32_out,
    gguf_q6_k_gemv_rowbatch16_bf16_bf16_out,
    gguf_q6_k_gemv_rowbatch16_bf16_f32_out,
    gguf_q6_k_gemv_rowbatch32_bf16_bf16_out,
    gguf_q6_k_gemv_rowbatch32_bf16_f32_out,
    gguf_q6_k_gemv_coltile2_rowbatch16_bf16_bf16_out,
    gguf_q6_k_gemv_coltile2_rowbatch16_bf16_f32_out,
    gguf_q6_k_gemv_coltile4_rowbatch8_bf16_bf16_out,
    gguf_q6_k_gemv_coltile4_rowbatch8_bf16_f32_out,
    gguf_q6_k_gemv_rowtile_bf16_bf16_out,
    gguf_q8_0_gemv_bf16_bf16_out,
    gguf_q8_0_gemv_bf16_f32_out,
    gguf_q8_0_gemv_rowtile_bf16_bf16_out,
    gguf_q8_0_gemv_rowtile_bf16_f32_out,
)
from hipengine.kernels.registry import resolve

QK_K = 256


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def make_q8_0_weight(out_f: int, in_f: int) -> np.ndarray:
    """Valid Q8_0 bytes: per 32-wide block = fp16 scale + 32 int8 quants."""
    blocks = in_f // 32
    rng = np.random.default_rng(out_f * 13 + in_f)
    data = np.empty((out_f, blocks * 34), dtype=np.uint8)
    for c in range(out_f):
        for b in range(blocks):
            base = b * 34
            d = np.float16(0.01 * (1 + ((c + b) % 7)))
            data[c, base : base + 2] = np.asarray([d], dtype=np.float16).view(np.uint8)
            q = rng.integers(-127, 128, size=32, dtype=np.int8)
            data[c, base + 2 : base + 34] = q.view(np.uint8)
    return data


def _q8_0_dequant(weight_row: np.ndarray, in_f: int) -> np.ndarray:
    blocks = in_f // 32
    out = np.empty(in_f, dtype=np.float32)
    for b in range(blocks):
        base = b * 34
        d = weight_row[base : base + 2].view(np.float16).astype(np.float32)[0]
        q = weight_row[base + 2 : base + 34].view(np.int8).astype(np.float32)
        out[b * 32 : (b + 1) * 32] = d * q
    return out


def _cpu_ref_q8_0(x: np.ndarray, qw: np.ndarray, in_f: int, out_f: int) -> np.ndarray:
    w = np.stack([_q8_0_dequant(qw[c], in_f) for c in range(out_f)], axis=0)
    return x.astype(np.float32) @ w.T.astype(np.float32)


def _run(wrapper, x_host, qw, out_host):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    rows, in_f = x_host.shape
    out_f = out_host.shape[1]
    bufs = []
    try:
        xd = malloc(x_host.nbytes, runtime=runtime)
        qd = malloc(qw.nbytes, runtime=runtime)
        od = malloc(out_host.nbytes, runtime=runtime)
        bufs.extend((xd, qd, od))
        copy_host_to_device(xd, host_array_ptr(np.ascontiguousarray(x_host)), runtime=runtime)
        copy_host_to_device(qd, host_array_ptr(np.ascontiguousarray(qw)), runtime=runtime)
        wrapper(xd.ptr, qd.ptr, od.ptr, rows, in_f, out_f, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), od, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)
    return out_host


def test_gguf_k_rowtile_registry_binds() -> None:
    register_gguf_k_gemv_kernels()
    for quant in ("gguf_q8_0", "gguf_q5_k", "gguf_q6_k"):
        for variant in ("rowtile_bf16_bf16_out", "rowtile_bf16_f32_out", "rowtile_f32_f32_out"):
            assert callable(resolve(backend="hip_gfx1100", layer="linear", quant=quant, variant=variant))


@pytest.mark.parametrize("quant", ("gguf_q5_k", "gguf_q6_k"))
def test_gguf_k_large_prefill_rowbatch_registry_binds(quant: str) -> None:
    register_gguf_k_gemv_kernels()
    for row_batch in (4, 8, 16, 32):
        for output_dtype in ("bf16", "f32"):
            assert callable(
                resolve(
                    backend="hip_gfx1100",
                    layer="linear",
                    quant=quant,
                    variant=f"rowbatch{row_batch}_bf16_{output_dtype}_out",
                )
            )


def test_q5k_q6k_output_coltile_registry_binds() -> None:
    register_gguf_k_gemv_kernels()
    for quant in ("gguf_q5_k", "gguf_q6_k"):
        for col_tile, row_batch in ((2, 16), (4, 8)):
            for output_dtype in ("bf16", "f32"):
                assert callable(
                    resolve(
                        backend="hip_gfx1100",
                        layer="linear",
                        quant=quant,
                        variant=(
                            f"coltile{col_tile}_rowbatch{row_batch}_"
                            f"bf16_{output_dtype}_out"
                        ),
                    )
                )


def test_raw_k_prefill_rowbatch16_32_are_not_aliased_to_gfx1151() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, is_registered

    register_gfx1151_kernels(replace=True)
    for quant in ("gguf_q5_k", "gguf_q6_k"):
        for row_batch in (16, 32):
            for output_dtype in ("bf16", "f32"):
                assert not is_registered(
                    KernelKey(
                        "hip_gfx1151",
                        "linear",
                        quant,
                        f"rowbatch{row_batch}_bf16_{output_dtype}_out",
                    )
                )
        for col_tile, row_batch in ((2, 16), (4, 8)):
            for output_dtype in ("bf16", "f32"):
                assert not is_registered(
                    KernelKey(
                        "hip_gfx1151",
                        "linear",
                        quant,
                        f"coltile{col_tile}_rowbatch{row_batch}_bf16_{output_dtype}_out",
                    )
                )


def test_raw_k_prefill_rowbatch_dispatch_is_exactly_scoped() -> None:
    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _raw_k_prefill_rowbatch_dispatch,
    )

    for quant in ("gguf_q5_k", "gguf_q6_k"):
        for output_dtype in ("bf16", "f32"):
            base = GGUFLinearDispatch(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    quant,
                    f"prefill_bf16_{output_dtype}_out",
                ),
                "raw",
            )
            for row_batch in (4, 8, 16, 32):
                selected = _raw_k_prefill_rowbatch_dispatch(
                    base,
                    rows=128,
                    in_features=3072,
                    out_features=72,
                    row_batch=row_batch,
                    variant="rowbatch",
                )
                assert selected.key.variant == (
                    f"rowbatch{row_batch}_bf16_{output_dtype}_out"
                )
                assert selected.abi == "raw"
            for rows in (1, 2, 8):
                assert (
                    _raw_k_prefill_rowbatch_dispatch(
                        base,
                        rows=rows,
                        in_features=3072,
                        out_features=72,
                        row_batch=8,
                        variant="rowbatch",
                    )
                    is base
                )
            assert (
                _raw_k_prefill_rowbatch_dispatch(
                    base,
                    rows=128,
                    in_features=3073,
                    out_features=72,
                    row_batch=8,
                    variant="rowbatch",
                )
                is base
            )
            assert (
                _raw_k_prefill_rowbatch_dispatch(
                    base,
                    rows=128,
                    in_features=3072,
                    out_features=72,
                    row_batch=0,
                    variant="rowbatch",
                )
                is base
            )

    q8 = GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q8_0",
            "prefill_bf16_bf16_out",
        ),
        "raw",
    )
    assert (
        _raw_k_prefill_rowbatch_dispatch(
            q8,
            rows=128,
            in_features=3072,
            out_features=72,
            row_batch=8,
            variant="rowbatch",
        )
        is q8
    )


def test_raw_k_prefill_coltile_dispatch_is_exactly_scoped() -> None:
    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _raw_k_prefill_rowbatch_dispatch,
    )

    qualified = {
        ("gguf_q5_k", "bf16_bf16_out", 12288),
        ("gguf_q5_k", "bf16_f32_out", 6144),
        ("gguf_q5_k", "bf16_f32_out", 9216),
        ("gguf_q6_k", "bf16_f32_out", 9216),
    }
    for quant in ("gguf_q5_k", "gguf_q6_k"):
        for output_dtype in ("bf16", "f32"):
            base = GGUFLinearDispatch(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    quant,
                    f"prefill_bf16_{output_dtype}_out",
                ),
                "raw",
            )
            for out_features in (72, 6144, 9216, 12288):
                selected = _raw_k_prefill_rowbatch_dispatch(
                    base,
                    rows=512,
                    in_features=3072,
                    out_features=out_features,
                    row_batch=32,
                    variant="coltile",
                )
                geometry = (
                    "coltile2_rowbatch16"
                    if (quant, f"bf16_{output_dtype}_out", out_features) in qualified
                    else "coltile4_rowbatch8"
                )
                assert selected.key.variant == (
                    f"{geometry}_bf16_{output_dtype}_out"
                )
                assert selected.abi == "raw"

            other_k = _raw_k_prefill_rowbatch_dispatch(
                base,
                rows=512,
                in_features=3328,
                out_features=9216,
                row_batch=32,
                variant="coltile",
            )
            assert other_k.key.variant == (
                f"coltile4_rowbatch8_bf16_{output_dtype}_out"
            )

            for row_batch, out_features in ((16, 72), (32, 73)):
                fallback = _raw_k_prefill_rowbatch_dispatch(
                    base,
                    rows=512,
                    in_features=3072,
                    out_features=out_features,
                    row_batch=row_batch,
                    variant="coltile",
                )
                assert fallback.key.variant == (
                    f"rowbatch{row_batch}_bf16_{output_dtype}_out"
                )

    unsupported = GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q5_k",
            "prefill_bf16_bf16_out",
        ),
        "raw",
    )
    assert (
        _raw_k_prefill_rowbatch_dispatch(
            unsupported,
            rows=512,
            in_features=3072,
            out_features=72,
            row_batch=32,
            variant="coltile",
        )
        is unsupported
    )


def test_raw_k_f32_ordered_prefill_dispatch_is_owner_and_role_scoped(
    monkeypatch,
) -> None:
    from hipengine.kernels import hip_gfx1100 as package
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_f32_rocblas_prefill import (
        register_gguf_q5_k_f32_rocblas_prefill_kernels,
    )
    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        Q5F32OrderedPrefillSession,
        _raw_k_f32_ordered_prefill_dispatch,
        q5_f32_ordered_prefill_session,
    )

    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    monkeypatch.setattr(
        package,
        "GGUF_F32_ORDERED_PREFILL_QUANTS",
        frozenset(("gguf_q5_k",)),
    )

    def base(quant: str, output_dtype: str) -> GGUFLinearDispatch:
        return GGUFLinearDispatch(
            KernelKey(
                "hip_gfx1100",
                "linear",
                quant,
                f"prefill_bf16_{output_dtype}_out",
            ),
            "raw",
        )

    session = Q5F32OrderedPrefillSession(
        min_rows=512,
        max_rows=512,
        weight_f32_ptr=1000,
        weight_f32_nbytes=150_994_944,
        library="ordered-library",
    )
    q5_qualified = {
        ("bf16", 3072, 1024): "weight_major_tile_k_col_coltile8_rowbatch4",
        ("bf16", 3072, 12288): "weight_major_coltile8_rowbatch12",
        ("bf16", 6144, 3072): "weight_major_tile_k_col_coltile16_rowbatch5",
        ("bf16", 9216, 3072): "weight_major_coltile12_rowbatch8",
        ("f32", 3072, 48): "coltile12_rowbatch4",
        ("f32", 3072, 72): "coltile8_rowbatch4",
        ("f32", 3072, 6144): "weight_major_tile_k_col_coltile16_rowbatch5",
        ("f32", 3072, 9216): "weight_major_tile_k_col_coltile8_rowbatch10",
    }
    q6_qualified = {
        ("bf16", 3072, 1024): "weight_major_coltile16_rowbatch5",
        ("bf16", 1024, 3072): "weight_major_coltile16_rowbatch4",
        ("f32", 3072, 72): "coltile8_rowbatch4",
        ("f32", 3072, 1024): "weight_major_coltile16_rowbatch5",
    }
    for output_dtype, in_features, out_features in q5_qualified:
        dispatch = base("gguf_q5_k", output_dtype)
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                dispatch,
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            is dispatch
        )

    with q5_f32_ordered_prefill_session(session):
        for role, geometry in q5_qualified.items():
            output_dtype, in_features, out_features = role
            selected = _raw_k_f32_ordered_prefill_dispatch(
                base("gguf_q5_k", output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected == GGUFLinearDispatch(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q5_k",
                    f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
                ),
                "raw_k_f32_ordered",
            )

        for output_dtype, in_features, out_features in q6_qualified:
            dispatch = base("gguf_q6_k", output_dtype)
            assert (
                _raw_k_f32_ordered_prefill_dispatch(
                    dispatch,
                    rows=512,
                    in_features=in_features,
                    out_features=out_features,
                )
                is dispatch
            )

        monkeypatch.setattr(
            package,
            "GGUF_F32_ORDERED_PREFILL_QUANTS",
            frozenset(("gguf_q5_k", "gguf_q6_k")),
        )
        for role, geometry in q6_qualified.items():
            output_dtype, in_features, out_features = role
            selected = _raw_k_f32_ordered_prefill_dispatch(
                base("gguf_q6_k", output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected == GGUFLinearDispatch(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q6_k",
                    f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
                ),
                "raw_k_f32_ordered",
            )

        for quant, rows, output_dtype, in_features, out_features in (
            ("gguf_q5_k", 511, "bf16", 3072, 1024),
            ("gguf_q5_k", 512, "bf16", 9216, 4096),
            ("gguf_q5_k", 512, "bf16", 3328, 1024),
            ("gguf_q5_k", 512, "f32", 3072, 96),
            ("gguf_q6_k", 511, "bf16", 3072, 1024),
            ("gguf_q6_k", 512, "bf16", 9216, 3072),
            ("gguf_q6_k", 512, "bf16", 12288, 3072),
            ("gguf_q6_k", 512, "f32", 3072, 9216),
        ):
            dispatch = base(quant, output_dtype)
            assert (
                _raw_k_f32_ordered_prefill_dispatch(
                    dispatch,
                    rows=rows,
                    in_features=in_features,
                    out_features=out_features,
                )
                is dispatch
            )
        unsupported = GGUFLinearDispatch(
            KernelKey(
                "hip_gfx1151",
                "linear",
                "gguf_q6_k",
                "prefill_bf16_f32_out",
            ),
            "raw",
        )
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                unsupported,
                rows=512,
                in_features=3072,
                out_features=72,
            )
            is unsupported
        )

    with pytest.raises(ValueError, match="min_rows"):
        Q5F32OrderedPrefillSession(
            min_rows=513,
            max_rows=512,
            weight_f32_ptr=1000,
            weight_f32_nbytes=150_994_944,
            library="ordered-library",
        )


@pytest.mark.parametrize("quant", ["gguf_q5_k", "gguf_q6_k"])
def test_launch_gguf_linear_honors_raw_k_f32_ordered_prefill_session(
    quant: str,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from hipengine.kernels import hip_gfx1100 as package
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_f32_rocblas_prefill import (
        register_gguf_q5_k_f32_rocblas_prefill_kernels,
    )
    from hipengine.kernels.registry import KernelKey, register
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
    from hipengine.runtime.gguf_linear import (
        Q5F32OrderedPrefillSession,
        clear_gguf_linear_dispatch_cache,
        launch_gguf_linear,
        q5_f32_ordered_prefill_session,
    )

    monkeypatch.setattr(
        package,
        "GGUF_F32_ORDERED_PREFILL_QUANTS",
        frozenset(("gguf_q5_k", "gguf_q6_k")),
    )
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        quant,
        "f32_ordered_coltile8_rowbatch4_bf16_f32_out",
    )
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def candidate(*args, **kwargs):
        calls.append((args, kwargs))

    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=200))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key=quant),
        allocation=lambda name: raw,
    )
    session = Q5F32OrderedPrefillSession(
        min_rows=512,
        max_rows=512,
        weight_f32_ptr=400,
        weight_f32_nbytes=150_994_944,
        library="ordered-library",
    )
    register(key, candidate, replace=True)
    clear_gguf_linear_dispatch_cache()
    try:
        with q5_f32_ordered_prefill_session(session):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=300,
                rows=512,
                in_features=3072,
                out_features=72,
                output_dtype="f32",
                backend="hip_gfx1100",
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        register(key, original, replace=True)
        clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 200, 300, 400, 512, 3072, 72),
            {
                "stream": 7,
                "library": "ordered-library",
                "runtime": "runtime-sentinel",
            },
        )
    ]


def test_raw_k_prefill_rowbatch_session_is_nested_and_fail_closed() -> None:
    from hipengine.runtime.gguf_linear import (
        raw_k_prefill_rowbatch,
        raw_k_prefill_rowbatch_session,
        raw_k_prefill_variant,
        raw_k_prefill_variant_session,
    )

    assert raw_k_prefill_rowbatch() == 0
    assert raw_k_prefill_variant() == "rowbatch"
    with raw_k_prefill_rowbatch_session(32):
        assert raw_k_prefill_rowbatch() == 32
        with raw_k_prefill_rowbatch_session(16):
            assert raw_k_prefill_rowbatch() == 16
        with raw_k_prefill_variant_session("coltile"):
            assert raw_k_prefill_variant() == "coltile"
        assert raw_k_prefill_rowbatch() == 32
        assert raw_k_prefill_variant() == "rowbatch"
    assert raw_k_prefill_rowbatch() == 0
    with pytest.raises(ValueError, match="row batch"):
        with raw_k_prefill_rowbatch_session(64):
            pass
    with pytest.raises(ValueError, match="variant"):
        with raw_k_prefill_variant_session("unknown"):
            pass


def test_launch_gguf_linear_honors_raw_k_prefill_rowbatch_session() -> None:
    from types import SimpleNamespace

    from hipengine.kernels.registry import KernelKey, register
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
    from hipengine.runtime.gguf_linear import (
        clear_gguf_linear_dispatch_cache,
        launch_gguf_linear,
        raw_k_prefill_rowbatch_session,
    )

    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k",
        "rowbatch8_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def candidate(*args, **kwargs):
        calls.append((args, kwargs))

    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=200))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k"),
        allocation=lambda name: raw,
    )
    register(key, candidate, replace=True)
    clear_gguf_linear_dispatch_cache()
    try:
        with raw_k_prefill_rowbatch_session(8):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=300,
                rows=128,
                in_features=3072,
                out_features=72,
                backend="hip_gfx1100",
                runtime="runtime-sentinel",
            )
    finally:
        register(key, original, replace=True)
        clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 200, 300, 128, 3072, 72),
            {"stream": 0, "runtime": "runtime-sentinel"},
        )
    ]


def test_launch_gguf_linear_honors_raw_k_prefill_coltile_session() -> None:
    from types import SimpleNamespace

    from hipengine.kernels.registry import KernelKey, register
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
    from hipengine.runtime.gguf_linear import (
        clear_gguf_linear_dispatch_cache,
        launch_gguf_linear,
        raw_k_prefill_rowbatch_session,
        raw_k_prefill_variant_session,
    )

    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k",
        "coltile4_rowbatch8_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def candidate(*args, **kwargs):
        calls.append((args, kwargs))

    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=200))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k"),
        allocation=lambda name: raw,
    )
    register(key, candidate, replace=True)
    clear_gguf_linear_dispatch_cache()
    try:
        with (
            raw_k_prefill_rowbatch_session(32),
            raw_k_prefill_variant_session("coltile"),
        ):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=300,
                rows=512,
                in_features=3072,
                out_features=72,
                backend="hip_gfx1100",
                runtime="runtime-sentinel",
            )
    finally:
        register(key, original, replace=True)
        clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 200, 300, 512, 3072, 72),
            {"stream": 0, "runtime": "runtime-sentinel"},
        )
    ]


_SHAPES = [(256, 16), (512, 48), (1024, 64)]
_ROWS = [2, 3, 4, 8]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("in_f,out_f", _SHAPES)
def test_q8_0_rowtile_bit_exact_and_oracle(rows, in_f, out_f) -> None:
    qw = make_q8_0_weight(out_f, in_f)
    rng = np.random.default_rng(7 + rows + in_f)
    x = rng.standard_normal((rows, in_f)).astype(np.float32) * 0.1
    xb = _bf16_bits(x).reshape(rows, in_f)

    ref = _run(gguf_q8_0_gemv_bf16_bf16_out, xb, qw, np.zeros((rows, out_f), np.uint16)).copy()
    got = _run(gguf_q8_0_gemv_rowtile_bf16_bf16_out, xb, qw, np.zeros((rows, out_f), np.uint16)).copy()
    np.testing.assert_array_equal(got, ref)  # bit-exact vs per-row

    ref32 = _run(gguf_q8_0_gemv_bf16_f32_out, xb, qw, np.zeros((rows, out_f), np.float32)).copy()
    got32 = _run(gguf_q8_0_gemv_rowtile_bf16_f32_out, xb, qw, np.zeros((rows, out_f), np.float32)).copy()
    np.testing.assert_array_equal(got32, ref32)
    cpu = _cpu_ref_q8_0(_bf16_to_f32(xb), qw, in_f, out_f)
    np.testing.assert_allclose(got32, cpu, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 4, 8])
def test_q5k_q6k_rowtile_bit_exact_vs_per_row(rows) -> None:
    # Random raw bytes are valid Q5_K/Q6_K superblocks; only kernel equivalence
    # is asserted here (the Q8_0 case covers the CPU oracle).
    in_f, out_f = 512, 32
    rng = np.random.default_rng(100 + rows)
    qw5 = rng.integers(0, 256, size=(out_f, (in_f // QK_K) * 176), dtype=np.uint8)  # Q5_K block = 176 B
    qw6 = rng.integers(0, 256, size=(out_f, (in_f // QK_K) * 210), dtype=np.uint8)  # Q6_K block = 210 B
    x = rng.standard_normal((rows, in_f)).astype(np.float32) * 0.1
    xb = _bf16_bits(x).reshape(rows, in_f)

    ref5 = _run(gguf_q5_k_gemv_bf16_bf16_out, xb, qw5, np.zeros((rows, out_f), np.uint16)).copy()
    got5 = _run(gguf_q5_k_gemv_rowtile_bf16_bf16_out, xb, qw5, np.zeros((rows, out_f), np.uint16)).copy()
    np.testing.assert_array_equal(got5, ref5)

    ref6 = _run(gguf_q6_k_gemv_bf16_bf16_out, xb, qw6, np.zeros((rows, out_f), np.uint16)).copy()
    got6 = _run(gguf_q6_k_gemv_rowtile_bf16_bf16_out, xb, qw6, np.zeros((rows, out_f), np.uint16)).copy()
    np.testing.assert_array_equal(got6, ref6)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [9, 17, 33])
def test_q5k_q6k_large_prefill_rowbatch_tails_are_bit_exact(rows: int) -> None:
    """Fixed rowbatch4/8/16/32 preserves scalar rows and partial tails."""

    from tests.test_gguf_k_gemv import make_q5_k_weight, make_q6_k_weight

    in_f, out_f = 512, 48
    rng = np.random.default_rng(20260728 + rows)
    xb = _bf16_bits(rng.standard_normal((rows, in_f)).astype(np.float32) * 0.1)
    cases = (
        (
            make_q5_k_weight(out_f, in_f),
            gguf_q5_k_gemv_bf16_bf16_out,
            gguf_q5_k_gemv_bf16_f32_out,
            gguf_q5_k_gemv_rowbatch4_bf16_bf16_out,
            gguf_q5_k_gemv_rowbatch4_bf16_f32_out,
            gguf_q5_k_gemv_rowbatch8_bf16_bf16_out,
            gguf_q5_k_gemv_rowbatch8_bf16_f32_out,
            gguf_q5_k_gemv_rowbatch16_bf16_bf16_out,
            gguf_q5_k_gemv_rowbatch16_bf16_f32_out,
            gguf_q5_k_gemv_rowbatch32_bf16_bf16_out,
            gguf_q5_k_gemv_rowbatch32_bf16_f32_out,
        ),
        (
            make_q6_k_weight(out_f, in_f),
            gguf_q6_k_gemv_bf16_bf16_out,
            gguf_q6_k_gemv_bf16_f32_out,
            gguf_q6_k_gemv_rowbatch4_bf16_bf16_out,
            gguf_q6_k_gemv_rowbatch4_bf16_f32_out,
            gguf_q6_k_gemv_rowbatch8_bf16_bf16_out,
            gguf_q6_k_gemv_rowbatch8_bf16_f32_out,
            gguf_q6_k_gemv_rowbatch16_bf16_bf16_out,
            gguf_q6_k_gemv_rowbatch16_bf16_f32_out,
            gguf_q6_k_gemv_rowbatch32_bf16_bf16_out,
            gguf_q6_k_gemv_rowbatch32_bf16_f32_out,
        ),
    )
    for weight, scalar_bf16, scalar_f32, *rowbatch in cases:
        expected_bf16 = _run(
            scalar_bf16, xb, weight, np.zeros((rows, out_f), np.uint16)
        ).copy()
        expected_f32 = _run(
            scalar_f32, xb, weight, np.zeros((rows, out_f), np.float32)
        ).copy()
        for index, wrapper in enumerate(rowbatch):
            dtype = np.uint16 if index % 2 == 0 else np.float32
            expected = expected_bf16 if index % 2 == 0 else expected_f32
            actual = _run(wrapper, xb, weight, np.zeros((rows, out_f), dtype)).copy()
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [17, 33])
def test_q5k_q6k_output_coltiles_are_bit_exact(rows: int) -> None:
    """Constant-accumulator column tiles preserve RB32 bytes and tails."""

    from tests.test_gguf_k_gemv import make_q5_k_weight, make_q6_k_weight

    in_f, out_f = 512, 48
    rng = np.random.default_rng(20260729 + rows)
    xb = _bf16_bits(rng.standard_normal((rows, in_f)).astype(np.float32) * 0.1)
    cases = (
        (
            make_q5_k_weight(out_f, in_f),
            gguf_q5_k_gemv_rowbatch32_bf16_bf16_out,
            gguf_q5_k_gemv_rowbatch32_bf16_f32_out,
            gguf_q5_k_gemv_coltile2_rowbatch16_bf16_bf16_out,
            gguf_q5_k_gemv_coltile2_rowbatch16_bf16_f32_out,
            gguf_q5_k_gemv_coltile4_rowbatch8_bf16_bf16_out,
            gguf_q5_k_gemv_coltile4_rowbatch8_bf16_f32_out,
        ),
        (
            make_q6_k_weight(out_f, in_f),
            gguf_q6_k_gemv_rowbatch32_bf16_bf16_out,
            gguf_q6_k_gemv_rowbatch32_bf16_f32_out,
            gguf_q6_k_gemv_coltile2_rowbatch16_bf16_bf16_out,
            gguf_q6_k_gemv_coltile2_rowbatch16_bf16_f32_out,
            gguf_q6_k_gemv_coltile4_rowbatch8_bf16_bf16_out,
            gguf_q6_k_gemv_coltile4_rowbatch8_bf16_f32_out,
        ),
    )
    for weight, control_bf16, control_f32, *candidates in cases:
        expected_bf16 = _run(
            control_bf16, xb, weight, np.zeros((rows, out_f), np.uint16)
        ).copy()
        expected_f32 = _run(
            control_f32, xb, weight, np.zeros((rows, out_f), np.float32)
        ).copy()
        for index, wrapper in enumerate(candidates):
            dtype = np.uint16 if index % 2 == 0 else np.float32
            expected = expected_bf16 if index % 2 == 0 else expected_f32
            actual = _run(wrapper, xb, weight, np.zeros((rows, out_f), dtype)).copy()
            np.testing.assert_array_equal(actual, expected)
