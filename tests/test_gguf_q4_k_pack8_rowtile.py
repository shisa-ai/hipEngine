"""Exact small-row reuse for resident-pack8 GGUF Q4_K projections."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q4_k_pack8_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_exact_prefill_tile8x8_bf16_bf16_out,
    gguf_q4_k_pack8_gemv_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _pack8_rowtile_dispatch,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    native_batch_decode_session,
)
from tests.test_gguf_q4_k_gemv import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).copy()
    words += 0x7FFF + ((words >> 16) & 1)
    return (words >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probs = np.exp(shifted, dtype=np.float64)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def test_pack8_rowtile_registry_and_dispatch_contract() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_rowtile_bf16_bf16_out",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is gguf_q4_k_pack8_rowtile_bf16_bf16_out
    full_k_batch_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_m",
        "pack8_full_k_grid_y_native_exact_bf16_bf16_out",
    )
    assert resolve(
        backend=full_k_batch_key.backend,
        layer=full_k_batch_key.layer,
        quant=full_k_batch_key.quant,
        variant=full_k_batch_key.variant,
    ) is gguf_q4_k_pack8_gemv_bf16_bf16_out

    base = GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k",
            "pack8_prefill_bf16_bf16_out",
        ),
        "pack8",
    )
    for rows in (2, 3, 4):
        selected = _pack8_rowtile_dispatch(
            base,
            rows=rows,
            use_rowtile=True,
            native_batch=True,
        )
        assert selected.key == key
        assert selected.abi == "pack8"
    for rows in (1, 5, 8):
        assert (
            _pack8_rowtile_dispatch(
                base,
                rows=rows,
                use_rowtile=True,
                native_batch=True,
            )
            is base
        )
    assert (
        _pack8_rowtile_dispatch(
            base,
            rows=4,
            use_rowtile=False,
            native_batch=True,
        )
        is base
    )
    assert (
        _pack8_rowtile_dispatch(
            base,
            rows=4,
            use_rowtile=True,
            native_batch=False,
        )
        is base
    )


_EXACT_PREFILL_CANDIDATES = (
    (
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
        gguf_q4_k_pack8_exact_prefill_tile8x8_bf16_bf16_out,
    ),
)


def test_pack8_exact_prefill_candidate_registry_contract() -> None:
    for variant, wrapper in _EXACT_PREFILL_CANDIDATES:
        assert resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant=variant,
        ) is wrapper


def _fake_pack8_weight(*, sidecar_name: str | None = None):
    allocations = {
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
    }
    if sidecar_name is not None:
        allocations[sidecar_name] = SimpleNamespace(
            tensor=SimpleNamespace(
                ptr=15 if sidecar_name == "decode_tiles" else 16
            )
        )

    class Weight:
        spec = SimpleNamespace(layout="q4_k_pack8", quant_key="gguf_q4_k")
        backend = "hip_gfx1100"

        def allocation(self, name: str):
            return allocations[name]

    return Weight()


@pytest.mark.parametrize("rows", (512, 1024))
def test_populated_pack8_prefill_routes_to_exact_tile8x8(rows: int) -> None:
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_prefill_bf16_bf16_out",
    )
    original_candidate = resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    )
    original_fallback = resolve(
        backend=fallback_key.backend,
        layer=fallback_key.layer,
        quant=fallback_key.quant,
        variant=fallback_key.variant,
    )
    calls: list[tuple[str, tuple, dict]] = []

    def fake_candidate(*args, **kwargs):
        calls.append(("tile8x8", args, kwargs))

    def fake_fallback(*args, **kwargs):
        calls.append(("fallback", args, kwargs))

    register(candidate_key, fake_candidate, replace=True)
    register(fallback_key, fake_fallback, replace=True)
    try:
        launch_gguf_linear(
            _fake_pack8_weight(),
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=5120,
            out_features=17408,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
            use_gemv_decode=False,
        )
    finally:
        register(candidate_key, original_candidate, replace=True)
        register(fallback_key, original_fallback, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            "tile8x8",
            (100, 11, 12, 13, 200, rows, 5120, 17408),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


@pytest.mark.parametrize(
    ("rows", "use_wmma_prefill"),
    ((511, True), (512, False)),
)
def test_populated_pack8_prefill_keeps_exact_fallback_outside_policy(
    rows: int,
    use_wmma_prefill: bool,
) -> None:
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_prefill_bf16_bf16_out",
    )
    original_candidate = resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    )
    original_fallback = resolve(
        backend=fallback_key.backend,
        layer=fallback_key.layer,
        quant=fallback_key.quant,
        variant=fallback_key.variant,
    )
    calls: list[str] = []
    register(candidate_key, lambda *args, **kwargs: calls.append("tile8x8"), replace=True)
    register(fallback_key, lambda *args, **kwargs: calls.append("fallback"), replace=True)
    try:
        launch_gguf_linear(
            _fake_pack8_weight(),
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=5120,
            out_features=17408,
            use_wmma_prefill=use_wmma_prefill,
            use_gemv_decode=False,
        )
    finally:
        register(candidate_key, original_candidate, replace=True)
        register(fallback_key, original_fallback, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == ["fallback"]


def test_populated_pack8_prefill_missing_tile8x8_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_prefill_bf16_bf16_out",
    )
    original_candidate = resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    )
    original_fallback = resolve(
        backend=fallback_key.backend,
        layer=fallback_key.layer,
        quant=fallback_key.quant,
        variant=fallback_key.variant,
    )
    calls: list[str] = []
    dual_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_dual_prefill_bf16_bf16_out",
        lambda *args, **kwargs: dual_calls.append((args, kwargs)),
    )
    register(candidate_key, None, replace=True)  # type: ignore[arg-type]
    register(fallback_key, lambda *args, **kwargs: calls.append("fallback"), replace=True)
    try:
        launch_gguf_linear(
            _fake_pack8_weight(),
            x_ptr=100,
            out_ptr=200,
            rows=512,
            in_features=5120,
            out_features=17408,
            use_wmma_prefill=True,
            use_gemv_decode=False,
        )
        assert launch_gguf_linear_pair(
            _fake_pack8_weight(),
            _fake_pack8_weight(),
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=512,
            in_features=5120,
            out_features=17408,
            use_wmma_prefill=True,
            use_gemv_decode=False,
        )
    finally:
        register(candidate_key, original_candidate, replace=True)
        register(fallback_key, original_fallback, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == ["fallback"]
    assert len(dual_calls) == 1


def test_populated_pack8_pair_declines_legacy_dual_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dual_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_dual_prefill_bf16_bf16_out",
        lambda *args, **kwargs: dual_calls.append((args, kwargs)),
    )
    assert not launch_gguf_linear_pair(
        _fake_pack8_weight(),
        _fake_pack8_weight(),
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        rows=512,
        in_features=5120,
        out_features=17408,
        use_wmma_prefill=True,
        use_gemv_decode=False,
    )
    assert dual_calls == []


def test_native_pack8_pair_uses_two_bounded_rowtiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations = {
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
    }

    class Weight:
        spec = SimpleNamespace(layout="q4_k_pack8", quant_key="gguf_q4_k")
        backend = "hip_gfx1100"

        def allocation(self, name: str):
            return allocations[name]

    rowtile_calls: list[tuple[tuple, dict]] = []
    dual_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: rowtile_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_dual_prefill_bf16_bf16_out",
        lambda *args, **kwargs: dual_calls.append((args, kwargs)),
    )
    with native_batch_decode_session(True):
        assert launch_gguf_linear_pair(
            Weight(),
            Weight(),
            100,
            200,
            300,
            4,
            5120,
            17408,
            use_wmma_prefill=False,
            use_gemv_decode=True,
        )
    assert len(rowtile_calls) == 2
    assert dual_calls == []

    assert launch_gguf_linear_pair(
        Weight(),
        Weight(),
        100,
        200,
        300,
        4,
        5120,
        17408,
        use_wmma_prefill=False,
        use_gemv_decode=True,
    )
    assert len(dual_calls) == 1


def test_native_pack8_single_uses_measured_t16_sidecar_policy() -> None:
    t16_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_bf16_out",
    )
    pack8_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_rowtile_bf16_bf16_out",
    )
    pack8_prefill_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_prefill_bf16_bf16_out",
    )
    original_t16 = resolve(
        backend=t16_key.backend,
        layer=t16_key.layer,
        quant=t16_key.quant,
        variant=t16_key.variant,
    )
    original_pack8 = resolve(
        backend=pack8_key.backend,
        layer=pack8_key.layer,
        quant=pack8_key.quant,
        variant=pack8_key.variant,
    )
    original_pack8_prefill = resolve(
        backend=pack8_prefill_key.backend,
        layer=pack8_prefill_key.layer,
        quant=pack8_prefill_key.quant,
        variant=pack8_prefill_key.variant,
    )
    calls: list[tuple[str, tuple]] = []
    register(
        t16_key,
        lambda *args, **_kwargs: calls.append(("t16", args)),
        replace=True,
    )
    register(
        pack8_key,
        lambda *args, **_kwargs: calls.append(("pack8", args)),
        replace=True,
    )
    register(
        pack8_prefill_key,
        lambda *args, **_kwargs: calls.append(("pack8", args)),
        replace=True,
    )
    try:
        with native_batch_decode_session(True):
            launch_gguf_linear(
                _fake_pack8_weight(sidecar_name="decode_tiles"),
                100,
                200,
                2,
                5120,
                6144,
                use_wmma_prefill=False,
                use_gemv_decode=True,
            )
            launch_gguf_linear(
                _fake_pack8_weight(sidecar_name="decode_tiles_r3plus"),
                100,
                200,
                2,
                5120,
                10240,
                use_wmma_prefill=False,
                use_gemv_decode=True,
            )
            for rows in (3, 4):
                launch_gguf_linear(
                    _fake_pack8_weight(sidecar_name="decode_tiles_r3plus"),
                    100,
                    200,
                    rows,
                    5120,
                    10240,
                    use_wmma_prefill=False,
                    use_gemv_decode=True,
                )
        launch_gguf_linear(
            _fake_pack8_weight(sidecar_name="decode_tiles"),
            100,
            200,
            3,
            5120,
            6144,
            use_wmma_prefill=False,
            use_gemv_decode=True,
        )
    finally:
        register(t16_key, original_t16, replace=True)
        register(pack8_key, original_pack8, replace=True)
        register(pack8_prefill_key, original_pack8_prefill, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert [kind for kind, _args in calls] == [
        "t16",
        "pack8",
        "t16",
        "t16",
        "pack8",
    ]
    assert calls[0][1] == (100, 15, 200, 2, 5120, 6144)
    assert calls[2][1] == (100, 16, 200, 3, 5120, 10240)
    assert calls[3][1] == (100, 16, 200, 4, 5120, 10240)


def test_pack8_rowtile_wrapper_rejects_non_verifier_shapes() -> None:
    for rows in (1, 5):
        with pytest.raises(ValueError, match="rows must be 2, 3, or 4"):
            gguf_q4_k_pack8_rowtile_bf16_bf16_out(
                1,
                2,
                3,
                4,
                5,
                rows,
                256,
                16,
            )
    with pytest.raises(ValueError, match="threads must be 0 or 32"):
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            4,
            256,
            16,
            threads=64,
        )


def _run_projection(
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = make_q4_k_weight(out_features, in_features)
    packed = repack_gguf_q4_k_pack8(raw)
    rng = np.random.default_rng(0xD27 + rows * 17 + in_features + out_features)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.15, size=(rows, in_features)).astype(np.float32)
    )
    control = np.empty((rows, out_features), dtype=np.uint16)
    candidate = np.empty_like(control)
    arrays = (x_bits, packed.qweight, packed.scales, packed.mins)
    inputs = [malloc(array.nbytes) for array in arrays]
    outputs = [malloc(control.nbytes), malloc(candidate.nbytes)]
    library = build_gguf_q4_k_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, q_d, s_d, m_d = inputs
        control_d, candidate_d = outputs
        common = (
            x_d.ptr,
            q_d.ptr,
            s_d.ptr,
            m_d.ptr,
        )
        gguf_q4_k_pack8_gemv_bf16_bf16_out(
            *common,
            control_d.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            *common,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        for host, allocation in zip((control, candidate), outputs, strict=True):
            copy_device_to_host(host_array_ptr(host), allocation, host.nbytes)
    finally:
        for allocation in (*outputs, *inputs):
            free(allocation)
    cpu = gguf_q4_k_pack8_gemv(
        _bf16_f32(x_bits),
        packed.qweight,
        packed.scales,
        packed.mins,
    )
    return control, candidate, cpu


def _run_exact_prefill_candidate(
    candidate,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = make_q4_k_weight(out_features, in_features)
    packed = repack_gguf_q4_k_pack8(raw)
    rng = np.random.default_rng(0xE27 + rows * 17 + in_features + out_features)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.15, size=(rows, in_features)).astype(np.float32)
    )
    control = np.empty((rows, out_features), dtype=np.uint16)
    candidate_out = np.empty_like(control)
    arrays = (x_bits, packed.qweight, packed.scales, packed.mins)
    inputs = [malloc(array.nbytes) for array in arrays]
    outputs = [malloc(control.nbytes), malloc(candidate_out.nbytes)]
    library = build_gguf_q4_k_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, q_d, s_d, m_d = inputs
        control_d, candidate_d = outputs
        common = (x_d.ptr, q_d.ptr, s_d.ptr, m_d.ptr)
        gguf_q4_k_pack8_gemv_bf16_bf16_out(
            *common,
            control_d.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        candidate(
            *common,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        for host, allocation in zip((control, candidate_out), outputs, strict=True):
            copy_device_to_host(host_array_ptr(host), allocation, host.nbytes)
    finally:
        for allocation in (*outputs, *inputs):
            free(allocation)
    return control, candidate_out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (5, 7))
@pytest.mark.parametrize(
    "variant,candidate",
    _EXACT_PREFILL_CANDIDATES,
    ids=[
        variant.removeprefix("pack8_exact_prefill_").removesuffix("_bf16_bf16_out")
        for variant, _ in _EXACT_PREFILL_CANDIDATES
    ],
)
def test_pack8_exact_prefill_candidates_are_bit_exact_for_partial_tiles(
    variant: str,
    candidate,
    rows: int,
) -> None:
    del variant
    control, candidate_out = _run_exact_prefill_candidate(
        candidate,
        rows=rows,
        in_features=512,
        out_features=64,
    )
    np.testing.assert_array_equal(candidate_out, control)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4))
@pytest.mark.parametrize(
    "in_features,out_features",
    ((256, 16), (512, 64), (1024, 128)),
)
def test_pack8_rowtile_is_bit_exact_and_passes_cpu_gate(
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    control, candidate, cpu = _run_projection(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    np.testing.assert_array_equal(candidate, control)

    gpu = _bf16_f32(candidate)
    p = _softmax(cpu)
    q = _softmax(gpu)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    top1 = np.mean(np.argmax(cpu, axis=-1) == np.argmax(gpu, axis=-1))
    assert float(np.max(kl)) <= 0.05
    assert float(top1) >= 0.90
