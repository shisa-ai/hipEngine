from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_f32_rocblas_prefill import (
    register_gguf_q5_k_f32_rocblas_prefill_kernels,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    Q5F32OrderedPrefillSession,
    _raw_k_f32_ordered_prefill_dispatch,
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear,
    q5_f32_ordered_prefill_session,
)
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch


_CAPABILITY = "GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY"
_H5W_POLICY = {
    ("bf16", 3072, 1024): "weight_major_coltile16_rowbatch5",
    ("bf16", 1024, 3072): "weight_major_coltile16_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): "weight_major_coltile16_rowbatch5",
}
_H6E_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
}
_H6E_CALL_WEIGHTS = {
    ("bf16", 3072, 1024): 2,
    ("bf16", 1024, 3072): 46,
    ("f32", 3072, 1024): 94,
}
_H6E_ROLES = tuple(_H6E_CALL_WEIGHTS)


def _base(output_dtype: str, *, backend: str = "hip_gfx1100") -> GGUFLinearDispatch:
    return GGUFLinearDispatch(
        KernelKey(
            backend,
            "linear",
            "gguf_q6_k",
            f"prefill_bf16_{output_dtype}_out",
        ),
        "raw",
    )


def _candidate_key(
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> KernelKey:
    geometry = _H6E_POLICY[(output_dtype, in_features, out_features)]
    return KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k",
        f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
    )


def _session(*, with_activation: bool = True) -> Q5F32OrderedPrefillSession:
    return Q5F32OrderedPrefillSession(
        min_rows=512,
        max_rows=512,
        weight_f32_ptr=0x4000,
        weight_f32_nbytes=150_994_944,
        activation_bf16_ptr=0x5000 if with_activation else 0,
        activation_bf16_nbytes=10_125_312 if with_activation else 0,
        library="ordered-library",
    )


def _install_candidate_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_F32_ORDERED_PREFILL_POLICIES",
        {
            "gguf_q5_k": hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY,
            "gguf_q6_k": _H6E_POLICY,
        },
    )


def test_h6e_runtime_capability_is_default_off_bounded_and_workspace_neutral() -> None:
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _H5W_POLICY
    assert getattr(hip_gfx1100, _CAPABILITY) == _H6E_POLICY
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert set(_H6E_POLICY) == set(_H5W_POLICY)
    assert {
        role for role in _H6E_POLICY if _H6E_POLICY[role] != _H5W_POLICY[role]
    } == set(_H6E_ROLES)
    assert sum(_H6E_CALL_WEIGHTS.values()) == 142
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert (
        LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(max_rows=512)
        == 10_125_312
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256


def test_h6e_runtime_dispatch_is_exact_role_owner_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    session = _session()

    with q5_f32_ordered_prefill_session(session):
        for output_dtype, in_features, out_features in _H6E_ROLES:
            selected = _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected == GGUFLinearDispatch(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q6_k",
                    "f32_ordered_"
                    f"{_H5W_POLICY[(output_dtype, in_features, out_features)]}_"
                    f"bf16_{output_dtype}_out",
                ),
                "raw_k_f32_ordered",
            )

    _install_candidate_policy(monkeypatch)
    with q5_f32_ordered_prefill_session(session):
        for output_dtype, in_features, out_features in _H6E_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == GGUFLinearDispatch(
                _candidate_key(output_dtype, in_features, out_features),
                "raw_k_f32_ordered_activation_tile_k_row",
            )

        retained_n72 = _raw_k_f32_ordered_prefill_dispatch(
            _base("f32"),
            rows=512,
            in_features=3072,
            out_features=72,
        )
        assert retained_n72 == GGUFLinearDispatch(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q6_k",
                "f32_ordered_coltile8_rowbatch4_bf16_f32_out",
            ),
            "raw_k_f32_ordered",
        )

        for rows, output_dtype, in_features, out_features in (
            (511, "bf16", 3072, 1024),
            (513, "bf16", 3072, 1024),
            (512, "bf16", 3072, 3072),
            (512, "bf16", 9216, 3072),
            (512, "f32", 3072, 9216),
        ):
            base = _base(output_dtype)
            assert (
                _raw_k_f32_ordered_prefill_dispatch(
                    base,
                    rows=rows,
                    in_features=in_features,
                    out_features=out_features,
                )
                is base
            )

    with q5_f32_ordered_prefill_session(_session(with_activation=False)):
        base = _base("bf16")
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                base,
                rows=512,
                in_features=3072,
                out_features=1024,
            )
            is base
        )

    missing_key = _candidate_key("bf16", 3072, 1024)
    original_is_registered = gguf_linear_module.is_registered
    monkeypatch.setattr(
        gguf_linear_module,
        "is_registered",
        lambda key: key != missing_key and original_is_registered(key),
    )
    with q5_f32_ordered_prefill_session(session):
        base = _base("bf16")
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                base,
                rows=512,
                in_features=3072,
                out_features=1024,
            )
            is base
        )

    gfx1151 = _base("bf16", backend="hip_gfx1151")
    with q5_f32_ordered_prefill_session(session):
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                gfx1151,
                rows=512,
                in_features=3072,
                out_features=1024,
            )
            is gfx1151
        )


def test_h6e_runtime_launch_reuses_owner_weight_activation_and_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_candidate_policy(monkeypatch)
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    session = _session()
    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k"),
        allocation=lambda name: raw,
    )

    for role_index, (output_dtype, in_features, out_features) in enumerate(
        _H6E_ROLES
    ):
        key = _candidate_key(output_dtype, in_features, out_features)
        original = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        calls: list[tuple[tuple, dict]] = []

        def candidate(*args, **kwargs):
            calls.append((args, kwargs))

        register(key, candidate, replace=True)
        clear_gguf_linear_dispatch_cache()
        try:
            with q5_f32_ordered_prefill_session(session):
                launch_gguf_linear(
                    weight,
                    x_ptr=0x1000 + role_index,
                    out_ptr=0x3000 + role_index,
                    rows=512,
                    in_features=in_features,
                    out_features=out_features,
                    output_dtype=output_dtype,
                    backend="hip_gfx1100",
                    stream=7,
                    runtime="runtime-sentinel",
                )
        finally:
            register(key, original, replace=True)
            clear_gguf_linear_dispatch_cache()

        assert calls == [
            (
                (
                    0x1000 + role_index,
                    0x2000,
                    0x3000 + role_index,
                    0x4000,
                    0x5000,
                    512,
                    in_features,
                    out_features,
                ),
                {
                    "stream": 7,
                    "library": "ordered-library",
                    "runtime": "runtime-sentinel",
                },
            )
        ]
