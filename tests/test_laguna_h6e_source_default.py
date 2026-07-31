from __future__ import annotations

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
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
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch


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
_H6E_ROLES = (
    ("bf16", 3072, 1024),
    ("bf16", 1024, 3072),
    ("f32", 3072, 1024),
)


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


def _selected(
    output_dtype: str,
    in_features: int,
    out_features: int,
    geometry: str,
) -> GGUFLinearDispatch:
    uses_activation = "_activation_tile_k_row_" in geometry
    return GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
        ),
        (
            "raw_k_f32_ordered_activation_tile_k_row"
            if uses_activation
            else "raw_k_f32_ordered"
        ),
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


def _install_policy(
    monkeypatch: pytest.MonkeyPatch,
    q6_policy: dict[tuple[str, int, int], str],
) -> None:
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_F32_ORDERED_PREFILL_POLICIES",
        {
            "gguf_q5_k": hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY,
            "gguf_q6_k": q6_policy,
        },
    )


def test_h6e_source_default_changes_only_qualified_q6_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)

    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _H6E_POLICY
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY == _H6E_POLICY
    assert hip_gfx1100.GGUF_F32_ORDERED_PREFILL_POLICIES == {
        "gguf_q5_k": hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY,
        "gguf_q6_k": _H6E_POLICY,
    }
    assert hip_gfx1151.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == {}
    assert not hasattr(hip_gfx1151, "GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY")
    assert {
        role for role in _H6E_POLICY if _H6E_POLICY[role] != _H5W_POLICY[role]
    } == set(_H6E_ROLES)
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H6E_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(
                output_dtype,
                in_features,
                out_features,
                _H6E_POLICY[(output_dtype, in_features, out_features)],
            )

        assert _raw_k_f32_ordered_prefill_dispatch(
            _base("f32"),
            rows=512,
            in_features=3072,
            out_features=72,
        ) == _selected("f32", 3072, 72, "coltile8_rowbatch4")

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

    _install_policy(monkeypatch, _H5W_POLICY)
    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H6E_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(
                output_dtype,
                in_features,
                out_features,
                _H5W_POLICY[(output_dtype, in_features, out_features)],
            )

    _install_policy(monkeypatch, _H6E_POLICY)
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

    missing_key = _selected(
        "bf16",
        3072,
        1024,
        _H6E_POLICY[("bf16", 3072, 1024)],
    ).key
    original_is_registered = gguf_linear_module.is_registered
    monkeypatch.setattr(
        gguf_linear_module,
        "is_registered",
        lambda key: key != missing_key and original_is_registered(key),
    )
    with q5_f32_ordered_prefill_session(_session()):
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
    with q5_f32_ordered_prefill_session(_session()):
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                gfx1151,
                rows=512,
                in_features=3072,
                out_features=1024,
            )
            is gfx1151
        )
