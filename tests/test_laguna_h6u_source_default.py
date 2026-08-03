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
_H6U_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
}
_H6U_CALL_WEIGHTS = {
    ("bf16", 3072, 1024): 2,
    ("bf16", 1024, 3072): 46,
    ("f32", 3072, 1024): 94,
}
_H6U_ROLES = tuple(_H6U_CALL_WEIGHTS)


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
    policy: dict[tuple[str, int, int], str],
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    geometry = policy[(output_dtype, in_features, out_features)]
    return GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
        ),
        (
            "raw_k_f32_ordered_activation_tile_k_row"
            if "_activation_tile_k_row_" in geometry
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


def test_h6u_source_default_promotes_only_qualified_q6_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)

    live_source = hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY
    assert live_source == _H6U_POLICY
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY == _H6E_POLICY
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_H6U_POLICY == _H6U_POLICY
    assert hip_gfx1100.GGUF_F32_ORDERED_PREFILL_POLICIES == {
        "gguf_q5_k": hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY,
        "gguf_q6_k": _H6U_POLICY,
    }
    assert hip_gfx1151.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == {}
    assert not hasattr(hip_gfx1151, "GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY")
    assert not hasattr(hip_gfx1151, "GGUF_Q6_F32_ORDERED_PREFILL_H6U_POLICY")
    assert set(_H6U_POLICY) == set(_H6E_POLICY)
    assert {
        role for role in _H6U_POLICY if _H6U_POLICY[role] != _H6E_POLICY[role]
    } == set(_H6U_ROLES)
    assert sum(_H6U_CALL_WEIGHTS.values()) == 142
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert (
        LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(max_rows=512)
        == 10_125_312
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H6U_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(
                _H6U_POLICY,
                output_dtype,
                in_features,
                out_features,
            )

        assert _raw_k_f32_ordered_prefill_dispatch(
            _base("f32"),
            rows=512,
            in_features=3072,
            out_features=72,
        ) == _selected(_H6U_POLICY, "f32", 3072, 72)

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

    _install_policy(monkeypatch, _H6E_POLICY)
    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H6U_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(
                _H6E_POLICY,
                output_dtype,
                in_features,
                out_features,
            )

    _install_policy(monkeypatch, _H6U_POLICY)
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

    missing_key = _selected(_H6U_POLICY, "bf16", 3072, 1024).key
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

    # Exactly three source-map values change; H6E remains explicit rollback.
    assert live_source == _H6U_POLICY
