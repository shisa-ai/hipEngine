"""WPF-H7G exact-Q5 source-default publication contract."""

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

_H5Y_CAPABILITY = "GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY"
_H7G_CAPABILITY = "GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY"
_H5Y_POLICY = {
    ("bf16", 3_072, 1_024): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch4"
    ),
    ("bf16", 3_072, 12_288): (
        "weight_major_row_major_activation_tile_k_row_coltile8_rowbatch12"
    ),
    ("bf16", 6_144, 3_072): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("bf16", 9_216, 3_072): (
        "weight_major_row_major_activation_tile_k_row_coltile12_rowbatch8"
    ),
    ("f32", 3_072, 48): "coltile12_rowbatch4",
    ("f32", 3_072, 72): "coltile8_rowbatch4",
    ("f32", 3_072, 6_144): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("f32", 3_072, 9_216): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch10"
    ),
}
_H7G_POLICY = {
    **_H5Y_POLICY,
    ("bf16", 3_072, 12_288): (
        "weight_major_row_major_activation_tile_k_row_"
        "padded_compute_coltile8_rowbatch12"
    ),
    ("bf16", 6_144, 3_072): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile16_rowbatch5"
    ),
    ("f32", 3_072, 6_144): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile16_rowbatch5"
    ),
    ("f32", 3_072, 9_216): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile8_rowbatch10"
    ),
}
_H7G_CALL_WEIGHTS = {
    ("bf16", 3_072, 12_288): 2,
    ("bf16", 6_144, 3_072): 12,
    ("f32", 3_072, 6_144): 12,
    ("f32", 3_072, 9_216): 35,
}
_H7G_ROLES = tuple(_H7G_CALL_WEIGHTS)
_EXACT_DIVISIBILITY_ROLES = (
    ("bf16", 3_072, 1_024),
    ("bf16", 9_216, 3_072),
    ("f32", 3_072, 48),
    ("f32", 3_072, 72),
)


def _base(output_dtype: str, *, backend: str = "hip_gfx1100") -> GGUFLinearDispatch:
    return GGUFLinearDispatch(
        KernelKey(
            backend,
            "linear",
            "gguf_q5_k",
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
            "gguf_q5_k",
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
    q5_policy: dict[tuple[str, int, int], str],
) -> None:
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_F32_ORDERED_PREFILL_POLICIES",
        {
            "gguf_q5_k": q5_policy,
            "gguf_q6_k": hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY,
        },
    )


def test_h7g_source_default_promotes_only_qualified_q5_roles_and_keeps_h5y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)

    live_source = hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY
    live_policies = hip_gfx1100.GGUF_F32_ORDERED_PREFILL_POLICIES
    h7h_policy = hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY
    assert live_source == h7h_policy
    assert getattr(hip_gfx1100, _H5Y_CAPABILITY) == _H5Y_POLICY
    assert getattr(hip_gfx1100, _H7G_CAPABILITY) == _H7G_POLICY
    assert live_policies == {
        "gguf_q5_k": h7h_policy,
        "gguf_q6_k": hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY,
    }
    assert not hasattr(hip_gfx1151, "GGUF_Q5_F32_ORDERED_PREFILL_POLICY")
    assert not hasattr(hip_gfx1151, _H5Y_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H7G_CAPABILITY)
    assert set(_H7G_POLICY) == set(_H5Y_POLICY)
    assert {
        role for role in _H7G_POLICY if _H7G_POLICY[role] != _H5Y_POLICY[role]
    } == set(_H7G_ROLES)
    assert sum(_H7G_CALL_WEIGHTS.values()) == 61
    assert all(
        _H7G_POLICY[role] == _H5Y_POLICY[role]
        and "padded_compute" not in _H7G_POLICY[role]
        for role in _EXACT_DIVISIBILITY_ROLES
    )
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert (
        LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(max_rows=512)
        == 10_125_312
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    _install_policy(monkeypatch, _H7G_POLICY)
    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H7G_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(
                _H7G_POLICY,
                output_dtype,
                in_features,
                out_features,
            )

        for output_dtype, in_features, out_features in _EXACT_DIVISIBILITY_ROLES:
            selected = _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected == _selected(
                _H5Y_POLICY,
                output_dtype,
                in_features,
                out_features,
            )
            assert "padded_compute" not in selected.key.variant

        for rows, output_dtype, in_features, out_features in (
            (511, "bf16", 3_072, 12_288),
            (513, "bf16", 3_072, 12_288),
            (512, "bf16", 3_072, 12_032),
            (512, "bf16", 6_400, 3_072),
            (512, "f32", 3_328, 6_144),
            (512, "f32", 3_072, 9_200),
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

    _install_policy(monkeypatch, _H5Y_POLICY)
    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H7G_ROLES:
            assert _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(
                _H5Y_POLICY,
                output_dtype,
                in_features,
                out_features,
            )

    _install_policy(monkeypatch, _H7G_POLICY)
    with q5_f32_ordered_prefill_session(_session(with_activation=False)):
        base = _base("bf16")
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                base,
                rows=512,
                in_features=3_072,
                out_features=12_288,
            )
            is base
        )

    missing_key = _selected(_H7G_POLICY, "bf16", 3_072, 12_288).key
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
                in_features=3_072,
                out_features=12_288,
            )
            is base
        )

    gfx1151 = _base("bf16", backend="hip_gfx1151")
    with q5_f32_ordered_prefill_session(_session()):
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                gfx1151,
                rows=512,
                in_features=3_072,
                out_features=12_288,
            )
            is gfx1151
        )

    # H7G and H5Y remain complete named rollbacks under the H7H source.
    assert (
        live_source,
        getattr(hip_gfx1100, _H5Y_CAPABILITY, None),
        getattr(hip_gfx1100, _H7G_CAPABILITY, None),
        live_policies["gguf_q5_k"],
    ) == (h7h_policy, _H5Y_POLICY, _H7G_POLICY, h7h_policy)
