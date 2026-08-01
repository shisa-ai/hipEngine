"""WPF-H7H exact-Q5 source-default publication contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_f32_rocblas_prefill import (
    register_gguf_q5_k_f32_rocblas_prefill_kernels,
)
from hipengine.runtime.gguf_linear import (
    _raw_k_f32_ordered_prefill_dispatch,
    q5_f32_ordered_prefill_session,
)
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_laguna_h7h_runtime_policy import (
    _H5Y_CAPABILITY,
    _H5Y_POLICY,
    _H7G_CAPABILITY,
    _H7G_POLICY,
    _H7G_ROLES,
    _H7H_CALL_WEIGHTS,
    _H7H_POLICY,
    _H7H_ROLES,
    _UNCHANGED_ROLES,
    _base,
    _install_q5_policy,
    _policy_key,
    _selected,
    _session,
)

_H7H_CAPABILITY = "GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY"
_ARTIFACT = Path(
    "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-candidate.json"
)


def test_h7h_source_default_promotes_only_two_roles_and_keeps_h7g_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)

    live_source = hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY
    live_policies = hip_gfx1100.GGUF_F32_ORDERED_PREFILL_POLICIES
    assert live_source == _H7H_POLICY
    assert live_policies == {
        "gguf_q5_k": _H7H_POLICY,
        "gguf_q6_k": hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY,
    }
    assert getattr(hip_gfx1100, _H5Y_CAPABILITY) == _H5Y_POLICY
    assert getattr(hip_gfx1100, _H7G_CAPABILITY) == _H7G_POLICY
    assert getattr(hip_gfx1100, _H7H_CAPABILITY) == _H7H_POLICY
    assert not hasattr(hip_gfx1151, "GGUF_Q5_F32_ORDERED_PREFILL_POLICY")
    assert not hasattr(hip_gfx1151, _H5Y_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H7G_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H7H_CAPABILITY)
    assert set(_H7H_POLICY) == set(_H7G_POLICY) == set(_H5Y_POLICY)
    assert {
        role for role in _H7H_POLICY if _H7H_POLICY[role] != _H7G_POLICY[role]
    } == set(_H7H_ROLES)
    assert {
        role for role in _H7G_POLICY if _H7G_POLICY[role] != _H5Y_POLICY[role]
    } == set(_H7G_ROLES)
    assert sum(_H7H_CALL_WEIGHTS.values()) == 127
    assert all(
        _H7H_POLICY[role] == _H7G_POLICY[role] for role in _UNCHANGED_ROLES
    )
    assert all(
        "full_group_compute" in _H7H_POLICY[role] for role in _H7H_ROLES
    )

    artifact = json.loads(_ARTIFACT.read_text())
    assert artifact["status"] == "qualified_bounded_default_off_runtime_owner"
    assert artifact["decision"]["runtime_capability_admitted"]
    assert not artifact["decision"]["source_owner_admitted"]
    assert artifact["scope"]["source_policy"] == "H7G"
    assert artifact["scope"]["source_promotion_pending"]
    assert artifact["scope"]["workspace_bytes"] == 161_120_256

    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert (
        LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(max_rows=512)
        == 10_125_312
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    _install_q5_policy(monkeypatch, _H7H_POLICY)
    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H7H_ROLES:
            selected = _raw_k_f32_ordered_prefill_dispatch(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected == _selected(
                _H7H_POLICY,
                output_dtype,
                in_features,
                out_features,
            )
            assert "full_group_compute" in selected.key.variant

        for output_dtype, in_features, out_features in _UNCHANGED_ROLES:
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

        for rows, output_dtype, in_features, out_features in (
            (511, "bf16", 3_072, 1_024),
            (513, "bf16", 3_072, 1_024),
            (511, "bf16", 9_216, 3_072),
            (513, "bf16", 9_216, 3_072),
            (512, "bf16", 3_328, 1_024),
            (512, "bf16", 3_072, 1_280),
            (512, "bf16", 8_960, 3_072),
            (512, "bf16", 9_216, 2_816),
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

    _install_q5_policy(monkeypatch, _H7G_POLICY)
    with q5_f32_ordered_prefill_session(_session()):
        for output_dtype, in_features, out_features in _H7H_ROLES:
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

    _install_q5_policy(monkeypatch, _H7H_POLICY)
    with q5_f32_ordered_prefill_session(_session(with_activation=False)):
        base = _base("bf16")
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                base,
                rows=512,
                in_features=3_072,
                out_features=1_024,
            )
            is base
        )

    missing_key = _policy_key(_H7H_POLICY, "bf16", 3_072, 1_024)
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
                out_features=1_024,
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
                out_features=1_024,
            )
            is gfx1151
        )

    # Source publication atomically selects H7H and keeps complete H7G rollback.
    assert (
        live_source,
        getattr(hip_gfx1100, _H7G_CAPABILITY),
        live_policies["gguf_q5_k"],
    ) == (_H7H_POLICY, _H7G_POLICY, _H7H_POLICY)
