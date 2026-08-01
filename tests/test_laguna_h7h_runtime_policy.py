"""WPF-H7H bounded exact full-group Q5 runtime ownership contract."""

from __future__ import annotations

import json
from pathlib import Path
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

_CAPABILITY = "GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY"
_H5Y_CAPABILITY = "GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY"
_H7G_CAPABILITY = "GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY"
_ARTIFACT = Path(
    "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-candidate.json"
)
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
_H7H_POLICY = {
    **_H7G_POLICY,
    ("bf16", 3_072, 1_024): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "full_group_compute_coltile8_rowbatch4"
    ),
    ("bf16", 9_216, 3_072): (
        "weight_major_row_major_activation_tile_k_row_"
        "full_group_compute_coltile12_rowbatch8"
    ),
}
_H7G_ROLES = (
    ("bf16", 3_072, 12_288),
    ("bf16", 6_144, 3_072),
    ("f32", 3_072, 6_144),
    ("f32", 3_072, 9_216),
)
_H7H_CALL_WEIGHTS = {
    ("bf16", 3_072, 1_024): 92,
    ("bf16", 9_216, 3_072): 35,
}
_H7H_ROLES = tuple(_H7H_CALL_WEIGHTS)
_UNCHANGED_ROLES = tuple(role for role in _H7G_POLICY if role not in _H7H_ROLES)
_ROW_BATCHES = {
    ("bf16", 3_072, 1_024): 4,
    ("bf16", 9_216, 3_072): 8,
}


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


def _policy_key(
    policy: dict[tuple[str, int, int], str],
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> KernelKey:
    geometry = policy[(output_dtype, in_features, out_features)]
    return KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k",
        f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
    )


def _selected(
    policy: dict[tuple[str, int, int], str],
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    geometry = policy[(output_dtype, in_features, out_features)]
    return GGUFLinearDispatch(
        _policy_key(policy, output_dtype, in_features, out_features),
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


def _install_q5_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[tuple[str, int, int], str],
) -> None:
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_F32_ORDERED_PREFILL_POLICIES",
        {
            "gguf_q5_k": policy,
            "gguf_q6_k": hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY,
        },
    )


def test_h7h_runtime_capability_is_default_off_exact_and_workspace_neutral() -> None:
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == _H7G_POLICY
    assert getattr(hip_gfx1100, _H5Y_CAPABILITY) == _H5Y_POLICY
    assert getattr(hip_gfx1100, _H7G_CAPABILITY) == _H7G_POLICY
    assert not hasattr(hip_gfx1151, _H5Y_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H7G_CAPABILITY)
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert set(_H7H_POLICY) == set(_H7G_POLICY) == set(_H5Y_POLICY)
    assert {
        role for role in _H7H_POLICY if _H7H_POLICY[role] != _H7G_POLICY[role]
    } == set(_H7H_ROLES)
    assert {
        role for role in _H7G_POLICY if _H7G_POLICY[role] != _H5Y_POLICY[role]
    } == set(_H7G_ROLES)
    assert sum(_H7H_CALL_WEIGHTS.values()) == 127
    assert all(512 % _ROW_BATCHES[role] == 0 for role in _H7H_ROLES)
    assert all(
        "full_group_compute" in _H7H_POLICY[role] for role in _H7H_ROLES
    )
    assert all(_H7H_POLICY[role] == _H7G_POLICY[role] for role in _UNCHANGED_ROLES)

    artifact = json.loads(_ARTIFACT.read_text())
    assert artifact["status"] == "admitted_standalone_exact_leaf"
    assert artifact["decision"]["leaf_admitted"]
    assert not artifact["decision"]["runtime_capability_admitted"]
    assert not artifact["decision"]["source_owner_admitted"]
    assert artifact["scope"]["workspace_bytes"] == 161_120_256
    assert [role["production_invocations"] for role in artifact["scope"]["owned_roles"]] == [92, 35]

    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    for output_dtype, in_features, out_features in _H7H_ROLES:
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q5_k",
            variant=_policy_key(
                _H7H_POLICY,
                output_dtype,
                in_features,
                out_features,
            ).variant,
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

    # Intentional RED only after source, leaf, role, and workspace controls.
    assert getattr(hip_gfx1100, _CAPABILITY) == _H7H_POLICY


def test_h7h_runtime_dispatch_owns_only_exact_m512_roles_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    session = _session()

    _install_q5_policy(monkeypatch, _H7G_POLICY)
    with q5_f32_ordered_prefill_session(session):
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
    with q5_f32_ordered_prefill_session(session):
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
    with q5_f32_ordered_prefill_session(session):
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
    with q5_f32_ordered_prefill_session(session):
        assert (
            _raw_k_f32_ordered_prefill_dispatch(
                gfx1151,
                rows=512,
                in_features=3_072,
                out_features=1_024,
            )
            is gfx1151
        )


def test_h7h_runtime_launch_reuses_owner_weight_activation_and_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_q5_policy(monkeypatch, _H7H_POLICY)
    register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    session = _session()
    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k"),
        allocation=lambda name: raw,
    )

    for role_index, (output_dtype, in_features, out_features) in enumerate(
        _H7H_ROLES
    ):
        key = _policy_key(
            _H7H_POLICY,
            output_dtype,
            in_features,
            out_features,
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
