"""WPF-H7I bounded raw-Q6 runtime ownership RED contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _raw_k_prefill_rowbatch_dispatch,
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear,
    raw_k_prefill_rowbatch_session,
    raw_k_prefill_variant_session,
)
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch

_ROOT = Path(__file__).parents[1]
_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json"
)
_ARTIFACT_SHA256 = (
    "5fd6e44d2b7c38b19da32cb6c05f0d1522fe4c4c7cf2f84be5dafd9f0e09f8a8"
)
_CAPABILITY = "GGUF_RAW_K_PREFILL_H7I_ROLE_VARIANTS"
_H7C_CAPABILITY = "GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS"
_GENERIC_CAPABILITY = "GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS"
_SOURCE_CAPABILITY = "GGUF_RAW_K_PREFILL_ROLE_VARIANTS"
# quant, output ABI, exact rows, exact K, exact N
_ROLES = (
    ("gguf_q6_k", "bf16_bf16_out", 512, 12_288, 3_072),
    ("gguf_q6_k", "bf16_f32_out", 512, 3_072, 9_216),
    ("gguf_q6_k", "bf16_bf16_out", 512, 9_216, 3_072),
)
_GENERIC_POLICY: dict[tuple[str, str, int, int, int], str] = {}
_H7C_POLICY = {
    _ROLES[0]: "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
    _ROLES[1]: "dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out",
    _ROLES[2]: "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
}
_H7I_POLICY = {
    _ROLES[0]: (
        "dpp_wave_reduction_full_group_compute_"
        "coltile4_rowbatch8_bf16_bf16_out"
    ),
    _ROLES[1]: (
        "dpp_wave_reduction_full_group_compute_"
        "coltile2_rowbatch16_bf16_f32_out"
    ),
    _ROLES[2]: (
        "dpp_wave_reduction_full_group_compute_"
        "coltile4_rowbatch8_bf16_bf16_out"
    ),
}
_GENERIC_VARIANTS = {
    _ROLES[0]: "coltile4_rowbatch8_bf16_bf16_out",
    _ROLES[1]: "coltile2_rowbatch16_bf16_f32_out",
    _ROLES[2]: "coltile4_rowbatch8_bf16_bf16_out",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(
    quant: str,
    output_variant: str,
    *,
    backend: str = "hip_gfx1100",
) -> GGUFLinearDispatch:
    return GGUFLinearDispatch(
        KernelKey(
            backend,
            "linear",
            quant,
            f"prefill_{output_variant}",
        ),
        "raw",
    )


def _selected(
    base: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    return _raw_k_prefill_rowbatch_dispatch(
        base,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        row_batch=32,
        variant="coltile",
    )


def _key(role: tuple[str, str, int, int, int], variant: str) -> KernelKey:
    quant, _, _, _, _ = role
    return KernelKey("hip_gfx1100", "linear", quant, variant)


def test_h7i_runtime_contract_binds_standalone_artifact_and_exact_role_set() -> None:
    artifact_bytes = _ARTIFACT.read_bytes()
    assert _sha256(_ARTIFACT) == _ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "admitted_standalone_exact_h7i_leaf"
    assert artifact["decision"]["h7i_standalone_admitted"]
    assert not artifact["decision"]["production_changed"]
    assert not artifact["decision"]["runtime_qualified"]
    assert not artifact["decision"]["source_promoted"]
    assert artifact["correctness"]["green"] == {
        "failed": 0,
        "nodes": 22,
        "passed": 22,
    }
    assert artifact["repository_replay"]["all_roles_both_clock_positive"]
    assert artifact["repository_replay"]["all_candidate_bytes_exact"]
    assert artifact["named_trace"]["trace_gate_passed"]
    assert len(_ROLES) == len(set(_ROLES)) == 3
    assert {role[0] for role in _ROLES} == {"gguf_q6_k"}
    assert {role[2] for role in _ROLES} == {512}
    assert sum(role[1] == "bf16_bf16_out" for role in _ROLES) == 2
    assert sum(role[1] == "bf16_f32_out" for role in _ROLES) == 1
    assert set(_H7C_POLICY) == set(_H7I_POLICY) == set(_ROLES)
    assert all("full_group_compute" in value for value in _H7I_POLICY.values())
    assert all("full_group_compute" not in value for value in _H7C_POLICY.values())
    assert _GENERIC_POLICY == {}
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256


def test_h7i_runtime_capability_is_default_off_and_gfx1100_only() -> None:
    load_backend_kernel_package("hip_gfx1100")
    load_backend_kernel_package("hip_gfx1151")

    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED is True
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROWBATCH == 32
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED is True
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_VARIANT == "coltile"
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED is False
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED is False
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_VARIANT == "rowbatch"
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _H7C_CAPABILITY)
    assert not hasattr(hip_gfx1151, _GENERIC_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    assert getattr(hip_gfx1100, _GENERIC_CAPABILITY) == _GENERIC_POLICY
    assert getattr(hip_gfx1100, _H7C_CAPABILITY) == _H7C_POLICY
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == _H7C_POLICY
    for role in _ROLES:
        h7c_key = _key(role, _H7C_POLICY[role])
        h7i_key = _key(role, _H7I_POLICY[role])
        assert is_registered(h7c_key)
        assert is_registered(h7i_key)
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                h7i_key.layer,
                h7i_key.quant,
                h7i_key.variant,
            )
        )

    # Intentional RED only after artifact, H7C source, registration, workspace,
    # and gfx1151 controls prove the standalone leaf is ready for ownership.
    assert getattr(hip_gfx1100, _CAPABILITY) == _H7I_POLICY


def test_h7i_in_memory_runtime_selection_rollback_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_backend_kernel_package("hip_gfx1100")

    # Complete named H7C rollback remains the source and is exercised first.
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(_H7C_POLICY))
    for role in _ROLES:
        quant, output_variant, rows, in_features, out_features = role
        assert _selected(
            _base(quant, output_variant),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        ) == GGUFLinearDispatch(_key(role, _H7C_POLICY[role]), "raw")

    # In-memory bounded candidate selection changes only the exact three roles.
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(_H7I_POLICY))
    for role in _ROLES:
        quant, output_variant, rows, in_features, out_features = role
        assert _selected(
            _base(quant, output_variant),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        ) == GGUFLinearDispatch(_key(role, _H7I_POLICY[role]), "raw")

    # Every unqualified axis retains the preceding generic route.
    for rows, quant, output_variant, in_features, out_features in (
        (511, "gguf_q6_k", "bf16_bf16_out", 12_288, 3_072),
        (513, "gguf_q6_k", "bf16_bf16_out", 12_288, 3_072),
        (512, "gguf_q5_k", "bf16_bf16_out", 12_288, 3_072),
        (512, "gguf_q6_k", "bf16_bf16_out", 12_032, 3_072),
        (512, "gguf_q6_k", "bf16_bf16_out", 12_288, 3_068),
    ):
        selected = _selected(
            _base(quant, output_variant),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        assert "full_group_compute" not in selected.key.variant
        assert selected.abi == "raw"

    gfx1151 = _base("gguf_q6_k", "bf16_bf16_out", backend="hip_gfx1151")
    assert (
        _selected(
            gfx1151,
            rows=512,
            in_features=12_288,
            out_features=3_072,
        )
        is gfx1151
    )

    missing = _key(_ROLES[0], _H7I_POLICY[_ROLES[0]])
    original_is_registered = gguf_linear_module.is_registered
    with monkeypatch.context() as scoped:
        scoped.setattr(
            gguf_linear_module,
            "is_registered",
            lambda key: key != missing and original_is_registered(key),
        )
        selected = _selected(
            _base("gguf_q6_k", "bf16_bf16_out"),
            rows=512,
            in_features=12_288,
            out_features=3_072,
        )
        assert selected.key.variant == _GENERIC_VARIANTS[_ROLES[0]]

    with monkeypatch.context() as scoped:
        scoped.setattr(hip_gfx1100, _SOURCE_CAPABILITY, {role: 7 for role in _ROLES})
        for role in _ROLES:
            quant, output_variant, rows, in_features, out_features = role
            selected = _selected(
                _base(quant, output_variant),
                rows=rows,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected.key.variant == _GENERIC_VARIANTS[role]


def test_h7i_in_memory_runtime_launch_preserves_raw_pointer_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_backend_kernel_package("hip_gfx1100")
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(_H7I_POLICY))
    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k"),
        allocation=lambda name: raw,
    )

    for role_index, role in enumerate(_ROLES):
        _, output_variant, rows, in_features, out_features = role
        output_dtype = "f32" if output_variant.endswith("f32_out") else "bf16"
        key = _key(role, _H7I_POLICY[role])
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
            with (
                raw_k_prefill_rowbatch_session(32),
                raw_k_prefill_variant_session("coltile"),
            ):
                launch_gguf_linear(
                    weight,
                    x_ptr=0x1000 + role_index,
                    out_ptr=0x3000 + role_index,
                    rows=rows,
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
                    rows,
                    in_features,
                    out_features,
                ),
                {"stream": 7, "runtime": "runtime-sentinel"},
            )
        ]
