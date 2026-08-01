"""WPF-H7C bounded raw-Q6 runtime ownership contract."""

from __future__ import annotations

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

_CAPABILITY = "GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS"
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
_GENERIC_VARIANTS = {
    _ROLES[0]: "coltile4_rowbatch8_bf16_bf16_out",
    _ROLES[1]: "coltile2_rowbatch16_bf16_f32_out",
    _ROLES[2]: "coltile4_rowbatch8_bf16_bf16_out",
}


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


def test_h7c_runtime_contract_is_exact_m512_shape_policy() -> None:
    assert len(_ROLES) == 3
    assert len(set(_ROLES)) == 3
    assert {role[0] for role in _ROLES} == {"gguf_q6_k"}
    assert {role[2] for role in _ROLES} == {512}
    assert sum(role[1] == "bf16_bf16_out" for role in _ROLES) == 2
    assert sum(role[1] == "bf16_f32_out" for role in _ROLES) == 1
    assert _GENERIC_POLICY == {}
    assert set(_H7C_POLICY) == set(_ROLES)
    assert set(_GENERIC_VARIANTS) == set(_ROLES)
    assert {
        variant.split("dpp_wave_reduction_", 1)[1]
        for variant in _H7C_POLICY.values()
    } == set(_GENERIC_VARIANTS.values())


def test_h7c_runtime_package_capability_is_default_off_and_backend_local() -> None:
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
    assert not hasattr(hip_gfx1151, _GENERIC_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    for role in _ROLES:
        candidate_key = _key(role, _H7C_POLICY[role])
        assert is_registered(candidate_key)
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                candidate_key.layer,
                candidate_key.quant,
                candidate_key.variant,
            )
        )

    # Intentional RED after leaf/backend controls: runtime qualification adds
    # only package-owned candidate/source/rollback policy metadata.
    assert getattr(hip_gfx1100, _CAPABILITY) == _H7C_POLICY
    assert getattr(hip_gfx1100, _GENERIC_CAPABILITY) == _GENERIC_POLICY
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == _GENERIC_POLICY


def test_h7c_runtime_dispatch_is_exact_role_owner_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_backend_kernel_package("hip_gfx1100")

    # Current source remains the exact generic coltile owner for all three
    # admitted leaf roles before the default-off capability is selected.
    for role in _ROLES:
        quant, output_variant, rows, in_features, out_features = role
        selected = _selected(
            _base(quant, output_variant),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        assert selected == GGUFLinearDispatch(
            _key(role, _GENERIC_VARIANTS[role]),
            "raw",
        )

    # Intentional RED only after retained source routing passes.
    capability = getattr(hip_gfx1100, _CAPABILITY)
    assert capability == _H7C_POLICY
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(capability))

    for role in _ROLES:
        quant, output_variant, rows, in_features, out_features = role
        assert _selected(
            _base(quant, output_variant),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        ) == GGUFLinearDispatch(_key(role, _H7C_POLICY[role]), "raw")

    # Every unqualified axis retains the preceding generic route.
    for role, rows, quant, output_variant, in_features, out_features in (
        (_ROLES[0], 511, "gguf_q6_k", "bf16_bf16_out", 12_288, 3_072),
        (_ROLES[0], 513, "gguf_q6_k", "bf16_bf16_out", 12_288, 3_072),
        (_ROLES[0], 512, "gguf_q5_k", "bf16_bf16_out", 12_288, 3_072),
        (_ROLES[0], 512, "gguf_q6_k", "bf16_bf16_out", 12_032, 3_072),
        (_ROLES[0], 512, "gguf_q6_k", "bf16_bf16_out", 12_288, 3_068),
    ):
        selected = _selected(
            _base(quant, output_variant),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        assert "dpp_wave_reduction" not in selected.key.variant
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

    missing = _key(_ROLES[0], _H7C_POLICY[_ROLES[0]])
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


def test_h7c_runtime_launch_preserves_raw_pointer_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_backend_kernel_package("hip_gfx1100")

    # Intentional RED after the registered leaf exists: the package capability
    # is the only new ownership surface needed by this launch path.
    capability = getattr(hip_gfx1100, _CAPABILITY)
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(capability))
    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k"),
        allocation=lambda name: raw,
    )

    for role_index, role in enumerate(_ROLES):
        quant, output_variant, rows, in_features, out_features = role
        del quant
        output_dtype = "f32" if output_variant.endswith("f32_out") else "bf16"
        key = _key(role, _H7C_POLICY[role])
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
