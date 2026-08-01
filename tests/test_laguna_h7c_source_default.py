"""WPF-H7C source-default promotion contract."""

from __future__ import annotations

import hashlib
import inspect

import pytest

from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _raw_k_prefill_rowbatch_dispatch,
)

_CAPABILITY = "GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS"
_GENERIC_CAPABILITY = "GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS"
_SOURCE_CAPABILITY = "GGUF_RAW_K_PREFILL_ROLE_VARIANTS"
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
_RUNTIME_DISPATCH_SHA256 = (
    "c12ea5317a53973204166ffe5589d44674e4f9d8df3b9f9964fb5656bce4df92"
)


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
    return KernelKey("hip_gfx1100", "linear", role[0], variant)


def _assert_policy_routes(policy: dict, *, candidate: bool) -> None:
    selected_variants = _H7C_POLICY if candidate else _GENERIC_VARIANTS
    original = getattr(hip_gfx1100, _SOURCE_CAPABILITY)
    setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(policy))
    try:
        for role in _ROLES:
            quant, output_variant, rows, in_features, out_features = role
            selected = _selected(
                _base(quant, output_variant),
                rows=rows,
                in_features=in_features,
                out_features=out_features,
            )
            assert selected == GGUFLinearDispatch(
                _key(role, selected_variants[role]),
                "raw",
            )
    finally:
        setattr(hip_gfx1100, _SOURCE_CAPABILITY, original)


def test_h7c_source_default_changes_only_the_exact_m512_role_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_backend_kernel_package("hip_gfx1100")
    load_backend_kernel_package("hip_gfx1151")

    generic = dict(getattr(hip_gfx1100, _GENERIC_CAPABILITY))
    capability = dict(getattr(hip_gfx1100, _CAPABILITY))
    live_source = dict(getattr(hip_gfx1100, _SOURCE_CAPABILITY))
    assert generic == _GENERIC_POLICY
    assert capability == _H7C_POLICY
    assert set(capability) == set(_ROLES)
    assert hashlib.sha256(
        inspect.getsource(_raw_k_prefill_rowbatch_dispatch).encode()
    ).hexdigest() == _RUNTIME_DISPATCH_SHA256
    assert not hasattr(hip_gfx1151, _GENERIC_CAPABILITY)
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    for role in _ROLES:
        candidate_key = _key(role, capability[role])
        assert is_registered(candidate_key)
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                candidate_key.layer,
                candidate_key.quant,
                candidate_key.variant,
            )
        )

    # Exercise both complete source topologies before checking the selected map.
    _assert_policy_routes(generic, candidate=False)
    _assert_policy_routes(capability, candidate=True)
    _assert_policy_routes(generic, candidate=False)

    # Candidate ownership stays bounded to exact M512 Q6 roles.
    with monkeypatch.context() as scoped:
        scoped.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(capability))
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

    # Intentional RED after candidate/rollback/backend/shape/registry controls:
    # source promotion changes only the live map from empty to H7C.
    assert (generic, live_source) == (_GENERIC_POLICY, _H7C_POLICY)
