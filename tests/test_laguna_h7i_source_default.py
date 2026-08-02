"""WPF-H7I exact raw-Q6 source-default publication RED contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _raw_k_prefill_rowbatch_dispatch,
)
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_laguna_h7i_runtime_policy import (
    _CAPABILITY,
    _GENERIC_CAPABILITY,
    _GENERIC_POLICY,
    _GENERIC_VARIANTS,
    _H7C_CAPABILITY,
    _H7C_POLICY,
    _H7I_POLICY,
    _ROLES,
    _SOURCE_CAPABILITY,
    _base,
    _key,
    _selected,
)

_ARTIFACT = Path(
    "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json"
)
_ARTIFACT_SHA256 = (
    "640fe7c58d71a775ebe98a0d7934d3ad46ecae56982cb324bd2e120f49154eea"
)
_RUNTIME_DISPATCH_SHA256 = (
    "c12ea5317a53973204166ffe5589d44674e4f9d8df3b9f9964fb5656bce4df92"
)


def _assert_policy_routes(policy: dict, selected_variants: dict) -> None:
    original = getattr(hip_gfx1100, _SOURCE_CAPABILITY)
    setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(policy))
    try:
        for role in _ROLES:
            quant, output_variant, rows, in_features, out_features = role
            assert _selected(
                _base(quant, output_variant),
                rows=rows,
                in_features=in_features,
                out_features=out_features,
            ) == GGUFLinearDispatch(
                _key(role, selected_variants[role]),
                "raw",
            )
    finally:
        setattr(hip_gfx1100, _SOURCE_CAPABILITY, original)


def test_h7i_source_default_promotes_only_three_roles_and_keeps_h7c_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_backend_kernel_package("hip_gfx1100")
    load_backend_kernel_package("hip_gfx1151")

    artifact_bytes = _ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "qualified_bounded_default_off_runtime_owner"
    assert artifact["decision"]["runtime_qualified"]
    assert artifact["decision"]["runtime_capability_admitted"]
    assert not artifact["decision"]["source_promoted"]
    assert not artifact["decision"]["production_changed"]
    assert artifact["runtime_scope"] == {
        "changed_roles": 3,
        "runtime_capability": True,
        "runtime_owner": "bounded_default_off",
        "source_owner": False,
        "source_policy": "H7C",
        "source_promotion_pending": True,
        "total_scratch_bytes": 600_141_856,
        "workspace_bytes": 161_120_256,
    }
    assert artifact["runtime_qualification"]["acceptance"]["passed"]
    assert artifact["runtime_qualification"]["integrated_trace"][
        "h7i_dispatches"
    ] == 3

    generic = dict(getattr(hip_gfx1100, _GENERIC_CAPABILITY))
    h7c = dict(getattr(hip_gfx1100, _H7C_CAPABILITY))
    h7i = dict(getattr(hip_gfx1100, _CAPABILITY))
    live_source = dict(getattr(hip_gfx1100, _SOURCE_CAPABILITY))
    assert generic == _GENERIC_POLICY
    assert h7c == _H7C_POLICY
    assert h7i == _H7I_POLICY
    assert live_source == _H7I_POLICY
    assert set(h7c) == set(h7i) == set(_ROLES)
    assert all("full_group_compute" not in value for value in h7c.values())
    assert all("full_group_compute" in value for value in h7i.values())
    assert hashlib.sha256(
        inspect.getsource(_raw_k_prefill_rowbatch_dispatch).encode()
    ).hexdigest() == _RUNTIME_DISPATCH_SHA256
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    assert not hasattr(hip_gfx1151, _GENERIC_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H7C_CAPABILITY)
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)
    for role in _ROLES:
        key = _key(role, h7i[role])
        assert is_registered(key)
        assert not is_registered(
            KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
        )

    # Exercise the complete candidate, source rollback, and generic fallback.
    _assert_policy_routes(h7i, _H7I_POLICY)
    _assert_policy_routes(h7c, _H7C_POLICY)
    _assert_policy_routes(generic, _GENERIC_VARIANTS)
    _assert_policy_routes(h7i, _H7I_POLICY)

    # Every unqualified axis remains outside H7I source ownership.
    with monkeypatch.context() as scoped:
        scoped.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(h7i))
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

    missing = _key(_ROLES[0], h7i[_ROLES[0]])
    original_is_registered = gguf_linear_module.is_registered
    with monkeypatch.context() as scoped:
        scoped.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(h7i))
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

    # Intentional RED after artifact/capability/rollback/fallback controls:
    # source publication atomically copies H7I into the live map.
    assert (live_source, h7c, h7i) == (_H7I_POLICY, _H7C_POLICY, _H7I_POLICY)
