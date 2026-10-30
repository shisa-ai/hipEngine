"""Move-only boundary contract for gfx1151 GGUF and capability policy data.

``policy_gguf`` owns the GGUF quant/layout/prefill/decode policy declarations
and ``policy_capability`` owns backend identity/capability data. The backend
package keeps explicit compatibility exports; registration, wrapper functions,
and kernel bodies stay in ``hipengine.kernels.hip_gfx1151``.

The package import is deferred to a fixture. Importing the backend package
before ``hipengine.runtime.gguf_linear`` exposes a pre-existing dispatch
ordering coupling; this module must not add another collector-order trigger.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.policy import (
    QWEN35_DENSE_H1024_GEOMETRY,
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)
from hipengine.kernels.registry import registered_keys, resolve

CAPABILITY_NAMES = {
    "BACKEND",
    "STAGED_F32_VARIANT",
    "TARGET_ARCH",
    "GGUF_CONSUMER_LAYERS",
}
GGUF_PRIVATE_NAMES = {
    "_IQ_DENSE_W4A16_VARIANT",
    "_IQ_DENSE_INTEGER_MMQ_VARIANT",
    "_GGUF_Q5_T16_ROWTILE_COL8_SHAPES",
}
GEOMETRY_NAMES = {
    "QWEN35_DENSE_H1024_GEOMETRY",
    "QWEN35_DENSE_H5120_GEOMETRY",
    "QWEN35_MOE_H2048_E256_GEOMETRY",
}
DATA_TYPES = (bool, int, str, tuple, frozenset, dict)


@pytest.fixture()
def policy():
    import hipengine.kernels.hip_gfx1151 as backend
    from hipengine.kernels.hip_gfx1151 import policy_capability, policy_gguf

    return backend, policy_capability, policy_gguf


def _implementation_identity(kernel: object) -> tuple[str, str]:
    return (
        getattr(kernel, "__module__", type(kernel).__module__),
        getattr(kernel, "__qualname__", type(kernel).__qualname__),
    )


def _registry_snapshot(backend_name: str) -> tuple[object, ...]:
    return tuple(
        (
            key,
            _implementation_identity(
                resolve(
                    backend=key.backend,
                    layer=key.layer,
                    quant=key.quant,
                    variant=key.variant,
                )
            ),
        )
        for key in sorted(registered_keys())
        if key.backend == backend_name
    )


def _initializer_assignments(backend: object) -> dict[str, ast.expr]:
    tree = ast.parse(Path(backend.__file__).read_text())
    assigned: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value
    return assigned


def test_policy_capability_owns_only_identity_and_capability_data(policy) -> None:
    _backend, policy_capability, _policy_gguf = policy
    declared = {
        name
        for name in vars(policy_capability)
        if name.isupper() and not name.startswith("_")
    }

    assert declared == set(policy_capability.__all__)
    assert CAPABILITY_NAMES <= declared
    assert all(
        name.startswith(("QWEN4_EXP_", "PARO_"))
        for name in declared - CAPABILITY_NAMES
    )
    for name in declared:
        value = getattr(policy_capability, name)
        assert not callable(value)
        assert isinstance(value, DATA_TYPES)

    # The policy module must not acquire a registry or import machinery.
    assert not hasattr(policy_capability, "register")
    assert not hasattr(policy_capability, "KernelKey")
    assert not hasattr(policy_capability, "import_module")


def test_policy_gguf_owns_only_gguf_policy_data(policy) -> None:
    _backend, _policy_capability, policy_gguf = policy
    public_gguf = {
        name
        for name in vars(policy_gguf)
        if name.startswith("GGUF_") and not name.startswith("_")
    }

    assert set(policy_gguf.__all__) == public_gguf
    assert "GGUF_CONSUMER_LAYERS" not in public_gguf
    assert GGUF_PRIVATE_NAMES <= set(vars(policy_gguf))
    for name in public_gguf | GGUF_PRIVATE_NAMES:
        value = getattr(policy_gguf, name)
        assert not callable(value)
        assert isinstance(value, DATA_TYPES)

    # Geometry identities are imported from the shared policy module, not
    # redefined as a second geometry type.
    assert {
        name for name in vars(policy_gguf) if name in GEOMETRY_NAMES
    } == GEOMETRY_NAMES
    assert policy_gguf.QWEN35_DENSE_H1024_GEOMETRY is QWEN35_DENSE_H1024_GEOMETRY
    assert policy_gguf.QWEN35_DENSE_H5120_GEOMETRY is QWEN35_DENSE_H5120_GEOMETRY
    assert policy_gguf.QWEN35_MOE_H2048_E256_GEOMETRY is QWEN35_MOE_H2048_E256_GEOMETRY
    assert any(
        geometry is QWEN35_MOE_H2048_E256_GEOMETRY
        for geometry, _quant in policy_gguf.GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY
    )

    # The policy module must not acquire a registry or import machinery.
    assert not hasattr(policy_gguf, "register")
    assert not hasattr(policy_gguf, "KernelKey")
    assert not hasattr(policy_gguf, "import_module")


def test_package_reexports_every_policy_declaration_explicitly(policy) -> None:
    backend, policy_capability, policy_gguf = policy
    for name in policy_capability.__all__:
        assert getattr(backend, name) is getattr(policy_capability, name)
    for name in set(policy_gguf.__all__) | GGUF_PRIVATE_NAMES:
        assert getattr(backend, name) is getattr(policy_gguf, name)

    from hipengine.kernels.hip_gfx1151 import (
        BACKEND,
        GGUF_IQ_DENSE_DECODE_POLICY,
        PARO_RETAINED_BATCH_DEFAULTS,
        _GGUF_Q5_T16_ROWTILE_COL8_SHAPES,
    )

    assert BACKEND is policy_capability.BACKEND
    assert PARO_RETAINED_BATCH_DEFAULTS is policy_capability.PARO_RETAINED_BATCH_DEFAULTS
    assert GGUF_IQ_DENSE_DECODE_POLICY is policy_gguf.GGUF_IQ_DENSE_DECODE_POLICY
    assert _GGUF_Q5_T16_ROWTILE_COL8_SHAPES is policy_gguf._GGUF_Q5_T16_ROWTILE_COL8_SHAPES

    # The historical star-import surface is unchanged: moved names already in
    # ``__all__`` still resolve, and the move added no new star-import names.
    listed = {name for name in backend.__all__ if name.startswith("GGUF_")}
    assert listed <= set(policy_capability.__all__) | set(policy_gguf.__all__)
    assert "BACKEND" in backend.__all__
    assert "TARGET_ARCH" in backend.__all__
    assert len(backend.__all__) == len(set(backend.__all__))


def test_package_capability_lookup_reads_the_package_aliases(policy, monkeypatch) -> None:
    backend, policy_capability, policy_gguf = policy
    assert (
        backend_package_capability("hip_gfx1151", "BACKEND")
        is policy_capability.BACKEND
    )
    assert (
        backend_package_capability("hip_gfx1151", "GGUF_CONSUMER_LAYERS")
        is policy_capability.GGUF_CONSUMER_LAYERS
    )
    assert (
        backend_package_capability("hip_gfx1151", "QWEN4_EXP_MOE_GRAPH")
        is policy_capability.QWEN4_EXP_MOE_GRAPH
    )
    assert (
        backend_package_capability("hip_gfx1151", "PARO_RETAINED_BATCH_DEFAULTS")
        is policy_capability.PARO_RETAINED_BATCH_DEFAULTS
    )

    # Reassignment through the package is the supported read path. The owning
    # module is not mutated, so a later replacement still restores the default.
    sentinel = {"gguf_iq4_xs": {"variant": "sentinel"}}
    monkeypatch.setattr(backend, "GGUF_IQ_DENSE_DECODE_POLICY", sentinel)
    assert (
        backend_package_capability("hip_gfx1151", "GGUF_IQ_DENSE_DECODE_POLICY")
        is sentinel
    )
    assert policy_gguf.GGUF_IQ_DENSE_DECODE_POLICY is not sentinel


def test_declared_policy_owner_list_is_one_literal_ordered_declaration(policy) -> None:
    backend, policy_capability, policy_gguf = policy
    from hipengine.kernels.hip_gfx1151 import policy_laguna

    # The package's module references are derived from this one literal list,
    # and the cold source reader reads the same declaration, so the runtime and
    # source views of ownership cannot drift apart.
    assert backend.POLICY_MODULE_NAMES == (
        "policy_capability",
        "policy_gguf",
        "policy_laguna",
    )
    assert backend._POLICY_MODULES == (
        policy_capability,
        policy_gguf,
        policy_laguna,
    )
    assert [module.__name__ for module in backend._POLICY_MODULES] == [
        f"{backend.__name__}.{name}" for name in backend.POLICY_MODULE_NAMES
    ]


def test_declared_policy_owners_are_collision_free(policy) -> None:
    backend, _policy_capability, _policy_gguf = policy

    owner: dict[str, object] = {}
    for module in backend._POLICY_MODULES:
        for name in vars(module):
            if not name.isupper():
                continue
            if name in owner:
                # The same declaration may be re-imported by a second owner;
                # only a divergent second value is an ownership collision.
                assert getattr(owner[name], name) is getattr(module, name), name
                continue
            owner[name] = module

    assert owner
    # Every declared name is still reachable through the package today, and it
    # is the owner's own object -- not a copy, wrapper, or second value.
    for name, module in owner.items():
        assert getattr(backend, name) is getattr(module, name), name


def test_every_owner_declaration_resolves_without_the_package_binding(
    policy, monkeypatch
) -> None:
    backend, _policy_capability, _policy_gguf = policy
    sentinel = object()
    checked = 0

    for module in backend._POLICY_MODULES:
        for name in vars(module):
            if not name.isupper():
                continue
            # Only this one package binding is absent; the owner fallback must
            # reproduce the exact owner value instead of the caller default.
            with monkeypatch.context() as scoped:
                scoped.delattr(backend, name, raising=False)
                assert (
                    backend_package_capability("hip_gfx1151", name, sentinel)
                    is getattr(module, name)
                ), name
            checked += 1

    assert checked


def test_package_binding_wins_even_when_falsy_or_none(policy, monkeypatch) -> None:
    backend, _policy_capability, policy_gguf = policy
    owner_value = policy_gguf.GGUF_IQ_DENSE_DECODE_POLICY
    assert owner_value

    for override in (False, None):
        with monkeypatch.context() as scoped:
            scoped.setattr(backend, "GGUF_IQ_DENSE_DECODE_POLICY", override)
            assert (
                backend_package_capability(
                    "hip_gfx1151", "GGUF_IQ_DENSE_DECODE_POLICY", "default"
                )
                is override
            )
            # The fallback read never writes back into the owner.
            assert policy_gguf.GGUF_IQ_DENSE_DECODE_POLICY is owner_value


def test_owner_value_mutation_is_visible_without_the_package_binding(
    policy, monkeypatch
) -> None:
    backend, _policy_capability, policy_gguf = policy
    capability_name = "GGUF_IQ_DENSE_DECODE_POLICY"
    replacement = {"gguf_iq4_xs": {"variant": "replacement"}}

    with monkeypatch.context() as scoped:
        scoped.delattr(backend, capability_name, raising=False)
        assert (
            backend_package_capability("hip_gfx1151", capability_name)
            is policy_gguf.GGUF_IQ_DENSE_DECODE_POLICY
        )
        scoped.setattr(policy_gguf, capability_name, replacement)
        # No cache: with the package binding absent, the very next read sees
        # the owner's current value, not a value captured on the first read.
        assert backend_package_capability("hip_gfx1151", capability_name) is replacement

    assert (
        backend_package_capability("hip_gfx1151", capability_name)
        is policy_gguf.GGUF_IQ_DENSE_DECODE_POLICY
    )
    assert backend_package_capability("hip_gfx1151", capability_name) is not replacement


def test_unknown_names_and_peer_backends_keep_caller_defaults(policy) -> None:
    import importlib

    _backend, _policy_capability, _policy_gguf = policy
    sentinel = object()
    gfx1100 = importlib.import_module("hipengine.kernels.hip_gfx1100")

    assert (
        backend_package_capability("hip_gfx1151", "NOT_A_DECLARED_NAME", sentinel)
        is sentinel
    )
    for peer in ("hip_gfx1100", "cuda_sm120a"):
        module = importlib.import_module(f"hipengine.kernels.{peer}")
        # No owner declaration is inherited across backend packages.
        assert not hasattr(module, "_POLICY_MODULES")
        assert (
            backend_package_capability(peer, "POLICY_MODULE_NAMES", sentinel)
            is sentinel
        )
    assert (
        backend_package_capability("hip_gfx1100", "LAGUNA_ACTIVATION_PACK_REUSE")
        is gfx1100.LAGUNA_ACTIVATION_PACK_REUSE
    )
    assert (
        backend_package_capability("hip_gfx1100", "NOT_A_DECLARED_NAME", sentinel)
        is sentinel
    )


def test_initializer_keeps_source_assignments_for_capability_ledger(policy) -> None:
    backend, policy_capability, policy_gguf = policy
    assigned = _initializer_assignments(backend)
    moved = set(policy_capability.__all__) | set(policy_gguf.__all__) | GGUF_PRIVATE_NAMES

    # The gfx1100/gfx1151 capability-ledger check unions the package's declared
    # policy owners, so it no longer depends on these aliases; the aliases stay
    # because the compatibility surface is frozen for RF-8.1.
    assert moved <= set(assigned)
    for name in moved:
        value = assigned[name]
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name)
        assert value.value.id in {"_policy_capability", "_policy_gguf"}
        assert value.attr == name


def test_registration_snapshot_is_stable_across_replacement(policy) -> None:
    backend, _policy_capability, _policy_gguf = policy
    # A pytest session may not have aliased every lazily imported peer module
    # yet; establish the complete alias state before comparing replacements.
    backend.register_gfx1151_kernels(replace=True)
    before = _registry_snapshot("hip_gfx1151")

    backend.register_gfx1151_kernels(replace=True)

    assert _registry_snapshot("hip_gfx1151") == before


def test_gfx1151_replacement_does_not_change_peer_registration(policy) -> None:
    backend, _policy_capability, _policy_gguf = policy
    backend.register_gfx1151_kernels(replace=True)
    peer_before = _registry_snapshot("hip_gfx1100")

    backend.register_gfx1151_kernels(replace=True)

    assert _registry_snapshot("hip_gfx1100") == peer_before
