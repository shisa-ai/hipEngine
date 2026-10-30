"""Move-only boundary contract for the gfx1151 Laguna policy declarations.

``policy_laguna`` owns the immutable Laguna capability/policy values; the
backend package keeps explicit compatibility exports. Registration stays in
``hipengine.kernels.hip_gfx1151`` and is not touched by this module.
"""

from __future__ import annotations

import hipengine.kernels.hip_gfx1151 as backend
from hipengine.kernels.hip_gfx1151 import policy_laguna
from hipengine.kernels.registry import registered_keys, resolve


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


def test_policy_module_owns_only_laguna_declarations() -> None:
    declared = {
        name
        for name in vars(policy_laguna)
        if name.isupper() and not name.startswith("_")
    }

    assert declared == set(policy_laguna.__all__)
    assert all(name.startswith("LAGUNA_") for name in declared)
    for name in declared:
        value = getattr(policy_laguna, name)
        # Declarations are data: no callables, modules, or registry handles.
        assert not callable(value)
        assert isinstance(value, (bool, int, str, tuple, frozenset, dict))

    # The policy module must not acquire a second registry or import machinery.
    assert not hasattr(policy_laguna, "register")
    assert not hasattr(policy_laguna, "KernelKey")
    assert not hasattr(policy_laguna, "import_module")


def test_package_reexports_every_policy_declaration_explicitly() -> None:
    for name in policy_laguna.__all__:
        assert getattr(backend, name) is getattr(policy_laguna, name)

    # The package keeps its historical star-import surface: every Laguna name
    # it already listed still resolves from the owning module, and the moved
    # block did not add names to ``__all__``.
    listed_laguna = {name for name in backend.__all__ if name.startswith("LAGUNA_")}
    assert listed_laguna <= set(policy_laguna.__all__)

    from hipengine.kernels.hip_gfx1151 import LAGUNA_F16_PREFILL_STRATEGY

    assert LAGUNA_F16_PREFILL_STRATEGY is policy_laguna.LAGUNA_F16_PREFILL_STRATEGY
    assert "LAGUNA_F16_PREFILL_STRATEGY" in backend.__all__


def test_registration_snapshot_is_stable_across_replacement() -> None:
    # A pytest session may not have aliased every lazily imported peer module
    # yet; establish the complete alias state before comparing replacements.
    backend.register_gfx1151_kernels(replace=True)
    before = _registry_snapshot("hip_gfx1151")

    backend.register_gfx1151_kernels(replace=True)

    assert _registry_snapshot("hip_gfx1151") == before


def test_gfx1151_replacement_does_not_change_peer_registration() -> None:
    backend.register_gfx1151_kernels(replace=True)
    peer_before = _registry_snapshot("hip_gfx1100")

    backend.register_gfx1151_kernels(replace=True)

    assert _registry_snapshot("hip_gfx1100") == peer_before
