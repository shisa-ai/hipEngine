"""The Gemma 4 prefill-attention variant selection, on CPU.

Three contracts, and they are separate ones:

* **Capability.** Each WMMA variant implements one geometry: ``gemma4_wmma_flash``
  the sliding one (head_dim 256, GQA ratio 2) and ``gemma4_wmma_flash_full`` the
  full one (head_dim 512, GQA ratio 8), and nothing else. A request for either
  anywhere else keeps the strict kernel and says why. This is a capability
  match, so it is exercised on values on both sides of the boundary and on
  values unrelated to it, rather than at the boundary alone.
* **Profile.** ``production`` requests both variants, ``strict`` requests
  neither, and an unset profile resolves to the shipped default. Registering
  both plans is what makes that default ``production``.
* **Fallback.** An unknown variant name, a missing plan, or an unregistered
  backend must reach the strict kernel rather than raise, because a profile is a
  performance decision and not a licence to fail a request.
"""

from __future__ import annotations

import pytest

from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.gemma4_gguf_profiles import (
    GEMMA4_GGUF_BACKEND,
    GEMMA4_GGUF_MODEL,
    GEMMA4_GGUF_QUANT,
    PREFILL_ATTENTION_PLAIN,
    PREFILL_ATTENTION_PRODUCTION_VARIANTS,
    PREFILL_ATTENTION_SCOPE,
    PREFILL_ATTENTION_SCOPE_FULL,
    PREFILL_ATTENTION_WMMA_FLASH,
    PREFILL_ATTENTION_WMMA_FLASH_FULL,
    gemma4_gguf_profiles_registered,
    register_gemma4_gguf_profiles,
    resolve_gemma4_prefill_attention_variants,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
    select_prefill_attention,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma import (
    gemma4_attention_prefill_wmma_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
    gemma4_attention_prefill_wmma_full_bf16,
)

# The sliding geometry, the full geometry, and shapes unrelated to either.
SLIDING = dict(num_heads=16, num_kv_heads=8, head_dim=256)
FULL = dict(num_heads=16, num_kv_heads=2, head_dim=512)
# What a production run requests, taken from the module that declares it so the
# resolver and the constant cannot drift apart.
PRODUCTION = PREFILL_ATTENTION_PRODUCTION_VARIANTS


def test_an_unrequested_variant_is_the_strict_kernel():
    selection = select_prefill_attention(requested_variant=None, **SLIDING)

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert selection.launcher is gemma4_attention_prefill_bf16
    assert selection.is_strict


def test_the_sliding_geometry_gets_the_wmma_variant():
    selection = select_prefill_attention(
        requested_variant=PREFILL_ATTENTION_WMMA_FLASH, **SLIDING
    )

    assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH
    assert selection.launcher is gemma4_attention_prefill_wmma_bf16
    assert not selection.is_strict
    assert "capability match" in selection.describe()


def test_the_full_geometry_gets_the_full_wmma_variant():
    selection = select_prefill_attention(
        requested_variant=PREFILL_ATTENTION_WMMA_FLASH_FULL, **FULL
    )

    assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH_FULL
    assert selection.launcher is gemma4_attention_prefill_wmma_full_bf16
    assert not selection.is_strict
    assert "capability match" in selection.describe()


def test_a_production_request_covers_both_geometries():
    """One request list, one layer at a time: each geometry gets its own variant.

    This is the contract that matters in production, where both geometries occur
    in the same model and the layer -- not the profile -- is what knows which
    one it is.
    """

    sliding = select_prefill_attention(requested_variant=PRODUCTION, **SLIDING)
    full = select_prefill_attention(requested_variant=PRODUCTION, **FULL)

    assert sliding.variant == PREFILL_ATTENTION_WMMA_FLASH
    assert sliding.launcher is gemma4_attention_prefill_wmma_bf16
    assert full.variant == PREFILL_ATTENTION_WMMA_FLASH_FULL
    assert full.launcher is gemma4_attention_prefill_wmma_full_bf16


@pytest.mark.parametrize(
    "geometry",
    [
        # Each half of the sliding geometry, wrong on its own. A selection that
        # checked only head_dim would admit the first and only the ratio the
        # second.
        dict(num_heads=16, num_kv_heads=2, head_dim=256),
        dict(num_heads=16, num_kv_heads=8, head_dim=128),
        dict(num_heads=16, num_kv_heads=8, head_dim=512),
        # Unrelated shapes, so the check is exercised away from its boundary.
        dict(num_heads=8, num_kv_heads=1, head_dim=64),
        dict(num_heads=32, num_kv_heads=4, head_dim=256),
        dict(num_heads=1, num_kv_heads=1, head_dim=256),
    ],
)
def test_a_geometry_the_variant_does_not_implement_keeps_the_strict_kernel(geometry):
    selection = select_prefill_attention(
        requested_variant=PREFILL_ATTENTION_WMMA_FLASH, **geometry
    )

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert selection.launcher is gemma4_attention_prefill_bf16
    # The refusal has to name what it wanted and what it got, or a silent
    # fallback is indistinguishable from a working one.
    assert "capability miss" in selection.reason
    assert str(geometry["head_dim"]) in selection.reason


@pytest.mark.parametrize(
    "geometry",
    [
        # The sliding geometry: head_dim 256 and a GQA ratio of 2, both wrong.
        SLIDING,
        # Each half of the full geometry, wrong on its own.
        dict(num_heads=16, num_kv_heads=8, head_dim=512),
        dict(num_heads=16, num_kv_heads=2, head_dim=256),
        # Unrelated shapes, so the check is exercised away from its boundary.
        dict(num_heads=8, num_kv_heads=1, head_dim=64),
        dict(num_heads=32, num_kv_heads=8, head_dim=512),
        dict(num_heads=16, num_kv_heads=2, head_dim=1024),
    ],
)
def test_a_geometry_the_full_variant_does_not_implement_keeps_the_strict_kernel(geometry):
    selection = select_prefill_attention(
        requested_variant=PREFILL_ATTENTION_WMMA_FLASH_FULL, **geometry
    )

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert selection.launcher is gemma4_attention_prefill_bf16
    assert "capability miss" in selection.reason
    assert str(geometry["head_dim"]) in selection.reason


def test_a_production_request_keeps_the_strict_kernel_off_both_geometries():
    """A geometry neither variant implements still runs, and says both misses."""

    selection = select_prefill_attention(
        requested_variant=PRODUCTION, num_heads=8, num_kv_heads=1, head_dim=64
    )

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert selection.launcher is gemma4_attention_prefill_bf16
    assert PREFILL_ATTENTION_WMMA_FLASH in selection.reason
    assert PREFILL_ATTENTION_WMMA_FLASH_FULL in selection.reason


def test_an_unknown_variant_name_keeps_the_strict_kernel():
    selection = select_prefill_attention(requested_variant="no_such_variant", **SLIDING)

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert "unknown variant" in selection.reason


def test_an_unknown_name_ahead_of_a_known_one_does_not_shadow_it():
    """The request list is tried in order, so a bad name is not a hard failure."""

    selection = select_prefill_attention(
        requested_variant=("no_such_variant", PREFILL_ATTENTION_WMMA_FLASH), **SLIDING
    )

    assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH


def test_the_two_launchers_accept_the_same_call_site_arguments():
    """The layer passes one argument list, so all three must accept all of it.

    ``scratch`` is the one that differs in substance: the strict kernel
    materialises logits and needs a caller-owned buffer, and the WMMA variants
    stage their K/V tile in LDS instead. It has to be accepted all the same.
    """

    import inspect

    strict = set(inspect.signature(gemma4_attention_prefill_bf16).parameters)
    sliding = set(inspect.signature(gemma4_attention_prefill_wmma_bf16).parameters)
    full = set(inspect.signature(gemma4_attention_prefill_wmma_full_bf16).parameters)

    assert strict <= sliding, f"the candidate is missing {sorted(strict - sliding)}"
    assert strict <= full, f"the full candidate is missing {sorted(strict - full)}"


def test_the_profiles_register_and_make_production_the_default():
    register_gemma4_gguf_profiles()

    assert gemma4_gguf_profiles_registered()

    from hipengine.execution_profiles import resolve_default_execution_profile

    assert (
        resolve_default_execution_profile(
            model=GEMMA4_GGUF_MODEL, backend=GEMMA4_GGUF_BACKEND, quant=GEMMA4_GGUF_QUANT
        )
        is ExecutionProfile.PRODUCTION
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (ExecutionProfile.STRICT, (PREFILL_ATTENTION_PLAIN,)),
        (ExecutionProfile.PRODUCTION, PRODUCTION),
        (None, PRODUCTION),
    ],
)
def test_the_profile_selects_the_variants(requested, expected):
    assert resolve_gemma4_prefill_attention_variants(requested_profile=requested) == expected


def test_the_environment_variable_selects_the_variants():
    assert resolve_gemma4_prefill_attention_variants(
        environ={"HIPENGINE_EXECUTION_PROFILE": "strict"}
    ) == (PREFILL_ATTENTION_PLAIN,)
    assert resolve_gemma4_prefill_attention_variants(
        environ={"HIPENGINE_EXECUTION_PROFILE": "production"}
    ) == PRODUCTION
    assert resolve_gemma4_prefill_attention_variants(environ={}) == PRODUCTION


def test_an_unregistered_backend_keeps_the_strict_kernel():
    """The plan is written for the Strix Halo alias, not for every backend.

    A backend with no plan is not an error: it is the migration path, and the
    strict kernel is what it has always run.
    """

    assert resolve_gemma4_prefill_attention_variants(backend="hip_gfx1100") == ()
    assert resolve_gemma4_prefill_attention_variants(backend="cpu_reference") == ()


def test_the_selected_variants_are_the_ones_the_registry_holds():
    """A plan may only name a variant the registry can actually resolve.

    ``resolve_runtime_profile`` enforces this, and the check is repeated here
    against the selection the layer would make, so a plan that names a variant
    nobody registered fails at test time rather than at request time.
    """

    from hipengine.kernels.registry import KernelKey, is_registered

    register_gemma4_gguf_profiles()

    for variant in (
        PREFILL_ATTENTION_PLAIN,
        PREFILL_ATTENTION_WMMA_FLASH,
        PREFILL_ATTENTION_WMMA_FLASH_FULL,
    ):
        assert is_registered(
            KernelKey(GEMMA4_GGUF_BACKEND, "prefill_attention", GEMMA4_GGUF_QUANT, variant)
        ), f"{variant} is not registered for {GEMMA4_GGUF_BACKEND}"


def test_the_selection_scopes_are_the_ones_the_plan_declares():
    """The layer's scope strings and the plan's have to be the same ones."""

    from hipengine.execution_profiles import resolve_runtime_profile

    register_gemma4_gguf_profiles()
    resolved = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
    )
    scopes = {
        (selection["layer"], selection["scope"])
        for selection in resolved.manifest["selections"]
    }

    assert ("prefill_attention", PREFILL_ATTENTION_SCOPE) in scopes
    assert ("prefill_attention", PREFILL_ATTENTION_SCOPE_FULL) in scopes


def test_the_production_plan_carries_one_selection_per_geometry():
    """Both geometries have to be named, or the full layers have no stated intent."""

    from hipengine.execution_profiles import resolve_runtime_profile

    register_gemma4_gguf_profiles()
    for profile, expected in (
        (ExecutionProfile.PRODUCTION, PRODUCTION),
        (ExecutionProfile.STRICT, (PREFILL_ATTENTION_PLAIN,)),
    ):
        resolved = resolve_runtime_profile(
            model=GEMMA4_GGUF_MODEL,
            backend=GEMMA4_GGUF_BACKEND,
            quant=GEMMA4_GGUF_QUANT,
            profile=profile,
        )
        rows = [
            selection
            for selection in resolved.manifest["selections"]
            if selection["layer"] == "prefill_attention"
        ]
        assert len(rows) == 2, f"{profile} declares {len(rows)} attention selections"
        assert {row["scope"] for row in rows} == {
            PREFILL_ATTENTION_SCOPE,
            PREFILL_ATTENTION_SCOPE_FULL,
        }
        assert tuple(dict.fromkeys(sorted(row["selected_variant"] for row in rows))) == expected
        for row in rows:
            assert row["strict_fallback_variant"] == PREFILL_ATTENTION_PLAIN
