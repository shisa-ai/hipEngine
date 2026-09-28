"""The Gemma 4 prefill-attention variant selection, on CPU.

Three contracts, and they are separate ones:

* **Capability.** The WMMA variant implements the sliding geometry (head_dim
  256, GQA ratio 2) and nothing else. A request for it anywhere else keeps the
  strict kernel and says why. This is a capability match, so it is exercised on
  values on both sides of the boundary and on values unrelated to it, rather
  than at the boundary alone.
* **Profile.** ``production`` requests the variant, ``strict`` does not, and an
  unset profile resolves to the shipped default. Registering both plans is what
  makes that default ``production``.
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
    PREFILL_ATTENTION_SCOPE,
    PREFILL_ATTENTION_WMMA_FLASH,
    gemma4_gguf_profiles_registered,
    register_gemma4_gguf_profiles,
    resolve_gemma4_prefill_attention_variant,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
    select_prefill_attention,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma import (
    gemma4_attention_prefill_wmma_bf16,
)

# The sliding geometry, the full geometry, and shapes unrelated to either.
SLIDING = dict(num_heads=16, num_kv_heads=8, head_dim=256)
FULL = dict(num_heads=16, num_kv_heads=2, head_dim=512)


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


@pytest.mark.parametrize(
    "geometry",
    [
        # The full layers: head_dim 512 and a GQA ratio of 8, both wrong.
        FULL,
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


def test_an_unknown_variant_name_keeps_the_strict_kernel():
    selection = select_prefill_attention(requested_variant="no_such_variant", **SLIDING)

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert "unknown variant" in selection.reason


def test_the_two_launchers_accept_the_same_call_site_arguments():
    """The layer passes one argument list, so both must accept all of it.

    ``scratch`` is the one that differs in substance: the strict kernel
    materialises logits and needs a caller-owned buffer, and the WMMA variant
    stages its K/V tile in LDS instead. It has to be accepted all the same.
    """

    import inspect

    strict = set(inspect.signature(gemma4_attention_prefill_bf16).parameters)
    candidate = set(inspect.signature(gemma4_attention_prefill_wmma_bf16).parameters)

    assert strict <= candidate, f"the candidate is missing {sorted(strict - candidate)}"


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
        (ExecutionProfile.STRICT, PREFILL_ATTENTION_PLAIN),
        (ExecutionProfile.PRODUCTION, PREFILL_ATTENTION_WMMA_FLASH),
        (None, PREFILL_ATTENTION_WMMA_FLASH),
    ],
)
def test_the_profile_selects_the_variant(requested, expected):
    assert (
        resolve_gemma4_prefill_attention_variant(requested_profile=requested)
        == expected
    )


def test_the_environment_variable_selects_the_variant():
    assert (
        resolve_gemma4_prefill_attention_variant(environ={"HIPENGINE_EXECUTION_PROFILE": "strict"})
        == PREFILL_ATTENTION_PLAIN
    )
    assert (
        resolve_gemma4_prefill_attention_variant(
            environ={"HIPENGINE_EXECUTION_PROFILE": "production"}
        )
        == PREFILL_ATTENTION_WMMA_FLASH
    )
    assert (
        resolve_gemma4_prefill_attention_variant(environ={})
        == PREFILL_ATTENTION_WMMA_FLASH
    )


def test_an_unregistered_backend_keeps_the_strict_kernel():
    """The plan is written for the Strix Halo alias, not for every backend.

    A backend with no plan is not an error: it is the migration path, and the
    strict kernel is what it has always run.
    """

    assert resolve_gemma4_prefill_attention_variant(backend="hip_gfx1100") is None
    assert resolve_gemma4_prefill_attention_variant(backend="cpu_reference") is None


def test_the_selected_variant_is_the_one_the_registry_holds():
    """A plan may only name a variant the registry can actually resolve.

    ``resolve_runtime_profile`` enforces this, and the check is repeated here
    against the selection the layer would make, so a plan that names a variant
    nobody registered fails at test time rather than at request time.
    """

    from hipengine.kernels.registry import KernelKey, is_registered

    register_gemma4_gguf_profiles()

    for variant in (PREFILL_ATTENTION_PLAIN, PREFILL_ATTENTION_WMMA_FLASH):
        assert is_registered(
            KernelKey(GEMMA4_GGUF_BACKEND, "prefill_attention", GEMMA4_GGUF_QUANT, variant)
        ), f"{variant} is not registered for {GEMMA4_GGUF_BACKEND}"


def test_the_selection_scope_is_the_one_the_plan_declares():
    """The layer's scope string and the plan's have to be the same one."""

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
