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
    _STRICT_DECODE_WINS_UNTIL_KEYS,
    gemma4_attention_prefill_bf16,
    gemma4_attention_serves_keys,
    gemma4_attention_shared_bytes,
    gemma4_decode_max_keys,
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


def test_a_one_token_block_keeps_the_strict_kernel():
    """A decode step is not a prefill, however well its geometry matches.

    The candidates tile 16 query rows wide. Serving one row with that tiling
    wastes fifteen sixteenths of every WMMA op, and the strict path already has
    a decode kernel for this case -- so the variant must not be selected, and
    the reason has to say why rather than looking like a capability miss.
    """

    for geometry in (SLIDING, FULL):
        selection = select_prefill_attention(
            requested_variant=PRODUCTION, tokens=1, **geometry
        )

        assert selection.variant == PREFILL_ATTENTION_PLAIN
        assert selection.launcher is gemma4_attention_prefill_bf16
        assert selection.is_strict
        assert selection.reason.startswith("decode:")


def test_the_decode_rule_is_the_block_width_and_not_the_geometry():
    """Both sides of the boundary, and a width unrelated to it.

    Two rows is already a block the tiling can use, and the token count that
    matters is 1, not any geometry constant -- so 2 and 512 must both select the
    variant, and a wide block must not be treated as a decode step.
    """

    for tokens in (2, 7, 512):
        selection = select_prefill_attention(
            requested_variant=PRODUCTION, tokens=tokens, **SLIDING
        )

        assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH, tokens
        assert selection.reason == "capability match"


def test_an_unconstrained_request_still_answers_on_geometry_alone():
    """``tokens=None`` is the geometry question, and it keeps its old answer."""

    selection = select_prefill_attention(requested_variant=PRODUCTION, **SLIDING)

    assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH


def test_a_one_token_block_takes_the_variant_above_the_measured_crossover():
    """The decode rule is a measured crossover, not the strict kernel's LDS bound.

    Below the crossover the strict decode kernel is the faster path and keeps the
    block; above it the WMMA decode shape is faster, by a margin that grows
    super-linearly. The rule used to run to ``gemma4_decode_max_keys()`` on the
    reasoning that a WMMA tile wastes fifteen sixteenths of a one-row block --
    which stopped being true when a one-token block started selecting the 2-row
    decode shape, and which measured 5.2x against the kernel it was keeping at
    8192 keys.

    Both sides of the crossover are exercised, and the crossover itself is the
    value that must keep the strict kernel.
    """

    for keys in (_STRICT_DECODE_WINS_UNTIL_KEYS, _STRICT_DECODE_WINS_UNTIL_KEYS + 1):
        selection = select_prefill_attention(
            requested_variant=PRODUCTION, tokens=1, keys=keys, **FULL
        )
        if keys <= _STRICT_DECODE_WINS_UNTIL_KEYS:
            assert selection.variant == PREFILL_ATTENTION_PLAIN, keys
            assert selection.reason.startswith("decode:"), keys
        else:
            assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH_FULL, keys
            assert selection.reason == "capability match", keys


def test_the_rule_never_selects_a_strict_launch_that_cannot_start():
    """The selection must not promise a path the launcher refuses.

    At head_dim 512 the strict path's *resident* requirement crosses the 64 KiB
    budget above 15616 keys, so when this rule was written the strict kernel
    could not launch there at all, and a context in [15617, 15856] was accepted
    at construction, prefilled successfully, and then raised on its first
    decode:

        NotImplementedError: Gemma 4 gfx1100 attention requires 66276 bytes of
        shared memory for head_dim=512, keys=15801

    On this tree that refusal is gone: the 256/512 class kernel moves its logits
    to request-owned global scratch past the budget, so the strict path launches
    at any key count and the rule is a routing choice rather than the only
    launchable path. The crossover sits below both bounds, so the window is still
    unreachable. Both halves are asserted here rather than trusting the
    arithmetic to stay put.
    """

    for keys in range(15_616, 15_857):
        selection = select_prefill_attention(
            requested_variant=PRODUCTION, tokens=1, keys=keys, **FULL
        )
        assert selection.variant == PREFILL_ATTENTION_WMMA_FLASH_FULL, keys

    # The bound the window above is defined against: the strict path's resident
    # requirement. One key past it the requirement stops growing with the key
    # count, which is the global-scratch route rather than a refusal.
    assert gemma4_attention_shared_bytes(head_dim=512, keys=15_616) <= 64 * 1024
    past = gemma4_attention_shared_bytes(head_dim=512, keys=15_617)
    assert past == gemma4_attention_shared_bytes(head_dim=512, keys=262_144)
    assert past < gemma4_attention_shared_bytes(head_dim=512, keys=15_616)


def test_the_sliding_geometry_keeps_the_strict_decode_kernel_at_any_context():
    """A sliding layer's decode presents its window's worth of keys, always.

    This is the half the bound must not move. At 262,144 configured positions a
    sliding layer still reads 1,024 keys, so it keeps the exact-tiling decode
    kernel; only the windowless layers cross the bound and take the candidate.
    """

    selection = select_prefill_attention(
        requested_variant=PRODUCTION, tokens=1, keys=1024, **SLIDING
    )

    assert selection.variant == PREFILL_ATTENTION_PLAIN
    assert selection.reason.startswith("decode:")


def test_an_unknown_key_count_keeps_the_incumbent_decode_routing():
    """``keys=None`` is not evidence that there are too many keys.

    A caller that did not say how many keys it has gets the exact-tiling path,
    because guessing the other way would move every decode step onto a tiling
    that wastes fifteen sixteenths of its work.
    """

    selection = select_prefill_attention(requested_variant=PRODUCTION, tokens=1, **FULL)

    assert selection.variant == PREFILL_ATTENTION_PLAIN


def test_the_decode_bound_is_the_kernels_own_lds_formula():
    """The bound is derived from the LDS budget, not written down a second time.

    Asserted against ``gemma4_attention_shared_bytes`` rather than against a
    literal, so a change to the decode kernel's LDS layout moves the bound and
    this fails where the two disagree rather than letting them drift.

    At the bound the resident row still fits the budget. One key past it the
    resident row does not, and the 256/512 class kernel moves its logits to
    request-owned global scratch -- so the requirement stops growing with the
    key count instead of refusing the launch.
    """

    bound = gemma4_decode_max_keys()

    assert gemma4_attention_shared_bytes(head_dim=256, keys=bound) == (bound + 528) * 4
    assert gemma4_attention_shared_bytes(head_dim=256, keys=bound) <= 64 * 1024
    past = gemma4_attention_shared_bytes(head_dim=256, keys=bound + 1)
    assert past == gemma4_attention_shared_bytes(head_dim=256, keys=262_144)
    assert past < gemma4_attention_shared_bytes(head_dim=256, keys=bound)


def test_the_strict_kernel_serves_gemma4_geometries_at_any_key_count():
    """The global-logits route, pinned where it is visible.

    ``gemma4_attention_shared_bytes`` is the strict family's own answer, and for
    the two head widths Gemma 4 uses it stops growing with the key count rather
    than refusing: beyond the LDS budget the logits move to request-owned global
    scratch. The selected-path question below is therefore about geometries that
    have no such route.
    """

    for head_dim in (256, 512):
        assert gemma4_attention_shared_bytes(head_dim=head_dim, keys=1024) <= 65_536
        assert gemma4_attention_shared_bytes(head_dim=head_dim, keys=262_144) <= 65_536
        assert gemma4_attention_serves_keys(
            head_dim=head_dim, num_heads=16, num_kv_heads=2, keys=262_144
        ) is None


def test_a_tiled_variant_serves_a_layer_the_strict_kernel_cannot(monkeypatch):
    """The capacity question is asked of the family, not of the strict kernel alone.

    This is what makes a long context reachable rather than merely configurable:
    where the strict kernel refuses 262,144 keys outright, the answer depends on
    whether the requested profile selects a variant that implements the geometry
    and walks keys without holding them.

    The strict kernel's bound is narrowed here rather than relied on, because on
    Gemma 4's own geometries it does not refuse -- see
    ``test_the_strict_kernel_serves_gemma4_geometries_at_any_key_count``. What is
    pinned is the mechanism for when it does.
    """

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention_module

    def _refuse(*, head_dim: int, keys: int) -> int:
        raise NotImplementedError(f"narrowed for this test: {head_dim}/{keys}")

    monkeypatch.setattr(attention_module, "gemma4_attention_shared_bytes", _refuse)

    long_context = dict(head_dim=512, num_heads=16, num_kv_heads=2, keys=262_144)

    assert gemma4_attention_serves_keys(**long_context) is not None
    assert gemma4_attention_serves_keys(
        requested_variant=PRODUCTION, **long_context
    ) is None
    # A variant that does not implement the geometry does not serve it either.
    assert gemma4_attention_serves_keys(
        requested_variant=(PREFILL_ATTENTION_WMMA_FLASH,), **long_context
    ) is not None
