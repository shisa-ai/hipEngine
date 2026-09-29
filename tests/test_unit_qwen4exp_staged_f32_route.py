"""Wiring contract for the opt-in LDS-staged f32 dense Q8_0 route.

The route is a production-gate candidate: it is 1.23x to 2.22x faster than the
exact coltile leaf across all six Q8_0 dense projections of a layer step, but its
output is not bit-identical to that leaf. That makes the *wiring* the safety
property under test here, not the arithmetic:

- it must be **off** when nothing binds it, because an unset environment that
  silently selected a non-bit-exact route would change every default;
- the variant name must be the same string in the runtime dispatch and in the
  backend registration, or the dispatch silently no-ops and the route looks
  enabled while doing nothing;
- the layer parser must read a scope, since a boolean cannot express the
  configuration a gate needs.

The arithmetic itself is covered by
``benchmarks/results/2026-09-29-dp8-staged-f32-dense-family/`` and is a
production gate's job, not this module's.
"""

import os

import pytest

from hipengine.kernels.registry import KernelKey, is_registered, resolve


def test_variant_name_agrees_between_runtime_and_backend_package():
    """A drifted name makes the dispatch a silent no-op.

    ``hipengine.runtime.gguf_linear`` keeps the name as a literal so it does not
    import a backend package eagerly. This is the check that the two stay equal.
    """

    from hipengine.kernels.hip_gfx1151 import STAGED_F32_VARIANT
    from hipengine.runtime import gguf_linear

    assert gguf_linear._Q8_STAGED_F32_VARIANT == STAGED_F32_VARIANT


def test_route_is_off_when_nothing_binds_it():
    """Unset must mean off, not every layer.

    This is the opposite of the sibling wide route, whose parser treats empty as
    "every layer". For a route that is not bit-exact, that default would change
    the engine's arithmetic with no opt-in.
    """

    from hipengine.runtime import gguf_linear

    saved = os.environ.pop(gguf_linear._Q8_STAGED_F32_LAYERS_ENV, None)
    try:
        assert gguf_linear._q8_staged_f32_layers() is None
        os.environ[gguf_linear._Q8_STAGED_F32_LAYERS_ENV] = ""
        assert gguf_linear._q8_staged_f32_layers() is None
        # Whitespace and separators only: still nothing named, still off.
        os.environ[gguf_linear._Q8_STAGED_F32_LAYERS_ENV] = " , ,"
        assert gguf_linear._q8_staged_f32_layers() is None
    finally:
        os.environ.pop(gguf_linear._Q8_STAGED_F32_LAYERS_ENV, None)
        if saved is not None:
            os.environ[gguf_linear._Q8_STAGED_F32_LAYERS_ENV] = saved


def test_layer_zero_is_layer_zero_not_off():
    """``=0`` selects layer 0. Pinned because the spelling is a footgun.

    There is no in-band way to spell both "layer 0" and "off", so this records
    which one the parser means. Someone who wants the route off must unset the
    variable rather than set it to ``0``.
    """

    from hipengine.runtime import gguf_linear

    env = gguf_linear._Q8_STAGED_F32_LAYERS_ENV
    saved = os.environ.get(env)
    try:
        os.environ[env] = "0"
        assert gguf_linear._q8_staged_f32_layers() == frozenset({0})
    finally:
        os.environ.pop(env, None)
        if saved is not None:
            os.environ[env] = saved


def test_layer_scope_parses_and_ignores_blanks():
    from hipengine.runtime import gguf_linear

    env = gguf_linear._Q8_STAGED_F32_LAYERS_ENV
    saved = os.environ.get(env)
    try:
        os.environ[env] = "0, 1 ,2,"
        assert gguf_linear._q8_staged_f32_layers() == frozenset({0, 1, 2})
    finally:
        os.environ.pop(env, None)
        if saved is not None:
            os.environ[env] = saved


def test_registration_resolves_to_the_staged_launcher():
    """The registered callable must be the staged launcher, not a sibling.

    ``resolve`` returning some other kernel would make the dispatch's swap
    silently select the wrong arithmetic under a name that says otherwise.
    """

    from hipengine.kernels.hip_gfx1151 import STAGED_F32_VARIANT, register_gfx1151_kernels
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_gr_staged_f32 as staged

    register_gfx1151_kernels(replace=True)
    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", STAGED_F32_VARIANT)
    assert is_registered(key)
    assert resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    ) is staged.gr_staged_f32_up


def test_staged_launcher_rejects_a_geometry_it_would_ignore():
    """``threads`` is accepted for the raw ABI but must not be silently ignored.

    The tile's block size is fixed at 256 by its BM=128/BN=64 geometry. A caller
    that passes a different count believes it is choosing a geometry; it should
    be told it is not, rather than getting a kernel that ignores it.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_gr_staged_f32 import (
        gr_staged_f32_up,
    )

    with pytest.raises(ValueError, match="fixed 256-thread block"):
        gr_staged_f32_up(0, 0, 0, 8, 32, 8, threads=128)


class _FakeSpec:
    def __init__(self, slot_path):
        self.slot_path = slot_path


class _FakeWeight:
    def __init__(self, slot_path):
        self.spec = _FakeSpec(slot_path)


def _exact_leaf_dispatch():
    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime.gguf_linear import GGUFLinearDispatch

    return GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out",
        ),
        "raw",
    )


def test_dispatch_is_a_no_op_when_the_route_is_unbound():
    """The load-bearing safety property: no opt-in, no arithmetic change.

    This is what makes the candidate safe to leave registered. If it ever fails,
    the engine's default dense Q8 arithmetic has silently changed.
    """

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.runtime import gguf_linear

    register_gfx1151_kernels(replace=True)
    env = gguf_linear._Q8_STAGED_F32_LAYERS_ENV
    saved = os.environ.pop(env, None)
    try:
        original = _exact_leaf_dispatch()
        weight = _FakeWeight("layers.0.attn_qkv.weight")
        result = gguf_linear._raw_k_staged_f32_dispatch(original, weight=weight)
        assert result is original, (
            "unbound route replaced the exact leaf; default arithmetic changed"
        )
    finally:
        if saved is not None:
            os.environ[env] = saved


def test_dispatch_swaps_only_inside_the_named_scope():
    from hipengine.kernels.hip_gfx1151 import (
        STAGED_F32_VARIANT,
        register_gfx1151_kernels,
    )
    from hipengine.runtime import gguf_linear

    register_gfx1151_kernels(replace=True)
    env = gguf_linear._Q8_STAGED_F32_LAYERS_ENV
    saved = os.environ.get(env)
    try:
        os.environ[env] = "4,5"
        original = _exact_leaf_dispatch()
        # Inside the scope: swapped.
        inside = gguf_linear._raw_k_staged_f32_dispatch(
            original, weight=_FakeWeight("layers.5.attn_qkv.weight")
        )
        assert inside.key.variant == STAGED_F32_VARIANT
        # Outside the scope: untouched.
        outside = gguf_linear._raw_k_staged_f32_dispatch(
            original, weight=_FakeWeight("layers.9.attn_qkv.weight")
        )
        assert outside is original
        # A weight with no layer index cannot be scoped, so it is left alone.
        unscoped = gguf_linear._raw_k_staged_f32_dispatch(
            original, weight=_FakeWeight("lm_head.weight")
        )
        assert unscoped is original
    finally:
        os.environ.pop(env, None)
        if saved is not None:
            os.environ[env] = saved


def test_selector_participates_in_the_dispatch_memoization_key():
    """The regression that produced a false gate pass.

    The two arms of a numerical gate are the same shapes on the same weight
    pointers and differ **only** in this selector. When the selector is absent
    from the dispatch memoization key, the teacher arm's exact-chain dispatch is
    replayed for the candidate arm. The symptom is not a wrong answer but a
    silent one: bit-identical candidate/teacher logits and a KL of exactly zero,
    which reads as a flawless pass.

    This records the key the real ``launch_gguf_linear`` actually uses rather
    than observing cache misses. Observing misses does **not** work here: the
    registry ``generation()`` is the first key element and the lazy kernel
    registration inside the call bumps it, so every call misses for an unrelated
    reason and a test written that way passes with the term removed.
    """

    from hipengine.runtime import gguf_linear

    env = gguf_linear._Q8_STAGED_F32_LAYERS_ENV
    saved = os.environ.get(env)
    recorded: list[tuple] = []

    class _Recording(dict):
        def get(self, key, default=None):
            recorded.append(key)
            return super().get(key, default)

    class _Spec:
        layout = "raw_gguf"
        quant_key = "gguf_q8_0"
        slot_path = "layers.3.attn_qkv.weight"

    class _Allocation:
        class tensor:  # noqa: N801 - mirrors the resident-weight attribute path
            ptr = 4096

    class _Weight:
        spec = _Spec()
        backend = "hip_gfx1151"

        @staticmethod
        def allocation(name):
            assert name == "raw"
            return _Allocation()

    def _key_for(selector: str) -> tuple:
        recorded.clear()
        os.environ[env] = selector
        try:
            gguf_linear.launch_gguf_linear(
                _Weight(), 0, 0, 1024, 2560, 10240, libraries={}
            )
        except Exception:
            # The launch cannot succeed on synthetic pointers. The memoization
            # key is computed before any of that, which is the part under test.
            pass
        assert recorded, (
            "the call never reached the dispatch memoization; the test did not "
            "exercise the path under test"
        )
        return recorded[-1]

    original = gguf_linear._DISPATCH_RESOLVE_CACHE
    try:
        gguf_linear._DISPATCH_RESOLVE_CACHE = _Recording()
        teacher = _key_for("")
        candidate = _key_for("3")
        assert teacher != candidate, (
            "the staged-f32 selector does not participate in the dispatch "
            "memoization key, so a gate's teacher arm can be replayed for its "
            "candidate arm and report a KL of exactly zero as a pass"
        )
        # And the scope must move the key too, or the first resolution for a
        # shape is replayed for every layer and the scope becomes an artifact
        # of cache-fill order rather than a contract.
        assert _key_for("3") != _key_for("4"), (
            "the layer scope does not participate in the memoization key"
        )
    finally:
        gguf_linear._DISPATCH_RESOLVE_CACHE = original
        os.environ.pop(env, None)
        if saved is not None:
            os.environ[env] = saved
