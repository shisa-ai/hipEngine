"""The f16 WMMA route's layer scope must participate in the dispatch cache key.

The cache key includes the layer scope as well as ``use_wmma``. These tests
pin that contract by recording keys, so lazy registry-generation changes cannot
hide an omitted scope behind unrelated cache misses.

The missing scope was a latent defect discovered during a gate investigation,
not the cause of that gate's false pass. The actual cause was dense-wide
claiming both comparison arms; reciprocal route-clearing tests cover that fix.
The corrected diagnosis is recorded in
``benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/WMMA-ROUTE-FALSE-PASS.md``.
"""

import os

import pytest


def _weight_module():
    from hipengine.runtime import gguf_linear

    class _Spec:
        layout = "raw_gguf"
        quant_key = "gguf_q8_0"
        slot_path = "layers.20.attn_qkv.weight"

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

    return gguf_linear, _Weight


def test_wmma_scope_empty_means_every_layer_not_off():
    """Empty means "no filter", matching what the engine does.

    This is deliberately the **opposite** of ``_q8_staged_f32_layers``, where
    empty means off. The runner reads this variable as
    ``wmma_prefill_session(True if q8_wmma_layers else None)`` - an empty scope
    removes the layer filter rather than disabling the route, and enablement is
    governed separately by ``_WMMA_PREFILL_ENV``. If this helper disagreed with
    the runner, the cache key would distinguish two configurations the engine
    treats as identical and the fix would be worse than the bug.
    """

    from hipengine.runtime import gguf_linear

    env = gguf_linear._Q8_WMMA_LAYERS_ENV
    saved = os.environ.get(env)
    try:
        os.environ.pop(env, None)
        assert gguf_linear._q8_wmma_layers_scope() is None
        os.environ[env] = ""
        assert gguf_linear._q8_wmma_layers_scope() is None
        os.environ[env] = "16,17,18"
        assert gguf_linear._q8_wmma_layers_scope() == frozenset({16, 17, 18})
    finally:
        os.environ.pop(env, None)
        if saved is not None:
            os.environ[env] = saved


def test_wmma_scope_participates_in_the_dispatch_memoization_key():
    """The regression that made a promoted route unverifiable at HEAD.

    Records the key the real ``launch_gguf_linear`` uses rather than observing
    cache misses: the registry ``generation()`` is the first key element and the
    lazy kernel registration inside the call bumps it, so every call misses for
    an unrelated reason and a miss-based test passes with the term removed.
    """

    gguf_linear, _Weight = _weight_module()

    env = gguf_linear._Q8_WMMA_LAYERS_ENV
    saved = os.environ.get(env)
    recorded: list[tuple] = []

    class _Recording(dict):
        def get(self, key, default=None):
            recorded.append(key)
            return super().get(key, default)

    def _key_for(scope: str) -> tuple:
        recorded.clear()
        os.environ[env] = scope
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
        scoped = _key_for("16,17,18")
        cleared = _key_for("")
        assert scoped != cleared, (
            "the WMMA layer scope does not participate in the dispatch "
            "memoization key, so a gate's teacher arm can replay its candidate "
            "arm and report a KL of exactly zero as a pass"
        )
        assert _key_for("16,17,18") != _key_for("16,17"), (
            "the layer scope does not move the key, so the first resolution for "
            "a shape is replayed for every layer"
        )
    finally:
        gguf_linear._DISPATCH_RESOLVE_CACHE = original
        os.environ.pop(env, None)
        if saved is not None:
            os.environ[env] = saved
