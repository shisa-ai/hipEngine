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
    """Empty removes the filter; explicit sessions preserve layer selection."""
    from hipengine.runtime import gguf_linear

    with gguf_linear.q8_wmma_layer_scope_session(gguf_linear.Q8WMMALayerScope.empty()):
        assert gguf_linear._q8_wmma_layers_scope() is None
    with gguf_linear.q8_wmma_layer_scope_session(gguf_linear.Q8WMMALayerScope.parse("16,17,18")):
        assert gguf_linear._q8_wmma_layers_scope() == frozenset({16, 17, 18})


def test_wmma_scope_participates_in_the_dispatch_memoization_key():
    """Record the key from the real launch rather than relying on cache misses.

    Lazy registration also changes registry generation, so cache misses alone
    cannot prove that a layer scope participates in the key.
    """

    gguf_linear, _Weight = _weight_module()

    recorded: list[tuple] = []

    class _Recording(dict):
        def get(self, key, default=None):
            recorded.append(key)
            return super().get(key, default)

    def _key_for(scope: str) -> tuple:
        recorded.clear()
        try:
            with gguf_linear.q8_wmma_layer_scope_session(
                    gguf_linear.Q8WMMALayerScope.parse(scope)):
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
