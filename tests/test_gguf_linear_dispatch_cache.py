"""RED-first gate for the launch_gguf_linear dispatch-resolve cache (task #9).

The per-launch dispatch-resolve (env reads + the 5-stage dispatch-transform
chain + registry resolve) is ~60% of the ~25us host cost of a dense GGUF linear
launch (scripts/gguf_launch_overhead_bench.py). It is memoized, keyed on the
resolution inputs plus the registry generation, so:

* repeated identical launches skip the resolve chain (cache hit), and
* any registry mutation (register/unregister) bumps the generation and
  invalidates the memo, so dispatch stays correct under the registry-swap idiom
  the existing dispatch tests rely on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gl
from hipengine.kernels.registry import KernelKey, _KERNELS, register, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import launch_gguf_linear

_KEY = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")


def _fake_weight(*, layout: str, quant_key: str):
    alloc = SimpleNamespace(tensor=SimpleNamespace(ptr=10))

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return alloc

    return Weight()


def test_dispatch_cache_memoizes_and_invalidates_on_registry_change(monkeypatch) -> None:
    gl.clear_gguf_linear_dispatch_cache()
    calls = {"n": 0}
    orig_resolve_dispatch = gl.resolve_gguf_linear_dispatch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig_resolve_dispatch(*args, **kwargs)

    monkeypatch.setattr(gl, "resolve_gguf_linear_dispatch", counting)

    weight = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    fired: list[str] = []
    saved = resolve(
        backend=_KEY.backend, layer=_KEY.layer, quant=_KEY.quant, variant=_KEY.variant, missing="none"
    )

    def fake1(*args, **kwargs):
        fired.append("fake1")

    register(_KEY, fake1, replace=True)
    try:
        for _ in range(2):
            launch_gguf_linear(
                weight, x_ptr=1, out_ptr=2, rows=1, in_features=1024, out_features=2048, runtime="rt"
            )
        # Second identical launch is a cache hit: the resolve chain ran once.
        assert calls["n"] == 1
        assert fired == ["fake1", "fake1"]

        # Registry mutation bumps the generation -> memo invalidated -> the new
        # kernel is picked up (this is what protects the dispatch-swap tests).
        def fake2(*args, **kwargs):
            fired.append("fake2")

        register(_KEY, fake2, replace=True)
        launch_gguf_linear(
            weight, x_ptr=1, out_ptr=2, rows=1, in_features=1024, out_features=2048, runtime="rt"
        )
        assert calls["n"] == 2
        assert fired[-1] == "fake2"
    finally:
        if saved is None:
            _KERNELS.pop(_KEY, None)
        else:
            register(_KEY, saved, replace=True)
        gl.clear_gguf_linear_dispatch_cache()
