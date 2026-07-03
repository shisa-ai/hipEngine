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
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_GGUF_Q8_0_T16, LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import launch_gguf_linear, launch_gguf_linear_pair

_KEY = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
_PAIR_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_dual_gemv_decode_bf16_bf16_out"
)


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


def test_pair_dispatch_cache_memoizes_and_invalidates_on_registry_change(monkeypatch) -> None:
    gl.clear_gguf_linear_dispatch_cache()
    calls = {"n": 0}
    orig_resolve_dispatch = gl.resolve_gguf_linear_dispatch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig_resolve_dispatch(*args, **kwargs)

    monkeypatch.setattr(gl, "resolve_gguf_linear_dispatch", counting)

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    fired: list[str] = []
    saved_kernel = resolve(
        backend=_PAIR_KEY.backend,
        layer=_PAIR_KEY.layer,
        quant=_PAIR_KEY.quant,
        variant=_PAIR_KEY.variant,
        missing="none",
    )
    saved_wrapper = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out

    def fake_wrapper(*args, **kwargs):
        fired.append("wrapper")

    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_wrapper  # type: ignore[assignment]
    try:
        for _ in range(2):
            assert launch_gguf_linear_pair(
                weight_a,
                weight_b,
                x_ptr=1,
                out_a_ptr=2,
                out_b_ptr=3,
                rows=2,
                in_features=2048,
                out_features=8192,
                out_features_b=4096,
                runtime="rt",
            )
        # First pair launch resolves both weights; the second identical launch
        # reuses the pair-kind cache and still calls the current wrapper.
        assert calls["n"] == 2
        assert fired == ["wrapper", "wrapper"]

        def fake_registered_kernel(*args, **kwargs):
            return None

        register(_PAIR_KEY, fake_registered_kernel, replace=True)
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=1,
            out_a_ptr=2,
            out_b_ptr=3,
            rows=2,
            in_features=2048,
            out_features=8192,
            out_features_b=4096,
            runtime="rt",
        )
        assert calls["n"] == 4
        assert fired[-1] == "wrapper"
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = saved_wrapper  # type: ignore[assignment]
        if saved_kernel is None:
            _KERNELS.pop(_PAIR_KEY, None)
        else:
            register(_PAIR_KEY, saved_kernel, replace=True)
        gl.clear_gguf_linear_dispatch_cache()


def test_q8_t16_pair_threads_env_reaches_wrapper(monkeypatch) -> None:
    gl.clear_gguf_linear_dispatch_cache()
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_THREADS", "64")
    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    calls: list[dict] = []
    saved_kernel = resolve(
        backend=_PAIR_KEY.backend,
        layer=_PAIR_KEY.layer,
        quant=_PAIR_KEY.quant,
        variant=_PAIR_KEY.variant,
        missing="none",
    )
    saved_wrapper = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out

    def fake_registered_kernel(*args, **kwargs):
        return None

    def fake_wrapper(*args, **kwargs):
        calls.append(dict(kwargs))

    register(_PAIR_KEY, fake_registered_kernel, replace=True)
    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_wrapper  # type: ignore[assignment]
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=1,
            out_a_ptr=2,
            out_b_ptr=3,
            rows=2,
            in_features=2048,
            out_features=8192,
            out_features_b=4096,
            runtime="rt",
        )
        assert calls == [{"threads": 64, "stream": 0, "runtime": "rt"}]
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = saved_wrapper  # type: ignore[assignment]
        if saved_kernel is None:
            _KERNELS.pop(_PAIR_KEY, None)
        else:
            register(_PAIR_KEY, saved_kernel, replace=True)
        gl.clear_gguf_linear_dispatch_cache()


def test_q8_t16_threads_env_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_THREADS", "256")
    with pytest.raises(ValueError, match="HIPENGINE_GGUF_Q8_T16_THREADS must be one of 64 or 128"):
        gl._resolve_q8_t16_threads()
