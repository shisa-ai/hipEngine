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
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_RAW_GGUF,
)
from hipengine.runtime.gguf_linear import launch_gguf_linear, launch_gguf_linear_pair

_KEY = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
_IQ_STRICT_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_iq4_xs", "gemv_bf16_bf16_out"
)
_IQ_LOCAL32_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_iq4_xs", "local32_gemv_bf16_bf16_out"
)
_PAIR_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_dual_gemv_decode_bf16_bf16_out"
)


def _fake_weight(*, layout: str, quant_key: str, slot_path: str | None = None):
    alloc = SimpleNamespace(tensor=SimpleNamespace(ptr=10))

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(
                layout=layout,
                quant_key=quant_key,
                slot_path=slot_path,
            )

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


def test_dispatch_cache_reuses_equivalent_iq_owner_contexts(monkeypatch) -> None:
    """Re-entering the same semantic owner must not invalidate every raw-IQ weight."""

    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_iq_source_mmq_prefill as iq_mmq,
    )

    gl.clear_gguf_linear_dispatch_cache()
    calls = {"n": 0}
    orig_resolve_dispatch = gl.resolve_gguf_linear_dispatch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig_resolve_dispatch(*args, **kwargs)

    monkeypatch.setattr(gl, "resolve_gguf_linear_dispatch", counting)
    weight = _fake_weight(
        layout=LAYOUT_RAW_GGUF,
        quant_key="gguf_iq4_xs",
        slot_path="layers.0.ffn_gate",
    )
    fired: list[str] = []
    saved_strict = resolve(
        backend=_IQ_STRICT_KEY.backend,
        layer=_IQ_STRICT_KEY.layer,
        quant=_IQ_STRICT_KEY.quant,
        variant=_IQ_STRICT_KEY.variant,
        missing="none",
    )
    saved_local32 = resolve(
        backend=_IQ_LOCAL32_KEY.backend,
        layer=_IQ_LOCAL32_KEY.layer,
        quant=_IQ_LOCAL32_KEY.quant,
        variant=_IQ_LOCAL32_KEY.variant,
        missing="none",
    )
    register(_IQ_STRICT_KEY, lambda *args, **kwargs: fired.append("strict"), replace=True)
    register(_IQ_LOCAL32_KEY, lambda *args, **kwargs: fired.append("local32"), replace=True)
    try:
        for _ in range(2):
            with iq_mmq.iq_dense_mmq_session(True):
                launch_gguf_linear(
                    weight,
                    x_ptr=1,
                    out_ptr=2,
                    rows=1,
                    in_features=5120,
                    out_features=17408,
                    runtime="rt",
                )
        assert fired == ["local32", "local32"]
        assert calls["n"] == 1
    finally:
        if saved_strict is None:
            _KERNELS.pop(_IQ_STRICT_KEY, None)
        else:
            register(_IQ_STRICT_KEY, saved_strict, replace=True)
        if saved_local32 is None:
            _KERNELS.pop(_IQ_LOCAL32_KEY, None)
        else:
            register(_IQ_LOCAL32_KEY, saved_local32, replace=True)
        gl.clear_gguf_linear_dispatch_cache()


def test_dispatch_cache_separates_iq_owner_and_decode_pin(monkeypatch) -> None:
    """A stable owner key must still keep disabled and slot-pinned routes distinct."""

    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_iq_source_mmq_prefill as iq_mmq,
    )

    gl.clear_gguf_linear_dispatch_cache()
    weight = _fake_weight(
        layout=LAYOUT_RAW_GGUF,
        quant_key="gguf_iq4_xs",
        slot_path="layers.0.ffn_gate",
    )
    fired: list[str] = []
    saved_strict = resolve(
        backend=_IQ_STRICT_KEY.backend,
        layer=_IQ_STRICT_KEY.layer,
        quant=_IQ_STRICT_KEY.quant,
        variant=_IQ_STRICT_KEY.variant,
        missing="none",
    )
    saved_local32 = resolve(
        backend=_IQ_LOCAL32_KEY.backend,
        layer=_IQ_LOCAL32_KEY.layer,
        quant=_IQ_LOCAL32_KEY.quant,
        variant=_IQ_LOCAL32_KEY.variant,
        missing="none",
    )
    register(_IQ_STRICT_KEY, lambda *args, **kwargs: fired.append("strict"), replace=True)
    register(_IQ_LOCAL32_KEY, lambda *args, **kwargs: fired.append("local32"), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=1,
            out_ptr=2,
            rows=1,
            in_features=5120,
            out_features=17408,
            runtime="rt",
        )
        with iq_mmq.iq_dense_mmq_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=1,
                out_ptr=2,
                rows=1,
                in_features=5120,
                out_features=17408,
                runtime="rt",
            )
        with iq_mmq.iq_dense_mmq_session(
            True,
            decode_strict_slots={"layers.0.ffn_gate"},
        ):
            launch_gguf_linear(
                weight,
                x_ptr=1,
                out_ptr=2,
                rows=1,
                in_features=5120,
                out_features=17408,
                runtime="rt",
            )
        assert fired == ["strict", "local32", "strict"]
    finally:
        if saved_strict is None:
            _KERNELS.pop(_IQ_STRICT_KEY, None)
        else:
            register(_IQ_STRICT_KEY, saved_strict, replace=True)
        if saved_local32 is None:
            _KERNELS.pop(_IQ_LOCAL32_KEY, None)
        else:
            register(_IQ_LOCAL32_KEY, saved_local32, replace=True)
        gl.clear_gguf_linear_dispatch_cache()


def test_iq_dispatch_cache_state_tracks_workspace_and_both_pin_sets() -> None:
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_iq_source_mmq_prefill as iq_mmq,
    )

    assert gl._iq_dense_dispatch_cache_state() is None
    with iq_mmq.iq_dense_mmq_session(
        True,
        strict_slots={"layers.0.ffn_up"},
        decode_strict_slots={"layers.1.ffn_down"},
    ):
        assert gl._iq_dense_dispatch_cache_state() == (
            False,
            frozenset({"layers.0.ffn_up"}),
            frozenset({"layers.1.ffn_down"}),
        )
    with iq_mmq.iq_dense_mmq_session(
        True,
        workspace_ptr=1 << 20,
        workspace_nbytes=1 << 24,
    ):
        assert gl._iq_dense_dispatch_cache_state() == (
            True,
            frozenset(),
            frozenset(),
        )


def test_prefill_f16_staging_works_after_bf16_dispatch_cache_hit(monkeypatch) -> None:
    """An env-on launch may reuse a dispatch cached by an env-off launch."""

    gl.clear_gguf_linear_dispatch_cache()
    key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    calls: list[str] = []
    saved = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    )

    register(key, lambda *args, **kwargs: None, replace=True)
    monkeypatch.setitem(
        gl._LAUNCH_ABI,
        "t16",
        lambda *args, **kwargs: calls.append("bf16"),
    )
    monkeypatch.setattr(
        gl,
        "_launch_prefill_f16_staged",
        lambda *args, **kwargs: calls.append("f16") or True,
    )
    try:
        monkeypatch.setenv(gl.PREFILL_F16_STAGING_ENV, "0")
        launch_gguf_linear(
            weight,
            x_ptr=1,
            out_ptr=2,
            rows=72,
            in_features=5_120,
            out_features=6_144,
            backend="hip_gfx1151",
            runtime="rt",
            use_wmma_prefill=True,
        )
        monkeypatch.setenv(gl.PREFILL_F16_STAGING_ENV, "1")
        launch_gguf_linear(
            weight,
            x_ptr=1,
            out_ptr=2,
            rows=72,
            in_features=5_120,
            out_features=6_144,
            backend="hip_gfx1151",
            runtime="rt",
            use_wmma_prefill=True,
        )
        assert calls == ["bf16", "f16"]
    finally:
        if saved is None:
            _KERNELS.pop(key, None)
        else:
            register(key, saved, replace=True)
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


def test_policy_reassignment_is_invisible_to_the_memo_until_it_is_cleared() -> None:
    """A policy table is an import-time constant, so it is not in the memo key.

    Reassigning ``GGUF_IQ_DENSE_DECODE_POLICY`` changes a dispatch-resolution
    input, but the memo keys on the registry generation and the dense-IQ
    session state rather than on the table. An already-memoized
    (weight, shape, rows, slot) therefore keeps its old owner. That is
    deliberate - the table cannot change in production - and it is why every
    gate that flips a policy between arms must drop the memo: without the
    clear, the second arm is a cache hit and the two arms measure the same
    kernels, which reads as a bit-exact route rather than as a broken probe.

    This pins both halves of the contract, so a future change to the key that
    silently makes one of them false fails here rather than in a gate result.
    """

    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_iq_source_mmq_prefill as iq_mmq,
    )
    import hipengine.kernels.hip_gfx1100 as backend

    gl.clear_gguf_linear_dispatch_cache()
    weight = _fake_weight(
        layout=LAYOUT_RAW_GGUF,
        quant_key="gguf_iq4_xs",
        slot_path="layers.3.ffn_gate",
    )
    fired: list[str] = []
    saved_strict = resolve(
        backend=_IQ_STRICT_KEY.backend, layer=_IQ_STRICT_KEY.layer,
        quant=_IQ_STRICT_KEY.quant, variant=_IQ_STRICT_KEY.variant,
        missing="none",
    )
    saved_local32 = resolve(
        backend=_IQ_LOCAL32_KEY.backend, layer=_IQ_LOCAL32_KEY.layer,
        quant=_IQ_LOCAL32_KEY.quant, variant=_IQ_LOCAL32_KEY.variant,
        missing="none",
    )
    register(_IQ_STRICT_KEY, lambda *a, **k: fired.append("strict"), replace=True)
    register(_IQ_LOCAL32_KEY, lambda *a, **k: fired.append("local32"), replace=True)
    shipped = backend.GGUF_IQ_DENSE_DECODE_POLICY
    try:
        def launch() -> None:
            with iq_mmq.iq_dense_mmq_session(True):
                launch_gguf_linear(
                    weight, x_ptr=1, out_ptr=2, rows=1,
                    in_features=5120, out_features=17408, runtime="rt",
                )

        backend.GGUF_IQ_DENSE_DECODE_POLICY = {}
        launch()
        backend.GGUF_IQ_DENSE_DECODE_POLICY = {
            "gguf_iq4_xs": {"variant": "local32_gemv_bf16_bf16_out"}}
        launch()
        assert fired == ["strict", "strict"], (
            "the memo must not observe a policy reassignment; a memo hit here "
            "is the documented contract, and a miss means the key changed"
        )

        gl.clear_gguf_linear_dispatch_cache()
        launch()
        assert fired == ["strict", "strict", "local32"], (
            "clearing the memo must be sufficient to pick up the new policy"
        )
    finally:
        backend.GGUF_IQ_DENSE_DECODE_POLICY = shipped
        if saved_strict is None:
            _KERNELS.pop(_IQ_STRICT_KEY, None)
        else:
            register(_IQ_STRICT_KEY, saved_strict, replace=True)
        if saved_local32 is None:
            _KERNELS.pop(_IQ_LOCAL32_KEY, None)
        else:
            register(_IQ_LOCAL32_KEY, saved_local32, replace=True)
        gl.clear_gguf_linear_dispatch_cache()
