"""Owner-gate tests for the fused IQ4_XS pair dispatch (2026-09-10 review).

The fused dual is only bit-exact with (and only authorized by) the admitted
cooperative prefill owner. The dispatch branch must therefore refuse to fuse
when either operand would resolve to a different execution owner - without a
dense-IQ session the singles keep the strict GEMV, and under a pinned
incumbent policy they keep the one-wave owner.
"""

from __future__ import annotations

import pytest

from hipengine.runtime import gguf_linear as gl
from hipengine.kernels.registry import KernelKey


class _FakeTensor:
    ptr = 0x10000


class _FakeAllocation:
    tensor = _FakeTensor()


class _FakeWeight:
    backend = "hip_gfx1100"

    def allocation(self, name):
        return _FakeAllocation()


@pytest.fixture()
def spy_resolve(monkeypatch):
    """Record every kernel the pair launcher resolves and launch nothing."""

    calls: list[KernelKey] = []
    real_resolve = gl.resolve

    def record(*, backend, layer, quant, variant):
        calls.append(KernelKey(backend, layer, quant, variant))
        return lambda *args, **kwargs: None

    monkeypatch.setattr(gl, "resolve", record)
    yield calls
    monkeypatch.setattr(gl, "resolve", real_resolve)


def _strict_dispatch():
    return gl.GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_iq4_xs",
                  "prefill_bf16_bf16_out"),
        "raw",
    )


def test_fused_pair_refuses_to_replace_the_strict_owner(spy_resolve, monkeypatch):
    """Without a dense-IQ session the pair must not fuse (review P1).

    The raw resolve names the strict single owner; the fused dual may not
    replace strict arithmetic regardless of quant, rows or dimensions.
    """

    monkeypatch.setattr(
        gl, "resolve_gguf_linear_dispatch",
        lambda weight, **kw: _strict_dispatch())
    launched = gl.launch_gguf_linear_pair_silu(
        _FakeWeight(), _FakeWeight(),
        0x20000, 0x30000, rows=512, in_features=5120, out_features=17408,
        backend="hip_gfx1100",
    )
    assert launched is False
    assert spy_resolve == []


def test_fused_pair_requires_the_admitted_cooperative_owner(spy_resolve, monkeypatch):
    """With the session and the shipped policy the pair fuses.

    The prefill rewrite resolves both operands to the admitted cooperative
    owner, which is the only path the dual is authorized to replace.
    """

    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq

    monkeypatch.setattr(
        gl, "resolve_gguf_linear_dispatch",
        lambda weight, **kw: _strict_dispatch())
    with iq_mmq.iq_dense_mmq_session(True):
        launched = gl.launch_gguf_linear_pair_silu(
            _FakeWeight(), _FakeWeight(),
            0x20000, 0x30000, rows=512, in_features=5120, out_features=17408,
            backend="hip_gfx1100",
        )
    assert launched is True
    assert spy_resolve == [
        KernelKey("hip_gfx1100", "linear_pair_silu", "gguf_iq4_xs",
                  "dense_iq_wmma_prefill_dual_silu_bf16_bf16_out"),
    ]


def test_fused_pair_refuses_a_pinned_incumbent_policy(spy_resolve, monkeypatch):
    """A pinned one-wave policy (the admission-gate incumbent arm) must not fuse.

    This is what keeps the combined-stack gate's incumbent arm on the
    two-singles path it declares.
    """

    import copy

    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq

    original = copy.deepcopy(be.GGUF_IQ_DENSE_PREFILL_POLICY)
    pinned = copy.deepcopy(original)
    pinned["gguf_iq4_xs"]["variant"] = "dense_wmma_w4a16_prefill_bf16_bf16_out"
    monkeypatch.setattr(
        gl, "resolve_gguf_linear_dispatch",
        lambda weight, **kw: _strict_dispatch())
    try:
        be.GGUF_IQ_DENSE_PREFILL_POLICY = pinned
        with iq_mmq.iq_dense_mmq_session(True):
            launched = gl.launch_gguf_linear_pair_silu(
                _FakeWeight(), _FakeWeight(),
                0x20000, 0x30000, rows=512, in_features=5120,
                out_features=17408, backend="hip_gfx1100",
            )
        assert launched is False
        assert spy_resolve == []
    finally:
        be.GGUF_IQ_DENSE_PREFILL_POLICY = original
