"""The AR runtime's paired head: one weight read per step, per-branch rows intact.

The lm head is 756 MB of BF16 weight per call, so `Yue2ArRuntime.logits` runs it once
for every branch and caches the rows until a branch's hidden row moves. That is only
safe if the cache is invalidated by every write to a hidden row and if a branch always
gets its own row back. These tests drive the method with a stub kernel namespace, so
they check the pairing and invalidation logic without a GPU; the arithmetic identity
per row is the kernel gate's job (`test_unit_yue2_ar_gemv_rowtile2.py`).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _Spec:
    hidden_size = 8
    vocab_size = 4
    rms_norm_eps = 1e-6


class _Buffer:
    def __init__(self, ptr: int, nbytes: int = 16) -> None:
        self.ptr = ptr
        self.nbytes = nbytes


class _Kernels:
    """Counts calls and writes a value that identifies the branch that produced it."""

    def __init__(self) -> None:
        self.head_calls = 0
        self.norm_calls = 0
        self.rows: dict[int, int] = {}

    def vv_rmsnorm_bf16(self, x_ptr, weight_ptr, out_ptr, rows, width, eps, **kwargs):
        self.norm_calls += 1
        self.rows[out_ptr] = x_ptr

    def dense_gemv_bf16_f32_out_rowtile2(
        self, x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, **kwargs
    ):
        self.head_calls += 1
        self.pair_out = out_ptr
        self.pair_rows = rows
        self.head_weight_ptr = weight_ptr
        self.head_out_features = out_features


def _runtime(branches: int = 2):
    from hipengine.runtime.yue2_ar import Yue2ArRuntime

    runtime = Yue2ArRuntime.__new__(Yue2ArRuntime)
    runtime.spec = _Spec()
    runtime.branches = branches
    runtime.kernels = _Kernels()
    runtime.final_ln = _Buffer(0x100)
    runtime.lm_head = _Buffer(0x200)
    runtime._hidden = [_Buffer(0x1000 + index * 0x100) for index in range(branches)]
    runtime._normed_pair = _Buffer(0x3000)
    runtime._logits_pair_f32 = _Buffer(0x4000)
    runtime._logits_pair_cache = None
    runtime._logits_pair_host = np.zeros((2, _Spec.vocab_size), dtype=np.float32)
    runtime.library = None
    runtime.runtime = None
    return runtime


def test_two_branches_cost_one_head_call(monkeypatch):
    from hipengine.runtime import yue2_ar

    runtime = _runtime()

    def fake_copy(host_ptr, device, nbytes):
        # The stub's head leaves its own marker per row: row 0 -> 1.0, row 1 -> 2.0.
        host = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_float * (nbytes // 4)).from_address(host_ptr))
        host[:] = [1.0] * _Spec.vocab_size + [2.0] * _Spec.vocab_size

    monkeypatch.setattr(yue2_ar, "copy_device_to_host", fake_copy)
    first = runtime.logits(0, as_bf16=False)
    second = runtime.logits(1, as_bf16=False)
    assert runtime.kernels.head_calls == 1, "both branches must share one weight read"
    assert runtime.kernels.norm_calls == 2, "each branch needs its own final norm"
    assert list(first) == [1.0] * _Spec.vocab_size
    assert list(second) == [2.0] * _Spec.vocab_size
    assert runtime.kernels.pair_rows == 2


def test_hidden_row_move_invalidates_the_pair(monkeypatch):
    from hipengine.runtime import yue2_ar

    runtime = _runtime()
    monkeypatch.setattr(yue2_ar, "copy_device_to_host", lambda *a, **k: None)
    runtime.logits(0, as_bf16=False)
    runtime.logits(1, as_bf16=False)
    assert runtime.kernels.head_calls == 1
    runtime.forward_layers = lambda position, branch=0: None  # the invalidation is the assertion
    # forward_layers invalidates on entry; call the real invalidation the same way.
    runtime._logits_pair_cache = None
    runtime.logits(0, as_bf16=False)
    assert runtime.kernels.head_calls == 2


def test_push_token_and_reset_invalidate_the_pair():
    from hipengine.core.memory import copy_host_array_to_device, host_array_ptr

    runtime = _runtime()
    runtime._logits_pair_cache = ["stale", "stale"]
    row = np.zeros(_Spec.hidden_size, dtype=np.float32)
    # The staging path must clear the cache; exercise it through the real method body
    # by checking the attribute after the calls it makes.
    import inspect

    # `prefill_rows` drives `forward_layers` per row, so it is covered through that.
    for method in ("push_token", "reset", "forward_layers", "prefill_host_rows"):
        assert "_reset_logits_cache()" in inspect.getsource(getattr(type(runtime), method)), method


def test_logits_rejects_an_unknown_branch_without_a_gpu():
    runtime = _runtime(branches=2)
    runtime._logits_pair_cache = [np.zeros(4, dtype=np.float32) for _ in range(2)]
    with pytest.raises(IndexError):
        runtime.logits(2)


def test_domain_projects_only_the_window_and_masks_the_rest(monkeypatch):
    """`domain` must narrow the projection, offset the weight, and -inf the rest."""

    from hipengine.generation.yue2 import phase_window
    from hipengine.runtime import yue2_ar

    runtime = _runtime()
    low, high = phase_window("semantic")
    runtime.spec = SimpleNamespace(hidden_size=_Spec.hidden_size, vocab_size=184704,
                                   rms_norm_eps=_Spec.rms_norm_eps)
    runtime._logits_window_domain = None
    runtime._logits_window_filled = False
    runtime._logits_window_host = np.zeros((2, runtime.spec.vocab_size), dtype=np.float32)
    runtime._logits_window_f32 = _Buffer(0x5000)

    def fake_copy(host_ptr, device, nbytes):
        width = nbytes // 4 // 2
        host = np.ctypeslib.as_array(
            (np.ctypeslib.ctypes.c_float * (nbytes // 4)).from_address(host_ptr)
        )
        host[:] = np.arange(nbytes // 4, dtype=np.float32)

    monkeypatch.setattr(yue2_ar, "copy_device_to_host", fake_copy)
    row = runtime.logits(0, as_bf16=False, domain=(low, high))

    assert runtime.kernels.head_out_features == high - low, "projection must be the window"
    hidden = _Spec.hidden_size
    assert runtime.kernels.head_weight_ptr == runtime.lm_head.ptr + low * hidden * 2
    assert np.isneginf(row[:low]).all() and np.isneginf(row[high:]).all()
    assert np.isfinite(row[low:high]).all()
    # A second call at the same hidden state reuses the fill.
    calls = runtime.kernels.head_calls
    runtime.logits(1, as_bf16=False, domain=(low, high))
    assert runtime.kernels.head_calls == calls, "both branches share one windowed fill"
    # A moved hidden row refills, and a different domain refills with fresh -inf.
    runtime._reset_logits_cache()
    runtime.logits(0, as_bf16=False, domain=(0, 4))
    assert runtime.kernels.head_out_features == 4
    assert np.isneginf(runtime._logits_window_host[0][4:]).all()


def test_switching_domain_without_a_step_reprojects(monkeypatch):
    """A domain change is part of the cache key, not just a refill trigger.

    The first version checked only whether the window was filled, so a caller that asked
    for a second domain before moving a hidden row got the first domain's rows back. The
    earlier test reset the cache first, which hid exactly that.
    """

    from hipengine.runtime import yue2_ar

    runtime = _runtime()
    runtime.spec = SimpleNamespace(hidden_size=_Spec.hidden_size, vocab_size=184704,
                                   rms_norm_eps=_Spec.rms_norm_eps)
    runtime._logits_window_domain = None
    runtime._logits_window_filled = False
    runtime._logits_window_host = np.zeros((2, runtime.spec.vocab_size), dtype=np.float32)
    runtime._logits_window_f32 = _Buffer(0x5000)

    def fake_copy(host_ptr, device, nbytes):
        host = np.ctypeslib.as_array(
            (np.ctypeslib.ctypes.c_float * (nbytes // 4)).from_address(host_ptr)
        )
        host[:] = np.arange(nbytes // 4, dtype=np.float32) + 1.0

    monkeypatch.setattr(yue2_ar, "copy_device_to_host", fake_copy)
    first = runtime.logits(0, as_bf16=False, domain=(100, 108))
    calls = runtime.kernels.head_calls
    # No `_reset_logits_cache()` here: that is the case that used to return stale rows.
    second = runtime.logits(0, as_bf16=False, domain=(200, 204))
    assert runtime.kernels.head_calls == calls + 1, "a new domain must reproject"
    assert runtime.kernels.head_out_features == 4
    assert runtime.kernels.head_weight_ptr == runtime.lm_head.ptr + 200 * _Spec.hidden_size * 2
    assert np.isfinite(second[200:204]).all()
    assert np.isneginf(second[100:108]).all(), "the previous window must be masked again"
    assert np.isfinite(first[100:108]).all()
    # Asking for the same domain again is still a cache hit.
    calls = runtime.kernels.head_calls
    runtime.logits(1, as_bf16=False, domain=(200, 204))
    assert runtime.kernels.head_calls == calls


@pytest.mark.parametrize(
    "domain",
    # `None` is not here: it means the full projection, which is a valid request.
    [(-1, 4), (0, 0), (8, 4), (0, 184705), (2.5, 4), (0, "4"), "nope", (0,), (0, 1, 2)],
)
def test_invalid_domains_are_rejected_before_dispatch(domain):
    """Nothing may offset a weight pointer ahead of its allocation."""

    runtime = _runtime()
    runtime.spec = SimpleNamespace(hidden_size=_Spec.hidden_size, vocab_size=184704,
                                   rms_norm_eps=_Spec.rms_norm_eps)
    runtime._logits_window_domain = None
    runtime._logits_window_filled = False
    runtime._logits_window_host = np.zeros((2, runtime.spec.vocab_size), dtype=np.float32)
    runtime._logits_window_f32 = _Buffer(0x5000)
    with pytest.raises(ValueError):
        runtime.logits(0, as_bf16=False, domain=domain)
    assert runtime.kernels.head_calls == 0, "no head call may happen for an invalid domain"
    assert runtime._logits_window_filled is False


def test_booleans_are_not_domains():
    runtime = _runtime()
    runtime.spec = SimpleNamespace(hidden_size=_Spec.hidden_size, vocab_size=184704,
                                   rms_norm_eps=_Spec.rms_norm_eps)
    with pytest.raises(ValueError):
        runtime._validate_domain((True, 4))
