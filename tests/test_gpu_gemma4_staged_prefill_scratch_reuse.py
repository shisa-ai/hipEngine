"""Shared-scratch reuse for the staged prefill across shapes and trim arms.

The staged launcher owns its workspace through a caller-supplied
:class:`~hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.Gemma4AttentionScratch`.
That owner keeps one buffer per stream, grows it by doubling, and never shrinks
it, so a run that walks an increasing then a decreasing sequence of shapes must
reuse the peak allocation rather than reallocate per call. This file drives that
sequence through a single owner and checks three things the per-call parity file
does not:

* **Reuse.** The buffer ``buffer()`` hands back is the same object while the
  requested size fits, and a new allocation happens only when it does not -- so
  an increasing shape grows the owner and a decreasing shape reuses the peak.
* **Equivalence.** Every launch through the shared owner is byte-identical to
  the same launch through a fresh, per-call workspace, for both the trimmed
  (``window > 0``) and untrimmed (``window == 0``) arms. The reuse changes no
  result.
* **Lifecycle.** The owner is closed exactly once at the end; a second
  ``close()`` is a no-op and ``buffer()`` after close is refused.

Reached with an explicit file target; the default suite collects ``test_unit_*``
only. No device work happens unless the module is actually run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable"
)

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as strict  # noqa: E402
from hipengine.kernels.hip_gfx1100.gemma4 import (  # noqa: E402
    gemma4_attention_staged as staged,
)
from tests.test_gpu_gemma4_attention_staged import (  # noqa: E402
    _Device,
    _assert_bit_identical,
    _inputs,
    _mask,
)

# One sliding geometry (16q/8kv, head_dim 256), window 1024. The workspace grows
# as ``tokens * num_heads * keys``, so the sequence below exercises an
# increasing owner and a decreasing one.
_HEAD_DIM = 256
_NUM_HEADS = 16
_NUM_KV_HEADS = 8
_WINDOW = 1024

# (label, tokens, keys, row_offset). Increasing, then back down to the first
# shape so the decreasing half must reuse the peak allocation.
_SHAPES = [
    ("small", 4, 1027, 1023),
    ("mid", 32, 1055, 1023),
    ("large", 256, 1279, 1023),
    ("small_again", 4, 1027, 1023),
]


def _run_one(
    shape,
    *,
    dtype,
    mask,
    window,
    row_offset,
    library,
    owner,
    runtime,
    seed=17,
):
    """One staged launch through ``owner`` (or a fresh per-call workspace)."""

    tokens, keys, num_heads, num_kv_heads, head_dim = shape
    query, key, value = _inputs(
        tokens, keys, num_heads, num_kv_heads, head_dim, dtype=dtype, seed=seed
    )
    out = np.zeros_like(query)
    device = _Device()
    try:
        query_buffer = device.put(query)
        key_buffer = device.put(key)
        value_buffer = device.put(value)
        mask_buffer = device.put(mask)
        out_buffer = device.take(out)
        entry = (
            staged.gemma4_attention_staged_bf16
            if dtype == "bf16"
            else staged.gemma4_attention_staged_f32
        )
        plan = entry(
            query_buffer.ptr,
            key_buffer.ptr,
            value_buffer.ptr,
            mask_buffer.ptr,
            out_buffer.ptr,
            library=library,
            scratch=owner,
            runtime=runtime,
            tokens=tokens,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            window=window,
            row_offset=row_offset,
        )
        device.read(out_buffer, out)
        return out, plan
    finally:
        device.close()


def _arms():
    """The trimmed and untrimmed arms of the same sliding shape."""

    return (("trimmed", _WINDOW, _WINDOW - 1), ("untrimmed", 0, 0))


@pytest.fixture(scope="module")
def staged_library():
    return staged.build_gemma4_attention_staged(load=True)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
def test_the_shared_owner_reuses_its_peak_allocation_across_shapes(
    staged_library, dtype,
) -> None:
    """Increasing then decreasing shapes reuse one owner, byte-identically.

    Every launch goes through the same :class:`Gemma4AttentionScratch`. The
    shared-owner output must equal a fresh per-call workspace's output for both
    trim arms, and the owner's buffer must be the same object once it has grown
    to the peak -- a decreasing shape must not allocate again.
    """

    runtime = get_hip_runtime()
    owner = strict.Gemma4AttentionScratch()
    previous_buffer = None
    previous_owned = 0
    peak_buffer = None
    try:
        for label, tokens, keys, row_offset in _SHAPES:
            shape = (tokens, keys, _NUM_HEADS, _NUM_KV_HEADS, _HEAD_DIM)
            mask = _mask(tokens, keys, "sliding", _WINDOW, row_offset, seed=3)
            for arm, window, arm_offset in _arms():
                shared_out, plan = _run_one(
                    shape, dtype=dtype, mask=mask, window=window,
                    row_offset=arm_offset, library=staged_library, owner=owner,
                    runtime=runtime,
                )
                fresh_out, _ = _run_one(
                    shape, dtype=dtype, mask=mask, window=window,
                    row_offset=arm_offset, library=staged_library, owner=None,
                    runtime=runtime,
                )
                _assert_bit_identical(shared_out, fresh_out, dtype)

                buffer = owner.buffer(
                    plan.workspace_bytes, stream=0, runtime=runtime
                )
                owned = len(owner._owned)
                if previous_buffer is None:
                    assert owned == 1
                elif buffer is previous_buffer:
                    assert owned == previous_owned, (
                        f"{label}/{arm}: a reused buffer must not allocate"
                    )
                else:
                    # A superseded buffer is retired behind a stream event, so
                    # the owned count does not grow with it; the replacement is
                    # a larger allocation and the stream's only owned buffer.
                    assert buffer.nbytes > previous_buffer.nbytes, (
                        f"{label}/{arm}: a new buffer must come from a larger allocation"
                    )
                    assert owned == 1
                assert buffer.nbytes >= plan.workspace_bytes
                previous_buffer, previous_owned = buffer, owned
                if label == "large":
                    peak_buffer = buffer

        # The decreasing shape must reuse the peak allocation, not a smaller one.
        assert peak_buffer is not None
        small_buffer = owner.buffer(
            staged.staged_workspace_bytes(4, _NUM_HEADS, 1027),
            stream=0, runtime=runtime,
        )
        assert small_buffer is peak_buffer
        assert len(owner._owned) == previous_owned
    finally:
        owner.close()

    # Closed exactly once: the flag is set, the buffers are gone, and a second
    # close is a no-op rather than a double free.
    assert owner._closed
    assert owner._owned == []
    owner.close()
    assert owner._owned == []
    with pytest.raises(RuntimeError):
        owner.buffer(1024, stream=0, runtime=runtime)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
def test_the_trim_is_a_noop_on_the_shared_owner(staged_library, dtype) -> None:
    """Both trim arms agree through one owner, so reuse does not perturb the trim.

    The trimmed and untrimmed launches share the same workspace and the same
    sliding mask; on a valid sliding-causal mask the window trim drops only
    zero-weight columns, so the two arms are byte-identical.
    """

    runtime = get_hip_runtime()
    owner = strict.Gemma4AttentionScratch()
    try:
        tokens, keys, row_offset = 64, 1087, 1023
        shape = (tokens, keys, _NUM_HEADS, _NUM_KV_HEADS, _HEAD_DIM)
        mask = _mask(tokens, keys, "sliding", _WINDOW, row_offset, seed=11)
        trimmed_out, _ = _run_one(
            shape, dtype=dtype, mask=mask, window=_WINDOW, row_offset=row_offset,
            library=staged_library, owner=owner, runtime=runtime,
        )
        untrimmed_out, _ = _run_one(
            shape, dtype=dtype, mask=mask, window=0, row_offset=0,
            library=staged_library, owner=owner, runtime=runtime,
        )
        _assert_bit_identical(trimmed_out, untrimmed_out, dtype)
    finally:
        owner.close()
    assert owner._closed
