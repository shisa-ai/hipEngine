"""Indexed Qwen4Exp PLE dilated-conv decode: independent rows over history slots.

The primitive is the multi-request form of
:func:`qwen4_exp_ple_dilated_depthwise_conv_f32`: row ``r`` is one token of an
independent request, reads its own token row and its own output row, and
reads/writes exactly the PLE history slot ``state_indices[r]`` of a
``[state_slots][(kernel_size - 1) * dilation][channels]`` FP32 slab. Slots no row
selects keep their exact bytes.

Two contracts are asserted separately, per ``docs/EXECUTION-PROFILES.md``:

* **Ownership (exact in every profile).** Row -> slot mapping, untouched slots,
  uniform batch rejection for a malformed slot vector, and the absence of writes
  past the launched channel extent are byte-level assertions.
* **Arithmetic.** The block body repeats the single-row owner's statements in the
  same order, so the declared expectation is *bit-exact* equality with the
  single-row owner (T0, layout-only). That is asserted, not assumed: if it ever
  fails, the primitive is T2 and needs the production numerical gate before use.

A CPU oracle (the committed PLE reference with one-token position vectors) is
also compared at FP32 round-off, so the primitive is not its own oracle.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEConvState,
    dilated_depthwise_conv,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    QWEN4_EXP_PLE_INDEXED_MAX_DILATION,
    QWEN4_EXP_PLE_INDEXED_MAX_ROWS,
    QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS,
    build_qwen4_exp_ple,
    qwen4_exp_ple_dilated_depthwise_conv_f32,
    qwen4_exp_ple_dilated_depthwise_conv_indexed_f32,
    qwen4_exp_ple_indexed_state_indices,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve

_INDEXED_KEY = KernelKey(
    "hip_gfx1151", "ple_dilated_depthwise_conv", "f32",
    "qwen4exp_ple_indexed_decode",
)
_RAW_ARG_TYPES = (ctypes.c_void_p,) * 5 + (ctypes.c_int64,) * 5 + (ctypes.c_void_p,)
_OUT_SENTINEL = np.float32(-12345.0)
_PAD_SENTINEL = np.float32(-777.0)


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


# ---------------------------------------------------------------------------
# Host fixtures and device helpers.
# ---------------------------------------------------------------------------


def _fixture(rng, rows, state_slots, channels, kernel_size, dilation):
    token = rng.normal(0.0, 0.4, (rows, channels)).astype(np.float32)
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)
    history_rows = (kernel_size - 1) * dilation
    slab = rng.normal(0.0, 0.25, (state_slots, history_rows, channels)).astype(
        np.float32
    )
    return token, weight, slab


def _upload(array, runtime, allocations):
    """Upload a packed host copy, keeping it alive until the buffers are freed.

    The H2D copy can outlive the call that issued it, so a contiguous copy made
    inline would be freed before it is read. ``allocations`` holds
    ``(device, host)`` pairs for that reason. A zero-row history still gets a real
    pointer.
    """

    host = np.ascontiguousarray(array)
    device = malloc(max(int(host.nbytes), 4), runtime=runtime)
    allocations.append((device, host))
    if host.nbytes:
        copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(reference, device, runtime):
    """Copy into a fresh C-contiguous buffer.

    ``np.empty_like`` preserves a strided view's strides, which would scramble a
    packed device copy, so the target is built from shape and dtype instead.
    """

    out = np.empty(reference.shape, reference.dtype)
    if out.nbytes:
        copy_device_to_host(host_array_ptr(out), device, runtime=runtime)
    return out


def _run_single(library, runtime, token_row, weight, slot_history, *, dilation,
                out_width):
    """Run the single-row owner on one token from one slot's history."""

    channels, kernel_size = weight.shape
    allocations = []
    try:
        token_dev = _upload(token_row, runtime, allocations)
        weight_dev = _upload(weight, runtime, allocations)
        history_dev = _upload(slot_history, runtime, allocations)
        out_host = np.full(out_width, _PAD_SENTINEL, np.float32)
        out_dev = _upload(out_host, runtime, allocations)
        qwen4_exp_ple_dilated_depthwise_conv_f32(
            token_dev.ptr, weight_dev.ptr, history_dev.ptr, out_dev.ptr,
            1, channels, kernel_size, dilation, library=library, runtime=runtime)
        runtime.device_synchronize()
        out = _download(np.empty(out_width, np.float32), out_dev, runtime)
        final = _download(slot_history, history_dev, runtime)
        return out, final
    finally:
        for device, _host in allocations:
            free(device, runtime=runtime)


def _run_indexed(library, runtime, token, weight, slab, slots, *, dilation,
                 raw_slots=None, out_pad=0, out_sentinel=_OUT_SENTINEL):
    """Run the indexed owner once over packed rows and the whole history slab.

    Returns ``(out, final)`` where ``out`` has ``rows * channels + out_pad``
    entries so a caller can assert the tail past the last channel is untouched.
    ``raw_slots`` bypasses host admission so the device guard is measured.
    """

    channels, kernel_size = weight.shape
    indices = (
        np.ascontiguousarray(raw_slots, dtype=np.int64)
        if raw_slots is not None
        else qwen4_exp_ple_indexed_state_indices(
            slots, rows=len(slots), state_slots=slab.shape[0])
    )
    rows = int(indices.size)
    allocations = []
    try:
        token_dev = _upload(token, runtime, allocations)
        weight_dev = _upload(weight, runtime, allocations)
        history_dev = _upload(slab, runtime, allocations)
        index_dev = _upload(indices, runtime, allocations)
        out_host = np.full(rows * channels + out_pad, out_sentinel, np.float32)
        out_dev = _upload(out_host, runtime, allocations)
        qwen4_exp_ple_dilated_depthwise_conv_indexed_f32(
            token_dev.ptr, weight_dev.ptr, history_dev.ptr, out_dev.ptr,
            index_dev.ptr,
            rows=rows, state_slots=slab.shape[0], channels=channels,
            kernel_size=kernel_size, dilation=dilation,
            library=library, runtime=runtime)
        runtime.device_synchronize()
        out = _download(out_host, out_dev, runtime)
        final = _download(slab, history_dev, runtime)
        return out, final
    finally:
        for device, _host in allocations:
            free(device, runtime=runtime)


def _expected_serial(library, runtime, token, weight, slab, slots, *, dilation):
    """Per-row single-row owner reference over a copy of the slab."""

    channels = weight.shape[0]
    out = np.empty((len(slots), channels), np.float32)
    state = slab.copy()
    for row, slot in enumerate(slots):
        out[row], state[slot] = _run_single(
            library, runtime, token[row], weight, slab[slot], dilation=dilation,
            out_width=channels)
    return out, state


def _slab_width(slots, state_slots=None):
    """A slab wide enough to hold the selected slots plus a spare canary slot."""

    if state_slots is not None:
        return state_slots
    return max(slots) + 1 + (1 if len(slots) > 1 else 0)


# ---------------------------------------------------------------------------
# Ownership contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size,dilation", [(4, 3), (2, 1), (4, 1)])
@pytest.mark.parametrize("slots", [(0,), (3, 1), (2, 0, 5, 4)])
def test_indexed_decode_is_bit_exact_to_the_single_row_owner(
    slots, kernel_size, dilation
):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels = 320
    rng = np.random.default_rng(101 * len(slots) + kernel_size * dilation)
    token, weight, slab = _fixture(
        rng, len(slots), _slab_width(slots), channels, kernel_size, dilation)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=dilation)

    expected_out, expected_state = _expected_serial(
        library, runtime, token, weight, slab, slots, dilation=dilation)
    assert np.isfinite(out).all() and np.isfinite(final).all()
    np.testing.assert_array_equal(
        out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


@pytest.mark.parametrize("slots", [(3, 1), (5, 0, 4)])
def test_indexed_decode_leaves_unselected_slots_untouched(slots):
    """Ragged active subset: unselected slots keep their exact bytes."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels, state_slots = 256, 6
    rng = np.random.default_rng(7 * len(slots))
    token, weight, slab = _fixture(rng, len(slots), state_slots, channels, 4, 2)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=2)

    untouched = [slot for slot in range(state_slots) if slot not in slots]
    assert untouched
    for slot in untouched:
        np.testing.assert_array_equal(
            final[slot].view(np.uint32), slab[slot].view(np.uint32))
    for slot in slots:
        assert not np.array_equal(final[slot], slab[slot])
    assert not np.array_equal(out, np.zeros_like(out))


def test_indexed_decode_does_not_write_past_the_channel_extent():
    """Launch fewer channels than the buffers hold; the tail stays a sentinel."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels, out_width, slots = 300, 320, (0, 2)
    rng = np.random.default_rng(313)
    token, weight, slab = _fixture(rng, len(slots), 3, out_width, 4, 1)
    # Only the first ``channels`` columns are inside the launched extent.
    token, weight, slab = token[:, :channels], weight[:channels], slab[..., :channels]

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=1,
        out_pad=out_width - channels, out_sentinel=_PAD_SENTINEL)
    expected_out, expected_state = _expected_serial(
        library, runtime, token, weight, slab, slots, dilation=1)
    np.testing.assert_array_equal(
        out[:len(slots) * channels].view(np.uint32),
        expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        out[len(slots) * channels:],
        np.full(out_width - channels, _PAD_SENTINEL, np.float32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


@pytest.mark.parametrize("raw_slots", [(-1, 0), (0, 5), (5, 0)])
def test_indexed_decode_rejects_an_out_of_range_slot_as_a_whole_batch(raw_slots):
    """A malformed batch writes no output row and no history byte."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels, state_slots = 128, 4
    rng = np.random.default_rng(5_003)
    token, weight, slab = _fixture(rng, 2, state_slots, channels, 4, 2)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, None, dilation=2,
        raw_slots=raw_slots)

    assert (out == _OUT_SENTINEL).all()
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


@pytest.mark.parametrize("bad_row,bad_value,label", [
    (40, 0, "duplicate in the second scan pass"),
    (63, -1, "negative in the second scan pass"),
    (63, 64, "out of range in the second scan pass"),
])
def test_indexed_decode_rejects_a_malformed_row_beyond_the_first_scan_pass(
    bad_row, bad_value, label
):
    """64 rows make each warp lane scan twice; the second pass is checked."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    rows, channels = QWEN4_EXP_PLE_INDEXED_MAX_ROWS, 64
    rng = np.random.default_rng(6_400 + bad_row)
    token, weight, slab = _fixture(rng, rows, rows, channels, 2, 1)
    raw_slots = list(range(rows))
    raw_slots[bad_row] = bad_value

    out, final = _run_indexed(
        library, runtime, token, weight, slab, None, dilation=1,
        raw_slots=raw_slots)

    assert (out == _OUT_SENTINEL).all(), label
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_rejects_duplicate_slots_as_a_whole_batch():
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels, state_slots = 128, 4
    rng = np.random.default_rng(5_009)
    token, weight, slab = _fixture(rng, 3, state_slots, channels, 4, 2)

    first = _run_indexed(
        library, runtime, token, weight, slab, None, dilation=2,
        raw_slots=(2, 0, 2))
    second = _run_indexed(
        library, runtime, token, weight, slab, None, dilation=2,
        raw_slots=(2, 0, 2))

    for out, final in (first, second):
        assert (out == _OUT_SENTINEL).all()
        np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))
    np.testing.assert_array_equal(first[0].view(np.uint32),
                                  second[0].view(np.uint32))
    np.testing.assert_array_equal(first[1].view(np.uint32),
                                  second[1].view(np.uint32))


@pytest.mark.parametrize("kernel_size,dilation", [(4, 3), (2, 1)])
def test_indexed_decode_repeated_steps_follow_permuted_slots(
    kernel_size, dilation
):
    """Three steps with a rotating slot map, checked against the serial owner."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels, state_slots = 192, 5
    rng = np.random.default_rng(8_801 + kernel_size)
    _, weight, slab = _fixture(
        rng, 2, state_slots, channels, kernel_size, dilation)

    state = slab.copy()
    for slots in ((0, 4), (2, 0), (4, 2)):
        token = rng.normal(0.0, 0.4, (2, channels)).astype(np.float32)
        previous = state
        out, state = _run_indexed(
            library, runtime, token, weight, previous, slots, dilation=dilation)
        expected_out, expected_state = _expected_serial(
            library, runtime, token, weight, previous, slots, dilation=dilation)
        np.testing.assert_array_equal(
            out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
        np.testing.assert_array_equal(
            state.view(np.uint32), expected_state.view(np.uint32))
    for slot in (1, 3):
        np.testing.assert_array_equal(state[slot].view(np.uint32),
                                      slab[slot].view(np.uint32))


def test_indexed_decode_is_deterministic_across_reruns():
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    slots = (2, 0, 3)
    rng = np.random.default_rng(4_242)
    token, weight, slab = _fixture(rng, len(slots), 4, 256, 4, 3)

    runs = [
        _run_indexed(library, runtime, token, weight, slab, slots, dilation=3)
        for _ in range(3)
    ]
    for out, final in runs[1:]:
        np.testing.assert_array_equal(out.view(np.uint32),
                                      runs[0][0].view(np.uint32))
        np.testing.assert_array_equal(final.view(np.uint32),
                                      runs[0][1].view(np.uint32))


@pytest.mark.parametrize("kernel_size,dilation", [(4, 1), (2, 1)])
def test_indexed_decode_history_edge_cases(kernel_size, dilation):
    """``history_rows`` of 1 and 3 exercise both ends of the shift loop."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    slots = (1, 0)
    rng = np.random.default_rng(61 + kernel_size)
    token, weight, slab = _fixture(rng, 2, 2, 128, kernel_size, dilation)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=dilation)
    expected_out, expected_state = _expected_serial(
        library, runtime, token, weight, slab, slots, dilation=dilation)
    np.testing.assert_array_equal(
        out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))
    # The last history row is always the row's own token.
    np.testing.assert_array_equal(final[1, -1], token[0])
    np.testing.assert_array_equal(final[0, -1], token[1])


def test_indexed_decode_kernel_size_one_writes_no_history():
    """``history_rows == 0``: a stateless pass-through that touches no slot."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    slots = (2, 0)
    rng = np.random.default_rng(4_001)
    token, weight, slab = _fixture(rng, 2, 3, 96, 1, 1)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=1)

    assert final.shape == (3, 0, 96)
    np.testing.assert_array_equal(
        out.view(np.uint32), (token * weight[:, 0]).reshape(-1).view(np.uint32))


def test_indexed_decode_admits_the_maximum_row_count():
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    rows, channels = QWEN4_EXP_PLE_INDEXED_MAX_ROWS, 64
    slots = tuple(reversed(range(rows)))
    rng = np.random.default_rng(64)
    token, weight, slab = _fixture(rng, rows, rows, channels, 2, 1)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=1)
    expected_out, expected_state = _expected_serial(
        library, runtime, token, weight, slab, slots, dilation=1)
    np.testing.assert_array_equal(
        out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


# ---------------------------------------------------------------------------
# Independent oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size,dilation", [(4, 3), (2, 1)])
@pytest.mark.parametrize("slots", [(0,), (2, 0), (3, 1, 2, 0)])
def test_indexed_decode_matches_the_cpu_oracle_to_round_off(
    slots, kernel_size, dilation
):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    channels = 160
    rng = np.random.default_rng(9_001 + kernel_size * dilation)
    token, weight, slab = _fixture(
        rng, len(slots), _slab_width(slots), channels, kernel_size, dilation)

    out, final = _run_indexed(
        library, runtime, token, weight, slab, slots, dilation=dilation)

    expected_out = np.empty((len(slots), channels), np.float32)
    expected_state = slab.copy()
    for row, slot in enumerate(slots):
        step_out, step_state = dilated_depthwise_conv(
            token[row][None, :], weight, dilation=dilation,
            positions=[13], state=PLEConvState(slab[slot].copy(), 13))
        expected_out[row] = step_out[0]
        expected_state[slot] = step_state.history
    np.testing.assert_allclose(out.reshape(len(slots), channels), expected_out,
                               rtol=0.0, atol=2.0e-6)
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


def test_indexed_decode_registers_with_the_strict_fallback():
    """The variant resolves on the device backend and keeps the owner."""

    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package("hip_gfx1151")
    for backend in (_INDEXED_KEY.backend, "hip_gfx1100"):
        key = KernelKey(
            backend, _INDEXED_KEY.layer, _INDEXED_KEY.quant, _INDEXED_KEY.variant)
        assert is_registered(key)
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is qwen4_exp_ple_dilated_depthwise_conv_indexed_f32

    # The existing single-row F32 owner stays registered as the strict fallback.
    for backend in (_INDEXED_KEY.backend, "hip_gfx1100"):
        assert is_registered(
            KernelKey(backend, "ple_dilated_depthwise_conv", "f32", "strict")
        )


# ---------------------------------------------------------------------------
# Raw C entry point: the device-side admission is not dead code.
# ---------------------------------------------------------------------------


def _call_c_launcher(library, runtime, *, token, weight, slab, slots,
                     rows=None, state_slots=None, channels=None,
                     kernel_size=None, dilation=None, slot_offset=0):
    channels = channels if channels is not None else weight.shape[0]
    kernel_size = kernel_size if kernel_size is not None else weight.shape[1]
    dilation = dilation if dilation is not None else 1
    rows = rows if rows is not None else len(slots)
    state_slots = state_slots if state_slots is not None else slab.shape[0]
    indices = np.ascontiguousarray(slots, dtype=np.int64)
    allocations = []
    try:
        token_dev = _upload(token, runtime, allocations)
        weight_dev = _upload(weight, runtime, allocations)
        history_dev = _upload(slab, runtime, allocations)
        index_dev = _upload(indices, runtime, allocations)
        out_dev = _upload(np.zeros(max(rows * channels, 1), np.float32),
                          runtime, allocations)
        fn = getattr(
            library, "hipengine_qwen4_exp_ple_dilated_depthwise_conv_indexed_f32")
        fn.argtypes = list(_RAW_ARG_TYPES)
        fn.restype = ctypes.c_int
        return int(fn(
            token_dev.ptr, weight_dev.ptr, history_dev.ptr, out_dev.ptr,
            index_dev.ptr + slot_offset,
            rows, state_slots, channels, kernel_size, dilation, 0))
    finally:
        for device, _host in allocations:
            free(device, runtime=runtime)


@pytest.mark.parametrize("label,overrides", [
    ("zero rows", {"rows": 0}),
    ("row bound", {"rows": QWEN4_EXP_PLE_INDEXED_MAX_ROWS + 1}),
    ("state slab too small", {"state_slots": 1}),
    ("state slot bound", {"state_slots": QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS + 1}),
    ("zero channels", {"channels": 0}),
    ("zero kernel", {"kernel_size": 0}),
    ("zero dilation", {"dilation": 0}),
    ("dilation bound", {"dilation": QWEN4_EXP_PLE_INDEXED_MAX_DILATION + 1}),
    ("history rows bound", {"kernel_size": 1024, "dilation": 1024}),
    ("misaligned slot vector", {"slot_offset": 4}),
])
def test_c_launcher_rejects_invalid_geometry(label, overrides):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    rng = np.random.default_rng(71)
    token, weight, slab = _fixture(rng, 2, 4, 32, 2, 1)

    error = _call_c_launcher(
        library, runtime, token=token, weight=weight, slab=slab, slots=(0, 1),
        **overrides)
    assert error != 0, label


def test_c_launcher_accepts_a_valid_call():
    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    rng = np.random.default_rng(73)
    token, weight, slab = _fixture(rng, 2, 3, 32, 2, 2)

    assert _call_c_launcher(
        library, runtime, token=token, weight=weight, slab=slab, slots=(2, 0),
        dilation=2) == 0
