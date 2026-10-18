"""Indexed Qwen4Exp F32 conv decode: independent rows over per-row state slots.

The primitive is the multi-row form of
:func:`qwen35_linear_attn_conv_decode_f32`: row ``r`` reads its own hidden row
and its own output row, and reads/writes exactly the Conv state slot
``state_indices[r]``. Every row therefore advances its own request state, and
slots no row selects keep their exact bytes.

Two contracts are asserted separately, per ``docs/EXECUTION-PROFILES.md``:

* **Ownership (exact in every profile).** Row -> slot mapping, untouched slots,
  uniform batch rejection for malformed slot vectors, determinism, and the
  absence of writes past the launched channel extent are byte-level assertions.
* **Arithmetic.** The block body repeats the single-row owner's statements in
  the same order, so the declared expectation is *bit-exact* equality with the
  single-row owner (T0, layout-only). That is asserted, not assumed: if it ever
  fails, the primitive is T2 and needs the production numerical gate before use.

A CPU oracle (the committed segmented convolution reference with one-token
segments) is also compared at FP32 round-off, so the primitive is not its own
oracle.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.cpu_reference.ops import linear_attn_conv_prefill_segments
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS,
    QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE,
    QWEN4_EXP_CONV_INDEXED_MAX_ROWS,
    QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
    build_qwen35_linear_attn_conv,
    qwen35_linear_attn_conv_decode_f32,
    qwen4_exp_conv_indexed_state_indices,
    qwen4_exp_linear_attn_conv_decode_indexed_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve

_INDEXED_KEY = KernelKey(
    "hip_gfx1151", "linear_attn_conv_decode", "f32_state",
    "qwen4exp_conv_indexed_decode",
)
_RAW_ARG_TYPES = (ctypes.c_void_p,) * 5 + (ctypes.c_int64,) * 4 + (ctypes.c_void_p,)
_SENTINEL = np.float32(-12345.0)
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


def _fixture(rng, rows, state_slots, channels, kernel_size):
    hidden = rng.normal(0.0, 0.4, (rows, channels)).astype(np.float32)
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)
    state = rng.normal(0.0, 0.25, (state_slots, channels, kernel_size)).astype(
        np.float32
    )
    return hidden, weight, state


def _run_single(library, runtime, hidden, weight, state0, *, row, out_width):
    """Run the single-row owner on ``row`` from ``state0``."""

    hidden_dev = malloc(hidden[row].nbytes, runtime=runtime)
    weight_dev = malloc(weight.nbytes, runtime=runtime)
    state_dev = malloc(state0.nbytes, runtime=runtime)
    out_dev = malloc(out_width * 4, runtime=runtime)
    try:
        copy_host_to_device(
            hidden_dev, host_array_ptr(np.ascontiguousarray(hidden[row])),
            runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight), runtime=runtime)
        copy_host_to_device(state_dev, host_array_ptr(state0), runtime=runtime)
        qwen35_linear_attn_conv_decode_f32(
            hidden_dev.ptr, state_dev.ptr, weight_dev.ptr, out_dev.ptr,
            weight.shape[0], weight.shape[1], library=library, runtime=runtime)
        runtime.device_synchronize()
        out = np.empty(out_width, np.float32)
        final = np.empty_like(state0)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state_dev, runtime=runtime)
        return out, final
    finally:
        for buffer in (hidden_dev, weight_dev, state_dev, out_dev):
            free(buffer, runtime=runtime)


def _run_indexed(
    library, runtime, hidden, weight, slab, slots, *, raw_slots=None,
    out_pad=0, out_sentinel=_PAD_SENTINEL,
):
    """Run the indexed owner once over packed rows and the whole state slab.

    Returns ``(out, final)`` where ``out`` has ``rows * channels + out_pad``
    entries so a caller can assert the tail past the last channel is untouched.
    ``raw_slots`` bypasses host admission so the device guard is measured.
    """

    channels, kernel_size = weight.shape
    indices = (
        np.ascontiguousarray(raw_slots, dtype=np.int64)
        if raw_slots is not None
        else qwen4_exp_conv_indexed_state_indices(
            slots, rows=len(slots), state_slots=slab.shape[0])
    )
    rows = int(indices.size)
    hidden_dev = malloc(hidden.nbytes, runtime=runtime)
    weight_dev = malloc(weight.nbytes, runtime=runtime)
    state_dev = malloc(slab.nbytes, runtime=runtime)
    index_dev = malloc(indices.nbytes, runtime=runtime)
    out_host = np.full(rows * channels + out_pad, out_sentinel, np.float32)
    out_dev = malloc(out_host.nbytes, runtime=runtime)
    try:
        copy_host_to_device(
            hidden_dev, host_array_ptr(np.ascontiguousarray(hidden)),
            runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight), runtime=runtime)
        copy_host_to_device(state_dev, host_array_ptr(slab), runtime=runtime)
        copy_host_to_device(index_dev, host_array_ptr(indices), runtime=runtime)
        copy_host_to_device(out_dev, host_array_ptr(out_host), runtime=runtime)
        qwen4_exp_linear_attn_conv_decode_indexed_f32(
            hidden_dev.ptr, state_dev.ptr, weight_dev.ptr, out_dev.ptr,
            index_dev.ptr,
            rows=rows, state_slots=slab.shape[0], channels=channels,
            kernel_size=kernel_size, library=library, runtime=runtime)
        runtime.device_synchronize()
        out = np.empty_like(out_host)
        final = np.empty_like(slab)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state_dev, runtime=runtime)
        return out, final
    finally:
        for buffer in (hidden_dev, weight_dev, state_dev, index_dev, out_dev):
            free(buffer, runtime=runtime)


def _expected_serial(library, runtime, hidden, weight, slab, slots):
    """Per-row single-row owner reference over a copy of the slab."""

    channels = weight.shape[0]
    out = np.empty((len(slots), channels), np.float32)
    state = slab.copy()
    for row, slot in enumerate(slots):
        out[row], state[slot] = _run_single(
            library, runtime, hidden, weight, slab[slot], row=row,
            out_width=channels)
    return out, state


# ---------------------------------------------------------------------------
# Ownership contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [4, 2])
@pytest.mark.parametrize("slots", [(0,), (3, 1), (2, 0, 5, 4)])
def test_indexed_decode_is_bit_exact_to_the_single_row_owner(slots, kernel_size):
    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    count, channels = len(slots), 320
    state_slots = max(slots) + 1 + (1 if count > 1 else 0)
    rng = np.random.default_rng(101 * count + kernel_size)
    hidden, weight, slab = _fixture(rng, count, state_slots, channels, kernel_size)

    out, final = _run_indexed(
        library, runtime, hidden, weight, slab, slots)

    expected_out, expected_state = _expected_serial(
        library, runtime, hidden, weight, slab, slots)
    assert np.isfinite(out).all() and np.isfinite(final).all()
    np.testing.assert_array_equal(
        out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


@pytest.mark.parametrize("slots", [(3, 1), (5, 0, 4)])
def test_indexed_decode_leaves_unselected_slots_untouched(slots):
    """Ragged active subset: unselected slots keep their exact bytes."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 6, 512, 4
    rng = np.random.default_rng(7 * state_slots + len(slots))
    hidden, weight, slab = _fixture(
        rng, len(slots), state_slots, channels, kernel_size)

    _, final = _run_indexed(library, runtime, hidden, weight, slab, slots)

    expected_out, expected_state = _expected_serial(
        library, runtime, hidden, weight, slab, slots)
    assert expected_out.shape == (len(slots), channels)
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))
    for slot in range(state_slots):
        if slot in slots:
            continue
        np.testing.assert_array_equal(
            final[slot].view(np.uint32), slab[slot].view(np.uint32))


def test_indexed_decode_does_not_write_past_the_channel_extent():
    """The last partial block must not touch memory past ``rows * channels``.

    ``channels`` is not a multiple of the 256-thread block, so the final block
    has inactive threads; a missing bounds check would write into the sentinel
    tail (and, on a real call, into the next row's first channels).
    """

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 4, 300, 4
    rng = np.random.default_rng(300)
    hidden, weight, slab = _fixture(rng, 3, state_slots, channels, kernel_size)
    slots = (2, 0, 1)

    out, _ = _run_indexed(
        library, runtime, hidden, weight, slab, slots, out_pad=64)

    body = out[: len(slots) * channels]
    tail = out[len(slots) * channels :]
    assert np.isfinite(body).all()
    np.testing.assert_array_equal(
        tail, np.full(64, _PAD_SENTINEL, np.float32))
    # Row 2 starts exactly at channels * 2; a row-stride error would show up as
    # row 1's tail bleeding into it.
    expected_out, _ = _expected_serial(
        library, runtime, hidden, weight, slab, slots)
    np.testing.assert_array_equal(
        body.reshape(-1, channels).view(np.uint32),
        expected_out.view(np.uint32))


def test_indexed_decode_rejects_an_out_of_range_slot_as_a_whole_batch():
    """One bad slot rejects the batch: no block writes state or output."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 4, 256, 4
    rng = np.random.default_rng(555)
    hidden, weight, slab = _fixture(rng, 2, state_slots, channels, kernel_size)

    out, final = _run_indexed(
        library, runtime, hidden, weight, slab, None,
        raw_slots=np.array([2, state_slots], dtype=np.int64))

    np.testing.assert_array_equal(
        out, np.full(2 * channels, _PAD_SENTINEL, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_rejects_duplicate_slots_as_a_whole_batch():
    """Two rows naming one slot is an ownership violation, not a race.

    The decode body shifts the slot in place, so both blocks would interleave
    their shifts. The batch is rejected instead: every slot and every output row
    keeps its exact bytes.
    """

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 4, 256, 4
    rng = np.random.default_rng(999)
    hidden, weight, slab = _fixture(rng, 2, state_slots, channels, kernel_size)

    out, final = _run_indexed(
        library, runtime, hidden, weight, slab, None,
        raw_slots=np.array([1, 1], dtype=np.int64))

    np.testing.assert_array_equal(
        out, np.full(2 * channels, _PAD_SENTINEL, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_duplicate_rejection_is_deterministic():
    """Repeated runs of a malformed batch produce identical (null) effects."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 5, 128, 4
    rng = np.random.default_rng(1234)
    hidden, weight, slab = _fixture(rng, 3, state_slots, channels, kernel_size)
    bad = np.array([4, 0, 4], dtype=np.int64)

    first_out, first_state = _run_indexed(
        library, runtime, hidden, weight, slab, None, raw_slots=bad)
    second_out, second_state = _run_indexed(
        library, runtime, hidden, weight, slab, None, raw_slots=bad)
    np.testing.assert_array_equal(
        first_out.view(np.uint32), second_out.view(np.uint32))
    np.testing.assert_array_equal(
        first_state.view(np.uint32), second_state.view(np.uint32))
    np.testing.assert_array_equal(first_state.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_rejects_a_duplicate_beyond_the_first_scan_pass():
    """A duplicate only the scan loop's second pass can see still rejects."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    limit = QWEN4_EXP_CONV_INDEXED_MAX_ROWS
    channels, kernel_size = 64, 4
    slots = np.arange(limit, dtype=np.int64)
    slots[40] = slots[0]
    rng = np.random.default_rng(4040)
    hidden, weight, slab = _fixture(rng, limit, limit, channels, kernel_size)

    out, final = _run_indexed(
        library, runtime, hidden, weight, slab, None, raw_slots=slots)

    np.testing.assert_array_equal(
        out, np.full(limit * channels, _PAD_SENTINEL, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_admits_the_maximum_row_count():
    """64 rows exercise the scan loop's second pass (``lane + 32``)."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    limit = QWEN4_EXP_CONV_INDEXED_MAX_ROWS
    channels, kernel_size = 64, 4
    slots = tuple(int(v) for v in np.random.default_rng(64).permutation(limit))
    rng = np.random.default_rng(6464)
    hidden, weight, slab = _fixture(rng, limit, limit, channels, kernel_size)

    out, final = _run_indexed(library, runtime, hidden, weight, slab, slots)

    expected_out, expected_state = _expected_serial(
        library, runtime, hidden, weight, slab, slots)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(
        out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


def test_indexed_decode_repeated_steps_follow_permuted_slots():
    """Three steps with a changing composition keep per-slot state separate."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 4, 256, 4
    steps = [(1, 0), (3, 1), (0, 2)]
    rng = np.random.default_rng(31337)
    _, weight, initial = _fixture(rng, 1, state_slots, channels, kernel_size)
    step_hidden = [
        rng.normal(0.0, 0.4, (len(slots), channels)).astype(np.float32)
        for slots in steps
    ]

    batched_state = initial.copy()
    reference_state = initial.copy()
    for hidden, slots in zip(step_hidden, steps):
        batched_out, batched_state = _run_indexed(
            library, runtime, hidden, weight, batched_state, slots)
        reference_out = np.empty((len(slots), channels), np.float32)
        for row, slot in enumerate(slots):
            reference_out[row], reference_state[slot] = _run_single(
                library, runtime, hidden, weight, reference_state[slot],
                row=row, out_width=channels)
        np.testing.assert_array_equal(
            batched_out.view(np.uint32), reference_out.reshape(-1).view(np.uint32))
        np.testing.assert_array_equal(
            batched_state.view(np.uint32), reference_state.view(np.uint32))


def test_indexed_decode_is_deterministic_across_reruns():
    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    state_slots, channels, kernel_size = 4, 512, 4
    rng = np.random.default_rng(606)
    hidden, weight, slab = _fixture(rng, 3, state_slots, channels, kernel_size)
    slots = (2, 0, 1)

    first_out, first_state = _run_indexed(
        library, runtime, hidden, weight, slab, slots)
    second_out, second_state = _run_indexed(
        library, runtime, hidden, weight, slab, slots)
    np.testing.assert_array_equal(
        first_out.view(np.uint32), second_out.view(np.uint32))
    np.testing.assert_array_equal(
        first_state.view(np.uint32), second_state.view(np.uint32))


# ---------------------------------------------------------------------------
# Arithmetic contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slots", [(0,), (2, 0), (3, 1, 2, 0)])
def test_indexed_decode_matches_the_cpu_oracle_to_round_off(slots):
    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    channels, kernel_size = 256, 4
    count = len(slots)
    rng = np.random.default_rng(8080 + count)
    hidden, weight, slab = _fixture(
        rng, count, max(slots) + 1, channels, kernel_size)

    out, final = _run_indexed(library, runtime, hidden, weight, slab, slots)
    expected_out, expected_state = linear_attn_conv_prefill_segments(
        hidden, slab, weight, np.arange(count + 1, dtype=np.int64),
        np.asarray(slots, dtype=np.int64))

    assert np.isfinite(out).all() and np.isfinite(final).all()
    # Measured on zbook/gfx1151 at N=1/2/4: output max abs deviation is at FP32
    # round-off. The gate keeps headroom over the measurement rather than being
    # tuned to it, matching the sibling GDN primitive's round-off gate.
    reference = expected_out.reshape(-1)
    assert np.linalg.norm(out - reference) <= 2.0e-6 * np.linalg.norm(reference)
    assert np.abs(out - reference).max() <= 1.0e-5 * max(
        1.0, np.abs(reference).max())
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


# ---------------------------------------------------------------------------
# Registry and the raw C entry point.
# ---------------------------------------------------------------------------


def test_indexed_decode_registers_with_the_single_row_fallback():
    """The variant resolves on the device backend and keeps the fallback."""

    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package("hip_gfx1151")
    for backend in (_INDEXED_KEY.backend, "hip_gfx1100"):
        key = KernelKey(
            backend, _INDEXED_KEY.layer, _INDEXED_KEY.quant, _INDEXED_KEY.variant
        )
        assert is_registered(key)
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is qwen4_exp_linear_attn_conv_decode_indexed_f32

    # The existing single-row F32 owner stays registered as the strict fallback.
    for backend in (_INDEXED_KEY.backend, "hip_gfx1100"):
        assert is_registered(
            KernelKey(backend, "linear_attn_conv_decode", "w4_paro", "f32")
        )


def _call_c_launcher(library, runtime, *, hidden, weight, slab, slots,
                     rows=None, state_slots=None, channels=None,
                     kernel_size=None, slot_offset=0):
    """Call the raw C symbol, returning (error, out, state)."""

    channels = weight.shape[0] if channels is None else channels
    kernel_size = weight.shape[1] if kernel_size is None else kernel_size
    rows = len(slots) if rows is None else rows
    state_slots = slab.shape[0] if state_slots is None else state_slots
    hidden_dev = malloc(hidden.nbytes, runtime=runtime)
    weight_dev = malloc(weight.nbytes, runtime=runtime)
    state_dev = malloc(slab.nbytes, runtime=runtime)
    index_dev = malloc(slots.nbytes, runtime=runtime)
    out_host = np.full(rows * channels, _PAD_SENTINEL, np.float32)
    out_dev = malloc(out_host.nbytes, runtime=runtime)
    fn = library.hipengine_qwen4_exp_linear_attn_conv_decode_indexed_f32
    fn.argtypes = _RAW_ARG_TYPES
    fn.restype = ctypes.c_int
    try:
        copy_host_to_device(
            hidden_dev, host_array_ptr(np.ascontiguousarray(hidden)),
            runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight), runtime=runtime)
        copy_host_to_device(state_dev, host_array_ptr(slab), runtime=runtime)
        copy_host_to_device(index_dev, host_array_ptr(slots), runtime=runtime)
        copy_host_to_device(out_dev, host_array_ptr(out_host), runtime=runtime)
        error = fn(
            hidden_dev.ptr, state_dev.ptr, weight_dev.ptr, out_dev.ptr,
            index_dev.ptr + slot_offset,
            rows, state_slots, channels, kernel_size, 0)
        runtime.device_synchronize()
        out = np.empty_like(out_host)
        final = np.empty_like(slab)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state_dev, runtime=runtime)
        return int(error), out, final
    finally:
        for buffer in (hidden_dev, weight_dev, state_dev, index_dev, out_dev):
            free(buffer, runtime=runtime)


@pytest.mark.parametrize(
    "label",
    ["slab_too_small", "slab_too_wide", "too_many_rows", "misaligned_slots",
     "channels_too_wide", "kernel_too_wide", "zero_rows"],
)
def test_c_launcher_rejects_invalid_geometry(label):
    """The C entry point must not rely on the Python validator.

    The host check runs first on every normal call, so these bounds are only
    exercised by a caller that builds its own argument list. Each of them would
    otherwise index outside the declared state slab or the launched extent.
    """

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    channels, kernel_size = 64, 4
    rng = np.random.default_rng(2468)
    hidden, weight, slab = _fixture(rng, 2, 2, channels, kernel_size)
    slots = np.array([0, 1], dtype=np.int64)
    kwargs = {}
    if label == "slab_too_small":
        kwargs = {"rows": 2, "state_slots": 1}
    elif label == "slab_too_wide":
        kwargs = {"state_slots": QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS + 1}
    elif label == "too_many_rows":
        kwargs = {"rows": QWEN4_EXP_CONV_INDEXED_MAX_ROWS + 1}
    elif label == "misaligned_slots":
        kwargs = {"slot_offset": 4}
    elif label == "channels_too_wide":
        kwargs = {"channels": QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS + 1}
    elif label == "kernel_too_wide":
        kwargs = {"kernel_size": QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE + 1}
    elif label == "zero_rows":
        kwargs = {"rows": 0}

    error, out, final = _call_c_launcher(
        library, runtime, hidden=hidden, weight=weight, slab=slab, slots=slots,
        **kwargs)

    assert error != 0, f"C launcher accepted invalid geometry: {label}"
    np.testing.assert_array_equal(
        out, np.full(out.size, _PAD_SENTINEL, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_c_launcher_accepts_a_valid_call():
    """Guard against a C check that rejects everything."""

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    channels, kernel_size = 256, 4
    rng = np.random.default_rng(1357)
    hidden, weight, slab = _fixture(rng, 3, 4, channels, kernel_size)
    slots = np.array([3, 0, 2], dtype=np.int64)

    error, out, final = _call_c_launcher(
        library, runtime, hidden=hidden, weight=weight, slab=slab, slots=slots)

    assert error == 0
    expected_out, expected_state = _expected_serial(
        library, runtime, hidden, weight, slab, tuple(slots.tolist()))
    np.testing.assert_array_equal(
        out.view(np.uint32), expected_out.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))
