"""Indexed Qwen4Exp GDN decode: independent rows over per-row state slots.

The primitive is the multi-row form of
:func:`qwen4_exp_gdn_decode_rowsplit_f32`: row ``r`` reads its own Conv row,
scalars, output gate and output row, and reads/writes exactly the recurrent
state slot ``state_indices[r]``. Every row therefore advances its own request
state, and slots no row selects are untouched.

Two contracts are asserted separately, per ``docs/EXECUTION-PROFILES.md``:

* **Ownership (exact in every profile).** Row -> slot mapping, untouched
  slots, ragged active subsets, permuted slots across repeated steps, and
  determinism are byte-level assertions. A failure is a bug, not drift.
* **Arithmetic.** Each block executes the single-row kernel's instruction
  sequence on its own row, so the declared expectation is *bit-exact* equality
  with the single-row owner (T0, layout-only). That claim is asserted, not
  assumed: if it ever fails, the primitive is T2 and needs the production
  numerical gate before use, and this test records the measured deviation
  instead of passing silently.

A CPU oracle (committed segmented recurrence plus the norm/gate epilogue) is
also compared, at FP32 round-off, so the primitive is not its own oracle.
"""
import ctypes
import math

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.cpu_reference.ops import gdn_prefill_recurrent_segments
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    QWEN4_EXP_GDN_INDEXED_MAX_ROWS,
    build_qwen4_exp_gdn_dpp,
    qwen4_exp_gdn_decode_rowsplit_f32,
    qwen4_exp_gdn_decode_rowsplit_indexed_f32,
    qwen4_exp_gdn_indexed_state_indices,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve

_K_HEADS = 16
_DIM = 128
_RAW_ARG_TYPES = (ctypes.c_void_p,) * 10 + (ctypes.c_int64,) * 10 + (ctypes.c_void_p,)
_INDEXED_KEY = KernelKey(
    "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
    "qwen4exp_gdn_indexed_decode",
)


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


def _rows(rng, rows: int, v_heads: int) -> list[np.ndarray]:
    """Packed per-row operands: conv, gate, alpha, beta logits."""

    qkv_width = 2 * _K_HEADS * _DIM + v_heads * _DIM
    return [
        rng.normal(0.0, 0.05, (rows, qkv_width)).astype(np.float32),
        rng.normal(0.0, 0.5, (rows, v_heads * _DIM)).astype(np.float32),
        rng.normal(-0.2, 0.1, (rows, v_heads)).astype(np.float32),
        rng.normal(0.0, 0.2, (rows, v_heads)).astype(np.float32),
    ]

def _weights(rng, v_heads: int) -> list[np.ndarray]:
    return [
        rng.normal(-1.0, 0.1, v_heads).astype(np.float32),
        (-np.exp(rng.normal(-0.5, 0.1, v_heads))).astype(np.float32),
        rng.normal(1.0, 0.05, _DIM).astype(np.float32),
    ]


def _run_single(library, runtime, rows, weights, state0, *, row, v_heads):
    """Run the single-row owner on row ``row`` from ``state0``, return outputs."""

    conv, gate, alpha, beta = rows
    dev = [
        malloc(values[row].nbytes, runtime=runtime)
        for values in rows
    ] + [malloc(values.nbytes, runtime=runtime) for values in weights]
    state = malloc(state0.nbytes, runtime=runtime)
    out = malloc(v_heads * _DIM * 4, runtime=runtime)
    try:
        for buffer, values in zip(dev[:4], rows):
            copy_host_to_device(
                buffer, host_array_ptr(np.ascontiguousarray(values[row])),
                runtime=runtime)
        for buffer, values in zip(dev[4:], weights):
            copy_host_to_device(buffer, host_array_ptr(values), runtime=runtime)
        copy_host_to_device(state, host_array_ptr(state0), runtime=runtime)
        qwen4_exp_gdn_decode_rowsplit_f32(
            *(buffer.ptr for buffer in dev), state.ptr, out.ptr,
            _K_HEADS, v_heads, _DIM, _DIM, library=library, runtime=runtime)
        runtime.device_synchronize()
        core = np.empty(v_heads * _DIM, np.float32)
        final = np.empty_like(state0)
        copy_device_to_host(host_array_ptr(core), out, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state, runtime=runtime)
        return core, final
    finally:
        for buffer in (*dev, state, out):
            free(buffer, runtime=runtime)


def _run_indexed(
    library, runtime, rows, weights, slab, slots, *, v_heads, strides=None,
    raw_slots=None, sentinel=np.float32(-777.0),
):
    """Run the indexed owner once over packed rows and the whole state slab.

    Returns the full output block (row stride width, so padded padding is
    visible) and the whole state slab.

    ``strides`` is an optional ``(conv, scalar, gate, output)`` tuple of padded
    row strides; the packed operands are then laid out with that padding filled
    by ``sentinel`` so a wrong row base shows up as a sentinel value.

    ``raw_slots`` bypasses :func:`qwen4_exp_gdn_indexed_state_indices` on purpose
    so the device-side guard is what gets measured.
    """

    indices = (
        np.ascontiguousarray(raw_slots, dtype=np.int64)
        if raw_slots is not None
        else qwen4_exp_gdn_indexed_state_indices(
            slots, rows=len(slots), state_slots=slab.shape[0])
    )
    count = indices.size
    conv, gate, alpha, beta = rows
    if strides is not None:
        conv, gate, alpha, beta = _pad_rows(rows, strides, v_heads, sentinel)
    packed = [conv, gate, alpha, beta]
    kwargs = {}
    if strides is not None:
        kwargs = dict(
            conv_row_stride=strides[0], scalar_row_stride=strides[1],
            gate_row_stride=strides[2], output_row_stride=strides[3],
        )
    dev = [
        malloc(values.nbytes, runtime=runtime) for values in packed
    ] + [malloc(values.nbytes, runtime=runtime) for values in weights]
    state = malloc(slab.nbytes, runtime=runtime)
    index_buffer = malloc(indices.nbytes, runtime=runtime)
    output_row_stride = strides[3] if strides is not None else v_heads * _DIM
    out_host = np.full((count, output_row_stride), sentinel, np.float32)
    out = malloc(out_host.nbytes, runtime=runtime)
    try:
        for buffer, values in zip(dev[:4], packed):
            copy_host_to_device(
                buffer, host_array_ptr(np.ascontiguousarray(values)),
                runtime=runtime)
        for buffer, values in zip(dev[4:], weights):
            copy_host_to_device(buffer, host_array_ptr(values), runtime=runtime)
        copy_host_to_device(state, host_array_ptr(slab), runtime=runtime)
        copy_host_to_device(
            index_buffer, host_array_ptr(indices), runtime=runtime)
        copy_host_to_device(out, host_array_ptr(out_host), runtime=runtime)
        qwen4_exp_gdn_decode_rowsplit_indexed_f32(
            *(buffer.ptr for buffer in dev), state.ptr, out.ptr,
            index_buffer.ptr,
            rows=count, state_slots=slab.shape[0],
            num_k_heads=_K_HEADS, num_v_heads=v_heads,
            head_k_dim=_DIM, head_v_dim=_DIM,
            library=library, runtime=runtime, **kwargs)
        runtime.device_synchronize()
        core = np.empty_like(out_host)
        final = np.empty_like(slab)
        copy_device_to_host(host_array_ptr(core), out, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state, runtime=runtime)
        return core, final
    finally:
        for buffer in (*dev, state, index_buffer, out):
            free(buffer, runtime=runtime)


def _pad_rows(rows, strides, v_heads, sentinel):
    """Lay the packed operands out with padded row strides.

    ``rows`` is ``(conv, gate, alpha, beta)`` and ``strides`` is
    ``(conv, scalar, gate, output)``, so alpha and beta both use the scalar
    stride and the output stride is not a packed operand.
    """

    widths = (
        2 * _K_HEADS * _DIM + v_heads * _DIM, v_heads * _DIM, v_heads, v_heads,
    )
    per_operand = (strides[0], strides[2], strides[1], strides[1])
    padded = []
    for values, width, stride in zip(rows, widths, per_operand):
        buffer = np.full((values.shape[0], stride), sentinel, np.float32)
        buffer[:, :width] = values
        padded.append(buffer)
    return padded


# ---------------------------------------------------------------------------
# CPU oracle: committed segmented recurrence plus the norm/gate epilogue.
# ---------------------------------------------------------------------------


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value.astype(np.float64)))


def _softplus(value):
    value = value.astype(np.float64)
    return np.where(value > 20.0, value, np.log1p(np.exp(value)))


def _cpu_oracle(rows, weights, slab, slots, *, v_heads, eps=1.0e-6):
    conv, gate, alpha, beta_logits = rows
    dt_bias, a_log, norm_weight = weights
    count = len(slots)
    key_offset = _K_HEADS * _DIM
    value_offset = 2 * _K_HEADS * _DIM

    query = np.zeros((count, v_heads, _DIM), np.float32)
    key = np.zeros((count, v_heads, _DIM), np.float32)
    value = np.zeros((count, v_heads, _DIM), np.float32)
    beta = np.zeros((count, v_heads), np.float32)
    decay = np.zeros((count, v_heads), np.float32)
    for row in range(count):
        for v_head in range(v_heads):
            k_head = v_head % _K_HEADS
            q_slice = conv[row, k_head * _DIM : (k_head + 1) * _DIM]
            k_slice = conv[row, key_offset + k_head * _DIM : key_offset + (k_head + 1) * _DIM]
            v_slice = conv[row, value_offset + v_head * _DIM : value_offset + (v_head + 1) * _DIM]
            q_scale = 1.0 / math.sqrt(float(np.sum(q_slice**2)) + eps) / math.sqrt(_DIM)
            k_scale = 1.0 / math.sqrt(float(np.sum(k_slice**2)) + eps)
            query[row, v_head] = (q_slice * q_scale).astype(np.float32)
            key[row, v_head] = (k_slice * k_scale).astype(np.float32)
            value[row, v_head] = v_slice
            beta[row, v_head] = np.float32(_sigmoid(beta_logits[row, v_head]))
            # ``a`` carries the sign already (Qwen4Exp stores ``-exp(log)``), so
            # this is the kernel's own expression, not a re-derivation.
            decay[row, v_head] = np.float32(
                np.exp(float(a_log[v_head])
                       * float(_softplus(np.float32(alpha[row, v_head] + dt_bias[v_head])))))
    core, state = gdn_prefill_recurrent_segments(
        query, key, value, beta, decay, slab,
        np.arange(count + 1, dtype=np.int64),
        np.asarray(slots, dtype=np.int64),
    )
    out = np.zeros_like(core)
    for row in range(count):
        for v_head in range(v_heads):
            head = core[row, v_head]
            rms = 1.0 / math.sqrt(float(np.sum(head**2)) / _DIM + eps)
            out[row, v_head] = (
                head * rms * norm_weight * _sigmoid(gate[row, v_head * _DIM : (v_head + 1) * _DIM])
            ).astype(np.float32)
    return out.reshape(count, v_heads * _DIM), state


# ---------------------------------------------------------------------------
# Ownership contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v_heads", [48, 32])
@pytest.mark.parametrize("slots", [(0,), (3, 1), (2, 0, 5, 4)])
def test_indexed_decode_is_bit_exact_to_the_single_row_owner(v_heads, slots):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    count = len(slots)
    rng = np.random.default_rng(101 * v_heads + count)
    rows = _rows(rng, count, v_heads)
    weights = _weights(rng, v_heads)
    state_slots = max(slots) + 1 + (1 if count > 1 else 0)
    slab = rng.normal(0.0, 0.01, (state_slots, v_heads, _DIM, _DIM)).astype(np.float32)

    core, final = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads)

    expected_core = np.empty_like(core)
    expected_state = slab.copy()
    for row, slot in enumerate(slots):
        single_core, single_state = _run_single(
            library, runtime, rows, weights, slab[slot], row=row, v_heads=v_heads)
        expected_core[row] = single_core
        expected_state[slot] = single_state

    assert np.isfinite(core).all() and np.isfinite(final).all()
    np.testing.assert_array_equal(core.view(np.uint32), expected_core.view(np.uint32))
    np.testing.assert_array_equal(final.view(np.uint32), expected_state.view(np.uint32))


def test_indexed_decode_rejects_an_out_of_range_slot_as_a_whole_batch():
    """One bad slot rejects the batch: no block writes state or output.

    The host validator normally catches this, so the device pointer is built by
    hand to measure the kernel's own guard. A per-row guard would let the valid
    row commit while the batch as a whole is malformed, which is a partial
    update; the contract here is all-or-nothing.
    """

    sentinel = np.float32(-12345.0)

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, state_slots = 48, 4
    rng = np.random.default_rng(555)
    rows = _rows(rng, 2, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (state_slots, v_heads, _DIM, _DIM)).astype(np.float32)
    bad_slots = np.array([2, state_slots], dtype=np.int64)
    core, final = _run_indexed(
        library, runtime, rows, weights, slab, None, v_heads=v_heads,
        raw_slots=bad_slots, sentinel=sentinel)

    np.testing.assert_array_equal(
        core, np.full((2, v_heads * _DIM), sentinel, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_rejects_duplicate_slots_as_a_whole_batch():
    """Two rows naming one slot is an ownership violation, not a race.

    Left to the device the two blocks would both read the same initial state and
    both write it, so the survivor depends on completion order. The batch is
    rejected instead: every slot and every output row keeps its exact bytes.
    """

    sentinel = np.float32(-12345.0)

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, state_slots = 48, 4
    rng = np.random.default_rng(999)
    rows = _rows(rng, 2, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (state_slots, v_heads, _DIM, _DIM)).astype(np.float32)

    core, final = _run_indexed(
        library, runtime, rows, weights, slab, None, v_heads=v_heads,
        raw_slots=np.array([1, 1], dtype=np.int64), sentinel=sentinel)

    np.testing.assert_array_equal(
        core, np.full((2, v_heads * _DIM), sentinel, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_duplicate_rejection_is_deterministic():
    """Repeated runs of a malformed batch produce identical (null) effects."""
    sentinel = np.float32(-12345.0)

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, state_slots = 48, 5
    rng = np.random.default_rng(1234)
    rows = _rows(rng, 3, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (state_slots, v_heads, _DIM, _DIM)).astype(np.float32)
    bad = np.array([4, 0, 4], dtype=np.int64)

    first_core, first_state = _run_indexed(
        library, runtime, rows, weights, slab, None, v_heads=v_heads,
        raw_slots=bad, sentinel=sentinel)
    second_core, second_state = _run_indexed(
        library, runtime, rows, weights, slab, None, v_heads=v_heads,
        raw_slots=bad, sentinel=sentinel)
    np.testing.assert_array_equal(
        first_core.view(np.uint32), second_core.view(np.uint32))
    np.testing.assert_array_equal(
        first_state.view(np.uint32), second_state.view(np.uint32))
    np.testing.assert_array_equal(first_state.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_admits_the_maximum_row_count():
    """64 rows exercise the scan loop's second pass (``lane + 32``)."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads = 16  # small slab: 64 slots x 1 MiB
    limit = QWEN4_EXP_GDN_INDEXED_MAX_ROWS
    slots = tuple(int(v) for v in np.random.default_rng(64).permutation(limit))
    rng = np.random.default_rng(6464)
    rows = _rows(rng, limit, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (limit, v_heads, _DIM, _DIM)).astype(np.float32)

    core, final = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads)

    expected_core = np.empty_like(core)
    expected_state = slab.copy()
    for row, slot in enumerate(slots):
        expected_core[row], expected_state[slot] = _run_single(
            library, runtime, rows, weights, slab[slot], row=row, v_heads=v_heads)
    assert np.isfinite(core).all()
    np.testing.assert_array_equal(
        core.view(np.uint32), expected_core.view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))


def test_indexed_decode_rejects_a_duplicate_beyond_the_first_scan_pass():
    """A duplicate only the scan loop's second pass can see still rejects."""

    sentinel = np.float32(-12345.0)
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads = 16
    limit = QWEN4_EXP_GDN_INDEXED_MAX_ROWS
    slots = np.arange(limit, dtype=np.int64)
    slots[40] = slots[0]  # lane 8 of the second pass collides with lane 0
    rng = np.random.default_rng(4040)
    rows = _rows(rng, limit, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (limit, v_heads, _DIM, _DIM)).astype(np.float32)

    core, final = _run_indexed(
        library, runtime, rows, weights, slab, None, v_heads=v_heads,
        raw_slots=slots, sentinel=sentinel)

    np.testing.assert_array_equal(
        core, np.full((limit, v_heads * _DIM), sentinel, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_indexed_decode_honours_padded_strides():
    """Padded rows address correctly and never spill into the padding."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, slots = 48, (3, 1, 0)
    count = len(slots)
    core_width = v_heads * _DIM
    conv_width = 2 * _K_HEADS * _DIM + core_width
    strides = (conv_width + 4, v_heads + 8, core_width + 4, core_width + 8)
    rng = np.random.default_rng(4711)
    rows = _rows(rng, count, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (4, v_heads, _DIM, _DIM)).astype(np.float32)

    core, final = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads,
        strides=strides)
    assert core.shape == (count, strides[3])

    expected_core = np.empty((count, core_width), np.float32)
    expected_state = slab.copy()
    for row, slot in enumerate(slots):
        expected_core[row], expected_state[slot] = _run_single(
            library, runtime, rows, weights, slab[slot], row=row, v_heads=v_heads)
    np.testing.assert_array_equal(
        core[:, :core_width].view(np.uint32), expected_core.view(np.uint32))
    np.testing.assert_array_equal(
        final.view(np.uint32), expected_state.view(np.uint32))

    # The output padding past the value-head row is never written. A stride that
    # was too small, or a row base computed without it, would show up here (or
    # already in the equality assertion above).
    np.testing.assert_array_equal(
        core[:, core_width:],
        np.full((count, strides[3] - core_width), np.float32(-777.0), np.float32))


def _call_c_launcher(library, runtime, *, conv_stride, state_slots, rows_host,
                     slots, slab, v_heads=48, conv_offset=0):
    """Call the raw C symbol, returning (error, output, state)."""

    conv_width = 2 * _K_HEADS * _DIM + v_heads * _DIM
    count = len(slots)
    dev = [malloc(values.nbytes, runtime=runtime) for values in rows_host]
    state = malloc(slab.nbytes, runtime=runtime)
    index_buffer = malloc(slots.nbytes, runtime=runtime)
    sentinel = np.float32(-12345.0)
    out_host = np.full((count, v_heads * _DIM), sentinel, np.float32)
    out = malloc(out_host.nbytes, runtime=runtime)
    fn = library.hipengine_qwen4_exp_gdn_decode_rowsplit_indexed_f32
    fn.argtypes = _RAW_ARG_TYPES
    fn.restype = ctypes.c_int
    try:
        for buffer, values in zip(dev, rows_host):
            copy_host_to_device(
                buffer, host_array_ptr(np.ascontiguousarray(values)),
                runtime=runtime)
        copy_host_to_device(state, host_array_ptr(slab), runtime=runtime)
        copy_host_to_device(index_buffer, host_array_ptr(slots), runtime=runtime)
        copy_host_to_device(out, host_array_ptr(out_host), runtime=runtime)
        error = fn(
            dev[0].ptr + conv_offset, dev[1].ptr, dev[2].ptr, dev[3].ptr,
            dev[4].ptr, dev[5].ptr, dev[6].ptr, state.ptr, out.ptr,
            index_buffer.ptr,
            count, state_slots, conv_stride, v_heads, v_heads * _DIM,
            v_heads * _DIM, _K_HEADS, v_heads, _DIM, _DIM, 0)
        runtime.device_synchronize()
        core = np.empty_like(out_host)
        final = np.empty_like(slab)
        copy_device_to_host(host_array_ptr(core), out, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state, runtime=runtime)
        return int(error), core, final, sentinel
    finally:
        for buffer in (*dev, state, index_buffer, out):
            free(buffer, runtime=runtime)


@pytest.mark.parametrize(
    "label,conv_stride,conv_offset,state_slots",
    [
        ("overlapping", None, 0, 2),        # conv_width - 4, filled in below
        ("unaligned_stride", None, 0, 2),   # conv_width + 1
        ("misaligned_base", None, 4, 2),    # 4-byte offset on the Conv pointer
        ("slab_too_wide", None, 0, 4097),
    ],
)
def test_c_launcher_rejects_invalid_geometry(
    label, conv_stride, conv_offset, state_slots
):
    """The C entry point must not rely on the Python validator.

    A stride shorter than one packed row makes row r's tail overlap row r+1, so
    the kernel would read another request's Conv values; a stride that is not a
    whole float4 breaks the 16-byte alignment of every row base; and a slab past
    the bitmap bound would index outside the shared scan. All three have to be
    rejected before the launch, otherwise any non-Python caller can trigger them.
    """

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads = 48
    conv_width = 2 * _K_HEADS * _DIM + v_heads * _DIM
    if label == "overlapping":
        conv_stride = conv_width - 4
    elif label == "unaligned_stride":
        conv_stride = conv_width + 1
    else:
        conv_stride = conv_width
    rng = np.random.default_rng(2468)
    packed = _rows(rng, 2, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (min(state_slots, 2), v_heads, _DIM, _DIM))
    slab = slab.astype(np.float32)
    slots = np.array([0, 1], dtype=np.int64)

    error, core, final, sentinel = _call_c_launcher(
        library, runtime, conv_stride=conv_stride, state_slots=state_slots,
        rows_host=[*packed, *weights], slots=slots, slab=slab,
        conv_offset=conv_offset)

    assert error != 0, f"C launcher accepted invalid geometry: {label}"
    np.testing.assert_array_equal(
        core, np.full((2, v_heads * _DIM), sentinel, np.float32))
    np.testing.assert_array_equal(final.view(np.uint32), slab.view(np.uint32))


def test_c_launcher_accepts_a_valid_padded_stride():
    """Guard against a C check that rejects everything."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads = 48
    conv_width = 2 * _K_HEADS * _DIM + v_heads * _DIM
    rng = np.random.default_rng(1357)
    packed = _rows(rng, 2, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (2, v_heads, _DIM, _DIM)).astype(np.float32)
    padded_conv = np.full((2, conv_width + 4), np.float32(-777.0), np.float32)
    padded_conv[:, :conv_width] = packed[0]
    rows_host = [padded_conv, packed[1], packed[2], packed[3], *weights]
    slots = np.array([1, 0], dtype=np.int64)

    error, core, _, _ = _call_c_launcher(
        library, runtime, conv_stride=conv_width + 4, state_slots=2,
        rows_host=rows_host, slots=slots, slab=slab)

    assert error == 0
    assert np.isfinite(core).all()
    np.testing.assert_array_equal(
        core[:, : v_heads * _DIM][0],
        _run_single(library, runtime, packed, weights, slab[1], row=0,
                    v_heads=v_heads)[0])


@pytest.mark.parametrize("slots", [(3, 1), (5, 0, 4)])
def test_indexed_decode_leaves_unselected_slots_and_rows_untouched(slots):
    """Ragged active subset: unselected slots keep their exact bytes."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, state_slots = 48, 6
    count = len(slots)
    rng = np.random.default_rng(7 * state_slots + count)
    rows = _rows(rng, count, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (state_slots, v_heads, _DIM, _DIM)).astype(np.float32)

    _, final = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads)

    expected_state = slab.copy()
    for row, slot in enumerate(slots):
        _, single_state = _run_single(
            library, runtime, rows, weights, slab[slot], row=row, v_heads=v_heads)
        expected_state[slot] = single_state

    np.testing.assert_array_equal(final.view(np.uint32), expected_state.view(np.uint32))
    for slot in range(state_slots):
        if slot in slots:
            continue
        np.testing.assert_array_equal(
            final[slot].view(np.uint32), slab[slot].view(np.uint32))


def test_indexed_decode_repeated_steps_follow_permuted_slots():
    """Three steps with a changing composition keep per-slot state separate."""

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, state_slots = 48, 4
    steps = [(1, 0), (3, 1), (0, 2)]
    rng = np.random.default_rng(31337)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (state_slots, v_heads, _DIM, _DIM)).astype(np.float32)
    step_rows = [_rows(rng, len(slots), v_heads) for slots in steps]

    batched_state = slab.copy()
    reference_state = slab.copy()
    for rows, slots in zip(step_rows, steps):
        batched_core, batched_state = _run_indexed(
            library, runtime, rows, weights, batched_state, slots, v_heads=v_heads)
        reference_core = np.empty_like(batched_core)
        for row, slot in enumerate(slots):
            reference_core[row], reference_state[slot] = _run_single(
                library, runtime, rows, weights, reference_state[slot],
                row=row, v_heads=v_heads)
        np.testing.assert_array_equal(
            batched_core.view(np.uint32), reference_core.view(np.uint32))
        np.testing.assert_array_equal(
            batched_state.view(np.uint32), reference_state.view(np.uint32))


def test_indexed_decode_is_deterministic_across_reruns():
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads, slots = 48, (2, 0, 1)
    rng = np.random.default_rng(606)
    rows = _rows(rng, len(slots), v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (4, v_heads, _DIM, _DIM)).astype(np.float32)

    first_core, first_state = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads)
    second_core, second_state = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads)
    np.testing.assert_array_equal(
        first_core.view(np.uint32), second_core.view(np.uint32))
    np.testing.assert_array_equal(
        first_state.view(np.uint32), second_state.view(np.uint32))


# ---------------------------------------------------------------------------
# Arithmetic contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slots", [(0,), (2, 0), (3, 1, 2, 0)])
def test_indexed_decode_matches_the_cpu_oracle_to_round_off(slots):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    v_heads = 48
    count = len(slots)
    rng = np.random.default_rng(8080 + count)
    rows = _rows(rng, count, v_heads)
    weights = _weights(rng, v_heads)
    slab = rng.normal(0.0, 0.01, (max(slots) + 1, v_heads, _DIM, _DIM)).astype(np.float32)

    core, final = _run_indexed(
        library, runtime, rows, weights, slab, slots, v_heads=v_heads)
    expected_core, expected_state = _cpu_oracle(
        rows, weights, slab, slots, v_heads=v_heads)

    assert np.isfinite(core).all() and np.isfinite(final).all()
    # Measured on gfx1100 at N=1/2/4 with permuted slots: output relL2 1.60e-7..1.63e-7
    # (max abs 2.4e-7..4.8e-7), state relL2 3.3e-8..4.4e-8. The gate keeps
    # ~12x / ~30x headroom over that, matching the neighbouring row-split
    # prefill round-off gate rather than being tuned to this measurement.
    assert np.linalg.norm(core - expected_core) <= 2.0e-6 * np.linalg.norm(expected_core)
    assert np.abs(core - expected_core).max() <= 1.0e-5 * max(1.0, np.abs(expected_core).max())
    scale = np.linalg.norm(expected_state)
    assert np.linalg.norm(final - expected_state) <= 2.0e-6 * scale


# ---------------------------------------------------------------------------
# Admission and registry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rows": 0}, "rows must be in 1"),
        ({"rows": QWEN4_EXP_GDN_INDEXED_MAX_ROWS + 1}, "rows must be in 1"),
        ({"state_slots": 0}, "state_slots must be positive"),
        ({"head_k_dim": 64}, "128x128 heads"),
        ({"head_v_dim": 256}, "128x128 heads"),
        ({"num_k_heads": 0}, "num_v_heads must be divisible"),
        ({"num_v_heads": 47}, "num_v_heads must be divisible"),
        ({"conv_row_stride": 0}, "conv_row_stride must be positive"),
        ({"scalar_row_stride": 0}, "scalar_row_stride must be positive"),
        ({"output_row_stride": -1}, "output_row_stride must be positive"),
    ],
)
def test_indexed_decode_admission_rejects_unbounded_shapes(kwargs, match):
    base = {
        "rows": 1,
        "state_slots": 1,
        "num_k_heads": _K_HEADS,
        "num_v_heads": 48,
        "head_k_dim": _DIM,
        "head_v_dim": _DIM,
    }
    base.update(kwargs)
    # 16-byte aligned placeholders so the pointer checks do not fire first and
    # mask the shape check under test.
    with pytest.raises(ValueError, match=match):
        qwen4_exp_gdn_decode_rowsplit_indexed_f32(
            *([16] * 10), runtime=object(), library=object(), **base)


def test_indexed_decode_admission_rejects_a_short_state_slab():
    with pytest.raises(ValueError, match="state slab"):
        qwen4_exp_gdn_decode_rowsplit_indexed_f32(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            rows=2, state_slots=1,
            num_k_heads=_K_HEADS, num_v_heads=48,
            head_k_dim=_DIM, head_v_dim=_DIM,
            runtime=object(), library=object())


def test_indexed_decode_registers_with_the_single_row_fallback():
    """The variant resolves on the device backend and keeps the fallback.

    The gfx1151 package mirrors every gfx1100 key under its own backend, so on a
    gfx1151 host the variant must resolve under that key too. Both are checked,
    because the source package registers ``hip_gfx1100`` and the mirror is what
    makes it reachable here.
    """

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
        ) is qwen4_exp_gdn_decode_rowsplit_indexed_f32

    # The existing single-row owner stays registered as the strict fallback on
    # both keys. ``qwen4exp_gdn_rowsplit_decode`` is a gfx1151 variant only; the
    # source package's own single-row owner is ``qwen4exp_sigmoid_strict``.
    for variant in (
        "qwen4exp_sigmoid_strict",
        "qwen4exp_gdn_rowsplit_decode",
    ):
        backends = (
            ("hip_gfx1151",)
            if variant == "qwen4exp_gdn_rowsplit_decode"
            else ("hip_gfx1151", "hip_gfx1100")
        )
        for backend in backends:
            assert is_registered(
                KernelKey(backend, "gdn_recurrence_norm_gate", "f32_state", variant)
            )
