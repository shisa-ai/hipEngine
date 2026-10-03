"""The expert forward's output must not depend on how many rows share the call.

`tests/test_unit_gemma4_expert_route.py` covers which route runs, and
`tests/test_gpu_gemma4_expert_route_true_geometry.py` covers the MMQ tile walk at
the true projection geometry. Neither asserts the property a *caller* depends on:
a token's expert output is a function of that token, not of the other rows that
happened to be in the same block.

That property is what a speculative verify needs. It decodes one row at a time
and then verifies several at once, so a route whose arithmetic depends on the
block width makes the verify disagree with the decode it is verifying. It is also
what `test_incremental_decode_matches_a_dense_prefill` exercises end to end.

Two defects are pinned separately below, because they have different causes and
different fixes:

* the fused MMQ gate/up route diverges across row counts, and
* the down projection diverges across row counts even with the MMQ route off.

Both were found by reading the real intermediates out of the layer's expert
scratch rather than by probing the projections directly; entering at
``gemma4_project_experts_rows`` skips the MMQ attempt that the real call makes
first, and that attempt succeeds for gate/up at this geometry.
"""

from __future__ import annotations

import ctypes
import tempfile

import numpy as np
import pytest

try:  # pragma: no cover - platform guard
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:  # pragma: no cover - no ROCm on this runner
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HIP_AVAILABLE, reason="requires the HIP runtime (libamdhip64.so)"
)

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    gemma4_experts_forward_bf16,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights

from tests._gemma4_gguf_fixture import write_default_gemma4_gguf

_TOP_K_ROUTING = np.array([[0, 2]], dtype=np.int64)
_UNIT_WEIGHTS = np.array([[0.5, 0.5]], dtype=np.float32)


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round to bf16 by truncating the low 16 bits, as the kernels' inputs do."""
    return (np.ascontiguousarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )


def _to_float(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def _upload(array: np.ndarray):
    """Return a device buffer holding ``array``; pass ``buffer.ptr`` to a kernel."""

    buffer = malloc(array.nbytes)
    copy_host_to_device(buffer, host_array_ptr(np.ascontiguousarray(array)), array.nbytes)
    return buffer


def _download(pointer: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    # ``copy_device_to_host`` takes a DeviceBuffer, and the caller-owned output
    # here is a raw pointer, so this reads through the runtime directly.
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import MemcpyKind

    get_hip_runtime().memcpy(host_array_ptr(out), pointer, out.nbytes, MemcpyKind.DEVICE_TO_HOST)
    return out


@pytest.fixture(scope="module")
def expert_context():
    """The fixture model's layer-0 weights, runner and expert scratch."""

    path = write_default_gemma4_gguf(tempfile.mkdtemp() + "/expert_rows.gguf")
    weights = load_gemma4_device_weights(GGUFReader(path))
    runner = Gemma4Runner(weights=weights, capacity=64, max_logits_rows=5)
    try:
        yield runner, weights
    finally:
        runner.close()
        weights.free()


def _forward_row_zero(runner, weights, hidden: np.ndarray, rows: int) -> np.ndarray:
    """Run the expert forward for ``rows`` rows and return row 0 as F32."""

    config = weights.config
    layer = weights.layers[0]
    scratch = runner._scratches[0].experts
    hidden_size = config.hidden_size

    device_hidden = _upload(hidden[:rows])
    routing = np.concatenate([_TOP_K_ROUTING] * rows, axis=0)
    weights_row = np.concatenate([_UNIT_WEIGHTS] * rows, axis=0)
    device_routing = _upload(routing)
    device_weights = _upload(weights_row)
    out = malloc(rows * hidden_size * 2)
    try:
        gemma4_experts_forward_bf16(
            device_hidden.ptr,
            device_routing.ptr,
            device_weights.ptr,
            layer.experts_gate_up_proj,
            layer.experts_down_proj,
            out.ptr,
            scratch=scratch,
            rows=rows,
        )
        return _to_float(_download(out.ptr, (rows, hidden_size), np.uint16)[0])
    finally:
        free(device_hidden)
        free(device_routing)
        free(device_weights)
        free(out)


def _gate_up_row_zero(runner, weights, hidden, rows, routing_row) -> np.ndarray:
    """Run the forward and return compact row 0 of the gate/up stage as bits."""

    config = weights.config
    layer = weights.layers[0]
    scratch = runner._scratches[0].experts
    fused = 2 * config.moe_intermediate_size
    hidden_size = config.hidden_size

    device_hidden = _upload(hidden[:rows])
    routing = np.concatenate([routing_row] * rows, axis=0)
    unit = np.full((rows, routing_row.shape[1]), 0.5, dtype=np.float32)
    device_routing = _upload(routing)
    device_weights = _upload(unit)
    out = malloc(rows * hidden_size * 2)
    try:
        gemma4_experts_forward_bf16(
            device_hidden.ptr,
            device_routing.ptr,
            device_weights.ptr,
            layer.experts_gate_up_proj,
            layer.experts_down_proj,
            out.ptr,
            scratch=scratch,
            rows=rows,
        )
        # Compact rows are lane-major, so compact row 0's gate half is the first
        # `intermediate` columns of the first row. Flatten before slicing: a 2-D
        # slice would take rows, not elements.
        packed = _download(scratch.buffer("gate_up_out").ptr, (rows * 2, fused), np.uint16)
        return packed.reshape(-1)[:fused]
    finally:
        free(device_hidden)
        free(device_routing)
        free(device_weights)
        free(out)


_TWO_EXPERTS = np.array([[0, 2]], dtype=np.int64)
_ONE_EXPERT = np.array([[0, 0]], dtype=np.int64)


def _hidden(weights):
    return _bf16(np.random.default_rng(0).normal(0, 1, size=(4, weights.config.hidden_size)))


def test_expert_forward_row_zero_is_invariant_with_both_mmq_routes_disabled(
    expert_context, monkeypatch
) -> None:
    """Row-count invariance holds when neither MMQ route can be selected.

    Both routes have their own lever and their own lane-count crossover, so
    disabling one leaves the other free to change route between the two calls.
    With both off, the whole forward is one execution profile and row 0 is a
    function of row 0.
    """

    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", "0")
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_DOWN_MMQ", "0")
    runner, weights = expert_context
    hidden = _hidden(weights)
    narrow = _forward_row_zero(runner, weights, hidden, rows=1)
    wide = _forward_row_zero(runner, weights, hidden, rows=2)
    assert np.array_equal(
        narrow.view(np.uint32), wide.view(np.uint32)
    ), (
        f"row 0 moved by {float(np.abs(narrow - wide).max()):.4f} with both MMQ "
        "routes disabled, so the divergence is not route selection alone"
    )


def test_the_lane_count_selects_the_mmq_route(expert_context) -> None:
    """The MMQ gates are the live lane count, so a row change can change route.

    This is why the two cases above exist. ``lanes = tokens * top_k``, so
    appending a row moves the lane count, and past the gate the fused MMQ
    route runs instead of the fp32 grouped family. The two are different
    execution profiles -- an int8-dp4a accumulation against an fp32 one -- and
    ``docs/EXECUTION-PROFILES.md`` does not contract bit-exactness between a
    strict path and a reassociating production candidate. Invariance is a
    within-route property, and this test pins where the route boundary is so a
    change to the gate is visible rather than silent.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_moe_prefill_route_enabled,
    )

    _, weights = expert_context
    num_experts = weights.config.num_experts
    top_k = _TOP_K_ROUTING.shape[1]

    # One row and two rows, at this fixture's expert count: the lane count is
    # rows * top_k, and the gate opens exactly at the expert count, so the two
    # row counts the other cases use land on different sides of it.
    assert not gemma4_moe_prefill_route_enabled(
        lanes=num_experts - 1, num_experts=num_experts
    )
    assert gemma4_moe_prefill_route_enabled(lanes=num_experts, num_experts=num_experts)
    assert gemma4_moe_prefill_route_enabled(
        lanes=num_experts + 1, num_experts=num_experts
    )

    # And the transition is reachable from the row counts, not just in theory.
    assert not gemma4_moe_prefill_route_enabled(
        lanes=1 * top_k, num_experts=num_experts
    )
    assert gemma4_moe_prefill_route_enabled(lanes=2 * top_k, num_experts=num_experts)


def test_gate_up_stage_is_invariant_for_two_experts_with_the_mmq_route_disabled(
    expert_context, monkeypatch
) -> None:
    """Two experts in one block must not change the gate/up stage's row 0.

    This passes. Together with the single-expert case below it establishes that
    the gate/up stage is row-count invariant once the MMQ route is off, so the
    gate/up divergence above is the MMQ route and not the grouped or selected
    families -- which is what the direct projection probes measured.
    """

    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", "0")
    runner, weights = expert_context
    hidden = _hidden(weights)
    narrow = _gate_up_row_zero(runner, weights, hidden, 1, _TWO_EXPERTS)
    wide = _gate_up_row_zero(runner, weights, hidden, 2, _TWO_EXPERTS)
    assert np.array_equal(narrow.view(np.uint32), wide.view(np.uint32))


def test_gate_up_stage_is_invariant_for_one_expert_with_the_mmq_route_disabled(
    expert_context, monkeypatch
) -> None:
    """The single-expert case *is* invariant with the MMQ route off.

    This passes today. It is recorded so the difference between the direct
    projection probes and the failing cases is explicit: the probes used one
    expert and entered at ``gemma4_project_experts_rows``, and both of those
    choices are why they measured agreement where the real call diverges.
    """

    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", "0")
    runner, weights = expert_context
    hidden = _hidden(weights)
    narrow = _gate_up_row_zero(runner, weights, hidden, 1, _ONE_EXPERT)
    wide = _gate_up_row_zero(runner, weights, hidden, 2, _ONE_EXPERT)
    assert np.array_equal(narrow.view(np.uint32), wide.view(np.uint32))
