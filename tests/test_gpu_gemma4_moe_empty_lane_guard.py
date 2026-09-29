"""The expert forward must survive a routing that selects no expert.

When every lane is rejected upstream, ``selected`` keeps the ``-1`` sentinel and
the group compaction finds ``active_count == 0``, so it writes nothing into
``sorted_lanes`` / ``sorted_experts`` / ``lane_to_row``. The downstream kernels
still run over ``tokens * top_k`` lanes, so before this guard they consumed
uninitialized heap:

- ``gemma4_moe_lane_to_row_kernel`` indexed ``lane_to_row[sorted_lanes[row]]``
  with an unchecked value, writing far outside the buffer;
- ``gemma4_moe_weighted_accumulate_kernel`` checked only ``row < 0``, so a large
  positive garbage row read outside ``expert_out`` and ``sorted_weights``.

The device reports that as an SQ privilege fault, and -- as the comment in
``gemma4_experts_forward_bf16`` records -- it "is silent and the next
synchronizing call simply blocks forever". The observable symptom is a test
that hangs rather than fails, which is what ``test_unit_gemma4_runner.py`` and
``test_unit_gemma4_generate.py`` did.

This test drives exactly that state: a real fixture artifact, a valid hidden
input, and an all-``-1`` selection. It passes when the call returns and the
output is finite, and it hangs without the range guards.

RED/GREEN note: the red state is a permanent device hang, not an assertion
failure, so running it first would wedge the GPU rather than report. The red
behaviour is therefore evidenced by the two hanging suites named above rather
than by executing this node pre-fix; this file is the green.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free, host_array_ptr, malloc
from hipengine.core.runtime import MemcpyKind
from hipengine.loading.gguf import GGUFReader
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    Gemma4ExpertScratch,
    gemma4_experts_forward_bf16,
)
from hipengine.runtime.gemma4 import load_gemma4_device_weights
from tests._gemma4_gguf_fixture import (
    FIXTURE_EXPERTS,
    FIXTURE_EXPERT_FF,
    FIXTURE_HIDDEN,
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)
from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable"
)

TOKENS = 4
TOP_K = 2  # FIXTURE_EXPERT_USED: two routed lanes per token
_intermediate = FIXTURE_EXPERT_FF


def _to_bf16(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


@pytest.fixture()
def artifact(tmp_path: pathlib.Path) -> GGUFReader:
    path = write_fixture_gguf(
        tmp_path / "gemma4.gguf", default_fixture_tensors(), fixture_metadata()
    )
    return GGUFReader(path)


def test_experts_forward_returns_when_no_expert_is_selected(
    artifact: GGUFReader,
) -> None:
    """An all-rejected routing must complete, not fault the device."""

    runtime = get_hip_runtime()
    weights = load_gemma4_device_weights(artifact)
    layer = weights.layers[0]

    hidden = malloc(TOKENS * FIXTURE_HIDDEN * 2)
    out = malloc(TOKENS * FIXTURE_HIDDEN * 2)
    selected = malloc(TOKENS * TOP_K * 8)
    routing = malloc(TOKENS * TOP_K * 4)

    # Finite, ordinary activations: this case is about the empty routing, not
    # about activation magnitude, so the input must not itself be degenerate.
    hidden_host = _to_bf16(
        np.linspace(-1.0, 1.0, TOKENS * FIXTURE_HIDDEN, dtype=np.float32)
    )
    runtime.memcpy(
        host_array_ptr(hidden_host), hidden.ptr, hidden_host.nbytes,
        MemcpyKind.HOST_TO_DEVICE,
    )
    # Every lane rejected -- the state that leaves compaction with nothing to do.
    selected_host = np.full(TOKENS * TOP_K, -1, dtype=np.int64)
    runtime.memcpy(
        host_array_ptr(selected_host), selected.ptr, selected_host.nbytes,
        MemcpyKind.HOST_TO_DEVICE,
    )
    routing_host = np.zeros(TOKENS * TOP_K, dtype=np.float32)
    runtime.memcpy(
        host_array_ptr(routing_host), routing.ptr, routing_host.nbytes,
        MemcpyKind.HOST_TO_DEVICE,
    )

    scratch = Gemma4ExpertScratch(
        tokens=TOKENS,
        top_k=TOP_K,
        hidden_size=FIXTURE_HIDDEN,
        intermediate=_intermediate,
        num_experts=FIXTURE_EXPERTS,
    )
    # Pre-load the compaction output with out-of-range values. A fresh
    # allocation may happen to be zero-filled, and a zeroed ``sorted_lanes`` is
    # in range -- so without this the node could pass on a lucky allocator and
    # exercise nothing. Because the routing below selects no expert, compaction
    # writes no lanes at all and this garbage is exactly what the downstream
    # kernels read, which is the state the guards exist for.
    garbage = np.full(TOKENS * TOP_K, 1 << 40, dtype=np.int64)
    runtime.memcpy(
        host_array_ptr(garbage),
        scratch.buffer("sorted_lanes").ptr,
        garbage.nbytes,
        MemcpyKind.HOST_TO_DEVICE,
    )
    result = np.zeros(TOKENS * FIXTURE_HIDDEN, dtype=np.uint16)
    try:
        gemma4_experts_forward_bf16(
            hidden.ptr,
            selected.ptr,
            routing.ptr,
            layer.experts_gate_up_proj,
            layer.experts_down_proj,
            out.ptr,
            scratch=scratch,
            rows=TOKENS,
        )
        # Without the range guards this synchronization never returns: the
        # scatter has written outside its buffer and the device has faulted.
        runtime.device_synchronize()
        runtime.memcpy(
            host_array_ptr(result), out.ptr, result.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        scratch.free()
        free(hidden)
        free(out)
        free(selected)
        free(routing)
        weights.free()

    # Completed rather than hung, and produced ordinary finite values instead
    # of faulting on the uninitialized compaction output.
    assert result.size == TOKENS * FIXTURE_HIDDEN
    bad = int(np.sum((result & 0x7F80) == 0x7F80))
    assert bad == 0, f"{bad}/{result.size} output elements are inf/nan"