"""GPU correctness coverage for the mixed-layout GDN and Conv state adapter.

``hipengine/runtime/qwen4_exp_mixed_gdn.py`` runs the request-indexed Conv and
GDN owners in depth waves over provisional state. This module asserts that
adapter against the real kernels:

* **Arithmetic.** The oracle is the existing segmented CPU recurrence plus the
  sigmoid-gated RMSNorm, reached through
  ``tests/test_gpu_qwen4exp_batch_gdn.py::_cpu_step`` — the same helper the
  independent-request mixer is measured against. Nothing here restates the
  recurrence, the Conv shift, or the norm/gate epilogue.
* **Ownership.** Which slot each physical row's snapshot came from, that a
  provisional pass leaves the live owner byte-identical, that a live slot no
  request owns keeps every byte, and that a commit writes exactly the accepted
  prefix's snapshot. These are byte-level assertions, so a failure is a bug.
* **Physical placement.** A permutation that keeps every chain ascending changes
  where a row sits, not what it computes: the same rows must produce the same
  bytes. Both indexed owners run one block per ``(row, head|channel)`` group and
  read only that row, so this is an exact claim rather than a tolerance.

The adapter consumes rows the caller already projected, so this module performs
the caller's shared input projection on the CPU once per pass (the same numpy
expressions the oracle uses internally) and uploads those rows. The recurrence
itself is the device kernel. The caller's ``ssm_out`` projection is likewise
applied on the CPU here, because the adapter deliberately stops at
``core`` — ``[total_rows, core_width]`` in physical row order.

Not covered here: attention, PLE and QSA state for mixed rows, the device
``ssm_out`` projection, and runner wiring. Those are the executor's stages.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
)
from hipengine.runtime.qwen4_exp_batch_gdn import Qwen4ExpBatchGdnState
from hipengine.runtime.qwen4_exp_mixed_gdn import (
    Qwen4ExpMixedGdnAdapter,
    Qwen4ExpMixedGdnProjectedRows,
)
from hipengine.runtime.qwen4_exp_mixed_layout import (
    Qwen4ExpMixedRequest,
    compile_mixed_layout,
)
from tests.test_gpu_qwen4exp_batch_gdn import (
    CORE_WIDTH,
    HEAD_DIM,
    HIDDEN,
    KERNEL,
    K_HEADS,
    ORACLE_ATOL,
    ORACLE_RTOL,
    QKV_WIDTH,
    V_HEADS,
    Weights,
    _cpu_step,
    _download,
    _initial_slabs,
    _upload,
)

# Live slots the three requests share. Slot 1 is deliberately owned by nobody, so
# "a slot no request owns keeps every byte" has something to point at.
LIVE_SLOTS = 4
UNOWNED_SLOT = 1
# Semantic rows are request 1 depths 0..2, request 2 depth 0 (an AR peer), request
# 3 depths 0..1. This permutation interleaves the requests and keeps every chain
# ascending, which the layout admits and the wave order then has to honour.
PERMUTATION = (3, 0, 4, 1, 5, 2)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()

pytestmark = pytest.mark.skipif(
    not HIP_AVAILABLE, reason="HIP runtime is not available"
)


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


# -- layout ------------------------------------------------------------------


def _request(
    request_id: int,
    *,
    slot: int,
    start: int = 2,
    root: int = 10,
    candidates=(),
) -> Qwen4ExpMixedRequest:
    return Qwen4ExpMixedRequest(
        request_id=request_id,
        slot=slot,
        owner_token=request_id + 100,
        start_position=start,
        root_token=root,
        candidates=tuple(candidates),
    )


def _layout(*, physical_row_order=None):
    """Two prepared chains of different lengths plus one ordinary AR peer."""

    return compile_mixed_layout(
        [
            _request(1, slot=3, start=2, root=10, candidates=(11, 12)),
            _request(2, slot=0, start=40, root=20),
            _request(3, slot=2, start=9, root=30, candidates=(31,)),
        ],
        physical_row_order=physical_row_order,
    )


def _owner_tokens(layout) -> dict[int, int]:
    return {request.slot: request.owner_token for request in layout.requests}


# -- harness -----------------------------------------------------------------


class Harness:
    """One layer's device weights, the live state owner, and its allocations."""

    def __init__(self, rng: np.random.Generator) -> None:
        from hipengine.core.hip import get_hip_runtime

        self.runtime = get_hip_runtime()
        self.weights = Weights(rng, self.runtime)
        self.allocations: list = []
        self.live = Qwen4ExpBatchGdnState.allocate(
            state_slots=LIVE_SLOTS,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            runtime=self.runtime,
        )

    def seed(self, conv_slab: np.ndarray, recurrent_slab: np.ndarray) -> None:
        copy_host_to_device(
            self.live.conv_state,
            host_array_ptr(np.ascontiguousarray(conv_slab)),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.live.recurrent_state,
            host_array_ptr(np.ascontiguousarray(recurrent_slab)),
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()

    def live_slabs(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _download(
                self.live.conv_state, (LIVE_SLOTS, QKV_WIDTH, KERNEL), np.float32,
                self.runtime,
            ),
            _download(
                self.live.recurrent_state,
                (LIVE_SLOTS, V_HEADS, HEAD_DIM, HEAD_DIM),
                np.float32,
                self.runtime,
            ),
        )

    def upload(self, array: np.ndarray) -> DeviceBuffer:
        return _upload(array, self.runtime, self.allocations)

    def adapter(self, layout) -> Qwen4ExpMixedGdnAdapter:
        # The layout's own tokens are the live admissions here; passing the
        # callback keeps the epoch check on the exercised path.
        return Qwen4ExpMixedGdnAdapter.allocate(
            layout=layout,
            live_state=self.live,
            runtime=self.runtime,
            owner_of=_owner_tokens(layout).get,
        )

    def close(self) -> None:
        self.live.close()
        self.weights.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


@pytest.fixture
def harness():
    built = Harness(np.random.default_rng(5301))
    try:
        yield built
    finally:
        built.close()


def _kernel_pointers(harness: Harness) -> dict[str, int]:
    weights = harness.weights
    return {
        "conv_weight_ptr": weights.d_conv_weight.ptr,
        "dt_bias_ptr": weights.d_dt_bias.ptr,
        "a_ptr": weights.d_a.ptr,
        "norm_weight_ptr": weights.d_norm.ptr,
    }


def _projected_rows(harness: Harness, hidden: np.ndarray) -> Qwen4ExpMixedGdnProjectedRows:
    """The caller's shared input projection: one packed row per physical row.

    The same numpy expressions ``_cpu_step`` uses internally, applied to the whole
    row set at once, which is the sharing the future caller owns.
    """

    arrays = harness.weights.arrays

    def project(name: str) -> np.ndarray:
        # One product per row: a BLAS matmul over the whole set may block rows
        # differently by position, which would hand a permuted layout
        # different input bits for the same row.
        weight = arrays[name].T
        return np.stack([row @ weight for row in hidden]).astype(np.float32)

    return Qwen4ExpMixedGdnProjectedRows(
        qkv=harness.upload(project("attn_qkv")),
        gate=harness.upload(project("attn_gate")),
        alpha=harness.upload(project("ssm_alpha")),
        beta=harness.upload(project("ssm_beta")),
    )


def _run_pass(
    harness: Harness, adapter: Qwen4ExpMixedGdnAdapter, hidden: np.ndarray
) -> None:
    projected = _projected_rows(harness, hidden)
    adapter.run(projected, **_kernel_pointers(harness))
    harness.runtime.device_synchronize()


def _pass_result(
    harness: Harness, adapter: Qwen4ExpMixedGdnAdapter
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gated recurrence output and every provisional snapshot, per row."""

    rows = adapter.plan.total_rows
    core = _download(adapter.core, (rows, CORE_WIDTH), np.float32, harness.runtime)
    conv = _download(
        adapter.working_state.conv_state, (rows, QKV_WIDTH, KERNEL), np.float32,
        harness.runtime,
    )
    recurrent = _download(
        adapter.working_state.recurrent_state,
        (rows, V_HEADS, HEAD_DIM, HEAD_DIM),
        np.float32,
        harness.runtime,
    )
    return core, conv, recurrent


def _oracle_pass(
    weights: Weights,
    adapter: Qwen4ExpMixedGdnAdapter,
    hidden: np.ndarray,
    conv_seed: np.ndarray,
    recurrent_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same pass on the CPU: root copies, parent copies, one step per wave.

    Slots of the CPU provisional slab are physical rows, exactly as on the device,
    and each wave is one ``_cpu_step`` over that wave's rows.
    """

    plan = adapter.plan
    layout = plan.layout
    rows = plan.total_rows
    conv = np.zeros((rows, QKV_WIDTH, KERNEL), np.float32)
    recurrent = np.zeros((rows, V_HEADS, HEAD_DIM, HEAD_DIM), np.float32)
    output = np.zeros((rows, HIDDEN), np.float32)
    for request in layout.requests:
        root = layout.physical_row(request.request_id, 0)
        conv[root] = conv_seed[request.slot]
        recurrent[root] = recurrent_seed[request.slot]
    for wave in plan.waves:
        if wave.depth:
            for row, parent in zip(wave.rows, wave.parent_rows, strict=True):
                conv[row] = conv[parent]
                recurrent[row] = recurrent[parent]
        wave_rows = list(wave.rows)
        wave_output, conv, recurrent = _cpu_step(
            weights, hidden[wave_rows], conv, recurrent, wave_rows
        )
        output[wave_rows] = wave_output
    return output, conv, recurrent


def _assert_live_unchanged(
    harness: Harness, conv_seed: np.ndarray, recurrent_seed: np.ndarray
) -> None:
    live_conv, live_recurrent = harness.live_slabs()
    np.testing.assert_array_equal(
        live_conv.view(np.uint32), conv_seed.view(np.uint32),
        err_msg="a verification pass wrote the live Conv state",
    )
    np.testing.assert_array_equal(
        live_recurrent.view(np.uint32), recurrent_seed.view(np.uint32),
        err_msg="a verification pass wrote the live recurrent state",
    )


def _assert_unowned_slot_unchanged(
    harness: Harness, conv_seed: np.ndarray, recurrent_seed: np.ndarray
) -> None:
    live_conv, live_recurrent = harness.live_slabs()
    np.testing.assert_array_equal(
        live_conv[UNOWNED_SLOT].view(np.uint32),
        conv_seed[UNOWNED_SLOT].view(np.uint32),
        err_msg=f"live slot {UNOWNED_SLOT} is owned by no request but its Conv state moved",
    )
    np.testing.assert_array_equal(
        live_recurrent[UNOWNED_SLOT].view(np.uint32),
        recurrent_seed[UNOWNED_SLOT].view(np.uint32),
        err_msg=f"live slot {UNOWNED_SLOT} is owned by no request but its recurrent state moved",
    )


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())


def _inputs(layout, rng: np.random.Generator):
    conv_seed, recurrent_seed = _initial_slabs(rng)
    hidden = rng.normal(0.0, 0.1, size=(layout.total_rows, HIDDEN)).astype(np.float32)
    return conv_seed, recurrent_seed, hidden


# -- one pass ----------------------------------------------------------------


@pytest.mark.parametrize(
    "physical_row_order",
    [None, PERMUTATION],
    ids=["semantic", "interleaved"],
)
def test_mixed_gdn_pass_matches_the_cpu_oracle(harness, physical_row_order) -> None:
    layout = _layout(physical_row_order=physical_row_order)
    rng = np.random.default_rng(6101)
    conv_seed, recurrent_seed, hidden = _inputs(layout, rng)
    harness.seed(conv_seed, recurrent_seed)
    adapter = harness.adapter(layout)
    try:
        # Root state comes from the request's own live slot; the plan is the map.
        for request in layout.requests:
            root = adapter.plan.snapshot_row(request.request_id, 0)
            assert adapter.plan.live_slot(request.request_id) == request.slot
            assert adapter.recurrent_snapshot_ptr(root) == (
                adapter.working_state.recurrent_slot_ptr(root)
            )
            assert adapter.conv_snapshot_ptr(root) == (
                adapter.working_state.conv_slot_ptr(root)
            )

        _run_pass(harness, adapter, hidden)
        core, conv, recurrent = _pass_result(harness, adapter)
        expected_output, expected_conv, expected_recurrent = _oracle_pass(
            harness.weights, adapter, hidden, conv_seed, recurrent_seed
        )
        # The caller's shared output projection, applied on the CPU for both sides.
        actual_output = core @ harness.weights.arrays["ssm_out"].T

        assert np.isfinite(core).all()
        assert np.isfinite(conv).all()
        assert np.isfinite(recurrent).all()
        np.testing.assert_allclose(
            actual_output, expected_output, rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )
        np.testing.assert_allclose(conv, expected_conv, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        np.testing.assert_allclose(
            recurrent, expected_recurrent, rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )
        # A provisional pass decides nothing: the live owner keeps every byte.
        _assert_live_unchanged(harness, conv_seed, recurrent_seed)
        print(
            f"rows={layout.total_rows} waves={[w.width for w in adapter.plan.waves]} "
            f"order={physical_row_order} "
            f"max|core-output|={np.abs(actual_output).max():.3e} "
            f"max|oracle-output|={_max_abs(actual_output, expected_output):.3e} "
            f"max|oracle-state|={_max_abs(recurrent, expected_recurrent):.3e}"
        )
    finally:
        adapter.close()


def test_physical_permutation_does_not_change_row_results(harness) -> None:
    rng = np.random.default_rng(6111)
    semantic_layout = _layout()
    interleaved_layout = _layout(physical_row_order=PERMUTATION)
    conv_seed, recurrent_seed, hidden = _inputs(semantic_layout, rng)
    harness.seed(conv_seed, recurrent_seed)

    results = {}
    for name, layout in (
        ("semantic", semantic_layout),
        ("interleaved", interleaved_layout),
    ):
        adapter = harness.adapter(layout)
        try:
            physical_hidden = hidden[[row.semantic_row for row in layout.rows]]
            _run_pass(harness, adapter, physical_hidden)
            results[name] = (adapter, _pass_result(harness, adapter))
        finally:
            adapter.close()
        # Both passes read the same live seed, which neither may change.
        _assert_live_unchanged(harness, conv_seed, recurrent_seed)

    semantic, (semantic_core, semantic_conv, semantic_recurrent) = results["semantic"]
    interleaved, (interleaved_core, interleaved_conv, interleaved_recurrent) = results[
        "interleaved"
    ]
    # The permutation has to actually move rows, or this test proves nothing.
    assert any(
        semantic.plan.snapshot_row(request.request_id, depth)
        != interleaved.plan.snapshot_row(request.request_id, depth)
        for request in semantic_layout.requests
        for depth in range(request.rows)
    )
    for request in semantic_layout.requests:
        for depth in range(request.rows):
            semantic_row = semantic.plan.snapshot_row(request.request_id, depth)
            interleaved_row = interleaved.plan.snapshot_row(request.request_id, depth)
            np.testing.assert_array_equal(
                semantic_recurrent[semantic_row].view(np.uint32),
                interleaved_recurrent[interleaved_row].view(np.uint32),
                err_msg=(
                    f"request {request.request_id} depth {depth} recurrent snapshot "
                    "depends on physical placement"
                ),
            )
            np.testing.assert_array_equal(
                semantic_conv[semantic_row].view(np.uint32),
                interleaved_conv[interleaved_row].view(np.uint32),
                err_msg=(
                    f"request {request.request_id} depth {depth} Conv snapshot "
                    "depends on physical placement"
                ),
            )
            np.testing.assert_array_equal(
                semantic_core[semantic_row].view(np.uint32),
                interleaved_core[interleaved_row].view(np.uint32),
                err_msg=(
                    f"request {request.request_id} depth {depth} output depends on "
                    "physical placement"
                ),
            )


def test_discarding_a_provisional_pass_leaves_no_residue(harness) -> None:
    layout = _layout(physical_row_order=PERMUTATION)
    rng = np.random.default_rng(6121)
    conv_seed, recurrent_seed, hidden_a = _inputs(layout, rng)
    hidden_b = rng.normal(0.0, 0.1, size=(layout.total_rows, HIDDEN)).astype(np.float32)
    harness.seed(conv_seed, recurrent_seed)
    adapter = harness.adapter(layout)
    try:
        _run_pass(harness, adapter, hidden_a)
        first_core, first_conv, first_recurrent = _pass_result(harness, adapter)
        assert adapter.verification_pending is True

        # An abandoned pass with different inputs, discarded without committing.
        _run_pass(harness, adapter, hidden_b)
        other_core, other_conv, other_recurrent = _pass_result(harness, adapter)
        assert not np.array_equal(
            other_recurrent.view(np.uint32), first_recurrent.view(np.uint32)
        ), "different inputs produced identical snapshots; the discard case is untested"
        _assert_live_unchanged(harness, conv_seed, recurrent_seed)

        # Rerunning the first pass reproduces it byte for byte: the discarded pass
        # left no state that a fresh pass inherits.
        _run_pass(harness, adapter, hidden_a)
        repeat_core, repeat_conv, repeat_recurrent = _pass_result(harness, adapter)
        np.testing.assert_array_equal(
            repeat_core.view(np.uint32), first_core.view(np.uint32)
        )
        np.testing.assert_array_equal(
            repeat_conv.view(np.uint32), first_conv.view(np.uint32)
        )
        np.testing.assert_array_equal(
            repeat_recurrent.view(np.uint32), first_recurrent.view(np.uint32)
        )
        _assert_live_unchanged(harness, conv_seed, recurrent_seed)

        # The surviving pass is the one that commits.
        adapter.commit([0, 0, 0])
        live_conv, live_recurrent = harness.live_slabs()
        for request in layout.requests:
            np.testing.assert_array_equal(
                live_recurrent[request.slot].view(np.uint32),
                repeat_recurrent[
                    layout.physical_row(request.request_id, 0)
                ].view(np.uint32),
            )
            np.testing.assert_array_equal(
                live_conv[request.slot].view(np.uint32),
                repeat_conv[layout.physical_row(request.request_id, 0)].view(np.uint32),
            )
        _assert_unowned_slot_unchanged(harness, conv_seed, recurrent_seed)
    finally:
        adapter.close()


# -- commit ------------------------------------------------------------------


@pytest.mark.parametrize(
    "accepted",
    [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (0, 0, 1),
        (2, 0, 1),
    ],
    ids=["roots", "one-of-two", "two-of-two", "peer-plus-one", "full"],
)
def test_commit_applies_exactly_the_accepted_prefix(harness, accepted) -> None:
    layout = _layout(physical_row_order=PERMUTATION)
    rng = np.random.default_rng(6201)
    conv_seed, recurrent_seed, hidden = _inputs(layout, rng)
    harness.seed(conv_seed, recurrent_seed)
    adapter = harness.adapter(layout)
    try:
        _run_pass(harness, adapter, hidden)
        _, snapshot_conv, snapshot_recurrent = _pass_result(harness, adapter)
        _assert_live_unchanged(harness, conv_seed, recurrent_seed)

        destinations = adapter.commit(list(accepted))
        assert destinations.accepted_counts == accepted
        assert destinations.commit_rows == tuple(
            layout.physical_row(request.request_id, count)
            for request, count in zip(layout.requests, accepted, strict=True)
        )

        live_conv, live_recurrent = harness.live_slabs()
        for request, count in zip(layout.requests, accepted, strict=True):
            row = layout.physical_row(request.request_id, count)
            # A commit is a device-to-device copy, so the live slot must hold the
            # snapshot's bytes exactly.
            np.testing.assert_array_equal(
                live_recurrent[request.slot].view(np.uint32),
                snapshot_recurrent[row].view(np.uint32),
                err_msg=(
                    f"request {request.request_id} accepted {count} but its live "
                    "recurrent state is not that snapshot"
                ),
            )
            np.testing.assert_array_equal(
                live_conv[request.slot].view(np.uint32),
                snapshot_conv[row].view(np.uint32),
                err_msg=(
                    f"request {request.request_id} accepted {count} but its live "
                    "Conv state is not that snapshot"
                ),
            )
            if count < request.depth:
                # The shallowest and deepest snapshots are distinct, so accepting a
                # prefix cannot be confused with accepting the whole chain.
                deepest = layout.physical_row(request.request_id, request.depth)
                assert not np.array_equal(
                    live_recurrent[request.slot].view(np.uint32),
                    snapshot_recurrent[deepest].view(np.uint32),
                ), (
                    f"request {request.request_id} accepted {count} of "
                    f"{request.depth} but its live state equals the deepest snapshot"
                )
        # A live slot no request owns is never a commit destination.
        _assert_unowned_slot_unchanged(harness, conv_seed, recurrent_seed)
    finally:
        adapter.close()
