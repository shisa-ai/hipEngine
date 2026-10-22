"""Device-resident sparse selection handoff for the N-row QSA mixer.

The c1 owner selects entirely on device: ``select_positions_device`` derives the
selected count analytically and hands its own ``selected_positions`` buffer to
the attention kernel, so a sparse decode step performs no host round-trip. The
batch owner's ``select_active`` published the same selection through host
metadata instead, which cost two blocking copies per QSA layer per step once a
row passed the dense-equivalence bound.

These tests pin both halves of the fix, at the dispatch level, against a fake
runtime:

* ``select_active_device`` performs the identical pool/score/select dispatch
  sequence and returns the owner's own device pointers with the owner's own
  ``output_stride``, and it reads nothing back to the host.
* the mixed mixer hands those device pointers to the rows attention kernel when
  every active row is sparse, and still stages host metadata for the mixed and
  dense-only row sets it keeps supporting.

``select_active`` keeps its host interface, so the readback path stays covered
here as well: a diagnostic caller and the mixed path must not change behaviour.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_qsa as batch_qsa
import hipengine.runtime.qwen4_exp_batch_qsa_index as batch_index
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_qsa_mixed_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import (
    Qwen4ExpBatchQsaDeviceSelection,
    Qwen4ExpBatchQsaIndexState,
)
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
)

SLOTS = 4
MAX_POSITIONS = 16
BLOCK_SIZE = 256
HIDDEN = 8
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
ROTARY_DIM = 4
THETA = 100.0
INDEX_HEADS = 2
INDEX_DIM = 4
INDEX_ROTARY_DIM = 4
RATIO = 2
BUDGET = 3
OUTPUT_STRIDE = BUDGET * RATIO + RATIO - 1
DENSE_POSITION = 3  # context 4 -> 2 complete blocks
SPARSE_POSITION = 8  # context 9 -> 4 complete blocks
QUERIES = 0x9000
NORM_KEY = 0xA000


class FakeRuntime:
    """Allocation bookkeeping and copy accounting only: no device work."""

    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, object]] = []
        self.syncs: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        # The owners copy into interior offsets of a slab, so check containment.
        def owned(pointer: int) -> bool:
            return any(
                base <= pointer < base + size
                for base, size in self.allocations.items()
            )

        assert owned(dst) or owned(src), (dst, src)
        self.copies.append((dst, int(nbytes), kind))

    def stream_synchronize(self, stream: int) -> None:
        self.syncs.append(int(stream))


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def stage(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def all(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for label, args, kwargs in self.calls if label == name]

    def one(self, name: str) -> tuple[tuple, dict]:
        matches = self.all(name)
        assert len(matches) == 1, f"expected one {name} dispatch, got {len(matches)}"
        return matches[0]


_INDEX_KERNELS = (
    "qwen4_exp_qsa_scatter_index_key_device_position_f32",
    "qwen4_exp_qsa_pool_norm_rope_f32",
    "qwen4_exp_qsa_score_f32",
    "qwen4_exp_qsa_topk_expand_rows_f32_i64",
)
_MIXER_KERNELS = (
    "qwen4_exp_qsa_split_norm_rope_rows_f32",
    "qwen4_exp_qsa_split_norm_mrope_rows_f32",
    "qwen4_exp_qsa_norm_rope_rows_f32",
    "qwen4_exp_qsa_norm_mrope_rows_f32",
    "qwen35_write_paged_kv_f32_batch_spans",
    "qwen35_paged_full_attn_decode_context_bf16_batch_spans",
    "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32",
    "qwen4_exp_qsa_gate_context_f32",
)


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    record = Recorder()
    for name in _MIXER_KERNELS:
        monkeypatch.setattr(batch_qsa, name, record.stage(name))
    monkeypatch.setattr(batch_qsa, "launch_gguf_linear", record.stage("projection"))
    for name in _INDEX_KERNELS:
        monkeypatch.setattr(batch_index, name, record.stage(name))
    return record


class Fixture:
    def __init__(self, *, selection_stride: int = OUTPUT_STRIDE) -> None:
        self.runtime = FakeRuntime()
        self.state = Qwen4ExpBatchAttentionState.allocate(
            state_slots=SLOTS,
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=self.runtime,
        )
        self.index = Qwen4ExpBatchQsaIndexState.allocate(
            attention_state=self.state,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=self.runtime,
        )
        self.selection = Qwen4ExpBatchQsaSelectionScratch.allocate(
            rows=SLOTS,
            stride=selection_stride,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpQSAScratch.allocate(
            rows=SLOTS,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=self.runtime,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
        )
        self.weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: f"{name}-weight"
                for name in (*batch_qsa._REQUIRED_PROJECTIONS, *batch_qsa._INDEX_PROJECTIONS)
            },
            q_norm_weight_ptr=0x900,
            k_norm_weight_ptr=0x904,
            index_q_norm_weight_ptr=0x908,
            index_k_norm_weight_ptr=0x90C,
        )
        self.mixed = DeviceBuffer(ptr=0x1000, nbytes=SLOTS * HIDDEN * 4)
        self.rows = 0

    def admit(self, positions, slots=None) -> None:
        row_slots = list(range(len(positions))) if slots is None else list(slots)
        for slot, position in zip(row_slots, positions):
            self.index.cursor_host[slot] = position
            self.index.pooled_count_host[slot] = position // RATIO
        self.state.prepare(row_slots, positions)
        self.index.prepare(row_slots)
        self.rows = len(row_slots)

    def close(self) -> None:
        self.scratch.close()
        self.selection.close()
        self.index.close()
        self.state.close()


@pytest.fixture
def fixture() -> Fixture:
    built = Fixture()
    try:
        yield built
    finally:
        built.close()


def _run(fixture: Fixture, **overrides) -> DeviceBuffer:
    kwargs = dict(
        attention_state=fixture.state,
        index_state=fixture.index,
        selection=fixture.selection,
        scratch=fixture.scratch,
        rows=fixture.rows,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        theta=THETA,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        index_rotary_dim=INDEX_ROTARY_DIM,
        runtime=fixture.runtime,
    )
    kwargs.update(overrides)
    weights = kwargs.pop("weights", fixture.weights)
    return run_qwen4_exp_batch_qsa_mixed_token_mixer(fixture.mixed.ptr, weights, **kwargs)


def _write(dst_ptr: int, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    ctypes.memmove(dst_ptr, contiguous.ctypes.data, contiguous.nbytes)


def _fake_readback(monkeypatch, *, counts: np.ndarray, positions: np.ndarray) -> None:
    rows = counts.size

    def fake_copy(dst_ptr, src, nbytes, *, runtime=None):
        if nbytes == rows * 4:
            _write(dst_ptr, counts.astype(np.int32))
        elif nbytes == positions.nbytes:
            _write(dst_ptr, positions)
        else:  # pragma: no cover - guards an unexpected read-back size
            raise AssertionError(f"unexpected read-back of {nbytes} bytes")

    monkeypatch.setattr(batch_index, "copy_device_to_host", fake_copy)


def _forbid_readback(monkeypatch) -> None:
    """Fail the test if the owner reads anything back to the host."""

    def explode(dst_ptr, src, nbytes, *, runtime=None):  # pragma: no cover - RED guard
        raise AssertionError(
            f"device-only selection performed a {nbytes}-byte device-to-host copy"
        )

    monkeypatch.setattr(batch_index, "copy_device_to_host", explode)


def _selection_payload_bytes(fixture: Fixture, rows: int) -> int:
    return rows * fixture.index.output_stride * np.dtype(np.int64).itemsize


# -- index owner: device-only selection -----------------------------------


def test_select_active_device_reads_nothing_back_and_returns_owner_pointers(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """The device form keeps the owner's own buffers and stride."""

    _forbid_readback(monkeypatch)
    fixture.admit([SPARSE_POSITION, SPARSE_POSITION + 1, SPARSE_POSITION + 2])
    selection = fixture.index.select_active_device(
        QUERIES,
        key_norm_weight_ptr=NORM_KEY,
        rotary_dim=INDEX_DIM,
        theta=THETA,
    )

    assert isinstance(selection, Qwen4ExpBatchQsaDeviceSelection)
    assert selection.positions_ptr == fixture.index.selected_positions.ptr
    assert selection.counts_ptr == fixture.index.selected_counts.ptr
    assert selection.rows == 3
    # The owner's own stride, which is what the topk kernel wrote.
    assert selection.stride == fixture.index.output_stride == OUTPUT_STRIDE
    assert selection.slot_ids == (0, 1, 2)
    assert selection.row_ptr(1) == (
        fixture.index.selected_positions.ptr + OUTPUT_STRIDE * 8
    )
    with pytest.raises(IndexError, match="outside"):
        selection.row_ptr(3)

    # Same dispatch sequence as the host form: one score per row, one selection.
    assert len(recorder.all("qwen4_exp_qsa_score_f32")) == 3
    (topk_args, _) = recorder.one("qwen4_exp_qsa_topk_expand_rows_f32_i64")
    assert topk_args[2] == fixture.index.selected_positions.ptr
    assert topk_args[3] == fixture.index.selected_counts.ptr
    assert topk_args[4] == 3
    assert topk_args[6] == OUTPUT_STRIDE
    assert recorder.names()[-1] == "qwen4_exp_qsa_topk_expand_rows_f32_i64"


def test_select_active_device_does_not_synchronize_the_caller_stream(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """Ordering comes from the shared stream, not from a drain."""

    _forbid_readback(monkeypatch)
    fixture.admit([SPARSE_POSITION])
    fixture.index.select_active_device(
        QUERIES,
        key_norm_weight_ptr=NORM_KEY,
        rotary_dim=INDEX_DIM,
        theta=THETA,
        stream=7,
    )
    assert fixture.runtime.syncs == []
    (topk_args, topk_kwargs) = recorder.one("qwen4_exp_qsa_topk_expand_rows_f32_i64")
    assert topk_kwargs["stream"] == 7


def test_select_active_still_synchronizes_and_returns_host_metadata(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """The diagnostic/mixed interface is unchanged, including its drain."""

    fixture.admit([SPARSE_POSITION, SPARSE_POSITION + 1])
    expected = np.arange(2 * OUTPUT_STRIDE, dtype=np.int64).reshape(2, OUTPUT_STRIDE)
    _fake_readback(monkeypatch, counts=np.array([5, 3], dtype=np.int32), positions=expected)
    counts, positions = fixture.index.select_active(
        QUERIES,
        key_norm_weight_ptr=NORM_KEY,
        rotary_dim=INDEX_DIM,
        theta=THETA,
        stream=7,
    )
    assert fixture.runtime.syncs == [7]
    assert counts.tolist() == [5, 3]
    assert positions.shape == (2, 5)
    assert positions.tolist() == expected[:, :5].tolist()
    assert len(recorder.all("qwen4_exp_qsa_score_f32")) == 2


def test_select_active_device_keeps_the_dense_row_refusal(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    _forbid_readback(monkeypatch)
    fixture.admit([DENSE_POSITION])
    with pytest.raises(ValueError, match="not more than the budget 3"):
        fixture.index.select_active_device(
            QUERIES, key_norm_weight_ptr=NORM_KEY, rotary_dim=INDEX_DIM, theta=THETA
        )
    assert recorder.calls == []


def test_select_active_device_rejects_a_closed_owner(fixture: Fixture, monkeypatch) -> None:
    _forbid_readback(monkeypatch)
    fixture.admit([SPARSE_POSITION])
    fixture.index.close()
    with pytest.raises(RuntimeError, match="closed"):
        fixture.index.select_active_device(
            QUERIES, key_norm_weight_ptr=NORM_KEY, rotary_dim=INDEX_DIM, theta=THETA
        )


# -- mixer: all-sparse rows use the device handoff -------------------------


def test_all_sparse_rows_attend_from_the_index_owners_device_buffers(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """No staging, no readback: the kernel reads the owner's own selection."""

    _forbid_readback(monkeypatch)
    fixture.admit([SPARSE_POSITION, SPARSE_POSITION + 1, SPARSE_POSITION + 2])
    counts_before = fixture.selection.counts_host.copy()
    positions_before = fixture.selection.positions_host.copy()
    copies_before = len(fixture.runtime.copies)
    payload = _selection_payload_bytes(fixture, 3)

    _run(fixture)

    (attention_args, attention_kwargs) = recorder.one(
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
    )
    assert attention_args[3] == fixture.index.selected_positions.ptr
    assert attention_args[4] == fixture.index.selected_counts.ptr
    assert attention_kwargs["rows"] == 3
    # The owner's stride, not the selection scratch's.
    assert attention_kwargs["selected_stride"] == fixture.index.output_stride

    step_copies = fixture.runtime.copies[copies_before:]
    assert all(
        not (fixture.selection.positions.ptr <= dst < fixture.selection.positions.ptr
             + fixture.selection.positions.nbytes)
        for dst, _, _ in step_copies
    ), "the all-sparse step must not upload selection positions"
    assert all(nbytes != payload for _, nbytes, _ in step_copies), (
        "the all-sparse step must not move a selection-sized payload"
    )
    # The staging mirrors are untouched, so a later dense step still restores.
    assert np.array_equal(fixture.selection.counts_host, counts_before)
    assert np.array_equal(fixture.selection.positions_host, positions_before)
    assert fixture.selection.row_is_arange.all()

    # The index append still ran for every row, and the owner holds the full set.
    assert len(recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")) == 3
    assert fixture.index.active.slot_ids == fixture.state.active.slot_ids == (0, 1, 2)
    assert fixture.index.cursor_host[:3].tolist() == [
        SPARSE_POSITION + 1,
        SPARSE_POSITION + 2,
        SPARSE_POSITION + 3,
    ]


def test_all_sparse_rows_keep_the_callers_row_order_for_permuted_slots(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    _forbid_readback(monkeypatch)
    fixture.admit(
        [SPARSE_POSITION, SPARSE_POSITION + 1, SPARSE_POSITION + 2],
        slots=[2, 0, 3],
    )
    _run(fixture)

    (attention_args, attention_kwargs) = recorder.one(
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
    )
    assert attention_kwargs["rows"] == 3
    assert attention_args[3] == fixture.index.selected_positions.ptr
    # Row r of the owner's buffers is slot 2, 0, 3 in that order.
    (topk_args, _) = recorder.one("qwen4_exp_qsa_topk_expand_rows_f32_i64")
    assert topk_args[4] == 3
    assert fixture.index.active.slot_ids == (2, 0, 3)


def test_device_handoff_uses_the_owner_stride_not_a_wider_selection_stride(
    recorder: Recorder, monkeypatch
) -> None:
    """A selection scratch wider than the owner's stride must not leak into it."""

    _forbid_readback(monkeypatch)
    wide = Fixture(selection_stride=OUTPUT_STRIDE + 5)
    try:
        wide.admit([SPARSE_POSITION, SPARSE_POSITION + 1])
        _run(wide)
        (attention_args, attention_kwargs) = recorder.one(
            "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
        )
        assert wide.selection.stride == OUTPUT_STRIDE + 5
        assert attention_kwargs["selected_stride"] == wide.index.output_stride
        assert attention_args[3] == wide.index.selected_positions.ptr
    finally:
        wide.close()


# -- mixer: the mixed and dense-only paths keep host staging ---------------


def test_mixed_row_set_still_stages_host_metadata(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    fixture.admit([DENSE_POSITION, SPARSE_POSITION, SPARSE_POSITION + 1])
    subset_positions = np.array(
        [[8, 7, 6, 5, 4, 3, 2], [9, 8, 7, 6, 5, 4, 3]], dtype=np.int64
    )
    _fake_readback(
        monkeypatch, counts=np.array([5, 3], dtype=np.int32), positions=subset_positions
    )
    _run(fixture)

    (attention_args, attention_kwargs) = recorder.one(
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
    )
    # The batch-wide staging buffer, at the scratch stride, as before.
    assert attention_args[3] == fixture.selection.positions.ptr
    assert attention_args[4] == fixture.selection.counts.ptr
    assert attention_kwargs["selected_stride"] == fixture.selection.stride
    assert fixture.selection.counts_host[:3].tolist() == [DENSE_POSITION + 1, 5, 3]
    assert fixture.selection.positions_host[1, :5].tolist() == [8, 7, 6, 5, 4]


def test_dense_only_row_set_still_stages_host_metadata(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    fixture.admit([DENSE_POSITION, DENSE_POSITION + 1])
    _run(fixture)

    (attention_args, attention_kwargs) = recorder.one(
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
    )
    assert attention_args[3] == fixture.selection.positions.ptr
    assert attention_args[4] == fixture.selection.counts.ptr
    assert attention_kwargs["selected_stride"] == fixture.selection.stride
    assert fixture.selection.counts_host[:2].tolist() == [
        DENSE_POSITION + 1,
        DENSE_POSITION + 2,
    ]
    assert recorder.all("qwen4_exp_qsa_topk_expand_rows_f32_i64") == []


def test_a_sparse_step_leaves_the_staging_mirrors_usable_for_a_dense_step(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """The device handoff must not leave the staging mirrors in a bad state."""

    _forbid_readback(monkeypatch)
    fixture.admit([SPARSE_POSITION])
    _run(fixture)
    (sparse_args, _) = recorder.one("qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32")
    assert sparse_args[3] == fixture.index.selected_positions.ptr
    assert fixture.selection.counts_host.tolist() == [0] * SLOTS
    recorder.calls.clear()

    # The next step's row is dense: the batch-wide staging buffer serves it, and
    # the untouched mirrors mean no restore of a sparse row is needed.
    fixture.admit([DENSE_POSITION])
    _run(fixture)
    (dense_args, _) = recorder.one("qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32")
    assert dense_args[3] == fixture.selection.positions.ptr
    assert fixture.selection.counts_host[:1].tolist() == [DENSE_POSITION + 1]
    assert fixture.selection.positions_host[0, : DENSE_POSITION + 1].tolist() == list(
        range(DENSE_POSITION + 1)
    )
    assert fixture.selection.row_is_arange.all()


def test_all_sparse_failure_propagates_and_dispatches_no_attention(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """A selection failure keeps today's containment: no attention, no rollback."""

    def explode(*args, **kwargs):
        raise RuntimeError("injected selection failure")

    monkeypatch.setattr(batch_index, "qwen4_exp_qsa_topk_expand_rows_f32_i64", explode)
    fixture.admit([SPARSE_POSITION, SPARSE_POSITION + 1])
    with pytest.raises(RuntimeError, match="injected selection failure"):
        _run(fixture)
    assert recorder.all("qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32") == []
    assert recorder.all("qwen4_exp_qsa_gate_context_f32") == []
    # The KV write and the index append already ran, so the step is not
    # transactional: the step owner rolls the state back through its own token.
    assert len(recorder.all("qwen35_write_paged_kv_f32_batch_spans")) == 1
    assert len(recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")) == 2
    # The owner was never restored to the full row set, because the failure
    # escaped before the restore; recovery is the caller's rollback.
    assert fixture.index.active.slot_ids == (0, 1)
