"""Packed decode-graph capture must preserve a non-identity packed page map.

A packed KV workspace backed by a ``GlobalDeviceKVPool`` lease hands
``_GGUFPackedTargetState`` arbitrary arena ``page_ids``, so
``_rebind_packed_verify_layout_pages`` rewrites the layout's slot-local identity
block table into the leased physical pages. The packed KV write and the paged
decode attention both read that table through ``append_spans``/``prefill_spans``,
so it is the authoritative mapping for the whole graph generation.

The in-graph metadata refresh kernel (``prepare_packed_decode_metadata_from_
positions``) unconditionally writes a slot-local identity table. That is correct
only when the page map already is the identity the kernel writes. These tests pin
that capture keeps the page-mapped table intact when it is not, and that the
identity path is unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipengine.runtime.gguf_packed_decode_graph as packed_graph
from hipengine.runtime.qwen35_gguf_graph_lifetime import Qwen35GGUFGraphLifetime
from hipengine.core.memory import DeviceBuffer

_BLOCK_TABLE_PTR = 0x401
_FIRST_MALLOC_PTR = 0x700


class _FakeDevice:
    """Pointer-addressed host arrays standing in for device allocations."""

    def __init__(self) -> None:
        self.arrays: dict[int, np.ndarray] = {}
        self.frees: list[int] = []

    def register(self, ptr: int, shape: tuple[int, ...], dtype) -> None:
        self.arrays[ptr] = np.zeros(shape, dtype=dtype)

    def read(self, ptr: int) -> np.ndarray:
        return self.arrays[ptr].copy()


def _identity_table(rows: int, blocks_per_slot: int) -> np.ndarray:
    return np.arange(rows * blocks_per_slot, dtype=np.int32).reshape(
        rows, blocks_per_slot
    )


def _capture(
    monkeypatch,
    *,
    page_ids: tuple[int, ...],
    backing_kind: str,
    model_step=None,
):
    """Capture a two-slot packed graph against a fake device and record writes."""

    device = _FakeDevice()
    metadata_calls: list[tuple[int, int, int, int | None]] = []
    manifest_kwargs: dict[str, object] = {}
    frees: list[int] = []
    mallocs: list[tuple[int, int]] = []
    next_ptr = [_FIRST_MALLOC_PTR]

    def fake_malloc(nbytes: int, *, runtime=None) -> DeviceBuffer:
        ptr = next_ptr[0]
        next_ptr[0] += 0x100
        device.register(ptr, (int(nbytes),), np.uint8)
        mallocs.append((ptr, int(nbytes)))
        return DeviceBuffer(ptr, int(nbytes))

    def fake_free(buffer, *, runtime=None) -> None:
        frees.append(int(buffer.ptr))
        device.arrays.pop(int(buffer.ptr), None)

    monkeypatch.setattr(packed_graph, "malloc", fake_malloc)
    monkeypatch.setattr(packed_graph, "free", fake_free)
    monkeypatch.setattr(
        packed_graph,
        "copy_host_array_to_device",
        lambda buffer, array, nbytes, **kwargs: None,
    )
    monkeypatch.setattr(packed_graph, "replace", lambda obj, **kwargs: obj)
    monkeypatch.setattr(
        packed_graph, "_packed_graph_buffer_ptrs", lambda *a, **k: (0x3000,)
    )
    monkeypatch.setattr(
        packed_graph,
        "build_qwen35_gguf_packed_decode_graph_key",
        lambda *a, **k: type(
            "_Key",
            (),
            {"as_dict": lambda self: {"stub": True}, "key_sha256": "stub"},
        )(),
    )

    def fake_manifest(**kwargs):
        manifest_kwargs.update(kwargs)
        return {"layer_families": {"sampler": {}}}

    monkeypatch.setattr(
        packed_graph, "build_packed_decode_execution_manifest", fake_manifest
    )
    monkeypatch.setattr(
        packed_graph,
        "create_graph_submission",
        lambda **kwargs: type(
            "_Submission",
            (),
            {
                "graph_exec": 0x55,
                "close": lambda self: None,
                "provenance": lambda self: {"stub": True},
                "launch": lambda self, stream: None,
            },
        )(),
    )

    def metadata_kernel(
        block_table_ptr,
        positions_ptr,
        contexts_ptr,
        cu_q_ptr,
        cu_k_ptr,
        atomic_ptr,
        gdn_cu_ptr,
        state_indices_ptr,
        rows,
        blocks_per_slot,
        *,
        active_mask_u8_ptr=None,
        stream=0,
        library=None,
        runtime=None,
    ) -> None:
        # Mirrors prepare_packed_decode_metadata_from_positions_kernel: a
        # slot-local identity table for every active row.
        metadata_calls.append(
            (int(block_table_ptr), int(rows), int(blocks_per_slot), active_mask_u8_ptr)
        )
        target = device.arrays[int(block_table_ptr)]
        if target.shape != (int(rows), int(blocks_per_slot)):
            target = np.ascontiguousarray(target).view(np.int32).reshape(
                int(rows), int(blocks_per_slot)
            )
            device.arrays[int(block_table_ptr)] = target
        target[...] = _identity_table(int(rows), int(blocks_per_slot))

    monkeypatch.setattr(
        packed_graph,
        "_resolve_packed_graph_kernels",
        lambda *a, **k: (
            metadata_kernel,
            lambda *a, **k: None,
            lambda *a, **k: None,
        ),
    )

    runtime = type(
        "_Runtime",
        (),
        {
            "stream_create": lambda self: 0x31,
            "stream_synchronize": lambda self, stream: None,
            "stream_begin_capture": lambda self, stream: None,
            "stream_end_capture": lambda self, stream: 0x77,
            "stream_destroy": lambda self, stream: None,
            "graph_destroy": lambda self, graph: None,
            "memset": lambda self, *a, **k: None,
        },
    )()

    packed_state = type(
        "_State",
        (),
        {
            "slot_count": 2,
            "blocks_per_slot": 4,
            "page_ids": tuple(int(page) for page in page_ids),
            "kv_backing_kind": backing_kind,
            "layer_conv_states": (),
            "layer_recurrent_states": (),
        },
    )()

    block_table = DeviceBuffer(_BLOCK_TABLE_PTR, 2 * 4 * 4)
    device.register(_BLOCK_TABLE_PTR, (2, 4), np.int32)
    packed_scratch = type(
        "_Scratch",
        (),
        {
            "block_size": 256,
            "max_positions": 1024,
            "block_table": block_table,
            "positions": DeviceBuffer(0x402, 16),
            "context_counts": DeviceBuffer(0x403, 16),
            "cu_q": DeviceBuffer(0x404, 8),
            "cu_k": DeviceBuffer(0x405, 8),
            "atomic": DeviceBuffer(0x406, 4),
            "gdn_cu_seqlens": DeviceBuffer(0x407, 12),
            "gdn_state_indices": DeviceBuffer(0x408, 16),
            "append_spans": type("_Spans", (), {"max_live_count": 0})(),
            "prefill_spans": type("_Spans", (), {"max_live_count": 0})(),
            "buffers": (),
        },
    )()

    uploaded_layouts: list[np.ndarray] = []

    def for_packed_verify_layout(layout, *, runtime, stream=0, metadata_prepare_fn=None):
        # Host-upload branch of the real scratch preparation: the rebound table
        # is what the paged KV write and decode attention must read.
        uploaded_layouts.append(np.asarray(layout.block_table).copy())
        device.arrays[_BLOCK_TABLE_PTR][...] = layout.block_table
        return packed_scratch

    scratch_base = SimpleNamespace()
    scratch_base.for_packed_verify_layout = for_packed_verify_layout

    owner = type(
        "_Owner",
        (),
        {
            "runner": type(
                "_Runner",
                (),
                {
                    "backend": "hip_gfx1100",
                    "target_arch": "gfx1100",
                    "hidden_size": 64,
                    "vocab_size": 128,
                    "_packed_decode_metadata_kernel": lambda self: metadata_kernel,
                    "weights": type(
                        "_Weights",
                        (),
                        {
                            "weights": (),
                            "config": type(
                                "_Config",
                                (),
                                {
                                    "layer_types": ("full_attention",),
                                    "is_moe": False,
                                    "expert_used_count": 0,
                                },
                            )(),
                        },
                    )(),
                    "_linear_attention_decode_batch_plan": lambda self: type(
                        "_Plan", (), {"gdn_decode_path": "indexed_batch"}
                    )(),
                },
            )(),
            "scratch": type("_OwnerScratch", (), {})(),
            "runtime": runtime,
            "host_token_embedding_enabled": False,
            "kv_storage_dtype": type("_DType", (), {"value": "bf16"})(),
            "kv_storage_layout": "uniform",
            "kv_scale_dtype": type("_ScaleDType", (), {"value": "fp16"})(),
            "kv_scale_granularity": "per_token_head",
            "use_expert_sidecar": False,
            "use_wmma_prefill": True,
            "use_gemv_decode": True,
            "_lm_head_threads": 128,
            "_lm_head_stage1_blocks": 970,
            "_packed_decode_state_dirty": False,
            "_prefill_token_buf": DeviceBuffer(0x1000, 16),
            "_bulk_prefill_scratch": object(),
            "_prefill_hidden_a": None,
            "_prefill_hidden_b": None,
            "_verify_logits_buf": None,
            "_verify_lm_block_values": None,
            "_verify_lm_block_indices_i32": None,
            "_verify_lm_out_indices_i32": DeviceBuffer(0x2000, 8),
            "_verify_lm_out_values": None,
            "_runtime_state_library": object(),
            "_last_packed_lm_head_decode_path": "stub",
            "_last_packed_sampler_decode_path": "stub",
            "_graph_lifetime": Qwen35GGUFGraphLifetime(),
            "_normalize_layer_output_capture": lambda self, value: tuple(value),
            "_sync_packed_decode_initial_state": lambda self, *a, **k: (),
            "_ensure_verify_lm_head_buffers": lambda self, *a, **k: None,
            "_enqueue_packed_decode_model_step": (
                model_step
                if model_step is not None
                else (lambda self, **k: ({"strict"}, {"strict"}))
            ),
            "_ensure_packed_verify_workspace": lambda self, **k: (
                packed_state,
                scratch_base,
            ),
        },
    )()

    sessions = tuple(
        type(
            "_Session",
            (),
            {"position": position, "runner": owner.runner, "scratch": owner.scratch},
        )()
        for position in (4, 8)
    )
    error = None
    graph = None
    try:
        graph = packed_graph.capture_qwen35_gguf_packed_decode_graph(
            owner,
            token_ids=(11, 22),
            sessions=sessions,
            steps_per_replay=1,
            max_replay_steps=4,
        )
    except Exception as exc:  # noqa: BLE001 - the failure-path test inspects it
        error = exc
    return type(
        "_Capture",
        (),
        {
            "graph": graph,
            "error": error,
            "device": device,
            "metadata_calls": metadata_calls,
            "manifest_kwargs": manifest_kwargs,
            "frees": frees,
            "mallocs": mallocs,
            "uploaded_layouts": uploaded_layouts,
        },
    )()


def test_packed_graph_capture_keeps_a_leased_page_map(monkeypatch) -> None:
    """A leased (non-identity) page map survives the in-graph metadata refresh."""

    leased_pages = tuple(range(100, 108))
    capture = _capture(monkeypatch, page_ids=leased_pages, backing_kind="pool_lease")

    # The rebound table is what the host upload staged before capture.
    rebound = capture.uploaded_layouts[-1]
    expected = np.asarray(leased_pages, dtype=np.int32).reshape(2, 4)
    np.testing.assert_array_equal(rebound, expected)

    # The identity-writing refresh kernel must not clobber the live table.
    np.testing.assert_array_equal(capture.device.read(_BLOCK_TABLE_PTR), expected)
    assert capture.metadata_calls, "capture must still refresh positions in-graph"
    discard_ptrs = {call[0] for call in capture.metadata_calls}
    assert _BLOCK_TABLE_PTR not in discard_ptrs, (
        "in-graph metadata refresh must not write the authoritative page table"
    )

    capture.graph.close()
    assert discard_ptrs <= set(capture.frees), (
        "the staged page table's discard target must be freed with the graph"
    )


def test_packed_graph_capture_identity_page_map_is_unchanged(monkeypatch) -> None:
    """The private-chunk identity map keeps the single-buffer fast path."""

    capture = _capture(monkeypatch, page_ids=(), backing_kind="private")

    identity = _identity_table(2, 4)
    np.testing.assert_array_equal(capture.uploaded_layouts[-1], identity)
    np.testing.assert_array_equal(capture.device.read(_BLOCK_TABLE_PTR), identity)
    assert capture.metadata_calls
    assert all(call[0] == _BLOCK_TABLE_PTR for call in capture.metadata_calls)
    assert capture.manifest_kwargs["metadata_prepare_path"] == (
        "device_positions_persistent"
    )
    assert capture.graph.discard_block_table is None

    capture.graph.close()


def test_packed_graph_capture_page_map_is_reported_in_the_manifest(
    monkeypatch,
) -> None:
    """Provenance must say the block table was staged, not rebuilt on device."""

    capture = _capture(
        monkeypatch, page_ids=tuple(range(100, 108)), backing_kind="pool_lease"
    )

    assert capture.manifest_kwargs["metadata_prepare_path"] == (
        "device_positions_persistent_host_block_table"
    )
    assert capture.graph.discard_block_table is not None

    capture.graph.close()


def test_packed_graph_capture_frees_the_discard_table_when_capture_fails(
    monkeypatch,
) -> None:
    """A failed capture must not leak the staged page table's discard target."""

    def explode(self, **kwargs):
        raise RuntimeError("model step failed")

    capture = _capture(
        monkeypatch,
        page_ids=tuple(range(100, 108)),
        backing_kind="pool_lease",
        model_step=explode,
    )

    assert capture.error is not None
    assert "model step failed" in str(capture.error)
    discard_ptrs = {call[0] for call in capture.metadata_calls}
    assert discard_ptrs
    assert discard_ptrs <= set(capture.frees)
