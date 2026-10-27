"""HIP regression: a packed graph replay must not remap leased KV pages.

``prepare_packed_decode_metadata_from_positions`` always writes the slot-local
identity block table (``row * blocks_per_slot + page``, ``-1`` for inactive
lanes). That is correct for a private packed chunk, whose ``page_ids`` are the
identity. A ``GlobalDeviceKVPool`` workspace lease rebinds the same table to
arbitrary arena pages, and the packed KV write plus paged decode attention read
it through ``append_spans``/``prefill_spans``.

``capture_qwen35_gguf_packed_decode_graph`` therefore redirects the in-graph
refresh's block-table output to a discard buffer when the layout is not the
identity the kernel rebuilds. These tests run the real kernels on a real HIP
graph: the authoritative page map must survive every replay, while the discard
target receives the identity table. The negative control pins the defect the
redirect exists to prevent.

No model weights are involved: the fixture is the metadata and commit kernels
alone, which is the whole captured control surface that touches the page map.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)

_ROWS = 8
_BLOCKS_PER_SLOT = 4
_TABLE_SHAPE = (_ROWS, _BLOCKS_PER_SLOT)
# Arbitrary arena pages, deliberately not the slot-local identity.
_LEASED_PAGES = np.arange(100, 100 + _ROWS * _BLOCKS_PER_SLOT, dtype=np.int32).reshape(
    _TABLE_SHAPE
)
_ALL_ACTIVE = np.ones((_ROWS,), dtype=np.uint8)
_HOLE_MASK = np.asarray([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
_ACTIVE_POSITIONS = np.arange(513, 513 + _ROWS * 4, 4, dtype=np.int64)


def _identity_table(mask: np.ndarray) -> np.ndarray:
    """The table ``prepare_packed_decode_metadata_from_positions`` writes."""

    table = np.arange(_ROWS * _BLOCKS_PER_SLOT, dtype=np.int32).reshape(_TABLE_SHAPE)
    table[np.asarray(mask) == 0] = -1
    return table


def _positions(mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mask) != 0, _ACTIVE_POSITIONS, -1).astype(np.int64)


class _KernelFixture:
    """The captured control surface: metadata refresh plus commit, no weights."""

    def __init__(self, *, mask: np.ndarray) -> None:
        from hipengine.core.dtype import DType
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import malloc
        from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state

        self.runtime = get_hip_runtime()
        self.library = build_runtime_state(load=True)
        self.mask = np.asarray(mask, dtype=np.uint8)
        self.positions_host = _positions(self.mask)
        self.authoritative = malloc(
            _ROWS * _BLOCKS_PER_SLOT * DType.INT32.itemsize, runtime=self.runtime
        )
        self.discard = malloc(
            _ROWS * _BLOCKS_PER_SLOT * DType.INT32.itemsize, runtime=self.runtime
        )
        self.positions = malloc(_ROWS * DType.INT64.itemsize, runtime=self.runtime)
        self.contexts = malloc(_ROWS * DType.INT64.itemsize, runtime=self.runtime)
        self.cu_q = malloc(2 * DType.INT32.itemsize, runtime=self.runtime)
        self.cu_k = malloc(2 * DType.INT32.itemsize, runtime=self.runtime)
        self.atomic = malloc(DType.INT32.itemsize, runtime=self.runtime)
        self.gdn_cu = malloc((_ROWS + 1) * DType.INT32.itemsize, runtime=self.runtime)
        self.state_indices = malloc(_ROWS * DType.INT64.itemsize, runtime=self.runtime)
        self.mask_device = malloc(self.mask.nbytes, runtime=self.runtime)
        self.token_i32 = malloc(_ROWS * DType.INT32.itemsize, runtime=self.runtime)
        self.token_i64 = malloc(_ROWS * DType.INT64.itemsize, runtime=self.runtime)
        self.buffers = (
            self.authoritative,
            self.discard,
            self.positions,
            self.contexts,
            self.cu_q,
            self.cu_k,
            self.atomic,
            self.gdn_cu,
            self.state_indices,
            self.mask_device,
            self.token_i32,
            self.token_i64,
        )
        self.stream = self.runtime.stream_create()
        self.graph = 0
        self.graph_exec = 0

    # -- host staging -----------------------------------------------------
    def stage(self) -> None:
        from hipengine.core.memory import copy_host_to_device, host_array_ptr

        copy_host_to_device(
            self.authoritative,
            host_array_ptr(_LEASED_PAGES),
            _LEASED_PAGES.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.positions,
            host_array_ptr(self.positions_host),
            self.positions_host.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.mask_device,
            host_array_ptr(self.mask),
            self.mask.nbytes,
            runtime=self.runtime,
        )
        tokens = np.where(self.mask != 0, np.arange(11, 11 + _ROWS), 0).astype(np.int32)
        copy_host_to_device(
            self.token_i32,
            host_array_ptr(tokens),
            tokens.nbytes,
            runtime=self.runtime,
        )
        self.runtime.stream_synchronize(self.stream)

    # -- kernel launches --------------------------------------------------
    def refresh(self, *, block_table_ptr: int, stream: int = 0) -> None:
        from hipengine.kernels.hip_gfx1100.runtime import (
            prepare_packed_decode_metadata_from_positions,
        )

        prepare_packed_decode_metadata_from_positions(
            block_table_ptr,
            self.positions.ptr,
            self.contexts.ptr,
            self.cu_q.ptr,
            self.cu_k.ptr,
            self.atomic.ptr,
            self.gdn_cu.ptr,
            self.state_indices.ptr,
            _ROWS,
            _BLOCKS_PER_SLOT,
            active_mask_u8_ptr=self.mask_device.ptr,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )

    def commit(self, *, stream: int = 0) -> None:
        from hipengine.kernels.hip_gfx1100.runtime import (
            commit_packed_decode_graph_step,
        )

        commit_packed_decode_graph_step(
            self.token_i32.ptr,
            self.token_i64.ptr,
            self.positions.ptr,
            self.contexts.ptr,
            _ROWS,
            active_mask_u8_ptr=self.mask_device.ptr,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )

    # -- capture / replay -------------------------------------------------
    def capture(self, *, block_table_ptr: int) -> None:
        self.runtime.stream_synchronize(self.stream)
        self.runtime.stream_begin_capture(self.stream)
        self.refresh(block_table_ptr=block_table_ptr, stream=self.stream)
        self.commit(stream=self.stream)
        self.graph = self.runtime.stream_end_capture(self.stream)
        self.graph_exec = self.runtime.graph_instantiate(self.graph)
        self.runtime.stream_synchronize(self.stream)

    def replay(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.runtime.graph_launch(self.graph_exec, self.stream)
        self.runtime.stream_synchronize(self.stream)

    # -- observation ------------------------------------------------------
    def read(self, buffer, host: np.ndarray) -> np.ndarray:
        from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(buffer.ptr, host.nbytes),
            runtime=self.runtime,
        )
        return host

    def table(self, buffer) -> np.ndarray:
        return self.read(buffer, np.empty(_TABLE_SHAPE, dtype=np.int32))

    def position_values(self) -> np.ndarray:
        return self.read(self.positions, np.empty(_ROWS, dtype=np.int64))

    def close(self) -> None:
        from hipengine.core.memory import free

        for attribute in ("graph_exec", "graph"):
            value = int(getattr(self, attribute))
            if value:
                destroy = (
                    self.runtime.graph_exec_destroy
                    if attribute == "graph_exec"
                    else self.runtime.graph_destroy
                )
                destroy(value)
                setattr(self, attribute, 0)
        if self.stream:
            self.runtime.stream_destroy(self.stream)
            self.stream = 0
        for buffer in self.buffers:
            free(buffer, runtime=self.runtime)


@pytest.fixture
def fixture_factory():
    created: list[_KernelFixture] = []

    def build(*, mask: np.ndarray) -> _KernelFixture:
        instance = _KernelFixture(mask=mask)
        created.append(instance)
        return instance

    yield build
    for instance in reversed(created):
        instance.close()


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(_ALL_ACTIVE, id="all_active"),
        pytest.param(_HOLE_MASK, id="masked_lanes"),
    ],
)
def test_metadata_refresh_writes_the_discard_target_only(
    fixture_factory, mask: np.ndarray
) -> None:
    """Two refreshes leave the leased page map untouched and fill the discard."""

    fixture = fixture_factory(mask=mask)
    fixture.stage()

    for _ in range(2):
        fixture.refresh(block_table_ptr=fixture.discard.ptr)
    fixture.runtime.stream_synchronize(fixture.stream)

    np.testing.assert_array_equal(
        fixture.table(fixture.authoritative),
        _LEASED_PAGES,
        err_msg="the in-graph refresh rewrote the authoritative page map",
    )
    np.testing.assert_array_equal(fixture.table(fixture.discard), _identity_table(mask))


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(_ALL_ACTIVE, id="all_active"),
        pytest.param(_HOLE_MASK, id="masked_lanes"),
    ],
)
def test_metadata_refresh_would_remap_the_authoritative_table(
    fixture_factory, mask: np.ndarray
) -> None:
    """Negative control: the same kernel does remap when given the live table."""

    fixture = fixture_factory(mask=mask)
    fixture.stage()

    fixture.refresh(block_table_ptr=fixture.authoritative.ptr)
    fixture.runtime.stream_synchronize(fixture.stream)

    np.testing.assert_array_equal(
        fixture.table(fixture.authoritative),
        _identity_table(mask),
        err_msg="the control no longer reproduces the identity remap",
    )


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(_ALL_ACTIVE, id="all_active"),
        pytest.param(_HOLE_MASK, id="masked_lanes"),
    ],
)
def test_graph_replay_keeps_the_leased_page_map_through_commit(
    fixture_factory, mask: np.ndarray
) -> None:
    """Three captured replays advance positions and preserve the page map."""

    fixture = fixture_factory(mask=mask)
    fixture.stage()
    fixture.capture(block_table_ptr=fixture.discard.ptr)

    fixture.replay(3)

    np.testing.assert_array_equal(
        fixture.table(fixture.authoritative),
        _LEASED_PAGES,
        err_msg="a replay step remapped the leased packed KV pages",
    )
    np.testing.assert_array_equal(fixture.table(fixture.discard), _identity_table(mask))

    expected_positions = fixture.positions_host.copy()
    expected_positions[mask != 0] += 3
    np.testing.assert_array_equal(
        fixture.position_values(),
        expected_positions,
        err_msg="the captured commit kernel did not advance replay state",
    )


def test_graph_replay_with_the_live_table_remaps_it(fixture_factory) -> None:
    """Negative control: capturing the refresh on the live table corrupts it."""

    fixture = fixture_factory(mask=_HOLE_MASK)
    fixture.stage()
    fixture.capture(block_table_ptr=fixture.authoritative.ptr)

    fixture.replay(2)

    np.testing.assert_array_equal(
        fixture.table(fixture.authoritative),
        _identity_table(_HOLE_MASK),
        err_msg="the control no longer reproduces the captured identity remap",
    )
