"""Low-perturbation persistent checkpoints for diagnosing silent GPU prefill stalls.

The recorder is default-off.  Host submissions are written into a fixed-size
file-backed mmap without per-checkpoint I/O.  A tiny kernel on the measured HIP
stream advances ``completed_sequence`` only after prior work on that stream has
retired.  The mmap can be decoded live from another process or immediately after
bounded process termination without loading the HIP runtime.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import Callable

from hipengine.core.hip import HIP_HOST_REGISTER_MAPPED, HipRuntime

_MAGIC = b"HPFR0001"
_VERSION = 1
_HEADER_SIZE = 256
_HEADER_STRUCT = struct.Struct("<8sIIIIQQQQQ")
_ENTRY_STRUCT = struct.Struct("<QQQqqqiiq")
_ENTRY_SIZE = _ENTRY_STRUCT.size
_SUBMITTED_SEQUENCE_OFFSET = 40
_COMPLETED_SEQUENCE_OFFSET = 48
_PREFILL_ID_OFFSET = 56
_DEFAULT_CAPACITY = 8192
_VALID_GRANULARITIES = frozenset({"chunk", "layer"})


class FlightRecorderPhase(IntEnum):
    PREFILL_BEGIN = 1
    CHUNK = 2
    EMBEDDING = 3
    LINEAR_ATTENTION_LAYER = 4
    FULL_ATTENTION_LAYER = 5
    PREFILL_FINALIZE = 6
    SAMPLE = 7
    PREFILL_END = 8
    RESET = 9


CompletionMarker = Callable[..., None]


class PrefillFlightRecorder:
    """Persistent host-submission ring plus same-stream GPU retirement cursor."""

    def __init__(
        self,
        path: str | Path,
        *,
        runtime: HipRuntime,
        completion_marker: CompletionMarker | None = None,
        marker_library: object | None = None,
        capacity: int = _DEFAULT_CAPACITY,
        granularity: str = "chunk",
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        granularity = str(granularity).strip().lower()
        if granularity not in _VALID_GRANULARITIES:
            choices = ", ".join(sorted(_VALID_GRANULARITIES))
            raise ValueError(f"granularity must be one of: {choices}")

        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbytes = _HEADER_SIZE + capacity * _ENTRY_SIZE
        fd = os.open(output_path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o644)
        try:
            os.ftruncate(fd, nbytes)
            mapped = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)

        self.path = output_path
        self.capacity = capacity
        self.granularity = granularity
        self.runtime = runtime
        self._mmap: mmap.mmap | None = mapped
        self._mapped_buffer = (ctypes.c_ubyte * nbytes).from_buffer(mapped)
        self._host_ptr = ctypes.addressof(self._mapped_buffer)
        self._device_ptr = 0
        self._registered = False
        self._closed = False
        self._lock = threading.Lock()
        self._submitted_sequence = 0
        self._prefill_id = 0
        if completion_marker is None:
            from hipengine.kernels.hip_gfx1100.runtime.state import flight_recorder_mark_i64

            completion_marker = flight_recorder_mark_i64
        self._completion_marker = completion_marker
        self._marker_library = marker_library

        _HEADER_STRUCT.pack_into(
            mapped,
            0,
            _MAGIC,
            _VERSION,
            _HEADER_SIZE,
            _ENTRY_SIZE,
            capacity,
            os.getpid(),
            time.time_ns(),
            0,
            0,
            0,
        )
        mapped.flush(0, _HEADER_SIZE)
        try:
            runtime.host_register(self._host_ptr, nbytes, flags=HIP_HOST_REGISTER_MAPPED)
            self._registered = True
            self._device_ptr = runtime.host_get_device_pointer(self._host_ptr)
            if self._device_ptr <= 0:
                raise RuntimeError("HIP returned a null mapped device pointer for the flight recorder")
        except BaseException:
            self._release_mapping(flush=False)
            raise

    @property
    def completed_device_ptr(self) -> int:
        return int(self._device_ptr) + _COMPLETED_SEQUENCE_OFFSET

    @property
    def should_complete_layers(self) -> bool:
        return self.granularity == "layer"

    def begin_prefill(self, *, total_rows: int, stream: int) -> int:
        if int(total_rows) <= 0:
            raise ValueError("total_rows must be positive")
        with self._lock:
            self._ensure_open()
            self._prefill_id += 1
            self._pack_u64(_PREFILL_ID_OFFSET, self._prefill_id)
            prefill_id = self._prefill_id
        self.submit(
            phase=FlightRecorderPhase.PREFILL_BEGIN,
            prefill_id=prefill_id,
            chunk_start=0,
            chunk_end=int(total_rows),
            layer_id=-1,
            layer_type=0,
            stream=int(stream),
        )
        return prefill_id

    def submit(
        self,
        *,
        phase: FlightRecorderPhase | int,
        prefill_id: int,
        chunk_start: int,
        chunk_end: int,
        layer_id: int,
        layer_type: int,
        stream: int,
    ) -> int:
        phase_value = int(FlightRecorderPhase(int(phase)))
        with self._lock:
            self._ensure_open()
            self._submitted_sequence += 1
            sequence = self._submitted_sequence
            slot = (sequence - 1) % self.capacity
            offset = _HEADER_SIZE + slot * _ENTRY_SIZE
            _ENTRY_STRUCT.pack_into(
                self._mapping,
                offset,
                sequence,
                time.monotonic_ns(),
                int(prefill_id),
                int(chunk_start),
                int(chunk_end),
                int(layer_id),
                phase_value,
                int(layer_type),
                int(stream),
            )
            # Publish the entry before the cursor. Readers reject ring slots whose
            # embedded sequence does not match the published cursor.
            self._pack_u64(_SUBMITTED_SEQUENCE_OFFSET, sequence)
            return sequence

    def complete(self, sequence: int, *, stream: int) -> None:
        sequence = int(sequence)
        if sequence <= 0:
            raise ValueError("sequence must be positive")
        self._ensure_open()
        self._completion_marker(
            self.completed_device_ptr,
            sequence,
            stream=int(stream),
            library=self._marker_library,
            runtime=self.runtime,
        )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._ensure_open()
            return _decode_bytes(bytes(self._mapping), path=self.path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._release_mapping(flush=True)

    def _pack_u64(self, offset: int, value: int) -> None:
        struct.pack_into("<Q", self._mapping, offset, int(value))

    @property
    def _mapping(self) -> mmap.mmap:
        if self._mmap is None:
            raise RuntimeError("prefill flight recorder is closed")
        return self._mmap

    def _ensure_open(self) -> None:
        if self._closed or self._mmap is None:
            raise RuntimeError("prefill flight recorder is closed")

    def _release_mapping(self, *, flush: bool) -> None:
        mapped = self._mmap
        if mapped is None:
            self._closed = True
            return
        try:
            if self._registered:
                self.runtime.host_unregister(self._host_ptr)
                self._registered = False
        finally:
            if flush:
                mapped.flush()
            self._mapped_buffer = None
            mapped.close()
            self._mmap = None
            self._closed = True

    def __enter__(self) -> "PrefillFlightRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


def read_prefill_flight_recorder(path: str | Path) -> dict[str, object]:
    input_path = Path(path).expanduser()
    try:
        payload = input_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read prefill flight recorder {input_path}: {exc}") from exc
    return _decode_bytes(payload, path=input_path)


def _decode_bytes(payload: bytes, *, path: Path) -> dict[str, object]:
    if len(payload) < _HEADER_SIZE:
        raise ValueError(f"invalid prefill flight recorder {path}: file is smaller than the header")
    try:
        (
            magic,
            version,
            header_size,
            entry_size,
            capacity,
            pid,
            created_ns,
            submitted_sequence,
            completed_sequence,
            prefill_id,
        ) = _HEADER_STRUCT.unpack_from(payload, 0)
    except struct.error as exc:
        raise ValueError(f"invalid prefill flight recorder {path}: malformed header") from exc
    if magic != _MAGIC or version != _VERSION:
        raise ValueError(f"invalid prefill flight recorder {path}: unknown magic or version")
    if header_size != _HEADER_SIZE or entry_size != _ENTRY_SIZE or capacity <= 0:
        raise ValueError(f"invalid prefill flight recorder {path}: incompatible layout")
    expected_size = header_size + capacity * entry_size
    if len(payload) < expected_size:
        raise ValueError(f"invalid prefill flight recorder {path}: truncated ring")

    first_sequence = max(1, int(submitted_sequence) - int(capacity) + 1)
    entries: list[dict[str, int | str]] = []
    by_sequence: dict[int, dict[str, int | str]] = {}
    for sequence in range(first_sequence, int(submitted_sequence) + 1):
        slot = (sequence - 1) % capacity
        offset = header_size + slot * entry_size
        values = _ENTRY_STRUCT.unpack_from(payload, offset)
        if int(values[0]) != sequence:
            continue
        entry = _entry_dict(values)
        entries.append(entry)
        by_sequence[sequence] = entry

    return {
        "kind": "hipengine_prefill_flight_recorder",
        "schema_version": int(version),
        "path": str(path),
        "pid": int(pid),
        "created_unix_ns": int(created_ns),
        "capacity": int(capacity),
        "prefill_id": int(prefill_id),
        "submitted_sequence": int(submitted_sequence),
        "completed_sequence": int(completed_sequence),
        "retirement_lag": max(0, int(submitted_sequence) - int(completed_sequence)),
        "overwritten_entries": max(0, int(submitted_sequence) - int(capacity)),
        "last_submitted": by_sequence.get(int(submitted_sequence)),
        "last_completed": by_sequence.get(int(completed_sequence)),
        "entries": entries,
    }


def _entry_dict(values: tuple[object, ...]) -> dict[str, int | str]:
    phase_value = int(values[6])
    try:
        phase_name = FlightRecorderPhase(phase_value).name.lower()
    except ValueError:
        phase_name = f"unknown_{phase_value}"
    return {
        "sequence": int(values[0]),
        "host_monotonic_ns": int(values[1]),
        "prefill_id": int(values[2]),
        "chunk_start": int(values[3]),
        "chunk_end": int(values[4]),
        "layer_id": int(values[5]),
        "phase": phase_name,
        "phase_code": phase_value,
        "layer_type": int(values[7]),
        "stream": int(values[8]),
    }
