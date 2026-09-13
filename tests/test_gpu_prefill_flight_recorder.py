from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hipengine.runtime.prefill_flight_recorder import (
    FlightRecorderPhase,
    PrefillFlightRecorder,
    read_prefill_flight_recorder,
)


def _hip_runtime_available() -> bool:
    try:
        library = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        library.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipGetDeviceCount.restype = ctypes.c_int
        return library.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0
    except OSError:
        return False


class FakeRuntime:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []
        self.device_offset = 0x100000

    def host_register(self, ptr: int, nbytes: int, *, flags: int = 0) -> None:
        self.registered.append((ptr, nbytes, flags))

    def host_get_device_pointer(self, ptr: int, *, flags: int = 0) -> int:
        assert flags == 0
        return ptr + self.device_offset

    def host_unregister(self, ptr: int) -> None:
        self.unregistered.append(ptr)


class HostVisibleMarker:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.calls: list[tuple[int, int, int]] = []

    def __call__(
        self,
        out_i64_ptr: int,
        value: int,
        *,
        stream: int = 0,
        library: object | None = None,
        runtime: object | None = None,
    ) -> None:
        del library
        assert runtime is self.runtime
        self.calls.append((out_i64_ptr, value, stream))
        host_ptr = out_i64_ptr - self.runtime.device_offset
        ctypes.c_int64.from_address(host_ptr).value = value


def test_flight_recorder_persists_last_submitted_and_gpu_completed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "prefill.flight"
    runtime = FakeRuntime()
    marker = HostVisibleMarker(runtime)

    recorder = PrefillFlightRecorder(
        path,
        runtime=runtime,  # type: ignore[arg-type]
        completion_marker=marker,
        capacity=8,
        granularity="chunk",
    )
    prefill_id = recorder.begin_prefill(total_rows=131072, stream=0)
    reset_seq = recorder.submit(
        phase=FlightRecorderPhase.RESET,
        prefill_id=prefill_id,
        chunk_start=0,
        chunk_end=131072,
        layer_id=-1,
        layer_type=0,
        stream=0,
    )
    recorder.complete(reset_seq, stream=0)
    chunk_seq = recorder.submit(
        phase=FlightRecorderPhase.CHUNK,
        prefill_id=prefill_id,
        chunk_start=65536,
        chunk_end=69632,
        layer_id=-1,
        layer_type=0,
        stream=0,
    )
    layer_seq = recorder.submit(
        phase=FlightRecorderPhase.FULL_ATTENTION_LAYER,
        prefill_id=prefill_id,
        chunk_start=65536,
        chunk_end=69632,
        layer_id=11,
        layer_type=2,
        stream=0,
    )
    recorder.complete(chunk_seq, stream=0)

    live = recorder.snapshot()
    assert live["prefill_id"] == 1
    assert live["submitted_sequence"] == layer_seq
    assert live["completed_sequence"] == chunk_seq
    assert live["last_submitted"]["phase"] == "full_attention_layer"
    assert live["last_submitted"]["layer_id"] == 11
    assert live["last_completed"]["phase"] == "chunk"
    assert live["last_completed"]["chunk_start"] == 65536
    assert marker.calls == [
        (recorder.completed_device_ptr, reset_seq, 0),
        (recorder.completed_device_ptr, chunk_seq, 0),
    ]

    recorder.close()
    persisted = read_prefill_flight_recorder(path)
    assert persisted["submitted_sequence"] == layer_seq
    assert persisted["completed_sequence"] == chunk_seq
    assert persisted["last_submitted"] == live["last_submitted"]
    assert persisted["last_completed"] == live["last_completed"]
    assert runtime.registered[0][1] == path.stat().st_size
    assert runtime.registered[0][2] == 2
    assert runtime.unregistered == [runtime.registered[0][0]]


def test_flight_recorder_ring_wrap_retains_sequence_identity(tmp_path: Path) -> None:
    path = tmp_path / "wrapped.flight"
    runtime = FakeRuntime()
    recorder = PrefillFlightRecorder(
        path,
        runtime=runtime,  # type: ignore[arg-type]
        completion_marker=HostVisibleMarker(runtime),
        capacity=4,
        granularity="layer",
    )
    prefill_id = recorder.begin_prefill(total_rows=16384, stream=7)
    sequences = []
    for layer in range(7):
        seq = recorder.submit(
            phase=FlightRecorderPhase.LINEAR_ATTENTION_LAYER,
            prefill_id=prefill_id,
            chunk_start=4096,
            chunk_end=8192,
            layer_id=layer,
            layer_type=1,
            stream=7,
        )
        recorder.complete(seq, stream=7)
        sequences.append(seq)

    snapshot = recorder.snapshot()
    assert snapshot["submitted_sequence"] == sequences[-1]
    assert snapshot["completed_sequence"] == sequences[-1]
    assert [entry["sequence"] for entry in snapshot["entries"]] == sequences[-4:]
    assert snapshot["last_completed"]["layer_id"] == 6
    assert snapshot["overwritten_entries"] == len(sequences) + 1 - 4
    recorder.close()


def test_flight_recorder_rejects_invalid_configuration(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="capacity"):
        PrefillFlightRecorder(tmp_path / "bad.flight", runtime=runtime, capacity=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="granularity"):
        PrefillFlightRecorder(
            tmp_path / "bad.flight",
            runtime=runtime,  # type: ignore[arg-type]
            capacity=4,
            granularity="dispatch",
        )


def test_flight_recorder_reader_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.flight"
    path.write_bytes(b"not-a-flight-recorder")
    with pytest.raises(ValueError, match="flight recorder"):
        read_prefill_flight_recorder(path)


def test_flight_recorder_decoder_limits_ring_entries(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.qwen35_prefill_flight_recorder import main

    path = tmp_path / "decode.flight"
    runtime = FakeRuntime()
    recorder = PrefillFlightRecorder(
        path,
        runtime=runtime,  # type: ignore[arg-type]
        completion_marker=HostVisibleMarker(runtime),
        capacity=8,
    )
    prefill_id = recorder.begin_prefill(total_rows=8192, stream=0)
    sequence = recorder.submit(
        phase=FlightRecorderPhase.CHUNK,
        prefill_id=prefill_id,
        chunk_start=4096,
        chunk_end=8192,
        layer_id=-1,
        layer_type=0,
        stream=0,
    )
    recorder.complete(sequence, stream=0)
    recorder.close()

    assert main([str(path), "--entries", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed_sequence"] == sequence
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["sequence"] == sequence


def test_flight_recorder_import_and_decoder_do_not_load_hip() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from hipengine.core.hip import is_default_runtime_loaded; "
                "import hipengine.runtime.prefill_flight_recorder; "
                "import scripts.qwen35_prefill_flight_recorder; "
                "print(is_default_runtime_loaded())"
            ),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.skipif(not _hip_runtime_available(), reason="HIP runtime/device unavailable")
def test_flight_recorder_mapped_completion_is_visible_cross_process(tmp_path: Path) -> None:
    from hipengine.core.hip import get_hip_runtime

    path = tmp_path / "gpu.flight"
    runtime = get_hip_runtime()
    stream = runtime.stream_create()
    try:
        recorder = PrefillFlightRecorder(path, runtime=runtime, capacity=8)
        prefill_id = recorder.begin_prefill(total_rows=4096, stream=stream)
        sequence = recorder.submit(
            phase=FlightRecorderPhase.CHUNK,
            prefill_id=prefill_id,
            chunk_start=0,
            chunk_end=4096,
            layer_id=-1,
            layer_type=0,
            stream=stream,
        )
        recorder.complete(sequence, stream=stream)
        runtime.stream_synchronize(stream)

        result = subprocess.run(
            [sys.executable, "scripts/qwen35_prefill_flight_recorder.py", str(path), "--entries", "2"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        snapshot = json.loads(result.stdout)
        assert snapshot["completed_sequence"] == sequence
        assert snapshot["last_completed"]["phase"] == "chunk"
        recorder.close()
    finally:
        runtime.stream_destroy(stream)


def test_readme_sweep_forwards_flight_recorder_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.qwen35_readme_sweep as sweep

    captured: dict[str, object] = {}

    def fake_run(args, model, workloads, warmup_decode_tokens, max_sequence_length, compiler_version, prefill_config):
        del model, workloads, warmup_decode_tokens, max_sequence_length, compiler_version, prefill_config
        captured["path"] = args.prefill_flight_recorder
        captured["granularity"] = args.prefill_flight_recorder_granularity
        return {"ok": True}

    recorder_path = tmp_path / "sweep.flight"
    monkeypatch.setattr(sweep, "_run_gguf_sweep", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_readme_sweep.py",
            "--engine",
            "gguf",
            "--model",
            str(tmp_path / "model.gguf"),
            "--workloads",
            "512/0",
            "--prefill-flight-recorder",
            str(recorder_path),
            "--prefill-flight-recorder-granularity",
            "layer",
        ],
    )

    assert sweep.main() == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert captured == {"path": recorder_path, "granularity": "layer"}
