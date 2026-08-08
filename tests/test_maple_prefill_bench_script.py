"""Unit contracts for the backend-neutral Maple prefill harness."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maple_prefill_bench.py"
_SPEC = importlib.util.spec_from_file_location("maple_prefill_bench_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_maple_prefill_harness_uses_backend_runner_capability(monkeypatch) -> None:
    class CudaRunner:
        pass

    monkeypatch.setattr(
        _MODULE,
        "backend_package_capability",
        lambda backend, name, default=None: (
            (lambda: CudaRunner)
            if (backend, name) == ("cuda_sm120a", "maple_runner_type")
            else default
        ),
    )

    assert _MODULE._maple_runner_type("cuda_sm120a") is CudaRunner
    assert _MODULE._maple_runner_type("hip_gfx1151") is _MODULE.MapleRunner


def test_maple_prefill_timing_uses_identical_native_and_serial_prompt_grid(
    monkeypatch,
) -> None:
    class Runner:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.native_rows: list[tuple[int, ...]] = []
            self.serial_rows: list[tuple[int, ...]] = []

        def reset(self) -> None:
            self.reset_calls += 1

        def prefill_native(self, tokens):
            self.native_rows.append(tuple(tokens))
            return SimpleNamespace(token_id=11)

        def prefill(self, tokens):
            self.serial_rows.append(tuple(tokens))
            return SimpleNamespace(token_id=11)

    shaped = [
        ({"id": "a", "category": "code", "heldout": False}, (1, 2, 3, 4)),
        ({"id": "b", "category": "code", "heldout": True}, (5, 6, 7, 8)),
    ]
    clock = itertools.cycle((0.0, 0.5))
    monkeypatch.setattr(_MODULE.time, "perf_counter", lambda: next(clock))
    runner = Runner()

    native = _MODULE._time_prefill_samples(runner, shaped, 2, native=True)
    serial = _MODULE._time_prefill_samples(runner, shaped, 2, native=False)

    expected_rows = [(1, 2, 3, 4), (5, 6, 7, 8)] * 2
    assert runner.native_rows == expected_rows
    assert runner.serial_rows == expected_rows
    assert runner.reset_calls == 8
    assert [row["prompt_id"] for row in native] == ["a", "b", "a", "b"]
    assert [row["prompt_id"] for row in serial] == ["a", "b", "a", "b"]
    assert all(row["tokens_per_second"] == 8.0 for row in native + serial)


def test_maple_prefill_harness_captures_physical_cuda_gpu0(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def capture(command, *, timeout=30.0):
        del timeout
        calls.append(tuple(command))
        output = (
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 610.43.03, 97887 MiB"
            if command[0] == "nvidia-smi"
            else "Cuda compilation tools, release 13.3, V13.3.73"
        )
        return {"command": " ".join(command), "returncode": 0, "output": output}

    monkeypatch.setattr(_MODULE, "_capture", capture)
    hardware = _MODULE._hardware_context("cuda_sm120a")

    assert hardware["gpu"] == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    assert hardware["architecture"] == "sm_120a"
    assert hardware["physical_gpu"] == 0
    assert calls[0][0:3] == ("nvidia-smi", "-i", "0")
    assert calls[1] == ("nvcc", "--version")
