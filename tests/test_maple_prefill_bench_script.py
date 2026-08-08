"""Unit contracts for the backend-neutral Maple prefill harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
