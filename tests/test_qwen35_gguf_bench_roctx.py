"""Profiler-region controls for ``scripts/qwen35_gguf_bench.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_bench_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "qwen35_gguf_bench.py"
    module_name = "_qwen35_gguf_bench_roctx_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load_bench_module()


class _FakeFn:
    def __init__(self, name: str, calls: list[tuple[str, int]]) -> None:
        self.name = name
        self.calls = calls
        self.argtypes = None
        self.restype = None

    def __call__(self, value: int) -> None:
        self.calls.append((self.name, int(value)))


class _FakeRoctx:
    def __init__(self, calls: list[tuple[str, int]]) -> None:
        self.roctxProfilerResume = _FakeFn("resume", calls)
        self.roctxProfilerPause = _FakeFn("pause", calls)


def test_selected_profiler_region_resumes_and_pauses_only_matching_phase(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(BENCH.ctypes, "CDLL", lambda _name: _FakeRoctx(calls))
    control = BENCH._RoctxProfilerControl(enabled=True)

    with control.region("prefill", selected="measured_decode_graph"):
        pass
    with control.region("measured_decode_graph", selected="measured_decode_graph"):
        pass

    assert calls == [("resume", 0), ("pause", 0)]


def test_selected_profiler_control_prefers_sdk_control_library(monkeypatch) -> None:
    loaded: list[str] = []
    calls: list[tuple[str, int]] = []

    def load(name: str):
        loaded.append(name)
        return _FakeRoctx(calls)

    monkeypatch.setattr(BENCH.ctypes, "CDLL", load)
    BENCH._RoctxProfilerControl(enabled=True)

    assert loaded == ["librocprofiler-sdk-roctx.so"]


def test_selected_profiler_control_falls_back_to_legacy_library(monkeypatch) -> None:
    loaded: list[str] = []
    calls: list[tuple[str, int]] = []

    def load(name: str):
        loaded.append(name)
        if name == "librocprofiler-sdk-roctx.so":
            raise OSError("SDK control library unavailable")
        return _FakeRoctx(calls)

    monkeypatch.setattr(BENCH.ctypes, "CDLL", load)
    control = BENCH._RoctxProfilerControl(enabled=True)
    control.resume()
    control.pause()

    assert loaded == ["librocprofiler-sdk-roctx.so", "libroctx64.so"]
    assert calls == [("resume", 0), ("pause", 0)]


def test_disabled_profiler_control_does_not_load_roctx(monkeypatch) -> None:
    def fail_load(_name: str):  # pragma: no cover - assertion helper
        raise AssertionError("disabled profiler control should not load ROCTX")

    monkeypatch.setattr(BENCH.ctypes, "CDLL", fail_load)
    control = BENCH._RoctxProfilerControl(enabled=False)

    with control.region("prefill", selected="prefill"):
        pass
