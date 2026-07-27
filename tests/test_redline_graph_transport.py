from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.hip import HipRuntime
from hipengine.core.redline_graph import GraphTransportError, RedlineHipGraphAdapter


class _Function:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.function(*args)


class _GraphLibrary:
    def __init__(self, *, graph: int, graph_exec: int):
        self.graph = graph
        self.graph_exec = graph_exec
        self.hipStreamBeginCapture = _Function(lambda stream, mode: 0)
        self.hipStreamEndCapture = _Function(self._end_capture)
        self.hipGraphInstantiate = _Function(self._instantiate)
        self.hipGraphLaunch = _Function(lambda graph_exec, stream: 0)
        self.hipGraphExecDestroy = _Function(lambda graph_exec: 0)
        self.hipGraphDestroy = _Function(lambda graph: 0)
        self.hipGetErrorString = _Function(lambda code: b"fake HIP graph error")

    def _end_capture(self, stream, out_graph):
        del stream
        out_graph._obj.value = self.graph
        return 0

    def _instantiate(self, out_exec, graph, error_node, log_buffer, buffer_size):
        del graph, error_node, log_buffer, buffer_size
        out_exec._obj.value = self.graph_exec
        return 0


class _ControlModule:
    def __init__(self, *, pm4: bool):
        self.pm4 = pm4
        self.queries: list[int] = []

    def available(self) -> bool:
        return True

    def is_pm4(self, graph_exec: int) -> bool:
        self.queries.append(graph_exec)
        return self.pm4


def _adapter(*, pm4: bool, require_pm4: bool = True) -> tuple[RedlineHipGraphAdapter, _GraphLibrary]:
    library = _GraphLibrary(graph=0xA000, graph_exec=0xB000)
    adapter = RedlineHipGraphAdapter.from_loaded(
        library=library,
        control_module=_ControlModule(pm4=pm4),
        library_path=Path("/opt/redline/libredline_hipgraph.so"),
        module_path=Path("/opt/redline/redline_hipgraph.so"),
        library_sha256="sha256:test",
        require_pm4=require_pm4,
    )
    return adapter, library


def test_redline_adapter_routes_graph_lifecycle_and_proves_pm4() -> None:
    native = _GraphLibrary(graph=0x6000, graph_exec=0x7000)
    adapter, redline = _adapter(pm4=True)
    runtime = HipRuntime(native, graph_adapter=adapter)  # type: ignore[arg-type]

    runtime.stream_begin_capture(0x5000, mode=2)
    graph = runtime.stream_end_capture(0x5000)
    graph_exec = runtime.graph_instantiate(graph)
    runtime.graph_launch(graph_exec, 0x5000)
    runtime.graph_exec_destroy(graph_exec)
    runtime.graph_destroy(graph)

    assert (graph, graph_exec) == (0xA000, 0xB000)
    assert runtime.graph_exec_transport(graph_exec) == "redline_pm4"
    assert len(redline.hipStreamBeginCapture.calls) == 1
    assert len(redline.hipGraphLaunch.calls) == 1
    assert len(redline.hipGraphExecDestroy.calls) == 1
    assert len(redline.hipGraphDestroy.calls) == 1
    assert native.hipStreamBeginCapture.calls == []
    assert native.hipGraphLaunch.calls == []
    assert adapter.provenance()["require_pm4"] is True
    assert adapter.provenance()["selected_exec_transports"] == {"redline_pm4": 1}


def test_redline_adapter_strict_mode_destroys_native_fallback_exec() -> None:
    native = _GraphLibrary(graph=0x6000, graph_exec=0x7000)
    adapter, redline = _adapter(pm4=False, require_pm4=True)
    runtime = HipRuntime(native, graph_adapter=adapter)  # type: ignore[arg-type]

    runtime.stream_begin_capture(0x5000)
    graph = runtime.stream_end_capture(0x5000)
    with pytest.raises(GraphTransportError, match="retained PM4"):
        runtime.graph_instantiate(graph)

    assert len(redline.hipGraphExecDestroy.calls) == 1
    assert runtime.graph_exec_transport(0xB000) == "redline_rejected"


def test_redline_adapter_explicit_fallback_mode_reports_actual_transport() -> None:
    native = _GraphLibrary(graph=0x6000, graph_exec=0x7000)
    adapter, redline = _adapter(pm4=False, require_pm4=False)
    runtime = HipRuntime(native, graph_adapter=adapter)  # type: ignore[arg-type]

    runtime.stream_begin_capture(0x5000)
    graph_exec = runtime.graph_instantiate(runtime.stream_end_capture(0x5000))
    runtime.graph_launch(graph_exec, 0x5000)

    assert runtime.graph_exec_transport(graph_exec) == "redline_native_fallback"
    assert len(redline.hipGraphLaunch.calls) == 1


def test_redline_loader_rejects_library_that_was_not_preloaded(tmp_path: Path) -> None:
    library = tmp_path / "libredline_hipgraph.so"
    module = tmp_path / "redline_hipgraph.so"
    library.write_bytes(b"not a shared object")
    module.symlink_to(library.name)

    with pytest.raises(GraphTransportError, match="preloaded"):
        RedlineHipGraphAdapter.load(
            library_path=library,
            module_path=module,
            require_pm4=True,
        )


def test_redline_loader_requires_module_and_preload_to_share_one_dso(tmp_path: Path) -> None:
    library = tmp_path / "libredline_hipgraph.so"
    module = tmp_path / "redline_hipgraph.so"
    library.write_bytes(b"library")
    module.write_bytes(b"copied module")

    with pytest.raises(GraphTransportError, match="same backing file"):
        RedlineHipGraphAdapter.load(
            library_path=library,
            module_path=module,
            require_pm4=True,
        )
