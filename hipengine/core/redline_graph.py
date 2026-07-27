"""Optional strict adapter for Redline's retained-PM4 hipGraph interposer.

The adapter has no import-time Redline dependency.  A caller must launch the
Python process with the Python-enabled ``libredline_hipgraph.so`` in
``LD_PRELOAD`` and pass the same backing file through its Python-module symlink.
Native HIP remains the default graph transport.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


class GraphTransportError(RuntimeError):
    """Raised when an optional graph transport cannot prove its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _same_file(lhs: Path, rhs: Path) -> bool:
    try:
        return os.path.samefile(lhs, rhs)
    except OSError:
        return False


def _is_preloaded(path: Path) -> bool:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return False
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        mapped = Path(fields[5].removesuffix(" (deleted)"))
        if _same_file(path, mapped):
            return True
    return False


def _load_control_module(module_path: Path) -> ModuleType:
    name = "redline_hipgraph"
    existing = sys.modules.get(name)
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", "")))
        if not _same_file(existing_path, module_path):
            raise GraphTransportError(
                "loaded redline_hipgraph module does not use the requested backing file"
            )
        return existing
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise GraphTransportError(f"cannot load Redline control module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass
class RedlineHipGraphAdapter:
    """Route graph lifecycle calls through one preloaded Redline DSO.

    ``require_pm4`` checks the selected transport after instantiation and after
    every launch.  Redline currently keeps a native-HIP shadow internally; if a
    replay failure makes upstream switch to that shadow, this adapter detects
    the transition and raises rather than silently accepting the fallback.
    """

    library: Any
    control_module: Any
    library_path: Path
    module_path: Path
    library_sha256: str
    require_pm4: bool = True
    _exec_transports: dict[int, str] = field(default_factory=dict, init=False)
    _selection_history: Counter[str] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        self.library_path = Path(self.library_path).expanduser().resolve()
        self.module_path = Path(self.module_path).expanduser().resolve()
        self._configure()

    @classmethod
    def load(
        cls,
        *,
        library_path: str | Path,
        module_path: str | Path,
        require_pm4: bool = True,
        expected_sha256: str | None = None,
    ) -> "RedlineHipGraphAdapter":
        """Load controls for an already-preloaded Python-enabled Redline DSO."""

        library_path = Path(library_path).expanduser().resolve()
        module_path = Path(module_path).expanduser().resolve()
        if not library_path.is_file():
            raise GraphTransportError(f"Redline hipGraph library is missing: {library_path}")
        if not module_path.is_file():
            raise GraphTransportError(f"Redline Python module is missing: {module_path}")
        if not _same_file(library_path, module_path):
            raise GraphTransportError(
                "Redline library and Python module must be the same backing file; "
                "use a symlink rather than a copied shared object"
            )
        if not _is_preloaded(library_path):
            raise GraphTransportError(
                "Redline hipGraph library was not preloaded; launch the process with "
                f"LD_PRELOAD={library_path}"
            )
        digest = _sha256(library_path)
        if expected_sha256 is not None:
            expected = expected_sha256
            if not expected.startswith("sha256:"):
                expected = "sha256:" + expected
            if digest != expected:
                raise GraphTransportError(
                    f"Redline library hash {digest} does not match required {expected}"
                )
        module = _load_control_module(module_path)
        required = ("available", "is_pm4")
        missing = [name for name in required if not callable(getattr(module, name, None))]
        if missing:
            raise GraphTransportError(
                f"Redline control module is missing required API: {', '.join(missing)}"
            )
        if not bool(module.available()):
            raise GraphTransportError("Redline retained-PM4 runtime is unavailable")
        library = ctypes.CDLL(str(library_path))
        return cls.from_loaded(
            library=library,
            control_module=module,
            library_path=library_path,
            module_path=module_path,
            library_sha256=digest,
            require_pm4=require_pm4,
        )

    @classmethod
    def from_loaded(
        cls,
        *,
        library: Any,
        control_module: Any,
        library_path: str | Path,
        module_path: str | Path,
        library_sha256: str,
        require_pm4: bool = True,
    ) -> "RedlineHipGraphAdapter":
        """Construct from existing handles; primarily useful for host-only tests."""

        return cls(
            library=library,
            control_module=control_module,
            library_path=Path(library_path),
            module_path=Path(module_path),
            library_sha256=str(library_sha256),
            require_pm4=bool(require_pm4),
        )

    def stream_begin_capture(self, stream: int, mode: int) -> None:
        self._check(
            "hipStreamBeginCapture",
            self.library.hipStreamBeginCapture(
                ctypes.c_void_p(stream), ctypes.c_int(mode)
            ),
        )

    def stream_end_capture(self, stream: int) -> int:
        graph = ctypes.c_void_p()
        self._check(
            "hipStreamEndCapture",
            self.library.hipStreamEndCapture(
                ctypes.c_void_p(stream), ctypes.byref(graph)
            ),
        )
        return int(graph.value or 0)

    def graph_instantiate(self, graph: int) -> int:
        graph_exec = ctypes.c_void_p()
        error_node = ctypes.c_void_p()
        log_buffer = ctypes.create_string_buffer(4096)
        self._check(
            "hipGraphInstantiate",
            self.library.hipGraphInstantiate(
                ctypes.byref(graph_exec),
                ctypes.c_void_p(graph),
                ctypes.byref(error_node),
                log_buffer,
                ctypes.c_size_t(len(log_buffer)),
            ),
        )
        value = int(graph_exec.value or 0)
        selected = self._selected_transport(value)
        self._record_transport(value, selected)
        if self.require_pm4 and selected != "redline_pm4":
            if value:
                self.library.hipGraphExecDestroy(ctypes.c_void_p(value))
            self._exec_transports[value] = "redline_rejected"
            raise GraphTransportError(
                "Redline graph instantiation did not select retained PM4"
            )
        return value

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        before = self._selected_transport(graph_exec)
        if self.require_pm4 and before != "redline_pm4":
            self._exec_transports[graph_exec] = "redline_rejected"
            raise GraphTransportError("Redline graph exec no longer owns retained PM4")
        self._check(
            "hipGraphLaunch",
            self.library.hipGraphLaunch(
                ctypes.c_void_p(graph_exec), ctypes.c_void_p(stream)
            ),
        )
        after = self._selected_transport(graph_exec)
        self._exec_transports[graph_exec] = after
        if self.require_pm4 and after != "redline_pm4":
            raise GraphTransportError(
                "Redline graph launch switched to native HIP fallback"
            )

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self._check(
            "hipGraphExecDestroy",
            self.library.hipGraphExecDestroy(ctypes.c_void_p(graph_exec)),
        )

    def graph_destroy(self, graph: int) -> None:
        self._check(
            "hipGraphDestroy",
            self.library.hipGraphDestroy(ctypes.c_void_p(graph)),
        )

    def graph_exec_transport(self, graph_exec: int) -> str:
        return self._exec_transports.get(int(graph_exec), "redline_unknown")

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_transport": "redline_pm4",
            "require_pm4": self.require_pm4,
            "library_path": str(self.library_path),
            "module_path": str(self.module_path),
            "library_sha256": self.library_sha256,
            "same_backing_file": _same_file(self.library_path, self.module_path),
            "selected_exec_transports": dict(sorted(self._selection_history.items())),
        }

    def _selected_transport(self, graph_exec: int) -> str:
        if not graph_exec:
            return "redline_unknown"
        try:
            is_pm4 = bool(self.control_module.is_pm4(int(graph_exec)))
        except Exception as exc:
            raise GraphTransportError(
                f"Redline is_pm4 proof failed for exec {graph_exec:#x}"
            ) from exc
        return "redline_pm4" if is_pm4 else "redline_native_fallback"

    def _record_transport(self, graph_exec: int, selected: str) -> None:
        self._exec_transports[int(graph_exec)] = selected
        self._selection_history[selected] += 1

    def _check(self, operation: str, status: int) -> None:
        status = int(status)
        if status == 0:
            return
        message = "<unknown>"
        error_string = getattr(self.library, "hipGetErrorString", None)
        if callable(error_string):
            raw = error_string(status)
            if raw:
                message = raw.decode("utf-8", errors="replace")
        raise GraphTransportError(f"{operation} failed with HIP status {status}: {message}")

    def _configure(self) -> None:
        vp = ctypes.c_void_p
        self.library.hipStreamBeginCapture.argtypes = [vp, ctypes.c_int]
        self.library.hipStreamBeginCapture.restype = ctypes.c_int
        self.library.hipStreamEndCapture.argtypes = [vp, ctypes.POINTER(vp)]
        self.library.hipStreamEndCapture.restype = ctypes.c_int
        self.library.hipGraphInstantiate.argtypes = [
            ctypes.POINTER(vp),
            vp,
            ctypes.POINTER(vp),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.library.hipGraphInstantiate.restype = ctypes.c_int
        self.library.hipGraphLaunch.argtypes = [vp, vp]
        self.library.hipGraphLaunch.restype = ctypes.c_int
        self.library.hipGraphExecDestroy.argtypes = [vp]
        self.library.hipGraphExecDestroy.restype = ctypes.c_int
        self.library.hipGraphDestroy.argtypes = [vp]
        self.library.hipGraphDestroy.restype = ctypes.c_int
