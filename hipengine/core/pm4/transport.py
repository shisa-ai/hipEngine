"""Architecture-keyed graph submission transports.

HIP remains the default graph frontend and replay transport.  Explicit ``aql``
or ``pm4`` selection inspects that captured graph once, creates one retained
public-HSA owner, and never falls back to HIP after selection or submission.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from hipengine.core.pm4.graph import HipGraphManifest, inspect_hip_graph
from hipengine.core.pm4.native import NativePm4Context, NativePm4Executable

_ENV_SUBMISSION_TRANSPORT = "HIPENGINE_SUBMISSION_TRANSPORT"
_DEFAULT_SUBMISSION_TRANSPORT = "hipgraph"
_NATIVE_TRANSPORTS = frozenset(("aql", "pm4"))


@dataclass(frozen=True, slots=True, order=True)
class SubmissionTransportKey:
    """Exact backend and transport registration key."""

    backend: str
    transport: str

    def __post_init__(self) -> None:
        for name in ("backend", "transport"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"submission transport {name} must be non-empty")
            object.__setattr__(self, name, value)


class DuplicateSubmissionTransportError(ValueError):
    """Raised when an exact transport key is registered twice."""


class MissingSubmissionTransportError(LookupError):
    """Raised when an explicit backend/transport pair is not registered."""

    def __init__(self, key: SubmissionTransportKey):
        self.key = key
        super().__init__(
            f"submission transport backend={key.backend!r}, transport={key.transport!r} "
            "is not registered"
        )


class GraphSubmission(Protocol):
    """Owned executable for one immutable captured graph generation."""

    name: str
    graph_exec: int

    def launch(self, stream: int) -> None: ...

    def provenance(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class GraphSubmissionContext(Protocol):
    """Persistent transport owner shared across graph generations."""

    backend: str
    gfx_arch: str
    name: str

    def instantiate(self, request: "GraphSubmissionRequest", selected: str) -> GraphSubmission: ...

    def provenance(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GraphSubmissionRequest:
    """Factory inputs captured once outside the replay hot loop."""

    backend: str
    gfx_arch: str
    runtime: Any
    graph: int
    capture_stream: int
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not str(self.backend).strip():
            raise ValueError("submission backend must be non-empty")
        if not str(self.gfx_arch).strip():
            raise ValueError("submission gfx_arch must be non-empty")
        if int(self.graph) <= 0:
            raise ValueError("captured HIP graph handle must be positive")
        if int(self.capture_stream) < 0:
            raise ValueError("capture stream handle must be non-negative")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("submission timeout_seconds must be positive")


SubmissionTransportFactory = Callable[[GraphSubmissionRequest, str], GraphSubmission]
SubmissionContextFactory = Callable[[str, str, Any, str], GraphSubmissionContext]
_TRANSPORTS: dict[SubmissionTransportKey, SubmissionTransportFactory] = {}
_CONTEXT_FACTORIES: dict[SubmissionTransportKey, SubmissionContextFactory] = {}
_BUILTINS_REGISTERED = False


def register_submission_transport(
    key: SubmissionTransportKey,
    factory: SubmissionTransportFactory,
    *,
    context_factory: SubmissionContextFactory | None = None,
    replace: bool = False,
) -> None:
    """Register one exact backend/transport factory."""

    if not isinstance(key, SubmissionTransportKey):
        raise TypeError("key must be a SubmissionTransportKey")
    if not callable(factory):
        raise TypeError("submission transport factory must be callable")
    if context_factory is not None and not callable(context_factory):
        raise TypeError("submission context factory must be callable")
    if key in _TRANSPORTS and not replace:
        raise DuplicateSubmissionTransportError(f"submission transport already registered: {key}")
    _TRANSPORTS[key] = factory
    if context_factory is None:
        _CONTEXT_FACTORIES.pop(key, None)
    else:
        _CONTEXT_FACTORIES[key] = context_factory


def _register_builtin_if_missing(
    key: SubmissionTransportKey,
    factory: SubmissionTransportFactory,
    context_factory: SubmissionContextFactory | None = None,
) -> None:
    if key not in _TRANSPORTS:
        _TRANSPORTS[key] = factory
        if context_factory is not None:
            _CONTEXT_FACTORIES[key] = context_factory


def register_builtin_submission_transports() -> None:
    """Register built-in exact architecture capabilities without loading ROCr."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        _register_builtin_if_missing(
            SubmissionTransportKey(backend, "hipgraph"),
            _create_hipgraph_submission,
        )
    for name in sorted(_NATIVE_TRANSPORTS):
        _register_builtin_if_missing(
            SubmissionTransportKey("hip_gfx1100", name),
            _create_gfx1100_native_submission,
            _create_gfx1100_submission_context,
        )
    _BUILTINS_REGISTERED = True


def registered_submission_transports() -> tuple[SubmissionTransportKey, ...]:
    register_builtin_submission_transports()
    return tuple(sorted(_TRANSPORTS))


def resolve_submission_transport_factory(
    backend: str,
    transport: str,
) -> SubmissionTransportFactory:
    """Resolve one exact capability or fail closed without backend fallback."""

    register_builtin_submission_transports()
    key = SubmissionTransportKey(backend, transport)
    try:
        return _TRANSPORTS[key]
    except KeyError as exc:
        raise MissingSubmissionTransportError(key) from exc


def select_submission_transport(
    transport: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Parse transport selection once at graph-owner construction."""

    env_map = os.environ if env is None else env
    if transport is None:
        selected = env_map.get(_ENV_SUBMISSION_TRANSPORT, _DEFAULT_SUBMISSION_TRANSPORT)
    else:
        selected = transport
    normalized = str(selected).strip().lower()
    if not normalized:
        raise ValueError("submission transport must be non-empty")
    return normalized


def create_graph_submission_context(
    *,
    backend: str,
    gfx_arch: str,
    runtime: Any,
    transport: str | None = None,
    env: Mapping[str, str] | None = None,
) -> GraphSubmissionContext | None:
    """Create one reusable transport context, or ``None`` for stateless HIP.

    The returned native context owns one queue across every graph generation
    instantiated through it.  Callers must close all child submissions before
    closing the context.
    """

    selected = select_submission_transport(transport, env=env)
    resolve_submission_transport_factory(backend, selected)
    key = SubmissionTransportKey(backend, selected)
    context_factory = _CONTEXT_FACTORIES.get(key)
    if context_factory is None:
        return None
    return context_factory(str(backend), str(gfx_arch), runtime, selected)


def create_graph_submission(
    *,
    backend: str,
    gfx_arch: str,
    runtime: Any,
    graph: int,
    stream: int,
    transport: str | None = None,
    timeout_seconds: float = 5.0,
    env: Mapping[str, str] | None = None,
    submission_context: GraphSubmissionContext | None = None,
) -> GraphSubmission:
    """Instantiate the selected transport for one captured graph.

    Selection and all capability checks happen before this returns.  An explicit
    native selection is never replaced with a HIP graph when inspection or
    native instantiation rejects the graph.
    """

    selected = select_submission_transport(transport, env=env)
    factory = resolve_submission_transport_factory(backend, selected)
    request = GraphSubmissionRequest(
        backend=str(backend),
        gfx_arch=str(gfx_arch),
        runtime=runtime,
        graph=int(graph),
        capture_stream=int(stream),
        timeout_seconds=float(timeout_seconds),
    )
    if submission_context is not None:
        return submission_context.instantiate(request, selected)
    return factory(request, selected)


@dataclass(slots=True)
class HipGraphSubmission:
    """Checked owner for a native HIP graph executable."""

    request: GraphSubmissionRequest
    graph_exec: int
    name: str = "hipgraph"
    launches: int = 0
    closed: bool = False

    def launch(self, stream: int) -> None:
        if self.closed or not self.graph_exec:
            raise RuntimeError("HIP graph submission is closed")
        self.request.runtime.graph_launch(self.graph_exec, int(stream))
        self.launches += 1

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "transport": self.name,
            "backend": self.request.backend,
            "gfx_arch": self.request.gfx_arch,
            "source": "hip_runtime_graph",
            "graph_handle": int(self.request.graph),
            "graph_exec": int(self.graph_exec),
            "launches": int(self.launches),
            "submission_started": bool(self.launches),
            "native_fallbacks": 0,
            "closed": bool(self.closed),
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.graph_exec:
            self.request.runtime.graph_exec_destroy(self.graph_exec)
            self.graph_exec = 0
        self.closed = True


@dataclass(slots=True)
class NativeGraphSubmission:
    """Retained direct-AQL or gfx1100 PM4 executable and queue owner."""

    request: GraphSubmissionRequest
    name: str
    manifest: HipGraphManifest
    context: NativePm4Context
    executable: NativePm4Executable
    stateful_registers: bool = False
    local_cache_dependencies: bool = False
    context_owner: NativeGraphSubmissionContext | None = None
    graph_exec: int = 0
    launches: int = 0
    launch_attempts: int = 0
    submission_started: bool = False
    closed: bool = False
    _final_provenance: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _released_from_context: bool = field(default=False, init=False, repr=False)

    def launch(self, stream: int) -> None:
        if self.closed or not self.executable.handle or not self.context.handle:
            raise RuntimeError(f"{self.name} graph submission is closed")
        # The public HSA queue is independent of HIP streams.  Drain the caller's
        # stream before publishing native packets; the native launch then waits
        # for its own finite-deadline completion before returning.
        self.request.runtime.stream_synchronize(int(stream))
        self.launch_attempts += 1
        self.submission_started = True
        self.executable.launch(self.name, timeout_seconds=self.request.timeout_seconds)
        self.launches += 1

    @staticmethod
    def _component_provenance(owner: Any) -> dict[str, Any]:
        try:
            return owner.provenance()
        except Exception as exc:
            return {"error_type": type(exc).__name__, "error": str(exc)}

    def _live_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "transport": self.name,
            "backend": self.request.backend,
            "gfx_arch": self.request.gfx_arch,
            "source": "hipengine_in_tree_rocr_pm4",
            "graph_handle": int(self.request.graph),
            "graph_exec": 0,
            "graph_fingerprint": self.manifest.fingerprint,
            "node_count": len(self.manifest.nodes),
            "hsaco_sha256": sorted({node.hsaco_sha256 for node in self.manifest.nodes}),
            "launches": int(self.launches),
            "launch_attempts": int(self.launch_attempts),
            "submission_started": bool(self.submission_started),
            "native_fallbacks": 0,
            "stateful_registers": bool(self.stateful_registers),
            "local_cache_dependencies": bool(self.local_cache_dependencies),
            "closed": bool(self.closed),
            "context": self._component_provenance(self.context),
            "executable": self._component_provenance(self.executable),
        }

    def provenance(self) -> dict[str, Any]:
        if self.closed:
            if self._final_provenance is None:
                raise RuntimeError(f"closed {self.name} graph submission has no provenance")
            value = dict(self._final_provenance)
        else:
            value = self._live_provenance()
        if self.context_owner is not None:
            value["transport_context"] = self.context_owner.provenance()
        return value

    def close(self) -> None:
        if self.closed:
            return
        snapshot = self._live_provenance()
        if self.executable.handle:
            self.executable.close()
        if self.context_owner is not None:
            if not self._released_from_context:
                self.context_owner.release(self)
                self._released_from_context = True
            context_retired = True
            snapshot["context_retained"] = True
        else:
            if self.context.handle:
                self.context.close()
            context_retired = not self.context.handle
            snapshot["context_retained"] = False
        self.closed = not self.executable.handle and context_retired
        if self.closed:
            snapshot["closed"] = True
            snapshot["graph_exec"] = 0
            self._final_provenance = snapshot


@dataclass(slots=True)
class NativeGraphSubmissionContext:
    """One persistent gfx1100 ROCr queue shared by graph generations."""

    backend: str
    gfx_arch: str
    runtime: Any
    name: str
    context: NativePm4Context
    stateful_registers: bool = False
    local_cache_dependencies: bool = False
    timestamps: bool = False
    context_create_ns: int = 0
    last_graph_inspection_ns: int = 0
    last_graph_inspection_phases_ns: dict[str, int] = field(default_factory=dict)
    last_native_instantiate_ns: int = 0
    graph_inspection_ns_total: int = 0
    native_instantiate_ns_total: int = 0
    children: int = 0
    generations: int = 0
    closed: bool = False
    _final_provenance: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def instantiate(
        self,
        request: GraphSubmissionRequest,
        selected: str,
    ) -> NativeGraphSubmission:
        if self.closed or not self.context.handle:
            raise RuntimeError(f"{self.name} submission context is closed")
        if (
            request.backend != self.backend
            or request.gfx_arch != self.gfx_arch
            or request.runtime is not self.runtime
            or selected != self.name
        ):
            raise RuntimeError("graph generation does not match its submission context")
        inspection_start_ns = time.perf_counter_ns()
        inspection_phases_ns: dict[str, int] = {}
        manifest = inspect_hip_graph(
            request.runtime,
            request.graph,
            gfx_arch=request.gfx_arch,
            stream=request.capture_stream,
            timings=inspection_phases_ns,
        )
        self.last_graph_inspection_phases_ns = inspection_phases_ns
        self.last_graph_inspection_ns = time.perf_counter_ns() - inspection_start_ns
        self.graph_inspection_ns_total += self.last_graph_inspection_ns
        instantiate_start_ns = time.perf_counter_ns()
        instantiate_options = {
            "stateful_registers": self.stateful_registers,
            "local_cache_dependencies": self.local_cache_dependencies,
        }
        if self.timestamps:
            instantiate_options["timestamps"] = True
        executable = self.context.instantiate(manifest, **instantiate_options)
        self.last_native_instantiate_ns = time.perf_counter_ns() - instantiate_start_ns
        self.native_instantiate_ns_total += self.last_native_instantiate_ns
        self.children += 1
        self.generations += 1
        return NativeGraphSubmission(
            request=request,
            name=selected,
            manifest=manifest,
            context=self.context,
            executable=executable,
            stateful_registers=self.stateful_registers,
            local_cache_dependencies=self.local_cache_dependencies,
            context_owner=self,
        )

    def release(self, submission: NativeGraphSubmission) -> None:
        if submission.context_owner is not self:
            raise RuntimeError("submission belongs to a different transport context")
        if self.children <= 0:
            raise RuntimeError("submission context child ledger underflow")
        self.children -= 1

    def provenance(self) -> dict[str, Any]:
        if self.closed:
            if self._final_provenance is None:
                raise RuntimeError("closed submission context has no provenance")
            return dict(self._final_provenance)
        return {
            "schema_version": 1,
            "transport": self.name,
            "backend": self.backend,
            "gfx_arch": self.gfx_arch,
            "source": "hipengine_in_tree_rocr_pm4",
            "children": int(self.children),
            "generations": int(self.generations),
            "stateful_registers": bool(self.stateful_registers),
            "local_cache_dependencies": bool(self.local_cache_dependencies),
            "timestamps": bool(self.timestamps),
            "context_create_ns": int(self.context_create_ns),
            "last_graph_inspection_ns": int(self.last_graph_inspection_ns),
            "last_graph_inspection_phases_ns": dict(self.last_graph_inspection_phases_ns),
            "last_native_instantiate_ns": int(self.last_native_instantiate_ns),
            "graph_inspection_ns_total": int(self.graph_inspection_ns_total),
            "native_instantiate_ns_total": int(self.native_instantiate_ns_total),
            "closed": False,
            "native": NativeGraphSubmission._component_provenance(self.context),
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.children:
            raise RuntimeError(
                f"cannot close {self.name} submission context with {self.children} live graphs"
            )
        snapshot = self.provenance()
        if self.context.handle:
            self.context.close()
        self.closed = not self.context.handle
        if self.closed:
            snapshot["closed"] = True
            snapshot["native_context_closed"] = True
            self._final_provenance = snapshot


def _create_hipgraph_submission(
    request: GraphSubmissionRequest,
    selected: str,
) -> HipGraphSubmission:
    if selected != "hipgraph":
        raise ValueError("HIP graph factory received a non-hipgraph transport")
    graph_exec = int(request.runtime.graph_instantiate(request.graph))
    if not graph_exec:
        raise RuntimeError("HIP returned a null graph executable")
    return HipGraphSubmission(request=request, graph_exec=graph_exec)


def _create_gfx1100_submission_context(
    backend: str,
    gfx_arch: str,
    runtime: Any,
    selected: str,
) -> NativeGraphSubmissionContext:
    if selected not in _NATIVE_TRANSPORTS:
        raise ValueError("native submission context requires aql or pm4")
    if backend != "hip_gfx1100" or gfx_arch != "gfx1100":
        raise RuntimeError(
            "in-tree native graph submission is admitted only for hip_gfx1100/gfx1100"
        )
    stateful_registers = selected == "pm4"
    local_cache_dependencies = selected == "pm4"
    context_create_start_ns = time.perf_counter_ns()
    context = NativePm4Context.create(
        pci_bdf=runtime.device_pci_bus_id(),
        gfx_arch=gfx_arch,
    )
    context_create_ns = time.perf_counter_ns() - context_create_start_ns
    return NativeGraphSubmissionContext(
        backend=backend,
        gfx_arch=gfx_arch,
        runtime=runtime,
        name=selected,
        context=context,
        stateful_registers=stateful_registers,
        local_cache_dependencies=local_cache_dependencies,
        context_create_ns=context_create_ns,
    )


def _create_gfx1100_native_submission(
    request: GraphSubmissionRequest,
    selected: str,
) -> NativeGraphSubmission:
    if selected not in _NATIVE_TRANSPORTS:
        raise ValueError("native graph factory requires aql or pm4")
    if request.backend != "hip_gfx1100" or request.gfx_arch != "gfx1100":
        raise RuntimeError(
            "in-tree native graph submission is admitted only for hip_gfx1100/gfx1100"
        )
    manifest = inspect_hip_graph(
        request.runtime,
        request.graph,
        gfx_arch=request.gfx_arch,
        stream=request.capture_stream,
    )
    stateful_registers = selected == "pm4"
    local_cache_dependencies = selected == "pm4"
    context: NativePm4Context | None = None
    try:
        context = NativePm4Context.create(
            pci_bdf=request.runtime.device_pci_bus_id(),
            gfx_arch=request.gfx_arch,
        )
        executable = context.instantiate(
            manifest,
            stateful_registers=stateful_registers,
            local_cache_dependencies=local_cache_dependencies,
        )
    except Exception as operation_error:
        if context is not None:
            try:
                context.close()
            except Exception as teardown_error:
                raise RuntimeError(
                    f"native graph instantiation failed: {operation_error}; "
                    f"context cleanup also failed: {teardown_error}"
                ) from operation_error
        raise
    return NativeGraphSubmission(
        request=request,
        name=selected,
        manifest=manifest,
        context=context,
        executable=executable,
        stateful_registers=stateful_registers,
        local_cache_dependencies=local_cache_dependencies,
    )


def clear_submission_transport_registry_for_tests() -> None:
    """Clear registrations; production code must not call this helper."""

    global _BUILTINS_REGISTERED
    _TRANSPORTS.clear()
    _CONTEXT_FACTORIES.clear()
    _BUILTINS_REGISTERED = False


__all__ = [
    "DuplicateSubmissionTransportError",
    "GraphSubmission",
    "GraphSubmissionContext",
    "GraphSubmissionRequest",
    "HipGraphSubmission",
    "MissingSubmissionTransportError",
    "NativeGraphSubmission",
    "NativeGraphSubmissionContext",
    "SubmissionContextFactory",
    "SubmissionTransportFactory",
    "SubmissionTransportKey",
    "clear_submission_transport_registry_for_tests",
    "create_graph_submission",
    "create_graph_submission_context",
    "register_builtin_submission_transports",
    "register_submission_transport",
    "registered_submission_transports",
    "resolve_submission_transport_factory",
    "select_submission_transport",
]
