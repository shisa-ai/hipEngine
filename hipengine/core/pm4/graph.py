"""Strict kernel-only HIP graph inspection and immutable PM4 manifests."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

from hipengine.core.hip import (
    HIP_GRAPH_NODE_TYPE_KERNEL,
    HipDim3,
    HipKernelNodeParams,
)
from hipengine.core.pm4.elf import extract_elf_section, select_amdgpu_code_object
from hipengine.core.pm4.errors import Pm4InspectionError
from hipengine.core.pm4.kernarg import LaunchContext, pack_kernel_node_params
from hipengine.core.pm4.metadata import (
    AmdgpuKernelMetadata,
    parse_amdgpu_kernels,
    resolve_kernel_metadata,
)


class _GraphRuntime(Protocol):
    def graph_nodes(self, graph: int) -> tuple[int, ...]: ...

    def graph_edges(self, graph: int) -> tuple[tuple[int, int], ...]: ...

    def graph_node_type(self, node: int) -> int: ...

    def graph_kernel_node_params(self, node: int) -> HipKernelNodeParams: ...

    def kernel_name_ref_by_ptr(self, function: int, stream: int = 0) -> str: ...


@dataclass(frozen=True, slots=True)
class KernelNodeManifest:
    handle: int
    name: str
    loader_symbol: str
    function: int
    grid_blocks: tuple[int, int, int]
    grid_workitems: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_bytes: int
    kernarg: bytes
    dso_path: Path
    dso_sha256: str
    fatbin_sha256: str
    target_id: str
    hsaco: bytes
    hsaco_sha256: str
    kernarg_align: int
    group_segment_size: int
    private_segment_size: int
    dynamic_stack: bool
    wavefront_size: int


@dataclass(frozen=True, slots=True)
class HipGraphManifest:
    graph_handle: int
    gfx_arch: str
    order: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    nodes: tuple[KernelNodeManifest, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _DsoImage:
    path: Path
    sha256: str
    fatbin_sha256: str
    target_id: str
    hsaco: bytes
    hsaco_sha256: str
    kernels: dict


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


_LIBDL: ctypes.CDLL | None = None
_DSO_CACHE: dict[tuple[Path, str, int, int, int, int, int], _DsoImage] = {}
_DSO_CACHE_LOCK = Lock()


def resolve_dso_for_function(function: int) -> Path:
    """Resolve a live host function pointer to its owning shared object."""

    if function <= 0:
        raise Pm4InspectionError("kernel function pointer is null")
    global _LIBDL
    if _LIBDL is None:
        name = ctypes.util.find_library("dl") or "libdl.so.2"
        try:
            library = ctypes.CDLL(name)
        except OSError as exc:
            raise Pm4InspectionError(f"cannot load dladdr provider {name!r}") from exc
        library.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        library.dladdr.restype = ctypes.c_int
        _LIBDL = library
    info = _DlInfo()
    if _LIBDL.dladdr(ctypes.c_void_p(function), ctypes.byref(info)) == 0 or not info.dli_fname:
        raise Pm4InspectionError(f"dladdr could not resolve kernel function {function:#x}")
    try:
        path = (
            Path(info.dli_fname.decode("utf-8", errors="strict")).expanduser().resolve(strict=True)
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise Pm4InspectionError("dladdr returned an invalid shared-object path") from exc
    if not path.is_file():
        raise Pm4InspectionError(f"kernel shared object is not a regular file: {path}")
    return path


def _read_stable(path: Path) -> bytes:
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise Pm4InspectionError(f"cannot read kernel shared object {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(data) != after.st_size:
        raise Pm4InspectionError(f"kernel shared object changed while being inspected: {path}")
    return data


def _dso_identity(path: Path, gfx_arch: str) -> tuple[Path, str, int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise Pm4InspectionError(f"cannot stat kernel shared object {path}") from exc
    return (
        path,
        gfx_arch,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _load_dso(path: Path, gfx_arch: str) -> _DsoImage:
    identity = _dso_identity(path, gfx_arch)
    with _DSO_CACHE_LOCK:
        cached = _DSO_CACHE.get(identity)
    if cached is not None:
        return cached

    data = _read_stable(path)
    fatbin = extract_elf_section(data, ".hip_fatbin")
    selected = select_amdgpu_code_object(fatbin, gfx_arch)
    kernels = parse_amdgpu_kernels(selected.image)
    result = _DsoImage(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        fatbin_sha256=hashlib.sha256(fatbin).hexdigest(),
        target_id=selected.target_id,
        hsaco=selected.image,
        hsaco_sha256=selected.sha256,
        kernels=kernels,
    )
    if _dso_identity(path, gfx_arch) != identity:
        raise Pm4InspectionError(f"kernel shared object changed while being cached: {path}")
    with _DSO_CACHE_LOCK:
        stale = [key for key in _DSO_CACHE if key[:2] == identity[:2] and key != identity]
        for key in stale:
            del _DSO_CACHE[key]
        _DSO_CACHE[identity] = result
    return result


def clear_dso_cache_for_tests() -> None:
    with _DSO_CACHE_LOCK:
        _DSO_CACHE.clear()


def topological_order(
    nodes: tuple[int, ...], edges: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    """Return a deterministic handle-sorted Kahn order for one DAG."""

    if not nodes:
        raise Pm4InspectionError("HIP graph contains no nodes")
    if any(node <= 0 for node in nodes):
        raise Pm4InspectionError("HIP graph contains a null node handle")
    node_set = set(nodes)
    if len(node_set) != len(nodes):
        raise Pm4InspectionError("HIP graph contains a duplicate node handle")
    if len(set(edges)) != len(edges):
        raise Pm4InspectionError("HIP graph contains a duplicate dependency edge")

    indegree = {node: 0 for node in node_set}
    outgoing: dict[int, list[int]] = {node: [] for node in node_set}
    for source, destination in edges:
        if source not in node_set or destination not in node_set:
            raise Pm4InspectionError("HIP graph edge references an unknown node")
        if source == destination:
            raise Pm4InspectionError("HIP graph contains a dependency cycle")
        outgoing[source].append(destination)
        indegree[destination] += 1

    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[int] = []
    while ready:
        source = heapq.heappop(ready)
        ordered.append(source)
        for destination in sorted(outgoing[source]):
            indegree[destination] -= 1
            if indegree[destination] == 0:
                heapq.heappush(ready, destination)
    if len(ordered) != len(nodes):
        raise Pm4InspectionError("HIP graph contains a dependency cycle")
    return tuple(ordered)


def _dimensions(value: HipDim3, label: str) -> tuple[int, int, int]:
    result = (int(value.x), int(value.y), int(value.z))
    if any(dimension <= 0 for dimension in result):
        raise Pm4InspectionError(f"HIP kernel {label} dimensions must be positive")
    return result


def _grid_workitems(
    grid_blocks: tuple[int, int, int], block: tuple[int, int, int]
) -> tuple[int, int, int]:
    result = tuple(grid_blocks[index] * block[index] for index in range(3))
    if any(value > 0xFFFFFFFF for value in result):
        raise Pm4InspectionError("HIP kernel global grid exceeds uint32")
    return result  # type: ignore[return-value]


def _fingerprint(
    gfx_arch: str,
    order: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    nodes: tuple[KernelNodeManifest, ...],
) -> str:
    digest = hashlib.sha256()

    def add_bytes(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)

    def add_text(value: str) -> None:
        add_bytes(value.encode("utf-8"))

    add_text("hipengine-pm4-graph-v1")
    add_text(gfx_arch)
    for node in order:
        digest.update(node.to_bytes(8, "little"))
    for source, destination in edges:
        digest.update(source.to_bytes(8, "little"))
        digest.update(destination.to_bytes(8, "little"))
    for node in nodes:
        add_text(node.name)
        add_text(node.loader_symbol)
        add_text(node.dso_sha256)
        add_text(node.hsaco_sha256)
        add_text(node.target_id)
        for value in (*node.grid_blocks, *node.block, node.dynamic_shared_bytes):
            digest.update(value.to_bytes(8, "little"))
        add_bytes(node.kernarg)
    return digest.hexdigest()


def inspect_hip_graph(
    runtime: _GraphRuntime,
    graph: int,
    *,
    gfx_arch: str,
    stream: int = 0,
    dso_resolver: Callable[[int], Path] = resolve_dso_for_function,
    timings: dict[str, int] | None = None,
) -> HipGraphManifest:
    """Inspect one live native graph and own every byte required for lowering."""

    total_start_ns = time.perf_counter_ns()
    enumeration_ns = 0
    node_api_ns = 0
    function_resolution_ns = 0
    dso_resolution_ns = 0
    dso_load_ns = 0
    metadata_resolution_ns = 0
    kernarg_pack_ns = 0
    revalidation_ns = 0

    if graph <= 0:
        raise Pm4InspectionError("HIP graph handle is null")
    if not gfx_arch.startswith("gfx"):
        raise Pm4InspectionError(f"invalid gfx architecture {gfx_arch!r}")
    try:
        phase_start_ns = time.perf_counter_ns()
        handles = tuple(int(node) for node in runtime.graph_nodes(graph))
        edges = tuple(
            (int(source), int(destination)) for source, destination in runtime.graph_edges(graph)
        )
        enumeration_ns += time.perf_counter_ns() - phase_start_ns
    except Exception as exc:
        if isinstance(exc, Pm4InspectionError):
            raise
        raise Pm4InspectionError("HIP graph enumeration failed") from exc
    order = topological_order(handles, edges)

    dso_cache: dict[Path, _DsoImage] = {}
    function_cache: dict[int, tuple[str, Path, _DsoImage, AmdgpuKernelMetadata]] = {}
    metadata_by_handle: dict[int, AmdgpuKernelMetadata] = {}
    manifests: list[KernelNodeManifest] = []
    for handle in order:
        phase_start_ns = time.perf_counter_ns()
        node_type = int(runtime.graph_node_type(handle))
        if node_type != HIP_GRAPH_NODE_TYPE_KERNEL:
            raise Pm4InspectionError(
                f"unsupported HIP graph node type {node_type} at handle {handle:#x}"
            )
        params = runtime.graph_kernel_node_params(handle)
        node_api_ns += time.perf_counter_ns() - phase_start_ns
        function = int(params.func or 0)
        if function <= 0:
            raise Pm4InspectionError(f"HIP kernel node {handle:#x} has a null function")
        resolved = function_cache.get(function)
        if resolved is None:
            try:
                phase_start_ns = time.perf_counter_ns()
                name = runtime.kernel_name_ref_by_ptr(function, stream)
                function_resolution_ns += time.perf_counter_ns() - phase_start_ns
            except Exception as exc:
                raise Pm4InspectionError(
                    f"cannot resolve HIP kernel node {handle:#x} name"
                ) from exc
            if not name or "\0" in name:
                raise Pm4InspectionError("HIP returned an invalid kernel name")
            try:
                phase_start_ns = time.perf_counter_ns()
                path = Path(dso_resolver(function)).expanduser().resolve(strict=True)
                dso_resolution_ns += time.perf_counter_ns() - phase_start_ns
            except (OSError, TypeError) as exc:
                raise Pm4InspectionError(f"cannot resolve DSO for kernel {name!r}") from exc
            dso = dso_cache.get(path)
            if dso is None:
                phase_start_ns = time.perf_counter_ns()
                dso = _load_dso(path, gfx_arch)
                dso_load_ns += time.perf_counter_ns() - phase_start_ns
                dso_cache[path] = dso
            phase_start_ns = time.perf_counter_ns()
            metadata = resolve_kernel_metadata(dso.kernels, name)
            metadata_resolution_ns += time.perf_counter_ns() - phase_start_ns
            resolved = (name, path, dso, metadata)
            function_cache[function] = resolved
        name, path, dso, metadata = resolved
        metadata_by_handle[handle] = metadata

        phase_start_ns = time.perf_counter_ns()
        grid_blocks = _dimensions(params.gridDim, "grid")
        block = _dimensions(params.blockDim, "block")
        dynamic_shared = int(params.sharedMemBytes)
        context = LaunchContext(grid_blocks, block, dynamic_shared)
        kernarg = pack_kernel_node_params(
            metadata,
            params.kernelParams,
            params.extra,
            context,
        )
        kernarg_pack_ns += time.perf_counter_ns() - phase_start_ns
        manifests.append(
            KernelNodeManifest(
                handle=handle,
                name=name,
                loader_symbol=metadata.symbol,
                function=function,
                grid_blocks=grid_blocks,
                grid_workitems=_grid_workitems(grid_blocks, block),
                block=block,
                dynamic_shared_bytes=dynamic_shared,
                kernarg=kernarg,
                dso_path=path,
                dso_sha256=dso.sha256,
                fatbin_sha256=dso.fatbin_sha256,
                target_id=dso.target_id,
                hsaco=dso.hsaco,
                hsaco_sha256=dso.hsaco_sha256,
                kernarg_align=metadata.kernarg_align,
                group_segment_size=metadata.group_segment_size,
                private_segment_size=metadata.private_segment_size,
                dynamic_stack=metadata.dynamic_stack,
                wavefront_size=metadata.wavefront_size,
            )
        )

    # Re-enumerate and recopy every launch field after the first complete pass.
    # Topology-only validation would miss concurrent scalar/pointer mutation and
    # could otherwise produce a mixed graph generation.
    phase_start_ns = time.perf_counter_ns()
    if tuple(runtime.graph_nodes(graph)) != handles or tuple(runtime.graph_edges(graph)) != edges:
        raise Pm4InspectionError("HIP graph changed during inspection")
    for node in manifests:
        try:
            if int(runtime.graph_node_type(node.handle)) != HIP_GRAPH_NODE_TYPE_KERNEL:
                raise Pm4InspectionError("kernel node type changed")
            params = runtime.graph_kernel_node_params(node.handle)
            function = int(params.func or 0)
            grid_blocks = _dimensions(params.gridDim, "grid")
            block = _dimensions(params.blockDim, "block")
            dynamic_shared = int(params.sharedMemBytes)
            kernarg = pack_kernel_node_params(
                metadata_by_handle[node.handle],
                params.kernelParams,
                params.extra,
                LaunchContext(grid_blocks, block, dynamic_shared),
            )
        except Exception as exc:
            raise Pm4InspectionError("HIP graph changed during inspection") from exc
        if (
            function != node.function
            or grid_blocks != node.grid_blocks
            or block != node.block
            or dynamic_shared != node.dynamic_shared_bytes
            or kernarg != node.kernarg
        ):
            raise Pm4InspectionError("HIP graph changed during inspection")
    revalidation_ns += time.perf_counter_ns() - phase_start_ns
    nodes = tuple(manifests)
    manifest = HipGraphManifest(
        graph_handle=graph,
        gfx_arch=gfx_arch,
        order=order,
        edges=edges,
        nodes=nodes,
        fingerprint=_fingerprint(gfx_arch, order, edges, nodes),
    )
    if timings is not None:
        timings.update(
            {
                "total_ns": time.perf_counter_ns() - total_start_ns,
                "enumeration_ns": enumeration_ns,
                "node_api_ns": node_api_ns,
                "function_resolution_ns": function_resolution_ns,
                "dso_resolution_ns": dso_resolution_ns,
                "dso_load_ns": dso_load_ns,
                "metadata_resolution_ns": metadata_resolution_ns,
                "kernarg_pack_ns": kernarg_pack_ns,
                "revalidation_ns": revalidation_ns,
            }
        )
    return manifest
