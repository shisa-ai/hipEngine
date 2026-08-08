"""ctypes ownership wrapper for the minimal public-ROCr/PM4 native core."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from typing import Any, Literal

from hipengine.core.pm4.graph import HipGraphManifest
from hipengine.core.pm4.native_build import build_pm4_native

NativeTransport = Literal["aql", "pm4"]
_ERROR_BYTES = 4096
_ABI_VERSION = 2
_EXECUTABLE_FLAG_TIMESTAMPS = 1 << 0
_EXECUTABLE_FLAG_STATEFUL_REGISTERS = 1 << 1
_EXECUTABLE_FLAG_LOCAL_CACHE_DEPENDENCIES = 1 << 2


class NativePm4Error(RuntimeError):
    """Raised when the checked native HSA/PM4 ABI reports failure."""


class _NativeNode(ctypes.Structure):
    _fields_ = [
        ("hsaco", ctypes.POINTER(ctypes.c_uint8)),
        ("hsaco_size", ctypes.c_size_t),
        ("symbol", ctypes.c_char_p),
        ("kernarg", ctypes.POINTER(ctypes.c_uint8)),
        ("kernarg_size", ctypes.c_uint32),
        ("kernarg_align", ctypes.c_uint32),
        ("grid", ctypes.c_uint32 * 3),
        ("block", ctypes.c_uint32 * 3),
        ("dynamic_lds", ctypes.c_uint32),
        ("expected_group_segment_size", ctypes.c_uint32),
        ("expected_private_segment_size", ctypes.c_uint32),
        ("expected_dynamic_stack", ctypes.c_uint32),
        ("expected_wavefront_size", ctypes.c_uint32),
    ]


def _configure(library: Any) -> None:
    vp = ctypes.c_void_p
    charp = ctypes.POINTER(ctypes.c_char)
    library.he_pm4_native_abi_version.argtypes = []
    library.he_pm4_native_abi_version.restype = ctypes.c_uint32
    library.he_pm4_node_size.argtypes = []
    library.he_pm4_node_size.restype = ctypes.c_size_t
    library.he_pm4_context_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(vp),
        charp,
        ctypes.c_size_t,
    ]
    library.he_pm4_context_create.restype = ctypes.c_int
    library.he_pm4_context_retire_queue.argtypes = [vp, charp, ctypes.c_size_t]
    library.he_pm4_context_retire_queue.restype = ctypes.c_int
    library.he_pm4_context_destroy.argtypes = [vp, charp, ctypes.c_size_t]
    library.he_pm4_context_destroy.restype = ctypes.c_int
    library.he_pm4_buffer_create.argtypes = [
        vp,
        ctypes.c_size_t,
        ctypes.POINTER(vp),
        ctypes.POINTER(ctypes.c_uint64),
        charp,
        ctypes.c_size_t,
    ]
    library.he_pm4_buffer_create.restype = ctypes.c_int
    library.he_pm4_buffer_write.argtypes = [
        vp,
        ctypes.c_size_t,
        vp,
        ctypes.c_size_t,
        charp,
        ctypes.c_size_t,
    ]
    library.he_pm4_buffer_write.restype = ctypes.c_int
    library.he_pm4_buffer_read.argtypes = [
        vp,
        ctypes.c_size_t,
        vp,
        ctypes.c_size_t,
        charp,
        ctypes.c_size_t,
    ]
    library.he_pm4_buffer_read.restype = ctypes.c_int
    library.he_pm4_buffer_destroy.argtypes = [vp, charp, ctypes.c_size_t]
    library.he_pm4_buffer_destroy.restype = ctypes.c_int
    library.he_pm4_executable_create_ex.argtypes = [
        vp,
        ctypes.POINTER(_NativeNode),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(vp),
        charp,
        ctypes.c_size_t,
    ]
    library.he_pm4_executable_create_ex.restype = ctypes.c_int
    library.he_pm4_executable_destroy.argtypes = [vp, charp, ctypes.c_size_t]
    library.he_pm4_executable_destroy.restype = ctypes.c_int
    for name in ("he_pm4_launch_aql", "he_pm4_launch_pm4"):
        function = getattr(library, name)
        function.argtypes = [vp, ctypes.c_uint64, charp, ctypes.c_size_t]
        function.restype = ctypes.c_int
    for name in ("he_pm4_context_json", "he_pm4_executable_json"):
        function = getattr(library, name)
        function.argtypes = [
            vp,
            charp,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            charp,
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


def _call(function: Any, *args: Any) -> None:
    error = ctypes.create_string_buffer(_ERROR_BYTES)
    status = int(function(*args, error, ctypes.c_size_t(len(error))))
    if status != 0:
        detail = error.value.decode("utf-8", errors="replace") or "unknown native PM4 error"
        raise NativePm4Error(detail)


def _json_query(function: Any, handle: int) -> dict[str, Any]:
    required = ctypes.c_size_t()
    error = ctypes.create_string_buffer(_ERROR_BYTES)
    status = int(
        function(
            ctypes.c_void_p(handle),
            None,
            ctypes.c_size_t(0),
            ctypes.byref(required),
            error,
            ctypes.c_size_t(len(error)),
        )
    )
    if status != 0:
        raise NativePm4Error(error.value.decode("utf-8", errors="replace"))
    if required.value <= 1 or required.value > (1 << 20):
        raise NativePm4Error("native PM4 JSON query returned an invalid size")
    output = ctypes.create_string_buffer(required.value)
    error = ctypes.create_string_buffer(_ERROR_BYTES)
    status = int(
        function(
            ctypes.c_void_p(handle),
            output,
            ctypes.c_size_t(len(output)),
            ctypes.byref(required),
            error,
            ctypes.c_size_t(len(error)),
        )
    )
    if status != 0:
        raise NativePm4Error(error.value.decode("utf-8", errors="replace"))
    value = json.loads(output.value.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise NativePm4Error("native PM4 provenance is not a JSON object")
    return value


@dataclass
class NativePm4Context:
    """One persistent public-HSA queue matched to an exact HIP physical GPU."""

    library: Any
    handle: int
    gfx_arch: str
    pci_bdf: str

    @classmethod
    def create(
        cls,
        *,
        pci_bdf: str,
        gfx_arch: str = "gfx1100",
        library: Any | None = None,
        compiler_version: str | None = None,
        require_cached: bool = False,
    ) -> "NativePm4Context":
        library = library or build_pm4_native(
            load=True,
            compiler_version=compiler_version,
            target_arch=gfx_arch,
            require_cached=require_cached,
        )
        _configure(library)
        version = int(library.he_pm4_native_abi_version())
        if version != _ABI_VERSION:
            raise NativePm4Error(
                f"native PM4 ABI version {version} does not match Python {_ABI_VERSION}"
            )
        native_node_size = int(library.he_pm4_node_size())
        python_node_size = ctypes.sizeof(_NativeNode)
        if native_node_size != python_node_size:
            raise NativePm4Error(
                f"native PM4 node ABI size {native_node_size} does not match Python {python_node_size}"
            )
        output = ctypes.c_void_p()
        _call(
            library.he_pm4_context_create,
            pci_bdf.encode("ascii"),
            gfx_arch.encode("ascii"),
            ctypes.byref(output),
        )
        if not output.value:
            raise NativePm4Error("native PM4 context creation returned null")
        return cls(library, int(output.value), gfx_arch, pci_bdf)

    def instantiate(
        self,
        manifest: HipGraphManifest,
        *,
        timestamps: bool = False,
        stateful_registers: bool = False,
        local_cache_dependencies: bool = False,
    ) -> "NativePm4Executable":
        if not self.handle:
            raise NativePm4Error("native PM4 context is closed")
        if manifest.gfx_arch != self.gfx_arch:
            raise NativePm4Error(
                f"manifest architecture {manifest.gfx_arch} does not match {self.gfx_arch}"
            )
        hsaco_buffers: dict[str, Any] = {}
        kernarg_buffers: list[Any] = []
        symbol_buffers: list[bytes] = []
        native_nodes = (_NativeNode * len(manifest.nodes))()
        for index, node in enumerate(manifest.nodes):
            hsaco_buffer = hsaco_buffers.get(node.hsaco_sha256)
            if hsaco_buffer is None:
                hsaco_buffer = (ctypes.c_uint8 * len(node.hsaco)).from_buffer_copy(node.hsaco)
                hsaco_buffers[node.hsaco_sha256] = hsaco_buffer
            if node.kernarg:
                kernarg_buffer = (ctypes.c_uint8 * len(node.kernarg)).from_buffer_copy(node.kernarg)
                kernarg_buffers.append(kernarg_buffer)
                kernarg_pointer = ctypes.cast(kernarg_buffer, ctypes.POINTER(ctypes.c_uint8))
            else:
                kernarg_pointer = ctypes.POINTER(ctypes.c_uint8)()
            symbol = node.loader_symbol.encode("utf-8")
            if b"\0" in symbol:
                raise NativePm4Error("kernel loader symbol contains NUL")
            symbol_buffers.append(symbol)
            native_nodes[index] = _NativeNode(
                ctypes.cast(hsaco_buffer, ctypes.POINTER(ctypes.c_uint8)),
                len(node.hsaco),
                symbol,
                kernarg_pointer,
                len(node.kernarg),
                node.kernarg_align,
                (ctypes.c_uint32 * 3)(*node.grid_workitems),
                (ctypes.c_uint32 * 3)(*node.block),
                node.dynamic_shared_bytes,
                node.group_segment_size,
                node.private_segment_size,
                int(node.dynamic_stack),
                node.wavefront_size,
            )
        output = ctypes.c_void_p()
        flags = 0
        if timestamps:
            flags |= _EXECUTABLE_FLAG_TIMESTAMPS
        if stateful_registers:
            flags |= _EXECUTABLE_FLAG_STATEFUL_REGISTERS
        if local_cache_dependencies:
            flags |= _EXECUTABLE_FLAG_LOCAL_CACHE_DEPENDENCIES
        _call(
            self.library.he_pm4_executable_create_ex,
            ctypes.c_void_p(self.handle),
            native_nodes,
            ctypes.c_size_t(len(native_nodes)),
            ctypes.c_uint32(flags),
            ctypes.byref(output),
        )
        if not output.value:
            raise NativePm4Error("native PM4 instantiation returned null")
        return NativePm4Executable(
            context=self,
            handle=int(output.value),
            fingerprint=manifest.fingerprint,
        )

    def allocate_buffer(self, nbytes: int) -> "NativePm4Buffer":
        if not self.handle:
            raise NativePm4Error("native PM4 context is closed")
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        output = ctypes.c_void_p()
        address = ctypes.c_uint64()
        _call(
            self.library.he_pm4_buffer_create,
            ctypes.c_void_p(self.handle),
            ctypes.c_size_t(nbytes),
            ctypes.byref(output),
            ctypes.byref(address),
        )
        if not output.value or not address.value:
            raise NativePm4Error("native HSA buffer creation returned null")
        return NativePm4Buffer(self, int(output.value), int(address.value), nbytes)

    def provenance(self) -> dict[str, Any]:
        if not self.handle:
            raise NativePm4Error("native PM4 context is closed")
        return _json_query(self.library.he_pm4_context_json, self.handle)

    def retire_queue(self) -> None:
        """Destroy the queue first while retaining signals and packet pointees."""

        if not self.handle:
            raise NativePm4Error("native PM4 context is closed")
        _call(self.library.he_pm4_context_retire_queue, ctypes.c_void_p(self.handle))

    def close(self) -> None:
        if not self.handle:
            return
        _call(self.library.he_pm4_context_destroy, ctypes.c_void_p(self.handle))
        self.handle = 0

    def __enter__(self) -> "NativePm4Context":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass
class NativePm4Buffer:
    """CPU-writable fine-grained HSA allocation accessible by the matched GPU."""

    context: NativePm4Context
    handle: int
    address: int
    nbytes: int

    def write(self, data: bytes, *, offset: int = 0) -> None:
        if not self.handle:
            raise NativePm4Error("native HSA buffer is closed")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        payload = bytes(data)
        source = ctypes.create_string_buffer(payload, len(payload)) if payload else None
        _call(
            self.context.library.he_pm4_buffer_write,
            ctypes.c_void_p(self.handle),
            ctypes.c_size_t(offset),
            None if source is None else ctypes.cast(source, ctypes.c_void_p),
            ctypes.c_size_t(len(payload)),
        )

    def read(self, nbytes: int | None = None, *, offset: int = 0) -> bytes:
        if not self.handle:
            raise NativePm4Error("native HSA buffer is closed")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        size = self.nbytes - offset if nbytes is None else nbytes
        if size < 0:
            raise ValueError("nbytes is invalid")
        output = ctypes.create_string_buffer(size)
        _call(
            self.context.library.he_pm4_buffer_read,
            ctypes.c_void_p(self.handle),
            ctypes.c_size_t(offset),
            ctypes.cast(output, ctypes.c_void_p),
            ctypes.c_size_t(size),
        )
        return bytes(output.raw)

    def close(self) -> None:
        if not self.handle:
            return
        _call(self.context.library.he_pm4_buffer_destroy, ctypes.c_void_p(self.handle))
        self.handle = 0
        self.address = 0

    def __enter__(self) -> "NativePm4Buffer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass
class NativePm4Executable:
    """One exact graph generation with retained kernargs, modules, and PM4 IB."""

    context: NativePm4Context
    handle: int
    fingerprint: str

    def launch(self, transport: NativeTransport, *, timeout_seconds: float = 5.0) -> None:
        if not self.handle:
            raise NativePm4Error("native PM4 executable is closed")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        timeout_ns = round(timeout_seconds * 1_000_000_000)
        if not 1 <= timeout_ns <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("timeout_seconds is outside uint64 nanoseconds")
        if transport == "aql":
            function = self.context.library.he_pm4_launch_aql
        elif transport == "pm4":
            function = self.context.library.he_pm4_launch_pm4
        else:
            raise ValueError("transport must be 'aql' or 'pm4'")
        _call(function, ctypes.c_void_p(self.handle), ctypes.c_uint64(timeout_ns))

    def provenance(self) -> dict[str, Any]:
        if not self.handle:
            raise NativePm4Error("native PM4 executable is closed")
        value = _json_query(self.context.library.he_pm4_executable_json, self.handle)
        value["graph_fingerprint"] = self.fingerprint
        return value

    def close(self) -> None:
        if not self.handle:
            return
        _call(
            self.context.library.he_pm4_executable_destroy,
            ctypes.c_void_p(self.handle),
        )
        self.handle = 0

    def __enter__(self) -> "NativePm4Executable":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
