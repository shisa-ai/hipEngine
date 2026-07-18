"""Fixed-B2 target graph submission through NativeSpecCycle ABI v1.

This N1 launcher is intentionally narrow: one chain request, two candidates,
three active target rows, int64 metadata, FP32 hidden rows, and BF16 KV.  The
provider owns a state-generation-bound graph executable and its allocations.
The launcher owns nothing; it performs one native call that submits the graph
and synchronizes the control block's session-owned stream.

Proposal and accept/commit remain on the exact Python path.  Broader shapes,
dynamic position/cursor handling, cancellation, deadlines, and complete cycles
belong to later ABI stages and must fall back instead of being approximated.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import threading

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.speculative.native_cycle import (
    NativeSpecCycleControl,
    NativeSpecCycleDType,
    NativeSpecCycleResult,
    NativeSpecCycleResultC,
    NativeSpecCycleStage,
)

_SOURCE = Path(__file__).with_name("native_cycle_graph.cpp")
_ABI_HEADER = Path(__file__).with_name("native_cycle_abi.h")
_OUTPUT_NAME = "native_spec_cycle_graph.so"
_SYMBOL = "hipengine_native_spec_target_graph_launch_v1"


def _abi_header_cache_flag() -> str:
    """Make the included ABI header bytes part of the JIT build identity."""

    digest = hashlib.sha256(_ABI_HEADER.read_bytes()).hexdigest()
    return f"-DHIPENGINE_NATIVE_SPEC_CYCLE_ABI_HEADER_SHA256_{digest}=1"


def plan_native_spec_cycle_graph_launcher_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    target_arch: str | None = None,
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="native_spec_cycle_graph",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=(_abi_header_cache_flag(),),
        target_arch=target_arch,
        output_name=_OUTPUT_NAME,
    )


def build_native_spec_cycle_graph_launcher(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    target_arch: str | None = None,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="native_spec_cycle_graph",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=(_abi_header_cache_flag(),),
        target_arch=target_arch,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


class NativeSpecTargetGraphLauncher:
    """Submit one pre-bound B2 target graph through one C++ boundary.

    ``graph_exec`` and both function addresses are borrowed.  The caller must
    keep the graph executable, HIP runtime library, stream, and every control
    allocation alive until :meth:`launch` returns. This class never destroys or
    assumes ownership of those borrowed resources.
    """

    def __init__(
        self,
        *,
        graph_exec: int,
        graph_launch_fn: int,
        stream_synchronize_fn: int,
        bound_control: NativeSpecCycleControl | None = None,
        library: ctypes.CDLL | object | None = None,
        runtime: HipRuntime | None = None,
        compiler_version: str | None = None,
        profile: ProfileName = "baseline",
        target_arch: str | None = None,
        require_cached: bool = False,
    ) -> None:
        self._graph_exec = _positive_address("graph_exec", graph_exec)
        self._graph_launch_fn = _positive_address("graph_launch_fn", graph_launch_fn)
        self._stream_synchronize_fn = _positive_address(
            "stream_synchronize_fn",
            stream_synchronize_fn,
        )
        if bound_control is not None:
            if not isinstance(bound_control, NativeSpecCycleControl):
                raise TypeError("bound_control must be NativeSpecCycleControl")
            bound_control.validate()
            _validate_fixed_b2_target(bound_control)
        self._bound_signature = (
            None if bound_control is None else _graph_binding_signature(bound_control)
        )
        self._library = library or build_native_spec_cycle_graph_launcher(
            load=True,
            compiler_version=compiler_version,
            profile=profile,
            target_arch=target_arch,
            require_cached=require_cached,
        )
        self._runtime = runtime
        self._lock = threading.Lock()
        self._launch_count = 0

    @classmethod
    def from_hip_graph(
        cls,
        *,
        graph_exec: int,
        runtime: HipRuntime | None = None,
        bound_control: NativeSpecCycleControl | None = None,
        library: ctypes.CDLL | object | None = None,
        compiler_version: str | None = None,
        profile: ProfileName = "baseline",
        target_arch: str | None = None,
        require_cached: bool = False,
    ) -> "NativeSpecTargetGraphLauncher":
        runtime = runtime or get_hip_runtime()
        return cls(
            graph_exec=graph_exec,
            graph_launch_fn=_function_address(runtime.library.hipGraphLaunch),
            stream_synchronize_fn=_function_address(runtime.library.hipStreamSynchronize),
            bound_control=bound_control,
            library=library,
            runtime=runtime,
            compiler_version=compiler_version,
            profile=profile,
            target_arch=target_arch,
            require_cached=require_cached,
        )

    @property
    def launch_count(self) -> int:
        return self._launch_count

    @property
    def graph_exec(self) -> int:
        return self._graph_exec

    def launch(self, control: NativeSpecCycleControl) -> NativeSpecCycleResult:
        if not isinstance(control, NativeSpecCycleControl):
            raise TypeError("control must be NativeSpecCycleControl")
        control.validate()
        _validate_fixed_b2_target(control)
        if (
            self._bound_signature is not None
            and _graph_binding_signature(control) != self._bound_signature
        ):
            raise RuntimeError("native target graph control drifted from its state-bound capture")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native target graph launcher already in flight")
        try:
            raw_control = control.to_ctypes()
            raw_result = NativeSpecCycleResultC()
            fn = getattr(self._library, _SYMBOL)
            try:
                fn.argtypes = [
                    ctypes.POINTER(type(raw_control)),
                    ctypes.POINTER(NativeSpecCycleResultC),
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                fn.restype = ctypes.c_int32
            except (AttributeError, TypeError):
                # Unit-test doubles are ordinary Python callables.  Real CDLL
                # symbols always take the exact fixed signature above.
                pass
            error = fn(
                ctypes.byref(raw_control),
                ctypes.byref(raw_result),
                ctypes.c_void_p(self._graph_exec),
                ctypes.c_void_p(self._graph_launch_fn),
                ctypes.c_void_p(self._stream_synchronize_fn),
            )
            if int(error) != 0:
                raise RuntimeError(f"native target graph launcher rejected its call boundary: {int(error)}")
            result = NativeSpecCycleResult.from_ctypes(raw_result)
            result.validate_for(control)
            self._launch_count += 1
            return result
        finally:
            self._lock.release()


def create_native_spec_target_graph_launcher(
    *,
    graph_exec: int,
    runtime: HipRuntime | None = None,
    bound_control: NativeSpecCycleControl | None = None,
    library: ctypes.CDLL | object | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    target_arch: str | None = None,
    require_cached: bool = False,
) -> NativeSpecTargetGraphLauncher:
    """Registry factory for the gfx1100 GGUF fixed-B2 target graph route."""

    return NativeSpecTargetGraphLauncher.from_hip_graph(
        graph_exec=graph_exec,
        runtime=runtime,
        bound_control=bound_control,
        library=library,
        compiler_version=compiler_version,
        profile=profile,
        target_arch=target_arch,
        require_cached=require_cached,
    )


def _validate_fixed_b2_target(control: NativeSpecCycleControl) -> None:
    shape = control.shape
    if control.stages != NativeSpecCycleStage.VERIFY:
        raise ValueError("native target graph N1 supports VERIFY only")
    if (
        shape.request_count != 1
        or shape.row_count != 3
        or shape.active_row_count != 3
        or shape.candidate_count != 2
        or shape.active_candidate_count != 2
        or shape.candidate_budget != 2
    ):
        raise ValueError("native target graph N1 supports exactly one B2 chain bucket (1 request, 3 rows)")
    if shape.metadata_dtype is not NativeSpecCycleDType.INT64:
        raise ValueError("native target graph N1 requires INT64 metadata")
    if shape.hidden_dtype is not NativeSpecCycleDType.FP32:
        raise ValueError("native target graph N1 requires FP32 hidden rows")
    if shape.kv_dtype is not NativeSpecCycleDType.BF16:
        raise ValueError("native target graph N1 requires BF16 KV")
    if control.stream == 0:
        raise ValueError("native target graph N1 requires a session-owned stream")
    if control.deadline_ns != 0:
        raise ValueError("native target graph N1 does not yet support a deadline")
    if control.pointers.outputs.cancel_flag != 0:
        raise ValueError("native target graph N1 does not yet support cancellation")


def _graph_binding_signature(control: NativeSpecCycleControl) -> tuple[object, ...]:
    """Return fields embedded in a captured graph, excluding result identity."""

    return (
        control.abi_version,
        control.stages,
        control.mode,
        control.shape,
        control.pointers,
        control.stream,
    )


def _positive_address(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer raw address")
    if value <= 0 or value >= 1 << 64:
        raise ValueError(f"{name} must be a positive uint64 address")
    return value


def _function_address(fn) -> int:
    value = ctypes.cast(fn, ctypes.c_void_p).value
    if value is None or int(value) == 0:
        raise RuntimeError("resolved native function has a null address")
    return int(value)


register(
    KernelKey(
        "hip_gfx1100",
        "speculative_cycle",
        "w4_gguf",
        "native_v1_b2_target_graph",
    ),
    create_native_spec_target_graph_launcher,
    replace=True,
)


__all__ = [
    "NativeSpecTargetGraphLauncher",
    "build_native_spec_cycle_graph_launcher",
    "create_native_spec_target_graph_launcher",
    "plan_native_spec_cycle_graph_launcher_build",
]
