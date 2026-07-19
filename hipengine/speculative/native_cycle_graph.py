"""Reusable small-chain target graph submission through NativeSpecCycle ABI v1.

This launcher is intentionally narrow: one chain request, one or two candidates,
two or three active target rows, int64 metadata, FP32 hidden rows, and BF16 KV.  The
provider owns a state-generation-bound graph executable and its allocations.
The launcher owns nothing; it performs one native call that submits the graph
and synchronizes the control block's session-owned stream.

The N2 bucket may also capture device acceptance, selected state/hidden commit,
and cursor update behind the same submission. A separate proposal-only bucket
can submit an existing strict B1/B2 NextN device chain through the same ABI.
The provider-target variant reuses the launcher for the shared PARO MTP/DFlash
single-request B1/B2/B3/B4/B5/B8 target+accept graph, with FP16 verifier rows
or BF16 sidecar hidden taps and the provider's existing Python commit path.
Unsupported shapes remain on the exact Python chain. Cancellation, deadlines,
and multi-cycle execution belong
to later ABI stages and must fall back instead of being approximated.
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
    NativeSpecCycleControlC,
    NativeSpecCycleDType,
    NativeSpecCycleResult,
    NativeSpecCycleResultC,
    NativeSpecCycleStage,
)

_SOURCE = Path(__file__).with_name("native_cycle_graph.cpp")
_ABI_HEADER = Path(__file__).with_name("native_cycle_abi.h")
_OUTPUT_NAME = "native_spec_cycle_graph.so"
_TARGET_SYMBOL = "hipengine_native_spec_target_graph_launch_v1"
_PROPOSAL_SYMBOL = "hipengine_native_spec_proposal_graph_launch_v1"
_GGUF_TARGET_GRAPH_BACKENDS = ("hip_gfx1100", "hip_gfx1151")


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
        _kind: str = "target",
    ) -> None:
        if _kind not in {"target", "provider_target", "proposal"}:
            raise ValueError(
                "native graph launcher kind must be target, provider_target, or proposal"
            )
        self._kind = str(_kind)
        self._symbol = _PROPOSAL_SYMBOL if self._kind == "proposal" else _TARGET_SYMBOL
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
            _validate_graph_control(self._kind, bound_control)
        self._bound_signature = (
            None if bound_control is None else _graph_binding_signature(bound_control)
        )
        self._bound_control = bound_control
        self._raw_bound_control = (
            None if bound_control is None else bound_control.to_ctypes()
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
        self._fn = getattr(self._library, self._symbol)
        try:
            self._fn.argtypes = [
                ctypes.POINTER(NativeSpecCycleControlC),
                ctypes.POINTER(NativeSpecCycleResultC),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._fn.restype = ctypes.c_int32
        except (AttributeError, TypeError):
            # Unit-test doubles are ordinary Python callables. Real CDLL
            # symbols always take the exact fixed signature above.
            pass

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
        _validate_graph_control(self._kind, control)
        if (
            self._bound_signature is not None
            and _graph_binding_signature(control) != self._bound_signature
        ):
            raise RuntimeError(
                f"native {self._kind} graph control drifted from its state-bound capture"
            )
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(f"native {self._kind} graph launcher already in flight")
        try:
            result = self._invoke(control.to_ctypes())
            result.validate_for(control)
            self._launch_count += 1
            return result
        finally:
            self._lock.release()

    def launch_bound(
        self,
        *,
        cycle_id: int,
        transaction_id: int,
    ) -> NativeSpecCycleResult:
        """Replay a validated binding while mutating only result identity fields."""

        if self._raw_bound_control is None or self._bound_control is None:
            raise RuntimeError("native graph launcher has no bound control")
        cycle = _uint64_identity("cycle_id", cycle_id)
        transaction = _uint64_identity("transaction_id", transaction_id)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(f"native {self._kind} graph launcher already in flight")
        try:
            self._raw_bound_control.cycle_id = cycle
            self._raw_bound_control.transaction_id = transaction
            result = self._invoke(self._raw_bound_control)
            result._validate_for_binding(
                cycle_id=cycle,
                transaction_id=transaction,
                request_count=self._bound_control.shape.request_count,
                stages=self._bound_control.stages,
                output_stride=self._bound_control.shape.output_stride,
            )
            self._launch_count += 1
            return result
        finally:
            self._lock.release()

    def _invoke(self, raw_control: NativeSpecCycleControlC) -> NativeSpecCycleResult:
        raw_result = NativeSpecCycleResultC()
        error = self._fn(
            ctypes.byref(raw_control),
            ctypes.byref(raw_result),
            ctypes.c_void_p(self._graph_exec),
            ctypes.c_void_p(self._graph_launch_fn),
            ctypes.c_void_p(self._stream_synchronize_fn),
        )
        if int(error) != 0:
            raise RuntimeError(
                f"native {self._kind} graph launcher rejected its call boundary: {int(error)}"
            )
        return NativeSpecCycleResult.from_ctypes(raw_result)


class NativeSpecProposalGraphLauncher(NativeSpecTargetGraphLauncher):
    """Submit one pre-bound B1/B2 NextN proposal graph through one C++ boundary."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, _kind="proposal")


class NativeSpecProviderTargetGraphLauncher(NativeSpecTargetGraphLauncher):
    """Submit a shared PARO MTP/DFlash chain target graph through ABI v1."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, _kind="provider_target")


def create_native_spec_proposal_graph_launcher(
    *,
    graph_exec: int,
    runtime: HipRuntime | None = None,
    bound_control: NativeSpecCycleControl | None = None,
    library: ctypes.CDLL | object | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    target_arch: str | None = None,
    require_cached: bool = False,
) -> NativeSpecProposalGraphLauncher:
    """Registry factory for a gfx11 GGUF B1/B2 NextN proposal graph."""

    runtime = runtime or get_hip_runtime()
    return NativeSpecProposalGraphLauncher(
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


def create_native_spec_provider_target_graph_launcher(
    *,
    graph_exec: int,
    runtime: HipRuntime | None = None,
    bound_control: NativeSpecCycleControl | None = None,
    library: ctypes.CDLL | object | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    target_arch: str | None = None,
    require_cached: bool = False,
) -> NativeSpecProviderTargetGraphLauncher:
    """Registry factory for shared PARO MTP/DFlash chain target graphs."""

    runtime = runtime or get_hip_runtime()
    return NativeSpecProviderTargetGraphLauncher(
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
    """Registry factory for the gfx11 GGUF fixed-B2 target graph route."""

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


def _validate_graph_control(kind: str, control: NativeSpecCycleControl) -> None:
    if kind == "proposal":
        _validate_small_chain_proposal(control)
    elif kind == "provider_target":
        _validate_provider_chain_target(control)
    else:
        _validate_small_chain_target(control)


def _validate_small_chain_proposal(control: NativeSpecCycleControl) -> None:
    shape = control.shape
    candidate_rows = int(shape.row_count) - 1
    if control.stages != NativeSpecCycleStage.PROPOSE:
        raise ValueError("native proposal graph supports only PROPOSE")
    if (
        shape.request_count != 1
        or shape.row_count not in {2, 3}
        or shape.active_row_count != shape.row_count
        or shape.candidate_count != candidate_rows
        or shape.active_candidate_count != candidate_rows
        or shape.candidate_budget != candidate_rows
    ):
        raise ValueError("native proposal graph supports one B1/B2 chain bucket (1 request, 2-3 rows)")
    if shape.metadata_dtype is not NativeSpecCycleDType.INT64:
        raise ValueError("native proposal graph requires INT64 dynamic metadata")
    if shape.hidden_dtype is not NativeSpecCycleDType.FP32:
        raise ValueError("native proposal graph requires FP32 hidden state")
    if shape.kv_dtype is not NativeSpecCycleDType.FP32:
        raise ValueError("native proposal graph requires FP32 draft KV")
    state = control.pointers.state
    for name in (
        "hidden_seed_in",
        "candidate_token_ids",
        "draft_key_cache",
        "draft_value_cache",
    ):
        if int(getattr(state, name)) == 0:
            raise ValueError(f"native proposal graph requires state.{name}")
    if control.stream == 0:
        raise ValueError("native proposal graph requires a runner-owned stream")
    if control.deadline_ns != 0:
        raise ValueError("native proposal graph does not yet support a deadline")
    if control.pointers.outputs.cancel_flag != 0:
        raise ValueError("native proposal graph does not yet support cancellation")


def _validate_provider_chain_target(control: NativeSpecCycleControl) -> None:
    shape = control.shape
    supported_stages = {
        NativeSpecCycleStage.VERIFY,
        NativeSpecCycleStage.VERIFY | NativeSpecCycleStage.ACCEPT,
    }
    candidate_rows = int(shape.row_count) - 1
    if control.stages not in supported_stages:
        raise ValueError("provider target graph supports VERIFY or VERIFY|ACCEPT")
    if (
        control.mode.name != "CHAIN"
        or shape.request_count != 1
        or candidate_rows not in {1, 2, 3, 4, 5, 8}
        or shape.active_row_count != shape.row_count
        or shape.candidate_count != candidate_rows
        or shape.active_candidate_count != candidate_rows
        or shape.candidate_budget != candidate_rows
    ):
        raise ValueError(
            "provider target graph supports one active B1/B2/B3/B4/B5/B8 chain bucket"
        )
    if shape.metadata_dtype is not NativeSpecCycleDType.INT32:
        raise ValueError("provider target graph requires INT32 verifier metadata")
    if shape.hidden_dtype not in {
        NativeSpecCycleDType.FP16,
        NativeSpecCycleDType.BF16,
    }:
        raise ValueError("provider target graph requires FP16/BF16 hidden rows")
    if shape.kv_dtype is not NativeSpecCycleDType.BF16:
        raise ValueError("provider target graph requires BF16 target KV")
    if control.deadline_ns != 0:
        raise ValueError("provider target graph does not yet support a deadline")
    if control.pointers.outputs.cancel_flag != 0:
        raise ValueError("provider target graph does not yet support cancellation")


def _validate_small_chain_target(control: NativeSpecCycleControl) -> None:
    shape = control.shape
    n2_stages = (
        NativeSpecCycleStage.VERIFY
        | NativeSpecCycleStage.ACCEPT
        | NativeSpecCycleStage.COMMIT
        | NativeSpecCycleStage.UPDATE_CURSORS
    )
    if control.stages not in {NativeSpecCycleStage.VERIFY, n2_stages}:
        raise ValueError("native target graph supports VERIFY or N2 accept/commit stages")
    candidate_rows = int(shape.row_count) - 1
    if (
        shape.request_count != 1
        or shape.row_count not in {2, 3}
        or shape.active_row_count != shape.row_count
        or shape.candidate_count != candidate_rows
        or shape.active_candidate_count != candidate_rows
        or shape.candidate_budget != candidate_rows
    ):
        raise ValueError("native target graph supports one B1/B2 chain bucket (1 request, 2-3 rows)")
    expected_metadata_dtype = (
        NativeSpecCycleDType.INT64
        if control.stages == NativeSpecCycleStage.VERIFY
        else NativeSpecCycleDType.INT32
    )
    if shape.metadata_dtype is not expected_metadata_dtype:
        raise ValueError(
            f"native target graph {control.stages!s} requires {expected_metadata_dtype.name} metadata"
        )
    if shape.hidden_dtype is not NativeSpecCycleDType.FP32:
        raise ValueError("native target graph requires FP32 hidden rows")
    if shape.kv_dtype is not NativeSpecCycleDType.BF16:
        raise ValueError("native target graph requires BF16 KV")
    if control.stream == 0:
        raise ValueError("native target graph requires a session-owned stream")
    if control.deadline_ns != 0:
        raise ValueError("native target graph does not yet support a deadline")
    if control.pointers.outputs.cancel_flag != 0:
        raise ValueError("native target graph does not yet support cancellation")


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


def _uint64_identity(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value >= 1 << 64:
        raise ValueError(f"{name} must fit uint64")
    return value


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


def register_native_spec_gguf_graphs(*, replace: bool = True) -> None:
    """Register backend-neutral GGUF graph launchers for admitted gfx11 peers."""

    for backend in _GGUF_TARGET_GRAPH_BACKENDS:
        register(
            KernelKey(
                backend,
                "speculative_cycle",
                "w4_gguf",
                "native_v1_b2_target_graph",
            ),
            create_native_spec_target_graph_launcher,
            replace=replace,
        )
    register(
        KernelKey(
            "hip_gfx1100",
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_proposal_graph",
        ),
        create_native_spec_proposal_graph_launcher,
        replace=replace,
    )


register_native_spec_gguf_graphs()


def register_native_spec_provider_target_graph(*, replace: bool = True) -> None:
    """Register the gfx1100 N4 provider only when its adapter is requested.

    Keeping this key lazy prevents the generic gfx1151 shared-body alias refresh
    from admitting an unvalidated provider merely because this module was
    imported for ABI tests or GGUF N1/N3P registration.
    """

    register(
        KernelKey(
            "hip_gfx1100",
            "speculative_cycle",
            "w4_paro",
            "native_v1_target_graph",
        ),
        create_native_spec_provider_target_graph_launcher,
        replace=replace,
    )


__all__ = [
    "NativeSpecProposalGraphLauncher",
    "NativeSpecProviderTargetGraphLauncher",
    "NativeSpecTargetGraphLauncher",
    "build_native_spec_cycle_graph_launcher",
    "create_native_spec_proposal_graph_launcher",
    "create_native_spec_provider_target_graph_launcher",
    "create_native_spec_target_graph_launcher",
    "plan_native_spec_cycle_graph_launcher_build",
    "register_native_spec_gguf_graphs",
    "register_native_spec_provider_target_graph",
]
