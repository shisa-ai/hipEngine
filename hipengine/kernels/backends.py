"""Backend metadata, target-arch mapping, and public backend selection.

Backend selection stays outside the engine hot path: model/runtime code receives a
backend key (for example ``hip_gfx1151`` or ``cuda_sm120a``), while this module
records the native HIP/CUDA architecture needed by the JIT build layer and maps
``backend="auto"`` to a concrete backend at load/serve time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import warnings
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

AUTO_BACKEND = "auto"
CPU_BACKEND = "cpu_reference"
_ENV_BACKEND = "HIPENGINE_BACKEND"
_ENV_HIP_ARCH = "HIPENGINE_HIP_ARCH"
_ENV_GPU_MAX_HW_QUEUES = "GPU_MAX_HW_QUEUES"
_ENV_GPU_MAX_HW_QUEUES_POLICY = "HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"
_ENV_PROCESS_ENV_REPORT_PATH = "HIPENGINE_PROCESS_ENV_REPORT_PATH"
_ENV_HSA_SCRATCH_SINGLE_LIMIT = "HSA_SCRATCH_SINGLE_LIMIT"
_GPU_MAX_HW_QUEUE_POLICIES = frozenset({"auto", "explicit", "backend_default", "runtime_default"})

HIP_BACKEND_TARGET_ARCH: dict[str, str] = {
    "hip_gfx1100": "gfx1100",
    "hip_gfx1151": "gfx1151",
}
HIP_TARGET_ARCH_BACKEND: dict[str, str] = {
    arch: backend for backend, arch in HIP_BACKEND_TARGET_ARCH.items()
}
CUDA_BACKEND_TARGET_ARCH: dict[str, str] = {
    "cuda_sm120a": "sm_120a",
}
CUDA_TARGET_ARCH_BACKEND: dict[str, str] = {
    arch: backend for backend, arch in CUDA_BACKEND_TARGET_ARCH.items()
}
# Process-start HIP runtime defaults are backend metadata, not dispatch logic.
# On gfx1100, rocprof identifies repeated 300-MiB use-once allocations for the
# 3,200-byte/thread AOTriton prefill kernel; 8 MiB preserves measured
# full-engine behavior while materially reducing whole-device peak.
# ROCm's documented default is four hardware queues. Clean Laguna branch-
# concurrency evidence admits two on gfx1151. Explicit user values always win.
HIP_BACKEND_PROCESS_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    "hip_gfx1100": {_ENV_HSA_SCRATCH_SINGLE_LIMIT: "8388608"},
    "hip_gfx1151": {_ENV_GPU_MAX_HW_QUEUES: "2"},
}

_ARCH_PATTERN = re.compile(r"\bgfx[0-9a-fA-F]+(?:[-_:][^\s]*)?")
_ARCH_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("amdgpu-arch",),
    ("/opt/rocm/bin/amdgpu-arch",),
    ("rocm_agent_enumerator",),
    ("/opt/rocm/bin/rocm_agent_enumerator",),
)


@dataclass(frozen=True)
class BackendSelection:
    """Resolved backend choice plus the evidence/source used for diagnostics."""

    requested: str
    backend: str
    source: str
    detected_arches: tuple[str, ...] = ()
    warning: str | None = None

    @property
    def detected_arch(self) -> str | None:
        """Return the primary detected target arch, if any."""

        return self.detected_arches[0] if self.detected_arches else None


def hip_target_arch_for_backend(backend: str) -> str:
    """Return the HIP offload arch for a supported HIP backend key."""

    try:
        return HIP_BACKEND_TARGET_ARCH[backend]
    except KeyError as exc:
        valid = ", ".join(sorted(HIP_BACKEND_TARGET_ARCH))
        raise ValueError(f"unsupported HIP backend {backend!r}; expected one of: {valid}") from exc


def cuda_target_arch_for_backend(backend: str) -> str:
    """Return the CUDA target arch for a supported CUDA backend key."""

    try:
        return CUDA_BACKEND_TARGET_ARCH[backend]
    except KeyError as exc:
        valid = ", ".join(sorted(CUDA_BACKEND_TARGET_ARCH))
        raise ValueError(
            f"unsupported CUDA backend {backend!r}; expected one of: {valid}"
        ) from exc


def load_backend_kernel_package(backend: str) -> ModuleType:
    """Load and refresh one concrete hardware backend's kernel registrations.

    Backend packages may expose a conventional ``register_backend_kernels``
    hook. Calling it with ``replace=False`` fills registrations missing after
    test isolation without overwriting a caller-provided registry fixture.
    """

    if backend in HIP_BACKEND_TARGET_ARCH:
        hip_target_arch_for_backend(backend)
    elif backend in CUDA_BACKEND_TARGET_ARCH:
        cuda_target_arch_for_backend(backend)
    else:
        valid = ", ".join(sorted((*HIP_BACKEND_TARGET_ARCH, *CUDA_BACKEND_TARGET_ARCH)))
        raise ValueError(f"unsupported hardware backend {backend!r}; expected one of: {valid}")
    module = import_module(f"hipengine.kernels.{backend}")
    registrar = getattr(module, "register_backend_kernels", None)
    if callable(registrar):
        registrar(replace=False)
    return module


def backend_package_capability(backend: str, name: str, default=None):
    """Read backend-package metadata without rerunning kernel registration."""

    if backend not in HIP_BACKEND_TARGET_ARCH and backend not in CUDA_BACKEND_TARGET_ARCH:
        valid = ", ".join(sorted((*HIP_BACKEND_TARGET_ARCH, *CUDA_BACKEND_TARGET_ARCH)))
        raise ValueError(f"unsupported hardware backend {backend!r}; expected one of: {valid}")
    module = import_module(f"hipengine.kernels.{backend}")
    return getattr(module, str(name), default)


def configure_hip_process_environment(
    *,
    detected_arches: Sequence[str] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply hardware-local HIP defaults before ``libamdhip64`` is loaded.

    gfx1100 caps reclaimable single-dispatch scratch at 8 MiB instead of
    ROCr's much larger default threshold; gfx1151 uses two hardware queues for
    the admitted Laguna shared/routed MoE overlap while remaining below ROCm's
    documented default of four. The policy is applied only when all recognized
    visible HIP architectures map to the same backend. Existing environment
    values are never overwritten, so users retain explicit rollback/control.
    The benchmark-only ``runtime_default`` policy suppresses the gfx1151 package
    queue limit while leaving ``GPU_MAX_HW_QUEUES`` absent for ROCm initialization.
    """

    env_map = os.environ if env is None else env
    if detected_arches is None:
        arch_hint = (env_map.get(_ENV_HIP_ARCH) or "").strip()
        raw_arches = (arch_hint,) if arch_hint else detect_hip_target_arches()
    else:
        raw_arches = detected_arches

    arches = tuple(_normalize_arch(arch) for arch in raw_arches)
    backends = {
        HIP_TARGET_ARCH_BACKEND[arch]
        for arch in arches
        if arch in HIP_TARGET_ARCH_BACKEND
    }
    if not backends:
        backend_hint = (env_map.get(_ENV_BACKEND) or "").strip()
        if backend_hint in HIP_BACKEND_TARGET_ARCH:
            backends.add(backend_hint)

    queue_policy = (
        env_map.get(_ENV_GPU_MAX_HW_QUEUES_POLICY) or "auto"
    ).strip().lower()
    if queue_policy not in _GPU_MAX_HW_QUEUE_POLICIES:
        valid = ", ".join(sorted(_GPU_MAX_HW_QUEUE_POLICIES))
        raise ValueError(
            f"invalid {_ENV_GPU_MAX_HW_QUEUES_POLICY}={queue_policy!r}; expected {valid}"
        )
    queue_before = env_map.get(_ENV_GPU_MAX_HW_QUEUES)
    if queue_policy == "runtime_default" and queue_before is not None:
        raise ValueError(
            f"{_ENV_GPU_MAX_HW_QUEUES_POLICY}=runtime_default requires "
            f"{_ENV_GPU_MAX_HW_QUEUES} to be absent"
        )
    if queue_policy == "explicit" and queue_before is None:
        raise ValueError(
            f"{_ENV_GPU_MAX_HW_QUEUES_POLICY}=explicit requires "
            f"{_ENV_GPU_MAX_HW_QUEUES}"
        )
    if queue_before is not None:
        try:
            if int(queue_before) < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"{_ENV_GPU_MAX_HW_QUEUES} must be a positive integer"
            ) from exc

    applied: dict[str, str] = {}
    backend = next(iter(backends)) if len(backends) == 1 else None
    if backend is not None:
        for name, value in HIP_BACKEND_PROCESS_ENV_DEFAULTS.get(backend, {}).items():
            if name == _ENV_GPU_MAX_HW_QUEUES and queue_policy == "runtime_default":
                continue
            if name in env_map:
                continue
            env_map[name] = value
            applied[name] = value

    queue_after = env_map.get(_ENV_GPU_MAX_HW_QUEUES)
    if queue_before is not None:
        queue_source = "explicit_environment"
    elif _ENV_GPU_MAX_HW_QUEUES in applied:
        queue_source = "backend_default"
    elif queue_policy == "runtime_default":
        queue_source = "rocm_runtime_default"
    else:
        queue_source = "not_applicable"
    report_path_raw = (env_map.get(_ENV_PROCESS_ENV_REPORT_PATH) or "").strip()
    if report_path_raw:
        report_path = Path(report_path_raw).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "kind": "hipengine_process_environment_report",
            "detected_arches": list(arches),
            "resolved_backend": backend,
            "applied_defaults": dict(applied),
            "gpu_max_hw_queues": {
                "requested_policy": queue_policy,
                "value_before_backend_defaults": queue_before,
                "effective_value": queue_after,
                "source": queue_source,
                "runtime_queue_ids": None,
                "runtime_queue_count": None,
                "runtime_observation": "requires rocprof queue trace",
            },
        }
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_path)
    return applied


def select_backend(
    backend: str | None = AUTO_BACKEND,
    *,
    detected_arches: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    fallback_backend: str = CPU_BACKEND,
) -> BackendSelection:
    """Resolve ``backend`` to a concrete backend key.

    ``backend='auto'`` is a selector, not a registry key. It detects exact native
    HIP targets that have peer backends (currently ``gfx1100`` and ``gfx1151``),
    honors ``HIPENGINE_BACKEND`` as an explicit force override, and otherwise
    returns ``cpu_reference`` with a warning message explaining how to force a
    nearby backend such as ``hip_gfx1100`` for gfx1101/gfx1102-class users.

    Explicit backend strings are returned unchanged so tests and future plugin
    backends can register their own keys without editing this module.
    """

    requested = (backend or AUTO_BACKEND).strip() or AUTO_BACKEND
    if requested != AUTO_BACKEND:
        return BackendSelection(requested=requested, backend=requested, source="explicit")

    env_map = os.environ if env is None else env
    env_backend = (env_map.get(_ENV_BACKEND) or "").strip()
    if env_backend and env_backend != AUTO_BACKEND:
        return BackendSelection(requested=requested, backend=env_backend, source=_ENV_BACKEND)

    raw_arches = detected_arches if detected_arches is not None else detect_hip_target_arches()
    arches = tuple(_normalize_arch(arch) for arch in raw_arches)
    arches = tuple(dict.fromkeys(arch for arch in arches if arch))
    for arch in arches:
        resolved = HIP_TARGET_ARCH_BACKEND.get(arch)
        if resolved is not None:
            return BackendSelection(
                requested=requested,
                backend=resolved,
                source="hip_arch",
                detected_arches=arches,
            )

    return BackendSelection(
        requested=requested,
        backend=fallback_backend,
        source="fallback",
        detected_arches=arches,
        warning=_auto_backend_warning(arches, fallback_backend),
    )


def resolve_backend(
    backend: str | None = AUTO_BACKEND,
    *,
    warn: bool = True,
    detected_arches: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    fallback_backend: str = CPU_BACKEND,
) -> str:
    """Return a concrete backend key, optionally emitting auto-fallback warnings."""

    selection = select_backend(
        backend,
        detected_arches=detected_arches,
        env=env,
        fallback_backend=fallback_backend,
    )
    if warn and selection.warning:
        warnings.warn(selection.warning, RuntimeWarning, stacklevel=2)
    return selection.backend


def detect_hip_target_arches() -> tuple[str, ...]:
    """Detect visible HIP GPU target architectures without importing torch.

    The ROCm command-line probes are intentionally outside import-time paths and
    cheap compared with model loading. They also avoid depending on the exact
    ``hipDeviceProp_t`` ABI layout across ROCm releases.
    """

    for command in _ARCH_COMMANDS:
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        arches = _parse_arches(result.stdout)
        if arches:
            return arches
    return ()


@contextmanager
def hip_target_arch_environment(target_arch: str | None) -> Iterator[None]:
    """Temporarily set ``HIPENGINE_HIP_ARCH`` for build calls in this scope."""

    if target_arch is None:
        yield
        return
    old = os.environ.get(_ENV_HIP_ARCH)
    os.environ[_ENV_HIP_ARCH] = target_arch
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(_ENV_HIP_ARCH, None)
        else:
            os.environ[_ENV_HIP_ARCH] = old


def _parse_arches(text: str) -> tuple[str, ...]:
    arches = [_normalize_arch(match.group(0)) for match in _ARCH_PATTERN.finditer(text)]
    return tuple(dict.fromkeys(arch for arch in arches if arch))


def _normalize_arch(value: str) -> str:
    match = _ARCH_PATTERN.search(value.strip())
    if match is None:
        return ""
    return match.group(0).split(":", 1)[0].split("_", 1)[0].split("-", 1)[0].lower()


def _auto_backend_warning(arches: Sequence[str], fallback_backend: str) -> str:
    supported = ", ".join(
        f"{arch}->{backend}" for arch, backend in sorted(HIP_TARGET_ARCH_BACKEND.items())
    )
    force = (
        "To force a HIP backend, pass backend='hip_gfx1100' or backend='hip_gfx1151' in Python, "
        "use --backend hip_gfx1100/hip_gfx1151 on CLI/server entry points, or set "
        f"{_ENV_BACKEND}=hip_gfx1100."
    )
    if arches:
        arch_list = ", ".join(arches)
        return (
            f"hipEngine detected HIP target arch(es) {arch_list}, but no native backend is "
            f"registered for them; using {fallback_backend!r}. gfx1101/gfx1102-class users may "
            f"want to force a nearby gfx1100 backend after validating correctness/performance. "
            f"{force} Supported auto mappings: {supported}."
        )
    return (
        f"hipEngine could not detect a supported HIP GPU target; using {fallback_backend!r}. "
        f"{force} Supported auto mappings: {supported}."
    )
