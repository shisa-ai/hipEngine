"""Backend metadata, target-arch mapping, and public backend selection.

Backend selection stays outside the engine hot path: model/runtime code receives a
backend key (for example ``hip_gfx1151`` or ``cuda_sm120a``), while this module
records the native HIP/CUDA architecture needed by the JIT build layer and maps
``backend="auto"`` to a concrete backend at load/serve time.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import warnings
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

AUTO_BACKEND = "auto"
CPU_BACKEND = "cpu_reference"
_ENV_BACKEND = "HIPENGINE_BACKEND"
_ENV_HIP_ARCH = "HIPENGINE_HIP_ARCH"
_ENV_GPU_MAX_HW_QUEUES = "GPU_MAX_HW_QUEUES"
_ENV_HSA_SCRATCH_SINGLE_LIMIT = "HSA_SCRATCH_SINGLE_LIMIT"
# Applied in this order: ROCR filters the physical device list, then HIP indexes
# the ROCR-visible result. They compose; one does not override the other.
_ENV_ROCR_VISIBLE_DEVICES = "ROCR_VISIBLE_DEVICES"
_ENV_HIP_VISIBLE_DEVICES = "HIP_VISIBLE_DEVICES"
_VISIBILITY_ENV_VARS = (_ENV_ROCR_VISIBLE_DEVICES, _ENV_HIP_VISIBLE_DEVICES)
_ENV_ROCM_TARGET_LST = "ROCM_TARGET_LST"

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
# ROCm's documented default is four hardware queues. Complete Qwen3.8
# fixed-width, arrival/soak, context/graph, prefix, and pressure matrices retain
# two on gfx1151. Explicit user values always win.
HIP_BACKEND_PROCESS_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    "hip_gfx1100": {_ENV_HSA_SCRATCH_SINGLE_LIMIT: "8388608"},
    "hip_gfx1151": {_ENV_GPU_MAX_HW_QUEUES: "2"},
}

_ARCH_PATTERN = re.compile(r"\bgfx[0-9a-fA-F]+(?:[-_:][^\s]*)?")
# Hardware-only architecture probes. ``rocm_agent_enumerator`` is deliberately
# absent: its installed implementation reads ``ROCM_TARGET_LST`` or an adjacent
# ``target.lst`` *before* querying hardware, so its output can be an offline
# target list rather than the installed devices. It stays in ``_ARCH_COMMANDS``
# for ambient reporting (auto-selection diagnostics, provenance, scripts), which
# is not a hardware-evidence claim; execution decisions use the hardware list.
_HARDWARE_ARCH_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("amdgpu-arch",),
    ("/opt/rocm/llvm/bin/amdgpu-arch",),
    ("/opt/rocm/bin/amdgpu-arch",),
)
_ARCH_COMMANDS: tuple[tuple[str, ...], ...] = (
    *_HARDWARE_ARCH_COMMANDS,
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


class BackendDeviceCapabilityError(RuntimeError):
    """An explicit HIP backend targets an architecture the selected device lacks.

    Raised at execution selection, before any model load or kernel launch. The
    requested backend's kernels are compiled for its declared target arch; the
    selected visible HIP device reports a different architecture, so those code
    objects cannot execute there. The class is a ``RuntimeError`` so existing
    callers keep working while tests and diagnostics can name the condition.
    """


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
    ROCr's much larger default threshold; gfx1151 uses two hardware queues after
    complete same-host queue-policy validation while remaining below ROCm's
    documented default of four. The policy is applied only when all recognized
    visible HIP architectures map to the same backend. Existing environment
    values are never overwritten, so users retain explicit rollback/control.
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

    queue_before = env_map.get(_ENV_GPU_MAX_HW_QUEUES)
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
            if name in env_map:
                continue
            env_map[name] = value
            applied[name] = value

    return applied


def select_backend(
    backend: str | None = AUTO_BACKEND,
    *,
    detected_arches: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    fallback_backend: str = CPU_BACKEND,
    execution_device: bool = False,
    physical_arches: Sequence[str] | None = None,
    current_device: int | None = None,
) -> BackendSelection:
    """Resolve ``backend`` to a concrete backend key.

    ``backend='auto'`` is a selector, not a registry key. It detects exact native
    HIP targets that have peer backends (currently ``gfx1100`` and ``gfx1151``),
    honors ``HIPENGINE_BACKEND`` as an explicit force override, and otherwise
    returns ``cpu_reference`` with a warning message explaining how to force a
    nearby backend such as ``hip_gfx1100`` for gfx1101/gfx1102-class users.

    Explicit backend strings are returned unchanged so tests and future plugin
    backends can register their own keys without editing this module.

    ``execution_device=True`` resolves automatic selection for the device this
    process will execute on, sharing one device identity with
    :func:`require_backend_device_compatibility`: the current logical device,
    derived from the physical per-device list (``physical_arches`` or
    :func:`detect_hip_physical_device_arches`) and the visibility variables.
    Without it, automatic selection takes the first ambient architecture, which
    is correct for registry resolution but can name a device other than the one
    the engine runs on. When the current device is *known* but has no native
    backend, selection takes the unsupported-architecture fallback for that
    device rather than another device's backend, which validation would refuse.
    Only an *unresolved* device identity falls back to the hardware-only ambient
    probe, so an offline target list is never accepted as installed hardware.
    """

    requested = (backend or AUTO_BACKEND).strip() or AUTO_BACKEND
    if requested != AUTO_BACKEND:
        return BackendSelection(requested=requested, backend=requested, source="explicit")

    env_map = os.environ if env is None else env
    env_backend = (env_map.get(_ENV_BACKEND) or "").strip()
    if env_backend and env_backend != AUTO_BACKEND:
        return BackendSelection(requested=requested, backend=env_backend, source=_ENV_BACKEND)

    if execution_device:
        device_arches = selected_device_arches(
            detected_arches=physical_arches,
            env=env_map,
            current_device=current_device,
        )
        if device_arches:
            for arch in device_arches:
                resolved = HIP_TARGET_ARCH_BACKEND.get(arch)
                if resolved is not None:
                    return BackendSelection(
                        requested=requested,
                        backend=resolved,
                        source="execution_device",
                        detected_arches=device_arches,
                    )
            # The execution device is known but has no native backend. Use the
            # unsupported-architecture fallback for that device rather than
            # another device's backend, which validation would then refuse.
            return BackendSelection(
                requested=requested,
                backend=fallback_backend,
                source="execution_device_fallback",
                detected_arches=device_arches,
                warning=_auto_backend_warning(device_arches, fallback_backend),
            )
        # Device identity is unresolved, so fall back to the ambient probe
        # below; compatibility validation stays permissive in this case.

    if detected_arches is not None:
        raw_arches = detected_arches
    elif execution_device:
        raw_arches = detect_hip_hardware_target_arches()
    else:
        raw_arches = detect_hip_target_arches()
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
    execution_device: bool = False,
    physical_arches: Sequence[str] | None = None,
    current_device: int | None = None,
) -> str:
    """Return a concrete backend key, optionally emitting auto-fallback warnings.

    ``execution_device=True`` makes automatic selection agree with
    :func:`require_backend_device_compatibility` about which device the engine
    will execute on. See :func:`select_backend`.
    """

    selection = select_backend(
        backend,
        detected_arches=detected_arches,
        env=env,
        fallback_backend=fallback_backend,
        execution_device=execution_device,
        physical_arches=physical_arches,
        current_device=current_device,
    )
    if warn and selection.warning:
        warnings.warn(selection.warning, RuntimeWarning, stacklevel=2)
    return selection.backend


def _probe_arches(
    commands: Sequence[Sequence[str]], *, env: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Run the first working architecture probe and return its arches in order.

    ``env=None`` inherits the process environment. The ROCm command-line probes
    are intentionally outside import-time paths and cheap compared with model
    loading. They also avoid depending on the exact ``hipDeviceProp_t`` ABI
    layout across ROCm releases.
    """

    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                env=None if env is None else dict(env),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        arches = _parse_arches_ordered(result.stdout)
        if arches:
            return arches
    return ()


def _hardware_probe_env() -> dict[str, str]:
    """Environment for a hardware-only probe.

    Removes the visibility variables so the result is the physical device list,
    and points ``ROCM_TARGET_LST`` at a path that cannot exist so a producer that
    consults an offline target list (including an adjacent ``target.lst``, which
    is only read when the variable is unset) cannot substitute a target list for
    installed hardware.
    """

    env = dict(os.environ)
    for name in _VISIBILITY_ENV_VARS:
        env.pop(name, None)
    env[_ENV_ROCM_TARGET_LST] = os.path.join(os.sep, "nonexistent", "hipengine-target.lst")
    return env


def detect_hip_target_arches() -> tuple[str, ...]:
    """Detect visible HIP GPU target architectures without importing torch.

    Ambient and deduplicated, for auto-selection diagnostics, provenance, and
    scripts. Execution decisions use :func:`detect_hip_hardware_target_arches`
    and :func:`detect_hip_physical_device_arches`, which do not accept a
    producer that can read an offline target list as hardware evidence.
    """

    return tuple(dict.fromkeys(_probe_arches(_ARCH_COMMANDS)))


def detect_hip_hardware_target_arches() -> tuple[str, ...]:
    """Deduplicated HIP target architectures from hardware-only probes.

    Same as :func:`detect_hip_target_arches` but restricted to producers that
    enumerate installed devices, so an offline ``ROCM_TARGET_LST``/``target.lst``
    cannot masquerade as hardware.
    """

    return tuple(dict.fromkeys(_probe_arches(_HARDWARE_ARCH_COMMANDS)))


def detect_hip_physical_device_arches() -> tuple[str, ...]:
    """Per-device HIP target architectures in physical device order.

    One entry per physical GPU, in device order, with duplicates preserved:
    ``("gfx1100", "gfx1100", "gfx1151")`` is three devices, not two. The probe
    runs with the visibility variables removed and the offline target list
    neutralized, so the result is the installed hardware whether or not a given
    ROCm release makes the arch tool visibility-aware; the caller applies the
    visibility remap exactly once over this list. Verified producer semantics on
    ROCm 7.2: ``amdgpu-arch`` prints one architecture per line in device order,
    keeps duplicates, and ignores ``ROCR_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES``.

    This differs from :func:`detect_hip_target_arches`, which deduplicates and
    reports the ambient set for auto-selection diagnostics.
    """

    return _probe_arches(_HARDWARE_ARCH_COMMANDS, env=_hardware_probe_env())


def _parse_visible_device_indices(raw: str | None) -> tuple[int, ...] | None:
    """Device indices from one visibility variable.

    ``None`` when the variable is unset. An empty tuple when it is set but its
    value is not a list of non-negative indices (empty, UUID or MIG-style
    handle), so the selected device is unknown and incompatibility is not
    proven.
    """

    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        return ()
    indices: list[int] = []
    for part in parts:
        if not part.isdigit():
            return ()
        indices.append(int(part))
    return tuple(indices)


def _visible_logical_device_arches(
    physical: Sequence[str], env: Mapping[str, str]
) -> tuple[str, ...] | None:
    """Visible device architectures in logical device order.

    ``ROCR_VISIBLE_DEVICES`` filters the physical device list; ``HIP_VISIBLE_DEVICES``
    then indexes that result. The two compose rather than one taking precedence:
    on ``(gfx1100, gfx1151)`` with ``ROCR_VISIBLE_DEVICES=1`` and
    ``HIP_VISIBLE_DEVICES=0``, logical device 0 is the gfx1151. ``None`` when a
    variable is set but cannot be resolved to device indices, or names an index
    outside the list it filters.
    """

    devices = list(physical)
    for name in _VISIBILITY_ENV_VARS:
        indices = _parse_visible_device_indices(env.get(name))
        if indices is None:
            continue
        if not indices:
            return None
        selected: list[str] = []
        for index in indices:
            if index >= len(devices):
                return None
            selected.append(devices[index])
        devices = selected
    return tuple(devices)


def _current_hip_device_ordinal() -> int | None:
    """Current logical HIP device ordinal from ``hipGetDevice``, or ``None``.

    Only scalar APIs are used, so there is no ``hipDeviceProp_t`` ABI hazard
    across ROCm releases. ``None`` means the HIP runtime is unavailable and the
    caller falls back to the default logical device. This is consulted only when
    the visible devices disagree on architecture, so a homogeneous host never
    initializes the HIP runtime from this path.

    HIP reads its queue/scratch policy during initialization, so the process
    defaults are applied before the library load or ``hipGetDevice`` call, the
    same ordering the runtime entry point in ``core/hip.py`` uses. Existing
    environment values are never overwritten.
    """

    configure_hip_process_environment()
    try:
        library = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return None
    try:
        library.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipGetDevice.restype = ctypes.c_int
        device = ctypes.c_int()
        if int(library.hipGetDevice(ctypes.byref(device))) != 0:
            return None
        return int(device.value)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def selected_device_arches(
    *,
    detected_arches: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    current_device: int | None = None,
) -> tuple[str, ...]:
    """Architecture(s) of the device the engine will execute on.

    ``detected_arches`` is the physical per-device architecture list in device
    order, as returned by :func:`detect_hip_physical_device_arches`; duplicates
    are meaningful and preserved. The visibility variables are applied exactly
    once over that list, and then the *current logical device* is selected. A
    matching architecture on some other device does not make the request
    runnable: with no visibility filter, device 1 being a gfx1151 does not
    excuse gfx1151 kernels launched on the default device 0.

    ``current_device`` overrides the logical ordinal. When it is omitted and the
    visible devices disagree on architecture, the ordinal is read with
    ``hipGetDevice``; a homogeneous visible set never loads the HIP runtime. If
    that query is unavailable or fails on a heterogeneous host, the current
    device is genuinely unknown: this returns an empty tuple rather than
    assuming ordinal zero, because an unresolved device is not evidence of
    incompatibility. An empty tuple means callers must not refuse the request.
    """

    env_map = os.environ if env is None else env
    raw = (
        tuple(detected_arches)
        if detected_arches is not None
        else detect_hip_physical_device_arches()
    )
    physical = tuple(arch for arch in (_normalize_arch(entry) for entry in raw) if arch)
    if not physical:
        return ()
    visible = _visible_logical_device_arches(physical, env_map)
    if not visible:
        return ()
    if current_device is not None:
        index = int(current_device)
    elif len(set(visible)) == 1:
        # Every visible device has the same architecture, so the ordinal cannot
        # change the answer and the HIP runtime is not needed.
        index = 0
    else:
        index = _current_hip_device_ordinal()
        if index is None:
            return ()
    if index < 0 or index >= len(visible):
        return ()
    return (visible[index],)


def require_backend_device_compatibility(
    backend: str,
    *,
    detected_arches: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    current_device: int | None = None,
) -> None:
    """Refuse an explicit HIP backend that the current device cannot execute.

    A non-HIP backend, an unrecognized backend key, or an unresolved device set
    is left alone: only a HIP backend whose declared target architecture is
    absent from the current logical device raises. ``HIPENGINE_HIP_ARCH`` is a
    build-time cross-compilation override and is deliberately not consulted as
    hardware evidence.
    """

    target_arch = HIP_BACKEND_TARGET_ARCH.get(backend)
    if target_arch is None:
        return
    arches = selected_device_arches(
        detected_arches=detected_arches,
        env=env,
        current_device=current_device,
    )
    if not arches or target_arch in arches:
        return
    raise BackendDeviceCapabilityError(
        f"backend {backend!r} targets HIP architecture {target_arch!r}, which is "
        f"not present on the selected HIP device(s) {arches!r}. Kernels built "
        f"for {target_arch!r} cannot execute on those devices, so this request "
        f"is refused before any model load or kernel launch. Select a backend "
        f"that matches the device or run on a {target_arch!r} device; "
        f"HIPENGINE_HIP_ARCH only cross-compiles offline and is not a device "
        f"override."
    )


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


def _parse_arches_ordered(text: str) -> tuple[str, ...]:
    """Normalized architectures in probe order, duplicates preserved."""

    return tuple(
        arch
        for arch in (_normalize_arch(match.group(0)) for match in _ARCH_PATTERN.finditer(text))
        if arch
    )


def _parse_arches(text: str) -> tuple[str, ...]:
    arches = _parse_arches_ordered(text)
    return tuple(dict.fromkeys(arches))


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
