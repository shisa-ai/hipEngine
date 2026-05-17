"""Backend metadata, target-arch mapping, and public backend selection.

Backend selection stays outside the engine hot path: model/runtime code receives a
backend key (for example ``hip_gfx1151``), while this module records the native
HIP offload architecture needed by the JIT build layer and maps ``backend="auto"``
to a concrete backend at load/serve time.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

AUTO_BACKEND = "auto"
CPU_BACKEND = "cpu_reference"
_ENV_BACKEND = "HIPENGINE_BACKEND"
_ENV_HIP_ARCH = "HIPENGINE_HIP_ARCH"

HIP_BACKEND_TARGET_ARCH: dict[str, str] = {
    "hip_gfx1100": "gfx1100",
    "hip_gfx1151": "gfx1151",
}
HIP_TARGET_ARCH_BACKEND: dict[str, str] = {
    arch: backend for backend, arch in HIP_BACKEND_TARGET_ARCH.items()
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
