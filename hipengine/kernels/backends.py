"""Backend metadata shared by kernel registration and build plumbing.

Backend selection stays outside the engine hot path: model/runtime code receives a
backend key (for example ``hip_gfx1151``), while this module only records the
native HIP offload architecture needed by the JIT build layer.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

HIP_BACKEND_TARGET_ARCH: dict[str, str] = {
    "hip_gfx1100": "gfx1100",
    "hip_gfx1151": "gfx1151",
}
_ENV_HIP_ARCH = "HIPENGINE_HIP_ARCH"


def hip_target_arch_for_backend(backend: str) -> str:
    """Return the HIP offload arch for a supported HIP backend key."""

    try:
        return HIP_BACKEND_TARGET_ARCH[backend]
    except KeyError as exc:
        valid = ", ".join(sorted(HIP_BACKEND_TARGET_ARCH))
        raise ValueError(f"unsupported HIP backend {backend!r}; expected one of: {valid}") from exc


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
