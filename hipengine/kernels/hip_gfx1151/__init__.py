"""gfx1151 / Strix Halo backend registration.

The initial gfx1151 backend intentionally reuses the proven gfx11 HIP kernel
bodies from ``hip_gfx1100`` and compiles them as native ``gfx1151`` code objects
through ``HIPENGINE_HIP_ARCH=gfx1151`` / ``--offload-arch=gfx1151``.  This gives
Strix Halo a peer backend key while keeping tuning changes separate from the
source-lineage port.
"""

from __future__ import annotations

from importlib import import_module

from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register, registered_keys, resolve

BACKEND = "hip_gfx1151"
TARGET_ARCH = hip_target_arch_for_backend(BACKEND)
_SOURCE_BACKEND = "hip_gfx1100"
_GFX1100_MODULES = (
    "hipengine.kernels.hip_gfx1100.attention",
    "hipengine.kernels.hip_gfx1100.convert",
    "hipengine.kernels.hip_gfx1100.fused",
    "hipengine.kernels.hip_gfx1100.linear",
    "hipengine.kernels.hip_gfx1100.linear_attn",
    "hipengine.kernels.hip_gfx1100.moe",
    "hipengine.kernels.hip_gfx1100.norm",
    "hipengine.kernels.hip_gfx1100.quant",
    "hipengine.kernels.hip_gfx1100.rotary",
    "hipengine.kernels.hip_gfx1100.runtime",
    "hipengine.kernels.hip_gfx1100.sampling",
    "hipengine.kernels.hip_gfx1100.smoke",
    "hipengine.kernels.hip_gfx1100.speculative",
    "hipengine.kernels.hip_gfx1100.wmma",
)


def register_gfx1151_kernels(*, replace: bool = True) -> None:
    """Register gfx1151 aliases for the current gfx1100 kernel key space."""

    for module_name in _GFX1100_MODULES:
        import_module(module_name)
    source_keys = [key for key in registered_keys() if key.backend == _SOURCE_BACKEND]
    for key in source_keys:
        register(
            KernelKey(BACKEND, key.layer, key.quant, key.variant),
            resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant),
            replace=replace,
        )


register_gfx1151_kernels()

__all__ = ["BACKEND", "TARGET_ARCH", "register_gfx1151_kernels"]
