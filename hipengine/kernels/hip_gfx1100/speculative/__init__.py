"""gfx1100 speculative decoding kernel wrappers."""

from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (
    build_dflash_accept,
    dflash_accept_chain_i32,
    plan_dflash_accept_build,
    register_dflash_accept_kernels,
)

__all__ = [
    "build_dflash_accept",
    "dflash_accept_chain_i32",
    "plan_dflash_accept_build",
    "register_dflash_accept_kernels",
]
