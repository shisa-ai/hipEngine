"""gfx1100 speculative decoding kernel wrappers."""

from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (
    build_dflash_accept,
    dflash_accept_chain_i32,
    plan_dflash_accept_build,
    register_dflash_accept_kernels,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash_commit import (
    build_dflash_commit,
    dflash_commit_chain_i32,
    plan_dflash_commit_build,
    register_dflash_commit_kernels,
)

__all__ = [
    "build_dflash_accept",
    "build_dflash_commit",
    "dflash_accept_chain_i32",
    "dflash_commit_chain_i32",
    "plan_dflash_accept_build",
    "plan_dflash_commit_build",
    "register_dflash_accept_kernels",
    "register_dflash_commit_kernels",
]
