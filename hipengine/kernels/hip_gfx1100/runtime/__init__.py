"""Small device-resident runtime state helpers."""

from hipengine.kernels.hip_gfx1100.runtime.state import (
    advance_decode_position_i64,
    build_runtime_state,
    embedding_lookup_bf16_i64,
    plan_runtime_state_build,
    register_runtime_state_kernels,
    set_decode_position_i64,
    set_i64_scalar,
)

__all__ = [
    "advance_decode_position_i64",
    "build_runtime_state",
    "embedding_lookup_bf16_i64",
    "plan_runtime_state_build",
    "register_runtime_state_kernels",
    "set_decode_position_i64",
    "set_i64_scalar",
]
