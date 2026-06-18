"""gfx1100 native sampler kernel wrappers."""

from hipengine.kernels.hip_gfx1100.sampling.sampler import (
    apply_processors_f32_rows,
    build_sampler,
    plan_sampler_build,
    register_sampler_kernels,
    sample_temperature_f32_rows_i32,
    sample_temperature_top_logprobs_f32_rows_i32,
    sample_top_p_temperature_f32_rows_i32,
    sample_topk_temperature_f32_rows_i32,
)

__all__ = [
    "apply_processors_f32_rows",
    "build_sampler",
    "plan_sampler_build",
    "register_sampler_kernels",
    "sample_temperature_f32_rows_i32",
    "sample_temperature_top_logprobs_f32_rows_i32",
    "sample_top_p_temperature_f32_rows_i32",
    "sample_topk_temperature_f32_rows_i32",
]
