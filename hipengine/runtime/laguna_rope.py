"""Exact host-table and registry integration for Laguna dual RoPE contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.cpu_reference import LagunaRopeConfig, laguna_rope_tables
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.materialize import DeviceTensorAllocation, load_host_array_to_device_as_dtype


@dataclass(frozen=True)
class LagunaDeviceRoPETables:
    config: LagunaRopeConfig
    max_positions: int
    cos: DeviceTensorAllocation
    sin: DeviceTensorAllocation

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.sin.free(runtime=runtime)
        self.cos.free(runtime=runtime)


def _reference_config(config) -> LagunaRopeConfig:
    if isinstance(config, LagunaRopeConfig):
        return config
    return LagunaRopeConfig(
        rope_type=str(config.rope_type),
        rotary_dim=int(config.dimension_count),
        freq_base=float(config.freq_base),
        scaling_factor=float(config.scaling_factor),
        original_context_length=int(config.original_context_length),
        yarn_attn_factor=float(config.yarn_attn_factor),
        yarn_beta_fast=float(config.yarn_beta_fast),
        yarn_beta_slow=float(config.yarn_beta_slow),
    )


def build_laguna_rope_table_data(max_positions: int, config) -> tuple[np.ndarray, np.ndarray]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    reference = _reference_config(config)
    return laguna_rope_tables(np.arange(max_positions, dtype=np.int64), reference)


def materialize_laguna_rope_tables(
    max_positions: int,
    config,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> LagunaDeviceRoPETables:
    reference = _reference_config(config)
    cos, sin = build_laguna_rope_table_data(max_positions, reference)
    cos_device = load_host_array_to_device_as_dtype(
        "laguna.rope.cos",
        cos,
        DType.FP32,
        device=device,
        runtime=runtime,
    )
    try:
        sin_device = load_host_array_to_device_as_dtype(
            "laguna.rope.sin",
            sin,
            DType.FP32,
            device=device,
            runtime=runtime,
        )
    except Exception:
        cos_device.free(runtime=runtime)
        raise
    return LagunaDeviceRoPETables(reference, max_positions, cos_device, sin_device)


def register_laguna_rope_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "head_rmsnorm+partial_rotary",
            "laguna_f32_weight",
            "positions_f32",
        ),
        gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
        replace=replace,
    )


def launch_laguna_head_rmsnorm_rope(
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    positions_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tables: LagunaDeviceRoPETables,
    *,
    backend: str = "hip_gfx1100",
    threads: int = 256,
    stream: int = 0,
    library=None,
    runtime=None,
) -> None:
    key = KernelKey(
        backend,
        "head_rmsnorm+partial_rotary",
        "laguna_f32_weight",
        "positions_f32",
    )
    fn = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    )
    if fn is None:
        register_laguna_rope_kernels()
        load_backend_kernel_package(backend)
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
    fn(
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        tables.cos.tensor.ptr,
        tables.sin.tensor.ptr,
        positions_ptr,
        query_out_ptr,
        key_out_ptr,
        eps,
        tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        tables.config.rotary_dim,
        tables.max_positions,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


register_laguna_rope_kernels()

__all__ = [
    "LagunaDeviceRoPETables",
    "build_laguna_rope_table_data",
    "launch_laguna_head_rmsnorm_rope",
    "materialize_laguna_rope_tables",
    "register_laguna_rope_kernels",
]
