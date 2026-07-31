"""Raw-pointer grouped scalar and compact-WMMA prefill for GGUF IQ experts.

Both routes consume hipEngine's existing compact-MoE scheduler ABI.  The
expert-major scalar kernels use ``expert_start_compact`` only and preserve the
selected-single BF16 projection boundary exactly.  The compact WMMA kernels
also consume ``expert_start_wmma``/``tile_expert`` and reuse each raw IQ weight
tile across up to 16 sorted routed rows.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from types import MappingProxyType

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq_selected_prefill.hip")
_PARENT_SOURCE = Path(__file__).with_name("gguf_iq_gemv.hip")
_OUTPUT_NAME = "gguf_iq_selected_prefill.so"
_QK_K = 256
_MAX_GROUPED_IN_FEATURES = 3072

_SYMBOL_IQ2_GROUPED_DUAL = (
    "hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ2_GROUPED_DUAL_ROWBATCH4 = (
    "hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out"
)
_SYMBOL_IQ2_GROUPED_DUAL_SILU_PAIR16_ROWBATCH8 = (
    "hipengine_gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_"
    "rowbatch8_bf16_bf16_out"
)
_SYMBOL_IQ2_GROUPED_DUAL_ADAPTIVE = (
    "hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out"
)
_SYMBOL_IQ2_WMMA_DUAL = (
    "hipengine_gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ3_GROUPED_DUAL = (
    "hipengine_gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ3_GROUPED_DUAL_ROWBATCH4 = (
    "hipengine_gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out"
)
_SYMBOL_IQ3_GROUPED_DUAL_SILU_K3072_N1024_E256_ROWBATCH4 = (
    "hipengine_gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_SYMBOL_IQ3_GROUPED_SINGLE = (
    "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ3_GROUPED_SINGLE_ROWBATCH8 = (
    "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out"
)
_SYMBOL_IQ3_GROUPED_SINGLE_K1024_RESIDENT_ROWBATCH8 = (
    "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_"
    "rowbatch8_bf16_bf16_out"
)
_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITION_VALUES = (64,)
_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITION_VALUES = (256,)
_SYMBOL_IQ3_ACTIVATION_RESIDENT_ROW_INTERLEAVED_VOPD = (
    "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_"
    "active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_"
    "rowbatch8_bf16_bf16_out"
)
_SYMBOL_IQ3_ACTIVATION_RESIDENT_ROW_INTERLEAVED_VOPD_PAIRED_OUTPUT = (
    "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_"
    "active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_"
    "paired_output_rowbatch8_bf16_bf16_out"
)


def _iq3_active_expert_persistent_symbol(partition: int) -> str:
    return (
        "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_"
        f"active_expert_p{partition}_resident_rowbatch8_bf16_bf16_out"
    )


def _iq3_active_expert_persistent_variant(partition: int) -> str:
    return (
        "selected_grouped_prefill_compact_k1024_active_expert_p"
        f"{partition}_resident_rowbatch8_bf16_bf16_out"
    )


def _iq3_activation_resident_output_symbol(output_partition: int) -> str:
    return (
        "hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_"
        "active_expert_p64_activation_resident_out_p"
        f"{output_partition}_rowbatch8_bf16_bf16_out"
    )


def _iq3_activation_resident_output_variant(output_partition: int) -> str:
    return (
        "selected_grouped_prefill_compact_k1024_active_expert_p64_"
        "activation_resident_out_p"
        f"{output_partition}_rowbatch8_bf16_bf16_out"
    )


_SYMBOL_IQ4_GROUPED_DUAL = (
    "hipengine_gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ4_GROUPED_SINGLE = (
    "hipengine_gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ4_GROUPED_SINGLE_K512_WAVE32 = (
    "hipengine_gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out"
)
_SYMBOL_IQ4_GROUPED_SINGLE_K1024_WAVE32 = (
    "hipengine_gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_"
    "bf16_bf16_out"
)
_SYMBOL_IQ3_WMMA_DUAL = (
    "hipengine_gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ4_WMMA_DUAL = (
    "hipengine_gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out"
)
_SYMBOL_IQ4_WMMA_SINGLE = (
    "hipengine_gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out"
)


def plan_gguf_iq_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq_selected_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_iq_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _extra_flags() -> tuple[str, ...]:
    # The HIP translation unit textually includes the direct IQ source for the
    # pinned lookup tables/helpers. Carry its digest in the build flags so a
    # parent edit invalidates this JIT artifact as well.
    parent_tag = int(hashlib.sha256(_PARENT_SOURCE.read_bytes()).hexdigest()[:8], 16)
    return ("-mcumode", f"-DHIPENGINE_IQ_GEMV_SOURCE_TAG={parent_tag}")


def gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ2_GROUPED_DUAL,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ2_GROUPED_DUAL_ROWBATCH4,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ2_GROUPED_DUAL_SILU_PAIR16_ROWBATCH8,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ2_GROUPED_DUAL_ADAPTIVE,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Select per-expert adaptive batching at Laguna width."""

    if in_features > 2048 and compact_rows < 4 * num_experts:
        fn = gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out
    elif compact_rows >= 4 * num_experts:
        fn = gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out
    else:
        fn = gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out
    fn(
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ3_GROUPED_DUAL,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ3_GROUPED_DUAL_ROWBATCH4,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_k3072_n1024_e256_rowbatch4_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if in_features != 3072:
        raise ValueError("in_features must be exactly 3072 for H6C grouped IQ3")
    if out_features != 1024:
        raise ValueError("out_features must be exactly 1024 for H6C grouped IQ3")
    if num_experts != 256:
        raise ValueError("num_experts must be exactly 256 for H6C grouped IQ3")
    _launch_grouped_dual(
        _SYMBOL_IQ3_GROUPED_DUAL_SILU_K3072_N1024_E256_ROWBATCH4,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_single(
        _SYMBOL_IQ3_GROUPED_SINGLE,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_single(
        _SYMBOL_IQ3_GROUPED_SINGLE_ROWBATCH8,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if in_features != 1024:
        raise ValueError(
            "in_features must be exactly 1024 for grouped IQ3 resident rowbatch8"
        )
    _launch_grouped_single(
        _SYMBOL_IQ3_GROUPED_SINGLE_K1024_RESIDENT_ROWBATCH8,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _make_iq3_active_expert_persistent_wrapper(partition: int):
    symbol = _iq3_active_expert_persistent_symbol(partition)

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        active_experts_ptr: int,
        active_count_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        *,
        compact_rows: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _validate_common(
            compact_rows=compact_rows,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
        )
        if in_features != 1024:
            raise ValueError(
                "in_features must be exactly 1024 for active-expert persistent IQ3"
            )
        if out_features != 3072:
            raise ValueError(
                "out_features must be exactly 3072 for active-expert persistent IQ3"
            )
        if num_experts != 256:
            raise ValueError(
                "num_experts must be exactly 256 for active-expert persistent IQ3"
            )
        _launch_active_expert_persistent_single(
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            active_experts_ptr,
            active_count_ptr,
            qweight_ptr,
            out_ptr,
            compact_rows=compact_rows,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = (
        "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p"
        f"{partition}_resident_rowbatch8_bf16_bf16_out"
    )
    wrapper.__qualname__ = wrapper.__name__
    return wrapper


GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS = MappingProxyType(
    {
        partition: _make_iq3_active_expert_persistent_wrapper(partition)
        for partition in _IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITION_VALUES
    }
)


def _make_iq3_activation_resident_output_wrapper(output_partition: int):
    symbol = _iq3_activation_resident_output_symbol(output_partition)

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        active_experts_ptr: int,
        active_count_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        *,
        compact_rows: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _validate_common(
            compact_rows=compact_rows,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
        )
        if in_features != 1024:
            raise ValueError(
                "in_features must be exactly 1024 for activation-resident IQ3"
            )
        if out_features != 3072:
            raise ValueError(
                "out_features must be exactly 3072 for activation-resident IQ3"
            )
        if num_experts != 256:
            raise ValueError(
                "num_experts must be exactly 256 for activation-resident IQ3"
            )
        _launch_active_expert_persistent_single(
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            active_experts_ptr,
            active_count_ptr,
            qweight_ptr,
            out_ptr,
            compact_rows=compact_rows,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = (
        "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_"
        "active_expert_p64_activation_resident_out_p"
        f"{output_partition}_rowbatch8_bf16_bf16_out"
    )
    wrapper.__qualname__ = wrapper.__name__
    return wrapper


GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS = MappingProxyType(
    {
        output_partition: _make_iq3_activation_resident_output_wrapper(
            output_partition
        )
        for output_partition in _IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITION_VALUES
    }
)


def gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if in_features != 1024:
        raise ValueError(
            "in_features must be exactly 1024 for row-interleaved VOPD IQ3"
        )
    if out_features != 3072:
        raise ValueError(
            "out_features must be exactly 3072 for row-interleaved VOPD IQ3"
        )
    if num_experts != 256:
        raise ValueError(
            "num_experts must be exactly 256 for row-interleaved VOPD IQ3"
        )
    _launch_active_expert_persistent_single(
        _SYMBOL_IQ3_ACTIVATION_RESIDENT_ROW_INTERLEAVED_VOPD,
        x_ptr,
        expert_start_compact_ptr,
        active_experts_ptr,
        active_count_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_paired_output_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if in_features != 1024:
        raise ValueError(
            "in_features must be exactly 1024 for paired-output IQ3"
        )
    if out_features != 3072:
        raise ValueError(
            "out_features must be exactly 3072 for paired-output IQ3"
        )
    if num_experts != 256:
        raise ValueError(
            "num_experts must be exactly 256 for paired-output IQ3"
        )
    _launch_active_expert_persistent_single(
        _SYMBOL_IQ3_ACTIVATION_RESIDENT_ROW_INTERLEAVED_VOPD_PAIRED_OUTPUT,
        x_ptr,
        expert_start_compact_ptr,
        active_experts_ptr,
        active_count_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Use row-batch 4 once there are at least four rows per expert on average."""

    fn = (
        gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out
        if compact_rows >= 4 * num_experts
        else gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out
    )
    fn(
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_grouped_dual(
        _SYMBOL_IQ4_GROUPED_DUAL,
        x_ptr,
        expert_start_compact_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_IQ4_GROUPED_SINGLE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if in_features != 512:
        raise ValueError("in_features must be exactly 512 for grouped IQ4 wave32")
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_IQ4_GROUPED_SINGLE_K512_WAVE32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if in_features != 1024:
        raise ValueError("in_features must be exactly 1024 for grouped IQ4 wave32")
    _launch_grouped_single(
        _SYMBOL_IQ4_GROUPED_SINGLE_K1024_WAVE32,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Select wave32 at K=512 and retain local128 for general shapes."""

    fn = (
        gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out
        if in_features == 512
        else gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out
    )
    fn(
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_wmma_dual(
        _SYMBOL_IQ2_WMMA_DUAL,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        wmma_total_rows=wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_wmma_dual(
        _SYMBOL_IQ3_WMMA_DUAL,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        wmma_total_rows=wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_wmma_dual(
        _SYMBOL_IQ4_WMMA_DUAL,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        wmma_total_rows=wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_wmma(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        wmma_total_rows=wmma_total_rows,
    )
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_IQ4_WMMA_SINGLE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_grouped_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(gate_weight_ptr),
        ctypes.c_void_p(up_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_grouped_single(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_active_expert_persistent_single(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(active_experts_ptr),
        ctypes.c_void_p(active_count_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_wmma_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate_wmma(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        wmma_total_rows=wmma_total_rows,
    )
    library = library or build_gguf_iq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(gate_weight_ptr),
        ctypes.c_void_p(up_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _validate_common(
    *, compact_rows: int, in_features: int, out_features: int, num_experts: int
) -> None:
    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if in_features > _MAX_GROUPED_IN_FEATURES:
        raise ValueError("in_features must be at most 3072 for grouped IQ prefill")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")


def _validate_wmma(
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    _validate_common(
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )
    if out_features % 16 != 0:
        raise ValueError("out_features must be a multiple of 16 for compact WMMA")
    if wmma_total_rows <= 0 or wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be positive and a multiple of 16")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_iq_selected_prefill_kernels(*, replace: bool = True) -> None:
    for quant, variant, fn in (
        (
            "gguf_iq2_xs",
            "selected_dual_grouped_prefill_compact_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq2_xs",
            "selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
        ),
        (
            "gguf_iq2_xs",
            "selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out,
        ),
        (
            "gguf_iq2_xs",
            "selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out,
        ),
        (
            "gguf_iq2_xs",
            "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
        ),
        (
            "gguf_iq2_xs",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_grouped_prefill_compact_bf16_bf16_out",
            gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
            gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_silu_grouped_prefill_compact_k3072_n1024_e256_rowbatch4_bf16_bf16_out",
            gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_k3072_n1024_e256_rowbatch4_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_bf16_bf16_out",
            gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
            gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out",
            gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out",
            gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_paired_output_rowbatch8_bf16_bf16_out",
            gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_paired_output_rowbatch8_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
            gguf_iq3_xxs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
            gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_dual_grouped_prefill_compact_bf16_bf16_out",
            gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
            gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_bf16_bf16_out",
            gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out",
            gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out",
            gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_auto_bf16_bf16_out",
            gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_wmma_prefill_compact_bf16_bf16_out",
            gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant, variant),
            fn,
            replace=replace,
        )
    for partition, fn in GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS.items():
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                "gguf_iq3_xxs",
                _iq3_active_expert_persistent_variant(partition),
            ),
            fn,
            replace=replace,
        )
    for output_partition, fn in (
        GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS.items()
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                "gguf_iq3_xxs",
                _iq3_activation_resident_output_variant(output_partition),
            ),
            fn,
            replace=replace,
        )


register_gguf_iq_selected_prefill_kernels()


__all__ = [
    "GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS",
    "GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS",
    "build_gguf_iq_selected_prefill",
    "gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out",
    "gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
    "gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out",
    "gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
    "gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out",
    "gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out",
    "gguf_iq3_xxs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
    "gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out",
    "gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_k3072_n1024_e256_rowbatch4_bf16_bf16_out",
    "gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out",
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_paired_output_rowbatch8_bf16_bf16_out",
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out",
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out",
    "gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
    "gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
    "gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out",
    "gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out",
    "gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out",
    "gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out",
    "gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out",
    "gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out",
    "gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out",
    "gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out",
    "plan_gguf_iq_selected_prefill_build",
    "register_gguf_iq_selected_prefill_kernels",
]
