from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hipengine.kernels.hip_gfx1151 as gfx1151_backend
from hipengine.core.build import plan_hip_build
from hipengine.generation import register_builtin_generators, resolve_text_generator
from hipengine.kernels.backends import (
    CPU_BACKEND,
    backend_package_capability,
    configure_hip_process_environment,
    hip_target_arch_for_backend,
    resolve_backend,
    select_backend,
)
from hipengine.kernels.policy import (
    QWEN35_DENSE_H1024_GEOMETRY,
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    laguna_global_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    laguna_swa_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.hip_gfx1100.norm import (
    paro_rmsnorm_out_fp16,
    register_qwen35_rmsnorm_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
    gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    register_gguf_k_t16_selected_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out,
    gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out,
    gguf_q6_k_wmma_prefill_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100 import (
    GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS as GFX1100_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS,
    GGUF_GDN_INDEXED_SINGLETON_DECODE as GFX1100_GGUF_GDN_INDEXED_SINGLETON_DECODE,
    GGUF_GDN_PREFILL_AUTO_MODE as GFX1100_GGUF_GDN_PREFILL_AUTO_MODE,
    GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE as GFX1100_GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE,
    GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS as GFX1100_GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS,
    GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS as GFX1100_GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS,
    GGUF_Q5_T16_SELECTED_QWEN_TILE8 as GFX1100_GGUF_Q5_T16_SELECTED_QWEN_TILE8,
    GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS as GFX1100_GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS,
    GGUF_Q6_LM_HEAD_MAX_CHUNK as GFX1100_GGUF_Q6_LM_HEAD_MAX_CHUNK,
    GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS as GFX1100_GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS,
    GGUF_Q8_T16_DECODE_ROWTILE_ALL as GFX1100_GGUF_Q8_T16_DECODE_ROWTILE_ALL,
    GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS as GFX1100_GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS,
    GGUF_GDN_PREFILL_EXACT_MODE as GFX1100_GGUF_GDN_PREFILL_EXACT_MODE,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE as GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT as GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT,
    GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK as GFX1100_GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK,
    GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS as GFX1100_GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS,
    GGUF_PREFILL_ROUTER_SELECT_THREADS as GFX1100_GGUF_PREFILL_ROUTER_SELECT_THREADS,
    GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE as GFX1100_GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE as GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS as GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS,
    GGUF_ROUTER_F32_BF16_HIDDEN_THREADS as GFX1100_GGUF_ROUTER_F32_BF16_HIDDEN_THREADS,
    LAGUNA_SWA_PREFILL_VARIANT as GFX1100_LAGUNA_SWA_PREFILL_VARIANT,
)
from hipengine.kernels.hip_gfx1151 import (
    GGUF_C2_PACKED_PREFILL_MAX_ROWS,
    GGUF_DIRECT_RESIDENT_LINEAR_STATE,
    GGUF_FUSED_LINEAR_STATE_TRANSFER,
    GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS,
    GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS,
    GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY,
    GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT,
    GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS,
    GGUF_PREFILL_ROUTER_SELECT_THREADS,
    GGUF_PREFILL_SCRATCH_ARENA_GROUPING,
    GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS,
    GGUF_PREFILL_SCRATCH_LIVENESS_MIN_ROWS,
    GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS,
    GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS,
    GGUF_Q5_T16_SELECTED_QWEN_TILE8,
    GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS,
    GGUF_Q6_LM_HEAD_MAX_CHUNK,
    GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS,
    GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT,
    GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS,
    GGUF_Q8_T16_DECODE_ROWTILE_ALL,
    GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS,
    GGUF_Q8_T16_PREFILL_FOUR_WAVE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS,
    GGUF_ROUTER_F32_BF16_HIDDEN_THREADS,
    LAGUNA_DENSE_Q4_PREFILL_MODE,
    LAGUNA_F16_ATTENTION_QUAD_DECODE,
    LAGUNA_F16_NONTEMPORAL_DECODE,
    LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE,
    LAGUNA_F16_PROJECTION_HEAD_KV_DECODE,
    LAGUNA_F16_BOUNDARY_FUSION,
    LAGUNA_F16_DECODE_FIXEDK,
    LAGUNA_F16_DECODE_ONEBARRIER,
    LAGUNA_Q4_PACK8_DUAL_SILU_DECODE,
    LAGUNA_SELECTED_NATURAL_DECODE,
    LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE,
    LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE,
    LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE,
    LAGUNA_SELECTED_NATURAL_TILE8_DECODE,
    LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_DECODE,
    LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_SILU_DECODE,
    LAGUNA_SELECTED_HALFDOT_DECODE,
    LAGUNA_F16_PREFILL_MIN_ROWS,
    LAGUNA_F16_PREFILL_MODE,
    LAGUNA_F16_PREFILL_STRATEGY,
    LAGUNA_GLOBAL_PREFILL_VARIANT,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_LAYER,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DIM_TILE,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DEFERREDNORM,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_SCORE,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LIVE,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LAYER,
    LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_TOKENLOOP4,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE64,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_MIN_LIVE,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH8_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE,
    LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PROBABILITY_VEC4_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE,
    LAGUNA_MOE_BRANCH_CONCURRENCY,
    LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY,
    LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY,
    LAGUNA_MOE_GROUP_COMPACT_MODE,
    LAGUNA_MOE_SHARED_AFTER_ROUTER,
    LAGUNA_MOE_SHARED_LOW_PRIORITY,
    LAGUNA_PREFILL_ATTENTION_HIPBLASLT,
    LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE,
    LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER,
    LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES,
    LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX,
    LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT,
    LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE,
    LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS,
    LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT,
    LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT,
    LAGUNA_Q6_WMMA_PREFETCH_WEIGHT,
    LAGUNA_Q6_WMMA_PREFETCH_ACTIVATION,
    LAGUNA_PREFILL_CACHED_META,
    LAGUNA_PREFILL_GLOBAL_QROW6,
    LAGUNA_PREFILL_KV_PREAPPEND,
    LAGUNA_PREFILL_MATRIX_ROWS,
    LAGUNA_ROUTER_LOGITS_MODE,
    LAGUNA_SELECTED_DOWN_MODE,
    LAGUNA_SELECTED_GATE_UP_MODE,
    LAGUNA_SWA_PREFILL_VARIANT,
    GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES,
    GGUF_GDN_INDEXED_SINGLETON_DECODE,
    GGUF_GDN_PREFILL_AUTO_MODE,
    GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE,
    GGUF_GDN_PREFILL_COMPACT_PEER_CHUNK_ROWS,
    GGUF_GDN_PREFILL_EXACT_MODE,
    GGUF_Q4_T16_F16_ROCBLAS_PREFILL_POLICIES,
    GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE,
    GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES,
    GGUF_Q6_T16_F16_ROCBLAS_PREFILL_POLICIES,
    GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS,
    GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES,
    GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS,
    GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES,
    GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE,
    GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES,
    GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT,
    GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS,
    GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES,
    GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT,
    GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT,
    GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS,
    GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS,
    TARGET_ARCH,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_gfx1151_bf16_bf16_out,
    register_gfx1151_kernels,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve


def test_rejected_unrouted_kernel_bodies_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    q4_source = (
        root / "hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.hip"
    ).read_text()
    q4_wrapper = (
        root / "hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.py"
    ).read_text()
    laguna_source = (
        root / "hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip"
    ).read_text()

    assert "pack8_wmma64_prefill" not in q4_source
    assert "pack8_wmma64_prefill" not in q4_wrapper
    assert not is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k",
            "pack8_wmma64_prefill_bf16_bf16_out",
        )
    )
    assert (
        "laguna_global_attention_split_exact_gated_mixed32_vstage64_reduce_kernel"
        not in laguna_source
    )


def test_auto_backend_selects_supported_hip_arches() -> None:
    assert select_backend("auto", detected_arches=["gfx1100"]).backend == "hip_gfx1100"
    assert (
        select_backend("auto", detected_arches=["gfx1151:sramecc+:xnack-"]).backend == "hip_gfx1151"
    )


def test_auto_backend_honors_force_env_override() -> None:
    selection = select_backend(
        "auto",
        detected_arches=["gfx1151"],
        env={"HIPENGINE_BACKEND": "hip_gfx1100"},
    )

    assert selection.backend == "hip_gfx1100"
    assert selection.source == "HIPENGINE_BACKEND"


def test_auto_backend_warns_and_falls_back_for_unknown_arch() -> None:
    selection = select_backend("auto", detected_arches=["gfx1102"], env={})

    assert selection.backend == CPU_BACKEND
    assert selection.detected_arches == ("gfx1102",)
    assert selection.warning is not None
    assert "gfx1102" in selection.warning
    assert "HIPENGINE_BACKEND=hip_gfx1100" in selection.warning

    with pytest.warns(RuntimeWarning, match="gfx1102"):
        assert resolve_backend("auto", detected_arches=["gfx1102"], env={}) == CPU_BACKEND


def test_explicit_backend_is_not_autodetected() -> None:
    selection = select_backend("custom_backend", detected_arches=["gfx1151"], env={})

    assert selection.backend == "custom_backend"
    assert selection.source == "explicit"
    assert selection.detected_arches == ()


def test_gfx1151_hip_process_environment_defaults_to_two_hardware_queues() -> None:
    env: dict[str, str] = {}

    applied = configure_hip_process_environment(
        detected_arches=["gfx1151:sramecc+:xnack-"],
        env=env,
    )

    assert applied == {"GPU_MAX_HW_QUEUES": "2"}
    assert env["GPU_MAX_HW_QUEUES"] == "2"


def test_gfx1151_hip_process_environment_preserves_explicit_queue_override() -> None:
    env = {"GPU_MAX_HW_QUEUES": "4"}

    applied = configure_hip_process_environment(detected_arches=["gfx1151"], env=env)

    assert applied == {}
    assert env["GPU_MAX_HW_QUEUES"] == "4"


def test_gfx1151_retired_matrix_policy_cannot_suppress_queue2_default() -> None:
    env = {"HIPENGINE_GPU_MAX_HW_QUEUES_POLICY": "runtime_default"}

    applied = configure_hip_process_environment(detected_arches=["gfx1151"], env=env)

    assert applied == {"GPU_MAX_HW_QUEUES": "2"}
    assert env["GPU_MAX_HW_QUEUES"] == "2"


def test_gfx1100_hip_process_environment_caps_reclaimable_scratch() -> None:
    env: dict[str, str] = {}

    applied = configure_hip_process_environment(detected_arches=["gfx1100"], env=env)

    assert applied == {"HSA_SCRATCH_SINGLE_LIMIT": "8388608"}
    assert env["HSA_SCRATCH_SINGLE_LIMIT"] == "8388608"
    assert "GPU_MAX_HW_QUEUES" not in env


def test_gfx1100_hip_process_environment_preserves_explicit_scratch_override() -> None:
    env = {"HSA_SCRATCH_SINGLE_LIMIT": "67108864"}

    applied = configure_hip_process_environment(detected_arches=["gfx1100"], env=env)

    assert applied == {}
    assert env["HSA_SCRATCH_SINGLE_LIMIT"] == "67108864"


def test_explicit_gfx1100_arch_hint_applies_scratch_default() -> None:
    env = {"HIPENGINE_HIP_ARCH": "gfx1100"}

    applied = configure_hip_process_environment(env=env)

    assert applied == {"HSA_SCRATCH_SINGLE_LIMIT": "8388608"}


def test_mixed_hip_arches_do_not_receive_a_process_wide_queue_default() -> None:
    env: dict[str, str] = {}

    applied = configure_hip_process_environment(
        detected_arches=["gfx1151", "gfx1100"],
        env=env,
    )

    assert applied == {}
    assert "GPU_MAX_HW_QUEUES" not in env
    assert "HSA_SCRATCH_SINGLE_LIMIT" not in env


def test_explicit_gfx1151_backend_hint_applies_when_arch_detection_is_empty() -> None:
    env = {"HIPENGINE_BACKEND": "hip_gfx1151"}

    applied = configure_hip_process_environment(detected_arches=[], env=env)

    assert applied == {"GPU_MAX_HW_QUEUES": "2"}


def test_explicit_gfx1100_backend_hint_applies_when_arch_detection_is_empty() -> None:
    env = {"HIPENGINE_BACKEND": "hip_gfx1100"}

    applied = configure_hip_process_environment(detected_arches=[], env=env)

    assert applied == {"HSA_SCRATCH_SINGLE_LIMIT": "8388608"}


def test_gfx1151_backend_does_not_alias_unvalidated_native_spec_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    source_keys = (
        KernelKey(
            "hip_gfx1100",
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_target_graph",
        ),
        KernelKey(
            "hip_gfx1100",
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_proposal_graph",
        ),
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_gemv_decode_tile4_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_gemv_decode_k1024_wave4_signbit_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out",
        ),
        *(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_grouped_prefill_compact_k1024_active_expert_p"
                f"{partition}_resident_rowbatch8_bf16_bf16_out",
            )
            for partition in (64,)
        ),
        KernelKey(
            "hip_gfx1100",
            "laguna_router_topk",
            "f32",
            "bf16_hidden_correction_bias_persistent_wave_top10",
        ),
    )
    registered: list[KernelKey] = []
    monkeypatch.setattr(backend, "import_module", lambda _name: None)
    monkeypatch.setattr(backend, "registered_keys", lambda: source_keys)
    monkeypatch.setattr(backend, "is_registered", lambda _key: False)
    monkeypatch.setattr(backend, "resolve", lambda **_kwargs: object())
    monkeypatch.setattr(
        backend,
        "register",
        lambda key, _kernel, *, replace=False: registered.append(key),
    )

    backend.register_gfx1151_kernels()

    assert registered == [
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_q8_1_planar_integer_mmq64x64_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_single_wave_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_smallm_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_shared_b_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear_pair",
            "gguf_q4_k",
            "pack8_dual_decode_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            "gguf_q4_k",
            "pack8_dual_decode_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            "gguf_q4_k",
            "pack8_dual_decode_t128_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            "gguf_q4_k",
            "t16_sidecar_dual_decode_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            "gguf_q4_k",
            "t16_dual_interleaved_sidecar_decode_bf16_bf16_out",
        ),
    ]


def test_gfx1151_backend_admits_only_q5_source_f16_prefill() -> None:
    assert GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES == {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): True,
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): True,
    }
    assert GGUF_Q4_T16_F16_ROCBLAS_PREFILL_POLICIES == {}
    assert GGUF_Q6_T16_F16_ROCBLAS_PREFILL_POLICIES == {}
    assert GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES == {
        (6_144, 5_120): {512: 1_280, 1_024: 1_280, 4_096: 1_024},
    }
    assert GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE == {}
    assert GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES == {
        "gguf_q5_k_t16_v1": {
            (6_144, 5_120): {
                (512, 4_096): "f16_rocblas_t16_octet_bf16_bf16_out",
            },
        },
    }


def test_gfx1151_backend_admits_dense_q6_qmicro_planar_exact_routes() -> None:
    register_gguf_q6_k_t16_gemv_kernels(replace=True)
    register_gguf_k_t16_selected_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS",
        (),
    ) == ("attn_qkv",)
    assert GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS == 96
    assert GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES == frozenset({(5_120, 10_240)})
    assert GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS == 256
    assert (
        GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES
        == gfx1151_backend.GGUF_Q4_T16_DENSE_LOWM_SHAPES
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q6_k_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q6_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q6_k_t16_qmicro_planar_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out
    )
    for layer, variant in (
        ("linear", "t16_gemv_decode_bf16_bf16_out"),
        ("linear", "t16_gemv_decode_bf16_f32_out"),
        ("linear", "t16_gemv_rowtile_bf16_bf16_out"),
        ("linear", "t16_gemv_rowtile_col8_bf16_bf16_out"),
        ("linear", "t16_gemv_rowtile_bf16_f32_out"),
        ("linear", "t16_wmma_prefill_bf16_bf16_out"),
        ("linear+argmax", "t16_gemv_decode_bf16_f32_top1_stage1"),
        ("linear+argmax", "proposal_top1_exact_bf16"),
    ):
        assert is_registered(
            KernelKey(
                "hip_gfx1151",
                layer,
                "gguf_q6_k_t16_qmicro_planar_v1",
                variant,
            )
        )


def test_gfx1151_target_verifier_admits_scoped_rowtile_rows_and_shapes() -> None:
    assert GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT == {
        "gguf_q5_k_t16_v1": frozenset({9, 12}),
        "gguf_q6_k_t16_v1": frozenset({9, 12, 16}),
        "gguf_q6_k_t16_qmicro_planar_v1": frozenset({9, 12, 16}),
    }
    assert GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS == frozenset(
        {6, 8, 9, 12}
    )
    assert GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES == frozenset(
        {
            (5_120, 6_144),
            (5_120, 10_240),
            (5_120, 12_288),
            (5_120, 17_408),
            (6_144, 5_120),
            (17_408, 5_120),
        }
    )
    assert GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT == {
        "gguf_q5_k_t16_v1": frozenset({(6_144, 5_120)}),
        "gguf_q6_k_t16_v1": frozenset({(5_120, 10_240)}),
        "gguf_q6_k_t16_qmicro_planar_v1": frozenset(
            {(5_120, 1_024), (17_408, 5_120)}
        ),
    }
    assert GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS[
        ("gguf_q5_k_t16_v1", 16, 6_144, 5_120)
    ] == "t16_gemv_rowtile16_col8_bf16_bf16_out"
    assert len(GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS) == 15
    assert set(GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS.values()) == {
        "t16_wmma_prefill_shared_b2w2_bf16_bf16_out",
        "t16_wmma_prefill_shared4_bf16_bf16_out",
    }
    assert {
        key[1] for key in GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS
    } == {20, 24, 32}
    assert GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT["gguf_q6_k_t16_v1"] == 8
    assert (
        GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT[
            "gguf_q6_k_t16_qmicro_planar_v1"
        ]
        == 8
    )


def test_gfx1151_q5_standard_prefill_shared8r3_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def retained(*args, **kwargs):
        calls.append(("retained", args, kwargs))

    def shared8r3(*args, **kwargs):
        calls.append(("shared8r3", args, kwargs))

    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q5_k_t16_wmma_prefill_bf16_bf16_out",
        retained,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out",
        retained,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out",
        shared8r3,
    )
    fn = gfx1151_backend.gguf_q5_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    fn(1, 2, 3, 288, 6_144, 5_120, stream=7)
    fn(1, 2, 3, 65, 6_144, 5_120, stream=13)
    fn(1, 2, 3, 96, 6_144, 5_120, stream=14)
    fn(1, 2, 3, 97, 6_144, 5_120, stream=15)
    fn(1, 2, 3, 256, 6_144, 5_120, stream=8)
    fn(1, 2, 3, 385, 6_144, 5_120, stream=9)
    fn(1, 2, 3, 1_024, 6_144, 5_120, stream=11)
    fn(1, 2, 3, 1_025, 6_144, 5_120, stream=12)
    fn(1, 2, 3, 288, 5_120, 10_240, stream=10)

    assert calls == [
        ("shared8r3", (1, 2, 3, 288, 6_144, 5_120), {"stream": 7}),
        ("retained", (1, 2, 3, 65, 6_144, 5_120), {"stream": 13}),
        ("retained", (1, 2, 3, 96, 6_144, 5_120), {"stream": 14}),
        ("retained", (1, 2, 3, 97, 6_144, 5_120), {"stream": 15}),
        ("shared8r3", (1, 2, 3, 256, 6_144, 5_120), {"stream": 8}),
        ("retained", (1, 2, 3, 385, 6_144, 5_120), {"stream": 9}),
        ("retained", (1, 2, 3, 1_024, 6_144, 5_120), {"stream": 11}),
        ("retained", (1, 2, 3, 1_025, 6_144, 5_120), {"stream": 12}),
        ("retained", (1, 2, 3, 288, 5_120, 10_240), {"stream": 10}),
    ]


def test_gfx1151_q6_shared3r1_is_scoped_to_rows33_48(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def retained(*args, **kwargs):
        calls.append(("retained", args, kwargs))

    def shared3r1(*args, **kwargs):
        calls.append(("shared3r1", args, kwargs))

    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_wmma_prefill_bf16_bf16_out",
        retained,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_wmma_prefill_shared3r1_bf16_bf16_out",
        shared3r1,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out",
        shared3r1,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out",
        retained,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out",
        retained,
    )
    standard = gguf_q6_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    planar = gfx1151_backend.gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out
    standard(1, 2, 3, 35, 5_120, 10_240, stream=7)
    standard(1, 2, 3, 32, 5_120, 10_240, stream=8)
    planar(1, 2, 3, 48, 17_408, 5_120, stream=9)
    planar(1, 2, 3, 49, 17_408, 5_120, stream=10)
    planar(1, 2, 3, 35, 5_120, 1_024, stream=11)

    assert calls == [
        ("shared3r1", (1, 2, 3, 35, 5_120, 10_240), {"stream": 7}),
        ("retained", (1, 2, 3, 32, 5_120, 10_240), {"stream": 8}),
        ("shared3r1", (1, 2, 3, 48, 17_408, 5_120), {"stream": 9}),
        ("retained", (1, 2, 3, 49, 17_408, 5_120), {"stream": 10}),
        ("shared3r1", (1, 2, 3, 35, 5_120, 1_024), {"stream": 11}),
    ]


def test_gfx1151_q6_planar_shared4r6_routes_high_rows_by_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared4r6"),
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("fallback"),
    )
    fn = gfx1151_backend.gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out
    fn(1, 2, 3, 1_024, 17_408, 5_120)
    fn(1, 2, 3, 536, 5_120, 1_024)
    fn(1, 2, 3, 1_024, 5_120, 1_024)
    assert calls == ["shared4r6", "shared4r6", "fallback"]


def test_gfx1151_q6_standard_prefill_shared4_is_qkv_shape_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def retained(*args, **kwargs):
        calls.append(("retained", args, kwargs))

    def shared4(*args, **kwargs):
        calls.append(("shared4", args, kwargs))

    def shared6r1(*args, **kwargs):
        calls.append(("shared6r1", args, kwargs))

    def shared8r3(*args, **kwargs):
        calls.append(("shared8r3", args, kwargs))

    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_wmma_prefill_bf16_bf16_out",
        retained,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out",
        shared4,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out",
        shared6r1,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out",
        shared8r3,
    )
    fn = gguf_q6_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    fn(1, 2, 3, 512, 5_120, 10_240, stream=7)
    fn(1, 2, 3, 1_024, 5_120, 10_240, stream=13)
    fn(1, 2, 3, 1_025, 5_120, 10_240, stream=14)
    fn(1, 2, 3, 288, 5_120, 10_240, stream=11)
    fn(1, 2, 3, 256, 5_120, 10_240, stream=16)
    fn(1, 2, 3, 96, 5_120, 10_240, stream=8)
    fn(1, 2, 3, 95, 5_120, 10_240, stream=9)
    fn(1, 2, 3, 49, 5_120, 10_240, stream=12)
    fn(1, 2, 3, 1_024, 5_120, 5_120, stream=10)

    assert calls == [
        ("shared8r3", (1, 2, 3, 512, 5_120, 10_240), {"stream": 7}),
        ("shared8r3", (1, 2, 3, 1_024, 5_120, 10_240), {"stream": 13}),
        ("shared4", (1, 2, 3, 1_025, 5_120, 10_240), {"stream": 14}),
        ("shared8r3", (1, 2, 3, 288, 5_120, 10_240), {"stream": 11}),
        ("shared8r3", (1, 2, 3, 256, 5_120, 10_240), {"stream": 16}),
        ("shared6r1", (1, 2, 3, 96, 5_120, 10_240), {"stream": 8}),
        ("shared6r1", (1, 2, 3, 95, 5_120, 10_240), {"stream": 9}),
        ("shared6r1", (1, 2, 3, 49, 5_120, 10_240), {"stream": 12}),
        ("retained", (1, 2, 3, 1_024, 5_120, 5_120), {"stream": 10}),
    ]


def test_gfx1151_q6_planar_prefill_shared4_covers_physical_shapes_from_row256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def retained(*args, **kwargs):
        calls.append(("retained", args, kwargs))

    def shared4(*args, **kwargs):
        calls.append(("shared4", args, kwargs))

    def shared4r3(*args, **kwargs):
        calls.append(("shared4r3", args, kwargs))

    def shared4r4(*args, **kwargs):
        calls.append(("shared4r4", args, kwargs))

    def shared4r6(*args, **kwargs):
        calls.append(("shared4r6", args, kwargs))

    def shared4r9(*args, **kwargs):
        calls.append(("shared4r9", args, kwargs))

    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out",
        retained,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out",
        shared4,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out",
        shared4r3,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out",
        shared4r4,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out",
        shared4r6,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out",
        shared4r9,
    )
    fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out
    fn(1, 2, 3, 512, 17_408, 5_120, stream=7)
    fn(1, 2, 3, 256, 17_408, 5_120, stream=8)
    fn(1, 2, 3, 256, 5_120, 1_024, stream=13)
    fn(1, 2, 3, 256, 5_120, 6_144, stream=9)
    fn(1, 2, 3, 255, 5_120, 6_144, stream=10)
    fn(1, 2, 3, 1_024, 5_120, 1_024, stream=11)
    fn(1, 2, 3, 512, 5_120, 248_320, stream=12)

    assert calls == [
        ("shared4r6", (1, 2, 3, 512, 17_408, 5_120), {"stream": 7}),
        ("shared4r4", (1, 2, 3, 256, 17_408, 5_120), {"stream": 8}),
        ("shared4r3", (1, 2, 3, 256, 5_120, 1_024), {"stream": 13}),
        ("shared4", (1, 2, 3, 256, 5_120, 6_144), {"stream": 9}),
        ("retained", (1, 2, 3, 255, 5_120, 6_144), {"stream": 10}),
        ("retained", (1, 2, 3, 1_024, 5_120, 1_024), {"stream": 11}),
        ("retained", (1, 2, 3, 512, 5_120, 248_320), {"stream": 12}),
    ]


def test_gfx1151_q6_planar_prefill_lowvgpr_bands_route_by_rows_and_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    for name, tag in (
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out",
            "plain",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out",
            "lowvgpr",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out",
            "lowvgpr48",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out",
            "shared4",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out",
            "shared4r3",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out",
            "shared4r4",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out",
            "shared4r6",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out",
            "shared4r9",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out",
            "shared3r1",
        ),
    ):
        monkeypatch.setattr(
            gfx1151_backend,
            name,
            (lambda tag: lambda *a, **k: calls.append(tag))(tag),
        )

    fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out
    shapes = sorted(gfx1151_backend.GGUF_Q4_T16_DENSE_LOWM_SHAPES)
    for rows in (17, 32):
        for in_f, out_f in shapes:
            fn(1, 2, 3, rows, in_f, out_f)
    for rows in (33, 48):
        fn(1, 2, 3, rows, 5_120, 17_408)
    others = [s for s in shapes if s != (5_120, 17_408)]
    # (17_408, 5_120) is a retained shared3r1 shape at rows 33-48; every
    # other low-M shape stays on the low-VGPR band there.
    def _rows33_48_tag(shape: tuple[int, int]) -> str:
        return "shared3r1" if shape == (17_408, 5_120) else "lowvgpr"

    for rows in (33, 48):
        for in_f, out_f in others:
            fn(1, 2, 3, rows, in_f, out_f)
    for rows in (49, 64):
        for in_f, out_f in shapes:
            fn(1, 2, 3, rows, in_f, out_f)
    lowvgpr80_shapes = gfx1151_backend.GGUF_Q6_PLANAR_LOWVGPR80_SHAPES
    for rows in (65, 80):
        for in_f, out_f in sorted(lowvgpr80_shapes):
            fn(1, 2, 3, rows, in_f, out_f)
        for in_f, out_f in sorted(set(shapes) - lowvgpr80_shapes):
            fn(1, 2, 3, rows, in_f, out_f)
    for rows, in_f, out_f in (
        (16, 5_120, 6_144),
        (145, 6_144, 5_120),
        (45, 5_120, 1_024),
        (512, 17_408, 5_120),
    ):
        fn(1, 2, 3, rows, in_f, out_f)

    expected = (
        ["lowvgpr"] * (2 * len(shapes))
        + ["lowvgpr48"] * 2
        + [_rows33_48_tag(s) for s in others] * 2
        + ["lowvgpr"] * (2 * len(shapes))
    )
    for _ in (65, 80):
        expected += ["lowvgpr"] * len(lowvgpr80_shapes)
        expected += ["lowvgpr48"] * len(set(shapes) - lowvgpr80_shapes)
    # rows45 (5_120, 1_024) is the other retained shared3r1 shape and rows512
    # (17_408, 5_120) is the retained shared4r6 band; both must stay stubbed.
    expected += ["plain", "plain", "shared3r1", "shared4r6"]
    assert calls == expected


def test_gfx1151_q5_prefill_lowvgpr_bands_and_registry_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    for name, tag in (
        ("gguf_q5_k_t16_wmma_prefill_bf16_bf16_out", "plain"),
        (
            "gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out",
            "lowvgpr",
        ),
        (
            "gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out",
            "lowvgpr48",
        ),
    ):
        monkeypatch.setattr(
            gfx1151_backend,
            name,
            (lambda tag: lambda *a, **k: calls.append(tag))(tag),
        )

    fn = gfx1151_backend.gguf_q5_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    shapes = sorted(gfx1151_backend.GGUF_Q5_T16_DENSE_LOWM_SHAPES)
    for rows in (17, 32):
        for in_f, out_f in shapes:
            fn(1, 2, 3, rows, in_f, out_f)
    for rows in (33, 48):
        fn(1, 2, 3, rows, 17_408, 5_120)
    lowvgpr48_shapes = sorted(
        gfx1151_backend.GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES
    )
    for rows in (33, 48):
        for in_f, out_f in lowvgpr48_shapes:
            fn(1, 2, 3, rows, in_f, out_f)
    for rows in (33, 48):
        fn(1, 2, 3, rows, 5_120, 17_408)
    lowvgpr64_shapes = gfx1151_backend.GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES
    for rows in (49, 64):
        for in_f, out_f in sorted(lowvgpr64_shapes):
            fn(1, 2, 3, rows, in_f, out_f)
        for in_f, out_f in sorted(set(shapes) - lowvgpr64_shapes):
            fn(1, 2, 3, rows, in_f, out_f)
    lowvgpr80_shapes = gfx1151_backend.GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES
    for rows in (65, 80):
        for in_f, out_f in sorted(lowvgpr80_shapes):
            fn(1, 2, 3, rows, in_f, out_f)
        for in_f, out_f in sorted(set(shapes) - lowvgpr80_shapes):
            fn(1, 2, 3, rows, in_f, out_f)
    for rows, in_f, out_f in (
        (16, 5_120, 6_144),
        (145, 6_144, 5_120),
        (45, 5_120, 1_024),
    ):
        fn(1, 2, 3, rows, in_f, out_f)

    expected = (
        ["lowvgpr"] * (2 * len(shapes))
        + ["lowvgpr"] * 2
        + ["lowvgpr48"] * (2 * len(lowvgpr48_shapes))
        + ["plain"] * 2
    )
    for _ in (49, 64):
        expected += ["lowvgpr"] * len(lowvgpr64_shapes)
        expected += ["plain"] * len(set(shapes) - lowvgpr64_shapes)
    for _ in (65, 80):
        expected += ["lowvgpr"] * len(lowvgpr80_shapes)
        expected += ["lowvgpr48"] * len(set(shapes) - lowvgpr80_shapes)
    expected += ["plain"] * 3
    assert calls == expected

    register_gfx1151_kernels()
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q5_k_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is fn
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="moe_linear",
            quant="gguf_q5_k_t16_v1",
            variant="selected_wmma_prefill_compact_bf16_bf16_out",
        )
        is gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out
    )


def test_gfx1151_registers_y1_q4_single_sweep_variant() -> None:
    register_gfx1151_kernels(replace=True)
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q4_k_t16_v1",
            variant="t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out",
        )
        is gfx1151_backend.gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out
    )


def test_gfx1151_highrow_prefill_bands_route_by_family_and_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    symbols = {
        # Q4 owners
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out": "q4_plain",
        "gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out": "q4_lv",
        "gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out": "q4_lv48",
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out": "q4_shared",
        "gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out": "q4_shared2w2",
        "gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out": "q4_shared2w4",
        "gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out": "q4_shared3w8r3",
        # Q5 owners
        "gguf_q5_k_t16_wmma_prefill_bf16_bf16_out": "q5_plain",
        "gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out": "q5_lv",
        "gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out": "q5_lv48",
        # Q6 planar owners
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out": "q6_plain",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out": "q6_lv",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out": "q6_lv48",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out": "q6_shared4",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out": "q6_shared4r3",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out": "q6_shared4r4",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out": "q6_shared4r6",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out": "q6_shared4r9",
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out": "q6_shared3r1",
    }
    for name, tag in symbols.items():
        monkeypatch.setattr(
            gfx1151_backend,
            name,
            (lambda tag: lambda *a, **k: calls.append(tag))(tag),
        )

    shapes = sorted(gfx1151_backend.GGUF_Q4_T16_DENSE_LOWM_SHAPES)

    def assert_routes(fn, rows: tuple[int, ...], expected: dict[tuple[int, int], str]):
        for row_count in rows:
            for shape in shapes:
                fn(1, 2, 3, row_count, *shape)
                assert calls.pop() == expected[shape]
        assert not calls

    q4 = gfx1151_backend.gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    assert_routes(
        q4,
        (81, 96),
        {
            (5_120, 6_144): "q4_lv48",
            (5_120, 10_240): "q4_lv48",
            (5_120, 12_288): "q4_lv",
            (5_120, 17_408): "q4_lv48",
            (6_144, 5_120): "q4_lv48",
            (17_408, 5_120): "q4_lv",
        },
    )
    assert_routes(
        q4,
        (97, 128),
        {
            (5_120, 6_144): "q4_plain",
            (5_120, 10_240): "q4_lv",
            (5_120, 12_288): "q4_lv48",
            (5_120, 17_408): "q4_shared",
            (6_144, 5_120): "q4_lv48",
            (17_408, 5_120): "q4_lv",
        },
    )
    assert_routes(
        q4,
        (129, 144),
        {
            (5_120, 6_144): "q4_lv48",
            (5_120, 10_240): "q4_shared",
            (5_120, 12_288): "q4_lv48",
            (5_120, 17_408): "q4_shared",
            (6_144, 5_120): "q4_lv48",
            (17_408, 5_120): "q4_lv48",
        },
    )
    assert_routes(
        q4,
        (145, 192),
        {
            **{shape: "q4_shared" for shape in shapes},
            (17_408, 5_120): "q4_shared2w4",
            (6_144, 5_120): "q4_shared2w2",
        },
    )
    assert_routes(
        q4,
        (193, 256),
        {
            **{shape: "q4_shared" for shape in shapes},
            (17_408, 5_120): "q4_shared2w2",
            (6_144, 5_120): "q4_shared2w2",
        },
    )
    assert_routes(
        q4,
        (257, 287),
        {
            **{shape: "q4_shared2w2" for shape in shapes},
            (5_120, 12_288): "q4_shared",
        },
    )
    assert_routes(
        q4,
        (288, 384),
        {
            **{shape: "q4_shared2w2" for shape in shapes},
            (5_120, 12_288): "q4_shared3w8r3",
            (5_120, 17_408): "q4_shared3w8r3",
            (17_408, 5_120): "q4_shared3w8r3",
        },
    )
    assert_routes(q4, (385,), {shape: "q4_shared" for shape in shapes})

    q5 = gfx1151_backend.gguf_q5_k_t16_wmma_prefill_gfx1151_bf16_bf16_out
    assert_routes(
        q5,
        (81, 96),
        {
            (5_120, 6_144): "q5_lv48",
            (5_120, 10_240): "q5_lv48",
            (5_120, 12_288): "q5_lv",
            (5_120, 17_408): "q5_lv48",
            (6_144, 5_120): "q5_lv48",
            (17_408, 5_120): "q5_lv",
        },
    )
    assert_routes(
        q5,
        (97, 128),
        {
            (5_120, 6_144): "q5_plain",
            (5_120, 10_240): "q5_lv48",
            (5_120, 12_288): "q5_lv48",
            (5_120, 17_408): "q5_plain",
            (6_144, 5_120): "q5_lv48",
            (17_408, 5_120): "q5_lv48",
        },
    )
    assert_routes(q5, (129, 144), {shape: "q5_lv48" for shape in shapes})
    assert_routes(q5, (145,), {shape: "q5_plain" for shape in shapes})

    q6 = gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out
    assert_routes(
        q6,
        (81, 96),
        {
            (5_120, 6_144): "q6_lv48",
            (5_120, 10_240): "q6_lv48",
            (5_120, 12_288): "q6_lv",
            (5_120, 17_408): "q6_lv48",
            (6_144, 5_120): "q6_lv48",
            (17_408, 5_120): "q6_lv",
        },
    )
    assert_routes(
        q6,
        (97, 128),
        {
            (5_120, 6_144): "q6_plain",
            (5_120, 10_240): "q6_shared4",
            (5_120, 12_288): "q6_plain",
            (5_120, 17_408): "q6_shared4",
            (6_144, 5_120): "q6_lv48",
            (17_408, 5_120): "q6_lv",
        },
    )
    assert_routes(
        q6,
        (129, 144),
        {
            (5_120, 6_144): "q6_shared4",
            (5_120, 10_240): "q6_shared4",
            (5_120, 12_288): "q6_shared4",
            (5_120, 17_408): "q6_shared4",
            (6_144, 5_120): "q6_lv48",
            (17_408, 5_120): "q6_lv48",
        },
    )
    assert_routes(q6, (145, 255), {shape: "q6_plain" for shape in shapes})
    # The retained Y2 high-row owners take (17_408, 5_120) at exact bands:
    # shared4r4 at rows256, shared4r6 from row288, shared4r9 at rows536.
    assert_routes(
        q6,
        (256,),
        {
            **{shape: "q6_shared4" for shape in shapes},
            (17_408, 5_120): "q6_shared4r4",
        },
    )
    assert_routes(
        q6,
        (288,),
        {
            **{shape: "q6_shared4" for shape in shapes},
            (17_408, 5_120): "q6_shared4r6",
        },
    )
    assert_routes(
        q6,
        (536,),
        {
            **{shape: "q6_shared4" for shape in shapes},
            (17_408, 5_120): "q6_shared4r9",
        },
    )


def test_gfx1151_backend_scopes_dense_down_residual_fusions() -> None:
    register_gfx1151_kernels()

    for quant, variant in (
        (
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_residual_bf16_out",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_residual_bf16_out",
        ),
    ):
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                "linear+residual",
                quant,
                variant,
            )
        )
    for quant, variant in (
        (
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_residual_bf16_out",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_residual_bf16_out",
        ),
    ):
        assert is_registered(
            KernelKey(
                "hip_gfx1151",
                "linear+residual",
                quant,
                variant,
            )
        )
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT",
        {},
    ) == {
        "gguf_q4_k_t16_v1": 4,
        "gguf_q6_k_t16_qmicro_planar_v1": 3,
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT",
        None,
    ) == {
        "gguf_q4_k_t16_v1": 4,
        "gguf_q6_k_t16_qmicro_planar_v1": 3,
        "bf16": 512,
    }


def test_gfx1151_backend_scopes_08b_short_attention_split_policy() -> None:
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_SHORT_C1_SPLIT_ATTN_POLICIES",
        {},
    ) == {
        (1_024, 24, 8, 2, 256, 256, 256): (514, 641),
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_SHORT_C1_SPLIT_ATTN_POLICIES",
        {},
    ) == {}


def test_gfx1151_backend_declares_q4_two_wave_shape_and_row_policy() -> None:
    policy = GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT[
        "gguf_q4_k_t16_v1"
    ]
    shapes = policy["shapes"]
    assert shapes[(5_120, 1_024)] == "dense_rowtile16_w2_bf16_bf16_out"
    assert shapes[(5_120, 12_288)] == "dense_rowtile16_w2_bf16_bf16_out"
    assert shapes[(17_408, 5_120)] == "dense_rowtile16_w2_bf16_bf16_out"
    assert (1_024, 4_096) not in shapes
    assert policy["rows_by_shape"] == {
        (5_120, 10_240): (3, 4, 8),
        (5_120, 12_288): (2, 3, 4),
    }
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile16_w2_bf16_bf16_out",
        )
    )


def test_gfx1151_backend_routes_admitted_physical_q4_shapes_to_smallm_wmma(
    monkeypatch,
) -> None:
    """Scaling-campaign M2j: rows<=16 physical cells supersede the one-row-tile
    smallm owner with the measured bit-exact siblings (low-VGPR everywhere,
    shared-B2W2 on the N5120 down-projection)."""
    selector = getattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out",
        None,
    )
    rows_policy = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_PHYSICAL_SMALLM_ROWS",
        None,
    )
    shape_policy = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES",
        None,
    )
    assert callable(selector)
    assert rows_policy == frozenset({6, 8, 12, 16})
    assert shape_policy == frozenset(
        {
            (5_120, 6_144),
            (5_120, 10_240),
            (5_120, 12_288),
            (5_120, 17_408),
            (6_144, 5_120),
            (17_408, 5_120),
        }
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("smallm"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("single"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("lowvgpr"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared_b2w2"),
        raising=False,
    )

    expected: list[str] = []
    for rows in (2, 4, 6, 8, 12, 16):
        for in_features, out_features in sorted(shape_policy):
            expected.append(
                "shared_b2w2" if (in_features, out_features) == (17_408, 5_120)
                else "lowvgpr"
            )
            selector(1, 2, 3, rows, in_features, out_features)
    tail: list[str] = []
    for rows, in_features, out_features in (
        (16, 5_120, 1_024),
        (16, 5_120, 5_120),
        (16, 1_024, 4_096),
    ):
        tail.append("shared")
        selector(1, 2, 3, rows, in_features, out_features)

    assert calls == expected + tail


def test_gfx1151_backend_routes_admitted_lowm_rows_to_single_wave_wmma(
    monkeypatch,
) -> None:
    selector = getattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out",
        None,
    )
    lowm_max_rows = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS",
        None,
    )
    lowm_shapes = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWM_SHAPES",
        None,
    )
    lowvgpr_max_rows = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS",
        None,
    )
    lowvgpr48_max_rows = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS",
        None,
    )
    lowvgpr48_shapes = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES",
        None,
    )
    lowvgpr64_shapes = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES",
        None,
    )
    lowvgpr80_shapes = getattr(
        gfx1151_backend,
        "GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES",
        None,
    )
    assert callable(selector)
    assert lowm_max_rows == 80
    assert lowvgpr_max_rows == 32
    assert lowvgpr48_max_rows == 48
    assert lowm_shapes == frozenset(
        {
            (5_120, 6_144),
            (5_120, 10_240),
            (5_120, 12_288),
            (5_120, 17_408),
            (6_144, 5_120),
            (17_408, 5_120),
        }
    )
    assert lowvgpr48_shapes == frozenset(
        {
            (5_120, 10_240),
            (5_120, 12_288),
            (5_120, 17_408),
        }
    )
    assert lowvgpr64_shapes == lowm_shapes - {(5_120, 17_408)}
    assert lowvgpr80_shapes == frozenset({(17_408, 5_120)})
    calls: list[str] = []
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("smallm"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("single"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("lowvgpr"),
        raising=False,
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("lowvgpr48"),
        raising=False,
    )

    # rows 17-32: lowvgpr on every admitted shape
    for rows in (17, 24, 32):
        for in_features, out_features in sorted(lowm_shapes):
            selector(1, 2, 3, rows, in_features, out_features)
    # rows 33-48: lowvgpr48 on its three shapes, lowvgpr on the other three
    for rows in (33, 45, 48):
        for in_features, out_features in sorted(lowvgpr48_shapes):
            selector(1, 2, 3, rows, in_features, out_features)
    others = sorted(lowm_shapes - lowvgpr48_shapes)
    for rows in (33, 45, 48):
        for in_features, out_features in others:
            selector(1, 2, 3, rows, in_features, out_features)
    # rows 49-64: lowvgpr on five shapes; wide 5120->17408 keeps single.
    for rows in (49, 64):
        for in_features, out_features in sorted(lowvgpr64_shapes):
            selector(1, 2, 3, rows, in_features, out_features)
        selector(1, 2, 3, rows, 5_120, 17_408)
    # rows 65-80: 17408->5120 uses lowvgpr; the other five use lowvgpr48.
    for rows in (65, 80):
        for in_features, out_features in sorted(lowvgpr80_shapes):
            selector(1, 2, 3, rows, in_features, out_features)
        for in_features, out_features in sorted(lowm_shapes - lowvgpr80_shapes):
            selector(1, 2, 3, rows, in_features, out_features)
    for rows, in_features, out_features in (
        (145, 5_120, 17_408),
        (145, 5_120, 12_288),
        (45, 5_120, 1_024),
        (45, 4_096, 4_096),
    ):
        selector(1, 2, 3, rows, in_features, out_features)

    expected = (
        ["lowvgpr"] * (3 * len(lowm_shapes))
        + ["lowvgpr48"] * (3 * len(lowvgpr48_shapes))
        + ["lowvgpr"] * (3 * len(others))
    )
    for _ in (49, 64):
        expected += ["lowvgpr"] * len(lowvgpr64_shapes) + ["single"]
    for _ in (65, 80):
        expected += ["lowvgpr"] * len(lowvgpr80_shapes)
        expected += ["lowvgpr48"] * len(lowm_shapes - lowvgpr80_shapes)
    expected += ["shared"] * 4
    assert calls == expected


def test_gfx1151_backend_registers_q4_physical_route_and_explicit_fallbacks() -> None:
    gfx1151_backend.register_gfx1151_kernels(replace=True)

    selected = resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_bf16_bf16_out",
    )
    single = resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_single_wave_bf16_bf16_out",
    )
    smallm = resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_smallm_bf16_bf16_out",
    )
    fallback = resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_shared_b_bf16_bf16_out",
    )
    peer = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_bf16_bf16_out",
    )

    assert selected is getattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out",
    )
    assert single is gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    assert smallm is getattr(
        gfx1151_backend,
        "gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out",
    )
    assert fallback is gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out
    assert peer is gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out


def test_gfx1151_backend_overrides_q5_rowtile_with_scoped_col8_wrapper(
    monkeypatch,
) -> None:
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q5_k_t16_v1",
        variant="t16_gemv_rowtile_bf16_bf16_out",
    ) is gfx1151_backend._gguf_q5_k_t16_gemv_rowtile_gfx1151_bf16_bf16_out
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_rowtile_col8_bf16_bf16_out",
        )
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("col8"),
    )
    monkeypatch.setattr(
        gfx1151_backend,
        "gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("col4"),
    )
    wrapper = gfx1151_backend._gguf_q5_k_t16_gemv_rowtile_gfx1151_bf16_bf16_out
    wrapper(1, 2, 3, 8, 6_144, 5_120)
    wrapper(1, 2, 3, 4, 2_048, 1_024)
    assert calls == ["col8", "col4"]


def test_gfx1151_backend_declares_generation2_physical_widths() -> None:
    assert GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS == (1, 2, 3, 4, 5, 6, 7, 8)
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", (1,)
    ) == (1, 2, 3, 4, 5, 6, 7, 8)
    assert GGUF_C2_PACKED_PREFILL_MAX_ROWS == 8
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_C2_PACKED_PREFILL_MAX_ROWS", 1
    ) == 8
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_C2_PACKED_PREFILL_MAX_ROWS", 1
    ) == 8
    assert GGUF_DIRECT_RESIDENT_LINEAR_STATE is True
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_DIRECT_RESIDENT_LINEAR_STATE", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_DIRECT_RESIDENT_LINEAR_STATE", False
    ) is False
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_DIRECT_RESIDENT_VERIFY_LINEAR_STATE_POLICY", {}
    ) == {
        "enabled_env": "HIPENGINE_GGUF_VERIFY_DIRECT_RESIDENT_LINEAR_STATE",
        "enabled_default": True,
    }
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_DIRECT_RESIDENT_VERIFY_LINEAR_STATE_POLICY", {}
    ) == {}
    assert GGUF_FUSED_LINEAR_STATE_TRANSFER is True
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_FUSED_LINEAR_STATE_TRANSFER", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_FUSED_LINEAR_STATE_TRANSFER", False
    ) is False


def test_gfx1151_backend_scopes_packed_prefill_final_output_mask() -> None:
    assert GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK is True
    assert GFX1100_GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK is False
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK", False
    ) is False


def test_gfx1151_backend_scopes_dense_grouped_gqa_split_policy() -> None:
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_PAGED_ATTN_GROUPED_GQA_MIN_CONTEXTS",
        {},
    ) == {
        (5_120, 64, 24, 4, 256, 256, 256): 4_096,
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_PAGED_ATTN_GROUPED_GQA_MIN_CONTEXTS",
        {},
    ) == {}


def test_gfx1151_backend_admits_dense_h5120_sole_q4_t16() -> None:
    register_gfx1151_kernels()

    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q4_T16",
        False,
    )


def test_gfx1151_backend_admits_dense_q5_t16_ssm_out_and_08b_roles() -> None:
    register_gguf_k_t16_selected_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_Q5_T16_SSM_OUT",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q5_T16_SSM_OUT",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_Q5_T16_H5120",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q5_T16_H5120",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_Q5_T16_QKV",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q5_T16_QKV",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
        False,
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
        (),
    ) == ("MOSTLY_Q4_K_S",)
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
        {},
    ) == {
        (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 1_024, 3_584): "pack8_dual_decode_t128_bf16_bf16_out",
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 5_120, 17_408): "dense_dual_local32_bf16_bf16_out",
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
            (1, 5_120, 17_408): (
                "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out"
            ),
        },
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
        {},
    ) == {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 5_120, 17_408): "dense_dual_local32_bf16_bf16_out",
        },
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_PAIR_SILU_NATIVE_DECODE_POLICIES",
        {},
    ) == {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 5_120, 17_408): "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
            (1, 5_120, 17_408): (
                "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out"
            ),
            **{
                (rows, 5_120, 17_408): (
                    "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out"
                )
                # rowtile8 chunks c>8 at the dispatch site; the decode regime
                # goes to rows 511 so no gate/up concurrency falls to WMMA.
                # c>=512 routes to the WMMA prefill owner before this policy.
                for rows in range(2, 512)
            },
        },
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_T16_NATIVE_SPLIT_ROW_CHUNKS_BY_QUANT_SHAPE",
        {},
    ) == {"gguf_q4_k_t16_v1": {(8, 1_024, 4_096): 4}}
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_T16_NATIVE_DIRECT_SHAPES_BY_QUANT",
        {},
    ) == {"gguf_q5_k_t16_v1": frozenset({(2_048, 1_024)})}
    for variant in (
        "t16_gemv_decode_bf16_bf16_out",
        "t16_gemv_rowtile_bf16_bf16_out",
        "t16_wmma_prefill_bf16_bf16_out",
    ):
        assert is_registered(
            KernelKey(
                "hip_gfx1151",
                "linear",
                "gguf_q5_k_t16_v1",
                variant,
            )
        )
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "gdn_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_lowp_f32_bf16_out",
        )
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "gdn_chain_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
        )
    )


def test_gfx1151_dense_pair_silu_t128_variant_binds_threads(monkeypatch) -> None:
    import hipengine.kernels.hip_gfx1151 as gfx1151

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gfx1151,
        "gguf_q4_k_pack8_dual_silu_bf16_bf16_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    register_gfx1151_kernels(replace=True)
    fn = resolve(
        backend="hip_gfx1151",
        layer="linear_pair_silu",
        quant="gguf_q4_k",
        variant="pack8_dual_decode_t128_bf16_bf16_out",
    )
    fn(1, 2, 3, 4, 5, 6, 7, 8, 1, 1_024, 3_584, stream=9, runtime="runtime")

    assert calls == [
        (
            (1, 2, 3, 4, 5, 6, 7, 8, 1, 1_024, 3_584),
            {"stream": 9, "runtime": "runtime", "threads": 128},
        )
    ]


def test_gfx1151_dense_down_residual_policies_are_exact() -> None:
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
        register_dense_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        register_gguf_q4_k_gemv_kernels,
    )

    register_dense_gemv_kernels()
    register_gguf_q4_k_gemv_kernels()
    register_gfx1151_kernels(replace=True)
    expected = {
        (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 3_584, 1_024): True
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 17_408, 5_120): True
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
            (1, 17_408, 5_120): True
        },
    }
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES", {}
    ) == expected
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES", {}
    ) == {}
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT", {}
    )["bf16"] == 512
    for quant, variant in (
        ("gguf_q4_k", "pack8_bf16_residual_bf16_out"),
        ("bf16", "out_bf16_residual_bf16_out"),
        ("bf16", "prefill_wmma_out_bf16_residual_bf16_out"),
    ):
        assert is_registered(
            KernelKey("hip_gfx1151", "linear+residual", quant, variant)
        )
        assert is_registered(
            KernelKey("hip_gfx1100", "linear+residual", quant, variant)
        )


def test_gfx1151_fixed_norm_residual_policies_are_exact() -> None:
    register_gfx1151_kernels(replace=True)
    expected = {
        (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 1_024): "bf16_out_fixed1024_wave256"
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            (1, 5_120): "bf16_out_fixed5120_wave256"
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
            (1, 5_120): "bf16_out_fixed5120_wave256"
        },
    }
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_NORM_RESIDUAL_DECODE_POLICIES", {}
    ) == expected
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_NORM_RESIDUAL_DECODE_POLICIES", {}
    ) == {}
    for layer in ("rmsnorm", "add_rmsnorm"):
        for variant in (
            "bf16_out_fixed1024_wave256",
            "bf16_out_fixed5120_wave256",
        ):
            assert is_registered(
                KernelKey(
                    "hip_gfx1151",
                    layer,
                    "gguf_f32_weight",
                    variant,
                )
            )


def test_gfx1151_backend_aliases_gfx1100_kernel_keys() -> None:
    register_qwen35_rmsnorm_kernels()
    register_gfx1151_kernels()

    assert TARGET_ARCH == "gfx1151"
    assert GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS == 128
    assert GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY == {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {2: 23},
    }
    assert LAGUNA_F16_PREFILL_STRATEGY == "wmma_comp_swa"
    assert LAGUNA_F16_PREFILL_MIN_ROWS == 16
    assert LAGUNA_F16_PREFILL_MODE == "hipblaslt_range_direct"
    assert LAGUNA_F16_BOUNDARY_FUSION is True
    assert LAGUNA_F16_ATTENTION_QUAD_DECODE is True
    assert LAGUNA_F16_NONTEMPORAL_DECODE is True
    assert LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE is True
    assert LAGUNA_F16_PROJECTION_HEAD_KV_DECODE is True
    assert LAGUNA_F16_DECODE_FIXEDK is True
    assert LAGUNA_F16_DECODE_ONEBARRIER is True
    assert LAGUNA_Q4_PACK8_DUAL_SILU_DECODE is True
    assert LAGUNA_SELECTED_NATURAL_DECODE is True
    assert LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE is True
    assert LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE is True
    assert LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE is True
    assert LAGUNA_SELECTED_NATURAL_TILE8_DECODE is True
    assert LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_DECODE is True
    assert LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_SILU_DECODE is True
    assert LAGUNA_SELECTED_HALFDOT_DECODE is True
    assert LAGUNA_DENSE_Q4_PREFILL_MODE == "wmma_pack8"
    assert LAGUNA_MOE_BRANCH_CONCURRENCY is True
    assert LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY is True
    assert LAGUNA_MOE_SHARED_AFTER_ROUTER is True
    assert LAGUNA_MOE_SHARED_LOW_PRIORITY is True
    assert LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY is True
    assert LAGUNA_PREFILL_ATTENTION_HIPBLASLT is True
    assert LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE is True
    assert LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER is True
    assert LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES is True
    assert LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX is True
    assert LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT is True
    assert LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE is True
    assert LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS == 2_048
    assert LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT is True
    assert LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT is True
    assert LAGUNA_Q6_WMMA_PREFETCH_WEIGHT is True
    assert LAGUNA_Q6_WMMA_PREFETCH_ACTIVATION is True
    assert LAGUNA_MOE_GROUP_COMPACT_MODE == "parallel"
    assert LAGUNA_ROUTER_LOGITS_MODE == "token_tile_8"
    assert (
        LAGUNA_SELECTED_GATE_UP_MODE
        == "mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512"
    )
    assert (
        LAGUNA_SELECTED_DOWN_MODE
        == "mmq64x64_d4_f32_q6_wavecols_direct_rawprefetch_q4_ge512"
    )
    assert LAGUNA_PREFILL_MATRIX_ROWS == 2048
    assert LAGUNA_PREFILL_CACHED_META is True
    assert LAGUNA_PREFILL_GLOBAL_QROW6 is True
    assert LAGUNA_PREFILL_KV_PREAPPEND is True
    assert (
        LAGUNA_GLOBAL_PREFILL_VARIANT
        == "global_context_rows_qrow4_m128_online_spans"
    )
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_LAYER == 28
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH
        == 16
    )
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DIM_TILE == 64
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DEFERREDNORM is True
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE
        is True
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH
        == 4
    )
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_SCORE is True
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LAYER == 32
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LIVE == 98_304
    assert LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_TOKENLOOP4 is True
    assert LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE64 is True
    assert LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80 is True
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX
        is True
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_MIN_LIVE
        == 65_536
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE
        is True
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH8_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE
        is True
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE
        is True
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE
        is True
    )
    assert (
        LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PROBABILITY_VEC4_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE
        is True
    )
    assert LAGUNA_SWA_PREFILL_VARIANT == "swa_context_rows_qrow4_m128_online_spans"
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_PREFILL_STRATEGY"
        )
        == "wmma_comp_swa"
    )
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_PREFILL_MIN_ROWS"
        )
        == 16
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_PREFILL_STRATEGY", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_BOUNDARY_FUSION", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_BOUNDARY_FUSION", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_DECODE_FIXEDK", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_DECODE_FIXEDK", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_ATTENTION_QUAD_DECODE", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_ATTENTION_QUAD_DECODE", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_NONTEMPORAL_DECODE", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_NONTEMPORAL_DECODE", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_PROJECTION_HEAD_KV_DECODE", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_PROJECTION_HEAD_KV_DECODE", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_SELECTED_NATURAL_DECODE", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_SELECTED_NATURAL_DECODE", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE",
            None,
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE",
            None,
        )
        is None
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE",
            None,
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE",
            None,
        )
        is None
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE",
            None,
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE",
            None,
        )
        is None
    )
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_F16_DECODE_ONEBARRIER", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_F16_DECODE_ONEBARRIER", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_MOE_BRANCH_CONCURRENCY", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_MOE_BRANCH_CONCURRENCY", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_MOE_SHARED_AFTER_ROUTER", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_MOE_SHARED_AFTER_ROUTER", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_MOE_SHARED_LOW_PRIORITY", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_MOE_SHARED_LOW_PRIORITY", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_PREFILL_ATTENTION_HIPBLASLT", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_PREFILL_ATTENTION_HIPBLASLT", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS",
            None,
        )
        == 2_048
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT",
            None,
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT",
        None,
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_PREFILL_MATRIX_ROWS", None
        )
        == 2048
    )
    assert (
        backend_package_capability(
            "hip_gfx1100", "LAGUNA_PREFILL_MATRIX_ROWS", None
        )
        == 512
    )
    assert (
        backend_package_capability(
            "hip_gfx1100", "LAGUNA_SELECTED_GATE_UP_MODE", None
        )
        == "grouped_pair16"
    )
    assert (
        backend_package_capability(
            "hip_gfx1100", "LAGUNA_SELECTED_DOWN_MODE", None
        )
        == "grouped_exact"
    )
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_PREFILL_CACHED_META", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_PREFILL_CACHED_META", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_PREFILL_GLOBAL_QROW6", None
        )
        is True
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_PREFILL_GLOBAL_QROW6", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_PREFILL_KV_PREAPPEND", None
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100", "LAGUNA_PREFILL_KV_PREAPPEND", None
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_GLOBAL_PREFILL_VARIANT", None
        )
        == "global_context_rows_qrow4_m128_online_spans"
    )
    assert backend_package_capability(
        "hip_gfx1100", "LAGUNA_GLOBAL_PREFILL_VARIANT", None
    ) is None
    assert (
        backend_package_capability(
            "hip_gfx1151", "LAGUNA_SWA_PREFILL_VARIANT", None
        )
        == "swa_context_rows_qrow4_m128_online_spans"
    )
    assert (
        backend_package_capability(
            "hip_gfx1100", "LAGUNA_SWA_PREFILL_VARIANT", None
        )
        == "swa_context_rows_qrow4_m128_c256_exact_spans"
        == GFX1100_LAGUNA_SWA_PREFILL_VARIANT
    )
    assert GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS == 4096
    assert GFX1100_GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS == 4096
    assert GGUF_PREFILL_ROUTER_SELECT_THREADS == 128
    assert GFX1100_GGUF_PREFILL_ROUTER_SELECT_THREADS == 128
    assert GGUF_PREFILL_SCRATCH_ARENA_GROUPING == "owner_slots"
    assert GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS is True
    assert GGUF_PREFILL_SCRATCH_LIVENESS_MIN_ROWS == 768
    assert GGUF_Q8_T16_PREFILL_FOUR_WAVE is True
    assert GGUF_Q8_T16_PREFILL_TWO_WAVE is True
    assert GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS == 65536
    assert GGUF_ROUTER_F32_BF16_HIDDEN_THREADS == 256
    assert GFX1100_GGUF_ROUTER_F32_BF16_HIDDEN_THREADS == 256
    assert GFX1100_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS == 4096
    assert GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS == 4096
    assert GFX1100_GGUF_GDN_INDEXED_SINGLETON_DECODE is False
    assert GGUF_GDN_INDEXED_SINGLETON_DECODE is True
    assert GFX1100_GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS == 0
    assert GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS == 8
    assert GFX1100_GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS == 0
    assert GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS == 8
    assert GFX1100_GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS == 0
    assert GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS == 8
    assert GFX1100_GGUF_Q5_T16_SELECTED_QWEN_TILE8 is False
    assert GGUF_Q5_T16_SELECTED_QWEN_TILE8 is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE", None
    ) == {}
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE", None
    ) == {
        "gguf_q4_k_t16_v1": {
            (5_120, 1_024): "dense_single_col4_bf16_bf16_out",
        },
        "gguf_q5_k_t16_v1": {
            (6_144, 5_120): "t16_gemv_decode_tile8_bf16_bf16_out",
        },
    }
    assert GFX1100_GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS == 0
    assert GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS == 8
    assert GFX1100_GGUF_Q6_LM_HEAD_MAX_CHUNK == 8
    assert GGUF_Q6_LM_HEAD_MAX_CHUNK == 8
    assert GFX1100_GGUF_Q8_T16_DECODE_ROWTILE_ALL is False
    assert GGUF_Q8_T16_DECODE_ROWTILE_ALL is False
    assert GFX1100_GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS == 0
    assert GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS == 4
    assert GFX1100_GGUF_GDN_PREFILL_AUTO_MODE == "chain_compact_peer_wave32"
    assert GFX1100_GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE == {}
    assert GFX1100_GGUF_GDN_PREFILL_EXACT_MODE == "chain_lds32_direct_nonvolatile"
    assert GGUF_GDN_PREFILL_AUTO_MODE == "chain_lds32_direct_nonvolatile"
    assert GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE == {
        ("MOSTLY_Q4_K_M", 16, 16, 128, 128): "chain_peer_cluster8",
        ("MOSTLY_Q4_K_M", 16, 48, 128, 128): "chain_compact_peer_wave32",
        ("MOSTLY_Q4_K_S", 16, 48, 128, 128): "chain_compact_peer_wave32",
        # D08-X2-K2: fresh five-block gate admitted Q8_0 after P2's 0.0108%
        # rejection was superseded by exact-core graph-decode evidence.
        ("MOSTLY_Q8_0", 16, 16, 128, 128): "chain_peer_cluster8",
    }
    assert GGUF_GDN_PREFILL_COMPACT_PEER_CHUNK_ROWS == 1024
    assert GGUF_GDN_PREFILL_EXACT_MODE == "chain_lds32_direct_nonvolatile"
    assert GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE is True
    assert GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT == 32768
    assert GGUF_PAGED_ATTN_PARALLEL_REDUCE is True
    assert GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT == 32768
    assert GFX1100_GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE == "shared_x"
    assert GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE == "baseline"
    assert GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE is True
    assert GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS == 4096
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_PREFILL_ROUTER_SELECT_THREADS",
        )
        == 128
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
        )
        == 65536
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "LAGUNA_Q4_SHARED_DOWN_T16_DECODE",
        )
        is True
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer=(
                "attention_projection+head_rmsnorm+partial_rotary+kv_write"
            ),
            quant="fp16_weight+laguna_f32_weight",
            variant="global_fixedk_nontemporal_bf16_f32_spans",
        )
        is laguna_global_f16_projection_head_kv_nontemporal_tile2_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer=(
                "attention_projection+head_rmsnorm+partial_rotary+kv_write"
            ),
            quant="fp16_weight+laguna_f32_weight",
            variant="swa_fixedk_nontemporal_bf16_f32_spans",
        )
        is laguna_swa_f16_projection_head_kv_nontemporal_tile2_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q4_k",
            variant="pack8_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="pack8_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear_pair",
            quant="gguf_q8_0_t16_v1",
            variant="t16_dual_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q6_k",
            variant="wmma_prefill_bf16_bf16_out",
        )
        is gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q6_k",
            variant="wmma_prefill_bf16_bf16_out",
        )
        is gguf_q6_k_wmma_prefill_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="router_logits",
            quant="f32",
            variant="bf16_hidden",
        )
        is qwen35_router_logits_bf16_f32w_auto_256
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_logits",
            quant="f32",
            variant="bf16_hidden",
        )
        is qwen35_router_logits_bf16_f32w_auto_256
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
        )
        == 128
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES",
    ) == {
        (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M"): {
            "transport": "hipgraph"
        }
    }
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_GDN_INDEXED_SINGLETON_DECODE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_GDN_INDEXED_SINGLETON_DECODE",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q6_LM_HEAD_MAX_CHUNK",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q6_LM_HEAD_MAX_CHUNK",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS",
        )
        == 4
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_GDN_PREFILL_AUTO_MODE",
        )
        == "chain_compact_peer_wave32"
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
        )
        == "shared_x"
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
        )
        == 32768
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
        )
        is True
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_bf16_parallel_reduce_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_RAW_K_PREFILL_ROWBATCH",
        )
        == 32
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_RAW_K_PREFILL_VARIANT",
        )
        == "coltile"
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    ) == frozenset(
        {
            ("gguf_q5_k", "bf16_bf16_out", 3072, 12288),
            ("gguf_q5_k", "bf16_f32_out", 3072, 6144),
            ("gguf_q5_k", "bf16_f32_out", 3072, 9216),
            ("gguf_q6_k", "bf16_f32_out", 3072, 9216),
        }
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_RAW_K_PREFILL_ROWBATCH",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_RAW_K_PREFILL_VARIANT",
        )
        == "rowbatch"
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
        )
        == frozenset()
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_col4_bf16_bf16_out",
        )
    )
    for quant in ("gguf_q5_k", "gguf_q6_k"):
        for row_batch in (4, 8):
            for output_dtype in ("bf16", "f32"):
                assert not is_registered(
                    KernelKey(
                        "hip_gfx1151",
                        "linear",
                        quant,
                        f"rowbatch{row_batch}_bf16_{output_dtype}_out",
                    )
                )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_GDN_PREFILL_AUTO_MODE",
        )
        == "chain_lds32_direct_nonvolatile"
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
        )
        == "baseline"
    )
    assert hip_target_arch_for_backend("hip_gfx1151") == "gfx1151"
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="rmsnorm",
            quant="w4_paro",
            variant="paro_out_fp16",
        )
        is paro_rmsnorm_out_fp16
    )


def test_plan_hip_build_target_arch_is_in_flags_and_cache_key(tmp_path: Path) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')

    gfx1100 = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1100",
    )
    gfx1151 = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1151",
    )

    assert gfx1100.cache_key != gfx1151.cache_key
    assert gfx1100.target_arch == "gfx1100"
    assert gfx1151.target_arch == "gfx1151"
    assert "--offload-arch=gfx1100" in gfx1100.flags
    assert "--offload-arch=gfx1151" in gfx1151.flags
    assert "--offload-arch=gfx1151" in gfx1151.command


def test_plan_hip_build_reads_target_arch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )

    assert artifact.target_arch == "gfx1151"
    assert "--offload-arch=gfx1151" in artifact.flags


def test_plan_hip_build_includes_device_lib_path_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')
    device_lib_path = tmp_path / "amdgcn" / "bitcode"
    monkeypatch.setenv("HIP_DEVICE_LIB_PATH", str(device_lib_path))

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1151",
    )

    assert f"--rocm-device-lib-path={device_lib_path}" in artifact.flags
    assert f"--rocm-device-lib-path={device_lib_path}" in artifact.command


def test_qwen35_paro_gfx1151_generation_factory_sets_backend() -> None:
    register_builtin_generators()
    factory = resolve_text_generator(
        model="qwen3_5_moe_paro",
        backend="hip_gfx1151",
        quant="w4_paro",
    )

    generator = factory(model_path="/tmp/fake", weight_index=object(), model_plugin=object())

    assert getattr(generator, "backend") == "hip_gfx1151"


def test_qwen35_gguf_gfx1151_generation_factory_sets_backend(monkeypatch) -> None:
    import hipengine.generation.qwen35_gguf as qwen35_gguf

    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFTokenizer,
        "from_gguf_info",
        classmethod(lambda cls, weight_index: object()),
    )
    register_builtin_generators()
    factory = resolve_text_generator(
        model="qwen3_5_moe_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    )

    generator = factory(
        model_path="/tmp/fake.gguf",
        weight_index=object(),
        model_plugin=object(),
    )

    assert getattr(generator, "backend") == "hip_gfx1151"
    assert generator.target_arch == "gfx1151"
    assert generator.engine_loop_config_defaults == {
        "prefill_decode_policy": "fair",
        "max_prefill_chunk_tokens": 256,
        "fair_prefill_burst_chunks": 2,
    }
    assert generator.server_plain_ar_max_active_requests == 8

    for quant in ("gguf_q4_k_s", "gguf_q8_0"):
        other_quant_factory = resolve_text_generator(
            model="qwen3_5_moe_gguf",
            backend="hip_gfx1151",
            quant=quant,
        )
        other_quant_generator = other_quant_factory(
            model_path=f"/tmp/fake-{quant}.gguf",
            weight_index=object(),
            model_plugin=object(),
        )
        assert other_quant_generator.engine_loop_config_defaults == {}
        assert other_quant_generator.server_plain_ar_max_active_requests is None


def test_gguf_weight_backend_drives_embedding_and_linear_dispatch() -> None:
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_BF16
    from hipengine.runtime.gguf_embedding import resolve_gguf_embedding_dispatch
    from hipengine.runtime.gguf_linear import resolve_gguf_linear_dispatch

    weight = SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(layout=LAYOUT_DENSE_BF16, quant_key="bf16"),
    )

    assert resolve_gguf_embedding_dispatch(weight).key.backend == "hip_gfx1151"
    assert resolve_gguf_linear_dispatch(weight).key.backend == "hip_gfx1151"


def test_gguf_router_resolve_uses_weight_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    resolved: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, **kwargs):
        del layer, quant, variant, kwargs
        resolved.append(backend)
        return lambda *args, **launch_kwargs: None

    monkeypatch.setattr(qwen35_gguf_runner, "resolve", fake_resolve)
    weight = SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(quant_key="f32"),
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=22)),
    )

    qwen35_gguf_runner._launch_qwen35_router_logits_bf16_hidden(
        11,
        weight,
        33,
        1,
        2048,
        256,
    )

    assert resolved == ["hip_gfx1151"]


def test_gguf_gdn_plan_resolves_every_key_for_runner_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    resolved: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, **kwargs):
        del layer, quant, variant, kwargs
        resolved.append(backend)
        return object()

    monkeypatch.setattr(qwen35_gguf_runner, "resolve", fake_resolve)
    runner = object.__new__(qwen35_gguf_runner.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1151"

    plan = runner._gdn_prefill_plan()

    assert plan.has_chain
    assert plan.has_exact_chain
    assert plan.has_fused
    assert len(resolved) == 35
    assert set(resolved) == {"hip_gfx1151"}


def test_gguf_runner_loads_backend_aliases_and_tags_resident_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    loaded: list[str] = []
    materialized: list[str] = []
    fake_weights = object()
    monkeypatch.setattr(
        qwen35_gguf_runner,
        "load_backend_kernel_package",
        lambda backend: loaded.append(backend),
    )

    def fake_materialize(model_path, *, runtime, backend):
        del model_path, runtime
        materialized.append(backend)
        return fake_weights

    monkeypatch.setattr(
        qwen35_gguf_runner,
        "materialize_qwen35_gguf_weights",
        fake_materialize,
    )

    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    runner = qwen35_gguf_runner.Qwen35GGUFFullStackRunner(
        "/tmp/fake.gguf",
        runtime=object(),
        backend="hip_gfx1151",
    )
    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE")

    assert runner.fp16_recurrent_state is True
    assert runner.backend == "hip_gfx1151"
    assert runner.target_arch == "gfx1151"
    assert runner.weights is fake_weights
    assert loaded == ["hip_gfx1151"]
    assert materialized == ["hip_gfx1151"]


def test_gguf_fused_linear_matching_uses_resident_backend() -> None:
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
    from hipengine.runtime.gguf_linear import _resolve_gguf_linear_pair_kind

    def weight():
        return SimpleNamespace(
            backend="hip_gfx1151",
            spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0"),
        )

    assert (
        _resolve_gguf_linear_pair_kind(
            weight(),
            weight(),
            rows=1,
            in_features=2048,
            out_features=4096,
            out_features_b=4096,
            activation_dtype="bf16",
            output_dtype="bf16",
            backend="hip_gfx1151",
            use_wmma=False,
            use_gemv=False,
            registered_decode_only=False,
        )
        == "q8_raw_dual"
    )


def test_gguf_runtime_has_no_literal_gfx1100_resolver_backend() -> None:
    import hipengine.runtime.gguf_embedding as gguf_embedding
    import hipengine.runtime.gguf_linear as gguf_linear
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    resolver_names = {
        "resolve",
        "resolve_gguf_embedding_dispatch",
        "resolve_gguf_linear_dispatch",
    }
    violations: list[str] = []
    for module in (gguf_embedding, gguf_linear, qwen35_gguf_runner):
        tree = ast.parse(inspect.getsource(module))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name):
                name = call.func.id
            elif isinstance(call.func, ast.Attribute):
                name = call.func.attr
            else:
                continue
            if name not in resolver_names:
                continue
            for keyword in call.keywords:
                if (
                    keyword.arg == "backend"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "hip_gfx1100"
                ):
                    violations.append(f"{module.__name__}:{call.lineno}:{name}")

    assert violations == []


def test_gfx1151_gguf_lazy_registration_rebinds_source_kernels() -> None:
    from hipengine.kernels.registry import (
        KernelKey,
        clear_registry_for_tests,
        is_registered,
        register,
        resolve,
    )
    from hipengine.runtime.gguf_embedding import _ensure_embedding_kernel_registered
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    embedding_key = KernelKey(
        "hip_gfx1151",
        "embedding",
        "gguf_q6_k",
        "lookup_bf16_out",
    )
    linear_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q6_k_t16_v1",
        "t16_gemv_decode_bf16_f32_out",
    )
    clear_registry_for_tests()
    register(KernelKey("cpu_reference", "embedding", "fp16"), lambda *args: args)
    register(KernelKey("cpu_reference", "linear", "fp16"), lambda *args: args)

    _ensure_embedding_kernel_registered(embedding_key)
    _ensure_linear_kernel_registered(linear_key)

    assert is_registered(embedding_key)
    assert is_registered(linear_key)
    assert callable(
        resolve(
            backend=embedding_key.backend,
            layer=embedding_key.layer,
            quant=embedding_key.quant,
            variant=embedding_key.variant,
        )
    )
    assert callable(
        resolve(
            backend=linear_key.backend,
            layer=linear_key.layer,
            quant=linear_key.quant,
            variant=linear_key.variant,
        )
    )
