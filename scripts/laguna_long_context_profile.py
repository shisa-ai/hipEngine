#!/usr/bin/env python3
"""Profile Laguna prefill and fixed-horizon decode at retained row lengths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LENGTHS = (512, 1024, 4096)
LAP0_LENGTHS = (128, 512, 1024, 4096)
ATTACK_LENGTHS = (4096, 16384, 65536, 131072)
ATTACK_DIRECTIONAL_LENGTHS = (4096, 16384, 65536)
LC1_128K_GATE_LENGTHS = (131072,)
LC0_TRACE_LENGTHS = (16384, 65536)
LC_D2_DECODE_TRACE_LENGTHS = (16384,)
FINAL_SWEEP_LENGTHS = (512, 1024, 4096, 32768, 65536, 131072)
MATCHED_LONG_SWEEP_LENGTHS = (1024, 4096, 16384, 32768, 65536, 131072)
STANDARD_DECODE_LENGTHS = (512,)
DECODE_OUTPUT_TOKENS = (1, 128)
PROFILE_LENGTH_SETS = (
    DEFAULT_LENGTHS,
    LAP0_LENGTHS,
    ATTACK_LENGTHS,
    ATTACK_DIRECTIONAL_LENGTHS,
    LC1_128K_GATE_LENGTHS,
    LC0_TRACE_LENGTHS,
    LC_D2_DECODE_TRACE_LENGTHS,
    FINAL_SWEEP_LENGTHS,
    MATCHED_LONG_SWEEP_LENGTHS,
    STANDARD_DECODE_LENGTHS,
)
DEFAULT_CHUNK_SIZE = 128
PROFILE_CHUNK_SIZES = (128, 256, 512, 1024, 2048)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-long-context-profile.json"
)
COMPARISON_ARGUMENTS = (
    "compare_long_attention_hipblaslt",
    "compare_block_attention_hipblaslt",
    "compare_dense_contiguous_cache",
    "compare_swa_attention_hipblaslt",
    "compare_f16_decode_onebarrier",
    "compare_f16_decode_fixedk",
    "compare_swa_fixed512_reduce",
    "compare_swa_fused_fixed512",
    "compare_swa_gqa3_local384",
    "compare_swa_gqa3_vstage64",
    "compare_swa_gqa3_vstage64_vec16",
    "compare_swa_gqa3_vstage64_vec16_direct",
    "compare_swa_assume_exp",
    "compare_swa_mixed32",
    "compare_swa_mixed32_exp4",
    "compare_swa_mixed32_exp8",
    "compare_swa_mixed32_exp16",
    "compare_swa_mixed32_exp32",
    "compare_global_fixedshape_reduce",
    "compare_global_fused_fixedshape",
    "compare_global_gqa2_vstage64",
    "compare_global_gqa2_vstage64_vec16",
    "compare_global_gqa2_vstage64_vec16_direct",
    "compare_global_assume_exp",
    "compare_global_exp32",
    "compare_global_mixed32",
    "compare_selected_natural_decode",
    "compare_selected_natural_tile8_decode",
    "compare_q4_decode_t16_sidecar",
    "compare_q4_decode_t16_dual_interleaved",
)


def _parse_chunk_size(value: str | int) -> int:
    chunk_size = int(value)
    if chunk_size not in PROFILE_CHUNK_SIZES:
        raise argparse.ArgumentTypeError(
            "matrix chunk size must be 128, 256, 512, 1024, or 2048"
        )
    return chunk_size


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or any(item <= 0 for item in lengths):
        raise argparse.ArgumentTypeError("context lengths must be positive integers")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("context lengths must be distinct")
    return lengths


def _parse_decode_output_tokens(value: str | int) -> int:
    output_tokens = int(value)
    if output_tokens not in DECODE_OUTPUT_TOKENS:
        raise argparse.ArgumentTypeError("decode output tokens must be 1 or 128")
    return output_tokens


def _active_comparison_count(args: argparse.Namespace) -> int:
    return sum(bool(getattr(args, name)) for name in COMPARISON_ARGUMENTS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=max(DEFAULT_LENGTHS))
    parser.add_argument("--lengths", type=_parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--chunk-size", type=_parse_chunk_size, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--attention-rows",
        type=_parse_chunk_size,
        help="global-attention query rows; SWA remains capped at 128",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-rows", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--decode-output-tokens",
        type=_parse_decode_output_tokens,
        default=1,
        help="total output horizon including the synchronized first prefill token",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--long-attention-hipblaslt",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compare-long-attention-hipblaslt",
        action="store_true",
    )
    parser.add_argument(
        "--compare-block-attention-hipblaslt",
        action="store_true",
    )
    parser.add_argument(
        "--block-attention-hipblaslt",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compare-dense-contiguous-cache",
        action="store_true",
    )
    parser.add_argument(
        "--dense-contiguous-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compare-swa-attention-hipblaslt",
        action="store_true",
    )
    parser.add_argument(
        "--swa-attention-hipblaslt",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--moe-branch-concurrency",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--q6-qmicro-permute",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--q6-qmicro-planar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--moe-shared-after-router",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--moe-shared-low-priority",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--head-kv-fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="select the exact decode-only head-RMSNorm+RoPE+KV-write composite",
    )
    parser.add_argument(
        "--moe-tail-next-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="select the exact c=1 MoE-tail+next-RMSNorm composite",
    )
    parser.add_argument(
        "--decode-split-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="screen the exact gfx11 global/SWA split-attention bundle",
    )
    parser.add_argument(
        "--compare-f16-decode-onebarrier",
        action="store_true",
        help="counterbalance exact source-F16 GEMV against the one-barrier sibling",
    )
    parser.add_argument(
        "--compare-f16-decode-fixedk",
        action="store_true",
        help="counterbalance one-barrier source-F16 decode against fixed-K",
    )
    parser.add_argument(
        "--compare-swa-fixed512-reduce",
        action="store_true",
        help="counterbalance exact SWA GQA3 against its saturated-512 reducer",
    )
    parser.add_argument(
        "--compare-swa-fused-fixed512",
        action="store_true",
        help="counterbalance saturated split SWA against exact fused attention",
    )
    parser.add_argument(
        "--compare-swa-gqa3-local384",
        action="store_true",
        help="counterbalance exact fused GQA2 against local384 GQA3",
    )
    parser.add_argument(
        "--compare-swa-gqa3-vstage64",
        action="store_true",
        help="counterbalance exact local384 GQA3 against exact 64-slot V staging",
    )
    parser.add_argument(
        "--compare-swa-gqa3-vstage64-vec16",
        action="store_true",
        help="counterbalance scalar and 16-byte copies in exact staged-V GQA3",
    )
    parser.add_argument(
        "--compare-swa-gqa3-vstage64-vec16-direct",
        action="store_true",
        help="counterbalance retained vec16 copies against direct LDS stores",
    )
    parser.add_argument(
        "--compare-swa-assume-exp",
        action="store_true",
        help="counterbalance exact generic-domain and softmax-domain expf",
    )
    parser.add_argument(
        "--compare-swa-mixed32",
        action="store_true",
        help="counterbalance exact retained GQA3 against mixed 32-block SWA",
    )
    parser.add_argument(
        "--compare-swa-mixed32-exp4",
        action="store_true",
        help="counterbalance retained mixed32 against exact four-lane expf",
    )
    parser.add_argument(
        "--compare-swa-mixed32-exp8",
        action="store_true",
        help="counterbalance retained exp4 against exact eight-lane expf",
    )
    parser.add_argument(
        "--compare-swa-mixed32-exp16",
        action="store_true",
        help="counterbalance retained exp8 against exact sixteen-lane expf",
    )
    parser.add_argument(
        "--compare-swa-mixed32-exp32",
        action="store_true",
        help="counterbalance retained exp16 against exact wave-wide expf",
    )
    parser.add_argument(
        "--compare-global-fixedshape-reduce",
        action="store_true",
        help="counterbalance exact global reduction against its natural shape",
    )
    parser.add_argument(
        "--compare-global-fused-fixedshape",
        action="store_true",
        help="counterbalance fixed-shape global reduction against one-head fusion",
    )
    parser.add_argument(
        "--compare-global-gqa2-vstage64",
        action="store_true",
        help="counterbalance fused global GQA1 against GQA2 with staged V reuse",
    )
    parser.add_argument(
        "--compare-global-gqa2-vstage64-vec16",
        action="store_true",
        help="counterbalance scalar and 16-byte copies in global staged V",
    )
    parser.add_argument(
        "--compare-global-gqa2-vstage64-vec16-direct",
        action="store_true",
        help="counterbalance global vec16 copies against direct LDS stores",
    )
    parser.add_argument(
        "--compare-global-assume-exp",
        action="store_true",
        help="counterbalance exact global generic-domain and softmax-domain expf",
    )
    parser.add_argument(
        "--compare-global-exp32",
        action="store_true",
        help="counterbalance serial and exact wave-wide global expf issue",
    )
    parser.add_argument(
        "--compare-global-mixed32",
        action="store_true",
        help="counterbalance exact 24-owner GQA2 and 32-owner mixed global attention",
    )
    parser.add_argument(
        "--compare-selected-natural-decode",
        action="store_true",
        help="counterbalance selected-MoE decode against natural-shape siblings",
    )
    parser.add_argument(
        "--compare-selected-natural-tile8-decode",
        action="store_true",
        help="counterbalance natural selected gate/up against exact tile8",
    )
    parser.add_argument(
        "--compare-q4-decode-t16-sidecar",
        action="store_true",
        help="counterbalance pack8 and compact T16 dense/shared Q4 decode",
    )
    parser.add_argument(
        "--compare-q4-decode-t16-dual-interleaved",
        action="store_true",
        help="counterbalance separate and paired T16 dense/shared Q4 decode",
    )
    parser.add_argument(
        "--ordinary-q4-expert-t16",
        action="store_true",
        help="materialize ordinary two-buffer expert T16 as a rollback control",
    )
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _timing_order(lengths: Sequence[int], repetition: int) -> tuple[int, ...]:
    ordered = tuple(int(value) for value in lengths)
    return ordered if int(repetition) % 2 == 0 else tuple(reversed(ordered))


def _summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("long-context summary requires at least one sample")
    lengths = {int(sample["length"]) for sample in samples}
    if len(lengths) != 1:
        raise ValueError("long-context summary samples must have one length")
    seconds = [float(sample["prefill_seconds"]) for sample in samples]
    if any(not math.isfinite(value) or value <= 0.0 for value in seconds):
        raise ValueError("long-context timing samples must be finite and positive")
    length = lengths.pop()
    median = statistics.median(seconds)
    summary = {
        "length": length,
        "samples_seconds": seconds,
        "median_seconds": median,
        "median_tok_s": length / median,
        "next_token_ids": [int(sample["next_token_id"]) for sample in samples],
        "repeat_deterministic": len({int(sample["next_token_id"]) for sample in samples}) == 1,
    }
    output_tokens = {int(sample.get("output_tokens", 1)) for sample in samples}
    if len(output_tokens) != 1:
        raise ValueError("long-context summary samples must have one output horizon")
    output_horizon = output_tokens.pop()
    summary["output_tokens"] = output_horizon
    if output_horizon > 1:
        decode_seconds = [float(sample["decode_seconds"]) for sample in samples]
        if any(not math.isfinite(value) or value <= 0.0 for value in decode_seconds):
            raise ValueError("long-context decode samples must be finite and positive")
        decode_calls = output_horizon - 1
        decode_median = statistics.median(decode_seconds)
        final_token_ids = [int(sample["final_token_id"]) for sample in samples]
        generated_hashes = [str(sample["generated_ids_sha256"]) for sample in samples]
        summary.update(
            {
                "decode_forward_calls": decode_calls,
                "decode_samples_seconds": decode_seconds,
                "decode_median_seconds": decode_median,
                "decode_median_tok_s": decode_calls / decode_median,
                "final_token_ids": final_token_ids,
                "generated_ids_sha256": generated_hashes,
                "repeat_generated_ids_deterministic": len(set(generated_hashes)) == 1,
            }
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    lengths = tuple(int(value) for value in args.lengths)
    if lengths not in PROFILE_LENGTH_SETS:
        raise ValueError(
            f"retained Laguna profiling requires one of {PROFILE_LENGTH_SETS}"
        )
    if args.chunk_size not in PROFILE_CHUNK_SIZES:
        raise ValueError(f"Laguna profiling chunk size must be one of {PROFILE_CHUNK_SIZES}")
    output_tokens = int(args.decode_output_tokens)
    required_context = max(lengths) + output_tokens - 1
    if args.context_length < required_context:
        raise ValueError(
            "largest Laguna prompt plus output horizon exceeds admitted context"
        )
    if args.repetitions <= 0:
        raise ValueError("LPF-5 repetitions must be positive")
    if args.warmup_rows <= 0 or args.warmup_rows > args.chunk_size:
        raise ValueError("LPF-5 warmup rows must fit one retained chunk")
    if (
        args.compare_long_attention_hipblaslt
        and args.long_attention_hipblaslt
    ):
        raise ValueError(
            "--compare-long-attention-hipblaslt and "
            "--long-attention-hipblaslt are mutually exclusive"
        )
    active_comparisons = _active_comparison_count(args)
    if active_comparisons > 1:
        raise ValueError("only one Laguna comparison may be active")
    if (
        args.compare_block_attention_hipblaslt
        and args.block_attention_hipblaslt
    ):
        raise ValueError(
            "--compare-block-attention-hipblaslt and "
            "--block-attention-hipblaslt are mutually exclusive"
        )
    if (
        args.compare_swa_attention_hipblaslt
        and args.swa_attention_hipblaslt
    ):
        raise ValueError(
            "--compare-swa-attention-hipblaslt and "
            "--swa-attention-hipblaslt are mutually exclusive"
        )
    if args.compare_dense_contiguous_cache and args.dense_contiguous_cache:
        raise ValueError(
            "--compare-dense-contiguous-cache and "
            "--dense-contiguous-cache are mutually exclusive"
        )
    comparison = active_comparisons > 0
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"] and not args.allow_dirty:
        raise RuntimeError("retained Laguna LPF-5 profiling requires a clean tracked worktree")

    # Runtime initialization applies backend process defaults such as
    # GPU_MAX_HW_QUEUES before libamdhip64 is loaded. Do it before provenance
    # capture so the artifact records the policy the measurement actually used.
    runtime = get_hip_runtime()
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile=(
            f"laguna_prefill_long_context_matrix{args.chunk_size}"
            if output_tokens == 1
            else f"laguna_p512_d{output_tokens}_matrix{args.chunk_size}"
        ),
        timing_protocol=(
            (
                "prefill_only_"
                + "_".join(str(length) for length in lengths)
                + f"_matrix{args.chunk_size}_attention128"
            )
            if output_tokens == 1
            else (
                f"p{lengths[0]}_d{output_tokens}_eager_c1_"
                f"matrix{args.chunk_size}_attention128"
            )
        ),
        warmups=1,
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(lengths))

    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    active_moe_branch_concurrency = False
    active_q6_qmicro_permute = False
    active_q6_qmicro_planar = False
    active_moe_shared_after_router = False
    active_moe_shared_low_priority = False
    active_moe_shared_priority_range: tuple[int, int] | None = None
    active_moe_tail_next_rmsnorm = False
    active_head_kv_fusion = False
    active_global_split_min_live: int | None = None
    active_swa_split_min_live: int | None = None
    active_swa_split_tile16_min_live: int | None = None
    active_split_gate_fusion = False
    active_swa_split_wave_local = False
    active_swa_split_fixed512_reduce = False
    active_swa_fused_fixed512 = False
    active_global_split_fixedshape_reduce = False
    active_global_split_gqa6_dim32_vstage64 = False
    active_global_split_gqa6_deferrednorm_dim32_vstage64 = False
    active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage64 = False
    active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80 = False
    active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix = False
    active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_min_live: int | None = None
    active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value = False
    active_global_split_gqa6_ctx4096_compensated_layer: int | None = None
    active_global_split_gqa6_ctx4096_dim_tile = 32
    active_global_split_gqa6_ctx4096_deferrednorm = False
    active_global_split_gqa6_ctx4096_tokenloop4 = False
    active_global_split_gqa6_ctx4096_min_live = 6_001
    active_global_split_gqa6_ctx4096_min_layer: int | None = None
    active_global_fused_fixedshape = False
    active_global_gqa2_vstage64_fixedshape = False
    active_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape = (
        False
    )
    active_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape = (
        False
    )
    active_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape = (
        False
    )
    active_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape = (
        False
    )
    active_global_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape = (
        False
    )
    active_global_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape = (
        False
    )
    active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
        False
    )
    active_long_attention_hipblaslt = False
    active_block_attention_hipblaslt = False
    active_swa_attention_hipblaslt = False
    active_dense_contiguous_cache = False
    active_q4_decode_t16_sidecar = False
    active_q4_decode_t16_dual_interleaved = False
    active_q4_shared_down_t16_decode = False
    active_q4_expert_t16_dual_interleaved = False
    active_attention_rows = 128
    active_global_attention_rows = 128
    rows: list[dict[str, Any]] = []
    original_f16_decode_mode = os.environ.get("HIPENGINE_LAGUNA_F16_DECODE")
    if args.compare_f16_decode_onebarrier:
        os.environ["HIPENGINE_LAGUNA_F16_DECODE"] = "gemv"
    if args.compare_f16_decode_fixedk:
        os.environ["HIPENGINE_LAGUNA_F16_DECODE"] = "onebarrier"
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
            prefill_global_attention_chunk_size=args.attention_rows,
            q6_qmicro_permute=args.q6_qmicro_permute,
            q6_qmicro_planar=args.q6_qmicro_planar,
            moe_branch_concurrency=args.moe_branch_concurrency,
            moe_shared_after_router=args.moe_shared_after_router,
            moe_shared_low_priority=args.moe_shared_low_priority,
            use_moe_tail_next_rmsnorm=args.moe_tail_next_rmsnorm,
            use_head_kv_fusion=args.head_kv_fusion,
            global_split_min_live=(
                127 if args.decode_split_attention is True else None
            ),
            swa_split_min_live=(
                65 if args.decode_split_attention is True else None
            ),
            swa_split_tile16_min_live=(
                257 if args.decode_split_attention is True else None
            ),
            use_swa_split_tile16=(
                True if args.decode_split_attention is True else None
            ),
            use_split_attention=args.decode_split_attention,
            use_split_gate_fusion=(
                True if args.decode_split_attention is True else None
            ),
            use_swa_split_wave_local=(
                True if args.decode_split_attention is True else None
            ),
            prefill_long_attention_hipblaslt=(
                False
                if args.compare_long_attention_hipblaslt
                else args.long_attention_hipblaslt
            ),
            prefill_block_attention_hipblaslt=(
                False
                if args.compare_block_attention_hipblaslt
                else args.block_attention_hipblaslt
            ),
            prefill_dense_contiguous_cache=(
                False
                if args.compare_dense_contiguous_cache
                else args.dense_contiguous_cache
            ),
            prefill_swa_attention_hipblaslt=(
                False
                if args.compare_swa_attention_hipblaslt
                else args.swa_attention_hipblaslt
            ),
            use_selected_natural_tile8_decode=(
                False
                if args.compare_selected_natural_tile8_decode
                else None
            ),
            use_q4_decode_t16_sidecar=(
                False if args.compare_q4_decode_t16_sidecar else None
            ),
            use_q4_decode_t16_dual_interleaved=(
                False
                if args.compare_q4_decode_t16_dual_interleaved
                else None
            ),
            use_q4_expert_t16_dual_interleaved=(
                False if args.ordinary_q4_expert_t16 else None
            ),
        )
        active_moe_branch_concurrency = owner.moe_branch_concurrency
        active_q6_qmicro_permute = owner.q6_qmicro_permute
        active_q6_qmicro_planar = owner.q6_qmicro_planar
        active_moe_shared_after_router = owner.moe_shared_after_router
        active_moe_shared_low_priority = owner.moe_shared_low_priority
        active_moe_shared_priority_range = owner.moe_shared_priority_range
        active_moe_tail_next_rmsnorm = (
            owner.kernel_plan.moe_tail_next_rmsnorm is not None
        )
        active_head_kv_fusion = owner.use_head_kv_fusion
        active_global_split_min_live = owner.kv_cache.global_split_min_live
        active_swa_split_min_live = owner.kv_cache.swa_split_min_live
        active_swa_split_tile16_min_live = (
            owner.kv_cache.swa_split_tile16_min_live
        )
        active_split_gate_fusion = owner.use_split_gate_fusion
        active_swa_split_wave_local = owner.use_swa_split_wave_local
        active_swa_split_fixed512_reduce = (
            owner.kv_cache.swa_split_fixed512_reduce
        )
        active_swa_fused_fixed512 = owner.kv_cache.swa_fused_fixed512
        active_swa_gqa3_local384_fixed512 = (
            owner.kv_cache.swa_gqa3_local384_fixed512
        )
        active_swa_gqa3_vstage64_fixed512 = (
            owner.kv_cache.swa_gqa3_vstage64_fixed512
        )
        active_swa_gqa3_vstage64_vec16_fixed512 = (
            owner.kv_cache.swa_gqa3_vstage64_vec16_fixed512
        )
        active_swa_gqa3_vstage64_vec16_direct_fixed512 = (
            owner.kv_cache.swa_gqa3_vstage64_vec16_direct_fixed512
        )
        active_swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512 = (
            owner.kv_cache.swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
        )
        active_swa_local1024 = owner.kv_cache.swa_local1024
        active_global_split_fixedshape_reduce = (
            owner.kv_cache.global_split_fixedshape_reduce
        )
        active_global_split_gqa6_dim32_vstage64 = (
            owner.kv_cache.global_split_gqa6_dim32_vstage64
        )
        active_global_split_gqa6_deferrednorm_dim32_vstage64 = (
            owner.kv_cache.global_split_gqa6_deferrednorm_dim32_vstage64
        )
        active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage64 = (
            owner.kv_cache.global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage64
        )
        active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80 = (
            owner.kv_cache.global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80
        )
        active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix = (
            owner.kv_cache.global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix
        )
        active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_min_live = (
            owner.kv_cache.global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_min_live
        )
        active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value = (
            owner.kv_cache.global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value
        )
        active_global_split_gqa6_ctx4096_compensated_layer = (
            owner.kv_cache.global_split_gqa6_ctx4096_compensated_layer
        )
        active_global_split_gqa6_ctx4096_dim_tile = (
            owner.kv_cache.global_split_gqa6_ctx4096_dim_tile
        )
        active_global_split_gqa6_ctx4096_deferrednorm = (
            owner.kv_cache.global_split_gqa6_ctx4096_deferrednorm
        )
        active_global_split_gqa6_ctx4096_tokenloop4 = (
            owner.kv_cache.global_split_gqa6_ctx4096_tokenloop4
        )
        active_global_split_gqa6_ctx4096_min_live = (
            owner.kv_cache.global_split_gqa6_ctx4096_min_live
        )
        active_global_split_gqa6_ctx4096_min_layer = (
            owner.kv_cache.global_split_gqa6_ctx4096_min_layer
        )
        active_global_fused_fixedshape = owner.kv_cache.global_fused_fixedshape
        active_global_gqa2_vstage64_fixedshape = (
            owner.kv_cache.global_gqa2_vstage64_fixedshape
        )
        active_global_gqa2_vstage64_vec16_fixedshape = (
            owner.kv_cache.global_gqa2_vstage64_vec16_fixedshape
        )
        active_global_gqa2_vstage64_vec16_direct_fixedshape = (
            owner.kv_cache.global_gqa2_vstage64_vec16_direct_fixedshape
        )
        active_global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape = (
            owner.kv_cache.global_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape
        )
        active_global_local1024 = owner.kv_cache.global_local1024
        active_long_attention_hipblaslt = (
            owner.prefill_long_attention_hipblaslt
        )
        active_block_attention_hipblaslt = (
            owner.prefill_block_attention_hipblaslt
        )
        active_swa_attention_hipblaslt = (
            owner.prefill_swa_attention_hipblaslt
        )
        active_dense_contiguous_cache = (
            owner.prefill_dense_contiguous_cache
        )
        active_q4_decode_t16_sidecar = (
            owner.use_q4_decode_t16_sidecar
        )
        active_q4_decode_t16_dual_interleaved = (
            owner.use_q4_decode_t16_dual_interleaved
        )
        active_q4_shared_down_t16_decode = (
            owner.use_q4_shared_down_t16_decode
        )
        active_q4_expert_t16_dual_interleaved = (
            owner.use_q4_expert_t16_dual_interleaved
        )
        active_attention_rows = owner.prefill_attention_chunk_size
        active_global_attention_rows = (
            owner.prefill_global_attention_chunk_size
        )
        load_seconds = time.perf_counter() - load_started
        owner.prefill(token_stream[: args.warmup_rows], use_bulk=True)
        runtime.device_synchronize()
        for repetition in range(args.repetitions):
            for order_index, length in enumerate(
                _timing_order(lengths, repetition)
            ):
                modes = ("production",)
                if comparison:
                    modes = (
                        ("control", "candidate")
                        if (repetition + order_index) % 2 == 0
                        else ("candidate", "control")
                    )
                for mode in modes:
                    if args.compare_long_attention_hipblaslt:
                        owner.set_prefill_long_attention_hipblaslt(
                            mode == "candidate"
                        )
                    if args.compare_block_attention_hipblaslt:
                        owner.set_prefill_block_attention_hipblaslt(
                            mode == "candidate"
                        )
                    if args.compare_swa_attention_hipblaslt:
                        owner.set_prefill_swa_attention_hipblaslt(
                            mode == "candidate"
                        )
                    if args.compare_dense_contiguous_cache:
                        owner.set_prefill_dense_contiguous_cache(
                            mode == "candidate"
                        )
                    if args.compare_f16_decode_onebarrier:
                        os.environ["HIPENGINE_LAGUNA_F16_DECODE"] = (
                            "onebarrier" if mode == "candidate" else "gemv"
                        )
                    if args.compare_f16_decode_fixedk:
                        os.environ["HIPENGINE_LAGUNA_F16_DECODE"] = (
                            "fixedk" if mode == "candidate" else "onebarrier"
                        )
                    if args.compare_swa_fixed512_reduce:
                        owner.kv_cache.swa_split_fixed512_reduce = (
                            mode == "candidate"
                        )
                    if args.compare_swa_fused_fixed512:
                        owner.kv_cache.swa_fused_fixed512 = mode == "candidate"
                    if args.compare_swa_gqa3_local384:
                        owner.kv_cache.swa_gqa3_local384_fixed512 = (
                            mode == "candidate"
                        )
                    if args.compare_swa_gqa3_vstage64:
                        owner.kv_cache.swa_gqa3_vstage64_fixed512 = (
                            mode == "candidate"
                        )
                    if args.compare_swa_gqa3_vstage64_vec16:
                        owner.kv_cache.swa_gqa3_vstage64_vec16_fixed512 = (
                            mode == "candidate"
                        )
                    if args.compare_swa_gqa3_vstage64_vec16_direct:
                        owner.kv_cache.swa_gqa3_vstage64_vec16_direct_fixed512 = (
                            mode == "candidate"
                        )
                    if args.compare_swa_assume_exp:
                        owner.set_decode_swa_assume_exp(mode == "candidate")
                    if args.compare_swa_mixed32:
                        owner.set_decode_swa_mixed32(mode == "candidate")
                    if args.compare_swa_mixed32_exp4:
                        owner.set_decode_swa_mixed32_exp4(mode == "candidate")
                    if args.compare_swa_mixed32_exp8:
                        owner.set_decode_swa_mixed32_exp8(mode == "candidate")
                    if args.compare_swa_mixed32_exp16:
                        owner.set_decode_swa_mixed32_exp16(mode == "candidate")
                    if args.compare_swa_mixed32_exp32:
                        owner.set_decode_swa_mixed32_exp32(mode == "candidate")
                    if args.compare_global_fixedshape_reduce:
                        owner.kv_cache.global_split_fixedshape_reduce = (
                            mode == "candidate"
                        )
                    if args.compare_global_fused_fixedshape:
                        owner.kv_cache.global_fused_fixedshape = (
                            mode == "candidate"
                        )
                    if args.compare_global_gqa2_vstage64:
                        owner.kv_cache.global_gqa2_vstage64_fixedshape = (
                            mode == "candidate"
                        )
                    if args.compare_global_gqa2_vstage64_vec16:
                        owner.kv_cache.global_gqa2_vstage64_vec16_fixedshape = (
                            mode == "candidate"
                        )
                    if args.compare_global_gqa2_vstage64_vec16_direct:
                        owner.kv_cache.global_gqa2_vstage64_vec16_direct_fixedshape = (
                            mode == "candidate"
                        )
                    if args.compare_global_assume_exp:
                        owner.set_decode_global_assume_exp(mode == "candidate")
                    if args.compare_global_exp32:
                        owner.set_decode_global_exp32(mode == "candidate")
                    if args.compare_global_mixed32:
                        owner.set_decode_global_mixed32(mode == "candidate")
                    if args.compare_selected_natural_decode:
                        owner.set_selected_natural_decode(
                            mode == "candidate"
                        )
                    if args.compare_selected_natural_tile8_decode:
                        owner.set_selected_natural_tile8_decode(
                            mode == "candidate"
                        )
                    if args.compare_q4_decode_t16_sidecar:
                        owner.set_q4_decode_t16_sidecar(
                            mode == "candidate"
                        )
                    if args.compare_q4_decode_t16_dual_interleaved:
                        owner.set_q4_decode_t16_dual_interleaved(
                            mode == "candidate"
                        )
                    owner.reset_state()
                    started = time.perf_counter()
                    result = owner.prefill(token_stream[:length], use_bulk=True)
                    runtime.device_synchronize()
                    elapsed = time.perf_counter() - started
                    prefill_final_position = int(owner.position)
                    generated = [int(result.next_token_id)]
                    decode_seconds = 0.0
                    if output_tokens > 1:
                        decode_started = time.perf_counter()
                        while len(generated) < output_tokens:
                            result = owner.forward_token(result.next_token_id)
                            generated.append(int(result.next_token_id))
                        runtime.device_synchronize()
                        decode_seconds = time.perf_counter() - decode_started
                    row = {
                        "length": length,
                        "chunks": math.ceil(length / args.chunk_size),
                        "prefill_seconds": elapsed,
                        "prefill_tok_s": length / elapsed,
                        "next_token_id": generated[0],
                        "prefill_final_position": prefill_final_position,
                        "final_position": int(owner.position),
                        "repetition": repetition,
                        "output_tokens": output_tokens,
                    }
                    if output_tokens > 1:
                        row.update(
                            {
                                "decode_forward_calls": output_tokens - 1,
                                "decode_seconds": decode_seconds,
                                "decode_tok_s": (output_tokens - 1) / decode_seconds,
                                "final_token_id": generated[-1],
                                "generated_ids_sha256": _sha256_json(generated),
                            }
                        )
                    if comparison:
                        row["mode"] = mode
                    rows.append(row)
                    mode_text = (
                        f" mode={mode}"
                        if comparison
                        else ""
                    )
                    print(
                        f"rep={repetition} length={length}{mode_text} "
                        f"chunks={row['chunks']} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s "
                        f"next={generated[0]}"
                        + (
                            f" decode={row['decode_tok_s']:.3f} tok/s "
                            f"final={generated[-1]}"
                            if output_tokens > 1
                            else ""
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
        if (
            args.compare_f16_decode_onebarrier
            or args.compare_f16_decode_fixedk
        ):
            if original_f16_decode_mode is None:
                os.environ.pop("HIPENGINE_LAGUNA_F16_DECODE", None)
            else:
                os.environ["HIPENGINE_LAGUNA_F16_DECODE"] = (
                    original_f16_decode_mode
                )
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna LPF-5 profiling")

    if comparison:
        summaries = {
            mode: {
                str(length): _summarize_samples(
                    [
                        row
                        for row in rows
                        if int(row["length"]) == length
                        and row["mode"] == mode
                    ]
                )
                for length in lengths
            }
            for mode in ("control", "candidate")
        }
        summary_rows = [
            summary
            for mode_summaries in summaries.values()
            for summary in mode_summaries.values()
        ]
        mode_next_tokens_match = all(
            summaries["control"][str(length)]["next_token_ids"]
            == summaries["candidate"][str(length)]["next_token_ids"]
            for length in lengths
        )
        mode_generated_ids_match = all(
            summaries["control"][str(length)].get("generated_ids_sha256")
            == summaries["candidate"][str(length)].get("generated_ids_sha256")
            for length in lengths
        )
    else:
        summaries = {
            str(length): _summarize_samples(
                [row for row in rows if int(row["length"]) == length]
            )
            for length in lengths
        }
        summary_rows = list(summaries.values())
        mode_next_tokens_match = True
        mode_generated_ids_match = True
    positions_exact = all(
        int(row["prefill_final_position"]) == int(row["length"]) - 1
        and int(row["final_position"])
        == int(row["length"]) + output_tokens - 2
        for row in rows
    )
    deterministic = all(
        bool(summary["repeat_deterministic"]) for summary in summary_rows
    )
    generated_deterministic = all(
        bool(summary.get("repeat_generated_ids_deterministic", True))
        for summary in summary_rows
    )
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(
        positions_exact
        and deterministic
        and generated_deterministic
        and mode_next_tokens_match
        and mode_generated_ids_match
        and recovered
    )
    prompt_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_long_context_profile",
        "status": "accepted_attribution_baseline" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "scope": (
            "Laguna S 2.1 c=1 prefill-only matrix-chunk attribution baseline"
            if output_tokens == 1
            else "Laguna S 2.1 c=1 prefill plus fixed-horizon eager-decode sweep"
        ),
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "lengths": list(lengths),
            "chunk_size": args.chunk_size,
            "matrix_rows": args.chunk_size,
            "attention_rows": active_global_attention_rows,
            "swa_attention_rows": min(
                args.chunk_size,
                128,
                active_attention_rows,
            ),
            "dense_contiguous_cache": active_dense_contiguous_cache,
            "chunks_per_length": {
                str(length): math.ceil(length / args.chunk_size) for length in lengths
            },
            "context_length": args.context_length,
            "output_tokens_including_first": output_tokens,
            "decode_forward_calls_per_run": output_tokens - 1,
            "repetitions": args.repetitions,
            "warmup_rows": args.warmup_rows,
            "q6_qmicro_permute": active_q6_qmicro_permute,
            "q6_qmicro_planar": active_q6_qmicro_planar,
            "moe_branch_concurrency": active_moe_branch_concurrency,
            "moe_shared_after_router": active_moe_shared_after_router,
            "moe_shared_low_priority": active_moe_shared_low_priority,
            "moe_shared_priority_range": active_moe_shared_priority_range,
            "moe_tail_next_rmsnorm": active_moe_tail_next_rmsnorm,
            "moe_tail_next_rmsnorm_requested": args.moe_tail_next_rmsnorm,
            "head_kv_fusion": active_head_kv_fusion,
            "head_kv_fusion_requested": args.head_kv_fusion,
            "decode_split_attention_requested": args.decode_split_attention,
            "compare_f16_decode_onebarrier": (
                args.compare_f16_decode_onebarrier
            ),
            "compare_f16_decode_fixedk": args.compare_f16_decode_fixedk,
            "compare_swa_fixed512_reduce": (
                args.compare_swa_fixed512_reduce
            ),
            "compare_swa_fused_fixed512": args.compare_swa_fused_fixed512,
            "compare_swa_gqa3_local384": args.compare_swa_gqa3_local384,
            "compare_swa_gqa3_vstage64": (
                args.compare_swa_gqa3_vstage64
            ),
            "compare_swa_gqa3_vstage64_vec16": (
                args.compare_swa_gqa3_vstage64_vec16
            ),
            "compare_swa_gqa3_vstage64_vec16_direct": (
                args.compare_swa_gqa3_vstage64_vec16_direct
            ),
            "compare_swa_assume_exp": args.compare_swa_assume_exp,
            "compare_swa_mixed32": args.compare_swa_mixed32,
            "compare_swa_mixed32_exp4": args.compare_swa_mixed32_exp4,
            "compare_swa_mixed32_exp8": args.compare_swa_mixed32_exp8,
            "compare_swa_mixed32_exp16": args.compare_swa_mixed32_exp16,
            "compare_swa_mixed32_exp32": args.compare_swa_mixed32_exp32,
            "compare_global_fixedshape_reduce": (
                args.compare_global_fixedshape_reduce
            ),
            "compare_global_fused_fixedshape": (
                args.compare_global_fused_fixedshape
            ),
            "compare_global_gqa2_vstage64": (
                args.compare_global_gqa2_vstage64
            ),
            "compare_global_gqa2_vstage64_vec16": (
                args.compare_global_gqa2_vstage64_vec16
            ),
            "compare_global_gqa2_vstage64_vec16_direct": (
                args.compare_global_gqa2_vstage64_vec16_direct
            ),
            "compare_global_assume_exp": args.compare_global_assume_exp,
            "compare_global_exp32": args.compare_global_exp32,
            "compare_global_mixed32": args.compare_global_mixed32,
            "compare_selected_natural_decode": (
                args.compare_selected_natural_decode
            ),
            "compare_selected_natural_tile8_decode": (
                args.compare_selected_natural_tile8_decode
            ),
            "compare_q4_decode_t16_sidecar": (
                args.compare_q4_decode_t16_sidecar
            ),
            "q4_decode_t16_sidecar": active_q4_decode_t16_sidecar,
            "compare_q4_decode_t16_dual_interleaved": (
                args.compare_q4_decode_t16_dual_interleaved
            ),
            "q4_decode_t16_dual_interleaved": (
                active_q4_decode_t16_dual_interleaved
            ),
            "q4_shared_down_t16_decode": (
                active_q4_shared_down_t16_decode
            ),
            "ordinary_q4_expert_t16": args.ordinary_q4_expert_t16,
            "q4_expert_t16_dual_interleaved": (
                active_q4_expert_t16_dual_interleaved
            ),
            "global_split_min_live": active_global_split_min_live,
            "swa_split_min_live": active_swa_split_min_live,
            "swa_split_tile16_min_live": active_swa_split_tile16_min_live,
            "split_gate_fusion": active_split_gate_fusion,
            "swa_split_wave_local": active_swa_split_wave_local,
            "swa_split_fixed512_reduce": (
                active_swa_split_fixed512_reduce
            ),
            "swa_fused_fixed512": active_swa_fused_fixed512,
            "swa_gqa3_local384_fixed512": (
                active_swa_gqa3_local384_fixed512
            ),
            "swa_gqa3_vstage64_fixed512": (
                active_swa_gqa3_vstage64_fixed512
            ),
            "swa_gqa3_vstage64_vec16_fixed512": (
                active_swa_gqa3_vstage64_vec16_fixed512
            ),
            "swa_gqa3_vstage64_vec16_direct_fixed512": (
                active_swa_gqa3_vstage64_vec16_direct_fixed512
            ),
            "swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512": (
                active_swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512
            ),
            "swa_local1024": active_swa_local1024,
            "global_split_fixedshape_reduce": (
                active_global_split_fixedshape_reduce
            ),
            "global_split_gqa6_dim32_vstage64": (
                active_global_split_gqa6_dim32_vstage64
            ),
            "global_split_gqa6_deferrednorm_dim32_vstage64": (
                active_global_split_gqa6_deferrednorm_dim32_vstage64
            ),
            "global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage64": (
                active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage64
            ),
            "global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80": (
                active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80
            ),
            "global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix": (
                active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix
            ),
            "global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_min_live": (
                active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_min_live
            ),
            "global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value": (
                active_global_split_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value
            ),
            "global_split_gqa6_ctx4096_min_layer": (
                active_global_split_gqa6_ctx4096_min_layer
            ),
            "global_split_gqa6_ctx4096_compensated_layer": (
                active_global_split_gqa6_ctx4096_compensated_layer
            ),
            "global_split_gqa6_ctx4096_dim_tile": (
                active_global_split_gqa6_ctx4096_dim_tile
            ),
            "global_split_gqa6_ctx4096_deferrednorm": (
                active_global_split_gqa6_ctx4096_deferrednorm
            ),
            "global_split_gqa6_ctx4096_tokenloop4": (
                active_global_split_gqa6_ctx4096_tokenloop4
            ),
            "global_split_gqa6_ctx4096_min_live": (
                active_global_split_gqa6_ctx4096_min_live
            ),
            "global_fused_fixedshape": active_global_fused_fixedshape,
            "global_gqa2_vstage64_fixedshape": (
                active_global_gqa2_vstage64_fixedshape
            ),
            "global_gqa2_vstage64_vec16_fixedshape": (
                active_global_gqa2_vstage64_vec16_fixedshape
            ),
            "global_gqa2_vstage64_vec16_direct_fixedshape": (
                active_global_gqa2_vstage64_vec16_direct_fixedshape
            ),
            "global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape": (
                active_global_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape
            ),
            "global_local1024": active_global_local1024,
            "long_attention_hipblaslt": active_long_attention_hipblaslt,
            "long_attention_hipblaslt_requested": (
                args.long_attention_hipblaslt
            ),
            "compare_long_attention_hipblaslt": (
                args.compare_long_attention_hipblaslt
            ),
            "compare_block_attention_hipblaslt": (
                args.compare_block_attention_hipblaslt
            ),
            "compare_dense_contiguous_cache": (
                args.compare_dense_contiguous_cache
            ),
            "dense_contiguous_cache_requested": (
                args.dense_contiguous_cache
            ),
            "block_attention_hipblaslt": (
                active_block_attention_hipblaslt
            ),
            "block_attention_hipblaslt_requested": (
                args.block_attention_hipblaslt
            ),
            "compare_swa_attention_hipblaslt": (
                args.compare_swa_attention_hipblaslt
            ),
            "swa_attention_hipblaslt": active_swa_attention_hipblaslt,
            "swa_attention_hipblaslt_requested": (
                args.swa_attention_hipblaslt
            ),
            "timed_order": "ascending then alternating direction by repetition",
            "timing_scope": (
                "prefill: reset complete through synchronized first-token projection; "
                "decode: exactly output_tokens-1 synchronized forward_token calls; "
                "load excluded"
            ),
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
        },
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "rows": rows,
        "timings": summaries,
        "correctness": {
            "pass": passed,
            "final_positions_exact": positions_exact,
            "repeat_next_token_deterministic": deterministic,
            "repeat_generated_ids_deterministic": generated_deterministic,
            "control_candidate_next_tokens_match": mode_next_tokens_match,
            "control_candidate_generated_ids_match": mode_generated_ids_match,
            "tracked_returned_to_baseline": recovered,
            "boundary_fixture_evidence": [
                "tests/test_laguna_cpu_reference.py::test_laguna_block_streaming_oracle_matches_dense_at_boundaries_and_tails",
                "tests/test_laguna_cpu_reference.py::test_laguna_block_streaming_oracle_handles_final_128k_position",
                "tests/test_laguna_cpu_reference.py::test_laguna_global_and_swa_masks_match_transformers_at_511_512_513",
                "tests/test_laguna_kv_attention.py::test_laguna_global_and_swa_token_serial_attention_match_cpu_across_wraps",
                "tests/test_laguna_kv_attention.py::test_laguna_bulk_global_and_swa_prefill_match_serial_across_ring_wrap",
            ],
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": (
            [
                "This is an attribution baseline, not a speedup or long-context support claim.",
                "The deterministic stream repeats the longest canonical prompt without its leading BOS.",
                "Run once under cached-only rocprofv3 and attach a raw-trace summary separately.",
            ]
            if output_tokens == 1
            else [
                "This is a fixed-shape eager c=1 decode snapshot, not a decode speedup claim.",
                "Decode throughput covers 127 synchronized forward_token calls after the first prefill-produced token.",
                "Dense-prefix global decode selects the capacity-independent "
                "exact fused specialization through the resource-qualified "
                "live-context bands (local1024 through 4000 slots, local512 "
                "through 6000); larger live contexts use GQA6 score "
                "ownership, ordered exp/sum reduction, deferred exact "
                "normalization, and dimension-sharded staged-V PV; gfx1151 "
                "quality-gates the 4,096-token split only at live >= "
                f"{active_global_split_gqa6_ctx4096_min_live:,}, with a "
                "compensated owner at global layer 28 and the ordinary split "
                "at layers "
                "32/36/40/44, with byte-identical D64 PV geometry for all "
                "five admitted layers.",
            ]
        ),
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
