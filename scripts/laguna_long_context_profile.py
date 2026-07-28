#!/usr/bin/env python3
"""Profile Laguna prefill at retained long-context and LAP-0 row lengths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
SHORT_FOCUS_LENGTHS = (512, 4096)
WPF_SHORT_LENGTHS = (512, 1024)
LAP0_LENGTHS = (128, 512, 1024, 4096)
ATTACK_LENGTHS = (4096, 16384, 65536, 131072)
ATTACK_DIRECTIONAL_LENGTHS = (4096, 16384, 65536)
LC1_128K_GATE_LENGTHS = (131072,)
LC0_TRACE_LENGTHS = (16384, 65536)
FINAL_SWEEP_LENGTHS = (512, 1024, 4096, 32768, 65536, 131072)
STANDARD_DECODE_LENGTHS = (512,)
DECODE_OUTPUT_TOKENS = (1, 128)
EAGER_DECODE_CONTEXT_LIMIT = 4096
PROFILE_LENGTH_SETS = (
    DEFAULT_LENGTHS,
    SHORT_FOCUS_LENGTHS,
    WPF_SHORT_LENGTHS,
    LAP0_LENGTHS,
    ATTACK_LENGTHS,
    ATTACK_DIRECTIONAL_LENGTHS,
    LC1_128K_GATE_LENGTHS,
    LC0_TRACE_LENGTHS,
    FINAL_SWEEP_LENGTHS,
    STANDARD_DECODE_LENGTHS,
)
DEFAULT_CHUNK_SIZE = 128
PROFILE_CHUNK_SIZES = (128, 256, 512, 1024, 2048)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-long-context-profile.json"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=max(DEFAULT_LENGTHS))
    parser.add_argument("--lengths", type=_parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--chunk-size", type=_parse_chunk_size, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--package-matrix-rows",
        action="store_true",
        help="resolve matrix rows from backend package capability instead of --chunk-size",
    )
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
        "--compare-raw-k-prefill-rowbatch",
        action="store_true",
        help="compare scalar raw Q5/Q6 prefill with fixed rowbatch4/8/16/32",
    )
    parser.add_argument(
        "--raw-k-prefill-rowbatch",
        type=int,
        choices=(0, 4, 8, 16, 32),
        help="explicit raw Q5/Q6 row slab; comparison defaults candidate to 8",
    )
    parser.add_argument(
        "--raw-k-prefill-rowbatch-control",
        type=int,
        choices=(0, 4, 8, 16, 32),
        default=0,
        help="control row slab for a rowbatch comparison; defaults to scalar 0",
    )
    parser.add_argument(
        "--compare-raw-k-prefill-mmq",
        action="store_true",
        help="compare exact rowbatch8 with the Q5/Q6 producer-row MMQ owner",
    )
    parser.add_argument(
        "--compare-grouped-exact-iq",
        action="store_true",
        help="compare retained route-major IQ with exact expert-major IQ down reuse",
    )
    parser.add_argument(
        "--compare-pair16-grouped-gate-up",
        action="store_true",
        help="compare retained grouped-down with exact pair16 grouped IQ2 gate/up",
    )
    parser.add_argument(
        "--raw-k-prefill-mmq",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="explicitly enable/disable the default-off Q5/Q6 MMQ owner",
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
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-gguf", action="store_true")
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--quant-label", default="Q4_K_M mixed GGUF v3")
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
    matrix_label = "package" if args.package_matrix_rows else str(args.chunk_size)
    output_tokens = int(args.decode_output_tokens)
    required_context = max(lengths) + output_tokens - 1
    if args.context_length < required_context:
        raise ValueError(
            "largest Laguna prompt plus output horizon exceeds admitted context"
        )
    if output_tokens > 1 and args.context_length > EAGER_DECODE_CONTEXT_LIMIT:
        raise ValueError(
            "fixed-horizon eager decode requires context length at most "
            f"{EAGER_DECODE_CONTEXT_LIMIT}"
        )
    if args.repetitions <= 0:
        raise ValueError("LPF-5 repetitions must be positive")
    if args.safety_reserve_gib < 0.0:
        raise ValueError("--safety-reserve-gib must be non-negative")
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
    active_comparisons = sum(
        bool(value)
        for value in (
            args.compare_long_attention_hipblaslt,
            args.compare_raw_k_prefill_rowbatch,
            args.compare_raw_k_prefill_mmq,
            args.compare_grouped_exact_iq,
            args.compare_pair16_grouped_gate_up,
            args.compare_block_attention_hipblaslt,
            args.compare_dense_contiguous_cache,
            args.compare_swa_attention_hipblaslt,
        )
    )
    if active_comparisons > 1:
        raise ValueError("only one Laguna prefill comparison may be active")
    if args.compare_raw_k_prefill_rowbatch and args.raw_k_prefill_rowbatch == 0:
        raise ValueError(
            "--compare-raw-k-prefill-rowbatch requires candidate rowbatch4/8/16/32"
        )
    if args.compare_raw_k_prefill_mmq and args.raw_k_prefill_mmq is not None:
        raise ValueError(
            "--compare-raw-k-prefill-mmq and --raw-k-prefill-mmq are mutually exclusive"
        )
    raw_k_candidate = (
        8
        if args.compare_raw_k_prefill_rowbatch
        and args.raw_k_prefill_rowbatch is None
        else args.raw_k_prefill_rowbatch
    )
    raw_k_control = int(args.raw_k_prefill_rowbatch_control)
    if not args.compare_raw_k_prefill_rowbatch and raw_k_control != 0:
        raise ValueError(
            "--raw-k-prefill-rowbatch-control requires "
            "--compare-raw-k-prefill-rowbatch"
        )
    if args.compare_raw_k_prefill_rowbatch and raw_k_candidate == raw_k_control:
        raise ValueError("raw-K rowbatch control and candidate must differ")
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
    comparison = bool(
        args.compare_long_attention_hipblaslt
        or args.compare_raw_k_prefill_rowbatch
        or args.compare_raw_k_prefill_mmq
        or args.compare_grouped_exact_iq
        or args.compare_pair16_grouped_gate_up
        or args.compare_block_attention_hipblaslt
        or args.compare_dense_contiguous_cache
        or args.compare_swa_attention_hipblaslt
    )
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
        quant=args.quant_label,
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile=(
            f"laguna_prefill_long_context_matrix{matrix_label}"
            if output_tokens == 1
            else f"laguna_p512_d{output_tokens}_matrix{matrix_label}"
        ),
        timing_protocol=(
            (
                "prefill_only_"
                + "_".join(str(length) for length in lengths)
                + f"_matrix{matrix_label}_attention128"
            )
            if output_tokens == 1
            else (
                f"p{lengths[0]}_d{output_tokens}_eager_c1_"
                f"matrix{matrix_label}_attention128"
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
    active_matrix_rows = args.chunk_size
    active_moe_branch_concurrency = False
    active_q6_qmicro_permute = False
    active_q6_qmicro_planar = False
    active_moe_shared_after_router = False
    active_moe_shared_low_priority = False
    active_moe_shared_priority_range: tuple[int, int] | None = None
    active_long_attention_hipblaslt = False
    active_block_attention_hipblaslt = False
    active_swa_attention_hipblaslt = False
    active_dense_contiguous_cache = False
    active_raw_k_prefill_mmq = False
    active_attention_rows = 128
    active_global_attention_rows = 128
    rows: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            safety_reserve_nbytes=int(args.safety_reserve_gib * 2**30),
            progress=_progress,
            repacked_cache=None if args.direct_gguf else args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=(None if args.package_matrix_rows else args.chunk_size),
            prefill_global_attention_chunk_size=args.attention_rows,
            q6_qmicro_permute=args.q6_qmicro_permute,
            q6_qmicro_planar=args.q6_qmicro_planar,
            moe_branch_concurrency=args.moe_branch_concurrency,
            moe_shared_after_router=args.moe_shared_after_router,
            moe_shared_low_priority=args.moe_shared_low_priority,
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
            raw_k_prefill_rowbatch=(
                raw_k_control
                if args.compare_raw_k_prefill_rowbatch
                else args.raw_k_prefill_rowbatch
            ),
            raw_k_prefill_mmq=(
                False
                if args.compare_raw_k_prefill_mmq
                else args.raw_k_prefill_mmq
            ),
        )
        active_matrix_rows = owner.prefill_chunk_size
        active_moe_branch_concurrency = owner.moe_branch_concurrency
        active_q6_qmicro_permute = owner.q6_qmicro_permute
        active_q6_qmicro_planar = owner.q6_qmicro_planar
        active_moe_shared_after_router = owner.moe_shared_after_router
        active_moe_shared_low_priority = owner.moe_shared_low_priority
        active_moe_shared_priority_range = owner.moe_shared_priority_range
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
        active_raw_k_prefill_mmq = owner.raw_k_prefill_mmq
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
                    if args.compare_raw_k_prefill_rowbatch:
                        owner.set_raw_k_prefill_rowbatch(
                            int(raw_k_candidate)
                            if mode == "candidate"
                            else raw_k_control
                        )
                    if args.compare_raw_k_prefill_mmq:
                        owner.set_raw_k_prefill_mmq(mode == "candidate")
                    if args.compare_grouped_exact_iq:
                        selected_mode = (
                            "grouped_exact" if mode == "candidate" else "direct"
                        )
                        owner.set_selected_gate_up_mode(selected_mode)
                        owner.set_selected_down_mode(selected_mode)
                    if args.compare_pair16_grouped_gate_up:
                        gate_up_mode = (
                            "grouped_pair16"
                            if mode == "candidate"
                            else "grouped_exact"
                        )
                        owner.set_selected_gate_up_mode(gate_up_mode)
                        owner.set_selected_down_mode("grouped_exact")
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
                        "chunks": math.ceil(length / active_matrix_rows),
                        "prefill_seconds": elapsed,
                        "prefill_tok_s": length / elapsed,
                        "next_token_id": generated[0],
                        "prefill_final_position": prefill_final_position,
                        "final_position": int(owner.position),
                        "repetition": repetition,
                        "output_tokens": output_tokens,
                        "raw_k_prefill_rowbatch": owner.raw_k_prefill_rowbatch,
                        "raw_k_prefill_mmq": owner.raw_k_prefill_mmq,
                        "selected_gate_up_mode": owner.selected_gate_up_mode,
                        "selected_down_mode": owner.selected_down_mode,
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
    active_cache = None if args.direct_gguf else args.repacked_cache
    manifest_path = None if active_cache is None else active_cache / "manifest.json"
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
            "quant": args.quant_label,
            "repacked_cache": None if active_cache is None else str(active_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes())
                if manifest_path is not None and manifest_path.is_file()
                else None
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
            "chunk_size": active_matrix_rows,
            "matrix_rows": active_matrix_rows,
            "package_matrix_rows_requested": bool(args.package_matrix_rows),
            "matrix_rows_cli_fallback": args.chunk_size,
            "attention_rows": active_global_attention_rows,
            "swa_attention_rows": min(
                active_matrix_rows,
                128,
                active_attention_rows,
            ),
            "dense_contiguous_cache": active_dense_contiguous_cache,
            "chunks_per_length": {
                str(length): math.ceil(length / active_matrix_rows) for length in lengths
            },
            "context_length": args.context_length,
            "direct_gguf": bool(args.direct_gguf),
            "safety_reserve_gib": float(args.safety_reserve_gib),
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
            "long_attention_hipblaslt": active_long_attention_hipblaslt,
            "long_attention_hipblaslt_requested": (
                args.long_attention_hipblaslt
            ),
            "compare_long_attention_hipblaslt": (
                args.compare_long_attention_hipblaslt
            ),
            "compare_raw_k_prefill_rowbatch": (
                args.compare_raw_k_prefill_rowbatch
            ),
            "raw_k_prefill_rowbatch_requested": args.raw_k_prefill_rowbatch,
            "raw_k_prefill_rowbatch_control": raw_k_control,
            "raw_k_prefill_rowbatch_candidate": raw_k_candidate,
            "compare_raw_k_prefill_mmq": args.compare_raw_k_prefill_mmq,
            "compare_grouped_exact_iq": args.compare_grouped_exact_iq,
            "compare_pair16_grouped_gate_up": (
                args.compare_pair16_grouped_gate_up
            ),
            "raw_k_prefill_mmq": active_raw_k_prefill_mmq,
            "raw_k_prefill_mmq_requested": args.raw_k_prefill_mmq,
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
                "The eager global-attention decode ABI currently admits cache capacity at most 4096.",
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
