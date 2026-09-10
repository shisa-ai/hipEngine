#!/usr/bin/env python3
"""Qualify natural Qwen4Exp QSA retrieval, control, and transaction ownership.

The selection oracle follows Transformers ``Qwen4ExpTextQSAIndexer`` at
huggingface/transformers@fc5c5bde8e656dad91cbf34e61940d984b1c7b91:
select a whole-block token budget and append only the actual incomplete tail.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.memory import (
    copy_device_to_host,
    host_array_ptr,
    memory_stats,
    reset_memory_stats,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.kernels.cpu_reference.qwen4_exp import (
    qsa_index_scores,
    qsa_prepare_index_keys,
    qsa_select_positions,
)
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4_exp_compare_logits import compare_logits


def build_retrieval_prompt(tokenizer: object, *, target_tokens: int) -> tuple[str, int]:
    prefix = (
        "<|im_start|>system\nReasoning effort is set to low. Keep your thinking "
        "brief and focused, moving directly to the conclusion without unnecessary "
        "elaboration. You retrieve exact facts from long archives.<|im_end|>\n"
        "<|im_start|>user\nYou are auditing an archive. Read all records and "
        "answer the final question exactly.\n"
    )
    records = [
        (
            f"Archive record {index:04d}: The unique access code is VIOLET-7391. "
            "Preserve this exact code for the audit.\n"
            if index == 24
            else f"Archive record {index:04d}: Inventory batch {10000 + index} is "
            "ordinary, stable, and requires no action.\n"
        )
        for index in range(100)
    ]
    base = prefix + "".join(records)
    suffix = (
        "\nFinal audit question: What is the unique access code? "
        "Answer with only the exact code.<|im_end|>\n<|im_start|>assistant\n"
    )
    filler = (
        "The archive checksum remains stable and the ordinary inventory records "
        "require no action. "
    )
    encode = getattr(tokenizer, "encode")
    while len(encode(base + filler + suffix)) <= target_tokens:
        base += filler
    prompt = base + suffix
    for count in range(64):
        candidate = base + " a" * count + suffix
        if len(encode(candidate)) <= target_tokens and len(encode(candidate)) > len(encode(prompt)):
            prompt = candidate
    needle = len(encode(prompt[: prompt.index("VIOLET-7391")]))
    return prompt, needle


def _selection_locality(
    selected_positions: np.ndarray, *, kv_row_bytes: int
) -> dict[str, int | float]:
    selected = np.sort(np.asarray(selected_positions, dtype=np.int64))
    if selected.size == 0:
        return {
            "selected_position_span": 0,
            "selected_contiguous_pairs": 0,
            "selected_gap_mean": 0.0,
            "selected_gap_max": 0,
            "selected_kv_pages_per_layer": 0,
        }
    gaps = np.diff(selected)
    selected_pages = np.unique(selected * int(kv_row_bytes) // 4096)
    return {
        "selected_position_span": int(selected[-1] - selected[0] + 1),
        "selected_contiguous_pairs": int(np.count_nonzero(gaps == 1)),
        "selected_gap_mean": float(np.mean(gaps)) if gaps.size else 0.0,
        "selected_gap_max": int(np.max(gaps)) if gaps.size else 0,
        "selected_kv_pages_per_layer": int(selected_pages.size),
    }


def _read_llama_tap(path: Path) -> tuple[list[int], np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 32:
        raise ValueError("llama tap is shorter than its shape header")
    shape = np.frombuffer(payload[:32], dtype=np.int64)
    values = np.frombuffer(payload[32:], dtype=np.int32)
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError("llama tap payload does not match its shape")
    tensor = values.reshape(tuple(int(value) for value in shape[::-1]))
    return [int(value) for value in shape], tensor[-1, -1, -1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--target-tokens", type=int, default=4_096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--teacher-logits", type=Path)
    parser.add_argument("--teacher-tokens", type=Path)
    parser.add_argument("--llama-topk-tap", type=Path)
    parser.add_argument(
        "--depth-profile", action="store_true",
        help="Record structural QSA locality, PLE I/O, graph reuse, and KV-byte census",
    )
    args = parser.parse_args()

    info = load_gguf_index(discover_gguf_files(args.model)[0])
    generator = None
    reset_memory_stats()
    try:
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=args.model,
            weight_index=info,
            model_plugin=resolve_model(info.architecture or ""),
            backend="hip_gfx1151",
            max_sequence_length=args.target_tokens + args.max_tokens + 4,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        runner = generator.runner
        config = runner.config
        graph_stats_before = (
            dict(runner.moe_graph_cache.stats)
            if args.depth_profile and runner.moe_graph_cache is not None
            else None
        )
        if args.depth_profile:
            runner.resident.ple_table.enable_telemetry()
        prompt, needle = build_retrieval_prompt(
            generator.tokenizer, target_tokens=args.target_tokens
        )
        token_ids = generator.tokenizer.encode(prompt)
        started = perf_counter()
        result = runner.prefill(token_ids)
        prefill_seconds = perf_counter() - started

        external = None
        tokenizer_match = None
        if args.teacher_logits is not None or args.teacher_tokens is not None:
            if args.teacher_logits is None or args.teacher_tokens is None:
                raise ValueError("teacher logits and tokens must be provided together")
            teacher_logits = np.fromfile(args.teacher_logits, dtype=np.float32)
            teacher_tokens = np.fromfile(args.teacher_tokens, dtype=np.int32)
            tokenizer_match = bool(
                np.array_equal(teacher_tokens, np.asarray(token_ids, dtype=np.int32))
            )
            external = compare_logits(teacher_logits, result.logits)

        ratio = config.qsa_compression_ratio
        needle_start = needle // ratio * ratio
        layer_selections = []
        selected_by_layer: dict[int, np.ndarray] = {}
        for layer, binding in sorted(runner.qsa_bindings.items()):
            state = runner.index_states[binding.qsa_state_index]
            count_host = np.empty(1, dtype=np.int32)
            copy_device_to_host(
                host_array_ptr(count_host), state.selected_count, runtime=runner.runtime
            )
            count = int(count_host[0])
            selected = np.empty(
                state.selected_positions.nbytes // np.dtype(np.int64).itemsize,
                dtype=np.int64,
            )
            copy_device_to_host(
                host_array_ptr(selected),
                state.selected_positions,
                runtime=runner.runtime,
            )
            selected = selected[:count]
            selected_by_layer[int(layer)] = selected
            layer_selections.append(
                {
                    "layer": int(layer),
                    "count": int(count),
                    "needle_selected": bool(
                        np.any(
                            (selected >= needle_start)
                            & (selected < needle_start + ratio)
                        )
                    ),
                }
            )

        layer, binding = max(runner.qsa_bindings.items())
        state = runner.index_states[binding.qsa_state_index]
        raw_physical = np.empty((state.capacity, state.index_dim), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(raw_physical), state.raw_keys, runtime=runner.runtime
        )
        raw_logical = raw_physical[state.physical_positions_host[: len(token_ids)]]
        norm = np.empty(state.index_dim, dtype=np.float32)
        runner.runtime.memcpy(
            host_array_ptr(norm),
            binding.mixer.index_k_norm_weight_ptr,
            norm.nbytes,
            MemcpyKind.DEVICE_TO_HOST,
        )
        query_rows = np.empty(
            (runner.prefill_chunk_size, state.index_heads, state.index_dim),
            dtype=np.float32,
        )
        copy_device_to_host(
            host_array_ptr(query_rows),
            runner.qsa_prefill_scratch.qsa.index_query,
            runtime=runner.runtime,
        )
        query = query_rows[(len(token_ids) - 1) % runner.prefill_chunk_size]
        prepared = qsa_prepare_index_keys(
            raw_logical,
            np.arange(len(token_ids), dtype=np.int64),
            norm,
            compression_ratio=ratio,
            rotary_dim=config.rope_dimension_count,
            theta=config.rope_freq_base,
        )
        cpu_scores = qsa_index_scores(query[None], prepared.keys)
        cpu_selection = qsa_select_positions(
            cpu_scores,
            prepared.block_starts,
            query_positions=[len(token_ids) - 1],
            available_positions=np.arange(len(token_ids), dtype=np.int64),
            compression_ratio=ratio,
            block_budget=config.qsa_block_budget,
        ).selected_positions[0]
        native = selected_by_layer[int(layer)][: cpu_selection.size].copy()
        native.sort()
        score_count = prepared.block_starts.size
        depth_profile = None
        if args.depth_profile:
            attention = runner.attention_states[binding.qsa_state_index]
            allocated_positions = len(attention.block_host) * int(attention.block_size)
            row_bytes = (
                int(attention.key_cache.nbytes) + int(attention.value_cache.nbytes)
            ) // allocated_positions
            locality = _selection_locality(native, kv_row_bytes=row_bytes)
            live_kv_bytes = sum(
                (
                    int(item.key_cache.nbytes) + int(item.value_cache.nbytes)
                )
                * len(token_ids)
                // int(item.max_positions)
                for item in runner.attention_states
            )
            depth_profile = {
                "score_rows": int(score_count),
                "score_rows_over_context": float(score_count / len(token_ids)),
                "selected_tokens": int(native.size),
                **locality,
                "selected_kv_row_bytes_per_layer": int(row_bytes),
                "live_kv_bytes_all_qsa_layers": int(live_kv_bytes),
                "allocated_kv_bytes_all_qsa_layers": int(sum(
                    int(item.key_cache.nbytes) + int(item.value_cache.nbytes)
                    for item in runner.attention_states
                )),
            }
        batched_selection = os.environ.get(
            "HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION", "1"
        ) not in {"", "0", "false", "False"}
        score_source = state.scores
        score_offset = 0
        if batched_selection:
            final_rows = (len(token_ids) - 1) % runner.prefill_chunk_size + 1
            final_start = len(token_ids) - final_rows
            dense_rows = max(
                0,
                min(final_rows, state.dense_equivalent_limit - final_start),
            )
            sparse_rows = final_rows - dense_rows
            if sparse_rows <= 0:
                raise RuntimeError("natural QSA gate did not reach a sparse score row")
            score_source = runner.qsa_prefill_metadata.scores
            score_offset = (sparse_rows - 1) * score_count * DType.FP32.itemsize
        runner.runtime.memcpy(
            host_array_ptr(state.scores_host),
            score_source.ptr + score_offset,
            score_count * np.dtype(np.float32).itemsize,
            MemcpyKind.DEVICE_TO_HOST,
        )
        scores = state.scores_host[:score_count]
        ranking = np.sort(scores)[::-1]
        cutoff_margin = float(
            ranking[config.qsa_block_budget - 1] - ranking[config.qsa_block_budget]
        )

        llama_tap = None
        if args.llama_topk_tap is not None:
            shape, llama_ids = _read_llama_tap(args.llama_topk_tap)
            llama_tap = {
                "shape": shape,
                "count": int(llama_ids.size),
                "unique_count": int(np.unique(llama_ids).size),
                "native_intersection": int(np.intersect1d(native, llama_ids).size),
                "llama_only_count": int(np.setdiff1d(llama_ids, native).size),
                "native_only_count": int(np.setdiff1d(native, llama_ids).size),
            }

        snapshot = runner.snapshot()
        initial = result

        def rollout() -> tuple[list[int], list[np.ndarray]]:
            current = initial
            generated: list[int] = []
            logits: list[np.ndarray] = []
            for index in range(args.max_tokens):
                logits.append(current.logits.copy())
                token = int(current.token_id)
                generated.append(token)
                if token == generator.tokenizer.eos_token_id:
                    break
                if index + 1 < args.max_tokens:
                    current = runner.step(token)
            return generated, logits

        generated, logits_a = rollout()
        runner.restore(snapshot)
        repeated, logits_b = rollout()
        repeat_exact = generated == repeated and all(
            np.array_equal(left, right) for left, right in zip(logits_a, logits_b)
        )
        runner.restore(snapshot)
        abandoned = (generated[0] + 17) % config.vocab_size
        runner.step(abandoned)
        runner.restore(snapshot)
        isolated, logits_c = rollout()
        isolation_exact = generated == isolated and all(
            np.array_equal(left, right) for left, right in zip(logits_a, logits_c)
        )
        text = generator.tokenizer.decode(generated, skip_special=False)
        answer = text.split("</think>", 1)[-1].replace("<|im_end|>", "").strip()
        peak = memory_stats()
        if depth_profile is not None:
            graph_stats_after = (
                dict(runner.moe_graph_cache.stats)
                if runner.moe_graph_cache is not None
                else None
            )
            depth_profile["graph_stats_before"] = graph_stats_before
            depth_profile["graph_stats_after"] = graph_stats_after
            depth_profile["ple_telemetry"] = runner.resident.ple_table.telemetry()
        generator.close()
        generator = None
        after = memory_stats()

        report = {
            "schema": 1,
            "model": str(args.model.resolve()),
            "target_tokens": int(args.target_tokens),
            "tokens": len(token_ids),
            "prefill_chunk_size": int(runner.prefill_chunk_size),
            "needle_token_offset": needle,
            "prefill_seconds": prefill_seconds,
            "tokenizer_match": tokenizer_match,
            "external_logits": external,
            "generated_ids": generated,
            "generated_text": text,
            "final_answer": answer,
            "retrieval_contains_code": "VIOLET-7391" in text,
            "retrieval_exact": answer == "VIOLET-7391",
            "needle_selected_layers": sum(
                item["needle_selected"] for item in layer_selections
            ),
            "qsa_layers": layer_selections,
            "cpu_index_oracle": {
                "source": "huggingface/transformers@fc5c5bde8e656dad91cbf34e61940d984b1c7b91",
                "layer": int(layer),
                "selected_count": int(cpu_selection.size),
                "exact_positions": bool(np.array_equal(native, cpu_selection)),
                "position_mismatches": int(np.count_nonzero(native != cpu_selection)),
                "score_mean_abs": float(np.mean(np.abs(scores - cpu_scores[0]))),
                "score_max_abs": float(np.max(np.abs(scores - cpu_scores[0]))),
                "cutoff_margin": cutoff_margin,
            },
            "llama_index_tap": llama_tap,
            "depth_profile": depth_profile,
            "transactional": {
                "repeat_exact": repeat_exact,
                "abandoned_token": int(abandoned),
                "isolation_exact": isolation_exact,
            },
            "memory_peak": peak,
            "memory_after_close": after,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        passed = (
            report["retrieval_exact"]
            and report["cpu_index_oracle"]["exact_positions"]
            and repeat_exact
            and isolation_exact
            and after["current_allocated_bytes"] == 0
        )
        if external is not None:
            passed = passed and bool(tokenizer_match) and external["top1_agreement"]
        return 0 if passed else 1
    finally:
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    raise SystemExit(main())
