#!/usr/bin/env python3
"""Qualify natural Qwen4Exp QSA retrieval, control, and transaction ownership.

The selection oracle follows Transformers ``Qwen4ExpTextQSAIndexer`` at
huggingface/transformers@fc5c5bde8e656dad91cbf34e61940d984b1c7b91:
select a whole-block token budget and append only the actual incomplete tail.

``--index-only`` runs the same prefill and index oracle at a full-capacity
prompt with no decode, and reports generation, repeat, rollback, and retrieval
as ``not_exercised``. Its pass predicate covers only exact CPU-reference index
positions, finite logits, cursor/count consistency, and a clean teardown.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
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
from hipengine.runtime.qwen4_exp_runner import qsa_batched_selection
from scripts.qwen4_exp_compare_logits import compare_logits


_NEEDLE_CODE = "VIOLET-7391"
_NEEDLE_RECORD_INDEX = 24
_ARCHIVE_RECORD_COUNT = 100
_PROMPT_PREFIX = (
    "<|im_start|>system\nReasoning effort is set to low. Keep your thinking "
    "brief and focused, moving directly to the conclusion without unnecessary "
    "elaboration. You retrieve exact facts from long archives.<|im_end|>\n"
    "<|im_start|>user\nYou are auditing an archive. Read all records and "
    "answer the final question exactly.\n"
)
_PROMPT_SUFFIX = (
    "\nFinal audit question: What is the unique access code? "
    "Answer with only the exact code.<|im_end|>\n<|im_start|>assistant\n"
)
_PROMPT_FILLER = (
    "The archive checksum remains stable and the ordinary inventory records "
    "require no action. "
)


def _archive_record(index: int) -> str:
    if index == _NEEDLE_RECORD_INDEX:
        return (
            f"Archive record {index:04d}: The unique access code is "
            f"{_NEEDLE_CODE}. Preserve this exact code for the audit.\n"
        )
    return (
        f"Archive record {index:04d}: Inventory batch {10000 + index} is "
        "ordinary, stable, and requires no action.\n"
    )


def _render_retrieval_prompt(records: list[str]) -> str:
    return _PROMPT_PREFIX + "".join(records) + _PROMPT_SUFFIX


def retrieval_prompt_minimum_tokens(tokenizer: object) -> int:
    """Tokens the mandatory payload needs: prefix, needle record, question."""

    encode = getattr(tokenizer, "encode")
    return len(
        encode(
            _render_retrieval_prompt([_archive_record(_NEEDLE_RECORD_INDEX)])
        )
    )


def _filler_count(encode, records: list[str], *, target_tokens: int) -> int:
    """Largest filler count whose rendered prompt still fits ``target_tokens``.

    A bounded exponential probe plus binary search costs O(log k) prompt
    encodes; the linear walk this replaces re-encoded the whole prompt once per
    filler, which is ~15k encodes at a 262K target.

    The probe assumes the tokenizer is monotone in the filler count, which holds
    for this prompt: every filler is the same literal sentence, and appending
    one adds at least one token. Under that assumption the search returns exactly
    the count the linear walk returned. A tokenizer that could make a longer
    prompt shorter is *not* detected - there is no scan below ``low`` - so the
    probe stops at ``target_tokens + 1`` and returns the last count it proved
    fits. Ordinary targets are compared against the linear walk in
    ``tests/test_unit_qwen4exp_qsa_gate_index_only.py``.
    """

    def fits(count: int) -> bool:
        return len(
            encode(_render_retrieval_prompt(records + [_PROMPT_FILLER] * count))
        ) <= target_tokens

    if not fits(0):
        return 0
    limit = int(target_tokens) + 1
    low, high = 0, 1
    while high < limit and fits(high):
        low, high = high, high * 2
    if high >= limit:
        # Only reachable when nothing in range ever fails to fit; keep the probe
        # bounded instead of doubling forever.
        high = limit
        if fits(high):
            return high
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def build_retrieval_prompt(tokenizer: object, *, target_tokens: int) -> tuple[str, int]:
    """Build a prompt that fits ``target_tokens`` and keeps the needle.

    The needle record and the final question are mandatory. Ordinary records are
    added around the needle in index order while they fit, then stable filler
    fills the remaining budget, so a small target drops records instead of
    returning an oversized prompt. A target below
    :func:`retrieval_prompt_minimum_tokens` raises rather than overrunning the
    sequence the caller sized from it.
    """

    encode = getattr(tokenizer, "encode")
    minimum = retrieval_prompt_minimum_tokens(tokenizer)
    if target_tokens < minimum:
        raise ValueError(
            f"QSA retrieval prompt needs at least {minimum} tokens for the "
            "system prefix, needle record, and final question; "
            f"target_tokens={target_tokens} cannot hold them"
        )
    needle_record = _archive_record(_NEEDLE_RECORD_INDEX)
    before: list[str] = []
    for index in range(_NEEDLE_RECORD_INDEX):
        candidate = before + [_archive_record(index)]
        if (
            len(encode(_render_retrieval_prompt(candidate + [needle_record])))
            > target_tokens
        ):
            break
        before = candidate
    after: list[str] = []
    for index in range(_NEEDLE_RECORD_INDEX + 1, _ARCHIVE_RECORD_COUNT):
        candidate = after + [_archive_record(index)]
        if len(
            encode(_render_retrieval_prompt(before + [needle_record] + candidate))
        ) > target_tokens:
            break
        after = candidate
    records = before + [needle_record] + after
    records += [_PROMPT_FILLER] * _filler_count(
        encode, records, target_tokens=target_tokens
    )
    prompt = _render_retrieval_prompt(records)
    prompt_tokens = len(encode(prompt))
    for count in range(64):
        candidate = _render_retrieval_prompt(records + [" a" * count])
        candidate_tokens = len(encode(candidate))
        if candidate_tokens <= target_tokens and candidate_tokens > prompt_tokens:
            prompt = candidate
            prompt_tokens = candidate_tokens
    needle = len(encode(prompt[: prompt.index(_NEEDLE_CODE)]))
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


def _prefill_progress_line(done: int, total: int, elapsed: float) -> str:
    """One sampled prefill line: percent, token counts, and elapsed seconds."""

    share = 0.0 if total <= 0 else 100.0 * done / total
    return f"prefill: {share:5.1f}% {done}/{total} tokens at {elapsed:7.1f}s"


@contextlib.contextmanager
def _prefill_progress(
    runner: object, total: int, progress, *, interval: float = 30.0
):
    """Sample ``runner.position`` while a long prefill runs.

    A full-capacity prefill takes hours, so silence is not acceptable. The
    sampler only reads the counter the runner already advances after each chunk;
    it adds no callback and changes no runtime path. Sampling lags by at most the
    chunk in flight.
    """

    stop = threading.Event()
    started = perf_counter()

    def report() -> None:
        while not stop.wait(interval):
            done = int(getattr(runner, "position", 0))
            progress(_prefill_progress_line(done, total, perf_counter() - started))

    thread = threading.Thread(target=report, name="qsa-prefill-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5.0)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--per-row-selection", action="store_true",
        help=(
            "Measure the strict per-row selection fallback; the scores then come "
            "from the index state rather than the batched prefill metadata"
        ),
    )
    parser.add_argument(
        "--index-only", action="store_true",
        help=(
            "Prefill an exactly --target-tokens prompt and stop: no decode, no "
            "repeat/rollback, no retrieval claim. Passes on exact CPU-reference "
            "index positions, finite logits, cursor/count consistency, and a "
            "clean teardown; generation and retrieval are reported not_exercised"
        ),
    )
    return parser


def _sequence_length(args: argparse.Namespace) -> int:
    """Context the generator is sized to.

    Index-only prefills the whole prompt and never decodes, so the sequence is
    exactly the prompt. That matters at a full-capacity target, which already
    equals the model's native context: generation headroom there would exceed
    the native context and be rejected instead of admitted.
    """

    if args.index_only:
        return int(args.target_tokens)
    return int(args.target_tokens) + int(args.max_tokens) + 4


def _index_state_exact(runner: object, token_count: int) -> dict[str, object]:
    """Runner cursor and per-layer index counts after a full prefill.

    This is cursor/count consistency, not tensor-state equality: it proves the
    runner cursor and every layer's contiguous index count reached
    ``token_count``, and reports the block rung that count covers
    (``count // ratio``) alongside the allocated index capacity. Device tensor
    contents are not compared here.
    """

    states = [
        runner.index_states[binding.qsa_state_index]
        for _, binding in sorted(runner.qsa_bindings.items())
    ]
    counts = [int(state.count) for state in states]
    capacities = sorted({int(state.capacity) for state in states})
    blocks = sorted(
        {
            int(state.count) // int(state.compression_ratio)
            for state in states
        }
    )
    position = int(runner.position)
    return {
        "position": position,
        "index_counts": counts,
        "index_capacity": capacities,
        "complete_blocks": blocks,
        "state_scope": "cursor_count_consistency",
        "state_exact": bool(
            position == token_count
            and counts
            and all(count == token_count for count in counts)
        ),
    }


def _require_exact_budget(token_count: int, target_tokens: int) -> None:
    """Index-only refuses a prompt that is not exactly the requested rung."""

    if token_count == target_tokens:
        return
    raise SystemExit(
        f"--index-only needs an exactly {target_tokens}-token prompt; the "
        f"builder produced {token_count}. Raise --target-tokens to the capacity "
        "rung you mean, or drop --index-only: a short prompt would misreport "
        "that rung"
    )


def index_only_passed(report: dict[str, object]) -> bool:
    """Index-only pass: index parity, finite logits, cursor/count consistency.

    Generation, repeat, rollback, and retrieval are not exercised in this mode
    and are reported as ``not_exercised``; they never enter this predicate. The
    state arm is cursor/count consistency (``state_scope``), not tensor-state
    equality. A teacher comparison, when supplied, is recorded but not gating
    here: index parity is the claim this mode makes. Each arm is compared with
    ``is True`` so a ``not_exercised`` marker can never pass for a measured
    boolean.
    """

    index = report["index_only"]
    return (
        report["cpu_index_oracle"]["exact_positions"] is True
        and index["logits_finite"] is True
        and index["state_exact"] is True
        and index["exact_token_budget"] is True
        and report["memory_after_close"]["current_allocated_bytes"] == 0
    )


def retrieval_passed(
    report: dict[str, object], external: dict[str, object] | None
) -> bool:
    """Retrieval pass: the default gate's published predicate, unchanged.

    ``is True`` keeps an unexercised marker from passing as a measured boolean.
    """

    passed = (
        report["retrieval_exact"] is True
        and report["cpu_index_oracle"]["exact_positions"] is True
        and report["transactional"]["repeat_exact"] is True
        and report["transactional"]["isolation_exact"] is True
        and report["memory_after_close"]["current_allocated_bytes"] == 0
    )
    if external is not None:
        passed = (
            passed
            and report["tokenizer_match"] is True
            and external["top1_agreement"] is True
        )
    return passed


_NOT_EXERCISED = "not_exercised"


def index_only_fields(
    *,
    index_state: dict[str, object],
    logits_finite: bool,
    token_count: int,
    target_tokens: int,
    tokenizer_match: bool | None,
    external: dict[str, object] | None,
) -> dict[str, object]:
    """Report fields for a run that prefills and never decodes.

    Generation, retrieval, repeat, and rollback carry the ``not_exercised``
    marker rather than ``true`` or ``false``: this mode makes no claim about
    them, and a boolean there would read as a measured result.
    """

    return {
        "tokenizer_match": tokenizer_match,
        "external_logits": external,
        "generated_ids": _NOT_EXERCISED,
        "generated_text": _NOT_EXERCISED,
        "final_answer": _NOT_EXERCISED,
        "retrieval_contains_code": _NOT_EXERCISED,
        "retrieval_exact": _NOT_EXERCISED,
        "transactional": {
            "status": _NOT_EXERCISED,
            "reason": (
                "index-only mode prefills a full-capacity prompt and never "
                "decodes, so the repeat and rollback arms do not run"
            ),
            "repeat_exact": _NOT_EXERCISED,
            "abandoned_token": _NOT_EXERCISED,
            "isolation_exact": _NOT_EXERCISED,
        },
        "index_only": {
            **index_state,
            "logits_finite": logits_finite,
            "exact_token_budget": bool(token_count == int(target_tokens)),
            "decode_exercised": False,
            "teacher_comparison_gating": False,
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    started = perf_counter()

    def progress(message: str) -> None:
        print(
            f"[qsa-gate] {perf_counter() - started:8.1f}s {message}",
            file=sys.stderr,
            flush=True,
        )

    info = load_gguf_index(discover_gguf_files(args.model)[0])
    generator = None
    reset_memory_stats()
    try:
        sequence_length = _sequence_length(args)
        progress(
            f"generator: context {sequence_length}, chunk {args.prefill_chunk_size}"
        )
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=args.model,
            weight_index=info,
            model_plugin=resolve_model(info.architecture or ""),
            backend="hip_gfx1151",
            max_sequence_length=sequence_length,
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
        if args.index_only:
            _require_exact_budget(len(token_ids), int(args.target_tokens))
        progress(
            f"prompt: {len(token_ids)} tokens, needle at {needle}, "
            f"prefill chunk {runner.prefill_chunk_size}"
        )
        selection = (
            qsa_batched_selection(runner, False)
            if args.per_row_selection
            else contextlib.nullcontext()
        )
        progress(f"prefill: start {len(token_ids)} tokens")
        prefill_started = perf_counter()
        with _prefill_progress(runner, len(token_ids), progress), selection:
            result = runner.prefill(token_ids)
            # Read the arm inside the scope: the runner's published value comes
            # back when it closes.
            batched_selection = bool(
                getattr(runner, "qsa_batched_selection", True)
            )
        prefill_seconds = perf_counter() - prefill_started
        progress(f"prefill: done in {prefill_seconds:.1f}s")

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

        progress("index oracle: reading per-layer selections")
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
        progress("index oracle: cpu reference selection")
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

        snapshot = None if args.index_only else runner.snapshot()
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

        generated: list[int] = []
        logits_a: list[np.ndarray] = []
        repeat_exact: object = "not_exercised"
        isolation_exact: object = "not_exercised"
        abandoned: object = "not_exercised"
        text: object = "not_exercised"
        answer: object = "not_exercised"
        if args.index_only:
            progress("index-only: skipping decode, repeat, and rollback arms")
        else:
            progress(f"rollout: {args.max_tokens} tokens, then repeat and rollback")
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
        index_state = (
            _index_state_exact(runner, len(token_ids)) if args.index_only else None
        )
        logits_finite = bool(np.all(np.isfinite(result.logits)))
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
        prefill_chunk_size = int(runner.prefill_chunk_size)
        generator.close()
        generator = None
        after = memory_stats()
        progress(
            "teardown: allocated bytes after close "
            f"{after['current_allocated_bytes']}"
        )

        report = {
            "schema": 1,
            "mode": "index_only" if args.index_only else "retrieval",
            "model": str(args.model.resolve()),
            "target_tokens": int(args.target_tokens),
            "tokens": len(token_ids),
            "prefill_chunk_size": prefill_chunk_size,
            "batched_selection": batched_selection,
            "needle_token_offset": needle,
            "prefill_seconds": prefill_seconds,
            "sequence_length": sequence_length,
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
            "memory_peak": peak,
            "memory_after_close": after,
        }
        if args.index_only:
            report.update(
                index_only_fields(
                    index_state=index_state or {},
                    logits_finite=logits_finite,
                    token_count=len(token_ids),
                    target_tokens=int(args.target_tokens),
                    tokenizer_match=tokenizer_match,
                    external=external,
                )
            )
        else:
            report.update(
                {
                    "tokenizer_match": tokenizer_match,
                    "external_logits": external,
                    "generated_ids": generated,
                    "generated_text": text,
                    "final_answer": answer,
                    "retrieval_contains_code": "VIOLET-7391" in text,
                    "retrieval_exact": answer == "VIOLET-7391",
                    "transactional": {
                        "repeat_exact": repeat_exact,
                        "abandoned_token": int(abandoned),
                        "isolation_exact": isolation_exact,
                    },
                }
            )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        passed = (
            index_only_passed(report)
            if args.index_only
            else retrieval_passed(report, external)
        )
        progress(f"pass: {passed}")
        return 0 if passed else 1
    finally:
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    raise SystemExit(main())
