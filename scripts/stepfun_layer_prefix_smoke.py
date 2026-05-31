#!/usr/bin/env python3
"""Run a partial StepFun GGUF resident layer-prefix prompt smoke.

This is a correctness bring-up diagnostic, not a full generation path or a
throughput benchmark: it executes a host-composed resident layer prefix and
explicitly reports which decoder layers are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.gguf import scan_gguf_splits
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.runtime.stepfun_gguf_runner import (
    StepFunResidentSession,
    stepfun_layer_prefix_slot_paths,
    stepfun_layer_slot_paths,
    stepfun_slot_tensor,
)

DEFAULT_GGUF_DIR = Path("/data/models/gguf")
DEFAULT_PATTERN = "Step-3.7-flash-Q3_K_L-*.gguf"
FORBIDDEN_TEXT_ONLY_FRAGMENTS = ("vision", "projector", "mmproj", "mtp", "nextn")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_GGUF_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--layer-count", type=int, default=4)
    parser.add_argument("--message", default="hello")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--sample-token-id",
        action="append",
        type=int,
        default=None,
        help="Token id to include in sampled final logits; may be repeated.",
    )
    parser.add_argument(
        "--max-resident-weight-gib",
        type=float,
        default=None,
        help="Refuse non-dry-run execution if selected resident weights exceed this GiB budget.",
    )
    parser.add_argument(
        "--stream-chunk-layers",
        type=int,
        default=None,
        help="Plan or execute root-plus-N-layer streaming chunks instead of all-resident layer weights.",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help="Scan metadata and print the layer-prefix slot/resource plan without HIP allocation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output to this path instead of stdout.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _scope(layer_count: int, block_count: int) -> str:
    if layer_count >= block_count:
        return f"layers_0_{block_count - 1}_prefix_no_skipped_layers"
    return (
        f"layers_0_{layer_count - 1}_prefix_only_layers_"
        f"{layer_count}_{block_count - 1}_skipped"
    )


def _skipped_layers(layer_count: int, block_count: int) -> list[int]:
    if layer_count >= block_count:
        return []
    return [layer_count, block_count - 1]


def _command(args: argparse.Namespace, *, dry_run: bool) -> str:
    parts = ["python3 scripts/stepfun_layer_prefix_smoke.py"]
    if dry_run:
        parts.append("--dry-run-plan")
    parts.extend(
        ["--layer-count", str(args.layer_count), "--message", json.dumps(args.message)]
    )
    if args.max_resident_weight_gib is not None:
        parts.extend(["--max-resident-weight-gib", f"{args.max_resident_weight_gib:g}"])
    if args.stream_chunk_layers is not None:
        parts.extend(["--stream-chunk-layers", str(args.stream_chunk_layers)])
    if args.output is not None:
        parts.extend(["--output", str(args.output)])
    if args.pretty:
        parts.append("--pretty")
    return " ".join(parts)


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def _slot_nbytes(model_map, slot: str) -> int:
    return int(stepfun_slot_tensor(model_map, slot).nbytes)


def _streaming_plan(model_map, *, layer_count: int, chunk_layers: int) -> dict[str, object]:
    if chunk_layers <= 0:
        raise ValueError("--stream-chunk-layers must be positive")
    root_slots = ("root.token_embedding", "root.output_norm", "root.lm_head")
    root_nbytes = sum(_slot_nbytes(model_map, slot) for slot in root_slots)
    chunks: list[dict[str, object]] = []
    for start in range(0, layer_count, chunk_layers):
        end = min(start + chunk_layers, layer_count)
        layer_slots: list[str] = []
        for layer_id in range(start, end):
            layer_slots.extend(stepfun_layer_slot_paths(model_map, layer_id))
        layer_nbytes = sum(_slot_nbytes(model_map, slot) for slot in layer_slots)
        chunks.append(
            {
                "start_layer": start,
                "end_layer_exclusive": end,
                "layer_count": end - start,
                "slot_count": len(layer_slots),
                "layer_weight_nbytes": int(layer_nbytes),
                "peak_with_roots_nbytes": int(root_nbytes + layer_nbytes),
            }
        )
    max_chunk = max(chunks, key=lambda chunk: int(chunk["peak_with_roots_nbytes"]))
    return {
        "chunk_layers": int(chunk_layers),
        "root_slots": list(root_slots),
        "root_resident_weight_nbytes": int(root_nbytes),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "max_chunk": max_chunk,
        "peak_resident_weight_nbytes": int(max_chunk["peak_with_roots_nbytes"]),
        "note": (
            "Metadata-only estimate for keeping root tensors resident and loading a fixed-size layer chunk "
            "at a time; non-dry-run prefix smokes can execute this path for small prefixes."
        ),
    }


def _run_chunked_prefix(
    *,
    paths: tuple[Path, ...],
    model_map,
    args: argparse.Namespace,
    runtime,
) -> tuple[SimpleNamespace, list[dict[str, object]], int, dict[str, object]]:
    root_slots = ("root.token_embedding", "root.output_norm", "root.lm_head")
    positions = None
    chunk_records: list[dict[str, object]] = []
    root_session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=root_slots,
        runtime=runtime,
    )
    peak_resident_nbytes = root_session.weights.allocated_nbytes
    try:
        prompt = root_session.embed_chat_prompt_bf16(
            [{"role": "user", "content": args.message}],
            reasoning_effort=args.reasoning_effort,
            runtime=runtime,
        )
        hidden_bits = prompt.embeddings_bf16
        positions = np.arange(prompt.prompt_length, dtype=np.int64)
        layer_hidden = None
        for start in range(0, args.layer_count, args.stream_chunk_layers):
            end = min(start + args.stream_chunk_layers, args.layer_count)
            chunk_slots: list[str] = []
            for layer_id in range(start, end):
                chunk_slots.extend(stepfun_layer_slot_paths(model_map, layer_id))
            layer_session = StepFunResidentSession.from_gguf_paths(
                paths,
                selected_slots=chunk_slots,
                runtime=runtime,
            )
            chunk_peak = root_session.weights.allocated_nbytes + layer_session.weights.allocated_nbytes
            peak_resident_nbytes = max(peak_resident_nbytes, chunk_peak)
            try:
                for layer_id in range(start, end):
                    layer_hidden = layer_session.layer_prefill_probe_bf16(
                        layer_id,
                        hidden_bits,
                        positions=positions,
                        runtime=runtime,
                    )
                    hidden_bits = float_array_to_bf16_bits(np.asarray(layer_hidden, dtype=np.float32))
            finally:
                layer_session.free(runtime=runtime)
            chunk_records.append(
                {
                    "start_layer": start,
                    "end_layer_exclusive": end,
                    "slot_count": len(chunk_slots),
                    "layer_weight_nbytes": layer_session.weights.allocated_nbytes,
                    "peak_with_roots_nbytes": chunk_peak,
                }
            )
        if layer_hidden is None:  # pragma: no cover - layer_count validation happens upstream
            raise RuntimeError("chunked StepFun prefix produced no hidden state")
        logits = root_session.final_logits_probe_bf16(hidden_bits[-1:].copy(), runtime=runtime)
        next_token_id = int(np.argmax(logits[-1]))
        probe = SimpleNamespace(
            prompt=prompt,
            layer_hidden=layer_hidden,
            logits=logits,
            next_token_id=next_token_id,
            next_token_logit=float(logits[-1, next_token_id]),
        )
        return probe, chunk_records, int(peak_resident_nbytes), memory_stats()
    finally:
        root_session.free(runtime=runtime)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = tuple(sorted(args.model_dir.glob(args.pattern)))
    if not paths:
        raise FileNotFoundError(f"no GGUF shards matched {args.model_dir / args.pattern}")

    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    selected_slots = stepfun_layer_prefix_slot_paths(model_map, args.layer_count)
    no_modal_slots = not any(
        fragment in slot for slot in selected_slots for fragment in FORBIDDEN_TEXT_ONLY_FRAGMENTS
    )
    if not no_modal_slots:
        raise RuntimeError("layer-prefix text smoke selected a vision/projector/MTP slot")
    resident_weight_nbytes = sum(_slot_nbytes(model_map, slot) for slot in selected_slots)
    streaming_plan = None
    if args.stream_chunk_layers is not None:
        streaming_plan = _streaming_plan(
            model_map,
            layer_count=args.layer_count,
            chunk_layers=args.stream_chunk_layers,
        )
    if args.dry_run_plan:
        result = {
            "status": "planned",
            "scope": _scope(args.layer_count, model_map.config.block_count),
            "model": args.pattern.removesuffix("-*.gguf"),
            "backend": "hip_gfx1151",
            "command": _command(args, dry_run=True),
            "paths": [str(path) for path in paths],
            "split_count": info.split_count,
            "tensor_count": info.tensor_count,
            "layer_count": args.layer_count,
            "skipped_layers": _skipped_layers(args.layer_count, model_map.config.block_count),
            "selected_slot_count": len(selected_slots),
            "selected_slots": list(selected_slots),
            "no_vision_projector_mtp_slots": True,
            "resident_weight_nbytes": int(resident_weight_nbytes),
            "resident_weight_gib": resident_weight_nbytes / 2**30,
            "note": (
                "Metadata-only layer-prefix prompt plan: no HIP runtime was initialized and no weights, "
                "KV buffers, prompt embeddings, or logits were allocated/computed."
            ),
        }
        if streaming_plan is not None:
            result["streaming_plan"] = streaming_plan
        _emit_json(result, pretty=args.pretty, output=args.output)
        return 0

    execution_resident_nbytes = (
        resident_weight_nbytes
        if streaming_plan is None
        else int(streaming_plan["peak_resident_weight_nbytes"])
    )
    if args.max_resident_weight_gib is not None:
        budget_nbytes = int(args.max_resident_weight_gib * 2**30)
        if execution_resident_nbytes > budget_nbytes:
            raise MemoryError(
                "layer-prefix smoke selected "
                f"{execution_resident_nbytes / 2**30:.3f} GiB of peak resident weights, "
                f"which exceeds --max-resident-weight-gib={args.max_resident_weight_gib:.3f}; "
                "rerun with --dry-run-plan or a larger explicit budget"
            )

    runtime = get_hip_runtime()
    reset_memory_stats()
    free_before, total = runtime.mem_get_info()
    started = time.perf_counter()
    session = None
    chunk_records: list[dict[str, object]] = []
    peak_resident_weight_nbytes = execution_resident_nbytes
    stats_before_free = None
    if args.stream_chunk_layers is None:
        session = StepFunResidentSession.from_gguf_paths(
            paths,
            selected_slots=selected_slots,
            runtime=runtime,
        )
        free_after_load, _ = runtime.mem_get_info()
        try:
            probe = session.layer_prefix_prompt_logits_probe_bf16(
                [{"role": "user", "content": args.message}],
                layer_count=args.layer_count,
                reasoning_effort=args.reasoning_effort,
                runtime=runtime,
            )
            loaded_resident_nbytes = session.weights.allocated_nbytes
        except Exception:
            session.free(runtime=runtime)
            session = None
            raise
    else:
        probe, chunk_records, peak_resident_weight_nbytes, stats_before_free = _run_chunked_prefix(
            paths=paths,
            model_map=model_map,
            args=args,
            runtime=runtime,
        )
        free_after_load, _ = runtime.mem_get_info()
        loaded_resident_nbytes = peak_resident_weight_nbytes
    sample_ids = args.sample_token_id or [0, 1, 128007, model_map.config.vocab_size - 1]
    sample_ids = [
        token_id if token_id >= 0 else model_map.config.vocab_size + token_id
        for token_id in sample_ids
    ]
    if any(token_id < 0 or token_id >= model_map.config.vocab_size for token_id in sample_ids):
        raise ValueError(
            f"sample token ids out of range for vocab_size={model_map.config.vocab_size}: {sample_ids}"
        )
    try:
        result = {
            "status": "partial_prompt_smoke",
            "scope": _scope(args.layer_count, model_map.config.block_count),
            "model": args.pattern.removesuffix("-*.gguf"),
            "backend": "hip_gfx1151",
            "command": _command(args, dry_run=False),
            "paths": [str(path) for path in paths],
            "split_count": info.split_count,
            "tensor_count": info.tensor_count,
            "layer_count": args.layer_count,
            "skipped_layers": _skipped_layers(args.layer_count, model_map.config.block_count),
            "execution_mode": "all_resident" if args.stream_chunk_layers is None else "chunked",
            "stream_chunk_layers": args.stream_chunk_layers,
            "selected_slot_count": len(selected_slots),
            "selected_slots": list(selected_slots),
            "no_vision_projector_mtp_slots": True,
            "resident_weight_nbytes": loaded_resident_nbytes,
            "all_resident_weight_nbytes": int(resident_weight_nbytes),
            "peak_resident_weight_nbytes": int(peak_resident_weight_nbytes),
            "chunk_records": chunk_records,
            "prompt": probe.prompt.rendered_prompt,
            "input_ids": [int(token_id) for token_id in probe.prompt.input_ids],
            "prompt_length": probe.prompt.prompt_length,
            "layer_hidden_shape": list(probe.layer_hidden.shape),
            "logits_shape": list(probe.logits.shape),
            "sampled_logits": {
                str(token_id): float(probe.logits[0, token_id]) for token_id in sample_ids
            },
            "next_token_id": probe.next_token_id,
            "next_token_logit": probe.next_token_logit,
            "hip_total_gib": total / 2**30,
            "hip_free_before_gib": free_before / 2**30,
            "hip_free_after_load_gib": free_after_load / 2**30,
            "elapsed_s": time.perf_counter() - started,
            "memory_stats_before_free": stats_before_free or memory_stats(),
            "hip_mem_get_info_note": (
                "hipMemGetInfo total/free are known to be contradictory under the configured Strix Halo GTT "
                "setup; use hipEngine allocation stats and relative before/load/free readings as diagnostic only."
            ),
            "note": (
                "Partial correctness smoke only: host-composed layer-prefix bridge uses resident root/layer "
                "weights and final sampled logits; skipped layers and KV-backed decode remain open, so this "
                "is not full next-token parity or throughput evidence."
            ),
        }
    finally:
        if session is not None:
            session.free(runtime=runtime)

    free_after_free, _ = runtime.mem_get_info()
    result["hip_free_after_free_gib"] = free_after_free / 2**30
    result["memory_stats_after_free"] = memory_stats()
    _emit_json(result, pretty=args.pretty, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
