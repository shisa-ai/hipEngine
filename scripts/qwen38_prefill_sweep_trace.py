#!/usr/bin/env python3
"""Capture Qwen3.8 prefill wall and HIP-event spans at fixed row points."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

from hipengine.loading.gguf import GGUFReader
from hipengine.runtime import qwen35_gguf_runner as qwen_runner
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows

DEFAULT_ROWS = (16, 35, 48, 72, 96, 256, 288, 536, 1024)


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item.strip())
    if not rows or len(set(rows)) != len(rows) or any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("rows must be distinct positive integers")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_family_inventory(reader: GGUFReader) -> dict[str, dict[str, int]]:
    inventory = {
        family: {"tensor_count": 0, "resident_bytes": 0, "logical_elements": 0}
        for family in ("q4", "q5", "q6", "other")
    }
    for tensor in reader.info.tensors:
        type_name = tensor.ggml_type_name.lower()
        family = next(
            (candidate for candidate in ("q4", "q5", "q6") if type_name.startswith(candidate)),
            "other",
        )
        inventory[family]["tensor_count"] += 1
        inventory[family]["resident_bytes"] += int(tensor.nbytes)
        inventory[family]["logical_elements"] += int(tensor.n_elements)
    return inventory


def _token_stream(
    reader: GGUFReader, prompts: Path, rows: int
) -> tuple[list[int], dict[str, Any]]:
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    prompt_rows = load_prompt_rows(prompts)
    encoded = [
        list(build_chat_prompt(tokenizer, str(prompt["prompt"]), reasoning="off"))
        for prompt in prompt_rows
    ]
    if not encoded or any(not tokens for tokens in encoded):
        raise RuntimeError("the frozen prompt suite must contain non-empty tokenized prompts")
    stream: list[int] = []
    index = 0
    while len(stream) < rows:
        tokens = encoded[index % len(encoded)]
        stream.extend(tokens[: rows - len(stream)])
        index += 1
    return stream, {
        "prompt_ids": [str(prompt["id"]) for prompt in prompt_rows],
        "prompt_categories": [str(prompt["category"]) for prompt in prompt_rows],
        "construction": "round_robin_complete_or_truncated_chat_prompts",
        "sha256": hashlib.sha256(
            json.dumps(stream, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _weight_family(weight: Any) -> str:
    type_name = str(weight.spec.source.ggml_type_name).lower()
    return next(
        (family for family in ("q4", "q5", "q6") if type_name.startswith(family)),
        "other",
    )


def _record_weights(ledger: dict[str, dict[str, Any]], weights: Sequence[Any], rows: int) -> None:
    grouped: dict[str, dict[str, int]] = {}
    for weight in weights:
        family = _weight_family(weight)
        source = weight.spec.source
        entry = grouped.setdefault(family, {"weight_count": 0, "active_weight_bytes": 0, "logical_elements": 0})
        entry["weight_count"] += 1
        entry["active_weight_bytes"] += int(source.nbytes)
        entry["logical_elements"] += int(source.n_elements)
    for family, entry in grouped.items():
        entry["rows"] = int(rows)
        ledger[family]["calls"] += int(entry["weight_count"])
        ledger[family]["active_weight_bytes"] += int(entry["active_weight_bytes"])
        ledger[family]["logical_elements"] += int(entry["logical_elements"])
        ledger[family]["logical_row_elements"] += int(entry["logical_elements"]) * int(rows)
        ledger[family]["entries"].append(entry)


@contextlib.contextmanager
def _linear_weight_ledger(ledger: dict[str, dict[str, int]]):
    specs = {
        "launch_gguf_linear": ((0,), 3, False),
        "launch_gguf_linear_residual": ((0,), 4, True),
        "launch_gguf_linear_pair": ((0, 1), 5, True),
        "launch_gguf_linear_pair_concat": ((0, 1), 4, True),
        "launch_gguf_linear_pair_silu": ((0, 1), 4, True),
        "launch_gguf_linear_triple": ((0, 1, 2), 7, True),
    }
    originals: dict[str, Any] = {}
    try:
        for name, (weight_indices, rows_index, conditional) in specs.items():
            original = getattr(qwen_runner, name)
            originals[name] = original

            def make_wrapper(original, weight_indices, rows_index, conditional):
                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if not conditional or bool(result):
                        rows = int(kwargs["rows"] if "rows" in kwargs else args[rows_index])
                        _record_weights(ledger, [args[index] for index in weight_indices], rows)
                    return result

                return wrapped

            setattr(
                qwen_runner,
                name,
                make_wrapper(original, weight_indices, rows_index, conditional),
            )
        yield
    finally:
        for name, original in originals.items():
            setattr(qwen_runner, name, original)


def capture(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.resolve()
    prompts = args.prompts.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    if not prompts.is_file():
        raise FileNotFoundError(f"prompt suite not found: {prompts}")
    if max(args.rows) > args.max_sequence_length:
        raise ValueError("the largest row point exceeds --max-sequence-length")
    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    reader = GGUFReader(model)
    tokens, token_source = _token_stream(reader, prompts, max(args.rows))
    family_inventory = _model_family_inventory(reader)
    records: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        model,
        backend=args.backend,
        max_sequence_length=args.max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        runtime = session.runtime
        if runtime is None:
            raise RuntimeError("the resident session did not expose a HIP runtime")
        for row in (args.rows[0],):
            session.prefill(tokens[:row], use_bulk=True, bulk_attention_mode="bulk")
            runtime.device_synchronize()
        for row in args.rows:
            weight_ledger = {
                family: {
                    "calls": 0,
                    "active_weight_bytes": 0,
                    "logical_elements": 0,
                    "logical_row_elements": 0,
                    "entries": [],
                }
                for family in ("q4", "q5", "q6", "other")
            }
            start_event = runtime.event_create()
            stop_event = runtime.event_create()
            try:
                session.reset()
                runtime.event_record(start_event, 0)
                start_ns = time.perf_counter_ns()
                with _linear_weight_ledger(weight_ledger):
                    result = session.prefill(
                        tokens[:row],
                        use_bulk=True,
                        bulk_attention_mode="bulk",
                        return_logits=False,
                        record_gpu_stage_timings=True,
                    )
                runtime.event_record(stop_event, 0)
                runtime.event_synchronize(stop_event)
                stop_ns = time.perf_counter_ns()
                gpu_ms = runtime.event_elapsed_time_ms(start_event, stop_event)
            finally:
                runtime.event_destroy(stop_event)
                runtime.event_destroy(start_event)
            records.append(
                {
                    "rows": row,
                    "start_monotonic_ns": start_ns,
                    "stop_monotonic_ns": stop_ns,
                    "wall_ms": (stop_ns - start_ns) / 1e6,
                    "gpu_span_ms": gpu_ms,
                    "wall_minus_gpu_ms": (stop_ns - start_ns) / 1e6 - gpu_ms,
                    "next_token_id": int(result.token_id),
                    "weight_ledger": weight_ledger,
                    "gpu_stage_timings_ms": dict(session.last_prefill_gpu_stage_timings_ms),
                }
            )
            print(
                f"rows={row} wall_ms={records[-1]['wall_ms']:.3f} "
                f"gpu_ms={gpu_ms:.3f}",
                flush=True,
            )
    return {
        "schema": 1,
        "kind": "qwen38-prefill-sweep-trace-raw",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "backend": args.backend,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "model": str(model),
        "model_sha256": _sha256(model),
        "prompts": str(prompts),
        "rows": list(args.rows),
        "token_source": token_source,
        "model_family_inventory": family_inventory,
        "timing": "host perf_counter_ns around prefill plus default-stream HIP events",
        "records": records,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"),
    )
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--max-sequence-length", type=int, default=1152)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
