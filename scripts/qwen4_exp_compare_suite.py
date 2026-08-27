#!/usr/bin/env python3
"""Run the canonical Qwen4Exp multi-prompt full-logit correctness gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import tempfile
from typing import Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4_exp_compare_logits import (
    _DEFAULT_LLAMA_DEBUG,
    _run_llama_debug,
    compare_logits,
)

_DEFAULT_PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get("text")
        if prompt is None:
            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"prompt line {number} has no text/messages")
            contents = [
                str(message["content"])
                for message in messages
                if message.get("role") in {"system", "developer", "user"}
            ]
            if not contents:
                raise ValueError(f"prompt line {number} has no input message")
            prompt = "\n".join(contents)
        rows.append(
            {
                "id": str(item.get("id", f"line_{number}")),
                "category": str(item.get("category", "uncategorized")),
                "prompt": str(prompt),
            }
        )
    if not rows:
        raise ValueError("prompt suite is empty")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    return rows


def aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one result row is required")
    kl = np.asarray([float(row["kl_teacher_to_hipengine"]) for row in rows])
    agreement = np.asarray([bool(row["top1_agreement"]) for row in rows])
    return {
        "count": len(rows),
        "mean_kl": float(np.mean(kl)),
        "p95_kl": float(np.percentile(kl, 95)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(np.max(kl)),
        "top1_agreement_rate": float(np.mean(agreement)),
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompts", type=Path, default=_DEFAULT_PROMPTS)
    parser.add_argument("--context", type=int, default=2051)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--llama-debug", type=Path, default=_DEFAULT_LLAMA_DEBUG)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--min-top1", type=float, default=0.9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    model_path = args.model.expanduser().resolve()
    parts = discover_gguf_files(model_path)
    prompts = load_prompts(args.prompts.expanduser().resolve())
    teachers: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    with tempfile.TemporaryDirectory(prefix="qwen4exp-suite-teacher-") as temporary:
        root = Path(temporary)
        for index, row in enumerate(prompts):
            output = root / f"{index:03d}-{row['id']}"
            output.mkdir()
            teachers.append(
                _run_llama_debug(
                    args.llama_debug.expanduser().resolve(),
                    parts[0],
                    row["prompt"],
                    output,
                    args.context,
                )
            )

        runtime = get_hip_runtime()
        free_before, total_bytes = runtime.mem_get_info()
        reset_memory_stats()
        generator = None
        try:
            info = load_gguf_index(parts[0])
            generator = Qwen4ExpGGUFTextGenerator(
                model_path=model_path,
                weight_index=info,
                model_plugin=resolve_model(info.architecture or ""),
                backend=args.backend,
                max_sequence_length=args.context,
            )
            free_resident, _ = runtime.mem_get_info()
            results: list[dict[str, object]] = []
            for row, (teacher_logits, teacher_tokens, command) in zip(
                prompts, teachers, strict=True
            ):
                tokens = np.asarray(generator.tokenizer.encode(row["prompt"]), dtype=np.int32)
                if not np.array_equal(tokens, teacher_tokens):
                    raise RuntimeError(
                        f"tokenizer mismatch for {row['id']}: "
                        f"llama={teacher_tokens.tolist()} hipengine={tokens.tolist()}"
                    )
                actual = generator.runner.prefill(tokens.tolist())
                results.append(
                    {
                        **row,
                        "token_count": int(tokens.size),
                        "token_ids": tokens.tolist(),
                        "llama_command": command,
                        **compare_logits(teacher_logits, actual.logits),
                    }
                )
            free_inference, _ = runtime.mem_get_info()
            peak = memory_stats()
            max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            generator.close()
            generator = None
            free_close, _ = runtime.mem_get_info()
            after_close = memory_stats()
        finally:
            if generator is not None:
                generator.close()

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_category[str(row["category"])].append(row)
    overall = aggregate(results)
    category = {name: aggregate(rows) for name, rows in sorted(by_category.items())}
    teardown = after_close["current_allocated_bytes"] == 0
    passed = (
        overall["max_kl"] <= args.max_kl
        and overall["top1_agreement_rate"] >= args.min_top1
        and teardown
    )
    report = {
        "schema": 1,
        "model_path": str(model_path),
        "parts": [{"path": str(path), "bytes": path.stat().st_size} for path in parts],
        "prompt_file": str(args.prompts.expanduser().resolve()),
        "backend": args.backend,
        "llama_debug": str(args.llama_debug.expanduser().resolve()),
        "thresholds": {"max_kl": args.max_kl, "min_top1": args.min_top1},
        "overall": overall,
        "by_category": category,
        "results": results,
        "memory": {
            "device_total_bytes": total_bytes,
            "device_free_before_bytes": free_before,
            "device_free_after_residency_bytes": free_resident,
            "device_free_after_inference_bytes": free_inference,
            "device_free_after_close_bytes": free_close,
            "hipengine_owned_peak": peak,
            "hipengine_owned_after_close": after_close,
            "process_max_rss_kib": max_rss_kib,
            "tracked_teardown_passed": teardown,
        },
        "passed": passed,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        output = args.json_out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
