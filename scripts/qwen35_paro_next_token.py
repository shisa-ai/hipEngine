#!/usr/bin/env python3
"""Minimal torch-free Qwen3.5/PARO one-token next-token smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=("auto", "hip_gfx1100", "hip_gfx1151"),
        default="auto",
        help=(
            "Kernel backend key; auto detects gfx1100/gfx1151, "
            "hip_gfx1151 builds native gfx1151 code objects."
        ),
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--token-id", type=int, default=None, help="Bypass tokenizer and decode this single token id")
    parser.add_argument("--max-layers", type=int, default=0, help="Debug limit; 0 means all layers")
    parser.add_argument("--lm-head-chunk", type=int, default=4096)
    parser.add_argument(
        "--lm-head",
        choices=("gpu_fp16_argmax", "cpu_numpy_argmax"),
        default="gpu_fp16_argmax",
        help="Final head path; default keeps projection+argmax on GPU and copies back only id/logit.",
    )
    parser.add_argument("--resident-layers", action="store_true", help="Materialize all layer weights before executing")
    parser.add_argument("--cpu-threads", type=int, default=None, help="Set BLAS CPU threads before importing NumPy")
    parser.add_argument("--progress", action="store_true", help="Emit progress events to stderr")
    parser.add_argument("--progress-format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            raise ValueError("--cpu-threads must be positive")
        threads = str(args.cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = threads
        os.environ["OMP_NUM_THREADS"] = threads
        os.environ["MKL_NUM_THREADS"] = threads

    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner

    def emit_progress(payload: dict) -> None:
        if args.progress_format == "json":
            print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
            return
        line = _format_progress(payload)
        if line is not None:
            print(line, file=sys.stderr, flush=True)

    runner = Qwen35ParoNextTokenRunner(args.model, backend=args.backend)
    with hip_target_arch_environment(runner.target_arch):
        result = runner.run_next_token(
            prompt=args.prompt,
            token_id=args.token_id,
            max_layers=args.max_layers,
            lm_head_chunk=args.lm_head_chunk,
            progress=emit_progress if args.progress else None,
            resident_layers=args.resident_layers,
            lm_head=args.lm_head,
        )
    print(json.dumps(result.to_json_dict(), ensure_ascii=False))
    return 0


def _format_progress(payload: dict) -> str | None:
    event = payload.get("event", "progress")
    layer = payload.get("layer")
    prefix = f"layer {layer}: " if layer is not None else ""
    if event in {"materialize_tensor_start", "materialize_prepared_tensor_start"}:
        return f"{prefix}{event} {payload.get('index')}/{payload.get('total')} {payload.get('name')}"
    if event in {
        "materialize_tensor_done",
        "materialize_prepared_tensor_done",
        "prepare_runtime_tensor_start",
        "prepare_runtime_tensor_done",
    }:
        return None
    if event == "expert_stack_progress":
        return (
            f"{prefix}stack {payload.get('proj')}.{payload.get('suffix')} "
            f"{payload.get('expert')}/{payload.get('total')}"
        )
    if event in {"layer_start", "layer_done", "materialize_layer_start", "materialize_layer_done"}:
        return f"{prefix}{event} {payload.get('type')}"
    if event == "layers_start":
        return f"layers_start total={payload.get('layers')} resident={payload.get('resident')}"
    if event == "lm_head_done":
        return f"lm_head_done id={payload.get('next_token_id')} logit={payload.get('next_token_logit')}"
    fields = " ".join(f"{key}={value}" for key, value in payload.items() if key != "event")
    return f"{prefix}{event} {fields}".rstrip()


if __name__ == "__main__":
    raise SystemExit(main())
