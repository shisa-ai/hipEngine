#!/usr/bin/env python3
"""Capture GGUF resident linear-attention post-FFN layer diagnostics."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from scripts.gguf_linear_boundary_capture import (  # noqa: E402
    _array_payload,
    _array_summary,
    _parse_array_keys,
    _parse_tokens,
    _resolve_position,
    _select_arrays,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter274-linear-layer-full-arrays.json")
DEFAULT_PROMPT_TOKENS = (
    248045,
    846,
    198,
    7734,
    264,
    2716,
    40719,
    13,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", default=",".join(str(token) for token in DEFAULT_PROMPT_TOKENS))
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=274)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compiler-version")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--include-arrays", action="store_true")
    parser.add_argument("--array-keys")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.array_keys and not args.include_arrays:
        parser.error("--array-keys requires --include-arrays")

    tokens = _parse_tokens(args.tokens)
    position = _resolve_position(args.position, len(tokens))
    array_keys = _parse_array_keys(args.array_keys) if args.array_keys else None
    if args.dry_run:
        artifact = _plan_artifact(
            model=args.model,
            tokens=tokens,
            position=position,
            layer=args.layer,
            status="dry_run",
            iteration=args.iteration,
        )
    elif not _hip_available():
        artifact = _plan_artifact(
            model=args.model,
            tokens=tokens,
            position=position,
            layer=args.layer,
            status="skipped_no_hip_runtime",
            iteration=args.iteration,
        )
    else:
        artifact = capture_linear_layer(
            model=args.model,
            tokens=tokens,
            position=position,
            layer=args.layer,
            compiler_version=args.compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=args.max_sequence_length,
            iteration=args.iteration,
            include_arrays=bool(args.include_arrays),
            array_keys=array_keys,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "position": artifact["position"],
                "token_id": artifact["token_id"],
                "layer_id": artifact["layer_id"],
            },
            indent=2,
        )
    )


def capture_linear_layer(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    layer: int,
    compiler_version: str | None,
    require_cached_build: bool,
    max_sequence_length: int | None,
    iteration: int = 274,
    include_arrays: bool = False,
    array_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    max_seq = int(max_sequence_length or max(len(tokens) + 8, 32))
    with Qwen35GGUFResidentSession(
        model,
        compiler_version=compiler_version,
        require_cached_build=bool(require_cached_build),
        max_sequence_length=max_seq,
    ) as session:
        session.reset()
        for pos, token_id in enumerate(tokens[:position]):
            session._run_token_to_final_hidden(int(token_id), position=pos)  # noqa: SLF001
        capture = session.capture_linear_attention_layer(
            int(tokens[position]),
            position=position,
            layer_id=layer,
        )

    artifact = _plan_artifact(
        model=model,
        tokens=tokens,
        position=position,
        layer=layer,
        status="captured",
        iteration=iteration,
    )
    artifact["capture_summary"] = capture.as_summary_dict()
    captured_arrays = {
        "hidden_in_f32": capture.hidden_in_f32,
        "attn_out_f32": capture.attn_out_f32,
        "post_norm_f32": capture.post_norm_f32,
        "residual_f32": capture.residual_f32,
        "ffn_or_moe_down_f32": capture.ffn_or_moe_down_f32,
        "layer_out_f32": capture.layer_out_f32,
    }
    if capture.moe_shared_out_f32 is not None:
        captured_arrays["moe_shared_out_f32"] = capture.moe_shared_out_f32
    artifact["buffers"] = {name: _array_summary(array) for name, array in captured_arrays.items()}
    if include_arrays:
        selected_arrays = _select_arrays(captured_arrays, array_keys)
        artifact["array_keys"] = list(selected_arrays)
        artifact["arrays"] = {
            name: _array_payload(array) for name, array in selected_arrays.items()
        }
    return artifact


def _plan_artifact(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    layer: int,
    status: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "mtp_gguf_linear_attention_layer_capture",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "model": str(model),
        "layer_id": int(layer),
        "position": int(position),
        "token_id": int(tokens[position]),
        "prompt_tokens": list(tokens),
        "warmup_tokens": list(tokens[:position]),
        "api": "Qwen35GGUFResidentSession.capture_linear_attention_layer",
        "note": (
            "Processes prompt tokens before position through the normal resident full-stack decode "
            "path, then captures the selected token after the requested linear-attention layer."
        ),
    }


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
