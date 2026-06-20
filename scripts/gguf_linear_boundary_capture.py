#!/usr/bin/env python3
"""Capture a real GGUF resident linear-attention boundary diagnostic.

The non-dry path requires HIP and a local GGUF model.  The default prompt tokens
are the captured reasoning-off greeting prompt used in the MTP-GGUF parity lane.
"""

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

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter258-linear-boundary-capture.json")
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
    parser.add_argument("--iteration", type=int, default=258)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compiler-version")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tokens = _parse_tokens(args.tokens)
    position = _resolve_position(args.position, len(tokens))
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
        artifact = capture_linear_boundary(
            model=args.model,
            tokens=tokens,
            position=position,
            layer=args.layer,
            compiler_version=args.compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=args.max_sequence_length,
            iteration=args.iteration,
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
            },
            indent=2,
        )
    )


def capture_linear_boundary(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    layer: int,
    compiler_version: str | None,
    require_cached_build: bool,
    max_sequence_length: int | None,
    iteration: int = 258,
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
        capture = session.capture_linear_attention_boundary(
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
    artifact["buffers"] = {
        "attn_norm_f32": _array_summary(capture.attn_norm_f32),
        "linear_qkv_f32": _array_summary(capture.linear_qkv_f32),
        "linear_z_f32": _array_summary(capture.linear_z_f32),
        "ssm_alpha_f32": _array_summary(capture.ssm_alpha_f32),
        "ssm_beta_f32": _array_summary(capture.ssm_beta_f32),
        "conv_out_f32": _array_summary(capture.conv_out_f32),
        "recurrent_out_f32": _array_summary(capture.recurrent_out_f32),
        "recurrent_bf16_f32": _array_summary(capture.recurrent_bf16_f32),
        "attn_out_f32": _array_summary(capture.attn_out_f32),
    }
    return artifact


def _plan_artifact(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    layer: int,
    status: str,
    iteration: int = 258,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "mtp_gguf_linear_attention_boundary_capture",
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
        "api": "Qwen35GGUFResidentSession.capture_linear_attention_boundary",
        "note": (
            "Processes prompt tokens before position through the normal resident full-stack decode "
            "path, then captures the selected token at the requested linear-attention layer."
        ),
    }


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    return {
        "shape": list(np.asarray(array).shape),
        "finite": bool(np.all(np.isfinite(values))),
        "min": float(np.min(values)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
        "mean": float(np.mean(values, dtype=np.float32)) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(values * values, dtype=np.float32))) if values.size else 0.0,
        "sample": [float(x) for x in values[:8]],
    }


def _parse_tokens(raw: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not tokens:
        raise ValueError("at least one token id is required")
    if any(token < 0 for token in tokens):
        raise ValueError("token ids must be non-negative")
    return tokens


def _resolve_position(raw_position: int, token_count: int) -> int:
    position = token_count - 1 if raw_position < 0 else int(raw_position)
    if position < 0 or position >= token_count:
        raise ValueError(f"position {position} outside prompt token range 0..{token_count - 1}")
    return position


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
