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
    parser.add_argument("--positions")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=258)
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
    array_keys = _parse_array_keys(args.array_keys) if args.array_keys else None
    positions = _parse_positions(args.positions, len(tokens)) if args.positions else None
    if positions is None:
        positions = (_resolve_position(args.position, len(tokens)),)
    if args.dry_run:
        artifact = _plan_for_positions(
            model=args.model,
            tokens=tokens,
            positions=positions,
            layer=args.layer,
            status="dry_run",
            iteration=args.iteration,
        )
    elif not _hip_available():
        artifact = _plan_for_positions(
            model=args.model,
            tokens=tokens,
            positions=positions,
            layer=args.layer,
            status="skipped_no_hip_runtime",
            iteration=args.iteration,
        )
    elif len(positions) == 1:
        artifact = capture_linear_boundary(
            model=args.model,
            tokens=tokens,
            position=positions[0],
            layer=args.layer,
            compiler_version=args.compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=args.max_sequence_length,
            iteration=args.iteration,
            include_arrays=bool(args.include_arrays),
            array_keys=array_keys,
        )
    else:
        artifact = capture_linear_boundaries(
            model=args.model,
            tokens=tokens,
            positions=positions,
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
    print(json.dumps(_stdout_summary(args.output, artifact), indent=2))


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
    include_arrays: bool = False,
    array_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return capture_linear_boundaries(
        model=model,
        tokens=tokens,
        positions=(position,),
        layer=layer,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        max_sequence_length=max_sequence_length,
        iteration=iteration,
        include_arrays=include_arrays,
        array_keys=array_keys,
    )["captures"][0]


def capture_linear_boundaries(
    *,
    model: Path,
    tokens: tuple[int, ...],
    positions: tuple[int, ...],
    layer: int,
    compiler_version: str | None,
    require_cached_build: bool,
    max_sequence_length: int | None,
    iteration: int = 258,
    include_arrays: bool = False,
    array_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    max_seq = int(max_sequence_length or max(len(tokens) + 8, 32))
    captures = []
    with Qwen35GGUFResidentSession(
        model,
        compiler_version=compiler_version,
        require_cached_build=bool(require_cached_build),
        max_sequence_length=max_seq,
    ) as session:
        for position in positions:
            session.reset()
            for pos, token_id in enumerate(tokens[:position]):
                session._run_token_to_final_hidden(int(token_id), position=pos)  # noqa: SLF001
            capture = session.capture_linear_attention_boundary(
                int(tokens[position]),
                position=position,
                layer_id=layer,
            )
            captures.append(
                _capture_artifact(
                    model=model,
                    tokens=tokens,
                    position=position,
                    layer=layer,
                    capture=capture,
                    iteration=iteration,
                    include_arrays=include_arrays,
                    array_keys=array_keys,
                )
            )

    return _batch_artifact(
        model=model,
        tokens=tokens,
        positions=positions,
        layer=layer,
        status="captured",
        iteration=iteration,
        captures=tuple(captures),
    )


def _capture_artifact(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    layer: int,
    capture: Any,
    iteration: int,
    include_arrays: bool,
    array_keys: tuple[str, ...] | None,
) -> dict[str, Any]:
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
        "attn_norm_f32": capture.attn_norm_f32,
        "linear_qkv_f32": capture.linear_qkv_f32,
        "linear_z_f32": capture.linear_z_f32,
        "ssm_alpha_f32": capture.ssm_alpha_f32,
        "ssm_beta_f32": capture.ssm_beta_f32,
        "conv_out_f32": capture.conv_out_f32,
        "recurrent_out_f32": capture.recurrent_out_f32,
        "recurrent_bf16_f32": capture.recurrent_bf16_f32,
        "attn_out_f32": capture.attn_out_f32,
    }
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


def _plan_for_positions(
    *,
    model: Path,
    tokens: tuple[int, ...],
    positions: tuple[int, ...],
    layer: int,
    status: str,
    iteration: int,
) -> dict[str, Any]:
    if len(positions) == 1:
        return _plan_artifact(
            model=model,
            tokens=tokens,
            position=positions[0],
            layer=layer,
            status=status,
            iteration=iteration,
        )
    captures = tuple(
        _plan_artifact(
            model=model,
            tokens=tokens,
            position=position,
            layer=layer,
            status=status,
            iteration=iteration,
        )
        for position in positions
    )
    return _batch_artifact(
        model=model,
        tokens=tokens,
        positions=positions,
        layer=layer,
        status=status,
        iteration=iteration,
        captures=captures,
    )


def _batch_artifact(
    *,
    model: Path,
    tokens: tuple[int, ...],
    positions: tuple[int, ...],
    layer: int,
    status: str,
    iteration: int,
    captures: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "mtp_gguf_linear_attention_boundary_capture_batch",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "model": str(model),
        "layer_id": int(layer),
        "positions": [int(position) for position in positions],
        "token_ids": [int(tokens[position]) for position in positions],
        "prompt_tokens": list(tokens),
        "capture_count": len(captures),
        "api": "Qwen35GGUFResidentSession.capture_linear_attention_boundary",
        "captures": list(captures),
    }


def _stdout_summary(output: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "output": str(output),
        "status": artifact["status"],
    }
    if "captures" in artifact:
        summary.update(
            {
                "positions": artifact["positions"],
                "token_ids": artifact["token_ids"],
                "capture_count": artifact["capture_count"],
            }
        )
    else:
        summary.update(
            {
                "position": artifact["position"],
                "token_id": artifact["token_id"],
            }
        )
    return summary


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


def _array_payload(array: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(array, dtype=np.float32).reshape(-1)]


def _select_arrays(
    arrays: dict[str, np.ndarray],
    keys: tuple[str, ...] | None,
) -> dict[str, np.ndarray]:
    if keys is None:
        return dict(arrays)
    missing = [key for key in keys if key not in arrays]
    if missing:
        available = ", ".join(sorted(arrays))
        raise ValueError(f"unknown --array-keys entries {missing}; available: {available}")
    return {key: arrays[key] for key in keys}


def _parse_array_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(dict.fromkeys(key.strip() for key in raw.split(",") if key.strip()))
    if not keys:
        raise ValueError("at least one array key is required")
    return keys


def _parse_tokens(raw: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not tokens:
        raise ValueError("at least one token id is required")
    if any(token < 0 for token in tokens):
        raise ValueError("token ids must be non-negative")
    return tokens


def _parse_positions(raw: str, token_count: int) -> tuple[int, ...]:
    positions: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = _resolve_position(int(start_raw), token_count)
            end = _resolve_position(int(end_raw), token_count)
            step = 1 if end >= start else -1
            positions.extend(range(start, end + step, step))
        else:
            positions.append(_resolve_position(int(item), token_count))
    if not positions:
        raise ValueError("at least one position is required")
    return tuple(dict.fromkeys(positions))


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
