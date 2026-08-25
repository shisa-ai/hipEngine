#!/usr/bin/env python3
"""Integrated no-shadow Qwen3.8 DMS long-context capacity/quality smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "working_tree_clean": not dirty}


def _tokens(path: Path, count: int) -> tuple[list[int], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequences = sorted(raw["sequences"], key=lambda row: str(row["sequence_id"]))
    stream = [int(token) for row in sequences for token in row["token_ids"]]
    if not stream:
        raise ValueError("data manifest contains no tokens")
    repetitions = (int(count) + len(stream) - 1) // len(stream)
    values = (stream * repetitions)[: int(count)]
    digest = hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()
    return values, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt, token_sha = _tokens(args.data_manifest, args.prompt_tokens)
    before = memory_stats()
    started = time.perf_counter()
    decode_rows: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        # Each decode step appends exactly one input token; the sampled output
        # is not appended until the following step.
        max_sequence_length=int(args.prompt_tokens) + int(args.decode_steps),
        dms_metadata_path=args.metadata,
        dms_max_new_tokens=int(args.decode_steps) + 1,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        constructed_at = time.perf_counter()
        constructed = memory_stats()
        first = session.prefill(
            prompt,
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=True,
            # Long chunked prefill exceeds the optional 4,096-boundary stage
            # recorder. Capacity qualification uses end-to-end wall time.
            record_gpu_stage_timings=False,
        )
        prefill_at = time.perf_counter()
        post_prefill = memory_stats()
        current = int(first.token_id)
        for step in range(int(args.decode_steps)):
            step_started = time.perf_counter()
            result = session.step(current, return_logits=True)
            decode_rows.append(
                {
                    "step": step,
                    "input_token": current,
                    "output_token": int(result.token_id),
                    "finite_logits": bool(np.isfinite(result.logits).all()),
                    "seconds": time.perf_counter() - step_started,
                }
            )
            current = int(result.token_id)
        post_decode = memory_stats()
        snapshot = session._dms_backend.observability_snapshot()
        prefill_gpu_stages = dict(session.last_prefill_gpu_stage_timings_ms)
    ended = time.perf_counter()
    after = memory_stats()
    result = {
        "schema_version": 1,
        "kind": "hipengine_qwen38_integrated_dms_long_smoke",
        "status": "passed" if all(row["finite_logits"] for row in decode_rows) else "failed",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "model": {
            "path": str(args.model.resolve()),
            "sha256": _sha256(args.model),
        },
        "metadata": {
            "path": str(args.metadata.resolve()),
            "sha256": _sha256(args.metadata),
        },
        "prompt": {
            "tokens": int(args.prompt_tokens),
            "token_ids_sha256": token_sha,
            "data_manifest_sha256": _sha256(args.data_manifest),
        },
        "timing": {
            "construct_seconds": constructed_at - started,
            "prefill_seconds": prefill_at - constructed_at,
            "decode_seconds": ended - prefill_at,
            "total_seconds": ended - started,
            "prefill_gpu_stages_ms": prefill_gpu_stages,
            "prefill_gpu_stage_note": "disabled: long chunked prefill exceeds recorder capacity",
        },
        "decode": decode_rows,
        "dms": snapshot,
        "memory": {
            "before": before,
            "constructed_dense_prefill_owner": constructed,
            "after_compact_pack_and_dense_release": post_prefill,
            "after_decode": post_decode,
            "after_close": after,
            "dense_release_delta_bytes": int(constructed["current_allocated_bytes"])
            - int(post_prefill["current_allocated_bytes"]),
        },
        "provenance": _git(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "prompt_tokens": result["prompt"]["tokens"],
                "timing": result["timing"],
                "capacity": result["dms"]["capacity"],
                "memory": result["memory"],
                "decode": result["decode"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
