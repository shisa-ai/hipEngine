#!/usr/bin/env python3
"""DMS C2/C4/C8 concurrency probe: N resident sessions through one runner.

Opens N simultaneous trained-DMS resident sessions on one shared
``Qwen35GGUFFullStackRunner``, gives each a distinct above-window prompt
sliced from the same manifest validation stream, prefills sequentially
(each prefill's dense owner coexists with the other sessions' live compact
owners), then decodes round-robin so all N sessions are interleaved.

KNOWN BLOCKER (recorded 2026-09-07): the shared runner routes external DMS
decode through a single ``_dms_decode_owner`` slot
(``Qwen35GGUFResidentSession`` setup overwrites it; only the owner itself
pops it on close). With two simultaneous DMS sessions the first session's
decode is routed through the second session's DMS backend and fails with
"DMS direct append device/host live-count mismatch". This probe reproduces
that blocker for evidence; DMS C>1 requires per-session owner routing.
"""

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
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)


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


def _validation_stream(path: Path) -> list[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequences = sorted(
        (row for row in raw["sequences"] if str(row.get("split")) == "validation"),
        key=lambda row: str(row["sequence_id"]),
    )
    stream = [int(token) for row in sequences for token in row["token_ids"]]
    if not stream:
        raise ValueError("manifest has no validation tokens")
    return stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    n = int(args.sessions)
    width = int(args.prompt_tokens)
    steps = int(args.decode_steps)
    stream = _validation_stream(args.data_manifest)
    if n * width > len(stream):
        raise ValueError(
            f"validation stream has {len(stream)} tokens; need {n * width}"
        )
    prompts = [stream[i * width : (i + 1) * width] for i in range(n)]
    prompt_sha = hashlib.sha256(
        np.asarray(prompts, dtype=np.int64).tobytes()
    ).hexdigest()

    before = memory_stats()
    started = time.perf_counter()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    sessions: list[Qwen35GGUFResidentSession] = []
    prefill_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    all_prefilled_at = 0.0
    post_prefill = before
    post_decode = before
    decode_done_at = 0.0
    close_started = 0.0
    ended = 0.0
    errors: list[str] = []
    try:
        # Sequential prefill: each dense prefill owner coexists with the
        # compact owners of every already-prefilled session.
        for index, prompt in enumerate(prompts):
            session = Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                shared_runner=runner,
                max_sequence_length=width + steps,
                dms_metadata_path=args.metadata,
                dms_max_new_tokens=steps + 1,
                use_wmma_prefill=True,
                use_gemv_decode=True,
            )
            session.__enter__()
            sessions.append(session)
            mark_started = time.perf_counter()
            session.prefill(
                prompt,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=False,
                record_gpu_stage_timings=False,
            )
            prefill_rows.append(
                {
                    "session": index,
                    "prefill_seconds": round(time.perf_counter() - mark_started, 3),
                    "capacity": session._dms_backend.observability_snapshot()["capacity"],
                    "active_sessions_at_pack": index + 1,
                }
            )
        all_prefilled_at = time.perf_counter()
        post_prefill = memory_stats()

        # Round-robin interleaved decode across all live sessions.
        currents = [int(prompts[index][-1]) for index in range(n)]
        for step in range(steps):
            for index, session in enumerate(sessions):
                step_started = time.perf_counter()
                result = session.step(currents[index], return_logits=True)
                currents[index] = int(result.token_id)
                decode_rows.append(
                    {
                        "step": step,
                        "session": index,
                        "output_token": int(result.token_id),
                        "finite_logits": bool(np.isfinite(result.logits).all()),
                        "seconds": round(time.perf_counter() - step_started, 4),
                    }
                )
        post_decode = memory_stats()
        decode_done_at = time.perf_counter()
        snapshots = [
            session._dms_backend.observability_snapshot() for session in sessions
        ]
    finally:
        close_started = time.perf_counter()
        errors = []
        for session in reversed(sessions):
            try:
                session.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                errors.append(repr(exc))
        close_errors_at = time.perf_counter()
        runner.close()
        ended = time.perf_counter()
    after = memory_stats()

    result = {
        "schema_version": 1,
        "kind": "hipengine_rx7900xtx_dms_concurrency_probe",
        "status": (
            "passed"
            if all(row["finite_logits"] for row in decode_rows) and not errors
            else "failed"
        ),
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "sessions": n,
        "prompt_tokens_each": width,
        "decode_steps_each": steps,
        "model": {"path": str(args.model.resolve()), "sha256": _sha256(args.model)},
        "metadata": {"path": str(args.metadata.resolve()), "sha256": _sha256(args.metadata)},
        "data_manifest": {"path": str(args.data_manifest.resolve()), "sha256": _sha256(args.data_manifest)},
        "prompts": {
            "disjoint_validation_slices": True,
            "sha256": prompt_sha,
        },
        "timing": {
            "load_seconds": round(loaded_at - started, 2),
            "prefill_all_seconds": round(all_prefilled_at - loaded_at, 2),
            "decode_all_seconds": round(decode_done_at - all_prefilled_at, 2),
            "close_seconds": round(ended - close_started, 2),
        },
        "prefill": prefill_rows,
        "decode": decode_rows,
        "dms_snapshots": [
            {
                "capacity": snap["capacity"],
                "extent_pool": {
                    "capacity_slots": snap["extent_pool"]["capacity_slots"],
                    "free_slots": snap["extent_pool"]["free_slots"],
                    "allocation_failures": snap["extent_pool"]["allocation_failures"],
                },
                "ledger_active_reservations": snap["ledger"]["active_reservations"],
            }
            for snap in snapshots
        ],
        "memory": {
            "before": before,
            "post_prefill_all_sessions": post_prefill,
            "post_decode": post_decode,
            "after_close": after,
        },
        "close_errors": errors,
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
                "sessions": result["sessions"],
                "prompt_tokens_each": result["prompt_tokens_each"],
                "memory_after_close_MiB": round(
                    result["memory"]["after_close"]["current_allocated_bytes"] / 2**20, 1
                ),
                "close_errors": result["close_errors"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
