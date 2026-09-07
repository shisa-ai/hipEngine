#!/usr/bin/env python3
"""DMS C2/C4/C8 concurrency probe: N resident sessions through one runner.

Opens N simultaneous trained-DMS resident sessions on one shared
``Qwen35GGUFFullStackRunner``, gives each a distinct above-window prompt
sliced from the same manifest validation stream, prefills sequentially
(each prefill's dense owner coexists with the other sessions' live compact
owners), then decodes round-robin so all N sessions are interleaved.

The shared runner binds the active session's DMS owner at each decode step.
Optional C1 verification compares every interleaved logit byte with an
independent single-session replay. INT8 mode is offline evaluation only.
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
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        required=True,
        help="uniform width when --heterogeneous is absent",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="give session i width prompt-tokens * 2^i (clamped to the stream), exercising heterogeneous lengths",
    )
    parser.add_argument(
        "--cancel-after-steps",
        type=int,
        default=None,
        help="close session 0 after this many interleaved decode rounds and continue the survivors (cancellation path)",
    )
    parser.add_argument(
        "--refill-cycles",
        type=int,
        default=1,
        help="repeat the open/prefill/decode/close cycle this many times, requiring 0.0 MiB allocated between cycles (pressure/refill)",
    )
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--codec", choices=("bf16", "int8_evaluation"), default="bf16")
    parser.add_argument("--verify-c1", action="store_true",
                        help="Require byte-exact logits against independent single-session replays.")
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    n = int(args.sessions)
    base_width = int(args.prompt_tokens)
    steps = int(args.decode_steps)
    stream = _validation_stream(args.data_manifest)
    widths = [
        min(base_width << i, len(stream)) if args.heterogeneous else base_width
        for i in range(n)
    ]
    if args.heterogeneous and len(set(widths)) != n:
        raise ValueError("heterogeneous widths collapsed; reduce session count or raise base width")
    offsets = [0] * n
    for i in range(1, n):
        offsets[i] = offsets[i - 1] + widths[i - 1]
    if offsets[-1] + widths[-1] > len(stream):
        raise ValueError(
            f"validation stream has {len(stream)} tokens; need {offsets[-1] + widths[-1]}"
        )
    prompts = [stream[offsets[i] : offsets[i] + widths[i]] for i in range(n)]
    prompt_sha = hashlib.sha256(
        np.asarray(sum(prompts, []), dtype=np.int64).tobytes()
    ).hexdigest()

    cycles: list[dict[str, Any]] = []
    cycle_errors: list[str] = []
    for cycle_index in range(int(args.refill_cycles)):
        cycles.append(
            _run_cycle(
                args,
                cycle_index,
                prompts,
                widths,
                steps,
                prompt_sha,
                cycle_errors,
            )
        )
        if cycle_errors:
            break
    result = {
        "schema_version": 1,
        "kind": "hipengine_rx7900xtx_dms_concurrency_probe",
        "status": (
            "passed"
            if not cycle_errors
            and all(
                row["finite_logits"] for cycle in cycles for row in cycle["decode"]
            )
            else "failed"
        ),
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "sessions": n,
        "codec": getattr(args, "codec", "bf16"),
        "verify_c1": getattr(args, "verify_c1", False),
        "heterogeneous_widths": widths,
        "cancel_after_steps": args.cancel_after_steps,
        "refill_cycles": int(args.refill_cycles),
        "prompt_tokens_each": base_width,
        "decode_steps_each": steps,
        "model": {"path": str(args.model.resolve()), "sha256": _sha256(args.model)},
        "metadata": {"path": str(args.metadata.resolve()), "sha256": _sha256(args.metadata)},
        "data_manifest": {"path": str(args.data_manifest.resolve()), "sha256": _sha256(args.data_manifest)},
        "prompts": {
            "disjoint_validation_slices": True,
            "sha256": prompt_sha,
        },
        "cycles": cycles,
        "cycle_errors": cycle_errors,
        "provenance": _git(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _run_cycle(
    args: argparse.Namespace,
    cycle_index: int,
    prompts: list[list[int]],
    widths: list[int],
    steps: int,
    prompt_sha: str,
    cycle_errors: list[str],
) -> dict[str, Any]:
    from hipengine.kvcache.dms import create_dms_bf16_backend, create_dms_int8_evaluation_backend
    backend_factory = {"bf16": create_dms_bf16_backend,
                       "int8_evaluation": create_dms_int8_evaluation_backend}[getattr(args, "codec", "bf16")]
    n = len(prompts)
    before = memory_stats()
    started = time.perf_counter()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    sessions: list[Qwen35GGUFResidentSession] = []
    prefill_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    cancellation: dict[str, Any] | None = None
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
                max_sequence_length=len(prompt) + steps,
                dms_metadata_path=args.metadata,
                dms_backend_factory=backend_factory,
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
                    "prompt_tokens": len(prompt),
                    "prefill_seconds": round(time.perf_counter() - mark_started, 3),
                    "capacity": session._dms_backend.observability_snapshot()["capacity"],
                    "active_sessions_at_pack": index + 1,
                }
            )
        all_prefilled_at = time.perf_counter()
        post_prefill = memory_stats()

        # Round-robin interleaved decode across all live sessions.
        currents = [int(prompts[index][-1]) for index in range(n)]
        cancelled = False
        for step in range(steps):
            for index in range(n):
                session = sessions[index]
                if session is None:
                    continue
                step_started = time.perf_counter()
                result_step = session.step(currents[index], return_logits=True)
                currents[index] = int(result_step.token_id)
                decode_rows.append(
                    {
                        "cycle": cycle_index,
                        "step": step,
                        "session": index,
                        "output_token": int(result_step.token_id),
                        "finite_logits": bool(np.isfinite(result_step.logits).all()),
                        "logits_sha256": hashlib.sha256(result_step.logits.tobytes()).hexdigest(),
                        "seconds": round(time.perf_counter() - step_started, 4),
                    }
                )
            if (
                args.cancel_after_steps is not None
                and not cancelled
                and step + 1 >= int(args.cancel_after_steps)
            ):
                # Cancellation: close session 0 mid-flight; survivors must
                # keep decoding with their own live counts intact.
                cancelled = True
                victim = sessions[0]
                cancel_started = time.perf_counter()
                victim.__exit__(None, None, None)
                sessions[0] = None
                cancellation = {
                    "cancelled_session": 0,
                    "after_decode_step": step,
                    "close_seconds": round(time.perf_counter() - cancel_started, 3),
                    "survivor_sessions": sum(1 for s in sessions if s is not None),
                    "memory_after_cancel_MiB": round(
                        memory_stats()["current_allocated_bytes"] / 2**20, 1
                    ),
                }
        post_decode = memory_stats()
        decode_done_at = time.perf_counter()
        for session in sessions:
            if session is not None:
                snapshots.append(session._dms_backend.observability_snapshot())
        if getattr(args, "verify_c1", False):
            for session in sessions:
                if session is not None:
                    session.__exit__(None, None, None)
            sessions.clear()
            for index, prompt in enumerate(prompts):
                expected = [row for row in decode_rows if row["session"] == index]
                with Qwen35GGUFResidentSession(
                    args.model, backend=str(args.backend), shared_runner=runner,
                    max_sequence_length=len(prompt) + steps,
                    dms_metadata_path=args.metadata, dms_backend_factory=backend_factory,
                    dms_max_new_tokens=steps + 1, use_wmma_prefill=True, use_gemv_decode=True,
                ) as reference:
                    reference.prefill(prompt, use_bulk=True, bulk_attention_mode="bulk", return_logits=False)
                    current = int(prompt[-1])
                    for row in expected:
                        replay = reference.step(current, return_logits=True)
                        current = int(replay.token_id)
                        row["c1_logits_exact"] = hashlib.sha256(replay.logits.tobytes()).hexdigest() == row["logits_sha256"]
                        if not row["c1_logits_exact"]:
                            raise AssertionError(f"DMS C1 logit mismatch at session {index}, step {row['step']}")
    finally:
        close_started = time.perf_counter()
        errors = []
        for session in sessions:
            if session is None:
                continue
            try:
                session.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                errors.append(repr(exc))
        runner.close()
        ended = time.perf_counter()
    after = memory_stats()

    cycle = {
        "cycle": cycle_index,
        "prefill": prefill_rows,
        "decode": decode_rows,
        "cancellation": cancellation,
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
            "after_close_MiB": round(after["current_allocated_bytes"] / 2**20, 1),
        },
        "close_errors": errors,
        "timing": {
            "load_seconds": round(loaded_at - started, 2),
            "prefill_all_seconds": round(all_prefilled_at - loaded_at, 2),
            "decode_all_seconds": round(decode_done_at - all_prefilled_at, 2),
            "close_seconds": round(ended - close_started, 2),
        },
    }
    if errors:
        cycle_errors.extend(errors)
    if not all(row["finite_logits"] for row in decode_rows):
        cycle_errors.append(f"cycle {cycle_index}: non-finite logits in decode")
    if after["current_allocated_bytes"] != 0:
        cycle_errors.append(
            f"cycle {cycle_index}: allocations after close: {after['current_allocated_bytes']}"
        )
    return cycle


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    last_cycle = result["cycles"][-1]
    print(
        json.dumps(
            {
                "status": result["status"],
                "sessions": result["sessions"],
                "heterogeneous_widths": result["heterogeneous_widths"],
                "cancel_after_steps": result["cancel_after_steps"],
                "refill_cycles": result["refill_cycles"],
                "final_after_close_MiB": last_cycle["memory"]["after_close_MiB"],
                "cycle_errors": result["cycle_errors"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
