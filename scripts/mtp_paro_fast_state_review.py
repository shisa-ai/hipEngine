#!/usr/bin/env python3
"""Review fast PARO B1 selected-state/KV ownership on a strict-owned schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.mtp_paro_verifier_numerics import (
    _FAST_FLAGS,
    _ROUTE_FLAGS,
    _prefill,
    _set_route,
    _target_batch,
)
from scripts.mtp_state_drift_audit import (
    _compare_mtp_scratch_to_resident,
    _copy_kv_cell,
    _copy_tensor_host,
)


def _bf16_finite(bits: np.ndarray) -> bool:
    values = np.asarray(bits, dtype=np.uint16)
    return bool(np.all((values & np.uint16(0x7F80)) != np.uint16(0x7F80)))


def _fp16_finite(bits: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(bits, dtype=np.uint16).view(np.float16)).all())


def _tensor_host_finite(array: np.ndarray, dtype_name: str) -> bool:
    if dtype_name == "bf16":
        return _bf16_finite(array)
    if dtype_name == "fp16":
        return _fp16_finite(array)
    return bool(np.isfinite(array).all())


def _audit_cycle_ids(cycles: Sequence[dict[str, Any]]) -> set[int]:
    if not cycles:
        raise ValueError("state review needs at least one cycle")
    selected = {int(cycles[0]["cycle"]), int(cycles[-1]["cycle"])}
    for accepted in (0, 1):
        match = next(
            (cycle for cycle in cycles if int(cycle["strict_accepted"]) == accepted),
            None,
        )
        if match is not None:
            selected.add(int(match["cycle"]))
    selected.update(
        int(cycle["cycle"])
        for cycle in cycles
        if bool(cycle.get("task_decision_mismatch", False))
    )
    return selected


def _finite_state_summary(session: Qwen35ParoResidentSession, *, position: int) -> dict[str, Any]:
    linear_records: list[dict[str, Any]] = []
    for layer_id in session.linear_layer_ids:
        conv, recurrent = session._slot_linear_state(int(layer_id), 0)
        for name, tensor in (("conv", conv), ("recurrent", recurrent)):
            host = _copy_tensor_host(session, tensor)
            linear_records.append(
                {
                    "layer": int(layer_id),
                    "state": name,
                    "finite": _tensor_host_finite(host, tensor.dtype.value),
                }
            )
    kv_records: list[dict[str, Any]] = []
    for layer_id in session.full_caches:
        for which in ("key", "value"):
            host = _copy_kv_cell(
                session,
                layer_id=int(layer_id),
                slot=0,
                position=int(position),
                which=which,
            )
            kv_records.append(
                {
                    "layer": int(layer_id),
                    "state": which,
                    "position": int(position),
                    "finite": _bf16_finite(host),
                }
            )
    return {
        "linear_checked": len(linear_records),
        "linear_failed": sum(not row["finite"] for row in linear_records),
        "kv_cells_checked": len(kv_records),
        "kv_cells_failed": sum(not row["finite"] for row in kv_records),
        "passed": all(row["finite"] for row in (*linear_records, *kv_records)),
    }


def run(
    *,
    model: Path,
    numerical_capture: Path,
    prompt_tokens_file: Path,
    backend: str,
) -> dict[str, Any]:
    source = json.loads(numerical_capture.read_text(encoding="utf-8"))
    prompt_tokens = [
        int(token)
        for token in prompt_tokens_file.read_text(encoding="utf-8")
        .replace(",", " ")
        .split()
    ]
    if not prompt_tokens:
        raise ValueError("prompt token fixture is empty")
    cycles = list(source["cycles"])
    audit_ids = _audit_cycle_ids(cycles)
    saved_env = {key: os.environ.get(key) for key in _ROUTE_FLAGS}
    reset_memory_stats()
    initial_memory = memory_stats()
    cycle_reviews: list[dict[str, Any]] = []
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    try:
        _set_route(_FAST_FLAGS)
        with Qwen35ParoResidentSession(
            runner,
            max_sequence_length=len(prompt_tokens) + int(source["prompt"]["decode_tokens"]) + 8,
            max_batch_size=2,
        ) as session:
            root = _prefill(session, prompt_tokens)
            if int(root) != int(cycles[0]["root"]):
                raise RuntimeError("candidate prefill root does not match strict schedule")
            for record in cycles:
                batch = _target_batch(
                    int(record["root"]),
                    int(record["context"]),
                    int(record["candidate"]),
                )
                verify = session.verify_chain_bulk_and_commit(
                    batch,
                    base_slot=0,
                    capture_layer_ids=(),
                    capture_hidden_concat=Tensor.from_handle(
                        0,
                        (2, 0),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                    capture_row_start=0,
                    chain_attn_mode="decode_batched",
                    graph_mode="off",
                    canonicalize_after=False,
                )
                candidate_bonus = int(
                    verify.next_token
                    if verify.next_token is not None
                    else record["candidate_bonus"]
                )
                candidate_decision_exact = (
                    int(verify.accepted_count) == int(record["candidate_accepted"])
                    and candidate_bonus == int(record["candidate_bonus"])
                )
                strict_row = int(record["strict_commit_row"])
                forced = int(verify.commit_row) != strict_row
                if forced:
                    session._commit_bulk_linear_states(strict_row, base_slot=0)
                    session._set_slot_position(
                        int(record["strict_commit_position"]), slot=0
                    )
                    session.runtime.device_synchronize()
                scratch = _compare_mtp_scratch_to_resident(
                    session,
                    selected_row=strict_row,
                )
                position_ok = (
                    int(session.position_arr[0]) == int(record["strict_commit_position"])
                    and int(session.context_arr[0])
                    == int(record["strict_commit_position"]) + 1
                )
                finite = None
                if int(record["cycle"]) in audit_ids:
                    finite = _finite_state_summary(
                        session,
                        position=int(record["strict_commit_position"]),
                    )
                cycle_reviews.append(
                    {
                        "cycle": int(record["cycle"]),
                        "strict_accepted": int(record["strict_accepted"]),
                        "strict_commit_row": strict_row,
                        "candidate_commit_row": int(verify.commit_row),
                        "candidate_accepted": int(verify.accepted_count),
                        "candidate_bonus": candidate_bonus,
                        "candidate_decision_matches_source": candidate_decision_exact,
                        "forced_to_strict_row": forced,
                        "scratch_to_resident": scratch,
                        "position_context_exact": position_ok,
                        "finite_state": finite,
                        "next_cycle_strict_schedule_isolation": bool(
                            candidate_decision_exact and scratch["passed"] and position_ok
                        ),
                    }
                )
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    final_memory = memory_stats()
    scratch_failures = [
        row["cycle"] for row in cycle_reviews if not row["scratch_to_resident"]["passed"]
    ]
    position_failures = [
        row["cycle"] for row in cycle_reviews if not row["position_context_exact"]
    ]
    finite_failures = [
        row["cycle"]
        for row in cycle_reviews
        if row["finite_state"] is not None and not row["finite_state"]["passed"]
    ]
    review_hash = hashlib.sha256(
        json.dumps(cycle_reviews, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checks = {
        "reject_and_full_accept_covered": (
            {int(row["strict_accepted"]) for row in cycle_reviews} >= {0, 1}
        ),
        "selected_scratch_to_resident_exact": not scratch_failures,
        "position_context_exact": not position_failures,
        "finite_state_and_kv": not finite_failures,
        "long_horizon": len(cycle_reviews) >= 32,
        "candidate_replay_deterministic": all(
            row["candidate_decision_matches_source"] for row in cycle_reviews
        ),
        "strict_schedule_isolation": all(
            row["next_cycle_strict_schedule_isolation"] for row in cycle_reviews
        ),
        "teardown_bounded": (
            int(final_memory["current_allocated_bytes"])
            - int(initial_memory["current_allocated_bytes"])
            <= 8
        ),
    }
    return {
        "schema": "hipengine.paro_mtp_fast_state_review.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "performance_claim": False,
        "model": str(model),
        "backend": backend,
        "source_capture": str(numerical_capture),
        "source_capture_sha256": source["capture_sha256"],
        "candidate_review_manifest_sha256": source["manifests"][
            "candidate_review_sha256"
        ],
        "prompt": source["prompt"],
        "audit_cycles": sorted(audit_ids),
        "checks": checks,
        "failures": {
            "scratch": scratch_failures,
            "position": position_failures,
            "finite": finite_failures,
        },
        "memory": {"initial": initial_memory, "final": final_memory},
        "state_review_sha256": review_hash,
        "cycles": cycle_reviews,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--numerical-capture", type=Path, required=True)
    parser.add_argument("--prompt-tokens-file", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        model=args.model,
        numerical_capture=args.numerical_capture,
        prompt_tokens_file=args.prompt_tokens_file,
        backend=args.backend,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
