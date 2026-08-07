#!/usr/bin/env python3
"""Safe persistent-queue P5 gate for GGUF HIP-graph versus in-tree PM4.

The gate captures exactly one graph per submitted transport and reuses it for
all launches.  It never runs submit-plus-queue-recreate stress.  Every token,
FP32 hidden seed, recurrent state, live BF16 K/V prefix, and final logit is
compared against one eager reference from the same resident session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_decode_graph_g5 import (  # noqa: E402
    _compact_eager_correctness,
    _prefill,
    _run_eager_correctness,
    _run_relaunch_correctness,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_MAX_RECOVERY_DELTA = 64 * 1024 * 1024


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty compiler-version file: {path}")
    return value


def _read_logits(session: Any) -> np.ndarray:
    result = session._read_sample(return_logits=True)
    logits = np.ascontiguousarray(result.logits, dtype=np.float32).reshape(-1)
    if logits.size == 0 or not np.isfinite(logits).all():
        raise RuntimeError("final GGUF logits are empty or non-finite")
    return logits


def _logits_summary(logits: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(logits, dtype=np.float32)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "top1": int(np.argmax(values)),
        "finite": bool(np.isfinite(values).all()),
    }


def _closed_transport_proof(run: dict[str, Any]) -> dict[str, Any]:
    proof = run.get("transport_provenance", {}).get("closed")
    if not isinstance(proof, dict):
        raise RuntimeError("decode graph did not return closed transport provenance")
    return proof


def _native_proof_passed(run: dict[str, Any], *, transport: str, steps: int) -> bool:
    live = run.get("transport_provenance", {}).get("live", {})
    closed = _closed_transport_proof(run)
    executable = live.get("executable", {})
    context = live.get("context", {})
    return bool(
        live.get("transport") == transport
        and live.get("source") == "hipengine_in_tree_rocr_pm4"
        and live.get("native_fallbacks") == 0
        and live.get("launches") == int(steps)
        and live.get("launch_attempts") == int(steps)
        and live.get("node_count", 0) > 0
        and len(live.get("hsaco_sha256", ())) > 0
        and executable.get(f"{transport}_submissions") == int(steps)
        and executable.get("retired") is True
        and context.get("submissions") == int(steps)
        and context.get("unretired_submissions") == 0
        and context.get("usable") is True
        and closed.get("closed") is True
        and closed.get("native_fallbacks") == 0
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend != "hip_gfx1100" and args.submission_transport != "hipgraph":
        raise ValueError("aql/pm4 P5 gate is admitted only on hip_gfx1100")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if int(args.prompt_length) <= 0 or int(args.steps) < 3:
        raise ValueError("prompt-length must be positive and steps must be at least three")

    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    os.environ["HIPENGINE_GGUF_MOE_GRAPH"] = "0"
    target_arch = {
        "hip_gfx1100": "gfx1100",
        "hip_gfx1151": "gfx1151",
    }[str(args.backend)]
    os.environ["HIPENGINE_HIP_ARCH"] = target_arch
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file.expanduser().resolve()
        )

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prompt_ids = [int(args.prompt_token_id)] * int(args.prompt_length)
    max_sequence_length = int(args.prompt_length) + int(args.steps) + 8
    stat = model.stat()
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached),
        backend=str(args.backend),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        eager = _run_eager_correctness(session, prompt_ids=prompt_ids, steps=int(args.steps))
        eager_logits = _read_logits(session)
        reference = eager["checkpoints"]

        hipgraph = _run_relaunch_correctness(
            session,
            prompt_ids=prompt_ids,
            steps=int(args.steps),
            reference=reference,
            submission_transport="hipgraph",
        )
        hipgraph_logits = _read_logits(session)

        session.runtime.device_synchronize()
        native_free_before, total_bytes = session.runtime.mem_get_info()
        candidate = _run_relaunch_correctness(
            session,
            prompt_ids=prompt_ids,
            steps=int(args.steps),
            reference=reference,
            submission_transport=str(args.submission_transport),
        )
        candidate_logits = _read_logits(session)
        session.runtime.device_synchronize()
        native_free_after, total_after = session.runtime.mem_get_info()

        # Cancellation/close control: instantiate one graph and close it before
        # submission.  This is a safe create/drop control, not queue-recreate
        # submit stress.
        _prefill(session, prompt_ids)
        cancelled = session.capture_decode_graph(
            position=int(session.position),
            max_replay_steps=1,
            steps_per_replay=1,
            attention_max_context_len=int(session.position) + 1,
            submission_transport=str(args.submission_transport),
        )
        cancelled_live = cancelled.transport_provenance()
        cancelled.close()
        cancelled_closed = cancelled.transport_provenance()
        transport_context_teardown = session.close_decode_graph_submission_contexts()
        session.runtime.device_synchronize()
        context_free_after, context_total_after = session.runtime.mem_get_info()
        pci_bdf = session.runtime.device_pci_bus_id()

    expected_tokens = [int(args.expected_token_id)] * int(args.steps)
    eager_tokens = [int(token) for token in eager["generated_token_ids"]]
    hipgraph_tokens = [int(token) for token in hipgraph["generated_token_ids"]]
    candidate_tokens = [int(token) for token in candidate["generated_token_ids"]]
    logits_exact = bool(
        np.array_equal(eager_logits, hipgraph_logits)
        and np.array_equal(eager_logits, candidate_logits)
    )
    memory_recovered = bool(
        total_bytes == total_after == context_total_after
        and context_free_after + _MAX_RECOVERY_DELTA >= native_free_before
    )
    context_teardown = transport_context_teardown.get(str(args.submission_transport))
    context_teardown_passed = bool(
        args.submission_transport == "hipgraph"
        or (
            isinstance(context_teardown, dict)
            and context_teardown.get("before", {}).get("children") == 0
            and context_teardown.get("after", {}).get("closed") is True
            and context_teardown.get("after", {}).get("native_context_closed") is True
        )
    )
    cancelled_passed = bool(
        cancelled_live.get("launches") == 0
        and cancelled_live.get("submission_started") is False
        and cancelled_closed.get("closed") is True
        and cancelled_closed.get("native_fallbacks") == 0
        and context_teardown_passed
    )
    candidate_proof_passed = (
        _native_proof_passed(
            candidate,
            transport=str(args.submission_transport),
            steps=int(args.steps),
        )
        if args.submission_transport in {"aql", "pm4"}
        else bool(
            _closed_transport_proof(candidate).get("transport") == "hipgraph"
            and _closed_transport_proof(candidate).get("native_fallbacks") == 0
        )
    )
    passed = bool(
        eager_tokens == expected_tokens
        and hipgraph_tokens == eager_tokens
        and candidate_tokens == eager_tokens
        and hipgraph["passed"]
        and candidate["passed"]
        and logits_exact
        and candidate_proof_passed
        and cancelled_passed
        and memory_recovered
    )
    return {
        "schema_version": 1,
        "kind": "hipengine_pm4_gguf_decode_p5_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if passed else "rejected_correctness",
        "performance_claim": False,
        "passed": passed,
        "model": {
            "path": str(model),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "hardware": {
            "backend": str(args.backend),
            "gfx_arch": target_arch,
            "pci_bdf": pci_bdf,
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "gpu_max_hw_queues": os.environ.get("GPU_MAX_HW_QUEUES"),
        },
        "workload": {
            "prompt_source": "repeated_token_correctness_fixture",
            "prompt_token_id": int(args.prompt_token_id),
            "expected_token_id": int(args.expected_token_id),
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.steps),
            "candidate_transport": str(args.submission_transport),
        },
        "checks": {
            "expected_tokens": eager_tokens == expected_tokens,
            "hipgraph_tokens_exact": hipgraph_tokens == eager_tokens,
            "candidate_tokens_exact": candidate_tokens == eager_tokens,
            "hipgraph_state_kv_exact": bool(hipgraph["passed"]),
            "candidate_state_kv_exact": bool(candidate["passed"]),
            "final_logits_bit_exact": logits_exact,
            "candidate_transport_proof": candidate_proof_passed,
            "cancel_close": cancelled_passed,
            "memory_recovered_within_64mib": memory_recovered,
        },
        "logits": {
            "eager": _logits_summary(eager_logits),
            "hipgraph": _logits_summary(hipgraph_logits),
            "candidate": _logits_summary(candidate_logits),
        },
        "memory": {
            "free_before_candidate": native_free_before,
            "free_after_candidate_graph_close_context_retained": native_free_after,
            "free_after_transport_context_close": context_free_after,
            "total_bytes": total_bytes,
            "recovery_tolerance_bytes": _MAX_RECOVERY_DELTA,
        },
        "eager": _compact_eager_correctness(eager),
        "hipgraph": hipgraph,
        "candidate": candidate,
        "cancel_control": {
            "live": cancelled_live,
            "closed": cancelled_closed,
            "transport_context_teardown": transport_context_teardown,
        },
        "notes": [
            "Correctness/lifecycle gate only; no throughput claim.",
            "One graph and one queue are reused for all candidate submissions.",
            "No submit-plus-queue-recreate stress is performed.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument(
        "--submission-transport",
        choices=("hipgraph", "aql", "pm4"),
        default="pm4",
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
    )
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    encoded = json.dumps(payload, sort_keys=True, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
