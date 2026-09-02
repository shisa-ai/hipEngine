#!/usr/bin/env python3
"""Profile the Qwen4Exp MTP full-vocabulary draft-head boundary on gfx1151."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata

DEFAULT_MODEL = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_SIDECAR = Path(
    "/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/"
    "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
)
DEFAULT_FULL_SUITE = ROOT / "benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-fullsuite-short.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--fullsuite-artifact", type=Path, default=DEFAULT_FULL_SUITE)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--candidate-budget", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _distribution(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    return {
        "samples_ms": values,
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model_root.is_dir() or not args.sidecar.is_file():
        raise ValueError("model root and MTP sidecar must exist")
    if args.warmups < 1 or args.iterations < 3:
        raise ValueError("at least one warmup and three measured iterations are required")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, host_array_ptr, memory_stats, reset_memory_stats
    from hipengine.loading.gguf import GGUFReader, discover_gguf_files
    from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
    from hipengine.loading.qwen4_exp_mtp_gguf import build_qwen4_exp_mtp_gguf_map
    from hipengine.loading.qwen4_exp_mtp_materialize import (
        materialize_qwen4_exp_mtp_weights,
        plan_qwen4_exp_mtp_residency,
    )
    from hipengine.runtime.gguf_linear import (
        GGUF_ACTIVATION_F32,
        GGUF_OUTPUT_F32,
        launch_gguf_linear,
    )
    from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner

    runtime = get_hip_runtime()
    reset_memory_stats()
    target_info = GGUFReader(discover_gguf_files(args.model_root)[0]).info
    target_config = qwen4_exp_gguf_config_from_metadata(target_info)
    reader = GGUFReader(args.sidecar)
    model_map = build_qwen4_exp_mtp_gguf_map((reader.info,))
    plan = plan_qwen4_exp_mtp_residency(model_map)
    resident = materialize_qwen4_exp_mtp_weights(
        (reader,), plan=plan, backend="hip_gfx1151", runtime=runtime
    )
    runner = None
    payload: dict[str, Any]
    try:
        runner = Qwen4ExpGGUFMTPDraftRunner(
            resident,
            target_config=target_config,
            max_sequence_length=max(64, args.warmups + args.iterations + 1),
            backend="hip_gfx1151",
            runtime=runtime,
        )
        hidden = np.zeros(target_config.residual_width, dtype=np.float32)
        token = 248_068
        for _ in range(args.warmups):
            result = runner.forward(token, hidden)
            token, hidden = result.token_id, result.hidden_seed

        runner.reset()
        hidden.fill(0)
        token = 248_068
        forward_ms: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            result = runner.forward(token, hidden)
            forward_ms.append((time.perf_counter() - started) * 1e3)
            token, hidden = result.token_id, result.hidden_seed

        weight = resident.weight("root.lm_head")
        logits = np.empty(target_config.vocab_size, dtype=np.float32)

        def launch_head() -> None:
            launch_gguf_linear(
                weight,
                runner.head_scratch.mixed.ptr,
                runner.logits_buffer.ptr,
                1,
                target_config.hidden_size,
                target_config.vocab_size,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32,
                runtime=runtime,
            )

        for _ in range(args.warmups):
            launch_head()
            runtime.device_synchronize()
        head_ms: list[float] = []
        for _ in range(args.iterations):
            runtime.device_synchronize()
            started = time.perf_counter()
            launch_head()
            runtime.device_synchronize()
            head_ms.append((time.perf_counter() - started) * 1e3)

        d2h_ms: list[float] = []
        for _ in range(args.iterations):
            runtime.device_synchronize()
            started = time.perf_counter()
            copy_device_to_host(
                host_array_ptr(logits), runner.logits_buffer, logits.nbytes, runtime=runtime
            )
            d2h_ms.append((time.perf_counter() - started) * 1e3)

        proposal_debug_ms: list[float] = []
        proposal_compact_ms: list[float] = []
        proposal_ids_exact = True
        for _ in range(args.warmups + args.iterations):
            runner.reset()
            started = time.perf_counter()
            debug = runner.propose_chain(
                start_token=248_068,
                target_hidden_seed=np.zeros(target_config.residual_width, dtype=np.float32),
                draft_n_max=args.candidate_budget,
                compact_output=False,
            )
            debug_elapsed = (time.perf_counter() - started) * 1e3
            runner.reset()
            started = time.perf_counter()
            compact = runner.propose_chain(
                start_token=248_068,
                target_hidden_seed=np.zeros(target_config.residual_width, dtype=np.float32),
                draft_n_max=args.candidate_budget,
                compact_output=True,
            )
            compact_elapsed = (time.perf_counter() - started) * 1e3
            proposal_ids_exact &= [row.token_id for row in debug] == [
                row.token_id for row in compact
            ]
            if len(proposal_debug_ms) >= args.warmups:
                proposal_debug_ms.append(debug_elapsed)
                proposal_compact_ms.append(compact_elapsed)
            else:
                proposal_debug_ms.append(debug_elapsed)
                proposal_compact_ms.append(compact_elapsed)
        proposal_debug_ms = proposal_debug_ms[args.warmups :]
        proposal_compact_ms = proposal_compact_ms[args.warmups :]

        head_median = statistics.median(head_ms)
        forward_median = statistics.median(forward_ms)
        bytes_per_row = int(weight.spec.device_nbytes) // target_config.vocab_size
        sizes = (8_192, 16_384, 32_768, 65_536)
        selected_head_bytes = {str(size): size * bytes_per_row for size in sizes}
        fullsuite = json.loads(args.fullsuite_artifact.read_text())
        result = fullsuite["result"]
        proposed = int(result["proposed_draft_tokens"])
        ar_seconds = float(result["ar_total_seconds"])
        mtp_seconds = float(result["mtp_total_seconds"])
        ideal_saved = proposed * float(head_median) / 1e3
        payload = {
            "schema": 1,
            "date": date.today().isoformat(),
            "kind": "qwen4exp_mtp_draft_head_profile",
            "status": "diagnostic_retained",
            "performance_claim": False,
            "command": list(command),
            "source": _git_metadata(ROOT),
            "host": _host_metadata(),
            "model": {
                "target_root": str(args.model_root),
                "sidecar": str(args.sidecar),
                "vocab": target_config.vocab_size,
                "hidden": target_config.hidden_size,
                "head_quant": weight.spec.source.ggml_type_name,
                "head_bytes": int(weight.spec.device_nbytes),
                "logit_bytes": int(logits.nbytes),
            },
            "protocol": {
                "warmups": int(args.warmups),
                "iterations": int(args.iterations),
                "start_token": 248_068,
                "hidden_seed": "zeros_fp32",
                "boundary": "synchronized c1 draft forward and isolated synchronized full Q8_0 head",
            },
            "measurement": {
                "full_draft_forward": _distribution(forward_ms),
                "full_head": _distribution(head_ms),
                "full_logits_d2h": _distribution(d2h_ms),
                "head_share_of_draft_forward": float(head_median / forward_median),
                "proposal_debug": _distribution(proposal_debug_ms),
                "proposal_compact": _distribution(proposal_compact_ms),
                "proposal_ids_exact": bool(proposal_ids_exact),
                "proposal_compact_speedup": float(
                    statistics.median(proposal_debug_ms)
                    / statistics.median(proposal_compact_ms)
                ),
            },
            "selected_q8_row_geometry": {
                "bytes_per_vocab_row": bytes_per_row,
                "incremental_head_bytes": selected_head_bytes,
                "note": "GGUF Q8_0 output rows are independently addressable; no EXL3 Hadamard-group constraint transfers.",
            },
            "current_fullsuite_ceiling": {
                "artifact": str(args.fullsuite_artifact.relative_to(ROOT)),
                "proposed_draft_tokens": proposed,
                "mtp_total_seconds": mtp_seconds,
                "ar_total_seconds": ar_seconds,
                "measured_mtp_vs_ar": float(ar_seconds / mtp_seconds),
                "ideal_zero_cost_head_saved_seconds": ideal_saved,
                "ideal_zero_cost_head_wall_reduction_percent": float(ideal_saved / mtp_seconds * 100),
                "ideal_zero_cost_head_mtp_vs_ar": float(ar_seconds / (mtp_seconds - ideal_saved)),
                "note": "Upper bound only; a selected head costs nonzero time and may lower acceptance.",
            },
            "decision": {
                "transfer": "useful_after_device-output cleanup, not the current first MTP bottleneck",
                "first": "replace full-logit D2H/NumPy argmax with existing device argmax and compact candidate packet",
                "candidate": "individual-row compact Q8_0 draft heads at 8K/16K/32K with local-to-global ID map",
                "binding_gate": "full category+heldout AR-equivalence and same-command true-AR economics",
            },
        }
    finally:
        if runner is not None:
            runner.close()
        resident.close()
    payload["lifecycle"] = {
        "after_close": memory_stats(),
        "passed": memory_stats()["current_allocated_bytes"] == 0,
    }
    return payload


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args, command=[Path(sys.argv[0]).name, *sys.argv[1:]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "head_median_ms": payload["measurement"]["full_head"]["median_ms"],
        "draft_median_ms": payload["measurement"]["full_draft_forward"]["median_ms"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
