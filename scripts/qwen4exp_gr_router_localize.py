#!/usr/bin/env python3
"""Localize the GR family's drift: projection boundary to model-visible flips.

The GR up projection's iu8 route differs from the strict parent by 1-3 fp32
ulps at the projection boundary (measured on actual model inputs), yet the
780-row production gate rejects it with 8/780 top-1 flips. This packet tests
the amplification path that the dataflow predicts: the GR output (`mixed`) is
the MoE router's input, and `qwen35_router_select` is a discrete top-k, so a
sub-ulp-scale perturbation of `mixed` can flip an expert selection when two
router logits are within the drift.

Per teacher-forced decode step, for both arms, it records:

* the final-logits drift at the teacher's top-2 tokens (the flip-driving
  quantity) and the teacher's top-2 margin;
* how many router selections differ from the other arm, over all layers;
* the router-logit drift at the layers whose selection differs.

Diagnostic only: no runtime default changes, no performance claim.

The continuous amplification is already measured: the gate is a cliff at a
prefill-boundary logits KL of ~5e-9, above which the 64-step decode horizon
saturates at 1.31e-3 - 1.61e-3 regardless of perturbation size (see
benchmarks/results/2026-09-22-gr-drift-localization/). This packet therefore
targets the one remaining *discrete* step in the dataflow rather than the
amplification factor itself.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--decode-steps", type=int, default=64)
    args = parser.parse_args()
    if not hip_available():
        parser.error("HIP runtime unavailable")

    from hipengine.core.runtime import MemcpyKind
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,
    )
    from scripts.qwen4exp_canonical_ar_bench import (
        DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata,
    )
    from scripts.qwen4exp_framework_family_refresh import model_identity
    from scripts.qwen4exp_layer2_profile_gate import (
        CANDIDATES, _make_generator, _strict_trajectory,
    )
    from scripts.qwen4exp_q8_repair_depth_gate import resolve_allocation_profile
    import hipengine.runtime.qwen4_exp_runner as runtime_module

    if args.decode_steps < 1:
        parser.error("positive decode-steps required")
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    spec = CANDIDATES["production_gr_up_restore"]
    fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    cases = fixture["cases"]
    if args.case_id:
        requested = set(args.case_id)
        if not requested <= {case["id"] for case in cases}:
            parser.error("unknown canonical case")
        cases = [case for case in cases if case["id"] in requested]
    # The context only has to cover the case's own prompt plus the forced
    # decode: the drift enters during the prefill, so a short prompt exercises
    # the same mechanism at a fraction of the KV footprint.
    args.max_sequence_length = max(
        case["prompt_tokens"] for case in cases
    ) + args.decode_steps + 8
    args.prefill_chunk_size = 1024

    report = {
        "schema": 1,
        "kind": "qwen4exp_gr_router_localization",
        "status": "running",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "model": model_identity(args.model_root),
        "command": sys.argv,
        "fixture_sha256": fixture_hash,
        "candidate": spec.name,
        "candidate_environment": spec.environment,
        "arithmetic_class": spec.classification,
        "performance_claim": False,
        "promotion_claim": False,
        "boundary": (
            "GR up projection (K320/N10240) -> mixed -> MoE router logits -> "
            "discrete top-k expert selection -> final logits"
        ),
        "cases": [],
    }

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resolve_allocation_profile()
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()

        def arm_trajectory(environment: dict[str, str], case):
            """Run one arm, recording per-step routing and logits evidence."""

            previous = {key: os.environ.get(key) for key in environment}
            os.environ.update(environment)
            generator, _profile, _ = _make_generator(args, "production")
            runtime = generator.runner.runtime
            original_select = runtime_module.qwen35_router_select
            steps: list[dict] = []
            pending: dict[str, object] = {"layers": [], "selected": [], "logits": []}

            def download(ptr: int, shape, dtype):
                value = np.empty(shape, dtype=dtype)
                runtime.memcpy(
                    value.ctypes.data, ptr, value.nbytes, MemcpyKind.DEVICE_TO_HOST
                )
                return value

            def wrapped(logits_ptr, selected_ptr, routing_ptr, tokens,
                        logits_stride, num_experts, top_k, **kwargs):
                result = original_select(
                    logits_ptr, selected_ptr, routing_ptr, tokens, logits_stride,
                    num_experts, top_k, **kwargs,
                )
                # Only the routed-MoE layers call this; capture the discrete
                # selection and the logits that produced it.
                runtime.device_synchronize()
                pending["selected"].append(
                    download(selected_ptr, (int(tokens), int(top_k)), np.int32)
                )
                pending["logits"].append(
                    download(logits_ptr, (int(tokens), int(logits_stride)), np.float32)
                )
                return result

            try:
                runtime_module.qwen35_router_select = wrapped
                trajectory = _strict_trajectory(
                    generator.runner, case["prompt_token_ids"], args.decode_steps
                )
                for index, sample in enumerate(trajectory):
                    if pending["selected"]:
                        steps.append({
                            "step": index,
                            "logits": sample["logits"],
                            "selected": list(pending["selected"]),
                            "router_logits": list(pending["logits"]),
                        })
                    pending["selected"] = []
                    pending["logits"] = []
            finally:
                runtime_module.qwen35_router_select = original_select
                generator.close()
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            return steps

        try:
            for case in cases:
                print("strict", case["id"], flush=True)
                strict = arm_trajectory({"HIPENGINE_QWEN4_EXP_GR_IU8": "0"}, case)
                print("candidate", case["id"], flush=True)
                candidate = arm_trajectory(dict(spec.environment), case)
                if len(strict) != len(candidate):
                    raise ValueError("arm step counts differ")

                rows = []
                for left, right in zip(strict, candidate):
                    if len(left["selected"]) != len(right["selected"]):
                        raise ValueError("routed-layer counts differ between arms")
                    differing_layers, selection_deltas = [], []
                    for layer_index, (a, b) in enumerate(
                        zip(left["selected"], right["selected"])
                    ):
                        delta = int(np.count_nonzero(a != b))
                        selection_deltas.append(delta)
                        if delta:
                            differing_layers.append(layer_index)
                    router_drift = []
                    for layer_index in differing_layers:
                        a = left["router_logits"][layer_index]
                        b = right["router_logits"][layer_index]
                        router_drift.append({
                            "layer": layer_index,
                            "max_abs": float(np.abs(b - a).max()),
                            "median_abs": float(np.median(np.abs(b - a))),
                        })
                    ref = np.asarray(left["logits"], dtype=np.float64)
                    cand = np.asarray(right["logits"], dtype=np.float64)
                    order = np.argsort(-ref)
                    top1, top2 = int(order[0]), int(order[1])
                    margin = float(ref[top1] - ref[top2])
                    delta_top1 = float(cand[top1] - ref[top1])
                    delta_top2 = float(cand[top2] - ref[top2])
                    rows.append({
                        "step": int(left["step"]),
                        "routed_layers": len(left["selected"]),
                        "routing_differing_layers": len(differing_layers),
                        "routing_differing_selections": int(sum(selection_deltas)),
                        "routing_max_selections_per_layer": (
                            int(max(selection_deltas)) if selection_deltas else 0
                        ),
                        "strict_margin": margin,
                        "strict_top1": top1,
                        "strict_top2": top2,
                        "candidate_top1": int(np.argmax(cand)),
                        "top1_equal": bool(int(np.argmax(cand)) == top1),
                        "delta_top1": delta_top1,
                        "delta_top2": delta_top2,
                        "flip_driving_delta": abs(delta_top1 - delta_top2),
                        "max_abs_logit_delta": float(np.abs(cand - ref).max()),
                        "median_abs_logit_delta": float(np.median(np.abs(cand - ref))),
                        "router_drift": router_drift,
                    })
                flips = [row for row in rows if not row["top1_equal"]]
                routing_layers = [row["routing_differing_layers"] for row in rows]
                report["cases"].append({
                    "id": case["id"],
                    "prompt_tokens": case["prompt_tokens"],
                    "steps": len(rows),
                    "top1_flips": len(flips),
                    "steps_with_routing_difference": int(sum(1 for v in routing_layers if v)),
                    "routing_differing_layers_total": int(sum(routing_layers)),
                    "flip_rows": flips,
                    "rows": rows,
                })
                summary = {
                    key: report["cases"][-1][key]
                    for key in ("id", "steps", "top1_flips",
                                "steps_with_routing_difference",
                                "routing_differing_layers_total")
                }
                print(json.dumps(summary), flush=True)
            report["status"] = "completed_diagnostic"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items()
                              if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
