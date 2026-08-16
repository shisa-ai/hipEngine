#!/usr/bin/env python3
"""Counterbalanced Qwen3.8 dense-pair route performance A/B.

This harness changes only the shape-scoped gfx1151 gate/up+SiLU owner inside a
single resident session. It is intentionally route-local: both variants run
under the registered strict GDN mode so unrelated changed arithmetic cannot
confound the comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _gguf_gdn_prefill_backend_exact_mode,
    _gguf_policy_identity,
)
from scripts.execution_profile_gdn_calibration import CalibrationError
from scripts.execution_profile_gguf_dense_pair_gate import (
    DEFAULT_CANDIDATE_VARIANT,
    DEFAULT_GDN_MODE,
    DEFAULT_MODEL,
    DEFAULT_STRICT_VARIANT,
    POLICY_CAPABILITY,
    REGISTRY_LAYER,
    REGISTRY_QUANT,
    dense_pair_policy_override,
    validate_route_variants,
)
from scripts.gguf_gdn_semantic_gate import _configure_gate_environment
from scripts.qwen35_gguf_bench import _RoctxProfilerControl, _run_existing_session_once

KIND = "qwen38_gfx1151_dense_pair_counterbalanced_perf"
SHAPE = (1, 5_120, 17_408)


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize measured strict/candidate rows and matched-pair ratios."""

    samples = {
        label: [
            float(row["decode_tok_s"])
            for row in runs
            if str(row["label"]) == label
        ]
        for label in ("strict", "candidate")
    }
    if not samples["strict"] or len(samples["strict"]) != len(samples["candidate"]):
        raise ValueError("strict and candidate samples must be non-empty and balanced")
    medians = {label: statistics.median(values) for label, values in samples.items()}
    pair_ids = sorted({int(row["pair"]) for row in runs})
    paired_ratios: list[float] = []
    for pair_id in pair_ids:
        by_label = {
            str(row["label"]): float(row["decode_tok_s"])
            for row in runs
            if int(row["pair"]) == pair_id
        }
        if set(by_label) != {"strict", "candidate"}:
            raise ValueError(f"pair {pair_id} does not contain one row per variant")
        paired_ratios.append(by_label["candidate"] / by_label["strict"])
    return {
        "samples": samples,
        "medians": medians,
        "candidate_over_strict": medians["candidate"] / medians["strict"],
        "paired_ratios": paired_ratios,
        "paired_median": statistics.median(paired_ratios),
        "candidate_wins": sum(ratio > 1.0 for ratio in paired_ratios),
    }


def _run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    validate_route_variants(args.strict_variant, args.candidate_variant)
    if not args.model.is_file():
        raise CalibrationError(f"model does not exist: {args.model}")
    if args.compiler_version_file is None or not args.compiler_version_file.is_file():
        raise CalibrationError("a readable compiler-version file is required")
    if int(args.pairs) < 1:
        raise CalibrationError("pairs must be positive")
    if int(args.prefill_tokens) <= 0 or int(args.decode_tokens) <= 0:
        raise CalibrationError("prefill and decode token counts must be positive")
    if not args.require_cached_build:
        raise CalibrationError("performance capture requires --require-cached-build")

    _configure_gate_environment(decode_repack=bool(args.decode_repack))
    os.environ["HIPENGINE_GGUF_GDN_PREFILL_MODE"] = str(args.gdn_mode)
    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package(str(args.backend))
    for variant in (str(args.strict_variant), str(args.candidate_variant)):
        key = KernelKey(str(args.backend), REGISTRY_LAYER, REGISTRY_QUANT, variant)
        if not is_registered(key):
            raise CalibrationError(f"dense pair route is not registered: {key.display()}")

    package = __import__(
        f"hipengine.kernels.{args.backend}", fromlist=[POLICY_CAPABILITY]
    )
    original_policy = copy.deepcopy(getattr(package, POLICY_CAPABILITY))
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    runtime = get_hip_runtime()
    runs: list[dict[str, Any]] = []
    identity_repr = "unresolved"
    owner_verified = False
    restored = False
    try:
        with Qwen35GGUFResidentSession(
            args.model,
            backend=str(args.backend),
            runtime=runtime,
            compiler_version=compiler_version,
            require_cached_build=True,
            max_sequence_length=(
                int(args.prefill_tokens) + int(args.decode_tokens) + 8
            ),
            use_wmma_prefill=True,
            use_gemv_decode=True,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=0),
        ) as session:
            if session.runner is None:
                raise CalibrationError("GGUF resident session closed during setup")
            backend = str(session.runner.backend)
            exact_mode = _gguf_gdn_prefill_backend_exact_mode(backend)
            if str(args.gdn_mode) != exact_mode:
                raise CalibrationError(
                    f"strict GDN mode mismatch: requested={args.gdn_mode!r}, "
                    f"backend_exact={exact_mode!r}"
                )
            identity = _gguf_policy_identity(session.runner.weights)
            if identity is None:
                raise CalibrationError("model does not expose a dense GGUF policy identity")
            identity_repr = repr(identity)
            current = original_policy.get(identity, {}).get(SHAPE)
            if current != args.candidate_variant:
                raise CalibrationError(
                    "candidate is not the current package owner: "
                    f"package={current!r}, candidate={args.candidate_variant!r}"
                )
            owner_verified = True
            roctx = _RoctxProfilerControl(enabled=False)
            prompt_tokens = [int(args.prompt_token_id)] * int(args.prefill_tokens)

            def run_once(
                label: str, variant: str, pair: int, *, measured: bool
            ) -> dict[str, Any]:
                setattr(
                    package,
                    POLICY_CAPABILITY,
                    dense_pair_policy_override(
                        original_policy,
                        identity=identity,
                        shape=SHAPE,
                        variant=variant,
                    ),
                )
                result = _run_existing_session_once(
                    session=session,
                    runtime=runtime,
                    model=args.model,
                    quant="gguf_q4_k_m",
                    prompt_tokens=prompt_tokens,
                    decode_tokens=int(args.decode_tokens),
                    warmup_decode_tokens=0,
                    graph_replay_decode=True,
                    graph_steps_per_replay=1,
                    use_bulk_prefill=True,
                    bulk_attention_mode="bulk",
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                    prefill_chunk_size=0,
                    measured=measured,
                    run_index=pair,
                    load_seconds=0.0,
                    persistent_session=True,
                    graph_holder=None,
                    roctx=roctx,
                    rocprof_selected_region="none",
                    gpu_stage_timings=False,
                )
                row = {
                    "label": label,
                    "variant": variant,
                    "pair": pair,
                    "measured": measured,
                    "prefill_tok_s": result["throughput"]["prefill_tok_s"],
                    "decode_tok_s": result["throughput"]["decode_tok_s"],
                    "decode_seconds": result["timings"][
                        "decode_seconds_excluding_graph_capture"
                    ],
                    "graph_capture_seconds": result["timings"][
                        "graph_capture_seconds"
                    ],
                    "correctness": result["correctness_sanity"],
                    "memory": result["memory"],
                }
                print(label, pair, row["decode_tok_s"], flush=True)
                return row

            run_once("strict", str(args.strict_variant), 0, measured=False)
            run_once("candidate", str(args.candidate_variant), 0, measured=False)
            for pair in range(1, int(args.pairs) + 1):
                order = (
                    (
                        ("strict", str(args.strict_variant)),
                        ("candidate", str(args.candidate_variant)),
                    )
                    if pair % 2
                    else (
                        ("candidate", str(args.candidate_variant)),
                        ("strict", str(args.strict_variant)),
                    )
                )
                for label, variant in order:
                    runs.append(run_once(label, variant, pair, measured=True))
    finally:
        setattr(package, POLICY_CAPABILITY, original_policy)
        restored = True

    summary = summarize_runs(runs)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch=str(args.backend).removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": os.environ.get(
                "HIPENGINE_GGUF_GDN_PREFILL_MODE"
            ),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get(
                "HIPENGINE_GGUF_DECODE_REPACK"
            ),
        },
        build_profile="qwen38_dense_pair_counterbalanced_perf",
        timing_protocol="persistent_counterbalanced_wall_v1",
        warmups=2,
        repetitions=int(args.pairs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    measurement_valid = bool(
        int(args.pairs) >= 7
        and owner_verified
        and restored
        and not provenance.get("dirty")
    )
    return {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "measurement_valid": measurement_valid,
        "performance_claim": measurement_valid,
        "model": str(args.model.resolve()),
        "hardware": {"device": "AMD Radeon 8060S Graphics", "arch": "gfx1151"},
        "route": {
            "policy_capability": POLICY_CAPABILITY,
            "policy_identity": identity_repr,
            "shape": list(SHAPE),
            "strict_variant": str(args.strict_variant),
            "candidate_variant": str(args.candidate_variant),
            "current_package_owner_verified": owner_verified,
            "policy_restored_after_capture": restored,
        },
        "protocol": {
            "persistent_session": True,
            "warmup_runs_per_variant": 1,
            "counterbalanced_pairs": int(args.pairs),
            "prefill_tokens": int(args.prefill_tokens),
            "decode_tokens": int(args.decode_tokens),
            "prompt_token_id": int(args.prompt_token_id),
            "graph_replay": True,
            "graph_steps_per_replay": 1,
            "strict_gdn_mode": str(args.gdn_mode),
            "timing": "decode wall time excluding graph capture",
        },
        "runs": runs,
        "summary": summary,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--strict-variant", default=DEFAULT_STRICT_VARIANT)
    parser.add_argument("--candidate-variant", default=DEFAULT_CANDIDATE_VARIANT)
    parser.add_argument("--gdn-mode", default=DEFAULT_GDN_MODE)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument(
        "--decode-repack", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = _run(args, command=command)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(artifact["summary"], indent=2))
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
