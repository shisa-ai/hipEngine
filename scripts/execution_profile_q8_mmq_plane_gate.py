#!/usr/bin/env python3
"""Whole-model execution-profile gate for the Q8 MMQ activation-plane change.

Evaluates the PF-1d MMQ plane candidate (``Q8MMQPrefillPolicy.planes``) against
the strict teacher on the Qwen4Exp UD-Q4_K_XL canonical exact-token fixture:
full-vocabulary logits trajectories with the candidate consuming the strict
generated prefix at every compared transition, evaluated with the calibrated
mean/tail/max KL and top-1 thresholds, plus same-schedule repeat determinism.
The adapter flips only the runner-resolved MMQ policy object (dense-pair-gate
pattern) and restores it on exit; it makes no performance claim and no runtime
profile-selection change.

Phases per canonical case:
- strict teacher: MMQ admission disabled (policy None -> exact coltile chain)
- candidate x N: policy copy with ``--planes`` (default 2, the d4x2 chain)
An optional ``--context-planes 3`` capture records the incumbent production
chain against the same teacher in a second capture set for calibration
context; it never relaxes the candidate verdict.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.execution_profiles import (
    ExecutionProfile,
    VariantSelection,
    build_variant_manifest,
    resolve_runtime_profile,
)
from scripts.execution_profile_gdn_calibration import (
    CalibrationError,
    PromptCalibrationCapture,
    build_candidate_quality,
)

KIND = "hipengine_execution_profile_q8_mmq_plane_gate"
SCHEMA_VERSION = 1
DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_FIXTURE = Path(
    "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"
)


def _load_fixture(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _trajectory_rows(result: Any) -> dict[str, Any]:
    logits = result.logits
    if logits is None:
        raise CalibrationError("gate requires full-vocabulary logits")
    return {
        "token_id": int(result.token_id),
        "logits": np.ascontiguousarray(logits, dtype=np.float32),
    }


def _strict_trajectory(runner: Any, prompt_ids: Sequence[int], decode_steps: int):
    runner.reset()
    result = runner.prefill([int(t) for t in prompt_ids], capture_logits=True)
    trajectory = [_trajectory_rows(result)]
    for _ in range(int(decode_steps)):
        result = runner.step(int(result.token_id), capture_logits=True)
        trajectory.append(_trajectory_rows(result))
    return tuple(trajectory)


def _forced_trajectory(
    runner: Any,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
):
    runner.reset()
    result = runner.prefill([int(t) for t in prompt_ids], capture_logits=True)
    trajectory = [_trajectory_rows(result)]
    for token_id in forced_input_ids:
        result = runner.step(int(token_id), capture_logits=True)
        trajectory.append(_trajectory_rows(result))
    return tuple(trajectory)


def _set_policy(runner: Any, policy: Any) -> None:
    runner._q8_mmq_policy = policy


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    if int(args.repeat_runs) < 3:
        raise CalibrationError("plane gate requires at least three repeats")

    fixture, fixture_sha256 = _load_fixture(REPO_ROOT / args.fixture)
    cases = list(fixture["cases"])
    if args.case_id:
        selected = set(str(cid) for cid in args.case_id)
        cases = [row for row in cases if str(row["id"]) in selected]
        if {str(row["id"]) for row in cases} != selected:
            raise CalibrationError("unknown canonical case id in --case-id")
    if args.limit:
        cases = cases[: int(args.limit)]
    decode_steps = int(args.decode_steps)
    transitions = int(fixture.get("decode_transitions", 128))
    if decode_steps > transitions:
        raise CalibrationError(
            f"decode steps {decode_steps} exceed fixture transitions {transitions}"
        )
    max_sequence_length = (
        max(int(row["prompt_tokens"]) for row in cases) + transitions + 8
    )

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    model_root = args.model_root.resolve()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(str(args.execution_profile)),
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    generator = resolved.construct_generator(factory)
    runner = generator.runner
    if runner is None:
        raise CalibrationError("runner is not resident")

    original_policy = runner._q8_mmq_policy
    if original_policy is None:
        raise CalibrationError(
            "production profile did not bind a Q8 MMQ policy; gate is vacuous"
        )
    candidate_policy = replace(original_policy, planes=int(args.planes))
    context_policy = (
        replace(original_policy, planes=int(args.context_planes))
        if args.context_planes
        else None
    )
    if int(args.planes) == 3:
        raise CalibrationError("candidate planes must differ from the incumbent 3")

    captures: list[PromptCalibrationCapture] = []
    context_captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    command = " ".join(["uv", "run", "python", __file__, *sys.argv[1:]])
    try:
        for index_, row in enumerate(cases):
            prompt_id = str(row["id"])
            tokens = [int(t) for t in row["prompt_token_ids"]]
            _set_policy(runner, None)
            strict = _strict_trajectory(runner, tokens, decode_steps)
            forced = [step["token_id"] for step in strict[:-1]]
            _set_policy(runner, candidate_policy)
            runs = tuple(
                _forced_trajectory(runner, tokens, forced)
                for _ in range(int(args.repeat_runs))
            )
            captures.append(
                PromptCalibrationCapture(
                    prompt_id=prompt_id,
                    category=str(row["category"]),
                    strict=strict,
                    candidate_runs={"candidate": runs},
                )
            )
            if context_policy is not None:
                _set_policy(runner, context_policy)
                context_runs = tuple(
                    _forced_trajectory(runner, tokens, forced)
                    for _ in range(int(args.repeat_runs))
                )
                context_captures.append(
                    PromptCalibrationCapture(
                        prompt_id=prompt_id,
                        category=str(row["category"]),
                        strict=strict,
                        candidate_runs={"candidate": context_runs},
                    )
                )
            _set_policy(runner, original_policy)
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256")
                    or hashlib.sha256(
                        np.asarray(tokens, dtype="<i8").tobytes()
                    ).hexdigest(),
                }
            )
            suffix = (
                f" + context x {args.repeat_runs}" if context_policy else ""
            )
            print(
                f"{index_ + 1}/{len(cases)} {prompt_id}: strict + "
                f"candidate(planes={args.planes}) x {args.repeat_runs}{suffix}",
                flush=True,
            )
    finally:
        _set_policy(runner, original_policy)

    evaluated = build_candidate_quality(
        captures,
        candidate_mode="candidate",
        scenario_id=f"qwen4exp-ud-q4-k-xl-mmq-plane{args.planes}-c1-teacher-forced",
        thresholds=EvaluationThresholds(),
    )
    context_evaluated = (
        build_candidate_quality(
            context_captures,
            candidate_mode="candidate",
            scenario_id=(
                "qwen4exp-ud-q4-k-xl-mmq-plane"
                f"{args.context_planes}-c1-teacher-forced-context"
            ),
            thresholds=EvaluationThresholds(),
        )
        if context_captures
        else None
    )

    manifests = {
        "candidate_evidentiary": build_variant_manifest(
            profile="production",
            backend="hip_gfx1151",
            model=QWEN4_EXP_MODEL,
            quant=QWEN4_EXP_QUANTS[1],
            kv_policy="paged_bf16",
            graph_policy="request_owned_exact_moe_graph_c1",
            selections=(
                VariantSelection(
                    layer="linear",
                    scope="prefill_policy_qwen4exp_dense_q8_shapes",
                    selected_variant=(
                        "mmq128_prefill_q8_1_d4x2_guarded_f32_f32_out"
                        if int(args.planes) == 2
                        else f"mmq128_prefill_q8_1_d4x{int(args.planes)}_guarded_f32_f32_out"
                    ),
                    strict_fallback_variant="coltile8_rowbatch4_f32_f32_out",
                    registry_quant="gguf_q8_0",
                ),
            ),
        )
    }

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1151",
        resolved_backend=str(getattr(runner, "backend", "hip_gfx1151")),
        target_arch=str(os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151")),
        model_path=model_root,
        quant="gguf_ud_q4_k_xl",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": os.environ.get(
                "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"
            ),
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": os.environ.get(
                "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE"
            ),
        },
        build_profile="execution_profile_q8_mmq_plane_gate",
        timing_protocol="none_full_logits_only_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    generator.close()

    quality = evaluated["quality"]
    quality_passed = bool(
        quality.get("hard_gates_passed") and quality.get("finite")
        and not quality.get("requires_outlier_review", False)
    )
    deterministic = bool(evaluated["repeat_determinism"]["passed"])
    measurement_valid = bool(quality_passed and deterministic and not provenance.get("dirty"))
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "profile_qualification_claim": quality_passed and deterministic,
        "qualification_blockers": [] if measurement_valid else [
            blocker
            for blocker, ok in (
                ("calibrated quality thresholds", quality_passed),
                ("same-schedule repeat determinism", deterministic),
                ("clean worktree provenance", not provenance.get("dirty")),
            )
            if not ok
        ],
        "route": {
            "surface": "runner _q8_mmq_policy (dense-pair-gate pattern)",
            "policy_restored_after_capture": True,
            "candidate_planes": int(args.planes),
            "context_planes": int(args.context_planes) if args.context_planes else None,
            "strict_teacher": "policy None -> exact coltile8_rowbatch4 chain",
            "incumbent_planes": 3,
        },
        "protocol": {
            "model_root": str(model_root),
            "fixture": str((REPO_ROOT / args.fixture).resolve()),
            "fixture_sha256": fixture_sha256,
            "case_count": len(cases),
            "decode_steps": decode_steps,
            "candidate_repeat_runs": int(args.repeat_runs),
            "execution_profile": str(args.execution_profile),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "same_context_rule": (
                "candidate consumes the strict generated token prefix at every "
                "compared transition"
            ),
            "thresholds_evaluated": EvaluationThresholds().to_dict(),
            "profile_manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
        },
        "quality": evaluated["quality"],
        "repeat_determinism": evaluated["repeat_determinism"],
        "strict_logits_sha256": evaluated["strict_logits_sha256"],
        "candidate_logits_sha256": evaluated["candidate_logits_sha256"],
        "context_quality": (
            context_evaluated["quality"] if context_evaluated else None
        ),
        "context_repeat_determinism": (
            context_evaluated["repeat_determinism"] if context_evaluated else None
        ),
        "variant_manifests": manifests,
        "prompts": prompt_manifest,
        "provenance": provenance,
    }
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planes", type=int, default=2, choices=(2,))
    parser.add_argument("--context-planes", type=int, default=None, choices=(3,))
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--execution-profile", choices=("production",), default="production")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=False) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "quality_passed": bool(
                    artifact["quality"].get("hard_gates_passed")
                    and artifact["quality"].get("finite")
                    and not artifact["quality"].get("requires_outlier_review", False)
                ),
                "repeat_deterministic": artifact["repeat_determinism"]["passed"],
                "output": str(args.output),
            },
            indent=1,
        )
    )
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
