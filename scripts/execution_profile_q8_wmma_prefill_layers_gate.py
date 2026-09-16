#!/usr/bin/env python3
"""Whole-model gate for the f16 WMMA Q8_0 dense prefill route.

Evaluates the layer-scoped f16 WMMA dense Q8_0 prefill selector
(``HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS``) against the exact coltile chain on the
Qwen4Exp UD-Q4_K_XL canonical exact-token fixture: full-vocabulary logits
trajectories with the candidate consuming the strict generated prefix at every
compared transition, evaluated with the calibrated mean/tail/max KL and top-1
thresholds, plus same-schedule repeat determinism.

The teacher is the named production profile with the selector cleared, so the
measured drift is the Q8 dense prefill route's own contribution and not the rest
of the production stack. That is the same candidate-local isolation the Q8 MMQ
plane gate uses.

The selector must be applied **after** the named production binder, which writes
it as ``""`` during its own pass; setting it before construction is silently
inert. This gate constructs the generator first and flips the selector per arm.

Phases per canonical case:
- teacher: selector empty -> exact ``coltile8_rowbatch4`` chain
- candidate x N: selector set to ``--layers`` -> ``wmma_prefill_f32_f32_out``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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

KIND = "hipengine_execution_profile_q8_wmma_prefill_layers_gate"
SCHEMA_VERSION = 1
SELECTOR_ENV = "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"
DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
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


def _set_selector(layers: str) -> None:
    """Flip the Q8 dense WMMA prefill selector. Empty means the exact chain."""

    os.environ[SELECTOR_ENV] = layers


def _warm(runner: Any, warmup_tokens: Sequence[int]) -> None:
    """Absorb a first-request dispatch before the measured trajectory.

    The dispatch resolution is memoized on the selector state, so a flip is
    expected to take effect immediately; the discarded request also warms the
    JIT so the measured trajectory is not the first launch of a new variant.
    """

    runner.reset()
    runner.prefill([int(t) for t in warmup_tokens[:32]], capture_logits=False)


def _parse_layers(raw: str) -> str:
    """Normalize ``--layers`` to the selector's comma-joined form."""

    values: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, _, end = item.partition("-")
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(item))
    if not values:
        raise CalibrationError("--layers must select at least one layer")
    unique = sorted(set(values))
    if any(layer < 0 or layer > 63 for layer in unique):
        raise CalibrationError("--layers must be within 0..63")
    return ",".join(str(layer) for layer in unique)


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
        raise CalibrationError("the gate requires at least three candidate repeats")

    fixture, fixture_sha256 = _load_fixture(REPO_ROOT / args.fixture)
    cases = list(fixture["cases"])
    if args.case_id:
        selected = {str(cid) for cid in args.case_id}
        cases = [row for row in cases if str(row["id"]) in selected]
        if {str(row["id"]) for row in cases} != selected:
            raise CalibrationError("unknown canonical case id in --case-id")
    if args.limit:
        cases = cases[: int(args.limit)]
    if not cases:
        raise CalibrationError("selected case set is empty")
    decode_steps = int(args.decode_steps)
    transitions = int(fixture.get("decode_transitions", 128))
    if decode_steps > transitions:
        raise CalibrationError(
            f"decode steps {decode_steps} exceed fixture transitions {transitions}"
        )
    max_sequence_length = (
        max(int(row["prompt_tokens"]) for row in cases) + transitions + 8
    )
    candidate_layers = _parse_layers(args.layers)

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
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    # Construct first: the named binder writes the selector as "" during its own
    # pass, so only a post-binder flip is effective.
    generator = resolved.construct_generator(factory)
    runner = generator.runner
    if runner is None:
        raise CalibrationError("runner is not resident")
    bound_selector = os.environ.get(SELECTOR_ENV)

    captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    command = ["python3", str(Path(__file__).resolve().relative_to(REPO_ROOT)), *sys.argv[1:]]
    try:
        for position, row in enumerate(cases):
            prompt_id = str(row["id"])
            tokens = [int(t) for t in row["prompt_token_ids"]]
            _set_selector("")
            _warm(runner, tokens)
            teacher = _strict_trajectory(runner, tokens, decode_steps)
            forced = [step["token_id"] for step in teacher[:-1]]
            _set_selector(candidate_layers)
            _warm(runner, tokens)
            runs = tuple(
                _forced_trajectory(runner, tokens, forced)
                for _ in range(int(args.repeat_runs))
            )
            captures.append(
                PromptCalibrationCapture(
                    prompt_id=prompt_id,
                    category=str(row["category"]),
                    strict=teacher,
                    candidate_runs={"candidate": runs},
                )
            )
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
            print(
                f"{position + 1}/{len(cases)} {prompt_id}: teacher + "
                f"candidate(layers={candidate_layers}) x {args.repeat_runs}",
                flush=True,
            )
    finally:
        _set_selector(bound_selector or "")

    evaluated = build_candidate_quality(
        captures,
        candidate_mode="candidate",
        scenario_id=(
            "qwen4exp-ud-q4-k-xl-q8-wmma-dense-prefill-layers-"
            f"{candidate_layers.replace(',', '_')}-c1-teacher-forced"
        ),
        thresholds=EvaluationThresholds(),
    )

    manifest = build_variant_manifest(
        profile="production",
        backend=QWEN4_EXP_BACKEND,
        model=QWEN4_EXP_MODEL,
        quant=QWEN4_EXP_QUANTS[1],
        kv_policy="paged_bf16_qsa_index_f32",
        graph_policy="request_owned_exact_moe_graph_c1",
        selections=(
            VariantSelection(
                layer="linear",
                scope="prefill_dense_q8_wmma_layers",
                selected_variant="wmma_prefill_f32_f32_out",
                strict_fallback_variant="coltile8_rowbatch4_f32_f32_out",
                registry_quant="gguf_q8_0",
            ),
        ),
    )
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
            SELECTOR_ENV: candidate_layers,
        },
        build_profile=KIND,
        timing_protocol="none_full_logits_only_v1",
        warmups=1,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    generator.close()

    quality = evaluated["quality"]
    quality_passed = bool(
        quality.get("hard_gates_passed")
        and quality.get("finite")
        and not quality.get("requires_outlier_review", False)
    )
    deterministic = bool(evaluated["repeat_determinism"]["passed"])
    # Validity turns on whether the dirty worktree can change what executed,
    # not on whether any file anywhere is dirty: another agent's untracked
    # notes cannot alter dispatch or arithmetic.
    clean = not provenance.get("execution_affecting_dirty", provenance.get("dirty"))
    measurement_valid = bool(quality_passed and deterministic and clean)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "profile_qualification_claim": quality_passed and deterministic,
        "qualification_blockers": [
            blocker
            for blocker, ok in (
                ("calibrated quality thresholds", quality_passed),
                ("same-schedule repeat determinism", deterministic),
                ("execution-affecting worktree provenance", clean),
            )
            if not ok
        ],
        "route": {
            "surface": f"post-binder {SELECTOR_ENV}",
            "candidate_layers": candidate_layers,
            "teacher_selector": "",
            "teacher_chain": "exact coltile8_rowbatch4_f32_f32_out",
            "candidate_chain": "wmma_prefill_f32_f32_out",
            "warmup_request_after_selector_flip": True,
            "selector_restored_after_capture": True,
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
            "teacher_scope": (
                "production profile with the Q8 dense prefill selector cleared; "
                "candidate-local isolation of the changed route"
            ),
            "thresholds_evaluated": EvaluationThresholds().to_dict(),
            "profile_manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
        },
        "quality": quality,
        "repeat_determinism": evaluated["repeat_determinism"],
        "strict_logits_sha256": evaluated["strict_logits_sha256"],
        "candidate_logits_sha256": evaluated["candidate_logits_sha256"],
        "variant_manifests": {"candidate_evidentiary": manifest},
        "prompts": prompt_manifest,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--layers",
        default="0-47",
        help="candidate layer set, e.g. 0-47 or 8,9,10 (default: every layer)",
    )
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--execution-profile", choices=("production",), default="production")
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=False) + "\n")
    quality = artifact["quality"]
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "mean_kl": quality.get("mean_kl"),
                "p95_kl": quality.get("p95_kl"),
                "max_kl": quality.get("max_kl"),
                "top1_rate": quality.get("top1_rate"),
                "repeat_deterministic": artifact["repeat_determinism"]["passed"],
                "blockers": artifact["qualification_blockers"],
                "output": str(args.output),
            },
            indent=1,
        )
    )
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
