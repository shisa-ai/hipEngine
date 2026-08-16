#!/usr/bin/env python3
"""Requalify one shape-scoped GGUF dense pair route against strict arithmetic.

The adapter runs full-vocabulary strict-teacher trajectories while logits are
resident. It changes only the backend package's declared dense-pair shape owner,
verifies both registry variants, and restores the package policy on exit. The
result is numerical/repeatability evidence; it deliberately does not invent
runtime control telemetry, task verdicts, BF16 evidence, or a performance claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.execution_profiles import VariantSelection, build_variant_manifest, manifest_sha256
from hipengine.kernels.registry import KernelKey, is_registered
from scripts.execution_profile_gdn_calibration import (
    CalibrationError,
    PromptCalibrationCapture,
    build_candidate_quality,
    validate_strict_baseline,
)
from scripts.gguf_gdn_semantic_gate import (
    DEFAULT_PROMPTS,
    _configure_gate_environment,
    _load_suites,
    _run_teacher_forced_candidate,
)
from scripts.gguf_gdn_trajectory_gate import _run_logits_trajectory
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256

KIND = "hipengine_execution_profile_gguf_dense_pair_requalification_capture"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_HISTORICAL_SOURCE = Path(
    "benchmarks/results/2026-08-15-gfx1151-qwen38-27b-q4-q8x2-dp4a.json"
)
POLICY_CAPABILITY = "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES"
REGISTRY_LAYER = "linear_pair_silu"
REGISTRY_QUANT = "gguf_q4_k_t16_v1"
DEFAULT_STRICT_VARIANT = "dense_dual_local32_bf16_bf16_out"
DEFAULT_CANDIDATE_VARIANT = (
    "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out"
)
DEFAULT_GDN_MODE = "chain_lds32_direct_nonvolatile"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense_pair_policy_override(
    policy: Mapping[object, Mapping[tuple[int, int, int], str]],
    *,
    identity: object,
    shape: tuple[int, int, int],
    variant: str,
) -> dict[object, dict[tuple[int, int, int], str]]:
    """Return a copied package policy with exactly one shape owner replaced."""

    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("dense pair variant must be non-empty")
    result = copy.deepcopy(dict(policy))
    shapes = dict(result.get(identity, {}))
    shapes[tuple(int(value) for value in shape)] = variant.strip()
    result[identity] = shapes
    return result


def validate_route_variants(strict_variant: str, candidate_variant: str) -> None:
    values = (strict_variant, candidate_variant)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("strict and candidate variants must be non-empty")
    if strict_variant == candidate_variant:
        raise ValueError("strict and candidate variants must differ")


def _verify_registry_variants(
    *, backend: str, strict_variant: str, candidate_variant: str
) -> None:
    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package(backend)
    missing = [
        key
        for key in (
            KernelKey(backend, REGISTRY_LAYER, REGISTRY_QUANT, strict_variant),
            KernelKey(backend, REGISTRY_LAYER, REGISTRY_QUANT, candidate_variant),
        )
        if not is_registered(key)
    ]
    if missing:
        raise CalibrationError(
            "dense pair route variants are not registered: "
            + ", ".join(key.display() for key in missing)
        )


def _variant_manifests(
    *,
    backend: str,
    model_identity: str,
    model_quant: str,
    scope: str,
    strict_variant: str,
    candidate_variant: str,
    evidence_artifact: str,
) -> dict[str, Any]:
    strict = build_variant_manifest(
        profile="strict",
        backend=backend,
        model=model_identity,
        quant=model_quant,
        kv_policy="paged_bf16",
        graph_policy="serial_c1",
        selections=(
            VariantSelection(
                layer=REGISTRY_LAYER,
                scope=scope,
                selected_variant=strict_variant,
                strict_fallback_variant=strict_variant,
                registry_quant=REGISTRY_QUANT,
            ),
        ),
    )
    production = build_variant_manifest(
        profile="production",
        backend=backend,
        model=model_identity,
        quant=model_quant,
        kv_policy="paged_bf16",
        graph_policy="serial_c1",
        selections=(
            VariantSelection(
                layer=REGISTRY_LAYER,
                scope=scope,
                selected_variant=candidate_variant,
                strict_fallback_variant=strict_variant,
                evidence_artifact=evidence_artifact,
                registry_quant=REGISTRY_QUANT,
            ),
        ),
    )
    return {
        "runtime_resolved": False,
        "strict": strict,
        "strict_sha256": manifest_sha256(strict),
        "candidate": production,
        "candidate_sha256": manifest_sha256(production),
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    validate_route_variants(args.strict_variant, args.candidate_variant)
    if not args.model.is_file():
        raise CalibrationError(f"model does not exist: {args.model}")
    if not args.historical_source.is_file():
        raise CalibrationError(
            f"historical source does not exist: {args.historical_source}"
        )
    if int(args.decode_steps) <= 0:
        raise CalibrationError("decode steps must be positive")
    if int(args.repeat_runs) < 3:
        raise CalibrationError("route requalification requires at least three repeats")
    if int(args.rows) != 1:
        raise CalibrationError("this adapter is restricted to a rows=1 route")
    shape = (int(args.rows), int(args.in_features), int(args.out_features))
    scope = (
        f"rows{shape[0]}_h{shape[1]}_ffn{shape[2]}"
    )
    historical = json.loads(args.historical_source.read_text(encoding="utf-8"))
    if not isinstance(historical, Mapping):
        raise CalibrationError("historical source root must be an object")

    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise CalibrationError("selected prompt suites are empty")
    complete_suite = args.limit is None
    _configure_gate_environment(decode_repack=bool(args.decode_repack))

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _gguf_gdn_prefill_backend_exact_mode,
        _gguf_policy_identity,
    )
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): build_chat_prompt(tokenizer, str(row["prompt"]))
        for row in prompt_rows
    }
    max_sequence_length = (
        max(len(tokens) for tokens in prompt_tokens.values())
        + int(args.decode_steps)
        + 2
    )
    _verify_registry_variants(
        backend=str(args.backend),
        strict_variant=str(args.strict_variant),
        candidate_variant=str(args.candidate_variant),
    )
    package = __import__(
        f"hipengine.kernels.{args.backend}", fromlist=[POLICY_CAPABILITY]
    )
    original_policy = copy.deepcopy(getattr(package, POLICY_CAPABILITY))
    captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    resolved_backend = str(args.backend)
    target_arch = str(args.backend).removeprefix("hip_")
    identity_repr = "unresolved"
    try:
        with Qwen35GGUFResidentSession(
            args.model,
            backend=str(args.backend),
            compiler_version=compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=max_sequence_length,
            use_wmma_prefill=True,
            use_gemv_decode=True,
            prefill_config=PrefillConfig(
                attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)
            ),
        ) as session:
            if session.runner is None:
                raise CalibrationError("GGUF resident session closed during setup")
            resolved_backend = str(session.runner.backend)
            target_arch = str(session.runner.target_arch)
            validate_strict_baseline(
                requested_mode=str(args.gdn_mode),
                backend_exact_mode=_gguf_gdn_prefill_backend_exact_mode(
                    resolved_backend
                ),
            )
            identity = _gguf_policy_identity(session.runner.weights)
            if identity is None:
                raise CalibrationError("model does not expose a dense GGUF policy identity")
            identity_repr = repr(identity)
            current_variant = original_policy.get(identity, {}).get(shape)
            if current_variant != args.candidate_variant:
                raise CalibrationError(
                    "candidate is not the current package owner for the declared "
                    f"identity/shape: package={current_variant!r}, "
                    f"candidate={args.candidate_variant!r}"
                )
            for index, row in enumerate(prompt_rows):
                prompt_id = str(row["id"])
                tokens = prompt_tokens[prompt_id]
                setattr(
                    package,
                    POLICY_CAPABILITY,
                    dense_pair_policy_override(
                        original_policy,
                        identity=identity,
                        shape=shape,
                        variant=str(args.strict_variant),
                    ),
                )
                strict = tuple(
                    _run_logits_trajectory(
                        session,
                        prompt_ids=tokens,
                        mode=str(args.gdn_mode),
                        decode_steps=int(args.decode_steps),
                        bulk_attention_mode="bulk",
                    )
                )
                forced = [int(step["token_id"]) for step in strict[:-1]]
                setattr(
                    package,
                    POLICY_CAPABILITY,
                    dense_pair_policy_override(
                        original_policy,
                        identity=identity,
                        shape=shape,
                        variant=str(args.candidate_variant),
                    ),
                )
                runs = tuple(
                    tuple(
                        _run_teacher_forced_candidate(
                            session,
                            prompt_ids=tokens,
                            forced_input_ids=forced,
                            mode=str(args.gdn_mode),
                            bulk_attention_mode="bulk",
                        )
                    )
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
                prompt_manifest.append(
                    {
                        "id": prompt_id,
                        "category": str(row["category"]),
                        "suite": str(row["suite"]),
                        "prompt_sha256": prompt_sha256(str(row["prompt"])),
                        "prompt_tokens": len(tokens),
                        "prompt_token_ids_sha256": hashlib.sha256(
                            np.asarray(tokens, dtype="<i8").tobytes()
                        ).hexdigest(),
                    }
                )
                print(
                    f"{index + 1}/{len(prompt_rows)} {prompt_id}: captured "
                    f"strict + candidate x {args.repeat_runs}",
                    flush=True,
                )
    finally:
        setattr(package, POLICY_CAPABILITY, original_policy)

    evaluated = build_candidate_quality(
        captures,
        candidate_mode="candidate",
        scenario_id=f"{args.model.stem}-dense-pair-c1-teacher-forced",
        thresholds=EvaluationThresholds(),
    )
    manifests = _variant_manifests(
        backend=resolved_backend,
        model_identity=str(args.model_identity),
        model_quant="gguf_q4_k_m",
        scope=scope,
        strict_variant=str(args.strict_variant),
        candidate_variant=str(args.candidate_variant),
        evidence_artifact=str(args.historical_source),
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": f"explicit {args.gdn_mode}",
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get(
                "HIPENGINE_GGUF_DECODE_REPACK"
            ),
        },
        build_profile="execution_profile_gguf_dense_pair_requalification",
        timing_protocol="none_full_logits_only_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    deterministic = bool(evaluated["repeat_determinism"]["passed"])
    measurement_valid = bool(
        complete_suite and deterministic and not provenance.get("dirty")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "profile_qualification_claim": False,
        "qualification_blockers": [
            "route manifest is evidentiary and is not resolved by a runtime profile plan",
            "adapter does not emit exact request/control ownership telemetry",
            "fresh task verdicts and BF16-relative logits are not available",
        ],
        "route": {
            "policy_capability": POLICY_CAPABILITY,
            "policy_identity": identity_repr,
            "shape": list(shape),
            "scope": scope,
            "registry_layer": REGISTRY_LAYER,
            "registry_quant": REGISTRY_QUANT,
            "strict_variant": str(args.strict_variant),
            "candidate_variant": str(args.candidate_variant),
            "current_package_owner_verified": True,
            "policy_restored_after_capture": True,
        },
        "protocol": {
            "model": str(args.model.resolve()),
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "decode_steps": int(args.decode_steps),
            "teacher_forced_rows": sum(len(capture.strict) for capture in captures),
            "candidate_repeat_runs": int(args.repeat_runs),
            "strict_gdn_mode": str(args.gdn_mode),
            "backend_strict_gdn_mode_verified": True,
            "same_context_rule": (
                "candidate consumes the strict generated token prefix at every "
                "compared transition"
            ),
            "thresholds_evaluated": EvaluationThresholds().to_dict(),
        },
        "quality": evaluated["quality"],
        "repeat_determinism": evaluated["repeat_determinism"],
        "strict_logits_sha256": evaluated["strict_logits_sha256"],
        "candidate_logits_sha256": evaluated["candidate_logits_sha256"],
        "strict_selected_token_ids_sha256": evaluated[
            "strict_selected_token_ids_sha256"
        ],
        "variant_manifests": manifests,
        "prompts": prompt_manifest,
        "historical_source": {
            "path": str(args.historical_source),
            "sha256": _file_sha256(args.historical_source),
            "kind": historical.get("kind"),
            "status": historical.get("status"),
            "label_does_not_affect_fresh_verdict": True,
        },
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-identity", default="qwen3_8_27b_gguf_dense_h5120")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--in-features", type=int, default=5_120)
    parser.add_argument("--out-features", type=int, default=17_408)
    parser.add_argument("--strict-variant", default=DEFAULT_STRICT_VARIANT)
    parser.add_argument("--candidate-variant", default=DEFAULT_CANDIDATE_VARIANT)
    parser.add_argument("--gdn-mode", default=DEFAULT_GDN_MODE)
    parser.add_argument("--historical-source", type=Path, default=DEFAULT_HISTORICAL_SOURCE)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--decode-repack", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "hard_gate_passed": artifact["quality"]["hard_gates_passed"],
                "requires_outlier_review": artifact["quality"][
                    "requires_outlier_review"
                ],
                "summary": artifact["quality"]["summary"],
                "top1_mismatch_rows": artifact["quality"]["top1_mismatch_rows"],
                "repeat_deterministic": artifact["repeat_determinism"]["passed"],
            },
            indent=2,
        )
    )
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
