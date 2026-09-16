#!/usr/bin/env python3
"""Whole-model gate for a layer-scoped f16 Q8_0 dense prefill route.

Evaluates one of the layer-scoped f16 WMMA-class dense Q8_0 prefill selectors
against the exact coltile chain on the Qwen4Exp UD-Q4_K_XL canonical
exact-token fixture: full-vocabulary logits trajectories with the candidate
consuming the strict generated prefix at every compared transition, evaluated
with the calibrated mean/tail/max KL and top-1 thresholds, plus same-schedule
repeat determinism. ``--route`` picks which route is measured; see
:data:`SELECTORS`.

The teacher is the named production profile with the route's selectors cleared,
so the measured drift is that route's own contribution and not the rest of the
production stack. That is the same candidate-local isolation the Q8 MMQ plane
gate uses.

Every route here is a post-binder selector: the named production binder writes
its env keys during its own pass, so setting them before construction is
silently inert. This gate constructs the generator first and flips the
selectors per arm.

Phases per canonical case:
- teacher: selectors cleared -> exact ``coltile8_rowbatch4`` chain
- candidate x N: selectors set to ``--layers`` -> the route's candidate chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
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

KIND = "hipengine_execution_profile_q8_wmma_prefill_layers_gate"
SCHEMA_VERSION = 1
WMMA_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"
DENSE_WIDE_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"
DENSE_WIDE_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"
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


@dataclass(frozen=True)
class Route:
    """One layer-scoped Q8_0 dense prefill route this gate can measure.

    Both routes are post-binder selectors: the named production binder writes
    their env keys during its own pass, so a value set before construction is
    inert and only a post-binder flip is effective.

    ``dense_wide`` clears the WMMA selector in **both** arms. The production
    default binds WMMA to the same layer window this route is gated at, so
    leaving it bound would make the comparison carry the WMMA route's
    arithmetic as well as the candidate's, and the two routes would contend for
    the window under test.
    """

    kind: str
    scenario_prefix: str
    candidate_chain: str
    candidate_env: Mapping[str, str]
    teacher_env: Mapping[str, str]
    surface: str

    def env_for(self, *, candidate: bool, layers: str) -> dict[str, str]:
        bindings = self.candidate_env if candidate else self.teacher_env
        return {
            key: value.format(layers=layers) for key, value in bindings.items()
        }

    @property
    def env_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.candidate_env, *self.teacher_env)))


# ``--route`` -> the route to gate. The teacher arm is the exact coltile chain
# in every case, so the measured drift is the route's own contribution and the
# figures are comparable across routes.
SELECTORS: dict[str, Route] = {
    "wmma": Route(
        kind=KIND,
        scenario_prefix="qwen4exp-ud-q4-k-xl-q8-wmma-dense-prefill-layers",
        candidate_chain="wmma_prefill_f32_f32_out",
        candidate_env={WMMA_LAYERS_ENV: "{layers}"},
        teacher_env={WMMA_LAYERS_ENV: ""},
        surface=f"post-binder {WMMA_LAYERS_ENV}",
    ),
    "dense_wide": Route(
        kind="hipengine_execution_profile_q8_dense_wide_prefill_layers_gate",
        scenario_prefix=(
            "qwen4exp-ud-q4-k-xl-q8-dense-wide256-dense-prefill-layers"
        ),
        candidate_chain="dense_wide256_f32_f32_out",
        candidate_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "1",
            DENSE_WIDE_LAYERS_ENV: "{layers}",
        },
        teacher_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
        },
        surface=(
            f"post-binder {DENSE_WIDE_ENV} + {DENSE_WIDE_LAYERS_ENV}; "
            f"{WMMA_LAYERS_ENV} cleared in both arms"
        ),
    ),
}


def _select_route(name: str) -> Route:
    try:
        return SELECTORS[str(name)]
    except KeyError as exc:
        raise CalibrationError(f"unknown Q8 prefill route {name!r}") from exc


def _apply_route_env(route: Route, *, candidate: bool, layers: str) -> None:
    """Bind every selector the route owns. Empty layers means the exact chain."""

    os.environ.update(route.env_for(candidate=candidate, layers=layers))


def _restore_env(bound: Mapping[str, str | None]) -> None:
    for key, value in bound.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        last_prebinder_conflicts,
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
    route = _select_route(str(args.route))

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
    bound_route_env = {key: os.environ.get(key) for key in route.env_keys}
    # A caller-set value the binder discarded means the intended route never
    # ran and the gate would report a clean null. Fail rather than measure it.
    discarded = last_prebinder_conflicts()
    if discarded:
        raise CalibrationError(
            "the profile binder discarded pre-binder route values, so the "
            f"measured route is not the requested one: {sorted(discarded)}"
        )

    captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    command = ["python3", str(Path(__file__).resolve().relative_to(REPO_ROOT)), *sys.argv[1:]]
    try:
        for position, row in enumerate(cases):
            prompt_id = str(row["id"])
            tokens = [int(t) for t in row["prompt_token_ids"]]
            _apply_route_env(route, candidate=False, layers="")
            _warm(runner, tokens)
            teacher = _strict_trajectory(runner, tokens, decode_steps)
            forced = [step["token_id"] for step in teacher[:-1]]
            _apply_route_env(route, candidate=True, layers=candidate_layers)
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
        _restore_env(bound_route_env)

    evaluated = build_candidate_quality(
        captures,
        candidate_mode="candidate",
        scenario_id=(
            f"{route.scenario_prefix}-"
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
                selected_variant=route.candidate_chain,
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
            **route.env_for(candidate=True, layers=candidate_layers),
        },
        build_profile=route.kind,
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
        "kind": route.kind,
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
            "name": str(args.route),
            "surface": route.surface,
            "candidate_layers": candidate_layers,
            "teacher_selector": "cleared",
            "teacher_chain": "exact coltile8_rowbatch4_f32_f32_out",
            "candidate_chain": route.candidate_chain,
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
                "production profile with the route's selectors cleared; "
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
        "--route",
        choices=tuple(SELECTORS),
        default="wmma",
        help=(
            "which layer-scoped Q8_0 dense prefill route to gate "
            "(default: wmma)"
        ),
    )
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
    # The evaluator nests its figures under "summary"; reading them from the
    # top level printed nulls for every metric.
    summary = quality.get("summary", {})
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "hard_gates_passed": quality.get("hard_gates_passed"),
                "mean_kl": summary.get("kl_mean"),
                "p95_kl": summary.get("kl_p95"),
                "max_kl": summary.get("kl_max"),
                "top1_rate": summary.get("top1_agreement"),
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
