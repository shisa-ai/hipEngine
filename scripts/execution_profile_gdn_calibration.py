#!/usr/bin/env python3
"""Capture full-logit GDN controls for execution-profile threshold calibration.

This is a calibration-only adapter.  It compares explicit strict GDN arithmetic
with independently labelled positive and negative reassociated routes, but it
does not invent profile manifests, exact control telemetry, task verdicts, or a
performance claim.  Historical labels are provenance only and never alter the
fresh evaluator result.
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

from hipengine.benchmark.execution_profiles import (
    EvaluationThresholds,
    RowDescriptor,
    compare_profile_logits,
)
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.gguf_gdn_semantic_gate import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    _configure_gate_environment,
    _load_suites,
    _run_teacher_forced_candidate,
)
from scripts.gguf_gdn_trajectory_gate import (
    CANDIDATE_MODES,
    SUPPORTED_MODES,
    _run_logits_trajectory,
)
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256


KIND = "hipengine_execution_profile_gdn_calibration_capture"
SCHEMA_VERSION = 1


class CalibrationError(RuntimeError):
    """Raised when a calibration capture cannot be evaluated honestly."""


@dataclass(frozen=True, slots=True)
class PromptCalibrationCapture:
    prompt_id: str
    category: str
    strict: tuple[Mapping[str, object], ...]
    candidate_runs: Mapping[str, tuple[tuple[Mapping[str, object], ...], ...]]

    def __post_init__(self) -> None:
        if not self.prompt_id or not self.category:
            raise ValueError("calibration prompt id/category must be non-empty")
        if not self.strict:
            raise ValueError("calibration prompt needs a strict trajectory")
        if not self.candidate_runs:
            raise ValueError("calibration prompt needs candidate trajectories")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    byte_view = value.view(np.uint8).reshape(-1)
    for offset in range(0, byte_view.size, 8 * 1024 * 1024):
        digest.update(memoryview(byte_view[offset : offset + 8 * 1024 * 1024]))
    return digest.hexdigest()


def _trajectory_arrays(
    trajectory: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, tuple[int, ...]]:
    if not trajectory:
        raise ValueError("calibration trajectory cannot be empty")
    logits = [np.asarray(step["logits"], dtype=np.float32) for step in trajectory]
    if any(row.ndim != 1 or row.size == 0 for row in logits):
        raise ValueError("calibration logits must be non-empty rank-1 rows")
    widths = {int(row.size) for row in logits}
    if len(widths) != 1:
        raise ValueError("calibration trajectory has inconsistent vocabulary widths")
    return np.stack(logits), tuple(int(step["token_id"]) for step in trajectory)


def _trajectory_sha256(trajectory: Sequence[Mapping[str, object]]) -> str:
    logits, token_ids = _trajectory_arrays(trajectory)
    digest = hashlib.sha256()
    digest.update(_matrix_sha256(logits).encode("ascii"))
    digest.update(np.asarray(token_ids, dtype="<i8").tobytes())
    return digest.hexdigest()


def build_candidate_quality(
    captures: Sequence[PromptCalibrationCapture],
    *,
    candidate_mode: str,
    scenario_id: str,
    thresholds: EvaluationThresholds | None = None,
) -> dict[str, Any]:
    """Run the profile numerical evaluator over one full-logit control."""

    prompt_captures = tuple(captures)
    if not prompt_captures:
        raise ValueError("candidate quality needs at least one prompt capture")
    rows: list[RowDescriptor] = []
    strict_chunks: list[np.ndarray] = []
    candidate_chunks: list[np.ndarray] = []
    repeat_hashes: list[list[str]] = []
    mismatches: list[dict[str, Any]] = []
    scenario_step = 0
    expected_repeat_runs: int | None = None
    for capture in prompt_captures:
        runs = capture.candidate_runs.get(candidate_mode)
        if runs is None:
            raise ValueError(
                f"candidate mode {candidate_mode!r} absent for prompt {capture.prompt_id!r}"
            )
        if len(runs) < 3:
            raise ValueError("calibration requires at least three candidate runs")
        if expected_repeat_runs is None:
            expected_repeat_runs = len(runs)
        elif expected_repeat_runs != len(runs):
            raise ValueError("candidate repeat count differs across prompts")
        strict_logits, strict_ids = _trajectory_arrays(capture.strict)
        first_logits, first_ids = _trajectory_arrays(runs[0])
        if strict_logits.shape != first_logits.shape:
            raise ValueError("strict and candidate trajectories are not aligned")
        strict_chunks.append(strict_logits)
        candidate_chunks.append(first_logits)
        prompt_hashes = [_trajectory_sha256(run) for run in runs]
        repeat_hashes.append(prompt_hashes)
        for repeat_index, repeat in enumerate(runs[1:], start=1):
            repeat_logits, repeat_ids = _trajectory_arrays(repeat)
            logits_exact = bool(np.array_equal(first_logits, repeat_logits))
            ids_exact = first_ids == repeat_ids
            if not logits_exact or not ids_exact:
                mismatches.append(
                    {
                        "prompt_id": capture.prompt_id,
                        "repeat_index": repeat_index,
                        "logits_exact": logits_exact,
                        "selected_token_ids_exact": ids_exact,
                    }
                )
        for teacher_step, teacher_token_id in enumerate(strict_ids):
            rows.append(
                RowDescriptor(
                    scenario_id=scenario_id,
                    scenario_step=scenario_step,
                    request_id=capture.prompt_id,
                    teacher_step=teacher_step,
                    category=capture.category,
                    shape="prefill_last" if teacher_step == 0 else "c1",
                    transition="prefill_to_c1" if teacher_step == 0 else "steady",
                    teacher_token_id=teacher_token_id,
                )
            )
            scenario_step += 1

    strict_matrix = np.concatenate(strict_chunks, axis=0)
    candidate_matrix = np.concatenate(candidate_chunks, axis=0)
    quality = compare_profile_logits(
        strict_matrix,
        candidate_matrix,
        rows,
        thresholds=thresholds,
    )
    return {
        "quality": quality,
        "repeat_determinism": {
            "runs": int(expected_repeat_runs or 0),
            "passed": not mismatches,
            "mismatches": mismatches,
            "trajectory_sha256_by_prompt": repeat_hashes,
        },
        "strict_logits_sha256": _matrix_sha256(strict_matrix),
        "candidate_logits_sha256": _matrix_sha256(candidate_matrix),
        "strict_selected_token_ids_sha256": hashlib.sha256(
            np.asarray(
                [row.teacher_token_id for row in rows], dtype="<i8"
            ).tobytes()
        ).hexdigest(),
    }


def parse_mode_sources(
    *,
    positive_modes: Sequence[str],
    negative_modes: Sequence[str],
    source_values: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Bind every declared control to one retained historical source artifact."""

    positives = tuple(str(mode) for mode in positive_modes)
    negatives = tuple(str(mode) for mode in negative_modes)
    declared = (*positives, *negatives)
    if not declared:
        raise ValueError("declare at least one positive or negative control")
    if len(set(declared)) != len(declared):
        raise ValueError("calibration control modes must be unique")
    labels = {mode: "positive" for mode in positives}
    labels.update({mode: "negative" for mode in negatives})
    paths: dict[str, Path] = {}
    for raw in source_values:
        mode, separator, value = str(raw).partition("=")
        if not separator or not mode or not value:
            raise ValueError("historical source must use MODE=PATH")
        if mode in paths:
            raise ValueError(f"duplicate historical source for mode {mode!r}")
        paths[mode] = Path(value)
    if set(paths) != set(declared):
        missing = sorted(set(declared) - set(paths))
        unknown = sorted(set(paths) - set(declared))
        raise ValueError(
            f"historical sources differ from controls; missing={missing}, unknown={unknown}"
        )
    result: dict[str, dict[str, Any]] = {}
    for mode in declared:
        path = paths[mode]
        if not path.is_file():
            raise ValueError(f"historical source does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"historical source is not valid JSON: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"historical source root is not an object: {path}")
        result[mode] = {
            "expected_label": labels[mode],
            "historical_artifact": str(path),
            "historical_artifact_sha256": _file_sha256(path),
            "historical_kind": payload.get("kind"),
            "historical_status": payload.get("status"),
        }
    return result


def _capture_controls(
    args: argparse.Namespace,
    *,
    prompt_rows: Sequence[Mapping[str, Any]],
    candidate_modes: Sequence[str],
) -> tuple[
    tuple[PromptCalibrationCapture, ...],
    list[dict[str, Any]],
    str,
    str,
]:
    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): build_chat_prompt(tokenizer, str(row["prompt"]))
        for row in prompt_rows
    }
    max_prompt_tokens = max(len(tokens) for tokens in prompt_tokens.values())
    max_sequence_length = max_prompt_tokens + int(args.decode_steps) + 2
    captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(
            attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)
        ),
    ) as session:
        if session.runner is None:
            raise CalibrationError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        for index, row in enumerate(prompt_rows):
            prompt_id = str(row["id"])
            tokens = prompt_tokens[prompt_id]
            strict = tuple(
                _run_logits_trajectory(
                    session,
                    prompt_ids=tokens,
                    mode=str(args.baseline_mode),
                    decode_steps=int(args.decode_steps),
                    bulk_attention_mode=str(args.bulk_attention_mode),
                )
            )
            forced = [int(step["token_id"]) for step in strict[:-1]]
            candidate_runs: dict[
                str, tuple[tuple[Mapping[str, object], ...], ...]
            ] = {}
            for mode in candidate_modes:
                runs = tuple(
                    tuple(
                        _run_teacher_forced_candidate(
                            session,
                            prompt_ids=tokens,
                            forced_input_ids=forced,
                            mode=mode,
                            bulk_attention_mode=str(args.bulk_attention_mode),
                        )
                    )
                    for _ in range(int(args.repeat_runs))
                )
                candidate_runs[mode] = runs
            captures.append(
                PromptCalibrationCapture(
                    prompt_id=prompt_id,
                    category=str(row["category"]),
                    strict=strict,
                    candidate_runs=candidate_runs,
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
                f"strict + {len(candidate_modes)} controls x {args.repeat_runs}",
                flush=True,
            )
    return tuple(captures), prompt_manifest, resolved_backend, target_arch


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise CalibrationError(f"model does not exist: {args.model}")
    if args.baseline_mode not in SUPPORTED_MODES:
        raise CalibrationError(f"unsupported baseline mode: {args.baseline_mode}")
    sources = parse_mode_sources(
        positive_modes=args.positive_mode,
        negative_modes=args.negative_mode,
        source_values=args.historical_source,
    )
    candidate_modes = tuple(sources)
    unknown = sorted(set(candidate_modes) - set(CANDIDATE_MODES))
    if unknown:
        raise CalibrationError(f"unsupported candidate modes: {unknown}")
    if args.baseline_mode in candidate_modes:
        raise CalibrationError("baseline mode cannot also be a candidate")
    if int(args.decode_steps) <= 0:
        raise CalibrationError("decode steps must be positive")
    if int(args.repeat_runs) < 3:
        raise CalibrationError("calibration requires at least three repeat runs")
    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise CalibrationError("selected prompt suites are empty")
    complete_suite = args.limit is None

    _configure_gate_environment(decode_repack=bool(args.decode_repack))
    captures, prompt_manifest, resolved_backend, target_arch = _capture_controls(
        args,
        prompt_rows=prompt_rows,
        candidate_modes=candidate_modes,
    )
    thresholds = EvaluationThresholds()
    controls: dict[str, Any] = {}
    for mode, source in sources.items():
        evaluated = build_candidate_quality(
            captures,
            candidate_mode=mode,
            scenario_id=f"gdn-{args.model.stem}-c1-teacher-forced",
            thresholds=thresholds,
        )
        hard_passed = bool(evaluated["quality"]["hard_gates_passed"])
        expected_pass = source["expected_label"] == "positive"
        controls[mode] = {
            **source,
            **evaluated,
            "fresh_hard_gate_passed": hard_passed,
            "historical_label_matches_fresh_hard_gate": hard_passed == expected_pass,
            "profile_qualified": False,
        }

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
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": (
                f"explicit {args.baseline_mode} versus {','.join(candidate_modes)}"
            ),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get(
                "HIPENGINE_GGUF_DECODE_REPACK"
            ),
            "HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE": os.environ.get(
                "HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE"
            ),
        },
        build_profile="execution_profile_gdn_calibration_capture",
        timing_protocol="none_calibration_only_full_logits_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    deterministic = all(
        bool(control["repeat_determinism"]["passed"])
        for control in controls.values()
    )
    measurement_valid = bool(not provenance.get("dirty") and deterministic and complete_suite)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "profile_qualification_claim": False,
        "qualification_blockers": [
            "calibration adapter does not emit resolved runtime profile manifests",
            "calibration adapter does not emit exact request/control ownership telemetry",
            "historical task verdicts are provenance only and are not fresh requalification",
        ],
        "protocol": {
            "model": str(args.model.resolve()),
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "baseline_mode": str(args.baseline_mode),
            "candidate_modes": list(candidate_modes),
            "decode_steps": int(args.decode_steps),
            "teacher_forced_rows": sum(len(capture.strict) for capture in captures),
            "candidate_repeat_runs": int(args.repeat_runs),
            "same_context_rule": (
                "each candidate consumes the strict generated token prefix at every "
                "compared transition"
            ),
            "thresholds_evaluated": thresholds.to_dict(),
            "labels_do_not_affect_metrics_or_verdicts": True,
        },
        "prompts": prompt_manifest,
        "controls": controls,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--baseline-mode", choices=SUPPORTED_MODES, default="chain_lds32_direct"
    )
    parser.add_argument(
        "--positive-mode", action="append", default=[], metavar="MODE"
    )
    parser.add_argument(
        "--negative-mode", action="append", default=[], metavar="MODE"
    )
    parser.add_argument(
        "--historical-source",
        action="append",
        default=[],
        metavar="MODE=PATH",
    )
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--bulk-attention-mode", choices=("bulk", "native"), default="bulk"
    )
    parser.add_argument(
        "--decode-repack", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True
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
                "controls": {
                    mode: {
                        "expected_label": control["expected_label"],
                        "hard_gate_passed": control["fresh_hard_gate_passed"],
                        "repeat_deterministic": control["repeat_determinism"]["passed"],
                        "summary": control["quality"]["summary"],
                    }
                    for mode, control in artifact["controls"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
