#!/usr/bin/env python3
"""P9 qwen35moe GGUF E2E correctness gate.

This is the formal P9 correctness contract for GGUF resident fast-path opt-ins.
The current fixture compares a 512-token qwen35moe generation from the resident
T16 decode-repack + GEMV path against the legacy row-GEMV reference path:

* reference: ``HIPENGINE_GGUF_WMMA_PREFILL=0`` and
  ``HIPENGINE_GGUF_GEMV_DECODE=0``
* candidate: ``HIPENGINE_GGUF_DECODE_REPACK=1`` and
  ``HIPENGINE_GGUF_GEMV_DECODE=1``; ``HIPENGINE_GGUF_WMMA_PREFILL=1`` remains
  in the fixture to prove the qwen35moe safety gate keeps unsafe WMMA disabled
  unless explicitly overridden.

The script runs the resident GGUF session directly so it can collect full logits
for the prefill sample plus every decode step.  It fails loudly when KL/top-1
drift exceeds the fixture thresholds, when final logits are not finite, or when
the candidate token tail is not deterministic across repeats.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/gguf/qwen36_35b_a3b_q4km_p9_e2e.json"
_WMMA_ENV = "HIPENGINE_GGUF_WMMA_PREFILL"
_GEMV_ENV = "HIPENGINE_GGUF_GEMV_DECODE"


@dataclass
class SequenceRun:
    generated_token_ids: list[int]
    logits: np.ndarray
    final_token_id: int
    final_logit: float
    finite_logits: bool
    memory: dict[str, Any]
    fastpath_safety: dict[str, Any] | None = None

    @property
    def tail_token_ids(self) -> list[int]:
        return self.generated_token_ids[-16:]


def load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text())
    required = {"schema_version", "model", "prompt", "generation", "reference", "candidate", "acceptance"}
    missing = sorted(required.difference(fixture))
    if missing:
        raise ValueError(f"fixture {path} missing required keys: {', '.join(missing)}")
    model = fixture["model"]
    for key in ("path", "quant", "architecture"):
        if key not in model:
            raise ValueError(f"fixture {path} model missing {key!r}")
    generation = fixture["generation"]
    for key in ("prompt_length", "decode_tokens", "repeats"):
        if key not in generation:
            raise ValueError(f"fixture {path} generation missing {key!r}")
    acceptance = fixture["acceptance"]
    for key in ("kl_threshold", "top1_threshold", "finite_final_logits_required", "deterministic_tail_required"):
        if key not in acceptance:
            raise ValueError(f"fixture {path} acceptance missing {key!r}")
    return fixture


def prompt_tokens_from_fixture(fixture: Mapping[str, Any]) -> list[int]:
    prompt = fixture["prompt"]
    if "prompt_ids" in prompt:
        tokens = [int(item) for item in prompt["prompt_ids"]]
    elif prompt.get("source") == "repeated_token_id":
        tokens = [int(prompt["token_id"])] * int(fixture["generation"]["prompt_length"])
    else:
        raise ValueError("fixture prompt must provide prompt_ids or source='repeated_token_id'")
    expected_len = int(fixture["generation"]["prompt_length"])
    if len(tokens) != expected_len:
        raise ValueError(f"fixture prompt has {len(tokens)} tokens, expected {expected_len}")
    return tokens


def expected_env_from_mode(mode: Mapping[str, Any]) -> dict[str, str]:
    env = {str(k): str(v) for k, v in dict(mode.get("env", {})).items()}
    if _WMMA_ENV not in env or _GEMV_ENV not in env:
        raise ValueError(f"mode {mode.get('name', '<unnamed>')!r} must set {_WMMA_ENV} and {_GEMV_ENV}")
    return env


@contextmanager
def patched_env(values: Mapping[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_sequence(
    *,
    model: Path,
    quant: str,
    prompt_tokens: list[int],
    decode_tokens: int,
    mode: Mapping[str, Any],
    compiler_version: str | None,
    require_cached_build: bool,
) -> SequenceRun:
    """Run prefill + ``decode_tokens`` eager decode steps and collect logits."""

    env = expected_env_from_mode(mode)
    reset_memory_stats()
    generated: list[int] = []
    logits_rows: list[np.ndarray] = []
    final_token_id: int | None = None
    final_logit = 0.0
    finite_logits = True
    max_sequence_length = len(prompt_tokens) + int(decode_tokens) + 2
    use_bulk_prefill = bool(mode.get("use_bulk_prefill", True))
    bulk_attention_mode = str(mode.get("bulk_prefill_attention_mode", "bulk"))
    with patched_env(env):
        with Qwen35GGUFResidentSession(
            model,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            max_sequence_length=max_sequence_length,
            # Keep these None so the fixture's env contract is what selects the path.
            use_wmma_prefill=None,
            use_gemv_decode=None,
        ) as session:
            first = session.prefill(
                prompt_tokens,
                use_bulk=use_bulk_prefill,
                bulk_attention_mode=bulk_attention_mode,
                return_logits=True,
            )
            generated.append(int(first.token_id))
            logits_rows.append(_checked_logits(first.logits, label="prefill"))
            current = first
            for step_index in range(int(decode_tokens)):
                current = session.step(int(current.token_id), return_logits=True)
                generated.append(int(current.token_id))
                logits_rows.append(_checked_logits(current.logits, label=f"decode[{step_index}]"))
            final_token_id = int(current.token_id)
            final_logit = float(current.logit)
            finite_logits = bool(np.all(np.isfinite(logits_rows[-1])))
            fastpath_safety = session.fastpath_safety.as_dict() if session.fastpath_safety is not None else None
    logits = np.vstack(logits_rows).astype(np.float32, copy=False)
    return SequenceRun(
        generated_token_ids=generated,
        logits=logits,
        final_token_id=int(final_token_id),
        final_logit=final_logit,
        finite_logits=finite_logits,
        memory=memory_stats(),
        fastpath_safety=fastpath_safety,
    )


def _checked_logits(logits: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise FloatingPointError(f"{label} logits are empty")
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{label} logits contain NaN or Inf")
    return arr.copy()


def evaluate_candidate_repeats(
    *,
    reference: SequenceRun,
    candidates: list[SequenceRun],
    kl_threshold: float,
    top1_threshold: float,
    deterministic_tail_required: bool,
    finite_final_logits_required: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    per_repeat: list[dict[str, Any]] = []
    reference_tail = reference.tail_token_ids
    reference_generated = reference.generated_token_ids
    candidate_tails = [candidate.tail_token_ids for candidate in candidates]
    deterministic_tail = bool(candidate_tails) and all(tail == candidate_tails[0] for tail in candidate_tails)
    deterministic_full = bool(candidates) and all(
        candidate.generated_token_ids == candidates[0].generated_token_ids for candidate in candidates
    )
    if deterministic_tail_required and not deterministic_tail:
        errors.append(f"candidate generated tails are not deterministic: {candidate_tails}")
    if finite_final_logits_required and not reference.finite_logits:
        errors.append("reference final logits are not finite")

    for index, candidate in enumerate(candidates):
        if finite_final_logits_required and not candidate.finite_logits:
            errors.append(f"candidate repeat {index} final logits are not finite")
        try:
            metrics = evaluate_logits(
                reference.logits,
                candidate.logits,
                kl_threshold=float(kl_threshold),
                top1_threshold=float(top1_threshold),
            )
        except Exception as exc:
            errors.append(f"candidate repeat {index} metric failure: {type(exc).__name__}: {exc}")
            per_repeat.append({"repeat_index": index, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        tokens_match_reference = candidate.generated_token_ids == reference_generated
        row_summary = _row_drift_summary(reference.logits, candidate.logits)
        if not metrics.passed:
            errors.append(
                f"candidate repeat {index} drift exceeds threshold: "
                f"kl_mean={metrics.kl_mean:.6g} <= {kl_threshold}, "
                f"top1={metrics.top1_agreement:.6g} >= {top1_threshold}, "
                f"first_top1_mismatch={row_summary['first_top1_mismatch']}"
            )
        per_repeat.append(
            {
                "repeat_index": int(index),
                "kl_mean": metrics.kl_mean,
                "kl_max": metrics.kl_max,
                "top1_agreement": metrics.top1_agreement,
                "passed": metrics.passed,
                "tokens_match_reference": tokens_match_reference,
                "generated_token_count": len(candidate.generated_token_ids),
                "generated_tail_token_ids": candidate.tail_token_ids,
                "final_token_id": candidate.final_token_id,
                "final_logit": candidate.final_logit,
                "finite_final_logits": candidate.finite_logits,
                "row_drift_summary": row_summary,
            }
        )
    passed = not errors and bool(candidates)
    return {
        "passed": passed,
        "errors": errors,
        "kl_threshold": float(kl_threshold),
        "top1_threshold": float(top1_threshold),
        "reference_generated_token_count": len(reference.generated_token_ids),
        "reference_tail_token_ids": reference_tail,
        "reference_final_token_id": reference.final_token_id,
        "reference_final_logit": reference.final_logit,
        "reference_finite_final_logits": reference.finite_logits,
        "candidate_repeat_count": len(candidates),
        "candidate_tails": candidate_tails,
        "candidate_deterministic_tail": deterministic_tail,
        "candidate_deterministic_full_sequence": deterministic_full,
        "per_repeat": per_repeat,
        "worst_kl_mean": max((float(row.get("kl_mean", float("inf"))) for row in per_repeat), default=None),
        "worst_kl_max": max((float(row.get("kl_max", float("inf"))) for row in per_repeat), default=None),
        "worst_top1_agreement": min((float(row.get("top1_agreement", 0.0)) for row in per_repeat), default=None),
    }


def _row_drift_summary(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference_logits, dtype=np.float32)
    candidate = np.asarray(candidate_logits, dtype=np.float32)
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)]}
    ref_argmax = np.argmax(reference, axis=-1).reshape(-1)
    cand_argmax = np.argmax(candidate, axis=-1).reshape(-1)
    matches = ref_argmax == cand_argmax
    mismatch_indices = np.flatnonzero(~matches)
    first_mismatch = None
    if mismatch_indices.size:
        row = int(mismatch_indices[0])
        first_mismatch = {
            "row": row,
            "reference_argmax": int(ref_argmax[row]),
            "candidate_argmax": int(cand_argmax[row]),
        }
    return {
        "rows": int(ref_argmax.size),
        "top1_match_count": int(np.count_nonzero(matches)),
        "top1_mismatch_count": int(mismatch_indices.size),
        "first_top1_mismatch": first_mismatch,
        "max_abs_logit_delta": float(np.max(np.abs(reference - candidate))) if reference.size else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = load_fixture(args.fixture)
    model = Path(args.model or fixture["model"]["path"])
    quant = str(args.quant or fixture["model"]["quant"])
    generation = fixture["generation"]
    acceptance = fixture["acceptance"]
    prompt_tokens = prompt_tokens_from_fixture(fixture)
    decode_tokens = int(args.decode_tokens if args.decode_tokens is not None else generation["decode_tokens"])
    repeats = int(args.repeats if args.repeats is not None else generation["repeats"])
    if decode_tokens <= 0:
        raise ValueError("decode_tokens must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None

    reference = run_sequence(
        model=model,
        quant=quant,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        mode=fixture["reference"],
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
    )
    candidates: list[SequenceRun] = []
    try:
        for _ in range(repeats):
            candidates.append(
                run_sequence(
                    model=model,
                    quant=quant,
                    prompt_tokens=prompt_tokens,
                    decode_tokens=decode_tokens,
                    mode=fixture["candidate"],
                    compiler_version=compiler_version,
                    require_cached_build=bool(args.require_cached_build),
                )
            )
        correctness = evaluate_candidate_repeats(
            reference=reference,
            candidates=candidates,
            kl_threshold=float(acceptance["kl_threshold"]),
            top1_threshold=float(acceptance["top1_threshold"]),
            deterministic_tail_required=bool(acceptance["deterministic_tail_required"]),
            finite_final_logits_required=bool(acceptance["finite_final_logits_required"]),
        )
    finally:
        # Keep peak host memory bounded when this script is invoked from benchmark gates.
        gc.collect()

    passed = bool(correctness["passed"])
    return {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "performance_claim": False,
        "mode": "qwen35moe_p9_t16_decode_repack_plus_gemv_decode_e2e_correctness",
        "fixture": str(args.fixture),
        "arguments": {
            "model_override": str(args.model) if args.model else "",
            "quant_override": str(args.quant) if args.quant else "",
            "decode_tokens_override": args.decode_tokens,
            "repeats_override": args.repeats,
            "compiler_version_file": None if args.compiler_version_file is None else str(args.compiler_version_file),
            "require_cached_build": bool(args.require_cached_build),
            "json": None if args.json is None else str(args.json),
        },
        "model": str(model),
        "quant": quant,
        "prompt": {
            "source": fixture["prompt"].get("source", "prompt_ids"),
            "token_id": fixture["prompt"].get("token_id"),
            "prompt_length": len(prompt_tokens),
        },
        "generation": {
            "decode_tokens": decode_tokens,
            "candidate_repeats": repeats,
            "total_logits_rows_per_run": decode_tokens + 1,
            "total_generated_tokens_per_run": decode_tokens + 1,
            "note": "Rows include the prefill sample plus every eager decode step; this mirrors 512/128 resident generation while collecting per-step logits.",
        },
        "reference": {
            "name": fixture["reference"].get("name"),
            "env": expected_env_from_mode(fixture["reference"]),
            "use_bulk_prefill": bool(fixture["reference"].get("use_bulk_prefill", True)),
            "bulk_prefill_attention_mode": str(fixture["reference"].get("bulk_prefill_attention_mode", "bulk")),
            "fastpath_safety": reference.fastpath_safety,
            "generated_tail_token_ids": reference.tail_token_ids,
            "final_token_id": reference.final_token_id,
            "final_logit": reference.final_logit,
            "finite_final_logits": reference.finite_logits,
            "memory": reference.memory,
        },
        "candidate": {
            "name": fixture["candidate"].get("name"),
            "env": expected_env_from_mode(fixture["candidate"]),
            "use_bulk_prefill": bool(fixture["candidate"].get("use_bulk_prefill", True)),
            "bulk_prefill_attention_mode": str(fixture["candidate"].get("bulk_prefill_attention_mode", "bulk")),
            "fastpath_safety_repeats": [candidate.fastpath_safety for candidate in candidates],
        },
        "correctness": correctness,
        "passed": passed,
        "notes": [
            "This gate intentionally uses the resident runner rather than public LLM.generate so it can compare full logits for every generated step.",
            "Reference is the legacy row-GEMV path with WMMA prefill and P9 GEMV decode disabled by env.",
            "Candidate is launched with HIPENGINE_GGUF_DECODE_REPACK=1 and HIPENGINE_GGUF_GEMV_DECODE=1 (and WMMA requested only to prove qwen35moe safety gating); inspect candidate.fastpath_safety_repeats for requested vs effective qwen35moe flags.",
            "Use this command as the correctness gate for P9.A3/P9.B7-style benchmark acceptance rows before reporting throughput.",
        ],
    }


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default="", help="Override fixture model path")
    parser.add_argument("--quant", default="", help="Override fixture quant key")
    parser.add_argument("--decode-tokens", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help="Optional precomputed hipcc --version text; omit to let the build cache probe the compiler when needed.",
    )
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = run(args)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
