#!/usr/bin/env python3
"""Is the gate's `prefill_last` row measuring the route, or prefill noise?

The X-7 long-form heldout run reports `shape=prefill_last` and
`transition=prefill_to_c1` at mean KL **1.439e-3**, and both `dense_wide` and
`wmma` fail those two scopes. Every canonical 12-case gate artifact - every
route, every scope-ladder rung, every commit, including the 2026-09-17
certificate - reports the same row at **1e-9 to 1e-11**. The two are five orders
of magnitude apart, so before the transition is localized as a defect it has to
be established what that row is a measurement *of*.

`_trajectory_rows(result)` is `result.logits` from `runner.prefill(...,
capture_logits=True)`, so step 0 is the prefill's own last-token logits. This
script asks three questions about that row, on the two fixtures whose artifacts
disagree:

1. **Noise floor.** Two prefills of the same prompt in the *same* arm. If the
   row moves between identical runs, the row is measuring prefill
   nondeterminism and no route attribution survives.
2. **Route effect.** Same prompt, selectors cleared vs `dense_wide` bound.
3. **Carry-over.** The first decode step after each prefill, so a difference in
   the prefill's logits can be separated from a difference in the state the
   prefill left behind.

Usage
-----
    bash scripts/run_qwen4exp_prefill_last_row_probe.sh
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
CANONICAL_FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"
LONGFORM_FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen4exp_longform_heldout_4prompt.json"
ADMISSION_FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen4exp_admission_suite_18prompt.json"

# The four protocols whose artifacts disagree about this row, in the order the
# recorded values run from flat to inverted:
#   canonical  code-p512                 512 tok  prefill row 5.8014e-12, decode 1.0184e-04
#   admission  code_merge_intervals       40 tok  prefill row 2.6345e-05, decode 6.0842e-06
#   admission  heldout_code_rate_limiter  55 tok  prefill row 2.06179e-03 (18-case mean)
#   longform   long_mixed_ja_en_acceptance 512 tok prefill row 1.4390e-03 (4-case mean)
SUBJECTS = (
    ("canonical-code-p512", CANONICAL_FIXTURE, "code-p512"),
    ("admission-code-merge-40", ADMISSION_FIXTURE, "code_merge_intervals"),
    ("admission-heldout-ratelimiter-55", ADMISSION_FIXTURE, "heldout_code_rate_limiter"),
    ("longform-mixed-512", LONGFORM_FIXTURE, "long_mixed_ja_en_acceptance"),
    ("longform-ja-424", LONGFORM_FIXTURE, "long_general_ja_retention_review"),
)

WMMA_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"
DENSE_WIDE_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"
DENSE_WIDE_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"
DENSE_WIDE_LAYERS = ",".join(str(layer) for layer in range(16, 48))


class ProbeError(RuntimeError):
    """A measurement precondition failed."""


@contextlib.contextmanager
def route_environment(*, candidate: bool):
    """Bind exactly the selectors the gate's `dense_wide` route binds."""

    names = (WMMA_LAYERS_ENV, DENSE_WIDE_ENV, DENSE_WIDE_LAYERS_ENV)
    previous = {name: os.environ.get(name) for name in names}
    if candidate:
        os.environ[WMMA_LAYERS_ENV] = ""
        os.environ[DENSE_WIDE_ENV] = "1"
        os.environ[DENSE_WIDE_LAYERS_ENV] = DENSE_WIDE_LAYERS
    else:
        os.environ[WMMA_LAYERS_ENV] = ""
        os.environ[DENSE_WIDE_ENV] = "0"
        os.environ[DENSE_WIDE_LAYERS_ENV] = ""
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _compare(reference: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if reference.shape != other.shape:
        raise ProbeError(f"logits shape mismatch: {reference.shape} vs {other.shape}")
    log_reference = reference - float(reference.max())
    log_other = other - float(other.max())
    p = np.exp(log_reference)
    p /= p.sum()
    q = np.exp(log_other)
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    kl = float(
        np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(midpoint, 1e-300))), 0.0))
    )
    return {
        "kl_symmetric": kl,
        "max_abs_diff": float(np.max(np.abs(reference - other))),
        "identical": bool(np.array_equal(reference, other)),
        "argmax_equal": bool(int(np.argmax(reference)) == int(np.argmax(other))),
    }


def _one_prefill(runner: Any, tokens: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (step-0 logits, step-1 logits, step-1 token id)."""

    runner.reset()
    result = runner.prefill([int(t) for t in tokens], capture_logits=True)
    if result.logits is None:
        raise ProbeError("prefill did not capture logits")
    step0 = np.asarray(result.logits, dtype=np.float32).copy()
    step1 = runner.step(int(result.token_id), capture_logits=True)
    if step1.logits is None:
        raise ProbeError("decode step did not capture logits")
    return step0, np.asarray(step1.logits, dtype=np.float32).copy(), int(step1.token_id)


def _warm(runner: Any, tokens: Sequence[int]) -> None:
    runner.reset()
    runner.prefill([int(t) for t in tokens[:32]], capture_logits=False)


def _case(fixture: Path, case_id: str) -> list[int]:
    data = json.loads(fixture.read_text())
    for row in data["cases"]:
        if str(row["id"]) == case_id:
            return [int(t) for t in row["prompt_token_ids"]]
    raise ProbeError(f"case {case_id!r} not found in {fixture}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
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

    model_root = args.model_root.resolve()
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile("production"),
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=4096 + 32 + 8,
            prefill_chunk_size=1024,
        )

    generator = resolved.construct_generator(factory)
    runner = generator.runner
    if runner is None:
        raise ProbeError("generator is not resident")

    subjects = list(SUBJECTS)

    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_prefill_last_row_probe",
        "model_root": str(model_root),
        "repeats": int(args.repeats),
        "dense_wide_layers": DENSE_WIDE_LAYERS,
        "subjects": [],
    }

    try:
        for label, fixture, case_id in subjects:
            tokens = _case(fixture, case_id)
            record: dict[str, Any] = {
                "subject": label,
                "fixture": str(fixture.relative_to(REPO_ROOT)),
                "case_id": case_id,
                "prompt_tokens": len(tokens),
                "arms": {},
            }
            captured: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {}
            for arm, candidate in (("cleared", False), ("dense_wide", True)):
                runs: list[tuple[np.ndarray, np.ndarray, int]] = []
                with route_environment(candidate=candidate):
                    _warm(runner, tokens)
                    for _ in range(int(args.repeats)):
                        runs.append(_one_prefill(runner, tokens))
                captured[arm] = runs
                record["arms"][arm] = {
                    "step0_token_ids": [run[2] for run in runs],
                    "step0_same_arm_repeats": [
                        _compare(runs[0][0], runs[i][0]) for i in range(1, len(runs))
                    ],
                    "step1_same_arm_repeats": [
                        _compare(runs[0][1], runs[i][1]) for i in range(1, len(runs))
                    ],
                }
            record["step0_route_effect"] = _compare(
                captured["cleared"][0][0], captured["dense_wide"][0][0]
            )
            record["step1_route_effect"] = _compare(
                captured["cleared"][0][1], captured["dense_wide"][0][1]
            )
            payload["subjects"].append(record)
            if args.json is not None:
                # Persist after every subject: a report-side bug must not be able
                # to discard measurements that already cost a model load.
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(
                    json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
                )
            print(f"[{label}] {len(tokens)} tokens")
            print(
                "   same-arm step0 (noise floor): "
                + "  ".join(
                    f"kl={c['kl_symmetric']:.3e} max={c['max_abs_diff']:.3e} ident={c['identical']}"
                    for c in record["arms"]["cleared"]["step0_same_arm_repeats"]
                )
            )
            print(
                "   same-arm step1 (noise floor): "
                + "  ".join(
                    f"kl={c['kl_symmetric']:.3e} max={c['max_abs_diff']:.3e} ident={c['identical']}"
                    for c in record["arms"]["cleared"]["step1_same_arm_repeats"]
                )
            )
            print(
                "   route effect step0: kl={:.3e} max={:.3e} ident={} argmax_eq={}".format(
                    record["step0_route_effect"]["kl_symmetric"],
                    record["step0_route_effect"]["max_abs_diff"],
                    record["step0_route_effect"]["identical"],
                    record["step0_route_effect"]["argmax_equal"],
                )
            )
            print(
                "   route effect step1: kl={:.3e} max={:.3e} ident={} argmax_eq={}".format(
                    record["step1_route_effect"]["kl_symmetric"],
                    record["step1_route_effect"]["max_abs_diff"],
                    record["step1_route_effect"]["identical"],
                    record["step1_route_effect"]["argmax_equal"],
                ),
                flush=True,
            )
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
