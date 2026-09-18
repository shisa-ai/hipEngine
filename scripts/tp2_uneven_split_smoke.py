#!/usr/bin/env python3
"""Smoke the TP2 uneven MLP split end to end and compare it to the even split.

The uneven split is a *production-profile* arithmetic change: ``ffn_gate`` and
``ffn_up`` move independent output rows, but the three MLP roles are coupled -
the rank owning intermediate rows ``[b, b')`` of gate/up must reduce over
exactly those columns of ``ffn_down`` - so the boundary moves
``ffn_down``'s summation grouping. This script answers the cheap question
before the full teacher-coverage gate: does the uneven session build, resolve
its per-rank shapes, generate the same tokens as the even split, and agree on
the top-1 token of every teacher-forced row?

It reports, for both arms:

* the per-rank FFN widths the session actually materialized;
* the resolved dense pair+SiLU decode variant per rank (the allowlist is
  exact-shape, so an unadmitted width silently falls back to the unfused chain
  and this field is how that shows up);
* the generated token ids, the teacher-forced logits, and the top-1 agreement.

The script exits non-zero when the two arms disagree on the generated tokens,
on the logits shape, or on any teacher-forced top-1. Logit values are not
required to match exactly - the split legitimately regroups the ``ffn_down``
summation - so the max/mean absolute logit difference is reported as context
for the numerical gate rather than as a pass/fail.

Usage::

    python scripts/tp2_uneven_split_smoke.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --fractions 0.417145/0.582855 \
        --json benchmarks/results/2026-09-18-w7900-tp2-uneven-split-smoke.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hipengine.distributed.tp2_generate import MlpTP2GenerationSession  # noqa: E402
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402

DEFAULT_PROMPT = (
    "def add(a, b):\n    return a + b\n\nExplain what this function does."
)
TEACHER_ROWS = (1, 2, 3, 4)


def _parse_fractions(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split("/")]
    if len(parts) < 2 or any(not part for part in parts):
        raise SystemExit(f"--fractions must be a '/' separated share per rank, got {text!r}")
    try:
        fractions = tuple(float(part) for part in parts)
    except ValueError as error:
        raise SystemExit(f"--fractions is not numeric: {text!r}") from error
    if any(fraction <= 0.0 for fraction in fractions):
        raise SystemExit(f"--fractions must be positive, got {text!r}")
    return fractions


def _variant_by_rank(variant: Any, world_size: int) -> dict[int, Any]:
    """Normalize the session's variant accessor to a per-rank mapping.

    ``MlpTP2ShardGroup.mlp_decode_variant`` returns a scalar when both ranks
    resolved the same variant and a mapping when they diverged; the report wants
    the per-rank view either way.
    """

    if isinstance(variant, dict):
        return {int(rank): value for rank, value in variant.items()}
    return {rank: variant for rank in range(world_size)}


def run_arm(
    *,
    model: str,
    fractions: tuple[float, ...] | None,
    prompt: str,
    max_new_tokens: int,
    devices: tuple[int, int],
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    session = MlpTP2GenerationSession(
        model,
        devices=devices,
        mode="tp2",
        uneven_split=fractions,
    )
    build_seconds = time.perf_counter() - started
    try:
        world_size = len(devices)
        widths = {
            int(rank): int(value)
            for rank, value in session._shard_group.per_rank_ffn.items()
        }
        variants = _variant_by_rank(session._shard_group.mlp_decode_variant, world_size)
        tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(model))
        prompt_ids = list(tokenizer.encode(prompt))
        started = time.perf_counter()
        result = session.generate(prompt_ids, max_new_tokens=int(max_new_tokens))
        generate_seconds = time.perf_counter() - started
        logits = np.asarray(session.teacher_forced_logits(list(TEACHER_ROWS)))
        tokens = [int(token) for token in result.token_ids]
        print(
            f"[{label}] build {build_seconds:.1f}s generate {generate_seconds:.2f}s "
            f"widths={widths} variants={variants}",
            flush=True,
        )
        print(f"[{label}] tokens={tokens}", flush=True)
        return {
            "label": label,
            "fractions": None if fractions is None else list(fractions),
            "per_rank_ffn": widths,
            "variants": {str(rank): value for rank, value in variants.items()},
            "build_seconds": round(build_seconds, 3),
            "generate_seconds": round(generate_seconds, 3),
            "tokens": tokens,
            "teacher_logits": logits.astype(np.float64).tolist(),
            "teacher_logits_shape": list(logits.shape),
        }
    finally:
        session.close()


def compare(even: dict[str, Any], uneven: dict[str, Any]) -> dict[str, Any]:
    even_logits = np.asarray(even["teacher_logits"], dtype=np.float64)
    uneven_logits = np.asarray(uneven["teacher_logits"], dtype=np.float64)
    shape_match = even_logits.shape == uneven_logits.shape
    report: dict[str, Any] = {
        "teacher_logits_shape_match": bool(shape_match),
        "tokens_identical": even["tokens"] == uneven["tokens"],
    }
    if shape_match and even_logits.size:
        diff = np.abs(even_logits - uneven_logits)
        even_top1 = even_logits.argmax(-1)
        uneven_top1 = uneven_logits.argmax(-1)
        report.update(
            {
                "max_abs_logit_diff": float(diff.max()),
                "mean_abs_logit_diff": float(diff.mean()),
                "top1_agreement": float((even_top1 == uneven_top1).mean()),
                "teacher_rows": int(even_logits.shape[0]) if even_logits.ndim else 0,
            }
        )
    passed = (
        report["teacher_logits_shape_match"]
        and report["tokens_identical"]
        and report.get("top1_agreement") == 1.0
    )
    report["passed"] = bool(passed)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fractions", required=True, type=_parse_fractions)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    devices = tuple(int(part) for part in str(args.devices).split(","))
    if len(devices) != 2:
        raise SystemExit(f"--devices must name exactly two devices, got {args.devices!r}")
    if len(args.fractions) != len(devices):
        raise SystemExit(
            f"--fractions names {len(args.fractions)} shares for {len(devices)} devices"
        )

    even = run_arm(
        model=str(args.model),
        fractions=None,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        devices=devices,
        label="even",
    )
    uneven = run_arm(
        model=str(args.model),
        fractions=args.fractions,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        devices=devices,
        label="uneven",
    )
    report = compare(even, uneven)
    for key in ("max_abs_logit_diff", "mean_abs_logit_diff", "top1_agreement"):
        if key in report:
            print(f"{key}: {report[key]:.6f}")
    print("tokens identical:", report["tokens_identical"])
    print("passed:", report["passed"])

    artifact = {
        "schema": 1,
        "kind": "tp2-uneven-split-smoke",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "devices": list(devices),
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new_tokens),
        "teacher_rows": list(TEACHER_ROWS),
        "arms": {"even": even, "uneven": uneven},
        "comparison": report,
        "note": (
            "A pass requires identical generated tokens, matching logits shape, "
            "and top-1 agreement of 1.0 on every teacher-forced row. Logit "
            "differences are expected and reported as context: the uneven split "
            "regroups the ffn_down summation."
        ),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2) + "\n")
        print("wrote", args.json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
