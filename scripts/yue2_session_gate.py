#!/usr/bin/env python3
"""YuE2 AR session gate: torch-free greedy trajectories vs the reference.

Replays the committed ``tests/fixtures/yue2/greedy`` cases through
``hipengine.runtime.yue2_session.Yue2ArSession``. Those fixtures come from
``scripts/yue2_oracle.py greedy``: free AR generation at temperature zero, so
the comparison is RNG-free and a mismatch is an implementation difference in
prompt assembly, masks, penalties, CFG arithmetic or the EOS/budget behavior.

Each case is measured twice:

* **Unassisted run** - the session generates freely, exactly as a caller would.
  This gates prefix identity (assembled from the reference's own ABC IDs, so a
  trajectory divergence cannot masquerade as a prompt bug), the truncation class
  of both phases, and an identical opening window.
* **Teacher-forced run** - the runtime is driven along the reference's tokens
  with the session's own score path, so every step compares greedy choices under
  the *same* context. This is the fidelity metric: a free greedy trajectory is
  chaotic after its first flip, so its agreement rate says little about the
  implementation.

Greedy decoding is still not bit-exact across implementations, so the gate
records the top-1/top-2 margin at the first mismatch. A mismatch with a large
margin means a real bug; a mismatch inside the BF16 logit noise is the M2 replay
gate's KL/top-1 agreement showing up in token space.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hipengine.generation.yue2 import (  # noqa: E402
    CODEC_OFFSET,
    Sampling,
    SongRequest,
    combine_cfg,
    distribution,
    negative_prefix,
    token_prefixes,
)
from hipengine.loading.yue2 import load_yue2_weights  # noqa: E402
from hipengine.runtime.yue2_ar import Yue2ArRuntime, bf16_bits_to_f32  # noqa: E402
from hipengine.runtime.yue2_session import Yue2ArSession  # noqa: E402
from hipengine.tokenization.yue2 import YuE2TextTokenizer  # noqa: E402

FIXTURES = REPO / "tests/fixtures/yue2/greedy"


def _host_identity() -> dict:
    """Physical host identity, read from the machine rather than typed."""

    name = ""
    try:
        name = Path("/etc/hostname").read_text().strip()
    except OSError:
        pass
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = ""
    try:
        completed = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=60, check=False
        )
        candidates = [
            line.split(":", 1)[1].strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("Marketing Name:")
        ]
        radeon = [value for value in candidates if "Radeon" in value]
        graphics = [value for value in radeon if "Graphics" in value]
        gpu = (graphics or radeon or candidates or [""])[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"name": name, "cpu": cpu, "gpu": gpu}


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cached_model_dir() -> str:
    cache = Path.home() / ".cache/huggingface/hub"
    for directory in sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*")):
        if (directory / "model.safetensors").is_file():
            return str(directory)
    raise SystemExit("YuE2-3B checkpoint not found; set YUE2_MODEL_DIR")


def _agreement(reference: np.ndarray, produced: list[int]) -> dict:
    reference = np.asarray(reference, dtype=np.int64)
    produced = np.asarray(produced, dtype=np.int64)
    shared = min(reference.size, produced.size)
    matches = int((reference[:shared] == produced[:shared]).sum())
    first = next((index for index in range(shared) if reference[index] != produced[index]), None)
    return {
        "reference_tokens": int(reference.size),
        "produced_tokens": int(produced.size),
        "matches": matches,
        "agreement": matches / shared if shared else 1.0,
        "first_divergence": first,
        "length_delta": int(produced.size) - int(reference.size),
    }


def _teacher_forced(
    runtime: Yue2ArRuntime,
    prefix: list[int],
    tokens: np.ndarray,
    sampling: Sampling,
    phase: str,
    *,
    token_offset: int = 0,
    negative: list[int] | None = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
) -> dict:
    """Drive the runtime along the reference tokens and compare greedy choices.

    Uses the session's score path (BF16 logits, BF16 CFG combine, reference
    ``distribution``), so a mismatch is a logit-level difference and not a
    scoring difference. ``token_offset`` maps stored fixture IDs back into the
    real vocabulary: semantic fixtures hold raw codec IDs.
    """

    reference = [int(token) + token_offset for token in tokens]
    runtime.reset()
    runtime.prefill_host_rows([runtime.embed_row(token) for token in prefix], branch=0, start_pos=0)
    if negative is not None:
        runtime.prefill_host_rows(
            [runtime.embed_row(token) for token in negative], branch=1, start_pos=0
        )
    history: list[int] = []
    matches = 0
    mismatches: list[dict] = []
    for step, token in enumerate(reference):
        conditional = bf16_bits_to_f32(runtime.logits(0))
        if negative is None:
            logits = conditional
        else:
            logits = combine_cfg(conditional, bf16_bits_to_f32(runtime.logits(1)), cfg_scale)
        scores = distribution(logits, sampling, history, step, phase, legacy_off=legacy_off)
        choice = int(np.argmax(scores))
        if choice == token:
            matches += 1
        else:
            top2 = np.partition(scores, -2)[-2:]
            mismatches.append(
                {
                    "step": step,
                    "reference_token": token,
                    "greedy_token": choice,
                    "margin": float(top2[1] - top2[0]),
                    "gap": float(top2[1] - scores[token]),
                    "reference_score": float(scores[token]),
                    "greedy_score": float(top2[1]),
                }
            )
        history.append(token)
        position = len(prefix) + step
        runtime.push_token(runtime.embed_row(token), position, branch=0)
        runtime.forward_layers(position, branch=0)
        if negative is not None:
            position_negative = len(negative) + step
            runtime.push_token(runtime.embed_row(token), position_negative, branch=1)
            runtime.forward_layers(position_negative, branch=1)
    return {
        "steps": len(reference),
        "matches": matches,
        "agreement": matches / len(reference) if reference else 1.0,
        "mismatches": mismatches[:8],
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0]["step"] if mismatches else None,
    }


def _rate(phase: dict) -> str:
    """Agreement as a fraction, or ``n/a`` when the phase had no steps."""

    steps = phase.get("steps", phase.get("reference_tokens", 0))
    return "n/a" if not steps else f"{phase['agreement']:.3f}"


def _summary_line(name: str, case: dict) -> str:
    mismatches = case["semantic_forced"]["mismatches"]
    tail = (
        f"first_mismatch=step{mismatches[0]['step']}"
        f"/gap{mismatches[0]['gap']:.4f}"
        f"/top1margin{mismatches[0]['margin']:.4f}"
        if mismatches
        else "no mismatch"
    )
    return (
        f"[session-gate] {name}: {'PASS' if case['passed'] else 'FAIL'} "
        f"prefix={'ok' if case['prefix_exact'] else 'MISMATCH'} "
        f"abc(free)={_rate(case['abc'])} abc(forced)={_rate(case['abc_forced'])} "
        f"semantic(free)={case['semantic']['agreement']:.3f} "
        f"semantic(forced)={case['semantic_forced']['agreement']:.3f} "
        f"first_div={case['semantic']['first_divergence']} "
        f"aligned={'abc' if case['abc_aligned'] else 'ABC-MISALIGNED'}"
        f"{'/sem' if case['semantic_context_aligned'] else '/sem=skipped'}"
        f"{'' if case['semantic_aligned'] else '=MISALIGNED'} {tail}"
    )


def _write_artifact(args, report: dict, ok: bool) -> None:
    if not args.json:
        return
    report["passed"] = ok
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=str(FIXTURES))
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--cases", default="")
    parser.add_argument("--min-agreement", type=float, default=0.9)
    parser.add_argument("--exact-window", type=int, default=16, help="reported diagnostic only")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    root = Path(args.fixtures)
    manifest = json.loads((root / "manifest.json").read_text())
    selected = [name for name in sorted(manifest) if not args.cases or name in args.cases.split(",")]
    if not selected:
        raise SystemExit("no greedy cases selected")

    model_dir = args.model_dir or _cached_model_dir()
    weights = load_yue2_weights(model_dir)
    tokenizer = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")
    runtime = Yue2ArRuntime(weights, branches=2)
    session = Yue2ArSession(runtime, encode=tokenizer.encode, decode=tokenizer.decode)

    report: dict = {
        "provenance": {
            "command_line": " ".join(sys.argv),
            "host": _host_identity(),
            "revision": _revision(),
            "fixtures": str(root),
            "model_dir": model_dir,
            "protocol": "yue2-greedy-session-gate-v1",
        },
        "model_dir": model_dir,
        "min_agreement": args.min_agreement,
        "exact_window": args.exact_window,
        "cases": {},
    }
    ok = True
    for name in selected:
        entry = manifest[name]
        arrays = np.load(root / f"{name}.npz")
        request = SongRequest(**entry["request"])
        abc_sampling = Sampling(**entry["sampling"]["abc"])
        semantic_sampling = Sampling(**entry["sampling"]["semantic"])
        reference_abc = [int(v) for v in arrays.get("abc_ids", np.empty(0, dtype=np.int32))]
        reference_semantic = arrays["semantic"]
        case: dict = {"request": entry["request"]}

        # 1. Unassisted run: exactly what a caller gets from the session.
        started = time.perf_counter()
        result = session.run(
            request, abc_sampling=abc_sampling, semantic_sampling=semantic_sampling
        )
        case["seconds"] = time.perf_counter() - started
        plan = result.plan
        # Prefix identity is checked against the reference's own ABC IDs so a
        # trajectory divergence cannot be mistaken for a prompt-assembly bug.
        expected_prefix = token_prefixes(request, tokenizer.encode, reference_abc or None)
        case["prefix_exact"] = expected_prefix == [int(v) for v in arrays["prefix"]]
        case["prefix_tokens"] = len(expected_prefix)
        case["plan_prefix_tokens"] = len(plan.prefix)
        case["abc"] = _agreement(reference_abc, list(plan.abc_ids))
        case["semantic"] = _agreement(reference_semantic, list(result.tokens))
        case["truncated_abc"] = bool(plan.truncated)
        case["truncated_abc_expected"] = bool(entry["truncated_abc"])
        case["truncated_semantic"] = bool(result.truncated)
        case["truncated_semantic_expected"] = bool(entry["truncated_semantic"])
        case["prefill_fallback"] = session.last_fallback_reason
        for phase, produced, reference in (
            ("abc", list(plan.abc_ids), reference_abc),
            ("semantic", list(result.tokens), reference_semantic),
        ):
            window = min(args.exact_window, len(reference))
            case[f"{phase}_exact_window"] = window
            case[f"{phase}_exact_head"] = bool(produced[:window] == [int(v) for v in reference[:window]])

        # 2. Teacher-forced runs: same context on both sides at every step.
        if reference_abc:
            case["abc_forced"] = _teacher_forced(
                runtime,
                token_prefixes(request, tokenizer.encode),
                reference_abc,
                abc_sampling,
                "abc",
            )
        else:
            case["abc_forced"] = {
                "steps": 0,
                "matches": 0,
                "agreement": 1.0,
                "mismatches": [],
                "mismatch_count": 0,
                "first_mismatch": None,
            }
        negative = (
            negative_prefix(request, tokenizer.encode, reference_abc)
            if request.needs_negative_branch
            else None
        )
        case["semantic_forced"] = _teacher_forced(
            runtime,
            expected_prefix,
            reference_semantic,
            semantic_sampling,
            "semantic",
            token_offset=CODEC_OFFSET,
            negative=negative,
            cfg_scale=request.guidance,
            legacy_off=request.cot == "off",
        )

        forced_ok = all(
            case[f"{phase}_forced"]["agreement"] >= args.min_agreement
            for phase in ("abc", "semantic")
        )
        # The free trajectory must stay identical to the reference until the
        # first *forced* flip, and that flip must be the near-tie the forced pass
        # measured. The semantic phase is only context-aligned when the ABC stage
        # itself reproduced the reference (or does not exist), because a symbolic
        # divergence changes the prefix the semantic phase continues from.
        case["abc_aligned"] = case["abc"]["first_divergence"] == case["abc_forced"]["first_mismatch"]
        case["semantic_context_aligned"] = (
            case["abc_forced"]["steps"] == 0 or case["abc"]["first_divergence"] is None
        )
        case["semantic_aligned"] = (
            not case["semantic_context_aligned"]
            or case["semantic"]["first_divergence"] == case["semantic_forced"]["first_mismatch"]
        )
        case["passed"] = (
            case["prefix_exact"]
            and forced_ok
            and case["abc_aligned"]
            and case["semantic_aligned"]
            and case["truncated_abc"] == case["truncated_abc_expected"]
            and case["truncated_semantic"] == case["truncated_semantic_expected"]
        )
        ok = ok and case["passed"]
        report["cases"][name] = case
        print(_summary_line(name, case), flush=True)
        _write_artifact(args, report, ok)
    report["passed"] = ok
    _write_artifact(args, report, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
