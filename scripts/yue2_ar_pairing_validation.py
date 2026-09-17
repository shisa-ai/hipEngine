"""Validate the two-row AR head on the production session loop, not on raw logits.

`dense_gemv_bf16_f32_out_rowtile2` and `Yue2ArRuntime.logits` are bit-identical per row
to the serial head, and the replay gate plus the matched timing run confirm that. This
script checks the layers above the kernel, because those are what a caller actually
consumes:

* per-branch logits, through the runtime's own `logits(branch)` path;
* the combined CFG score row (`combine_cfg`), including the branch-1 draw;
* the sampler's score row and softmax distribution (`distribution`, `softmax_f32`),
  compared as distributions rather than as tokens, with the top-1 and the KL between
  them;
* the tokens the session loop actually emits, with temperature 1.0 so the sampler and
  the RNG stream are exercised, and the RNG state after the run;
* session lifecycle: case A, then case B, then case A again in one process, with a
  `runtime.reset()` in between, so a leaked or stale pair cannot hide behind a fresh
  process.

The serial side is a subclass whose `logits` is the pre-pairing implementation (final
norm into the single-row head per branch). Bit identity is reported as a diagnostic; the
acceptance bar is that the tokens, the RNG state and the distributions agree.

    python3 scripts/yue2_ar_pairing_validation.py [--json PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.generation.yue2 import Sampling  # noqa: E402
from hipengine.kernels.hip_gfx1100.linear import dense_gemv  # noqa: E402
from hipengine.loading.yue2 import load_yue2_weights  # noqa: E402
from hipengine.runtime.yue2_ar import (  # noqa: E402
    Yue2ArRuntime,
    bf16_bits_to_f32,
    f32_to_bf16_bits,
)
from hipengine.runtime.yue2_session import (  # noqa: E402
    combine_cfg,
    distribution,
    generate_tokens,
    softmax_f32,
)

FIXTURES = REPO / "tests/fixtures/yue2/greedy"
CASES = ("mandarin-off-s1234", "english-melody-s1234")
PROBE_TOKENS = 64


class _RecordingRuntime(Yue2ArRuntime):
    """The production runtime, recording the rows each `logits` call returns."""

    def __init__(self, *args, serial_head: bool = False, **kwargs) -> None:
        self.serial_head = serial_head
        self.recorded: list[np.ndarray] = []
        super().__init__(*args, **kwargs)

    def logits(self, branch: int = 0, *, as_bf16: bool = True) -> np.ndarray:
        if not self.serial_head:
            row = super().logits(branch, as_bf16=False)
        else:
            # The pre-pairing implementation, verbatim: per-branch norm and one
            # single-row head call, then a device-to-host copy.
            hidden = self.spec.hidden_size
            self.kernels.vv_rmsnorm_bf16(
                self._hidden[branch].ptr, self.final_ln.ptr, self._normed.ptr, 1, hidden,
                self.spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            dense_gemv.dense_gemv_bf16_f32_out(
                self._normed.ptr, self.lm_head.ptr, self._logits_f32.ptr, 1, hidden,
                self.spec.vocab_size,
            )
            row = np.empty(self.spec.vocab_size, dtype=np.float32)
            copy_device_to_host(host_array_ptr(row), self._logits_f32, self.spec.vocab_size * 4)
        self.recorded.append(row.copy())
        return f32_to_bf16_bits(row) if as_bf16 else row


def _run(runtime, prefix, negative, sampling, seed, cfg_scale, phase):
    runtime.recorded = []
    ids, timing, truncated = generate_tokens(
        runtime, prefix, sampling, seed, phase,
        negative=negative, cfg_scale=cfg_scale,
    )
    return ids, timing, truncated, runtime.recorded


def _distribution_probe(recorded, sampling, cfg_scale, negative_present):
    """The first step's combined scores and softmax row, rebuilt from the recorded rows."""

    conditional = recorded[0]
    if not negative_present:
        logits = conditional
    else:
        logits = combine_cfg(conditional, recorded[1], cfg_scale)
    scores = distribution(logits, sampling, [], 0, "semantic")
    return logits, scores, softmax_f32(scores)


def _masked_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs difference over the entries both sides keep finite.

    `distribution` masks disallowed tokens with `-inf`, so a plain subtraction turns
    those into `nan`. The masks are structural, so they are compared separately.
    """

    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    return float(np.abs(a[finite] - b[finite]).max())


def _compare(paired, serial, sampling, cfg_scale, negative_present):
    out = {"steps_recorded": [len(paired), len(serial)]}
    p_logits, p_scores, p_probs = _distribution_probe(paired, sampling, cfg_scale, negative_present)
    s_logits, s_scores, s_probs = _distribution_probe(serial, sampling, cfg_scale, negative_present)
    out["logits_max_abs"] = float(np.abs(p_logits - s_logits).max())
    out["logits_exact"] = bool(np.array_equal(p_logits, s_logits))
    out["cfg_scores_max_abs"] = _masked_max_abs(p_scores, s_scores)
    out["cfg_scores_exact"] = bool(np.array_equal(p_scores, s_scores))
    out["mask_identical"] = bool(np.array_equal(np.isfinite(p_scores), np.isfinite(s_scores)))
    out["probs_max_abs"] = float(np.abs(p_probs - s_probs).max())
    out["probs_exact"] = bool(np.array_equal(p_probs, s_probs))
    finite = (p_probs > 0) & (s_probs > 0)
    out["probs_kl"] = float(
        np.sum(p_probs[finite] * np.log(p_probs[finite] / s_probs[finite]))
    )
    out["top1_agreement"] = bool(int(p_scores.argmax()) == int(s_scores.argmax()))
    out["top1_paired"] = int(p_scores.argmax())
    out["top1_serial"] = int(s_scores.argmax())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    parser.add_argument("--fixtures", default=str(FIXTURES))
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--tokens", type=int, default=PROBE_TOKENS)
    args = parser.parse_args()

    model_dir = os.environ.get("YUE2_MODEL_DIR") or sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/*"))
    )[0]
    weights = load_yue2_weights(model_dir)
    cases = [name.strip() for name in args.cases.split(",") if name.strip()]

    report = {
        "protocol": "yue2-ar-pairing-validation-v1",
        "date": time.strftime("%Y-%m-%d"),
        "host": Path("/etc/hostname").read_text().strip(),
        "model_dir": model_dir,
        "tokens_per_run": args.tokens,
        "cases": {},
        "notes": [
            "Both sides run the production session loop (generate_tokens) with "
            "temperature 1.0, so the sampler and the RNG stream are exercised.",
            "Bit identity is a diagnostic here; the acceptance bar is that tokens, the "
            "RNG state and the distributions agree.",
        ],
    }

    for name in cases:
        fixture = np.load(Path(args.fixtures) / f"{name}.npz")
        prefix = [int(t) for t in fixture["prefix"]]
        negative = [int(t) for t in fixture["negative_prefix"]] if "negative_prefix" in fixture else None
        sampling = Sampling(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
                            penalty_window=50, min_tokens=0, max_tokens=args.tokens)
        cfg_scale = 1.01 if negative is not None else 1.0
        entry = {"prefix_tokens": len(prefix),
                 "negative_tokens": len(negative) if negative is not None else 0,
                 "cfg_scale": cfg_scale, "seed": 1234}

        runs = {}
        for label, serial in (("paired", False), ("serial", True)):
            runtime = _RecordingRuntime(
                weights, branches=2 if negative is not None else 1, max_context=2056,
                serial_head=serial,
            )
            try:
                ids, timing, truncated, recorded = _run(
                    runtime, prefix, negative, sampling, 1234, cfg_scale, "semantic"
                )
                runs[label] = {
                    "ids": ids, "timing": timing, "truncated": truncated,
                    "recorded": recorded,
                }
            finally:
                runtime.close()

        entry["tokens_identical"] = runs["paired"]["ids"] == runs["serial"]["ids"]
        entry["token_count"] = [len(runs["paired"]["ids"]), len(runs["serial"]["ids"])]
        entry["rng_state_identical"] = (
            runs["paired"]["timing"]["random"] == runs["serial"]["timing"]["random"]
        )
        entry["truncated_identical"] = runs["paired"]["truncated"] == runs["serial"]["truncated"]
        entry["first_token"] = [runs["paired"]["ids"][:1], runs["serial"]["ids"][:1]]
        entry["step"] = _compare(
            runs["paired"]["recorded"], runs["serial"]["recorded"], sampling, cfg_scale,
            negative is not None,
        )
        # Every recorded row must match between the two sides, not just the first step.
        rows = min(len(runs["paired"]["recorded"]), len(runs["serial"]["recorded"]))
        worst = 0.0
        for index in range(rows):
            worst = max(worst, float(np.abs(
                runs["paired"]["recorded"][index] - runs["serial"]["recorded"][index]
            ).max()))
        entry["all_steps_max_abs"] = worst
        entry["all_steps_exact"] = worst == 0.0
        report["cases"][name] = entry
        print(f"{name}: tokens identical {entry['tokens_identical']} "
              f"({entry['token_count'][0]} tokens), rng identical {entry['rng_state_identical']}, "
              f"step KL {entry['step']['probs_kl']:.3e}, "
              f"logits exact {entry['step']['logits_exact']}, all steps exact {entry['all_steps_exact']}")

    # Lifecycle: A, B, A again in one process, with a reset between, so a leaked pair
    # cannot hide behind a fresh process.
    if len(cases) >= 2:
        runtime = _RecordingRuntime(weights, branches=2, max_context=2056)
        try:
            first = {}
            order = [cases[0], cases[1], cases[0]]
            for index, name in enumerate(order):
                fixture = np.load(Path(args.fixtures) / f"{name}.npz")
                prefix = [int(t) for t in fixture["prefix"]]
                negative = ([int(t) for t in fixture["negative_prefix"]]
                            if "negative_prefix" in fixture else None)
                sampling = Sampling(temperature=1.0, top_p=0.95, top_k=100,
                                    repetition_penalty=1.2, penalty_window=50, min_tokens=0,
                                    max_tokens=args.tokens)
                cfg_scale = 1.01 if negative is not None else 1.0
                ids, timing, truncated, _ = _run(
                    runtime, prefix, negative, sampling, 1234, cfg_scale, "semantic"
                )
                runtime.reset()
                if name not in first:
                    first[name] = ids
                elif ids != first[name]:
                    report["lifecycle_repeat_identical"] = False
                    report["lifecycle_note"] = f"{name} differed when replayed after {order[index-1]}"
                    break
            else:
                report["lifecycle_repeat_identical"] = True
                report["lifecycle_note"] = f"order {order}; each case's tokens repeated exactly"
        finally:
            runtime.close()
        print(f"lifecycle: {report.get('lifecycle_note')}")

    ok = all(
        case["tokens_identical"] and case["rng_state_identical"] and case["all_steps_exact"]
        for case in report["cases"].values()
    ) and report.get("lifecycle_repeat_identical", False)
    report["status"] = "pass" if ok else "fail"
    print()
    print(f"status: {report['status']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
