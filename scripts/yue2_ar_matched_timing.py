#!/usr/bin/env python3
"""Fixed-token AR decode timing for the torch-free YuE2 runtime.

The product-path comparison this replaces timed our run against the reference's
*recorded* run, and the two generated different numbers of tokens, so only per-token
arithmetic was comparable and the whole-path ratio was not. This harness removes that
freedom: both sides consume the same recorded token trajectory for the same number of
steps, with the same positive and negative prefixes and the same two CFG branches, so
each step is the same work on both sides.

Each side drives its own product decode loop, with sampling replaced by the recorded
token and nothing else changed, so the per-step host interaction each implementation
really pays is included:

* hipEngine: ``logits(0)`` then ``logits(1)`` (final norm, full-vocabulary head, and a
  bf16 row to the host, as ``generate_tokens`` does), then one ``embed_row`` /
  ``push_token`` / ``forward_layers`` per branch.
* reference: ``GraphAR.step(token)``, whose single batched forward covers both branches
  and returns both branches' logits.

Run ``--side hipengine`` under this repo's interpreter and
``scripts/yue2_reference_ar_timing.py --side reference`` under the oracle venv, then
compare the two JSONs with ``--compare``. Running them alternately keeps both sides on
the same host state.

    python3 scripts/yue2_ar_matched_timing.py --case mandarin-off-s1234 \
        --json /tmp/yue2_ar_hip.json
    PYTHONPATH=~/yue2-shootout/shared/upstream ~/venvs/vibevoice-tts-oracle/bin/python \
        scripts/yue2_reference_ar_timing.py --case mandarin-off-s1234 --json /tmp/yue2_ar_ref.json
    python3 scripts/yue2_ar_matched_timing.py --compare /tmp/yue2_ar_hip.json /tmp/yue2_ar_ref.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROTOCOL = "yue2-ar-fixed-token-v1"


def _resolve(env: str, pattern: str) -> Path:
    value = os.environ.get(env)
    if value:
        return Path(value)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(cache.glob(pattern))
    if not matches:
        raise SystemExit(f"no model matching {pattern}; set {env}")
    return matches[-1]


def _digest(tokens) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:16]


def load_case(oracle: Path, case: str, encode) -> dict:
    """The shared inputs: both prefixes and the recorded token trajectory."""

    from hipengine.generation.yue2 import CODEC_OFFSET, SongRequest, negative_prefix

    case_dir = oracle / case
    request = json.loads((case_dir / "request.json").read_text())
    config = json.loads((case_dir / "config.json").read_text())
    prefix = [int(t) for t in np.load(case_dir / "prefix.npy")]
    semantic = np.load(case_dir / "semantic.npy")
    # The stored trajectory is content-only; the loop feeds the raw token.
    trajectory = [int(t) + CODEC_OFFSET for t in semantic]
    song = SongRequest(
        style=request["style"],
        lyrics=request["lyrics"],
        cot=request.get("cot", "off"),
        seed=int(request.get("seed", 1234)),
        abc=request.get("abc"),
        cfg_scale=float(config["cfg_scale"]) if config.get("cfg_scale") is not None else None,
        id=request["id"],
    )
    negative = negative_prefix(song, encode, ())
    return {
        "case": case,
        "case_dir": str(case_dir),
        "prefix": prefix,
        "negative": [int(t) for t in negative],
        "trajectory": trajectory,
        "song": song,
        "guidance": float(song.guidance),
    }


def run_hipengine(case_data: dict, *, steps: int, warmup: int, repeats: int) -> dict:
    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime
    from hipengine.runtime.yue2_session import bf16_bits_to_f32

    model_dir = _resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*")
    weights = load_yue2_weights(model_dir)
    runtime = Yue2ArRuntime(weights, branches=2)
    prefix, negative, trajectory = case_data["prefix"], case_data["negative"], case_data["trajectory"]

    def decode_once() -> tuple[float, float]:
        """One timed decode: prefill both branches, then ``steps`` steps."""

        runtime.reset()
        started = time.perf_counter()
        runtime.prefill_host_rows([runtime.embed_row(t) for t in prefix], branch=0, start_pos=0)
        runtime.prefill_host_rows([runtime.embed_row(t) for t in negative], branch=1, start_pos=0)
        runtime.runtime.device_synchronize()
        prefill_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for step in range(steps - 1):
            # One iteration is the reference's one ``step()``: both branches' logits
            # (final norm, full-vocabulary head, bf16 row to the host) and both
            # branches' next token fed.
            bf16_bits_to_f32(runtime.logits(0))
            bf16_bits_to_f32(runtime.logits(1))
            row = runtime.embed_row(trajectory[step])
            position = runtime.context_length(0)
            runtime.push_token(row, position, branch=0)
            runtime.forward_layers(position, branch=0)
            negative_position = runtime.context_length(1)
            runtime.push_token(row, negative_position, branch=1)
            runtime.forward_layers(negative_position, branch=1)
        runtime.runtime.device_synchronize()
        return prefill_seconds, time.perf_counter() - started

    for _ in range(warmup):
        decode_once()
    prefill_seconds, decode_seconds = decode_once()
    samples = []
    for _ in range(max(0, repeats - 1)):
        samples.append(decode_once())
    return {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "repeat_prefill_seconds": [s[0] for s in samples],
        "repeat_decode_seconds": [s[1] for s in samples],
    }


def _reading(ours: float, theirs: float) -> str:
    """Which side is ahead, stated the way the rollups state it."""

    if ours <= 0 or theirs <= 0:
        return "n/a"
    if theirs < ours:
        return f"reference {ours / theirs:.2f}x faster"
    return f"hipEngine {theirs / ours:.2f}x faster"


def compare(paths: list[str]) -> int:
    sides = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        sides[data["side"]] = data
    if set(sides) != {"hipengine", "reference"}:
        raise SystemExit(f"need one hipengine and one reference JSON, got {sorted(sides)}")
    ours, theirs = sides["hipengine"], sides["reference"]
    problems = []
    for key in ("case", "steps", "prefix_digest", "negative_digest", "trajectory_digest"):
        if ours[key] != theirs[key]:
            problems.append(f"{key}: {ours[key]!r} != {theirs[key]!r}")
    if problems:
        raise SystemExit("the two sides did not run the same protocol:\n  " + "\n  ".join(problems))
    steps = ours["steps"]
    decode_steps = max(1, steps - 1)
    our_ms = ours["decode_seconds"] / decode_steps * 1000.0
    their_ms = theirs["decode_seconds"] / decode_steps * 1000.0
    print(f"case {ours['case']}: {decode_steps} decode steps, "
          f"prefix {ours['prefix_tokens']} / negative {ours['negative_tokens']} tokens")
    print()
    print("| Quantity | hipEngine | torch reference | Reading |")
    print("| --- | ---: | ---: | --- |")
    print(f"| prefill (2 branches) | {ours['prefill_seconds']:.3f} s | "
          f"{theirs['prefill_seconds']:.3f} s | "
          f"{_reading(ours['prefill_seconds'], theirs['prefill_seconds'])} |")
    print(f"| decode, {decode_steps} steps | {ours['decode_seconds']:.2f} s | "
          f"{theirs['decode_seconds']:.2f} s | "
          f"{_reading(ours['decode_seconds'], theirs['decode_seconds'])} |")
    print(f"| per decode step | {our_ms:.2f} ms | {their_ms:.2f} ms | "
          f"{_reading(our_ms, their_ms)} |")
    print()
    print(f"torch imported on the hipEngine side: {ours.get('torch_imported')}")
    print(f"host: {ours['host'].get('hostname')} / {ours['host'].get('gpu')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="hipengine", choices=["hipengine"])
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--oracle", default=str(REPO / "artifacts" / "yue2" / "oracle" / "cases"))
    parser.add_argument("--steps", type=int, default=0, help="0 = the recorded trajectory's own length")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", default="")
    parser.add_argument("--compare", nargs=2, default=None)
    parser.add_argument("--merge", nargs=3, default=None,
                        metavar=("OUT", "HIPENGINE_JSON", "REFERENCE_JSON"),
                        help="write one artifact holding both sides' passes and the derived ratios")
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare)
    if args.merge:
        out, hip_path, ref_path = args.merge
        hip = json.loads(Path(hip_path).read_text())
        ref = json.loads(Path(ref_path).read_text())
        steps = max(1, int(hip["steps"]) - 1)
        for side, data in (("hipengine", hip), ("reference", ref)):
            decode = [data["decode_seconds"], *data.get("repeat_decode_seconds", [])]
            data["decode_seconds_per_pass"] = decode
            data["ms_per_decode_step_per_pass"] = [v / steps * 1000.0 for v in decode]
        our_best = min(hip["ms_per_decode_step_per_pass"])
        their_best = min(ref["ms_per_decode_step_per_pass"])
        artifact = {
            "protocol": hip["protocol"],
            "case": hip["case"],
            "decode_steps": steps,
            "prefix_tokens": hip["prefix_tokens"],
            "negative_tokens": hip["negative_tokens"],
            "prefix_digest": hip["prefix_digest"],
            "negative_digest": hip["negative_digest"],
            "trajectory_digest": hip["trajectory_digest"],
            "guidance": hip["guidance"],
            "host": hip["host"],
            "torch_imported_on_hipengine_side": hip["torch_imported"],
            "hipengine": hip,
            "reference": ref,
            "comparison": {
                "ms_per_decode_step": {"hipengine": our_best, "reference": their_best},
                "reference_advantage": their_best and our_best / their_best,
                "note": (
                    "Both sides drove the same recorded token trajectory for the same "
                    "number of steps with the same 98-token positive and 12-token "
                    "negative prefixes and the same two CFG branches, so each step is "
                    "the same work. Sampling is excluded by design (the token is "
                    "recorded), so this is model work only; the product loop adds the "
                    "host sampling path measured by scripts/yue2_ar_sampling_cost.py."
                ),
            },
        }
        Path(out).write_text(json.dumps(artifact, indent=1))
        print(f"wrote {out}: hipEngine {our_best:.2f} ms/step, reference {their_best:.2f} ms/step, "
              f"reference {our_best / their_best:.2f}x faster")
        return 0

    from hipengine.tokenization.yue2 import YuE2TextTokenizer

    model_dir = _resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*")
    tokenizer = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")
    case_data = load_case(Path(args.oracle), args.case, tokenizer.encode)
    steps = int(args.steps) or len(case_data["trajectory"])

    host = {}
    try:
        host["hostname"] = Path("/etc/hostname").read_text().strip()
    except OSError:
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                host["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60, check=False).stdout
        for line in out.splitlines():
            if line.strip().startswith("Name:") and "gfx" in line:
                host["gpu"] = line.split(":", 1)[1].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass

    result = run_hipengine(case_data, steps=steps, warmup=args.warmup, repeats=args.repeats)
    payload = {
        "protocol": PROTOCOL,
        "side": "hipengine",
        "case": case_data["case"],
        "steps": steps,
        "prefix_tokens": len(case_data["prefix"]),
        "negative_tokens": len(case_data["negative"]),
        "prefix_digest": _digest(case_data["prefix"]),
        "negative_digest": _digest(case_data["negative"]),
        "trajectory_digest": _digest(case_data["trajectory"][:steps]),
        "guidance": case_data["guidance"],
        "host": host,
        "torch_imported": "torch" in sys.modules,
        "revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                   text=True, check=False).stdout.strip(),
        **result,
    }
    our_ms = payload["decode_seconds"] / max(1, steps - 1) * 1000.0
    print(f"case {payload['case']}: {steps} steps, prefix {payload['prefix_tokens']} tokens, "
          f"negative {payload['negative_tokens']} tokens")
    print(f"prefill {payload['prefill_seconds']:.3f} s, decode {payload['decode_seconds']:.2f} s, "
          f"{our_ms:.2f} ms per step")
    if payload["repeat_decode_seconds"]:
        print(f"repeat decode: {['%.2f' % v for v in payload['repeat_decode_seconds']]}")
    print(f"torch imported: {payload['torch_imported']}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
