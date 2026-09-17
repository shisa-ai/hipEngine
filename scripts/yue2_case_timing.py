#!/usr/bin/env python3
"""Time the whole YuE2 product path on one recorded case, stage by stage.

The reference's own per-stage timings are recorded next to each oracle case
(``artifacts/yue2/oracle/cases/<case>/result.json``), so this runs the same request
through the torch-free path and prints both side by side. The semantic stage is
capped at the reference's own token count so the NAR stage solves the same number of
frames on both sides.

Usage:
    python3 scripts/yue2_case_timing.py --case mandarin-off-s1234 --steps 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _resolve(env: str, pattern: str) -> Path:
    import os

    value = os.environ.get(env)
    if value:
        return Path(value)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(cache.glob(pattern))
    if not matches:
        raise SystemExit(f"no model matching {pattern}; set {env}")
    return matches[-1]


def _host_identity() -> dict:
    import subprocess

    def first_line(path: str, prefix: str) -> str:
        try:
            for line in Path(path).read_text().splitlines():
                if line.lower().startswith(prefix):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return ""

    name = ""
    try:
        name = Path("/etc/hostname").read_text().strip()
    except OSError:
        pass
    gpu = ""
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60,
                             check=False).stdout
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.startswith("gfx"):
                gpu = stripped
                break
    except (OSError, subprocess.SubprocessError):
        pass
    return {"hostname": name, "cpu": first_line("/proc/cpuinfo", "model name"),
            "gpu": gpu, "platform": sys.platform}


def _revision() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--oracle", default=str(REPO / "artifacts" / "yue2" / "oracle" / "cases"))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from hipengine.loading.yue2 import load_yue2_vae_decoder, load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime
    from hipengine.runtime.yue2_nar import Yue2NarRuntime
    from hipengine.runtime.yue2_session import (
        GenerationConfig,
        SongRequest,
        Yue2ArSession,
        Yue2Session,
    )
    from hipengine.runtime.yue2_vae import Yue2VaeRuntime
    from hipengine.tokenization.yue2 import YuE2TextTokenizer

    case_dir = Path(args.oracle) / args.case
    request = json.loads((case_dir / "request.json").read_text())
    config = json.loads((case_dir / "config.json").read_text())
    reference = json.loads((case_dir / "result.json").read_text())
    reference_timing = reference["timing"]
    semantic_tokens = int(reference_timing["semantic"]["content_tokens"])

    model_dir = _resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*")
    vae_dir = _resolve("YUE2_VAE_DIR", "models--m-a-p--YuE2-Vae/snapshots/*")
    weights = load_yue2_weights(model_dir)
    vae_weights = load_yue2_vae_decoder(vae_dir)
    tokenizer = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")

    ar = Yue2ArRuntime(weights, branches=2)
    nar = Yue2NarRuntime(weights, ar)
    vae = Yue2VaeRuntime(vae_weights)

    generation = config["generation"]
    session_config = GenerationConfig.from_dict({
        **generation,
        "semantic": {**generation["semantic"], "max_tokens": semantic_tokens + 1},
    })
    session = Yue2Session(
        Yue2ArSession(ar, encode=tokenizer.encode, decode=tokenizer.decode),
        nar,
        vae,
        config=session_config,
    )

    song = SongRequest(
        style=request["style"],
        lyrics=request["lyrics"],
        cot=request.get("cot", "off"),
        seed=int(request.get("seed", 1234)),
        abc=request.get("abc"),
        cfg_scale=float(config["cfg_scale"]) if config.get("cfg_scale") is not None else None,
        id=request["id"],
    )

    started = time.perf_counter()
    result = session.generate(song, steps=int(args.steps))
    total_seconds = time.perf_counter() - started

    timing = result.timing
    semantic_seconds = float(timing.get("semantic", {}).get("seconds", 0.0))
    nar_seconds = float(timing.get("nar_seconds", 0.0))
    vae_seconds = float(timing.get("vae_seconds", 0.0))

    def row(label: str, ours: float, theirs: float) -> str:
        ratio = (theirs / ours) if ours > 0 else float("nan")
        return f"| {label} | {ours:8.2f} s | {theirs:8.2f} s | {ratio:5.2f}x |"

    print(f"case {args.case}: {reference['audio_seconds']:.1f} s of audio, "
          f"{int(reference_timing['semantic']['content_tokens'])} semantic tokens, "
          f"{int(args.steps)} ODE steps")
    print(f"semantic tokens produced: {len(result.semantic.tokens)}")
    print()
    print("| Stage | hipEngine | torch reference | ratio |")
    print("| --- | ---: | ---: | ---: |")
    print(row("AR semantic", semantic_seconds, float(reference_timing["semantic"]["seconds"])))
    print(row("NAR solve", nar_seconds, float(reference_timing["nar_seconds"])))
    print(row("VAE decode", vae_seconds, float(reference_timing["vae_seconds"])))
    print(row("total", total_seconds, float(reference_timing["e2e_seconds"])))
    print()
    print(f"torch imported: {'torch' in sys.modules}")
    print(f"audio: {result.frames} frames, {result.audio.shape[-1]} samples, "
          f"peak {abs(result.audio).max():.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "protocol": "yue2-product-case-timing-v1",
            "command_line": " ".join(["scripts/yue2_case_timing.py", *sys.argv[1:]]),
            "revision": _revision(),
            "host": _host_identity(),
            "case": args.case,
            "steps": int(args.steps),
            "audio_seconds": reference["audio_seconds"],
            "semantic_tokens": len(result.semantic.tokens),
            "frames": int(result.frames),
            "hipengine": {
                "semantic_seconds": semantic_seconds,
                "nar_seconds": nar_seconds,
                "vae_seconds": vae_seconds,
                "total_seconds": total_seconds,
            },
            "reference": {
                "semantic_seconds": reference_timing["semantic"]["seconds"],
                "nar_seconds": reference_timing["nar_seconds"],
                "vae_seconds": reference_timing["vae_seconds"],
                "total_seconds": reference_timing["e2e_seconds"],
                "semantic_tokens": int(reference_timing["semantic"]["content_tokens"]),
                "execution": reference_timing["semantic"].get("execution"),
                "cfg_branches": reference_timing["semantic"].get("cfg_branches"),
            },
            "timing": timing,
        }, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
