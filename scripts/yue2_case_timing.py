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
            if line.strip().startswith("Name:") and "gfx" in line:
                gpu = line.split(":", 1)[1].strip()
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
    parser.add_argument(
        "--full-ar-paths",
        action="store_true",
        help="pre-change AR arithmetic: full-vocabulary head and full-row sampler, "
             "for a matched A/B on the same protocol",
    )
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

    if args.full_ar_paths:
        # Pre-change arithmetic for a matched A/B: project the whole vocabulary and sample
        # from the full row, which is what the session did before the phase window and the
        # window-relative sampler. The reconstructed row is -inf outside the window, which
        # is what the head's full row carries there.
        import numpy as _np

        from hipengine.generation.yue2 import VOCAB_SIZE, PhaseScores
        from hipengine.generation.yue2 import distribution as _distribution
        from hipengine.runtime import yue2_session as _session

        def _full_row_sampler(values, sampling, history, step, phase, window,
                              legacy_off=False):
            low, high = window
            row = _np.full(VOCAB_SIZE, -_np.inf, dtype=_np.float32)
            row[low:high] = values
            scores = _distribution(row, sampling, history, step, phase, legacy_off=legacy_off)
            return PhaseScores(values=scores[low:high], offset=low)

        _session.phase_window = lambda phase: (0, VOCAB_SIZE)
        _session.distribution_windowed = _full_row_sampler

    started = time.perf_counter()
    result = session.generate(song, steps=int(args.steps))
    total_seconds = time.perf_counter() - started

    timing = result.timing
    semantic_seconds = float(timing.get("semantic", {}).get("seconds", 0.0))
    nar_seconds = float(timing.get("nar_seconds", 0.0))
    vae_seconds = float(timing.get("vae_seconds", 0.0))

    sample_rate = int(getattr(vae, "sample_rate", 48000))
    our_frames = int(result.frames)
    our_samples = int(result.audio.shape[-1])
    our_audio_seconds = our_samples / sample_rate
    our_tokens = len(result.semantic.tokens)
    ref_tokens = int(reference_timing["semantic"]["content_tokens"])
    ref_frames = ref_tokens
    latent_path = case_dir / "latent.npy"
    if latent_path.exists():
        import numpy as _np

        ref_frames = int(_np.load(latent_path, mmap_mode="r").shape[0])
    ref_audio_seconds = float(reference["audio_seconds"])
    ref_samples = int(round(ref_audio_seconds * sample_rate))
    ref_total = float(reference_timing["e2e_seconds"])

    # The harness caps the semantic phase at the reference's own token count so both
    # sides solve the same number of frames. That only holds if the cap reached
    # generation, which is a contract this harness checks rather than assumes.
    if our_tokens > int(session_config.semantic.max_tokens):
        raise SystemExit(
            f"semantic budget not applied: {our_tokens} tokens for a cap of "
            f"{session_config.semantic.max_tokens}"
        )

    def per_unit(seconds: float, units: int) -> float:
        return (seconds / units * 1000.0) if units else float("nan")

    def row(label: str, ours: float, theirs: float, unit: str) -> str:
        ratio = (theirs / ours) if ours > 0 else float("nan")
        return (f"| {label} | {ours:8.2f} ms | {theirs:8.2f} ms | {ratio:5.2f}x | {unit} |")

    elapsed_ratio = total_seconds / ref_total
    rtf_ratio = (total_seconds / our_audio_seconds) / (ref_total / ref_audio_seconds)
    print(f"case {args.case}: reference {ref_audio_seconds:.2f} s of audio / {ref_tokens} "
          f"semantic tokens, ours {our_audio_seconds:.2f} s / {our_tokens} tokens, "
          f"{int(args.steps)} ODE steps")
    print()
    print("| Stage | hipEngine | torch reference | ratio | per unit |")
    print("| --- | ---: | ---: | ---: | --- |")
    print(row("AR semantic", per_unit(semantic_seconds, our_tokens),
              per_unit(float(reference_timing["semantic"]["seconds"]), ref_tokens), "per token"))
    print(row("NAR solve", per_unit(nar_seconds, our_frames),
              per_unit(float(reference_timing["nar_seconds"]), ref_frames), "per frame"))
    print(row("VAE decode", per_unit(vae_seconds, our_frames),
              per_unit(float(reference_timing["vae_seconds"]), ref_frames), "per frame"))
    print()
    print(f"elapsed: ours {total_seconds:.2f} s for {our_audio_seconds:.2f} s of audio "
          f"({total_seconds / our_audio_seconds:.2f}x real time) against the reference's "
          f"{ref_total:.2f} s for {ref_audio_seconds:.2f} s "
          f"({ref_total / ref_audio_seconds:.2f}x real time)")
    print(f"raw elapsed ratio: {elapsed_ratio:.3f}x slower; "
          f"real-time-factor ratio: {rtf_ratio:.3f}x slower "
          f"(different outputs: {our_frames} frames against {ref_frames})")
    print(f"torch imported: {'torch' in sys.modules}")
    print(f"audio: {our_frames} frames, {our_samples} samples, peak {abs(result.audio).max():.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "protocol": "yue2-product-case-timing-v2",
            "command_line": " ".join(["scripts/yue2_case_timing.py", *sys.argv[1:]]),
            "revision": _revision(),
            "host": _host_identity(),
            "case": args.case,
            "steps": int(args.steps),
            "full_ar_paths": bool(args.full_ar_paths),
            "sample_rate": sample_rate,
            "hipengine": {
                "semantic_seconds": semantic_seconds,
                "nar_seconds": nar_seconds,
                "vae_seconds": vae_seconds,
                "total_seconds": total_seconds,
                "semantic_tokens": our_tokens,
                "frames": our_frames,
                "samples": our_samples,
                "audio_seconds": our_audio_seconds,
                "semantic_ms_per_token": per_unit(semantic_seconds, our_tokens),
                "nar_ms_per_frame": per_unit(nar_seconds, our_frames),
                "vae_ms_per_frame": per_unit(vae_seconds, our_frames),
            },
            "reference": {
                "semantic_seconds": reference_timing["semantic"]["seconds"],
                "nar_seconds": reference_timing["nar_seconds"],
                "vae_seconds": reference_timing["vae_seconds"],
                "total_seconds": ref_total,
                "semantic_tokens": ref_tokens,
                "frames": ref_frames,
                "samples": ref_samples,
                "audio_seconds": ref_audio_seconds,
                "semantic_ms_per_token": per_unit(float(reference_timing["semantic"]["seconds"]), ref_tokens),
                "nar_ms_per_frame": per_unit(float(reference_timing["nar_seconds"]), ref_frames),
                "vae_ms_per_frame": per_unit(float(reference_timing["vae_seconds"]), ref_frames),
                "execution": reference_timing["semantic"].get("execution"),
                "cfg_branches": reference_timing["semantic"].get("cfg_branches"),
            },
            "comparison": {
                "raw_elapsed_ratio": elapsed_ratio,
                "real_time_factor_ratio": rtf_ratio,
                "outputs_matched": our_frames == ref_frames and our_tokens == ref_tokens,
                "note": (
                    "The two sides produced different amounts of audio, so the elapsed "
                    "ratio is not a like-for-like comparison; the per-unit rows are. "
                    "Neither total is an estimate: both are measured wall clock."
                ),
            },
            "executed_config": session.effective_config(song, steps=int(args.steps)),
            "timing": timing,
        }, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
