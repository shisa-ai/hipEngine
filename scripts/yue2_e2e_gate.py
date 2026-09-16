#!/usr/bin/env python3
"""YuE2 end-to-end product gate.

Two parts, both torch-free:

**Recorded 12-case replay.** Each committed case fixture
(``tests/fixtures/yue2/cases/``) holds the reference pipeline's own assembled
prefix and semantic codes plus the statistics of the audio it produced. The gate
replays those through the native NAR and VAE stages and checks that the product
path reproduces the recorded shape of every case: the same latent frame count, the
same audio length, finite latents and audio, and a repeat of one case that is
bit-identical. Arithmetic parity against the reference is established by the M4
solver gate and the M5 decoder gate; this gate is about the product path wiring.

**Live end-to-end request.** A short greedy request runs through the full
``Yue2Session`` (AR plan -> semantic -> NAR -> VAE) twice. Greedy decoding is
deterministic, so both runs must produce identical latent and audio identities.

The gate also asserts that ``torch`` was never imported: the product path is
required to be torch-free, and a silently imported torch would invalidate that
claim rather than fail loudly.

Usage:
    python3 scripts/yue2_e2e_gate.py [--steps 4] [--json OUT] [--skip-live]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CASES = REPO / "tests/fixtures/yue2/cases"
PROTOCOL = "yue2-e2e-product-gate-v1"
CACHE = Path.home() / ".cache/huggingface/hub"
SEED_PATTERN = re.compile(r"-s(\d+)$")


def _host_identity() -> dict:
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
        for line in completed.stdout.splitlines():
            if line.strip().startswith("Name:") and "gfx" in line:
                gpu = line.split(":", 1)[1].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    return {"hostname": name, "cpu": cpu, "gpu": gpu, "platform": sys.platform}


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
            timeout=30, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _resolve(env: str, pattern: str) -> Path:
    import os

    override = os.environ.get(env)
    if override:
        return Path(override)
    for directory in sorted(CACHE.glob(pattern)):
        if (directory / "model.safetensors").is_file():
            return directory
    raise SystemExit(f"checkpoint for {pattern} not found; set {env}")


def _seed_of(name: str) -> int:
    match = SEED_PATTERN.search(name)
    if not match:
        raise SystemExit(f"case {name!r} does not end in -s<seed>")
    return int(match.group(1))


def _live_request():
    from hipengine.generation.yue2 import Sampling, SongRequest

    return (
        SongRequest(
            style="Warm acoustic pop, clear lead vocal, fingerpicked guitar",
            lyrics="[verse]\nMorning light across the floor\nA quiet room, an open door",
            cot="off",
            seed=20260916,
            id="e2e-gate",
        ),
        Sampling(temperature=0.0, top_p=1.0, top_k=1, repetition_penalty=1.0,
                 penalty_window=50, min_tokens=0, max_tokens=96),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--steps", type=int, default=4,
                        help="ODE steps for the 12-case replay (the M4 gate owns solver parity)")
    parser.add_argument("--live-steps", type=int, default=32)
    parser.add_argument("--only", default="", help="comma-separated case names")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from hipengine.loading.yue2 import load_yue2_vae_decoder, load_yue2_weights
    from hipengine.runtime.yue2_nar import Yue2NarRuntime
    from hipengine.runtime.yue2_vae import Yue2VaeRuntime

    cases_dir = Path(args.cases)
    manifest = json.loads((cases_dir / "manifest.json").read_text())
    names = args.only.split(",") if args.only else sorted(manifest)
    if args.skip_replay:
        names = []
    steps = int(args.steps)

    report: dict = {
        "provenance": {
            "command_line": " ".join(sys.argv),
            "host": _host_identity(),
            "revision": _revision(),
            "cases_dir": str(cases_dir),
            "protocol": PROTOCOL,
        },
        "ode_steps": steps,
        "cases": {},
    }

    model_dir = _resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*")
    vae_dir = _resolve("YUE2_VAE_DIR", "models--m-a-p--YuE2-Vae/snapshots/*")
    weights = load_yue2_weights(model_dir)
    vae_weights = load_yue2_vae_decoder(vae_dir)

    from hipengine.runtime.yue2_ar import Yue2ArRuntime

    # The product configuration: the semantic phase uses classifier-free guidance,
    # which needs both branches. The NAR only ever reads branch 0 and re-prefills
    # its own conditioning, so the extra branch costs AR time and nothing else.
    ar = Yue2ArRuntime(weights, branches=2)
    nar = Yue2NarRuntime(weights, ar)
    vae = Yue2VaeRuntime(vae_weights)

    ok = True
    replay_started = time.perf_counter()
    for name in names:
        entry = manifest[name]
        arrays = np.load(cases_dir / f"{name}.npz")
        prefix = [int(v) for v in arrays["prefix"]]
        codec = [int(v) for v in arrays["semantic"]]
        seed = _seed_of(name)
        started = time.perf_counter()
        latents = nar.synthesize(prefix, codec, seed, steps=steps)
        audio = vae.decode_tiled(latents.T[None, ...], core_frames=1024, halo_frames=16)[0]
        case = {
            "prefix_tokens": len(prefix),
            "semantic_tokens": len(codec),
            "frames": int(latents.shape[0]),
            "expected_frames": int(entry["latent_frames"]),
            "samples": int(audio.shape[-1]),
            "expected_samples": int(round(float(entry["audio_seconds"]) * 48000)),
            "latents_finite": bool(np.isfinite(latents).all()),
            "audio_finite": bool(np.isfinite(audio).all()),
            "audio_peak": float(np.abs(audio).max()),
            "seconds": time.perf_counter() - started,
        }
        case["frames_match"] = case["frames"] == case["expected_frames"]
        case["samples_match"] = case["samples"] == case["expected_samples"]
        case["passed"] = bool(
            case["frames_match"]
            and case["samples_match"]
            and case["latents_finite"]
            and case["audio_finite"]
            and case["audio_peak"] > 0.0
        )
        ok = ok and case["passed"]
        report["cases"][name] = case
        print(
            f"[e2e] {name:26s} frames={case['frames']}/{case['expected_frames']} "
            f"samples={case['samples']}/{case['expected_samples']} "
            f"peak={case['audio_peak']:.4f} {case['seconds']:.1f}s "
            f"{'ok' if case['passed'] else 'FAIL'}"
        )
    report["replay_seconds"] = time.perf_counter() - replay_started
    report["replay_skipped"] = bool(args.skip_replay)

    # Determinism: replay one case again and require bit-identical latents.
    if names:
        probe = names[0]
        arrays = np.load(cases_dir / f"{probe}.npz")
        again = nar.synthesize(
            [int(v) for v in arrays["prefix"]], [int(v) for v in arrays["semantic"]],
            _seed_of(probe), steps=steps,
        )
        first = nar.synthesize(
            [int(v) for v in arrays["prefix"]], [int(v) for v in arrays["semantic"]],
            _seed_of(probe), steps=steps,
        )
        report["determinism"] = {
            "case": probe,
            "bit_identical": bool(np.array_equal(again, first)),
        }
        ok = ok and report["determinism"]["bit_identical"]
        print(
            f"[e2e] determinism {probe}: bit-identical="
            f"{report['determinism']['bit_identical']}"
        )

    if not args.skip_live:
        from hipengine.tokenization.yue2 import YuE2TextTokenizer
        from hipengine.runtime.yue2_session import Yue2Session, Yue2ArSession

        tokenizer = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")
        session = Yue2Session(
            Yue2ArSession(ar, encode=tokenizer.encode, decode=tokenizer.decode),
            nar,
            vae,
        )
        request, greedy = _live_request()
        try:
            started = time.perf_counter()
            first = session.generate(request, abc_sampling=greedy, semantic_sampling=greedy,
                                     steps=int(args.live_steps))
            first_seconds = time.perf_counter() - started
            started = time.perf_counter()
            second = session.generate(request, abc_sampling=greedy, semantic_sampling=greedy,
                                      steps=int(args.live_steps))
            second_seconds = time.perf_counter() - started
            live = {
                "request": request.to_dict(),
                "ode_steps": int(args.live_steps),
                "abc_tokens": len(first.semantic.plan.abc_ids),
                "semantic_tokens": len(first.semantic.tokens),
                "frames": first.frames,
                "samples": int(first.audio.shape[-1]),
                "duration_seconds": first.duration_seconds,
                "audio_peak": float(np.abs(first.audio).max()),
                "audio_finite": bool(np.isfinite(first.audio).all()),
                "request_id": first.request_id,
                "repeat_request_id_equal": first.request_id == second.request_id,
                "repeat_latent_identical": first.latent_identity == second.latent_identity,
                "repeat_audio_identical": first.audio_identity == second.audio_identity,
                "first_seconds": first_seconds,
                "second_seconds": second_seconds,
                "timing": first.timing,
            }
            live["passed"] = bool(
                live["audio_finite"]
                and live["audio_peak"] > 0.0
                and live["samples"] > 0
                and live["repeat_latent_identical"]
                and live["repeat_audio_identical"]
            )
            report["live"] = live
            ok = ok and live["passed"]
            print(
                f"[e2e] live greedy: abc={live['abc_tokens']} semantic={live['semantic_tokens']} "
                f"frames={live['frames']} seconds={live['duration_seconds']:.2f} "
                f"peak={live['audio_peak']:.4f} repeat_identical={live['repeat_latent_identical']} "
                f"{first_seconds:.1f}s/{second_seconds:.1f}s "
                f"{'ok' if live['passed'] else 'FAIL'}"
            )
        finally:
            session.close()
    else:
        nar.close()
        vae.close()

    # The product path must never pull torch in.
    report["torch_imported"] = "torch" in sys.modules
    ok = ok and not report["torch_imported"]
    print(f"[e2e] torch imported: {report['torch_imported']}")

    report["passed"] = bool(ok)
    report["cases_passed"] = sum(1 for case in report["cases"].values() if case["passed"])
    report["cases_total"] = len(report["cases"])
    print(
        f"[e2e] {report['cases_passed']}/{report['cases_total']} cases "
        f"replay_seconds={report['replay_seconds']:.1f} -> "
        f"{'PASS' if report['passed'] else 'FAIL'}"
    )
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[e2e] wrote {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
