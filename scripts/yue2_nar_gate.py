#!/usr/bin/env python3
"""YuE2 NAR gate: native flow-matching solver vs the recorded reference fixture.

The fixture in ``tests/fixtures/yue2/nar/`` was recorded from the pinned upstream
``yue2`` package (``CachedNAR``) and holds, for one chunk of a fixed song, the
per-step state and first velocity of the 4-step midpoint solve plus the final
latents. This gate drives :class:`hipengine.runtime.yue2_nar.Yue2NarRuntime`
through the same schedule on the same noise and compares both traces.

The comparison is a numerical one, not a token-agreement one: the acoustic
latents are continuous, so the report carries the error relative to the recorded
latent norm, the cosine similarity, and the fraction of BF16 entries that match
exactly. The BF16 exact-match fraction is a *diagnostic*: the AR conditioning
cache and the BF16 projection chain differ from the reference by design in the
last bit, and one BF16 ulp in the velocity is amplified by the solver's 4 steps.

Usage:
    python3 scripts/yue2_nar_gate.py [--fixture PATH] [--steps 4] [--json OUT]
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

from hipengine.loading.yue2 import load_yue2_weights  # noqa: E402
from hipengine.runtime.yue2_ar import Yue2ArRuntime, bf16_bits_to_f32  # noqa: E402
from hipengine.runtime.yue2_nar import (  # noqa: E402
    Yue2NarRuntime,
    solver_schedule,
    song_chunks,
    to_bf16_bits,
)

FIXTURE = REPO / "tests/fixtures/yue2/nar/chunk0.npz"
PROTOCOL = "yue2-nar-fixture-gate-v1"


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
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cached_model_dir() -> str:
    cache = Path.home() / ".cache/huggingface/hub"
    for directory in sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*")):
        if (directory / "model.safetensors").is_file():
            return str(directory)
    raise SystemExit("YuE2-3B checkpoint not found; set --model-dir or YUE2_MODEL_DIR")


def _compare(reference: np.ndarray, produced: np.ndarray, *, scale: float) -> dict:
    """Error metrics between an FP32 reference tensor and BF16 produced bits."""

    reference = np.asarray(reference, dtype=np.float64)
    got = bf16_bits_to_f32(np.asarray(produced, dtype=np.uint16)).astype(np.float64)
    if reference.shape != got.shape:
        raise SystemExit(f"shape mismatch: reference {reference.shape} vs produced {got.shape}")
    delta = got - reference
    reference_norm = float(np.linalg.norm(reference))
    got_norm = float(np.linalg.norm(got))
    exact = int(
        (np.asarray(produced, dtype=np.uint16) == to_bf16_bits(np.asarray(reference, dtype=np.float32))).sum()
    )
    denominator = reference_norm * got_norm
    return {
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else 0.0,
        "relative_to_scale": float(np.abs(delta).max() / scale) if scale else 0.0,
        "cosine": float((reference * got).sum() / denominator) if denominator else 1.0,
        "bf16_exact": exact / reference.size,
        "reference_norm": reference_norm,
        "produced_norm": got_norm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--steps", type=int, default=4, help="must match the recorded fixture")
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    arrays = np.load(fixture)
    manifest_path = fixture.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    prefix = [int(v) for v in arrays["prefix"]]
    codec = [int(v) for v in arrays["codec"]]
    noise = np.asarray(arrays["noise"], dtype=np.float32)
    steps = int(args.steps)
    if steps != int(arrays["velocities"].shape[0]):
        raise SystemExit(
            f"--steps {steps} does not match the recorded {arrays['velocities'].shape[0]} steps"
        )

    model_dir = args.model_dir or _cached_model_dir()
    weights = load_yue2_weights(model_dir)
    ar = Yue2ArRuntime(weights, branches=1)
    runtime = Yue2NarRuntime(weights, ar)
    chunks = song_chunks(prefix, codec, int(manifest.get("seed", 1234)), noise=noise)
    if len(chunks) != 1:
        raise SystemExit(f"fixture is a single-chunk case, got {len(chunks)} chunks")

    report: dict = {
        "provenance": {
            "command_line": " ".join(sys.argv),
            "host": _host_identity(),
            "revision": _revision(),
            "model_dir": model_dir,
            "fixture": str(fixture),
            "protocol": PROTOCOL,
        },
        "steps": steps,
        "frames": int(manifest.get("frames", noise.shape[0])),
        "ar_tokens": len(chunks[0].ar_tokens),
        "nar_rows": chunks[0].nar_length,
        "latent_scale": float(manifest.get("latent_norm", 0.0)),
        "thresholds": {"max_relative_l2": args.max_relative_l2, "min_cosine": args.min_cosine},
        "steps_report": [],
    }

    started = time.perf_counter()
    runtime.condition(chunks[0])
    report["condition_seconds"] = time.perf_counter() - started
    scale = float(manifest.get("latent_norm", 0.0)) or float(np.linalg.norm(arrays["latents"]))

    recorded_velocities = arrays["velocities"]
    recorded_states = arrays["states"]
    # Teacher-forced attribution: evaluate the native velocity at each *recorded*
    # state, so a velocity defect is separated from the solver's own trajectory.
    started = time.perf_counter()
    for index, (raw, raw_mid) in enumerate(solver_schedule(steps)):
        runtime.load_state(recorded_states[index])
        report["steps_report"].append(
            {
                "step": index,
                "raw_t": raw,
                "raw_mid": raw_mid,
                "state": _compare(recorded_states[index], runtime.state_bits(), scale=scale),
                "velocity": _compare(
                    recorded_velocities[index], runtime.velocity_bits(raw), scale=scale
                ),
            }
        )
    report["teacher_forced_seconds"] = time.perf_counter() - started
    # End-to-end: the native solver's own trajectory from the recorded noise.
    started = time.perf_counter()
    runtime.load_state(np.asarray(arrays["states"][0], dtype=np.float32))
    latents = runtime.solve(steps)
    report["solve_seconds"] = time.perf_counter() - started
    report["latents"] = _compare(arrays["latents"], to_bf16_bits(latents), scale=scale)

    latents = report["latents"]
    worst_step = max(
        (entry["velocity"]["relative_l2"] for entry in report["steps_report"]), default=0.0
    )
    report["worst_velocity_relative_l2"] = worst_step
    report["passed"] = bool(
        latents["relative_l2"] <= args.max_relative_l2
        and latents["cosine"] >= args.min_cosine
        and worst_step <= args.max_relative_l2
    )

    print(
        f"[nar-gate] steps={steps} frames={report['frames']} ar={report['ar_tokens']} "
        f"rows={report['nar_rows']}"
    )
    for entry in report["steps_report"]:
        state = entry["state"]
        velocity = entry["velocity"]
        print(
            f"[nar-gate] step {entry['step']} raw={entry['raw_t']:+.6f} "
            f"state rel_l2={state['relative_l2']:.5f} max={state['max_abs']:.4f} "
            f"exact={state['bf16_exact']:.3f} | velocity rel_l2={velocity['relative_l2']:.5f} "
            f"max={velocity['max_abs']:.4f} exact={velocity['bf16_exact']:.3f} "
            f"cos={velocity['cosine']:.6f}"
        )
    print(
        f"[nar-gate] latents rel_l2={latents['relative_l2']:.5f} "
        f"max={latents['max_abs']:.4f} cosine={latents['cosine']:.6f} "
        f"exact={latents['bf16_exact']:.3f} | "
        f"reference norm={latents['reference_norm']:.3f} produced={latents['produced_norm']:.3f}"
    )
    print(
        f"[nar-gate] condition={report['condition_seconds']:.2f}s solve={report['solve_seconds']:.2f}s "
        f"-> {'PASS' if report['passed'] else 'FAIL'}"
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[nar-gate] wrote {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
