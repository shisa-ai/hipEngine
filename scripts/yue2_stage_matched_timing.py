#!/usr/bin/env python3
"""Matched NAR/VAE timing for the torch-free YuE2 runtime.

The AR side already has a fixed-token harness; these two stages had the same defect in
their comparison - each side ran a different amount of work, so only per-unit arithmetic
was comparable. Here both sides get identical inputs:

* ``nar``: the case's own prefix and semantic codes with the case's seed, so
  ``song_chunks`` derives the same initial noise and the same chunking on both sides,
  solved for the same number of steps. The chunk-noise digest is compared, not assumed.
* ``vae``: the case's own recorded latents (``latent.npy``), decoded by both decoders,
  with the full decode and the tiled decode (core 1024 / halo 16, the product's tiling)
  timed separately so the tiling choice is visible instead of mixed into one number.

    python3 scripts/yue2_stage_matched_timing.py --stage nar --case mandarin-off-s1234 \
        --json /tmp/yue2_nar_hip.json
    PYTHONPATH=~/yue2-shootout/shared/upstream ~/venvs/vibevoice-tts-oracle/bin/python \
        scripts/yue2_reference_stage_timing.py --stage nar --case mandarin-off-s1234 \
        --json /tmp/yue2_nar_ref.json
    python3 scripts/yue2_stage_matched_timing.py --compare /tmp/yue2_nar_hip.json /tmp/yue2_nar_ref.json
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

PROTOCOL = "yue2-stage-matched-v1"
SAMPLE_RATE = 48000


def _resolve(env: str, pattern: str) -> Path:
    value = os.environ.get(env)
    if value:
        return Path(value)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(cache.glob(pattern))
    if not matches:
        raise SystemExit(f"no model matching {pattern}; set {env}")
    return matches[-1]


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def load_case(oracle: Path, case: str) -> dict:
    case_dir = oracle / case
    request = json.loads((case_dir / "request.json").read_text())
    config = json.loads((case_dir / "config.json").read_text())
    prefix = [int(t) for t in np.load(case_dir / "prefix.npy")]
    semantic = [int(t) for t in np.load(case_dir / "semantic.npy")]
    latent = np.load(case_dir / "latent.npy")
    return {
        "case": case,
        "case_dir": str(case_dir),
        "prefix": prefix,
        "semantic": semantic,
        "latent": latent,
        "seed": int(request.get("seed", 1234)),
        "steps": int(config["generation"]["ode_steps"]),
        "prefix_digest": array_digest(np.asarray(prefix, dtype=np.int64)),
        "semantic_digest": array_digest(np.asarray(semantic, dtype=np.int64)),
        "latent_digest": array_digest(np.asarray(latent, dtype=np.float32)),
    }


def _chunk_noise_digest(prefix, semantic, seed: int, context: int,
                        noise: np.ndarray | None = None) -> str:
    """The noise the solver starts from, so both sides can prove they share it."""

    from hipengine.runtime.yue2_nar import song_chunks

    chunks = song_chunks(prefix, semantic, seed, context=context, noise=noise)
    return array_digest(np.asarray(chunks[0].noise, dtype=np.float32))


def _warm(data: dict, key: str, repeat_key: str) -> float:
    """The warm pass: the reference's first decode in a process is 8x its steady state.

    Torch's first use of these kernels on this device costs far more than the decode
    itself (163.1 s against 19.6 s for the same 1 297 frames), and the recorded
    product numbers in the campaign artifacts carry that same first-use cost plus a
    per-request decoder transfer. Comparing a warm pass against a cold one is not a
    decode comparison, so the steady-state pass is the one that gets compared and the
    first pass is reported alongside it.
    """

    values = [data[key], *data.get(repeat_key, [])]
    return min(values)


def run_nar(case_data: dict, *, repeats: int, noise_path: str = "") -> dict:
    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime
    from hipengine.runtime.yue2_nar import Yue2NarRuntime

    model_dir = _resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*")
    weights = load_yue2_weights(model_dir)
    nar = Yue2NarRuntime(weights, Yue2ArRuntime(weights, branches=1))
    prefix, semantic, seed, steps = (case_data["prefix"], case_data["semantic"],
                                     case_data["seed"], case_data["steps"])
    # Our own seeded draw is not the reference's (see song_chunks' caveat), so a
    # matched comparison has to consume the reference's own whole-song noise.
    noise = np.load(noise_path).astype(np.float32) if noise_path else None
    if noise is not None and noise.shape != (len(semantic), 64):
        raise SystemExit(f"noise shape {noise.shape} does not match {len(semantic)} frames")

    def solve_once() -> tuple[float, np.ndarray]:
        started = time.perf_counter()
        latents = nar.synthesize(prefix, semantic, seed, steps=steps, noise=noise)
        return time.perf_counter() - started, np.asarray(latents, dtype=np.float32)

    first_seconds, latents = solve_once()
    repeats_seconds = [solve_once()[0] for _ in range(max(0, repeats - 1))]
    noise_digest = _chunk_noise_digest(prefix, semantic, seed, 24576, noise=noise)
    nar.close()
    return {
        "solve_seconds": first_seconds,
        "repeat_solve_seconds": repeats_seconds,
        "warm_solve_seconds": min([first_seconds, *repeats_seconds]),
        "noise_source": noise_path or "own seeded draw (not the reference's)",
        "ms_per_step": first_seconds / steps * 1000.0,
        "latents": latents,
        "latent_digest": array_digest(latents),
        "noise_digest": noise_digest,
        "latent_norm": float(np.linalg.norm(latents.astype(np.float64))),
    }


def run_vae(case_data: dict, *, repeats: int) -> dict:
    from hipengine.loading.yue2 import load_yue2_vae_decoder
    from hipengine.runtime.yue2_vae import Yue2VaeRuntime

    vae_dir = _resolve("YUE2_VAE_DIR", "models--m-a-p--YuE2-Vae/snapshots/*")
    vae = Yue2VaeRuntime(load_yue2_vae_decoder(vae_dir))
    latent = np.asarray(case_data["latent"], dtype=np.float32)
    values = latent.T[None, ...]

    def decode_tiled() -> np.ndarray:
        return np.asarray(vae.decode_tiled(values, core_frames=1024, halo_frames=16)[0],
                          dtype=np.float32)

    started = time.perf_counter()
    tiled = decode_tiled()
    tiled_seconds = time.perf_counter() - started
    tiled_repeats = []
    for _ in range(max(0, repeats - 1)):
        started = time.perf_counter()
        decode_tiled()
        tiled_repeats.append(time.perf_counter() - started)
    full = None
    full_seconds = None
    full_note = ""
    try:
        started = time.perf_counter()
        full = np.asarray(vae.decode(values)[0], dtype=np.float32)
        full_seconds = time.perf_counter() - started
    except Exception as error:  # device residency, not correctness
        full_note = f"full decode unavailable: {type(error).__name__}: {error}"
    vae.close()
    frames = int(latent.shape[0])
    return {
        "frames": frames,
        "tiled_seconds": tiled_seconds,
        "repeat_tiled_seconds": tiled_repeats,
        "tiled_ms_per_frame": tiled_seconds / frames * 1000.0,
        "full_seconds": full_seconds,
        "full_ms_per_frame": (full_seconds / frames * 1000.0) if full_seconds else None,
        "full_note": full_note,
        "tiled_digest": array_digest(tiled),
        "full_digest": array_digest(full) if full is not None else "",
        "tiled_vs_full_max_abs": (float(np.abs(tiled - full).max()) if full is not None else None),
        "samples": int(tiled.shape[-1]),
        "audio_seconds": int(tiled.shape[-1]) / SAMPLE_RATE,
        "peak": float(np.abs(tiled).max()),
        "tiled": tiled,
    }


def compare(paths: list[str]) -> int:
    sides = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        sides[data["side"]] = data
    if set(sides) != {"hipengine", "reference"}:
        raise SystemExit(f"need one hipengine and one reference JSON, got {sorted(sides)}")
    ours, theirs = sides["hipengine"], sides["reference"]
    stage = ours["stage"]
    if stage != theirs["stage"]:
        raise SystemExit(f"stage mismatch: {stage} != {theirs['stage']}")
    problems = [key for key in ("case", "steps", "prefix_digest", "semantic_digest")
                if ours.get(key) != theirs.get(key)]
    if problems:
        raise SystemExit("the two sides did not run the same protocol: " + ", ".join(problems))
    print(f"{stage} on {ours['case']}: {ours['steps']} ODE steps, "
          f"prefix {len(ours['prefix'])} / semantic {len(ours['semantic'])} tokens")
    print()
    if stage == "nar":
        if ours["noise_digest"] != theirs["noise_digest"]:
            print(f"** the two sides did not start from the same noise: "
                  f"{ours['noise_digest']} != {theirs['noise_digest']} **")
            print(f"   (ours from: {ours.get('noise_source')})")
        else:
            print(f"initial noise digest matches: {ours['noise_digest']}")
        print()
        print("| Quantity | hipEngine | torch reference | Reading |")
        print("| --- | ---: | ---: | --- |")
        our_warm = _warm(ours, "solve_seconds", "repeat_solve_seconds")
        their_warm = _warm(theirs, "solve_seconds", "repeat_solve_seconds")
        reading = ("reference %.2fx faster" % (our_warm / their_warm) if their_warm < our_warm
                   else "hipEngine %.2fx faster" % (their_warm / our_warm))
        print(f"| warm solve | {our_warm:.3f} s | {their_warm:.3f} s | {reading} |")
        print(f"| first solve (cold) | {ours['solve_seconds']:.3f} s | "
              f"{theirs['solve_seconds']:.3f} s | not compared |")
        print()
        print(f"latent norm: ours {ours['latent_norm']:.3f} against {theirs['latent_norm']:.3f}")
        if ours["noise_digest"] == theirs["noise_digest"]:
            print(f"latent digest: {'identical' if ours['latent_digest'] == theirs['latent_digest'] else 'differs (BF16 rounding, expected)'}")
    else:
        print("| Quantity | hipEngine | torch reference | Reading |")
        print("| --- | ---: | ---: | --- |")
        our_warm = _warm(ours, "tiled_seconds", "repeat_tiled_seconds")
        their_warm = _warm(theirs, "tiled_seconds", "repeat_tiled_seconds")
        reading = ("reference %.2fx faster" % (our_warm / their_warm) if their_warm < our_warm
                   else "hipEngine %.2fx faster" % (their_warm / our_warm))
        print(f"| warm tiled decode (1024/16) | {our_warm:.3f} s | {their_warm:.3f} s | {reading} |")
        print(f"| first tiled decode (cold) | {ours['tiled_seconds']:.3f} s | "
              f"{theirs['tiled_seconds']:.3f} s | not compared |")
        frames = ours["frames"]
        print(f"| per frame, warm | {our_warm / frames * 1000:.2f} ms | "
              f"{their_warm / frames * 1000:.2f} ms | {reading} |")
        print()
        print(f"frames {ours['frames']}, samples {ours['samples']} (both sides), "
              f"peak {ours['peak']:.4f}; ours tiled vs full max abs "
              f"{ours.get('tiled_vs_full_max_abs')}, theirs "
              f"{theirs.get('tiled_vs_full_max_abs')}")
    print()
    print(f"torch imported on the hipEngine side: {ours.get('torch_imported')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["nar", "vae"], default="nar")
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--oracle", default=str(REPO / "artifacts" / "yue2" / "oracle" / "cases"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--noise", default="",
                        help="reference whole-song noise .npy, for a matched NAR comparison")
    parser.add_argument("--json", default="")
    parser.add_argument("--merge", nargs=3, default=None,
                        metavar=("OUT", "HIPENGINE_JSON", "REFERENCE_JSON"))
    parser.add_argument("--compare", nargs=2, default=None)
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare)
    if args.merge:
        out, hip_path, ref_path = args.merge
        hip = json.loads(Path(hip_path).read_text())
        ref = json.loads(Path(ref_path).read_text())
        for data in (hip, ref):
            data.pop("latents", None)
            data.pop("tiled", None)
        artifact = {"protocol": hip["protocol"], "stage": hip["stage"], "case": hip["case"],
                    "steps": hip["steps"], "hipengine": hip, "reference": ref}
        if hip["stage"] == "nar":
            our_warm = _warm(hip, "solve_seconds", "repeat_solve_seconds")
            their_warm = _warm(ref, "solve_seconds", "repeat_solve_seconds")
            artifact["comparison"] = {
                "hipengine_warm_solve_seconds": our_warm,
                "reference_warm_solve_seconds": their_warm,
                "reference_advantage": our_warm / their_warm,
                "hipengine_first_solve_seconds": hip["solve_seconds"],
                "reference_first_solve_seconds": ref["solve_seconds"],
                "noise_digests_match": hip["noise_digest"] == ref["noise_digest"],
                "latent_norm": {"hipengine": hip["latent_norm"], "reference": ref["latent_norm"]},
                "note": "Identical prefix, codes and seed, and the reference's own whole-song "
                        "noise is handed to the torch-free solver, so both sides start from the "
                        "same draw (digests compared). Warm passes are the comparison; the first "
                        "pass is kept for reference.",
            }
        else:
            our_warm = _warm(hip, "tiled_seconds", "repeat_tiled_seconds")
            their_warm = _warm(ref, "tiled_seconds", "repeat_tiled_seconds")
            artifact["comparison"] = {
                "hipengine_warm_tiled_seconds": our_warm,
                "reference_warm_tiled_seconds": their_warm,
                "reference_advantage": our_warm / their_warm,
                "hipengine_first_tiled_seconds": hip["tiled_seconds"],
                "reference_first_tiled_seconds": ref["tiled_seconds"],
                "frames": hip["frames"],
                "samples": hip["samples"],
                "tiled_vs_full_max_abs": {"hipengine": hip.get("tiled_vs_full_max_abs"),
                                          "reference": ref.get("tiled_vs_full_max_abs")},
                "note": "Identical latents (the case's recorded latent.npy) and the same tiling "
                        "(core 1024 / halo 16), so both sides do the same work. The reference's "
                        "first decode in a process costs 8x its steady state on this device, so "
                        "the warm pass is the comparison and the first pass is kept for "
                        "reference; the recorded product numbers in the campaign artifacts "
                        "carry that first-use cost plus a per-request decoder transfer.",
            }
        Path(out).write_text(json.dumps(artifact, indent=1))
        print(f"wrote {out}")
        return 0

    case_data = load_case(Path(args.oracle), args.case)
    result = run_nar(case_data, repeats=args.repeats, noise_path=args.noise) if args.stage == "nar" else \
        run_vae(case_data, repeats=args.repeats)
    if args.stage == "nar":
        result.pop("latents", None)
    if args.stage == "vae":
        result.pop("tiled", None)
    try:
        host = {"hostname": Path("/etc/hostname").read_text().strip()}
    except OSError:
        host = {}
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60, check=False).stdout
        for line in out.splitlines():
            if line.strip().startswith("Name:") and "gfx" in line:
                host["gpu"] = line.split(":", 1)[1].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    payload = {
        "protocol": PROTOCOL,
        "side": "hipengine",
        "stage": args.stage,
        "case": case_data["case"],
        "steps": case_data["steps"],
        "prefix": case_data["prefix"],
        "semantic": case_data["semantic"],
        "prefix_digest": case_data["prefix_digest"],
        "semantic_digest": case_data["semantic_digest"],
        "latent_digest": case_data["latent_digest"],
        "host": host,
        "torch_imported": "torch" in sys.modules,
        "revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                   text=True, check=False).stdout.strip(),
        **result,
    }
    if args.stage == "nar":
        print(f"nar {payload['case']}: {payload['steps']} steps, "
              f"solve {payload['solve_seconds']:.3f} s "
              f"({payload['ms_per_step']:.2f} ms per step), "
              f"latent norm {payload['latent_norm']:.3f}")
        if payload["repeat_solve_seconds"]:
            print(f"repeat solve: {['%.3f' % v for v in payload['repeat_solve_seconds']]}")
    else:
        print(f"vae {payload['case']}: {payload['frames']} frames, "
              f"tiled {payload['tiled_seconds']:.3f} s "
              f"({payload['tiled_ms_per_frame']:.2f} ms per frame), "
              f"full {payload['full_seconds'] and '%.3f s' % payload['full_seconds'] or payload['full_note']}")
        if payload["repeat_tiled_seconds"]:
            print(f"repeat tiled: {['%.3f' % v for v in payload['repeat_tiled_seconds']]}")
    print(f"torch imported: {payload['torch_imported']}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
