#!/usr/bin/env python3
"""YuE2 VAE gate: native FP32 decoder vs the recorded reference fixture.

``tests/fixtures/yue2/vae/decode.npz`` holds a small latent recorded from the
pinned upstream ``yue2`` package and the waveform its FP32 Oobleck decoder
produced. When the full oracle artifact is present
(``artifacts/yue2/oracle/vae/decode.npz``) the gate also checks the 64-frame case
against both the reference's full decode and its own tiled decode, and asserts
that the native tiled decode is bit-identical to the native full decode: with
sufficient halo, tiling must not change the waveform.

The comparison is absolute and relative to the waveform scale. FP32 rounding is
expected to differ from torch by a few ulp per stage - the reference's own tiled
and full decodes already differ by ~1.3e-6 - so the bound is on the recorded
scale rather than bit equality.

Usage:
    python3 scripts/yue2_vae_gate.py [--json OUT] [--max-abs 1e-4]
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

from hipengine.loading.yue2 import load_yue2_vae_decoder  # noqa: E402
from hipengine.runtime.yue2_vae import Yue2VaeRuntime  # noqa: E402

FIXTURE = REPO / "tests/fixtures/yue2/vae/decode.npz"
FULL_FIXTURE = REPO / "artifacts/yue2/oracle/vae/decode.npz"
PROTOCOL = "yue2-vae-fixture-gate-v1"
CACHE = Path.home() / ".cache/huggingface/hub"


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


def _vae_dir() -> Path:
    for directory in sorted(CACHE.glob("models--m-a-p--YuE2-Vae/snapshots/*")):
        if (directory / "model.safetensors").is_file():
            return directory
    raise SystemExit("YuE2-Vae checkpoint not found in the local cache")


def _compare(reference: np.ndarray, produced: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    produced = np.asarray(produced, dtype=np.float64)
    if reference.shape != produced.shape:
        raise SystemExit(f"shape mismatch: reference {reference.shape} vs produced {produced.shape}")
    delta = produced - reference
    reference_norm = float(np.linalg.norm(reference))
    return {
        "max_abs": float(np.abs(delta).max()),
        "rms": float(np.sqrt((delta**2).mean())),
        "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else 0.0,
        "reference_peak": float(np.abs(reference).max()),
        "reference_rms": float(np.sqrt((reference**2).mean())),
        "produced_peak": float(np.abs(produced).max()),
        "produced_rms": float(np.sqrt((produced**2).mean())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--full-fixture", default=str(FULL_FIXTURE))
    parser.add_argument("--max-abs", type=float, default=1e-4)
    parser.add_argument("--core-frames", type=int, default=16)
    parser.add_argument("--halo-frames", type=int, default=16)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    arrays = np.load(fixture)
    manifest_path = fixture.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    decoder = load_yue2_vae_decoder(_vae_dir())
    runtime = Yue2VaeRuntime(decoder)
    report: dict = {
        "provenance": {
            "command_line": " ".join(sys.argv),
            "host": _host_identity(),
            "revision": _revision(),
            "fixture": str(fixture),
            "protocol": PROTOCOL,
        },
        "decoder": {
            "latent_dim": decoder.latent_dim,
            "downsampling_ratio": decoder.downsampling_ratio,
            "blocks": len(decoder.blocks),
            "bytes": decoder.bytes,
            "required_halo_core16": decoder.required_halo(16),
        },
        "thresholds": {"max_abs": args.max_abs},
        "cases": [],
    }

    def run_case(name: str, latent: np.ndarray, reference_full: np.ndarray,
                 reference_tiled: np.ndarray | None, *, tiled: bool) -> dict:
        started = time.perf_counter()
        produced_full = runtime.decode(latent)
        full_seconds = time.perf_counter() - started
        entry = {
            "case": name,
            "frames": int(latent.shape[-1]),
            "natural_length": runtime.natural_output_length(int(latent.shape[-1])),
            "full_seconds": full_seconds,
            "full": _compare(reference_full, produced_full),
        }
        if tiled:
            core = int(args.core_frames)
            halo = int(args.halo_frames)
            started = time.perf_counter()
            produced_tiled = runtime.decode_tiled(
                latent, core_frames=core, halo_frames=halo
            )
            entry["tiled_seconds"] = time.perf_counter() - started
            entry["tiled"] = {
                "seconds": entry["tiled_seconds"],
                "core_frames": core,
                "halo_frames": halo,
                "required_halo": runtime.required_halo(core),
                "vs_full_bit_exact": bool(np.array_equal(produced_tiled, produced_full)),
                "vs_reference": _compare(
                    reference_tiled if reference_tiled is not None else reference_full,
                    produced_tiled,
                ),
            }
        entry["passed"] = bool(
            entry["full"]["max_abs"] <= args.max_abs
            and (not tiled or entry["tiled"]["vs_full_bit_exact"])
            and (not tiled or entry["tiled"]["vs_reference"]["max_abs"] <= args.max_abs)
        )
        report["cases"].append(entry)
        return entry

    small = run_case(
        "recorded small latent",
        np.asarray(arrays["latent"], dtype=np.float32),
        arrays["full"],
        None,
        tiled=False,
    )
    print(
        f"[vae-gate] small: frames={small['frames']} length={small['natural_length']} "
        f"max_abs={small['full']['max_abs']:.3e} rel_l2={small['full']['relative_l2']:.3e} "
        f"peak={small['full']['produced_peak']:.6f} (reference {small['full']['reference_peak']:.6f}) "
        f"{small['full_seconds']:.2f}s"
    )

    full_path = Path(args.full_fixture)
    if full_path.is_file():
        full = np.load(full_path)
        large = run_case(
            "recorded 64-frame latent",
            np.asarray(full["latent"], dtype=np.float32),
            full["full"],
            full["tiled"],
            tiled=True,
        )
        print(
            f"[vae-gate] 64 frames: length={large['natural_length']} "
            f"max_abs={large['full']['max_abs']:.3e} rel_l2={large['full']['relative_l2']:.3e} "
            f"{large['full_seconds']:.2f}s"
        )
        tiled = large["tiled"]
        print(
            f"[vae-gate] tiled core={tiled['core_frames']} halo={tiled['halo_frames']} "
            f"(required {tiled['required_halo']}): bit-exact vs full="
            f"{tiled['vs_full_bit_exact']} max_abs_vs_reference="
            f"{tiled['vs_reference']['max_abs']:.3e} {tiled['seconds']:.2f}s"
        )
    else:
        print(f"[vae-gate] full oracle artifact absent ({full_path}); small case only")
        report["full_fixture_present"] = False

    report["passed"] = all(entry["passed"] for entry in report["cases"])
    report["worst_max_abs"] = max(entry["full"]["max_abs"] for entry in report["cases"])
    print(
        f"[vae-gate] {len(report['cases'])} case(s): worst max_abs={report['worst_max_abs']:.3e} "
        f"-> {'PASS' if report['passed'] else 'FAIL'}"
    )
    runtime.close()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[vae-gate] wrote {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
