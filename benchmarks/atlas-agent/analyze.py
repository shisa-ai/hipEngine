#!/usr/bin/env python3
"""Build the hipEngine vs atlas comparison artifact from a run directory.

Reads the per-engine arm JSON written by ``http_1to1_bench.py`` and emits one
artifact that states, for every number, which engine produced it, on which host,
with which quant and which speculation depth -- plus the caveats that make the
comparison interpretable. Refuses to invent a number: an arm with no data is
reported as missing rather than as zero.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CAVEATS = [
    "CROSS-QUANT: atlas runs nvidia/Qwen3.8-27B-NVFP4 (NVFP4, W4A8 DP4A decode arm) and "
    "hipEngine runs Qwen3.8-27B Q4_K_M GGUF. Atlas carries no k-quant kernels and hipEngine "
    "has no NVFP4 execution path, so an identical quant is unavailable on either side. These "
    "are not same-quant rates.",
    "SPECULATION DEPTH DIFFERS: atlas runs K=4 MTP (NUM_DRAFTS=4), hipEngine runs K=3, which "
    "is the deepest candidate budget its serving evidence qualifies.",
    "LONG-CONTEXT MTP: hipEngine's dense MTP adapter admits only inside a 1,023-token window, "
    "so above it rows decode autoregressively by design. At the context lengths measured here "
    "the hipEngine arm is therefore AR unless FORCE_LONG_MTP=1 was set, and that forced arm is "
    "an unqualified diagnostic: raising the window alone measures 0.57x because the target and "
    "draft graphs decline into their eager paths per cycle (docs/REFACTOR.md).",
    "Both engines were measured on this one host in this one run. No number here is taken from "
    "either project's published results, and none may be compared against them as if same-host.",
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def arm_view(report: dict | None, arm: str) -> dict | None:
    """Extract the comparable summary of one arm, or None when absent."""
    if not report:
        return None
    arms = report.get("arms") or {}
    if arm not in arms:
        return None
    data = arms[arm]
    if arm == "single":
        return {
            "requests": data.get("requests"),
            "errors": data.get("errors"),
            "reached_target": data.get("reached_target"),
            "ttft_ms": data.get("ttft_ms"),
            "itl_ms": data.get("itl_ms"),
            "decode_tok_s": data.get("decode_tok_s"),
            "e2e_s": data.get("e2e_s"),
        }
    if arm == "multi":
        return {
            "requests": data.get("requests"),
            "errors": data.get("errors"),
            "turn0_ttft_ms": data.get("turn0_ttft_ms"),
            "later_turn_ttft_ms": data.get("later_turn_ttft_ms"),
            "turn0_decode_tok_s": data.get("turn0_decode_tok_s"),
            "later_turn_decode_tok_s": data.get("later_turn_decode_tok_s"),
        }
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--engines", default="hipengine,atlas")
    args = parser.parse_args()

    run_dir = args.run_dir
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    env_file = run_dir / "environment.txt"
    environment = env_file.read_text().strip().splitlines() if env_file.exists() else []

    artifact: dict[str, object] = {
        "schema": 1,
        "kind": "hipengine_vs_atlas_1to1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "run_dir": str(run_dir),
        "environment": environment,
        "engines": {},
        "caveats": CAVEATS,
        "gpu": {},
    }
    try:
        artifact["gpu"]["rocm"] = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=60
        ).stdout.splitlines()[:6]
    except Exception:  # noqa: BLE001 - provenance is best-effort
        artifact["gpu"]["rocm"] = []

    for engine in engines:
        entry: dict[str, object] = {"arms": {}}
        cap = load(run_dir / f"{engine}-capabilities.json")
        if cap:
            entry["capabilities"] = cap
        for arm in ("single", "multi", "conc"):
            view = arm_view(load(run_dir / f"{engine}-{arm}.json"), arm)
            entry["arms"][arm] = view if view is not None else {"missing": True}
        artifact["engines"][engine] = entry

    # A side-by-side of the headline decode rate, only where both sides exist.
    comparison: dict[str, object] = {}
    for arm, key in (("single", "decode_tok_s"), ("multi", "later_turn_decode_tok_s")):
        row: dict[str, object] = {}
        for engine in engines:
            data = (artifact["engines"][engine]["arms"] or {}).get(arm)  # type: ignore[index]
            value = None
            if isinstance(data, dict) and isinstance(data.get(key), dict):
                value = data[key].get("median")
            row[engine] = value
        if all(isinstance(v, (int, float)) for v in row.values()) and row:
            hip, atl = row.get("hipengine"), row.get("atlas")
            if isinstance(hip, (int, float)) and isinstance(atl, (int, float)) and hip:
                row["atlas_over_hipengine"] = atl / hip
        comparison[arm] = row
    artifact["comparison_median"] = comparison

    out = args.json or (run_dir / "artifact.json")
    out.write_text(json.dumps(artifact, indent=1, default=str))
    print(f"# wrote {out}")
    for arm, row in comparison.items():
        print(f"# {arm}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
