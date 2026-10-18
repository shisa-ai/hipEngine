#!/usr/bin/env python3
"""A/B the GGUF WMMA prefill opt-in on Gemma 4 prefill, one shape, one sample.

Runs the campaign bench twice on the same device: once with the dispatch
defaults, once with ``HIPENGINE_GGUF_WMMA_PREFILL=1``. Decode rows are
unaffected by the opt-in, so a decode difference is noise and prefill is the
signal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "scripts" / "gemma4_campaign_bench.py"


def run(env_extra: dict[str, str], out: Path, prompt: int, output: int, samples: int, device: str):
    env = dict(os.environ)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["ROCR_VISIBLE_DEVICES"] = device
    env["PYTHONPATH"] = "."
    env.update(env_extra)
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(BENCH),
        "--prompt", str(prompt),
        "--output", str(output),
        "--samples", str(samples),
        "--expect-gpu", "W7900" if device == "0" else "RX 7900 XTX",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-3000:])
        raise SystemExit(f"bench failed rc={proc.returncode}")
    body = json.loads(out.read_text())
    stats = body["stats"]
    return {
        "prefill_tps": stats["prefill_tps"],
        "decode_tps": stats["decode_tps"],
        "samples": stats.get("samples"),
        "public_path_parity": body.get("public", {}).get("public_path_parity"),
    }


def main() -> int:
    prompt = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    output = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    device = sys.argv[4] if len(sys.argv) > 4 else "1"
    tmp = Path("/tmp")
    arms = {
        "default": {},
        "wmma_prefill": {"HIPENGINE_GGUF_WMMA_PREFILL": "1"},
    }
    results = {}
    for name, extra in arms.items():
        results[name] = run(
            extra, tmp / f"gemma4-prefill-ab-{name}.json", prompt, output, samples, device
        )
        print(name, json.dumps(results[name]), flush=True)
    base = results["default"]["prefill_tps"]
    new = results["wmma_prefill"]["prefill_tps"]
    print(f"prefill {base:.4f} -> {new:.4f} ({100.0 * (new / base - 1.0):+.2f}%)", flush=True)
    (tmp / "gemma4-prefill-ab.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
