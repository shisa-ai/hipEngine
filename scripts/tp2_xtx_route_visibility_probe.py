#!/usr/bin/env python3
"""Bounded XTX TP1-route visibility probe (no model weights, no kernels).

Answers one narrow question before any heavier diagnostic: when the XTX is
selected by ``HIP_VISIBLE_DEVICES`` or ``ROCR_VISIBLE_DEVICES``, what does HIP
report as logical device 0, and does it match the physical RX 7900 XTX
(PCI ``0000:10:00.0``, uuid ``cc4d02090dc9c3ff...``)?

Each environment combination runs in a fresh subprocess because HIP caches the
visible-device set at library load; setting the variables in-process after the
first load does not re-enumerate. No model session is built and no kernel is
launched, so this cannot fault the GPU.

Usage: python3 scripts/tp2_xtx_route_visibility_probe.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ENUM_SNIPPET = r"""
import json, sys
sys.path.insert(0, %r)
from hipengine.core.hip import HipRuntime
rt = HipRuntime.load()
out = {"count": rt.device_count(), "devices": []}
for i in range(out["count"]):
    info = rt.device_info(i)
    out["devices"].append(
        {"index": i, "name": info.name, "uuid": info.uuid, "uuid_hex": info.uuid_hex,
         "pci_bus_id": info.pci_bus_id}
    )
print(json.dumps(out))
""" % str(REPO_ROOT)

COMBINATIONS = (
    {},
    {"HIP_VISIBLE_DEVICES": "0"},
    {"HIP_VISIBLE_DEVICES": "1"},
    {"HIP_VISIBLE_DEVICES": "0,1"},
    {"ROCR_VISIBLE_DEVICES": "1"},
)


def enumerate_devices(env_overrides: dict[str, str]) -> dict[str, object]:
    env = dict(os.environ)
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(key, None)
    env.update(env_overrides)
    completed = subprocess.run(
        [sys.executable, "-c", _ENUM_SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        return {"error": f"rc={completed.returncode}", "stderr": completed.stderr.strip()[-2000:]}
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    results = []
    for overrides in COMBINATIONS:
        entry = {"env": overrides, "observed": enumerate_devices(overrides)}
        results.append(entry)
        label = overrides or {"(inherit)": "1"}
        observed = entry["observed"]
        if "error" in observed:
            print(f"{label}: ERROR {observed['error']}", flush=True)
        else:
            names = [f"{d['index']}:{d['name']}@{d['pci_bus_id']}" for d in observed["devices"]]
            print(f"{label}: count={observed['count']} {names}", flush=True)

    # A logical-0 mapping to the XTX under HIP_VISIBLE_DEVICES=1 is what the
    # optimized TP1 route needs; anything else is a remapping problem, not a
    # kernel problem.
    xgpu = next((r for r in results if r["env"] == {"HIP_VISIBLE_DEVICES": "1"}), None)
    mapping_ok = None
    if xgpu and "error" not in xgpu["observed"]:
        devices = xgpu["observed"]["devices"]
        mapping_ok = bool(devices) and "7900 XTX" in devices[0]["name"]
    artifact = {
        "kind": "tp2_xtx_route_visibility_probe",
        "combinations": results,
        "hip_visible_devices_1_logical0_is_xtx": mapping_ok,
        "note": (
            "visibility/remapping only; no model session and no kernel launched. "
            "An idle-healthy card does not certify a healthy fault context."
        ),
    }
    if args.json:
        args.json.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"HIP_VISIBLE_DEVICES=1 logical0 is XTX: {mapping_ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
