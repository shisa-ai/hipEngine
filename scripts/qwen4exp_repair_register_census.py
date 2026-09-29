"""Static register/occupancy/instruction census for the IU8 repair kernels.

Ledger ``EG-3`` leaves the repair parked until "a measured mechanism (registers,
instruction count, occupancy)" reopens it. This builds the kernel family with
``--save-temps`` and reports, per kernel:

* VGPR/SGPR counts from the ``.amdhsa_next_free_vgpr`` / ``next_free_sgpr``
  assembler directives;
* occupancy from the gfx11 register file (65536 VGPR per CU, 32-wave ceiling);
* static instruction count and opcode mix from the device-code disassembly.

Run::

    python3 scripts/qwen4exp_repair_register_census.py --output <path.json>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# gfx11 (RDNA3/3.5): 65536 VGPR per CU, register granules of 8, 32 waves/CU max,
# 8 waves/SIMD max across 4 SIMDs.
VGPR_PER_CU = 65536
MAX_WAVES_PER_CU = 32
WAVE_WIDTH = 32
GRANULE = 8

DEFAULT_SOURCE = Path(
    "hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_selected_prefill.hip"
)
FLAGS = ("-O3", "-mllvm", "-amdgpu-unroll-threshold-local=600",
         "--offload-arch=gfx1151", "-mcumode")


def _occupancy(vgpr: int, threads_per_block: int) -> dict[str, object]:
    granule = ((vgpr + GRANULE - 1) // GRANULE) * GRANULE
    waves_by_vgpr = VGPR_PER_CU // (granule * WAVE_WIDTH)
    waves = min(waves_by_vgpr, MAX_WAVES_PER_CU)
    return {
        "vgpr": vgpr,
        "vgpr_granule": granule,
        "waves_per_cu_by_vgpr": waves_by_vgpr,
        "waves_per_cu": waves,
        "waves_per_simd": waves // 4,
        "occupancy_fraction": waves / MAX_WAVES_PER_CU,
        "threads_per_cu": waves * WAVE_WIDTH,
        "blocks_per_cu": (waves * WAVE_WIDTH) // threads_per_block,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--threads-per-block", type=int, default=128)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="repair-census-"))
    workdir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    subprocess.run(
        ["hipcc", "-c", "--save-temps", *FLAGS, str(source),
         "-o", str(workdir / "out.o")],
        check=True, cwd=workdir,
    )

    asm = next(workdir.glob("*-amdgcn-amd-amdhsa-gfx1151.s"))
    devobj = next(p for p in workdir.glob("*-amdgcn-amd-amdhsa-gfx1151.o"))
    llvm = Path(shutil.which("llvm-objdump") or "llvm-objdump")
    dis = subprocess.run(
        [str(llvm), "-d", "--mcpu=gfx1151", str(devobj)],
        capture_output=True, text=True, check=True,
    ).stdout

    text = asm.read_text()
    kernels: list[dict[str, object]] = []
    hdr = re.compile(r"^[0-9a-f]+ <(.+)>:$")
    ins = re.compile(r"^\s+([A-Za-z_][\w.]*)\s.*?//\s*[0-9a-f]+:\s")
    cur = None
    counts: dict[str, int] = {}
    ops: dict[str, dict[str, int]] = {}
    for line in dis.splitlines():
        m = hdr.match(line.strip())
        if m:
            cur = m.group(1)
            counts[cur] = 0
            ops[cur] = {}
            continue
        m = ins.match(line)
        if cur and m:
            counts[cur] += 1
            ops[cur][m.group(1)] = ops[cur].get(m.group(1), 0) + 1

    for block in re.split(r"\n\t\.type\s+", text):
        m = re.search(r"^(\S+?),@function", block)
        if not m:
            continue
        mangled = m.group(1)
        vg = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", block)
        sg = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", block)
        match = next((k for k in counts if mangled.endswith(k) or k == mangled), None)
        entry = {
            "symbol": mangled,
            "vgpr": int(vg.group(1)) if vg else None,
            "sgpr": int(sg.group(1)) if sg else None,
            "static_instructions": counts.get(match, 0) if match else 0,
            "top_opcodes": sorted(
                (ops.get(match, {})).items(), key=lambda kv: -kv[1]
            )[:12],
        }
        if entry["vgpr"] is not None:
            entry["occupancy"] = _occupancy(entry["vgpr"], args.threads_per_block)
        kernels.append(entry)

    artifact = {
        "schema": 1,
        "kind": "qwen4exp_repair_register_census",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "boundary": (
            "static census of one compiled profile; occupancy is the register "
            "limit only (no LDS/scratch), not a runtime measurement"
        ),
        "source": str(source),
        "flags": list(FLAGS),
        "threads_per_block": args.threads_per_block,
        "model": {
            "vgpr_per_cu": VGPR_PER_CU,
            "max_waves_per_cu": MAX_WAVES_PER_CU,
            "wave_width": WAVE_WIDTH,
            "vgpr_granule": GRANULE,
        },
        "kernels": kernels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    for k in kernels:
        name = re.sub(r"_ZN12_GLOBAL__N_1\d+", "", k["symbol"])[:52]
        occ = k.get("occupancy") or {}
        print(
            f"{name:<54} VGPR={str(k['vgpr']):>4} SGPR={str(k['sgpr']):>4} "
            f"ins={k['static_instructions']:>4} "
            f"waves/CU={occ.get('waves_per_cu','?')} "
            f"occ={occ.get('occupancy_fraction',0)*100:>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
