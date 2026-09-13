#!/usr/bin/env python3
"""Report per-kernel VGPR/SGPR/LDS and the gfx11 occupancy they imply.

Compiles a HIP source with ``hipcc -save-temps`` and reads the device
assembly's ``.amdhsa_kernel`` metadata, which is the same metadata the runtime
profiler reports: allocated VGPR and SGPR counts, the LDS (group segment) byte
footprint, and the private-segment footprint.

Occupancy model (gfx11, wave32; one workgroup's waves spread over the CU's
SIMDs):

* ``waves_per_simd_vgpr`` = ``min(8, 512 // round_up_8(next_free_vgpr))``
* ``workgroups_per_cu_lds`` = ``floor(65536 / group_segment_fixed_size)``
* ``waves_per_cu`` = ``min(32, waves_per_simd_vgpr * 4,
  workgroups_per_cu_lds * waves_per_workgroup)``

LDS bounds resident workgroups, VGPR bounds resident waves per SIMD. Both must
clear the next step for occupancy to move, so a tile change that only lowers
LDS does not by itself raise resident waves.

Examples:
    python3 scripts/gguf_prefill_kernel_resources.py \
        --source hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.hip \
        --arch gfx1151 --waves-per-workgroup 4 --filter dense_dual_wmma \
        --json /tmp/resources.json

    python3 scripts/gguf_prefill_kernel_resources.py --asm /tmp/temps/*.s \
        --filter shared_b --waves-per-workgroup 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL_RE = re.compile(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", re.S)

META_KEYS = (
    "next_free_vgpr",
    "next_free_sgpr",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "kernarg_size",
    "sgpr_count",
    "vgpr_count",
    "max_flat_workgroup_size",
    "wavefront_size32",
)


def parse_metadata(block: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in META_KEYS:
        match = re.search(rf"\.amdhsa_{key}\s+(\S+)", block)
        if match:
            meta[key] = match.group(1)
    return meta


def round_up_8(value: int) -> int:
    return (value + 7) // 8 * 8


def derived_occupancy(
    vgpr: int, lds_bytes: int, waves_per_workgroup: int
) -> dict[str, int]:
    waves_per_simd = min(8, 512 // round_up_8(vgpr)) if vgpr else 0
    workgroups_per_cu = max(1, 65536 // lds_bytes) if lds_bytes else 32
    waves_per_cu = min(
        32, waves_per_simd * 4, workgroups_per_cu * waves_per_workgroup
    )
    return {
        "waves_per_simd_vgpr": waves_per_simd,
        "workgroups_per_cu_lds": workgroups_per_cu,
        "waves_per_cu": waves_per_cu,
    }


def kernel_rows(
    asm_text: str, *, waves_per_workgroup: int, name_filter: str | None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, block in KERNEL_RE.findall(asm_text):
        if name_filter and name_filter not in name:
            continue
        meta = parse_metadata(block)
        vgpr = int(meta.get("vgpr_count") or meta.get("next_free_vgpr", "0"))
        sgpr = int(meta.get("sgpr_count") or meta.get("next_free_sgpr", "0"))
        lds_bytes = int(meta.get("group_segment_fixed_size", "0"))
        row: dict[str, object] = {
            "kernel": name,
            "vgpr": vgpr,
            "vgpr_allocated": round_up_8(vgpr),
            "sgpr": sgpr,
            "lds_bytes": lds_bytes,
            "private_bytes": int(meta.get("private_segment_fixed_size", "0")),
            "max_flat_workgroup_size": int(
                meta.get("max_flat_workgroup_size", "0")
            ),
        }
        row.update(derived_occupancy(vgpr, lds_bytes, waves_per_workgroup))
        rows.append(row)
    rows.sort(key=lambda row: str(row["kernel"]))
    return rows


def compile_device_asm(
    source: Path, arch: str, temp_dir: Path, extra_flags: list[str]
) -> Path:
    """Compile ``source`` with ``-save-temps`` and return the device .s path."""

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise SystemExit("hipcc is not on PATH; activate the ROCm environment")
    command = [
        hipcc,
        "-save-temps",
        "-O3",
        f"--offload-arch={arch}",
        *extra_flags,
        "-c",
        str(source),
        "-o",
        str(temp_dir / "resources.o"),
    ]
    print("+ " + " ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=temp_dir, check=True)
    candidates = sorted(temp_dir.glob("*-amdgcn-amd-amdhsa-*.s"))
    if not candidates:
        raise SystemExit(f"no device assembly produced in {temp_dir}")
    return candidates[-1]


def compiler_version() -> str:
    hipcc = shutil.which("hipcc") or "hipcc"
    try:
        return subprocess.check_output(
            (hipcc, "--version"), text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="HIP source to compile")
    source.add_argument("--asm", type=Path, help="existing device .s file")
    parser.add_argument("--arch", default="gfx1151")
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="compile directory (default: a fresh temporary directory)",
    )
    parser.add_argument(
        "--extra-flag",
        action="append",
        default=None,
        help="extra hipcc flag (default: -mcumode plus the prefill unroll "
        "threshold used by hipengine.core.build)",
    )
    parser.add_argument("--filter", default=None, help="kernel name substring")
    parser.add_argument(
        "--waves-per-workgroup",
        type=int,
        default=4,
        help="waves the owning launcher puts in one workgroup",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    extra_flags = args.extra_flag or [
        "-mcumode",
        "-mllvm",
        "-amdgpu-unroll-threshold-local=600",
    ]
    provenance: dict[str, object] = {
        "arch": args.arch,
        "waves_per_workgroup": args.waves_per_workgroup,
        "compiler_version": compiler_version(),
    }
    temp_root: tempfile.TemporaryDirectory[str] | None = None
    if args.source is not None:
        temp_root = (
            None
            if args.temp_dir is not None
            else tempfile.TemporaryDirectory(prefix="kernel-resources-")
        )
        directory = Path(args.temp_dir or temp_root.name)
        directory.mkdir(parents=True, exist_ok=True)
        asm_path = compile_device_asm(
            args.source.resolve(), args.arch, directory, extra_flags
        )
        provenance["source"] = str(args.source.resolve())
        provenance["source_sha256"] = hashlib.sha256(
            args.source.read_bytes()
        ).hexdigest()
        provenance["hipcc_flags"] = extra_flags
    else:
        asm_path = args.asm
    provenance["device_asm"] = str(asm_path)

    rows = kernel_rows(
        asm_path.read_text(errors="replace"),
        waves_per_workgroup=args.waves_per_workgroup,
        name_filter=args.filter,
    )
    payload = {"provenance": provenance, "kernels": rows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json} ({len(rows)} kernels)")

    for row in rows:
        print(
            f"{str(row['kernel'])[:96]:96s} vgpr={row['vgpr']:4d}"
            f" sgpr={row['sgpr']:4d} lds={row['lds_bytes']:6d}"
            f" private={row['private_bytes']:4d}"
            f" wg/cu={row['workgroups_per_cu_lds']}"
            f" waves/simd={row['waves_per_simd_vgpr']}"
            f" waves/cu={row['waves_per_cu']}"
        )
    if temp_root is not None:
        temp_root.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
