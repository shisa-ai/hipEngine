#!/usr/bin/env python3
"""Cached-only rocprof smoke for all RDNA3 compact-DMS HIP kernels.

Prebuild ``dms_compact.so`` outside rocprof, then run this child under
``rocprofv3 --kernel-trace``.  The child loads the exact cached DSO with
``require_cached=True``, binds that library into all registered wrappers,
and executes representative strict/production-shaped pytest nodes in-process.
No compiler subprocess can be launched from the profiled process.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.attention import dms_compact
from hipengine.kernels.registry import KernelKey, register, resolve

_KERNEL_KEYS = (
    KernelKey("hip_gfx1100", "dms_extract_decision", "bf16", "corrected_mask"),
    KernelKey(
        "hip_gfx1100",
        "dms_decision_source",
        "bf16",
        "external_linear_sidecar_v1",
    ),
    KernelKey(
        "hip_gfx1151",
        "dms_decision_source",
        "bf16",
        "external_linear_sidecar_v1",
    ),
    KernelKey("hip_gfx1100", "dms_streaming_pack", "bf16", "count_rank_scatter"),
    KernelKey("hip_gfx1100", "dms_append_decode", "bf16", "compact_append_evict"),
    KernelKey("hip_gfx1100", "dms_compact_attn_decode", "bf16", "grouped_gqa"),
)
_TEST_NODES = (
    "tests/test_dms_extract_decision_hip.py::test_dms_extract_decision_production_head_geometry_bit_exact",
    "tests/test_dms_external_linear_hip.py::test_external_linear_device_projector_matches_bf16_cpu_decisions_production_geometry",
    "tests/test_dms_streaming_pack_hip.py::test_dms_streaming_pack_long_prompt_multi_tile_bit_exact",
    "tests/test_dms_append_decode_hip.py::test_dms_append_decode_batched_rows_bit_exact",
    "tests/test_dms_compact_attn_decode_hip.py::test_dms_compact_attn_decode_kl_top1_gate_production_shape",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    version_path = args.compiler_version_file.expanduser().resolve()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(version_path)
    compiler_version = version_path.read_text(encoding="utf-8").strip()
    library = dms_compact.build_dms_compact(
        load=True,
        require_cached=True,
        compiler_version=compiler_version,
    )
    load_backend_kernel_package("hip_gfx1151")
    for key in _KERNEL_KEYS:
        wrapper = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        register(key, functools.partial(wrapper, library=library), replace=True)
    return int(pytest.main(["-q", *_TEST_NODES]))


if __name__ == "__main__":
    raise SystemExit(main())
