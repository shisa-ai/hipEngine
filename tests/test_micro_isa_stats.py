from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "isa_stats.py"
    )
    spec = importlib.util.spec_from_file_location("micro_isa_stats", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_hip_metadata_and_disassembly_stats() -> None:
    module = _load_runner_module()
    notes = """
    .name:           _ZN12_GLOBAL__N_124f32_gemv_geometry_kernelEPKfS1_Pfjj
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .vgpr_count:     11
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
    """
    disasm = """
0000000000000000 <kernel>:
    s_load_b32 s3, s[0:1], 0x2c
    s_waitcnt lgkmcnt(0)
    v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
    global_load_b32 v1, v[9:10], off
    global_load_b32 v7, v[7:8], off
    s_waitcnt vmcnt(0)
    v_dual_fmac_f32 v3, v1, v7 :: v_dual_add_nc_u32 v6, s1, v6
    ds_store_b32 v1, v3
    s_barrier
    ds_load_b32 v2, v2
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v2, v2, v3
    global_store_b32 v0, v1, s[0:1]
    s_endpgm
    """

    metadata = module.parse_hip_metadata(notes)
    stats = module.parse_disassembly_stats(disasm)

    assert metadata["kernel_name"].startswith("_ZN12_GLOBAL")
    assert metadata["target"] == "amdgcn-amd-amdhsa--gfx1151"
    assert metadata["sgpr"] == 18
    assert metadata["vgpr"] == 11
    assert metadata["scratch_bytes"] == 0
    assert metadata["sgpr_spill_count"] == 0
    assert metadata["vgpr_spill_count"] == 0
    assert metadata["wave_size"] == 32
    assert stats["waitcnt_count"] == 3
    assert stats["vopd_count"] == 2
    assert stats["vopd_op_count"] == 4
    assert stats["global_load_count"] == 2
    assert stats["global_store_count"] == 1
    assert stats["ds_load_count"] == 1
    assert stats["ds_store_count"] == 1
    assert stats["barrier_count"] == 1
    assert stats["estimated_vgpr_span"] == 11
    assert stats["estimated_sgpr_span"] == 4


def test_parse_radv_shader_dump_sections() -> None:
    module = _load_runner_module()
    dump = """
shader: MESA_SHADER_COMPUTE
workgroup_size: 64, 1, 1
shared_size: 1024
api_subgroup_size: 64
max_subgroup_size: 64
min_subgroup_size: 64
After RA:
  v1: %39:v[8] = buffer_load_dword %36:s[8-11], %38:v[8], 0 offen
Compute Shader
disasm:
BB0:
    s_load_b256 s[8:15], s[6:7], null
    buffer_load_b32 v8, v8, s[8:11], 0 offen
    buffer_load_b32 v3, v3, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_fmac_f32_e32 v1, v8, v3
    ds_store_b32 v2, v1
    s_waitcnt_depctr 0xffe3
    s_barrier
    ds_load_2addr_b32 v[4:5], v2 offset1:32
    s_waitcnt lgkmcnt(0)
    buffer_store_b32 v0, off, s[0:3], s5
    s_endpgm
shader: MESA_SHADER_COMPUTE
workgroup_size: 256, 1, 1
shared_size: 1024
api_subgroup_size: 64
max_subgroup_size: 64
min_subgroup_size: 64
After RA:
Compute Shader
disasm:
BB0:
    v_mov_b32_e32 v1, 0
    s_endpgm
    """

    sections = module.parse_radv_shader_dump(dump)

    assert len(sections) == 2
    assert sections[0]["workgroup_size"] == 64
    assert sections[0]["shared_size"] == 1024
    assert sections[0]["api_subgroup_size"] == 64
    assert sections[0]["has_after_ra"] is True
    assert sections[0]["has_final_disasm"] is True
    assert sections[0]["buffer_load_count"] == 2
    assert sections[0]["buffer_store_count"] == 1
    assert sections[0]["waitcnt_count"] == 3
    assert sections[0]["waitcnt_depctr_count"] == 1
    assert sections[0]["vopd_count"] == 0
    assert sections[0]["fma_or_fmac_count"] == 1
    assert sections[0]["estimated_vgpr_span"] == 9
    assert sections[0]["estimated_sgpr_span"] == 16
    assert sections[1]["workgroup_size"] == 256
    json.dumps(sections, allow_nan=False)


def test_build_isa_comparison_matches_workgroups() -> None:
    module = _load_runner_module()
    hip = {
        "source": {"commit": "c" * 40},
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "k": 2048,
                    "rows": 1,
                    "body_repeats": 128,
                    "workgroup_size": 64,
                    "vgpr": 11,
                    "sgpr": 18,
                    "scratch_bytes": 0,
                    "wave_size": 32,
                    "instruction_count": 100,
                    "waitcnt_count": 5,
                    "vopd_count": 2,
                    "vopd_op_count": 4,
                    "dot4_count": 0,
                }
            ]
        },
    }
    vulkan = {
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "k": 2048,
                    "rows": 1,
                    "body_repeats": 128,
                    "workgroup_size": 64,
                    "estimated_vgpr_span": 9,
                    "estimated_sgpr_span": 16,
                    "wave_size": 64,
                    "shared_size": 1024,
                    "instruction_count": 120,
                    "waitcnt_count": 8,
                    "waitcnt_depctr_count": 7,
                    "vopd_count": 0,
                    "vopd_op_count": 0,
                    "dot4_count": 0,
                    "correctness_pass": True,
                }
            ]
        },
    }

    comparison = module.build_comparison(
        hip,
        vulkan,
        command=["python3", "isa_stats.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert len(comparison["matched_rows"]) == 1
    row = comparison["matched_rows"][0]
    assert row["workgroup_size"] == 64
    assert row["hip_vgpr"] == 11
    assert row["vulkan_official_register_counts"] is False
    assert row["vulkan_estimated_sgpr_span"] == 16
    json.dumps(comparison, allow_nan=False)
