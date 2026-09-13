"""Regression contracts for the September prefill review repairs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import gguf_prefill_kernel_resources as resources
from scripts import qwen38_gfx1151_prefill_kernel_profile as profile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"


def load_assembler(directory):
    path = RESULTS / directory / "assemble.py"
    spec = importlib.util.spec_from_file_location(directory.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("vgpr,allocated,waves", [(248, 264, 5), (224, 240, 6), (96, 96, 16)])
def test_gfx1151_register_ceiling_is_not_measured_occupancy(vgpr, allocated, waves):
    result = resources.derived_occupancy(vgpr, 32768, 4)
    assert result["vgpr_allocated"] == allocated
    assert result["waves_per_simd_vgpr"] == waves
    assert result["waves_per_cu"] is None
    assert result["workgroups_per_cu_lds"] is None


def test_resource_model_rejects_unknown_arch_and_impossible_lds():
    with pytest.raises(ValueError):
        resources.derived_occupancy(64, 131072, 4)
    with pytest.raises(ValueError):
        resources.derived_occupancy(64, 1024, 4, arch="gfx9999")


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_profile_requires_nonempty_finite_logits(values):
    class Session:
        def _read_sample(self, *, return_logits):
            assert return_logits
            return type("Sample", (), {"token_id": 7, "logits": np.array(values)})()

    with pytest.raises(FloatingPointError):
        profile.validate_profile_logits(Session(), 7)


def test_profile_accepts_finite_zero_and_checks_sample_identity():
    class Session:
        def _read_sample(self, *, return_logits):
            return type("Sample", (), {"token_id": 7, "logits": np.array([0., 1.])})()

    assert profile.validate_profile_logits(Session(), 7)
    with pytest.raises(ValueError):
        profile.validate_profile_logits(Session(), 8)


def test_q6_assembler_rejects_failed_or_empty_gate(monkeypatch):
    module = load_assembler("2026-09-13-q6-planar-prefill-large-row-screen")
    original = module.load

    def corrupted(name):
        data = original(name)
        if name == "raw/prefill-after-fca92ac-128decode.json":
            data["graph_eager_gate"][0]["passed"] = False
        return data

    monkeypatch.setattr(module, "load", corrupted)
    with pytest.raises(ValueError, match="gate"):
        module.build()
    with pytest.raises(ValueError, match="gate"):
        module.gate_summary({"graph_eager_gate": []})


def test_q6_assembler_rejects_nonexact_screen(monkeypatch):
    module = load_assembler("2026-09-13-q6-planar-prefill-large-row-screen")
    original = module.load

    def corrupted(name):
        data = original(name)
        if name == "screen-bands.json":
            data["cases"][0]["rows"][0]["shared4r4_bit_equal"] = False
        return data

    monkeypatch.setattr(module, "load", corrupted)
    with pytest.raises(ValueError, match="screen"):
        module.build()


def test_rejected_assembler_freezes_measurement_provenance():
    name = "2026-09-13-q4-dual-prefill-col16-arm-ab-rejected"
    module = load_assembler(name)
    path = RESULTS / name
    stored = json.loads((path / "artifact.json").read_text())
    rebuilt = module.build(path, Path(stored["model"]), generated_at=stored["generated_at"])
    assert rebuilt["cells"] == stored["cells"]
    assert rebuilt["provenance"]["git_head_at_measurement"] == (
        "4620b9cf5712f99c1e03dee9cbf92ab26e8e8cc8"
    )
    assert rebuilt["provenance"]["model_hash_source"] == "recorded_measurement"
    assert rebuilt == stored


def test_rejected_geometry_distinguishes_token_and_column_axes():
    module = load_assembler("2026-09-13-q4-dual-prefill-col16-arm-ab-rejected")
    assert module.launch_geometry(512, columns=32, tile_rows=256)["blocks"] == 1088
    assert module.launch_geometry(512, columns=16, tile_rows=512)["blocks"] == 1088
    assert module.launch_geometry(512, columns=16, tile_rows=256)["blocks"] == 2176


def test_corrected_tile_screen_reproduces_and_does_not_reuse_old_occupancy():
    name = "2026-09-13-q4-dual-prefill-tile-knob-screen"
    module = load_assembler(name)
    path = RESULTS / name
    result = module.build(
        path / "resource_table.json", path / "rowtile_prefill.json", path / "col_tile_owners.json",
    )
    assert result == json.loads((path / "artifact.json").read_text())
    assert all(row["waves_per_cu"] is None for row in result["dual_ladder_resources"])
    assert all(row["launched_waves_per_block"] == 4 for row in result["dual_ladder_resources"])


def test_historical_profile_does_not_certify_unread_logits():
    artifact = json.loads((RESULTS / "2026-09-13-qwen38-gfx1151-prefill-kernel-profile/artifact.json").read_text())
    assert artifact["correctness"]["logits_finite"] == dict.fromkeys(("512", "1024", "4096"))


def test_resource_parser_requires_wave32_metadata():
    asm = """
.amdhsa_kernel fixture
  .amdhsa_next_free_vgpr 248
  .amdhsa_next_free_sgpr 32
  .amdhsa_group_segment_fixed_size 32768
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
"""
    rows = resources.kernel_rows(asm, waves_per_workgroup=4, name_filter=None)
    assert rows[0]["vgpr_allocated"] == 264
    with pytest.raises(ValueError, match="wave32"):
        resources.kernel_rows(
            asm.replace("size32 1", "size32 0"), waves_per_workgroup=4, name_filter=None,
        )
