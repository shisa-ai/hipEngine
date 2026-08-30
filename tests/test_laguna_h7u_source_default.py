"""WPF-H7U stable parallel MoE compaction source-default RED contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.moe import (
    qwen35_moe_group_compact_active_parallel,
    qwen35_moe_group_compact_active_source_rows_parallel,
    register_qwen35_moe_group_scatter_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.runtime.laguna_moe import resolve_laguna_group_compact_mode

_ROOT = Path(__file__).parents[1]
_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-candidate.json"
)
_ARTIFACT_SHA256 = (
    "1555285431d0ae5ee2771a0840ccd4fa2eb101ff0513cf595ab97711d74c1caf"
)
_PACKAGE = _ROOT / "hipengine/kernels/hip_gfx1100/__init__.py"
_TARGET_PACKAGE_SHA256 = (
    "9cf784e41d9f77983373f60c543342cfb7659a736ce84cdc762bb2bc93ab6abf"
)
_POST_MERGE_PACKAGE_SHA256 = (
    "a7365e583064e581744760a4723cccaea8fa9a8c9ece7584f2e6ea6ccb291981"
)
_CANDIDATE_CAPABILITY = "LAGUNA_MOE_GROUP_COMPACT_H7U_MODE"
_SOURCE_CAPABILITY = "LAGUNA_MOE_GROUP_COMPACT_MODE"
_CANDIDATE_BLOCK = (
    "# WPF-H7U exposes the exact registered stable parallel active-route compactor\n"
    "# only as a bounded default-off W7900 candidate. The live source owner remains\n"
    "# serial until complete standalone/runtime/source qualification.\n"
    'LAGUNA_MOE_GROUP_COMPACT_H7U_MODE = "parallel"\n'
)
_SOURCE_BLOCK = (
    "# WPF-H7U promotes exact stable parallel active-route compaction after full\n"
    "# standalone, bounded-runtime, fixed, length, and source-trace qualification.\n"
    "# Explicit serial remains the registered rollback; peer backends stay local.\n"
    'LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"\n'
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_state() -> str:
    source = _PACKAGE.read_text()
    candidate_count = source.count(_CANDIDATE_BLOCK)
    promoted_count = source.count(_SOURCE_BLOCK)
    assert candidate_count + promoted_count == 1
    normalized = source.replace(_CANDIDATE_BLOCK, "").replace(_SOURCE_BLOCK, "")
    assert hashlib.sha256(normalized.encode()).hexdigest() == _POST_MERGE_PACKAGE_SHA256
    return "candidate" if candidate_count else "source"


def test_h7u_source_red_pins_admitted_standalone_and_source_gate() -> None:
    artifact_bytes = _ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "qualified_bounded_default_off_standalone"
    assert artifact["decision"] == {
        "bounded_default_off_capability_retained": True,
        "next_action": (
            "freeze a separate runtime/source RED, then run bounded complete-state "
            "and fixed C4096/M512 before clean source 512/1K/4K qualification"
        ),
        "production_changed": False,
        "production_speed_claim": False,
        "runtime_qualified": False,
        "source_promoted": False,
        "standalone_admitted": True,
    }
    assert artifact["red_green"]["green"] == {"failed": 0, "passed": 9}
    assert artifact["natural_m512_correctness"]["metadata_layers"] == 47
    assert artifact["natural_m512_correctness"]["gather_layers"] == 47
    assert artifact["natural_m512_correctness"]["all_48_hidden_boundaries_exact"]
    assert artifact["named_trace"]["profile"]["application_dispatches"] == 2_286
    assert artifact["named_trace"]["profile"]["contiguous_stage_triples"] == 47
    assert artifact["named_trace"]["checks"]["serial_calls_eq_0"]
    assert artifact["immutable_actual_routing_screen"][
        "all_47_layers_both_clock_positive"
    ]
    assert artifact["immutable_actual_routing_screen"]["first_and_only_timing_run"]
    aggregate = artifact["immutable_actual_routing_screen"]["aggregate"]
    assert aggregate["event_speedup"] > 15.0
    assert aggregate["wall_speedup"] > 14.0
    assert artifact["production"]["source_group_compact_mode"] == "serial"
    assert artifact["production"]["unchanged"]
    assert _package_state() in {"candidate", "source"}


def test_h7u_source_red_preserves_registered_parallel_leaf_and_serial_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_qwen35_moe_group_scatter_kernels(replace=True)
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_group_compact",
        quant="generic",
        variant="active_experts_parallel",
    ) is qwen35_moe_group_compact_active_parallel
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_group_compact",
        quant="generic",
        variant="active_experts_source_rows_parallel",
    ) is qwen35_moe_group_compact_active_source_rows_parallel
    assert resolve_laguna_group_compact_mode("hip_gfx1100", "serial") == "serial"
    assert resolve_laguna_group_compact_mode("hip_gfx1100", "parallel") == "parallel"
    with pytest.raises(ValueError, match="group compact"):
        resolve_laguna_group_compact_mode("hip_gfx1100", "atomic")

    # Exercise the future source owner and retained explicit rollback in memory.
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, "parallel", raising=False)
    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "parallel"
    assert resolve_laguna_group_compact_mode("hip_gfx1100", "serial") == "serial"
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, "serial")
    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "serial"

    # gfx1151 keeps its independently qualified source and no H7U seam.
    assert getattr(hip_gfx1151, _SOURCE_CAPABILITY) == "parallel"
    assert not hasattr(hip_gfx1151, _CANDIDATE_CAPABILITY)


def test_h7u_source_default_replaces_candidate_seam_atomically() -> None:
    # Intentional RED after artifact, package normalization, leaf, explicit
    # rollback, resolver, and peer-backend controls all pass.
    assert _package_state() == "source"
    assert not hasattr(hip_gfx1100, _CANDIDATE_CAPABILITY)
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == "parallel"
    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "parallel"
    assert resolve_laguna_group_compact_mode("hip_gfx1100", "serial") == "serial"
