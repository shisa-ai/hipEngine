"""The router SGEMM comparison seam after the env lever's removal (R0048).

The promotion packet separates its router arms in-process by moving
``gemma4_router._ROUTER_SGEMM_MIN_TOKENS``: the strict arm raises it above any
block width so the tier's own ``tokens >= threshold`` guard is false, and the
candidate arm restores the shipped value. These tests pin that switch, its
restore on a raising capture, and the absence of an environment read on both
sides of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.gemma4_wmma_promotion_packet as packet
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_router

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_SOURCE = REPO_ROOT / "hipengine/kernels/hip_gfx1100/gemma4/gemma4_router.py"
PACKET_SOURCE = REPO_ROOT / "scripts/gemma4_wmma_promotion_packet.py"

# Block widths a router projection can see: the production full block, the
# packet's widest case, and a wide prefill chunk.
FULL_BLOCK_WIDTHS = (1024, 2053, 4093, 8191, 32768)


@pytest.fixture(autouse=True)
def _restore_threshold():
    original = gemma4_router._ROUTER_SGEMM_MIN_TOKENS
    yield
    gemma4_router._ROUTER_SGEMM_MIN_TOKENS = original


def test_the_tier_guard_is_the_threshold_the_packet_moves():
    """The arm switch moves the real guard, not a second selector."""

    assert "if tokens >= _ROUTER_SGEMM_MIN_TOKENS:" in ROUTER_SOURCE.read_text()


def test_strict_arm_excludes_full_blocks():
    arm = packet.RouterThresholdArm()
    arm.select(candidate=False)
    threshold = gemma4_router._ROUTER_SGEMM_MIN_TOKENS
    assert threshold == packet.ROUTER_STRICT_MIN_TOKENS
    assert not any(tokens >= threshold for tokens in FULL_BLOCK_WIDTHS)


def test_candidate_arm_restores_the_shipped_threshold():
    arm = packet.RouterThresholdArm()
    baseline = gemma4_router._ROUTER_SGEMM_MIN_TOKENS
    assert baseline == 1024, "the shipped threshold is the production full block"
    arm.select(candidate=False)
    arm.select(candidate=True)
    assert gemma4_router._ROUTER_SGEMM_MIN_TOKENS == baseline


def test_restore_puts_the_threshold_back_and_is_idempotent():
    arm = packet.RouterThresholdArm()
    baseline = gemma4_router._ROUTER_SGEMM_MIN_TOKENS
    arm.select(candidate=False)
    arm.restore()
    assert gemma4_router._ROUTER_SGEMM_MIN_TOKENS == baseline
    arm.restore()  # main's finally may run after any exit
    assert gemma4_router._ROUTER_SGEMM_MIN_TOKENS == baseline


def test_a_raising_capture_still_restores_the_threshold():
    arm = packet.RouterThresholdArm()
    baseline = gemma4_router._ROUTER_SGEMM_MIN_TOKENS
    with pytest.raises(RuntimeError):
        with arm.strict():
            assert gemma4_router._ROUTER_SGEMM_MIN_TOKENS == packet.ROUTER_STRICT_MIN_TOKENS
            raise RuntimeError("capture failed")
    assert gemma4_router._ROUTER_SGEMM_MIN_TOKENS == baseline


def test_the_router_projection_reads_no_environment():
    # The retired lever's name is deliberately not spelled in this file:
    # scripts/check_envs_docs.py treats every `HIPENGINE_*` literal under
    # tests/ as an environment read, so naming it would demand documenting it
    # again. Any reintroduced read needs `getenv`/`environ`, which this pins.
    source = ROUTER_SOURCE.read_text()
    assert "os.environ" not in source
    assert "getenv" not in source


def test_the_packet_arms_router_without_an_environment_variable():
    source = PACKET_SOURCE.read_text()
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "import os" not in source
    assert "RouterThresholdArm" in source


def test_router_is_still_a_candidate_kind(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gemma4_wmma_promotion_packet.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        packet.main()
    assert exit_info.value.code == 0
    assert "router" in capsys.readouterr().out
