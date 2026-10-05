"""Qualification harness contracts for ordered router DPP (CPU only)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.qwen4exp_halo_box_campaign_ab import (
    _apply_mode, build_parser, router_dpp_expected_calls,
)
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
from scripts.qwen4exp_router_dpp_gate_support import assert_router_outputs_exact

PARENT = "f32_hidden_token_tile4_dense_exact"
DPP = "f32_hidden_token_tile4_dpp_exact"
ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"


@pytest.mark.parametrize("mode,enabled,variant", [("before", "0", PARENT), ("after", "1", DPP)])
def test_router_gate_changes_only_the_producer(mode, enabled, variant):
    environment = {"unrelated": "unchanged"}
    _apply_mode(mode, route_package="router-dpp", environment=environment)
    assert environment == {"unrelated": "unchanged", ENV: variant}
    other = {"unrelated": "unchanged"}
    apply_state_gate_mode(None, "router-dpp", enabled, ENV, environment=other)
    assert other == environment
    args = build_parser().parse_args([
        "--model-root", "/unused", "--output", "/unused.json",
        "--route-package", "router-dpp", "--prefill-chunk-size", "4096",
    ])
    assert args.route_package == "router-dpp"


def test_router_ab_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _apply_mode("unknown", route_package="router-dpp", environment={})


@pytest.mark.parametrize("tokens,chunk,expected", [
    (1, 4096, 0), (2, 4096, 48), (512, 512, 48),
    (1024, 512, 96), (4096, 1024, 192), (4096, 4096, 48),
    (4097, 4096, 48), (4098, 4096, 96), (2, 1, 0),
])
def test_router_engagement_excludes_single_row_tail(tokens, chunk, expected):
    assert router_dpp_expected_calls(tokens, chunk) == expected


@pytest.mark.parametrize("tokens,chunk", [(0, 4096), (512, 0)])
def test_router_engagement_rejects_invalid_shape(tokens, chunk):
    with pytest.raises(ValueError):
        router_dpp_expected_calls(tokens, chunk)


def _outputs():
    return [np.array([[0., -0., 1.]], np.float32),
            np.array([[2, 1]], np.int64), np.array([[.75, .25]], np.float32)]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_private_checker_rejects_parent_and_repeat_bit_mismatch(which):
    parent = _outputs()
    candidate = [a.copy() for a in parent]
    repeat = [a.copy() for a in parent]
    assert set(assert_router_outputs_exact(parent, candidate, repeat)) == {
        "logits_sha256", "expert_ids_sha256", "routing_sha256"}
    candidate[which].view(np.uint8).flat[0] ^= 1
    with pytest.raises(AssertionError, match="differs from parent bits"):
        assert_router_outputs_exact(parent, candidate, repeat)
    candidate = [a.copy() for a in parent]
    repeat[which].view(np.uint8).flat[0] ^= 1
    with pytest.raises(AssertionError, match="repeat differs"):
        assert_router_outputs_exact(parent, candidate, repeat)


def test_private_checker_rejects_signed_zero_flip_and_nonfinite():
    parent = _outputs()
    candidate = _outputs()
    candidate[0][0, 1] = 0.
    with pytest.raises(AssertionError, match="parent bits"):
        assert_router_outputs_exact(parent, candidate, candidate)
    candidate[0][0, 1] = np.nan
    with pytest.raises(AssertionError, match="nonfinite"):
        assert_router_outputs_exact(parent, candidate, candidate)


def test_named_candidate_manifest_and_strict_binder_in_fresh_process():
    # Import-time selection must be tested in a fresh interpreter; reloading
    # shared profile modules would mutate another test's module/registry state.
    code = '''
import json, os, warnings
from types import SimpleNamespace
from scripts.qwen4exp_q8_repair_depth_gate import resolve_allocation_profile
from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.generation import qwen4_exp_profiles as p
resolve_allocation_profile()
p.register_qwen4_exp_gfx1151_profiles()
out = {}
for mode in ("strict", "production"):
 r = resolve_runtime_profile(model=p.QWEN4_EXP_MODEL,backend=p.QWEN4_EXP_BACKEND,
                            quant="gguf_ud_q4_k_xl",profile=mode)
 with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  r.construct_generator(lambda: SimpleNamespace())
 row = next(s for s in r.manifest["selections"] if s["layer"]=="router_logits")
 out[mode] = dict(selection=dict(row), bound=os.environ[p._ROUTER_VARIANT_ENV])
print(json.dumps(out))
'''
    env = dict(os.environ, **{ENV: DPP})
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(result.stdout)
    assert data["strict"]["bound"] == PARENT
    assert data["strict"]["selection"]["selected_variant"] == PARENT
    assert data["production"]["bound"] == DPP
    assert data["production"]["selection"]["selected_variant"] == DPP
    assert data["production"]["selection"]["strict_fallback_variant"] == PARENT
    assert data["production"]["selection"]["evidence_artifact"].endswith("qwen4exp-router-dpp/README.md")
