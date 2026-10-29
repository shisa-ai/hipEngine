"""Threshold comparison children must isolate singleton and fused pack8 routes."""

import json
import os
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear
from scripts import qwen35_08b_threshold_sweep as sweep


@pytest.mark.parametrize("quant,role", [
    ("q4", "current"), ("q4", "pre_x2"), ("q4", "strict_x2"),
    ("q8", "current"), ("q8", "strict_x2"),
])
@pytest.mark.parametrize("inherited", [None, "0", "1"])
def test_threshold_child_pins_fused_pair_and_records_effective_environment(monkeypatch, tmp_path, quant, role, inherited):
    name = "HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL"
    if inherited is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, inherited)
    monkeypatch.setattr(sweep, "OUT_DIR", tmp_path)
    launches = []
    def run(command, **kwargs):
        launches.append((command, kwargs))
        payload = {"summary_by_workload": {"512/1": {"prefill_tok_s": {"median": 1.0}}},
            "runs_by_workload": {"512/1": [{"measured": True,
                "timings": {"prefill_seconds": 1.0},
                "correctness_sanity": {"finite_final_logits": True, "final_token_id": 1},
                "prefill_chunk_sizes": [512],
                "memory_snapshots": {"after_prefill": {"scratch_max_positions": 512}}}]},
            "provenance": {"hipengine_commit": "synthetic", "dirty": False}}
        from pathlib import Path
        Path(command[command.index("--json") + 1]).write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(sweep.subprocess, "run", run)
    result = sweep.run_one(quant, role, 0, 0, 512)
    env = {key: launches[0][1]["env"].get(key) for key in (
        *sweep.ROLE_ENVS[quant][role], name)}
    expected = None if role == "current" else "0"
    assert env.get(name) == expected
    assert os.environ.get(name) == inherited
    assert result["environment"] == {key: env.get(key) for key in sweep.ROLE_ENVS[quant][role]}

    # Exercise the real dispatch predicate, not just the spelling of the env.
    with monkeypatch.context() as patch:
        for key in (name, "HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK"):
            if env.get(key) is not None:
                patch.setenv(key, env[key])
            else:
                patch.delenv(key, raising=False)
        dispatch = SimpleNamespace(abi="pack8", key=KernelKey("hip_gfx1151",
            "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"))
        patch.setattr(gguf_linear, "_ensure_linear_kernel_registered", lambda key: None)
        patch.setattr(gguf_linear, "is_registered", lambda key: True)
        with gguf_linear.q4_pack8_dual_wmma_silu_prefill_session(True):
            key = gguf_linear._pack8_dual_wmma_silu_dispatch(dispatch, dispatch,
                rows=512, in_features=1024, out_features=3584, use_wmma=True)
        assert (key is not None) == (role == "current")


def test_threshold_role_keeps_each_child_effective_environment(monkeypatch):
    monkeypatch.setattr(sweep, "LENGTHS", (17, 511, 513))
    def child(quant, role, block, order_index, length):
        return {"artifact": str(length), "command": ["synthetic", str(length)],
            "environment": {"effective": str(length)}, "source_commit": "synthetic",
            "clean_tree": True, "row": {"length": length}}
    monkeypatch.setattr(sweep, "run_one", child)
    result = sweep.run_role("q4", "current", 0, 0)
    assert result["environments"] == {
        str(length): {"effective": str(length)} for length in (17, 511, 513)}
