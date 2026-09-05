"""Live grouped-row4 route: original row ordering and exact MoE publication."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import qwen4_exp_runner as runner


def test_row4_benchmark_arm_and_engagement():
    from scripts.qwen4exp_halo_box_campaign_ab import (
        _apply_mode, validate_row4_engagement, ROW4_ENV, FORKB_ENV, Q5_M1_ENV,
    )
    env = {FORKB_ENV: "1", Q5_M1_ENV: "1"}
    _apply_mode("before", environment=env, route_package="q5k-row4")
    assert env == {FORKB_ENV: "1", Q5_M1_ENV: "1", ROW4_ENV: "0"}
    _apply_mode("after", environment=env, route_package="q5k-row4")
    assert env[ROW4_ENV] == "1"
    validate_row4_engagement("before", 0)
    validate_row4_engagement("after", 2)
    with pytest.raises(ValueError):
        validate_row4_engagement("after", 0)
    with pytest.raises(ValueError):
        validate_row4_engagement("before", 2)

def test_row4_admission(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL", "1")
    monkeypatch.setattr(runner, "is_registered", lambda key: key.quant == "candidate")
    weight = SimpleNamespace(spec=SimpleNamespace(quant_key="candidate"))
    weights = {"expert_gate": weight, "expert_up": weight}
    assert runner.qwen4_exp_grouped_row4_prefill_selected(weights, rows=64, backend="fake")
    assert not runner.qwen4_exp_grouped_row4_prefill_selected(weights, rows=63, backend="fake")
    weights["expert_up"] = SimpleNamespace(spec=SimpleNamespace(quant_key="missing"))
    assert not runner.qwen4_exp_grouped_row4_prefill_selected(weights, rows=64, backend="fake")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL", "0")
    assert not runner.qwen4_exp_grouped_row4_prefill_selected(weights, rows=64, backend="fake")


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _hip_available(), reason="HIP unavailable")
def test_live_row4_matches_parent(monkeypatch):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import register, KernelKey, resolve
    from tests.test_qwen4_exp_runner_moe import (
        _quant_weight, _q8_0_weight, _dense_f32_weight, _upload, _download,
    )
    from tests.test_qwen4exp_q5k_grouped_row4 import make_weights
    from tests.test_qwen4exp_pf1_forkb_selected_down import make_q8_0_weight_large
    from hipengine.quant.gguf import GGMLQuantizationType

    register_gfx1151_kernels(replace=True)
    runtime = get_hip_runtime()
    key = KernelKey("hip_gfx1151", "linear", "gguf_q5_k",
                    "selected_grouped_row4_gemv_bf16_bf16_out")
    candidate = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return candidate(*args, **kwargs)

    register(key, counted, replace=True)
    rng = np.random.default_rng(986)
    rows, hidden, ffn, experts, topk = 65, 256, 256, 8, 3
    allocations = []
    scratch = None
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL", "0")
    try:
        mixed = _upload(rng.normal(0, 0.1, (rows, hidden)).astype(np.float32), runtime, allocations)
        weights = {
            name: _dense_f32_weight(name, rng.normal(0, 0.02, shape).astype(np.float32), runtime, allocations)
            for name, shape in {
                "router": (experts, hidden), "shared_gate": (ffn, hidden),
                "shared_up": (ffn, hidden), "shared_down": (hidden, ffn),
                "shared_gate_weight": (1, hidden),
            }.items()
        }
        for i, name in enumerate(("expert_gate", "expert_up")):
            weights[name] = _quant_weight(
                name, make_weights(experts, ffn, hidden, seed=931+i),
                "gguf_q5_k", GGMLQuantizationType.Q5_K, 176, runtime, allocations)
        weights["expert_down"] = _q8_0_weight(
            "expert_down", make_q8_0_weight_large(experts * hidden, ffn).reshape(experts, hidden, -1),
            runtime, allocations)
        scratch = runner.Qwen4ExpMoEScratch.allocate(
            rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=topk, runtime=runtime)

        def run(enabled):
            monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL", enabled)
            result = runner.run_qwen4_exp_moe(
                mixed.ptr, weights, scratch=scratch, rows=rows, hidden=hidden,
                ffn=ffn, experts=experts, top_k=topk, runtime=runtime)
            runtime.device_synchronize()
            return [
                _download(result.output, (rows, hidden), np.uint16, runtime),
                _download(scratch.expert_gate, (rows * topk, ffn), np.uint16, runtime),
                _download(scratch.expert_up, (rows * topk, ffn), np.uint16, runtime),
                _download(scratch.expert_down, (rows * topk, hidden), np.uint16, runtime),
                _download(scratch.selected, (rows, topk), np.int64, runtime),
                _download(scratch.routing, (rows, topk), np.float32, runtime),
            ]

        parent = run("0")
        assert not calls
        for _ in range(2):
            actual = run("1")
            for a, b in zip(actual, parent):
                np.testing.assert_array_equal(a, b)
        assert len(calls) == 4
        replay = run("0")
        for a, b in zip(replay, parent):
            np.testing.assert_array_equal(a, b)
    finally:
        if scratch:
            scratch.close()
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
