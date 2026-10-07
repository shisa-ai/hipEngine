"""HIP-guarded live smoke for the F16 Q5_1 grouped down tile-map prerequisite.

The down-only configuration (production exact Q4 gate/up + F16 Q5_1 down) used
to raise ``ValueError: compact_rows, num_experts, and wmma_total_rows must be
positive`` because the F16 down owner was handed a stale zero host tile-row
count. This smoke loads the real model, selects the down-only route, and proves
two things at once:

- the F16 down owner engages and receives a positive multiple-of-16 row count,
  and
- the gate/up leg still runs the production exact iu8-risk+repair owner.

It is a bounded reduced protocol (one prompt, default chunk), not a numerical
gate; no rejection or promotion follows from it.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py"
MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)

_BACKEND = "hip_gfx1151"
_RISK_GATE_UP_KEY = (
    _BACKEND, "moe_linear", "gguf_q4_k",
    "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
)
_REPAIR_KEY = (
    _BACKEND, "moe_linear", "gguf_q4_k",
    "selected_dual_sparse_exact_repair_bf16",
)
_DOWN_ALIAS = "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_layer2_profile_gate_smoke", GATE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pytestmark = [
    pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable"),
    pytest.mark.skipif(not MODEL_ROOT.is_dir(), reason="model root absent"),
]


def test_down_only_route_engages_with_current_map_and_production_gate_up() -> None:
    gate = _load_gate()

    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        register_gguf_q4_k_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, register, resolve

    register_gfx1151_kernels(replace=True)
    register_gguf_q4_k_selected_prefill_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()

    args = SimpleNamespace(
        model_root=MODEL_ROOT,
        max_sequence_length=2051,
        prefill_chunk_size=512,
    )
    generator, _profile, _index = gate._make_generator(args, "production")

    counts = {"down": 0, "down_rows": [], "risk": 0, "repair": 0}
    runner_module = sys.modules["hipengine.runtime.qwen4_exp_runner"]

    def counted_down(*call_args, **kwargs):
        counts["down"] += 1
        counts["down_rows"].append(int(call_args[10]))
        return original_down(*call_args, **kwargs)

    def counted_risk(*call_args, **kwargs):
        counts["risk"] += 1
        return original_risk(*call_args, **kwargs)

    def counted_repair(*call_args, **kwargs):
        counts["repair"] += 1
        return original_repair(*call_args, **kwargs)

    original_down = getattr(runner_module, _DOWN_ALIAS)
    original_risk = resolve(
        backend=_RISK_GATE_UP_KEY[0], layer=_RISK_GATE_UP_KEY[1],
        quant=_RISK_GATE_UP_KEY[2], variant=_RISK_GATE_UP_KEY[3],
    )
    original_repair = resolve(
        backend=_REPAIR_KEY[0], layer=_REPAIR_KEY[1],
        quant=_REPAIR_KEY[2], variant=_REPAIR_KEY[3],
    )

    saved = {
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN": os.environ.get(
            "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN"
        ),
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA": os.environ.get(
            "HIPENGINE_QWEN4_EXP_Q5_1_WMMA"
        ),
    }
    try:
        os.environ["HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN"] = "0"
        os.environ["HIPENGINE_QWEN4_EXP_Q5_1_WMMA"] = "1"
        setattr(runner_module, _DOWN_ALIAS, counted_down)
        register(KernelKey(*_RISK_GATE_UP_KEY), counted_risk, replace=True)
        register(KernelKey(*_REPAIR_KEY), counted_repair, replace=True)

        rows = gate._load_suites(list(gate.DEFAULT_PROMPTS))[:1]
        tokens = gate.build_chat_prompt(generator.tokenizer, str(rows[0]["prompt"]))
        generator.runner.prefill([int(token) for token in tokens])
    finally:
        setattr(runner_module, _DOWN_ALIAS, original_down)
        register(KernelKey(*_RISK_GATE_UP_KEY), original_risk, replace=True)
        register(KernelKey(*_REPAIR_KEY), original_repair, replace=True)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        generator.close()

    # The former ValueError would have aborted before this point.
    assert counts["down"] > 0, "F16 Q5_1 down owner did not engage"
    assert counts["risk"] > 0, "production exact gate/up risk owner did not engage"
    assert counts["repair"] > 0, "production gate/up repair did not engage"
    assert all(row > 0 and row % 16 == 0 for row in counts["down_rows"]), counts[
        "down_rows"
    ]
