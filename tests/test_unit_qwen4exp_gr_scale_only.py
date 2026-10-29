"""Scale-only GR selection follows capability, not a process opt-out."""
import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

import pytest
from hipengine.runtime.qwen4_exp_runner import run_qwen4_exp_gr_read


@pytest.mark.parametrize("stale", ["0", "1", "false"])
@pytest.mark.parametrize("case", ["supported", "no_direct", "no_wide", "iu8",
                                 "branches", "hidden", "width", "up_quant",
                                 "inject_quant", "no_inject"])
def test_scale_only_capability_ignores_retired_control(monkeypatch, stale, case):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_SCALE_ONLY", stale)
    def weight(quant):
        return SimpleNamespace(spec=SimpleNamespace(quant_key=quant))
    scope = {
        "os": os, "rows": 300, "low_rank": 64,
        "_qwen4_exp_gr_f16_direct_enabled": lambda *args: case != "no_direct",
        "_qwen4_exp_gr_wide_enabled": lambda *args: case != "no_wide",
        "_qwen4_exp_gr_iu8_enabled": lambda *args: case == "iu8",
        "branches": 3 if case == "branches" else 4,
        "hidden": 513 if case == "hidden" else 512,
        "residual_width": 2049 if case == "width" else 2048,
        "up_weight": weight("f32" if case == "up_quant" else "gguf_q8_0"),
        "inject_weight": None if case == "no_inject" else weight(
            "gguf_q8_0" if case == "inject_quant" else "f32"),
    }
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_qwen4_exp_gr_read)))
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "scale_only"
                              for target in node.targets))
    actual = eval(compile(ast.Expression(assignment.value), "scale_only", "eval"), scope)
    assert actual is (case in {"supported", "no_inject"})
