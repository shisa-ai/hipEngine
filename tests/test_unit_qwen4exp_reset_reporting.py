"""Reset diagnostic environment reporting must match the runtime default."""
import ast
import os
from pathlib import Path

import pytest

from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_bounded_kv_reset_enabled

ENV = "HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["qwen4exp_rt4_kv_compare.py", "qwen4exp_de5_reset_split.py"])
@pytest.mark.parametrize("value", [None, "", "0", "1", "false", "False", "true"])
def test_reported_reset_setting_matches_runtime(monkeypatch, script, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    # Execute the reporting expression without loading a model or issuing GPU
    # work. Inspect both main paths rather than duplicating their defaults here.
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "os.environ.get"
             and node.args and isinstance(node.args[0], ast.Constant)
             and node.args[0].value == ENV]
    assert len(calls) == 1
    reported = eval(compile(ast.Expression(calls[0]), script, "eval"), {"os": os})
    assert reported == ("1" if value is None else value)
    assert (reported not in {"", "0", "false", "False"}) == _qwen4_exp_bounded_kv_reset_enabled()
