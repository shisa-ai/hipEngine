"""The exact norm/inject fusion has no process-wide opt-out."""
import ast
import inspect
import textwrap

import pytest

from hipengine.runtime.qwen4_exp_runner import run_qwen4_exp_gr_read


@pytest.mark.parametrize("stale", ["0", "1", "false"])
@pytest.mark.parametrize("scale_only,has_inject", [(False, False), (False, True),
                                                   (True, False), (True, True)])
def test_norm_inject_follows_supported_scale_path(monkeypatch, stale, scale_only, has_inject):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_NORM_INJECT", stale)
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_qwen4_exp_gr_read)))
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "norm_inject"
                              for target in node.targets))
    expression = compile(ast.Expression(assignment.value), "norm_inject", "eval")
    result = eval(expression, {"scale_only": scale_only,
                              "inject_weight": object() if has_inject else None})
    assert result is (scale_only and has_inject)
