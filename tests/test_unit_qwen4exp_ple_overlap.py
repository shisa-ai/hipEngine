"""PLE overlap retains memory and first-consumer preconditions."""
import ast
import inspect
import os
from types import SimpleNamespace

import pytest
from hipengine.runtime import qwen4_exp_runner as runtime


@pytest.mark.parametrize('pinned,layers,expected', [
    (True, (3, 7), True), (False, (3, 7), False),
    (True, (0, 3), False), (True, (), False),
])
@pytest.mark.parametrize('legacy', ['0', '1'])
def test_overlap_policy(monkeypatch, pinned, layers, expected, legacy):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_PLE_OVERLAP', legacy)
    tree = ast.parse(inspect.getsource(runtime))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'overlap_ple'
                              for t in n.targets))
    expression = ast.Expression(assignment.value)
    owner = SimpleNamespace(resident=SimpleNamespace(ple_staging=SimpleNamespace(pinned=pinned)))
    result = eval(compile(expression, '<overlap-policy>', 'eval'),
                  dict(self=owner, cfg=SimpleNamespace(ple_layers=layers), os=os))
    assert result is expected
