"""Direct QSA scoring depends on geometry, not a process rollback."""
import ast
import inspect

import pytest

from hipengine.runtime import qwen4_exp_runner as runtime


@pytest.mark.parametrize('legacy', ['0', 'false', '1'])
@pytest.mark.parametrize('heads,dim,expected', [(4,128,True),(3,128,False),(4,64,False)])
def test_direct_selection_geometry_ignores_retired_key(monkeypatch, legacy, heads, dim, expected):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_DIRECT_SELECT', legacy)
    tree = ast.parse(inspect.getsource(runtime.run_qwen4_exp_qsa_prefill_token_mixer))
    decision = next(node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'direct_select' for t in node.targets))
    expression = compile(ast.Expression(decision), '<direct-select>', 'eval')
    assert eval(expression, vars(runtime), dict(index_heads=heads, index_dim=dim)) is expected


def test_retired_direct_selector_has_no_runtime_reader():
    assert 'HIPENGINE_QWEN4_EXP_QSA_DIRECT_SELECT' not in inspect.getsource(runtime)
