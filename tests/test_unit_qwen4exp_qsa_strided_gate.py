"""QSA prefill always consumes the projected gate without a copy."""
import ast
import inspect

from hipengine.runtime import qwen4_exp_runner as runtime


def test_prefill_has_no_strided_gate_rollback():
    assert 'HIPENGINE_QWEN4_EXP_QSA_STRIDED_GATE' not in inspect.getsource(runtime)
    tree=ast.parse(inspect.getsource(runtime.run_qwen4_exp_qsa_prefill_token_mixer))
    split=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='split_rope_rows')
    assert isinstance(split.args[7],ast.Constant) and split.args[7].value==0
    names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert 'qwen4_exp_qsa_gate_context_strided_f32' in names
    assert 'qwen4_exp_qsa_gate_context_f32' not in names
    assert {'declare_f16_mirror','invalidate_f16_mirror'} <= names
