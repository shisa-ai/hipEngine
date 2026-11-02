"""GR IU8 choices travel with their owner without changing arithmetic."""
from types import SimpleNamespace
import inspect
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


@pytest.mark.parametrize('rows', [17, 255, 256, 257, 389])
@pytest.mark.parametrize('up,down', [(False, False), (True, False), (False, True), (True, True)])
def test_gr_iu8_policy_beats_ambient(monkeypatch, rows, up, down):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8', '0' if up else '1')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8_DOWN', '0' if down else '1')
    policy = runtime.GRIU8Policy(up, down)
    assert runtime._qwen4_exp_gr_iu8_enabled(rows, policy) is (up and rows > 256)
    assert runtime._qwen4_exp_gr_iu8_down_enabled(rows, policy) is (down and rows > 256)


def test_gr_iu8_scope_restores_distinct_values_and_absence():
    prior = runtime.GRIU8Policy(False, True)
    runner = SimpleNamespace(gr_iu8_policy=prior)
    owner = SimpleNamespace(runner=runner)
    active = runtime.GRIU8Policy(True, False)
    with pytest.raises(RuntimeError, match='restore'):
        with runtime.gr_iu8(owner, active):
            assert owner.gr_iu8_policy is runner.gr_iu8_policy is active
            assert _inherit_lane_decisions(owner, SimpleNamespace()).gr_iu8_policy is active
            raise RuntimeError('restore')
    assert not hasattr(owner, 'gr_iu8_policy')
    assert runner.gr_iu8_policy is prior


@pytest.mark.parametrize('up', ['0', 'false', 'yes'])
@pytest.mark.parametrize('down', ['0', 'false', 'yes'])
def test_gr_iu8_unbound_ignores_ambient_pair(monkeypatch, up, down):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8', up)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8_DOWN', down)
    assert runtime.resolve_gr_iu8_policy(None) == runtime.GRIU8Policy(False, False)
    assert runtime._qwen4_exp_gr_iu8_enabled(389) is False
    assert runtime._qwen4_exp_gr_iu8_down_enabled(389) is False


def test_gr_iu8_constructor_accepts_policy_before_allocation():
    policy = runtime.GRIU8Policy(True, False)
    runner = runtime.Qwen4ExpGGUFResidentModelRunner.__new__(runtime.Qwen4ExpGGUFResidentModelRunner)
    resident = SimpleNamespace(plan=SimpleNamespace(config=SimpleNamespace(context_length=8)))
    with pytest.raises(ValueError, match='max_sequence_length'):
        runner.__init__(resident, max_sequence_length=0, gr_iu8_policy=policy)
    assert runner.gr_iu8_policy is policy


def test_gr_iu8_all_read_carriers_forward_policy():
    import ast
    tree = ast.parse(inspect.getsource(runtime))
    targets = {'run_qwen4_exp_gr_read', '_run_qwen4_exp_gr_norm_down',
               'run_qwen4_exp_dense_qsa_layer', 'run_qwen4_exp_qsa_prefill_layer',
               'run_qwen4_exp_gdn_layer'}
    reads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in targets]
    assert reads
    for call in reads:
        assert 'gr_iu8_policy' in {kw.arg for kw in call.keywords}, (call.func.id, call.lineno)
