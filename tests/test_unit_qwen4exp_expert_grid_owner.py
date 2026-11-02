"""Expert-grid family selection belongs to independent execution owners."""
from contextlib import ExitStack
from types import SimpleNamespace
import inspect
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


@pytest.mark.parametrize('q4,q5', [(False, False), (True, False), (False, True), (True, True)])
def test_explicit_family_overrides_ambient(monkeypatch, q4, q5):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID', '0' if q4 or q5 else '64')
    policy = runtime.ExpertGridPolicy(q4=q4, q5=q5)
    four = runtime._qwen4_exp_q4_gate_up_key(backend='hip_gfx1151', quant='gguf_q4_k',
        rows=17, in_features=4096, expert_grid_policy=policy)
    five = runtime._qwen4_exp_q51_down_key(backend='hip_gfx1151', quant='gguf_q5_1',
        rows=17, in_features=640, expert_grid_policy=policy)
    assert ('expertgrid64' in four.variant) is q4
    assert ('expertgrid64' in five.variant) is q5


@pytest.mark.parametrize('raw,q4,q5', [('64', True, True), ('q4', True, False),
                                     ('q5', False, True), ('0', False, False)])
def test_unbound_legacy_family_still_runs(monkeypatch, raw, q4, q5):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID', raw)
    assert runtime.resolve_expert_grid_policy(None) == runtime.ExpertGridPolicy(q4, q5)


def test_scope_restores_missing_and_distinct_values_and_late_clone():
    policy = runtime.ExpertGridPolicy(False, True)
    prior = runtime.ExpertGridPolicy(True, False)
    owner = SimpleNamespace(runner=SimpleNamespace(expert_grid_policy=prior))
    peer = SimpleNamespace()
    with pytest.raises(RuntimeError, match='injected'):
        with runtime.expert_grid_policy(owner, policy):
            assert owner.expert_grid_policy is owner.runner.expert_grid_policy is policy
            assert not hasattr(peer, 'expert_grid_policy')
            assert _inherit_lane_decisions(owner, SimpleNamespace()).expert_grid_policy is policy
            with runtime.expert_grid_policy(owner, prior):
                assert owner.expert_grid_policy is prior
            assert owner.expert_grid_policy is policy
            raise RuntimeError('injected')
    assert not hasattr(owner, 'expert_grid_policy')
    assert owner.runner.expert_grid_policy is prior


@pytest.mark.parametrize('function', [runtime.run_qwen4_exp_moe,
    runtime.run_qwen4_exp_gdn_layer, runtime.run_qwen4_exp_dense_qsa_layer,
    runtime.run_qwen4_exp_qsa_prefill_layer, batch.run_qwen4_exp_batch_gdn_layer,
    batch.run_qwen4_exp_batch_qsa_layer, batch._run_layer_ffn])
def test_forwarders_carry_policy(function):
    assert 'expert_grid_policy' in inspect.signature(function).parameters
    assert 'expert_grid_policy=expert_grid_policy' in inspect.getsource(function)


def test_batch_capture_key_includes_policy():
    owner = SimpleNamespace(q51_down_m1=False, q51_fold128_prefill=False,
        q51_fold_pair_prefill=False, q51_exact_grouped_down=True, q51_wmma_down=False,
        q51_out8=True, q4_exact_grouped_gate=True, q4_exact_grouped_all=True, q4_out4=True,
        forkb_grouped_down=False, q8_down_row4_prefill=False, q8_down_bundle_prefill=False,
        q5k_row4_prefill=False)
    method = batch.Qwen4ExpGGUFBatchRunner._expert_down_policy
    owner.expert_grid_policy = runtime.ExpertGridPolicy(False, True)
    before = method(owner)
    owner.expert_grid_policy = runtime.ExpertGridPolicy(True, False)
    assert before != method(owner)
    assert before == method(SimpleNamespace(**{**vars(owner),
        'expert_grid_policy': runtime.ExpertGridPolicy(False, True)}))


def test_every_dispatch_call_forwards_policy():
    import ast
    from pathlib import Path
    names = {'run_qwen4_exp_moe', 'run_qwen4_exp_gdn_layer',
        'run_qwen4_exp_dense_qsa_layer', 'run_qwen4_exp_qsa_prefill_layer',
        'run_qwen4_exp_batch_gdn_layer', 'run_qwen4_exp_batch_qsa_layer',
        '_run_layer_ffn', '_qwen4_exp_q4_gate_up_key', '_qwen4_exp_q51_down_key'}
    from hipengine.runtime import qwen4_exp_mixed_target as mixed
    for module in (runtime, batch, mixed):
        for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
                assert any(k.arg == 'expert_grid_policy' for k in node.keywords), (node.func.id, node.lineno)


@pytest.mark.parametrize('interface', ['profile_gap', 'candidate_ar'])
@pytest.mark.parametrize('raw,expected', [('64', (True, True)), ('q4', (True, False)),
    ('q5', (False, True)), ('0', (False, False)), ('other', (False, False))])
def test_generic_override_binds_without_process_writes(monkeypatch, interface, raw, expected):
    import importlib
    import os
    script = importlib.import_module('scripts.qwen4exp_' + interface)
    key = 'HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID'
    monkeypatch.setenv(key, 'stale')
    policy = runtime.ExpertGridPolicy(False, True)
    owner = SimpleNamespace(expert_grid_policy=policy, runner=SimpleNamespace(expert_grid_policy=policy))
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='injected'):
        with ExitStack() as stack:
            report = script.bind_expert_grid_override(stack, owner, {key: raw})
            selected = runtime.resolve_expert_grid_policy(owner.expert_grid_policy)
            assert selected == runtime.ExpertGridPolicy(*expected)
            assert owner.runner.expert_grid_policy is selected
            assert report['expert_grid_policy'] == selected.to_report()
            assert report['expert_grid_override'] == raw
            assert _inherit_lane_decisions(owner, SimpleNamespace()).expert_grid_policy is selected
            assert dict(os.environ) == before
            raise RuntimeError('injected')
    assert owner.expert_grid_policy is owner.runner.expert_grid_policy is policy


def test_generic_override_main_transport(monkeypatch):
    from scripts import qwen4exp_candidate_ar as ar, qwen4exp_profile_gap as gap
    import os
    key = 'HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID'
    overrides = {key: 'q4', 'HIPENGINE_HIP_ARCH': 'same'}
    monkeypatch.setenv(key, 'prior')
    monkeypatch.setenv('HIPENGINE_HIP_ARCH', 'prior')
    environment, selected = ar.split_expert_grid_override(overrides)
    assert environment == {'HIPENGINE_HIP_ARCH': 'same'}
    assert selected == {key: 'q4'}
    source = inspect.getsource(ar.main)
    assert source.index('split_expert_grid_override(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_expert_grid_override(" in source
    assert 'report.update(bind_expert_grid_override(' in inspect.getsource(gap.main)
    gap._apply_post_binder_overrides(overrides)
    assert os.environ[key] == 'prior'
    assert os.environ['HIPENGINE_HIP_ARCH'] == 'same'


@pytest.mark.parametrize('interface', ['profile_gap', 'candidate_ar'])
def test_omitted_override_keeps_owner_and_reports_it(interface):
    import importlib
    script = importlib.import_module('scripts.qwen4exp_' + interface)
    policy = runtime.ExpertGridPolicy(True, False)
    owner = SimpleNamespace(expert_grid_policy=policy)
    with ExitStack() as stack:
        report = script.bind_expert_grid_override(stack, owner, {})
        assert owner.expert_grid_policy is policy
        assert report == {'expert_grid_override': None, 'expert_grid_policy': policy.to_report()}


def test_constructor_carries_policy(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner',
        lambda resident, **kw: calls.append(kw) or SimpleNamespace(**kw))
    owner = generation.Qwen4ExpGGUFTextGenerator(model_path='unused.gguf',
        weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    owner.expert_grid_policy = runtime.ExpertGridPolicy(False, True)
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['expert_grid_policy'] is clone.expert_grid_policy is owner.expert_grid_policy
