import pytest
from collections import Counter
from types import SimpleNamespace

from scripts.qwen4exp_q8_repair_tasks import completion, select_task_prompts, task_candidate


def test_completion_observes_phase_local_dispatch_without_changing_outputs():
    counter = {'calls': 99, 'shapes': Counter({(999,): 99})}

    class Runner:
        def reset(self):
            self.steps = 0

        def prefill(self, ids):
            counter['calls'] += 2
            counter['shapes'][(len(ids),)] += 2
            return SimpleNamespace(token_id=1)

        def step(self, token):
            self.steps += 1
            counter['calls'] += 1
            counter['shapes'][(1,)] += 1
            return SimpleNamespace(token_id=2 if self.steps == 1 else 0)

    tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda ids, **kwargs: str(ids))
    observation = {}
    result = completion(Runner(), tokenizer, [5] * 300, 8,
                        counter=counter, observation=observation)
    assert result == {'ids': [1, 2, 0], 'finish_reason': 'eos', 'text': '[1, 2, 0]'}
    assert observation['prefill'] == {'calls': 2, 'shapes': [{'arguments': [300], 'calls': 2}]}
    assert observation['first_decode']['calls'] == 1
    assert observation['later_decode']['calls'] == 1
    assert observation['candidate_observed'] is True
    assert observation['gpu_engagement_certified'] is False
    assert all(999 not in row['arguments'] for phase in ('prefill', 'first_decode', 'later_decode')
               for row in observation[phase]['shapes'])


def test_completion_does_not_borrow_engagement_from_previous_case():
    class Runner:
        def reset(self):
            pass

        def prefill(self, ids):
            return SimpleNamespace(token_id=0)

        def step(self, token):
            raise AssertionError('EOS must not decode')

    observation = {}
    completion(Runner(), SimpleNamespace(eos_token_id=0, decode=lambda *a, **k: ''),
               [7], 8, counter={'calls': 12, 'shapes': Counter({(300,): 12})},
               observation=observation)
    assert observation['candidate_observed'] is False
    assert all(observation[phase] == {'calls': 0, 'shapes': []}
               for phase in ('prefill', 'first_decode', 'later_decode'))



def test_completion_without_counter_marks_engagement_unknown():
    class Runner:
        def reset(self):
            pass

        def prefill(self, ids):
            return SimpleNamespace(token_id=3)

        def step(self, token):
            raise AssertionError('one-token cap must not decode')

    observation = {}
    tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda *a, **k: 'answer')
    result = completion(Runner(), tokenizer, [7], 1, observation=observation)
    assert result['finish_reason'] == 'length'
    assert result['ids'] == [3]
    assert observation['counted'] is False
    assert observation['candidate_observed'] is None
    assert observation['gpu_engagement_certified'] is False
    assert observation['first_decode'] == {'calls': 0, 'shapes': []}


def test_task_dispatch_uses_declared_registry_and_restores_on_error():
    from scripts import qwen4exp_q8_repair_tasks as tasks
    from hipengine.kernels.registry import KernelKey, register, resolve

    _, spec = task_candidate("production_gdn_multi_restore")
    key = KernelKey(*spec.candidate_key)
    original = lambda *args, **kwargs: "ran"
    register(key, original, replace=True)
    with pytest.raises(RuntimeError, match="failure"):
        with tasks.task_dispatch_context(spec) as counter:
            fn = resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                         variant=key.variant)
            assert fn(1) == "ran"
            assert counter["calls"] == 1
            raise RuntimeError("failure")
    assert resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                   variant=key.variant) is original


def test_uncounted_task_candidate_keeps_no_dispatch_requirement():
    from scripts.qwen4exp_q8_repair_tasks import task_dispatch_context

    _, spec = task_candidate("production_q8_fallback")
    with task_dispatch_context(spec) as counter:
        assert counter["calls"] == 0


def test_task_candidate_keeps_historical_label_and_accepts_gdn():
    label, spec = task_candidate("production_q8_fallback")
    assert label == "q8_fallback"
    assert spec.base_profile == "production"
    label, spec = task_candidate("production_gdn_restore")
    assert label == "production_gdn_restore"
    assert spec.environment["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"


def test_task_selection_preserves_fixture_order_and_marks_subsets():
    prompts = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert select_task_prompts(prompts, None) == (prompts, True)
    assert select_task_prompts(prompts, ["c", "a"]) == (
        [prompts[0], prompts[2]], False)
    with pytest.raises(ValueError, match="duplicate"):
        select_task_prompts(prompts, ["a", "a"])
    with pytest.raises(ValueError, match="unknown"):
        select_task_prompts(prompts, ["missing"])
