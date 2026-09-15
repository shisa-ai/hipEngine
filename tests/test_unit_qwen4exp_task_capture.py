import pytest

from scripts.qwen4exp_q8_repair_tasks import select_task_prompts, task_candidate


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
