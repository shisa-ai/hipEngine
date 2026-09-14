import pytest

from scripts.qwen4exp_q8_repair_tasks import select_task_prompts, task_candidate


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
