import json
from types import SimpleNamespace

from scripts import qwen4exp_q8_localize as module


def test_interventions_count_real_selection_and_restore_hooks(monkeypatch, tmp_path):
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as runner

    parent, selected = object(), object()
    dispatch = lambda *a, **kw: selected
    down_flag = module.localize.PREFIX + "Q8_0_SELECTED_WMMA_DOWN"
    observed = []
    cleared = []
    moe = lambda *a, **kw: observed.append(module.os.environ.get(down_flag))
    monkeypatch.setattr(linear, "_q8_mmq_prefill_dispatch", dispatch)
    monkeypatch.setattr(runner, "run_qwen4_exp_moe", moe)
    output = tmp_path / "result.json"
    monkeypatch.setattr(module.sys, "argv", ["probe", "--output", str(output)])
    monkeypatch.setenv(down_flag, "1")
    monkeypatch.setenv(module.SHAPE_FLAG, "2560x640")
    monkeypatch.setenv(module.LAYER_FLAG, "4")
    monkeypatch.setattr(module.localize, "clear_arm_graphs",
                        lambda runner: cleared.append("graphs"))
    monkeypatch.setattr(linear, "clear_gguf_linear_dispatch_cache",
                        lambda: cleared.append("dispatch"))

    def run():
        module.localize.clear_arm_graphs(None)
        assert linear._q8_mmq_prefill_dispatch(
            parent, rows=64, in_features=2560, out_features=640) is parent
        assert linear._q8_mmq_prefill_dispatch(
            parent, rows=64, in_features=2560, out_features=512) is selected
        weight = SimpleNamespace(spec=SimpleNamespace(slot_path="layers.4.expert_down"))
        runner.run_qwen4_exp_moe(0, {"expert_down": weight})
        assert module.os.environ[down_flag] == "1"

    monkeypatch.setattr(module.localize, "main", run)
    module.main()
    assert observed == ["0"]
    assert cleared == ["graphs", "dispatch"]
    assert linear._q8_mmq_prefill_dispatch is dispatch
    assert runner.run_qwen4_exp_moe is moe
    counts = json.loads(output.with_suffix(".counts.json").read_text())
    assert counts["mmq:2560x640:2560x640:disabled=True"] == 1
    assert counts["mmq:2560x640:2560x512:disabled=False"] == 1


def test_boundary_replay_dequant_uses_fp16_scales_and_signed_codes():
    import numpy as np
    from scripts.qwen4exp_q8_boundary_replay import dequant

    raw = np.zeros((2, 2, 34), dtype=np.uint8)
    scales = np.array([[0.5, -0.25], [2, 0.125]], dtype=np.float16)
    raw[:, :, :2] = scales.view(np.uint8).reshape(2, 2, 2)
    codes = np.arange(-16, 16, dtype=np.int8)
    raw[:, :, 2:] = codes.view(np.uint8)
    expected = (scales.astype(np.float64)[:, :, None] * codes).reshape(2, 64)
    np.testing.assert_array_equal(dequant(raw, 64), expected)


def test_task_completion_stops_at_eos_and_labels_truncation():
    from scripts.qwen4exp_q8_repair_tasks import completion

    calls = []
    runner = SimpleNamespace(
        reset=lambda: calls.append("reset"),
        prefill=lambda ids: SimpleNamespace(token_id=3),
        step=lambda token: (calls.append(token) or SimpleNamespace(token_id=4)),
    )
    tokenizer = SimpleNamespace(eos_token_id=4, decode=lambda ids, **kw: str(ids))
    result = completion(runner, tokenizer, [1, 2], 5)
    assert result["ids"] == [3, 4]
    assert result["finish_reason"] == "eos"
    assert calls == ["reset", 3]
    assert completion(runner, tokenizer, [1], 1)["finish_reason"] == "length"
