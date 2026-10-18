from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter, _commit_eager_cycle_row
from hipengine.generation.sampling import RowSamplingState
from hipengine.llm import SamplingParams
from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.speculative.serving import resolve_speculative_mtp_serving_plan
from tests.test_unit_speculative_mtp_serving_capability import _key


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_sampled_concurrent_policy_is_automatic(width):
    decision = resolve_speculative_mtp_serving_plan(
        Qwen35GGUFModel().speculative_mtp_serving_evidence,
        key=_key(resident_capacity=4, realized_group_rows=width, sampling_mode="sampled"),
    )
    assert decision.admitted and decision.automatic_eligible


def test_packed_sampler_uses_each_request_span_seed_and_step():
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    states = (RowSamplingState(seed=17), RowSamplingState(seed=29))
    states[1].observe(4)
    rows = tuple(SimpleNamespace(
        request_id=rid, native_sampler=True, sampling_state=state,
        sampling_request=SamplingParams(temperature=0.7, top_p=0.95),
    ) for rid, state in zip((9, 3), states))
    owner = SimpleNamespace(
        _verify_logits_buf=SimpleNamespace(ptr=4096, nbytes=8*100*4),
        runner=SimpleNamespace(vocab_size=100),
        runtime=object(),
    )
    results = (
        SimpleNamespace(request_id=9, transaction_id=7, row_start=0, rows=4, target_top1=SimpleNamespace(ptr=8192)),
        SimpleNamespace(request_id=3, transaction_id=7, row_start=4, rows=3, target_top1=SimpleNamespace(ptr=8208)),
    )
    calls = []
    class Sampler:
        def stage(self, inputs):
            calls.append(("stage", inputs.seed, inputs.step_index, inputs.rows))
        def enqueue(self, ptr, out):
            calls.append(("enqueue", ptr, out))
    adapter._packed_native_sampler = lambda owner, count: Sampler()
    adapter._sample_packed_target_rows(owner, results, rows, transaction_id=7)
    assert calls == [
        ("stage", 17, 0, 4), ("enqueue", 4096, 8192),
        ("stage", 29, 1, 3), ("enqueue", 5696, 8208),
    ]
    assert [state.step_index for state in states] == [0, 1]
    assert all(row.full_vocab_logits_d2h is False for row in rows)


def test_packed_sampler_validates_all_identities_before_launch():
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter._packed_native_sampler = lambda *args: pytest.fail("launched before validation")
    rows = (SimpleNamespace(request_id=9), SimpleNamespace(request_id=3))
    results = (SimpleNamespace(request_id=9, transaction_id=7), SimpleNamespace(request_id=8, transaction_id=7))
    with pytest.raises(ValueError, match="identit"):
        adapter._sample_packed_target_rows(SimpleNamespace(), results, rows, transaction_id=7)


def test_batched_commit_advances_only_its_sampler_history():
    from hipengine.generation.qwen35_gguf import _GGUFResidentLoopRow
    from hipengine.generation import GenerationRequest

    row = _GGUFResidentLoopRow(
        request_id=9, batch_id=1, row_index=2, prompt_ids=(5,),
        request=GenerationRequest(prompts=("p",), max_tokens=8, temperature=0.7, top_p=0.95, ignore_eos=True),
        native_greedy=False, native_sampled=True, native_sampler=True, submitted_at=0,
    )
    row.sampling_state = RowSamplingState(seed=17)
    row.sampling_state.observe(2)
    row.slot = SimpleNamespace(generated_ids=[2], native_decode_steps=0)
    _commit_eager_cycle_row(
        row, visible=(3, 4), accepted=1, candidate_count=3,
        plan_reason="speculative_qualified", target_position=4,
    )
    assert tuple(row.sampling_state.generated_tokens) == (2, 3, 4)
    assert row.sampling_state.step_index == 3


def test_packed_verifier_requires_logits_when_any_job_samples():
    import ast
    import inspect
    import textwrap
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    tree = ast.parse(textwrap.dedent(inspect.getsource(
        Qwen35GGUFResidentSession.verify_target_blocks_batch)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_enqueue_target_block_rows_from_hidden"
    ]
    assert len(calls) == 1
    keyword = next((kw for kw in calls[0].keywords if kw.arg == "require_logits"), None)
    assert keyword is not None, "sampling must not use the fused greedy-only LM head"
    expression = ast.Expression(keyword.value)
    for jobs, expected in (
        ([{}], False), ([{"require_logits": True}], True),
        ([{}, {"require_logits": True}], True),
    ):
        assert eval(compile(expression, "<require_logits>", "eval"), {"job_list": jobs}) is expected


def test_packed_verifier_preserves_resident_initial_state_owner():
    import ast
    import inspect
    import textwrap
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    tree = ast.parse(textwrap.dedent(inspect.getsource(
        Qwen35GGUFResidentSession.verify_target_blocks_batch)))
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp)
                      and any(isinstance(t, ast.Name) and t.id == "linear_state_owner" for t in node.targets))
    resident, packed, explicit = object(), object(), object()
    for direct, expected in ((None, resident), (((), explicit), explicit)):
        assert eval(compile(ast.Expression(assignment.value), "<state-owner>", "eval"), {
            "linear_state_owner": resident, "packed_state": packed, "direct_linear_state": direct,
        }) is expected
