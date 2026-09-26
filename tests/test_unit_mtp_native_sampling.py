from types import SimpleNamespace
from collections import defaultdict
from itertools import product

import numpy as np
import pytest

from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.speculative.serving import resolve_speculative_mtp_serving_plan
from tests.test_unit_speculative_mtp_serving_capability import _key


def test_native_sampled_row_is_admitted_to_mtp():
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter._sampled_route_qualified = lambda: True
    adapter._sampling_mode_by_request = {7: "sampled"}
    row = SimpleNamespace(native_sampler=True)
    adapter.owner = SimpleNamespace(_row=lambda rid: row)
    assert adapter._sampled_route_request(7)


def test_sampled_mtp_is_automatic_without_widening_greedy_policy():
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    sampled = resolve_speculative_mtp_serving_plan(
        evidence, key=_key(resident_capacity=4, sampling_mode="sampled"),
    )
    assert sampled.admitted and sampled.automatic_eligible
    wider = resolve_speculative_mtp_serving_plan(
        evidence, key=_key(resident_capacity=4, realized_group_rows=2, sampling_mode="sampled"),
    )
    assert wider.automatic_eligible


@pytest.mark.parametrize("seed", [0, 17, 2**64 - 1])
@pytest.mark.parametrize("step", [0, 1, 31, 1025])
def test_chain_seeds_are_exact_singleton_ar_rng_inputs(seed, step):
    from hipengine.runtime.native_sampler import native_chain_seeds

    mask = (1 << 64) - 1
    row_factor = 0xBF58476D1CE4E5B9
    step_factor = 0x9E3779B97F4A7C15
    actual = native_chain_seeds(seed, step, 4)
    assert actual.dtype == np.uint64
    for row, folded in enumerate(actual):
        graph_input = int(folded) ^ (((row + 1) * row_factor) & mask) ^ step_factor
        ar_input = seed ^ row_factor ^ (((step + row + 1) * step_factor) & mask)
        assert graph_input == ar_input


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_target_draw_prefix_walk_has_the_target_law(budget):
    from tests.test_unit_mtp_sampled_accept import _chain_batch

    batch = _chain_batch((0, 1))
    probabilities = ((0.3, 0.7), (0.6, 0.4), (0.8, 0.2))
    observed = defaultdict(float)
    for draws in product(range(2), repeat=3):
        mass = float(np.prod([probabilities[row][token] for row, token in enumerate(draws)]))
        result = batch.accept_from_top1(draws, remaining_decode=(budget,))
        tokens = result.accepted_tokens[0]
        bonus = result.next_tokens[0]
        visible = (*tokens, *(() if bonus is None else (bonus,)))
        observed[visible] += mass
    expected = {(1,): 0.7}
    if budget == 1:
        expected[(0,)] = 0.3
    else:
        expected[(0, 0)] = 0.3 * 0.6
        if budget == 2:
            expected[(0, 1)] = 0.3 * 0.4
        else:
            expected[(0, 1, 0)] = 0.3 * 0.4 * 0.8
            expected[(0, 1, 1)] = 0.3 * 0.4 * 0.2
    assert dict(observed) == pytest.approx(expected, abs=1e-15)


def test_native_plan_stages_without_advancing_live_rng_even_on_rollback():
    from hipengine.generation.qwen35_gguf_mtp2 import _DeviceSampledAcceptPlan
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.llm import SamplingParams

    state = RowSamplingState(seed=17)
    before = state.clone()
    plan = _DeviceSampledAcceptPlan(
        seed=state.seed, step_index=state.step_index,
        params=SamplingParams(temperature=0.7), rows=4,
    )
    assert state.step_index == before.step_index == plan.step_index
    assert state.random_unit() == before.random_unit()
    state.observe(5)
    assert plan.step_index == 0 and state.step_index == 1


@pytest.mark.parametrize("changes", [{"top_k": 8}, {"repetition_penalty": 1.1}])
def test_captured_chain_rejects_eager_shapes_before_staging(changes):
    from hipengine.llm import SamplingParams
    from hipengine.runtime.native_sampler import NativeSamplerChainWorkspace

    chain = NativeSamplerChainWorkspace.__new__(NativeSamplerChainWorkspace)
    chain.closed = False
    chain.rows = 4
    chain._upload = lambda *args: pytest.fail("unsupported shape staged data")
    with pytest.raises(NotImplementedError, match="full-vocabulary"):
        chain.stage(SimpleNamespace(
            params=SamplingParams(temperature=0.7, **changes), seed=17, step_index=0,
        ))


def test_chain_allocation_failure_releases_partial_workspace(monkeypatch):
    from hipengine.core.memory import DeviceBuffer
    import hipengine.runtime.native_sampler as native

    allocated = []
    freed = []

    def allocate(nbytes, *, runtime):
        if len(allocated) == 2:
            raise RuntimeError("allocation failed")
        buffer = DeviceBuffer(len(allocated) + 1, nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(native, "malloc", allocate)
    monkeypatch.setattr(native, "free", lambda buffer, **kwargs: freed.append(buffer.ptr))
    with pytest.raises(RuntimeError, match="allocation failed"):
        native.NativeSamplerChainWorkspace(
            runtime=object(), vocab_size=1027, rows=4, sampler_library=object(),
        )
    assert freed == [2, 1]


def test_sampled_mtp_output_decodes_all_committed_tokens(monkeypatch):
    import hipengine.generation.qwen35_gguf as gguf
    from hipengine.generation import GenerationRequest

    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(chr(64 + i) for i in ids))
    runner = gguf.Qwen35GGUFResidentModelRunner.__new__(gguf.Qwen35GGUFResidentModelRunner)
    runner.generator = SimpleNamespace(tokenizer=tokenizer)
    runner._request_diagnostics = lambda *args, **kwargs: {}
    monkeypatch.setattr(gguf, "_gguf_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(gguf, "_gguf_finish_details", lambda *args, **kwargs: None)
    row = gguf._GGUFResidentLoopRow(
        request_id=1, batch_id=1, row_index=0, prompt_ids=(7,),
        request=GenerationRequest(
            prompts=("p",), max_tokens=3, temperature=0.7, top_p=0.95, ignore_eos=True,
        ),
        native_greedy=False, native_sampled=True, native_sampler=True, submitted_at=0.0,
    )
    row.slot = SimpleNamespace(
        generated_ids=[1, 2, 3], timing={}, done=True,
        native_compact_prefill=True, native_decode_steps=1, serial_decode_steps=0,
    )
    row.samples = [SimpleNamespace(token_id=1, logprob=None, top_logprobs=())]
    row.mtp2_cycles = 1
    output = runner._native_output(row, SimpleNamespace(finish_reason="length"))
    assert output.text == "ABC"
    assert output.generated_token_ids == (1, 2, 3)
    assert output.token_logprobs == ()
    assert runner._native_stream_chunk(row).text == "C"


def test_eager_native_accept_reads_device_logits_without_uploading_or_advancing_state():
    from hipengine.core.memory import DeviceBuffer
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.llm import SamplingParams
    from tests.test_unit_mtp_sampled_accept import _chain_batch

    batch = _chain_batch((3, 4))
    state = RowSamplingState(seed=17, generated_tokens=(2,))
    observed = []
    draws = iter((3, 7, 2))

    def sample(ptr, params, prefix):
        observed.append((ptr, tuple(prefix.generated_tokens)))
        token = next(draws)
        prefix.observe(token)
        return SimpleNamespace(token_id=token)

    workspace = SimpleNamespace(
        vocab_size=8, sample=sample,
        _upload=lambda *args: pytest.fail("resident logits took a host round trip"),
    )
    row = SimpleNamespace(
        sampling_state=state, sampling_request=SamplingParams(temperature=0.7, top_k=8),
        native_sampler=True,
        lease=SimpleNamespace(session=SimpleNamespace(_native_sampler=lambda: workspace)),
    )
    prepared = SimpleNamespace(
        target_logits=np.empty((0, 0), dtype=np.float32),
        target_logits_device=DeviceBuffer(4096, batch.rows * 8 * 4),
    )
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    # The double carries the shape the method reads: no tokenizer,
    # so the walk builds no text observer and the native report stands.
    adapter.generator = SimpleNamespace(tokenizer=None)
    summary, row_metadata, forced_ids = adapter._sampled_accept_summary(
        row, prepared, batch, transaction_id=5, remaining_decode=3,
    )
    # This row asked for no logprobs, so the native per-row report is not kept.
    assert row_metadata is None
    assert summary.accepted_counts == (1,)
    assert summary.next_tokens == (7,)
    assert observed == [(4096, (2,)), (4128, (2, 3)), (4160, (2, 3, 4))]
    assert tuple(state.generated_tokens) == (2,)


def test_native_row_keeps_the_samplers_own_logprob_report():
    """A logprobs request reports the native sampler's value, not a rebuild.

    The autoregressive route for a native-sampler row reports the value the
    native sampler returned, so the speculative route has to keep that same
    report rather than recompute the distribution from the logits rows -- a host
    rebuild is a different implementation of the same law and is not what the
    request would have been served on the autoregressive path.
    """

    from hipengine.core.memory import DeviceBuffer
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.llm import SamplingParams
    from tests.test_unit_mtp_sampled_accept import _chain_batch

    batch = _chain_batch((3, 4))
    state = RowSamplingState(seed=17, generated_tokens=(2,))
    draws = iter((3, 7, 2))
    reported = {
        0: (0.25, ((3, 0.25), (4, 0.75))),
        1: (0.5, ((7, 0.5),)),
        2: (0.75, ((2, 0.75),)),
    }

    index = iter(range(3))

    def sample_indexed(ptr, params, prefix):
        position = next(index)
        token = next(draws)
        prefix.observe(token)
        logprob, top = reported[position]
        return SimpleNamespace(token_id=token, logprob=logprob, top_logprobs=top)

    workspace = SimpleNamespace(vocab_size=8, sample=sample_indexed)
    row = SimpleNamespace(
        sampling_state=state,
        sampling_request=SamplingParams(temperature=0.7, top_k=8, logprobs=True, top_logprobs=1),
        native_sampler=True,
        lease=SimpleNamespace(session=SimpleNamespace(_native_sampler=lambda: workspace)),
    )
    prepared = SimpleNamespace(
        target_logits=np.empty((0, 0), dtype=np.float32),
        target_logits_device=DeviceBuffer(4096, batch.rows * 8 * 4),
    )
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    # The double carries the shape the method reads: no tokenizer,
    # so the walk builds no text observer and the native report stands.
    adapter.generator = SimpleNamespace(tokenizer=None)
    _, row_metadata, forced_ids = adapter._sampled_accept_summary(
        row, prepared, batch, transaction_id=5, remaining_decode=3,
    )
    # The native path resolves no forced overrides: a request that carries one is
    # refused by ``supports_native_gpu_sampling`` before it can reach this shape.
    assert forced_ids is None
    assert row_metadata == (
        (0.25, ((3, 0.25), (4, 0.75))),
        (0.5, ((7, 0.5),)),
        (0.75, ((2, 0.75),)),
    )


def test_host_sampled_accept_resolves_forced_overrides_without_consuming_them():
    """The host path must resolve overrides from module scope, not a local import.

    A function-local import of the walk helper binds that name for the whole
    function, so the host path -- which shares the function with the native
    branch -- raised ``UnboundLocalError`` and fell back to autoregressive
    decoding. The gate caught it live; this pins it without a model load.
    """

    from hipengine.generation.sampling import RowSamplingState
    from hipengine.llm import SamplingParams
    from tests.test_unit_mtp_sampled_accept import _chain_batch

    batch = _chain_batch((2,))
    logits = np.full((2, 8), -8.0, dtype=np.float32)
    logits[:, 2] = 0.0
    state = RowSamplingState(
        seed=17, generated_tokens=(1,), forced_tokens_pending=(5,)
    )
    row = SimpleNamespace(
        sampling_state=state,
        sampling_request=SamplingParams(temperature=0.7, top_k=8),
        native_sampler=False,
    )
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(tokenizer=None)
    summary, row_metadata, forced_ids = adapter._sampled_accept_summary(
        row,
        SimpleNamespace(target_logits=logits),
        batch,
        transaction_id=5,
        remaining_decode=3,
    )
    assert forced_ids == (5, None)
    # The draft said 2 and the request forces 5, so the cycle corrects to 5.
    assert summary.accepted_counts == (0,)
    assert summary.next_tokens == (5,)
    assert row_metadata is None
    # Resolving the walk must not consume the live queue: only the commit does.
    assert state.forced_tokens == (5,)
