"""CPU regressions for sampled MTP logits ownership across graph fallback."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.generation.sampling import RowSamplingState
from hipengine.kvcache.backend import ResourceClaimSet
from hipengine.llm import SamplingParams
from tests.test_unit_mtp_sampled_accept import _chain_batch
from tests.test_unit_qwen35_gguf_mtp2_c1_physical import _c1_plan


@pytest.mark.parametrize("adapter_budget", [2, 3])
def test_sampled_plan_preserves_logits_even_when_graph_is_unavailable(adapter_budget):
    batch = _chain_batch((3, 4))
    plan = replace(_c1_plan(5), candidate_counts=(2,), target_row_decomposition=(3,))
    claims = ResourceClaimSet(claim_id=plan.operation_id)
    observed = {}

    class PreparedBoundary(Exception):
        pass

    def prepare(batch, **kwargs):
        observed.update(kwargs)
        raise PreparedBoundary

    verifier = SimpleNamespace(graph_bucket=lambda *args: object(), prepare=prepare)
    state = SimpleNamespace(verifier=verifier, sampled_accept_plan=object(), proposal_device=None)
    row = SimpleNamespace(
        lease=SimpleNamespace(session=SimpleNamespace(position=0)),
        slot=SimpleNamespace(generated_ids=[2]),
        request=SamplingParams(max_tokens=12, temperature=0.7, top_p=0.9),
        native_sampler=True,
    )
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(_row=lambda rid: row)
    adapter._states = {5: state}
    adapter._active_claims = claims
    adapter._transaction_sequence = 0
    adapter.candidate_budget = adapter_budget
    adapter._sampled_route_request = lambda rid: True
    adapter._restore_provider_checkpoint = lambda state: None
    with pytest.raises(PreparedBoundary):
        adapter.execute_target_frontier(
            plan, SimpleNamespace(target_batch=batch), claims,
            commit=True, cancelled_request_ids=lambda: (),
        )
    assert observed["allow_graph"] is (adapter_budget == 2)
    # A staged sampled plan is not proof that a graph will execute. The eager
    # accept walk must receive resident logits, not an empty host placeholder.
    assert observed.get("return_device_logits") is True
    assert observed["return_logits"] is False


def test_native_accept_uploads_complete_host_logits_when_no_device_view_exists():
    batch = _chain_batch((3, 4))
    state = RowSamplingState(seed=17, generated_tokens=(2,))
    uploaded = []
    sampled = []
    draws = iter((3, 7, 2))

    def upload(name, logits):
        uploaded.append((name, logits.shape, logits.dtype))
        return DeviceBuffer(4096, logits.nbytes)

    def sample(ptr, params, prefix):
        sampled.append(ptr)
        token = next(draws)
        prefix.observe(token)
        return SimpleNamespace(token_id=token)

    workspace = SimpleNamespace(vocab_size=8, sample=sample, _upload=upload)
    row = SimpleNamespace(
        sampling_state=state,
        sampling_request=SamplingParams(temperature=0.7, top_p=0.9),
        native_sampler=True,
        lease=SimpleNamespace(session=SimpleNamespace(_native_sampler=lambda: workspace)),
    )
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(tokenizer=None)
    prepared = SimpleNamespace(target_logits=np.zeros((3, 8), dtype=np.float64))
    summary, metadata, forced = adapter._sampled_accept_summary(
        row, prepared, batch, transaction_id=5, remaining_decode=3,
    )
    assert uploaded == [("mtp_eager_logits", (3, 8), np.dtype("float32"))]
    assert sampled == [4096, 4128, 4160]
    assert summary.accepted_counts == (1,)
    assert summary.next_tokens == (7,)
    assert metadata is forced is None
    assert tuple(state.generated_tokens) == (2,)


@pytest.mark.parametrize("source", ["empty", "wrong_width", "short_rows", "flat", "short_device", "null_device"])
def test_native_accept_rejects_incomplete_logits_before_upload_or_sampling(source):
    batch = _chain_batch((3, 4))
    state = RowSamplingState(seed=17, generated_tokens=(2,))
    workspace = SimpleNamespace(
        vocab_size=8,
        sample=lambda *args: pytest.fail("invalid logits reached native sampler"),
        _upload=lambda *args: pytest.fail("invalid logits reached device upload"),
    )
    row = SimpleNamespace(
        sampling_state=state,
        sampling_request=SamplingParams(temperature=0.7, top_p=0.9),
        native_sampler=True,
        lease=SimpleNamespace(session=SimpleNamespace(_native_sampler=lambda: workspace)),
    )
    shapes = {"empty": (3, 0), "wrong_width": (3, 7), "short_rows": (2, 8), "flat": (24,)}
    device = None
    if source == "short_device":
        device = DeviceBuffer(4096, batch.rows * 8 * 4 - 4)
    elif source == "null_device":
        device = DeviceBuffer(0, batch.rows * 8 * 4)
    prepared = SimpleNamespace(
        target_logits=np.zeros(shapes.get(source, (3, 8)), dtype=np.float32),
        target_logits_device=device,
    )
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(tokenizer=None)
    with pytest.raises(RuntimeError, match="sampled MTP.*logits"):
        adapter._sampled_accept_summary(
            row, prepared, batch, transaction_id=5, remaining_decode=3,
        )
    assert tuple(state.generated_tokens) == (2,)
