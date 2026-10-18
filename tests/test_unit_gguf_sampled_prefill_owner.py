"""The logits producer owns packed sampling; the slot owns the selected token."""

from collections import Counter
from types import SimpleNamespace

import pytest

from hipengine.generation import GenerationRequest
from hipengine.generation import qwen35_gguf as gguf
from hipengine.generation.sampling import SampleResult, SamplingMode


@pytest.mark.parametrize("shared_owner", [False, True])
def test_native_prefill_samples_from_logits_owner_and_commits_to_slot(
    monkeypatch, shared_owner
):
    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    calls = []

    class Session:
        position = 0
        # Force block-table prefill even for a standalone session.
        kv_attention_source = "int8_direct"
        _device_kv_allocation = SimpleNamespace(block_ids=(8,), chunk_start_block_id=8)
        logits_ready = False

        def prefill_batch_native(self, prompts, *, sessions, **kwargs):
            assert kwargs["require_logits"] is True
            assert kwargs["return_logits"] is False
            assert sessions == [slot]
            sessions[0].position = len(prompts[0])
            self.logits_ready = True
            return [SimpleNamespace(token_id=0, logits=None)]

        def sample_native_from_packed_logits(self, row, params, state, *, output_session):
            assert self.logits_ready, "sampling read a slot instead of the packed logits owner"
            assert row == 0
            assert output_session is slot
            calls.append((self, output_session))
            state.observe(7)
            return SampleResult(
                token_id=7, logit=1.0, logprob=-0.2,
                mode=SamplingMode.GPU_SAMPLE, candidate_count=2,
            )

    slot = Session()
    owner = Session() if shared_owner else slot
    request = GenerationRequest(
        prompts=((10, 11),), max_tokens=2, temperature=0.7, top_p=0.95,
        ignore_eos=True,
    )
    row = gguf._GGUFResidentLoopRow(
        request_id=1, batch_id=0, row_index=0, request=request,
        prompt_ids=(10, 11), native_greedy=False, native_sampled=True,
        submitted_at=0.0,
        lease=gguf._GGUFResidentSessionLease(session=slot, pool_key=("test",)),
    )
    runner = gguf.Qwen35GGUFResidentModelRunner.__new__(gguf.Qwen35GGUFResidentModelRunner)
    runner.generator = SimpleNamespace(tokenizer=SimpleNamespace(
        eos_token_id=99, decode=lambda ids, **kwargs: "sample",
    ))
    runner._resident_batch_owner = owner if shared_owner else None
    runner._route_counts = Counter()
    runner._fallback_reasons = Counter()
    runner._refresh_prefix_cache = lambda row: None

    runner._prefill_sampled_row(row)

    assert calls == [(owner, slot)]
    assert row.slot.generated_ids == [7]
    assert row.sampling_state.generated_tokens == [7]
    assert row.full_vocab_logits_d2h is False
    assert row.logits_d2h_bytes == 0
    assert runner._route_counts["native_sampler_row_launches"] == 1
