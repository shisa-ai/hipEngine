"""The compatibility wrapper preserves cold validation and provider width."""
from __future__ import annotations

import pytest

from hipengine.generation.engine_loop import SubmitPollTextGenerator
from hipengine.generation.registry import GenerationRequest


class _ValidatingGenerator:
    server_mtp_batch_max_active_requests = 1

    def generate(self, request):
        raise AssertionError("invalid request reached generation")

    def validate_request(self, request):
        raise NotImplementedError("unsupported control before admission")


def test_legacy_provider_width_is_forwarded():
    wrapper = SubmitPollTextGenerator(_ValidatingGenerator(), capacity=4)
    assert wrapper.server_mtp_batch_max_active_requests == 1


def test_ar_validation_precedes_scheduler_admission():
    wrapper = SubmitPollTextGenerator(_ValidatingGenerator(), capacity=4)
    with pytest.raises(NotImplementedError, match="before admission"):
        wrapper.submit_detailed(GenerationRequest(prompts=("hello",), max_tokens=4, temperature=0, top_p=1, ignore_eos=False))
    assert not wrapper._loop.scheduler.pending_requests
