from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import (
    SpeculativeProviderConfig,
    register_builtin_speculative_providers,
    resolve_speculative_provider,
)


_RUNTIME = object()


class _Tokenizer:
    eos_token_id = 99_999

    def encode(self, text: str):
        return [1, 2]

    def decode(self, token_ids, *, skip_special=False):
        return " ".join(str(int(token)) for token in token_ids)


class _TargetRunner:
    def __init__(self) -> None:
        self.max_sequence_length = 1_024
        self.position = 0
        self.truth = (10, 11, 12, 13, 14)
        self.cursor = 0
        self.runtime = _RUNTIME
        self.last_target_hidden = SimpleNamespace(ptr=123)
        self.capture_hidden_seed_calls = []

    def _hidden(self, value: int) -> np.ndarray:
        return np.full(8, float(value), dtype=np.float32)

    def prefill(self, token_ids, *, capture_hidden_seeds=False):
        self.position = len(token_ids)
        self.cursor = 0
        hidden = np.stack([self._hidden(token) for token in token_ids])
        return SimpleNamespace(
            token_id=self.truth[0],
            hidden_seeds=hidden,
            hidden_seed=hidden[-1],
        )

    def step(self, token_id: int, *, capture_hidden_seed=False):
        self.capture_hidden_seed_calls.append(bool(capture_hidden_seed))
        assert int(token_id) == self.truth[self.cursor]
        self.cursor += 1
        self.position += 1
        hidden = self._hidden(token_id)
        return SimpleNamespace(
            token_id=self.truth[self.cursor],
            hidden_seeds=hidden.reshape(1, -1),
            hidden_seed=hidden,
        )


class _DraftRunner:
    max_sequence_length = 1_024

    def __init__(self) -> None:
        self.position = 0
        self.proposals = [(11, 98), (13, 14)]
        self.proposal_index = 0
        self.trimmed = []
        self.last_proposal_stage_timings_ms = {}
        self.runtime = _RUNTIME
        self.target_hidden_seed_ptrs = []

    def prime_prompt(self, token_ids, hidden_rows):
        assert hidden_rows.shape == (len(token_ids), 8)
        self.position = len(token_ids)
        self.proposal_index = 0

    def propose_chain(
        self,
        *,
        start_token,
        target_hidden_seed,
        draft_n_max,
        target_hidden_seed_ptr=None,
    ):
        self.target_hidden_seed_ptrs.append(target_hidden_seed_ptr)
        values = self.proposals[self.proposal_index][:draft_n_max]
        self.proposal_index += 1
        self.position += len(values)
        self.last_proposal_stage_timings_ms = {
            "draft_input_fusion": 1.0,
            "draft_layer": 2.0,
            "draft_head": 3.0,
            "draft_logits_d2h": 4.0,
            "draft_hidden_d2h": 5.0,
            "draft_sampler": 0.5,
        }
        return tuple(
            SimpleNamespace(
                token_id=token,
                hidden_seed=np.full(8, token, dtype=np.float32),
            )
            for token in values
        )

    def trim(self, position):
        self.position = int(position)
        self.trimmed.append(self.position)

    def close(self):
        pass


class _TargetGenerator:
    backend = "hip_gfx1151"

    def __init__(self) -> None:
        self.runner = _TargetRunner()
        self.tokenizer = _Tokenizer()


def _request() -> GenerationRequest:
    return GenerationRequest(
        prompts=((1, 2),),
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        top_k=1,
    )


def test_qwen4_exp_mtp_provider_keeps_exact_target_output_and_trims_draft() -> None:
    target = _TargetGenerator()
    draft = _DraftRunner()
    provider = Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model="/tmp/fake.gguf", candidate_budget=2
        ),
        draft_runner=draft,
        draft_resident=object(),
    )

    output = provider.generate_detailed(_request())[0]

    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert output.text == "10 11 12 13 14"
    assert [cycle.accepted for cycle in provider.last_cycles] == [1, 2]
    assert provider.last_cycles[0].mismatch_token == 12
    assert output.telemetry is not None
    assert output.telemetry.diagnostics["proposed_draft_tokens"] == 4
    assert output.telemetry.diagnostics["accepted_draft_tokens"] == 3
    assert output.telemetry.diagnostics["draft_acceptance"] == 0.75
    phase = output.telemetry.diagnostics["phase_census"]
    assert phase["cycles"] == 2
    assert phase["target_verify_rows"] == 4
    assert phase["proposal"]["calls"] == 2
    assert phase["target_verify"]["calls"] == 4
    assert phase["acceptance_control"]["calls"] == 4
    assert phase["draft_commit_or_rollback"]["calls"] == 2
    assert phase["draft_stages_ms"]["draft_input_fusion"] == 2.0
    assert phase["draft_stages_ms"]["draft_layer"] == 4.0
    assert phase["draft_stages_ms"]["draft_head"] == 6.0
    assert phase["draft_stages_ms"]["draft_logits_d2h"] == 8.0
    assert phase["draft_stages_ms"]["draft_hidden_d2h"] == 10.0
    assert phase["draft_stages_ms"]["draft_sampler"] == 1.0
    assert draft.trimmed == [4, 6]
    assert draft.position == target.runner.position == 6
    capability = provider.capabilities()
    assert capability["strict_fallback"] == "target_ar"
    assert capability["streaming_mode"] == "buffered_public"

    chunks = list(provider.stream_detailed(_request()))
    assert len(chunks) == 1
    assert chunks[0].generated_token_ids == (10, 11, 12, 13, 14)


def test_qwen4_exp_mtp_provider_uses_resident_target_hidden_when_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT", "1")
    target = _TargetGenerator()
    draft = _DraftRunner()
    provider = Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model="/tmp/fake.gguf", candidate_budget=2
        ),
        draft_runner=draft,
        draft_resident=object(),
    )

    output = provider.generate_detailed(_request())[0]

    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert draft.target_hidden_seed_ptrs == [123, 123]
    assert target.runner.capture_hidden_seed_calls == [False, False, False, False]
    assert output.telemetry.diagnostics["target_hidden_handoff"] == "device_to_device"


def test_qwen4_exp_mtp_provider_is_registered_for_operational_quant() -> None:
    register_builtin_speculative_providers()
    factory = resolve_speculative_provider(
        provider="qwen4_exp_mtp",
        target_model="qwen4_exp_gguf",
        backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl",
    )
    assert callable(factory)
