from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pytest

from hipengine.generation.deadline import (
    GenerationCancelled,
    GenerationCancellationToken,
)
from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import (
    SpeculativeProviderConfig,
    register_builtin_speculative_providers,
    resolve_speculative_provider,
)


class _Tokenizer:
    eos_token_id = 99_999

    def encode(self, text: str):
        return [1, 2]

    def decode(self, token_ids, *, skip_special=False):
        return " ".join(str(int(token)) for token in token_ids)


class _TargetRunner:
    def __init__(self) -> None:
        self.max_sequence_length = 4_096
        self.position = 0
        self.truth = (10, 11, 12, 13, 14)
        self.cursor = 0

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
    max_sequence_length = 4_096

    def __init__(self) -> None:
        self.position = 0
        self.proposals = [(11, 98), (13, 14)]
        self.proposal_index = 0
        self.trimmed = []
        self.calls = []
        self.last_proposal_stage_timings_ms = {}

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
        compact_output=False,
    ):
        values = self.proposals[self.proposal_index][:draft_n_max]
        self.proposal_index += 1
        self.position += len(values)
        self.calls.append(
            {
                "start_token": int(start_token),
                "draft_n_max": int(draft_n_max),
                "compact_output": bool(compact_output),
                "hidden_seed": (
                    None if target_hidden_seed is None
                    else tuple(float(value) for value in np.asarray(target_hidden_seed).reshape(-1))
                ),
            }
        )
        self.last_proposal_stage_timings_ms = {
            "draft_input_fusion": 1.0,
            "draft_layer": 2.0,
            "draft_head": 3.0,
            "draft_logits_d2h": 4.0,
            "draft_hidden_d2h": 5.0,
            "draft_sampler": 0.5,
        }
        if compact_output:
            # The runner's compact contract: token ids only, no logits and no
            # per-candidate hidden seed.
            return tuple(_CompactRow(token) for token in values)
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


class _CompactRow:
    """A compact draft row: token id only, and reading anything else fails.

    The runtime's ``compact_output`` path returns
    ``Qwen4ExpMTPDraftResult(token_id, None, None)``. Raising on the unused
    fields proves the provider consumes nothing but the token id.
    """

    def __init__(self, token_id: int) -> None:
        self.token_id = int(token_id)

    def __getattr__(self, name):
        raise AssertionError(
            f"the provider read {name!r} from a compact proposal row")


class _TargetGenerator:
    backend = "hip_gfx1151"

    def __init__(self) -> None:
        self.runner = _TargetRunner()
        self.tokenizer = _Tokenizer()


def _request(**overrides) -> GenerationRequest:
    return GenerationRequest(
        prompts=((1, 2),),
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        top_k=1,
        **overrides,
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
    from hipengine.server.api import _mtp_response_summary, _mtp_accepted_rejected_counts

    summary = _mtp_response_summary("speculative_mtp", [output])
    assert summary["used"] is True
    assert summary["draft_tokens"] == 4
    assert summary["accepted_draft_tokens"] == 3
    assert summary["rejected_draft_tokens"] == 1
    assert summary["draft_cycles"] == 2
    assert _mtp_accepted_rejected_counts([output]) == (3, 1)
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
    assert capability["streaming_mode"] == "committed_tokens"
    assert capability["context_limit"] == 4_096

    chunks = list(provider.stream_detailed(_request()))
    assert len(chunks) > 1
    assert ''.join(chunk.text for chunk in chunks) == output.text
    assert chunks[-1].generated_token_ids == output.generated_token_ids
    assert chunks[-1].telemetry.timing == output.telemetry.timing


def test_qwen4_exp_mtp_crosses_the_old_1k_boundary() -> None:
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

    request = GenerationRequest(
        prompts=(tuple(range(1_020)),),
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        top_k=1,
    )
    output = provider.generate_detailed(request)[0]
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert provider.capabilities()["context_limit"] == 4_096


def test_qwen4_exp_mtp_provider_is_registered_for_operational_quant() -> None:
    register_builtin_speculative_providers()
    factory = resolve_speculative_provider(
        provider="qwen4_exp_mtp",
        target_model="qwen4_exp_gguf",
        backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl",
    )
    assert callable(factory)


# --- compact proposals are the provider's own contract ----------------------


def _provider():
    target = _TargetGenerator()
    draft = _DraftRunner()
    provider = Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model="/tmp/fake.gguf",
            candidate_budget=2),
        draft_runner=draft,
        draft_resident=object(),
    )
    return provider, target, draft


@pytest.mark.parametrize("value", [None, "", "0", "false", "False", "1", "yes"])
def test_provider_proposals_are_compact_independent_of_the_process_env(
        monkeypatch, value):
    # The rollback lever is gone: the provider states the contract itself, so a
    # leftover process value cannot select the logits path.
    if value is None:
        monkeypatch.delenv("HIPENGINE_QWEN4_EXP_MTP_COMPACT_PROPOSAL", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MTP_COMPACT_PROPOSAL", value)
    provider, target, draft = _provider()
    output = provider.generate_detailed(_request())[0]
    assert [call["compact_output"] for call in draft.calls] == [True, True]
    # Token behaviour is unchanged by the contract: the target still owns
    # every emitted token, and the acceptance accounting is the same.
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert [cycle.accepted for cycle in provider.last_cycles] == [1, 2]
    assert output.telemetry.diagnostics["proposed_draft_tokens"] == 4
    assert output.telemetry.diagnostics["accepted_draft_tokens"] == 3
    assert draft.position == target.runner.position


def test_compact_proposal_rows_are_consumed_by_token_id_alone():
    # ``_CompactRow`` raises on any read other than ``token_id``, which is the
    # runtime's compact contract (``Qwen4ExpMTPDraftResult(token, None, None)``).
    provider, _target, draft = _provider()
    output = provider.generate_detailed(_request())[0]
    assert all(call["compact_output"] for call in draft.calls)
    assert output.generated_token_ids == (10, 11, 12, 13, 14)


def test_each_compact_proposal_receives_the_previous_accepted_target_hidden():
    provider, _target, draft = _provider()
    provider.generate_detailed(_request())[0]
    # Cycle one seeds from the last prompt hidden; cycle two seeds from the
    # hidden of the token the target accepted as the next root.
    assert [call["start_token"] for call in draft.calls] == [10, 12]
    assert [set(call["hidden_seed"]) for call in draft.calls] == [
        {2.0}, {11.0}]


def test_cancellation_between_compact_cycles_leaves_the_draft_at_the_trim():
    token = GenerationCancellationToken()
    provider, target, draft = _provider()
    steps = provider.iter_generate(
        (1, 2), _request(cancellation_token=token))
    assert next(steps).token_ids == (10,)
    assert draft.calls == []

    token.cancel()
    with pytest.raises(GenerationCancelled):
        next(steps)
    # No proposal ran for the cancelled cycle, and the cursor still matches the
    # target's committed position, so a replay starts from the same state.
    assert draft.calls == []
    assert draft.position == target.runner.position == 2
    steps.close()
