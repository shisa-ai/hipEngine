"""YuE2 AR generation-loop contracts, exercised on a deterministic CPU stand-in.

``generate_tokens`` is the loop that turns logits into a token trajectory: masks,
penalties, CFG, EOS, budget truncation, cancellation and the per-phase random
stream. None of that needs a GPU, so these tests drive it with a fake runtime
that returns hand-built logits and records every call. The GPU gate for the same
loop is the greedy oracle comparison in ``scripts/yue2_session_gate.py``.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest

import hipengine.generation.yue2 as generation_module
import hipengine.runtime.yue2_session as session_module

REPO = Path(__file__).resolve().parents[1]
VOCAB = 184704
ABC_END = 151848
MUSIC_END = 151852
MUSIC_START = 151851
CODEC_OFFSET = 151853


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    wide = array.view(np.uint32).astype(np.uint64)
    return (
        (wide + np.uint64(0x7FFF) + ((wide >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    ).astype(np.uint16)


class FakeRuntime:
    """Deterministic stand-in for ``Yue2ArRuntime`` with a scripted logit source."""

    branches = 2
    prefill_fallback_reason: str | None = None

    def __init__(self, logits=None, *, script=None, vocab: int = VOCAB):
        if (logits is None) == (script is None):
            raise ValueError("pass either a logits function or a script")
        self._logits = logits
        self._script = [np.asarray(row, dtype=np.float32) for row in (script or [])]
        self._calls = {0: 0, 1: 0}
        self.vocab = vocab
        self.lengths = {0: 0, 1: 0}
        self.prefills: list[tuple[int, int, int]] = []
        self.decoded: list[tuple[int, int, int]] = []
        self.resets = 0
        self.closed = False

    def reset(self, branch: int | None = None) -> None:
        self.resets += 1
        for index in list(self.lengths):
            if branch is None or branch == index:
                self.lengths[index] = 0

    def context_length(self, branch: int = 0) -> int:
        return self.lengths[branch]

    def prefill_host_rows(self, rows, *, branch: int = 0, start_pos: int = 0) -> None:
        rows = list(rows)
        self.prefills.append((len(rows), branch, start_pos))
        self.lengths[branch] = start_pos + len(rows)

    def embed_row(self, token_id: int) -> np.ndarray:
        row = np.zeros(8, dtype=np.float32)
        row[0] = float(token_id)
        return row

    def push_token(self, row, position: int, branch: int = 0) -> None:
        assert position == self.lengths[branch], "append-only positions"
        self.lengths[branch] += 1

    def forward_layers(self, position: int, branch: int = 0) -> None:
        self.decoded.append((position, branch, self.lengths[branch]))

    def logits(self, branch: int = 0, *, as_bf16: bool = True) -> np.ndarray:
        if self._script:
            index = min(self._calls[branch], len(self._script) - 1)
            self._calls[branch] += 1
            return _bf16_bits(self._script[index])
        return _bf16_bits(self._logits(branch, self.lengths[branch]))

    def close(self) -> None:
        self.closed = True


def _logits_for(token: int, vocab: int = VOCAB) -> np.ndarray:
    """A row whose only finite entry is ``token`` with the largest score."""

    row = np.full(vocab, -20.0, dtype=np.float32)
    row[token] = 5.0
    return row


def _sampling(**overrides):
    values = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": VOCAB,
        "repetition_penalty": 1.0,
        "penalty_window": 1,
        "min_tokens": 0,
        "max_tokens": 8,
    }
    values.update(overrides)
    return generation_module.Sampling(**values)


def test_end_token_exits_without_entering_history():
    runtime = FakeRuntime(lambda branch, length: _logits_for(ABC_END))
    tokens, timing, truncated = session_module.generate_tokens(
        runtime, [1, 2, 3], _sampling(), 7, "abc"
    )
    assert tokens == []
    assert truncated is False
    assert timing["output_tokens"] == 1
    assert timing["content_tokens"] == 0
    assert timing["cfg_branches"] == 1
    assert runtime.prefills == [(3, 0, 0)]


def test_budget_exit_is_reported_as_truncation():
    runtime = FakeRuntime(lambda branch, length: _logits_for(1000 + length))
    tokens, timing, truncated = session_module.generate_tokens(
        runtime, [1], _sampling(max_tokens=5), 7, "abc"
    )
    assert tokens == [1001, 1002, 1003, 1004, 1005]
    assert truncated is True
    assert timing["content_tokens"] == 5
    assert timing["output_tokens"] == 5


def test_end_token_is_callback_visible_once():
    seen = []
    runtime = FakeRuntime(lambda branch, length: _logits_for(ABC_END))
    session_module.generate_tokens(
        runtime,
        [1],
        _sampling(),
        7,
        "abc",
        on_token=lambda phase, token: seen.append((phase, token)),
    )
    assert seen == [("abc", ABC_END)]


def test_minimum_length_masks_the_end_token():
    runtime = FakeRuntime(lambda branch, length: _logits_for(ABC_END if length > 3 else 55))
    tokens, _, truncated = session_module.generate_tokens(
        runtime, [1], _sampling(min_tokens=3, max_tokens=6), 7, "abc"
    )
    # Three content tokens are forced before the end token is unmasked.
    assert tokens == [55, 55, 55]
    assert truncated is False


def test_phase_mask_blocks_the_other_phases_domain():
    runtime = FakeRuntime(lambda branch, length: _logits_for(CODEC_OFFSET + 5))
    tokens, _, truncated = session_module.generate_tokens(
        runtime, [1], _sampling(max_tokens=2), 7, "abc"
    )
    # The codec token is masked in the ABC phase, so the argmax stays in [0, EOD).
    assert all(0 <= token < 151643 for token in tokens)
    assert truncated is True


def test_semantic_phase_blocks_ordinary_text():
    runtime = FakeRuntime(lambda branch, length: _logits_for(10))
    tokens, _, _ = session_module.generate_tokens(
        runtime, [1], _sampling(max_tokens=2), 7, "semantic"
    )
    assert all(CODEC_OFFSET <= token < CODEC_OFFSET + 32768 for token in tokens)


def test_cfg_combines_both_branches_before_sampling():
    positive = _logits_for(11)
    negative = _logits_for(22)
    runtime = FakeRuntime(lambda branch, length: positive if branch == 0 else negative)
    tokens, timing, _ = session_module.generate_tokens(
        runtime,
        [1],
        _sampling(max_tokens=2),
        7,
        "abc",
        negative=[9],
        cfg_scale=1.01,
    )
    expected = generation_module.combine_cfg(positive, negative, 1.01)
    assert tokens == [int(np.argmax(expected)), int(np.argmax(expected))]
    assert timing["cfg_branches"] == 2
    assert runtime.prefills == [(1, 0, 0), (1, 1, 0)]
    # Both branches advance one position per emitted token.
    assert runtime.lengths == {0: 2, 1: 2}


def test_same_seed_replays_and_different_seed_diverges():
    row = np.full(VOCAB, -10.0, dtype=np.float32)
    row[CODEC_OFFSET : CODEC_OFFSET + 50] = 0.0
    sampling = _sampling(temperature=1.0, top_p=0.9, top_k=50, max_tokens=6)
    trajectories = []
    for seed in (1234, 1234, 5678):
        runtime = FakeRuntime(lambda branch, length, row=row: row)
        tokens, _, _ = session_module.generate_tokens(runtime, [1], sampling, seed, "semantic")
        trajectories.append(tuple(tokens))
    assert trajectories[0] == trajectories[1]
    assert trajectories[0] != trajectories[2]


def test_phase_streams_are_independent():
    row = np.full(VOCAB, -10.0, dtype=np.float32)
    row[CODEC_OFFSET : CODEC_OFFSET + 50] = 0.0
    sampling = _sampling(temperature=1.0, top_p=0.9, top_k=50, max_tokens=4)
    first = FakeRuntime(lambda branch, length: row)
    tokens_a, _, _ = session_module.generate_tokens(first, [1], sampling, 99, "semantic")
    # A different request in between must not perturb the next stream: the seed is
    # reset per phase, so the same seed and logits reproduce the same trajectory.
    other = FakeRuntime(lambda branch, length: row)
    session_module.generate_tokens(other, [2, 3], sampling, 100, "semantic")
    second = FakeRuntime(lambda branch, length: row)
    tokens_b, _, _ = session_module.generate_tokens(second, [1], sampling, 99, "semantic")
    assert tokens_a == tokens_b


def test_cancellation_before_prefill_and_mid_loop():
    runtime = FakeRuntime(lambda branch, length: _logits_for(1000 + length))
    with pytest.raises(InterruptedError):
        session_module.generate_tokens(
            runtime, [1], _sampling(), 7, "abc", cancelled=lambda: True
        )
    assert runtime.prefills == [], "cancelled request must not prefill"

    calls = {"count": 0}

    def cancelled() -> bool:
        calls["count"] += 1
        return calls["count"] > 3

    runtime = FakeRuntime(lambda branch, length: _logits_for(1000 + length))
    with pytest.raises(InterruptedError):
        session_module.generate_tokens(
            runtime, [1], _sampling(max_tokens=20), 7, "abc", cancelled=cancelled
        )
    assert runtime.lengths[0] < 20, "mid-loop cancellation stops the decode"


def test_budget_and_cfg_validation():
    runtime = FakeRuntime(lambda branch, length: _logits_for(1))
    sampling = _sampling()
    with pytest.raises(ValueError, match="empty prefix"):
        session_module.generate_tokens(runtime, [], sampling, 7, "abc")
    with pytest.raises(ValueError, match="exceeds 24576"):
        session_module.generate_tokens(runtime, list(range(24570)), sampling, 7, "abc")
    with pytest.raises(ValueError, match="CFG requires a negative prefix"):
        session_module.generate_tokens(runtime, [1], sampling, 7, "abc", cfg_scale=1.5)
    with pytest.raises(ValueError, match="Negative prefix"):
        session_module.generate_tokens(
            runtime, [1], sampling, 7, "abc", negative=list(range(24570)), cfg_scale=1.5
        )
    with pytest.raises(ValueError, match="phase must be"):
        session_module.generate_tokens(runtime, [1], sampling, 7, "nar")
    assert runtime.prefills == [], "validation failures must not touch the runtime"


def test_cfg_needs_two_branches():
    class SingleBranch(FakeRuntime):
        branches = 1

    runtime = SingleBranch(lambda branch, length: _logits_for(1))
    with pytest.raises(ValueError, match="branches=1"):
        session_module.generate_tokens(
            runtime, [1], _sampling(), 7, "abc", negative=[2], cfg_scale=1.5
        )


# ---------------------------------------------------------------------------
# staged session API
# ---------------------------------------------------------------------------


def _request(**overrides):
    values = {"style": "jazz", "lyrics": "la la", "cot": "off", "seed": 5, "id": "case"}
    values.update(overrides)
    return session_module.SongRequest(**values)


def test_off_mode_plan_has_no_abc_and_no_generation():
    runtime = FakeRuntime(lambda branch, length: _logits_for(1))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "unused"
    )
    plan = session.plan(_request())
    assert plan.abc is None and plan.abc_ids == ()
    assert plan.truncated is False
    assert plan.prefix == tuple(
        generation_module.token_prefixes(plan.request, lambda text: [7, 8])
    )
    assert runtime.prefills == [], "off mode must not run the AR loop"


def test_provided_abc_is_encoded_without_generation():
    runtime = FakeRuntime(lambda branch, length: _logits_for(1))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [11, 12, 13], decode=lambda ids: "unused"
    )
    plan = session.plan(_request(cot="melody", abc="X:1\nK:C\nCDEF"))
    assert plan.abc == "X:1\nK:C\nCDEF"
    assert plan.abc_ids == (11, 12, 13)
    assert plan.timing["external_prefix_tokens"] == 3
    assert runtime.prefills == []


def test_semantic_rejects_a_plan_whose_prefix_disagrees():
    runtime = FakeRuntime(lambda branch, length: _logits_for(MUSIC_END))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc"
    )
    plan = session.plan(_request())
    forged = session_module.SymbolicPlan(
        request=plan.request,
        abc=plan.abc,
        abc_ids=plan.abc_ids,
        prefix=(1, 2, 3),
        timing=plan.timing,
        truncated=plan.truncated,
    )
    with pytest.raises(ValueError, match="Plan prefix disagrees"):
        session.generate_semantic(forged)


def test_semantic_result_strips_the_codec_offset():
    runtime = FakeRuntime(
        script=[_logits_for(CODEC_OFFSET + 42), _logits_for(CODEC_OFFSET + 42), _logits_for(MUSIC_END)]
    )
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc"
    )
    plan = session.plan(_request())
    result = session.generate_semantic(plan, sampling=_sampling(max_tokens=4))
    assert result.tokens == (42, 42)
    assert result.truncated is False
    assert result.timing["content_tokens"] == 2


def test_plan_and_semantic_round_trip_through_disk(tmp_path):
    runtime = FakeRuntime(lambda branch, length: _logits_for(CODEC_OFFSET + 9))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc-text"
    )
    plan = session.plan(_request())
    result = session.generate_semantic(plan, sampling=_sampling(max_tokens=4))
    result.save(tmp_path)
    reloaded = session_module.SemanticResult.load(tmp_path)
    assert reloaded.tokens == result.tokens
    assert reloaded.truncated == result.truncated
    assert reloaded.plan.prefix == plan.prefix
    assert reloaded.plan.abc_ids == plan.abc_ids
    assert reloaded.plan.request.to_dict() == plan.request.to_dict()
    # A reloaded plan must still be accepted by the semantic stage.
    again = session.generate_semantic(reloaded.plan, sampling=_sampling(max_tokens=4))
    assert again.tokens == result.tokens


def test_loaded_plan_rejects_broken_token_domains(tmp_path):
    runtime = FakeRuntime(lambda branch, length: _logits_for(ABC_END))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc"
    )
    plan = session.plan(_request())
    payload = plan.to_dict()
    payload["prefix"] = [VOCAB] + list(payload["prefix"][1:])
    (tmp_path / "plan.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="vocabulary"):
        session_module.SymbolicPlan.load(tmp_path)
    payload = plan.to_dict()
    payload["abc_ids"] = [ABC_END] + list(payload["abc_ids"][1:])
    (tmp_path / "plan.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ordinary text"):
        session_module.SymbolicPlan.load(tmp_path)
    symbolic = session.plan(_request(cot="melody", abc="X:1\nK:C\nCDEF"))
    payload = symbolic.to_dict()
    payload["prefix"] = list(payload["prefix"])[:-2] + [3, MUSIC_START]
    (tmp_path / "plan.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact ABC IDs"):
        session_module.SymbolicPlan.load(tmp_path)


def test_loaded_plan_rejects_a_missing_symbolic_stage(tmp_path):
    runtime = FakeRuntime(lambda branch, length: _logits_for(ABC_END))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc"
    )
    payload = session.plan(_request(cot="off")).to_dict()
    payload["abc_ids"] = [7]
    (tmp_path / "plan.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="off mode"):
        session_module.SymbolicPlan.load(tmp_path)


def test_loaded_semantic_rejects_non_codec_tokens(tmp_path):
    runtime = FakeRuntime(script=[_logits_for(CODEC_OFFSET + 5), _logits_for(MUSIC_END)])
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc"
    )
    result = session.generate_semantic(
        session.plan(_request()), sampling=_sampling(max_tokens=2)
    )
    result.save(tmp_path)
    payload = json.loads((tmp_path / "semantic.json").read_text())
    payload["tokens"] = [CODEC_OFFSET]
    (tmp_path / "semantic.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="raw codec"):
        session_module.SemanticResult.load(tmp_path)


def test_session_resets_between_requests():
    runtime = FakeRuntime(lambda branch, length: _logits_for(MUSIC_END))
    session = session_module.Yue2ArSession(
        runtime, encode=lambda text: [7, 8], decode=lambda ids: "abc"
    )
    plan = session.plan(_request())
    session.generate_semantic(plan, sampling=_sampling())
    assert runtime.resets >= 1, "each phase starts from a clean context"
    assert runtime.lengths[0] == len(plan.prefix), "the second request starts from its own prefix"


def test_session_module_imports_no_torch():
    source = (REPO / "hipengine/runtime/yue2_session.py").read_text()
    assert "import torch" not in source
    assert "from torch" not in source
