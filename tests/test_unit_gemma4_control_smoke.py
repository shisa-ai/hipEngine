"""Unit coverage for the Gemma 4 control-smoke packet producer.

The GPU surface is the script's own run; these tests pin the parts that must
be right before any run is trusted: the route pin must be active *during*
every forward (it is the strict/production switch the manifests claim), the
schedule must assemble the shared control/row-spec contract, and the prompt
loader must keep messages so rows render through the artifact's chat template.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.gemma4_profiles import MOE_PREFILL_ENV
from scripts.gemma4_control_smoke import (
    _moe_prefill_width,
    _prompt_rows,
    _prompt_token_ids,
    _route_env,
    _trajectory_with_controls,
)

_VOCAB = 6


class _StubSession:
    """Minimal prefill/step contract that records the route pin per call."""

    def __init__(self) -> None:
        self.position = 0
        self.prefill_envs: list[str | None] = []
        self.step_envs: list[str | None] = []
        self.step_inputs: list[int] = []
        self._token_counter = 100

    def reset(self) -> None:
        self.position = 0

    def prefill(self, tokens, *, return_logits: bool = True):
        self.prefill_envs.append(os.environ.get(MOE_PREFILL_ENV))
        tokens = list(tokens)
        self.position += len(tokens)
        self._token_counter += 1
        return SimpleNamespace(
            logits=np.full(_VOCAB, float(self._token_counter), dtype=np.float32),
            token_id=self._token_counter,
        )

    def step(self, token_id: int, *, return_logits: bool = True):
        self.step_envs.append(os.environ.get(MOE_PREFILL_ENV))
        self.step_inputs.append(int(token_id))
        self.position += 1
        self._token_counter += 1
        return SimpleNamespace(
            logits=np.full(_VOCAB, float(self._token_counter), dtype=np.float32),
            token_id=self._token_counter,
        )


_PROMPT_IDS = [11, 12, 13, 14]


def _trajectory(
    session,
    *,
    forced,
    decode_steps,
    route_env,
    scenario_id="sc",
    teacher_chain=None,
    step_offset=0,
):
    return _trajectory_with_controls(
        session,
        prompt_ids=list(_PROMPT_IDS),
        forced_input_ids=forced,
        teacher_row_ids=teacher_chain,
        decode_steps=decode_steps,
        scenario_id=scenario_id,
        request_id="prompt-p0",
        route_top_k=8,
        graph_bucket="c1",
        rng_seed=0,
        route_env=route_env,
        step_offset=step_offset,
    )


def test_route_env_pins_during_the_block_and_restores_both_states() -> None:
    # Restores a pre-existing pin (the binder's production write).
    os.environ[MOE_PREFILL_ENV] = "auto"
    with _route_env({MOE_PREFILL_ENV: "grouped"}):
        assert os.environ[MOE_PREFILL_ENV] == "grouped"
    assert os.environ[MOE_PREFILL_ENV] == "auto"
    del os.environ[MOE_PREFILL_ENV]

    # And clears when the caller had nothing set.
    with _route_env({MOE_PREFILL_ENV: "auto"}):
        assert os.environ[MOE_PREFILL_ENV] == "auto"
    assert MOE_PREFILL_ENV not in os.environ


def test_route_env_restores_even_when_the_block_raises() -> None:
    os.environ[MOE_PREFILL_ENV] = "auto"
    with pytest.raises(RuntimeError):
        with _route_env({MOE_PREFILL_ENV: "grouped"}):
            raise RuntimeError("trajectory failed")
    assert os.environ[MOE_PREFILL_ENV] == "auto"
    del os.environ[MOE_PREFILL_ENV]


def test_greedy_trajectory_pins_the_route_during_every_forward() -> None:
    session = _StubSession()
    logits, controls, row_specs = _trajectory(
        session, forced=None, decode_steps=2,
        route_env={MOE_PREFILL_ENV: "grouped"},
    )

    # The pin must be live inside prefill and every decode step: it is the
    # strict/production switch the variant manifests claim for the capture.
    assert session.prefill_envs == ["grouped"]
    assert session.step_envs == ["grouped", "grouped"]

    assert logits.shape == (3, _VOCAB)  # prefill row + two decode rows
    assert len(controls) == len(row_specs) == 3
    assert [control["position"] for control in controls] == [
        len(_PROMPT_IDS) - 1, len(_PROMPT_IDS), len(_PROMPT_IDS) + 1,
    ]
    assert [control["context_length"] for control in controls] == [
        len(_PROMPT_IDS), len(_PROMPT_IDS) + 1, len(_PROMPT_IDS) + 2,
    ]
    assert [control["route_top_k"] for control in controls] == [8, 8, 8]
    assert [spec["shape"] for spec in row_specs] == ["prefill_last", "c1", "c1"]
    assert [spec["transition"] for spec in row_specs] == [
        "prefill_to_c1", "steady", "steady",
    ]
    assert [spec["teacher_step"] for spec in row_specs] == [0, 1, 2]
    # Greedy chain: each step's input is the previous emitted token.
    assert session.step_inputs == [
        row_specs[0]["teacher_token_id"], row_specs[1]["teacher_token_id"],
    ]


def test_forced_trajectory_records_the_teachers_rows_under_the_production_pin() -> None:
    session = _StubSession()
    # The strict chain as the strict capture recorded it: prefill emission
    # first, then one token per decode step. The candidate replays all but the
    # last as inputs and must record ALL of them as row teachers so
    # ``strict.rows == candidate.rows`` holds even when the arms diverge.
    teacher_chain = [55, 66, 77]
    _, controls, row_specs = _trajectory(
        session, forced=teacher_chain[:-1], decode_steps=2,
        teacher_chain=teacher_chain,
        route_env={MOE_PREFILL_ENV: "auto"},
    )

    assert session.prefill_envs == ["auto"]
    assert session.step_envs == ["auto", "auto"]
    assert session.step_inputs == teacher_chain[:-1]
    assert [control["input_token_id"] for control in controls] == [
        _PROMPT_IDS[-1], *teacher_chain[:-1],
    ]
    # The rows carry the teacher chain, NOT this arm's own emissions (the stub
    # emits 101, 102, 103 -- none of which may appear).
    assert [spec["teacher_token_id"] for spec in row_specs] == teacher_chain
    assert len(row_specs) == 3


def test_forced_trajectory_requires_a_wellformed_teacher_chain() -> None:
    session = _StubSession()
    with pytest.raises(ValueError, match="teacher chain"):
        _trajectory(
            session, forced=[1, 2], decode_steps=2,
            route_env={MOE_PREFILL_ENV: "auto"},
        )
    with pytest.raises(ValueError, match="prefill emission"):
        _trajectory(
            session, forced=[1, 2], decode_steps=2, teacher_chain=[55, 66],
            route_env={MOE_PREFILL_ENV: "auto"},
        )
    # A teacher chain without forced inputs makes no sense either.
    with pytest.raises(ValueError, match="only valid with forced"):
        _trajectory(
            session, forced=None, decode_steps=2, teacher_chain=[55, 66, 77],
            route_env={MOE_PREFILL_ENV: "grouped"},
        )


def _write_suite(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_prompt_rows_keeps_messages_and_enforces_exactly_one_field(tmp_path) -> None:
    suite = _write_suite(
        tmp_path / "suite.jsonl",
        [
            {"id": "a", "category": "code",
             "messages": [{"role": "user", "content": "hi"}]},
            {"id": "b", "prompt": "plain", "category": "general_en"},
        ],
    )
    rows = _prompt_rows([suite], limit=10)
    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert rows[0]["prompt"] is None
    assert rows[1]["messages"] is None
    assert rows[1]["prompt"] == "plain"

    assert len(_prompt_rows([suite], limit=1)) == 1

    both = _write_suite(
        tmp_path / "both.jsonl",
        [{"id": "c", "prompt": "x", "messages": [{"role": "user", "content": "y"}]}],
    )
    with pytest.raises(SystemExit, match="exactly one"):
        _prompt_rows([both], limit=10)

    duplicate = tmp_path / "dup.jsonl"
    duplicate.write_text(
        '{"id": "a", "prompt": "one"}\n{"id": "a", "prompt": "two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="duplicate prompt id"):
        _prompt_rows([duplicate], limit=10)


def test_prompt_token_ids_renders_messages_and_encodes_raw_prompts() -> None:
    class _Tokenizer:
        chat_template = "{% for m in messages %}{{ m.role }}:{{ m.content }}|{% endfor %}"
        seen: list[str] = []

        def encode(self, text: str, *, add_special_tokens: bool = False):
            assert add_special_tokens is False
            self.seen.append(text)
            return [ord(char) for char in text]

    tokenizer = _Tokenizer()
    rendered = _prompt_token_ids(
        tokenizer,
        {"messages": [{"role": "user", "content": "hello"}], "prompt": None},
    )
    assert tokenizer.seen[-1] == "user:hello|"
    assert rendered == [ord(char) for char in "user:hello|"]

    raw = _prompt_token_ids(tokenizer, {"messages": None, "prompt": "plain text"})
    assert tokenizer.seen[-1] == "plain text"
    assert raw == [ord(char) for char in "plain text"]


class _CountingTokenizer:
    """Encodes one token per whitespace word so padding counts are checkable."""

    chat_template = "{{ messages[0].content }}"

    def encode(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return list(range(len(text.split())))


def test_prompt_token_ids_pads_by_cycling_the_rows_own_text() -> None:
    tokenizer = _CountingTokenizer()
    # Natural length: 3 tokens ("aa bb cc").
    natural = _prompt_token_ids(tokenizer, {"messages": None, "prompt": "aa bb cc"})
    assert natural == [0, 1, 2]

    padded = _prompt_token_ids(
        tokenizer, {"messages": None, "prompt": "aa bb cc"}, min_tokens=10
    )
    assert len(padded) >= 10

    # Messages rows re-render through the template with the repeated content.
    padded_messages = _prompt_token_ids(
        tokenizer,
        {"messages": [{"role": "user", "content": "aa bb cc"}], "prompt": None},
        min_tokens=10,
    )
    assert len(padded_messages) >= 10

    # Short enough prompts are untouched.
    untouched = _prompt_token_ids(
        tokenizer, {"messages": None, "prompt": "aa bb cc"}, min_tokens=2
    )
    assert untouched == [0, 1, 2]


def test_moe_prefill_width_follows_the_artifacts_routing() -> None:
    # lanes >= 16 * experts and lanes = tokens * used, so tokens = 16*e/u:
    # the shipped 128/8 artifact first engages the MMQ/WMMA plans at 256.
    assert _moe_prefill_width(
        {"gemma4.expert_count": 128, "gemma4.expert_used_count": 8}
    ) == 256
    assert _moe_prefill_width(
        {"gemma4.expert_count": 8, "gemma4.expert_used_count": 4}
    ) == 32
    # No usable routing metadata: nothing to derive, caller stays on the
    # campaign chain width.
    assert _moe_prefill_width({}) == 0
    assert _moe_prefill_width({"gemma4.expert_count": 128}) == 0
    assert _moe_prefill_width({"gemma4.expert_count": "n/a"}) == 0
    assert _moe_prefill_width(
        {"gemma4.expert_count": 128, "gemma4.expert_used_count": "n/a"}
    ) == 0


def test_default_prompt_tokens_is_protocol_fixed_at_the_campaign_chain() -> None:
    from scripts.gemma4_control_smoke import CAMPAIGN_GATE_CHAIN_TOKENS, _default_prompt_tokens

    # Normal artifacts: the registered 2048-token gate chain, never the
    # route minimum, so the width cannot move with a verdict already seen.
    target, source = _default_prompt_tokens(
        {"gemma4.expert_count": 128, "gemma4.expert_used_count": 8}
    )
    assert target == CAMPAIGN_GATE_CHAIN_TOKENS == 2048
    assert "campaign-gate-chain" in source and "route min 256" in source
    # An artifact whose MoE gate sits above the campaign chain raises the
    # target and switches the source.
    target, source = _default_prompt_tokens(
        {"gemma4.expert_count": 1024, "gemma4.expert_used_count": 4}
    )
    assert target == 4096  # ceil(16 * 1024 / 4)
    assert source.startswith("route-min")
    # Missing metadata: fall back to the campaign chain.
    target, source = _default_prompt_tokens({})
    assert target == CAMPAIGN_GATE_CHAIN_TOKENS


def test_trajectory_places_its_rows_in_the_requested_step_band() -> None:
    session = _StubSession()
    _, controls, row_specs = _trajectory(
        session, forced=None, decode_steps=2,
        route_env={MOE_PREFILL_ENV: "grouped"}, step_offset=6,
    )
    assert [control["scenario_step"] for control in controls] == [6, 7, 8]
    assert [spec["scenario_step"] for spec in row_specs] == [6, 7, 8]
    # teacher_step stays per-request: it indexes the chain, not the scenario.
    assert [spec["teacher_step"] for spec in row_specs] == [0, 1, 2]


def test_multi_prompt_fixture_offsets_avoid_the_slot_collision_the_gate_raises():
    """Two prompts sharing one scenario must not both occupy step 0 slot 0.

    This is the exact failure the gate reported on the first multi-prompt
    packet: ``summarize_scenario`` rejects two active records on the same
    physical slot within one step, and the default schedule starts every
    request at step 0.
    """

    from hipengine.benchmark.control_capture import schedule_c1_control_records
    from hipengine.benchmark.execution_profiles import summarize_scenario

    def _schedule(*, step_offset: int, request_id: str):
        return schedule_c1_control_records(
            scenario_id="sc",
            request_id=request_id,
            prompt_ids=[11, 12, 13, 14],
            teacher_token_ids=[7, 9],
            route_top_k=8,
            graph_bucket="c1",
            rng_seed=0,
            step_offset=step_offset,
        )

    # Distinct requests, both starting at step 0: the gate's actual failure.
    first = _schedule(step_offset=0, request_id="prompt-a")
    with pytest.raises(ValueError, match="physical-slot collision"):
        summarize_scenario(
            first + _schedule(step_offset=0, request_id="prompt-b")
        )

    # Three records wide, so the next request starts at band 3.
    second = _schedule(step_offset=len(first), request_id="prompt-b")
    summary = summarize_scenario(first + second)
    assert summary["width_sequence"] == [1] * (2 * len(first))