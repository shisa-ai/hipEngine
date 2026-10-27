"""Index-only capacity mode and the bounded filler search in the QSA gate.

``--index-only`` prefills an exactly ``--target-tokens`` prompt and stops. It
makes one claim: the device index selection equals the CPU reference at that
depth. Generation, retrieval, repeat, and rollback are not exercised, so they
must be reported as ``not_exercised`` and must never satisfy a pass predicate.
"""

from __future__ import annotations

import pytest

from scripts import qwen4_exp_qsa_retrieval_gate as gate


class _WhitespaceTokenizer:
    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()


class _CharCountTokenizer:
    """O(1) tokenizer: one token per character, so a 262K prompt stays cheap."""

    calls = 0

    def encode(self, text: str) -> object:
        type(self).calls += 1

        class _Counted:
            def __len__(self) -> int:
                return len(text)

        return _Counted()


class _Binding:
    def __init__(self, index: int) -> None:
        self.qsa_state_index = index


def _reference_prompt(tokenizer: object, *, target_tokens: int) -> tuple[str, int]:
    """The linear builder this change replaced, kept as the oracle.

    Same record placement, same one-filler-at-a-time walk, same ``" a"``
    refinement. The optimized builder must return byte-identical output for any
    target it can serve.
    """

    encode = tokenizer.encode
    needle_record = gate._archive_record(gate._NEEDLE_RECORD_INDEX)
    before: list[str] = []
    for index in range(gate._NEEDLE_RECORD_INDEX):
        candidate = before + [gate._archive_record(index)]
        if (
            len(encode(gate._render_retrieval_prompt(candidate + [needle_record])))
            > target_tokens
        ):
            break
        before = candidate
    after: list[str] = []
    for index in range(gate._NEEDLE_RECORD_INDEX + 1, gate._ARCHIVE_RECORD_COUNT):
        candidate = after + [gate._archive_record(index)]
        if (
            len(
                encode(
                    gate._render_retrieval_prompt(
                        before + [needle_record] + candidate
                    )
                )
            )
            > target_tokens
        ):
            break
        after = candidate
    records = before + [needle_record] + after
    while (
        len(encode(gate._render_retrieval_prompt(records + [gate._PROMPT_FILLER])))
        <= target_tokens
    ):
        records.append(gate._PROMPT_FILLER)
    prompt = gate._render_retrieval_prompt(records)
    for count in range(64):
        candidate = gate._render_retrieval_prompt(records + [" a" * count])
        if (
            len(encode(candidate)) <= target_tokens
            and len(encode(candidate)) > len(encode(prompt))
        ):
            prompt = candidate
    needle = len(encode(prompt[: prompt.index(gate._NEEDLE_CODE)]))
    return prompt, needle


def test_bounded_filler_search_matches_the_linear_walk_exactly() -> None:
    tokenizer = _WhitespaceTokenizer()
    minimum = gate.retrieval_prompt_minimum_tokens(tokenizer)

    for target in (minimum, minimum + 7, minimum + 25, minimum + 100, minimum + 900, 2_000):
        prompt, needle = gate.build_retrieval_prompt(tokenizer, target_tokens=target)
        reference, reference_needle = _reference_prompt(tokenizer, target_tokens=target)
        assert prompt == reference, target
        assert needle == reference_needle, target


def test_filler_search_is_logarithmic_not_one_encode_per_filler() -> None:
    _CharCountTokenizer.calls = 0
    tokenizer = _CharCountTokenizer()
    records = [gate._archive_record(index) for index in range(5)]

    count = gate._filler_count(tokenizer.encode, records, target_tokens=262_144)

    # A linear walk would need ~2_900 encodes at this target; the probe needs
    # one per bit of the answer plus the closing window.
    assert _CharCountTokenizer.calls <= 40
    assert count > 2_000
    rendered = gate._render_retrieval_prompt(records + [gate._PROMPT_FILLER] * count)
    assert len(rendered) <= 262_144
    rendered_more = gate._render_retrieval_prompt(
        records + [gate._PROMPT_FILLER] * (count + 1)
    )
    assert len(rendered_more) > 262_144


def test_full_builder_stays_bounded_at_a_full_capacity_target() -> None:
    _CharCountTokenizer.calls = 0
    prompt, needle = gate.build_retrieval_prompt(
        _CharCountTokenizer(), target_tokens=262_144
    )

    assert len(prompt) <= 262_144
    assert prompt[needle : needle + len(gate._NEEDLE_CODE)] == gate._NEEDLE_CODE
    # Record placement is bounded by the archive, the filler probe is
    # logarithmic, and the " a" refinement is a fixed 64-step walk.
    assert _CharCountTokenizer.calls < 250


def test_prefill_progress_line_reports_share_counts_and_elapsed() -> None:
    line = gate._prefill_progress_line(131_072, 262_144, 3_600.0)

    assert "50.0%" in line
    assert "131072/262144" in line
    assert "3600.0s" in line
    assert gate._prefill_progress_line(0, 0, 0.0).startswith("prefill:")


def test_prefill_progress_sampler_reads_position_and_stops_with_the_block() -> None:
    import threading
    import time

    class _Runner:
        position = 0

    runner = _Runner()
    lines: list[str] = []
    with gate._prefill_progress(runner, 10, lines.append, interval=0.01):
        runner.position = 4
        time.sleep(0.06)
    sampled = len(lines)
    assert sampled >= 1
    assert "4/10" in lines[-1]

    time.sleep(0.05)
    assert len(lines) == sampled, "sampler kept running after the block"
    assert not any(
        thread.name == "qsa-prefill-progress" for thread in threading.enumerate()
    )


def test_index_only_sizes_the_sequence_to_the_prompt_exactly() -> None:
    parser = gate.build_parser()
    index_only = parser.parse_args(
        ["model", "--json-out", "out.json", "--index-only", "--target-tokens", "262144"]
    )
    retrieval = parser.parse_args(
        ["model", "--json-out", "out.json", "--target-tokens", "262012"]
    )

    # A full-capacity target already equals the native context, so any
    # generation headroom would be rejected by the admission instead of
    # admitted; index-only asks for the prompt and nothing more.
    assert gate._sequence_length(index_only) == 262_144
    # The default retrieval gate keeps its published headroom, which lands
    # exactly on native for its 262_012 rung.
    assert gate._sequence_length(retrieval) == 262_144
    assert parser.parse_args(["model", "--json-out", "out.json"]).index_only is False


def test_index_only_refuses_a_short_prompt() -> None:
    gate._require_exact_budget(262_144, 262_144)

    with pytest.raises(SystemExit, match="exactly 262144-token prompt"):
        gate._require_exact_budget(262_143, 262_144)
    with pytest.raises(SystemExit, match="produced 100"):
        gate._require_exact_budget(100, 262_144)


def test_index_state_reports_the_count_rung_not_the_allocated_capacity() -> None:
    class _State:
        def __init__(self, count: int) -> None:
            self.count = count
            self.capacity = 262_144
            self.compression_ratio = 4

    class _Runner:
        position = 262_144
        index_states = [_State(262_144), _State(262_144)]
        qsa_bindings = {0: _Binding(0), 1: _Binding(1)}

    exact = gate._index_state_exact(_Runner(), 262_144)

    assert exact["state_exact"] is True
    assert exact["complete_blocks"] == [65_536]
    assert exact["index_capacity"] == [262_144]
    assert exact["index_counts"] == [262_144, 262_144]
    # The state arm is cursor/count consistency, not tensor equality.
    assert exact["state_scope"] == "cursor_count_consistency"

    _Runner.index_states = [_State(262_143), _State(262_144)]
    short = gate._index_state_exact(_Runner(), 262_144)
    assert short["state_exact"] is False
    # The rung follows the count, so a short prefill cannot borrow the
    # allocated capacity's 65_536 blocks.
    assert short["complete_blocks"] == [65_535, 65_536]

    _Runner.index_states = [_State(1_000), _State(1_000)]
    low = gate._index_state_exact(_Runner(), 262_144)
    assert low["complete_blocks"] == [250]
    assert low["index_capacity"] == [262_144]

    _Runner.index_states = [_State(262_144), _State(262_144)]
    _Runner.position = 262_143
    assert gate._index_state_exact(_Runner(), 262_144)["state_exact"] is False


def test_filler_probe_is_bounded_when_nothing_ever_stops_fitting() -> None:
    class _Flat:
        calls = 0

        def encode(self, text: str) -> object:
            type(self).calls += 1

            class _Counted:
                def __len__(self) -> int:
                    return 0

            return _Counted()

    tokenizer = _Flat()
    count = gate._filler_count(
        tokenizer.encode, [gate._archive_record(0)], target_tokens=50
    )

    # A tokenizer whose prompt never grows cannot make the linear walk stop
    # either; the probe stops at target_tokens + 1 instead of doubling forever.
    assert count == 51
    assert _Flat.calls <= 52


def _index_only_report(**overrides: object) -> dict[str, object]:
    index_state = {"position": 262_144, "index_counts": [262_144], "state_exact": True}
    report: dict[str, object] = {
        "mode": "index_only",
        "cpu_index_oracle": {"exact_positions": True},
        "memory_after_close": {"current_allocated_bytes": 0},
        **gate.index_only_fields(
            index_state=index_state,
            logits_finite=True,
            token_count=262_144,
            target_tokens=262_144,
            tokenizer_match=None,
            external=None,
        ),
    }
    for key, value in overrides.items():
        if key in ("exact_positions", "allocated"):
            continue
        report["index_only"][key] = value  # type: ignore[index]
    if "exact_positions" in overrides:
        report["cpu_index_oracle"] = {"exact_positions": overrides["exact_positions"]}
    if "allocated" in overrides:
        report["memory_after_close"] = {"current_allocated_bytes": overrides["allocated"]}
    return report


def test_index_only_pass_requires_all_four_measured_arms() -> None:
    assert gate.index_only_passed(_index_only_report()) is True

    for arm in ("logits_finite", "state_exact", "exact_token_budget"):
        report = _index_only_report(**{arm: False})
        assert gate.index_only_passed(report) is False, arm
    assert gate.index_only_passed(_index_only_report(exact_positions=False)) is False
    assert gate.index_only_passed(_index_only_report(allocated=4096)) is False


def test_unexercised_markers_never_satisfy_a_pass_predicate() -> None:
    report = _index_only_report()
    assert report["retrieval_exact"] == "not_exercised"
    assert report["transactional"]["repeat_exact"] == "not_exercised"
    assert report["transactional"]["isolation_exact"] == "not_exercised"
    assert report["generated_ids"] == "not_exercised"
    assert report["final_answer"] == "not_exercised"

    # The marker is a non-empty string, so a truthiness test would pass it. The
    # default gate must reject an index-only report outright.
    assert gate.retrieval_passed(report, None) is False

    # An index-only report whose measured arm carries the marker also fails.
    marked = _index_only_report()
    marked["index_only"]["logits_finite"] = "not_exercised"  # type: ignore[index]
    assert gate.index_only_passed(marked) is False


def test_retrieval_predicate_is_unchanged_and_still_gates_the_teacher() -> None:
    report = {
        "retrieval_exact": True,
        "tokenizer_match": True,
        "cpu_index_oracle": {"exact_positions": True},
        "transactional": {"repeat_exact": True, "isolation_exact": True},
        "memory_after_close": {"current_allocated_bytes": 0},
    }
    assert gate.retrieval_passed(report, None) is True
    assert gate.retrieval_passed(report, {"top1_agreement": True}) is True
    assert gate.retrieval_passed(report, {"top1_agreement": False}) is False
    assert gate.retrieval_passed({**report, "tokenizer_match": False}, {"top1_agreement": True}) is False
    assert gate.retrieval_passed({**report, "retrieval_exact": False}, None) is False
    assert (
        gate.retrieval_passed(
            {**report, "transactional": {"repeat_exact": False, "isolation_exact": True}},
            None,
        )
        is False
    )
