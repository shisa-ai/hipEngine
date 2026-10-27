from pathlib import Path

import numpy as np
import pytest

from scripts.qwen4_exp_qsa_retrieval_gate import (
    _read_llama_tap,
    _selection_locality,
    build_retrieval_prompt,
    retrieval_prompt_minimum_tokens,
)


class _WhitespaceTokenizer:
    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()


def test_qsa_retrieval_prompt_is_bounded_and_retains_early_needle() -> None:
    tokenizer = _WhitespaceTokenizer()
    prompt, needle = build_retrieval_prompt(tokenizer, target_tokens=2_000)

    tokens = tokenizer.encode(prompt)
    assert len(tokens) <= 2_000
    assert len(tokens) > 1_900
    assert tokens[needle] == "VIOLET-7391."
    assert "Reasoning effort is set to low" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_selection_locality_counts_gaps_and_kv_pages() -> None:
    actual = _selection_locality(
        np.asarray([1025, 0, 1, 1024], dtype=np.int64), kv_row_bytes=16
    )

    assert actual == {
        "selected_position_span": 1026,
        "selected_contiguous_pairs": 2,
        "selected_gap_mean": 1025 / 3,
        "selected_gap_max": 1023,
        "selected_kv_pages_per_layer": 2,
    }


def test_selection_locality_accepts_an_empty_selection() -> None:
    actual = _selection_locality(np.empty(0, dtype=np.int64), kv_row_bytes=16)

    assert actual == {
        "selected_position_span": 0,
        "selected_contiguous_pairs": 0,
        "selected_gap_mean": 0.0,
        "selected_gap_max": 0,
        "selected_kv_pages_per_layer": 0,
    }


def test_read_llama_tap_returns_last_query_vector(tmp_path: Path) -> None:
    shape = np.asarray([4, 3, 1, 1], dtype=np.int64)
    values = np.arange(12, dtype=np.int32).reshape(1, 1, 3, 4)
    path = tmp_path / "tap.bin"
    path.write_bytes(shape.tobytes() + values.tobytes())

    actual_shape, actual = _read_llama_tap(path)

    assert actual_shape == [4, 3, 1, 1]
    np.testing.assert_array_equal(actual, [8, 9, 10, 11])


def test_gate_takes_the_selection_from_the_runner_not_the_environment() -> None:
    """The retired process switch must not decide where the gate reads scores."""

    import ast
    import inspect

    from scripts import qwen4_exp_qsa_retrieval_gate as gate

    source = inspect.getsource(gate.main)
    assert "HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION" not in source
    assert "qsa_batched_selection(runner, False)" in source
    # The flag lives with the rest of the CLI surface, which moved into
    # ``build_parser`` so the capacity mode is testable without a model.
    assert "--per-row-selection" in inspect.getsource(gate.build_parser)
    tree = ast.parse(inspect.getsource(gate))

    def is_field_read(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "runner"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "qsa_batched_selection"
        )

    def names(node: ast.AST, name: str) -> bool:
        return any(
            isinstance(child, ast.Attribute) and child.attr == name
            for child in ast.walk(node)
        )

    # The arm is a runner field read, not a process lookup, and it is read
    # inside the scope that binds it: the published value returns on exit.
    with_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With) and names(node, "prefill")
    )
    assert any(is_field_read(child) for child in ast.walk(with_node))


def _ordinary_record_count(prompt: str) -> int:
    return sum(
        1 for index in range(100) if f"Archive record {index:04d}:" in prompt
    )


def test_prompt_budget_below_at_and_above_the_mandatory_payload() -> None:
    tokenizer = _WhitespaceTokenizer()
    minimum = retrieval_prompt_minimum_tokens(tokenizer)

    with pytest.raises(ValueError, match=f"at least {minimum} tokens"):
        build_retrieval_prompt(tokenizer, target_tokens=minimum - 1)

    prompt, needle = build_retrieval_prompt(tokenizer, target_tokens=minimum)
    tokens = tokenizer.encode(prompt)
    assert len(tokens) == minimum
    assert tokens[needle] == "VIOLET-7391."
    # At the minimum the ordinary archive is gone, not the needle or question.
    assert "Archive record 0000:" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

    above, _ = build_retrieval_prompt(tokenizer, target_tokens=minimum + 400)
    assert len(tokenizer.encode(above)) <= minimum + 400
    assert "VIOLET-7391" in above
    assert above.endswith("<|im_start|>assistant\n")


def test_builder_never_exceeds_the_requested_budget() -> None:
    """A target below the full archive drops records instead of overrunning."""

    tokenizer = _WhitespaceTokenizer()
    minimum = retrieval_prompt_minimum_tokens(tokenizer)
    for target in range(minimum, minimum + 900, 25):
        prompt, needle = build_retrieval_prompt(tokenizer, target_tokens=target)
        tokens = tokenizer.encode(prompt)
        assert len(tokens) <= target
        assert tokens[needle] == "VIOLET-7391."
        assert prompt.count("VIOLET-7391") == 1
        assert prompt.endswith("<|im_start|>assistant\n")


def test_the_reported_2080_token_target_fits_its_sequence() -> None:
    """The natural-retrieval gate sizes its sequence from this budget."""

    tokenizer = _WhitespaceTokenizer()

    prompt, _ = build_retrieval_prompt(tokenizer, target_tokens=2_080)

    assert len(tokenizer.encode(prompt)) <= 2_080


def test_smaller_budgets_drop_ordinary_records_not_the_needle() -> None:
    tokenizer = _WhitespaceTokenizer()
    minimum = retrieval_prompt_minimum_tokens(tokenizer)

    small, _ = build_retrieval_prompt(tokenizer, target_tokens=minimum + 200)
    large, _ = build_retrieval_prompt(tokenizer, target_tokens=minimum + 1_200)

    assert _ordinary_record_count(small) < _ordinary_record_count(large) <= 100
    assert "VIOLET-7391" in small
    assert "VIOLET-7391" in large


@pytest.mark.parametrize("target", [0, 1, 40, -5])
def test_unrealistic_targets_fail_loudly(target: int) -> None:
    tokenizer = _WhitespaceTokenizer()
    minimum = retrieval_prompt_minimum_tokens(tokenizer)
    assert target < minimum

    with pytest.raises(ValueError, match="at least .* tokens"):
        build_retrieval_prompt(tokenizer, target_tokens=target)


def test_every_prefill_runs_inside_the_selection_scope() -> None:
    """A repeat or isolation prefill must not escape the bound selection arm."""

    import ast
    import inspect

    from scripts import qwen4_exp_qsa_retrieval_gate as gate

    tree = ast.parse(inspect.getsource(gate))
    prefills = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prefill"
    ]
    assert prefills

    def binds_selection(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name) and node.id == "selection"
        ) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "qsa_batched_selection"
        )

    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(binds_selection(item.context_expr) for item in node.items)
    ]
    for call in prefills:
        assert any(
            any(child is call for child in ast.walk(scope)) for scope in scopes
        ), "prefill call outside the QSA selection scope"
