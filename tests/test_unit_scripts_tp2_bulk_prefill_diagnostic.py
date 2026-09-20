"""The gate must refuse to score a context the teacher never saw.

This is the check whose absence invalidated a whole gate run: the student prompt
was padded to 512 tokens to reach a row count the source-F16 policy admits, but
the reference teacher still held logits for the original 52-64 token prompt and a
different forced trajectory. The reported KL and top-1 therefore compared
different contexts, not implementation drift, and both the "the owner regresses
the heldout" and "the 512-row baseline is broken" readings drawn from them had to
be retracted.

Padding is legitimate only when the reference was captured with the same padding,
so the diagnostic asserts prompt and trajectory identity before scoring. These
tests exercise that assertion directly.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "tp2_bulk_prefill_diagnostic.py"
)
TEACHER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "tp2_teacher_coverage_broad.py"
)


@pytest.fixture(scope="module")
def diagnostic():
    spec = importlib.util.spec_from_file_location("tp2_bulk_prefill_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnostic_asserts_prompt_and_trajectory_identity() -> None:
    source = SCRIPT.read_text()
    assert "assert_prompt_matches_reference" in source
    assert "assert_trajectory_matches_reference" in source
    # ... and they must actually gate the scoring loop, not sit unused.
    assert source.index("assert_prompt_matches_reference(prompt_id, prompt)") > 0
    assert source.index("assert_trajectory_matches_reference(prompt_id, forced)") > 0


def test_a_length_mismatch_is_refused() -> None:
    """The exact mistake: a padded student against an unpadded teacher."""

    source = SCRIPT.read_text()
    block = source.split("def assert_prompt_matches_reference", 1)[1]
    body = block.split("def assert_trajectory_matches_reference", 1)[0]
    assert "len(teacher_tokens) != len(prompt)" in body
    # The message must say what to do, not just that it failed.
    assert "pad-prompt-tokens" in body


def test_a_token_mismatch_at_the_same_length_is_refused() -> None:
    source = SCRIPT.read_text()
    block = source.split("def assert_prompt_matches_reference", 1)[1]
    body = block.split("def assert_trajectory_matches_reference", 1)[0]
    assert "teacher_tokens != prompt" in body


def test_teacher_and_diagnostic_share_one_padding_rule() -> None:
    """Two divergent padding rules would silently reintroduce the bug.

    The teacher's padded prompts land in the artifact's ``suite.tokens``, and the
    diagnostic reads its prompts from there, so the teacher is the single source
    of the padded context.
    """

    teacher = TEACHER.read_text()
    assert "_pad_prompt_tokens" in teacher
    block = teacher.split("def _pad_prompt_tokens", 1)[1].split("def ", 1)[0]
    # Repeating the last token is the rule both sides use.
    assert "row[-1]" in block
    assert "it never truncates" in block or "pad < len(row)" in block
    # The padded tokens must reach the artifact the diagnostic reads.
    assert "tokens=_pad_prompt_tokens(tokens," in teacher
