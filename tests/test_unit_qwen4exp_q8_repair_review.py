import ast

import pytest

from scripts.qwen4exp_q8_repair_code_checks import extract_code


def test_review_code_extraction_removes_only_outer_fence_and_eos():
    source = "```python\ndef f():\n    return 1\n```<|im_end|>"
    assert extract_code(source) == "def f():\n    return 1\n"
    assert isinstance(ast.parse(extract_code(source)), ast.Module)


def test_review_code_extraction_rejects_truncated_python():
    with pytest.raises(SyntaxError):
        extract_code("```python\ndef f(")
