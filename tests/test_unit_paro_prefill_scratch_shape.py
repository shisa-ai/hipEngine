"""Full-attention split views must follow the executing row count."""
from types import SimpleNamespace

import pytest

from hipengine.core import DType
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


@pytest.mark.parametrize('reserved,requested', [(2, 1), (32, 1), (32, 7)])
def test_prefill_full_scratch_rebinds_row_dependent_views(reserved, requested):
    session = object.__new__(Qwen35ParoResidentSession)
    old = SimpleNamespace(attn_input=SimpleNamespace(shape=(reserved, 16)),
                          query=SimpleNamespace(dtype=DType.FP32))
    session.prefill_full_scratch = old
    calls = []
    expected = object()
    def reserve(**kwargs):
        calls.append(kwargs)
        return expected
    session._prefill_scratch_owner = lambda: SimpleNamespace(reserve_full_attention_scratch=reserve)
    assert session._ensure_full_prefill_scratch(tokens=requested) is expected
    assert calls[0]['tokens'] == requested
    assert session.prefill_full_scratch is expected


def test_prefill_full_scratch_reuses_same_layout():
    session = object.__new__(Qwen35ParoResidentSession)
    expected = SimpleNamespace(attn_input=SimpleNamespace(shape=(7, 16)),
                               query=SimpleNamespace(dtype=DType.FP32))
    session.prefill_full_scratch = expected
    assert session._ensure_full_prefill_scratch(tokens=7) is expected
