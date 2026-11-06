"""Explicit packed-decode owner selection ignores retired environment input."""
import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator


@pytest.mark.parametrize('selected', [None, False, True])
@pytest.mark.parametrize('ambient', ['0', '1'])
def test_packed_decode_owner_ignores_retired_environment(monkeypatch, selected, ambient):
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    owner.ar_packed_decode = selected
    monkeypatch.setenv('HIPENGINE_GGUF_AR_PACKED_DECODE', ambient)
    assert owner._ar_packed_decode_enabled() is (True if selected is None else selected)
