"""Owner-bound packed prefill selection, independent of retired ambient input."""

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator


@pytest.mark.parametrize("selected, expected", [(None, True), (True, True), (False, False)])
@pytest.mark.parametrize("ambient", ["0", "1"])
def test_packed_prefill_selection_ignores_retired_environment(monkeypatch, selected, expected, ambient):
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", ambient)
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    owner.ar_packed_prefill = selected
    assert owner._ar_packed_prefill_enabled() is expected
