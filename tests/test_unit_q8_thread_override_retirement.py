"""Retired ambient Q8 thread control must not override launch policy."""

import pytest

from hipengine.runtime import gguf_linear


@pytest.mark.parametrize("value", ["64", "128", "invalid"])
def test_retired_environment_does_not_select_threads(monkeypatch, value):
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_THREADS", value)
    assert gguf_linear._resolve_q8_t16_threads() == 0
    assert not gguf_linear._q8_t16_threads_override_active()


@pytest.mark.parametrize("threads", [64, 128])
def test_explicit_thread_arguments_remain_supported(monkeypatch, threads):
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_THREADS", "invalid")
    assert gguf_linear._resolve_q8_t16_threads(threads) == threads
    assert gguf_linear._q8_t16_threads_override_active(threads)
