"""Ngram experiments use explicit per-adapter configuration."""

from types import SimpleNamespace

from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.speculative.ngram_mod import NgramModConfig


def test_ngram_configuration_is_explicit(monkeypatch):
    for name, value in (
        ("HIPENGINE_GGUF_SPECDEC2_NGRAM_MOD", "1"),
        ("HIPENGINE_GGUF_SPECDEC2_NGRAM_MATCH", "bad"),
        ("HIPENGINE_GGUF_SPECDEC2_NGRAM_MIN", "bad"),
        ("HIPENGINE_GGUF_SPECDEC2_NGRAM_PROBE_MAX", "bad"),
    ):
        monkeypatch.setenv(name, value)
    owner = SimpleNamespace(generator=SimpleNamespace(backend="hip_gfx1100", execution_profile="production"),
                            capacity=2, _shared_runner=SimpleNamespace(hidden_size=4))
    kwargs = dict(enabled=True, target_verify_mode="native", candidate_budget=2)
    default = Qwen35GGUFMTP2Adapter(owner, **kwargs)
    diagnostic = None
    try:
        assert default._ngram is None
        diagnostic = Qwen35GGUFMTP2Adapter(owner, ngram_enabled=True,
            ngram_config=NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=24), **kwargs)
        assert diagnostic._ngram is not None
        assert default._ngram is None
    finally:
        default.close()
        if diagnostic is not None:
            diagnostic.close()
