"""User-surface check for the explicit ``speculative_mtp`` flag (M7).

The unit tests prove the dispatch; this proves what a caller actually gets on
the real gemma 4 artifact: an explicit request raises naming the capability
instead of quietly returning plain autoregressive text. Before the flag
existed, ``SamplingParams(speculative_mtp=True)`` did not even construct, so
there was no way to ask -- and therefore no way to be told no.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams

_MODEL_DIR = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF")
_TARGET = _MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _TARGET.exists(),
    reason="needs ROCm (libamdhip64.so) and the gemma4 fixture artifact",
)


def test_live_gemma4_explicit_mtp_request_fails_naming_the_capability():
    llm = LLM(model=str(_TARGET))

    with pytest.raises(NotImplementedError, match="speculative MTP"):
        llm.generate("Name one colour.", SamplingParams(speculative_mtp=True))

    # The same prompt without the flag must still be served, so the refusal is
    # attributable to the request and not to the model being broken.
    plain = llm.generate("Name one colour.", SamplingParams(max_tokens=8))
    assert plain and plain[0]