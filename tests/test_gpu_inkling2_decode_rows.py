"""Several prompts in one request decode together on the packed Inkling GGUF route.

The resident runner advances a request's sessions through ``decode_rows``; in ``fast`` two to four
rows go through the counted expert projections. The model is a multi-gigabyte local file, so these
tests run only when ``HIPENGINE_INKLING2_MINI_GGUF`` points at it and a HIP runtime loads.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import closing
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams

PROMPTS = (
    "The capital of France is",
    "Write two sentences about tide pools.",
    "12 times 17 is",
    "名古屋の有名な食べ物をひとつ挙げてください。",
)
PARAMS = dict(max_tokens=24, temperature=0.0, ignore_eos=True)


def _skip_unless_runnable() -> str:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    model = os.environ.get("HIPENGINE_INKLING2_MINI_GGUF", "")
    if not model or not Path(model).is_file():
        pytest.skip("HIPENGINE_INKLING2_MINI_GGUF must point at the Inkling 2 GGUF model")
    return model


@pytest.mark.parametrize("profile", ["strict", "fast"])
def test_batched_prompts_match_single_prompts_and_decode_together(profile):
    model = _skip_unless_runnable()
    params = SamplingParams(**PARAMS)
    with closing(LLM(model=model, backend="hip_gfx1100", execution_profile=profile, max_sequence_length=256,
                     prefix_cache="off", speculative_mtp_serving="off")) as llm:
        text = llm._get_text_generator().model
        alone = [llm.generate_detailed([prompt], params)[0] for prompt in PROMPTS]
        for rows in (2, 3, 4):
            widths = []
            decode_rows = text.decode_rows

            def counted(rows, _decode_rows=decode_rows, _widths=widths):
                _widths.append(len(rows))
                return _decode_rows(rows)

            text.decode_rows = counted
            try:
                together = llm.generate_detailed(list(PROMPTS[:rows]), params)
            finally:
                text.decode_rows = decode_rows
            assert [out.generated_token_ids for out in together] == [
                out.generated_token_ids for out in alone[:rows]]
            assert all(out.telemetry.diagnostics["execution_path"] == "inkling2_gguf_packed_hip"
                       for out in together)
            # Every row asks for the same number of tokens, so each decode step advances all of them.
            assert rows in widths and max(widths) == rows
