"""Live tier: Muse Glimmer DFlash exactness — greedy output == true AR.

Speculative decoding is exact by construction (target-corrected greedy); this
gate proves the implementation: on the ten-prompt public mtp-bench suite, the
DFlash provider's generated IDs must equal the plain AR generator's IDs
token-for-token. Also reports acceptance economics (accepted/proposed) and
exercises the public LLM.generate_speculative_detailed surface once.

Requires ROCm + the local target/draft artifacts.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _HIP_AVAILABLE, reason="ROCm runtime not available")

TARGET = Path("/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
DRAFT = Path("/models/gguf/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
SUITE = Path(__file__).parent.parent / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
MAX_TOKENS = 48


@pytest.fixture(scope="module")
def generator():
    if not TARGET.exists() or not DRAFT.exists():
        pytest.skip("local Muse Glimmer artifacts not found")
    from hipengine.generation.muse_glimmer_dflash import make_muse_glimmer_dflash_text_provider
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.loading.gguf import scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF
    from hipengine.speculative.registry import SpeculativeProviderConfig

    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=TARGET, weight_index=scan_gguf(TARGET),
        model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=2048,
    )
    provider = make_muse_glimmer_dflash_text_provider(
        target_generator=gen,
        config=SpeculativeProviderConfig(
            provider="dflash", draft_model=DRAFT, candidate_budget=15,
        ),
    )
    gen.attach_speculative_provider(provider)
    yield gen, provider
    gen.close()


def _suite_prompts() -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in SUITE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row["id"], row["messages"][0]["content"]) for row in rows]


def _request(gen, prompt: str):
    from hipengine.generation.registry import GenerationRequest

    return GenerationRequest(
        prompts=(prompt,), max_tokens=MAX_TOKENS, temperature=0.0,
        top_p=1.0, ignore_eos=False,
    )


def test_dflash_ids_equal_true_ar_on_ten_prompt_suite(generator) -> None:
    gen, provider = generator
    prompts = _suite_prompts()
    assert len(prompts) == 10
    for pid, prompt in prompts:
        request = _request(gen, prompt)
        ar = gen.generate_detailed(request)[0]
        request = _request(gen, prompt)
        spec = gen.generate_speculative_detailed(request)[0]
        assert spec.generated_token_ids == ar.generated_token_ids, (
            f"{pid}: DFlash diverged from true AR\n"
            f"  ar:   {ar.generated_token_ids}\n  spec: {spec.generated_token_ids}"
        )
        assert spec.text == ar.text
    c = provider.counters
    assert c.proposed > 0
    # Economics sanity: the draft must accept something on the suite (quality
    # is measured, not gated here; exactness is the gate above).
    print(f"\nDFlash economics: accepted {c.accepted}/{c.proposed} "
          f"({c.accepted / max(c.proposed, 1):.1%}), committed {c.committed}, "
          f"target forwards {c.target_forwards}, draft forwards {c.draft_forwards}")


def test_public_llm_speculative_surface(generator) -> None:
    import hipengine
    from hipengine import SamplingParams

    llm = hipengine.LLM(
        model=str(TARGET), speculative_provider="dflash",
        draft_model=str(DRAFT), max_sequence_length=2048,
    )
    try:
        assert llm.supports_speculative
        caps = llm.speculative_capabilities
        assert caps.get("provider") == "dflash"
        outputs = llm.generate_speculative_detailed(
            ["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=24)
        )
        assert len(outputs) == 1
        assert outputs[0].generated_token_ids[0] == 13796  # oracle argmax " Paris"
        assert "Paris" in outputs[0].text
    finally:
        llm.close()