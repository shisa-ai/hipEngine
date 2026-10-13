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
# Load the Gemma kernel families before conftest snapshots the collection-time
# registry. Otherwise the first live request imports them, teardown removes
# their registrations, and Python's import cache prevents the second request
# from restoring them. This imports wrappers only; it does not allocate a GPU.
from hipengine.runtime.gemma4 import Gemma4Runner  # noqa: F401

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


@pytest.fixture
def live_llm():
    llm = LLM(model=str(_TARGET))
    try:
        yield llm
    finally:
        llm.close()


def test_live_gemma4_explicit_mtp_request_fails_naming_the_capability(live_llm):
    llm = live_llm

    with pytest.raises(NotImplementedError, match="speculative MTP"):
        llm.generate("Name one colour.", SamplingParams(speculative_mtp=True))

    # The same prompt without the flag must still be served, so the refusal is
    # attributable to the request and not to the model being broken.
    plain = llm.generate("Name one colour.", SamplingParams(max_tokens=8))
    assert plain and plain[0]


def test_live_gemma4_attached_mtp_preserves_plain_request_state():
    draft = _MODEL_DIR / "MTP" / "mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    if not draft.exists():
        pytest.skip("needs the Gemma assistant GGUF sidecar")
    llm = LLM(
        model=str(_TARGET),
        backend="hip_gfx1100",
        speculative_provider="gemma4_mtp",
        draft_model=str(draft),
        speculative_candidate_budget=2,
    )
    try:
        prompt = "Explain in a few sentences why the sky is blue."
        plain_params = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
        spec_params = SamplingParams(
            max_tokens=32, temperature=0.0, ignore_eos=True, speculative_mtp=True,
        )
        before = llm.generate_detailed([prompt], plain_params)[0]
        speculative = llm.generate_detailed([prompt], spec_params)[0]
        after = llm.generate_detailed([prompt], plain_params)[0]
        assert len(speculative.generated_token_ids) == 32
        assert after.generated_token_ids == before.generated_token_ids
        diagnostics = speculative.telemetry.diagnostics
        assert diagnostics["proposed_draft_tokens"] > 0
        # The profile decides the verify contract, and the telemetry has to say
        # which one ran: only the exact one promises plain greedy's tokens.
        if llm._get_text_generator().verify_dense_exact:
            assert before.generated_token_ids == speculative.generated_token_ids
            assert diagnostics["verify_equivalence"] == "bit_identical_to_target_ar_greedy"
            assert diagnostics["target_verify"] == "batched_exact"
            assert (
                speculative.telemetry.decode_state.execution_path
                == "gemma4_mtp_batched_exact_verify"
            )
        else:
            assert diagnostics["verify_equivalence"] == "production_envelope_of_target_ar_greedy"
            assert diagnostics["target_verify"] == "batched_wmma"
            assert speculative.telemetry.decode_state.execution_path == "gemma4_mtp_batched_verify"
    finally:
        llm.close()


def test_live_gemma4_sampled_requests_run_plain_and_through_mtp():
    """A temperature above zero samples on both routes, and a seed reproduces it."""

    draft = _MODEL_DIR / "MTP" / "mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    if not draft.exists():
        pytest.skip("needs the Gemma assistant GGUF sidecar")
    llm = LLM(
        model=str(_TARGET),
        backend="hip_gfx1100",
        speculative_provider="gemma4_mtp",
        draft_model=str(draft),
        speculative_candidate_budget=4,
    )
    try:
        # A chat-formatted prompt: on bare text the model's continuation is
        # near-deterministic and two seeds can draw the same tokens.
        prompt = llm._get_text_generator().render_chat_prompt(
            [{"role": "user", "content": "Suggest five unusual names for a cat."}]
        )

        def run(seed: int, *, mtp: bool, **sampler):
            params = SamplingParams(
                max_tokens=40, ignore_eos=True, seed=seed, speculative_mtp=mtp,
                **({"temperature": 1.0, "top_k": 64, "top_p": 0.95} | sampler),
            )
            return llm.generate_detailed([prompt], params)[0]

        for mtp in (False, True):
            first, again, other = run(3, mtp=mtp), run(3, mtp=mtp), run(4, mtp=mtp)
            assert len(first.generated_token_ids) == 40
            assert first.generated_token_ids == again.generated_token_ids
            assert first.generated_token_ids != other.generated_token_ids
        plain, speculative = run(3, mtp=False), run(3, mtp=True)
        assert plain.finish_details.sampler_mode == "host_logits_sample"
        assert speculative.finish_details.sampler_mode == "sampled_speculative_mtp"
        diagnostics = speculative.telemetry.diagnostics
        assert diagnostics["draft_law"] == "sampled"
        assert 0 < diagnostics["accepted_draft_tokens"] <= diagnostics["proposed_draft_tokens"]
        # A penalty and a whole-vocabulary law take the other sampler paths.
        penalized = run(3, mtp=True, top_k=0, top_p=1.0, temperature=0.8,
                        repetition_penalty=1.1)
        assert len(penalized.generated_token_ids) == 40
        assert penalized.generated_token_ids != run(
            4, mtp=True, top_k=0, top_p=1.0, temperature=0.8, repetition_penalty=1.1,
        ).generated_token_ids
    finally:
        llm.close()
