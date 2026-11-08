"""Live tier: Muse Glimmer resident prefix reuse and context snapshots.

The resident session keeps one conversation's KV. A prompt that extends it
prefills only its new suffix, and ``save_context``/``load_context`` move that
KV through disk. Gates:

* a prefill resumed after a cached prefix gives bit-identical logits to a
  fresh one (resume points are chunk-aligned for exactly this reason);
* an agent-style second turn (turn 1 prompt + its output + new text) reuses the
  cached rows and decodes the same greedy tokens as a fresh prefill;
* DFlash on a reused prefix still equals AR token-for-token;
* a snapshot written to disk and loaded back serves the same continuation.

Requires ROCm + the local target/draft artifacts.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _HIP_AVAILABLE, reason="ROCm runtime not available")

TARGET = Path("/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
DRAFT = Path("/models/gguf/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
SOURCE = Path(__file__).resolve().parents[1] / "hipengine" / "runtime" / "muse_glimmer.py"
MAX_TOKENS = 32


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
        model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=8192,
    )
    provider = make_muse_glimmer_dflash_text_provider(
        target_generator=gen,
        config=SpeculativeProviderConfig(provider="dflash", draft_model=DRAFT, candidate_budget=4),
    )
    gen.attach_speculative_provider(provider)
    yield gen, provider
    gen.close()


def _turn1_ids(gen) -> tuple[int, ...]:
    # ~2.6K tokens: spans two prefill-chunk boundaries.
    text = SOURCE.read_text(encoding="utf-8")[:9000]
    return tuple(gen.tokenize(text + "\n\nSummarize what this module does in one sentence."))


def _turn2_ids(gen, turn1: tuple[int, ...], out1: tuple[int, ...]) -> tuple[int, ...]:
    return turn1 + out1 + tuple(gen.tokenize("\nNow name the class it defines."))


def _request(prompt_ids):
    from hipengine.generation.registry import GenerationRequest

    return GenerationRequest(
        prompts=(tuple(prompt_ids),), max_tokens=MAX_TOKENS, temperature=0.0,
        top_p=1.0, ignore_eos=True,
    )


def _reused(output) -> int:
    return int(output.telemetry.diagnostics["prefix_cache"]["reused_tokens"])


def _aligned(gen, tokens: int) -> int:
    return gen._ensure_session().align_resume(tokens)


def _fresh(gen, prompt_ids):
    gen.configure_engine_loop(SimpleNamespace(prefix_cache="off"))
    try:
        return gen.generate_detailed(_request(prompt_ids))[0]
    finally:
        gen.configure_engine_loop(SimpleNamespace(prefix_cache="radix"))


def test_resumed_prefill_matches_fresh_prefill(generator) -> None:
    gen, _ = generator
    ids = np.asarray(_turn1_ids(gen), dtype=np.int64)
    session = gen._ensure_session()
    session.reset()
    fresh = session.prefill(ids).copy()
    assert session.reusable_prefix(ids) == _aligned(gen, ids.size - 1)
    # A common prefix that ends mid-chunk resumes at the chunk boundary below it.
    session.truncate(1500)
    start = session.reusable_prefix(ids)
    assert start == 1024
    resumed = session.prefill(ids, start=start)
    np.testing.assert_array_equal(resumed, fresh)


def test_second_turn_reuses_the_first_and_matches_fresh(generator) -> None:
    gen, _ = generator
    turn1 = _turn1_ids(gen)
    out1 = gen.generate_detailed(_request(turn1))[0].generated_token_ids
    turn2 = _turn2_ids(gen, turn1, out1)
    cached = gen.generate_detailed(_request(turn2))[0]
    # Every row through out1[-2] is cached (out1[-1] was sampled last and
    # never fed back); reuse resumes at the chunk boundary below that.
    assert _reused(cached) == _aligned(gen, len(turn1) + len(out1) - 1)
    fresh = _fresh(gen, turn2)
    assert _reused(fresh) == 0
    assert cached.generated_token_ids == fresh.generated_token_ids


def test_dflash_on_a_reused_prefix_equals_ar(generator) -> None:
    gen, provider = generator
    turn1 = _turn1_ids(gen)
    spec1 = gen.generate_speculative_detailed(_request(turn1))[0]
    turn2 = _turn2_ids(gen, turn1, spec1.generated_token_ids)
    spec2 = gen.generate_speculative_detailed(_request(turn2))[0]
    assert _reused(spec2) == _aligned(gen, len(turn1) + len(spec1.generated_token_ids) - 1)
    ar2 = _fresh(gen, turn2)
    assert spec2.generated_token_ids == ar2.generated_token_ids


def test_context_snapshot_round_trip(generator, tmp_path) -> None:
    gen, provider = generator
    turn1 = _turn1_ids(gen)
    gen.generate_speculative_detailed(_request(turn1))
    saved = gen.save_context(tmp_path / "ctx")
    assert saved >= len(turn1)
    expected = _fresh(gen, turn1)  # overwrites the resident cache
    loaded = gen.load_context(tmp_path / "ctx")
    assert loaded == saved
    restored = gen.generate_detailed(_request(turn1))[0]
    assert _reused(restored) == _aligned(gen, len(turn1) - 1)
    assert restored.generated_token_ids == expected.generated_token_ids
    # The drafter's rows came back too: DFlash on the snapshot reuses them.
    gen.load_context(tmp_path / "ctx")
    spec = gen.generate_speculative_detailed(_request(turn1))[0]
    assert _reused(spec) == _aligned(gen, len(turn1) - 1)
    assert spec.generated_token_ids == expected.generated_token_ids
