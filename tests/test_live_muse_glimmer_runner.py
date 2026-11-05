"""Live tier: Muse Glimmer resident runner correctness on gfx11.

Gates (declared before the formal run; the bring-up smoke informed the bf16
tolerances, which surround the observed bf16-vs-f16-KV drift with margin):

* llama.cpp oracle fixture (build 12005 e0f142d5d, same artifact): greedy
  argmax exact; top-20 id overlap >= 19/20; per-id top-20 logit |err| <= 0.15.
  The 0.15 bound is a measured cross-engine floor, not a semantic tolerance:
  the f32 CPU reference itself drifts up to 0.088 from llama.cpp on the rank-20
  tail (f16 KV + accumulation order), so no correct implementation that does
  not replicate llama.cpp's exact arithmetic can hold 0.1. The semantic anchor
  is the CPU-reference gate below, where the runner stays within 0.05.
* CPU f32 reference (validated against the same oracle): top-1 exact,
  per-id top-20 |err| <= 0.05, and KL(runner || reference) <= 0.05 on the
  final-row distribution.
* Prefill/decode consistency: a decode step's logits equal a fresh prefill over
  prompt+token's final row (the block and decode kernels share arithmetic),
  max |err| <= 1e-3. This catches KV-cache append/position bugs without an
  external oracle.

Requires ROCm and the local artifact; skips otherwise.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _HIP_AVAILABLE, reason="ROCm runtime not available")

MODEL = Path(
    "/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "muse_glimmer"
    / "llamacpp_debug_e0f142d5d.json"
)


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def session():
    if not MODEL.exists():
        pytest.skip(f"local Muse Glimmer GGUF not found: {MODEL}")
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.muse_glimmer_gguf import materialize_muse_glimmer_weights
    from hipengine.runtime.muse_glimmer import MuseGlimmerResidentSession

    reader = GGUFReader(MODEL)
    loaded = materialize_muse_glimmer_weights(reader, backend="hip_gfx1151")
    sess = MuseGlimmerResidentSession(loaded, max_context=2048)
    yield sess
    sess.close()
    loaded.free()


def test_final_logits_match_llamacpp_oracle(session, oracle) -> None:
    logits = session.prefill(oracle["token_ids"])
    assert np.isfinite(logits).all()
    expected = oracle["final_logits"]
    assert int(np.argmax(logits)) == int(expected["argmax"])

    top20 = expected["top20"]
    expected_ids = {int(row["id"]) for row in top20}
    runner_ids = set(np.argsort(logits)[-20:].tolist())
    assert len(expected_ids & runner_ids) >= 19, (
        f"top-20 overlap {len(expected_ids & runner_ids)}/20"
    )
    for row in top20:
        err = abs(float(logits[int(row["id"])]) - float(row["logit"]))
        assert err <= 0.15, f"id {row['id']}: |err| {err} > 0.15"


def test_final_logits_match_cpu_reference(session, oracle) -> None:
    from hipengine.kernels.cpu_reference.muse_glimmer import (
        MuseGlimmerConfig, forward, gguf_weight_source,
    )
    from hipengine.loading.gguf import GGUFReader

    logits = session.prefill(oracle["token_ids"])
    reader = GGUFReader(MODEL)
    config = MuseGlimmerConfig.from_gguf_info(reader.info, vocab_size=len(logits))
    ref_logits, _, _ = forward(oracle["token_ids"], gguf_weight_source(reader), config)
    ref = ref_logits[-1]
    assert int(np.argmax(logits)) == int(np.argmax(ref))
    for row in oracle["final_logits"]["top20"]:
        i = int(row["id"])
        err = abs(float(logits[i]) - float(ref[i]))
        assert err <= 0.05, f"id {i}: runner vs f32 reference |err| {err} > 0.05"

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        e = np.exp(x - x.max())
        return e / e.sum()

    p, q = _softmax(logits), _softmax(ref)
    kl = float(np.sum(p * np.log((p + 1e-30) / (q + 1e-30))))
    assert kl <= 0.05, f"KL(runner||reference) = {kl}"


def test_decode_step_matches_fresh_prefill(session, oracle) -> None:
    ids = list(oracle["token_ids"])
    session.reset()
    logits = session.prefill(ids)
    token = int(np.argmax(logits))
    decode_logits = session.decode_step(token)
    assert np.isfinite(decode_logits).all()

    session.reset()
    prefill_logits = session.prefill(ids + [token])
    err = float(np.max(np.abs(decode_logits - prefill_logits)))
    assert err <= 1e-3, f"decode vs fresh-prefill max |err| = {err}"
    assert int(np.argmax(decode_logits)) == int(np.argmax(prefill_logits))


def test_greedy_generation_is_coherent(session, oracle) -> None:
    from hipengine.tokenization.gguf import MuseGlimmerGGUFTokenizer
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(MODEL)
    tokenizer = MuseGlimmerGGUFTokenizer.from_gguf_info(reader.info)
    session.reset()
    logits = session.prefill(oracle["token_ids"])
    generated = []
    cur = int(np.argmax(logits))
    for _ in range(8):
        generated.append(cur)
        if cur in tokenizer.stop_token_ids:
            break
        logits = session.decode_step(cur)
        assert np.isfinite(logits).all()
        cur = int(np.argmax(logits))
    text = tokenizer.decode(generated)
    # The oracle prompt is "The capital of France is"; the validated llama.cpp
    # argmax is " Paris". The continuation must at least start there.
    assert generated[0] == int(oracle["final_logits"]["argmax"])
    assert "Paris" in text