"""Cross-runner KV save/restore exactness (GPU).

The save/restore contract is bitwise: a restored runner holds the same cache
bytes and position the saving runner held, so decoding from the restore must
produce the same greedy token IDs as decoding the fresh runner -- including the
first token, which comes from the logits saved alongside the planes because a
restored context has no prefill pass to recompute them.

Guarded on the real artifact and on HIP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader, scan_gguf
from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights
from tests._rocm_guard import hip_runtime_available

ARTIFACT = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)

_needs = pytest.mark.skipif(
    not (hip_runtime_available() and ARTIFACT.exists()),
    reason="the Gemma 4 artifact and HIP are both required",
)

PROMPT_TOKENS = 512
DECODE_TOKENS = 24


def _prompt_ids() -> list[int]:
    # Any well-formed id range serves the exactness contract; a low fixed range
    # avoids depending on metadata accessors just to build the prompt.
    rng = np.random.default_rng(20261006)
    return [int(token) for token in rng.integers(0, 2000, size=PROMPT_TOKENS)]


def _decode(runner: Gemma4Runner, first_token: int, count: int) -> list[int]:
    generated = [int(first_token)]
    for _ in range(count - 1):
        logits = runner.forward([generated[-1]])
        generated.append(int(np.argmax(logits)))
    return generated


@_needs
def test_restored_decode_matches_fresh_decode_bitwise(tmp_path: Path) -> None:
    reader = GGUFReader(str(ARTIFACT))
    weights = load_gemma4_device_weights(reader)
    try:
        prompt = _prompt_ids()
        identity = {"prompt_tokens_sha256": "cafe", "artifact": "the-campaign-gguf"}

        fresh = Gemma4Runner(weights=weights, capacity=4096)
        try:
            logits = fresh.forward(prompt)
            first = int(np.argmax(logits))

            state = tmp_path / "kv_state.bin"
            fresh.save_kv_state(state, identity=identity, logits=logits)

            fresh_ids = _decode(fresh, first, DECODE_TOKENS)

            restored = Gemma4Runner(weights=weights, capacity=4096)
            try:
                saved_logits = restored.restore_kv_state(state, identity=identity)
                assert isinstance(saved_logits, np.ndarray)
                # The saved logits are the fresh prefill's own last row,
                # byte for byte, and reproduce its first sampled token.
                assert saved_logits.tobytes() == np.ascontiguousarray(
                    logits, dtype=np.float32
                ).tobytes()
                assert int(np.argmax(saved_logits)) == first
                assert restored.position == len(prompt)
                restored_ids = _decode(restored, first, DECODE_TOKENS)
                assert restored_ids == fresh_ids
                assert restored.position == len(prompt) + DECODE_TOKENS - 1
            finally:
                restored.close()

            # A mismatched identity is refused before any byte lands.
            with pytest.raises(ValueError, match="identity mismatch"):
                fresh.restore_kv_state(state, identity={"prompt_tokens_sha256": "other"})

            # A file saved without identity is not restorable under one.
            anonymous = tmp_path / "kv_anonymous.bin"
            fresh.save_kv_state(anonymous)
            with pytest.raises(ValueError, match="identity mismatch"):
                fresh.restore_kv_state(anonymous, identity=identity)
        finally:
            fresh.close()
    finally:
        weights.free()


@_needs
def test_restore_refuses_a_capacity_mismatch(tmp_path: Path) -> None:
    reader = GGUFReader(str(ARTIFACT))
    weights = load_gemma4_device_weights(reader)
    try:
        prompt = _prompt_ids()
        identity = {"prompt_tokens_sha256": "cafe", "artifact": "the-campaign-gguf"}

        wide = Gemma4Runner(weights=weights, capacity=4096)
        try:
            logits = wide.forward(prompt)
            state = tmp_path / "kv_state.bin"
            wide.save_kv_state(state, identity=identity, logits=logits)

            narrow = Gemma4Runner(weights=weights, capacity=2048)
            try:
                with pytest.raises(ValueError, match="capacity"):
                    narrow.restore_kv_state(state, identity=identity)
                # The refused restore must not have moved the runner's state.
                assert narrow.position == 0
            finally:
                narrow.close()
        finally:
            wide.close()
    finally:
        weights.free()
