"""Live tier: Muse Glimmer GPU DFlash drafter vs the llama.cpp draft oracle.

Oracle: scripts/muse_glimmer_dflash_oracle.cpp (build 12005 e0f142d5d), target
prefill with layer-input extraction -> feature injection -> non-causal noise
block. The oracle's Q4_K MMQ int8 activations drift from f32 semantics (the
documented RDNA3 pattern); the CPU reference matches it 16/16 argmax at cosine
0.9998. Gates: argmax exact on all 16 block rows, cosine >= 0.999, and the
first mask-row prediction stable. Requires ROCm + local artifacts + the local
full-logits dump (git-ignored; regenerate with the oracle tool).
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

TARGET = Path("/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
DRAFT = Path("/models/gguf/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
ORACLE_DIR = Path(__file__).parent.parent / "fixtures" / "muse_glimmer" / "dflash_oracle_e0f142d5d"
PROMPT_IDS = [200000, 954, 7963, 323, 11698, 373]


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads((ORACLE_DIR / "draft_oracle.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def setup():
    if not TARGET.exists() or not DRAFT.exists():
        pytest.skip("local Muse Glimmer artifacts not found")
    from hipengine.kernels.cpu_reference.muse_glimmer_dflash import MuseGlimmerDFlashConfig
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.muse_glimmer_gguf import materialize_muse_glimmer_weights
    from hipengine.runtime.muse_glimmer import MuseGlimmerResidentSession
    from hipengine.runtime.muse_glimmer_dflash import MuseGlimmerDFlashDrafter

    target_loaded = materialize_muse_glimmer_weights(GGUFReader(TARGET), backend="hip_gfx1151")
    session = MuseGlimmerResidentSession(target_loaded, max_context=2048)
    draft_reader = GGUFReader(DRAFT)
    dcfg = MuseGlimmerDFlashConfig.from_gguf_info(draft_reader.info, mask_token_id=201818)
    drafter = MuseGlimmerDFlashDrafter(
        draft_reader, dcfg, target_weights=target_loaded,
        backend="hip_gfx1151", max_context=2048,
    )
    yield session, drafter, dcfg, target_loaded
    drafter.close()
    session.close()
    target_loaded.free()


def test_draft_block_matches_oracle(setup, oracle) -> None:
    session, drafter, dcfg, _ = setup
    logits, taps = session.prefill_with_taps(PROMPT_IDS, dcfg.target_layers)
    assert taps.shape == (len(PROMPT_IDS), len(dcfg.target_layers), 6656)
    assert int(np.argmax(logits)) == oracle["target_argmax"]
    drafter.reset()
    drafter.inject(taps, 0)
    assert drafter.context_length == len(PROMPT_IDS)
    block = np.asarray([PROMPT_IDS[-1]] + [dcfg.mask_token_id] * (dcfg.block_size - 1), dtype=np.int64)
    draft_logits = drafter.draft_block(block)
    assert np.isfinite(draft_logits).all()
    assert drafter.context_length == len(PROMPT_IDS)  # block KV discarded

    argmax = [int(np.argmax(draft_logits[i])) for i in range(dcfg.block_size)]
    expected = [oracle["target_argmax"]] + oracle["draft_argmax"]
    assert argmax == expected, f"draft argmax {argmax} != oracle {expected}"

    full = ORACLE_DIR / "draft_logits.f32"
    if full.exists():
        reference = np.fromfile(full, dtype=np.float32).reshape(dcfg.block_size, -1)
        cos = float(np.sum(draft_logits * reference) / (
            np.linalg.norm(draft_logits) * np.linalg.norm(reference)))
        assert cos >= 0.999, f"draft logits cosine {cos}"


def test_draft_block_long_context_uses_window_slice(setup) -> None:
    """Past the strict kernel's resident-logit LDS bound (~16K keys) the draft
    still runs: its SWA attention only reads the window slice. The result must
    match a short-context draft over the same windowed keys at the same
    relative positions (RoPE is relative; only the absolute tables differ)."""

    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.muse_glimmer_dflash import MuseGlimmerDFlashDrafter

    session, short_drafter, dcfg, target_loaded = setup
    _, taps = session.prefill_with_taps(PROMPT_IDS, dcfg.target_layers)
    tile = np.tile(taps, (16, 1, 1))  # 96 rows, period 6
    long_len, short_len = 96 * 209, 96 * 32
    assert (long_len - short_len) % len(PROMPT_IDS) == 0
    block = np.asarray([PROMPT_IDS[-1]] + [dcfg.mask_token_id] * 4, dtype=np.int64)

    def draft(max_context: int, length: int) -> np.ndarray:
        drafter = MuseGlimmerDFlashDrafter(
            GGUFReader(DRAFT), dcfg, target_weights=target_loaded,
            backend="hip_gfx1151", max_context=max_context,
        )
        try:
            for pos in range(0, length, tile.shape[0]):
                drafter.inject(tile, pos)
            return drafter.draft_block(block)
        finally:
            drafter.close()

    long_logits = draft(long_len + 64, long_len)
    short_logits = draft(short_len + 64, short_len)
    assert np.isfinite(long_logits).all()
    for row in range(block.size):
        a, b = long_logits[row], short_logits[row]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos >= 0.999, f"row {row} cosine {cos}"
        assert int(np.argmax(a)) == int(np.argmax(b)), f"row {row} argmax differs"
