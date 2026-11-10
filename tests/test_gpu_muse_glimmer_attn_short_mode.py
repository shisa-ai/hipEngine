"""GPU tier: Muse Glimmer adaptive attention split-block config.

Short live ranges run fewer split blocks (empty blocks still launch and the
combine still scans them). The sticky session config is shared by the decode
graph, the verify graphs, and the eager fallback because row bit-identity
requires the same split config on both sides. Crossing the threshold
invalidates both graph families so they re-capture together at 64 blocks.
These tests exercise the config choice, the crossing re-capture, and
verify-row == decode-step bit identity in both modes through the real model.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")

TARGET = Path("/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")


@pytest.fixture(scope="module")
def gen():
    if not TARGET.exists():
        pytest.skip("Muse Glimmer target artifact not available")
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.loading.gguf import scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF

    return make_muse_glimmer_gguf_generator_gfx1151(
        model_path=TARGET, weight_index=scan_gguf(TARGET),
        model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=4096)


def _prefill(session, generator, n: int) -> None:
    ids = (generator.tokenize("hello world ") * (n // 2 + 1))[:n]
    session.reset()
    session.prefill(ids)


def test_short_session_reduced_blocks_and_row_identity(gen) -> None:
    s = gen._ensure_session()
    _prefill(s, gen, 64)
    s.decode_step(13796)
    assert s._attn_split_cfg == (s._DECODE_ATTN_SPLIT_KEYS, s._ATTN_SHORT_MIN_BLOCKS)
    assert getattr(s, "_graph_exec", 0), "decode graph must engage in short mode"
    # Verify row 0 equals the decode step at the same position, bit-for-bit,
    # with both sides on the same reduced-block config.
    length = s._length
    v_logits, _ = s.verify_with_taps([13796, 954], (2, 14))
    assert getattr(s, "_verify_graph_exec", None), "verify graph must engage"
    s.truncate(length)
    d_logits = s.decode_step(13796)
    np.testing.assert_array_equal(v_logits[0], d_logits)


def test_threshold_crossing_recaptures_and_keeps_row_identity(gen) -> None:
    s = gen._ensure_session()
    thr = s._ATTN_SHORT_THRESHOLD
    _prefill(s, gen, thr - 8)
    for _ in range(4):
        s.decode_step(13796)
    assert s._ATTN_SHORT_MIN_BLOCKS <= s._attn_split_cfg[1] < s._DECODE_ATTN_BLOCKS
    old_exec = s._graph_exec
    while s._length <= thr:
        s.decode_step(13796)
    s.decode_step(13796)  # first step starting beyond the threshold flips the config
    assert s._attn_split_cfg == (s._DECODE_ATTN_SPLIT_KEYS, s._DECODE_ATTN_BLOCKS)
    assert getattr(s, "_graph_exec", 0) and s._graph_exec != old_exec, \
        "crossing must re-capture the decode graph with the 64-block config"
    length = s._length
    v_logits, _ = s.verify_with_taps([13796, 954, 323], (2, 14))
    assert getattr(s, "_verify_graph_exec", None), "verify graph must re-capture"
    s.truncate(length)
    d_logits = s.decode_step(13796)
    np.testing.assert_array_equal(v_logits[0], d_logits)