"""VibeVoice-ASR Q4_K_M backbone runner gates.

The Q4 GGUF is produced by scripts/vibevoice_asr_to_gguf.py +
llama-quantize + scripts/vibevoice_asr_gguf_merge.py. Skipped cleanly
when the artifact is absent (no-ROCm CI and machines without the file).

Gates:
- first-position logits: top-1 must match the torch fixture, with a
  Q4-calibrated relative bound (0.05) on the logit drift
- greedy chain: 16/16 tokens vs the torch fixture (strict prefix
  regression, same as the dense bf16 runner's chain gate)
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.vibevoice_qwen2 import greedy_generate

Q4_GGUF = Path("/tmp/vibevoice-asr-q4km.gguf")
LM_FIXTURE = Path(__file__).parent / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available() or not Q4_GGUF.is_file() or not LM_FIXTURE.is_file(),
    reason="needs ROCm, the merged Q4_K_M GGUF and the LM fixture",
)


@pytest.fixture(scope="module")
def runtime():
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    runner = VibevoiceQwen2Q4Runtime(weights, max_context=512)
    yield runner
    runner.close()


@pytest.fixture(scope="module")
def lm() -> dict[str, np.ndarray]:
    with np.load(LM_FIXTURE) as data:
        return {k: data[k] for k in data.files}


def _prompt_rows(runner, lm) -> list[np.ndarray]:
    input_ids = np.asarray(lm["input_ids"])[0]
    positions = np.asarray(lm["audio_placeholder_positions"])
    audio = lm["audio_embeds"].astype(np.float32)
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for p in positions:
        rows[p] = audio[p - positions[0]]
    return rows


def test_first_position_logits_q4(runtime, lm) -> None:
    rows = _prompt_rows(runtime, lm)
    runtime.reset()
    runtime.push_token(rows[0], 0)
    runtime.forward_layers(0)
    logits, token = runtime.logits_argmax()
    ref = lm["logits_pos0"]
    diff = np.abs(logits - ref).max()
    scale = max(np.abs(ref).max(), 1e-9)
    assert diff / scale <= 5e-2, f"q4 logits_pos0 rel {diff / scale:.3e}"
    assert token == int(ref.argmax())


def test_greedy_chain_matches_torch_q4(runtime, lm) -> None:
    """Prefix regression: the Q4 backbone reproduces the torch greedy chain."""
    rows = _prompt_rows(runtime, lm)
    runtime.reset()
    generated = greedy_generate(runtime, rows, max_new_tokens=16)
    fixture = [int(t) for t in np.asarray(lm["greedy_tokens"])]
    assert generated == fixture, f"{generated} != {fixture}"


def test_weight_type_routing(runtime) -> None:
    """All backbone GEMM weights are routed K-quant blocks the kernels own."""
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES

    routed = {GGUF_WEIGHT_TYPES[buf.ptr] for layer in runtime.layers
              for buf in (layer.q_w, layer.k_w, layer.v_w, layer.o_w,
                          layer.gate_w, layer.up_w, layer.down_w)}
    assert routed <= {12, 13, 14}, routed
    assert 12 in routed  # Q4_K backbone present
