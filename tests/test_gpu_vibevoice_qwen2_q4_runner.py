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


def test_weights_own_buffers_and_runners_borrow() -> None:
    """Loader buffers are owned by the weights handle, not by a runner.

    Previously every runner keep()ed the loader's buffers into its own
    free list, so two runners over one weights object double-freed the
    same device pointers (HIP error 1 on the second close).
    """
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        first = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        second = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        assert all(buf not in first._buffers for buf in weights.buffers)
        assert all(buf not in second._buffers for buf in weights.buffers)
        first.close()
        second.close()  # must not free the other runner's weights
        assert weights.buffers and not weights._closed
    finally:
        weights.close()
        weights.close()  # idempotent


def test_dual_gemv_scratch_is_per_runner() -> None:
    """Emulated dual GEMV scratch must not be shared across runners."""
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        first = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        second = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        need = weights.spec.intermediate_size * 4
        for runner in (first, second):
            gate, up = runner._dual_scratch
            assert gate.nbytes >= need and up.nbytes >= need
            # Runner-owned: released by runner.close().
            assert gate in runner._buffers and up in runner._buffers
        assert first._dual_scratch[0].ptr != second._dual_scratch[0].ptr
        first.close()
        second.close()
    finally:
        weights.close()


def test_prefill_manifest_records_q4_quant() -> None:
    """The execution manifest must not claim bf16 for a Q4 backbone."""
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        runner = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        assert runner.quant_name == "q4_k_m"
        from hipengine.core.memory import malloc, free

        rows = 2
        hidden = runner.spec.hidden_size
        buf = malloc(rows * hidden * 2)
        try:
            runner.reset()
            runner.prefill_rows(buf, rows, 0)
            manifest = runner.variant_manifest
            quant = manifest["quant"] if isinstance(manifest, dict) else manifest.quant
            model = manifest["model"] if isinstance(manifest, dict) else manifest.model
            assert quant == "q4_k_m", manifest
            assert model == "vibevoice_asr"
        finally:
            free(buf)
        runner.close()
    finally:
        weights.close()


def test_loader_buffers_are_bf16_sized() -> None:
    """Dense device buffers must be bf16-sized, not 4x oversized.

    ``malloc(host_f32.nbytes * 2)`` allocated 8 bytes per element for a
    buffer that holds 2, wasting ~6.5 GB on the embedding and lm_head.
    """
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        exact = weights.spec.vocab_size * weights.spec.hidden_size * 2
        assert weights.embed.nbytes == exact
        assert weights.lm_head.nbytes == exact
        assert weights.final_norm.nbytes == weights.spec.hidden_size * 2
    finally:
        weights.close()
