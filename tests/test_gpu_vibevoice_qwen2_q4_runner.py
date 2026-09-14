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


def test_batched_prefill_matches_row_by_row(runtime, lm) -> None:
    """The batched WMMA prefill must reproduce the row-by-row result.

    ``q4_prefill`` was a per-row loop through the decode path; it now runs
    the raw-block Q4_K/Q6_K WMMA prefill kernels. Both routes must land on
    the same post-prefill hidden row, otherwise the batched GEMMs changed
    the arithmetic rather than just the schedule.
    """
    from hipengine.core.memory import free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    total = len(rows)
    prompt = malloc(total * hidden * 2)
    try:
        import numpy as np

        from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, host_array_ptr

        copy_host_array_to_device(prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))

        runtime.reset()
        runtime.prefill_rows(prompt, total, 0)
        runtime.runtime.memcpy(runtime._hidden.ptr,
                               prompt.ptr + (total - 1) * hidden * 2, hidden * 2, 3)
        batched_logits, batched_top = runtime.logits_argmax()

        runtime.reset()
        for i, row in enumerate(rows):
            runtime.push_token(row, i)
            runtime.forward_layers(i)
        sequential_logits, sequential_top = runtime.logits_argmax()

        assert batched_top == sequential_top, (
            f"batched prefill top-1 {batched_top} != row-by-row {sequential_top}")
        # Same schedule class: the batched WMMA tiles reassociate, so require
        # closeness rather than bit equality. Normalise by the logit scale,
        # not per-element (near-zero logits make a per-element ratio useless).
        scale = float(np.max(np.abs(sequential_logits)))
        rel = float(np.max(np.abs(batched_logits - sequential_logits))) / scale
        assert rel < 0.05, f"batched prefill logit drift {rel:.3e} (scale {scale:.3f})"
        assert set(np.argsort(sequential_logits)[-5:]) == set(np.argsort(batched_logits)[-5:]), \
            "batched prefill changed the top-5 set"
    finally:
        free(prompt)


def test_batched_prefill_refuses_unsupported_quant_types(runtime, lm, monkeypatch) -> None:
    """A non-Q4_K/Q6_K weight must fail loudly, never decode as Q4_K.

    The prefill route used to default any unregistered type to the Q4_K
    decoder, so a Q5_K / Q8_0 / IQ4_XS tensor would have produced silently
    wrong logits. Only the types with a WMMA prefill kernel may route.
    """
    from hipengine.core.memory import malloc
    from hipengine.kernels.hip_gfx1100.vibevoice import q4 as q4mod
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES

    rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    total = len(rows)
    prompt = malloc(total * hidden * 2)
    try:
        import numpy as np

        from hipengine.core.memory import copy_host_array_to_device
        from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

        copy_host_array_to_device(prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
        layer = runtime.layers[0]

        # Claim the q_proj weight is Q5_K: supported by the decode GEMV path,
        # but with no WMMA prefill kernel.
        GGML_Q5_K = 13
        original = GGUF_WEIGHT_TYPES.get(layer.q_w.ptr)
        monkeypatch.setitem(GGUF_WEIGHT_TYPES, layer.q_w.ptr, GGML_Q5_K)
        try:
            runtime.reset()
            with pytest.raises(ValueError, match="no WMMA prefill kernel"):
                runtime.prefill_rows(prompt, total, 0)
        finally:
            if original is None:
                GGUF_WEIGHT_TYPES.pop(layer.q_w.ptr, None)
            else:
                GGUF_WEIGHT_TYPES[layer.q_w.ptr] = original

        # An unregistered pointer must not be guessed at either.
        monkeypatch.setitem(GGUF_WEIGHT_TYPES, layer.q_w.ptr, None)
        try:
            runtime.reset()
            with pytest.raises(ValueError, match="no GGUF type registered"):
                runtime.prefill_rows(prompt, total, 0)
        finally:
            if original is not None:
                GGUF_WEIGHT_TYPES[layer.q_w.ptr] = original
    finally:
        from hipengine.core.memory import free

        free(prompt)
