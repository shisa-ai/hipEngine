"""Dense Llama prefill end to end against a BF16-rounded NumPy reference.

The reference rounds activations to BF16 at every op boundary and normalizes
attention in FP64 over the BF16 Q/K the kernel consumes. Two levels of agreement
are asserted, because they mean different things:

* the first layer's DMS tap (normalized hidden, post-RoPE Q, post-RoPE K) is
  **bit-identical**, which is what pins the embedding gather, RMSNorm, projection
  layout, and rotary convention;
* end-to-end logits agree to BF16 representation noise. Each op is bit-exact, but
  a value that lands on a rounding boundary in one association and not the other
  flips by one BF16 ULP, and those flips accumulate across layers. A tolerance
  below one ULP would assert that two different F32 accumulation orders round
  identically, which is not a property of the kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._llama_tiny import (
    TINY_LLAMA_CONFIG,
    tiny_llama_reference,
    write_tiny_llama_checkpoint,
)
from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

_VOCAB = int(TINY_LLAMA_CONFIG["vocab_size"])
_HIDDEN = int(TINY_LLAMA_CONFIG["hidden_size"])
_HEADS = int(TINY_LLAMA_CONFIG["num_attention_heads"])
_KV_HEADS = int(TINY_LLAMA_CONFIG["num_key_value_heads"])
_HEAD_DIM = int(TINY_LLAMA_CONFIG["head_dim"])


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-llama")
    return write_tiny_llama_checkpoint(directory)


def _runner(checkpoint_path):
    from hipengine.loading.llama_safetensors import load_llama_spec, materialize_llama_weights
    from hipengine.loading.safetensors import load_weight_index
    from hipengine.runtime.llama_prefill import LlamaPrefillRunner

    index = load_weight_index(checkpoint_path)
    loaded = materialize_llama_weights(index, load_llama_spec(index))
    return LlamaPrefillRunner(loaded), loaded


class _RecordingSink:
    hidden_size = _HIDDEN
    num_q_heads = _HEADS
    num_kv_heads = _KV_HEADS
    head_dim = _HEAD_DIM
    input_stage = "post_attn_rmsnorm_pre_q_projection"
    physical_layer_ids = tuple(range(int(TINY_LLAMA_CONFIG["num_hidden_layers"])))

    def __init__(self) -> None:
        self.chunks: dict[int, dict[str, np.ndarray]] = {}
        self.teacher: np.ndarray | None = None

    def capture_chunk(self, **kwargs) -> None:
        self.chunks[int(kwargs["physical_layer_id"])] = kwargs

    def capture_teacher_logits(self, logits: np.ndarray) -> None:
        self.teacher = np.asarray(logits)


@pytest.mark.parametrize("prompt", [(3, 9, 14, 27), tuple(range(11))])
def test_prefill_logits_track_the_bf16_reference(tiny_checkpoint, prompt):
    path, _f32, bf16 = tiny_checkpoint
    runner, loaded = _runner(path)
    try:
        tokens = np.asarray(prompt, dtype=np.int64)
        logits = runner.prefill(tokens, logits="last")
        reference = tiny_llama_reference(bf16, TINY_LLAMA_CONFIG, tokens).logits()
    finally:
        runner.close()
        loaded.free()

    assert logits.shape == (1, _VOCAB)
    assert np.all(np.isfinite(logits))
    peak = float(np.max(np.abs(reference)))
    assert peak > 0.0
    assert float(np.max(np.abs(logits[0] - reference))) <= 1.0e-2 * peak
    assert int(np.argmax(logits[0])) == int(np.argmax(reference))


def test_first_layer_tap_is_bit_identical_to_the_reference(tiny_checkpoint):
    path, _f32, bf16 = tiny_checkpoint
    runner, loaded = _runner(path)
    tokens = np.asarray((5, 11, 17, 23), dtype=np.int64)
    sink = _RecordingSink()
    try:
        logits = runner.prefill(tokens, capture=sink, logits="last")
    finally:
        runner.close()
        loaded.free()

    reference = tiny_llama_reference(bf16, TINY_LLAMA_CONFIG, tokens)
    assert sorted(sink.chunks) == list(range(int(TINY_LLAMA_CONFIG["num_hidden_layers"])))

    first = sink.chunks[0]
    normed, query, key = reference.taps[0]
    assert first["hidden_bf16"].shape == (tokens.size, _HIDDEN)
    assert first["hidden_bf16"].dtype == np.uint16
    assert first["query_f32"].dtype == np.float32
    assert first["query_f32"].shape == (tokens.size, _HEADS, _HEAD_DIM)
    assert first["key_f32"].shape == (tokens.size, _KV_HEADS, _HEAD_DIM)
    assert np.array_equal(first["positions"], np.arange(tokens.size, dtype=np.int32))

    from tests._llama_tiny import bf16_bits

    assert np.array_equal(first["hidden_bf16"], bf16_bits(normed))
    assert np.array_equal(first["query_f32"], query)
    assert np.array_equal(first["key_f32"], key)

    # The tap is the normalized pre-projection stage: unit RMS per row times the
    # norm weight, which the fixture sets to ones.
    hidden = (first["hidden_bf16"].astype(np.uint32) << np.uint32(16)).view(np.float32)
    assert np.allclose(np.sqrt(np.mean(hidden.astype(np.float64) ** 2, axis=-1)), 1.0, atol=5.0e-2)

    # Every later layer stays inside BF16 representation noise of the reference.
    for layer in range(1, int(TINY_LLAMA_CONFIG["num_hidden_layers"])):
        tap_norm, tap_query, tap_key = reference.taps[layer]
        captured = (sink.chunks[layer]["hidden_bf16"].astype(np.uint32) << np.uint32(16)).view(
            np.float32
        )
        assert float(np.max(np.abs(captured - tap_norm))) <= 1.0e-2 * float(
            np.sqrt((tap_norm**2).mean())
        )
        assert float(np.max(np.abs(sink.chunks[layer]["query_f32"] - tap_query))) <= 1.0e-2
        assert float(np.max(np.abs(sink.chunks[layer]["key_f32"] - tap_key))) <= 1.0e-2

    assert sink.teacher is not None
    assert np.array_equal(sink.teacher, logits[0])


def test_row_subset_logits_agree_with_the_full_projection(tiny_checkpoint):
    path, _f32, bf16 = tiny_checkpoint
    runner, loaded = _runner(path)
    tokens = np.asarray((2, 4, 6, 8, 10), dtype=np.int64)
    try:
        subset = runner.prefill(tokens, logits=(1, 3))
        every = runner.prefill(tokens, logits="all")
    finally:
        runner.close()
        loaded.free()

    reference = tiny_llama_reference(bf16, TINY_LLAMA_CONFIG, tokens)
    expected = np.stack([reference.logits(1), reference.logits(3)])
    peak = float(np.max(np.abs(expected)))
    assert subset.shape == (2, _VOCAB)
    assert float(np.max(np.abs(subset - expected))) <= 1.0e-2 * peak
    # "all" runs the BF16 bulk projection, so it is compared as a distribution.
    peak_all = float(np.max(np.abs(every)))
    assert every.shape == (tokens.size, _VOCAB)
    assert float(np.max(np.abs(every[[1, 3]] - subset))) <= 5.0e-2 * peak_all
