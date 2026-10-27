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


def test_row_subset_logits_agree_with_the_full_projection(tiny_checkpoint, monkeypatch):
    path, _f32, bf16 = tiny_checkpoint
    runner, loaded = _runner(path)
    tokens = np.asarray((2, 4, 6, 8, 10), dtype=np.int64)

    def reject_host_roundtrip(*args, **kwargs):
        pytest.fail("selected logits must not download the full normalized prompt")

    monkeypatch.setattr(runner, "_copy_bf16_rows", reject_host_roundtrip)
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


@pytest.mark.parametrize("tokens,keys", [(1, 1), (1, 37), (7, 7), (7, 128), (8, 129), (9, 129), (17, 1024), (33, 65)])
@pytest.mark.parametrize("mask_kind", ["causal", "holes", "empty"])
def test_tiled_attention_matches_the_exact_kernel(tokens, keys, mask_kind):
    """The tiled route is the default for head_dim 128; the exact one is strict.

    Shared memory in the tiled kernel is a fixed K/V tile, so ``keys`` only
    changes how many tiles are walked -- the point of the parameterization is
    that 1024 keys and 1 key take the same code path, and that a key count which
    is not a multiple of the tile exercises the partial tail.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention import (
        causal_keep_mask,
        dense_causal_prefill_bf16,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(keys)
    heads, kv_heads, head_dim = 4, 2, 128
    width = heads * head_dim
    kv_width = kv_heads * head_dim

    q = _f32_to_bf16_bits(rng.standard_normal((tokens, width)).astype(np.float32) * 0.5)
    k = _f32_to_bf16_bits(rng.standard_normal((keys, kv_width)).astype(np.float32) * 0.5)
    v = _f32_to_bf16_bits(rng.standard_normal((keys, kv_width)).astype(np.float32) * 0.5)
    mask = causal_keep_mask(tokens, keys=keys)
    if mask_kind == "holes":
        mask[:, ::3] = 0
    elif mask_kind == "empty":
        mask[:] = 0

    def run(q_buf, k_buf, v_buf, mask_ptr, strict):
        out = malloc(tokens * width * 2, runtime=runtime)
        try:
            dense_causal_prefill_bf16(
                q_buf.ptr,
                k_buf.ptr,
                v_buf.ptr,
                mask_ptr,
                out.ptr,
                tokens=tokens,
                keys=keys,
                num_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                scale=head_dim**-0.5,
                strict=strict,
                runtime=runtime,
            )
            host = np.empty((tokens, width), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(host), out, tokens * width * 2, runtime=runtime)
            return host
        finally:
            free(out, runtime=runtime)

    buffers = []
    try:
        for host in (q, k, v, mask):
            buf = malloc(host.nbytes, runtime=runtime)
            copy_host_to_device(buf, host_array_ptr(host), host.nbytes, runtime=runtime)
            buffers.append(buf)
        q_buf, k_buf, v_buf, mask_buf = buffers

        exact = run(q_buf, k_buf, v_buf, mask_buf.ptr, True)
        tiled = run(q_buf, k_buf, v_buf, mask_buf.ptr, False)
        # A null mask asserts the causal precondition; it must agree with the
        # explicit causal mask, because the walk already excludes those keys.
        implicit = run(q_buf, k_buf, v_buf, 0, False) if mask_kind == "causal" else None
        repeat = run(q_buf, k_buf, v_buf, mask_buf.ptr, False)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    reference = _bf16_bits_to_f32(exact)
    tiled_f32 = _bf16_bits_to_f32(tiled)

    # An independent oracle, so a shared bug in both kernels cannot pass: FP64
    # softmax over the same BF16 operands, in the same causal frame.
    oracle = _attention_oracle(q, k, v, mask, heads=heads, kv_heads=kv_heads, head_dim=head_dim)
    scale = float(np.abs(oracle).max()) or 1.0
    # Both kernels emit BF16, whose ULP at the largest magnitude here is 2**-8
    # of it. A tolerance below that would assert that the FP64 oracle and a
    # BF16 store round identically, which is not a property of either.
    assert np.max(np.abs(tiled_f32 - oracle)) <= 5e-3 * scale
    live_rows = np.any(mask, axis=1)
    if np.any(live_rows):
        assert np.max(np.abs(reference[live_rows] - oracle[live_rows])) <= 5e-3 * scale
    # Gemma's strict fallback explicitly defines an empty softmax as NaN;
    # the tiled kernel defines it as zero. Assert both contracts, not parity
    # between undefined strict distributions and the tiled empty-row result.
    assert np.all(np.isnan(reference[~live_rows]))
    assert np.all(tiled_f32[~live_rows] == 0)
    if implicit is not None:
        assert np.array_equal(implicit, tiled), "a null mask must equal the causal mask"
    assert np.array_equal(repeat, tiled)


def _attention_oracle(q, k, v, mask, *, heads, kv_heads, head_dim):
    """FP64 causal attention over BF16 operands, grouped-query aware."""

    qf = _bf16_bits_to_f32(q).astype(np.float64)
    kf = _bf16_bits_to_f32(k).astype(np.float64)
    vf = _bf16_bits_to_f32(v).astype(np.float64)
    group = heads // kv_heads
    out = np.zeros_like(qf)
    for head in range(heads):
        kv_head = head // group
        qh = qf[:, head * head_dim : (head + 1) * head_dim]
        kh = kf[:, kv_head * head_dim : (kv_head + 1) * head_dim]
        vh = vf[:, kv_head * head_dim : (kv_head + 1) * head_dim]
        scores = qh @ kh.T * head_dim**-0.5
        scores = np.where(mask.astype(bool), scores, -np.inf)
        maxima = scores.max(axis=1, keepdims=True)
        scores -= np.where(np.isfinite(maxima), maxima, 0.0)
        weights = np.exp(scores)
        denominator = weights.sum(axis=1, keepdims=True)
        weights /= np.where(denominator > 0, denominator, 1.0)
        out[:, head * head_dim : (head + 1) * head_dim] = weights @ vh
    return out


def _bf16_bits_to_f32(host: np.ndarray) -> np.ndarray:
    return (host.astype(np.uint32) << 16).view(np.float32).reshape(host.shape)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> BF16 bit patterns, as uint16."""

    bits = values.astype(np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)
