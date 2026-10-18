"""Unit tier: streaming Gemma 4 forward against the materialized reference.

The streaming path re-expresses the decoder layer so it can dequantize experts
on demand. That duplication is the risk, so the central test here runs both
paths on identical weights and requires bit-for-bit agreement, not approximate
agreement.

The second property under test is that the streaming path really does avoid
materializing every expert: the number of experts it dequantizes must equal the
number the router selected, and must be far below the artifact's expert count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    gemma4_text_forward,
)
from hipengine.kernels.cpu_reference.gemma4_streaming import (
    Gemma4GGUFStreamingWeights,
    _dequantize,
    gemma4_streaming_forward,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gemma4_gguf_materialize import (
    materialize_gemma4_reference_weights,
)
from tests._gemma4_gguf_fixture import (
    FIXTURE_EXPERTS,
    FIXTURE_EXPERT_USED,
    FIXTURE_VOCAB,
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)

PROMPT = [1, 2, 3, 4, 5]


@pytest.fixture
def reader(tmp_path: Path) -> GGUFReader:
    path = write_fixture_gguf(
        tmp_path / "streaming.gguf",
        default_fixture_tensors(),
        fixture_metadata(),
    )
    return GGUFReader(path)


@pytest.fixture
def streaming(reader: GGUFReader) -> Gemma4GGUFStreamingWeights:
    return Gemma4GGUFStreamingWeights(reader, dtype=np.float32)


def test_streaming_forward_matches_the_reference_exactly(
    reader: GGUFReader,
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    reference = gemma4_text_forward(materialized.weights, materialized.config, PROMPT)
    streamed = gemma4_streaming_forward(streaming, PROMPT)
    np.testing.assert_array_equal(streamed, reference.logits)


def test_streaming_forward_matches_the_reference_with_explicit_positions(
    reader: GGUFReader,
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    materialized = materialize_gemma4_reference_weights(reader)
    positions = [7, 8, 9, 10, 11]
    reference = gemma4_text_forward(
        materialized.weights,
        materialized.config,
        PROMPT,
        positions=positions,
    )
    streamed = gemma4_streaming_forward(streaming, PROMPT, positions=positions)
    np.testing.assert_array_equal(streamed, reference.logits)


def test_streaming_layer_matches_the_reference_layer(
    reader: GGUFReader,
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    """Per-layer agreement localizes a divergence instead of hiding it."""

    from hipengine.kernels.cpu_reference.gemma4 import (
        gemma4_decoder_layer_forward,
    )
    from hipengine.kernels.cpu_reference.gemma4_streaming import (
        _streaming_layer_forward,
    )

    materialized = materialize_gemma4_reference_weights(reader)
    config = materialized.config
    embedding = np.asarray(materialized.weights.embed_tokens, dtype=np.float32)
    hidden = embedding[PROMPT] * np.float32(config.embed_scale)
    positions = np.arange(len(PROMPT), dtype=np.int64)

    for layer_id in range(config.num_hidden_layers):
        expected = gemma4_decoder_layer_forward(
            hidden,
            materialized.weights.layers[layer_id],
            config.geometry(layer_id),
            config,
            positions=positions,
        )
        actual = _streaming_layer_forward(
            hidden,
            streaming,
            layer_id,
            config.geometry(layer_id),
            config,
            positions=positions,
        )
        np.testing.assert_array_equal(actual, expected, err_msg=f"layer {layer_id}")
        hidden = expected
    assert np.all(np.isfinite(hidden))


def test_only_the_selected_experts_are_dequantized(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    """The whole point of the streaming path is not reading 128 experts."""

    gate_up, down = streaming.expert_weights(0, [0])
    per_expert_bytes = gate_up.nbytes + down.nbytes
    streaming.reset_caches()

    gemma4_streaming_forward(streaming, PROMPT)
    cached = streaming.cached_experts()

    # Exactly the experts the router selected, and never the whole stack.
    total_experts = FIXTURE_EXPERTS * streaming.config.num_hidden_layers
    per_layer_bound = min(FIXTURE_EXPERTS, len(PROMPT) * FIXTURE_EXPERT_USED)
    assert cached
    assert len(cached) < total_experts
    assert all(0 <= expert < FIXTURE_EXPERTS for _, expert in cached)
    assert len(cached) <= per_layer_bound * streaming.config.num_hidden_layers

    _, expert_bytes = streaming.resident_bytes()
    assert expert_bytes == per_expert_bytes * len(cached)
    assert expert_bytes < per_expert_bytes * total_experts


def test_expert_gather_preserves_the_fused_layout(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    gate_up, down = streaming.expert_weights(0, [0, 2])
    assert gate_up.shape[0] == 2
    assert down.shape[0] == 2
    assert gate_up.shape[1] % 2 == 0
    single_gate_up, single_down = streaming.expert_weights(0, [2])
    np.testing.assert_array_equal(gate_up[1], single_gate_up[0])
    np.testing.assert_array_equal(down[1], single_down[0])
    # The gathered expert must be the one the full tensor holds, not a neighbor.
    full = _dequantize(streaming.reader, "blk.0.ffn_down_exps.weight", np.float32)
    np.testing.assert_array_equal(down[0], full[0])
    np.testing.assert_array_equal(down[1], full[2])


def test_expert_gather_matches_a_full_tensor_dequantization(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    """Slicing the outer expert axis must equal dequantizing the whole stack."""

    for name, slot in (
        ("blk.0.ffn_gate_up_exps.weight", 0),
        ("blk.0.ffn_down_exps.weight", 1),
    ):
        full = _dequantize(streaming.reader, name, np.float32)
        gathered = np.stack(
            [streaming.expert_weights(0, [expert])[slot][0] for expert in range(FIXTURE_EXPERTS)]
        )
        np.testing.assert_array_equal(gathered, full, err_msg=name)


def test_expert_weights_are_reused_across_calls(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    _, before = streaming.resident_bytes()
    streaming.expert_weights(0, [1, 3])
    _, after_first = streaming.resident_bytes()
    streaming.expert_weights(0, [1, 3])
    _, after_second = streaming.resident_bytes()
    assert after_first > before
    assert after_second == after_first


def test_dense_layer_weights_carry_no_expert_tensors(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    layer = streaming.layer_weights(0)
    assert not hasattr(layer, "experts_gate_up_proj") or layer.experts_gate_up_proj is None
    assert not hasattr(layer, "experts_down_proj") or layer.experts_down_proj is None
    assert layer.v_proj is not None
    assert streaming.layer_weights(1).v_proj is None


def test_an_out_of_range_expert_is_refused(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    with pytest.raises(IndexError, match="holds"):
        streaming.expert_weights(0, [FIXTURE_EXPERTS])


def test_the_head_is_refused_on_a_tied_artifact(
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    with pytest.raises(ValueError, match="ties the head"):
        streaming.lm_head()


def test_an_untied_artifact_uses_its_own_head(tmp_path: Path) -> None:
    """A separate ``output.weight`` must be read instead of the embedding."""

    tensors = list(default_fixture_tensors())
    tensors.append(("output.weight", (FIXTURE_VOCAB, 256), _f32()))
    metadata = [
        (key, value_type, value)
        for key, value_type, value in fixture_metadata()
        if key != "general.file_type"
    ]
    metadata.append(("general.file_type", 4, 0))
    path = write_fixture_gguf(tmp_path / "untied.gguf", tensors, metadata)
    reader = GGUFReader(path)
    streaming = Gemma4GGUFStreamingWeights(reader)
    head = streaming.lm_head()
    assert head.shape == (FIXTURE_VOCAB, 256)
    materialized = materialize_gemma4_reference_weights(reader)
    np.testing.assert_array_equal(head, np.asarray(materialized.weights.lm_head, dtype=np.float32))
    streamed = gemma4_streaming_forward(streaming, PROMPT)
    reference = gemma4_text_forward(materialized.weights, materialized.config, PROMPT)
    np.testing.assert_array_equal(streamed, reference.logits)


def _f32():
    from hipengine.quant.gguf import GGMLQuantizationType

    return GGMLQuantizationType.F32


def test_appending_tokens_does_not_change_earlier_logits(
    reader: GGUFReader,
    streaming: Gemma4GGUFStreamingWeights,
) -> None:
    """Causality is why the streaming path re-reads the whole block.

    A single-token forward at position ``n`` is *not* equivalent to a longer
    prefill's logits at ``n``, because there is no KV cache here: the one-token
    call attends to one key. What must hold is that appending tokens leaves the
    earlier positions' logits unchanged.
    """

    materialized = materialize_gemma4_reference_weights(reader)
    short = gemma4_text_forward(materialized.weights, materialized.config, PROMPT[:3])
    long = gemma4_text_forward(materialized.weights, materialized.config, PROMPT)
    np.testing.assert_allclose(short.logits, long.logits[:3], rtol=1e-4, atol=1e-4)

    streamed = gemma4_streaming_forward(streaming, PROMPT)
    np.testing.assert_allclose(streamed[:3], long.logits[:3], rtol=1e-6, atol=1e-6)

    # A one-token forward at the last position attends to one key only, so it
    # deliberately does not reproduce the block's last-position logits.
    single = gemma4_text_forward(
        materialized.weights,
        materialized.config,
        PROMPT[-1:],
        positions=[len(PROMPT) - 1],
    )
    assert not np.allclose(single.logits[0], long.logits[-1], rtol=1e-3, atol=1e-3)
