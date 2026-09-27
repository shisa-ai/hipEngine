"""Gemma 4 runner: config derivation, weight load, and forward-pass parity.

The runner is compared against the CPU reference on a tiny quantized GGUF
artifact. Both sides read the *same* weights: the reference state dict is built
from the GGUF tensors dequantized, which is exactly what the device path reads.
That makes the only difference between the two the arithmetic precision, so a
divergence is a real defect rather than a fixture mismatch.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    gemma4_text_forward,
    gemma4_text_weights_from_hf,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.gemma4 import (
    Gemma4Runner,
    gemma4_text_config_from_gguf,
    load_gemma4_device_weights,
)
from tests._gemma4_gguf_fixture import (
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)
from tests._rocm_guard import hip_runtime_available

_needs_hip = pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable; skipping Gemma 4 runner"
)

# GGUF tensor name -> HF state-dict name, with `{layer}` for the block index.
# The reference adapter reads HF names, so this is the bridge between the two
# naming schemes rather than a second source of truth about the model.
_GGUF_TO_HF = {
    "attn_norm": "input_layernorm.weight",
    "attn_q": "self_attn.q_proj.weight",
    "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight",
    "attn_k_norm": "self_attn.k_norm.weight",
    "post_attention_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight",
    "ffn_gate": "mlp.gate_proj.weight",
    "ffn_up": "mlp.up_proj.weight",
    "ffn_down": "mlp.down_proj.weight",
    "ffn_gate_inp": "router.proj.weight",
    "ffn_gate_inp_scale": "router.scale",
    "ffn_down_exps_scale": "router.per_expert_scale",
    "ffn_gate_up_exps": "experts.gate_up_proj",
    "ffn_down_exps": "experts.down_proj",
    "pre_ffw_norm_2": "pre_feedforward_layernorm_2.weight",
    "post_ffw_norm": "post_feedforward_layernorm.weight",
    "post_ffw_norm_1": "post_feedforward_layernorm_1.weight",
    "post_ffw_norm_2": "post_feedforward_layernorm_2.weight",
    "layer_output_scale": "layer_scalar",
}


@pytest.fixture()
def artifact(tmp_path: pathlib.Path) -> GGUFReader:
    path = write_fixture_gguf(
        tmp_path / "gemma4.gguf", default_fixture_tensors(), fixture_metadata()
    )
    return GGUFReader(path)


def _reference_state_dict(reader: GGUFReader) -> dict[str, np.ndarray]:
    """Build an HF-shaped state dict from the artifact's dequantized tensors.

    Reading the GGUF rather than the generator's source values is deliberate: it
    is the same data the device path consumes, so the comparison isolates
    precision instead of also measuring quantization.
    """

    state: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.asarray(
            reader.dequantize_tensor("token_embd.weight"), dtype=np.float32
        ),
        "model.norm.weight": np.asarray(
            reader.dequantize_tensor("output_norm.weight"), dtype=np.float32
        ),
    }
    for tensor in reader.info.tensors:
        name = tensor.name
        if not name.startswith("blk."):
            continue
        _, block, rest = name.split(".", 2)
        # The loader distinguishes `ffn_gate_inp.weight` from
        # `ffn_gate_inp.scale` by suffix, so collapsing to the stem here would
        # map both to the same HF name and silently overwrite one with the other.
        if rest.endswith(".scale"):
            slot = rest[: -len(".scale")] + "_scale"
        elif rest.endswith(".weight"):
            slot = rest[: -len(".weight")]
        else:
            continue
        suffix = _GGUF_TO_HF.get(slot)
        if suffix is None:
            continue
        state[f"model.layers.{block}.{suffix}"] = np.asarray(
            reader.dequantize_tensor(name), dtype=np.float32
        )
    return state


def _logits_from_reference(
    reader: GGUFReader, token_ids: list[int], *, softcap: bool = True
) -> np.ndarray:
    """Reference logits, optionally with softcapping disabled.

    The tiny fixture's random weights produce logits far larger than the cap, so
    with capping on every value saturates to +-cap and a parity comparison would
    be comparing saturated values. Disabling it on both sides keeps the
    comparison sharp.
    """

    import dataclasses

    from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata
    from hipengine.runtime.gemma4 import gemma4_text_config_from_gguf

    config = gemma4_text_config_from_gguf(
        gemma4_gguf_config_from_metadata(reader.info),
        tensor_names=tuple(t.name for t in reader.info.tensors),
    )
    if not softcap:
        config = dataclasses.replace(config, final_logit_softcapping=None)
    weights = gemma4_text_weights_from_hf(_reference_state_dict(reader), config)
    return np.asarray(
        gemma4_text_forward(weights, config, token_ids).logits, dtype=np.float32
    )


def test_the_config_comes_from_the_artifact_metadata(artifact: GGUFReader) -> None:
    """A user with a .gguf has a .gguf; no HF config.json is required."""

    from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata

    config = gemma4_text_config_from_gguf(
        gemma4_gguf_config_from_metadata(artifact.info),
        tensor_names=tuple(t.name for t in artifact.info.tensors),
    )

    assert config.num_hidden_layers == 2
    assert config.tie_word_embeddings is True
    assert config.embed_scale == pytest.approx(16.0)

    sliding = config.geometry(0)
    assert sliding.layer_type == "sliding_attention"
    assert (sliding.num_heads, sliding.num_kv_heads, sliding.head_dim) == (4, 2, 64)
    assert sliding.sliding_window == 16
    assert sliding.k_eq_v is False

    # The global layer carries no attn_v tensor, and the config must reflect
    # that rather than defaulting to "a v_proj exists".
    global_layer = config.geometry(1)
    assert global_layer.layer_type == "full_attention"
    assert global_layer.sliding_window is None
    assert global_layer.k_eq_v is True
    assert global_layer.head_dim == 128
    # Proportional RoPE: 16 of 64 pairs rotate on the global layer.
    assert global_layer.rope.rope_angles == 16


def test_the_two_layer_types_get_different_rope_tables(artifact: GGUFReader) -> None:
    """Per-layer geometry, not one table for the model.

    A runner that built the rope tables once from layer 0 would rotate the global
    layers as though they were sliding layers: wrong head width, wrong angle
    count, wrong base. The two geometries must not coincide, or this fixture
    could not detect that.
    """

    from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata

    config = gemma4_text_config_from_gguf(
        gemma4_gguf_config_from_metadata(artifact.info),
        tensor_names=tuple(t.name for t in artifact.info.tensors),
    )
    sliding, global_layer = config.geometry(0), config.geometry(1)

    assert sliding.head_dim != global_layer.head_dim
    assert sliding.rope.rope_angles != global_layer.rope.rope_angles
    assert sliding.rope.rope_theta != global_layer.rope.rope_theta


@_needs_hip
def test_runner_prefill_matches_the_reference(artifact: GGUFReader) -> None:
    """Prefill parity against the CPU reference."""

    token_ids = [1, 5, 9, 13]
    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=32)
    try:
        got = runner.forward(token_ids, apply_softcap=False)
        expected = _logits_from_reference(artifact, token_ids, softcap=False)
        # The reference returns logits for every position; the runner's contract
        # is the last row, which is what generation reads. Compare that row.
        assert got.shape == expected[-1].shape
        expected = expected[-1]
        assert np.isfinite(got).all()
        scale = float(np.abs(expected).max())
        assert scale > 0, "reference logits are all zero; the fixture is degenerate"
        # bf16 activations through 2 layers; the bound is on relative error
        # against the logit magnitude, not an absolute tolerance that would pass
        # trivially on small logits.
        assert np.allclose(got, expected, rtol=5e-2, atol=5e-2 * scale), (
            f"prefill diverged: max abs diff {np.abs(got - expected).max():.4g} "
            f"against scale {scale:.4g}"
        )
    finally:
        runner.close()
        weights.free()


@_needs_hip
def test_incremental_decode_matches_a_dense_prefill(artifact: GGUFReader) -> None:
    """Decode one token at a time and prefill the same sequence in one call.

    This is the check the KV cache exists for. It fails if the cache is written
    at the wrong offset, if the mask does not cover the cached range, or if a
    decode step is treated as though it were a wide block.
    """

    token_ids = [3, 7, 11, 2, 5]
    weights = load_gemma4_device_weights(artifact)
    try:
        dense = Gemma4Runner(weights=weights, capacity=32)
        try:
            dense_logits = dense.forward(token_ids)
        finally:
            dense.close()

        incremental = Gemma4Runner(weights=weights, capacity=32)
        try:
            step_logits = None
            for token in token_ids:
                step_logits = incremental.forward([token])
            assert incremental.position == len(token_ids)
        finally:
            incremental.close()

        scale = float(np.abs(dense_logits).max())
        assert scale > 0
        assert np.allclose(step_logits, dense_logits, rtol=5e-2, atol=5e-2 * scale), (
            f"incremental decode diverged from dense prefill: "
            f"max abs diff {np.abs(step_logits - dense_logits).max():.4g} "
            f"against scale {scale:.4g}"
        )
    finally:
        weights.free()


@_needs_hip
def test_a_prompt_wider_than_max_block_is_chunked_exactly(
    artifact: GGUFReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt wider than the block bound must not need a wider scratch.

    The per-layer scratch is sized from ``max_block``, so a runner that refused
    a wider prompt would force every caller to size scratch for the whole
    context. The chunked path has to produce the same logits as one wide block:
    the mask is absolute-position based, so block boundaries are an execution
    detail, not part of the computation.

    The fixture is 12 tokens long, so the wide side is what a real 8192-token
    context would be relative to a 512-token block -- a prompt several times the
    block bound.

    The subject here is scratch sizing and absolute-position masking, not
    arithmetic equality, so the exact route is pinned. The fixture's weights are
    synthetic, which puts it far off any trained manifold where a changed
    arithmetic association amplifies instead of staying bounded; arithmetic
    equality between the routes belongs to the campaign teacher-forced gate,
    which measures it on the real artifact.
    """

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    token_ids = [3, 7, 11, 2, 5, 9, 1, 4, 6, 8, 10, 12]
    weights = load_gemma4_device_weights(artifact)
    try:
        wide = Gemma4Runner(weights=weights, capacity=32, max_block=len(token_ids))
        try:
            wide_logits = wide.forward(token_ids)
            assert wide.position == len(token_ids)
        finally:
            wide.close()

        chunked = Gemma4Runner(weights=weights, capacity=32, max_block=4)
        try:
            chunked_logits = chunked.forward(token_ids)
            assert chunked.position == len(token_ids), (
                "a chunked forward advanced the position by "
                f"{chunked.position} instead of {len(token_ids)}"
            )
        finally:
            chunked.close()

        scale = float(np.abs(wide_logits).max())
        assert scale > 0
        assert np.allclose(chunked_logits, wide_logits, rtol=5e-2, atol=5e-2 * scale), (
            f"chunked prefill diverged from a single wide block: "
            f"max abs diff {np.abs(chunked_logits - wide_logits).max():.4g} "
            f"against scale {scale:.4g}"
        )
    finally:
        weights.free()


@_needs_hip
def test_a_sliding_layer_ignores_positions_beyond_its_window(artifact: GGUFReader) -> None:
    """The sliding window must actually bound attention.

    Runs a prompt longer than the fixture's window and checks the logits against
    a reference whose window is enforced. If the runner ignored the window, the
    layer would attend to older positions and the logits would differ.
    """


    window = 16
    token_ids = list(range(1, window + 9))
    assert len(token_ids) > window, "prompt must exceed the window for this to test anything"

    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=64)
    try:
        got = runner.forward(token_ids, apply_softcap=False)
    finally:
        runner.close()
        weights.free()

    expected = _logits_from_reference(artifact, token_ids, softcap=False)[-1]
    scale = float(np.abs(expected).max())
    assert scale > 0
    assert np.allclose(got, expected, rtol=5e-2, atol=5e-2 * scale), (
        f"sliding-window run diverged: max abs diff "
        f"{np.abs(got - expected).max():.4g} against scale {scale:.4g}"
    )


@_needs_hip
def test_the_softcap_is_applied_and_matches_the_reference(artifact: GGUFReader) -> None:
    """``forward`` returns the model's distribution, which is capped.

    Capping belongs in the forward pass, not the sampler: every consumer of the
    logits must see the distribution the model defines, and both CPU references
    apply it at the same point.
    """

    token_ids = [1, 5, 9, 13]
    cap = np.float32(30.0)
    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=32)
    try:
        capped = runner.forward(token_ids)
        runner.reset()
        raw = runner.forward(token_ids, apply_softcap=False)
    finally:
        runner.close()
        weights.free()

    assert np.abs(capped).max() <= float(cap) + 1e-3, "cap not applied"
    # The fixture's raw logits exceed the cap, so capping must actually bite;
    # otherwise this test would pass on a runner that ignored the cap.
    assert np.abs(raw).max() > float(cap) * 2, (
        f"fixture logits ({np.abs(raw).max():.4g}) do not exceed the cap, so this "
        "test cannot tell whether capping happened"
    )
    expected = (np.tanh(raw / cap) * cap).astype(np.float32)
    assert np.allclose(capped, expected, rtol=1e-5, atol=1e-5)


@_needs_hip
def test_reset_rewinds_the_cache(artifact: GGUFReader) -> None:
    """A reset runner must reproduce its first run exactly."""

    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=32)
    try:
        first = runner.forward([2, 4, 6])
        assert runner.position == 3
        runner.reset()
        assert runner.position == 0
        second = runner.forward([2, 4, 6])
        # Not merely close: the same inputs and the same code path, so any
        # difference at all means stale cache state leaked across the reset.
        assert np.array_equal(first, second), (
            f"reset left stale state: max abs diff {np.abs(first - second).max():.4g}"
        )
    finally:
        runner.close()
        weights.free()


@_needs_hip
def test_the_runner_rejects_tokens_outside_the_vocabulary(artifact: GGUFReader) -> None:
    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=8)
    try:
        with pytest.raises(ValueError):
            runner.forward([])
        with pytest.raises(ValueError):
            runner.forward([int(weights.config.vocab_size)])
        with pytest.raises(ValueError):
            runner.forward([-1])
        # A refusal must not have advanced the position.
        assert runner.position == 0
    finally:
        runner.close()
        weights.free()


@_needs_hip
def test_the_runner_refuses_to_overrun_its_capacity(artifact: GGUFReader) -> None:
    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=4)
    try:
        runner.forward([1, 2, 3])
        with pytest.raises(ValueError):
            runner.forward([4, 5])
        assert runner.position == 3
    finally:
        runner.close()
        weights.free()


@_needs_hip
def test_staging_buffers_are_one_per_geometry_kind_not_per_layer(
    tmp_path: pathlib.Path,
) -> None:
    """One staging pair per distinct rope contract, one mask per window.

    The forward pass used to rebuild and re-upload the RoPE tables and keep
    mask inside the layer loop: for layers of the same kind those bytes are
    identical, so a 30-layer production decode step pushed 90 tiny H2D copies
    per step where two rope contracts and two sliding windows need at most six.
    The buffer *content* is byte-identical either way (parity is covered by the
    reference tests); what must hold here is the sharing itself, keyed by
    geometry value rather than layer index.
    """

    layer_types = (
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    )
    reader = GGUFReader(
        write_fixture_gguf(
            tmp_path / "gemma4_kinds.gguf",
            default_fixture_tensors(layer_types=layer_types),
            fixture_metadata(layer_types=layer_types),
        )
    )
    weights = load_gemma4_device_weights(reader)
    runner = Gemma4Runner(weights=weights, capacity=32)
    try:
        runner.forward([1, 5, 9, 13])
        runner.forward([4])  # rows == 1, the decode shape
        config = weights.config
        geometries = [config.geometry(i) for i in range(len(layer_types))]
        ropes = {g.rope for g in geometries}
        windows = {g.sliding_window for g in geometries}
        expected = 2 * len(ropes) + len(windows)
        assert len(runner._staging) == expected, (
            f"expected {expected} staging buffers "
            f"({len(ropes)} rope kinds x cos/sin + {len(windows)} mask kinds), "
            f"got {len(runner._staging)}: {sorted(runner._staging)}"
        )
    finally:
        runner.close()
        weights.free()
