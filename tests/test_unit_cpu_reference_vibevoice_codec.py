"""Unit tests for the VibeVoice codec CPU reference (decoder).

Covers the primitives against manual loops and the structural contract:
streaming and non-streaming decodes of whole-frame latents agree, and the
weight-shape contract matches the checkpoint's recorded decoder inventory.
The full-checkpoint oracle comparison lives in
``scripts/vibevoice_codec_cpu_reference_check.py`` (needs torch + checkpoint);
the tests here are checkpoint-free and use a small synthetic geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import vibevoice_codec as ref

SMALL_GEOMETRY = ref.DecoderGeometry(
    n_filters=4,
    ratios=(4, 2),
    depths=(2, 1, 1),
    kernel_size=3,
    last_kernel_size=3,
)


def synthetic_weights(geometry: ref.DecoderGeometry, seed: int = 7) -> dict[str, np.ndarray]:
    """Random FP32 weights matching ``expected_decoder_weight_shapes``."""

    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for name, shape in ref.expected_decoder_weight_shapes(geometry).items():
        weights[name] = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    return weights


def manual_dense_conv(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Triple-loop reference for one dense conv over pre-padded input."""

    out_c, in_c, kernel = weight.shape
    length = x.shape[1]
    out = np.zeros((out_c, length - kernel + 1), dtype=np.float64)
    for o in range(out_c):
        for j in range(out.shape[1]):
            acc = 0.0
            for i in range(in_c):
                for k in range(kernel):
                    acc += float(x[i, j + k]) * float(weight[o, i, k])
            out[o, j] = acc + float(bias[o])
    return out.astype(np.float32)


def manual_conv_transpose(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    """Triple-loop causal transposed conv with the right ``kernel - stride`` trim."""

    in_c, length = x.shape
    _, out_c, kernel = weight.shape
    raw = (length - 1) * stride + kernel
    out = np.zeros((out_c, raw), dtype=np.float64)
    for o in range(out_c):
        for j in range(raw):
            acc = 0.0
            for i in range(length):
                for k in range(kernel):
                    if i * stride + k == j:
                        for ic in range(in_c):
                            acc += float(x[ic, i]) * float(weight[ic, o, k])
            out[o, j] = acc + float(bias[o])
    trimmed = raw - (kernel - stride)
    return out[:, :trimmed].astype(np.float32)


class TestPrimitives:
    def test_dense_conv_matches_manual(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((5, 12)).astype(np.float32)
        weight = (rng.standard_normal((7, 5, 3)) * 0.1).astype(np.float32)
        bias = (rng.standard_normal(7) * 0.1).astype(np.float32)
        ours = ref.conv1d_valid(x, weight, bias, "test")
        assert ours.shape == (7, 10)
        np.testing.assert_allclose(ours, manual_dense_conv(x, weight, bias), atol=1e-5)

    def test_depthwise_conv_matches_manual(self):
        rng = np.random.default_rng(2)
        channels, length, kernel = 6, 10, 3
        x = rng.standard_normal((channels, length)).astype(np.float32)
        weight = (rng.standard_normal((channels, 1, kernel)) * 0.1).astype(np.float32)
        bias = (rng.standard_normal(channels) * 0.1).astype(np.float32)
        ours = ref.depthwise_conv1d_valid(x, weight, bias, "test")
        assert ours.shape == (channels, length - kernel + 1)
        expected = np.zeros_like(ours)
        for c in range(channels):
            for j in range(ours.shape[1]):
                expected[c, j] = sum(
                    float(x[c, j + k]) * float(weight[c, 0, k]) for k in range(kernel)
                ) + float(bias[c])
        np.testing.assert_allclose(ours, expected, atol=1e-6)

    def test_conv_transpose_matches_manual(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((4, 5)).astype(np.float32)
        weight = (rng.standard_normal((4, 6, 8)) * 0.1).astype(np.float32)  # [in, out, k]
        bias = (rng.standard_normal(6) * 0.1).astype(np.float32)
        stride = 4
        ours = ref.conv_transpose1d_causal(x, weight, bias, stride, "test")
        # Trimmed length is exactly length * stride.
        assert ours.shape == (6, 5 * stride)
        np.testing.assert_allclose(ours, manual_conv_transpose(x, weight, bias, stride), atol=1e-4)

    def test_rms_norm_matches_formula(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((8, 5)).astype(np.float32)
        weight = (rng.standard_normal(8) * 0.5 + 1.0).astype(np.float32)
        ours = ref.rms_norm_channels(x, weight, 1e-5, "test")
        expected = np.zeros_like(x)
        for j in range(x.shape[1]):
            ms = float(np.mean(x[:, j].astype(np.float64) ** 2))
            scale = 1.0 / math.sqrt(ms + 1e-5)
            expected[:, j] = x[:, j] * scale * weight
        np.testing.assert_allclose(ours, expected, rtol=1e-5, atol=1e-6)

    def test_gelu_matches_erf_formula(self):
        x = np.linspace(-4, 4, 33, dtype=np.float32)
        ours = ref.gelu_exact(x, "test")
        expected = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        np.testing.assert_allclose(ours, expected, rtol=1e-6, atol=1e-6)

    def test_linear_matches_manual(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((6, 4)).astype(np.float32)
        weight = (rng.standard_normal((9, 6)) * 0.1).astype(np.float32)
        bias = (rng.standard_normal(9) * 0.1).astype(np.float32)
        ours = ref.linear(x, weight, bias, "test")
        expected = (weight.astype(np.float64) @ x.astype(np.float64)) + bias[:, None].astype(
            np.float64
        )
        np.testing.assert_allclose(ours, expected.astype(np.float32), atol=1e-5)


class TestDecodeStructure:
    def test_streaming_frame_by_frame_matches_full(self):
        geometry = SMALL_GEOMETRY
        weights = synthetic_weights(geometry)
        rng = np.random.default_rng(11)
        frames = (rng.standard_normal((3, geometry.latent_dim)) * 0.3).astype(np.float32)

        full = ref.decode_full_sequence(frames, weights, geometry)
        assert full.shape == (3 * geometry.hop_length,)

        streams = ref.DecoderStreams()
        chunks = [
            ref.decode_latent_frames(frames[i : i + 1], weights, streams, geometry)
            for i in range(frames.shape[0])
        ]
        streamed = np.concatenate(chunks)
        assert streamed.shape == full.shape
        # Same windows, different BLAS shapes: FP32 accumulation-order noise.
        assert np.abs(full - streamed).max() < 1e-5

    def test_chunked_streaming_matches_per_frame(self):
        """A multi-frame chunk and per-frame chunks must agree."""

        geometry = SMALL_GEOMETRY
        weights = synthetic_weights(geometry, seed=13)
        rng = np.random.default_rng(17)
        frames = (rng.standard_normal((4, geometry.latent_dim)) * 0.3).astype(np.float32)

        per_frame_streams = ref.DecoderStreams()
        per_frame = np.concatenate(
            [
                ref.decode_latent_frames(frames[i : i + 1], weights, per_frame_streams, geometry)
                for i in range(frames.shape[0])
            ]
        )
        chunked_streams = ref.DecoderStreams()
        chunked = ref.decode_latent_frames(frames, weights, chunked_streams, geometry)
        assert np.abs(per_frame - chunked).max() < 1e-5

    def test_fresh_streams_restart_the_utterance(self):
        """A fresh DecoderStreams reproduces the first chunk exactly."""

        geometry = SMALL_GEOMETRY
        weights = synthetic_weights(geometry, seed=19)
        rng = np.random.default_rng(23)
        frames = (rng.standard_normal((2, geometry.latent_dim)) * 0.3).astype(np.float32)

        streams = ref.DecoderStreams()
        first = ref.decode_latent_frames(frames[:1], weights, streams, geometry)
        ref.decode_latent_frames(frames[1:], weights, streams, geometry)
        fresh = ref.decode_latent_frames(frames[:1], weights, ref.DecoderStreams(), geometry)
        np.testing.assert_array_equal(first, fresh)

    def test_output_length_is_frames_times_hop(self):
        geometry = SMALL_GEOMETRY
        weights = synthetic_weights(geometry, seed=29)
        rng = np.random.default_rng(31)
        frames = (rng.standard_normal((5, geometry.latent_dim)) * 0.3).astype(np.float32)
        out = ref.decode_full_sequence(frames, weights, geometry)
        assert out.shape == (5 * geometry.hop_length,)
        assert np.isfinite(out).all()

    def test_bf16_widening_is_bit_exact(self):
        payload = np.array([1.0, -2.0, 3.5], dtype=np.float32).view(np.uint32) >> 16
        raw = payload.astype("<u2").tobytes()
        out = ref.bf16_bytes_to_f32(raw)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, np.array([1.0, -2.0, 3.5], dtype=np.float32))


class TestWeightContract:
    def test_expected_shapes_match_recorded_checkpoint_inventory(self):
        """A few recorded decoder tensors pin the contract to the checkpoint."""

        geometry = ref.DecoderGeometry()
        shapes = ref.expected_decoder_weight_shapes(geometry)
        assert shapes["upsample_layers.0.0.conv.conv.weight"] == (2048, 64, 7)
        assert shapes["upsample_layers.0.0.conv.conv.bias"] == (2048,)
        assert shapes["upsample_layers.1.0.convtr.convtr.weight"] == (2048, 1024, 16)
        assert shapes["upsample_layers.6.0.convtr.convtr.weight"] == (64, 32, 4)
        assert shapes["head.conv.conv.weight"] == (1, 32, 7)
        assert shapes["stages.0.0.mixer.conv.conv.conv.weight"] == (2048, 1, 7)
        assert shapes["stages.0.0.ffn.linear1.weight"] == (8192, 2048)
        assert shapes["stages.5.2.ffn.linear1.weight"] == (256, 64)

    def test_expected_name_count_matches_checkpoint(self):
        """The decoder has 276 tensors in the checkpoint (recorded inventory)."""

        geometry = ref.DecoderGeometry()
        total = len(ref.expected_decoder_weight_shapes(geometry))
        assert total == 276


@pytest.mark.parametrize("bad_shape", [(3, 3), (2, 5, 3)])
def test_load_rejects_wrong_shapes(bad_shape):
    class FakeIndex:
        def require(self, names):
            return [object()] * len(names)

    def read_tensor(full_name):
        return np.zeros(int(np.prod(bad_shape)), dtype="<u2").tobytes()

    with pytest.raises(ValueError, match="shape"):
        ref.load_decoder_weights(read_tensor, SMALL_GEOMETRY)
