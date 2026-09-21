"""M5 runtime gate: YuE2 FP32 VAE decoder chain and bounded tiled decode.

The device decoder must reproduce the NumPy chain built from the same folded
weights. The chain is checked end to end rather than op by op because the
residual units are where wiring errors hide: the reference keeps each unit's
input as its residual, so an activation applied in place silently changes what is
added back and every individual kernel still looks correct.

Tiling is checked against the same runtime's own full decode: with enough halo
the crops must be bit-identical, which is what "exact interior crops, no
crossfade" means in practice.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import yue2 as reference
from hipengine.loading.yue2 import load_yue2_vae_decoder

CACHE = Path.home() / ".cache/huggingface/hub"


def _has_hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(not _has_hip(), reason="ROCm/HIP runtime is not available")


def _resolve(env: str, pattern: str) -> Path:
    override = os.environ.get(env)
    if override:
        return Path(override)
    for directory in sorted(CACHE.glob(pattern)):
        if (directory / "model.safetensors").is_file():
            return directory
    pytest.skip(f"checkpoint for {pattern} not present in the local cache")


@pytest.fixture(scope="module")
def decoder():
    return load_yue2_vae_decoder(_resolve("YUE2_VAE_DIR", "models--m-a-p--YuE2-Vae/snapshots/*"))


@pytest.fixture(scope="module")
def runtime(decoder):
    from hipengine.runtime.yue2_vae import Yue2VaeRuntime

    instance = Yue2VaeRuntime(decoder)
    yield instance
    instance.close()


def _numpy_decode(decoder, latent: np.ndarray) -> np.ndarray:
    """The reference chain, built from the same folded weights."""

    def conv(weights, x):
        if weights.transposed:
            return reference.conv_transpose1d(
                x, weights.weight, weights.bias, stride=weights.stride, padding=weights.padding
            )
        return reference.conv1d(
            x,
            weights.weight,
            weights.bias,
            stride=weights.stride,
            dilation=weights.dilation,
            padding=weights.padding,
        )

    x = conv(decoder.input_conv, latent)
    for block in decoder.blocks:
        x = reference.snake_beta(x, block.activation.alpha, block.activation.beta)
        x = conv(block.upsample, x)
        for unit in block.residual_units:
            h = reference.snake_beta(x, unit.activation_in.alpha, unit.activation_in.beta)
            h = conv(unit.conv, h)
            h = reference.snake_beta(h, unit.activation_out.alpha, unit.activation_out.beta)
            h = conv(unit.pointwise, h)
            x = x + h
    x = reference.snake_beta(x, decoder.output_activation.alpha, decoder.output_activation.beta)
    return conv(decoder.output_conv, x)


@requires_hip
def test_decode_matches_the_numpy_chain(runtime, decoder):
    rng = np.random.default_rng(11)
    latent = rng.standard_normal((1, 64, 1)).astype(np.float32)
    expected = _numpy_decode(decoder, latent[0])
    got = runtime.decode(latent)[0]
    assert got.shape == expected.shape
    assert got.shape[-1] == decoder.natural_output_length(1)
    delta = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    assert float(delta.max()) <= 1e-4 * max(1.0, float(np.abs(expected).max()))


@requires_hip
def test_decode_residual_units_add_the_original_input(runtime, decoder):
    """A unit's residual must be its input, not its own activation.

    Guards the exact defect this test was written after: applying the first
    SnakeBeta in place left the activation in the residual branch.
    """

    rng = np.random.default_rng(12)
    latent = rng.standard_normal((1, 64, 1)).astype(np.float32)
    got = runtime.decode(latent)[0]
    expected = _numpy_decode(decoder, latent[0])
    assert float(np.abs(got - expected).max()) <= 1e-4 * max(
        1.0, float(np.abs(expected).max())
    )


@requires_hip
def test_tiled_decode_matches_the_full_decode_bit_exactly(runtime):
    rng = np.random.default_rng(13)
    latent = rng.standard_normal((1, 64, 4)).astype(np.float32)
    full = runtime.decode(latent)
    tiled = runtime.decode_tiled(latent, core_frames=1, halo_frames=16)
    assert tiled.shape == full.shape
    assert np.array_equal(tiled, full)


@requires_hip
def test_tiled_decode_handles_a_partial_last_tile(runtime):
    rng = np.random.default_rng(14)
    latent = rng.standard_normal((1, 64, 5)).astype(np.float32)
    full = runtime.decode(latent)
    # 5 frames over a 3-frame core gives tiles 3 + 2, so the last crop is short.
    tiled = runtime.decode_tiled(latent, core_frames=3, halo_frames=16)
    assert np.array_equal(tiled, full)


@requires_hip
def test_natural_length_matches_the_reference_formula(runtime, decoder):
    for frames in (1, 2, 3, 64):
        assert runtime.natural_output_length(frames) == decoder.natural_output_length(frames)
    assert runtime.natural_output_length(3) == 5696
    assert runtime.natural_output_length(64) == 122816


@requires_hip
def test_tiled_decode_rejects_an_insufficient_halo(runtime):
    latent = np.zeros((1, 64, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="halo_frames must be at least"):
        runtime.decode_tiled(latent, core_frames=16, halo_frames=11)
    # The recorded dependency interval for a 16-frame core is 12.
    assert runtime.required_halo(16) == 12
    runtime.decode_tiled(latent, core_frames=16, halo_frames=12)


@requires_hip
def test_decode_rejects_bad_latents(runtime):
    with pytest.raises(ValueError, match="latent must be"):
        runtime.decode(np.zeros((1, 32, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="latent must have at least one frame"):
        runtime.decode(np.zeros((1, 64, 0), dtype=np.float32))
    bad = np.zeros((1, 64, 2), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        runtime.decode(bad)
