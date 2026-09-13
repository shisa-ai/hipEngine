"""VibeVoice-ASR audio front-end CPU-reference parity vs the torch oracle.

Compares the torch-free NumPy reference (``hipengine/kernels/cpu_reference/
vibevoice_asr.py``) against fixtures produced by ``scripts/
vibevoice_asr_oracle_torch.py`` from the HF artifact, using encoder and
connector weights read from the original ``microsoft/VibeVoice-ASR``
checkpoint. Skips when the fixture or the local checkpoint is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_asr"
TRACE = FIXTURE_DIR / "vibevoice_asr_trace.npz"
BOUNDARY = FIXTURE_DIR / "vibevoice_asr_boundary.npz"
PINNED_MODEL_ID = "microsoft/VibeVoice-ASR"

if not TRACE.is_file():
    pytest.skip("VibeVoice-ASR trace fixture not present", allow_module_level=True)


def _cached_snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


@pytest.fixture(scope="module")
def snapshot() -> Path:
    path = _cached_snapshot()
    if path is None:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    return path


@pytest.fixture(scope="module")
def trace() -> dict[str, np.ndarray]:
    with np.load(TRACE) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def encoders(snapshot):
    from hipengine.loading.vibevoice_asr import load_vibevoice_encoder

    return {
        tok: load_vibevoice_encoder(str(snapshot), tok)
        for tok in ("acoustic", "semantic")
    }


def _assert_close(name: str, got: np.ndarray, ref: np.ndarray, atol: float = 2e-3) -> None:
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    scale = np.maximum(np.abs(ref.astype(np.float32)).max(), 1e-6)
    assert diff.max() <= max(atol, 4e-3 * scale), (
        f"{name}: max|diff| {diff.max():.3e} vs scale {scale:.3e}"
    )


def test_encoder_stage_trace(encoders, trace) -> None:
    """Per-stage outputs (stem, downsamples, head) match the torch oracle."""
    from hipengine.kernels.cpu_reference.vibevoice_asr import (
        vibevoice_causal_conv1d,
        _convnext_block,
        vibevoice_rmsnorm,
    )

    pcm = trace["pcm_short"].astype(np.float32)[None, None, :]
    for tok in ("acoustic", "semantic"):
        spec, w = encoders[tok]
        prefix = tok if tok == "semantic" else "acoustic"
        x = vibevoice_causal_conv1d(pcm, w.stem_conv_weight, w.stem_conv_bias)
        _assert_close(f"{prefix}_stem_conv", x, trace[f"{prefix}_stem_conv"])
        width = spec.num_filters
        block_iter = iter(w.blocks)
        for _ in range(spec.depths[0]):
            x = _convnext_block(x, next(block_iter), spec, width, dtype=None)
        _assert_close(f"{prefix}_stem_out", x, trace[f"{prefix}_stem_out"])
        for s, ratio in enumerate(spec.ratios):
            x = vibevoice_causal_conv1d(
                x, w.stage_conv_weights[s], w.stage_conv_biases[s], stride=ratio
            )
            _assert_close(f"{prefix}_stage{s}_conv", x, trace[f"{prefix}_stage{s}_conv"])
            width *= 2
            for _ in range(spec.depths[s + 1]):
                x = _convnext_block(x, next(block_iter), spec, width, dtype=None)
            _assert_close(f"{prefix}_stage{s}_out", x, trace[f"{prefix}_stage{s}_out"])
        x = vibevoice_causal_conv1d(x, w.head_conv_weight, w.head_conv_bias)
        _assert_close(f"{prefix}_head_out", x, trace[f"{prefix}_head_out"])
        _assert_close(
            f"{prefix}_latent_raw", x.transpose(0, 2, 1), trace[f"{prefix}_latent_raw"]
        )


def test_full_forward_matches_raw_latents(encoders, trace) -> None:
    """Single-call reference forward equals the traced raw latents."""
    from hipengine.kernels.cpu_reference.vibevoice_asr import (
        vibevoice_tokenizer_encoder_forward,
    )

    pcm = trace["pcm_short"].astype(np.float32)
    for tok in ("acoustic", "semantic"):
        spec, w = encoders[tok]
        lat = vibevoice_tokenizer_encoder_forward(spec, w, pcm)
        _assert_close(f"{tok}_latent", lat, trace[f"{tok}_latent_raw"], atol=4e-3)


def test_sampling_and_connector(encoders, snapshot, trace) -> None:
    """Recorded-noise sampling and both connector paths match the oracle."""
    from hipengine.kernels.cpu_reference.vibevoice_asr import (
        vibevoice_connector,
        vibevoice_sample_acoustic,
    )
    from hipengine.loading.vibevoice_asr import load_vibevoice_connector

    sampled = vibevoice_sample_acoustic(
        trace["acoustic_latent_raw"], trace["acoustic_noise_scale"], trace["acoustic_noise"]
    )
    _assert_close("acoustic_latent_sampled", sampled, trace["acoustic_latent_sampled"])

    ac = vibevoice_connector(load_vibevoice_connector(str(snapshot), "acoustic"), sampled)
    _assert_close("connector_acoustic_out", ac, trace["connector_acoustic_out"], atol=6e-3)
    se = vibevoice_connector(
        load_vibevoice_connector(str(snapshot), "semantic"), trace["semantic_latent_raw"]
    )
    _assert_close("connector_semantic_out", se, trace["connector_semantic_out"], atol=6e-3)
    _assert_close("connector_combined", ac + se, trace["connector_combined"], atol=8e-3)


def test_boundary_frame_counts(encoders) -> None:
    """3199/3200/3201-sample clips produce floor(L/3200) latent frames."""
    import numpy as np

    from hipengine.kernels.cpu_reference.vibevoice_asr import (
        vibevoice_tokenizer_encoder_forward,
    )

    if not BOUNDARY.is_file():
        pytest.skip("boundary fixture not present")
    with np.load(BOUNDARY) as data:
        bnd = {k: data[k] for k in data.files}
    spec_ac, w_ac = encoders["acoustic"]
    for n in (3199, 3200, 3201):
        pcm = bnd[f"pcm_{n}"].astype(np.float32)
        frames = spec_ac.frame_count(n)
        assert frames == n // 3200, (n, frames)
        lat = vibevoice_tokenizer_encoder_forward(spec_ac, w_ac, pcm)
        assert lat.shape == (1, frames, spec_ac.hidden_size)
        _assert_close(f"acoustic_latent_{n}", lat, bnd[f"acoustic_latent_raw_{n}"], atol=4e-3)
