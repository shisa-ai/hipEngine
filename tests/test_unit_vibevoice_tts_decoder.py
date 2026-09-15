"""VibeVoice-TTS acoustic decoder CPU-reference parity vs the frozen oracle.

Replays the schema-2 fixture chain end to end with the torch-free reference
(``hipengine/kernels/cpu_reference/vibevoice_tts.py``): every recorded
``scaled_latent`` in call order, cache resets at the speech-boundary positions
derived from ``generated_ids``, compared against the recorded decoder chunks and
the concatenated waveform. Skips when the fixture or the local checkpoint is
absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.vibevoice_tts import (
    SPEECH_DIFFUSION_ID,
    SPEECH_END_ID,
    VibevoiceDecoderStream,
    decode_frames,
)
from hipengine.loading.hf_cache import resolve_model_path
from hipengine.loading.vibevoice_tts import load_vibevoice_tts_decoder

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_tts"
PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"
REQUESTS = ("single", "two")

if not (FIXTURE_DIR / "manifest.json").is_file():
    pytest.skip("VibeVoice-TTS fixtures not present", allow_module_level=True)


def _cached_snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


@pytest.fixture(scope="module")
def decoder_bundle():
    snapshot = _cached_snapshot()
    if snapshot is None:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    return load_vibevoice_tts_decoder(snapshot)


def _request_fixture(name: str) -> dict[str, np.ndarray]:
    diffusion = np.load(FIXTURE_DIR / f"{name}_diffusion.npz")
    audio = np.load(FIXTURE_DIR / f"{name}_audio.npz")
    lm = np.load(FIXTURE_DIR / f"{name}_lm.npz")
    manifest = (FIXTURE_DIR / "manifest.json").read_text()
    import json

    request = next(r for r in json.loads(manifest)["requests"] if r["name"] == name)
    n_calls = int(diffusion["num_calls_recorded"])
    latents = np.stack(
        [diffusion[f"call{i}_scaled_latent"].reshape(-1) for i in range(n_calls)], axis=0
    )
    # Cache resets: the loop zeros the streaming caches at each generated
    # speech-end token, so the next decode call after an end token starts from
    # zero context. Decode calls are the diffusion tokens after the prompt.
    ids = lm["generated_ids"].reshape(-1)[request["prompt_tokens"]:]
    is_decode = ids == SPEECH_DIFFUSION_ID
    is_end = ids == SPEECH_END_ID
    if int(is_decode.sum()) != n_calls:
        raise AssertionError(
            f"{name}: {int(is_decode.sum())} diffusion tokens but {n_calls} recorded calls"
        )
    reset_calls = []
    call_index = -1
    pending_reset = False
    for token_is_decode, token_is_end in zip(is_decode, is_end):
        if token_is_end:
            pending_reset = True
        elif token_is_decode:
            call_index += 1
            if pending_reset:
                reset_calls.append(call_index)
                pending_reset = False
    chunks = audio["chunks"].reshape(n_calls, -1)
    return {
        "latents": latents,
        "reset_calls": tuple(reset_calls),
        "chunks": chunks,
        "pcm": audio["pcm"].reshape(-1),
    }


@pytest.mark.parametrize("name", REQUESTS)
def test_decoder_replay_chunk_parity(decoder_bundle, name):
    spec, weights, scale, bias = decoder_bundle
    data = _request_fixture(name)
    if data["reset_calls"]:
        pass  # exercise both the helper and the explicit stream below
    replayed = decode_frames(
        spec, weights, data["latents"], reset_before=data["reset_calls"], dtype="bfloat16"
    )
    assert replayed.shape == data["chunks"].shape

    exact = np.array(
        [
            np.array_equal(a.view(np.uint32), b.view(np.uint32))
            for a, b in zip(replayed, data["chunks"])
        ]
    )
    # Calibrated eager-bf16 envelope. The oracle records eager bf16 GPU kernels
    # whose fp32 accumulation order inside each GEMM differs from exact math,
    # and the difference compounds through 26 residual blocks and the
    # cross-call streaming caches. Measured against exact (fp64-internal)
    # math, the oracle's own spread is ~1.2-1.4% pooled RMS / ~11% worst
    # frame-peak; this reference matches exact math, so parity vs the oracle
    # is bounded by the oracle's noise floor, not by this implementation.
    # Quiet frames are gated on absolute difference.
    per_frame_peak = np.abs(data["chunks"].astype(np.float64)).max(axis=1)
    per_frame_diff = np.abs(replayed.astype(np.float64) - data["chunks"].astype(np.float64)).max(axis=1)
    envelope = np.maximum(1e-4, 0.15 * per_frame_peak)
    assert bool((per_frame_diff <= envelope).all()), (
        f"{name}: frame diff exceeds envelope; worst "
        f"{(per_frame_diff / np.maximum(per_frame_peak, 1e-9)).max():.3f} of frame peak"
    )
    pcm = replayed.reshape(-1)
    ref_pcm = data["pcm"]
    rms_rel = np.sqrt(((pcm.astype(np.float64) - ref_pcm.astype(np.float64)) ** 2).mean()) / np.sqrt(
        (ref_pcm.astype(np.float64) ** 2).mean()
    )
    assert rms_rel < 0.02, f"{name}: pooled waveform RMS relative error {rms_rel:.4f}"
    if not exact.all():
        # Diagnostic, not a gate: bit-parity across accumulation orders is a
        # debugging oracle, not the promotion bar (docs/EXECUTION-PROFILES.md).
        print(f"{name}: {int(exact.sum())}/{len(exact)} frames bit-exact (diagnostic)")


def test_decoder_stream_matches_batch_helper(decoder_bundle):
    """The per-call stream and the batch helper produce identical chunks."""
    spec, weights, scale, bias = decoder_bundle
    data = _request_fixture("single")
    stream = VibevoiceDecoderStream(spec, weights, dtype="bfloat16")
    chunks = []
    for i, row in enumerate(data["latents"]):
        if i in data["reset_calls"]:
            stream.reset()
        chunks.append(stream.decode(row))
    helper = decode_frames(
        spec, weights, data["latents"], reset_before=data["reset_calls"], dtype="bfloat16"
    )
    assert np.array_equal(np.stack(chunks), helper)


def test_decoder_scale_bias_loaded(decoder_bundle):
    """The checkpoint scaling factors are finite and match the manifest."""
    import json

    _, _, scale, bias = decoder_bundle
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    expected = manifest["requests"][0]
    assert scale == expected["scaling_factor"]
    assert bias == expected["bias_factor"]


def test_decoder_reset_changes_output(decoder_bundle):
    """A cache reset at the first frame must change the second frame's output."""
    spec, weights, scale, bias = decoder_bundle
    data = _request_fixture("single")
    lat = data["latents"][:2]
    with_reset = decode_frames(spec, weights, lat, reset_before=(1,), dtype="bfloat16")
    without_reset = decode_frames(spec, weights, lat, reset_before=(), dtype="bfloat16")
    assert not np.array_equal(with_reset[1], without_reset[1])
