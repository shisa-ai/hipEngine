"""GPU parity test for the VibeVoice-TTS torch-free HIP decoder runtime.

Requires the cached ``microsoft/VibeVoice-1.5B`` snapshot, a working ROCm
stack, and the frozen schema-2 oracle fixtures. Skipped otherwise.

Gates (mirroring tests/test_unit_vibevoice_tts_decoder.py):
- per-frame chunk parity vs the CPU reference within the measured eager-bf16
  envelope, plus pooled waveform RMS relative error < 2%;
- decode_bulk must stay bit-identical to per-frame streaming (the causal
  prefix rolls are pure row-independent arithmetic);
- reset() at the generated speech boundaries must change the output exactly
  as the CPU stream's reset does.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_tts"
MANIFEST = FIXTURE_DIR / "manifest.json"
PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"

if not MANIFEST.is_file():
    pytest.skip("VibeVoice-TTS trace fixtures not present", allow_module_level=True)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if not _hip_available():
    pytest.skip("ROCm/HIP runtime not available", allow_module_level=True)


def _snapshot():
    from hipengine.loading.hf_cache import resolve_model_path

    return resolve_model_path(PINNED_MODEL_ID)


def _speech_resets(lm, request):
    ids = lm["generated_ids"].reshape(-1)[request["prompt_tokens"]:]
    resets = set()
    call = -1
    pending = False
    for token in ids:
        if token == 151653:
            pending = True
        elif token == 151654:
            call += 1
            if pending:
                resets.add(call)
                pending = False
    return resets


def _load_latents(name):
    d = np.load(FIXTURE_DIR / f"{name}_diffusion.npz")
    n = int(d["num_calls_recorded"])
    return np.stack([d[f"call{i}_scaled_latent"].reshape(-1) for i in range(n)]), n


@pytest.fixture(scope="module")
def bundle():
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_decoder

    try:
        return load_vibevoice_tts_decoder(_snapshot())
    except (FileNotFoundError, ValueError):
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache", allow_module_level=True)


@pytest.fixture(scope="module")
def gpu_runner(bundle):
    from hipengine.runtime.vibevoice_tts_decoder import VibevoiceTTSDecoderGPU

    spec, weights, _, _ = bundle
    runner = VibevoiceTTSDecoderGPU(spec, weights)
    yield runner
    runner.close()


@pytest.mark.parametrize("name", ["single", "two"])
def test_gpu_decoder_replay_parity(bundle, gpu_runner, name):
    import hipengine.kernels.cpu_reference.vibevoice_tts as tts_ref

    spec, weights, _, _ = bundle
    manifest = json.loads(MANIFEST.read_text())
    request = next(r for r in manifest["requests"] if r["name"] == name)
    data = np.load(FIXTURE_DIR / f"{name}_audio.npz")
    lm = np.load(FIXTURE_DIR / f"{name}_lm.npz")
    lat, n = _load_latents(name)
    resets = _speech_resets(lm, request)

    cpu_out = tts_ref.decode_frames(
        spec, weights, lat, reset_before=tuple(sorted(resets)), dtype="bfloat16"
    )

    gpu_runner.reset()  # the module-scoped runner may carry state from other tests
    gpu_out = np.empty_like(cpu_out)
    for i in range(n):
        if i in resets:
            gpu_runner.reset()
        gpu_out[i] = gpu_runner.decode(lat[i])

    assert gpu_out.shape == cpu_out.shape, f"{name}: {gpu_out.shape} != {cpu_out.shape}"

    # Calibrated eager-bf16 envelope (see the CPU test for the derivation):
    # the oracle's own fp32-accumulation spread compounds through 26 residual
    # blocks; quiet frames are gated on absolute difference.
    per_frame_peak = np.abs(cpu_out.astype(np.float64)).max(axis=1)
    per_frame_diff = np.abs(gpu_out.astype(np.float64) - cpu_out.astype(np.float64)).max(axis=1)
    envelope = np.maximum(2e-3, 0.06 * per_frame_peak)
    worst = (per_frame_diff / np.maximum(per_frame_peak, 1e-9)).max()
    assert bool((per_frame_diff <= envelope).all()), (
        f"{name}: GPU frame diff exceeds envelope; worst {worst:.3f} of frame peak"
    )

    pcm_gpu = gpu_out.reshape(-1)
    pcm_cpu = cpu_out.reshape(-1)
    rms_rel = np.sqrt(((pcm_gpu - pcm_cpu) ** 2).mean()) / np.sqrt(
        (pcm_cpu.astype(np.float64) ** 2).mean()
    )
    assert rms_rel < 0.02, f"{name}: pooled waveform RMS relative error {rms_rel:.4f}"


@pytest.mark.parametrize("name", ["single", "two"])
def test_gpu_bulk_matches_per_frame_streaming(bundle, gpu_runner, name):
    import hipengine.kernels.cpu_reference.vibevoice_tts as tts_ref

    spec, weights, _, _ = bundle
    manifest = json.loads(MANIFEST.read_text())
    request = next(r for r in manifest["requests"] if r["name"] == name)
    lm = np.load(FIXTURE_DIR / f"{name}_lm.npz")
    lat, n = _load_latents(name)
    resets = _speech_resets(lm, request)

    # Split into reset-delimited segments; decode each segment in bulk and by
    # per-frame streaming. A segment shorter than the FFN dispatch threshold stays
    # bit-identical, because every stage runs the same kernel at the same row count.
    # A longer segment runs the batched GEMM at its stage-0 rows where streaming
    # runs the per-output GEMV, so the outputs reassociate; there the bound is the
    # same eager-bf16 envelope the oracle parity test uses, and a prefix-roll bug
    # would move the waveform far past a reassociation.
    from hipengine.runtime import vibevoice_tts_decoder as dec

    bounds = [0, *(r + 1 for r in sorted(resets)), n]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo >= hi:
            continue
        gpu_runner.reset()
        bulk = gpu_runner.decode_bulk(lat[lo:hi])
        gpu_runner.reset()
        stream = np.stack([gpu_runner.decode(lat[i]) for i in range(lo, hi)])
        if hi - lo < dec._FFN_GEMM_MIN_ROWS:
            assert np.array_equal(bulk.view(np.uint32), stream.view(np.uint32)), (
                f"{name}: bulk decode diverges from per-frame streaming in [{lo}, {hi})"
            )
            continue
        peak = np.abs(stream.astype(np.float64)).max(axis=1)
        diff = np.abs(bulk.astype(np.float64) - stream.astype(np.float64)).max(axis=1)
        assert bool((diff <= np.maximum(2e-3, 0.06 * peak)).all()), (
            f"{name}: bulk vs streaming exceeds the oracle envelope in [{lo}, {hi}): "
            f"worst {diff.max():.3e} of frame peak {peak.max():.3e}"
        )
        rms_rel = np.sqrt(((bulk - stream) ** 2).mean()) / np.sqrt((stream.astype(np.float64) ** 2).mean())
        assert rms_rel < 0.01, (
            f"{name}: bulk-vs-streaming RMS relative error {rms_rel:.3e} in [{lo}, {hi}); "
            f"the reassociation measured 5.8e-3 (single) and 6.7e-3 (two) on 2026-09-15"
        )


def test_gpu_reset_changes_output(bundle, gpu_runner):
    spec, weights, _, _ = bundle
    lat, _ = _load_latents("single")
    gpu_runner.reset()
    fresh = gpu_runner.decode(lat[8])
    gpu_runner.decode(lat[0])
    carried = gpu_runner.decode(lat[8])
    assert not np.array_equal(fresh.view(np.uint32), carried.view(np.uint32)), (
        "reset() did not change the decode output; caches are not rolling"
    )


def test_gpu_decoder_resolves_through_registry(bundle):
    """The convtr primitive must come from the four-axis registry, not a hard path."""
    from hipengine.kernels.vibevoice import resolve_vibevoice_kernels

    ops = resolve_vibevoice_kernels()
    assert callable(ops.vv_convtr_gemm_bf16)
    assert callable(ops.vv_conv_gemm_bf16)


def test_gpu_ffn_dispatch_boundary():
    """`_ffn_linear` picks the batched GEMM above the row threshold, GEMV below."""
    from hipengine.runtime import vibevoice_tts_decoder as dec
    from hipengine.kernels.hip_gfx1100.linear import dense_gemv

    calls = []
    real_gemv = dense_gemv.dense_gemv_out_bf16
    real_gemm = dense_gemv.dense_prefill_gemm_out_bf16
    dense_gemv.dense_gemv_out_bf16 = lambda *a, **k: calls.append("gemv")
    dense_gemv.dense_prefill_gemm_out_bf16 = lambda *a, **k: calls.append("gemm")
    try:
        dec._ffn_linear(0, 0, 0, dec._FFN_GEMM_MIN_ROWS - 1, 32, 128, runtime=None)
        dec._ffn_linear(0, 0, 0, dec._FFN_GEMM_MIN_ROWS, 32, 128, runtime=None)
    finally:
        dense_gemv.dense_gemv_out_bf16 = real_gemv
        dense_gemv.dense_prefill_gemm_out_bf16 = real_gemm

    assert calls == ["gemv", "gemm"], calls


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [(3200, 32, 128), (1600, 64, 256), (800, 128, 512), (200, 256, 1024), (8, 1024, 4096)],
)
def test_gpu_ffn_gemm_agrees_with_gemv_within_one_ulp(rows, in_features, out_features):
    """The reassociation is bounded: at most one bf16 ULP on the FFN shapes.

    `_ffn_linear` swaps the per-output-element GEMV for the batched GEMM on the
    decoder's ConvNeXt FFN linears. The GEMM sums in a different order, so the
    outputs are not bit-identical; this pins how far they are allowed to move.
    Measured on these shapes it is 6-27 elements of 409,600 at one ULP, and the
    fixture chain stays exact (see test_live_vibevoice_tts_session_gpu.py).
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear import dense_gemv

    rt = get_hip_runtime()
    rng = np.random.default_rng(0)

    def to_bf16(a):
        a = np.ascontiguousarray(a.astype(np.float32))
        u = a.view(np.uint32)
        return ((u + 0x8000 + ((u >> 16) & 1)) >> 16).astype(np.uint16)

    x = to_bf16(rng.standard_normal((rows, in_features))).reshape(-1)
    w = to_bf16(rng.standard_normal((out_features, in_features)) * 0.05).reshape(-1)

    def run(fn):
        xb, wb, ob = malloc(x.nbytes), malloc(w.nbytes), malloc(rows * out_features * 2)
        copy_host_array_to_device(xb, x)
        copy_host_array_to_device(wb, w)
        fn(xb.ptr, wb.ptr, ob.ptr, rows, in_features, out_features, runtime=rt)
        out = np.empty(rows * out_features, np.uint16)
        copy_device_to_host(out.ctypes.data, ob, out.nbytes, runtime=rt)
        return out

    gemv = run(dense_gemv.dense_gemv_out_bf16)
    gemm = run(dense_gemv.dense_prefill_gemm_out_bf16)

    a = (gemv.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = (gemm.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    # Both operands of a pair share a sign here, so the raw pattern distance is the
    # bf16 ULP distance. A sign-crossing pair would show up as ~0x8000.
    ulp = np.abs(gemv.astype(np.int32) - gemm.astype(np.int32))
    differing = int(np.count_nonzero(ulp))
    assert int(ulp.max()) <= 4, f"max bf16 ULP distance {int(ulp.max())} exceeds 4"
    assert differing / gemv.size < 1e-3, (
        f"{differing} of {gemv.size} elements differ, above the 1e-3 bound"
    )
    assert np.abs(a - b).max() <= 0.02, f"max abs diff {np.abs(a - b).max():.3e}"
