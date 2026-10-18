"""VibeVoice-ASR front-end HIP kernels and runtime parity vs CPU reference.

Per-kernel gates (plain-weight RMSNorm, erf-GELU, scale-residual, depthwise
causal conv, strided causal conv-as-gemm, bias add, recorded-noise sampling)
against ``hipengine/kernels/cpu_reference/vibevoice_asr.py``, then the full
two-encoder + connector runtime against the CPU reference forward on real
checkpoint weights. Skips without a usable HIP runtime or without the local
``microsoft/VibeVoice-ASR`` snapshot.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

if not hip_runtime_available():
    pytest.skip("no usable HIP runtime for VibeVoice front-end tests", allow_module_level=True)

from hipengine.core.memory import copy_device_to_host, free, host_array_ptr
from hipengine.kernels.cpu_reference.vibevoice_asr import (
    _gelu_erf,
    vibevoice_causal_conv1d,
    vibevoice_rmsnorm,
    vibevoice_tokenizer_encoder_forward,
    vibevoice_connector,
)
from hipengine.kernels.hip_gfx1100.vibevoice.encoder import (
    build_vibevoice_encoder,
    conv_rows_out,
    f32_to_bf16_bits,
    transpose_conv_weight_t,
    vv_add_bias_bf16,
    vv_add_scaled_noise_bf16,
    vv_conv_gemm_bf16,
    vv_depthwise_conv_bf16,
    vv_gelu_bf16,
    vv_rmsnorm_bf16,
    vv_scale_residual_bf16,
)
from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime, _upload_u16

PINNED_MODEL_ID = "microsoft/VibeVoice-ASR"


def _snapshot_or_skip():
    from hipengine.loading.hf_cache import resolve_model_path

    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    return path


@pytest.fixture(scope="module")
def library():
    lib = build_vibevoice_encoder()
    assert lib is not None
    return lib


def _d2h(buf, count):
    host = np.empty(count, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(host), buf)
    return (host.astype(np.uint32) << 16).view(np.float32)


def _assert_rel(name, got, ref, tol):
    diff = np.abs(got - ref).max()
    scale = max(np.abs(ref).max(), 1e-9)
    assert diff / scale <= tol, f"{name}: rel {diff / scale:.3e} > {tol}"


def test_rmsnorm_plain_weight(library) -> None:
    rng = np.random.default_rng(0)
    rows, hidden = 13, 320
    x = rng.standard_normal((rows, hidden)).astype(np.float32)
    w = rng.standard_normal(hidden).astype(np.float32)
    gx = _upload_u16(f32_to_bf16_bits(x))
    gw = _upload_u16(f32_to_bf16_bits(w))
    out = _upload_u16(np.zeros(rows * hidden, dtype=np.uint16))
    vv_rmsnorm_bf16(gx.ptr, gw.ptr, out.ptr, rows, hidden, 1e-5)
    _assert_rel("rmsnorm", _d2h(out, rows * hidden).reshape(rows, hidden), vibevoice_rmsnorm(x, w, 1e-5), 2e-2)
    free(gx); free(gw); free(out)


def test_gelu_and_elementwise(library) -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal(1000).astype(np.float32)
    gz = _upload_u16(f32_to_bf16_bits(z))
    out = _upload_u16(np.zeros(1000, dtype=np.uint16))
    vv_gelu_bf16(gz.ptr, out.ptr, 1000)
    _assert_rel("gelu", _d2h(out, 1000), _gelu_erf(z), 2e-2)
    free(out)

    x = rng.standard_normal((7, 64)).astype(np.float32)
    y = rng.standard_normal((7, 64)).astype(np.float32)
    gamma = rng.standard_normal(64).astype(np.float32)
    gx = _upload_u16(f32_to_bf16_bits(x))
    gy = _upload_u16(f32_to_bf16_bits(y))
    gg = _upload_u16(f32_to_bf16_bits(gamma))
    out = _upload_u16(np.zeros(7 * 64, dtype=np.uint16))
    vv_scale_residual_bf16(gx.ptr, gy.ptr, gg.ptr, out.ptr, 7 * 64, 64)
    _assert_rel("scale_residual", _d2h(out, 7 * 64).reshape(7, 64), x + y * gamma.reshape(1, 64), 2e-2)
    free(out); free(gx); free(gy); free(gg)

    b = rng.standard_normal(64).astype(np.float32)
    gb = _upload_u16(f32_to_bf16_bits(b))
    out = _upload_u16(np.zeros(7 * 64, dtype=np.uint16))
    vv_add_bias_bf16(gx.ptr if False else _upload_u16(f32_to_bf16_bits(x)).ptr, gb.ptr, out.ptr, 7 * 64, 64)
    _assert_rel("add_bias", _d2h(out, 7 * 64).reshape(7, 64), x + b.reshape(1, 64), 2e-2)


def test_depthwise_conv(library) -> None:
    rng = np.random.default_rng(2)
    rows, c, k = 40, 64, 7
    normed = rng.standard_normal((rows, c)).astype(np.float32)
    resid = rng.standard_normal((rows, c)).astype(np.float32)
    w = rng.standard_normal((c, 1, k)).astype(np.float32)
    b = rng.standard_normal(c).astype(np.float32)
    gamma = rng.standard_normal(c).astype(np.float32)
    gp = _upload_u16(np.zeros((k - 1) * c, dtype=np.uint16))
    gn = _upload_u16(f32_to_bf16_bits(normed))
    gr = _upload_u16(f32_to_bf16_bits(resid))
    gw = _upload_u16(f32_to_bf16_bits(w.reshape(c, k)))
    gbw = _upload_u16(f32_to_bf16_bits(b))
    gg = _upload_u16(f32_to_bf16_bits(gamma))
    out = _upload_u16(np.zeros(rows * c, dtype=np.uint16))
    vv_depthwise_conv_bf16(gp.ptr, gn.ptr, gr.ptr, gw.ptr, gbw.ptr, gg.ptr, out.ptr, k - 1, rows, c, k)
    conv = vibevoice_causal_conv1d(normed.T[None], w, b, groups=c)
    ref = (resid.T[None] + conv * gamma.reshape(1, c, 1)).transpose(0, 2, 1)[0]
    _assert_rel("depthwise", _d2h(out, rows * c).reshape(rows, c), ref, 2e-2)


def test_conv_gemm_strided(library) -> None:
    rng = np.random.default_rng(3)
    rows, c_in, k, stride, c_out = 40, 64, 4, 2, 128
    x = rng.standard_normal((rows, c_in)).astype(np.float32)
    w = rng.standard_normal((c_out, c_in, k)).astype(np.float32)
    b = rng.standard_normal(c_out).astype(np.float32)
    gx = _upload_u16(f32_to_bf16_bits(x))
    gw = _upload_u16(transpose_conv_weight_t(w))
    gb = _upload_u16(f32_to_bf16_bits(b))
    prefix_rows = k - stride
    gp = _upload_u16(np.zeros(prefix_rows * c_in, dtype=np.uint16))
    rows_out = conv_rows_out(prefix_rows, rows, k, stride)
    out = _upload_u16(np.zeros(rows_out * c_out, dtype=np.uint16))
    vv_conv_gemm_bf16(gp.ptr, gx.ptr, gw.ptr, gb.ptr, out.ptr, prefix_rows, rows_out, c_in, c_out, k, stride)
    ref = vibevoice_causal_conv1d(x.T[None], w, b, stride=stride).transpose(0, 2, 1)[0]
    _assert_rel("conv_gemm", _d2h(out, rows_out * c_out).reshape(rows_out, c_out), ref, 2e-2)


def test_add_scaled_noise(library) -> None:
    rng = np.random.default_rng(4)
    frames, hidden = 5, 64
    lat = rng.standard_normal((1, frames, hidden)).astype(np.float32)
    scale = rng.standard_normal(1).astype(np.float32)
    noise = rng.standard_normal((1, frames, hidden)).astype(np.float32)
    gl = _upload_u16(f32_to_bf16_bits(lat))
    from hipengine.core.memory import malloc, copy_host_array_to_device

    gs = malloc(4)
    copy_host_array_to_device(gs, scale)
    gn = _upload_u16(f32_to_bf16_bits(noise))
    out = _upload_u16(np.zeros(frames * hidden, dtype=np.uint16))
    vv_add_scaled_noise_bf16(gl.ptr, gs.ptr, gn.ptr, out.ptr, frames * hidden, frames * hidden)
    _assert_rel("add_noise", _d2h(out, frames * hidden).reshape(frames, hidden), (lat + scale.reshape(1, 1, 1) * noise)[0], 2e-2)


@pytest.fixture(scope="module")
def frontend():
    _snapshot_or_skip()
    from hipengine.loading.vibevoice_asr import load_vibevoice_connector, load_vibevoice_encoder

    specs, conns = {}, {}
    for tok in ("acoustic", "semantic"):
        specs[tok] = load_vibevoice_encoder(PINNED_MODEL_ID, tok)
        conns[tok] = load_vibevoice_connector(PINNED_MODEL_ID, tok)
    runtime = VibevoiceFrontendRuntime(
        specs["acoustic"][0], specs["acoustic"][1],
        specs["semantic"][0], specs["semantic"][1],
        conns["acoustic"], conns["semantic"],
    )
    yield runtime, specs, conns
    runtime.close()


def test_frontend_e2e_matches_cpu_reference(frontend) -> None:
    runtime, specs, conns = frontend
    rng = np.random.default_rng(5)
    pcm = np.pad((0.1 * rng.standard_normal(12000)).astype(np.float32), (0, 800))
    got = runtime.forward(pcm)
    lat_ac = vibevoice_tokenizer_encoder_forward(specs["acoustic"][0], specs["acoustic"][1], pcm)
    lat_se = vibevoice_tokenizer_encoder_forward(specs["semantic"][0], specs["semantic"][1], pcm)
    ref = vibevoice_connector(conns["acoustic"], lat_ac) + vibevoice_connector(conns["semantic"], lat_se)
    _assert_rel("frontend_e2e", got, ref, 5e-2)


GPU_FIXTURE = Path(__file__).parent / "fixtures" / "vibevoice_asr_gpu" / "vibevoice_asr_trace.npz"


def test_frontend_matches_torch_gpu_bf16(frontend) -> None:
    """HIP front-end vs torch-GPU bf16 oracle on the same PCM and noise."""
    if not GPU_FIXTURE.is_file():
        pytest.skip("torch-GPU bf16 fixture not present")
    runtime, specs, conns = frontend
    with np.load(GPU_FIXTURE) as data:
        fx = {k: data[k] for k in data.files}
    pcm = fx["pcm_short"].astype(np.float32)
    # This legacy fixture captures bare encoders without processor padding,
    # so it contains only complete frames. Its unused causal tail contributes
    # no outputs. Final partial-frame behavior has its own boundary tests.
    pcm = pcm[:(len(pcm) // 3200) * 3200]
    hidden_ac = specs["acoustic"][0].hidden_size
    noise = fx["acoustic_noise"].astype(np.float32).reshape(-1, hidden_ac)
    scale = fx["acoustic_noise_scale"].astype(np.float32).reshape(1)
    got = runtime.forward(pcm, noise=noise, noise_scale=scale[0])
    ref = fx["connector_combined"].astype(np.float32)
    _assert_rel("frontend_vs_torch_gpu_bf16", got, ref, 5e-2)
