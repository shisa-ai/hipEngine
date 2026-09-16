"""GPU parity test for the VibeVoice-TTS diffusion head HIP runtime.

Requires the cached ``microsoft/VibeVoice-1.5B`` snapshot, a working ROCm
stack, and the frozen schema-2 oracle fixtures. Skipped otherwise.

Gates (mirroring the decoder lane):
- per-step head outputs and solver trajectory vs the numpy CPU reference over
  the pre-amplification steps, on both fixture requests;
- the whole 20-step trajectory plus the final latent and its scaled form vs
  the CPU reference and the frozen fixture chain, on the single-speaker
  request;
- replay determinism (two runs bit-identical);
- the head primitives resolve through the four-axis registry.

The two-speaker request carries no late-step threshold: its ``call0``
trajectory amplifies a one-bf16-ULP input change past this envelope from step 6
on, so agreement there measures the fixture's conditioning rather than the
kernel. ``tests/test_unit_vibevoice_tts_diffusion.py`` documents the same split.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_tts"
MANIFEST = FIXTURE_DIR / "manifest.json"
PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"

# Frozen acceptance thresholds, measured 2026-09-15 from this head against the
# numpy CPU reference on the committed fixtures (see worklog entry
# 20260915T101241.017890Z-lhl-vibevoice-tts-chain-bifurcation-48804e.md).
# Constants rather than a function of the run under test, so a kernel that gets
# less stable cannot earn itself a wider gate. Measured pre-onset worst cases:
# eps 0.0085 and speech 0.0144; whole-trajectory single-speaker eps 0.039,
# latent 0.006 vs CPU and 0.009 vs the fixture, scaled 0.009.
_PRE_AMPLIFICATION_STEPS = 6
_FROZEN_EPS_EARLY = 0.025
_FROZEN_SPEECH_EARLY = 0.025
_FROZEN_EPS_SINGLE = 0.08
_FROZEN_SPEECH_SINGLE = 0.05
_FROZEN_LATENT_VS_CPU_SINGLE = 0.05
_FROZEN_LATENT_VS_FIXTURE_SINGLE = 0.06
_FROZEN_SCALED_VS_FIXTURE_SINGLE = 0.08

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


@pytest.fixture(scope="module")
def bundle():
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_diffusion_head

    try:
        return load_vibevoice_tts_diffusion_head(resolve_model_path(PINNED_MODEL_ID))
    except (FileNotFoundError, ValueError):
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache", allow_module_level=True)


@pytest.fixture(scope="module")
def gpu_head(bundle):
    from hipengine.runtime.vibevoice_tts_diffusion import VibevoiceTTSDiffusionHeadGPU

    spec, weights, _, _ = bundle
    head = VibevoiceTTSDiffusionHeadGPU(spec, weights)
    yield head
    head.close()


def _replay(head, data, collect):
    return head.sample_speech_tokens(
        data["call0_condition"],
        data["call0_neg_condition"],
        float(data["call0_cfg_scale"]),
        data["call0_initial_noise"],
        collect=collect,
    )


@pytest.mark.parametrize("name", ["single", "two"])
def test_gpu_diffusion_replay_parity(bundle, gpu_head, name):
    import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref

    spec, weights, scale, bias = bundle
    data = np.load(FIXTURE_DIR / f"{name}_diffusion.npz")

    steps: list[dict[str, np.ndarray]] = []
    final_g, _ = _replay(gpu_head, data, steps)
    eps_g = np.stack([s["eps"] for s in steps])
    speech_g = np.stack([s["speech"] for s in steps])

    csteps: list[dict[str, np.ndarray]] = []
    final_c, _ = diff_ref.sample_speech_tokens(
        spec,
        weights,
        data["call0_condition"],
        data["call0_neg_condition"],
        float(data["call0_cfg_scale"]),
        data["call0_initial_noise"],
        collect=csteps,
    )
    eps_c = np.stack([s["eps"] for s in csteps])
    speech_c = np.stack([s["speech"] for s in csteps])

    eps_ratio = (
        np.abs(eps_g - eps_c).max(axis=(1, 2)) / np.maximum(np.abs(eps_c).max(axis=(1, 2)), 1e-9)
    )
    speech_ratio = (
        np.abs(speech_g - speech_c).max(axis=(1, 2))
        / np.maximum(np.abs(speech_c).max(axis=(1, 2)), 1e-9)
    )
    assert eps_ratio[:_PRE_AMPLIFICATION_STEPS].max() < _FROZEN_EPS_EARLY, (
        f"{name}: GPU eps step {int(eps_ratio[:_PRE_AMPLIFICATION_STEPS].argmax())} off CPU by "
        f"{eps_ratio[:_PRE_AMPLIFICATION_STEPS].max():.4f} of peak against {_FROZEN_EPS_EARLY}"
    )
    assert speech_ratio[:_PRE_AMPLIFICATION_STEPS].max() < _FROZEN_SPEECH_EARLY, (
        f"{name}: GPU speech step {int(speech_ratio[:_PRE_AMPLIFICATION_STEPS].argmax())} off CPU "
        f"by {speech_ratio[:_PRE_AMPLIFICATION_STEPS].max():.4f} against {_FROZEN_SPEECH_EARLY}"
    )

    if name != "single":
        # Past the amplification onset this replay is on a different trajectory
        # than the CPU reference (0.32 of peak on eps at step 19), so no
        # threshold on that comparison separates a kernel defect from the
        # fixture's conditioning. These bounds only catch a collapsed head.
        assert np.abs(eps_g).max() <= 4 * np.abs(eps_c).max(), f"{name}: eps magnitude blew up"
        assert np.abs(final_g).max() <= 4 * max(np.abs(final_c).max(), 1e-9), (
            f"{name}: latent magnitude blew up"
        )
        return

    assert eps_ratio.max() < _FROZEN_EPS_SINGLE, (
        f"{name}: GPU eps step {int(eps_ratio.argmax())} off CPU by {eps_ratio.max():.4f} "
        f"of peak against {_FROZEN_EPS_SINGLE}"
    )
    assert speech_ratio.max() < _FROZEN_SPEECH_SINGLE, (
        f"{name}: GPU speech step {int(speech_ratio.argmax())} off CPU by "
        f"{speech_ratio.max():.4f} against {_FROZEN_SPEECH_SINGLE}"
    )

    latent_peak = max(np.abs(final_c).max(), 1e-9)
    assert np.abs(final_g - final_c).max() / latent_peak < _FROZEN_LATENT_VS_CPU_SINGLE, (
        f"{name}: GPU final latent off CPU by "
        f"{np.abs(final_g - final_c).max() / latent_peak:.4f} relative"
    )

    # Fixture chain: the frozen oracle latent and its scaled decoder input.
    fx_rel = np.abs(final_g - data["call0_speech_latent"]).max() / max(
        np.abs(data["call0_speech_latent"]).max(), 1e-9
    )
    assert fx_rel < _FROZEN_LATENT_VS_FIXTURE_SINGLE, (
        f"{name}: GPU final latent off fixture by {fx_rel:.4f} relative"
    )
    scaled_g = diff_ref.scale_speech_latent(final_g, scale, bias)
    fx_scaled = data["call0_scaled_latent"].reshape(-1)
    scaled_rel = np.abs(scaled_g - fx_scaled).max() / max(np.abs(fx_scaled).max(), 1e-9)
    assert scaled_rel < _FROZEN_SCALED_VS_FIXTURE_SINGLE, (
        f"{name}: GPU scaled latent off fixture by {scaled_rel:.4f} relative"
    )


def test_gpu_diffusion_replay_is_deterministic(bundle, gpu_head):
    data = np.load(FIXTURE_DIR / "single_diffusion.npz")
    a: list[dict[str, np.ndarray]] = []
    final_a, _ = _replay(gpu_head, data, a)
    b: list[dict[str, np.ndarray]] = []
    final_b, _ = _replay(gpu_head, data, b)
    assert np.array_equal(final_a.view(np.uint32), final_b.view(np.uint32))
    for sa, sb in zip(a, b):
        assert np.array_equal(sa["eps"].view(np.uint32), sb["eps"].view(np.uint32))
        assert np.array_equal(sa["speech"].view(np.uint32), sb["speech"].view(np.uint32))


def test_gpu_head_step0_matches_cpu_reference(bundle, gpu_head):
    """Isolated head call: duplicated first branch at the first timestep."""
    import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref
    from hipengine.runtime.vibevoice_tts_diffusion import _bf16_u16

    spec, weights, _, _ = bundle
    data = np.load(FIXTURE_DIR / "single_diffusion.npz")
    cond = np.concatenate([data["call0_condition"], data["call0_neg_condition"]], axis=0)
    noise = data["call0_initial_noise"]
    combined = np.concatenate([noise[:1], noise[:1]], axis=0)
    t0 = int(data["call0_scheduler_timesteps"][0])

    gpu_eps = gpu_head.forward(_bf16_u16(combined), t0, _bf16_u16(cond))
    cpu_eps = diff_ref.diffusion_head_forward(spec, weights, combined, np.full(2, t0), cond)
    peak = max(np.abs(cpu_eps).max(), 1e-9)
    assert np.abs(gpu_eps - cpu_eps).max() / peak < 0.02, (
        f"isolated head call off CPU by {np.abs(gpu_eps - cpu_eps).max() / peak:.4f} of peak"
    )


def test_gpu_dpm_step_kernel_matches_cpu_solver(bundle):
    """The fused solver-step kernel must be bit-identical to the numpy solver.

    Runs the whole 20-step schedule through the kernel, feeding each step's
    device ``x0`` back as the next step's ``x0_prev``, and compares both the
    returned sample and ``x0`` against ``DPMSolverMultistepScheduler.step``.
    Order 1 and order 2 are both exercised by the schedule itself.
    """
    import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.vibevoice import diffusion as hip_diff
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    from hipengine.runtime.vibevoice_tts_diffusion import _bf16_bits_to_f32

    spec, _, _, _ = bundle
    runtime = get_hip_runtime()
    shape = (2, spec.latent_size)
    count = int(np.prod(shape))

    rng = np.random.default_rng(0)
    sample = diff_ref.bf16_round(rng.standard_normal(shape).astype(np.float32))
    eps = diff_ref.bf16_round(rng.standard_normal(shape).astype(np.float32))

    s_dev, e_dev, p_dev, out_dev, x0_dev = (malloc(count * 2) for _ in range(5))

    def upload(buffer, array):
        copy_host_array_to_device(
            buffer, f32_to_bf16_bits(np.ascontiguousarray(array).reshape(-1))
        )

    def download(buffer):
        raw = np.empty(count, dtype=np.uint16)
        copy_device_to_host(raw.ctypes.data, buffer, count * 2, runtime=runtime)
        return _bf16_bits_to_f32(raw, shape)

    upload(s_dev, sample)
    upload(e_dev, eps)
    upload(p_dev, np.zeros_like(sample))

    ref_sched = diff_ref.DPMSolverMultistepScheduler(spec)
    ref_sched.set_timesteps(spec.num_inference_steps)
    scale_sched = diff_ref.DPMSolverMultistepScheduler(spec)
    scale_sched.set_timesteps(spec.num_inference_steps)

    orders = []
    for timestep in ref_sched.timesteps:
        expected_x0 = ref_sched._convert_model_output(eps, sample.astype(np.float32))
        order, a_b, s_b, scale, coef_b, half_b, inv_r0 = scale_sched.step_scalars()
        expected = ref_sched.step(eps, int(timestep), sample)

        hip_diff.vv_diff_dpm_step_bf16(
            s_dev.ptr, e_dev.ptr, p_dev.ptr, out_dev.ptr, x0_dev.ptr,
            a_b, s_b, scale, coef_b, half_b, inv_r0, count, order,
            runtime=runtime,
        )
        runtime.device_synchronize()
        got = download(out_dev)
        got_x0 = download(x0_dev)

        orders.append(order)
        assert np.array_equal(got_x0, expected_x0), (
            f"x0 diverged at timestep {int(timestep)} (order {order})"
        )
        assert np.array_equal(got, expected), (
            f"sample diverged at timestep {int(timestep)} (order {order}): "
            f"max |diff| {np.abs(got - expected).max()}"
        )

        # Feed this step's device x0 back as the next step's x0_prev.
        upload(p_dev, got_x0)
        upload(s_dev, got)

        scale_sched.model_outputs = scale_sched.model_outputs[1:] + [got_x0]
        if scale_sched.lower_order_nums < spec.solver_order:
            scale_sched.lower_order_nums += 1
        scale_sched._step_index += 1
        sample = expected

    # The schedule must exercise both solver orders, or the test proves nothing
    # about the second-order path.
    assert set(orders) == {1, 2}, f"schedule only exercised orders {sorted(set(orders))}"


def test_gpu_cfg_combine_matches_sampling_oracle(bundle):
    """The device CFG combine must use the sampling path's rounding points.

    The oracle is ``bf16_round(u + bf16_round(cfg * (c - u)))``: the ``(c - u)``
    difference stays in fp32, so there are exactly two roundings. An earlier
    version of this kernel rounded the difference too (three roundings); it had
    no caller, so nothing caught it until the device solver loop started using
    it. This test pins the contract against the oracle formula directly rather
    than against another kernel.
    """
    import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.vibevoice import diffusion as hip_diff
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    from hipengine.runtime.vibevoice_tts_diffusion import _bf16_bits_to_f32

    runtime = get_hip_runtime()
    rng = np.random.default_rng(7)
    n = 64
    # Include a case where the unrounded difference and the rounded one disagree
    # after scaling, so a three-rounding kernel cannot pass by luck.
    cond = diff_ref.bf16_round(rng.standard_normal(n).astype(np.float32) * 3.0)
    uncond = diff_ref.bf16_round(rng.standard_normal(n).astype(np.float32) * 3.0)
    cfg = 1.3

    c_dev, u_dev, out_dev = (malloc(n * 2) for _ in range(3))
    copy_host_array_to_device(c_dev, f32_to_bf16_bits(cond))
    copy_host_array_to_device(u_dev, f32_to_bf16_bits(uncond))
    hip_diff.vv_diff_cfg_combine_bf16(
        c_dev.ptr, u_dev.ptr, cfg, out_dev.ptr, n, runtime=runtime
    )
    runtime.device_synchronize()
    raw = np.empty(n, dtype=np.uint16)
    copy_device_to_host(raw.ctypes.data, out_dev, raw.nbytes, runtime=runtime)
    got = _bf16_bits_to_f32(raw, (n,))

    expected = diff_ref.bf16_round(
        uncond + diff_ref.bf16_round(np.float32(cfg) * (cond - uncond))
    )
    three_rounding = diff_ref.bf16_round(
        uncond
        + diff_ref.bf16_round(np.float32(cfg) * diff_ref.bf16_round(cond - uncond))
    )

    assert not np.array_equal(expected, three_rounding), (
        "test inputs do not distinguish the two- and three-rounding forms"
    )
    assert np.array_equal(got, expected), (
        "device CFG combine is not the sampling oracle's rounding chain: "
        f"{int((got != expected).sum())}/{n} elements differ, "
        f"max |diff| {np.abs(got - expected).max()}"
    )


def test_gpu_diffusion_resolves_through_registry():
    from hipengine.kernels.vibevoice import resolve_vibevoice_kernels

    ops = resolve_vibevoice_kernels()
    for name in (
        "vv_diff_rmsnorm_bf16",
        "vv_diff_silu_bf16",
        "vv_diff_add_bf16",
        "vv_diff_modulate_bf16",
        "vv_diff_gated_residual_bf16",
        "vv_diff_cfg_combine_bf16",
        "vv_diff_dpm_step_bf16",
    ):
        assert callable(getattr(ops, name)), f"{name} not registered in the vibevoice family"
