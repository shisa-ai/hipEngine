"""CPU reference for the VibeVoice-TTS diffusion head and DPMSolver sampler.

Mirrors the community fork's ``VibevoiceDiffusionHead``
(``modular_vibevoice_diffusion_head.py``) and its vendored
``vibevoice/schedule/dpm_solver.py`` ``DPMSolverMultistepScheduler`` for the
``microsoft/VibeVoice-1.5B`` checkpoint configuration:

- prediction: ``v_prediction``, schedule ``cosine`` over 1000 train steps,
  solver ``dpmsolver++`` order 2 type ``midpoint``, ``linspace`` spacing,
  ``final_sigmas_type='zero'``, ``lower_order_final=True``;
- sampling: ``sample_speech_tokens`` -- two-branch CFG (condition, then
  negative condition stacked on axis 0), the head runs on the duplicated
  first branch, ``eps = uncond + cfg * (cond - uncond)`` is re-duplicated and
  the solver advances both branches identically.

Rounding structure is the torch-eager one, reproduced op by op:

- head activations live on the bfloat16 grid; each linear is an FP32
  accumulation rounded once (``vibevoice_linear``);
- the timestep is cast to bfloat16 *before* ``timestep_embedding`` (torch
  eager passes ``t.repeat(n).to(bf16)``; 999 -> 1000, 949 -> 948);
- RMSNorm computes in FP32 and rounds the normalized product back to bfloat16;
- in the solver, a 0-dim FP32 torch tensor times an N-dim bfloat16 tensor
  casts the scalar to bfloat16 first (verified against eager torch), so every
  ``alpha * x0_pred``-style product is a bfloat16-grid multiply.

The solver itself upcasts the sample to FP32 inside ``step`` and rounds the
result once at the end, exactly as the fork does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from hipengine.kernels.cpu_reference.maple import bf16_round


def _scalar(value: object) -> float:
    """Extract a Python float from a bf16-round result (which is 0-d or 1-element)."""
    return float(np.asarray(value).reshape(-1)[0])
from hipengine.kernels.cpu_reference.vibevoice_asr import _finite, vibevoice_linear

ArrayLike = Any


@dataclass(frozen=True)
class VibevoiceDiffusionSpec:
    """Static geometry of the diffusion head and its sampler."""

    hidden_size: int
    head_layers: int
    head_ffn_ratio: float
    latent_size: int
    rms_norm_eps: float
    frequency_embedding_size: int
    num_train_timesteps: int
    num_inference_steps: int
    prediction_type: str
    algorithm_type: str
    solver_order: int
    solver_type: str
    beta_schedule: str
    timestep_spacing: str
    final_sigmas_type: str
    lower_order_final: bool
    euler_at_final: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VibevoiceDiffusionSpec":
        return cls(
            hidden_size=int(config["hidden_size"]),
            head_layers=int(config["head_layers"]),
            head_ffn_ratio=float(config["head_ffn_ratio"]),
            latent_size=int(config["latent_size"]),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-5)),
            frequency_embedding_size=256,
            num_train_timesteps=int(config.get("ddpm_num_steps", 1000)),
            num_inference_steps=int(config.get("ddpm_num_inference_steps", 20)),
            prediction_type=str(config.get("prediction_type", "v_prediction")),
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            beta_schedule="cosine",
            timestep_spacing="linspace",
            final_sigmas_type="zero",
            lower_order_final=True,
            euler_at_final=False,
        )

    @property
    def ffn_dim(self) -> int:
        return int(self.hidden_size * self.head_ffn_ratio)


@dataclass(frozen=True)
class VibevoiceDiffusionWeights:
    """Flat weight bundle for the diffusion head, all FP32 host arrays.

    Linear weights keep the checkpoint layout ``[out, in]``.
    """

    noisy_images_proj: np.ndarray
    cond_proj: np.ndarray
    t_mlp_0: np.ndarray
    t_mlp_2: np.ndarray
    layers: tuple[dict[str, np.ndarray], ...]
    final_adaLN: np.ndarray
    final_linear: np.ndarray

    @property
    def num_params(self) -> int:
        n = (
            self.noisy_images_proj.size
            + self.cond_proj.size
            + self.t_mlp_0.size
            + self.t_mlp_2.size
            + self.final_adaLN.size
            + self.final_linear.size
        )
        for lw in self.layers:
            n += sum(a.size for a in lw.values())
        return int(n)


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """Glide cosine beta schedule (the fork's ``alpha_transform_type='cosine'``)."""

    def alpha_bar_fn(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = np.empty(num_diffusion_timesteps, dtype=np.float32)
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas[i] = min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta)
    return betas


def timestep_embedding(t_value: float, dim: int, max_period: int = 10000) -> np.ndarray:
    """The fork's static ``timestep_embedding`` for a scalar timestep.

    ``t_value`` must already be the bfloat16-rounded value the head receives.
    Computation matches eager: FP32 frequency constants, FP32 args, and one
    bfloat16 rounding of the concatenated ``[cos, sin]`` embedding.
    """
    half = dim // 2
    exponent = -math.log(max_period) * np.arange(0, half, dtype=np.float32) / half
    freqs = np.exp(exponent)
    args = np.float32(t_value) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = np.concatenate([embedding, np.zeros(1, dtype=np.float32)], axis=-1)
    return bf16_round(embedding)


def _silu(x: np.ndarray, dtype: str) -> np.ndarray:
    """Eager-faithful SiLU: FP32 math, one rounding at the boundary."""
    from hipengine.kernels.cpu_reference.vibevoice_asr import _round

    return _round("silu", x.astype(np.float32) / (1.0 + np.exp(-x.astype(np.float32))), dtype)


def _rmsnorm_affine_bf16(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    """RMSNorm with the fork's two-stage eager rounding.

    ``self._norm(x.float()).type_as(x)`` rounds the normalized value to bf16
    first; the ``* weight`` product is a separate bf16 kernel that rounds
    again. The no-affine final-layer norm has only the first rounding.
    """
    xf = x.astype(np.float32)
    normed = bf16_round(xf * np.reciprocal(np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)))
    if weight is not None:
        normed = bf16_round(normed * weight.astype(np.float32))
    return normed


def _modulate(x: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``x * (1 + scale) + shift`` with one rounding per binary op (eager)."""
    return bf16_round(bf16_round(x * bf16_round(1.0 + scale)) + shift)


def diffusion_head_forward(
    spec: VibevoiceDiffusionSpec,
    weights: VibevoiceDiffusionWeights,
    x: ArrayLike,
    t: ArrayLike,
    condition: ArrayLike,
    *,
    dtype: str | None = "bfloat16",
) -> np.ndarray:
    """Predict the noise/velocity ``eps`` for batched branches.

    ``x``: ``(N, latent_size)`` noisy latents; ``t``: ``(N,)`` timesteps (any
    float dtype -- cast to the bfloat16 grid exactly as eager does);
    ``condition``: ``(N, hidden_size)``. Returns ``(N, latent_size)``.
    """
    xf = np.asarray(x, dtype=np.float32)
    cf = np.asarray(condition, dtype=np.float32)
    if dtype is None:
        raise ValueError("this reference models the bf16 checkpoint; pass dtype='bfloat16'")

    # x = noisy_images_proj(x) -- FP32 accumulation, one rounding.
    h = vibevoice_linear(xf, weights.noisy_images_proj, None, dtype=dtype)

    # t = t_embedder(t): cast the timestep to the activation dtype first.
    tb = bf16_round(np.asarray(t, dtype=np.float32))
    t_freq = np.stack([timestep_embedding(v, spec.frequency_embedding_size) for v in tb])
    t_hidden = vibevoice_linear(t_freq, weights.t_mlp_0, None, dtype=dtype)
    t_hidden = _silu(t_hidden, dtype)
    t_hidden = vibevoice_linear(t_hidden, weights.t_mlp_2, None, dtype=dtype)

    cond = vibevoice_linear(cf, weights.cond_proj, None, dtype=dtype)
    c = bf16_round(cond + t_hidden)

    for lw in weights.layers:
        # adaLN modulation: silu(c) -> linear -> chunk(3).
        ada = vibevoice_linear(_silu(c, dtype), lw["adaLN_weight"], None, dtype=dtype)
        shift, scale, gate = (
            ada[..., 0 : spec.hidden_size],
            ada[..., spec.hidden_size : 2 * spec.hidden_size],
            ada[..., 2 * spec.hidden_size :],
        )
        normed = _rmsnorm_affine_bf16(h, lw["norm_weight"], spec.rms_norm_eps)
        modulated = _modulate(normed, shift, scale)
        gate_h = vibevoice_linear(modulated, lw["gate_proj"], None, dtype=dtype)
        up_h = vibevoice_linear(modulated, lw["up_proj"], None, dtype=dtype)
        ffn_out = vibevoice_linear(
            bf16_round(_silu(gate_h, dtype) * up_h), lw["down_proj"], None, dtype=dtype
        )
        h = bf16_round(h + bf16_round(gate * ffn_out))

    # final_layer: no-affine RMSNorm, adaLN chunk(2), linear to latent size.
    ada = vibevoice_linear(_silu(c, dtype), weights.final_adaLN, None, dtype=dtype)
    shift, scale = ada[..., 0 : spec.hidden_size], ada[..., spec.hidden_size :]
    normed = _rmsnorm_affine_bf16(h, None, spec.rms_norm_eps)
    out = vibevoice_linear(_modulate(normed, shift, scale), weights.final_linear, None, dtype=dtype)
    _finite("diffusion_head_out", out)
    return out


_ALPHA_BAR_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _alpha_bar_tables(num_train_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alphas_cumprod, init_sigmas)`` for a training schedule.

    Both arrays depend only on ``num_train_timesteps``, but the cumulative
    product below is a Python-level loop over every training step (1000 for
    this checkpoint) using fp32 scalar arithmetic. It matches torch's
    sequential fp32 accumulation, so it cannot be vectorised without changing
    the values. Rebuilding it per scheduler instance made the TTS session pay
    ~495 us on every diffusion frame -- 25 times per request -- to recompute a
    constant. The scheduler never mutates either table, so sharing them across
    instances is safe.
    """
    cached = _ALPHA_BAR_TABLES.get(num_train_timesteps)
    if cached is not None:
        return cached
    betas = betas_for_alpha_bar(num_train_timesteps)
    alphas = 1.0 - betas
    # FP32 cumprod matches torch's sequential FP32 accumulation.
    ac = np.empty_like(alphas)
    acc = np.float32(1.0)
    for i, a in enumerate(alphas):
        acc = np.float32(acc * a)
        ac[i] = acc
    init_sigmas = (((1.0 - ac) / ac) ** 0.5).astype(np.float32)
    _ALPHA_BAR_TABLES[num_train_timesteps] = (ac, init_sigmas)
    return ac, init_sigmas


class DPMSolverMultistepScheduler:
    """The fork's vendored scheduler, narrowed to the checkpoint configuration.

    Reproduces ``set_timesteps`` / ``step`` numerics: FP32 sigmas from the
    cosine schedule with ``np.interp`` (FP64 interpolation, then one FP32
    cast), FP32 internal solver math, bfloat16-grid products wherever eager
    torch promotes a 0-dim FP32 scalar against bfloat16 activations.
    """

    def __init__(self, spec: VibevoiceDiffusionSpec):
        if spec.prediction_type != "v_prediction" or spec.algorithm_type != "dpmsolver++":
            raise ValueError("only dpmsolver++ / v_prediction is modelled")
        if spec.solver_order != 2 or spec.solver_type != "midpoint":
            raise ValueError("only order-2 midpoint solver is modelled")
        self.spec = spec
        self.alphas_cumprod, self.init_sigmas = _alpha_bar_tables(
            spec.num_train_timesteps
        )

    def set_timesteps(self, num_inference_steps: int) -> None:
        # lambda_min_clipped = -inf -> clipped_idx = 0 -> last_timestep = T.
        timesteps = np.linspace(0, self.spec.num_train_timesteps - 1, num_inference_steps + 1)
        timesteps = timesteps.round()[::-1][:-1].copy().astype(np.int64)
        sigmas = np.interp(timesteps, np.arange(0, len(self.init_sigmas)), self.init_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.timesteps = timesteps
        self.sigmas = sigmas
        self.num_inference_steps = num_inference_steps
        self.model_outputs: list[np.ndarray | None] = [None] * self.spec.solver_order
        self.lower_order_nums = 0
        self._step_index = 0

    @staticmethod
    def _alpha_sigma(sigma: np.ndarray | float):
        alpha = 1.0 / np.sqrt(sigma**2 + 1.0)
        return np.float32(alpha), np.float32(sigma * alpha)
    def step_scalars(self) -> tuple[int, float, float, float, float, float, float]:
        """Order and per-step scalars for the current position, read-only.

        Shared by the numpy ``_first_order``/``_second_order`` path and the
        device solver so the two cannot drift. Returns
        ``(order, alpha_b, sigma_b, scale, coef_b, half_coef_b, inv_r0_b)``;
        the bf16-rounded scalars are exactly what eager would produce, and
        ``half_coef_b``/``inv_r0_b`` are 0.0 at order 1. Does not advance
        ``_step_index`` or ``lower_order_nums``.
        """
        last = self._step_index == len(self.timesteps) - 1
        lower_order_final = last and (
            self.spec.euler_at_final
            or (self.spec.lower_order_final and len(self.timesteps) < 15)
            or self.spec.final_sigmas_type == "zero"
        )
        order = 1 if (
            self.spec.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final
        ) else 2

        alpha_s, sigma_s = self._alpha_sigma(np.float32(self.sigmas[self._step_index]))
        alpha_t, sigma_t = self._alpha_sigma(np.float32(self.sigmas[self._step_index + 1]))
        alpha_b = bf16_round(np.float32(alpha_s))
        sigma_b = bf16_round(np.float32(sigma_s))
        with np.errstate(divide="ignore"):
            lambda_t = np.float32(np.log(alpha_t) - np.log(sigma_t))
            lambda_s = np.float32(np.log(alpha_s) - np.log(sigma_s))
        h = np.float32(lambda_t - lambda_s)
        scale = np.float32(sigma_t / sigma_s)

        if order == 1:
            coef_b = bf16_round(np.float32(alpha_t * (np.exp(-h) - 1.0)))
            return (
                1, _scalar(alpha_b), _scalar(sigma_b), _scalar(scale),
                _scalar(coef_b), 0.0, 0.0,
            )

        alpha_s1, sigma_s1 = self._alpha_sigma(np.float32(self.sigmas[self._step_index - 1]))
        with np.errstate(divide="ignore"):
            lambda_s1 = np.float32(np.log(alpha_s1) - np.log(sigma_s1))
        h_0 = np.float32(lambda_s - lambda_s1)
        r0 = np.float32(h_0 / h)
        coef = np.float32(alpha_t * (np.exp(-h) - 1.0))
        return (
            2,
            _scalar(alpha_b),
            _scalar(sigma_b),
            _scalar(scale),
            _scalar(bf16_round(np.float32(coef))),
            _scalar(bf16_round(np.float32(0.5 * coef))),
            _scalar(bf16_round(np.float32(1.0 / r0))),
        )

    def advance_step(self) -> None:
        """Advance the schedule position without computing an update.

        The device-path counterpart of the state advance inside :meth:`step`,
        which calls this so the two cannot drift. ``model_outputs`` is not
        maintained here: :meth:`step_scalars`, the device path's only consumer,
        reads only ``sigmas``, ``_step_index`` and ``lower_order_nums``.
        """
        if self.lower_order_nums < self.spec.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

    def _convert_model_output(self, model_output: np.ndarray, sample: np.ndarray) -> np.ndarray:
        """``x0_pred = alpha_t * sample - sigma_t * eps`` on the bf16 grid."""
        sigma = np.float32(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._alpha_sigma(sigma)
        alpha_b = bf16_round(np.float32(alpha_t))
        sigma_b = bf16_round(np.float32(sigma_t))
        # Two bf16-grid multiplies, then a bf16 add: three eager kernels.
        t1 = bf16_round(alpha_b * sample)
        t2 = bf16_round(sigma_b * model_output)
        return bf16_round(t1 - t2)

    def _first_order(self, model_output: np.ndarray, sample: np.ndarray) -> np.ndarray:
        sigma_t = np.float32(self.sigmas[self._step_index + 1])
        sigma_s = np.float32(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s, sigma_s = self._alpha_sigma(sigma_s)
        with np.errstate(divide="ignore"):
            lambda_t = np.float32(np.log(alpha_t) - np.log(sigma_t))
            lambda_s = np.float32(np.log(alpha_s) - np.log(sigma_s))
        h = np.float32(lambda_t - lambda_s)
        # (sigma_t/sigma_s) is an FP32 scalar times the FP32-upcast sample.
        scale = np.float32(sigma_t / sigma_s)
        coef = bf16_round(np.float32(alpha_t * (np.exp(-h) - 1.0)))
        # FP32 term minus a bf16-grid product -> FP32, rounded once at the end.
        return scale * sample - bf16_round(coef * model_output)

    def _second_order(self, sample: np.ndarray) -> np.ndarray:
        sigma_t = np.float32(self.sigmas[self._step_index + 1])
        sigma_s0 = np.float32(self.sigmas[self._step_index])
        sigma_s1 = np.float32(self.sigmas[self._step_index - 1])
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        alpha_s1, sigma_s1 = self._alpha_sigma(sigma_s1)
        with np.errstate(divide="ignore"):
            lambda_t = np.float32(np.log(alpha_t) - np.log(sigma_t))
            lambda_s0 = np.float32(np.log(alpha_s0) - np.log(sigma_s0))
            lambda_s1 = np.float32(np.log(alpha_s1) - np.log(sigma_s1))
        h = np.float32(lambda_t - lambda_s0)
        h_0 = np.float32(lambda_s0 - lambda_s1)
        r0 = np.float32(h_0 / h)
        m0 = self.model_outputs[-1]
        m1 = self.model_outputs[-2]
        d0 = m0
        # eager: D1 = (1/r0) * (m0 - m1) -- the difference rounds first, then
        # the 0-dim FP32 scalar casts to bf16 for the product.
        d1 = bf16_round(bf16_round(np.float32(1.0 / r0)) * bf16_round(m0 - m1))
        scale = np.float32(sigma_t / sigma_s0)
        coef = np.float32(alpha_t * (np.exp(-h) - 1.0))
        term0 = scale * sample
        term1 = bf16_round(bf16_round(np.float32(coef)) * d0)
        term2 = bf16_round(bf16_round(np.float32(0.5 * coef)) * d1)
        return term0 - term1 - term2

    def step(self, model_output: np.ndarray, timestep: int, sample: np.ndarray) -> np.ndarray:
        if self._step_index >= self.num_inference_steps:
            raise RuntimeError("scheduler exhausted; call set_timesteps")
        if timestep != int(self.timesteps[self._step_index]):
            raise ValueError(
                f"timestep {timestep} does not match schedule position "
                f"{self._step_index} ({int(self.timesteps[self._step_index])})"
            )
        last = self._step_index == len(self.timesteps) - 1
        lower_order_final = last and (
            self.spec.euler_at_final
            or (self.spec.lower_order_final and len(self.timesteps) < 15)
            or self.spec.final_sigmas_type == "zero"
        )

        x0 = self._convert_model_output(model_output, sample)
        for i in range(self.spec.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = x0

        sample32 = sample.astype(np.float32)
        if self.spec.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev = self._first_order(x0, sample32)
        elif self.spec.solver_order == 2 or self.lower_order_nums < 2:
            prev = self._second_order(sample32)
        else:
            raise ValueError("order-3 solver is not modelled")

        self.advance_step()
        return bf16_round(prev)


def sample_speech_tokens(
    spec: VibevoiceDiffusionSpec,
    weights: VibevoiceDiffusionWeights,
    condition: ArrayLike,
    neg_condition: ArrayLike,
    cfg_scale: float,
    initial_noise: ArrayLike,
    *,
    scheduler: DPMSolverMultistepScheduler | None = None,
    collect: list[dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The fork's ``sample_speech_tokens`` for one sample.

    ``condition``/``neg_condition``: ``(1, hidden_size)``; ``initial_noise``:
    ``(2, latent_size)`` (the recorded draw, both branches). Returns
    ``(final speech row (1, latent), final eps row (1, latent))``. When
    ``collect`` is given, appends one dict per step with ``eps`` (2, latent)
    and ``speech`` (2, latent) snapshots.
    """
    cf = np.asarray(condition, dtype=np.float32)
    ncf = np.asarray(neg_condition, dtype=np.float32)
    cond = bf16_round(np.concatenate([cf, ncf], axis=0))
    speech = bf16_round(np.asarray(initial_noise, dtype=np.float32))
    sched = scheduler if scheduler is not None else DPMSolverMultistepScheduler(spec)
    sched.set_timesteps(spec.num_inference_steps)
    for t in sched.timesteps:
        half = speech[: len(speech) // 2]
        combined = np.concatenate([half, half], axis=0)
        raw_eps = diffusion_head_forward(spec, weights, combined, np.full(2, int(t)), cond)
        cond_eps, uncond_eps = np.split(raw_eps, 2, axis=0)
        half_eps = bf16_round(uncond_eps + bf16_round(cfg_scale * (cond_eps - uncond_eps)))
        eps = np.concatenate([half_eps, half_eps], axis=0)
        speech = sched.step(eps, int(t), speech)
        if collect is not None:
            collect.append({"eps": raw_eps.copy(), "eps_cfg": eps.copy(), "speech": speech.copy()})
    return speech[: len(speech) // 2], raw_eps[: len(raw_eps) // 2]


def scale_speech_latent(
    speech_latent: ArrayLike, scaling_factor: float, bias_factor: float
) -> np.ndarray:
    """``latent / scale - bias`` with one bf16 rounding per checkpoint op."""
    scaled = bf16_round(
        bf16_round(np.asarray(speech_latent, dtype=np.float32) / bf16_round(np.float32(scaling_factor)))
        - bf16_round(np.float32(bias_factor))
    )
    return scaled
