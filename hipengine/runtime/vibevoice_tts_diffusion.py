"""Torch-free HIP runtime for the VibeVoice-TTS diffusion head.

Drives the registered ``vibevoice`` primitives for the MLP-DiT head: linears
via ``dense_gemv_out_bf16`` (fp32 accumulation), RMSNorm via
``vv_rmsnorm_bf16`` (the final layer's no-affine norm passes an all-ones
weight -- an exact fp32 no-op), and the eager-rounded elementwise chain via
``vv_diff_*``. The DPMSolver itself runs on the host through the CPU
reference's scheduler (per-step solver tensors are 2x64, so the solver's
device transfers are a few hundred bytes per step); only the head runs on
device.

Bit-faithfulness contract: the GPU head must match the numpy CPU reference
within the eager-bf16 envelope (occasional 1-2 ulp flips from fp32
accumulation-order differences, exactly like the decoder lane), and the
host-side solver math is shared verbatim with the CPU reference.
"""

from __future__ import annotations

import numpy as np

import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    malloc,
)
from hipengine.kernels.vibevoice import resolve_vibevoice_kernels
from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

# Block size for the head's GEMVs. The shared default of 256 spends most of the
# call in the block-wide reduction tree: this head is 123M params over 17 GEMVs
# with only 2 rows, so each block reduces 1536-4608 values for one output. At 64
# threads a full head call measures 2.08 ms against 3.33 ms at 256.
_GEMV_THREADS = 64


def _bf16_u16(array: np.ndarray) -> np.ndarray:
    """bf16-rounded FP32 values as uint16 bit patterns."""
    return f32_to_bf16_bits(np.ascontiguousarray(array, dtype=np.float32))


def _bf16_bits_to_f32(bits: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return (
        np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    ).view(np.float32).reshape(shape).copy()


class VibevoiceTTSDiffusionHeadGPU:
    """MLP-DiT diffusion head on HIP with the host-side DPMSolver."""

    def __init__(self, spec, weights, *, runtime=None) -> None:
        from hipengine.core.hip import get_hip_runtime

        self.spec = spec
        self.weights = weights
        self.runtime = runtime or get_hip_runtime()
        self.kernels = resolve_vibevoice_kernels()
        self._linears: dict[str, tuple[DeviceBuffer, int, int]] = {}
        self._buffers: dict[str, DeviceBuffer] = {}
        self._upload_weights()
        self._alloc_state()

    # -- setup -------------------------------------------------------------
    def _linear(self, key: str, weight: np.ndarray) -> None:
        array = _bf16_u16(np.asarray(weight))
        buffer = malloc(array.nbytes)
        copy_host_array_to_device(buffer, array)
        self._linears[key] = (buffer, int(array.shape[0]), int(array.shape[1]))

    def _upload_weights(self) -> None:
        spec, w = self.spec, self.weights
        self._linear("noisy_proj", w.noisy_images_proj)
        self._linear("cond_proj", w.cond_proj)
        self._linear("t_mlp_0", w.t_mlp_0)
        self._linear("t_mlp_2", w.t_mlp_2)
        for i, lw in enumerate(w.layers):
            self._linear(f"L{i}_adaLN", lw["adaLN_weight"])
            self._linear(f"L{i}_gate", lw["gate_proj"])
            self._linear(f"L{i}_up", lw["up_proj"])
            self._linear(f"L{i}_down", lw["down_proj"])
            norm_bits = _bf16_u16(lw["norm_weight"])
            norm_buf = malloc(norm_bits.nbytes)
            copy_host_array_to_device(norm_buf, norm_bits)
            self._buffers[f"normw{i}"] = norm_buf
        self._linear("final_adaLN", w.final_adaLN)
        self._linear("final_linear", w.final_linear)

    def _alloc_state(self) -> None:
        spec = self.spec
        h = spec.hidden_size
        # rows=2 bf16 row buffers for every activation shape the head touches.
        for key, width in (
            ("h", h),
            ("t_mlp", h),
            ("cond", h),
            ("cond_proj", h),
            ("c", h),
            ("normed", h),
            ("modulated", h),
            ("ffn_out", h),
            ("h_out", h),
            ("silu", spec.ffn_dim),
            ("ffn_gated", spec.ffn_dim),
            ("eps", spec.latent_size),
        ):
            self._buffers[key] = malloc(2 * width * 2)
        # Per-layer and final adaLN outputs, gate/up hidden rows.
        for i in range(spec.head_layers):
            self._buffers[f"ada{i}"] = malloc(2 * 3 * h * 2)
            self._buffers[f"gate_h{i}"] = malloc(2 * spec.ffn_dim * 2)
            self._buffers[f"up_h{i}"] = malloc(2 * spec.ffn_dim * 2)
        self._buffers["final_ada"] = malloc(2 * 2 * h * 2)
        self._buffers["t_freq_rows"] = malloc(2 * spec.frequency_embedding_size * 2)
        self._buffers["x_rows"] = malloc(2 * spec.latent_size * 2)
        # Caches for work that is invariant either across the 20 solver steps of a
        # frame (the condition projection) or across the 25 frames of a request
        # (the timestep MLP, whose 20-step schedule repeats). Both skip weight
        # reads, not merely a copy: cond_proj is 4.7 MB and the two t-MLP
        # matrices are 5.5 MB of the head's 246.6 MB per step.
        self._t_mlp_cache: dict[float, DeviceBuffer] = {}
        self._cond_proj_key: np.ndarray | None = None

    def close(self) -> None:
        for buffer in self._t_mlp_cache.values():
            free(buffer)
        self._t_mlp_cache.clear()
        self._cond_proj_key = None
        for buffer in self._buffers.values():
            free(buffer)
        self._buffers.clear()
        for buffer, _, _ in self._linears.values():
            free(buffer)
        self._linears.clear()

    # -- primitives --------------------------------------------------------
    def _gemv(self, key: str, x_ptr: int, out_ptr: int, rows: int) -> None:
        weight, out_features, in_features = self._linears[key]
        self.kernels.dense_gemv_out_bf16(
            x_ptr,
            weight.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            threads=_GEMV_THREADS,
        )

    def _silu(self, x: DeviceBuffer, out: DeviceBuffer, n: int) -> None:
        self.kernels.vv_diff_silu_bf16(x.ptr, out.ptr, n)

    def _rmsnorm(self, x: DeviceBuffer, weight: DeviceBuffer | None, out: DeviceBuffer, rows: int) -> None:
        self.kernels.vv_diff_rmsnorm_bf16(
            x.ptr,
            weight.ptr if weight is not None else None,
            out.ptr,
            rows,
            self.spec.hidden_size,
            self.spec.rms_norm_eps,
        )

    # -- head forward ------------------------------------------------------
    def forward(
        self,
        x_bf16_u16: np.ndarray,
        t_value: float,
        condition_bf16_u16: np.ndarray,
    ) -> np.ndarray:
        """One head call on device.

        ``x_bf16_u16``: (2, latent) uint16 bf16 rows; ``t_value``: raw
        timestep (rounded to the bf16 grid here, exactly as eager casts the
        scheduler timestep before the sinusoid); ``condition_bf16_u16``:
        (2, hidden) uint16. Returns the raw eps (2, latent) as FP32 values on
        the bf16 grid.
        """
        spec = self.spec
        h_dim = spec.hidden_size
        rows = 2
        t_bf = float(np.asarray(diff_ref.bf16_round(np.float32(t_value))).reshape(-1)[0])

        b = self._buffers
        # x = noisy_images_proj(x)
        copy_host_array_to_device(b["x_rows"], np.ascontiguousarray(x_bf16_u16))
        self._gemv("noisy_proj", b["x_rows"].ptr, b["h"].ptr, rows)
        h = b["h"]

        # t = t_embedder(t): host sinusoid (fp32 constants + one rounding),
        # device MLPs. The 20-step schedule repeats across the request's frames,
        # so each distinct timestep's MLP output is computed once and kept.
        t_mlp = self._t_mlp_cache.get(t_bf)
        if t_mlp is None:
            t_freq = diff_ref.timestep_embedding(t_bf, spec.frequency_embedding_size)
            t_freq_u16 = np.repeat(_bf16_u16(t_freq.reshape(1, -1)), rows, axis=0)
            copy_host_array_to_device(b["t_freq_rows"], t_freq_u16)
            self._gemv("t_mlp_0", b["t_freq_rows"].ptr, b["t_mlp"].ptr, rows)
            self._silu(b["t_mlp"], b["silu"], rows * h_dim)
            t_mlp = malloc(2 * h_dim * 2)
            self._gemv("t_mlp_2", b["silu"].ptr, t_mlp.ptr, rows)
            self._t_mlp_cache[t_bf] = t_mlp

        # condition = cond_proj(condition); c = r(condition + t). The condition
        # is fixed for all 20 steps of a frame, so its projection is computed
        # once per frame and the buffer reused thereafter.
        cond_rows = np.ascontiguousarray(condition_bf16_u16)
        if self._cond_proj_key is None or not np.array_equal(cond_rows, self._cond_proj_key):
            copy_host_array_to_device(b["cond"], cond_rows)
            self._gemv("cond_proj", b["cond"].ptr, b["cond_proj"].ptr, rows)
            self._cond_proj_key = cond_rows
        self.kernels.vv_diff_add_bf16(b["cond_proj"].ptr, t_mlp.ptr, b["c"].ptr, rows * h_dim)
        c_buf = b["c"]

        # silu(c) is invariant across the head layers and the final projection,
        # so it is computed once per step instead of once per layer. Nothing in
        # the loop below writes b["silu"].
        self._silu(c_buf, b["silu"], rows * h_dim)
        silu_c = b["silu"]

        for i in range(spec.head_layers):
            # adaLN modulation: silu(c) -> linear -> chunk(3)
            ada = b[f"ada{i}"]
            self._gemv(f"L{i}_adaLN", silu_c.ptr, ada.ptr, rows)
            gate = ada.ptr + 2 * h_dim * 2  # column 2H of each (rows, 3H) row

            # normed = rmsnorm(h, layer weight); modulated = r(r(normed * r(1 + scale)) + shift)
            # in one bit-identical launch; the adaLN row layout is (rows, 3H):
            # shift at col 0, scale at col H.
            self.kernels.vv_diff_rmsnorm_modulate_bf16(
                h.ptr, b[f"normw{i}"].ptr, ada.ptr, b["modulated"].ptr,
                rows, h_dim, 3 * h_dim, self.spec.rms_norm_eps,
            )

            # ffn: gate/up -> silu_mul -> down. Two single GEMVs beat the fused
            # dual variant here: measured 2.08 vs 2.95 ms per head call at the
            # same thread count, because the dual kernel's two-output block
            # costs more than the x re-read it saves (x is 2x1536 bf16 against
            # 28 MB of weights per layer).
            self._gemv(f"L{i}_gate", b["modulated"].ptr, b[f"gate_h{i}"].ptr, rows)
            self._gemv(f"L{i}_up", b["modulated"].ptr, b[f"up_h{i}"].ptr, rows)
            self.kernels.silu_mul_separate_out_bf16(
                b[f"gate_h{i}"].ptr, b[f"up_h{i}"].ptr, b["ffn_gated"].ptr, rows, spec.ffn_dim
            )
            self._gemv(f"L{i}_down", b["ffn_gated"].ptr, b["ffn_out"].ptr, rows)

            # h = r(h + r(gate * ffn_out)); gate lives at column 2H of each
            # (rows, 3H) adaLN row.
            self.kernels.vv_diff_gated_residual_bf16(
                h.ptr, gate, b["ffn_out"].ptr, b["h_out"].ptr, rows, h_dim, 3 * h_dim
            )
            h = b["h_out"]

        # final layer: no-affine norm, adaLN chunk(2), modulate, linear
        fada = b["final_ada"]
        self._gemv("final_adaLN", silu_c.ptr, fada.ptr, rows)
        self.kernels.vv_diff_rmsnorm_modulate_bf16(
            h.ptr, None, fada.ptr, b["modulated"].ptr, rows, h_dim,
            2 * h_dim, self.spec.rms_norm_eps,
        )
        self._gemv("final_linear", b["modulated"].ptr, b["eps"].ptr, rows)

        raw = np.empty(rows * spec.latent_size, dtype=np.uint16)
        copy_device_to_host(raw.ctypes.data, b["eps"], raw.nbytes, runtime=self.runtime)
        return _bf16_bits_to_f32(raw, (rows, spec.latent_size))

    # -- sampling ----------------------------------------------------------
    def sample_speech_tokens(
        self,
        condition: np.ndarray,
        neg_condition: np.ndarray,
        cfg_scale: float,
        initial_noise: np.ndarray,
        *,
        collect: list[dict[str, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The fork's two-branch CFG loop: head on device, solver on host.

        ``condition``/``neg_condition``: ``(1, hidden)`` each (stacked
        internally); ``initial_noise``: ``(2, latent)``. Returns
        ``(speech_first_row, raw_eps_first_row)``; per-step snapshots
        land in ``collect`` when given (raw head eps, CFG-combined eps, and
        the post-step speech, both branches).
        """
        spec = self.spec
        cf = np.asarray(condition, dtype=np.float32)
        ncf = np.asarray(neg_condition, dtype=np.float32)
        if cf.shape != (1, spec.hidden_size) or ncf.shape != (1, spec.hidden_size):
            raise ValueError(
                f"conditions must be (1, {spec.hidden_size}), got {cf.shape}/{ncf.shape}"
            )
        if np.asarray(initial_noise).shape != (2, spec.latent_size):
            raise ValueError(
                f"initial_noise must be (2, {spec.latent_size}), got {np.asarray(initial_noise).shape}"
            )
        cond = diff_ref.bf16_round(np.concatenate([cf, ncf], axis=0))
        cond_u16 = _bf16_u16(cond)
        speech = diff_ref.bf16_round(np.asarray(initial_noise, dtype=np.float32))
        sched = diff_ref.DPMSolverMultistepScheduler(spec)
        sched.set_timesteps(spec.num_inference_steps)

        raw_eps = None
        for t in sched.timesteps:
            half = speech[: len(speech) // 2]
            combined = np.concatenate([half, half], axis=0)
            raw_eps = self.forward(_bf16_u16(combined), int(t), cond_u16)
            cond_eps, uncond_eps = np.split(raw_eps, 2, axis=0)
            half_eps = diff_ref.bf16_round(
                uncond_eps + diff_ref.bf16_round(cfg_scale * (cond_eps - uncond_eps))
            )
            eps = np.concatenate([half_eps, half_eps], axis=0)
            speech = sched.step(eps, int(t), speech)
            if collect is not None:
                collect.append(
                    {"eps": raw_eps.copy(), "eps_cfg": eps.copy(), "speech": speech.copy()}
                )
        return speech[: len(speech) // 2], raw_eps[: len(raw_eps) // 2]
