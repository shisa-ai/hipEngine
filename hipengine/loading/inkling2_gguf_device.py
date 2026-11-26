"""Packed projection ownership for the Inkling GGUF decoder."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import numpy as np

from hipengine.core.hip import HipMemcpyKind, capture_stream
from hipengine.core.memory import free, malloc
from hipengine.kernels.backends import hip_target_arch_for_backend, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.moe.inkling2_route_tiles import launch_route_ids_i32
from hipengine.kernels.hip_gfx1100.moe.laguna_router import laguna_sigmoid_correction_topk_f32
from hipengine.kernels.hip_gfx1100.quant import (  # noqa: F401
    gguf_nvfp4_gemv, gguf_nvfp4_q8_dot, gguf_q8_0_dense_wide, inkling2_experts,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_nvfp4_selected_wmma import launch_route_schedule
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.loading.inkling2_device_experts import build_expert_route_tiles
from hipengine.quant.gguf import GGMLQuantizationType as Q

# Do not hand HIP a whole file-backed expert bank. Read each window before
# submitting its transfer, and reuse this host staging across physical banks.
_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_SELECTED_ROUTES = 65535
# Leave narrow projections on the scalar kernel: pack8 can lose on W7900.
_Q8_DECODE_MIN_OUTPUTS = 1024
# The rows from which a prompt block's Q8_0 projection takes a wide tile, by tile: `fast` reads the
# tile with one F16 product, `production` the one that adds a residual F16 part to each operand and
# is 1.2 to 2.2 times slower. Below these rows the exact kernel is as fast or faster (W7900, best
# of 20, the exact kernel then each tile). K2304 by N3072, 64 rows: 153, 90 and 121 us; by N1024,
# 64 rows: 69, 71 and 107, 256 rows: 188, 128 and 175. K2304 by N384, 256 rows: 89, 99 and 165;
# 575 rows: 160, 109 and 129 (worklog/entries/20261126T172532.740394Z-lhl-inkling2-mini-3ddb67.md
# and worklog/entries/20261126T175737.613472Z-lhl-inkling2-mini-652e29.md).
_WIDE_DENSE_MIN_ROWS = {"dense_wide64x128_f32_f32_out": 64, "dense_wide64x128_residual_f32_f32_out": 192}
_WIDE_DENSE_NARROW_MIN_ROWS = {"dense_wide64x128_f32_f32_out": 384, "dense_wide64x128_residual_f32_f32_out": 512}
_WIDE_DENSE_WIDE_OUTPUTS = 1024
# The grouped WMMA expert projection runs from this many token rows. Below it the FP32 tiles are
# as fast or faster: a block of 192 rows puts about twelve routes on each of 128 experts.
_WMMA_MIN_ROWS = 192
_WMMA_GROUP_ROWS = 64


class Inkling2GGUFDeviceProjections(Inkling2DeviceDense):
    """Share dense transfer scratch; upload physical expert banks only once."""

    def __init__(self, weights, *, runtime, blas, backend, fast_experts=False, fast_controllers=False,
                 device_routes=False, q8_dot_experts=None, wide_dense=None):
        super().__init__(weights, runtime=runtime, blas=blas)
        load_backend_kernel_package(backend)
        if q8_dot_experts is None:
            q8_dot_experts = fast_experts
        self._tensors = {t.name: t for t in weights.reader.info.tensors}
        self._backend = str(backend)
        self._target_arch = hip_target_arch_for_backend(backend)
        self._launches = {q: resolve(backend=backend, layer="linear", quant=f"gguf_{q.name.lower()}",
                                    variant="gemv_f32_f32_out") for q in (Q.NVFP4, Q.Q8_0, Q.Q5_K)}
        self._bound_launches = {}
        self._selected_launches = {}
        self._tiled_launches = {}
        self._shared_tiled_launches = {}
        self._q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                variant="selected_q8_dot_f32_f32_out") if q8_dot_experts else None)
        self._shared_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                       variant="selected_shared_input_q8_dot_f32_f32_out") if q8_dot_experts else None)
        self._counted_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                        variant="selected_counted_shared_input_q8_dot_f32_f32_out")
                                 if q8_dot_experts else None)
        self._counted_split_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                              variant="selected_counted_split_activation_q8_dot_f32_f32_out")
                                       if q8_dot_experts else None)
        self._split_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                      variant="selected_split_activation_q8_dot_f32_f32_out") if q8_dot_experts else None)
        self._q8_dot_scratch = None
        self._k_library = None
        self._device_routes_requested = bool(device_routes)
        self._device_routes = None
        self._device_route_hidden = None
        self._device_route_hidden_rep = None
        self._router_logits = None
        self._device_route_interim = None
        self._device_route_out = None
        self._router_bias_cache: dict[int, int] = {}
        self._route_scale_cache: dict[str, int] = {}
        self._route_buffer_handles = []
        self._route_state = None
        self._schedule_state = None
        prefill = KernelKey(backend, "linear", "gguf_q8_0", "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out")
        self._q8_prefill = (self._bind_packed(Q.Q8_0, resolve(backend=prefill.backend, layer=prefill.layer,
                                                           quant=prefill.quant, variant=prefill.variant))
                            if is_registered(prefill) else None)
        decode = KernelKey(backend, "linear", "gguf_q8_0", "pack8_gemv_f32_f32_out")
        self._q8_decode = (self._bind_packed(Q.Q8_0, resolve(backend=decode.backend, layer=decode.layer,
                                                           quant=decode.quant, variant=decode.variant))
                           if is_registered(decode) else None)
        # `wide_dense` names the tile a prompt block's Q8_0 projections take, or None for the exact kernel.
        self.wide_dense = wide_dense
        if wide_dense is not None and wide_dense not in _WIDE_DENSE_MIN_ROWS:
            raise ValueError(f"unknown Inkling wide dense tile: {wide_dense}")
        wide = KernelKey(backend, "linear", "gguf_q8_0", wide_dense or "dense_wide64x128_f32_f32_out")
        self._q8_wide = (resolve(backend=wide.backend, layer=wide.layer, quant=wide.quant, variant=wide.variant)
                         if wide_dense and is_registered(wide) else None)
        self._wide_min_rows = (_WIDE_DENSE_MIN_ROWS.get(wide_dense, 0), _WIDE_DENSE_NARROW_MIN_ROWS.get(wide_dense, 0))
        self._wide_libraries = {}
        self._head = weights.mapping.root["output.weight"]
        self._vocab = weights.mapping.vocab_size
        padded = (self._vocab + 7) // 8 * 8
        self._tiled_head_rows = (padded if self._head.ggml_type == Q.Q8_0 and self._q8_prefill is not None
                                 and padded <= self._head.shape[0] else 0)
        for quant in self._launches:
            for variant, launches in (("selected_f32_f32_out", self._selected_launches),
                                      ("selected_tile4_f32_f32_out", self._tiled_launches),
                                      ("selected_tile4_shared_input_f32_f32_out", self._shared_tiled_launches)):
                key = KernelKey(backend, "moe_linear", f"gguf_{quant.name.lower()}", variant)
                if is_registered(key):
                    launch = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
                    launches[quant] = self._bind_packed(quant, launch)
            key = KernelKey(backend, "moe_linear", f"gguf_{quant.name.lower()}", "selected_register_fp4_f32_f32_out")
            if is_registered(key):
                launch = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
                self._selected_launches[quant] = self._bind_packed(quant, launch)
        # Routes that share one input row read it through the shared-input projection, so a decode
        # row is not copied once per route. Banks without it replicate the row in one launch.
        self._shared_selected_launches = {}
        self._replicate = None
        shared = KernelKey(backend, "moe_linear", "gguf_nvfp4", "selected_shared_register_fp4_f32_f32_out")
        if is_registered(shared):
            self._shared_selected_launches[Q.NVFP4] = resolve(backend=shared.backend, layer=shared.layer,
                                                               quant=shared.quant, variant=shared.variant)
        replicate = KernelKey(backend, "inkling2_expert_pointwise", "inkling2_mixed", "replicate_row")
        if is_registered(replicate):
            self._replicate = partial(resolve(backend=replicate.backend, layer=replicate.layer,
                                               quant=replicate.quant, variant=replicate.variant),
                                      target_arch=self._target_arch)
        self._mix = None
        key = KernelKey(backend, "inkling2_expert_pointwise", "inkling2_mixed", "ordered_mix_exact")
        if (is_registered(key) and all(Q(layer.down.weight.ggml_type) in self._selected_launches
                                       for layer in weights.mapping.layers)):
            self._mix = partial(resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant),
                                target_arch=self._target_arch)
        self._activation = None
        if fast_experts:
            if self._mix is None or any(Q(layer.gate_up.weight.ggml_type) not in self._selected_launches
                                        for layer in weights.mapping.layers):
                raise ValueError("Inkling GGUF fast experts require selected gate/up and down projections and ordered mix")
            self._activation = partial(resolve(backend=backend, layer="inkling2_expert_pointwise",
                                               quant="inkling2_mixed", variant="split_activation_device"),
                                       target_arch=self._target_arch)
        self._wmma = self._wmma_arena = self._wmma_down = self._wmma_parts = None
        # The grouped kernel of each expert layout the backend registers it for: NVFP4, and the Q5_K
        # and Q8_0 banks a mixed file keeps in its last layers.
        self._wmma_by_quant = {}
        if fast_experts:
            split = KernelKey(backend, "inkling2_expert_pointwise", "inkling2_mixed", "fp16_parts")
            if is_registered(split):
                for quant in (Q.NVFP4, Q.Q5_K, Q.Q8_0):
                    key = KernelKey(backend, "moe_linear", f"gguf_{quant.name.lower()}",
                                    "selected_wmma_f16_parts_f32_out")
                    if is_registered(key):
                        self._wmma_by_quant[quant] = partial(resolve(backend=key.backend, layer=key.layer,
                                                                     quant=key.quant, variant=key.variant),
                                                             target_arch=self._target_arch)
                if Q.NVFP4 in self._wmma_by_quant:
                    self._wmma = (self._wmma_by_quant[Q.NVFP4],
                                  partial(resolve(backend=split.backend, layer=split.layer, quant=split.quant,
                                                  variant=split.variant), target_arch=self._target_arch))
                    self._wmma_parts = 2
        self._controller_input = self._controller_blend_norm = None
        self._controller_direct = None
        if fast_controllers:
            for variant, name in (("input", "_controller_input"), ("blend_norm", "_controller_blend_norm")):
                setattr(self, name, partial(resolve(backend=backend, layer="inkling2_controller",
                                                    quant="inkling2_mixed", variant=variant),
                                            target_arch=self._target_arch))
            # The same two kernels over input, message and weight rows that stay where they are.
            direct = [KernelKey(backend, "inkling2_controller", "inkling2_mixed", variant)
                      for variant in ("input_direct", "blend_norm_direct")]
            if all(is_registered(key) for key in direct):
                self._controller_direct = tuple(partial(resolve(backend=key.backend, layer=key.layer,
                                                                 quant=key.quant, variant=key.variant),
                                                        target_arch=self._target_arch) for key in direct)
        self._prepare_clip = self._prepare_rms = self._prepare_group = None
        if fast_controllers:
            for variant, name in (("clip", "_prepare_clip"), ("rms", "_prepare_rms"),
                                  ("group", "_prepare_group")):
                key = KernelKey(backend, "inkling2_attention_prepare", "inkling2_mixed", variant)
                if is_registered(key):
                    setattr(self, name, partial(resolve(backend=backend, layer=key.layer, quant=key.quant,
                                                        variant=variant), target_arch=self._target_arch))
        self._vectors = {}
        self._resident = None
        self._bank_scales = {}
        self._upload_staging = None
        self._residual = None
        keys = {variant: KernelKey(backend, "inkling2_residual", "inkling2_mixed", variant)
                for variant in ("row_rms", "row_norm_mul", "row_add", "residual_read")}
        if all(is_registered(key) for key in keys.values()):
            self._residual = {variant: partial(resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                                                       variant=variant), target_arch=self._target_arch)
                              for variant, key in keys.items()}
            for variant, name in (("residual_update", "update"), ("row_norm_mul_group", "norm_group")):
                fused = KernelKey(backend, "inkling2_residual", "inkling2_mixed", variant)
                if is_registered(fused):
                    self._residual[name] = partial(resolve(backend=fused.backend, layer=fused.layer,
                                                           quant=fused.quant, variant=fused.variant),
                                                   target_arch=self._target_arch)

    @property
    def backend(self):
        return self._backend

    @property
    def target_arch(self):
        return self._target_arch

    @property
    def supports_device_mix(self):
        return self._mix is not None

    @property
    def supports_device_activation(self):
        return self._activation is not None

    @property
    def supports_resident_controllers(self):
        return self._controller_input is not None

    @property
    def controller_rows_in_place(self):
        """The in-place controller kernels are registered for this backend."""
        return getattr(self, "_controller_direct", None) is not None

    @property
    def supports_resident_attention(self):
        return self._prepare_clip is not None and self._prepare_rms is not None

    @property
    def supports_resident_block(self):
        return (self.supports_resident_attention and self._residual is not None and self._mix is not None
                and self._activation is not None)

    def residual_kernel(self, variant):
        """Return one registered residual-stack launcher."""
        if self._residual is None:
            raise RuntimeError("Inkling GGUF residual kernels are not available")
        return self._residual[variant]

    def optional_residual_kernel(self, variant):
        """Return a registered residual-stack launcher, or None when this backend has no such variant."""
        return None if self._residual is None else self._residual.get(variant)

    def attention_resident_extent(self, plan):
        """Input, output and scratch bytes one resident attention launch needs."""
        rows, item = plan.rows, np.dtype(np.float32).itemsize
        if plan.controllers:
            plans, norm_weights = self._controller_plans(plan.controllers, plan.norm_weights)
            inp_bytes, out_bytes = self._controller_extent(plans, norm_weights)
        else:
            inp_bytes, out_bytes = rows * plan.hidden * item * len(plan.sources), 0
        matrices = [[self._matrix(name, None) for name in names] for _, names, _ in plan.groups]
        prep, output = plan.prepare, self._matrix(plan.output[0], None)
        query_bytes = rows * prep.nh * prep.dim * item
        kv_bytes = rows * prep.nkv * prep.dim * item
        scratch = (sum(rows * matrix[1][0] * item for group in matrices for matrix in group)
                   + query_bytes + 2 * kv_bytes + rows * prep.nh * prep.rel_width * item
                   + query_bytes + rows * output[1][0] * item)
        return inp_bytes, out_bytes, scratch

    def upload(self, dst, host):
        """Stage a host array into device memory on stream zero."""
        self._transfer.enqueue_upload(dst, np.ascontiguousarray(host, dtype=np.float32))

    def project_controllers_normed(self, controllers, *, norm_weights, eps):
        """Exchange raw controller inputs once: RMS and concatenate, project, then blend and normalize on the device."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF dense residency is closed")
            plans, norm_weights = self._controller_plans(controllers, norm_weights)
            staging = self._controller_staging(plans, norm_weights)
            rows, hidden_size = plans[0][0].shape
            output = np.empty((len(plans), rows, hidden_size), dtype=np.float32)
            inp = self._scratch("_input", self._controller_extent(plans, norm_weights)[0])
            out = self._scratch("_output", self._controller_extent(plans, norm_weights)[1])
            with self._transfer.exchange(inp.ptr, out.ptr, staging, output, pinned=True):
                self._controller_kernels(inp.ptr, out.ptr, plans, norm_weights, eps)
            return output

    def _controller_plans(self, controllers, norm_weights):
        """Validate controller geometry and bind the down, forget and gain matrices."""
        controllers = tuple(controllers)
        norm_weights = tuple(np.ascontiguousarray(w, dtype=np.float32) for w in norm_weights)
        if not controllers or len(controllers) != len(norm_weights):
            raise ValueError("Inkling GGUF resident controllers need one norm weight per controller")
        plans = []
        for (down, forget, gain), hidden, message in controllers:
            # The resident path reads these rows where they are, and the staging path concatenates
            # them: neither needs a contiguous copy of a 9.4 MB message column per controller.
            hidden = np.asarray(hidden, dtype=np.float32)
            message = np.asarray(message, dtype=np.float32)
            first = self._matrix(down, None)
            tails = (self._matrix(forget, None), self._matrix(gain, None))
            width, inputs = first[1]
            if (hidden.ndim != 2 or not all(hidden.shape) or hidden.shape != message.shape
                    or inputs != 2 * hidden.shape[1] or any(m != (hidden.shape[1], width) for _, m, _ in tails)
                    or norm_weights[len(plans)].shape != (hidden.shape[1],)):
                raise ValueError("Inkling GGUF resident controller geometry differs from its matrices")
            plans.append((hidden, message, first, tails))
        rows, hidden_size = plans[0][0].shape
        if any(plan[0].shape != (rows, hidden_size) for plan in plans):
            raise ValueError("Inkling GGUF resident controllers need one row count")
        return plans, norm_weights

    @staticmethod
    def _controller_staging(plans, norm_weights):
        return np.concatenate([plan[0].ravel() for plan in plans] + [plan[1].ravel() for plan in plans]
                              + [w.ravel() for w in norm_weights])

    @staticmethod
    def _controller_extent(plans, norm_weights):
        """Input and output bytes of one controller launch."""
        rows, hidden_size = plans[0][0].shape
        count = len(plans)
        item = np.dtype(np.float32).itemsize
        gate = count * rows * hidden_size * item
        staging = sum(plan[0].nbytes + plan[1].nbytes for plan in plans) + sum(w.nbytes for w in norm_weights)
        return (staging + 4 * gate, gate + count * rows * plans[0][2][1][0] * item)

    def _controller_kernels(self, inp, out, plans, norm_weights, eps):
        """Form controller inputs, project, then blend and normalize, all on the device."""
        rows, hidden_size = plans[0][0].shape
        count = len(plans)
        item = np.dtype(np.float32).itemsize
        gate = count * rows * hidden_size * item
        down_bytes = 2 * gate
        staging_bytes = self._controller_extent(plans, norm_weights)[0] - down_bytes - 2 * gate
        raw, down_inputs = inp, inp + staging_bytes
        forgets, gains = down_inputs + down_bytes, down_inputs + down_bytes + gate
        intermediate = out + count * rows * hidden_size * item
        self._controller_input(raw, raw + gate, down_inputs, rows, hidden_size, count, eps)
        for i, (_, _, first, tails) in enumerate(plans):
            width = first[1][0]
            mid = intermediate + i * rows * width * item
            self._launch_projection(down_inputs + i * down_bytes // count, mid, rows, 2, first, None)
            self._launch_projection(mid, forgets + i * rows * hidden_size * item, rows, 2, tails[0], None)
            self._launch_projection(mid, gains + i * rows * hidden_size * item, rows, 2, tails[1], None)
        self._controller_blend_norm(raw, raw + gate, forgets, gains, raw + 2 * gate, out, rows, hidden_size, count, eps)

    def _controller_kernels_direct(self, inputs, messages, out, plans, eps, weights, scratch):
        """The controller chain over input, message and weight rows that stay where they are.

        `scratch` holds the down inputs, the forget and gain rows; `out` the blended rows and, after
        them, the gate values. Same kernels as `_controller_kernels` over unpacked rows.
        """
        rows, hidden_size = plans[0][0].shape
        count = len(plans)
        item = np.dtype(np.float32).itemsize
        gate = count * rows * hidden_size * item
        down_inputs, forgets, gains = scratch, scratch + 2 * gate, scratch + 3 * gate
        mids = out + gate
        form, blend = self._controller_direct
        form(inputs, messages, down_inputs, rows, hidden_size, count, eps)
        for i, (_, _, first, tails) in enumerate(plans):
            mid = mids + i * rows * first[1][0] * item
            self._launch_projection(down_inputs + i * 2 * rows * hidden_size * item, mid, rows, 2, first, None)
            self._launch_projection(mid, forgets + i * rows * hidden_size * item, rows, 2, tails[0], None)
            self._launch_projection(mid, gains + i * rows * hidden_size * item, rows, 2, tails[1], None)
        blend(inputs, messages, forgets, gains, weights, out, rows, hidden_size, count, eps)

    @contextmanager
    def resident(self, nbytes):
        """Hold the projection lock over a device arena of at least `nbytes`."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            yield self._scratch("_resident", nbytes).ptr

    def blend_controllers_resident(self, base, chains, norms, eps, hidden, message, *, rows, width):
        """Blend and normalize resident controller rows and leave the result in the caller's arena.

        `chains` are the down/forget/gain tensor names, `norms` are `(vector name, host row)` pairs,
        and `hidden`/`message` are device rows. The blended rows land at `base + inp_bytes` on the
        staging path, and after the chain's own scratch (4 gates) on the in-place path.
        """
        plans, norm_weights = self._controller_plans(
            [((down, forget, gain), np.empty((rows, width), np.float32), np.empty((rows, width), np.float32))
             for down, forget, gain in chains], [host for _, host in norms])
        if getattr(self, "_controller_direct", None) is not None:
            # Nothing is staged: the rows are read where they are and the arena holds only what the
            # chain writes (down inputs, gates, blended rows and gate values).
            gate = len(plans) * rows * width * np.dtype(np.float32).itemsize
            out = base + 4 * gate
            self._controller_kernels_direct(list(hidden), list(message), out, plans, eps,
                                            [self.resident_vector(name, host) for name, host in norms], base)
            return out
        inp_bytes, out_bytes = self._controller_extent(plans, norm_weights)
        item = np.dtype(np.float32).itemsize
        gate = len(plans) * rows * width * item
        out = base + inp_bytes
        for i in range(len(plans)):
            self.copy_device(base + i * gate, hidden[i], gate)
            self.copy_device(base + gate + i * gate, message[i], gate)
        cursor = base + 2 * gate
        for name, host in norms:
            self.copy_device(cursor, self.resident_vector(name, host), host.nbytes)
            cursor += host.nbytes
        self._controller_kernels(base, out, plans, norm_weights, eps)
        return out

    def download(self, src, host):
        """Read a resident region back to the host and fence preceding launches."""
        self._transfer.download(src, host)

    def copy_device(self, dst, src, nbytes):
        """Copy inside device memory on stream zero, after any launch already queued."""
        if nbytes > 0:
            self._runtime.memcpy_async(dst, src, nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, capture_stream())

    def project_from_device(self, name, rows, source):
        """Project rows that are already resident and return the result on the host."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            matrix = self._matrix(name, None)
            out = self._scratch("_output", rows * matrix[1][0] * np.dtype(np.float32).itemsize).ptr
            self._launch_projection(source, out, rows, 2, matrix, None)
            result = np.empty((rows, matrix[1][0]), dtype=np.float32)
            self._transfer.download(out, result)
            return result

    def resident_vector(self, name, host):
        """Upload an immutable coefficient row once and return its device pointer."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            if name not in self._vectors:
                self._vectors[name] = self._upload(np.ascontiguousarray(host, dtype=np.float32))
            return self._vectors[name].ptr

    def project_attention_resident(self, plan):
        """Run one block's attention half on the device and download only its branch output.

        Controllers, attention-input projections, convolution, attention and the output projection
        read and write one device arena; nothing is staged through the host in between. With
        `plan.device` the caller owns the arena, the inputs are already resident, and the branch
        output stays on the device.
        """
        rows, item = plan.rows, np.dtype(np.float32).itemsize
        device = getattr(plan, "device", None)
        inp_bytes = out_bytes = 0
        plans = norm_weights = staging = None
        if plan.controllers:
            plans, norm_weights = self._controller_plans(plan.controllers, plan.norm_weights)
            inp_bytes, out_bytes = self._controller_extent(plans, norm_weights)
            if device is None:
                staging = self._controller_staging(plans, norm_weights)
        else:
            inp_bytes = rows * plan.hidden * item * len(plan.sources)
        matrices = [[self._matrix(name, None) for name in names] for _, names, _ in plan.groups]
        output = self._matrix(plan.output[0], None)
        if device is not None:
            if plans is not None:
                direct = getattr(self, "_controller_direct", None)
                if direct is not None and getattr(device, "messages", None) is not None:
                    # Inputs, messages and norm weights are read where they are; the cursor holds the
                    # down inputs, gates, blended rows and gate values.
                    gate = len(plans) * rows * plan.hidden * item
                    out = device.cursor + 4 * gate
                    self._controller_kernels_direct(list(device.inputs), list(device.messages), out, plans, plan.eps,
                                                    device.norm_weights, device.cursor)
                    attention = out + out_bytes
                else:
                    weights = np.concatenate([np.ascontiguousarray(w, dtype=np.float32).ravel() for w in norm_weights])
                    self._transfer.enqueue_upload(device.cursor + 2 * len(plans) * rows * plan.hidden * item, weights)
                    self._controller_kernels(device.cursor, device.cursor + inp_bytes, plans, norm_weights, plan.eps)
                    out = device.cursor + inp_bytes
                    attention = out + out_bytes
                sources = [out + i * rows * plan.hidden * item for i in range(len(plans))]
            else:
                sources = list(device.sources)
                attention = device.cursor + inp_bytes + out_bytes
            self._attention_kernels(plan, matrices, sources, attention, output, device.result, rows, item)
            return None
        prep = plan.prepare
        query_bytes = rows * prep.nh * prep.dim * item
        kv_bytes = rows * prep.nkv * prep.dim * item
        out_bytes_total = rows * output[1][0] * item
        total = (inp_bytes + out_bytes
                 + sum(rows * matrix[1][0] * item for group in matrices for matrix in group)
                 + query_bytes + 2 * kv_bytes + rows * prep.nh * prep.rel_width * item
                 + query_bytes + out_bytes_total)
        with self.resident(total) as base:
            if plans is not None:
                self._transfer.enqueue_upload(base, staging)
                self._controller_kernels(base, base + inp_bytes, plans, norm_weights, plan.eps)
                sources = [base + inp_bytes + i * rows * plan.hidden * item for i in range(len(plans))]
            else:
                self._transfer.enqueue_upload(base, np.concatenate([np.asarray(s, dtype=np.float32).ravel()
                                                                    for s in plan.sources]))
                sources = [base + i * rows * plan.hidden * item for i in range(len(plan.sources))]
            result_ptr = self._attention_kernels(plan, matrices, sources, base + inp_bytes + out_bytes, output,
                                                 None, rows, item)
            result = np.empty((rows, output[1][0]), dtype=np.float32)
            self._transfer.download(result_ptr, result)
            return result

    def _attention_kernels(self, plan, matrices, sources, cursor, output, result_ptr, rows, item):
        """Project, clip, attend and project out, all inside one arena."""
        prep = plan.prepare
        query_bytes = rows * prep.nh * prep.dim * item
        kv_bytes = rows * prep.nkv * prep.dim * item
        alpha, floor = prep.logscale or (0.0, 1)
        projected = []
        for (index, _, consumers), group in zip(plan.groups, matrices, strict=True):
            src = sources[0 if index is None else index]
            for matrix, consume in zip(group, consumers, strict=True):
                self._launch_projection(src, cursor, rows, 2, matrix, None)
                if consume is not None:
                    consume(cursor, (rows, matrix[1][0]))
                projected.append(cursor)
                cursor += rows * matrix[1][0] * item
        if plan.convolve is not None:
            plan.convolve(projected)
        query = cursor
        key, value = query + query_bytes, query + query_bytes + kv_bytes
        rel = value + kv_bytes
        scaled = dict(first=plan.position + 1, target_arch=self._target_arch)
        if self._prepare_group is not None:
            # One launch for the four steps; each part is the separate kernel's arithmetic.
            self._prepare_group(projected[prep.query], query, prep.clip, prep.nh, prep.threshold,
                                alpha=alpha, floor=floor, logscale=prep.logscale is not None, **scaled,
                                key=projected[prep.key], key_out=key, key_heads=prep.nkv,
                                key_threshold=prep.threshold, value=projected[prep.value], value_out=value,
                                value_heads=prep.nkv, eps=prep.eps, relative=projected[prep.relative],
                                relative_out=rel, relative_scale=prep.rel_scale, relative_heads=prep.nh,
                                relative_threshold=prep.rel_threshold, relative_width=prep.rel_width,
                                dim=prep.dim, rows=rows)
        else:
            self._prepare_clip(projected[prep.query], query, prep.clip, rows, prep.dim, prep.nh, prep.threshold,
                               alpha=alpha, floor=floor, logscale=prep.logscale is not None, **scaled)
            self._prepare_clip(projected[prep.key], key, 0, rows, prep.dim, prep.nkv, prep.threshold,
                               logscale=False, **scaled)
            self._prepare_rms(projected[prep.value], value, rows, prep.dim, prep.nkv, prep.eps,
                              target_arch=self._target_arch)
            self._prepare_clip(projected[prep.relative], rel, prep.rel_scale, rows, prep.rel_width, prep.nh,
                               prep.rel_threshold, alpha=alpha, floor=floor, logscale=prep.logscale is not None,
                               **scaled)
        attended = rel + rows * prep.nh * prep.rel_width * item
        plan.attend(query, key, value, rel, projected[prep.sink], projected[prep.gate], attended)
        result = attended + query_bytes if result_ptr is None else result_ptr
        self._launch_projection(attended, result, rows, 2, output, None)
        if plan.output[1] is not None:
            plan.output[1](result, (rows, output[1][0]))
        return result

    def _bind_packed(self, quant, launch):
        if quant == Q.NVFP4:
            return launch

        def run(*args, **kwargs):
            if self._k_library is None:
                self._k_library = build_gguf_k_gemv(target_arch=self._target_arch)
            return launch(*args, library=self._k_library, runtime=self._runtime, **kwargs)
        return run

    def _packed_launch(self, quant):
        if quant not in self._bound_launches:
            self._bound_launches[quant] = self._bind_packed(quant, self._launches[quant])
        return self._bound_launches[quant]

    def _upload(self, array):
        host = np.ascontiguousarray(array)
        buf = malloc(host.nbytes, runtime=self._runtime)
        try:
            if isinstance(array, np.memmap) and host.nbytes > _UPLOAD_CHUNK_BYTES:
                if self._upload_staging is None:
                    self._upload_staging = np.empty(_UPLOAD_CHUNK_BYTES, dtype=np.uint8)
                raw = host.view(np.uint8).reshape(-1)
                for begin in range(0, raw.size, _UPLOAD_CHUNK_BYTES):
                    payload = raw[begin:begin + _UPLOAD_CHUNK_BYTES]
                    staging = self._upload_staging[:payload.size]
                    staging[:] = payload
                    self._runtime.memcpy(buf.ptr + begin, staging.ctypes.data, staging.nbytes,
                                         HipMemcpyKind.HOST_TO_DEVICE)
            else:
                self._runtime.memcpy(buf.ptr, host.ctypes.data, host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        except BaseException:
            free(buf, runtime=self._runtime)
            raise
        return buf

    def _matrix(self, name, limit, *, preserve_bf16=False):
        key = name, limit, False
        if key not in self._matrices:
            tensor = self._tensors[name]
            if len(tensor.shape) != 2 or (limit is not None and not 0 < limit <= tensor.shape[0]):
                raise ValueError(f"invalid Inkling GGUF matrix or row limit for {name}")
            rows = slice(limit)
            if tensor.ggml_type == Q.Q8_0:
                # Keep the logical width for scalar and verifier launches, but
                # upload the head's existing padding once for tiled decode.
                if name == self._head.name and limit == self._vocab and self._tiled_head_rows:
                    rows = slice(self._tiled_head_rows)
                data, packed = self._weights.raw(tensor, cache=False)[rows], Q.Q8_0
            else:
                data, packed = self._weights.decode(tensor, rows), False
            outputs = data.shape[0] if limit is None else limit
            self._matrices[key] = self._upload(data), (outputs, tensor.shape[1]), packed
        return self._matrices[key]

    def project_decode_head(self, hidden):
        """Share a packed head across independent rows and hide physical padding."""
        if hidden.ndim != 2 or not len(hidden) or hidden.shape[1] != self._head.shape[1]:
            raise ValueError("Inkling GGUF decode head expects a nonempty matrix of hidden rows")
        padded = self._tiled_head_rows
        if len(hidden) == 1 or not padded:
            return np.stack([self.project(self._head.name, row, limit=self._vocab, pinned=True) for row in hidden])
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            weight, (_, inputs), _ = self._matrix(self._head.name, self._vocab)
            x = np.ascontiguousarray(hidden, dtype=np.float32)
            y = np.empty((len(x), padded), np.float32)
            inp = self._scratch("_input", x.nbytes).ptr
            out = self._scratch("_output", y.nbytes).ptr
            with self._transfer.exchange(inp, out, x, y, pinned=True):
                self._q8_prefill(inp, weight.ptr, out, len(x), inputs, padded)
            return y[:, :self._vocab]

    def project_shared_attention_rows(self, inputs):
        """Project independent attention rows in one exchange, retaining row-local consumers."""
        plans = []
        locations = [[] for _ in inputs]
        for aligned in zip(*(groups for groups, _ in inputs), strict=True):
            names = aligned[0][0]
            if any(other != names or x.ndim != 1 for other, x in aligned):
                raise ValueError("Inkling GGUF shared attention requires matching vector projection groups")
            grouped, scalar = [], []
            index = []
            for name in names:
                tensor = self._tensors[name]
                share = self._q8_prefill is not None and tensor.ggml_type == Q.Q8_0 and tensor.shape[0] % 8 == 0
                target = grouped if share else scalar
                index.append((share, len(target)))
                target.append(name)
            common = len(plans)
            if grouped:
                plans.append((None, grouped, np.stack([x for _, x in aligned])))
            for row, (_, x) in enumerate(aligned):
                private = len(plans)
                if scalar:
                    plans.append((None, scalar, x))
                locations[row].append(tuple((common if share else private, j, row if share else None)
                                            for share, j in index))

        def transform_batch(resident):
            for (_, transforms), row in zip(inputs, locations, strict=True):
                if transforms is None:
                    continue
                for consume, group in zip(transforms, row, strict=True):
                    if consume is not None:
                        outputs = []
                        for i, j, idx in group:
                            ptr, shape = resident[i][j]
                            if idx is not None:
                                ptr += idx * shape[1] * 4
                                shape = shape[1:]
                            outputs.append((ptr, shape))
                        consume(*outputs)

        has_transforms = any(transforms is not None for _, transforms in inputs)
        outputs = self._project_groups(plans, None, True, transform_batch=transform_batch if has_transforms else None)
        return tuple(tuple(tuple(outputs[i][j] if idx is None else outputs[i][j][idx] for i, j, idx in group)
                           for group in row) for row in locations)

    def project_resident_attention_rows(self, name, width, inputs):
        """Project borrowed attention vectors, consume each row, and fence the result."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            matrix = self._matrix(name, None)
            _, (m, n), packed = matrix
            if not inputs or n != width:
                raise ValueError("Inkling GGUF resident attention output projection has invalid geometry")
            y = np.empty((len(inputs), m), dtype=np.float32)
            out = self._scratch("_output", y.nbytes).ptr
            if self._q8_prefill is not None and len(inputs) > 1 and packed == Q.Q8_0 and m % 8 == 0:
                row_bytes = n * y.itemsize
                inp = self._scratch("_input", len(inputs) * row_bytes).ptr
                for i, (ptr, _) in enumerate(inputs):
                    self._runtime.memcpy_async(inp + i * row_bytes, ptr, row_bytes, HipMemcpyKind.DEVICE_TO_DEVICE, capture_stream())
                self._launch_projection(inp, out, len(inputs), 2, matrix, None)
            else:
                for i, (ptr, _) in enumerate(inputs):
                    self._launch_projection(ptr, out + i * y.strides[0], 1, 1, matrix, None)
            for i, (_, consume) in enumerate(inputs):
                consume(out + i * y.strides[0], (m,))
            self._transfer.download(out, y)
            return y

    def _launch_wide_dense(self, src, weight, dst, rows, n, m):
        """Project through the wide tile, keeping each build of its library by the rows it serves."""
        profile = ("prefill" if gguf_q8_0_dense_wide._WGP_BUILD_MIN_ROWS <= rows
                   <= gguf_q8_0_dense_wide._WGP_BUILD_MAX_ROWS else "decode")
        if profile not in self._wide_libraries:
            self._wide_libraries[profile] = gguf_q8_0_dense_wide.build_gguf_q8_0_dense_wide(
                load=True, profile=profile)
        self._q8_wide(src, weight.ptr, dst, rows, n, m, library=self._wide_libraries[profile],
                      runtime=self._runtime, stream=capture_stream())

    def _launch_projection(self, src, dst, rows, ndim, matrix, native_bf16):
        weight, (m, n), packed = matrix
        if not packed:
            super()._launch_projection(src, dst, rows, ndim, matrix, native_bf16)
            return
        if packed == Q.Q8_0:
            if (self._q8_wide is not None and rows >= self._wide_min_rows[m < _WIDE_DENSE_WIDE_OUTPUTS]
                    and n % gguf_q8_0_dense_wide.K_TILE == 0):
                self._launch_wide_dense(src, weight, dst, rows, n, m)
                return
            if self._q8_prefill is not None and rows > 1 and m % 8 == 0:
                self._q8_prefill(src, weight.ptr, dst, rows, n, m)
                return
            if self._q8_decode is not None and rows == 1 and m >= _Q8_DECODE_MIN_OUTPUTS:
                aligned = m - m % 8
                self._q8_decode(src, weight.ptr, dst, rows, n, aligned)
                if aligned != m:
                    row_bytes = n // 32 * 34
                    self._packed_launch(packed)(src, weight.ptr + aligned * row_bytes, dst + 4 * aligned,
                                                rows, n, m - aligned)
                return
        self._packed_launch(packed)(src, weight.ptr, dst, rows, n, m)

    def _bank(self, bank):
        weight = bank.weight
        key = weight.name, None, "experts"
        if key not in self._matrices:
            scales = None if bank.scale is None else self._weights.decode(bank.scale)
            if scales is not None and not np.isfinite(scales).all():
                raise ValueError(f"non-finite Inkling GGUF expert scales for {weight.name}")
            packed = self._upload(self._weights.raw(weight, cache=False))
            self._matrices[key] = packed, weight.shape, weight.ggml_type
            self._bank_scales[key] = scales
        return self._matrices[key][0], self._bank_scales[key]


    # ----- Device-side route selection + expert projection ------------------

    def device_routes_enabled(self):
        if self._device_routes is None:
            # Device routes need the device split activation, which exists only with the fast expert
            # chain, so without that chain (the strict profile) the request is inert.
            self._device_routes = (self._device_routes_requested and self._activation is not None
                                   and self._mix is not None)
        return self._device_routes

    def device_routes_supported(self, visit):
        """True when this visit's expert banks fit the device-route projection."""
        if not self.device_routes_enabled():
            return False
        cached = self._device_route_support.get(visit.weight_layer) if hasattr(self, "_device_route_support") else None
        if cached is not None:
            return cached
        try:
            layer = self._weights.mapping.layers[visit.weight_layer]
            gate_q = Q(layer.gate_up.weight.ggml_type)
            down_q = Q(layer.down.weight.ggml_type)
            shape_ok = (layer.gate_up.weight.shape[1] == 2 * layer.down.weight.shape[2]
                        and layer.down.weight.shape[2] == 1152 and layer.down.weight.shape[1] == 2304
                        and layer.gate_up.weight.shape[1] == 2304 and layer.gate_up.weight.shape[2] == 2304)
            # Every bank runs the selected F32 kernels from device routes. The Q8-dot chain
            # moved individual logits by up to four nats and failed the section 2.11 envelope.
            selected = (gate_q in self._selected_launches and down_q in self._selected_launches
                        and self._activation is not None)
            ok = shape_ok and selected
        except (AttributeError, IndexError, ValueError):
            ok = False
        if not hasattr(self, "_device_route_support"):
            self._device_route_support = {}
        self._device_route_support[visit.weight_layer] = ok
        return ok

    def upload_scratch(self, name, host):
        """Stage a host array into a cached device scratch buffer."""
        array = np.ascontiguousarray(host, dtype=np.float32)
        buf = self._scratch(name, array.nbytes)
        self._transfer.enqueue_upload(buf.ptr, array)
        return buf.ptr

    def project_router_logits_ptr(self, name, rows, source):
        """Project router logits and keep them on the device."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF device projection residency is closed")
            matrix = self._matrix(name, None)
            outputs = matrix[1][0]
            out = self._scratch("_router_logits", rows * outputs * 4).ptr
            self._launch_projection(source, out, rows, 2, matrix, None)
            return int(out), outputs

    def _bank_scale_device(self, bank):
        key = bank.weight.name
        if key in self._route_scale_cache:
            return self._route_scale_cache[key]
        _, scales = self._bank(bank)
        if scales is None:
            self._route_scale_cache[key] = 0
            return 0
        scales = np.ascontiguousarray(np.asarray(scales, dtype=np.float32).ravel())
        buf = self._upload(scales)
        self._route_buffer_handles.append(buf)
        self._route_scale_cache[key] = buf.ptr
        return buf.ptr

    def _router_bias_ptr(self, layer_index, layer, experts):
        ptr = self._router_bias_cache.get(layer_index)
        if ptr is not None:
            return ptr
        bias = layer.tensors.get("exp_probs_b.bias")
        if bias is None:
            data = np.zeros(experts, dtype=np.float32)
        else:
            data = np.ascontiguousarray(self._weights.decode(bias), dtype=np.float32).ravel()
            if data.size != experts:
                raise ValueError(f"router bias size {data.size} != {experts} at layer {layer_index}")
        buf = self._upload(data)
        self._route_buffer_handles.append(buf)
        ptr = buf.ptr
        self._router_bias_cache[layer_index] = ptr
        return ptr

    def _route_buffers(self, rows, top_k, experts):
        state = self._route_state
        need = rows * top_k
        if state is not None and state["routes"] >= need and state["experts"] == experts:
            return state
        if state is not None:
            for buf in state["buffers"]:
                free(buf, runtime=self._runtime)
        capacity = max(need, 8)
        sizes = {
            "routing": rows * experts * 4,
            "selection": rows * experts * 4,
            "ids64": capacity * 8,
            "ids32": capacity * 4,
            "norm": capacity * 4,
            "scaled": capacity * 4,
            "gate_scales": capacity * 4,
            "down_scales": capacity * 4,
        }
        buffers = {}
        state = {"routes": capacity, "experts": experts, "buffers": []}
        for name, nbytes in sizes.items():
            buf = malloc(nbytes, runtime=self._runtime)
            buffers[name] = buf
            state["buffers"].append(buf)
        state.update({name: buf.ptr for name, buf in buffers.items()})
        self._route_state = state
        return state

    def device_routes(self, visit, logits_ptr, rows=1):
        """Select routes on the device; returns pointers + route metadata."""
        layer = self._weights.mapping.layers[visit.weight_layer]
        experts = int(layer.gate_up.weight.shape[0])
        top_k = int(layer.top_k)
        routes = rows * top_k
        if routes > _MAX_SELECTED_ROUTES:
            raise ValueError(f"device routes exceed {_MAX_SELECTED_ROUTES} selected routes")
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF device projection residency is closed")
            state = self._route_buffers(rows, top_k, experts)
            bias_ptr = self._router_bias_ptr(visit.weight_layer, layer, experts)
            laguna_sigmoid_correction_topk_f32(
                int(logits_ptr), int(bias_ptr), state["routing"], state["selection"],
                state["ids64"], state["norm"], state["scaled"],
                int(rows), experts, top_k, float(self._weights.mapping.route_scale), stream=capture_stream())
            gate_table = self._bank_scale_device(layer.gate_up)
            down_table = self._bank_scale_device(layer.down)
            launch_route_ids_i32(state["ids64"], routes, gate_table or None,
                                 state["gate_scales"] if gate_table else None,
                                 state["ids32"], stream=capture_stream())
            launch_route_ids_i32(state["ids64"], routes, down_table or None,
                                 state["down_scales"] if down_table else None,
                                 state["ids32"], stream=capture_stream())
            return {"ids": state["ids32"], "mix": state["scaled"],
                    "gate_scales": state["gate_scales"] if gate_table else 0,
                    "down_scales": state["down_scales"] if down_table else 0,
                    "top_k": top_k, "experts": experts, "routes": routes}

    def project_experts_device_routes(self, visit, *, hidden_ptr, route, out_ptr=None,
                                      transform_output=None):
        """Run the selected-expert chain from device-resident route pointers."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF device projection residency is closed")
            layer = self._weights.mapping.layers[visit.weight_layer]
            mapping = self._weights.mapping
            gate, down = layer.gate_up, layer.down
            gate_q, down_q = Q(gate.weight.ggml_type), Q(down.weight.ggml_type)
            gate_weight, _ = self._bank(gate)
            down_weight, _ = self._bank(down)
            experts, gm, gn = gate.weight.shape
            de, dm, dn = down.weight.shape
            if not (gm == 2 * dn and dm == gm and de == experts):
                raise ValueError("device route projection bank geometry mismatch: "
                                 f"gate {gate.weight.shape} down {down.weight.shape}")
            ids_ptr, mix_ptr = int(route["ids"]), int(route["mix"])
            routes, k = int(route["routes"]), int(route["top_k"])
            if out_ptr is None:
                out_ptr = self._scratch("_device_route_out", routes // k * dm * 4).ptr
            out_ptr = int(out_ptr)
            interim = self._scratch("_device_route_interim", routes * gm * 4 + routes * dn * 4).ptr
            if gate_q in self._selected_launches and down_q in self._selected_launches:
                if self._activation is None:
                    raise RuntimeError("device route projection requires the split activation")
                # The selected kernels read one src row per route. With the shared-input projection every
                # route of a token reads that token's resident hidden row; other banks get the row written
                # once per route by one launch. Each takes the per-route outer scales.
                routes_per_input = 1
                shared = getattr(self, "_shared_selected_launches", {}).get(gate_q)
                if shared is not None:
                    hidden_rep = int(hidden_ptr)
                    routes_per_input = k
                else:
                    hidden_rep = self._scratch("_device_route_hidden_rep", routes * gn * 4).ptr
                    if getattr(self, "_replicate", None) is None:
                        for i in range(routes):
                            self.copy_device(hidden_rep + i * gn * 4, int(hidden_ptr), gn * 4)
                    else:
                        self._replicate(int(hidden_ptr), hidden_rep, gn, routes)
                active = interim + routes * gm * 4
                gate_src = (hidden_rep, gate_weight.ptr, interim, ids_ptr, int(route["gate_scales"] or 0),
                            routes, experts, gn, gm)
                if shared is not None:
                    shared(*gate_src, routes_per_input=routes_per_input)
                else:
                    self._selected_launches[gate_q](*gate_src)
                self._activation(interim, active, routes, dn, mapping.gate_cap, mapping.up_cap)
                self._selected_launches[down_q](active, down_weight.ptr, interim, ids_ptr,
                                                int(route["down_scales"] or 0), routes, experts, dn, dm)
            else:
                raise ValueError(f"device route projection does not support {gate_q.name} gate/up and "
                                 f"{down_q.name} down expert banks")
            self._mix(interim, mix_ptr, out_ptr, k, dm, tokens=routes // k)
            if transform_output is not None:
                rows = routes // k
                transform_output(int(out_ptr), (dm,) if rows == 1 else (rows, dm))
            return int(out_ptr)

    def close(self):
        try:
            super().close()
        finally:
            self._upload_staging = None
            with self._lock:
                for name in ("_q8_dot_scratch", "_resident", "_wmma_arena", "_wmma_down", "_device_route_hidden",
                         "_device_route_hidden_rep", "_router_logits",
                         "_device_route_interim", "_device_route_out"):
                    buffer = getattr(self, name)
                    if buffer is not None:
                        free(buffer, runtime=self._runtime)
                        setattr(self, name, None)
                if getattr(self, "_route_state", None) is not None:
                    for buffer in self._route_state["buffers"]:
                        free(buffer, runtime=self._runtime)
                    self._route_state = None
                if self._schedule_state is not None:
                    for buffer in self._schedule_state["buffers"]:
                        free(buffer, runtime=self._runtime)
                    self._schedule_state = None
                for buffer in getattr(self, "_route_buffer_handles", ()):
                    free(buffer, runtime=self._runtime)
                self._route_buffer_handles = []
                for buffer in self._vectors.values():
                    free(buffer, runtime=self._runtime)
                self._vectors.clear()

    @property
    def packed_bank_bytes(self):
        return sum(self._matrices[key][0].nbytes for key in self._bank_scales if key in self._matrices)

    def project_selected(self, visit, projection, expert_ids, hidden, *, single_token=False, routes_per_input=1):
        """Project selected routes in order, optionally sharing one FP32 row per token."""
        layer = self._weights.mapping.layers[visit.weight_layer]
        bank = {"gate_up": layer.gate_up, "down": layer.down}[projection]
        experts, m, n = bank.weight.shape
        ids = np.asarray(expert_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or np.any(ids < 0) or np.any(ids >= experts)
                or routes_per_input <= 0 or ids.size % routes_per_input
                or hidden.shape != (len(ids) // routes_per_input, n)):
            raise ValueError("invalid Inkling GGUF selected projection geometry")
        quant = Q(bank.weight.ggml_type)
        selected = self._selected_launches.get(quant) if len(ids) <= _MAX_SELECTED_ROUTES else None
        if selected is not None:
            ids = np.ascontiguousarray(ids, dtype=np.int32)
            with self._lock:
                if self._closed:
                    raise RuntimeError("Inkling GGUF projection residency is closed")
                weight, scales = self._bank(bank)
                return self._project_routes(quant, selected, weight.ptr, scales, ids, hidden, experts, m, n, single_token,
                                            routes_per_input)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        x = np.ascontiguousarray(hidden[order // routes_per_input], dtype=np.float32)
        y = np.empty((len(ids), m), np.float32)
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            weight, scales = self._bank(bank)
            launch = self._packed_launch(quant)
            inp = self._scratch("_input", x.nbytes)
            out = self._scratch("_output", y.nbytes)
            stride = bank.weight.nbytes // experts
            with self._transfer.exchange(inp.ptr, out.ptr, x, y, pinned=True):
                for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                    expert = int(ids[start])
                    scale = {} if quant != Q.NVFP4 or scales is None else {"outer_scale": float(scales[expert])}
                    launch(inp.ptr + int(start) * n * x.itemsize, weight.ptr + expert * stride,
                           out.ptr + int(start) * m * y.itemsize, int(stop - start), n, m, **scale)
            if scales is not None and quant != Q.NVFP4:
                y *= scales[ids, None]
        result = np.empty_like(y)
        result[order] = y
        return result

    def _route_tiles(self, quant, ids):
        if quant not in self._tiled_launches or not 16 <= ids.size <= _MAX_SELECTED_ROUTES:
            return None
        tiles = build_expert_route_tiles(ids, 4)
        return tiles if len(tiles) < ids.size else None

    @staticmethod
    def _stage_routes(ids, hidden, scales, extra=0, tiles=None):
        # Activations, route metadata and any mix coefficients share one upload.
        count = 0 if hidden is None else hidden.size
        records = ids if tiles is None else tiles.ravel()
        scales_start = count + records.size
        x = np.empty(scales_start + (ids.size if scales is not None else 0) + extra, np.float32)
        if hidden is not None:
            x[:count].reshape(hidden.shape)[...] = hidden
        x[count:scales_start].view(np.int32)[:] = records
        if scales is not None:
            x[scales_start:scales_start + ids.size] = scales[ids]
        return x, records.nbytes

    def _q8_scratch_ptr(self, values):
        # Scratch is owned by the projection lock and is not transferred.
        return self._scratch("_q8_dot_scratch", values + values // 32 * 4).ptr

    def _run_selected(self, quant, launch, src, weight, out, ids, scales, routes, experts, n, m, *, single_token):
        if (self._q8_dot is not None and quant == Q.NVFP4 and single_token and routes == 8
                and n in (1152, 2304) and m == 2304):
            self._q8_dot(src, weight, out, ids, scales, routes, experts, n, m,
                         scratch=self._q8_scratch_ptr(routes * n))
        else:
            launch(src, weight, out, ids, scales, routes, experts, n, m)

    def _route_input(self, quant, hidden, tiles, repeats):
        if repeats > 1 and (tiles is None or quant not in self._shared_tiled_launches):
            return np.repeat(hidden, repeats, axis=0), 1
        return hidden, repeats

    def _project_routes(self, quant, launch, weight, scales, ids, hidden, experts, m, n, single_token, repeats):
        tiles = self._route_tiles(quant, ids)
        hidden, repeats = self._route_input(quant, hidden, tiles, repeats)
        x, record_bytes = self._stage_routes(ids, hidden, scales, tiles=tiles)
        y = np.empty((ids.size, m), np.float32)
        inp = self._scratch("_input", x.nbytes)
        out = self._scratch("_output", y.nbytes)
        ids_ptr = inp.ptr + hidden.size * x.itemsize
        scales_ptr = ids_ptr + record_bytes if scales is not None else 0
        with self._transfer.exchange(inp.ptr, out.ptr, x, y, pinned=True):
            self._launch_routes(quant, launch, inp.ptr, weight, out.ptr, ids_ptr, scales_ptr, ids.size,
                                experts, n, m, tiles, single_token, routes_per_input=repeats)
        return y

    def _launch_routes(self, quant, launch, src, weight, out, ids_ptr, scales_ptr, routes, experts, n, m,
                       tiles, single_token, *, routes_per_input=1):
        if tiles is None:
            self._run_selected(quant, launch, src, weight, out, ids_ptr, scales_ptr, routes, experts, n, m,
                               single_token=single_token)
        elif routes_per_input > 1:
            self._shared_tiled_launches[quant](src, weight, out, ids_ptr, scales_ptr, len(tiles), routes, experts, n, m,
                                               routes_per_input=routes_per_input)
        else:
            self._tiled_launches[quant](src, weight, out, ids_ptr, scales_ptr, len(tiles), routes, experts, n, m)

    def project_experts_mixed(self, visit, ids, mix, hidden, *, transform_output=None, device=None):
        return self.project_experts_mixed_rows(visit, ((ids, mix, hidden, transform_output),), device=device)[0]

    def project_experts_mixed_rows(self, visit, inputs, *, device=None):
        """Share an exchange and the gate weights where the scalar arithmetic matches.

        With `device=(hidden, result)` the routed rows are already resident and the mixed
        projection is left at `result`; nothing crosses the host in either direction.
        """
        if self._activation is None:
            raise RuntimeError("Inkling GGUF split expert activation is not enabled")
        if not inputs:
            return ()
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            layer = self._weights.mapping.layers[visit.weight_layer]
            bank = layer.gate_up
            _, gm, gn = bank.weight.shape
            counted = (self._counted_q8_dot is not None and 2 <= len(inputs) <= 4
                       and bank.weight.ggml_type == Q.NVFP4 and gm == 2304 and gn in (1152, 2304)
                       and all(ids.shape == (8,) for ids, *_ in inputs))
            counted_down = (counted and self._counted_split_q8_dot is not None
                            and layer.down.weight.ggml_type == Q.NVFP4 and layer.down.weight.shape[1:] == (2304, 1152))
            if self._wmma_rows(visit, inputs):
                return self._project_experts_wmma(visit, inputs[0], device=device)
            plans = [self._plan_expert_chain(visit, ids, mix, hidden, transform, counted=counted, counted_down=counted_down)
                     for ids, mix, hidden, transform in inputs]
            payload = [data for parts, *_ in plans for data in parts]
            gate_row_bytes = gate_input_bytes = pack_values = 0
            if counted:
                gate_data, launch_counted = self._plan_counted_gate(bank, inputs, down=layer.down if counted_down else None)
                payload.insert(0, gate_data)
                gate_input_bytes = gate_data.nbytes
                gate_row_bytes = 8 * gm * 4
                pack_values = len(inputs) * (8 * 1152 if counted_down else gn)
            packed = np.concatenate(payload)
            result = np.empty(sum(size for _, _, size, *_ in plans), np.float32)
            if pack_values := max(pack_values, max(plan[4] for plan in plans)):
                self._q8_scratch_ptr(pack_values)
            if device is not None:
                quant = Q(bank.weight.ggml_type)
                if len(inputs) != 1 or self._route_tiles(quant, np.ascontiguousarray(inputs[0][0].ravel())) is None \
                        or quant not in self._shared_tiled_launches:
                    raise ValueError("Inkling GGUF device expert input needs one shared-tile route set")
            inp = self._scratch("_input", packed.nbytes).ptr
            temp_bytes = max(plan[3] for plan in plans)
            slab_bytes = len(plans) * gate_row_bytes
            out = self._scratch("_output", result.nbytes + temp_bytes + slab_bytes * (2 if counted_down else 1)).ptr
            interim = out + result.nbytes
            gate = interim + temp_bytes if counted else 0
            down_out = gate + slab_bytes if counted_down else 0
            projection = down_out or gate
            results = []
            input_offset, output_offset = gate_input_bytes, 0
            if device is None:
                self._transfer.enqueue_upload(inp, packed)
            else:
                hidden_ptr, result_ptr = device
                staging = packed.copy()
                staging[:inputs[0][2].size] = 0.0
                self._transfer.enqueue_upload(inp, staging)
                self.copy_device(inp, hidden_ptr, inputs[0][2].nbytes)
            if counted:
                launch_counted(inp, gate, down_out)
            for parts, shape, size, _, _, launch in plans:
                launch(inp + input_offset, out + output_offset * result.itemsize, interim, projection)
                results.append(result[output_offset:output_offset + size].reshape(shape))
                input_offset += sum(part.nbytes for part in parts)
                output_offset += size
                projection += gate_row_bytes
            if device is not None:
                self.copy_device(device[1], out, result.nbytes)
                return ()
            self._transfer.download(out, result)
            return tuple(results)

    def _wmma_banks(self, visit):
        """True when the grouped kernel reads both of this visit's expert banks.

        NVFP4 is read in one FP16 part; the Q5_K and Q8_0 banks of a mixed file's last layers in a
        high and a residual part.
        """
        layer = self._weights.mapping.layers[visit.weight_layer]
        return (Q(layer.gate_up.weight.ggml_type) in self._wmma_by_quant
                and Q(layer.down.weight.ggml_type) in self._wmma_by_quant)

    def _wmma_rows(self, visit, inputs):
        """One prompt block of at least 192 rows, on banks the grouped kernel reads."""
        if self._wmma is None or len(inputs) != 1:
            return False
        ids = inputs[0][0]
        return ids.ndim == 2 and ids.shape[0] >= _WMMA_MIN_ROWS and self._wmma_banks(visit)

    def _project_experts_wmma(self, visit, item, *, device):
        """Project one prompt block through the grouped WMMA kernels.

        The block's routes are grouped by expert on the host. Gate/up reads each token row once per
        route over the grouped routes, the activation runs on the grouped rows, and down writes
        route-ordered rows for the ordered mix. Activations are FP16 in two parts, so the arithmetic
        differs from the FP32 tile kernels.
        """
        ids, mix, hidden, transform_output = item
        mapping = self._weights.mapping
        layer = mapping.layers[visit.weight_layer]
        gate, down = layer.gate_up, layer.down
        ge, gm, gn = gate.weight.shape
        de, dm, dn = down.weight.shape
        if (ids.ndim != 2 or not 0 < ids.size <= _MAX_SELECTED_ROUTES or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= min(ge, de)) or mix.shape != ids.shape
                or hidden.shape != (ids.shape[0], gn) or gm != 2 * dn
                or gn % 64 or dn % 64 or gm % 32 or dm % 32):
            raise ValueError("invalid Inkling GGUF grouped WMMA expert projection geometry")
        experts = min(ge, de)
        flat = np.ascontiguousarray(ids.ravel(), dtype=np.int32)
        tokens, k = ids.shape
        routes = flat.size
        order = np.argsort(flat, kind="stable").astype(np.int32)
        count = np.bincount(flat, minlength=experts)
        start = np.r_[0, np.cumsum(count)].astype(np.int64)
        group_expert = np.repeat(np.arange(experts, dtype=np.int32), -(-count // _WMMA_GROUP_ROWS))
        group_first = np.concatenate([np.arange(0, int(c), _WMMA_GROUP_ROWS, dtype=np.int32) for c in count])
        source = (order // k).astype(np.int32)
        gate_weight, gate_scales = self._bank(gate)
        down_weight, down_scales = self._bank(down)
        # One upload carries the schedule and the scales after the token rows. The int64 expert
        # starts follow rows whose size is a multiple of eight bytes.
        scale_values = [None if values is None else np.ascontiguousarray(values, dtype=np.float32)
                        for values in (gate_scales, down_scales)]
        schedule = [("start", start), ("group", group_expert), ("first", group_first), ("source", source),
                    ("route", order), ("gate_scale", scale_values[0]), ("down_scale", scale_values[1]),
                    ("mix", np.ascontiguousarray(mix, dtype=np.float32).ravel())]
        schedule = [(name, values) for name, values in schedule if values is not None]
        rows = None if device is not None else np.ascontiguousarray(hidden, dtype=np.float32)
        rows_bytes = 0 if rows is None else rows.nbytes
        packed = np.empty(rows_bytes + sum(values.nbytes for _, values in schedule), np.uint8)
        if rows is not None:
            packed[:rows_bytes] = rows.view(np.uint8).reshape(-1)
        at, cursor = {}, rows_bytes
        for name, values in schedule:
            at[name] = cursor
            packed[cursor:cursor + values.nbytes] = values.view(np.uint8)
            cursor += values.nbytes
        result = np.empty((tokens, dm), np.float32)
        inp = self._scratch("_input", packed.nbytes).ptr
        out = self._scratch("_output", result.nbytes + routes * dm * 4).ptr
        schedule = {"start": inp + at["start"], "group": inp + at["group"], "first": inp + at["first"],
                    "source": inp + at["source"], "route": inp + at["route"],
                    "gate_scale": inp + at["gate_scale"] if "gate_scale" in at else 0,
                    "down_scale": inp + at["down_scale"] if "down_scale" in at else 0,
                    "mix": inp + at["mix"], "down": out + result.nbytes, "groups": len(group_expert)}
        geometry = dict(tokens=tokens, routes=routes, k=k, gn=gn, gm=gm, dn=dn, dm=dm,
                        gate_weight=gate_weight.ptr, down_weight=down_weight.ptr,
                        gate_quant=Q(gate.weight.ggml_type), down_quant=Q(down.weight.ggml_type))
        if device is None:
            self._transfer.enqueue_upload(inp, packed)
            self._run_wmma(schedule, x_ptr=inp, out_ptr=out, transform_output=transform_output, **geometry)
            self._transfer.download(out, result)
            return (result,)
        hidden_ptr, result_ptr = device
        self._transfer.enqueue_upload(inp, packed[rows_bytes:])
        self._run_wmma(schedule, x_ptr=hidden_ptr, out_ptr=out, transform_output=transform_output, **geometry)
        self.copy_device(result_ptr, out, result.nbytes)
        return ()

    def _run_wmma(self, schedule, *, tokens, routes, k, gn, gm, dn, dm, gate_weight, down_weight, gate_quant,
                  down_quant, x_ptr, out_ptr, transform_output=None):
        """Launch the grouped projection: FP16 parts, gate/up, the activation, down, the ordered mix."""
        mapping = self._weights.mapping
        split = self._wmma[1]
        gate_wmma, down_wmma = self._wmma_by_quant[gate_quant], self._wmma_by_quant[down_quant]
        parts = self._wmma_parts
        itemsize, half = np.dtype(np.float32).itemsize, np.dtype(np.float16).itemsize
        x_bytes = tokens * gn * half
        gate_bytes, act_bytes, act_half = routes * gm * itemsize, routes * dn * itemsize, routes * dn * half
        arena = self._scratch("_wmma_arena", parts * x_bytes + gate_bytes + act_bytes + parts * act_half).ptr
        x_hi, x_lo = arena, arena + x_bytes
        gate_out = arena + parts * x_bytes
        act, act_hi = gate_out + gate_bytes, gate_out + gate_bytes + act_bytes
        act_lo = act_hi + act_half
        # A schedule built on the device brings its group count.
        count = schedule.get("count")
        split(x_ptr, x_hi, x_lo, tokens * gn, parts)
        gate_wmma(x_hi, x_lo, schedule["start"], schedule["group"], schedule["first"], schedule["source"], 0,
                  schedule["gate_scale"], gate_weight, gate_out, schedule["groups"], gn, gm, group_count=count)
        self._activation(gate_out, act, routes, dn, mapping.gate_cap, mapping.up_cap)
        split(act, act_hi, act_lo, routes * dn, parts)
        down_wmma(act_hi, act_lo, schedule["start"], schedule["group"], schedule["first"], 0, schedule["route"],
                  schedule["down_scale"], down_weight, schedule["down"], schedule["groups"], dn, dm,
                  group_count=count)
        self._mix(schedule["down"], schedule["mix"], out_ptr, k, dm, tokens=tokens)
        if transform_output is not None:
            transform_output(out_ptr, (tokens, dm))

    def block_device_routes_supported(self, visit, rows):
        """True when a prompt block can select its routes and build its schedule on the device."""
        return (self._wmma is not None and rows >= _WMMA_MIN_ROWS and self.device_routes_enabled()
                and self.device_routes_supported(visit) and self._wmma_banks(visit))

    def project_experts_block_device_routes(self, visit, *, rows, router, hidden_ptr, out_ptr, transform_output=None):
        """Project one prompt block from device-resident routes; nothing crosses the host.

        The router logits are projected from the resident normalized rows, the routes are selected
        on the device (:meth:`device_routes`) and the grouped schedule is built there, so the stage
        has no download of the block and no upload of a schedule. The schedule puts a row's routes
        in expert order for the ordered mix, as the host-routed block does.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF device projection residency is closed")
            layer = self._weights.mapping.layers[visit.weight_layer]
            gate, down = layer.gate_up, layer.down
            ge, gm, gn = gate.weight.shape
            de, dm, dn = down.weight.shape
            gate_q, down_q = Q(gate.weight.ggml_type), Q(down.weight.ggml_type)
            if (self._wmma is None or rows < _WMMA_MIN_ROWS or not (gm == 2 * dn and dm == gn and de == ge)
                    or gn % 64 or dn % 64 or gm % 32 or dm % 32
                    or gate_q not in self._wmma_by_quant or down_q not in self._wmma_by_quant):
                raise ValueError("Inkling GGUF device-routed expert blocks need the grouped WMMA projection "
                                 f"of NVFP4, Q5_K or Q8_0 experts ({gate_q.name} gate/up, {down_q.name} down) "
                                 f"for {rows} rows")
            gate_weight, _ = self._bank(gate)
            down_weight, _ = self._bank(down)
            logits_ptr, _ = self.project_router_logits_ptr(router, rows, hidden_ptr)
            route = self.device_routes(visit, logits_ptr, rows)
            grouped = self._device_schedule(route)
            schedule = {"start": grouped["start"], "group": grouped["group"], "first": grouped["first"],
                        "source": grouped["source"], "route": grouped["route"], "mix": grouped["mix"],
                        "gate_scale": self._bank_scale_device(gate), "down_scale": self._bank_scale_device(down),
                        "down": self._scratch("_wmma_down", route["routes"] * dm * 4).ptr,
                        "groups": grouped["capacity"], "count": grouped["total"]}
            self._run_wmma(schedule, tokens=rows, routes=route["routes"], k=route["top_k"], gn=gn, gm=gm, dn=dn,
                           dm=dm, gate_weight=gate_weight.ptr, down_weight=down_weight.ptr, gate_quant=gate_q,
                           down_quant=down_q, x_ptr=int(hidden_ptr), out_ptr=int(out_ptr),
                           transform_output=transform_output)
            return int(out_ptr)

    def _device_schedule(self, route):
        """Group the block's routes by expert on the device; returns the schedule's device pointers.

        The grid of the grouped projection has a fixed height: no split can need more groups than
        ``min(experts, routes) + ceil(routes / 64)``, and the kernel skips the groups beyond the count.
        """
        experts, routes, top = route["experts"], route["routes"], route["top_k"]
        state = getattr(self, "_schedule_state", None)
        capacity = min(experts, routes) + -(-routes // _WMMA_GROUP_ROWS)
        if state is not None and (state["routes"] < routes or state["experts"] != experts
                                  or state["capacity"] < capacity):
            for buffer in state["buffers"]:
                free(buffer, runtime=self._runtime)
            state = None
        if state is None:
            buffers = {name: malloc(nbytes, runtime=self._runtime) for name, nbytes in
                       (("start", (experts + 1) * 8), ("group", capacity * 4), ("first", capacity * 4),
                        ("source", routes * 4), ("route", routes * 4), ("mix", routes * 4), ("total", 4))}
            state = {"routes": routes, "experts": experts, "capacity": capacity,
                     "buffers": list(buffers.values())}
            state.update({name: buffer.ptr for name, buffer in buffers.items()})
            self._schedule_state = state
        launch_route_schedule(route["ids"], route["mix"], routes, experts, top, state["start"], state["group"],
                              state["first"], state["source"], state["route"], state["mix"], state["total"],
                              capacity, target_arch=self._target_arch, stream=capture_stream())
        return state

    def _plan_counted_gate(self, bank, inputs, *, down=None):
        weight, scales = self._bank(bank)
        experts, m, n = bank.weight.shape
        ids = np.concatenate([item[0] for item in inputs], dtype=np.int32)
        hidden = np.stack([item[2] for item in inputs])
        tiles = build_expert_route_tiles(ids, 4)
        down_scales = None
        if down is not None:
            down_weight, down_scales = self._bank(down)
            de, dm, dn = down.weight.shape
            caps = self._weights.mapping.gate_cap, self._weights.mapping.up_cap
        data, rec_bytes = self._stage_routes(ids, hidden, scales, tiles=tiles,
                                             extra=ids.size if down_scales is not None else 0)
        if down_scales is not None:
            data[-ids.size:] = down_scales[ids]

        def launch(inp, out, down_out):
            records = inp + hidden.size * 4
            outer = records + rec_bytes if scales is not None else 0
            self._counted_q8_dot(inp, weight.ptr, out, records, outer, len(tiles), len(inputs), experts, n, m,
                                 scratch=self._q8_scratch_ptr(hidden.size))
            if down is not None:
                down_outer = inp + data.nbytes - ids.nbytes if down_scales is not None else 0
                self._counted_split_q8_dot(out, down_weight.ptr, down_out, records, down_outer,
                                           len(tiles), len(inputs), de, dn, dm, *caps,
                                           scratch=self._q8_scratch_ptr(ids.size * dn))

        return data, launch

    def _plan_expert_chain(self, visit, ids, mix, hidden, transform_output, *, counted=False, counted_down=False):
        mapping = self._weights.mapping
        layer = mapping.layers[visit.weight_layer]
        gate, down = layer.gate_up, layer.down
        ge, gm, gn = gate.weight.shape
        de, dm, dn = down.weight.shape
        if (ids.ndim not in (1, 2) or not 0 < ids.size <= _MAX_SELECTED_ROUTES or ids.dtype.kind not in "iu"
                or gm != 2 * dn or np.any(ids < 0) or np.any(ids >= min(ge, de))
                or mix.shape != ids.shape or hidden.shape != (*ids.shape[:-1], gn)):
            raise ValueError("invalid Inkling GGUF fused expert projection geometry")
        flat_ids = np.ascontiguousarray(ids.ravel(), dtype=np.int32)
        routes, k = ids.size, ids.shape[-1]
        shape = (*ids.shape[:-1], dm)
        if counted_down:
            def mix_projected(inp, out, interim, projected):
                self._mix(projected, inp, out, k, dm, tokens=routes // k)
                if transform_output is not None:
                    transform_output(out, shape)

            return ((np.ascontiguousarray(mix, dtype=np.float32),), shape, (routes // k) * dm, 0, 0, mix_projected)
        itemsize = np.dtype(np.float32).itemsize
        single_top8 = ids.ndim == 1 and routes == 8
        shared_q8 = (self._shared_q8_dot is not None and single_top8 and gn in (1152, 2304) and gm == 2304
                     and gate.weight.ggml_type == Q.NVFP4)
        split_q8 = (self._split_q8_dot is not None and single_top8 and dn == 1152 and dm == 2304
                    and down.weight.ggml_type == Q.NVFP4)

        def prepare(bank, x, *, repeats=1, extra=0, fused=False, shared=False):
            quant = Q(bank.weight.ggml_type)
            weight, scales = self._bank(bank)
            tiles = self._route_tiles(quant, flat_ids)
            if not shared:
                x, repeats = self._route_input(quant, x, tiles, repeats)
            data, rec_bytes = self._stage_routes(flat_ids, x, scales, extra=extra, tiles=tiles)
            input_bytes = 0 if x is None else x.size * itemsize
            experts, m, n = bank.weight.shape

            def launch(src, dst, uploaded):
                metadata = uploaded + input_bytes
                outer = metadata + rec_bytes if scales is not None else 0
                if shared:
                    self._shared_q8_dot(src, weight.ptr, dst, metadata, outer, routes, experts, n, m,
                                        scratch=self._q8_scratch_ptr(n))
                elif fused:
                    self._split_q8_dot(src, weight.ptr, dst, metadata, outer, routes, experts, n, m,
                                       mapping.gate_cap, mapping.up_cap, scratch=self._q8_scratch_ptr(routes * n))
                else:
                    self._launch_routes(quant, self._selected_launches[quant], src, weight.ptr, dst, metadata, outer,
                                        routes, experts, n, m, tiles, ids.ndim == 1, routes_per_input=repeats)

            return data, launch

        if not counted:
            gate_data, launch_gate = prepare(gate, np.atleast_2d(hidden), repeats=k, shared=shared_q8)
        down_data, launch_down = prepare(down, None, extra=routes, fused=split_q8)
        down_data[-routes:] = mix.ravel()
        # Reserve one maximum before any launch: it stays live across the
        # exchange, and stream order makes interim storage reusable.
        gate_pack = gn if shared_q8 else (
            routes * gn if self._q8_dot is not None and single_top8 and gn in (1152, 2304) and gm == 2304
            and gate.weight.ggml_type == Q.NVFP4 else 0)
        down_pack = (routes * dn if split_q8 or (
            self._q8_dot is not None and single_top8 and dn in (1152, 2304) and dm == 2304
            and down.weight.ggml_type == Q.NVFP4) else 0)
        intermediate_bytes = routes * max(gm, dm) * itemsize
        activation_bytes = 0 if split_q8 else routes * dn * itemsize
        gate_bytes = 0 if counted else gate_data.nbytes
        mix_offset = gate_bytes + down_data.nbytes - routes * itemsize

        def launch(inp, out, interim, gate_ptr):
            if not counted:
                launch_gate(inp, interim, inp)
                gate_ptr = interim
            active = gate_ptr
            if not split_q8:
                active = interim + intermediate_bytes
                self._activation(gate_ptr, active, routes, dn, mapping.gate_cap, mapping.up_cap)
            launch_down(active, interim, inp + gate_bytes)
            self._mix(interim, inp + mix_offset, out, k, dm, tokens=routes // k)
            if transform_output is not None:
                transform_output(out, shape)

        parts = (down_data,) if counted else (gate_data, down_data)
        return (parts, shape, (routes // k) * dm, intermediate_bytes + activation_bytes,
                max(0 if counted else gate_pack, down_pack), launch)

    def project_down_mixed(self, visit, ids, mix, activated, *, transform_output=None):
        """Keep the selected down projection, ordered mix and optional consumer on device."""
        bank = self._weights.mapping.layers[visit.weight_layer].down
        experts, m, n = bank.weight.shape
        if (ids.ndim not in (1, 2) or not ids.size or not 1 <= ids.shape[-1] <= _MAX_SELECTED_ROUTES
                or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= experts)
                or mix.shape != ids.shape or activated.shape != (ids.size, n)):
            raise ValueError("invalid Inkling GGUF mixed down projection geometry")
        k = ids.shape[-1]
        flat_ids = np.ascontiguousarray(ids.ravel(), dtype=np.int32)
        tokens = ids.size // k
        tile = _MAX_SELECTED_ROUTES // k
        result = np.empty((*ids.shape[:-1], m), dtype=np.float32)
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            weight, scales = self._bank(bank)
            quant = Q(bank.weight.ggml_type)
            launch = self._selected_launches[quant]
            tiles = self._route_tiles(quant, flat_ids)
            x, record_bytes = self._stage_routes(flat_ids, activated, scales, extra=ids.size, tiles=tiles)
            x[-ids.size:].reshape(ids.shape)[...] = mix
            inp = self._scratch("_input", x.nbytes)
            itemsize = x.itemsize
            out = self._scratch("_output", result.nbytes + min(tile, tokens) * k * m * itemsize)
            route_out = out.ptr + result.nbytes
            ids_ptr = inp.ptr + activated.size * itemsize
            scales_ptr = ids_ptr + record_bytes if scales is not None else 0
            mix_ptr = inp.ptr + x.nbytes - flat_ids.nbytes
            with self._transfer.exchange(inp.ptr, out.ptr, x, result, pinned=True):
                for row in range(0, tokens, tile):
                    count = min(tile, tokens - row)
                    offset = row * k * itemsize
                    if tiles is None:
                        self._run_selected(quant, launch, inp.ptr + offset * n, weight.ptr, route_out, ids_ptr + offset,
                                           scales_ptr + offset if scales_ptr else 0, count * k, experts, n, m,
                                           single_token=ids.ndim == 1)
                    else:
                        # The tile builder admits only route lists that fit one selected launch.
                        self._tiled_launches[quant](inp.ptr, weight.ptr, route_out, ids_ptr, scales_ptr, len(tiles),
                                                   flat_ids.size, experts, n, m)
                    self._mix(route_out, mix_ptr + offset, out.ptr + row * m * itemsize, k, m, tokens=count)
                if transform_output is not None:
                    transform_output(out.ptr, result.shape)
        return result
