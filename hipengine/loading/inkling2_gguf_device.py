"""Packed projection ownership for the Inkling GGUF decoder."""

from __future__ import annotations

from functools import partial

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.kernels.backends import hip_target_arch_for_backend, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_gemv, gguf_nvfp4_q8_dot, inkling2_experts  # noqa: F401
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


class Inkling2GGUFDeviceProjections(Inkling2DeviceDense):
    """Share dense transfer scratch; upload physical expert banks only once."""

    def __init__(self, weights, *, runtime, blas, backend, fast_experts=False):
        super().__init__(weights, runtime=runtime, blas=blas)
        load_backend_kernel_package(backend)
        self._tensors = {t.name: t for t in weights.reader.info.tensors}
        self._target_arch = hip_target_arch_for_backend(backend)
        self._launches = {q: resolve(backend=backend, layer="linear", quant=f"gguf_{q.name.lower()}",
                                    variant="gemv_f32_f32_out") for q in (Q.NVFP4, Q.Q8_0, Q.Q5_K)}
        self._bound_launches = {}
        self._selected_launches = {}
        self._tiled_launches = {}
        self._shared_tiled_launches = {}
        self._q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                variant="selected_q8_dot_f32_f32_out") if fast_experts else None)
        self._shared_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                       variant="selected_shared_input_q8_dot_f32_f32_out") if fast_experts else None)
        self._counted_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                        variant="selected_counted_shared_input_q8_dot_f32_f32_out")
                                 if fast_experts else None)
        self._counted_split_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                              variant="selected_counted_split_activation_q8_dot_f32_f32_out")
                                       if fast_experts else None)
        self._split_q8_dot = (resolve(backend=backend, layer="moe_linear", quant="gguf_nvfp4",
                                      variant="selected_split_activation_q8_dot_f32_f32_out") if fast_experts else None)
        self._q8_dot_scratch = None
        self._k_library = None
        prefill = KernelKey(backend, "linear", "gguf_q8_0", "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out")
        self._q8_prefill = (self._bind_packed(Q.Q8_0, resolve(backend=prefill.backend, layer=prefill.layer,
                                                           quant=prefill.quant, variant=prefill.variant))
                            if is_registered(prefill) else None)
        decode = KernelKey(backend, "linear", "gguf_q8_0", "pack8_gemv_f32_f32_out")
        self._q8_decode = (self._bind_packed(Q.Q8_0, resolve(backend=decode.backend, layer=decode.layer,
                                                           quant=decode.quant, variant=decode.variant))
                           if is_registered(decode) else None)
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
        self._bank_scales = {}
        self._upload_staging = None

    @property
    def supports_device_mix(self):
        return self._mix is not None

    @property
    def supports_device_activation(self):
        return self._activation is not None

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
                    self._runtime.memcpy_async(inp + i * row_bytes, ptr, row_bytes, HipMemcpyKind.DEVICE_TO_DEVICE, 0)
                self._launch_projection(inp, out, len(inputs), 2, matrix, None)
            else:
                for i, (ptr, _) in enumerate(inputs):
                    self._launch_projection(ptr, out + i * y.strides[0], 1, 1, matrix, None)
            for i, (_, consume) in enumerate(inputs):
                consume(out + i * y.strides[0], (m,))
            self._transfer.download(out, y)
            return y

    def _launch_projection(self, src, dst, rows, ndim, matrix, native_bf16):
        weight, (m, n), packed = matrix
        if not packed:
            super()._launch_projection(src, dst, rows, ndim, matrix, native_bf16)
            return
        if packed == Q.Q8_0:
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

    def close(self):
        try:
            super().close()
        finally:
            self._upload_staging = None
            with self._lock:
                if self._q8_dot_scratch is not None:
                    free(self._q8_dot_scratch, runtime=self._runtime)
                    self._q8_dot_scratch = None

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

    def project_experts_mixed(self, visit, ids, mix, hidden, *, transform_output=None):
        return self.project_experts_mixed_rows(visit, ((ids, mix, hidden, transform_output),))[0]

    def project_experts_mixed_rows(self, visit, inputs):
        """Share an exchange and the gate weights where the scalar arithmetic matches."""
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
            with self._transfer.exchange(inp, out, packed, result, pinned=True):
                if counted:
                    launch_counted(inp, gate, down_out)
                for parts, shape, size, _, _, launch in plans:
                    launch(inp + input_offset, out + output_offset * result.itemsize, interim, projection)
                    results.append(result[output_offset:output_offset + size].reshape(shape))
                    input_offset += sum(part.nbytes for part in parts)
                    output_offset += size
                    projection += gate_row_bytes
            return tuple(results)

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
