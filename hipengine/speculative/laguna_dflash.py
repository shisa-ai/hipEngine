"""Resident Poolside Laguna DFlash drafter over a Laguna GGUF target session."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import hip_target_arch_environment, hip_target_arch_for_backend
from hipengine.kernels.cpu_reference.laguna import LagunaRopeConfig
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_add_rmsnorm_f32_bf16_f32_weight,
    gguf_f32_bf16_add_out_f32,
    gguf_rmsnorm_f32_f32_weight,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import build_dense_gemv
from hipengine.kernels.hip_gfx1100.linear.lm_head import topk_f32_rows_i32
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import build_dflash_drafter
from hipengine.loading.dflash import (
    DFlashDrafterDeviceWeights,
    load_dflash_drafter_bf16_weights,
)
from hipengine.loading.safetensors import load_weight_index
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.laguna_gguf_runner import (
    LagunaGGUFResidentSession,
    LagunaHiddenCaptureTargets,
)
from hipengine.runtime.laguna_kv import LagunaKVCache, allocate_laguna_kv_cache
from hipengine.runtime.laguna_rope import (
    LagunaDeviceRoPETables,
    materialize_laguna_rope_tables,
)
from hipengine.speculative.dflash_drafter import (
    dflash_head_rmsnorm_rotary_f32,
    dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
    gate_laguna_dflash_attention_bf16,
    project_dflash_bf16_to_bf16,
    project_dflash_bf16_to_f32,
    project_laguna_dflash_target_hidden_bf16,
)


@dataclass(frozen=True)
class LagunaDFlashDraftResult:
    """Borrowed device outputs plus compact host top-k rows."""

    candidate_token_ids: tuple[int, ...]
    candidate_values: tuple[float, ...]
    topk_token_ids: tuple[tuple[int, ...], ...]
    topk_values: tuple[tuple[float, ...], ...]
    query_rows: int
    candidate_budget: int
    logits: Tensor
    final_hidden: Tensor


@dataclass
class LagunaDFlashCaptureOwner:
    """Caller-owned target hidden destinations at the normalized DFlash depths."""

    depths: tuple[int, ...]
    hidden_size: int
    rows: int
    buffers: tuple[DeviceBuffer, ...]
    tensors: tuple[Tensor, ...]
    targets: LagunaHiddenCaptureTargets
    runtime: HipRuntime
    _closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        depths: Sequence[int],
        hidden_size: int,
        rows: int,
        runtime: HipRuntime,
        device: Device | None = None,
    ) -> "LagunaDFlashCaptureOwner":
        normalized_depths = tuple(int(depth) for depth in depths)
        hidden = int(hidden_size)
        row_count = int(rows)
        if not normalized_depths:
            raise ValueError("Laguna DFlash capture depths must be non-empty")
        if len(set(normalized_depths)) != len(normalized_depths):
            raise ValueError("Laguna DFlash capture depths must be unique")
        if hidden <= 0 or row_count <= 0:
            raise ValueError("Laguna DFlash capture hidden_size/rows must be positive")
        capture_device = device or Device("hip", 0)
        nbytes = row_count * hidden * DType.BF16.itemsize
        buffers: list[DeviceBuffer] = []
        try:
            for _ in normalized_depths:
                buffers.append(malloc(nbytes, runtime=runtime))
            tensors = tuple(
                Tensor.from_handle(
                    buffer.ptr,
                    (row_count, hidden),
                    DType.BF16,
                    capture_device,
                )
                for buffer in buffers
            )
            targets = LagunaHiddenCaptureTargets(
                hidden_size=hidden,
                rows=row_count,
                buffers=dict(zip(normalized_depths, buffers, strict=True)),
            )
            return cls(
                normalized_depths,
                hidden,
                row_count,
                tuple(buffers),
                tensors,
                targets,
                runtime,
            )
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    def free(self) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in reversed(self.buffers):
            free(buffer, runtime=self.runtime)

    def __enter__(self) -> "LagunaDFlashCaptureOwner":
        if self._closed:
            raise RuntimeError("Laguna DFlash capture owner is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()


class LagunaDFlashResidentDrafter:
    """Correctness-first six-layer Laguna DFlash chain proposer.

    Target captures are committed separately from transient root/mask query
    blocks. The projected target context owns six bounded 512-token K/V rings;
    a proposal prepares causal query positions, attends over those rings plus
    current query K/V, then discards the pending query transaction.
    """

    def __init__(
        self,
        target_session: LagunaGGUFResidentSession,
        drafter_model: str | Path,
        *,
        candidate_budget: int = 7,
        top_k: int = 1,
        max_append_rows: int = 64,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
    ) -> None:
        if target_session.closed:
            raise ValueError("Laguna target session must be open")
        self.target = target_session
        self.runtime = target_session.runtime
        self.device = target_session.device
        self.backend = target_session.backend
        self.candidate_budget = int(candidate_budget)
        self.top_k = int(top_k)
        self.max_append_rows = int(max_append_rows)
        self.weights: DFlashDrafterDeviceWeights | None = None
        self.kv_cache: LagunaKVCache | None = None
        self.rope: LagunaDeviceRoPETables | None = None
        self._buffers: list[DeviceBuffer] = []
        self._f32_norm_weights: dict[str, Tensor] = {}
        self._closed = False
        self._draft_library = None
        self._dense_library = None

        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.top_k <= 0 or self.top_k > 8:
            raise ValueError("top_k must be within [1, 8]")
        if self.max_append_rows <= 0 or self.max_append_rows > 512:
            raise ValueError("max_append_rows must be within [1, 512]")
        try:
            index = load_weight_index(drafter_model)
            self.weights = load_dflash_drafter_bf16_weights(
                index,
                runtime=self.runtime,
                device=self.device,
            )
            self.config = self.weights.config
            self._validate_pair()
            self._materialize_f32_norm_weights()
            if self.candidate_budget >= self.config.block_size:
                raise ValueError("candidate_budget must be below the DFlash block size")
            self.query_rows = self.candidate_budget + 1
            target_arch = hip_target_arch_for_backend(self.backend)
            with hip_target_arch_environment(target_arch):
                self._draft_library = build_dflash_drafter(
                    load=True,
                    compiler_version=compiler_version,
                    require_cached=require_cached_build,
                )
                self._dense_library = build_dense_gemv(
                    load=True,
                    compiler_version=compiler_version,
                    require_cached=require_cached_build,
                )
            self.rope = materialize_laguna_rope_tables(
                self.target.context_length,
                LagunaRopeConfig(
                    rope_type="default",
                    rotary_dim=self.config.head_dim,
                    freq_base=self.config.rope_theta,
                ),
                device=self.device,
                runtime=self.runtime,
            )
            self.kv_cache = allocate_laguna_kv_cache(
                self._draft_kv_config(),
                context_length=self.target.context_length,
                backend=self.backend,
                device=self.device,
                runtime=self.runtime,
            )
            self._allocate_scratch()
        except BaseException:
            self._close(suppress_errors=True)
            raise

    @property
    def capture_depths(self) -> tuple[int, ...]:
        return self.config.target_capture_depths

    @property
    def committed_context_tokens(self) -> int:
        self._check_open()
        assert self.kv_cache is not None
        return self.kv_cache.position + 1

    @property
    def resident_nbytes(self) -> int:
        self._check_open()
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.rope is not None
        weight_bytes = sum(
            allocation.buffer.nbytes
            for allocation in self.weights.weights.tensors.values()
            if allocation.owns_buffer
        )
        return (
            weight_bytes
            + self.kv_cache.resident_nbytes
            + self.rope.cos.buffer.nbytes
            + self.rope.sin.buffer.nbytes
            + sum(buffer.nbytes for buffer in self._buffers)
        )

    def allocate_captures(self, *, rows: int = 1) -> LagunaDFlashCaptureOwner:
        self._check_open()
        return LagunaDFlashCaptureOwner.allocate(
            depths=self.capture_depths,
            hidden_size=self.config.target_hidden_size,
            rows=rows,
            runtime=self.runtime,
            device=self.device,
        )

    def append_target_hidden(
        self,
        captures: LagunaDFlashCaptureOwner | Sequence[Tensor],
        *,
        positions: Sequence[int],
        stream: int = 0,
    ) -> None:
        """Normalize/project and transactionally append committed target rows."""

        self._check_open()
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.rope is not None
        taps = captures.tensors if isinstance(captures, LagunaDFlashCaptureOwner) else tuple(captures)
        row_positions = tuple(int(position) for position in positions)
        rows = len(row_positions)
        if rows <= 0 or rows > self.max_append_rows:
            raise ValueError(
                f"append rows must be within [1, {self.max_append_rows}]"
            )
        if any(tap.shape[0] != rows for tap in taps):
            raise ValueError("capture tap rows must match append positions")
        expected_start = self.kv_cache.position + 1
        if row_positions != tuple(range(expected_start, expected_start + rows)):
            raise ValueError(f"append positions must be consecutive from {expected_start}")

        normalized_concat = self._rows_view(
            self.normalized_concat,
            rows,
            self.config.target_hidden_concat_size,
        )
        projected_scratch = self._rows_view(
            self.projected_scratch,
            rows,
            self.config.hidden_size,
        )
        projected = self._rows_view(
            self.projected_context,
            rows,
            self.config.hidden_size,
        )
        project_laguna_dflash_target_hidden_bf16(
            taps,
            normalized_concat,
            projected,
            projected_scratch,
            self.weights,
            stream=stream,
            libraries={"dense": self._dense_library, "norm": self._draft_library},
        )
        positions_i32 = np.ascontiguousarray(row_positions, dtype=np.int32)
        copy_host_to_device(
            self._buffer_for(self.append_positions),
            host_array_ptr(positions_i32),
            positions_i32.nbytes,
            runtime=self.runtime,
        )
        self.kv_cache.prepare_rows(row_positions)
        try:
            for layer in range(self.config.num_hidden_layers):
                self._append_projected_layer(layer, projected, rows=rows, stream=stream)
            self.kv_cache.commit_rows()
        except BaseException:
            if self.kv_cache.pending_positions:
                self.kv_cache.discard_rows()
            raise

    def propose(
        self,
        *,
        root_token_id: int,
        root_position: int,
        stream: int = 0,
    ) -> LagunaDFlashDraftResult:
        """Run one root + mask block and return compact greedy/top-k rows."""

        self._check_open()
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.target.weights is not None
        assert self.target.libraries is not None
        root = int(root_token_id)
        position = int(root_position)
        if root < 0 or root >= self.config.vocab_size:
            raise ValueError("root token is outside the shared target vocabulary")
        if position != self.kv_cache.position + 1:
            raise ValueError(
                f"root_position must equal next committed context position {self.kv_cache.position + 1}"
            )
        tokens = np.full(self.query_rows, self.config.mask_token_id, dtype=np.int64)
        tokens[0] = root
        positions = np.arange(position, position + self.query_rows, dtype=np.int32)
        if int(positions[-1]) >= self.target.context_length:
            raise ValueError("DFlash query block exceeds target context admission")
        copy_host_to_device(
            self._buffer_for(self.query_token_ids),
            host_array_ptr(tokens),
            tokens.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self._buffer_for(self.query_positions),
            host_array_ptr(positions),
            positions.nbytes,
            runtime=self.runtime,
        )
        launch_gguf_embedding(
            self.target.weights.root("token_embedding"),
            self.query_token_ids.ptr,
            self.query_embedding_bf16.ptr,
            self.query_rows,
            self.config.hidden_size,
            self.config.vocab_size,
            backend=self.backend,
            stream=stream,
            libraries={"gguf_q4_k": self.target.libraries.embedding},
            runtime=self.runtime,
        )
        gguf_f32_bf16_add_out_f32(
            self.query_zero_f32.ptr,
            self.query_embedding_bf16.ptr,
            self.query_hidden_a.ptr,
            self.query_rows * self.config.hidden_size,
            stream=stream,
            library=self.target.libraries.gguf_ops,
            runtime=self.runtime,
        )

        self.kv_cache.prepare_rows(tuple(int(value) for value in positions))
        query_in = self.query_hidden_a
        query_out = self.query_hidden_b
        try:
            for layer in range(self.config.num_hidden_layers):
                self._run_query_layer(
                    layer,
                    query_in=query_in,
                    query_out=query_out,
                    stream=stream,
                )
                query_in, query_out = query_out, query_in
        finally:
            if self.kv_cache.pending_positions:
                self.kv_cache.discard_rows()

        gguf_rmsnorm_f32_f32_weight(
            query_in.ptr,
            self._f32_norm_weight("norm.weight").ptr,
            self.final_hidden.ptr,
            self.query_rows,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            stream=stream,
            library=self.target.libraries.gguf_ops,
            runtime=self.runtime,
        )
        candidate_hidden_ptr = (
            self.final_hidden.ptr + self.config.hidden_size * DType.BF16.itemsize
        )
        launch_gguf_linear(
            self.target.weights.root("lm_head"),
            candidate_hidden_ptr,
            self.logits.ptr,
            self.candidate_budget,
            self.config.hidden_size,
            self.config.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            backend=self.backend,
            stream=stream,
            libraries=self.target.libraries.linear,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=False,
        )
        topk_f32_rows_i32(
            self.logits.ptr,
            self.topk_values.ptr,
            self.topk_ids.ptr,
            self.candidate_budget,
            self.config.vocab_size,
            self.top_k,
            stream=stream,
            library=self.target.libraries.argmax,
            runtime=self.runtime,
        )
        if stream:
            self.runtime.stream_synchronize(stream)
        else:
            self.runtime.device_synchronize()
        ids = np.empty((self.candidate_budget, self.top_k), dtype=np.int32)
        values = np.empty((self.candidate_budget, self.top_k), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(ids),
            self._buffer_for(self.topk_ids),
            ids.nbytes,
            runtime=self.runtime,
        )
        copy_device_to_host(
            host_array_ptr(values),
            self._buffer_for(self.topk_values),
            values.nbytes,
            runtime=self.runtime,
        )
        return LagunaDFlashDraftResult(
            candidate_token_ids=tuple(int(value) for value in ids[:, 0]),
            candidate_values=tuple(float(value) for value in values[:, 0]),
            topk_token_ids=tuple(
                tuple(int(value) for value in row)
                for row in ids
            ),
            topk_values=tuple(
                tuple(float(value) for value in row)
                for row in values
            ),
            query_rows=self.query_rows,
            candidate_budget=self.candidate_budget,
            logits=self.logits,
            final_hidden=self.final_hidden,
        )

    def close(self) -> None:
        self._close(suppress_errors=False)

    def __enter__(self) -> "LagunaDFlashResidentDrafter":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _validate_pair(self) -> None:
        target = self.target.config
        config = self.config
        errors: list[str] = []
        if config.decoder_arch != "laguna":
            errors.append(f"drafter decoder architecture is {config.decoder_arch!r}")
        if config.target_hidden_size != target.hidden_size:
            errors.append(
                f"target hidden mismatch: drafter={config.target_hidden_size} target={target.hidden_size}"
            )
        if config.vocab_size != target.vocab_size:
            errors.append(
                f"target vocab mismatch: drafter={config.vocab_size} target={target.vocab_size}"
            )
        if config.num_target_layers != target.block_count:
            errors.append(
                f"target layer mismatch: drafter={config.num_target_layers} target={target.block_count}"
            )
        if config.target_capture_depths != tuple(depth for depth in (2, 11, 20, 30, 39, 48)):
            errors.append(f"unsupported capture depths: {config.target_capture_depths}")
        if config.num_key_value_heads != 8 or config.head_dim != 128:
            errors.append("Laguna DFlash KV ABI requires 8 heads of width 128")
        if config.sliding_windows != (512,) * config.num_hidden_layers:
            errors.append("Laguna DFlash requires six 512-token sliding windows")
        for slot in ("token_embedding", "lm_head"):
            try:
                assert self.target.weights is not None
                self.target.weights.root(slot)
            except KeyError:
                errors.append(f"target-owned root {slot!r} is unavailable")
        if errors:
            raise ValueError("incompatible Laguna target/DFlash pair: " + "; ".join(errors))

    def _materialize_f32_norm_weights(self) -> None:
        """Expand tiny BF16 norm vectors for the FP32 residual path."""

        assert self.weights is not None
        names = ["norm.weight"]
        for layer in range(self.config.num_hidden_layers):
            names.extend(
                (
                    f"layers.{layer}.input_layernorm.weight",
                    f"layers.{layer}.post_attention_layernorm.weight",
                )
            )
        for name in names:
            source = self.weights.tensor(name)
            if source.dtype != DType.BF16 or source.ndim != 1:
                raise ValueError(f"DFlash norm {name!r} must be a BF16 vector")
            host_bf16 = np.empty(source.shape, dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(host_bf16),
                DeviceBuffer(source.ptr, host_bf16.nbytes),
                host_bf16.nbytes,
                runtime=self.runtime,
            )
            host_f32 = np.ascontiguousarray(
                (host_bf16.astype(np.uint32) << 16).view(np.float32)
            )
            expanded = self._empty(source.shape, DType.FP32)
            copy_host_to_device(
                self._buffer_for(expanded),
                host_array_ptr(host_f32),
                host_f32.nbytes,
                runtime=self.runtime,
            )
            self._f32_norm_weights[name] = expanded

    def _f32_norm_weight(self, name: str) -> Tensor:
        try:
            return self._f32_norm_weights[name]
        except KeyError as exc:
            raise KeyError(f"missing expanded DFlash norm weight: {name}") from exc

    def _draft_kv_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            block_count=self.config.num_hidden_layers,
            layer_types=("sliding_attention",) * self.config.num_hidden_layers,
            head_counts=(self.config.num_attention_heads,) * self.config.num_hidden_layers,
            head_count_kv=self.config.num_key_value_heads,
            key_length=self.config.head_dim,
            value_length=self.config.head_dim,
            sliding_window=512,
        )

    def _allocate_scratch(self) -> None:
        c = self.config
        qr = self.query_rows
        ar = self.max_append_rows
        h = c.hidden_size
        q = c.q_features
        kv = c.kv_features
        f = c.intermediate_size
        v = c.vocab_size
        self.normalized_concat = self._empty((ar, c.target_hidden_concat_size), DType.BF16)
        self.projected_scratch = self._empty((ar, h), DType.BF16)
        self.projected_context = self._empty((ar, h), DType.BF16)
        self.append_norm = self._empty((ar, h), DType.BF16)
        self.append_key_raw = self._empty((ar, kv), DType.FP32)
        self.append_value = self._empty((ar, kv), DType.FP32)
        self.append_key_rotated = self._empty((ar, kv), DType.FP32)
        self.append_positions = self._empty((ar,), DType.INT32)

        self.query_token_ids = self._empty((qr,), DType.INT64)
        self.query_positions = self._empty((qr,), DType.INT32)
        self.query_embedding_bf16 = self._empty((qr, h), DType.BF16)
        self.query_zero_f32 = self._empty((qr, h), DType.FP32)
        zeros = np.zeros((qr, h), dtype=np.float32)
        copy_host_to_device(
            self._buffer_for(self.query_zero_f32),
            host_array_ptr(zeros),
            zeros.nbytes,
            runtime=self.runtime,
        )
        self.query_hidden_a = self._empty((qr, h), DType.FP32)
        self.query_hidden_b = self._empty((qr, h), DType.FP32)
        self.query_norm = self._empty((qr, h), DType.BF16)
        self.query_raw = self._empty((qr, q), DType.FP32)
        self.key_raw = self._empty((qr, kv), DType.FP32)
        self.value_raw = self._empty((qr, kv), DType.FP32)
        self.query_rotated = self._empty((qr, q), DType.FP32)
        self.key_rotated = self._empty((qr, kv), DType.FP32)
        self.attention_context = self._empty((qr, q), DType.FP32)
        self.gate_logits = self._empty((qr, c.num_attention_heads), DType.FP32)
        self.gated_context = self._empty((qr, q), DType.BF16)
        self.attention_output = self._empty((qr, h), DType.BF16)
        self.post_attention = self._empty((qr, h), DType.FP32)
        self.ffn_norm = self._empty((qr, h), DType.BF16)
        self.ffn_gate = self._empty((qr, f), DType.BF16)
        self.ffn_up = self._empty((qr, f), DType.BF16)
        self.ffn_intermediate = self._empty((qr, f), DType.BF16)
        self.ffn_output = self._empty((qr, h), DType.BF16)
        self.final_hidden = self._empty((qr, h), DType.BF16)
        self.logits = self._empty((self.candidate_budget, v), DType.FP32)
        self.topk_values = self._empty((self.candidate_budget, self.top_k), DType.FP32)
        self.topk_ids = self._empty((self.candidate_budget, self.top_k), DType.INT32)

    def _append_projected_layer(
        self,
        layer: int,
        projected: Tensor,
        *,
        rows: int,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.rope is not None
        prefix = f"layers.{layer}"
        norm = self._rows_view(self.append_norm, rows, self.config.hidden_size)
        key = self._rows_view(self.append_key_raw, rows, self.config.kv_features)
        value = self._rows_view(self.append_value, rows, self.config.kv_features)
        key_rotated = self._rows_view(
            self.append_key_rotated,
            rows,
            self.config.kv_features,
        )
        dflash_rmsnorm_bf16(
            projected,
            self.weights.tensor(f"{prefix}.input_layernorm.weight"),
            norm,
            eps=self.config.rms_norm_eps,
            stream=stream,
            library=self._draft_library,
        )
        project_dflash_bf16_to_f32(
            norm,
            self.weights.tensor(f"{prefix}.self_attn.k_proj.weight"),
            key,
            stream=stream,
            library=self._draft_library,
        )
        project_dflash_bf16_to_f32(
            norm,
            self.weights.tensor(f"{prefix}.self_attn.v_proj.weight"),
            value,
            stream=stream,
            library=self._draft_library,
        )
        from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
            dflash_key_rmsnorm_rotary_f32,
        )

        dflash_key_rmsnorm_rotary_f32(
            key.ptr,
            self.weights.tensor(f"{prefix}.self_attn.k_norm.weight").ptr,
            self.rope.cos.tensor.ptr,
            self.rope.sin.tensor.ptr,
            self.append_positions.ptr,
            key_rotated.ptr,
            rows,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.head_dim,
            self.rope.max_positions,
            stream=stream,
            library=self._draft_library,
            runtime=self.runtime,
        )
        self.kv_cache.append_rows(
            layer,
            key_rotated.ptr,
            value.ptr,
            rows,
            stream=stream,
            library=self.target.libraries.kv_attention if self.target.libraries else None,
        )

    def _run_query_layer(
        self,
        layer: int,
        *,
        query_in: Tensor,
        query_out: Tensor,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.rope is not None
        assert self.target.libraries is not None
        prefix = f"layers.{layer}"
        gguf_rmsnorm_f32_f32_weight(
            query_in.ptr,
            self._f32_norm_weight(f"{prefix}.input_layernorm.weight").ptr,
            self.query_norm.ptr,
            self.query_rows,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            stream=stream,
            library=self.target.libraries.gguf_ops,
            runtime=self.runtime,
        )
        for name, output in (
            ("q_proj", self.query_raw),
            ("k_proj", self.key_raw),
            ("v_proj", self.value_raw),
        ):
            project_dflash_bf16_to_f32(
                self.query_norm,
                self.weights.tensor(f"{prefix}.self_attn.{name}.weight"),
                output,
                stream=stream,
                library=self._draft_library,
            )
        query = Tensor.from_handle(
            self.query_raw.ptr,
            (
                1,
                self.query_rows,
                self.config.num_attention_heads,
                self.config.head_dim,
            ),
            DType.FP32,
            self.device,
        )
        key = Tensor.from_handle(
            self.key_raw.ptr,
            (
                1,
                self.query_rows,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ),
            DType.FP32,
            self.device,
        )
        query_rotated = Tensor.from_handle(
            self.query_rotated.ptr,
            query.shape,
            DType.FP32,
            self.device,
        )
        key_rotated = Tensor.from_handle(
            self.key_rotated.ptr,
            key.shape,
            DType.FP32,
            self.device,
        )
        positions = Tensor.from_handle(
            self.query_positions.ptr,
            (1, self.query_rows),
            DType.INT32,
            self.device,
        )
        dflash_head_rmsnorm_rotary_f32(
            query,
            key,
            self.weights.tensor(f"{prefix}.self_attn.q_norm.weight"),
            self.weights.tensor(f"{prefix}.self_attn.k_norm.weight"),
            self.rope.cos.tensor,
            self.rope.sin.tensor,
            positions,
            positions,
            query_rotated,
            key_rotated,
            eps=self.config.rms_norm_eps,
            stream=stream,
            library=self._draft_library,
        )
        self.kv_cache.attend_prefill(
            layer,
            self.query_rotated.ptr,
            self.key_rotated.ptr,
            self.value_raw.ptr,
            self.attention_context.ptr,
            self.query_rows,
            stream=stream,
            library=self.target.libraries.kv_attention,
        )
        gate_laguna_dflash_attention_bf16(
            self.query_norm,
            self.attention_context,
            self.gate_logits,
            self.gated_context,
            self.weights,
            layer=layer,
            gate_kernel=self.target.kernel_plan.attention_gate,
            stream=stream,
            projection_library=self._draft_library,
            library=self.target.libraries.attention_gate,
            runtime=self.runtime,
        )
        project_dflash_bf16_to_bf16(
            self.gated_context,
            self.weights.tensor(f"{prefix}.self_attn.o_proj.weight"),
            self.attention_output,
            stream=stream,
            library=self._draft_library,
        )
        gguf_add_rmsnorm_f32_bf16_f32_weight(
            query_in.ptr,
            self.attention_output.ptr,
            self._f32_norm_weight(f"{prefix}.post_attention_layernorm.weight").ptr,
            self.ffn_norm.ptr,
            self.post_attention.ptr,
            self.query_rows,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            stream=stream,
            library=self.target.libraries.gguf_ops,
            runtime=self.runtime,
        )
        project_dflash_bf16_to_bf16(
            self.ffn_norm,
            self.weights.tensor(f"{prefix}.mlp.gate_proj.weight"),
            self.ffn_gate,
            stream=stream,
            library=self._draft_library,
        )
        project_dflash_bf16_to_bf16(
            self.ffn_norm,
            self.weights.tensor(f"{prefix}.mlp.up_proj.weight"),
            self.ffn_up,
            stream=stream,
            library=self._draft_library,
        )
        dflash_silu_mul_bf16(
            self.ffn_gate,
            self.ffn_up,
            self.ffn_intermediate,
            stream=stream,
            library=self._draft_library,
        )
        project_dflash_bf16_to_bf16(
            self.ffn_intermediate,
            self.weights.tensor(f"{prefix}.mlp.down_proj.weight"),
            self.ffn_output,
            stream=stream,
            library=self._draft_library,
        )
        gguf_f32_bf16_add_out_f32(
            self.post_attention.ptr,
            self.ffn_output.ptr,
            query_out.ptr,
            self.query_rows * self.config.hidden_size,
            stream=stream,
            library=self.target.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _empty(self, shape: tuple[int, ...], dtype: DType) -> Tensor:
        nbytes = math.prod(shape) * dtype.itemsize
        buffer = malloc(nbytes, runtime=self.runtime)
        self._buffers.append(buffer)
        return Tensor.from_handle(buffer.ptr, shape, dtype, self.device)

    @staticmethod
    def _rows_view(tensor: Tensor, rows: int, width: int) -> Tensor:
        if tensor.shape[0] < rows or tensor.shape[1] != width:
            raise ValueError("scratch tensor cannot cover requested rows/width")
        return Tensor.from_handle(
            tensor.ptr,
            (int(rows), int(width)),
            tensor.dtype,
            tensor.device,
        )

    def _buffer_for(self, tensor: Tensor) -> DeviceBuffer:
        for buffer in self._buffers:
            if buffer.ptr == tensor.ptr:
                return buffer
        raise KeyError(f"no Laguna DFlash owner for tensor 0x{tensor.ptr:x}")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna DFlash drafter is closed")

    def _close(self, *, suppress_errors: bool) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for buffer in reversed(self._buffers):
            try:
                free(buffer, runtime=self.runtime)
            except BaseException as exc:
                errors.append(exc)
        self._buffers.clear()
        for close in (
            lambda: self.kv_cache.free() if self.kv_cache is not None else None,
            lambda: self.rope.free(runtime=self.runtime) if self.rope is not None else None,
            lambda: self.weights.free(runtime=self.runtime) if self.weights is not None else None,
        ):
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
        if errors and not suppress_errors:
            raise errors[0]


__all__ = [
    "LagunaDFlashCaptureOwner",
    "LagunaDFlashDraftResult",
    "LagunaDFlashResidentDrafter",
]
