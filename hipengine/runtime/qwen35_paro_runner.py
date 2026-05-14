"""Bring-up runner for real Qwen3.5/PARO one-token decode smokes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
)
from hipengine.kernels.hip_gfx1100.norm import paro_rmsnorm_out_bf16
from hipengine.kvcache import KVLiveSpans
from hipengine.loading import (
    WeightIndex,
    float_array_to_bf16_bits,
    load_weight_index,
    materialize_qwen35_paro_full_attention_moe_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer,
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
)
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
    load_tensor_info_to_device,
)
from hipengine.runtime.qwen35_paro import Qwen35ParoDecodeState
from hipengine.runtime.workspace import RuntimeWorkspace


@dataclass(frozen=True)
class Qwen35ParoLayerRecord:
    """One layer executed by the one-token Qwen3.5/PARO smoke path."""

    layer: int
    type: str

    def to_json_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "type": self.type}


@dataclass(frozen=True)
class Qwen35ParoNextTokenResult:
    """Structured result from the one-token Qwen3.5/PARO bring-up runner."""

    model: str
    prompt: str
    prompt_ids: tuple[int, ...]
    input_token_id: int
    layers_run: tuple[Qwen35ParoLayerRecord, ...]
    next_token_id: int
    next_token_text: str
    next_token_logit: float
    lm_head: str = "cpu_numpy_argmax"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "prompt_ids": list(self.prompt_ids),
            "input_token_id": self.input_token_id,
            "layers_run": [record.to_json_dict() for record in self.layers_run],
            "next_token_id": self.next_token_id,
            "next_token_text": self.next_token_text,
            "next_token_logit": self.next_token_logit,
            "lm_head": self.lm_head,
        }


class Qwen35ParoNextTokenRunner:
    """Torch-free one-token next-token runner for the real Qwen3.5/PARO checkpoint.

    This is a correctness/bring-up path, not a performance path: it materializes one
    layer at a time, runs the c=1 decode layer chain on HIP, applies final RMSNorm on
    HIP, and computes the lm-head argmax on CPU with NumPy chunks.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        index: WeightIndex | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.model = Path(model)
        self.index = index or load_weight_index(self.model)
        self.config = qwen35_paro_config_from_hf(self.index.config)
        self.normalized_infos = _normalized_infos(self.index)
        self.runtime = runtime or get_hip_runtime()

    def run_next_token(
        self,
        *,
        prompt: str = "Hello",
        token_id: int | None = None,
        max_layers: int = 0,
        lm_head_chunk: int = 4096,
        progress: Callable[[dict[str, Any]], None] | None = None,
        resident_layers: bool = False,
        lm_head: str = "gpu_fp16_argmax",
    ) -> Qwen35ParoNextTokenResult:
        if lm_head_chunk <= 0:
            raise ValueError("lm_head_chunk must be positive")
        if lm_head not in {"gpu_fp16_argmax", "cpu_numpy_argmax"}:
            raise ValueError("lm_head must be 'gpu_fp16_argmax' or 'cpu_numpy_argmax'")

        def emit(event: str, **fields: Any) -> None:
            if progress is not None:
                progress({"event": event, **fields})

        token_id, prompt_ids = _select_token(self.model, prompt, token_id)
        emit("token_selected", token_id=token_id, prompt_ids=list(prompt_ids))
        runtime = self.runtime
        device = Device("hip", 0)
        buffers: list[DeviceBuffer] = []
        allocations: list[DeviceTensorAllocation] = []

        def dev(array: np.ndarray) -> DeviceBuffer:
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), runtime=runtime)
            return buf

        hidden_bits = float_array_to_bf16_bits(
            _read_tensor(self.normalized_infos, "language_model.embed_tokens.weight")[
                token_id : token_id + 1
            ]
        )
        if hidden_bits.shape != (1, self.config.hidden_size):
            raise ValueError(
                f"unexpected embedding row shape {hidden_bits.shape}, "
                f"expected (1, {self.config.hidden_size})"
            )
        hidden_a = dev(hidden_bits)
        hidden_b = malloc(hidden_bits.nbytes, runtime=runtime)
        buffers.append(hidden_b)
        hidden = Tensor.from_handle(hidden_a.ptr, hidden_bits.shape, DType.BF16, device)
        next_hidden = Tensor.from_handle(hidden_b.ptr, hidden_bits.shape, DType.BF16, device)

        # One-token decode smoke: all full-attention layers can reuse the same temporary
        # KV page, and all linear layers can reuse zeroed recurrent/conv state inputs.
        block_size = 256
        block_table_arr = np.asarray([0], dtype=np.int32)
        position_arr = np.asarray([0], dtype=np.int64)
        context_arr = np.asarray([1], dtype=np.int64)
        block_table_buf = dev(block_table_arr)
        position_buf = dev(position_arr)
        context_buf = dev(context_arr)
        block_table = Tensor.from_handle(block_table_buf.ptr, block_table_arr.shape, DType.INT32, device)
        position = Tensor.from_handle(position_buf.ptr, position_arr.shape, DType.INT64, device)
        context = Tensor.from_handle(context_buf.ptr, context_arr.shape, DType.INT64, device)
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=position,
            max_live_count=0,
            storage_dtype=DType.BF16,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context,
            max_live_count=1,
            storage_dtype=DType.BF16,
        )
        cos_arr, sin_arr = _rope_tables(
            max_positions=1,
            rotary_dim=self.config.rotary_dim or self.config.head_dim,
            base=self.config.rope_theta,
        )
        cos_buf = dev(cos_arr)
        sin_buf = dev(sin_arr)
        cos = Tensor.from_handle(cos_buf.ptr, cos_arr.shape, DType.FP32, device)
        sin = Tensor.from_handle(sin_buf.ptr, sin_arr.shape, DType.FP32, device)

        key_cache_arr = np.zeros(
            (1, block_size, self.config.num_key_value_heads, self.config.head_dim),
            dtype=np.uint16,
        )
        value_cache_arr = np.zeros_like(key_cache_arr)
        key_cache_buf = dev(key_cache_arr)
        value_cache_buf = dev(value_cache_arr)
        key_cache = Tensor.from_handle(key_cache_buf.ptr, key_cache_arr.shape, DType.BF16, device)
        value_cache = Tensor.from_handle(value_cache_buf.ptr, value_cache_arr.shape, DType.BF16, device)

        qkv_width = (
            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        conv_zero = np.zeros((qkv_width, self.config.linear_conv_kernel_dim), dtype=np.float32)
        recurrent_zero = np.zeros(
            (
                self.config.linear_num_value_heads,
                self.config.linear_key_head_dim,
                self.config.linear_value_head_dim,
            ),
            dtype=np.float32,
        )
        conv_buf = dev(conv_zero)
        recurrent_buf = dev(recurrent_zero)
        conv_state = Tensor.from_handle(conv_buf.ptr, conv_zero.shape, DType.FP32, device)
        recurrent_state = Tensor.from_handle(recurrent_buf.ptr, recurrent_zero.shape, DType.FP32, device)

        layer_limit = (
            self.config.num_hidden_layers
            if max_layers <= 0
            else min(max_layers, self.config.num_hidden_layers)
        )
        layer_records: list[Qwen35ParoLayerRecord] = []
        resident_states: list[Qwen35ParoDecodeState] = []
        emit("layers_start", layers=layer_limit, resident=resident_layers)
        try:
            if resident_layers:
                resident_states = self._materialize_resident_states(layer_limit, emit=emit)
            for layer_id in range(layer_limit):
                layer_type = self.config.layer_types[layer_id]
                emit("layer_start", layer=layer_id, type=layer_type)
                state = (
                    resident_states[layer_id]
                    if resident_layers
                    else self._materialize_state(layer_id, layer_type, progress=_progress_forwarder(emit))
                )
                try:
                    out = self._run_layer_state(
                        state,
                        layer_type,
                        hidden,
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        conv_buf=conv_buf,
                        recurrent_buf=recurrent_buf,
                        conv_zero=conv_zero,
                        recurrent_zero=recurrent_zero,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        key_cache_buf=key_cache_buf,
                        value_cache_buf=value_cache_buf,
                        key_cache_zero=key_cache_arr,
                        value_cache_zero=value_cache_arr,
                        append_spans=append_spans,
                        decode_spans=decode_spans,
                        cos=cos,
                        sin=sin,
                        position=position,
                    )
                    runtime.memcpy(
                        next_hidden.ptr,
                        out.ptr,
                        hidden_bits.nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                    )
                finally:
                    if not resident_layers:
                        state.free()
                hidden, next_hidden = next_hidden, hidden
                layer_records.append(Qwen35ParoLayerRecord(layer=layer_id, type=layer_type))
                emit("layer_done", layer=layer_id, type=layer_type)

            emit("final_norm_start")
            norm_bits = float_array_to_bf16_bits(
                _read_tensor(self.normalized_infos, "language_model.norm.weight")
            )
            norm_weight = load_host_array_to_device_as_dtype(
                "model.norm.weight",
                norm_bits,
                DType.BF16,
                runtime=runtime,
            )
            allocations.append(norm_weight)
            norm_out_buf = malloc(hidden_bits.nbytes, runtime=runtime)
            buffers.append(norm_out_buf)
            norm_out = Tensor.from_handle(norm_out_buf.ptr, hidden_bits.shape, DType.BF16, device)
            paro_rmsnorm_out_bf16(
                hidden.ptr,
                norm_weight.tensor.ptr,
                norm_out.ptr,
                1,
                self.config.hidden_size,
                self.config.rms_norm_eps,
                runtime=runtime,
            )
            runtime.device_synchronize()
            emit("final_norm_done")
            emit("lm_head_start", mode=lm_head, chunk_size=lm_head_chunk)
            if lm_head == "gpu_fp16_argmax":
                next_id, next_logit = self._gpu_lm_head_argmax(norm_out, allocations, buffers)
            else:
                final_bits = np.empty(hidden_bits.shape, dtype=np.uint16)
                copy_device_to_host(
                    host_array_ptr(final_bits),
                    DeviceBuffer(norm_out.ptr, final_bits.nbytes),
                    runtime=runtime,
                )
                final_hidden = _bf16_bits_to_float32(final_bits.reshape(-1))
                next_id, next_logit = _lm_head_argmax(
                    self.normalized_infos,
                    final_hidden,
                    chunk_size=lm_head_chunk,
                )
            emit("lm_head_done", next_token_id=next_id, next_token_logit=next_logit)
            return Qwen35ParoNextTokenResult(
                model=str(self.model),
                prompt=prompt,
                prompt_ids=tuple(prompt_ids),
                input_token_id=token_id,
                layers_run=tuple(layer_records),
                next_token_id=next_id,
                next_token_text=_decode_token(self.model, next_id),
                next_token_logit=next_logit,
                lm_head=lm_head,
            )
        finally:
            for state in reversed(resident_states):
                state.free()
            for allocation in reversed(allocations):
                allocation.free(runtime=runtime)
            for buf in reversed(buffers):
                free(buf, runtime=runtime)

    def _materialize_state(
        self,
        layer_id: int,
        layer_type: str,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen35ParoDecodeState:
        if layer_type == "linear_attention":
            return self._materialize_linear_state(layer_id, progress=progress)
        if layer_type == "full_attention":
            return self._materialize_full_state(layer_id, progress=progress)
        raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")

    def _materialize_resident_states(
        self,
        layer_limit: int,
        *,
        emit: Callable[..., None],
    ) -> list[Qwen35ParoDecodeState]:
        states: list[Qwen35ParoDecodeState] = []
        try:
            for layer_id in range(layer_limit):
                layer_type = self.config.layer_types[layer_id]
                emit("materialize_layer_start", layer=layer_id, type=layer_type)
                states.append(self._materialize_state(layer_id, layer_type, progress=_progress_forwarder(emit)))
                emit("materialize_layer_done", layer=layer_id, type=layer_type)
        except Exception:
            for state in reversed(states):
                state.free()
            raise
        return states

    def _run_layer_state(
        self,
        state: Qwen35ParoDecodeState,
        layer_type: str,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        conv_buf: DeviceBuffer,
        recurrent_buf: DeviceBuffer,
        conv_zero: np.ndarray,
        recurrent_zero: np.ndarray,
        key_cache: Tensor,
        value_cache: Tensor,
        key_cache_buf: DeviceBuffer,
        value_cache_buf: DeviceBuffer,
        key_cache_zero: np.ndarray,
        value_cache_zero: np.ndarray,
        append_spans: KVLiveSpans,
        decode_spans: KVLiveSpans,
        cos: Tensor,
        sin: Tensor,
        position: Tensor,
    ) -> Tensor:
        if layer_type == "linear_attention":
            _copy_zero(self.runtime, conv_buf, conv_zero)
            _copy_zero(self.runtime, recurrent_buf, recurrent_zero)
            return state.run_linear_attention_moe_c1_layer_bf16(
                hidden,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )
        if layer_type == "full_attention":
            _copy_zero(self.runtime, key_cache_buf, key_cache_zero)
            _copy_zero(self.runtime, value_cache_buf, value_cache_zero)
            return state.run_full_attention_moe_c1_layer_bf16(
                hidden,
                key_cache=key_cache,
                value_cache=value_cache,
                append_spans=append_spans,
                decode_spans=decode_spans,
                cos_table=cos,
                sin_table=sin,
                position=position,
                max_positions=1,
            )
        raise ValueError(f"unsupported layer type {layer_type!r}")

    def _gpu_lm_head_argmax(
        self,
        hidden: Tensor,
        allocations: list[DeviceTensorAllocation],
        buffers: list[DeviceBuffer],
    ) -> tuple[int, float]:
        info = self.normalized_infos["lm_head.weight"]
        lm_head_weight = load_tensor_info_to_device(info, runtime=self.runtime)
        allocations.append(lm_head_weight)
        vocab_size, hidden_size = lm_head_weight.tensor.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(f"lm_head hidden size {hidden_size} does not match {self.config.hidden_size}")
        threads = 256
        stage1_blocks = lm_head_argmax_stage1_blocks(vocab_size, threads=threads)
        logits = malloc(vocab_size * DType.FP32.itemsize, runtime=self.runtime)
        block_values = malloc(stage1_blocks * DType.FP32.itemsize, runtime=self.runtime)
        block_indices = malloc(stage1_blocks * DType.INT64.itemsize, runtime=self.runtime)
        out_index = malloc(DType.INT64.itemsize, runtime=self.runtime)
        out_value = malloc(DType.FP32.itemsize, runtime=self.runtime)
        buffers.extend((logits, block_values, block_indices, out_index, out_value))
        lm_head_fp16_argmax_bf16(
            hidden.ptr,
            lm_head_weight.tensor.ptr,
            logits.ptr,
            block_values.ptr,
            block_indices.ptr,
            out_index.ptr,
            out_value.ptr,
            self.config.hidden_size,
            vocab_size,
            threads=threads,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        index_host = np.empty((1,), dtype=np.int64)
        value_host = np.empty((1,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(index_host), out_index, runtime=self.runtime)
        copy_device_to_host(host_array_ptr(value_host), out_value, runtime=self.runtime)
        return int(index_host[0]), float(value_host[0])

    def _materialize_linear_state(
        self,
        layer_id: int,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen35ParoDecodeState:
        weights = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(
            self.index,
            layer_id=layer_id,
            runtime=self.runtime,
            progress=progress,
        )
        return Qwen35ParoDecodeState(
            layer_weights=weights,
            workspace=RuntimeWorkspace(runtime=self.runtime),
            runtime=self.runtime,
        )

    def _materialize_full_state(
        self,
        layer_id: int,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen35ParoDecodeState:
        weights = materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(
            self.index,
            layer_id=layer_id,
            runtime=self.runtime,
            progress=progress,
        )
        return Qwen35ParoDecodeState(
            layer_weights=weights,
            workspace=RuntimeWorkspace(runtime=self.runtime),
            runtime=self.runtime,
        )


def _progress_forwarder(emit: Callable[..., None]) -> Callable[[dict[str, Any]], None]:
    def forward(payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "loader"))
        fields = {key: value for key, value in payload.items() if key != "event"}
        emit(event, **fields)

    return forward


def _normalized_infos(index: WeightIndex) -> dict[str, Any]:
    out = {}
    for name, info in index.tensors.items():
        out[normalize_qwen35_weight_name(name)] = info
    return out


def _read_tensor(normalized: dict[str, Any], name: str) -> np.ndarray:
    key = normalize_qwen35_weight_name(name)
    info = normalized[key]
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        return np.ascontiguousarray(handle.get_tensor(info.name))


def _select_token(model: Path, prompt: str, token_id: int | None) -> tuple[int, list[int]]:
    if token_id is not None:
        return int(token_id), [int(token_id)]
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency guard
        raise RuntimeError("tokenizers is required unless --token-id is supplied") from exc
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt produced no tokens")
    return int(ids[-1]), [int(x) for x in ids]


def _decode_token(model: Path, token_id: int) -> str:
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _copy_zero(runtime: HipRuntime, buffer: DeviceBuffer, zeros: np.ndarray) -> None:
    copy_host_to_device(buffer, host_array_ptr(zeros), runtime=runtime)


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _lm_head_argmax(
    normalized: dict[str, Any],
    hidden: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[int, float]:
    info = normalized["lm_head.weight"]
    best_id = -1
    best_logit = -float("inf")
    hidden_f32 = hidden.astype(np.float32, copy=False)
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        weight = handle.get_tensor(info.name)
        rows = int(weight.shape[0])
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            logits = weight[start:end].astype(np.float32) @ hidden_f32
            local = int(np.argmax(logits))
            value = float(logits[local])
            if value > best_logit:
                best_logit = value
                best_id = start + local
    return best_id, best_logit
