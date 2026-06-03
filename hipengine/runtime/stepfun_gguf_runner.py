"""Torch-free StepFun GGUF short-context decode planning helpers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    host_buffer_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.dispatch.kv import (
    PagedAttnDecodeKind,
    PagedKVWriteKind,
    plan_paged_attn_decode,
    plan_paged_kv_write,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.gguf import GGUFSplitModelInfo, scan_gguf_splits
from hipengine.loading.stepfun_gguf import (
    SLIDING_ATTENTION,
    StepFunGGUFConfig,
    StepFunGGUFModelMap,
    build_stepfun_gguf_tensor_map,
)
from hipengine.loading.stepfun_gguf_materialize import (
    StepFunGGUFResidentWeights,
    materialize_stepfun_gguf_weights,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_BF16, GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.tokenization import StepFunGGUFTokenizer

DEFAULT_STEPFUN_SHORT_CONTEXT = 512
DEFAULT_STEPFUN_MAX_NEW_TOKENS = 1
STEPFUN_GGUF_KERNEL_QUANT = "gguf_step35"
STEPFUN_KV_ATTENTION_BLOCK_SIZE = 256
BF16_BYTES = 2


def _stable_json_sha256(value: object) -> str:
    """Return a stable SHA-256 digest for JSON-serializable runtime metadata."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stepfun_streaming_runner_blockers() -> tuple[dict[str, object], ...]:
    """Return current blockers for marking the StepFun streaming runner ready."""

    return (
        {
            "name": "streaming_decode_loop_not_wired",
            "ready": False,
            "required_evidence": (
                "A StepFunResidentSession decode loop must launch prompt KV writes, one-token "
                "decode KV writes, and gated paged attention from resident buffers."
            ),
        },
        {
            "name": "kv_kernel_trace_artifact_missing",
            "ready": False,
            "required_evidence": (
                "A retained rocprofv3 or equivalent trace must show the prompt KV write, "
                "decode KV write, and gated decode-attention kernels for the canonical prompt."
            ),
        },
        {
            "name": "kv_backed_next_token_artifact_missing",
            "ready": False,
            "required_evidence": (
                "A KV-backed one-token decode artifact must record the generated token/logit path "
                "without host-composed layer-prefix outputs."
            ),
        },
    )


@dataclass(frozen=True)
class StepFunDecodePlan:
    """Validated prompt-side plan for StepFun c=1 bring-up."""

    input_ids: tuple[int, ...]
    rendered_prompt: str
    stop_token_ids: tuple[int, ...]
    max_context: int
    max_new_tokens: int
    backend: str
    quant_dispatch_keys: Mapping[str, KernelKey]
    kv_dispatch_keys: Mapping[str, KernelKey]

    @property
    def prompt_length(self) -> int:
        return len(self.input_ids)

    def should_stop(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids


@dataclass(frozen=True)
class StepFunMoERouterResult:
    """Host-visible Step MoE router outputs for correctness probes."""

    routing_weights: object
    selected_experts: object
    logits: object


@dataclass(frozen=True)
class StepFunRootOnlyLogitsProbe:
    """Root-only prompt embedding plus final logits smoke result."""

    prompt: "StepFunPromptEmbedding"
    logits: object

    @property
    def next_token_id(self) -> int:
        import numpy as np

        return int(np.argmax(self.logits[-1]))

    @property
    def next_token_logit(self) -> float:
        return float(self.logits[-1, self.next_token_id])


@dataclass(frozen=True)
class StepFunOneLayerLogitsProbe:
    """Prompt embedding plus one resident layer and final logits smoke result."""

    prompt: "StepFunPromptEmbedding"
    layer_hidden: object
    logits: object

    @property
    def next_token_id(self) -> int:
        import numpy as np

        return int(np.argmax(self.logits[-1]))

    @property
    def next_token_logit(self) -> float:
        return float(self.logits[-1, self.next_token_id])


@dataclass(frozen=True)
class StepFunLayerPrefixLogitsProbe:
    """Prompt embedding plus a contiguous resident layer prefix and logits."""

    prompt: "StepFunPromptEmbedding"
    layer_count: int
    layer_hidden: object
    logits: object

    @property
    def next_token_id(self) -> int:
        import numpy as np

        return int(np.argmax(self.logits[-1]))

    @property
    def next_token_logit(self) -> float:
        return float(self.logits[-1, self.next_token_id])


@dataclass(frozen=True)
class StepFunPromptEmbedding:
    """Rendered/tokenized Step prompt plus resident BF16 embedding rows."""

    rendered_prompt: str
    input_ids: tuple[int, ...]
    embeddings_bf16: object

    @property
    def prompt_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class StepFunKVCacheAllocation:
    """Owned synthetic BF16 KV-cache buffers for StepFun decode bring-up."""

    buffers: tuple[DeviceBuffer, ...]
    context_pages: int
    page_size: int
    layer_nbytes: tuple[tuple[int, int], ...]

    @property
    def tokens(self) -> int:
        return self.context_pages * self.page_size

    @property
    def nbytes(self) -> int:
        return sum(key_bytes + value_bytes for key_bytes, value_bytes in self.layer_nbytes)

    @property
    def buffer_count(self) -> int:
        return len(self.buffers)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


@dataclass(frozen=True)
class StepFunInputIDDeviceUpload:
    """Owned device buffer for planned StepFun input-token uploads."""

    buffer: DeviceBuffer
    payload_sha256: str
    token_count: int
    dtype: str = "int32"

    def to_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "dtype": self.dtype,
            "ptr": self.buffer.ptr,
            "nbytes": self.buffer.nbytes,
            "sha256": self.payload_sha256,
        }

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        free(self.buffer, runtime=runtime)


@dataclass(frozen=True)
class StepFunKVSpanInputDeviceUpload:
    """Owned device buffers for planned StepFun KV span-input uploads."""

    buffers: Mapping[str, DeviceBuffer]
    payload_sha256: Mapping[str, str]
    total_nbytes: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.buffers.keys())

    @property
    def buffer_count(self) -> int:
        return len(self.buffers)

    def to_dict(self) -> dict[str, object]:
        return {
            "buffer_count": self.buffer_count,
            "total_nbytes": self.total_nbytes,
            "buffers": {
                name: {
                    "ptr": buffer.ptr,
                    "nbytes": buffer.nbytes,
                    "sha256": self.payload_sha256[name],
                }
                for name, buffer in self.buffers.items()
            },
        }

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for buffer in reversed(tuple(self.buffers.values())):
            free(buffer, runtime=runtime)


@dataclass(frozen=True)
class StepFunKVDecodeDeviceInputs:
    """Owned device buffers for metadata-only StepFun KV decode inputs."""

    input_ids: StepFunInputIDDeviceUpload
    span_inputs: StepFunKVSpanInputDeviceUpload

    @property
    def buffer_count(self) -> int:
        return 1 + self.span_inputs.buffer_count

    @property
    def total_nbytes(self) -> int:
        return self.input_ids.buffer.nbytes + self.span_inputs.total_nbytes

    def to_dict(self) -> dict[str, object]:
        return {
            "buffer_count": self.buffer_count,
            "total_nbytes": self.total_nbytes,
            "input_ids": self.input_ids.to_dict(),
            "span_inputs": self.span_inputs.to_dict(),
        }

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.span_inputs.free(runtime=runtime)
        self.input_ids.free(runtime=runtime)


@dataclass(frozen=True)
class StepFunKVDecodeKernelPlan:
    """Registry-key plan for the StepFun BF16 KV-backed decode path."""

    backend: str
    model_quant: str
    kv_storage_dtype: str
    dispatch_keys: Mapping[str, KernelKey]
    registered: Mapping[str, bool]
    decode_attention_kind: str
    max_context: int
    max_new_tokens: int
    attention_block_size: int
    attention_block_table_len: int
    max_prompt_rows: int
    decode_max_live_count: int

    @property
    def all_registered(self) -> bool:
        return all(bool(value) for value in self.registered.values())

    @property
    def attention_capacity_tokens(self) -> int:
        return self.attention_block_size * self.attention_block_table_len

    @property
    def decode_span_shape_compatible(self) -> bool:
        return 0 <= self.decode_max_live_count < self.attention_capacity_tokens

    @property
    def prompt_span_shape_compatible(self) -> bool:
        return self.max_prompt_rows > 0 and self.max_prompt_rows + self.max_new_tokens <= self.max_context

    @property
    def span_shape_compatible(self) -> bool:
        return self.decode_span_shape_compatible and self.prompt_span_shape_compatible

    @property
    def decode_span_contract(self) -> dict[str, object]:
        return {
            "block_size": self.attention_block_size,
            "block_table_len": self.attention_block_table_len,
            "live_counts_len": 1,
            "max_live_count": self.decode_max_live_count,
            "capacity_tokens": self.attention_capacity_tokens,
            "shape_compatible": self.decode_span_shape_compatible,
        }

    @property
    def prompt_span_contract(self) -> dict[str, object]:
        return {
            "block_size": self.attention_block_size,
            "max_prompt_rows": self.max_prompt_rows,
            "block_table_len_per_row": self.attention_block_table_len,
            "base_offsets_len_formula": f"rows * {self.attention_block_table_len}",
            "live_counts_len_formula": "rows",
            "row_positions_required": True,
            "shape_compatible": self.prompt_span_shape_compatible,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model_quant": self.model_quant,
            "kv_storage_dtype": self.kv_storage_dtype,
            "decode_attention_kind": self.decode_attention_kind,
            "max_context": self.max_context,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_rows": self.max_prompt_rows,
            "attention_block_size": self.attention_block_size,
            "attention_block_table_len": self.attention_block_table_len,
            "attention_capacity_tokens": self.attention_capacity_tokens,
            "decode_span": self.decode_span_contract,
            "prompt_span": self.prompt_span_contract,
            "decode_span_shape_compatible": self.decode_span_shape_compatible,
            "prompt_span_shape_compatible": self.prompt_span_shape_compatible,
            "span_shape_compatible": self.span_shape_compatible,
            "dispatch_keys": {
                name: _kernel_key_to_dict(key) for name, key in self.dispatch_keys.items()
            },
            "registered": dict(self.registered),
            "all_registered": self.all_registered,
            "note": (
                "These are registry-key checks for the planned StepFun BF16 KV write/decode path. "
                "They do not claim that the streaming runner or oracle parity is complete."
            ),
        }


@dataclass(frozen=True)
class StepFunTextDecodeResourcePlan:
    """Resident-weight and KV-cache byte plan for text-only Step decode."""

    slot_paths: tuple[str, ...]
    resident_weight_nbytes: int
    kv_layer_nbytes: tuple[tuple[int, int], ...]
    context_pages: int
    page_size: int
    max_new_tokens: int
    backend: str
    kv_decode_kernel_plan: StepFunKVDecodeKernelPlan

    @classmethod
    def from_model_map(
        cls,
        model_map: StepFunGGUFModelMap,
        *,
        backend: str,
        context_pages: int,
        page_size: int,
        max_new_tokens: int = DEFAULT_STEPFUN_MAX_NEW_TOKENS,
    ) -> "StepFunTextDecodeResourcePlan":
        slots = stepfun_text_decode_slot_paths(model_map)
        resident_nbytes = sum(stepfun_slot_tensor(model_map, slot).nbytes for slot in slots)
        return cls(
            slot_paths=slots,
            resident_weight_nbytes=int(resident_nbytes),
            kv_layer_nbytes=stepfun_kv_cache_layer_nbytes(
                model_map.config,
                context_pages=context_pages,
                page_size=page_size,
            ),
            context_pages=int(context_pages),
            page_size=int(page_size),
            max_new_tokens=int(max_new_tokens),
            backend=backend,
            kv_decode_kernel_plan=stepfun_kv_decode_kernel_plan(
                backend=backend,
                max_context=int(context_pages) * int(page_size),
                max_new_tokens=int(max_new_tokens),
            ),
        )

    @property
    def slot_count(self) -> int:
        return len(self.slot_paths)

    @property
    def kv_nbytes(self) -> int:
        return sum(key_bytes + value_bytes for key_bytes, value_bytes in self.kv_layer_nbytes)

    @property
    def total_nbytes(self) -> int:
        return self.resident_weight_nbytes + self.kv_nbytes

    @property
    def resident_weight_gib(self) -> float:
        return self.resident_weight_nbytes / 2**30

    @property
    def kv_gib(self) -> float:
        return self.kv_nbytes / 2**30

    @property
    def total_gib(self) -> float:
        return self.total_nbytes / 2**30

    @property
    def streaming_runner_blockers(self) -> tuple[dict[str, object], ...]:
        return stepfun_streaming_runner_blockers()

    @property
    def kv_decode_launch_schedule(self) -> dict[str, object]:
        """Return the planned per-layer KV launch order for streaming decode."""

        layer_count = len(self.kv_layer_nbytes)
        per_layer_order = ["prompt_kv_write", "decode_kv_write", "decode_attention"]
        streaming_runner_blockers = list(self.streaming_runner_blockers)
        streaming_runner_blocker_names = [str(blocker["name"]) for blocker in streaming_runner_blockers]
        first_streaming_runner_blocker = (
            streaming_runner_blocker_names[0] if streaming_runner_blocker_names else None
        )
        first_streaming_runner_blocker_sha256 = (
            _stable_json_sha256(first_streaming_runner_blocker)
            if first_streaming_runner_blocker is not None
            else None
        )
        return {
            "source": "text_decode_resource_plan",
            "layer_count": layer_count,
            "operation_count": layer_count * len(per_layer_order),
            "per_layer_order": per_layer_order,
            "stages": [
                {
                    "name": "prompt_prefill_kv_write",
                    "dispatch_key": "prompt_kv_write",
                    "span_contract": "prompt_span",
                    "layer_count": layer_count,
                    "ready": self.kv_decode_kernel_plan.registered.get("prompt_kv_write") is True,
                },
                {
                    "name": "one_token_decode_kv_write",
                    "dispatch_key": "decode_kv_write",
                    "span_contract": "decode_span",
                    "layer_count": layer_count,
                    "ready": self.kv_decode_kernel_plan.registered.get("decode_kv_write") is True,
                },
                {
                    "name": "one_token_gated_attention_decode",
                    "dispatch_key": "decode_attention",
                    "span_contract": "decode_span",
                    "layer_count": layer_count,
                    "ready": self.kv_decode_kernel_plan.registered.get("decode_attention") is True,
                },
            ],
            "first_layer_ops": [f"layers.0.{name}" for name in per_layer_order],
            "last_layer_ops": [f"layers.{layer_count - 1}.{name}" for name in per_layer_order]
            if layer_count
            else [],
            "all_stage_dispatch_ready": self.kv_decode_kernel_plan.all_registered,
            "streaming_runner_ready": False,
            "streaming_runner_blocker_count": len(streaming_runner_blockers),
            "streaming_runner_blocker_names": streaming_runner_blocker_names,
            "streaming_runner_blocker_names_sha256": _stable_json_sha256(
                streaming_runner_blocker_names
            ),
            "first_streaming_runner_blocker": first_streaming_runner_blocker,
            "first_streaming_runner_blocker_sha256": first_streaming_runner_blocker_sha256,
            "streaming_runner_blockers": streaming_runner_blockers,
            "note": (
                "Planned launch order for the future StepFun streaming KV-backed decode runner. "
                "Current prompt smokes remain host-composed until streaming_runner_ready is true."
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "slot_count": self.slot_count,
            "slot_paths": list(self.slot_paths),
            "resident_weight_nbytes": self.resident_weight_nbytes,
            "resident_weight_gib": self.resident_weight_gib,
            "context_pages": self.context_pages,
            "page_size": self.page_size,
            "max_new_tokens": self.max_new_tokens,
            "kv_buffer_count": len(self.kv_layer_nbytes) * 2,
            "kv_layer_nbytes": [
                {"layer": idx, "key_nbytes": key, "value_nbytes": value}
                for idx, (key, value) in enumerate(self.kv_layer_nbytes)
            ],
            "kv_nbytes": self.kv_nbytes,
            "kv_gib": self.kv_gib,
            "total_nbytes": self.total_nbytes,
            "total_gib": self.total_gib,
            "kv_decode_kernel_plan": self.kv_decode_kernel_plan.to_dict(),
            "kv_decode_launch_schedule": self.kv_decode_launch_schedule,
        }


@dataclass(frozen=True)
class StepFunKVDecodeRunPlan:
    """Metadata-only prompt/resource binding for StepFun KV decode bring-up."""

    decode_plan: StepFunDecodePlan
    resource_plan: StepFunTextDecodeResourcePlan

    @property
    def prompt_length(self) -> int:
        return self.decode_plan.prompt_length

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.decode_plan.input_ids

    @property
    def input_ids_dtype(self) -> str:
        return "int32"

    @property
    def input_ids_payload_bytes(self) -> bytes:
        return _pack_integer_payload(self.input_ids_dtype, self.input_ids)

    @property
    def input_ids_nbytes(self) -> int:
        return len(self.input_ids_payload_bytes)

    @property
    def input_ids_sha256(self) -> str:
        return hashlib.sha256(self.input_ids_payload_bytes).hexdigest()

    @property
    def rendered_prompt_nchars(self) -> int:
        return len(self.decode_plan.rendered_prompt)

    @property
    def rendered_prompt_sha256(self) -> str:
        return hashlib.sha256(self.decode_plan.rendered_prompt.encode("utf-8")).hexdigest()

    @property
    def prompt_positions(self) -> tuple[int, ...]:
        return tuple(range(self.prompt_length))

    @property
    def decode_position(self) -> int:
        return self.prompt_length

    @property
    def decode_live_count(self) -> int:
        return self.prompt_length

    @property
    def required_context_tokens(self) -> int:
        return self.prompt_length + self.decode_plan.max_new_tokens

    @property
    def max_prompt_rows(self) -> int:
        return self.resource_plan.kv_decode_kernel_plan.max_prompt_rows

    @property
    def attention_block_size(self) -> int:
        return self.resource_plan.kv_decode_kernel_plan.attention_block_size

    @property
    def attention_block_table_len(self) -> int:
        return self.resource_plan.kv_decode_kernel_plan.attention_block_table_len

    @property
    def prompt_span_base_offsets(self) -> tuple[int, ...]:
        row_block_table = tuple(range(self.attention_block_table_len))
        return tuple(value for _ in self.prompt_positions for value in row_block_table)

    @property
    def decode_span_base_offsets(self) -> tuple[int, ...]:
        return tuple(range(self.attention_block_table_len))

    @property
    def prompt_span_inputs(self) -> dict[str, object]:
        base_offsets_len = len(self.prompt_span_base_offsets)
        live_counts_len = self.prompt_length
        base_offsets_nbytes = base_offsets_len * 4
        live_counts_nbytes = live_counts_len * 8
        return {
            "rows": self.prompt_length,
            "block_size": self.attention_block_size,
            "block_table_len_per_row": self.attention_block_table_len,
            "base_offsets": list(self.prompt_span_base_offsets),
            "base_offsets_dtype": "int32",
            "base_offsets_len": base_offsets_len,
            "base_offsets_nbytes": base_offsets_nbytes,
            "live_counts": list(self.prompt_positions),
            "live_counts_dtype": "int64",
            "live_counts_len": live_counts_len,
            "live_counts_nbytes": live_counts_nbytes,
            "position_tensor_role": "prompt_row_positions",
            "max_live_count": max(self.prompt_positions) if self.prompt_positions else 0,
            "total_span_input_nbytes": base_offsets_nbytes + live_counts_nbytes,
        }

    @property
    def decode_span_inputs(self) -> dict[str, object]:
        base_offsets_len = len(self.decode_span_base_offsets)
        base_offsets_nbytes = base_offsets_len * 4
        attention_live_counts_nbytes = 8
        return {
            "block_size": self.attention_block_size,
            "block_table_len": self.attention_block_table_len,
            "base_offsets": list(self.decode_span_base_offsets),
            "base_offsets_dtype": "int32",
            "base_offsets_len": base_offsets_len,
            "base_offsets_nbytes": base_offsets_nbytes,
            "kv_write_position": self.decode_position,
            "kv_write_position_dtype": "int64",
            "kv_write_position_nbytes": 8,
            "attention_live_counts": [self.decode_live_count],
            "attention_live_counts_dtype": "int64",
            "attention_live_counts_len": 1,
            "attention_live_counts_nbytes": attention_live_counts_nbytes,
            "max_live_count": self.decode_live_count,
            "total_span_input_nbytes": base_offsets_nbytes + attention_live_counts_nbytes,
        }

    @property
    def span_input_total_nbytes(self) -> int:
        return int(self.prompt_span_inputs["total_span_input_nbytes"]) + int(
            self.decode_span_inputs["total_span_input_nbytes"]
        )

    @property
    def span_input_upload_manifest(self) -> dict[str, object]:
        prompt_inputs = self.prompt_span_inputs
        decode_inputs = self.decode_span_inputs
        entries = [
            {
                "name": "prompt_base_offsets",
                "source": "prompt_span_inputs.base_offsets",
                "kernel_args": ["prompt_kv_write.base_offsets"],
                "dtype": prompt_inputs["base_offsets_dtype"],
                "shape": [self.prompt_length, self.attention_block_table_len],
                "nbytes": prompt_inputs["base_offsets_nbytes"],
            },
            {
                "name": "prompt_live_counts",
                "source": "prompt_span_inputs.live_counts",
                "kernel_args": ["prompt_kv_write.live_counts"],
                "dtype": prompt_inputs["live_counts_dtype"],
                "shape": [self.prompt_length],
                "nbytes": prompt_inputs["live_counts_nbytes"],
            },
            {
                "name": "decode_base_offsets",
                "source": "decode_span_inputs.base_offsets",
                "kernel_args": ["decode_kv_write.base_offsets", "decode_attention.base_offsets"],
                "dtype": decode_inputs["base_offsets_dtype"],
                "shape": [self.attention_block_table_len],
                "nbytes": decode_inputs["base_offsets_nbytes"],
            },
            {
                "name": "decode_kv_write_position",
                "source": "decode_span_inputs.kv_write_position",
                "kernel_args": ["decode_kv_write.position"],
                "dtype": decode_inputs["kv_write_position_dtype"],
                "shape": [],
                "nbytes": decode_inputs["kv_write_position_nbytes"],
            },
            {
                "name": "decode_attention_live_counts",
                "source": "decode_span_inputs.attention_live_counts",
                "kernel_args": ["decode_attention.live_counts"],
                "dtype": decode_inputs["attention_live_counts_dtype"],
                "shape": [1],
                "nbytes": decode_inputs["attention_live_counts_nbytes"],
            },
        ]
        return {
            "entries": entries,
            "entry_count": len(entries),
            "total_nbytes": sum(int(entry["nbytes"]) for entry in entries),
            "note": "Host-side upload manifest for metadata-only StepFun KV decode planning.",
        }

    def _span_input_values_for_source(self, source: str) -> list[int]:
        if source == "prompt_span_inputs.base_offsets":
            return list(self.prompt_span_base_offsets)
        if source == "prompt_span_inputs.live_counts":
            return list(self.prompt_positions)
        if source == "decode_span_inputs.base_offsets":
            return list(self.decode_span_base_offsets)
        if source == "decode_span_inputs.kv_write_position":
            return [self.decode_position]
        if source == "decode_span_inputs.attention_live_counts":
            return [self.decode_live_count]
        raise KeyError(f"unknown StepFun KV upload source: {source}")

    def upload_input_ids_payload(
        self,
        *,
        runtime: HipRuntime | None = None,
    ) -> StepFunInputIDDeviceUpload:
        """Allocate/copy the planned input-token payload to a device buffer."""

        payload = self.input_ids_payload_bytes
        buffer = malloc(len(payload), runtime=runtime)
        try:
            host_payload = ctypes.create_string_buffer(payload, len(payload))
            copy_host_to_device(
                buffer,
                host_buffer_ptr(host_payload),
                len(payload),
                runtime=runtime,
            )
        except Exception:
            free(buffer, runtime=runtime)
            raise
        return StepFunInputIDDeviceUpload(
            buffer=buffer,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            token_count=self.prompt_length,
            dtype=self.input_ids_dtype,
        )

    @property
    def span_input_host_payload_bytes(self) -> dict[str, bytes]:
        """Return little-endian host bytes for each planned KV span upload."""

        payloads: dict[str, bytes] = {}
        for manifest_entry in self.span_input_upload_manifest["entries"]:
            source = str(manifest_entry["source"])
            dtype = str(manifest_entry["dtype"])
            values = self._span_input_values_for_source(source)
            payloads[str(manifest_entry["name"])] = _pack_integer_payload(dtype, values)
        return payloads

    @property
    def span_input_host_payloads(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        payload_bytes = self.span_input_host_payload_bytes
        for manifest_entry in self.span_input_upload_manifest["entries"]:
            source = str(manifest_entry["source"])
            dtype = str(manifest_entry["dtype"])
            values = self._span_input_values_for_source(source)
            payload = payload_bytes[str(manifest_entry["name"])]
            entries.append(
                {
                    "name": manifest_entry["name"],
                    "source": source,
                    "dtype": dtype,
                    "byte_order": "little",
                    "value_count": len(values),
                    "nbytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "preview_values": values[:8],
                }
            )
        return {
            "entries": entries,
            "entry_count": len(entries),
            "total_nbytes": sum(int(entry["nbytes"]) for entry in entries),
            "note": "Deterministic little-endian host payload hashes for StepFun KV span inputs.",
        }

    def upload_span_input_payloads(
        self,
        *,
        runtime: HipRuntime | None = None,
    ) -> StepFunKVSpanInputDeviceUpload:
        """Allocate/copy planned span-input payloads to device buffers."""

        payloads = self.span_input_host_payload_bytes
        buffers: dict[str, DeviceBuffer] = {}
        payload_sha256: dict[str, str] = {}
        try:
            for name, payload in payloads.items():
                buffer = malloc(len(payload), runtime=runtime)
                buffers[name] = buffer
                host_payload = ctypes.create_string_buffer(payload, len(payload))
                copy_host_to_device(
                    buffer,
                    host_buffer_ptr(host_payload),
                    len(payload),
                    runtime=runtime,
                )
                payload_sha256[name] = hashlib.sha256(payload).hexdigest()
        except Exception:
            for buffer in reversed(tuple(buffers.values())):
                free(buffer, runtime=runtime)
            raise
        return StepFunKVSpanInputDeviceUpload(
            buffers=buffers,
            payload_sha256=payload_sha256,
            total_nbytes=sum(len(payload) for payload in payloads.values()),
        )

    def upload_decode_inputs(
        self,
        *,
        runtime: HipRuntime | None = None,
    ) -> StepFunKVDecodeDeviceInputs:
        """Allocate/copy all planned pre-runner KV decode inputs."""

        input_upload: StepFunInputIDDeviceUpload | None = None
        try:
            input_upload = self.upload_input_ids_payload(runtime=runtime)
            span_upload = self.upload_span_input_payloads(runtime=runtime)
        except Exception:
            if input_upload is not None:
                input_upload.free(runtime=runtime)
            raise
        return StepFunKVDecodeDeviceInputs(
            input_ids=input_upload,
            span_inputs=span_upload,
        )

    @property
    def prompt_fits_resource_plan(self) -> bool:
        return self.prompt_length <= self.max_prompt_rows

    @property
    def context_fits_resource_plan(self) -> bool:
        return self.required_context_tokens <= self.resource_plan.kv_decode_kernel_plan.max_context

    @property
    def streaming_runner_ready(self) -> bool:
        return False

    @property
    def streaming_runner_blockers(self) -> tuple[dict[str, object], ...]:
        return self.resource_plan.streaming_runner_blockers

    @property
    def decode_input_upload_plan(self) -> dict[str, object]:
        span_manifest = self.span_input_upload_manifest
        entries = [
            {
                "name": "input_ids",
                "source": "input_ids",
                "upload_group": "input_tokens",
                "dtype": self.input_ids_dtype,
                "shape": [self.prompt_length],
                "nbytes": self.input_ids_nbytes,
                "sha256": self.input_ids_sha256,
            },
            *[
                {
                    "name": str(entry["name"]),
                    "source": str(entry["source"]),
                    "upload_group": "kv_span_inputs",
                    "dtype": str(entry["dtype"]),
                    "shape": list(entry["shape"]),
                    "nbytes": int(entry["nbytes"]),
                    "sha256": str(payload_entry["sha256"]),
                }
                for entry, payload_entry in zip(
                    span_manifest["entries"],
                    self.span_input_host_payloads["entries"],
                    strict=True,
                )
            ],
        ]
        upload_order = [str(entry["name"]) for entry in entries]
        cleanup_order = [str(entry["name"]) for entry in reversed(entries)]
        input_token_nbytes = self.input_ids_nbytes
        span_input_nbytes = int(span_manifest["total_nbytes"])
        total_nbytes = input_token_nbytes + span_input_nbytes
        span_payload_entries = list(self.span_input_host_payloads["entries"])
        consistency_checks = {
            "entry_count_matches_upload_order": len(entries) == len(upload_order),
            "cleanup_order_reverses_upload_order": cleanup_order == list(reversed(upload_order)),
            "entry_total_nbytes_matches": sum(int(entry["nbytes"]) for entry in entries)
            == total_nbytes,
            "input_token_hash_matches": str(entries[0]["sha256"]) == self.input_ids_sha256,
            "span_payload_hashes_match_manifest": all(
                str(entry["sha256"]) == str(payload_entry["sha256"])
                for entry, payload_entry in zip(entries[1:], span_payload_entries, strict=True)
            ),
        }
        return {
            "entries": entries,
            "entry_count": len(entries),
            "upload_order": upload_order,
            "cleanup_order": cleanup_order,
            "input_token_nbytes": input_token_nbytes,
            "span_input_nbytes": span_input_nbytes,
            "total_nbytes": total_nbytes,
            "consistency_checks": consistency_checks,
            "all_consistency_checks_passed": all(consistency_checks.values()),
            "streaming_runner_ready": self.streaming_runner_ready,
            "note": "Metadata-only combined upload plan; no kernels are launched.",
        }

    @property
    def streaming_decode_loop_blueprint(self) -> dict[str, object]:
        """Return the metadata-only contract for the future KV streaming loop."""

        launch_schedule = self.resource_plan.kv_decode_launch_schedule
        per_layer_order = list(launch_schedule["per_layer_order"])
        layer_count = int(launch_schedule["layer_count"])
        operation_sequence = [
            f"layers.{layer_id}.{op_name}"
            for layer_id in range(layer_count)
            for op_name in per_layer_order
        ]
        upload_plan = self.decode_input_upload_plan
        streaming_runner_blockers = list(self.streaming_runner_blockers)
        streaming_runner_blocker_names = [
            str(blocker["name"]) for blocker in streaming_runner_blockers
        ]
        first_streaming_runner_blocker = (
            streaming_runner_blocker_names[0] if streaming_runner_blocker_names else None
        )
        first_streaming_runner_blocker_sha256 = (
            _stable_json_sha256(first_streaming_runner_blocker)
            if first_streaming_runner_blocker is not None
            else None
        )
        return {
            "source": "kv_decode_run_plan",
            "executable": False,
            "blocked_by": first_streaming_runner_blocker,
            "blocked_by_sha256": first_streaming_runner_blocker_sha256,
            "streaming_runner_ready": self.streaming_runner_ready,
            "layer_count": layer_count,
            "operation_count": len(operation_sequence),
            "per_layer_order": per_layer_order,
            "operation_sequence_sha256": _stable_json_sha256(operation_sequence),
            "first_layer_ops": operation_sequence[: len(per_layer_order)],
            "last_layer_ops": operation_sequence[-len(per_layer_order) :] if operation_sequence else [],
            "pre_run_upload_order": list(upload_plan["upload_order"]),
            "pre_run_cleanup_order": list(upload_plan["cleanup_order"]),
            "pre_run_upload_checks_passed": upload_plan["all_consistency_checks_passed"],
            "stages": [
                {
                    "name": "upload_decode_inputs",
                    "source": "decode_input_upload_plan",
                    "ready": upload_plan["all_consistency_checks_passed"],
                    "entry_count": upload_plan["entry_count"],
                    "total_nbytes": upload_plan["total_nbytes"],
                },
                *[
                    {
                        "name": str(stage["name"]),
                        "dispatch_key": str(stage["dispatch_key"]),
                        "span_contract": str(stage["span_contract"]),
                        "layer_count": int(stage["layer_count"]),
                        "ready": bool(stage["ready"]),
                    }
                    for stage in launch_schedule["stages"]
                ],
            ],
            "stage_count": 1 + len(launch_schedule["stages"]),
            "note": (
                "Metadata-only StepFun KV streaming decode loop contract. "
                "It records the upload and launch order but does not launch kernels."
            ),
        }

    @property
    def streaming_decode_loop_status(self) -> dict[str, object]:
        """Return compact readiness metadata for the future KV streaming loop."""

        blueprint = self.streaming_decode_loop_blueprint
        streaming_runner_blockers = list(self.streaming_runner_blockers)
        blocker_names = [str(blocker["name"]) for blocker in streaming_runner_blockers]
        return {
            "source": "kv_decode_run_plan",
            "ready": self.streaming_runner_ready,
            "executable": bool(blueprint["executable"]),
            "blocked_by": blueprint["blocked_by"],
            "blocked_by_sha256": blueprint["blocked_by_sha256"],
            "blocker_count": len(streaming_runner_blockers),
            "blocker_names": blocker_names,
            "blocker_names_sha256": _stable_json_sha256(blocker_names),
            "blueprint_operation_count": blueprint["operation_count"],
            "blueprint_stage_count": blueprint["stage_count"],
            "blueprint_sha256": _stable_json_sha256(blueprint),
            "next_action": (
                "wire_streaming_decode_loop" if not self.streaming_runner_ready else None
            ),
            "note": (
                "Metadata-only readiness summary for the future StepFun KV streaming "
                "decode loop; no kernels are launched."
            ),
        }

    @property
    def streaming_decode_launch_trace(self) -> dict[str, object]:
        """Return the metadata-only per-layer launch trace for the KV loop."""

        blueprint = self.streaming_decode_loop_blueprint
        launch_schedule = self.resource_plan.kv_decode_launch_schedule
        per_layer_order = list(blueprint["per_layer_order"])
        layer_count = int(blueprint["layer_count"])
        stage_by_dispatch_key = {
            str(stage["dispatch_key"]): stage for stage in launch_schedule["stages"]
        }
        span_uploads_by_operation = {
            "prompt_kv_write": ["prompt_base_offsets", "prompt_live_counts"],
            "decode_kv_write": ["decode_base_offsets", "decode_kv_write_position"],
            "decode_attention": ["decode_base_offsets", "decode_attention_live_counts"],
        }
        runtime_inputs_by_operation = {
            "prompt_kv_write": ["layer_prompt_key", "layer_prompt_value"],
            "decode_kv_write": ["layer_decode_key", "layer_decode_value"],
            "decode_attention": ["layer_decode_query", "layer_decode_attention_gate"],
        }
        records: list[dict[str, object]] = []
        for layer_id in range(layer_count):
            for op_name in per_layer_order:
                stage = dict(stage_by_dispatch_key[op_name])
                operation_name = f"layers.{layer_id}.{op_name}"
                records.append(
                    {
                        "op_index": len(records),
                        "operation": operation_name,
                        "layer": layer_id,
                        "name": op_name,
                        "stage_name": stage["name"],
                        "dispatch_key_name": op_name,
                        "kernel_key": _kernel_key_to_dict(
                            self.decode_plan.kv_dispatch_keys[op_name]
                        ),
                        "span_contract": stage["span_contract"],
                        "pre_run_uploads": list(span_uploads_by_operation[op_name]),
                        "expected_runtime_inputs": list(
                            runtime_inputs_by_operation[op_name]
                        ),
                        "launch_ready": bool(stage["ready"]),
                        "execution_status": "not_launched_metadata_only",
                        "blocked_by": blueprint["blocked_by"],
                    }
                )
        operation_sequence = [str(record["operation"]) for record in records]
        return {
            "schema_version": 1,
            "source": "kv_decode_run_plan",
            "executable": False,
            "ready": self.streaming_runner_ready,
            "blocked_by": blueprint["blocked_by"],
            "blocked_by_sha256": blueprint["blocked_by_sha256"],
            "layer_count": layer_count,
            "per_layer_order": per_layer_order,
            "operation_count": len(records),
            "operation_sequence_sha256": _stable_json_sha256(operation_sequence),
            "operation_records_sha256": _stable_json_sha256(records),
            "first_operation": records[0] if records else None,
            "last_operation": records[-1] if records else None,
            "span_uploads_by_operation": span_uploads_by_operation,
            "pre_run_upload_order": list(blueprint["pre_run_upload_order"]),
            "all_launches_have_dispatch_keys": all(
                bool(record["kernel_key"]) for record in records
            ),
            "all_launches_ready": all(bool(record["launch_ready"]) for record in records),
            "no_kernel_launches": True,
            "operation_records": records,
            "note": (
                "Metadata-only per-layer launch trace for the future StepFun KV "
                "streaming decode loop; it records dispatch/span/upload contracts but "
                "does not launch kernels or produce a token."
            ),
        }

    @property
    def kv_decode_blocker_summary(self) -> dict[str, object]:
        """Return machine-readable blocker evidence for the KV decode runner."""

        upload_plan = self.decode_input_upload_plan
        blueprint = self.streaming_decode_loop_blueprint
        streaming_runner_blockers = list(self.streaming_runner_blockers)
        blocker_names = [str(blocker["name"]) for blocker in streaming_runner_blockers]
        first_blocker = streaming_runner_blockers[0] if streaming_runner_blockers else None
        first_blocker_name = str(first_blocker["name"]) if first_blocker else None
        artifacts_needed = [
            {
                "name": "kv_kernel_trace_artifact",
                "required_for": "kv_kernel_trace_artifact_missing",
                "evidence": (
                    "rocprofv3 or equivalent trace showing prompt KV write, decode KV write, "
                    "and gated decode-attention kernels for the canonical prompt"
                ),
            },
            {
                "name": "kv_backed_next_token_artifact",
                "required_for": "kv_backed_next_token_artifact_missing",
                "evidence": (
                    "one-token decode artifact recording generated token/logit path from KV-backed "
                    "runtime execution, not host-composed layer-prefix outputs"
                ),
            },
        ]
        return {
            "schema_version": 1,
            "source": "kv_decode_run_plan",
            "status": "blocked" if streaming_runner_blockers else "ready",
            "ready": self.streaming_runner_ready,
            "executable": bool(blueprint["executable"]),
            "next_action": "wire_streaming_decode_loop" if streaming_runner_blockers else None,
            "blocker_count": len(streaming_runner_blockers),
            "blocker_names": blocker_names,
            "blocker_names_sha256": _stable_json_sha256(blocker_names),
            "first_blocker": first_blocker,
            "first_blocker_name": first_blocker_name,
            "first_blocker_sha256": _stable_json_sha256(first_blocker)
            if first_blocker is not None
            else None,
            "upload_plan_ready": bool(upload_plan["all_consistency_checks_passed"]),
            "upload_entry_count": int(upload_plan["entry_count"]),
            "upload_total_nbytes": int(upload_plan["total_nbytes"]),
            "launch_blueprint_ready": bool(blueprint["pre_run_upload_checks_passed"]),
            "launch_stage_count": int(blueprint["stage_count"]),
            "launch_operation_count": int(blueprint["operation_count"]),
            "per_layer_order": list(blueprint["per_layer_order"]),
            "artifacts_needed": artifacts_needed,
            "artifacts_needed_sha256": _stable_json_sha256(artifacts_needed),
            "artifact_count": len(artifacts_needed),
            "no_claim_policy": {
                "oracle_parity_claim_allowed": False,
                "kv_backed_decode_claim_allowed": False,
                "performance_claim_allowed": False,
                "reason": (
                    "metadata-only KV decode planning is not a streaming decode execution and "
                    "does not generate a token/logit artifact"
                ),
            },
        }

    def to_dict(self) -> dict[str, object]:
        launch_schedule = self.resource_plan.kv_decode_launch_schedule
        streaming_runner_blockers = list(self.streaming_runner_blockers)
        streaming_runner_blocker_names = [str(blocker["name"]) for blocker in streaming_runner_blockers]
        first_streaming_runner_blocker = (
            streaming_runner_blocker_names[0] if streaming_runner_blocker_names else None
        )
        first_streaming_runner_blocker_sha256 = (
            _stable_json_sha256(first_streaming_runner_blocker)
            if first_streaming_runner_blocker is not None
            else None
        )
        return {
            "prompt_length": self.prompt_length,
            "input_ids": list(self.input_ids),
            "input_ids_dtype": self.input_ids_dtype,
            "input_ids_nbytes": self.input_ids_nbytes,
            "input_ids_sha256": self.input_ids_sha256,
            "input_id_count": len(self.input_ids),
            "input_id_preview": list(self.input_ids[:8]),
            "rendered_prompt_nchars": self.rendered_prompt_nchars,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "max_new_tokens": self.decode_plan.max_new_tokens,
            "required_context_tokens": self.required_context_tokens,
            "max_context": self.resource_plan.kv_decode_kernel_plan.max_context,
            "max_prompt_rows": self.max_prompt_rows,
            "attention_block_size": self.attention_block_size,
            "attention_block_table_len": self.attention_block_table_len,
            "prompt_positions": list(self.prompt_positions),
            "decode_position": self.decode_position,
            "decode_live_count": self.decode_live_count,
            "prompt_span_inputs": self.prompt_span_inputs,
            "decode_span_inputs": self.decode_span_inputs,
            "span_input_total_nbytes": self.span_input_total_nbytes,
            "span_input_upload_manifest": self.span_input_upload_manifest,
            "span_input_host_payloads": self.span_input_host_payloads,
            "decode_input_upload_plan": self.decode_input_upload_plan,
            "streaming_decode_loop_blueprint": self.streaming_decode_loop_blueprint,
            "streaming_decode_loop_status": self.streaming_decode_loop_status,
            "streaming_decode_launch_trace": self.streaming_decode_launch_trace,
            "kv_decode_blocker_summary": self.kv_decode_blocker_summary,
            "prompt_fits_resource_plan": self.prompt_fits_resource_plan,
            "context_fits_resource_plan": self.context_fits_resource_plan,
            "stop_token_ids": list(self.decode_plan.stop_token_ids),
            "kv_dispatch_keys": {
                name: _kernel_key_to_dict(key) for name, key in self.decode_plan.kv_dispatch_keys.items()
            },
            "kv_decode_launch_operation_count": launch_schedule["operation_count"],
            "kv_decode_launch_per_layer_order": list(launch_schedule["per_layer_order"]),
            "streaming_runner_ready": self.streaming_runner_ready,
            "streaming_runner_blocker_count": len(streaming_runner_blockers),
            "streaming_runner_blocker_names": streaming_runner_blocker_names,
            "streaming_runner_blocker_names_sha256": _stable_json_sha256(
                streaming_runner_blocker_names
            ),
            "first_streaming_runner_blocker": first_streaming_runner_blocker,
            "first_streaming_runner_blocker_sha256": first_streaming_runner_blocker_sha256,
            "streaming_runner_blockers": streaming_runner_blockers,
            "note": (
                "Metadata-only prompt/resource binding for the future StepFun KV-backed decode runner. "
                "It does not launch KV kernels or claim oracle parity."
            ),
        }


@dataclass(frozen=True)
class StepFunShortContextDecodePlanner:
    """Pre-run planner for StepFun text-only c=1 decode.

    This is intentionally not the full model runner. It binds the pieces that
    must be stable before streaming decode: split GGUF metadata, tokenizer/chat
    rendering, short-context limits, multi-EOS stopping, and mixed-quant kernel
    registry keys. The full P11 runner can consume this plan once resident weight
    materialization and all layer dispatch paths are wired.
    """

    info: GGUFSplitModelInfo
    model_map: StepFunGGUFModelMap
    tokenizer: StepFunGGUFTokenizer
    backend: str = "hip_gfx1151"
    max_context: int = DEFAULT_STEPFUN_SHORT_CONTEXT
    max_new_tokens: int = DEFAULT_STEPFUN_MAX_NEW_TOKENS

    @classmethod
    def from_gguf_paths(
        cls,
        paths: Sequence[str | Path],
        *,
        backend: str = "hip_gfx1151",
        max_context: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
        max_new_tokens: int = DEFAULT_STEPFUN_MAX_NEW_TOKENS,
    ) -> "StepFunShortContextDecodePlanner":
        if max_context <= 0:
            raise ValueError("max_context must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        info = scan_gguf_splits(tuple(Path(path) for path in paths))
        model_map = build_stepfun_gguf_tensor_map(info)
        tokenizer = StepFunGGUFTokenizer.from_gguf_info(info)
        return cls(
            info=info,
            model_map=model_map,
            tokenizer=tokenizer,
            backend=backend,
            max_context=int(max_context),
            max_new_tokens=int(max_new_tokens),
        )

    def plan_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
    ) -> StepFunDecodePlan:
        rendered = self.tokenizer.render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )
        input_ids = tuple(self.tokenizer.encode(rendered, add_bos=False))
        self._validate_short_context(input_ids)
        return StepFunDecodePlan(
            input_ids=input_ids,
            rendered_prompt=rendered,
            stop_token_ids=self.tokenizer.eos_token_ids,
            max_context=self.max_context,
            max_new_tokens=self.max_new_tokens,
            backend=self.backend,
            quant_dispatch_keys=self.resolve_quant_dispatch_keys(),
            kv_dispatch_keys=self.resolve_kv_dispatch_keys(),
        )

    def plan_kv_decode_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
        context_pages: int = 1,
        page_size: int | None = None,
    ) -> StepFunKVDecodeRunPlan:
        """Plan the prompt/resource inputs for the future KV-backed decode runner."""

        decode_plan = self.plan_chat(
            messages,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
        )
        resource_plan = self.text_decode_resource_plan(
            context_pages=context_pages,
            page_size=page_size,
        )
        run_plan = StepFunKVDecodeRunPlan(
            decode_plan=decode_plan,
            resource_plan=resource_plan,
        )
        if not run_plan.prompt_fits_resource_plan:
            raise ValueError(
                "StepFun prompt does not fit KV prompt span: "
                f"prompt_length={run_plan.prompt_length} max_prompt_rows={run_plan.max_prompt_rows}"
            )
        if not run_plan.context_fits_resource_plan:
            raise ValueError(
                "StepFun prompt+decode does not fit KV context span: "
                f"required_context_tokens={run_plan.required_context_tokens} "
                f"max_context={resource_plan.kv_decode_kernel_plan.max_context}"
            )
        return run_plan

    def text_decode_resource_plan(
        self,
        *,
        context_pages: int = 1,
        page_size: int | None = None,
    ) -> StepFunTextDecodeResourcePlan:
        """Estimate resident text weights plus BF16 KV-cache bytes.

        The plan is metadata-only and torch-free; it does not allocate HIP
        memory. It mirrors the slot set used by the text-only resident runner so
        load-smoke memory snapshots can be compared against expected bytes.
        """

        page = self.max_context if page_size is None else int(page_size)
        return StepFunTextDecodeResourcePlan.from_model_map(
            self.model_map,
            backend=self.backend,
            context_pages=context_pages,
            page_size=page,
            max_new_tokens=self.max_new_tokens,
        )

    def resolve_quant_dispatch_keys(self) -> Mapping[str, KernelKey]:
        """Return representative mixed-GGUF linear dispatch keys for this model."""

        _register_backend_plugin(self.backend)
        required = {
            "gguf_q3_k": KernelKey(self.backend, "linear", "gguf_q3_k", "gemv_bf16_bf16_out"),
            "gguf_q5_k": KernelKey(self.backend, "linear", "gguf_q5_k", "gemv_bf16_bf16_out"),
            "gguf_q8_0": KernelKey(self.backend, "linear", "gguf_q8_0", "gemv_bf16_bf16_out"),
        }
        _raise_for_missing_stepfun_dispatch(required, "mixed-quant")
        return required

    def resolve_kv_dispatch_keys(self) -> Mapping[str, KernelKey]:
        """Return BF16 KV write/decode dispatch keys for this model."""

        return stepfun_kv_decode_kernel_plan(
            backend=self.backend,
            max_context=self.max_context,
            max_new_tokens=self.max_new_tokens,
        ).dispatch_keys

    def _validate_short_context(self, input_ids: tuple[int, ...]) -> None:
        if len(input_ids) + self.max_new_tokens > self.max_context:
            raise ValueError(
                "StepFun short-context bring-up exceeded max_context: "
                f"prompt={len(input_ids)} max_new_tokens={self.max_new_tokens} "
                f"max_context={self.max_context}"
            )


@dataclass
class StepFunResidentSession:
    """Owned resident StepFun state for incremental GGUF decode bring-up.

    The session is intentionally still below the full streaming runner: it owns
    materialized split-GGUF weights and exposes correctness bridges for prompt
    embeddings, KV allocation, per-layer prefill probes, and sampled logits.
    Full KV-backed streaming decode is still wired in later P11 iterations.
    """

    info: GGUFSplitModelInfo
    model_map: StepFunGGUFModelMap
    tokenizer: StepFunGGUFTokenizer
    weights: StepFunGGUFResidentWeights
    backend: str = "hip_gfx1151"
    _closed: bool = False

    @classmethod
    def from_gguf_paths(
        cls,
        paths: Sequence[str | Path],
        *,
        backend: str = "hip_gfx1151",
        selected_slots: Sequence[str] | None = None,
        runtime: HipRuntime | None = None,
    ) -> "StepFunResidentSession":
        info = scan_gguf_splits(tuple(Path(path) for path in paths))
        model_map = build_stepfun_gguf_tensor_map(info)
        tokenizer = StepFunGGUFTokenizer.from_gguf_info(info)
        weights = materialize_stepfun_gguf_weights(
            info,
            selected_slots=selected_slots,
            runtime=runtime,
        )
        return cls(
            info=info,
            model_map=model_map,
            tokenizer=tokenizer,
            weights=weights,
            backend=backend,
        )

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self._closed:
            return
        self.weights.free(runtime=runtime)
        self._closed = True

    def __enter__(self) -> "StepFunResidentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.free()

    def allocate_kv_cache(
        self,
        *,
        context_pages: int,
        page_size: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
        runtime: HipRuntime | None = None,
    ) -> StepFunKVCacheAllocation:
        """Allocate per-layer BF16 K/V buffers for StepFun decode bring-up."""

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        if context_pages <= 0:
            raise ValueError("context_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        runtime = runtime or get_hip_runtime()
        layer_nbytes = stepfun_kv_cache_layer_nbytes(
            self.model_map.config,
            context_pages=context_pages,
            page_size=page_size,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for key_nbytes, value_nbytes in layer_nbytes:
                buffers.append(malloc(key_nbytes, runtime=runtime))
                buffers.append(malloc(value_nbytes, runtime=runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise
        return StepFunKVCacheAllocation(
            buffers=tuple(buffers),
            context_pages=int(context_pages),
            page_size=int(page_size),
            layer_nbytes=layer_nbytes,
        )

    def weight_for_slot(self, slot_path: str):
        """Return a resident weight by StepFun materialization slot path."""

        if slot_path.startswith("root."):
            return self.weights.root(slot_path.removeprefix("root."))
        if slot_path.startswith("layers."):
            parts = slot_path.split(".", 2)
            if len(parts) != 3:
                raise ValueError(f"invalid StepFun layer slot path: {slot_path!r}")
            return self.weights.layer(int(parts[1])).weight(parts[2])
        raise ValueError(f"invalid StepFun materialization slot path: {slot_path!r}")

    def embed_token_ids_bf16(
        self,
        token_ids: Sequence[int],
        *,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Launch resident Q8_0 token embedding and return BF16 bit rows."""

        import numpy as np

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        if "token_embedding" not in self.weights.root_weights:
            raise RuntimeError("token_embedding weight is not resident in this session")
        runtime = runtime or get_hip_runtime()
        _register_backend_plugin(self.backend)
        ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("token_ids must not be empty")
        vocab_size = int(self.model_map.config.vocab_size)
        if np.any(ids < 0) or np.any(ids >= vocab_size):
            raise ValueError("token_ids contain out-of-range StepFun token IDs")
        rows = int(ids.shape[0])
        hidden_size = int(self.model_map.config.hidden_size)
        out = np.empty((rows, hidden_size), dtype=np.uint16)
        token_buf = malloc(ids.nbytes, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        try:
            copy_host_to_device(token_buf, host_array_ptr(ids), runtime=runtime)
            launch_gguf_embedding(
                self.weights.root("token_embedding"),
                token_buf.ptr,
                out_buf.ptr,
                rows=rows,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                backend=self.backend,
                stream=stream,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        finally:
            free(out_buf, runtime=runtime)
            free(token_buf, runtime=runtime)
        return out

    def embed_chat_prompt_bf16(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> StepFunPromptEmbedding:
        """Render/tokenize a Step chat prompt and launch resident embeddings."""

        rendered = self.tokenizer.render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )
        input_ids = tuple(self.tokenizer.encode(rendered, add_bos=False))
        embeddings = self.embed_token_ids_bf16(input_ids, runtime=runtime, stream=stream)
        return StepFunPromptEmbedding(
            rendered_prompt=rendered,
            input_ids=input_ids,
            embeddings_bf16=embeddings,
        )

    def root_only_prompt_logits_probe_bf16(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> StepFunRootOnlyLogitsProbe:
        """Run tokenizer -> embedding -> final logits without transformer layers."""

        prompt = self.embed_chat_prompt_bf16(
            messages,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            runtime=runtime,
            stream=stream,
        )
        logits = self.final_logits_probe_bf16(
            prompt.embeddings_bf16[-1:].copy(),
            runtime=runtime,
            stream=stream,
        )
        return StepFunRootOnlyLogitsProbe(prompt=prompt, logits=logits)

    def first_layer_prompt_logits_probe_bf16(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> StepFunOneLayerLogitsProbe:
        """Run tokenizer -> embeddings -> layer 0 prefill -> final logits.

        This is a correctness smoke for orchestration only. It deliberately
        skips layers 1-44, so its logits are not next-token parity evidence.
        """

        prefix = self.layer_prefix_prompt_logits_probe_bf16(
            messages,
            layer_count=1,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            runtime=runtime,
            stream=stream,
        )
        return StepFunOneLayerLogitsProbe(
            prompt=prefix.prompt,
            layer_hidden=prefix.layer_hidden,
            logits=prefix.logits,
        )

    def layer_prefix_prompt_logits_probe_bf16(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        layer_count: int,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> StepFunLayerPrefixLogitsProbe:
        """Run tokenizer -> embeddings -> contiguous layer prefix -> logits.

        This host-composed prefill bridge applies layers ``0..layer_count-1``
        with BF16 boundaries between layers, then runs the final root logits
        probe on the last prompt row. It is not a native decode/KV-cache path
        and is not full next-token parity unless ``layer_count`` covers every
        decoder layer and an external oracle comparison is recorded.
        """

        import numpy as np
        from hipengine.loading.materialize import float_array_to_bf16_bits

        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if layer_count > self.model_map.config.block_count:
            raise ValueError(
                f"layer_count={layer_count} exceeds StepFun block_count={self.model_map.config.block_count}"
            )
        prompt = self.embed_chat_prompt_bf16(
            messages,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            runtime=runtime,
            stream=stream,
        )
        positions = np.arange(prompt.prompt_length, dtype=np.int64)
        hidden_bits = np.ascontiguousarray(prompt.embeddings_bf16, dtype=np.uint16)
        layer_hidden = None
        for layer_id in range(int(layer_count)):
            layer_hidden = self.layer_prefill_probe_bf16(
                layer_id,
                hidden_bits,
                positions=positions,
                runtime=runtime,
                stream=stream,
            )
            hidden_bits = float_array_to_bf16_bits(np.asarray(layer_hidden, dtype=np.float32))
        if layer_hidden is None:  # pragma: no cover - guarded by layer_count validation
            raise RuntimeError("StepFun layer prefix produced no hidden state")
        logits = self.final_logits_probe_bf16(hidden_bits[-1:].copy(), runtime=runtime, stream=stream)
        return StepFunLayerPrefixLogitsProbe(
            prompt=prompt,
            layer_count=int(layer_count),
            layer_hidden=layer_hidden,
            logits=logits,
        )

    def linear_slot_bf16(
        self,
        slot_path: str,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Launch a resident GGUF linear slot with BF16-bit activations."""

        import numpy as np

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        if output_dtype not in {GGUF_OUTPUT_BF16, GGUF_OUTPUT_F32}:
            raise ValueError(f"unsupported StepFun resident linear output dtype {output_dtype!r}")
        runtime = runtime or get_hip_runtime()
        _register_backend_plugin(self.backend)
        weight = self.weight_for_slot(slot_path)
        if len(weight.spec.source.shape) != 2:
            raise ValueError(f"StepFun linear slot must be rank-2, got {slot_path!r}")
        out_features, in_features = (int(dim) for dim in weight.spec.source.shape)
        x = np.ascontiguousarray(x_bf16_bits, dtype=np.uint16)
        if x.ndim != 2:
            raise ValueError("x_bf16_bits must have shape [rows, in_features]")
        rows = int(x.shape[0])
        if rows <= 0:
            raise ValueError("x_bf16_bits must have at least one row")
        if int(x.shape[1]) != in_features:
            raise ValueError(
                f"x_bf16_bits.shape[1]={x.shape[1]} does not match {slot_path} in_features={in_features}"
            )
        out_dtype = np.uint16 if output_dtype == GGUF_OUTPUT_BF16 else np.float32
        out = np.empty((rows, out_features), dtype=out_dtype)
        x_buf = malloc(x.nbytes, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        try:
            copy_host_to_device(x_buf, host_array_ptr(x), runtime=runtime)
            launch_gguf_linear(
                weight,
                x_buf.ptr,
                out_buf.ptr,
                rows=rows,
                in_features=in_features,
                out_features=out_features,
                output_dtype=output_dtype,
                backend=self.backend,
                stream=stream,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        finally:
            free(out_buf, runtime=runtime)
            free(x_buf, runtime=runtime)
        return out

    def selected_expert_linear_bf16(
        self,
        slot_path: str,
        x_bf16_bits,
        selected_experts,
        *,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Launch a resident selected-expert GGUF linear slot with BF16 output."""

        import numpy as np

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        runtime = runtime or get_hip_runtime()
        _register_backend_plugin(self.backend)
        weight = self.weight_for_slot(slot_path)
        if len(weight.spec.source.shape) != 3:
            raise ValueError(f"StepFun selected-expert slot must be rank-3, got {slot_path!r}")
        num_experts, out_features, in_features = (int(dim) for dim in weight.spec.source.shape)
        x = np.ascontiguousarray(x_bf16_bits, dtype=np.uint16)
        selected = np.ascontiguousarray(selected_experts, dtype=np.int64).reshape(-1)
        if x.ndim != 2:
            raise ValueError("x_bf16_bits must have shape [x_rows, in_features]")
        x_rows = int(x.shape[0])
        if x_rows <= 0:
            raise ValueError("x_bf16_bits must have at least one row")
        if int(x.shape[1]) != in_features:
            raise ValueError(
                f"x_bf16_bits.shape[1]={x.shape[1]} does not match {slot_path} in_features={in_features}"
            )
        rows = int(selected.shape[0])
        if rows <= 0 or rows % x_rows != 0:
            raise ValueError("selected_experts length must be positive and divisible by x rows")
        if np.any(selected < 0) or np.any(selected >= num_experts):
            raise ValueError("selected_experts contain out-of-range expert IDs")
        out = np.empty((rows, out_features), dtype=np.uint16)
        x_buf = malloc(x.nbytes, runtime=runtime)
        selected_buf = malloc(selected.nbytes, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        try:
            copy_host_to_device(x_buf, host_array_ptr(x), runtime=runtime)
            copy_host_to_device(selected_buf, host_array_ptr(selected), runtime=runtime)
            key = KernelKey(self.backend, "linear", weight.spec.quant_key, "selected_gemv_bf16_bf16_out")
            fn = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
            fn(
                x_buf.ptr,
                selected_buf.ptr,
                weight.allocation().buffer.ptr,
                out_buf.ptr,
                x_rows=x_rows,
                rows=rows,
                num_experts=num_experts,
                in_features=in_features,
                out_features=out_features,
                runtime=runtime,
                stream=stream,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        finally:
            free(out_buf, runtime=runtime)
            free(selected_buf, runtime=runtime)
            free(x_buf, runtime=runtime)
        return out

    def project_moe_expert_inputs_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        selected_experts,
        *,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> Mapping[str, object]:
        """Launch selected and shared MoE gate/up input projections."""

        self._validate_layer_id(layer_id)
        layer = self.model_map.layer(layer_id)
        required = ("ffn_gate_exps", "ffn_up_exps", "ffn_gate_shexp", "ffn_up_shexp")
        if any(slot not in layer.tensors for slot in required):
            raise RuntimeError(f"layer {layer_id} does not expose MoE expert input weights")
        prefix = f"layers.{layer_id}"
        return {
            "expert_gate": self.selected_expert_linear_bf16(
                f"{prefix}.ffn_gate_exps",
                x_bf16_bits,
                selected_experts,
                runtime=runtime,
                stream=stream,
            ),
            "expert_up": self.selected_expert_linear_bf16(
                f"{prefix}.ffn_up_exps",
                x_bf16_bits,
                selected_experts,
                runtime=runtime,
                stream=stream,
            ),
            "shared_gate": self.linear_slot_bf16(
                f"{prefix}.ffn_gate_shexp",
                x_bf16_bits,
                output_dtype=GGUF_OUTPUT_BF16,
                runtime=runtime,
                stream=stream,
            ),
            "shared_up": self.linear_slot_bf16(
                f"{prefix}.ffn_up_shexp",
                x_bf16_bits,
                output_dtype=GGUF_OUTPUT_BF16,
                runtime=runtime,
                stream=stream,
            ),
        }

    def project_attention_inputs_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> Mapping[str, object]:
        """Launch resident StepFun Q/K/V/gate input projections for one layer."""

        self._validate_layer_id(layer_id)
        prefix = f"layers.{layer_id}"
        return {
            "q": self.linear_slot_bf16(
                f"{prefix}.attn_q",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "k": self.linear_slot_bf16(
                f"{prefix}.attn_k",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "v": self.linear_slot_bf16(
                f"{prefix}.attn_v",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "gate": self.linear_slot_bf16(
                f"{prefix}.attn_gate",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
        }

    def attention_prefill_probe_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        positions=None,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Correctness probe for one resident Step attention prefill block.

        Q/K/V/gate and output projections run through resident GGUF weights.
        Q/K norms, RoPE, causal GQA attention, and head-wise gating happen on
        the host until native Step attention/KV-cache execution is wired.
        """

        import numpy as np
        from hipengine.kernels.cpu_reference.ops import (
            step_apply_rope,
            step_gqa_attention_prefill,
            step_headwise_attention_gate,
            step_rmsnorm,
        )
        from hipengine.loading.materialize import float_array_to_bf16_bits
        from hipengine.quant.gguf import bf16_to_float32

        self._validate_layer_id(layer_id)
        layer = self.model_map.layer(layer_id)
        required = (
            "attn_norm",
            "attn_q_norm",
            "attn_k_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_gate",
            "attn_output",
        )
        if any(slot not in layer.tensors for slot in required):
            raise RuntimeError(f"layer {layer_id} does not expose all Step attention weights")
        runtime = runtime or get_hip_runtime()
        x = np.ascontiguousarray(x_bf16_bits, dtype=np.uint16)
        if x.ndim != 2:
            raise ValueError("x_bf16_bits must have shape [rows, hidden_size]")
        rows = int(x.shape[0])
        hidden_size = int(self.model_map.config.hidden_size)
        if rows <= 0:
            raise ValueError("x_bf16_bits must have at least one row")
        if int(x.shape[1]) != hidden_size:
            raise ValueError(f"x_bf16_bits.shape[1]={x.shape[1]} does not match hidden_size={hidden_size}")
        if positions is None:
            pos = np.arange(rows, dtype=np.int64)
        else:
            pos = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
            if pos.shape != (rows,):
                raise ValueError("positions must have one entry per x row")
        head_dim = int(self.model_map.config.head_dim)
        value_dim = int(self.model_map.config.value_dim)
        if value_dim != head_dim:
            raise RuntimeError("StepFun attention prefill probe currently requires value_dim == head_dim")
        query_heads = int(self.model_map.config.head_counts[layer_id])
        kv_heads = int(self.model_map.config.kv_head_counts[layer_id])

        hidden = bf16_to_float32(x)
        attn_norm_weight = self._copy_resident_f32_weight(f"layers.{layer_id}.attn_norm", runtime=runtime)
        normed_bits = float_array_to_bf16_bits(
            step_rmsnorm(hidden, attn_norm_weight, eps=self.model_map.config.rms_norm_eps)
        )
        projections = self.project_attention_inputs_bf16(
            layer_id,
            normed_bits,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=runtime,
            stream=stream,
        )
        q = np.asarray(projections["q"], dtype=np.float32).reshape(rows, query_heads, head_dim)
        k = np.asarray(projections["k"], dtype=np.float32).reshape(rows, kv_heads, head_dim)
        v = np.asarray(projections["v"], dtype=np.float32).reshape(rows, kv_heads, value_dim)
        gate_logits = np.asarray(projections["gate"], dtype=np.float32)
        if gate_logits.shape != (rows, query_heads):
            raise ValueError(f"attn_gate output shape {gate_logits.shape} does not match {(rows, query_heads)}")

        q_norm_weight = self._copy_resident_f32_weight(f"layers.{layer_id}.attn_q_norm", runtime=runtime)
        k_norm_weight = self._copy_resident_f32_weight(f"layers.{layer_id}.attn_k_norm", runtime=runtime)
        q = step_rmsnorm(q, q_norm_weight, eps=self.model_map.config.rms_norm_eps)
        k = step_rmsnorm(k, k_norm_weight, eps=self.model_map.config.rms_norm_eps)
        if layer.attention_type == SLIDING_ATTENTION:
            partial_factor = 1.0
            theta = self.model_map.config.rope_freq_base_swa
            llama3_scaling = False
            sliding_window = self.model_map.config.sliding_window
        else:
            partial_factor = 0.5
            theta = self.model_map.config.rope_freq_base
            llama3_scaling = True
            sliding_window = None
        q_rope = step_apply_rope(
            q,
            pos,
            head_dim=head_dim,
            partial_factor=partial_factor,
            theta=theta,
            llama3_scaling=llama3_scaling,
        )
        k_rope = step_apply_rope(
            k,
            pos,
            head_dim=head_dim,
            partial_factor=partial_factor,
            theta=theta,
            llama3_scaling=llama3_scaling,
        )
        attention = step_gqa_attention_prefill(q_rope, k_rope, v, sliding_window=sliding_window)
        gated = step_headwise_attention_gate(attention, gate_logits)
        gated_bits = float_array_to_bf16_bits(gated.reshape(rows, query_heads * head_dim))
        return self.linear_slot_bf16(
            f"layers.{layer_id}.attn_output",
            gated_bits,
            output_dtype=output_dtype,
            runtime=runtime,
            stream=stream,
        )

    def layer_prefill_probe_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        positions=None,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Correctness probe for one resident Step layer prefill block.

        This composes the attention prefill probe, residual add, FFN RMSNorm,
        and the dense or MoE MLP probe on host-visible arrays. It is a bridge
        toward the streaming layer loop, not the final fused/device-side path.
        """

        import numpy as np
        from hipengine.kernels.cpu_reference.ops import step_rmsnorm
        from hipengine.loading.materialize import float_array_to_bf16_bits
        from hipengine.quant.gguf import bf16_to_float32

        self._validate_layer_id(layer_id)
        runtime = runtime or get_hip_runtime()
        x = np.ascontiguousarray(x_bf16_bits, dtype=np.uint16)
        hidden = bf16_to_float32(x)
        attention_out = self.attention_prefill_probe_bf16(
            layer_id,
            x,
            positions=positions,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=runtime,
            stream=stream,
        )
        attention_residual = (hidden + np.asarray(attention_out, dtype=np.float32)).astype(np.float32)
        ffn_norm_weight = self._copy_resident_f32_weight(f"layers.{layer_id}.ffn_norm", runtime=runtime)
        ffn_norm_bits = float_array_to_bf16_bits(
            step_rmsnorm(attention_residual, ffn_norm_weight, eps=self.model_map.config.rms_norm_eps)
        )
        layer = self.model_map.layer(layer_id)
        if "ffn_gate" in layer.tensors:
            ffn = self.dense_mlp_probe_bf16(
                layer_id,
                ffn_norm_bits,
                output_dtype=GGUF_OUTPUT_F32,
                runtime=runtime,
                stream=stream,
            )
        elif "ffn_gate_inp" in layer.tensors:
            ffn = self.moe_mlp_probe_bf16(
                layer_id,
                ffn_norm_bits,
                runtime=runtime,
                stream=stream,
            )
        else:
            raise RuntimeError(f"layer {layer_id} does not expose a Step MLP path")
        return (attention_residual + np.asarray(ffn, dtype=np.float32)).astype(np.float32)

    def project_dense_mlp_inputs_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> Mapping[str, object]:
        """Launch resident dense-SwiGLU gate/up projections for one layer."""

        self._validate_layer_id(layer_id)
        layer = self.model_map.layer(layer_id)
        if "ffn_gate" not in layer.tensors or "ffn_up" not in layer.tensors:
            raise RuntimeError(f"layer {layer_id} does not expose dense ffn_gate/ffn_up weights")
        prefix = f"layers.{layer_id}"
        return {
            "gate": self.linear_slot_bf16(
                f"{prefix}.ffn_gate",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "up": self.linear_slot_bf16(
                f"{prefix}.ffn_up",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
        }

    def final_logits_probe_bf16(
        self,
        x_bf16_bits,
        *,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Correctness probe for final RMSNorm + output projection logits."""

        import numpy as np
        from hipengine.kernels.cpu_reference.ops import step_rmsnorm
        from hipengine.loading.materialize import float_array_to_bf16_bits
        from hipengine.quant.gguf import bf16_to_float32

        runtime = runtime or get_hip_runtime()
        hidden = bf16_to_float32(np.ascontiguousarray(x_bf16_bits, dtype=np.uint16))
        norm_weight = self._copy_resident_f32_weight("root.output_norm", runtime=runtime)
        normed = step_rmsnorm(hidden, norm_weight, eps=self.model_map.config.rms_norm_eps)
        normed_bits = float_array_to_bf16_bits(normed)
        return self.linear_slot_bf16(
            "root.lm_head",
            normed_bits,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=runtime,
            stream=stream,
        )

    def moe_mlp_probe_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Correctness probe for a resident Step MoE MLP layer.

        Routing plus gate/up/down projections run through resident weights.
        SwiGLU activation and expert aggregation happen on the host until a
        device-side MoE composition path is available.
        """

        import numpy as np
        from hipengine.quant.gguf import bf16_to_float32

        runtime = runtime or get_hip_runtime()
        router = self.moe_router_probe_bf16(layer_id, x_bf16_bits, runtime=runtime)
        selected = np.asarray(router.selected_experts, dtype=np.int64)
        routing = np.asarray(router.routing_weights, dtype=np.float32)
        top_k = int(routing.shape[-1])
        projections = self.project_moe_expert_inputs_bf16(
            layer_id,
            x_bf16_bits,
            selected.reshape(-1),
            runtime=runtime,
            stream=stream,
        )
        expert_fused_bits = _swiglu_bf16_bits(
            bf16_to_float32(np.asarray(projections["expert_gate"], dtype=np.uint16)),
            bf16_to_float32(np.asarray(projections["expert_up"], dtype=np.uint16)),
            self.model_map.config.swiglu_clamp_exp[layer_id],
        )
        expert_down_bits = self.selected_expert_linear_bf16(
            f"layers.{layer_id}.ffn_down_exps",
            expert_fused_bits,
            selected.reshape(-1),
            runtime=runtime,
            stream=stream,
        )
        expert_down = bf16_to_float32(np.asarray(expert_down_bits, dtype=np.uint16)).reshape(
            routing.shape[0],
            top_k,
            -1,
        )
        out = np.sum(expert_down * routing[..., None], axis=1, dtype=np.float32)
        shared_fused_bits = _swiglu_bf16_bits(
            bf16_to_float32(np.asarray(projections["shared_gate"], dtype=np.uint16)),
            bf16_to_float32(np.asarray(projections["shared_up"], dtype=np.uint16)),
            self.model_map.config.swiglu_clamp_shexp[layer_id],
        )
        shared_down_bits = self.linear_slot_bf16(
            f"layers.{layer_id}.ffn_down_shexp",
            shared_fused_bits,
            output_dtype=GGUF_OUTPUT_BF16,
            runtime=runtime,
            stream=stream,
        )
        return (out + bf16_to_float32(np.asarray(shared_down_bits, dtype=np.uint16))).astype(np.float32)

    def moe_router_probe_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        runtime: HipRuntime | None = None,
    ) -> StepFunMoERouterResult:
        """Correctness probe for a resident Step MoE router.

        Router weights/bias are resident F32 tensors; this probe copies them
        through hipEngine's memory API and applies the CPU-reference routing
        math on the host until a device-side router is introduced.
        """

        import numpy as np
        from hipengine.kernels.cpu_reference import step_moe_router
        from hipengine.quant.gguf import bf16_to_float32

        self._validate_layer_id(layer_id)
        layer = self.model_map.layer(layer_id)
        if "ffn_gate_inp" not in layer.tensors or "exp_probs_bias" not in layer.tensors:
            raise RuntimeError(f"layer {layer_id} does not expose MoE router weights")
        runtime = runtime or get_hip_runtime()
        hidden = bf16_to_float32(np.ascontiguousarray(x_bf16_bits, dtype=np.uint16))
        router_weight = self._copy_resident_f32_weight(f"layers.{layer_id}.ffn_gate_inp", runtime=runtime)
        router_bias = self._copy_resident_f32_weight(f"layers.{layer_id}.exp_probs_bias", runtime=runtime)
        routing_weights, selected_experts, logits = step_moe_router(
            hidden,
            router_weight,
            router_bias=router_bias,
            top_k=self.model_map.config.expert_used_count,
            routing_scale=self.model_map.config.expert_weights_scale,
            normalize_selected=self.model_map.config.expert_weights_norm,
        )
        return StepFunMoERouterResult(
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            logits=logits,
        )

    def dense_mlp_probe_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Correctness probe for a resident dense SwiGLU MLP layer.

        Gate/up/down projections run through resident GGUF linears. SwiGLU and
        BF16 rounding happen on the host until a device-side fused MLP path is
        available, so this is not the final streaming hot path.
        """

        import numpy as np
        from hipengine.loading.materialize import float_array_to_bf16_bits

        projections = self.project_dense_mlp_inputs_bf16(
            layer_id,
            x_bf16_bits,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=runtime,
            stream=stream,
        )
        gate = np.asarray(projections["gate"], dtype=np.float32)
        up = np.asarray(projections["up"], dtype=np.float32)
        activated_gate = gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32))
        limit = float(self.model_map.config.swiglu_clamp_exp[layer_id])
        if limit > 0.0:
            activated_gate = np.minimum(activated_gate, np.float32(limit))
            up = np.clip(up, np.float32(-limit), np.float32(limit))
        fused_bits = float_array_to_bf16_bits(activated_gate * up)
        return self.linear_slot_bf16(
            f"layers.{layer_id}.ffn_down",
            fused_bits,
            output_dtype=output_dtype,
            runtime=runtime,
            stream=stream,
        )

    def _copy_resident_f32_weight(self, slot_path: str, *, runtime: HipRuntime):
        import numpy as np

        weight = self.weight_for_slot(slot_path)
        if weight.spec.quant_key != "f32":
            raise ValueError(f"resident slot {slot_path!r} is not an F32 tensor")
        out = np.empty(tuple(int(dim) for dim in weight.spec.source.shape), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), weight.allocation().buffer, runtime=runtime)
        return out

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.model_map.config.block_count:
            raise ValueError(f"layer_id out of range: {layer_id}")


def stepfun_kv_cache_layer_nbytes(
    config: StepFunGGUFConfig,
    *,
    context_pages: int,
    page_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return per-layer BF16 key/value cache byte counts for StepFun."""

    if context_pages <= 0:
        raise ValueError("context_pages must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    tokens = int(context_pages) * int(page_size)
    return tuple(
        (
            tokens * int(kv_heads) * int(config.head_dim) * BF16_BYTES,
            tokens * int(kv_heads) * int(config.value_dim) * BF16_BYTES,
        )
        for kv_heads in config.kv_head_counts
    )


def stepfun_kv_cache_nbytes(
    config: StepFunGGUFConfig,
    *,
    context_pages: int,
    page_size: int,
) -> int:
    """Return total BF16 KV-cache bytes for StepFun."""

    return sum(
        key_nbytes + value_nbytes
        for key_nbytes, value_nbytes in stepfun_kv_cache_layer_nbytes(
            config,
            context_pages=context_pages,
            page_size=page_size,
        )
    )


def stepfun_kv_decode_kernel_plan(
    *,
    backend: str = "hip_gfx1151",
    max_context: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
    max_new_tokens: int = DEFAULT_STEPFUN_MAX_NEW_TOKENS,
    attention_block_size: int = STEPFUN_KV_ATTENTION_BLOCK_SIZE,
) -> StepFunKVDecodeKernelPlan:
    """Return registered BF16 KV dispatch keys planned for StepFun decode."""

    if max_context <= 0:
        raise ValueError("max_context must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if int(max_context) <= int(max_new_tokens):
        raise ValueError("max_context must leave at least one prompt token before decode")
    if attention_block_size <= 0:
        raise ValueError("attention_block_size must be positive")
    attention_block_table_len = _ceil_div(int(max_context), int(attention_block_size))
    max_prompt_rows = int(max_context) - int(max_new_tokens)
    decode_max_live_count = max_prompt_rows
    _register_backend_plugin(backend)
    decode_spans = _planning_bf16_kv_spans(
        base_offsets_len=attention_block_table_len,
        live_counts_len=1,
        max_live_count=decode_max_live_count,
    )
    prompt_route_spans = _planning_bf16_kv_spans(
        base_offsets_len=attention_block_table_len,
        live_counts_len=1,
        max_live_count=0,
    )
    dispatch_keys = {
        "prompt_kv_write": plan_paged_kv_write(
            prompt_route_spans,
            kind=PagedKVWriteKind.PROMPT,
            source_dtype=DType.BF16,
            model_quant=STEPFUN_GGUF_KERNEL_QUANT,
        ).key(backend),
        "decode_kv_write": plan_paged_kv_write(
            decode_spans,
            kind=PagedKVWriteKind.DECODE,
            source_dtype=DType.BF16,
            model_quant=STEPFUN_GGUF_KERNEL_QUANT,
        ).key(backend),
        "decode_attention": plan_paged_attn_decode(
            decode_spans,
            kind=PagedAttnDecodeKind.SPLITK_GATE_F32,
            model_quant=STEPFUN_GGUF_KERNEL_QUANT,
        ).key(backend),
    }
    registered = {name: _can_resolve_kernel_key(key) for name, key in dispatch_keys.items()}
    plan = StepFunKVDecodeKernelPlan(
        backend=backend,
        model_quant=STEPFUN_GGUF_KERNEL_QUANT,
        kv_storage_dtype=DType.BF16.value,
        dispatch_keys=dispatch_keys,
        registered=registered,
        decode_attention_kind=PagedAttnDecodeKind.SPLITK_GATE_F32.value,
        max_context=int(max_context),
        max_new_tokens=int(max_new_tokens),
        attention_block_size=int(attention_block_size),
        attention_block_table_len=attention_block_table_len,
        max_prompt_rows=max_prompt_rows,
        decode_max_live_count=decode_max_live_count,
    )
    if not plan.all_registered:
        missing = {name: key for name, key in dispatch_keys.items() if not registered[name]}
        joined = ", ".join(f"{name}={key}" for name, key in missing.items())
        raise RuntimeError(f"missing StepFun KV dispatch keys: {joined}")
    return plan


def _planning_bf16_kv_spans(
    *,
    base_offsets_len: int,
    live_counts_len: int,
    max_live_count: int,
) -> KVLiveSpans:
    """Construct no-allocation span metadata for registry-key planning."""

    if base_offsets_len <= 0:
        raise ValueError("base_offsets_len must be positive")
    if live_counts_len <= 0:
        raise ValueError("live_counts_len must be positive")
    if max_live_count < 0:
        raise ValueError("max_live_count must be non-negative")
    device = Device("hip", 0)
    block_table = Tensor.from_handle(0, (int(base_offsets_len),), DType.INT32, device)
    live_counts = Tensor.from_handle(0, (int(live_counts_len),), DType.INT64, device)
    return KVLiveSpans.paged_uniform(
        block_table=block_table,
        live_counts=live_counts,
        max_live_count=int(max_live_count),
        storage_dtype=DType.BF16,
        span_role="decode",
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pack_integer_payload(dtype: str, values: Sequence[int]) -> bytes:
    if dtype == "int32":
        code = "i"
    elif dtype == "int64":
        code = "q"
    else:
        raise ValueError(f"unsupported StepFun KV span payload dtype: {dtype}")
    return struct.pack("<" + code * len(values), *[int(value) for value in values])


def _can_resolve_kernel_key(key: KernelKey) -> bool:
    return (
        resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )
        is not None
    )


def _raise_for_missing_stepfun_dispatch(keys: Mapping[str, KernelKey], label: str) -> None:
    missing = [key for key in keys.values() if not _can_resolve_kernel_key(key)]
    if missing:
        joined = ", ".join(str(key) for key in missing)
        raise RuntimeError(f"missing StepFun {label} dispatch keys: {joined}")


def _kernel_key_to_dict(key: KernelKey) -> dict[str, str]:
    return {
        "backend": key.backend,
        "layer": key.layer,
        "quant": key.quant,
        "variant": key.variant,
    }


def stepfun_slot_tensor(model_map: StepFunGGUFModelMap, slot_path: str):
    """Resolve a StepFun materialization slot path to its GGUF tensor info."""

    if slot_path.startswith("root."):
        parts = slot_path.split(".")
        if len(parts) != 2:
            raise ValueError(f"invalid StepFun root slot path: {slot_path!r}")
        return model_map.root(parts[1])
    if slot_path.startswith("layers."):
        parts = slot_path.split(".")
        if len(parts) != 3:
            raise ValueError(f"invalid StepFun layer slot path: {slot_path!r}")
        return model_map.layer(int(parts[1])).tensor(parts[2])
    raise ValueError(f"invalid StepFun materialization slot path: {slot_path!r}")


def stepfun_layer_slot_paths(model_map: StepFunGGUFModelMap, layer_id: int) -> tuple[str, ...]:
    """Return resident slot paths required to execute one StepFun layer."""

    if layer_id < 0 or layer_id >= model_map.config.block_count:
        raise ValueError(f"layer_id out of range: {layer_id}")
    layer = model_map.layer(layer_id)
    slots = [
        "attn_norm",
        "attn_q_norm",
        "attn_k_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "ffn_norm",
    ]
    if "ffn_gate" in layer.tensors:
        slots.extend(["ffn_gate", "ffn_up", "ffn_down"])
    elif "ffn_gate_inp" in layer.tensors:
        slots.extend(
            [
                "ffn_gate_inp",
                "exp_probs_bias",
                "ffn_gate_exps",
                "ffn_up_exps",
                "ffn_down_exps",
                "ffn_gate_shexp",
                "ffn_up_shexp",
                "ffn_down_shexp",
            ]
        )
    else:
        raise RuntimeError(f"layer {layer_id} does not expose a Step MLP path")
    missing = [slot for slot in slots if slot not in layer.tensors]
    if missing:
        raise RuntimeError(f"layer {layer_id} is missing required Step slots: {missing}")
    return tuple(f"layers.{layer_id}.{slot}" for slot in slots)


def stepfun_layer_prefix_slot_paths(model_map: StepFunGGUFModelMap, layer_count: int) -> tuple[str, ...]:
    """Return root + layer slot paths for a contiguous prompt-logits prefix."""

    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if layer_count > model_map.config.block_count:
        raise ValueError(f"layer_count={layer_count} exceeds StepFun block_count={model_map.config.block_count}")
    slots: list[str] = ["root.token_embedding", "root.output_norm", "root.lm_head"]
    for layer_id in range(int(layer_count)):
        slots.extend(stepfun_layer_slot_paths(model_map, layer_id))
    return tuple(slots)


def stepfun_text_decode_slot_paths(model_map: StepFunGGUFModelMap) -> tuple[str, ...]:
    """Return every resident GGUF slot needed by the text-only Step runner.

    This full-model slot plan is intentionally separate from the shorter
    layer-prefix prompt-logits probe: native text decode also owns the GGUF root
    RoPE-frequency table even though the current host-composed probes derive
    RoPE frequencies from metadata.
    """

    root_slots = ("token_embedding", "rope_freqs", "output_norm", "lm_head")
    missing_roots = [slot for slot in root_slots if slot not in model_map.root_tensors]
    if missing_roots:
        raise RuntimeError(f"model is missing required Step root slots: {missing_roots}")
    slots: list[str] = [f"root.{slot}" for slot in root_slots]
    for layer_id in range(model_map.config.block_count):
        slots.extend(stepfun_layer_slot_paths(model_map, layer_id))
    return tuple(slots)


def _swiglu_bf16_bits(gate, up, limit: float):
    import numpy as np
    from hipengine.loading.materialize import float_array_to_bf16_bits

    gate_arr = np.asarray(gate, dtype=np.float32)
    up_arr = np.asarray(up, dtype=np.float32)
    activated = gate_arr / (np.float32(1.0) + np.exp(-gate_arr).astype(np.float32))
    if float(limit) > 0.0:
        clamp = np.float32(limit)
        activated = np.minimum(activated, clamp)
        up_arr = np.clip(up_arr, -clamp, clamp)
    return float_array_to_bf16_bits(activated * up_arr)


def _register_backend_plugin(backend: str) -> None:
    # Import-time backend plugins populate aliases/registrations. Resolve by
    # backend module name instead of branching on a concrete backend key.
    backend_module = import_module(f"hipengine.kernels.{backend}")
    registrar_name = f"register_{backend.removeprefix('hip_')}_kernels"
    registrar = getattr(backend_module, registrar_name, None)
    if callable(registrar):
        registrar()


__all__ = [
    "DEFAULT_STEPFUN_MAX_NEW_TOKENS",
    "DEFAULT_STEPFUN_SHORT_CONTEXT",
    "STEPFUN_GGUF_KERNEL_QUANT",
    "STEPFUN_KV_ATTENTION_BLOCK_SIZE",
    "StepFunDecodePlan",
    "StepFunInputIDDeviceUpload",
    "StepFunKVCacheAllocation",
    "StepFunKVDecodeDeviceInputs",
    "StepFunKVDecodeKernelPlan",
    "StepFunKVDecodeRunPlan",
    "StepFunKVSpanInputDeviceUpload",
    "StepFunLayerPrefixLogitsProbe",
    "StepFunMoERouterResult",
    "StepFunOneLayerLogitsProbe",
    "StepFunPromptEmbedding",
    "StepFunResidentSession",
    "StepFunRootOnlyLogitsProbe",
    "StepFunShortContextDecodePlanner",
    "StepFunTextDecodeResourcePlan",
    "stepfun_kv_cache_layer_nbytes",
    "stepfun_kv_cache_nbytes",
    "stepfun_kv_decode_kernel_plan",
    "stepfun_layer_prefix_slot_paths",
    "stepfun_layer_slot_paths",
    "stepfun_slot_tensor",
    "stepfun_streaming_runner_blockers",
    "stepfun_text_decode_slot_paths",
]
