from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import sys
from pathlib import Path

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime.stepfun_gguf_runner import (
    STEPFUN_GGUF_KERNEL_QUANT,
    STEPFUN_KV_ATTENTION_BLOCK_SIZE,
    StepFunShortContextDecodePlanner,
    stepfun_kv_cache_nbytes,
    stepfun_kv_decode_kernel_plan,
    stepfun_text_decode_slot_paths,
)

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


class _FakeHipRuntime:
    def __init__(self, *, fail_on_copy_index: int | None = None) -> None:
        self.next_ptr = 0x1000
        self.fail_on_copy_index = fail_on_copy_index
        self.copy_count = 0
        self.allocations: dict[int, int] = {}
        self.copies: dict[int, bytes] = {}
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        assert int(kind) == 1
        assert self.allocations[int(dst)] >= int(nbytes)
        self.copy_count += 1
        if self.copy_count == self.fail_on_copy_index:
            raise RuntimeError("simulated host-to-device copy failure")
        self.copies[int(dst)] = ctypes.string_at(int(src), int(nbytes))

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))


def test_stepfun_short_context_decode_plan_preserves_chat_prefix_and_multi_eos() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )

    plan = planner.plan_chat([{"role": "user", "content": "hello"}], reasoning_effort="low")

    assert plan.prompt_length + plan.max_new_tokens <= 512
    assert plan.max_new_tokens == 1
    assert plan.rendered_prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert plan.stop_token_ids == (1, 2, 128007)
    assert plan.should_stop(1)
    assert plan.should_stop(2)
    assert plan.should_stop(128007)
    assert not plan.should_stop(128006)
    assert plan.quant_dispatch_keys["gguf_q3_k"] == KernelKey(
        "hip_gfx1151", "linear", "gguf_q3_k", "gemv_bf16_bf16_out"
    )
    assert plan.quant_dispatch_keys["gguf_q5_k"] == KernelKey(
        "hip_gfx1151", "linear", "gguf_q5_k", "gemv_bf16_bf16_out"
    )
    assert plan.quant_dispatch_keys["gguf_q8_0"] == KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0", "gemv_bf16_bf16_out"
    )
    assert plan.kv_dispatch_keys["prompt_kv_write"] == KernelKey(
        "hip_gfx1151", "paged_kv_write", STEPFUN_GGUF_KERNEL_QUANT, "mixed_bf16_prompt_spans"
    )
    assert plan.kv_dispatch_keys["decode_kv_write"] == KernelKey(
        "hip_gfx1151", "paged_kv_write", STEPFUN_GGUF_KERNEL_QUANT, "mixed_bf16_spans"
    )
    assert plan.kv_dispatch_keys["decode_attention"] == KernelKey(
        "hip_gfx1151", "paged_attn_decode", STEPFUN_GGUF_KERNEL_QUANT, "bf16_split_k_gate_f32_spans"
    )


def test_stepfun_kv_decode_run_plan_binds_prompt_to_resource_spans() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )

    run_plan = planner.plan_kv_decode_chat(
        [{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        context_pages=1,
        page_size=512,
    )

    assert run_plan.prompt_length == run_plan.decode_plan.prompt_length > 0
    assert run_plan.input_ids == run_plan.decode_plan.input_ids
    input_ids_payload = struct.pack(
        "<" + "i" * run_plan.prompt_length,
        *run_plan.input_ids,
    )
    assert run_plan.input_ids_dtype == "int32"
    assert run_plan.input_ids_payload_bytes == input_ids_payload
    assert run_plan.input_ids_nbytes == run_plan.prompt_length * 4
    assert run_plan.input_ids_sha256 == hashlib.sha256(input_ids_payload).hexdigest()
    assert run_plan.rendered_prompt_nchars == len(run_plan.decode_plan.rendered_prompt)
    assert run_plan.rendered_prompt_sha256 == hashlib.sha256(
        run_plan.decode_plan.rendered_prompt.encode("utf-8")
    ).hexdigest()
    assert run_plan.prompt_positions == tuple(range(run_plan.prompt_length))
    assert run_plan.decode_position == run_plan.prompt_length
    assert run_plan.decode_live_count == run_plan.prompt_length
    assert run_plan.required_context_tokens == run_plan.prompt_length + 1
    assert run_plan.max_prompt_rows == 511
    assert run_plan.attention_block_size == 256
    assert run_plan.attention_block_table_len == 2
    assert run_plan.prompt_span_base_offsets == tuple(
        value for _ in range(run_plan.prompt_length) for value in (0, 1)
    )
    assert run_plan.decode_span_base_offsets == (0, 1)
    assert run_plan.prompt_fits_resource_plan is True
    assert run_plan.context_fits_resource_plan is True
    assert run_plan.streaming_runner_ready is False
    assert run_plan.streaming_runner_blockers[0]["name"] == "streaming_decode_loop_not_wired"
    payload = run_plan.to_dict()
    assert payload["prompt_length"] == run_plan.prompt_length
    assert payload["input_ids"] == list(run_plan.decode_plan.input_ids)
    assert payload["input_ids_dtype"] == "int32"
    assert payload["input_ids_nbytes"] == run_plan.prompt_length * 4
    assert payload["input_ids_sha256"] == hashlib.sha256(input_ids_payload).hexdigest()
    assert payload["input_id_count"] == run_plan.prompt_length
    assert payload["input_id_preview"] == list(run_plan.decode_plan.input_ids[:8])
    assert payload["rendered_prompt_nchars"] == len(run_plan.decode_plan.rendered_prompt)
    assert payload["rendered_prompt_sha256"] == hashlib.sha256(
        run_plan.decode_plan.rendered_prompt.encode("utf-8")
    ).hexdigest()
    assert payload["prompt_positions"] == list(range(run_plan.prompt_length))
    assert payload["decode_position"] == run_plan.prompt_length
    assert payload["decode_live_count"] == run_plan.prompt_length
    assert payload["required_context_tokens"] == run_plan.required_context_tokens
    assert payload["max_context"] == 512
    assert payload["max_prompt_rows"] == 511
    assert payload["attention_block_size"] == 256
    assert payload["attention_block_table_len"] == 2
    prompt_base_offsets_len = run_plan.prompt_length * 2
    prompt_base_offsets_nbytes = prompt_base_offsets_len * 4
    prompt_live_counts_nbytes = run_plan.prompt_length * 8
    assert payload["prompt_span_inputs"] == {
        "rows": run_plan.prompt_length,
        "block_size": 256,
        "block_table_len_per_row": 2,
        "base_offsets": [value for _ in range(run_plan.prompt_length) for value in (0, 1)],
        "base_offsets_dtype": "int32",
        "base_offsets_len": prompt_base_offsets_len,
        "base_offsets_nbytes": prompt_base_offsets_nbytes,
        "live_counts": list(range(run_plan.prompt_length)),
        "live_counts_dtype": "int64",
        "live_counts_len": run_plan.prompt_length,
        "live_counts_nbytes": prompt_live_counts_nbytes,
        "position_tensor_role": "prompt_row_positions",
        "max_live_count": run_plan.prompt_length - 1,
        "total_span_input_nbytes": prompt_base_offsets_nbytes + prompt_live_counts_nbytes,
    }
    assert payload["decode_span_inputs"] == {
        "block_size": 256,
        "block_table_len": 2,
        "base_offsets": [0, 1],
        "base_offsets_dtype": "int32",
        "base_offsets_len": 2,
        "base_offsets_nbytes": 8,
        "kv_write_position": run_plan.prompt_length,
        "kv_write_position_dtype": "int64",
        "kv_write_position_nbytes": 8,
        "attention_live_counts": [run_plan.prompt_length],
        "attention_live_counts_dtype": "int64",
        "attention_live_counts_len": 1,
        "attention_live_counts_nbytes": 8,
        "max_live_count": run_plan.prompt_length,
        "total_span_input_nbytes": 16,
    }
    assert payload["span_input_total_nbytes"] == (
        prompt_base_offsets_nbytes + prompt_live_counts_nbytes + 16
    )
    assert payload["span_input_upload_manifest"] == {
        "entries": [
            {
                "name": "prompt_base_offsets",
                "source": "prompt_span_inputs.base_offsets",
                "kernel_args": ["prompt_kv_write.base_offsets"],
                "dtype": "int32",
                "shape": [run_plan.prompt_length, 2],
                "nbytes": prompt_base_offsets_nbytes,
            },
            {
                "name": "prompt_live_counts",
                "source": "prompt_span_inputs.live_counts",
                "kernel_args": ["prompt_kv_write.live_counts"],
                "dtype": "int64",
                "shape": [run_plan.prompt_length],
                "nbytes": prompt_live_counts_nbytes,
            },
            {
                "name": "decode_base_offsets",
                "source": "decode_span_inputs.base_offsets",
                "kernel_args": ["decode_kv_write.base_offsets", "decode_attention.base_offsets"],
                "dtype": "int32",
                "shape": [2],
                "nbytes": 8,
            },
            {
                "name": "decode_kv_write_position",
                "source": "decode_span_inputs.kv_write_position",
                "kernel_args": ["decode_kv_write.position"],
                "dtype": "int64",
                "shape": [],
                "nbytes": 8,
            },
            {
                "name": "decode_attention_live_counts",
                "source": "decode_span_inputs.attention_live_counts",
                "kernel_args": ["decode_attention.live_counts"],
                "dtype": "int64",
                "shape": [1],
                "nbytes": 8,
            },
        ],
        "entry_count": 5,
        "total_nbytes": prompt_base_offsets_nbytes + prompt_live_counts_nbytes + 24,
        "note": "Host-side upload manifest for metadata-only StepFun KV decode planning.",
    }
    prompt_base_payload = struct.pack(
        "<" + "i" * prompt_base_offsets_len,
        *[value for _ in range(run_plan.prompt_length) for value in (0, 1)],
    )
    prompt_live_payload = struct.pack(
        "<" + "q" * run_plan.prompt_length,
        *range(run_plan.prompt_length),
    )
    host_payload_bytes = run_plan.span_input_host_payload_bytes
    assert set(host_payload_bytes) == {
        "prompt_base_offsets",
        "prompt_live_counts",
        "decode_base_offsets",
        "decode_kv_write_position",
        "decode_attention_live_counts",
    }
    assert host_payload_bytes["prompt_base_offsets"] == prompt_base_payload
    assert host_payload_bytes["prompt_live_counts"] == prompt_live_payload
    assert host_payload_bytes["decode_base_offsets"] == struct.pack("<ii", 0, 1)
    assert host_payload_bytes["decode_kv_write_position"] == struct.pack("<q", run_plan.prompt_length)
    assert host_payload_bytes["decode_attention_live_counts"] == struct.pack("<q", run_plan.prompt_length)
    assert sum(len(payload_bytes) for payload_bytes in host_payload_bytes.values()) == (
        prompt_base_offsets_nbytes + prompt_live_counts_nbytes + 24
    )
    input_runtime = _FakeHipRuntime()
    input_upload = run_plan.upload_input_ids_payload(runtime=input_runtime)
    try:
        assert input_upload.token_count == run_plan.prompt_length
        assert input_upload.dtype == "int32"
        assert input_upload.buffer.nbytes == len(input_ids_payload)
        assert input_runtime.copies[input_upload.buffer.ptr] == input_ids_payload
        assert input_upload.payload_sha256 == hashlib.sha256(input_ids_payload).hexdigest()
        assert input_upload.to_dict() == {
            "token_count": run_plan.prompt_length,
            "dtype": "int32",
            "ptr": input_upload.buffer.ptr,
            "nbytes": len(input_ids_payload),
            "sha256": hashlib.sha256(input_ids_payload).hexdigest(),
        }
    finally:
        input_upload.free(runtime=input_runtime)
    assert input_runtime.freed == [input_upload.buffer.ptr]
    failing_input_runtime = _FakeHipRuntime(fail_on_copy_index=1)
    with pytest.raises(RuntimeError, match="simulated host-to-device copy failure"):
        run_plan.upload_input_ids_payload(runtime=failing_input_runtime)
    assert len(failing_input_runtime.allocations) == 1
    assert failing_input_runtime.freed == list(failing_input_runtime.allocations)
    assert failing_input_runtime.copies == {}
    host_payloads = payload["span_input_host_payloads"]
    assert host_payloads["entry_count"] == 5
    assert host_payloads["total_nbytes"] == prompt_base_offsets_nbytes + prompt_live_counts_nbytes + 24
    assert host_payloads["entries"][0] == {
        "name": "prompt_base_offsets",
        "source": "prompt_span_inputs.base_offsets",
        "dtype": "int32",
        "byte_order": "little",
        "value_count": prompt_base_offsets_len,
        "nbytes": prompt_base_offsets_nbytes,
        "sha256": hashlib.sha256(prompt_base_payload).hexdigest(),
        "preview_values": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    assert host_payloads["entries"][1]["sha256"] == hashlib.sha256(prompt_live_payload).hexdigest()
    assert host_payloads["entries"][3]["preview_values"] == [run_plan.prompt_length]
    upload_plan = payload["decode_input_upload_plan"]
    assert upload_plan["entry_count"] == 6
    assert upload_plan["upload_order"] == [
        "input_ids",
        "prompt_base_offsets",
        "prompt_live_counts",
        "decode_base_offsets",
        "decode_kv_write_position",
        "decode_attention_live_counts",
    ]
    assert upload_plan["cleanup_order"] == list(reversed(upload_plan["upload_order"]))
    assert upload_plan["input_token_nbytes"] == len(input_ids_payload)
    assert upload_plan["span_input_nbytes"] == host_payloads["total_nbytes"]
    assert upload_plan["total_nbytes"] == len(input_ids_payload) + host_payloads["total_nbytes"]
    assert upload_plan["consistency_checks"] == {
        "cleanup_order_reverses_upload_order": True,
        "entry_count_matches_upload_order": True,
        "entry_total_nbytes_matches": True,
        "input_token_hash_matches": True,
        "span_payload_hashes_match_manifest": True,
    }
    assert upload_plan["all_consistency_checks_passed"] is True
    assert upload_plan["entries"][0] == {
        "name": "input_ids",
        "source": "input_ids",
        "upload_group": "input_tokens",
        "dtype": "int32",
        "shape": [run_plan.prompt_length],
        "nbytes": len(input_ids_payload),
        "sha256": hashlib.sha256(input_ids_payload).hexdigest(),
    }
    assert upload_plan["entries"][1]["source"] == "prompt_span_inputs.base_offsets"
    assert upload_plan["streaming_runner_ready"] is False
    fake_runtime = _FakeHipRuntime()
    device_upload = run_plan.upload_span_input_payloads(runtime=fake_runtime)
    try:
        assert device_upload.names == (
            "prompt_base_offsets",
            "prompt_live_counts",
            "decode_base_offsets",
            "decode_kv_write_position",
            "decode_attention_live_counts",
        )
        assert device_upload.buffer_count == 5
        assert device_upload.total_nbytes == host_payloads["total_nbytes"]
        for name, buffer in device_upload.buffers.items():
            assert buffer.nbytes == len(host_payload_bytes[name])
            assert fake_runtime.copies[buffer.ptr] == host_payload_bytes[name]
            assert device_upload.payload_sha256[name] == hashlib.sha256(host_payload_bytes[name]).hexdigest()
        upload_payload = device_upload.to_dict()
        assert upload_payload["buffer_count"] == 5
        assert upload_payload["total_nbytes"] == host_payloads["total_nbytes"]
        assert upload_payload["buffers"]["prompt_base_offsets"]["nbytes"] == prompt_base_offsets_nbytes
    finally:
        device_upload.free(runtime=fake_runtime)
    assert fake_runtime.freed == [buffer.ptr for buffer in reversed(tuple(device_upload.buffers.values()))]
    combined_runtime = _FakeHipRuntime()
    decode_inputs = run_plan.upload_decode_inputs(runtime=combined_runtime)
    try:
        assert decode_inputs.buffer_count == 6
        assert decode_inputs.total_nbytes == len(input_ids_payload) + host_payloads["total_nbytes"]
        assert combined_runtime.copies[decode_inputs.input_ids.buffer.ptr] == input_ids_payload
        for name, buffer in decode_inputs.span_inputs.buffers.items():
            assert combined_runtime.copies[buffer.ptr] == host_payload_bytes[name]
        combined_payload = decode_inputs.to_dict()
        assert combined_payload["buffer_count"] == 6
        assert combined_payload["total_nbytes"] == len(input_ids_payload) + host_payloads["total_nbytes"]
        assert combined_payload["input_ids"]["token_count"] == run_plan.prompt_length
        assert combined_payload["span_inputs"]["buffer_count"] == 5
    finally:
        decode_inputs.free(runtime=combined_runtime)
    assert combined_runtime.freed == [
        *[buffer.ptr for buffer in reversed(tuple(decode_inputs.span_inputs.buffers.values()))],
        decode_inputs.input_ids.buffer.ptr,
    ]
    assert payload["stop_token_ids"] == [1, 2, 128007]
    assert payload["kv_dispatch_keys"]["decode_attention"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_attn_decode",
        "quant": STEPFUN_GGUF_KERNEL_QUANT,
        "variant": "bf16_split_k_gate_f32_spans",
    }
    assert payload["kv_decode_launch_operation_count"] == 135
    assert payload["kv_decode_launch_per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert payload["streaming_runner_ready"] is False
    assert payload["streaming_runner_blocker_count"] == 3
    assert payload["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert [blocker["name"] for blocker in payload["streaming_runner_blockers"]] == [
        "streaming_decode_loop_not_wired",
        "kv_kernel_trace_artifact_missing",
        "kv_backed_next_token_artifact_missing",
    ]
    assert all(blocker["ready"] is False for blocker in payload["streaming_runner_blockers"])


def test_stepfun_kv_decode_run_plan_frees_partial_uploads_after_copy_failure() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )
    run_plan = planner.plan_kv_decode_chat(
        [{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        context_pages=1,
        page_size=512,
    )
    fake_runtime = _FakeHipRuntime(fail_on_copy_index=2)

    with pytest.raises(RuntimeError, match="simulated host-to-device copy failure"):
        run_plan.upload_span_input_payloads(runtime=fake_runtime)

    allocated_ptrs = tuple(fake_runtime.allocations)
    assert fake_runtime.copy_count == 2
    assert len(allocated_ptrs) == 2
    assert fake_runtime.freed == [allocated_ptrs[1], allocated_ptrs[0]]
    assert fake_runtime.copies == {allocated_ptrs[0]: run_plan.span_input_host_payload_bytes["prompt_base_offsets"]}


def test_stepfun_kv_decode_run_plan_frees_combined_uploads_after_span_copy_failure() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )
    run_plan = planner.plan_kv_decode_chat(
        [{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        context_pages=1,
        page_size=512,
    )
    fake_runtime = _FakeHipRuntime(fail_on_copy_index=3)

    with pytest.raises(RuntimeError, match="simulated host-to-device copy failure"):
        run_plan.upload_decode_inputs(runtime=fake_runtime)

    allocated_ptrs = tuple(fake_runtime.allocations)
    assert fake_runtime.copy_count == 3
    assert len(allocated_ptrs) == 3
    assert fake_runtime.freed == [allocated_ptrs[2], allocated_ptrs[1], allocated_ptrs[0]]
    assert fake_runtime.copies == {
        allocated_ptrs[0]: run_plan.input_ids_payload_bytes,
        allocated_ptrs[1]: run_plan.span_input_host_payload_bytes["prompt_base_offsets"],
    }


def test_stepfun_kv_decode_run_plan_rejects_resource_span_too_small() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )

    with pytest.raises(ValueError, match="KV prompt span"):
        planner.plan_kv_decode_chat(
            [{"role": "user", "content": "hello"}],
            reasoning_effort="low",
            context_pages=1,
            page_size=16,
        )


def test_stepfun_short_context_decode_plan_rejects_long_prompts() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=32,
        max_new_tokens=1,
    )

    with pytest.raises(ValueError, match="max_context"):
        planner.plan_chat([{"role": "user", "content": "hello " * 128}], reasoning_effort="low")


def test_stepfun_text_decode_slot_paths_cover_validated_text_model_without_extra_modal_slots() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())

    slots = stepfun_text_decode_slot_paths(planner.model_map)

    assert slots[:4] == (
        "root.token_embedding",
        "root.rope_freqs",
        "root.output_norm",
        "root.lm_head",
    )
    assert len(slots) == len(set(slots)) == planner.info.tensor_count
    assert "layers.0.ffn_down" in slots
    assert "layers.3.ffn_gate_inp" in slots
    assert "layers.44.ffn_down_shexp" in slots
    forbidden_fragments = ("vision", "projector", "mmproj", "mtp", "nextn")
    assert not any(fragment in slot for slot in slots for fragment in forbidden_fragments)

    tensor_names: list[str] = []
    for slot in slots:
        parts = slot.split(".")
        if parts[0] == "root":
            tensor_names.append(planner.model_map.root(parts[1]).name)
        else:
            assert parts[0] == "layers"
            tensor_names.append(planner.model_map.layer(int(parts[1])).tensor(parts[2]).name)
    assert set(tensor_names) == set(planner.model_map.tensor_names)


def test_stepfun_text_decode_resource_plan_estimates_weight_and_kv_bytes() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())

    plan = planner.text_decode_resource_plan(context_pages=1, page_size=512)

    assert plan.backend == "hip_gfx1151"
    assert plan.context_pages == 1
    assert plan.page_size == 512
    assert plan.slot_paths == stepfun_text_decode_slot_paths(planner.model_map)
    assert plan.slot_count == planner.info.tensor_count == 754
    assert plan.resident_weight_nbytes == planner.info.total_tensor_nbytes
    assert plan.resident_weight_gib == pytest.approx(planner.info.total_tensor_nbytes / 2**30)
    assert len(plan.kv_layer_nbytes) == planner.model_map.config.block_count == 45
    assert plan.kv_layer_nbytes[0] == (1_048_576, 1_048_576)
    assert plan.kv_nbytes == 94_371_840
    assert plan.kv_gib == pytest.approx(94_371_840 / 2**30)
    assert plan.kv_nbytes == stepfun_kv_cache_nbytes(
        planner.model_map.config,
        context_pages=1,
        page_size=512,
    )
    assert plan.total_nbytes == plan.resident_weight_nbytes + plan.kv_nbytes
    payload = plan.to_dict()
    assert payload["backend"] == "hip_gfx1151"
    assert payload["slot_count"] == 754
    assert payload["slot_paths"][:4] == [
        "root.token_embedding",
        "root.rope_freqs",
        "root.output_norm",
        "root.lm_head",
    ]
    assert payload["resident_weight_nbytes"] == 102_499_149_312
    assert payload["context_pages"] == 1
    assert payload["page_size"] == 512
    assert payload["max_new_tokens"] == 1
    assert payload["kv_buffer_count"] == 90
    assert payload["kv_layer_nbytes"][0] == {
        "layer": 0,
        "key_nbytes": 1_048_576,
        "value_nbytes": 1_048_576,
    }
    assert payload["kv_nbytes"] == 94_371_840
    assert payload["total_nbytes"] == plan.total_nbytes
    kv_kernel_plan = payload["kv_decode_kernel_plan"]
    assert kv_kernel_plan["model_quant"] == STEPFUN_GGUF_KERNEL_QUANT
    assert kv_kernel_plan["kv_storage_dtype"] == "bf16"
    assert kv_kernel_plan["decode_attention_kind"] == "splitk_gate_f32"
    assert kv_kernel_plan["max_context"] == 512
    assert kv_kernel_plan["max_new_tokens"] == 1
    assert kv_kernel_plan["max_prompt_rows"] == 511
    assert kv_kernel_plan["attention_block_size"] == STEPFUN_KV_ATTENTION_BLOCK_SIZE == 256
    assert kv_kernel_plan["attention_block_table_len"] == 2
    assert kv_kernel_plan["attention_capacity_tokens"] == 512
    assert kv_kernel_plan["decode_span"] == {
        "block_size": 256,
        "block_table_len": 2,
        "live_counts_len": 1,
        "max_live_count": 511,
        "capacity_tokens": 512,
        "shape_compatible": True,
    }
    assert kv_kernel_plan["prompt_span"] == {
        "block_size": 256,
        "max_prompt_rows": 511,
        "block_table_len_per_row": 2,
        "base_offsets_len_formula": "rows * 2",
        "live_counts_len_formula": "rows",
        "row_positions_required": True,
        "shape_compatible": True,
    }
    assert kv_kernel_plan["decode_span_shape_compatible"] is True
    assert kv_kernel_plan["prompt_span_shape_compatible"] is True
    assert kv_kernel_plan["span_shape_compatible"] is True
    assert kv_kernel_plan["all_registered"] is True
    assert kv_kernel_plan["dispatch_keys"]["prompt_kv_write"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_kv_write",
        "quant": STEPFUN_GGUF_KERNEL_QUANT,
        "variant": "mixed_bf16_prompt_spans",
    }
    assert kv_kernel_plan["dispatch_keys"]["decode_attention"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_attn_decode",
        "quant": STEPFUN_GGUF_KERNEL_QUANT,
        "variant": "bf16_split_k_gate_f32_spans",
    }
    launch_schedule = payload["kv_decode_launch_schedule"]
    assert launch_schedule["source"] == "text_decode_resource_plan"
    assert launch_schedule["layer_count"] == 45
    assert launch_schedule["operation_count"] == 135
    assert launch_schedule["per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert launch_schedule["first_layer_ops"] == [
        "layers.0.prompt_kv_write",
        "layers.0.decode_kv_write",
        "layers.0.decode_attention",
    ]
    assert launch_schedule["last_layer_ops"] == [
        "layers.44.prompt_kv_write",
        "layers.44.decode_kv_write",
        "layers.44.decode_attention",
    ]
    assert launch_schedule["stages"] == [
        {
            "name": "prompt_prefill_kv_write",
            "dispatch_key": "prompt_kv_write",
            "span_contract": "prompt_span",
            "layer_count": 45,
            "ready": True,
        },
        {
            "name": "one_token_decode_kv_write",
            "dispatch_key": "decode_kv_write",
            "span_contract": "decode_span",
            "layer_count": 45,
            "ready": True,
        },
        {
            "name": "one_token_gated_attention_decode",
            "dispatch_key": "decode_attention",
            "span_contract": "decode_span",
            "layer_count": 45,
            "ready": True,
        },
    ]
    assert launch_schedule["all_stage_dispatch_ready"] is True
    assert launch_schedule["streaming_runner_ready"] is False
    assert launch_schedule["streaming_runner_blocker_count"] == 3
    assert launch_schedule["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert [blocker["name"] for blocker in launch_schedule["streaming_runner_blockers"]] == [
        "streaming_decode_loop_not_wired",
        "kv_kernel_trace_artifact_missing",
        "kv_backed_next_token_artifact_missing",
    ]

    with pytest.raises(ValueError, match="context_pages"):
        planner.text_decode_resource_plan(context_pages=0, page_size=512)


def test_stepfun_kv_decode_kernel_plan_resolves_step35_registry_keys() -> None:
    plan = stepfun_kv_decode_kernel_plan(backend="hip_gfx1151")

    assert plan.model_quant == STEPFUN_GGUF_KERNEL_QUANT
    assert plan.kv_storage_dtype == "bf16"
    assert plan.decode_attention_kind == "splitk_gate_f32"
    assert plan.max_context == 512
    assert plan.max_new_tokens == 1
    assert plan.max_prompt_rows == 511
    assert plan.decode_max_live_count == 511
    assert plan.attention_block_size == STEPFUN_KV_ATTENTION_BLOCK_SIZE == 256
    assert plan.attention_block_table_len == 2
    assert plan.attention_capacity_tokens == 512
    assert plan.decode_span_contract == {
        "block_size": 256,
        "block_table_len": 2,
        "live_counts_len": 1,
        "max_live_count": 511,
        "capacity_tokens": 512,
        "shape_compatible": True,
    }
    assert plan.prompt_span_contract == {
        "block_size": 256,
        "max_prompt_rows": 511,
        "block_table_len_per_row": 2,
        "base_offsets_len_formula": "rows * 2",
        "live_counts_len_formula": "rows",
        "row_positions_required": True,
        "shape_compatible": True,
    }
    assert plan.decode_span_shape_compatible is True
    assert plan.prompt_span_shape_compatible is True
    assert plan.span_shape_compatible is True
    assert plan.all_registered is True
    assert plan.registered == {
        "prompt_kv_write": True,
        "decode_kv_write": True,
        "decode_attention": True,
    }
    assert plan.dispatch_keys["decode_attention"] == KernelKey(
        "hip_gfx1151", "paged_attn_decode", STEPFUN_GGUF_KERNEL_QUANT, "bf16_split_k_gate_f32_spans"
    )


def test_stepfun_kv_decode_kernel_plan_rounds_block_table_to_attention_block_size() -> None:
    plan = stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=513)

    assert plan.attention_block_size == STEPFUN_KV_ATTENTION_BLOCK_SIZE
    assert plan.attention_block_table_len == 3
    assert plan.max_prompt_rows == 512
    assert plan.decode_max_live_count == 512
    assert plan.attention_capacity_tokens == 768
    assert plan.prompt_span_contract["base_offsets_len_formula"] == "rows * 3"
    assert plan.span_shape_compatible is True

    with pytest.raises(ValueError, match="max_context"):
        stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=512, max_new_tokens=0)
    with pytest.raises(ValueError, match="at least one prompt token"):
        stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=1, max_new_tokens=1)


def test_stepfun_decode_planner_does_not_import_torch() -> None:
    had_torch = "torch" in sys.modules

    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())
    planner.plan_chat([{"role": "user", "content": "hello"}], reasoning_effort="low")

    if not had_torch:
        assert "torch" not in sys.modules
