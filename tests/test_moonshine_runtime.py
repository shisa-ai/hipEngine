from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import free, malloc, memory_stats, reset_memory_stats
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    alias_device_allocation,
)
from hipengine.loading.moonshine import MoonshineLoadedModel
from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models.moonshine import (
    expected_moonshine_weight_shapes,
    parse_moonshine_model_spec,
)
from hipengine.runtime.moonshine import (
    MoonshineDecoderLibraries,
    MoonshineResidentRuntime,
    NoAllocationError,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000
        self.malloc_calls: list[int] = []
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, int, int]] = []
        self.sets: list[tuple[int, int, int, int]] = []
        self.created_streams: list[int] = []
        self.destroyed_streams: list[int] = []
        self.created_events: list[int] = []
        self.destroyed_events: list[int] = []
        self.synchronized_streams: list[int] = []
        self.fail_malloc_at: int | None = None

    def malloc(self, nbytes: int) -> int:
        if self.fail_malloc_at is not None and len(self.malloc_calls) >= self.fail_malloc_at:
            raise RuntimeError("injected malloc failure")
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.malloc_calls.append(int(nbytes))
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(kind)))

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        self.sets.append((int(dst), int(value), int(nbytes), int(stream)))

    def stream_create(self, *, nonblocking: bool = True, priority=None) -> int:
        del nonblocking, priority
        stream = 0x5000 + len(self.created_streams)
        self.created_streams.append(stream)
        return stream

    def stream_destroy(self, stream: int) -> None:
        self.destroyed_streams.append(stream)

    def stream_synchronize(self, stream: int) -> None:
        self.synchronized_streams.append(stream)

    def event_create(self, *, flags: int = 0) -> int:
        del flags
        event = 0x6000 + len(self.created_events)
        self.created_events.append(event)
        return event

    def event_destroy(self, event: int) -> None:
        self.destroyed_events.append(event)

    def event_record(self, event: int, stream: int = 0) -> None:
        del event, stream

    def event_synchronize(self, event: int) -> None:
        del event

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        del start, stop
        return 0.0


def model_config() -> dict:
    return {
        "architectures": ["MoonshineForConditionalGeneration"],
        "attention_bias": False,
        "bos_token_id": 1,
        "decoder_hidden_act": "silu",
        "decoder_num_attention_heads": 8,
        "decoder_num_hidden_layers": 8,
        "decoder_num_key_value_heads": 8,
        "decoder_start_token_id": 1,
        "dtype": "float32",
        "encoder_hidden_act": "gelu",
        "encoder_num_attention_heads": 8,
        "encoder_num_hidden_layers": 8,
        "encoder_num_key_value_heads": 8,
        "eos_token_id": 2,
        "hidden_size": 416,
        "intermediate_size": 1664,
        "is_encoder_decoder": True,
        "max_position_embeddings": 194,
        "model_type": "moonshine",
        "pad_head_dim_to_multiple_of": 8,
        "pad_token_id": 2,
        "partial_rotary_factor": 0.62,
        "rope_parameters": {
            "partial_rotary_factor": 0.62,
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
        "tie_word_embeddings": True,
        "vocab_size": 36_864,
    }


def generation_config() -> dict:
    return {
        "bos_token_id": 1,
        "decoder_start_token_id": 1,
        "do_sample": False,
        "eos_token_id": [2],
        "max_length": 195,
        "num_beams": 5,
        "pad_token_id": 2,
        "use_cache": True,
    }


def fake_loaded_model(runtime: FakeRuntime) -> MoonshineLoadedModel:
    spec = parse_moonshine_model_spec(model_config(), generation_config())
    baseline = memory_stats()
    nbytes = spec.vocab_size * spec.hidden_size * DType.FP16.itemsize
    buffer = malloc(nbytes, runtime=runtime)  # type: ignore[arg-type]
    source = TensorInfo(
        spec.embedding_weight_name,
        Path("fake.safetensors"),
        "F32",
        (spec.vocab_size, spec.hidden_size),
    )
    tensor = Tensor.from_handle(
        buffer.ptr,
        source.shape,
        DType.FP16,
        Device("hip", 0),
    )
    owner = DeviceTensorAllocation(spec.embedding_weight_name, source, buffer, tensor)
    alias = alias_device_allocation(
        spec.lm_head_alias_name,
        owner,
        source.shape,
        DType.FP16,
    )
    allocations = {
        spec.embedding_weight_name: owner,
        spec.lm_head_alias_name: alias,
    }
    for name, shape in expected_moonshine_weight_shapes(spec).items():
        if name == spec.embedding_weight_name:
            continue
        weight_source = TensorInfo(name, Path("fake.safetensors"), "F32", shape)
        allocations[name] = DeviceTensorAllocation(
            name,
            weight_source,
            owner.buffer,
            Tensor.from_handle(owner.buffer.ptr, shape, DType.FP16, Device("hip", 0)),
            owns_buffer=False,
        )
    weights = DeviceWeightMap(allocations)
    index = WeightIndex(Path("/fake/moonshine"), model_config(), {source.name: source}, (source.shard_path,))
    return MoonshineLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
    )


def setup_function() -> None:
    reset_memory_stats()


def test_resident_runtime_owns_fixed_buffers_aliases_stream_events_and_bucket() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
        device=Device("hip", 0),
    )
    try:
        assert resident.encoder_frames == 40
        assert resident.stream in runtime.created_streams
        assert (resident.start_event, resident.stop_event) == tuple(runtime.created_events)
        assert resident.weights[resident.spec.embedding_weight_name].ptr == resident.weights[
            resident.spec.lm_head_alias_name
        ].ptr
        assert resident.weights.allocation(resident.spec.lm_head_alias_name).owns_buffer is False
        assert resident.tensor("rope_cos").shape == (194, 16)
        assert resident.tensor("rope_sin").shape == (194, 16)
        assert resident.tensor("encoder_hidden").shape == (1, 40, 416)
        assert resident.tensor("encoder_attention_mask").shape == (1, 40)
        assert resident.tensor("self_kv").shape == (8, 2, 1, 8, 194, 52)
        assert resident.tensor("cross_kv").shape == (8, 2, 1, 8, 40, 52)
        assert resident.tensor("logits").shape == (1, 36_864)
        assert resident.self_cache(7).key.shape == (1, 8, 194, 52)
        assert resident.cross_cache(0).value.shape == (1, 8, 40, 52)
        assert resident.resident_nbytes > resident.loaded_model.owned_weight_bytes
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == resident.resident_nbytes
        assert stats["active_allocations"] == 1 + len(resident.workspace.names)
    finally:
        resident.close()

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0
    assert runtime.destroyed_events == list(reversed(runtime.created_events))
    assert runtime.destroyed_streams == runtime.created_streams
    freed_once = list(runtime.freed)
    resident.close()
    assert runtime.freed == freed_once


def test_reset_and_cross_cache_state_reuse_fixed_addresses_without_allocating() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=207,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        pointers = {name: resident.tensor(name).ptr for name in resident.workspace.names}
        malloc_count = len(runtime.malloc_calls)
        resident.mark_cross_cache_ready(207)
        assert resident.cross_cache_valid is True
        resident.set_self_cache_length(17)
        resident.reset_generation(clear_cross_cache=False)
        assert resident.cross_cache_valid is True
        assert resident.self_cache_length == 0
        resident.reset_generation(clear_cross_cache=True)
        assert resident.cross_cache_valid is False
        assert {name: resident.tensor(name).ptr for name in resident.workspace.names} == pointers
        assert len(runtime.malloc_calls) == malloc_count
        assert runtime.synchronized_streams
    finally:
        resident.close()


def test_no_allocation_region_detects_allocate_free_and_unprepared_token_step() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with resident.no_allocation_region("empty-token-step"):
            pass
        with pytest.raises(NoAllocationError, match="allocated"):
            with resident.no_allocation_region("bad-token-step"):
                temporary = malloc(16, runtime=runtime)  # type: ignore[arg-type]
                free(temporary, runtime=runtime)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="prepared"):
            resident.token_step()
    finally:
        resident.close()


def test_decoder_precompute_and_token_step_follow_the_unfused_fixed_address_chain() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []

    class FakeKernel:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, *args):
            trace.append((self.name, args))
            return 0

    class FakeLibrary:
        def __init__(self, *symbols: str) -> None:
            for symbol in symbols:
                setattr(self, symbol, FakeKernel(symbol))

    libraries = MoonshineDecoderLibraries(
        projection=FakeLibrary(
            "hipengine_moonshine_f16_lm_head_projection",
            "hipengine_moonshine_f16_lm_head_projection_wave8",
            "hipengine_moonshine_f16_projection",
            "hipengine_moonshine_f16_projection_bias",
            "hipengine_moonshine_f16_projection_pair_head_major",
            "hipengine_moonshine_f16_projection_triple",
        ),
        dense_projection=FakeLibrary("hipengine_dense_gemv_out_fp16"),
        layernorm=FakeLibrary("hipengine_moonshine_layernorm_fp16"),
        glue=FakeLibrary(
            "hipengine_moonshine_argmax_fp16",
            "hipengine_moonshine_embedding_lookup_fp16",
            "hipengine_moonshine_partial_rope_cache_append_fp16",
            "hipengine_moonshine_residual_fp16",
        ),
        mlp=FakeLibrary("hipengine_moonshine_gated_silu_fp16"),
        attention=FakeLibrary(
            "hipengine_moonshine_cross_attention_fp16",
            "hipengine_moonshine_cross_attention_parallel_fp16",
            "hipengine_moonshine_self_attention_fp16",
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="prepared"):
            resident.precompute_cross_kv()
        resident.prepare_decoder_kernels(libraries=libraries)
        with pytest.raises(RuntimeError, match="encoder state"):
            resident.precompute_cross_kv()
        with pytest.raises(ValueError, match="encoder hidden"):
            resident.set_encoder_state(
                np.zeros((1, 39, 416), dtype=np.float16),
                np.ones((1, 40), dtype=np.int32),
            )
        resident.set_encoder_state(
            np.zeros((1, 40, 416), dtype=np.float16),
            np.ones((1, 40), dtype=np.int32),
        )
        resident.precompute_cross_kv()
        assert resident.cross_cache_valid is True
        assert [name for name, _ in trace] == [
            "hipengine_moonshine_f16_projection_pair_head_major"
        ] * 8
        assert [args[-2] for _, args in trace] == [32] * 8
        trace.clear()
        malloc_count = len(runtime.malloc_calls)
        resident.set_decode_state(token_id=1, position=0)
        with resident.no_allocation_region("decoder-token-step"):
            resident.token_step()
        assert len(runtime.malloc_calls) == malloc_count
        assert resident.self_cache_length == 1
        assert resident.decode_position is None
        per_layer = [
            "hipengine_moonshine_layernorm_fp16",
            "hipengine_moonshine_f16_projection_triple",
            "hipengine_moonshine_partial_rope_cache_append_fp16",
            "hipengine_moonshine_self_attention_fp16",
            "hipengine_dense_gemv_out_fp16",
            "hipengine_moonshine_residual_fp16",
            "hipengine_moonshine_layernorm_fp16",
            "hipengine_dense_gemv_out_fp16",
            "hipengine_moonshine_cross_attention_parallel_fp16",
            "hipengine_dense_gemv_out_fp16",
            "hipengine_moonshine_residual_fp16",
            "hipengine_moonshine_layernorm_fp16",
            "hipengine_moonshine_f16_projection_bias",
            "hipengine_moonshine_gated_silu_fp16",
            "hipengine_moonshine_f16_projection_bias",
            "hipengine_moonshine_residual_fp16",
        ]
        expected = ["hipengine_moonshine_embedding_lookup_fp16"]
        for _ in range(8):
            expected.extend(per_layer)
        expected.extend(
            [
                "hipengine_moonshine_layernorm_fp16",
                "hipengine_moonshine_f16_lm_head_projection_wave8",
                "hipengine_moonshine_argmax_fp16",
            ]
        )
        assert [name for name, _ in trace] == expected
        projection_threads = {
            name: [args[-2] for call_name, args in trace if call_name == name]
            for name in (
                "hipengine_dense_gemv_out_fp16",
                "hipengine_moonshine_f16_projection_bias",
                "hipengine_moonshine_f16_projection_triple",
            )
        }
        assert projection_threads == {
            "hipengine_dense_gemv_out_fp16": [64] * 24,
            "hipengine_moonshine_f16_projection_bias": [32, 64] * 8,
            "hipengine_moonshine_f16_projection_triple": [32] * 8,
        }
        with pytest.raises(RuntimeError, match="reset generation"):
            resident.set_encoder_state(
                np.zeros((1, 40, 416), dtype=np.float16),
                np.ones((1, 40), dtype=np.int32),
            )
        with pytest.raises(ValueError, match="sequential"):
            resident.set_decode_state(token_id=1, position=2)
        resident.set_decode_state(token_id=1, position=1)
        resident.reset_generation(clear_cross_cache=False)
        assert resident.cross_cache_valid is True
        assert resident.decode_position is None
        assert resident.self_cache_length == 0
    finally:
        resident.close()


def test_runtime_rejects_unknown_encoder_bucket_and_cache_state_drift() -> None:
    runtime = FakeRuntime()
    loaded = fake_loaded_model(runtime)
    with pytest.raises(ValueError, match="encoder frame bucket"):
        MoonshineResidentRuntime(
            loaded_model=loaded,
            encoder_frames=41,
            runtime=runtime,  # type: ignore[arg-type]
        )
    assert memory_stats()["current_allocated_bytes"] == 0

    loaded = fake_loaded_model(runtime)
    resident = MoonshineResidentRuntime(
        loaded_model=loaded,
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(ValueError, match="encoder_frames"):
            resident.mark_cross_cache_ready(42)
        with pytest.raises(ValueError, match="self cache length"):
            resident.set_self_cache_length(195)
    finally:
        resident.close()


def test_partial_initialization_failure_unwinds_weights_workspace_events_and_stream() -> None:
    runtime = FakeRuntime()
    loaded = fake_loaded_model(runtime)
    runtime.fail_malloc_at = len(runtime.malloc_calls) + 3
    with pytest.raises(RuntimeError, match="injected malloc failure"):
        MoonshineResidentRuntime(
            loaded_model=loaded,
            encoder_frames=40,
            runtime=runtime,  # type: ignore[arg-type]
        )
    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0
    assert runtime.destroyed_events == list(reversed(runtime.created_events))
    assert runtime.destroyed_streams == runtime.created_streams
