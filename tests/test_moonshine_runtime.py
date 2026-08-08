from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc, memory_stats, reset_memory_stats
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    alias_device_allocation,
)
from hipengine.loading.moonshine import (
    MoonshineLoadedModel,
    MoonshineW8A16Tensor,
    MoonshineW8A16Weights,
    moonshine_w8a16_source_names,
    normalize_moonshine_w8a16_families,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models.moonshine import (
    expected_moonshine_weight_shapes,
    parse_moonshine_model_spec,
)
from hipengine.runtime.moonshine import (
    MoonshineDecoderLibraries,
    MoonshineResidentRuntime,
    NoAllocationError,
    _moonshine_self_attention_threads,
    _moonshine_token_graph_bucket,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000
        self.malloc_calls: list[int] = []
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, int, int]] = []
        self.async_copies: list[tuple[int, int, int, int, int]] = []
        self.sets: list[tuple[int, int, int, int]] = []
        self.created_streams: list[int] = []
        self.destroyed_streams: list[int] = []
        self.created_events: list[int] = []
        self.destroyed_events: list[int] = []
        self.synchronized_streams: list[int] = []
        self.capture_begins: list[tuple[int, int]] = []
        self.capture_ends: list[int] = []
        self.instantiated_graphs: list[int] = []
        self.launched_graphs: list[tuple[int, int]] = []
        self.destroyed_graph_execs: list[int] = []
        self.destroyed_graphs: list[int] = []
        self.fail_malloc_at: int | None = None
        self.fail_graph_instantiate_at: int | None = None

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

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind, stream: int) -> None:
        self.async_copies.append(
            (int(dst), int(src), int(nbytes), int(kind), int(stream))
        )

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

    def stream_begin_capture(self, stream: int, mode: int = 2) -> None:
        self.capture_begins.append((int(stream), int(mode)))

    def stream_end_capture(self, stream: int) -> int:
        self.capture_ends.append(int(stream))
        return 0x7000 + len(self.capture_ends) - 1

    def graph_instantiate(self, graph: int) -> int:
        self.instantiated_graphs.append(int(graph))
        if (
            self.fail_graph_instantiate_at is not None
            and len(self.instantiated_graphs) >= self.fail_graph_instantiate_at
        ):
            raise RuntimeError("injected graph instantiate failure")
        return int(graph) + 0x1000

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        self.launched_graphs.append((int(graph_exec), int(stream)))

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self.destroyed_graph_execs.append(int(graph_exec))

    def graph_destroy(self, graph: int) -> None:
        self.destroyed_graphs.append(int(graph))

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


class FakeKernel:
    def __init__(self, name: str, trace: list[tuple[str, tuple[object, ...]]]) -> None:
        self.name = name
        self.trace = trace

    def __call__(self, *args):
        self.trace.append((self.name, args))
        return 0


class FakeLibrary:
    def __init__(self, trace: list[tuple[str, tuple[object, ...]]], *symbols: str) -> None:
        for symbol in symbols:
            setattr(self, symbol, FakeKernel(symbol, trace))


def fake_decoder_libraries(trace: list[tuple[str, tuple[object, ...]]]) -> MoonshineDecoderLibraries:
    return MoonshineDecoderLibraries(
        projection=FakeLibrary(
            trace,
            "hipengine_moonshine_f16_lm_head_projection",
            "hipengine_moonshine_f16_lm_head_projection_wave8",
            "hipengine_moonshine_f16_lm_head_projection_wave8_top1",
            "hipengine_moonshine_f16_projection",
            "hipengine_moonshine_f16_projection_bias",
            "hipengine_moonshine_f16_projection_bias_gated_silu",
            "hipengine_moonshine_f16_projection_bias_residual",
            "hipengine_moonshine_f16_projection_pair_head_major",
            "hipengine_moonshine_f16_projection_triple",
        ),
        dense_projection=FakeLibrary(trace, "hipengine_dense_gemv_out_fp16"),
        layernorm=FakeLibrary(
            trace,
            "hipengine_moonshine_layernorm_fp16",
            "hipengine_moonshine_residual_layernorm_fp16",
        ),
        glue=FakeLibrary(
            trace,
            "hipengine_moonshine_argmax_fp16",
            "hipengine_moonshine_embedding_lookup_fp16",
            "hipengine_moonshine_partial_rope_cache_append_fp16",
            "hipengine_moonshine_residual_fp16",
        ),
        mlp=FakeLibrary(trace, "hipengine_moonshine_gated_silu_fp16"),
        attention=FakeLibrary(
            trace,
            "hipengine_moonshine_cross_attention_fp16",
            "hipengine_moonshine_cross_attention_parallel_fp16",
            "hipengine_moonshine_self_attention_branch_fp16",
            "hipengine_moonshine_self_attention_parallel_fp16",
            "hipengine_moonshine_self_attention_fp16",
        ),
        w8a16=FakeLibrary(
            trace,
            "hipengine_moonshine_w8a16_lm_head_wave8",
            "hipengine_moonshine_w8a16_projection",
            "hipengine_moonshine_w8a16_mlp_fc1_gated_silu",
            "hipengine_moonshine_w8a16_mlp_fc2_residual",
            "hipengine_moonshine_w8a16_qkv_triple",
            "hipengine_moonshine_w8a16_cross_kv_pair_head_major",
        ),
    )


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


def fake_loaded_model(
    runtime: FakeRuntime,
    *,
    w8a16_families=(),
) -> MoonshineLoadedModel:
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
    selected = normalize_moonshine_w8a16_families(w8a16_families)
    w8_tensors = {}
    for name in moonshine_w8a16_source_names(spec, selected):
        shape = expected_moonshine_weight_shapes(spec)[name]
        qbuffer = malloc(int(np.prod(shape, dtype=np.int64)), runtime=runtime)  # type: ignore[arg-type]
        scalebuffer = malloc(shape[0] * DType.FP32.itemsize, runtime=runtime)  # type: ignore[arg-type]
        weight_source = TensorInfo(name, Path("fake.safetensors"), "F32", shape)
        qsource = TensorInfo(f"{name}.w8a16.qweight", Path("fake.safetensors"), "I8", shape)
        scalesource = TensorInfo(
            f"{name}.w8a16.scale", Path("fake.safetensors"), "F32", (shape[0],)
        )
        qallocation = DeviceTensorAllocation(
            qsource.name,
            qsource,
            qbuffer,
            Tensor.from_handle(qbuffer.ptr, shape, DType.INT8, Device("hip", 0)),
        )
        scaleallocation = DeviceTensorAllocation(
            scalesource.name,
            scalesource,
            scalebuffer,
            Tensor.from_handle(
                scalebuffer.ptr, (shape[0],), DType.FP32, Device("hip", 0)
            ),
        )
        if name == spec.embedding_weight_name:
            family = "lm_head"
        elif name.endswith(".mlp.fc1.weight"):
            family = "mlp_fc1"
        elif name.endswith(".mlp.fc2.weight"):
            family = "mlp_fc2"
        elif ".self_attn." in name:
            family = "self_attention"
        else:
            family = "cross_attention"
        w8_tensors[name] = MoonshineW8A16Tensor(
            source_name=name,
            family=family,
            qweight=qallocation,
            scales=scaleallocation,
            source_fp16_nbytes=int(np.prod(shape, dtype=np.int64)) * DType.FP16.itemsize,
        )
    w8a16 = MoonshineW8A16Weights(selected, w8_tensors) if selected else None
    index = WeightIndex(Path("/fake/moonshine"), model_config(), {source.name: source}, (source.shard_path,))
    return MoonshineLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
        w8a16=w8a16,
    )


def setup_function() -> None:
    reset_memory_stats()


def test_moonshine_self_attention_thread_buckets() -> None:
    with pytest.raises(ValueError, match="positive"):
        _moonshine_self_attention_threads(0)
    assert _moonshine_self_attention_threads(1) == 64
    assert _moonshine_self_attention_threads(2) == 128
    assert _moonshine_self_attention_threads(3) == 128
    assert _moonshine_self_attention_threads(4) == 256
    assert _moonshine_self_attention_threads(193) == 256


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


def test_device_encoder_handoff_zero_pads_and_copies_fixed_prefix_without_allocating() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        pointers = {
            name: resident.tensor(name).ptr
            for name in ("encoder_hidden", "encoder_attention_mask")
        }
        malloc_count = len(runtime.malloc_calls)
        initial_set_count = len(runtime.sets)
        resident.set_encoder_state_from_device(
            hidden_fp16_ptr=0x900000,
            attention_mask_int32_ptr=0xA00000,
            source_frames=24,
        )

        assert resident.encoder_state_valid is True
        assert resident.cross_cache_valid is False
        assert runtime.sets[initial_set_count:] == [
            (pointers["encoder_hidden"], 0, 40 * 416 * 2, resident.stream),
            (pointers["encoder_attention_mask"], 0, 40 * 4, resident.stream),
        ]
        assert runtime.async_copies[-2:] == [
            (
                pointers["encoder_hidden"],
                0x900000,
                24 * 416 * 2,
                int(HipMemcpyKind.DEVICE_TO_DEVICE),
                resident.stream,
            ),
            (
                pointers["encoder_attention_mask"],
                0xA00000,
                24 * 4,
                int(HipMemcpyKind.DEVICE_TO_DEVICE),
                resident.stream,
            ),
        ]
        assert len(runtime.malloc_calls) == malloc_count
        assert resident.tensor("encoder_hidden").ptr == pointers["encoder_hidden"]
        assert resident.tensor("encoder_attention_mask").ptr == pointers[
            "encoder_attention_mask"
        ]
        assert runtime.synchronized_streams[-1] == resident.stream

        for kwargs, message in (
            ({"hidden_fp16_ptr": 0, "attention_mask_int32_ptr": 1, "source_frames": 24}, "pointers"),
            ({"hidden_fp16_ptr": 1, "attention_mask_int32_ptr": 0, "source_frames": 24}, "pointers"),
            ({"hidden_fp16_ptr": 1, "attention_mask_int32_ptr": 2, "source_frames": 0}, "source_frames"),
            ({"hidden_fp16_ptr": 1, "attention_mask_int32_ptr": 2, "source_frames": 41}, "source_frames"),
        ):
            with pytest.raises(ValueError, match=message):
                resident.set_encoder_state_from_device(**kwargs)
        resident.set_self_cache_length(1)
        with pytest.raises(RuntimeError, match="reset generation"):
            resident.set_encoder_state_from_device(
                hidden_fp16_ptr=1,
                attention_mask_int32_ptr=2,
                source_frames=24,
            )
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

    libraries = fake_decoder_libraries(trace)
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
            "hipengine_moonshine_residual_layernorm_fp16",
            "hipengine_dense_gemv_out_fp16",
            "hipengine_moonshine_cross_attention_parallel_fp16",
            "hipengine_dense_gemv_out_fp16",
            "hipengine_moonshine_residual_layernorm_fp16",
            "hipengine_moonshine_f16_projection_bias_gated_silu",
            "hipengine_moonshine_f16_projection_bias_residual",
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
                "hipengine_moonshine_f16_projection_bias_gated_silu",
                "hipengine_moonshine_f16_projection_bias_residual",
                "hipengine_moonshine_f16_projection_triple",
            )
        }
        assert projection_threads == {
            "hipengine_dense_gemv_out_fp16": [64] * 24,
            "hipengine_moonshine_f16_projection_bias_gated_silu": [32] * 8,
            "hipengine_moonshine_f16_projection_bias_residual": [64] * 8,
            "hipengine_moonshine_f16_projection_triple": [32] * 8,
        }
        with pytest.raises(RuntimeError, match="reset generation"):
            resident.set_encoder_state(
                np.zeros((1, 40, 416), dtype=np.float16),
                np.ones((1, 40), dtype=np.int32),
            )
        with pytest.raises(ValueError, match="sequential"):
            resident.set_decode_state(token_id=1, position=2)
        trace.clear()
        resident.set_decode_state(token_id=1, position=1)
        with resident.no_allocation_region("decoder-token-step-position-1"):
            resident.token_step()
        names = [name for name, _ in trace]
        assert names.count("hipengine_moonshine_self_attention_parallel_fp16") == 8
        self_attention_threads = [
            args[-2]
            for name, args in trace
            if name == "hipengine_moonshine_self_attention_parallel_fp16"
        ]
        assert self_attention_threads == [64] * 8
        assert "hipengine_moonshine_self_attention_branch_fp16" not in names
        resident.reset_generation(clear_cross_cache=False)
        assert resident.cross_cache_valid is True
        assert resident.decode_position is None
        assert resident.self_cache_length == 0
    finally:
        resident.close()


def test_exact_wave8_top1_route_rejects_unknown_route_and_int8_lm_head() -> None:
    with pytest.raises(ValueError, match="lm_head_route"):
        MoonshineResidentRuntime(
            model_path="/not-loaded",
            encoder_frames=40,
            runtime=FakeRuntime(),  # type: ignore[arg-type]
            lm_head_route="unknown",
        )

    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="requires the exact FP16"):
        MoonshineResidentRuntime(
            loaded_model=fake_loaded_model(runtime, w8a16_families=("lm_head",)),
            encoder_frames=40,
            runtime=runtime,  # type: ignore[arg-type]
            lm_head_route="wave8_top1",
        )
    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


def test_exact_wave8_top1_route_owns_bounded_scratch_and_keeps_fallback_explicit() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
        lm_head_route="wave8_top1",
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    try:
        contract = resident.lm_head_contract()
        assert contract == {
            "route": "wave8_top1",
            "materializes_full_fp16_logits": True,
            "stable_lowest_id_top1": True,
            "partial_count": 4_608,
            "partial_value_dtype": "fp16",
            "partial_index_dtype": "int64",
            "fallback": "wave8_argmax",
        }
        assert resident.tensor("lm_head_partial_values").shape == (4_608,)
        assert resident.tensor("lm_head_partial_indices").shape == (4_608,)
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries(trace))
        resident.set_encoder_state(
            np.zeros((1, 40, 416), dtype=np.float16),
            np.ones((1, 40), dtype=np.int32),
        )
        resident.precompute_cross_kv()
        trace.clear()
        malloc_count = len(runtime.malloc_calls)
        resident.set_decode_state(token_id=1, position=0)
        with resident.no_allocation_region("wave8-top1-token-step"):
            resident.token_step()
        assert len(runtime.malloc_calls) == malloc_count
        names = [name for name, _ in trace]
        assert names.count(
            "hipengine_moonshine_f16_lm_head_projection_wave8_top1"
        ) == 1
        assert "hipengine_moonshine_f16_lm_head_projection_wave8" not in names
        assert "hipengine_moonshine_argmax_fp16" not in names
        resident.reset_generation(clear_cross_cache=False)
        assert resident.cross_cache_valid is True
        trace.clear()
        captures = resident.capture_token_graphs()
        assert len(captures) == 4
        capture_names = [name for name, _ in trace]
        assert capture_names.count(
            "hipengine_moonshine_f16_lm_head_projection_wave8_top1"
        ) == 4
        assert "hipengine_moonshine_f16_lm_head_projection_wave8" not in capture_names
        assert "hipengine_moonshine_argmax_fp16" not in capture_names
    finally:
        resident.close()

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


def test_selective_w8a16_routes_expected_fixed_address_kernels_and_reports_bytes() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(
            runtime, w8a16_families=("lm_head", "mlp", "attention")
        ),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    try:
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries(trace))
        resident.set_encoder_state(
            np.zeros((1, 40, 416), dtype=np.float16),
            np.ones((1, 40), dtype=np.int32),
        )
        resident.precompute_cross_kv()
        assert [name for name, _ in trace] == [
            "hipengine_moonshine_w8a16_cross_kv_pair_head_major"
        ] * 8
        trace.clear()
        malloc_count = len(runtime.malloc_calls)
        resident.set_decode_state(token_id=1, position=0)
        with resident.no_allocation_region("w8a16-token-step"):
            resident.token_step()
        assert len(runtime.malloc_calls) == malloc_count
        names = [name for name, _ in trace]
        assert names.count("hipengine_moonshine_w8a16_qkv_triple") == 8
        assert names.count("hipengine_moonshine_w8a16_projection") == 24
        assert names.count("hipengine_moonshine_w8a16_mlp_fc1_gated_silu") == 8
        assert names.count("hipengine_moonshine_w8a16_mlp_fc2_residual") == 8
        assert names.count("hipengine_moonshine_w8a16_lm_head_wave8") == 1
        assert "hipengine_moonshine_f16_lm_head_projection_wave8" not in names
        assert "hipengine_dense_gemv_out_fp16" not in names
        contract = resident.quantization_contract()
        assert contract["families"] == [
            "lm_head",
            "mlp_fc1",
            "mlp_fc2",
            "self_attention",
            "cross_attention",
        ]
        assert contract["tensor_count"] == 81
        assert contract["layout"] == (
            "row_major_int8_per_output_channel_symmetric_f32_scale"
        )
        assert contract["active_read_byte_reduction"] > 0
        assert contract["fp16_fallback_resident"] is True
        allocation = resident.allocation_contract()
        assert allocation["w8a16_sidecar_nbytes"] == contract["packed_nbytes"]
        assert allocation["resident_nbytes"] > allocation["fp16_weight_nbytes"]
    finally:
        resident.close()

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


def test_token_graphs_capture_four_buckets_replay_sequential_state_and_close() -> None:
    assert _moonshine_token_graph_bucket(0) == ("position_0", 0, 0)
    assert _moonshine_token_graph_bucket(1) == ("position_1", 1, 1)
    assert _moonshine_token_graph_bucket(2) == ("positions_2_3", 2, 3)
    assert _moonshine_token_graph_bucket(3) == ("positions_2_3", 2, 3)
    assert _moonshine_token_graph_bucket(4) == ("positions_4_193", 4, 193)
    assert _moonshine_token_graph_bucket(193) == ("positions_4_193", 4, 193)
    with pytest.raises(ValueError, match="capacity"):
        _moonshine_token_graph_bucket(194)

    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    try:
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries(trace))
        resident.set_encoder_state(
            np.zeros((1, 40, 416), dtype=np.float16),
            np.ones((1, 40), dtype=np.int32),
        )
        resident.precompute_cross_kv()
        trace.clear()
        malloc_count = len(runtime.malloc_calls)

        resident.set_decode_state(token_id=1, position=0)
        with pytest.raises(RuntimeError, match="not captured"):
            resident.graph_token_step()
        resident.reset_generation(clear_cross_cache=False)
        captures = resident.capture_token_graphs()
        assert [capture.bucket for capture in captures] == [
            "position_0",
            "position_1",
            "positions_2_3",
            "positions_4_193",
        ]
        assert [capture.capture_position for capture in captures] == [0, 1, 2, 4]
        assert [capture.position_range for capture in captures] == [
            (0, 0),
            (1, 1),
            (2, 3),
            (4, 193),
        ]
        assert len(runtime.capture_begins) == len(runtime.capture_ends) == 4
        assert runtime.instantiated_graphs == [capture.graph for capture in captures]
        assert resident.self_cache_length == 0
        assert resident.decode_position is None
        assert len(trace) == 4 * 100
        for offset, expected_self_symbol in zip(
            range(0, 4 * 100, 100),
            (
                "hipengine_moonshine_self_attention_fp16",
                "hipengine_moonshine_self_attention_parallel_fp16",
                "hipengine_moonshine_self_attention_parallel_fp16",
                "hipengine_moonshine_self_attention_parallel_fp16",
            ),
            strict=True,
        ):
            graph_names = [name for name, _ in trace[offset : offset + 100]]
            assert graph_names.count(expected_self_symbol) == 8
        assert len(runtime.malloc_calls) == malloc_count
        assert all(capture.capture_wall_ms >= 0.0 for capture in captures)
        assert all(capture.instantiate_wall_ms >= 0.0 for capture in captures)

        resident.set_decode_state(token_id=1, position=0)
        with resident.no_allocation_region("graph-position-0"):
            resident.graph_token_step()
        assert runtime.launched_graphs[-1] == (captures[0].graph_exec, resident.stream)
        assert resident.self_cache_length == 1
        assert resident.decode_position is None

        resident.set_decode_state(token_id=1, position=1)
        with resident.no_allocation_region("graph-position-1"):
            resident.graph_token_step()
        assert runtime.launched_graphs[-1] == (captures[1].graph_exec, resident.stream)

        resident.set_decode_state(token_id=1, position=2)
        resident.graph_token_step()
        assert runtime.launched_graphs[-1] == (captures[2].graph_exec, resident.stream)
        resident.set_decode_state(token_id=1, position=3)
        resident.graph_token_step()
        assert runtime.launched_graphs[-1] == (captures[2].graph_exec, resident.stream)
        resident.set_decode_state(token_id=1, position=4)
        resident.graph_token_step()
        assert runtime.launched_graphs[-1] == (captures[3].graph_exec, resident.stream)

        contract = resident.token_graph_contract()
        assert contract["captured"] is True
        assert contract["graph_count"] == 4
        assert contract["replay_count"] == 5
        assert contract["buckets"] == [
            "position_0",
            "position_1",
            "positions_2_3",
            "positions_4_193",
        ]
        assert resident.capture_token_graphs() == captures
        assert len(runtime.capture_begins) == 4
        resident.reset_generation(clear_cross_cache=False)
        resident.set_decode_state(token_id=1, position=0)
        resident.graph_token_step()
        assert runtime.launched_graphs[-1] == (captures[0].graph_exec, resident.stream)
    finally:
        resident.close()

    assert runtime.destroyed_graph_execs == [capture.graph_exec for capture in reversed(captures)]
    assert runtime.destroyed_graphs == [capture.graph for capture in reversed(captures)]


def test_token_graph_partial_capture_failure_destroys_every_created_handle() -> None:
    runtime = FakeRuntime()
    resident = MoonshineResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries([]))
        resident.set_encoder_state(
            np.zeros((1, 40, 416), dtype=np.float16),
            np.ones((1, 40), dtype=np.int32),
        )
        resident.precompute_cross_kv()
        runtime.fail_graph_instantiate_at = 2
        with pytest.raises(RuntimeError, match="injected graph instantiate failure"):
            resident.capture_token_graphs()
        assert resident.token_graph_contract()["captured"] is False
        assert runtime.destroyed_graph_execs == [0x8000]
        assert runtime.destroyed_graphs == [0x7001, 0x7000]
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
