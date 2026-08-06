"""C2: resident eager CUDA decoder composition (MoonshineCudaResidentRuntime).

CPU-side tests drive the fixed-address token DAG through fake libraries and a
fake CUDA runtime, verifying dispatch order and measured schedules without a
GPU.  GPU-gated tests run the full 194-position autonomous generation on the
real sm_120a backend against the model-derived golden fixtures and assert the
exact token stream plus FP16-tolerance final_hidden at the retained positions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.cuda import CudaRuntime
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    MemcpyKind,
    host_array_ptr,
    memory_stats,
    reset_memory_stats,
)
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    alias_device_allocation,
)
from hipengine.loading.moonshine import (
    MoonshineLoadedModel,
    moonshine_w8a16_source_names,
    normalize_moonshine_w8a16_families,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models.moonshine import (
    expected_moonshine_weight_shapes,
    parse_moonshine_model_spec,
)
from hipengine.runtime.moonshine_cuda import (
    MoonshineCudaDecoderLibraries,
    MoonshineCudaResidentRuntime,
    NoAllocationError,
    _self_attention_threads,
)

_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures",
)
_CHECKPOINT = os.environ.get(
    "HIPENGINE_MOONSHINE_CHECKPOINT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d/model.safetensors",
)
_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
)
_FIXTURES = ("audio-konichiwa-fp16", "synthetic-1s-seed1234-fp16")
_RETAINED = (0, 1, 8, 32, 64, 128, 193)
_FINAL_HIDDEN_MAX_ABS = 0.02  # FP16 compose vs FP32-accumulated fixture reference


def _cuda_sm120a_enabled() -> bool:
    import ctypes

    if os.environ.get("HIPENGINE_RUN_CUDA_SM120A") != "1":
        return False
    if os.environ.get("HIPENGINE_CUDA_ARCH") != "sm_120a":
        return False
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError:
        return False
    return True


def _fixtures_available() -> bool:
    return all(
        os.path.isfile(os.path.join(_FIXTURE_DIR, f"{name}.npz"))
        and os.path.isfile(os.path.join(_FIXTURE_DIR, f"{name}.json"))
        for name in _FIXTURES
    ) and os.path.isfile(_CHECKPOINT) and os.path.isdir(_SNAPSHOT)


class FakeCudaRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000
        self.malloc_calls: list[int] = []
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, int, int]] = []
        self.sets: list[tuple[int, int, int, int]] = []
        self.created_streams: list[int] = []
        self.destroyed_streams: list[int] = []
        self.synchronized_streams: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.malloc_calls.append(int(nbytes))
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))

    def memcpy(self, dst: int, src: int, nbytes: int, kind: MemcpyKind) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(kind)))

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(kind), int(stream)))

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


def fake_decoder_libraries(
    trace: list[tuple[str, tuple[object, ...]]],
) -> MoonshineCudaDecoderLibraries:
    return MoonshineCudaDecoderLibraries(
        glue=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16",
            "hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_fp16",
        ),
        layernorm=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16",
        ),
        projection=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_f16_projection",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias_gated_silu",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual",
            "hipengine_cuda_sm120a_moonshine_f16_projection_triple",
        ),
        attention=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_self_attention_fp16",
            "hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16",
            "hipengine_cuda_sm120a_moonshine_cross_attention_parallel_fp16",
        ),
        lm_head=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16",
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


def fake_loaded_model(runtime: FakeCudaRuntime) -> MoonshineLoadedModel:
    spec = parse_moonshine_model_spec(model_config(), generation_config())
    baseline = memory_stats()
    nbytes = spec.vocab_size * spec.hidden_size * DType.FP16.itemsize
    buffer = DeviceBuffer(runtime.malloc(nbytes), nbytes)
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
        Device("cuda", 0),
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
            buffer,
            Tensor.from_handle(buffer.ptr, shape, DType.FP16, Device("cuda", 0)),
            owns_buffer=False,
        )
    weights = DeviceWeightMap(allocations)
    index = WeightIndex(
        Path("/fake/moonshine"),
        model_config(),
        {source.name: source},
        (source.shard_path,),
    )
    return MoonshineLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
        w8a16=None,
    )


def setup_function() -> None:
    reset_memory_stats()


# ---------------------------------------------------------------- CPU-side


def test_cuda_resident_runtime_self_attention_thread_buckets() -> None:
    assert _self_attention_threads(0) == 32  # below-threshold bucket
    assert _self_attention_threads(1) == 32
    assert _self_attention_threads(7) == 32
    assert _self_attention_threads(8) == 256
    assert _self_attention_threads(193) == 256


def test_cuda_resident_runtime_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MoonshineCudaResidentRuntime(encoder_frames=40)
    with pytest.raises(ValueError, match="exactly one"):
        MoonshineCudaResidentRuntime(
            encoder_frames=40,
            model_path="/fake",
            loaded_model=object(),  # type: ignore[arg-type]
        )


def test_cuda_resident_runtime_rejects_nonpositive_encoder_frames() -> None:
    runtime = FakeCudaRuntime()
    loaded = fake_loaded_model(runtime)
    with pytest.raises(ValueError, match="must be positive"):
        MoonshineCudaResidentRuntime(
            loaded_model=loaded,
            encoder_frames=0,
            runtime=runtime,  # type: ignore[arg-type]
        )


def test_cuda_resident_runtime_rejects_bad_cache_layer() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(ValueError, match="layer"):
            resident.self_cache(-1)
        with pytest.raises(ValueError, match="layer"):
            resident.cross_cache(8)
    finally:
        resident.close()


def test_cuda_resident_runtime_rejects_nonmatching_cross_cache_shapes() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        good = np.zeros((1, 8, 40, 52), dtype=np.float16)
        with pytest.raises(ValueError, match="cross cache needs 8 layers"):
            resident.load_cross_cache([good] * 7, [good] * 8)
        with pytest.raises(ValueError, match="layer 0 cross cache shape"):
            resident.load_cross_cache(
                [np.zeros((1, 8, 39, 52), dtype=np.float16)] * 8,
                [good] * 8,
            )
        with pytest.raises(ValueError, match="mask size"):
            resident.load_cross_cache([good] * 8, [good] * 8, mask=np.ones(39, dtype=np.int32))
        # A valid load marks the cache valid and resets generation state.
        resident.load_cross_cache([good] * 8, [good] * 8)
        assert resident.cross_cache_valid is True
        assert resident.encoder_state_valid is True
        assert resident.self_cache_length == 0
        assert resident.decode_position is None
    finally:
        resident.close()


def test_cuda_resident_runtime_cache_views_use_layer_offsets() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        self_view0 = resident.self_cache(0)
        self_view1 = resident.self_cache(1)
        # Each layer owns both key and value slots.
        self_bytes = 2 * 8 * resident.spec.self_cache_capacity * 52 * DType.FP16.itemsize
        assert self_view1.key.ptr - self_view0.key.ptr == self_bytes
        assert self_view0.key.ptr != self_view0.value.ptr
        cross_view0 = resident.cross_cache(0)
        cross_view1 = resident.cross_cache(1)
        cross_bytes = 2 * 8 * 40 * 52 * DType.FP16.itemsize
        assert cross_view1.key.ptr - cross_view0.key.ptr == cross_bytes
        # Self and cross caches never alias.
        assert cross_view0.key.ptr != self_view0.key.ptr
    finally:
        resident.close()


def test_cuda_resident_runtime_no_allocation_region_detects_allocations() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(ValueError, match="non-empty"):
            with resident.no_allocation_region(""):
                pass
        with resident.no_allocation_region("clean"):
            pass
        with pytest.raises(NoAllocationError, match="bad"):
            with resident.no_allocation_region("bad"):
                from hipengine.core.memory import malloc as core_malloc

                core_malloc(16, runtime=runtime)  # type: ignore[arg-type]
    finally:
        resident.close()


def test_cuda_resident_runtime_token_step_dispatch_order_and_schedules() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    libraries = fake_decoder_libraries(trace)
    try:
        with pytest.raises(RuntimeError, match="prepared"):
            resident.set_decode_state(token_id=1, position=0)
        resident.prepare_decoder_kernels(libraries=libraries)
        good = np.zeros((1, 8, 40, 52), dtype=np.float16)
        with pytest.raises(RuntimeError, match="cross cache"):
            resident.set_decode_state(token_id=1, position=0)
        resident.load_cross_cache([good] * 8, [good] * 8)
        resident.set_decode_state(token_id=1, position=0)
        malloc_count = len(runtime.malloc_calls)
        with resident.no_allocation_region("token-step"):
            resident.token_step()
        assert len(runtime.malloc_calls) == malloc_count
        assert resident.self_cache_length == 1
        assert resident.decode_position is None
        names = [name for name, _ in trace]
        # Layer count is one per decoder layer.
        assert names[0] == "hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16"
        per_layer = [
            "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection_triple",
            "hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_fp16",
            "hipengine_cuda_sm120a_moonshine_self_attention_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection",
            "hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection",
            "hipengine_cuda_sm120a_moonshine_cross_attention_parallel_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection",
            "hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias_gated_silu",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual",
        ]
        expected = ["hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16"]
        for _ in range(8):
            expected.extend(per_layer)
        expected.extend(
            [
                "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
                "hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16",
            ]
        )
        assert names == expected
        # Position 0 uses the one-wave (t32) self-attention variant.
        assert names.count("hipengine_cuda_sm120a_moonshine_self_attention_fp16") == 8
        assert "hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16" not in names
        self_threads = [
            args[-2]
            for name, args in trace
            if name == "hipengine_cuda_sm120a_moonshine_self_attention_fp16"
        ]
        assert self_threads == [32] * 8
        # Fused LM head carries the measured rows-per-block default.
        lm_head_rows = [
            args[-2]
            for name, args in trace
            if name == "hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16"
        ]
        assert lm_head_rows == [8]

        # Position 7 (visible=8) switches to the parallel t256 variant.
        for intermediate in range(1, 7):
            resident.set_decode_state(token_id=1, position=intermediate)
            with resident.no_allocation_region(f"token-step-{intermediate}"):
                resident.token_step()
        trace.clear()
        resident.set_decode_state(token_id=1, position=7)
        with resident.no_allocation_region("token-step-position-7"):
            resident.token_step()
        names2 = [name for name, _ in trace]
        assert names2.count("hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16") == 8
        parallel_threads = [
            args[-2]
            for name, args in trace
            if name == "hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16"
        ]
        assert parallel_threads == [256] * 8
        assert "hipengine_cuda_sm120a_moonshine_self_attention_fp16" not in names2
    finally:
        resident.close()


def test_cuda_resident_runtime_reset_and_close_parity() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    baseline = resident._allocation_baseline
    try:
        resident.reset_generation(clear_cross_cache=True)
        assert resident.cross_cache_valid is False
        assert resident.self_cache_length == 0
        # reset_generation zeroes scratch but does not allocate.
        assert resident.allocation_contract()["workspace_nbytes"] > 0
        assert resident.allocation_contract()["resident_nbytes"] > 0
    finally:
        resident.close()
    assert resident.closed is True
    assert resident.teardown_returned_to_baseline is True
    assert memory_stats()["current_allocated_bytes"] <= baseline


# ---------------------------------------------------------------- GPU gate


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_generates_exact_token_stream_on_fixtures() -> None:
    from hipengine.core.cuda import get_cuda_runtime

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    for fixture_name in _FIXTURES:
        with open(os.path.join(_FIXTURE_DIR, f"{fixture_name}.json")) as handle:
            manifest = json.load(handle)
        frames = int(manifest["input"]["encoder_frames"])
        reference = [int(token) for token in manifest["decoder"]["token_ids"]]
        with np.load(os.path.join(_FIXTURE_DIR, f"{fixture_name}.npz")) as fixture:
            keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
            values = [fixture[f"cross.layer_{layer}.value"] for layer in range(8)]
            mask = fixture["encoder.attention_mask"]

        resident = MoonshineCudaResidentRuntime(
            model_path=_SNAPSHOT,
            encoder_frames=frames,
        )
        resident.prepare_decoder_kernels()
        resident.load_cross_cache(keys, values, mask=mask)
        mismatches: list[tuple[int, int, int]] = []
        token_id = reference[0]
        try:
            for position in range(194):
                resident.set_decode_state(token_id=token_id, position=position)
                with resident.no_allocation_region("token-step"):
                    resident.token_step()
                token_id = resident.read_token()
                expected = (
                    reference[position + 1]
                    if position + 1 < len(reference)
                    else reference[position]
                )
                if token_id != expected:
                    mismatches.append((position, token_id, expected))
        finally:
            resident.close()
        assert mismatches == [], (
            f"{fixture_name}: {len(mismatches)} token mismatches: {mismatches[:10]}"
        )


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_final_hidden_within_fp16_tolerance() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    for fixture_name in _FIXTURES:
        with open(os.path.join(_FIXTURE_DIR, f"{fixture_name}.json")) as handle:
            manifest = json.load(handle)
        frames = int(manifest["input"]["encoder_frames"])
        with np.load(os.path.join(_FIXTURE_DIR, f"{fixture_name}.npz")) as fixture:
            keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
            values = [fixture[f"cross.layer_{layer}.value"] for layer in range(8)]
            mask = fixture["encoder.attention_mask"]

        resident = MoonshineCudaResidentRuntime(
            model_path=_SNAPSHOT,
            encoder_frames=frames,
        )
        resident.prepare_decoder_kernels()
        resident.load_cross_cache(keys, values, mask=mask)
        captured: dict[int, np.ndarray] = {}
        with np.load(os.path.join(_FIXTURE_DIR, f"{fixture_name}.npz")) as fixture:
            reference_tokens = [int(t) for t in manifest["decoder"]["token_ids"]]
            token_id = reference_tokens[0]
            for position in range(194):

                def capture(name: str, tensor: Tensor, pos: int = position) -> None:
                    if name != "final_hidden" or pos not in _RETAINED:
                        return
                    cuda_runtime.stream_synchronize(resident.stream)
                    host = np.empty(tensor.shape, dtype=np.float16)
                    copy_device_to_host(
                        host_array_ptr(host),
                        DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize),
                        runtime=cuda_runtime,
                    )
                    captured[pos] = host.copy()

                resident.set_decode_state(token_id=token_id, position=position)
                with resident.no_allocation_region("token-step"):
                    resident.token_step(boundary_callback=capture)
                token_id = resident.read_token()
                assert token_id >= 0
            assert set(captured) == set(_RETAINED)
            for pos in _RETAINED:
                reference = fixture[f"decoder.position_{pos}.final_hidden"]
                actual = captured[pos]
                diff = np.abs(
                    actual.astype(np.float32) - reference.astype(np.float32)
                )
                assert float(diff.max()) <= _FINAL_HIDDEN_MAX_ABS, (
                    f"{fixture_name} pos {pos}: final_hidden max_abs={float(diff.max())}"
                )
        resident.close()
