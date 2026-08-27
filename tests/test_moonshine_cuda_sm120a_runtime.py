"""C2/C3/C4: resident eager CUDA decoder composition (MoonshineCudaResidentRuntime).

CPU-side tests drive the fixed-address token DAG through fake libraries and a
fake CUDA runtime, verifying dispatch order and measured schedules without a
GPU.  GPU-gated tests run the full 194-position autonomous generation on the
real sm_120a backend against the model-derived golden fixtures and assert the
exact token stream plus FP16-tolerance final_hidden at the retained positions.

The C3 token-graph tests exercise the two captured CUDA self-attention bucket
DAGs (t32 positions 0-6, parallel t256 positions 7-193): CPU-side capture
contract, bucket dispatch, graph launch, and close teardown through the fake
runtime, plus a GPU-gated gate that the replayed graphs reproduce the eager
token stream exactly across all 194 positions on both fixtures.

The C4 encoder-handoff tests cover set_encoder_state_from_device (D2D copy of
a caller-owned device encoder prefix into the fixed padded bucket) and
precompute_cross_kv (projecting all eight head-major cross K/V caches from the
resident encoder hidden): CPU-side validation/dispatch/contract tests through
the fake runtime, plus a GPU-gated gate that uploads each fixture's real
encoder hidden + int32 mask, hands it off D2D, precomputes the caches, and
reproduces the exact 194-position token stream.
"""

from __future__ import annotations

import ctypes
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
    copy_device_to_host,
    copy_host_to_device,
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
from hipengine.runtime.moonshine_encoder_cuda import (
    MoonshineCudaEncoderLibraries,
    MoonshineCudaEncoderRuntime,
    moonshine_encoder_frames_from_audio,
)

_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures",
)
_SIX_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures-six",
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
# Cross-cache gate for the six-file torch-free encoder E2E.  Each head-major
# cross cache has ~150K FP16 elements with magnitude up to ~10; a 2-3 ULP
# outlier (measured max 0.027 across the six files) is normal FP16 compose
# noise, so the max-abs gate is set with generous headroom over that while
# still catching genuine correctness errors (wrong weight/layout/order).
_CROSS_CACHE_MAX_ABS = 0.05


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


# The six audio fixtures used by the C4 torch-free encoder E2E gate.  ``None``
# entries have no documented borderline token positions (exact match expected);
# the ``audio-konichiwa`` entry lists a single position whose fixture decision
# is a sub-0.05 top-2 logit coin flip (see the GPU-gated test for the margin
# check).
_SIX_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)
_SIX_BORDERLINE = {"audio-konichiwa-fp16": {88}}
_BORDERLINE_MARGIN = 0.05  # fixture top-2 logit gap that counts as a coin flip


def _six_fixtures_available() -> bool:
    return all(
        os.path.isfile(os.path.join(_SIX_FIXTURE_DIR, f"{name}.npz"))
        and os.path.isfile(os.path.join(_SIX_FIXTURE_DIR, f"{name}.json"))
        for name in _SIX_FIXTURES
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
        self.capture_events: list[tuple[str, int, ...]] = []
        self.graphs: list[int] = []
        self.graph_instantiations: list[tuple[int, int]] = []
        self.graph_execs: list[int] = []
        self.graph_launches: list[tuple[int, int]] = []
        self.graph_exec_destroyed: list[int] = []
        self.graph_destroyed: list[int] = []

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

    def stream_begin_capture(self, stream: int, mode: int = 2) -> None:
        self.capture_events.append(("begin", int(stream), int(mode)))

    def stream_end_capture(self, stream: int) -> int:
        self.capture_events.append(("end", int(stream)))
        graph = 0x6000 + len(self.graphs)
        self.graphs.append(graph)
        return graph

    def graph_instantiate(self, graph: int, *, flags: int = 0) -> int:
        self.graph_instantiations.append((int(graph), int(flags)))
        graph_exec = 0x7000 + len(self.graph_execs)
        self.graph_execs.append(graph_exec)
        return graph_exec

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        self.graph_launches.append((int(graph_exec), int(stream)))

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self.graph_exec_destroyed.append(int(graph_exec))

    def graph_destroy(self, graph: int) -> None:
        self.graph_destroyed.append(int(graph))


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
            "hipengine_cuda_sm120a_moonshine_publish_result_fp16",
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
            "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major",
            "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major_batch",
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


def fake_encoder_libraries(
    trace: list[tuple[str, tuple[object, ...]]],
) -> MoonshineCudaEncoderLibraries:
    return MoonshineCudaEncoderLibraries(
        encoder=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_conv1_tanh_fp16",
            "hipengine_cuda_sm120a_moonshine_conv2_gelu_fp16",
            "hipengine_cuda_sm120a_moonshine_conv3_gelu_fp16",
            "hipengine_cuda_sm120a_moonshine_groupnorm_fp16",
            "hipengine_cuda_sm120a_moonshine_gelu_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_rope_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_attention_fp16",
        ),
        layernorm=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16",
        ),
        projection=FakeLibrary(
            trace,
            "hipengine_cuda_sm120a_moonshine_f16_projection",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual",
            "hipengine_cuda_sm120a_moonshine_f16_projection_triple",
        ),
    )


def _encoder_ready(
    runtime: FakeCudaRuntime,
    *,
    audio_samples: int = 16000,
    owns_weights: bool = False,
):
    encoder = MoonshineCudaEncoderRuntime(
        audio_samples=audio_samples,
        loaded_model=fake_loaded_model(runtime),
        runtime=runtime,  # type: ignore[arg-type]
        owns_weights=owns_weights,
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    encoder.prepare_encoder_kernels(libraries=fake_encoder_libraries(trace))
    return encoder, trace


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


def test_cuda_resident_runtime_read_result_tokens_limits_no_eos_to_generated_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    resident._device_owned_decode = True
    resident.self_cache_length = 3

    def fake_copy_device_to_host(host_ptr, _buffer, _nbytes=None, *, runtime=None):
        del runtime
        host = (ctypes.c_int64 * resident.spec.self_cache_capacity).from_address(
            host_ptr
        )
        host[:] = [11, 12, 13] + [999] * (resident.spec.self_cache_capacity - 3)

    monkeypatch.setattr(
        "hipengine.runtime.moonshine_cuda.copy_device_to_host",
        fake_copy_device_to_host,
    )
    try:
        assert resident.read_result_tokens() == [11, 12, 13]
    finally:
        resident.close()


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


def test_cuda_resident_runtime_load_cross_cache_optional_mask_means_all_valid() -> None:
    import ctypes

    import hipengine.runtime.moonshine_cuda as runtime_module

    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=4,
        runtime=runtime,  # type: ignore[arg-type]
    )
    mask_buffer = resident.workspace.allocation("encoder_attention_mask").buffer
    uploaded: list[np.ndarray] = []
    real_copy = runtime_module.copy_host_to_device

    def spy_copy(buffer, host_ptr, nbytes=None, *, runtime=None):
        count = buffer.nbytes if nbytes is None else nbytes
        if buffer.ptr == mask_buffer.ptr:
            array_type = ctypes.c_int32 * (count // ctypes.sizeof(ctypes.c_int32))
            uploaded.append(
                np.ctypeslib.as_array(array_type.from_address(int(host_ptr))).copy()
            )
        real_copy(buffer, host_ptr, nbytes, runtime=runtime)

    runtime_module.copy_host_to_device = spy_copy  # type: ignore[assignment]
    try:
        good = np.zeros((1, 8, 4, 52), dtype=np.float16)
        # Omitted mask must install an all-valid (all-ones) encoder mask, never
        # the zero-initialized all-masked buffer.
        resident.load_cross_cache([good] * 8, [good] * 8, mask=None)
        assert len(uploaded) == 1, uploaded
        assert uploaded[0].tolist() == [1, 1, 1, 1]
        assert resident.cross_cache_valid is True
        assert resident.encoder_state_valid is True
    finally:
        runtime_module.copy_host_to_device = real_copy
        resident.close()


def test_cuda_resident_runtime_set_encoder_state_from_device_validation() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(ValueError, match="positive integers"):
            resident.set_encoder_state_from_device(
                hidden_fp16_ptr=0, attention_mask_int32_ptr=0x2000, source_frames=40
            )
        with pytest.raises(ValueError, match="positive integers"):
            resident.set_encoder_state_from_device(
                hidden_fp16_ptr=0x1000, attention_mask_int32_ptr=0, source_frames=40
            )
        with pytest.raises(ValueError, match="source_frames"):
            resident.set_encoder_state_from_device(
                hidden_fp16_ptr=0x1000, attention_mask_int32_ptr=0x2000, source_frames=41
            )
        # Valid handoff copies D2D into the fixed padded bucket, marks encoder
        # valid and invalidates any prior cross cache.
        resident.set_encoder_state_from_device(
            hidden_fp16_ptr=0x1000,
            attention_mask_int32_ptr=0x2000,
            source_frames=40,
        )
        assert resident.encoder_state_valid is True
        assert resident.cross_cache_valid is False
        hidden_buf = resident.workspace.allocation("encoder_hidden").buffer
        mask_buf = resident.workspace.allocation("encoder_attention_mask").buffer
        d2d = [
            (dst, src, nbytes, kind)
            for dst, src, nbytes, kind, _stream in (
                entry for entry in runtime.copies if len(entry) == 5
            )
            if kind == int(MemcpyKind.DEVICE_TO_DEVICE)
        ]
        assert (
            hidden_buf.ptr,
            0x1000,
            40 * 416 * DType.FP16.itemsize,
            int(MemcpyKind.DEVICE_TO_DEVICE),
        ) in d2d
        assert (
            mask_buf.ptr,
            0x2000,
            40 * DType.INT32.itemsize,
            int(MemcpyKind.DEVICE_TO_DEVICE),
        ) in d2d
    finally:
        resident.close()


def test_cuda_resident_runtime_set_encoder_state_requires_reset() -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)
    try:
        resident.set_decode_state(token_id=1, position=0)
        with pytest.raises(RuntimeError, match="reset generation"):
            resident.set_encoder_state_from_device(
                hidden_fp16_ptr=0x1000, attention_mask_int32_ptr=0x2000, source_frames=40
            )
    finally:
        resident.close()


def test_cuda_resident_runtime_precompute_cross_kv_requires_prepared_and_state() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(RuntimeError, match="prepared"):
            resident.precompute_cross_kv()
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries([]))
        with pytest.raises(RuntimeError, match="encoder state"):
            resident.precompute_cross_kv()
    finally:
        resident.close()


def test_cuda_resident_runtime_precompute_cross_kv_dispatch_and_contract() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    try:
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries(trace))
        resident.set_encoder_state_from_device(
            hidden_fp16_ptr=0x1000, attention_mask_int32_ptr=0x2000, source_frames=40
        )
        resident.precompute_cross_kv()
        assert resident.cross_cache_valid is True
        assert resident.encoder_state_valid is True
        assert resident.self_cache_length == 0
        names = [name for name, _ in trace]
        assert names == [
            "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major"
        ] * 8
        per_layer = [
            args
            for name, args in trace
            if name == "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major"
        ]
        assert len(per_layer) == 8
        for layer, args in enumerate(per_layer):
            # args: input, wA, wB, outA, outB, rows, in_feat, outA, outB, head_dim,
            #       threads, stream
            cache = resident.cross_cache(layer)
            assert args[0] == resident.tensor("encoder_hidden").ptr
            assert args[3] == cache.key.ptr
            assert args[4] == cache.value.ptr
            assert args[5] == 40  # rows = encoder frames
            assert args[6] == 416
            assert args[9] == 52  # head_major head_dim
        # Decode is now ready through the normal eager path.
        resident.set_decode_state(token_id=1, position=0)
        with resident.no_allocation_region("token-step"):
            resident.token_step()
        assert resident.self_cache_length == 1
    finally:
        resident.close()


def test_cuda_resident_runtime_precompute_cross_kv_uses_exact_source_rows() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    try:
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries(trace))
        resident.set_encoder_state_from_device(
            hidden_fp16_ptr=0x1000,
            attention_mask_int32_ptr=0x2000,
            source_frames=24,
        )
        resident.precompute_cross_kv()

        names = [name for name, _args in trace]
        assert names == [
            "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major_batch"
        ] * 8
        for _name, args in trace:
            # input, wA, wB, outA, outB, batch, rows, output_frames,
            # in_features, outA, outB, head_dim, threads, stream
            assert args[5] == 1
            assert args[6] == 24
            assert args[7] == 40
        cross = resident.workspace.allocation("cross_kv").buffer
        assert (cross.ptr, 0, cross.nbytes, resident.stream) in runtime.sets
        assert resident.allocation_contract()["encoder_source_frames"] == 24
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


def test_cuda_resident_runtime_token_graph_bucket_boundaries() -> None:
    from hipengine.runtime.moonshine_cuda import _moonshine_cuda_token_graph_bucket

    assert _moonshine_cuda_token_graph_bucket(0) == ("positions_0_6", 0, 6)
    assert _moonshine_cuda_token_graph_bucket(6) == ("positions_0_6", 0, 6)
    assert _moonshine_cuda_token_graph_bucket(7) == ("positions_7_193", 7, 193)
    assert _moonshine_cuda_token_graph_bucket(128) == ("positions_7_193", 7, 193)
    assert _moonshine_cuda_token_graph_bucket(193) == ("positions_7_193", 7, 193)
    with pytest.raises(ValueError, match="integer"):
        _moonshine_cuda_token_graph_bucket("7")
    with pytest.raises(ValueError, match="capacity"):
        _moonshine_cuda_token_graph_bucket(0, capacity=0)
    with pytest.raises(ValueError, match="outside"):
        _moonshine_cuda_token_graph_bucket(194)
    with pytest.raises(ValueError, match="outside"):
        _moonshine_cuda_token_graph_bucket(-1)


def _graph_ready_resident(runtime: FakeCudaRuntime):
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    trace: list[tuple[str, tuple[object, ...]]] = []
    resident.prepare_decoder_kernels(libraries=fake_decoder_libraries(trace))
    good = np.zeros((1, 8, 40, 52), dtype=np.float16)
    resident.load_cross_cache([good] * 8, [good] * 8)
    return resident, trace


def test_cuda_resident_runtime_capture_token_graphs_contract() -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)
    try:
        assert resident.token_graph_contract()["captured"] is False
        with pytest.raises(RuntimeError, match="capture token graphs before setting"):
            resident.set_decode_state(token_id=1, position=0)
            resident.capture_token_graphs()
        assert resident.decode_position == 0
        # A fresh resident captures exactly the two measured CUDA buckets.
        resident2, _ = _graph_ready_resident(runtime)
        try:
            captures = resident2.capture_token_graphs()
            assert len(captures) == 2
            assert [c.bucket for c in captures] == [
                "positions_0_6",
                "positions_7_193",
            ]
            assert [c.capture_position for c in captures] == [0, 7]
            assert captures[0].position_range == (0, 6)
            assert captures[1].position_range == (7, 193)
            assert captures[0].accepts(0) and captures[0].accepts(6)
            assert not captures[0].accepts(7)
            assert captures[1].accepts(7) and captures[1].accepts(193)
            assert runtime.graphs == [0x6000, 0x6001]
            assert runtime.graph_execs == [0x7000, 0x7001]
            # Capture is idempotent: re-calling returns the same two graphs.
            again = resident2.capture_token_graphs()
            assert again == captures
            assert len(runtime.graphs) == 2
            contract = resident2.token_graph_contract()
            assert contract["captured"] is True
            assert contract["graph_count"] == 2
            assert contract["buckets"] == ["positions_0_6", "positions_7_193"]
            assert contract["capture_positions"] == [0, 7]
            assert contract["capture_wall_ms"] > 0
            assert contract["instantiate_wall_ms"] > 0
            assert contract["replay_count"] == 0
        finally:
            resident2.close()
    finally:
        resident.close()


def test_cuda_resident_runtime_capture_requires_cross_cache_and_prepare() -> None:
    runtime = FakeCudaRuntime()
    resident = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(RuntimeError, match="prepared"):
            resident.capture_token_graphs()
        resident.prepare_decoder_kernels(libraries=fake_decoder_libraries([]))
        with pytest.raises(RuntimeError, match="cross cache"):
            resident.capture_token_graphs()
    finally:
        resident.close()


def test_cuda_resident_runtime_graph_token_step_dispatches_bucket_and_advances() -> None:
    runtime = FakeCudaRuntime()
    # Without capture, graph_token_step refuses once decode state is set.
    resident, _ = _graph_ready_resident(runtime)
    try:
        resident.set_decode_state(token_id=1, position=0)
        with pytest.raises(RuntimeError, match="not captured"):
            resident.graph_token_step()
    finally:
        resident.close()

    resident, _ = _graph_ready_resident(runtime)
    try:
        resident.capture_token_graphs()

        # Graph replay selects the t32 bucket for route positions 0-6.
        for position in range(7):
            resident.set_decode_state(token_id=1, position=position)
            malloc_count = len(runtime.malloc_calls)
            with resident.no_allocation_region(f"graph-{position}"):
                resident.graph_token_step()
            assert len(runtime.malloc_calls) == malloc_count
            assert resident.self_cache_length == position + 1
            assert resident.decode_position is None
        assert [exec for exec, _ in runtime.graph_launches] == [0x7000] * 7

        # Route positions 7+ switch to the parallel t256 bucket graph.
        for position in range(7, 12):
            resident.set_decode_state(token_id=1, position=position)
            with resident.no_allocation_region(f"graph-{position}"):
                resident.graph_token_step()
        assert [exec for exec, _ in runtime.graph_launches] == [0x7000] * 7 + [
            0x7001
        ] * 5

        # The captured DAG is replayed as one launch, not per-kernel dispatch.
        contract = resident.token_graph_contract()
        assert contract["replay_count"] == 12

        with pytest.raises(ValueError, match="sequential"):
            resident.set_decode_state(token_id=1, position=0)
    finally:
        resident.close()


def test_cuda_resident_runtime_conditional_eos_graph_contract_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)
    resident.set_device_owned_decode(True)
    resident.capture_token_graphs()
    calls = []

    def fake_create(first, rest, eos, position, capacity, **kwargs):
        calls.append((first, rest, eos, position, capacity, kwargs["library"]))
        return (0x8000, 0x9000)

    monkeypatch.setattr(
        "hipengine.kernels.cuda_sm120a.fused.moonshine_glue.moonshine_create_eos_decode_graph",
        fake_create,
    )
    graph = resident.capture_eos_decode_graph()
    assert graph.graph == 0x8000
    assert graph.graph_exec == 0x9000
    assert graph.replay_count == 0
    assert calls[0][:5] == (
        0x6000,
        0x6001,
        resident.tensor("result_eos").ptr,
        resident.tensor("position").ptr,
        194,
    )
    assert calls[0][5] is resident.decoder_libraries.glue
    assert resident.capture_eos_decode_graph() is graph
    contract = resident.token_graph_contract()
    assert contract["eos_conditional_captured"] is True
    assert contract["eos_conditional_replay_count"] == 0

    resident.close()
    assert runtime.graph_exec_destroyed[0] == 0x9000
    assert runtime.graph_destroyed[0] == 0x8000
    assert sorted(runtime.graph_exec_destroyed[1:]) == [0x7000, 0x7001]
    assert sorted(runtime.graph_destroyed[1:]) == [0x6000, 0x6001]


def test_cuda_resident_runtime_conditional_eos_decode_updates_exact_host_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)
    resident.set_device_owned_decode(True)
    resident.capture_token_graphs()
    monkeypatch.setattr(
        "hipengine.kernels.cuda_sm120a.fused.moonshine_glue.moonshine_create_eos_decode_graph",
        lambda *_args, **_kwargs: (0x8000, 0x9000),
    )
    resident.capture_eos_decode_graph()
    resident.set_decode_seed(token_id=1)
    monkeypatch.setattr(resident, "_read_device_position", lambda: 23)
    monkeypatch.setattr(resident, "read_result_tokens", lambda: [7] * 22 + [2])

    tokens = resident.graph_decode_to_eos()

    assert tokens == [7] * 22 + [2]
    assert resident.self_cache_length == 23
    assert runtime.graph_launches[-1] == (0x9000, resident.stream)
    contract = resident.token_graph_contract()
    assert contract["eos_conditional_replay_count"] == 1
    assert contract["replay_count"] == 23
    resident.close()


def test_cuda_resident_runtime_conditional_eos_graph_requires_device_owned() -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)
    resident.capture_token_graphs()
    try:
        with pytest.raises(RuntimeError, match="device-owned"):
            resident.capture_eos_decode_graph()
    finally:
        resident.close()


def test_cuda_resident_runtime_graph_close_destroys_graphs_and_clears_registry() -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)
    resident.capture_token_graphs()
    assert len(runtime.graphs) == 2
    assert resident._token_graphs
    resident.close()
    assert sorted(runtime.graph_destroyed) == [0x6000, 0x6001]
    assert sorted(runtime.graph_exec_destroyed) == [0x7000, 0x7001]
    assert not resident._token_graphs
    # A second close is a no-op and does not double-destroy.
    resident.close()
    assert sorted(runtime.graph_destroyed) == [0x6000, 0x6001]


def test_cuda_resident_runtime_graph_close_after_partial_capture_failure() -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)

    def boom_graph_instantiate(graph, *, flags=0):
        raise RuntimeError("instantiate failed")

    runtime.graph_instantiate = boom_graph_instantiate  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="instantiate failed"):
            resident.capture_token_graphs()
        assert not resident._token_graphs
    finally:
        resident.close()
    # The first graph that failed to instantiate was destroyed, but never exec.
    assert runtime.graph_destroyed == [0x6000]
    assert runtime.graph_exec_destroyed == []


def test_cuda_resident_runtime_graph_close_destroys_graph_on_enqueue_failure() -> None:
    runtime = FakeCudaRuntime()
    resident, _ = _graph_ready_resident(runtime)

    def boom_enqueue(*, route_position: int, stream: int, **kwargs):
        raise RuntimeError("enqueue failed")

    resident._enqueue_token_step = boom_enqueue  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="enqueue failed"):
            resident.capture_token_graphs()
        assert not resident._token_graphs
    finally:
        resident.close()
    # An enqueue-time failure after stream_begin_capture must destroy the graph
    # handle returned by the unwinding stream_end_capture (no leaked 0x6000).
    assert runtime.capture_events == [("begin", 0x5000, 2), ("end", 0x5000)]
    assert runtime.graph_destroyed == [0x6000]
    assert runtime.graph_exec_destroyed == []



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


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_graph_replay_generates_exact_token_stream_on_fixtures() -> None:
    """C3: the two captured token graphs replay bit-exact across all 194 positions.

    CUDA graph capture fixes the launch geometry at the representative route
    position (0 for the t32 bucket, 7 for the parallel t256 bucket) but every
    kernel reads the current position from the fixed position tensor at launch,
    so replaying at each later position must reproduce the eager token stream
    exactly on both retained fixtures.
    """

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
        captures = resident.capture_token_graphs()
        assert len(captures) == 2
        mismatches: list[tuple[int, int, int]] = []
        token_id = reference[0]
        try:
            for position in range(194):
                resident.set_decode_state(token_id=token_id, position=position)
                with resident.no_allocation_region("graph-token-step"):
                    resident.graph_token_step()
                token_id = resident.read_token()
                expected = (
                    reference[position + 1]
                    if position + 1 < len(reference)
                    else reference[position]
                )
                if token_id != expected:
                    mismatches.append((position, token_id, expected))
            contract = resident.token_graph_contract()
            assert contract["graph_count"] == 2
            assert contract["replay_count"] == 194
        finally:
            resident.close()
        assert mismatches == [], (
            f"{fixture_name}: {len(mismatches)} graph token mismatches: {mismatches[:10]}"
        )


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_encoder_handoff_precompute_and_decode_on_fixtures() -> None:
    """C4: D2D encoder handoff + precomputed cross K/V reproduce the token stream.

    Uploads the fixture's real encoder hidden state and int32 mask to device
    buffers, hands them to the resident decoder with
    set_encoder_state_from_device (D2D into the fixed padded bucket), projects
    all eight head-major cross K/V caches with precompute_cross_kv, then runs
    the eager token DAG across all 194 positions.  The precomputed caches and
    the token stream must match the model-derived fixture gates.
    """

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    for fixture_name in _FIXTURES:
        with open(os.path.join(_FIXTURE_DIR, f"{fixture_name}.json")) as handle:
            manifest = json.load(handle)
        frames = int(manifest["input"]["encoder_frames"])
        reference = [int(token) for token in manifest["decoder"]["token_ids"]]
        with np.load(os.path.join(_FIXTURE_DIR, f"{fixture_name}.npz")) as fixture:
            ref_keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
            ref_values = [fixture[f"cross.layer_{layer}.value"] for layer in range(8)]
            enc_hidden = fixture["encoder.output"]
            enc_mask = fixture["encoder.attention_mask"]

        hidden_nbytes = frames * 416 * np.dtype(np.float16).itemsize
        mask_nbytes = frames * np.dtype(np.int32).itemsize
        hidden_ptr = cuda_runtime.malloc(hidden_nbytes)
        mask_ptr = cuda_runtime.malloc(mask_nbytes)
        try:
            copy_host_to_device(
                DeviceBuffer(hidden_ptr, hidden_nbytes),
                host_array_ptr(np.ascontiguousarray(enc_hidden, dtype=np.float16)),
                runtime=cuda_runtime,
            )
            copy_host_to_device(
                DeviceBuffer(mask_ptr, mask_nbytes),
                host_array_ptr(np.ascontiguousarray(enc_mask, dtype=np.int32)),
                runtime=cuda_runtime,
            )
            resident = MoonshineCudaResidentRuntime(
                model_path=_SNAPSHOT,
                encoder_frames=frames,
            )
            resident.prepare_decoder_kernels()
            resident.set_encoder_state_from_device(
                hidden_fp16_ptr=hidden_ptr,
                attention_mask_int32_ptr=mask_ptr,
                source_frames=frames,
            )
            assert resident.cross_cache_valid is False
            resident.precompute_cross_kv()
            assert resident.cross_cache_valid is True
            # The precomputed head-major caches must match the fixture gates.
            for layer in range(8):
                cache = resident.cross_cache(layer)
                for side, ref_array in (
                    (cache.key, ref_keys[layer]),
                    (cache.value, ref_values[layer]),
                ):
                    cuda_runtime.stream_synchronize(resident.stream)
                    host = np.empty(ref_array.shape, dtype=np.float16)
                    copy_device_to_host(
                        host_array_ptr(host),
                        DeviceBuffer(
                            side.ptr, side.numel * side.dtype.itemsize
                        ),
                        runtime=cuda_runtime,
                    )
                    diff = np.abs(
                        host.astype(np.float32) - ref_array.astype(np.float32)
                    )
                    assert float(diff.max()) <= _FINAL_HIDDEN_MAX_ABS, (
                        f"{fixture_name} layer {layer} precomputed cross cache "
                        f"max_abs={float(diff.max())}"
                    )
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
                f"{fixture_name}: {len(mismatches)} handoff token mismatches: "
                f"{mismatches[:10]}"
            )
        finally:
            cuda_runtime.free(hidden_ptr)
            cuda_runtime.free(mask_ptr)


# ---------------------------------------------------------------- C4 encoder


def _device_tensor_to_host(cuda_runtime, tensor, dtype=np.float16) -> np.ndarray:
    host = np.empty(tensor.shape, dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize),
        runtime=cuda_runtime,
    )
    return host


def test_cuda_encoder_runtime_frames_from_audio() -> None:
    assert moonshine_encoder_frames_from_audio(16_000) == 40
    assert moonshine_encoder_frames_from_audio(16_896) == 42
    assert moonshine_encoder_frames_from_audio(80_000) == 207
    assert moonshine_encoder_frames_from_audio(480_000) == 1248
    with pytest.raises(ValueError, match="audio_samples"):
        moonshine_encoder_frames_from_audio(0)
    with pytest.raises(ValueError, match="conv1 kernel"):
        moonshine_encoder_frames_from_audio(100)
    with pytest.raises(ValueError, match="conv2 kernel"):
        moonshine_encoder_frames_from_audio(127)  # L1 == 1 is too short for conv2


def test_cuda_encoder_runtime_requires_exactly_one_source() -> None:
    runtime = FakeCudaRuntime()
    with pytest.raises(ValueError, match="exactly one"):
        MoonshineCudaEncoderRuntime(
            audio_samples=16000,
            runtime=runtime,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly one"):
        MoonshineCudaEncoderRuntime(
            audio_samples=16000,
            model_path="x",
            loaded_model=fake_loaded_model(runtime),
            runtime=runtime,  # type: ignore[arg-type]
        )


def test_cuda_encoder_runtime_rejects_nonpositive_audio_samples() -> None:
    runtime = FakeCudaRuntime()
    with pytest.raises(ValueError, match="audio_samples"):
        MoonshineCudaEncoderRuntime(
            audio_samples=0,
            loaded_model=fake_loaded_model(runtime),
            runtime=runtime,  # type: ignore[arg-type]
        )


def test_cuda_encoder_runtime_encode_validation() -> None:
    runtime = FakeCudaRuntime()
    encoder, _ = _encoder_ready(runtime)
    try:
        with pytest.raises(ValueError, match="too short for the encoder bucket"):
            encoder.encode(np.zeros((1, 100), dtype=np.float32))
        with pytest.raises(ValueError, match=r"must be in 1\.\.16000"):
            encoder.encode(np.zeros((1, 20000), dtype=np.float32))
        with pytest.raises(ValueError, match="float32 or float16"):
            encoder.encode(np.zeros((1, 16000), dtype=np.float64))
        values = np.ones((1, 16000), dtype=np.float32)
        values[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            encoder.encode(values)
        with pytest.raises(ValueError, match=r"\(1, 16000\)"):
            encoder.encode(
                np.ones((1, 16000), dtype=np.float32),
                np.ones((1, 100), dtype=np.int64),
            )
        bad_mask = np.ones((1, 16000), dtype=np.int64)
        bad_mask[0, 5] = 2
        with pytest.raises(ValueError, match="binary"):
            encoder.encode(np.ones((1, 16000), dtype=np.float32), bad_mask)
    finally:
        encoder.close()


def test_cuda_encoder_runtime_bucket_padding_contract() -> None:
    """C4-R2: a bucket-capacity arena reuses fixed sizes and masks the tail.

    The 40-frame arena (16,000-sample capacity) accepts a 24-frame file (9,727
    samples).  The audio is copied at its exact (unpadded) length — Moonshine
    GroupNorm(1, C) normalizes across positions, so zero-padded audio would
    corrupt the encoder statistics — while the downsampled encoder mask is
    zero-padded to the bucket so the decoder sees only the real 24 frames as
    valid.  ``_enqueue_encode`` processes the exact sample/frame counts.
    """

    import hipengine.runtime.moonshine_encoder_cuda as encoder_module

    runtime = FakeCudaRuntime()
    encoder, trace = _encoder_ready(runtime, audio_samples=16000)
    try:
        real_samples = 9727  # 24 frames
        audio = np.ones((1, real_samples), dtype=np.float32)
        encoder.upload_input(audio)
        assert encoder.real_frames == 24
        # The audio H2D is the exact (unpadded) length: 9727 * 2 bytes.
        h2d = [
            (dst, nbytes)
            for dst, src, nbytes, kind in runtime.copies
            if kind == int(MemcpyKind.HOST_TO_DEVICE)
        ]
        audio_buf = encoder.workspace.allocation("audio").buffer
        mask_buf = encoder.workspace.allocation("encoder_attention_mask").buffer
        assert (audio_buf.ptr, real_samples * DType.FP16.itemsize) in h2d
        # The encoder mask is bucket-sized: real 24 frames valid, tail masked.
        assert (mask_buf.ptr, 40 * DType.INT32.itemsize) in h2d
        # The encoder mask is bucket-sized: real 24 frames valid, tail masked.
        assert (mask_buf.ptr, 40 * DType.INT32.itemsize) in h2d
        encoder.run_encode()
        conv1 = trace[0][1]
        assert conv1[3] == real_samples  # conv1 reads the exact sample count
        attention = trace[10][1]
        assert (attention[5], attention[6], attention[7]) == (8, 52, 24)
        # Bucket selection picks the smallest bucket that fits.
        from hipengine.runtime.moonshine_encoder_cuda import (
            moonshine_encoder_bucket_for_frames,
        )

        assert moonshine_encoder_bucket_for_frames(24) == 40
        assert moonshine_encoder_bucket_for_frames(40) == 40
        assert moonshine_encoder_bucket_for_frames(42) == 207
        assert moonshine_encoder_bucket_for_frames(105) == 207
        assert moonshine_encoder_bucket_for_frames(207) == 207
        assert moonshine_encoder_bucket_for_frames(1248) == 1248
        with pytest.raises(ValueError, match="exceeds"):
            moonshine_encoder_bucket_for_frames(1249)
        with pytest.raises(ValueError, match="positive"):
            moonshine_encoder_bucket_for_frames(0)
    finally:
        encoder.close()



def test_cuda_encoder_runtime_upload_run_encode_split() -> None:
    """C4/C5: ``upload_input`` + ``run_encode`` is equivalent to ``encode`` but
    lets a timing harness exclude the (KB-scale) initial H2D."""

    runtime = FakeCudaRuntime()
    encoder, trace = _encoder_ready(runtime)
    try:
        with pytest.raises(RuntimeError, match="input is not uploaded"):
            encoder.run_encode()
        encoder.upload_input(np.ones((1, 16000), dtype=np.float32))
        assert encoder._input_uploaded is True
        encoder.run_encode()
        # The split dispatches the same full 101-launch DAG as ``encode``.
        assert len(trace) == 101
        # Upload validation errors are preserved on the split path.
        with pytest.raises(ValueError, match="too short for the encoder bucket"):
            encoder.upload_input(np.zeros((1, 100), dtype=np.float32))
    finally:
        encoder.close()


def test_cuda_encoder_runtime_encode_dispatch_and_contract() -> None:
    runtime = FakeCudaRuntime()
    encoder, trace = _encoder_ready(runtime, audio_samples=16000)
    try:
        encoder.encode(
            np.ones((1, 16000), dtype=np.float32),
            np.ones((1, 16000), dtype=np.int64),
        )
        names = [name for name, _ in trace]
        layer_sequence = [
            "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection_triple",
            "hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_rope_fp16",
            "hipengine_cuda_sm120a_moonshine_encoder_attention_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection",
            "hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias",
            "hipengine_cuda_sm120a_moonshine_gelu_fp16",
            "hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual",
        ]
        expected = [
            "hipengine_cuda_sm120a_moonshine_conv1_tanh_fp16",
            "hipengine_cuda_sm120a_moonshine_groupnorm_fp16",
            "hipengine_cuda_sm120a_moonshine_conv2_gelu_fp16",
            "hipengine_cuda_sm120a_moonshine_conv3_gelu_fp16",
            *(layer_sequence * 8),
            "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
        ]
        assert names == expected

        # conv1: (audio, conv1_w, conv1_out, audio_samples, L1=249, threads, stream)
        conv1 = trace[0][1]
        assert conv1[0] == encoder.tensor("audio").ptr
        assert conv1[3] == 16000
        assert conv1[4] == 249
        # conv3 writes the fused row-major hidden: (conv2_out, ..., hidden, L2=81, L3=40)
        conv3 = trace[3][1]
        assert conv3[3] == encoder.tensor("hidden").ptr
        assert conv3[4] == 81
        assert conv3[5] == 40

        # First layer's input LayerNorm reads hidden -> normalized (40 rows, 416).
        layer0_layernorm = trace[4][1]
        assert layer0_layernorm[0] == encoder.tensor("hidden").ptr
        assert layer0_layernorm[2] == encoder.tensor("normalized").ptr
        assert layer0_layernorm[3] == 40
        assert layer0_layernorm[4] == 416
        # QKV triple writes the row-major query/key/value.
        triple = trace[5][1]
        assert triple[4] == encoder.tensor("query_row").ptr
        assert triple[5] == encoder.tensor("key_row").ptr
        assert triple[6] == encoder.tensor("value_row").ptr
        assert triple[7] == 40
        # First transpose bridges query_row (row-major) -> query (head-major).
        transpose_q = trace[6][1]
        assert transpose_q[0] == encoder.tensor("query_row").ptr
        assert transpose_q[1] == encoder.tensor("query").ptr
        assert (transpose_q[2], transpose_q[3], transpose_q[4]) == (40, 8, 52)
        # Attention attends head-major Q/K/V with the encoder mask at 52**-0.5.
        attention = trace[10][1]
        assert attention[0] == encoder.tensor("query").ptr
        assert attention[3] == encoder.tensor("encoder_attention_mask").ptr
        assert attention[4] == encoder.tensor("attention").ptr
        assert (attention[5], attention[6], attention[7]) == (8, 52, 40)
        assert abs(float(attention[8]) - 52.0**-0.5) < 1e-9
        # Encoder MLP fc1 projects normalized -> mlp_fc1 at the 1664 intermediate.
        fc1 = trace[13][1]
        assert fc1[0] == encoder.tensor("normalized").ptr
        assert fc1[3] == encoder.tensor("mlp_fc1").ptr
        assert (fc1[4], fc1[5], fc1[6]) == (40, 416, 1664)
        # GELU is exact-erf over the intermediate plane.
        gelu = trace[14][1]
        assert gelu[1] == encoder.tensor("mlp_gelu").ptr
        assert gelu[2] == 40 * 1664
        # Final LayerNorm writes hidden -> encoder_output.
        final_layernorm = trace[100][1]
        assert final_layernorm[0] == encoder.tensor("hidden").ptr
        assert final_layernorm[2] == encoder.tensor("encoder_output").ptr
        assert final_layernorm[3] == 40

        # The downsampled mask is copied H2D once at 40 * 4 bytes regardless of
        # the 16000-sample audio length, and the audio copy matches.
        h2d = [
            (dst, nbytes)
            for dst, src, nbytes, kind in runtime.copies
            if kind == int(MemcpyKind.HOST_TO_DEVICE)
        ]
        mask_buf = encoder.workspace.allocation("encoder_attention_mask").buffer
        assert (mask_buf.ptr, 40 * DType.INT32.itemsize) in h2d
        audio_buf = encoder.workspace.allocation("audio").buffer
        assert (audio_buf.ptr, 16000 * DType.FP16.itemsize) in h2d

        # Encode must not allocate through hipEngine memory on the hot path.
        n_malloc = len(runtime.malloc_calls)
        encoder.encode(np.ones((1, 16000), dtype=np.float32))
        assert len(runtime.malloc_calls) == n_malloc
    finally:
        encoder.close()


def test_cuda_encoder_runtime_mask_downsample_contract() -> None:
    runtime = FakeCudaRuntime()
    encoder, _ = _encoder_ready(runtime, audio_samples=16000)
    try:
        # A strided binary mask downsamples to 40 frames (every 384th sample).
        pattern = np.ones((1, 16000), dtype=np.int64)
        pattern[0, ::384] = 1
        encoder.encode(np.zeros((1, 16000), dtype=np.float32), pattern)
        mask_buf = encoder.workspace.allocation("encoder_attention_mask").buffer
        h2d = [
            (dst, nbytes)
            for dst, src, nbytes, kind in runtime.copies
            if kind == int(MemcpyKind.HOST_TO_DEVICE)
        ]
        assert (mask_buf.ptr, 40 * DType.INT32.itemsize) in h2d
    finally:
        encoder.close()


def test_cuda_encoder_runtime_handoff_contract() -> None:
    runtime = FakeCudaRuntime()
    encoder, _ = _encoder_ready(runtime, audio_samples=16000)
    decoder = MoonshineCudaResidentRuntime(
        loaded_model=fake_loaded_model(runtime),
        encoder_frames=40,
        runtime=runtime,  # type: ignore[arg-type]
    )
    dtrace: list[tuple[str, tuple[object, ...]]] = []
    decoder.prepare_decoder_kernels(libraries=fake_decoder_libraries(dtrace))
    try:
        with pytest.raises(TypeError, match="MoonshineCudaResidentRuntime"):
            encoder.handoff_to(object())  # type: ignore[arg-type]
        encoder.encode(np.ones((1, 16000), dtype=np.float32))
        encoder.handoff_to(decoder)
        assert decoder.encoder_state_valid is True
        assert decoder.cross_cache_valid is True
        assert decoder.self_cache_length == 0
        # The handoff precomputes all eight head-major cross K/V caches.
        names = [name for name, _ in dtrace]
        assert names == [
            "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major"
        ] * 8
        # Decode is ready through the normal eager path.
        decoder.set_decode_state(token_id=1, position=0)
        with decoder.no_allocation_region("token-step"):
            decoder.token_step()
        assert decoder.self_cache_length == 1
    finally:
        decoder.close()
        encoder.close()


def test_cuda_encoder_runtime_close_parity() -> None:
    runtime = FakeCudaRuntime()
    encoder = MoonshineCudaEncoderRuntime(
        audio_samples=16000,
        loaded_model=fake_loaded_model(runtime),
        runtime=runtime,  # type: ignore[arg-type]
        owns_weights=True,
    )
    freed_before = len(runtime.freed)
    encoder.close()
    assert encoder.closed is True
    assert len(runtime.freed) > freed_before  # owned weights were freed
    assert encoder.teardown_returned_to_baseline is True
    freed_after = len(runtime.freed)
    encoder.close()  # second close is a no-op
    assert len(runtime.freed) == freed_after


def _oracle_top2_margins(
    cuda_runtime,
    loaded: MoonshineLoadedModel,
    frames: int,
    ref_keys: list[np.ndarray],
    ref_values: list[np.ndarray],
    ref_mask: np.ndarray,
    reference: list[int],
    lm_head_np: np.ndarray,
    positions: list[int],
) -> dict[int, float]:
    """Return the fixture-oracle top-2 logit gap at the requested positions.

    Runs the resident decoder on the fixture cross K/V (the byte-identical
    PyTorch encoder path) and computes, for each requested position, the gap
    between the top-1 and top-2 logits from the fixture-oracle's own final
    hidden.  A small gap means the fixture stream decision was a coin flip, so
    a distinct-but-within-tolerance encoder may legitimately land on the other
    token.
    """

    target = set(int(position) for position in positions)
    decoder = MoonshineCudaResidentRuntime(
        encoder_frames=frames, loaded_model=loaded, owns_weights=False
    )
    decoder.prepare_decoder_kernels()
    captured: dict[int, np.ndarray] = {}
    try:
        decoder.load_cross_cache(ref_keys, ref_values, mask=ref_mask)
        token_id = reference[0]
        margins: dict[int, float] = {}
        for position in range(194):

            def callback(name: str, tensor: Tensor) -> None:
                if name == "final_hidden" and position in target:
                    host = np.empty(tensor.shape, dtype=np.float16)
                    copy_device_to_host(
                        host_array_ptr(host),
                        DeviceBuffer(
                            tensor.ptr, tensor.numel * tensor.dtype.itemsize
                        ),
                        runtime=cuda_runtime,
                    )
                    captured[position] = host.reshape(-1)

            decoder.set_decode_state(token_id=token_id, position=position)
            decoder.token_step(boundary_callback=callback)
            token_id = int(decoder.read_token())
            if position in target and position in captured:
                logits = captured[position].astype(np.float32) @ lm_head_np.T
                order = np.argsort(logits)[::-1]
                margins[position] = float(logits[order[0]] - logits[order[1]])
        return margins
    finally:
        decoder.close()


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _six_fixtures_available(),
    reason="CUDA sm_120a gate or six audio fixtures are not available",
)
def test_cuda_encoder_runtime_e2e_encode_handoff_decode_on_six_fixtures() -> None:
    """C4: torch-free encoder -> handoff -> eager decode on the six audio files.

    Each file's raw audio is encoded by the standalone CUDA encoder, handed off
    to the resident decoder (D2D + precomputed cross K/V), and decoded across
    all 194 positions.  The encoder hidden and the precomputed cross caches must
    be within the FP16 compose tolerance of the fixture gates, the token stream
    must match the fixture stream exactly except at documented borderline
    positions, and every borderline mismatch must be a true coin flip: the
    fixture-oracle's own top-2 logit gap at that position must be below
    ``_BORDERLINE_MARGIN``.
    """

    import safetensors.numpy

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=cuda_runtime
    )
    with safetensors.numpy.safe_open(
        f"{_SNAPSHOT}/model.safetensors", framework="np"
    ) as handle:
        lm_head_np = handle.get_tensor(loaded.spec.embedding_weight_name).astype(
            np.float32
        )
    try:
        for fixture_name in _SIX_FIXTURES:
            with open(
                os.path.join(_SIX_FIXTURE_DIR, f"{fixture_name}.json")
            ) as handle:
                manifest = json.load(handle)
            with np.load(
                os.path.join(_SIX_FIXTURE_DIR, f"{fixture_name}.npz")
            ) as fixture:
                audio = fixture["input.values"]
                amask = fixture["input.attention_mask"]
                ref_mask = fixture["encoder.attention_mask"]
                reference = [int(token) for token in manifest["decoder"]["token_ids"]]
                frames = int(manifest["input"]["encoder_frames"])
                ref_hidden = fixture["encoder.output"]
                ref_keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
                ref_values = [
                    fixture[f"cross.layer_{layer}.value"] for layer in range(8)
                ]

            enc = MoonshineCudaEncoderRuntime(
                audio_samples=audio.shape[1],
                loaded_model=loaded,
                owns_weights=False,
            )
            enc.prepare_encoder_kernels()
            dec = MoonshineCudaResidentRuntime(
                encoder_frames=frames, loaded_model=loaded, owns_weights=False
            )
            dec.prepare_decoder_kernels()
            try:
                enc.encode(audio, amask)
                got_hidden = _device_tensor_to_host(cuda_runtime, enc.encoder_output())
                hidden_abs = np.abs(
                    got_hidden.astype(np.float32) - ref_hidden.astype(np.float32)
                ).max()
                assert float(hidden_abs) <= _FINAL_HIDDEN_MAX_ABS, (
                    f"{fixture_name} encoder hidden max_abs={float(hidden_abs)}"
                )
                got_mask = _device_tensor_to_host(
                    cuda_runtime,
                    enc.attention_mask(),
                    dtype=np.int32,
                )
                assert np.array_equal(got_mask, ref_mask), (
                    f"{fixture_name} encoder mask differs"
                )

                enc.handoff_to(dec)
                for layer in range(8):
                    cache = dec.cross_cache(layer)
                    for side, ref_array in (
                        (cache.key, ref_keys[layer]),
                        (cache.value, ref_values[layer]),
                    ):
                        host = _device_tensor_to_host(cuda_runtime, side)
                        diff = np.abs(
                            host.astype(np.float32) - ref_array.astype(np.float32)
                        ).max()
                        assert float(diff) <= _CROSS_CACHE_MAX_ABS, (
                            f"{fixture_name} layer {layer} precomputed cross cache "
                            f"max_abs={float(diff)}"
                        )

                mismatches: list[tuple[int, int, int]] = []
                token_id = reference[0]
                for position in range(194):
                    dec.set_decode_state(token_id=token_id, position=position)
                    dec.token_step()
                    token_id = int(dec.read_token())
                    expected = (
                        reference[position + 1]
                        if position + 1 < len(reference)
                        else reference[position]
                    )
                    if token_id != expected:
                        mismatches.append((position, token_id, expected))
                allowed = _SIX_BORDERLINE.get(fixture_name, set())
                unexpected = [
                    (position, got, expected)
                    for (position, got, expected) in mismatches
                    if position not in allowed
                ]
                assert unexpected == [], (
                    f"{fixture_name}: {len(unexpected)} unexpected token "
                    f"mismatches: {unexpected[:10]}"
                )
                # Every borderline mismatch must be a coin flip in the
                # fixture-oracle's own top-2 logit gap at that position.
                if mismatches:
                    margins = _oracle_top2_margins(
                        cuda_runtime,
                        loaded,
                        frames,
                        ref_keys,
                        ref_values,
                        ref_mask,
                        reference,
                        lm_head_np,
                        [position for position, _, _ in mismatches],
                    )
                    for position, _, _ in mismatches:
                        margin = margins[position]
                        assert margin < _BORDERLINE_MARGIN, (
                            f"{fixture_name} position {position} mismatch but the "
                            f"fixture-oracle top-2 logit gap is {margin:.4f} "
                            f"(not a coin flip)"
                        )
            finally:
                enc.close()
                dec.close()
    finally:
        loaded.weights.free(runtime=cuda_runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _six_fixtures_available(),
    reason="CUDA sm_120a gate or six audio fixtures are not available",
)
def test_cuda_encoder_runtime_bucket_arenas_e2e_on_six_fixtures() -> None:
    """C4-R1/C4-R2: fixed-bucket arenas reproduce the exact-shape token route.

    Each file is encoded through a reusable bucket-capacity arena (40/207/1,248
    frames) selected by ``moonshine_encoder_bucket_for_frames`` and handed off
    to a bucket-sized resident decoder.  Because the arena processes the exact
    (unpadded) audio length, the encoder hidden, downsampled mask, precomputed
    cross K/V, and token stream must match the exact-shape fixture gates exactly
    (the decoder masks the unused bucket tail via ``source_frames=real_frames``).
    """

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model
    from hipengine.runtime.moonshine_encoder_cuda import (
        MoonshineCudaEncoderRuntime,
        moonshine_encoder_bucket_audio_samples,
        moonshine_encoder_bucket_for_frames,
    )

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=cuda_runtime
    )
    try:
        for fixture_name in _SIX_FIXTURES:
            with open(
                os.path.join(_SIX_FIXTURE_DIR, f"{fixture_name}.json")
            ) as handle:
                manifest = json.load(handle)
            with np.load(
                os.path.join(_SIX_FIXTURE_DIR, f"{fixture_name}.npz")
            ) as fixture:
                audio = fixture["input.values"]
                amask = fixture["input.attention_mask"]
                ref_mask = fixture["encoder.attention_mask"]
                reference = [int(token) for token in manifest["decoder"]["token_ids"]]
                frames = int(manifest["input"]["encoder_frames"])
                ref_hidden = fixture["encoder.output"]
                ref_keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
                ref_values = [
                    fixture[f"cross.layer_{layer}.value"] for layer in range(8)
                ]

            bucket = moonshine_encoder_bucket_for_frames(frames)
            enc = MoonshineCudaEncoderRuntime(
                audio_samples=moonshine_encoder_bucket_audio_samples(bucket),
                loaded_model=loaded,
                owns_weights=False,
            )
            enc.prepare_encoder_kernels()
            dec = MoonshineCudaResidentRuntime(
                encoder_frames=bucket, loaded_model=loaded, owns_weights=False
            )
            dec.prepare_decoder_kernels()
            try:
                enc.encode(audio, amask)
                got_hidden = _device_tensor_to_host(cuda_runtime, enc.encoder_output())
                hidden_abs = np.abs(
                    got_hidden[0, :frames].astype(np.float32)
                    - ref_hidden[0].astype(np.float32)
                ).max()
                assert float(hidden_abs) <= _FINAL_HIDDEN_MAX_ABS, (
                    f"{fixture_name} bucket {bucket} hidden max_abs={float(hidden_abs)}"
                )
                got_mask = _device_tensor_to_host(
                    cuda_runtime, enc.attention_mask(), dtype=np.int32
                )
                assert np.array_equal(got_mask[0, :frames], ref_mask[0]), (
                    f"{fixture_name} bucket {bucket} mask differs"
                )
                assert bool((got_mask[0, frames:] == 0).all()), (
                    f"{fixture_name} bucket {bucket} mask tail is not masked"
                )

                enc.handoff_to(dec)
                for layer in range(8):
                    cache = dec.cross_cache(layer)
                    for side, ref_array in (
                        (cache.key, ref_keys[layer]),
                        (cache.value, ref_values[layer]),
                    ):
                        host = _device_tensor_to_host(cuda_runtime, side)
                        diff = np.abs(
                            host[..., :frames, :].astype(np.float32)
                            - ref_array[0].astype(np.float32)
                        ).max()
                        assert float(diff) <= _CROSS_CACHE_MAX_ABS, (
                            f"{fixture_name} bucket {bucket} layer {layer} "
                            f"cross cache max_abs={float(diff)}"
                        )

                mismatches: list[tuple[int, int, int]] = []
                token_id = reference[0]
                for position in range(194):
                    dec.set_decode_state(token_id=token_id, position=position)
                    dec.token_step()
                    token_id = int(dec.read_token())
                    expected = (
                        reference[position + 1]
                        if position + 1 < len(reference)
                        else reference[position]
                    )
                    if token_id != expected:
                        mismatches.append((position, token_id, expected))
                allowed = _SIX_BORDERLINE.get(fixture_name, set())
                unexpected = [
                    (position, got, expected)
                    for (position, got, expected) in mismatches
                    if position not in allowed
                ]
                assert unexpected == [], (
                    f"{fixture_name} bucket {bucket}: {len(unexpected)} unexpected "
                    f"token mismatches: {unexpected[:10]}"
                )
            finally:
                enc.close()
                dec.close()
    finally:
        loaded.weights.free(runtime=cuda_runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled(),
    reason="CUDA sm_120a gate is not enabled",
)
def test_cuda_encoder_runtime_bucket_207_1248_execution() -> None:
    """C4-R1: the 207- and 1,248-frame buckets execute end to end.

    Previously the encoder RoPE tables were sized to the decoder self-cache
    limit (194) and the ``sequence >= max_positions`` guard rejected any
    sequence at/above that, so the certified 207/1,248-frame buckets could not
    run at all.  Now the tables are sized to the admitted bucket and the guard
    accepts ``sequence == max_positions``; this test encodes full-length
    synthetic audio in both large buckets, checks finite hidden + all-valid
    mask, then decodes deterministically to the decoder's 194-position limit.
    """

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model
    from hipengine.runtime.moonshine_encoder_cuda import (
        MoonshineCudaEncoderRuntime,
        moonshine_encoder_bucket_audio_samples,
    )

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=cuda_runtime
    )
    try:
        for bucket in (207, 1248):
            samples = moonshine_encoder_bucket_audio_samples(bucket)
            rng = np.random.default_rng(bucket)
            audio = rng.normal(0.0, 0.02, size=(1, samples)).astype(np.float32)
            enc = MoonshineCudaEncoderRuntime(
                audio_samples=samples, loaded_model=loaded, owns_weights=False
            )
            enc.prepare_encoder_kernels()
            dec = MoonshineCudaResidentRuntime(
                encoder_frames=bucket, loaded_model=loaded, owns_weights=False
            )
            dec.prepare_decoder_kernels()
            try:
                enc.encode(audio)
                hidden = _device_tensor_to_host(cuda_runtime, enc.encoder_output())
                assert hidden.shape == (1, bucket, 416)
                assert bool(np.isfinite(hidden).all())
                mask = _device_tensor_to_host(
                    cuda_runtime, enc.attention_mask(), dtype=np.int32
                )
                assert np.array_equal(mask[0], np.ones(bucket, dtype=np.int32))
                enc.handoff_to(dec)
                token_id = 1
                stream_a: list[int] = []
                for position in range(194):
                    dec.set_decode_state(token_id=token_id, position=position)
                    dec.token_step()
                    token_id = int(dec.read_token())
                    stream_a.append(token_id)
                # Deterministic: a second decode is byte-identical.
                dec.reset_generation(clear_cross_cache=False)
                token_id = 1
                stream_b: list[int] = []
                for position in range(194):
                    dec.set_decode_state(token_id=token_id, position=position)
                    dec.token_step()
                    token_id = int(dec.read_token())
                    stream_b.append(token_id)
                assert stream_a == stream_b
                assert all(0 <= token_id for token_id in stream_a)
            finally:
                enc.close()
                dec.close()
    finally:
        loaded.weights.free(runtime=cuda_runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_device_owned_decode_exact_token_stream_on_fixtures() -> None:
    """C5/§7.3: device-owned decode reproduces the exact eager token stream.

    With device-owned decode the token/position buffers are seeded once and the
    captured graphs include the graph-tail position-advance state kernel; the
    replayed DAGs must produce a byte-identical token stream to the host-upload
    eager path across all 194 positions on both retained fixtures.
    """

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
        # Device-owned must be enabled before graph capture so the captured
        # DAGs include the graph-tail position-advance state kernel.
        resident.set_device_owned_decode(True)
        captures = resident.capture_token_graphs()
        assert len(captures) == 2
        mismatches: list[tuple[int, int, int]] = []
        try:
            resident.set_decode_seed(token_id=reference[0])
            token_id = reference[0]
            for position in range(194):
                resident.graph_token_step()
                token_id = resident.read_token()
                expected = (
                    reference[position + 1]
                    if position + 1 < len(reference)
                    else reference[position]
                )
                if token_id != expected:
                    mismatches.append((position, token_id, expected))
            contract = resident.token_graph_contract()
            assert contract["graph_count"] == 2
            assert contract["replay_count"] == 194
        finally:
            resident.close()
        assert mismatches == [], (
            f"{fixture_name}: {len(mismatches)} device-owned token mismatches: {mismatches[:10]}"
        )


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_device_owned_batched_decode_exact_stream_on_fixtures() -> None:
    """RR-8: batched device-owned readback reproduces the exact token stream.

    The graph tail publishes each token to the device result buffer and sets a
    sticky EOS flag, so the host launches several token steps back-to-back with
    no per-token D2H, reads only the tiny EOS status per batch, and recovers
    the full token stream from the result buffer.  The recovered stream must be
    identical to the per-step readback path (the existing exact device-owned
    gate) and to the retained reference (reference[0] is the seeded BOS).
    """

    from hipengine.core.cuda import get_cuda_runtime

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    batch = 8
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
        resident.set_device_owned_decode(True)
        captures = resident.capture_token_graphs()
        assert len(captures) == 2
        try:
            resident.set_decode_seed(token_id=reference[0])
            max_positions = resident.spec.self_cache_capacity
            steps = 0
            eos_seen = False
            for start in range(0, max_positions, batch):
                count = min(batch, max_positions - start)
                resident.graph_token_step_batch(count)
                steps += count
                if resident.read_eos_flag():
                    eos_seen = True
                    break
            result = resident.read_result_tokens()
            assert eos_seen, f"{fixture_name}: device EOS flag was never published"
            assert result[-1] == resident.spec.eos_token_ids[0], (
                f"{fixture_name}: result does not end in EOS"
            )
            expected = reference[1 : 1 + len(result)]
            assert result == expected, (
                f"{fixture_name}: batched result {result} != reference {expected}"
            )
            contract = resident.token_graph_contract()
            assert contract["replay_count"] == steps
        finally:
            resident.close()


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_cuda_resident_runtime_conditional_eos_decode_exact_stream_on_fixtures() -> None:
    """RT-2: one conditional graph replay stops on device at exact EOS."""

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
        resident.set_device_owned_decode(True)
        resident.capture_token_graphs()
        resident.capture_eos_decode_graph()
        try:
            resident.set_decode_seed(token_id=reference[0])
            result = resident.graph_decode_to_eos()
            expected_steps = reference.index(resident.spec.eos_token_ids[0], 1)
            assert result == reference[1 : expected_steps + 1]
            assert resident.self_cache_length == expected_steps
            contract = resident.token_graph_contract()
            assert contract["eos_conditional_replay_count"] == 1
            assert contract["replay_count"] == expected_steps
        finally:
            resident.close()


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _six_fixtures_available(),
    reason="CUDA sm_120a gate or six audio fixtures are not available",
)
def test_cuda_encoder_runtime_async_chain_e2e_on_six_fixtures() -> None:
    """C5/§7.3: the no-terminal-sync encoder->handoff->cross-KV chain is exact.

    The async chain enqueues the standalone encoder DAG onto the decoder stream
    (``synchronize=False``) and hands off D2D + precomputes cross K/V without
    intermediate host syncs; the decode loop's first sync covers the whole
    chain.  The produced encoder hidden and cross caches must match the fixture
    gates and the token stream must match the fixture stream exactly except at
    documented borderline positions (each a sub-margin top-2 coin flip).
    """

    import safetensors.numpy

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=cuda_runtime
    )
    with safetensors.numpy.safe_open(
        f"{_SNAPSHOT}/model.safetensors", framework="np"
    ) as handle:
        lm_head_np = handle.get_tensor(loaded.spec.embedding_weight_name).astype(
            np.float32
        )
    try:
        for fixture_name in _SIX_FIXTURES:
            with open(
                os.path.join(_SIX_FIXTURE_DIR, f"{fixture_name}.json")
            ) as handle:
                manifest = json.load(handle)
            with np.load(
                os.path.join(_SIX_FIXTURE_DIR, f"{fixture_name}.npz")
            ) as fixture:
                audio = fixture["input.values"]
                amask = fixture["input.attention_mask"]
                ref_mask = fixture["encoder.attention_mask"]
                reference = [int(token) for token in manifest["decoder"]["token_ids"]]
                frames = int(manifest["input"]["encoder_frames"])
                ref_keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
                ref_values = [
                    fixture[f"cross.layer_{layer}.value"] for layer in range(8)
                ]

            enc = MoonshineCudaEncoderRuntime(
                audio_samples=audio.shape[1],
                loaded_model=loaded,
                owns_weights=False,
            )
            enc.prepare_encoder_kernels()
            dec = MoonshineCudaResidentRuntime(
                encoder_frames=frames, loaded_model=loaded, owns_weights=False
            )
            dec.prepare_decoder_kernels()
            try:
                enc.upload_input(audio, amask)
                # Async chain: encoder DAG on the decoder stream, D2D handoff and
                # cross-K/V with no terminal host syncs; the first decode sync
                # (inside set_decode_state) covers the whole chain.  The direct
                # cache reads below sync the decoder stream first because the
                # D2H copy issues on the default stream.
                enc.run_encode(stream=dec.stream, synchronize=False)
                enc.handoff_to(dec, synchronize=False)
                cuda_runtime.stream_synchronize(dec.stream)
                for layer in range(8):
                    cache = dec.cross_cache(layer)
                    for side, ref_array in (
                        (cache.key, ref_keys[layer]),
                        (cache.value, ref_values[layer]),
                    ):
                        host = _device_tensor_to_host(cuda_runtime, side)
                        diff = np.abs(
                            host.astype(np.float32) - ref_array.astype(np.float32)
                        ).max()
                        assert float(diff) <= _CROSS_CACHE_MAX_ABS, (
                            f"{fixture_name} layer {layer} async cross cache "
                            f"max_abs={float(diff)}"
                        )

                mismatches: list[tuple[int, int, int]] = []
                token_id = reference[0]
                for position in range(194):
                    dec.set_decode_state(token_id=token_id, position=position)
                    dec.token_step()
                    token_id = int(dec.read_token())
                    expected = (
                        reference[position + 1]
                        if position + 1 < len(reference)
                        else reference[position]
                    )
                    if token_id != expected:
                        mismatches.append((position, token_id, expected))
                allowed = _SIX_BORDERLINE.get(fixture_name, set())
                unexpected = [
                    (position, got, expected)
                    for (position, got, expected) in mismatches
                    if position not in allowed
                ]
                assert unexpected == [], (
                    f"{fixture_name}: {len(unexpected)} unexpected token "
                    f"mismatches: {unexpected[:10]}"
                )
                if mismatches:
                    margins = _oracle_top2_margins(
                        cuda_runtime,
                        loaded,
                        frames,
                        ref_keys,
                        ref_values,
                        ref_mask,
                        reference,
                        lm_head_np,
                        [position for position, _, _ in mismatches],
                    )
                    for position, _, _ in mismatches:
                        margin = margins[position]
                        assert margin < _BORDERLINE_MARGIN, (
                            f"{fixture_name} position {position} mismatch but the "
                            f"fixture-oracle top-2 logit gap is {margin:.4f} "
                            f"(not a coin flip)"
                        )
            finally:
                enc.close()
                dec.close()
    finally:
        loaded.weights.free(runtime=cuda_runtime)
