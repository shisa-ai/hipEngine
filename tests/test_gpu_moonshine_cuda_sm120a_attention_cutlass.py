"""AOT CuTe attention source-identity, deployment, and GPU route gates."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)

HEADS = 8
HEAD_DIM = 52
HIDDEN = HEADS * HEAD_DIM
_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    os.environ.get(
        "SNAPSHOT_DIR",
        "/home/lhl/.cache/huggingface/hub/"
        "models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
        "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
    ),
)


def _cuda_cutlass_gate_enabled() -> bool:
    if os.environ.get("HIPENGINE_RUN_CUDA_SM120A") != "1":
        return False
    if os.environ.get("HIPENGINE_CUDA_ARCH") != "sm_120a":
        return False
    if os.environ.get("HIPENGINE_RUN_CUDA_CUTLASS_ATTENTION_GATE") != "1":
        return False
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError:
        return False
    return bool(os.environ.get("HIPENGINE_CUTLASS_DIR"))


def _fake_include(root: Path) -> Path:
    include = root / "include"
    (include / "cute").mkdir(parents=True)
    (include / "cutlass").mkdir()
    (include / "cute" / "tensor.hpp").write_text("// cute tensor\n")
    (include / "cutlass" / "cutlass.h").write_text("// cutlass\n")
    return include


def test_cutlass_source_identity_hashes_all_consumed_headers(tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention_cutlass import (
        cutlass_source_identity,
    )

    include = _fake_include(tmp_path)
    first = cutlass_source_identity(include)
    assert first.git_commit is None
    assert first.git_revision.startswith("content-")
    assert first.consumed_headers_dirty is False
    assert first.header_file_count == 2

    (include / "cutlass" / "detail.hpp").write_text("// transitive detail\n")
    second = cutlass_source_identity(include)
    assert second.header_file_count == 3
    assert second.headers_sha256 != first.headers_sha256


def test_cutlass_build_plan_key_covers_header_content(monkeypatch, tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.attention import moonshine_attention_cutlass as module

    include = _fake_include(tmp_path / "source")
    monkeypatch.setattr(module, "cutlass_include_dir", lambda: include)
    first = module.plan_moonshine_attention_cutlass_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc provenance test",
    )
    flags = " ".join(first.flags)
    assert "MOONSHINE_CUTLASS_REVISION" in flags
    assert "MOONSHINE_CUTLASS_COMMIT" in flags
    assert "MOONSHINE_CUTLASS_HEADERS_SHA256" in flags

    (include / "cute" / "tensor.hpp").write_text("// changed consumed header\n")
    second = module.plan_moonshine_attention_cutlass_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc provenance test",
    )
    assert second.cache_key != first.cache_key


def test_cutlass_build_rejects_dirty_consumed_headers(monkeypatch, tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.attention import moonshine_attention_cutlass as module

    include = _fake_include(tmp_path)
    clean = module.cutlass_source_identity(include)
    dirty = module.CutlassSourceIdentity(
        root=clean.root,
        git_commit="a" * 40,
        git_revision="dirty-test",
        consumed_headers_dirty=True,
        headers_sha256=clean.headers_sha256,
        header_file_count=clean.header_file_count,
    )
    monkeypatch.setattr(module, "cutlass_source_identity", lambda _include: dirty)
    with pytest.raises(RuntimeError, match="dirty consumed CUTLASS headers"):
        module._cutlass_flags(include)


def test_batch_encoder_attention_route_validates_before_loading() -> None:
    from hipengine.runtime.moonshine_encoder_cuda_batch import (
        MoonshineCudaBatchEncoderRuntime,
    )

    with pytest.raises(ValueError, match="attention_route"):
        MoonshineCudaBatchEncoderRuntime(
            max_batch=1,
            audio_samples=16384,
            model_path="unused",
            attention_route="invalid",
        )


def test_cutlass_batch_wrapper_keeps_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention_cutlass import (
        moonshine_encoder_attention_cutlass_batch_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_encoder_attention_cutlass_batch_fp16 = (
            FakeKernel()
        )

    library = FakeLibrary()
    scale = HEAD_DIM**-0.5
    moonshine_encoder_attention_cutlass_batch_fp16(
        1,
        2,
        3,
        4,
        5,
        8,
        HEADS,
        HEAD_DIM,
        1248,
        library=library,
        runtime=object(),
        stream=7,
    )
    assert library.hipengine_cuda_sm120a_moonshine_encoder_attention_cutlass_batch_fp16.calls == [
        (1, 2, 3, 4, 5, 8, HEADS, HEAD_DIM, 1248, scale, 512, 7)
    ]
    with pytest.raises(ValueError, match="batch"):
        moonshine_encoder_attention_cutlass_batch_fp16(
            1, 2, 3, 4, 5, 0, HEADS, HEAD_DIM, 40, library=library
        )


def test_prebuilt_so_mode_bypasses_cutlass_source_tree(monkeypatch, tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.attention import moonshine_attention_cutlass as module

    prebuilt = tmp_path / "attention.so"
    prebuilt.write_bytes(b"test prebuilt identity")
    loaded: list[str] = []
    sentinel = object()

    def fake_cdll(path: str):
        loaded.append(path)
        return sentinel

    monkeypatch.delenv("HIPENGINE_CUTLASS_DIR", raising=False)
    monkeypatch.setattr(module.ctypes, "CDLL", fake_cdll)
    assert module.build_moonshine_attention_cutlass(prebuilt_path=prebuilt) is sentinel
    assert loaded == [str(prebuilt)]


@pytest.mark.skipif(
    not _cuda_cutlass_gate_enabled(),
    reason="CUDA CUTLASS attention gate is not enabled",
)
def test_cutlass_attention_matches_custom_at_certified_buckets() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention_cutlass import (
        build_moonshine_attention_cutlass,
        moonshine_encoder_attention_cutlass_batch_fp16,
        moonshine_encoder_attention_cutlass_fp16,
    )
    from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
        build_moonshine_encoder,
        moonshine_encoder_attention_batch_fp16,
        moonshine_encoder_attention_fp16,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    custom_library = build_moonshine_encoder(load=True)
    aot_library = build_moonshine_attention_cutlass(load=True)
    # Exercise deployment-mode loading of the exact generated binary too.
    prebuilt_library = build_moonshine_attention_cutlass(
        prebuilt_path=Path(aot_library._name)
    )
    allocations = []
    try:
        for sequence in (40, 207, 1248):
            rng = np.random.default_rng(0xC07E + sequence)
            q = rng.uniform(-5, 5, (HEADS, sequence, HEAD_DIM)).astype(np.float16)
            k = rng.uniform(-5, 5, (HEADS, sequence, HEAD_DIM)).astype(np.float16)
            v = rng.uniform(-5, 5, (HEADS, sequence, HEAD_DIM)).astype(np.float16)
            mask = np.ones(sequence, dtype=np.int32)
            mask[3::7] = 0
            d_q = _upload(q, runtime, allocations)
            d_k = _upload(k, runtime, allocations)
            d_v = _upload(v, runtime, allocations)
            d_mask = _upload(mask, runtime, allocations)
            d_custom = _alloc((sequence, HIDDEN), runtime, allocations)
            d_aot = _alloc((sequence, HIDDEN), runtime, allocations)

            moonshine_encoder_attention_fp16(
                d_q.ptr,
                d_k.ptr,
                d_v.ptr,
                d_mask.ptr,
                d_custom.ptr,
                HEADS,
                HEAD_DIM,
                sequence,
                library=custom_library,
                runtime=runtime,
            )
            moonshine_encoder_attention_cutlass_fp16(
                d_q.ptr,
                d_k.ptr,
                d_v.ptr,
                d_mask.ptr,
                d_aot.ptr,
                HEADS,
                HEAD_DIM,
                sequence,
                library=prebuilt_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            custom = _download(d_custom, (sequence, HIDDEN), runtime)
            aot = _download(d_aot, (sequence, HIDDEN), runtime)
            assert np.max(np.abs(custom.astype(np.float32) - aot.astype(np.float32))) <= 2**-8

        for batch in (1, 2, 4, 8):
            for sequence in (40, 207, 1248):
                rng = np.random.default_rng(0xBA7C + batch * 2000 + sequence)
                shape = (batch, HEADS, sequence, HEAD_DIM)
                q = rng.uniform(-5, 5, shape).astype(np.float16)
                k = rng.uniform(-5, 5, shape).astype(np.float16)
                v = rng.uniform(-5, 5, shape).astype(np.float16)
                mask = np.ones((batch, sequence), dtype=np.int32)
                for row in range(batch):
                    mask[row, (row + 3) :: 7] = 0
                d_q = _upload(q, runtime, allocations)
                d_k = _upload(k, runtime, allocations)
                d_v = _upload(v, runtime, allocations)
                d_mask = _upload(mask, runtime, allocations)
                d_custom = _alloc((batch, sequence, HIDDEN), runtime, allocations)
                d_aot = _alloc((batch, sequence, HIDDEN), runtime, allocations)
                moonshine_encoder_attention_batch_fp16(
                    d_q.ptr,
                    d_k.ptr,
                    d_v.ptr,
                    d_mask.ptr,
                    d_custom.ptr,
                    batch,
                    HEADS,
                    HEAD_DIM,
                    sequence,
                    library=custom_library,
                    runtime=runtime,
                )
                moonshine_encoder_attention_cutlass_batch_fp16(
                    d_q.ptr,
                    d_k.ptr,
                    d_v.ptr,
                    d_mask.ptr,
                    d_aot.ptr,
                    batch,
                    HEADS,
                    HEAD_DIM,
                    sequence,
                    library=prebuilt_library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                custom = _download(
                    d_custom,
                    (batch, sequence, HIDDEN),
                    runtime,
                )
                aot = _download(d_aot, (batch, sequence, HIDDEN), runtime)
                max_abs = np.max(
                    np.abs(custom.astype(np.float32) - aot.astype(np.float32))
                )
                assert max_abs <= 2**-8
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(
    not _cuda_cutlass_gate_enabled() or not os.path.isdir(_SNAPSHOT),
    reason="CUDA CUTLASS attention gate or snapshot is not available",
)
def test_cutlass_batch_runtime_graph_handoff_and_tokens_match_custom() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model
    from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime
    from hipengine.runtime.moonshine_encoder_cuda_batch import (
        MoonshineCudaBatchEncoderRuntime,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT,
        device=Device("cuda", 0),
        runtime=runtime,
    )
    batch = 2
    samples = 16_000
    frames = 40
    audio = np.stack(
        [
            np.random.default_rng(500 + row).normal(0, 0.02, samples)
            for row in range(batch)
        ]
    ).astype(np.float16)
    mask = np.ones((batch, samples), dtype=np.int64)
    mask[1, 12_000:] = 0

    def run(attention_route: str) -> dict[str, object]:
        encoder = MoonshineCudaBatchEncoderRuntime(
            max_batch=batch,
            audio_samples=samples,
            loaded_model=loaded,
            owns_weights=False,
            attention_route=attention_route,
        )
        decoder = MoonshineCudaBatchRuntime(
            max_batch=batch,
            encoder_frames=frames,
            loaded_model=loaded,
            owns_weights=False,
        )
        try:
            encoder.prepare_encoder_kernels()
            decoder.prepare_decoder_kernels()
            encoder.upload_input(audio, mask)
            encoder.run_encode()
            encoder.handoff_to(decoder)
            eager_hidden = _tensor_to_host(runtime, encoder.encoder_output())
            eager_mask = _tensor_to_host(
                runtime,
                encoder.attention_mask(),
                dtype=np.int32,
            )
            eager_cross = _tensor_to_host(runtime, decoder.cross_cache(0).key)

            encoder.capture_encoder_chain(decoder)
            encoder.graph_encode_and_handoff(decoder)
            runtime.stream_synchronize(decoder.stream)
            graph_hidden = _tensor_to_host(runtime, encoder.encoder_output())
            graph_cross = _tensor_to_host(runtime, decoder.cross_cache(0).key)
            assert np.array_equal(graph_hidden, eager_hidden)
            assert np.array_equal(graph_cross, eager_cross)

            tokens = np.ones(batch, dtype=np.int64)
            transcript: list[list[int]] = [[] for _ in range(batch)]
            for _ in range(24):
                decoder.set_batch_decode_state(
                    tokens=tokens.tolist(),
                    position=decoder.self_cache_length,
                )
                decoder.batch_token_step()
                tokens = decoder.read_tokens()
                for row, token in enumerate(tokens.tolist()):
                    transcript[row].append(int(token))
            return {
                "hidden": graph_hidden,
                "mask": eager_mask,
                "cross": graph_cross,
                "tokens": transcript,
                "graph": encoder.encoder_chain_graph_contract(),
            }
        finally:
            decoder.close()
            encoder.close()

    try:
        custom = run("custom")
        cutlass = run("cutlass")
        assert np.array_equal(custom["mask"], cutlass["mask"])
        hidden_diff = np.abs(
            custom["hidden"].astype(np.float32)
            - cutlass["hidden"].astype(np.float32)
        )
        cross_diff = np.abs(
            custom["cross"].astype(np.float32)
            - cutlass["cross"].astype(np.float32)
        )
        assert float(hidden_diff.max()) <= 0.01
        # Eight-layer AOT reduction-order differences compound through the
        # cross projection; measured B=2 max is 0.05176 FP16, so retain the
        # next exact power-of-two envelope while token parity remains exact.
        assert float(cross_diff.max()) <= 0.0625
        assert custom["tokens"] == cutlass["tokens"]
        assert custom["graph"]["captured"] is True
        assert cutlass["graph"]["captured"] is True
        assert custom["graph"]["replay_count"] == 1
        assert cutlass["graph"]["replay_count"] == 1
    finally:
        loaded.weights.free(runtime=runtime)


def _tensor_to_host(runtime, tensor, *, dtype=np.float16) -> np.ndarray:
    host = np.empty(tensor.shape, dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize),
        runtime=runtime,
    )
    return host


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(np.float16).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
