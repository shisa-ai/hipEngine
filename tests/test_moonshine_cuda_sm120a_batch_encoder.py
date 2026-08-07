"""C8 phase-2 batch encoder kernels/runtime bit-exactness for CUDA ``sm_120a``.

The C8 batch milestones add static-B variants of the encoder primitives and a
``MoonshineCudaBatchEncoderRuntime`` whose rows must be **bit-exact** to B
independent batch-one encoder sessions at the conv front end, per-layer hidden
state, attention-mask, and batch cross-KV levels.  A batch encoder ->
``MoonshineCudaBatchRuntime`` handoff closes the torch-free batch path; its
decode tokens must equal B independent c=1 encoder+decoder sessions.

Two retained gates live here:

1. Kernel-level: the batch conv front end (conv1+tanh, GroupNorm, conv2/conv3)
   and the non-causal full-sequence self-attention reproduce B independent
   single-row calls exactly.
2. Runtime-level: at B=2/4, ``MoonshineCudaBatchEncoderRuntime`` equals B
   independent ``MoonshineCudaEncoderRuntime`` sessions (encoder output +
   mask + batch cross cache), and a subsequent batch decoder token window
   equals B independent c=1 encoder+decoder sessions, plus reset/teardown.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.loading.moonshine import load_moonshine_model

_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
)
_AUDIO_SAMPLES = 16_000  # 40 encoder frames
_EOS = 2


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


def _snapshot_available() -> bool:
    return os.path.isdir(_SNAPSHOT)


def _tensor_to_host(runtime, tensor, dtype=np.float16) -> np.ndarray:
    host = np.empty(tensor.shape, dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize),
        runtime=runtime,
    )
    return host


def _batch_audio(runtime, batch: int, *, mask_row: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ``[B, AUDIO_SAMPLES]`` FP16 audio + ``[B, ...]`` masks.

    Every row gets a distinct seed; optionally one row gets a half-length mask
    so the masked-attention path is exercised inside the batch.
    """
    audio = np.stack(
        [np.random.default_rng(seed).standard_normal(_AUDIO_SAMPLES) for seed in range(1, batch + 1)],
        axis=0,
    ).astype(np.float16)
    mask = np.ones((batch, _AUDIO_SAMPLES), dtype=np.int64)
    if mask_row is not None and 0 <= mask_row < batch:
        mask[mask_row, _AUDIO_SAMPLES // 2 :] = 0
    return audio, mask


def _run_c1_encoder(runtime, loaded, audio, mask):
    """One independent batch-one encoder session; returns output + mask host arrays."""
    from hipengine.runtime.moonshine_encoder_cuda import MoonshineCudaEncoderRuntime

    enc = MoonshineCudaEncoderRuntime(
        audio_samples=_AUDIO_SAMPLES, loaded_model=loaded, owns_weights=False
    )
    enc.prepare_encoder_kernels()
    try:
        enc.encode(audio, mask)
        out = _tensor_to_host(runtime, enc.encoder_output())
        mask_out = _tensor_to_host(runtime, enc.attention_mask(), dtype=np.int32)
        return out, mask_out
    finally:
        enc.close()


def _run_c1_decode_window(runtime, loaded, audio, mask, seed: int, steps: int) -> list[int]:
    """One c=1 encoder -> decoder session for ``steps`` positions; returns tokens."""
    from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime
    from hipengine.runtime.moonshine_encoder_cuda import MoonshineCudaEncoderRuntime

    enc = MoonshineCudaEncoderRuntime(
        audio_samples=_AUDIO_SAMPLES, loaded_model=loaded, owns_weights=False
    )
    enc.prepare_encoder_kernels()
    dec = MoonshineCudaResidentRuntime(
        encoder_frames=40, loaded_model=loaded, owns_weights=False
    )
    dec.prepare_decoder_kernels()
    try:
        enc.encode(audio, mask)
        enc.handoff_to(dec)
        tokens: list[int] = []
        token_id = seed
        for position in range(steps):
            dec.set_decode_state(token_id=token_id, position=position)
            dec.token_step()
            token_id = int(dec.read_token())
            tokens.append(token_id)
        return tokens
    finally:
        enc.close()
        dec.close()


# ---------------------------------------------------------------------------
# Kernel-level gates: batch conv front end and batch self-attention
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _snapshot_available(),
    reason="CUDA sm_120a gate or snapshot is not available",
)
def test_moonshine_cuda_batch_encoder_conv_frontend_bit_exact() -> None:
    """Batch conv1/GroupNorm/conv2/conv3 reproduce B single-row stages exactly."""
    from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
        build_moonshine_encoder,
        moonshine_conv1_tanh_batch_fp16,
        moonshine_conv2_gelu_batch_fp16,
        moonshine_conv3_gelu_batch_fp16,
        moonshine_groupnorm_batch_fp16,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=runtime
    )
    library = build_moonshine_encoder(load=True)
    batch = 2
    audio, mask = _batch_audio(runtime, batch, mask_row=1)
    hidden = loaded.spec.hidden_size
    try:
        length = (_AUDIO_SAMPLES - 127) // 64 + 1
        conv2_length = (length - 7) // 3 + 1
        frames = (conv2_length - 3) // 2 + 1
        from hipengine.core.memory import copy_host_to_device, malloc

        allocations = []

        def upload(values):
            host = np.ascontiguousarray(values)
            device = malloc(host.nbytes, runtime=runtime)
            allocations.append(device)
            copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
            return device

        def download(device, shape, dtype):
            host = np.empty(shape, dtype=dtype)
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
            return host

        audio_batch = upload(audio.astype(np.float16))
        batch_conv1 = upload(np.zeros((batch, hidden, length), dtype=np.float16))
        batch_gn_partial = upload(np.zeros((batch, hidden, 2), dtype=np.float32))
        batch_gn_mean = upload(np.zeros((batch, 2), dtype=np.float32))
        batch_conv2 = upload(np.zeros((batch, 2 * hidden, conv2_length), dtype=np.float16))
        batch_hidden = upload(np.zeros((batch, frames, hidden), dtype=np.float16))

        # B independent single-row stages
        c1_conv1 = []
        c1_conv2 = []
        c1_hidden = []
        for b in range(batch):
            a = upload(audio[b : b + 1].astype(np.float16))
            conv1 = upload(np.zeros((hidden, length), dtype=np.float16))
            gn_partial = upload(np.zeros((hidden, 2), dtype=np.float32))
            gn_mean = upload(np.zeros((2,), dtype=np.float32))
            conv2 = upload(np.zeros((2 * hidden, conv2_length), dtype=np.float16))
            hid = upload(np.zeros((frames, hidden), dtype=np.float16))
            moonshine_conv1_tanh_batch_fp16(
                a.ptr, loaded.weights["model.encoder.conv1.weight"].ptr,
                conv1.ptr, 1, _AUDIO_SAMPLES, length,
                library=library, runtime=runtime,
            )
            moonshine_groupnorm_batch_fp16(
                conv1.ptr, loaded.weights["model.encoder.groupnorm.weight"].ptr,
                loaded.weights["model.encoder.groupnorm.bias"].ptr, conv1.ptr,
                gn_partial.ptr, gn_mean.ptr, 1, hidden, length,
                library=library, runtime=runtime,
            )
            moonshine_conv2_gelu_batch_fp16(
                conv1.ptr, loaded.weights["model.encoder.conv2.weight"].ptr,
                loaded.weights["model.encoder.conv2.bias"].ptr, conv2.ptr,
                1, length, conv2_length, library=library, runtime=runtime,
            )
            moonshine_conv3_gelu_batch_fp16(
                conv2.ptr, loaded.weights["model.encoder.conv3.weight"].ptr,
                loaded.weights["model.encoder.conv3.bias"].ptr, hid.ptr,
                1, conv2_length, frames, library=library, runtime=runtime,
            )
            c1_conv1.append(download(conv1, (hidden, length), np.float16))
            c1_conv2.append(download(conv2, (2 * hidden, conv2_length), np.float16))
            c1_hidden.append(download(hid, (frames, hidden), np.float16))

        moonshine_conv1_tanh_batch_fp16(
            audio_batch.ptr, loaded.weights["model.encoder.conv1.weight"].ptr,
            batch_conv1.ptr, batch, _AUDIO_SAMPLES, length,
            library=library, runtime=runtime,
        )
        moonshine_groupnorm_batch_fp16(
            batch_conv1.ptr, loaded.weights["model.encoder.groupnorm.weight"].ptr,
            loaded.weights["model.encoder.groupnorm.bias"].ptr, batch_conv1.ptr,
            batch_gn_partial.ptr, batch_gn_mean.ptr, batch, hidden, length,
            library=library, runtime=runtime,
        )
        moonshine_conv2_gelu_batch_fp16(
            batch_conv1.ptr, loaded.weights["model.encoder.conv2.weight"].ptr,
            loaded.weights["model.encoder.conv2.bias"].ptr, batch_conv2.ptr,
            batch, length, conv2_length, library=library, runtime=runtime,
        )
        moonshine_conv3_gelu_batch_fp16(
            batch_conv2.ptr, loaded.weights["model.encoder.conv3.weight"].ptr,
            loaded.weights["model.encoder.conv3.bias"].ptr, batch_hidden.ptr,
            batch, conv2_length, frames, library=library, runtime=runtime,
        )
        runtime.device_synchronize()

        got_conv1 = download(batch_conv1, (batch, hidden, length), np.float16)
        got_conv2 = download(batch_conv2, (batch, 2 * hidden, conv2_length), np.float16)
        got_hidden = download(batch_hidden, (batch, frames, hidden), np.float16)
        for b in range(batch):
            assert np.array_equal(got_conv1[b], c1_conv1[b]), f"row {b} conv1 diverged"
            assert np.array_equal(got_conv2[b], c1_conv2[b]), f"row {b} conv2 diverged"
            assert np.array_equal(got_hidden[b], c1_hidden[b]), f"row {b} conv3 hidden diverged"
        for allocation in allocations:
            from hipengine.core.memory import free

            free(allocation, runtime=runtime)
    finally:
        loaded.weights.free(runtime=runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _snapshot_available(),
    reason="CUDA sm_120a gate or snapshot is not available",
)
def test_moonshine_cuda_batch_encoder_attention_bit_exact() -> None:
    """Batch non-causal self-attention reproduces B single-row calls exactly."""
    from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
        build_moonshine_encoder,
        moonshine_encoder_attention_batch_fp16,
        moonshine_encoder_attention_fp16,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=runtime
    )
    library = build_moonshine_encoder(load=True)
    batch = 2
    frames = 40
    heads = 8
    head_dim = 52
    hidden = loaded.spec.hidden_size
    from hipengine.core.memory import copy_host_to_device, free, malloc

    rng = np.random.default_rng(3)
    scale = head_dim**-0.5
    try:
        allocations = []

        def upload(values, dtype):
            host = np.ascontiguousarray(values.astype(dtype))
            device = malloc(host.nbytes, runtime=runtime)
            allocations.append(device)
            copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
            return device

        def download(device, shape, dtype):
            host = np.empty(shape, dtype=dtype)
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
            return host

        mask = np.ones((batch, frames), dtype=np.int32)
        mask[1, frames // 2 :] = 0
        rng_q = np.random.default_rng(11)
        rng_k = np.random.default_rng(12)
        rng_v = np.random.default_rng(13)
        # Per-row arrays generated once so the batch and c=1 rows see identical inputs.
        row_q = [rng_q.standard_normal((heads, frames, head_dim)).astype(np.float16) for _ in range(batch)]
        row_k = [rng_k.standard_normal((heads, frames, head_dim)).astype(np.float16) for _ in range(batch)]
        row_v = [rng_v.standard_normal((heads, frames, head_dim)).astype(np.float16) for _ in range(batch)]
        batch_q = upload(np.stack(row_q, axis=0), np.float16)
        batch_k = upload(np.stack(row_k, axis=0), np.float16)
        batch_v = upload(np.stack(row_v, axis=0), np.float16)
        batch_mask = upload(mask, np.int32)
        batch_out = upload(np.zeros((batch, frames, hidden)), np.float16)
        moonshine_encoder_attention_batch_fp16(
            batch_q.ptr, batch_k.ptr, batch_v.ptr, batch_mask.ptr, batch_out.ptr,
            batch, heads, head_dim, frames, scale=scale, library=library, runtime=runtime,
        )

        c1_outs = []
        for b in range(batch):
            q = upload(row_q[b], np.float16)
            k = upload(row_k[b], np.float16)
            v = upload(row_v[b], np.float16)
            row_mask = upload(mask[b : b + 1], np.int32)
            out = upload(np.zeros((frames, hidden)), np.float16)
            moonshine_encoder_attention_fp16(
                q.ptr, k.ptr, v.ptr, row_mask.ptr, out.ptr, heads, head_dim,
                frames, scale=scale, library=library, runtime=runtime,
            )
            c1_outs.append(download(out, (frames, hidden), np.float16))
        runtime.device_synchronize()

        got = download(batch_out, (batch, frames, hidden), np.float16)
        for b in range(batch):
            assert np.array_equal(got[b], c1_outs[b]), f"row {b} attention diverged"
        for allocation in allocations:
            free(allocation, runtime=runtime)
    finally:
        loaded.weights.free(runtime=runtime)


# ---------------------------------------------------------------------------
# Runtime-level gates
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _snapshot_available(),
    reason="CUDA sm_120a gate or snapshot is not available",
)
@pytest.mark.parametrize("batch_size", [2, 4])
def test_moonshine_cuda_batch_encoder_runtime_bit_exact_vs_c1_sessions(
    batch_size: int,
) -> None:
    """C8 phase-2: batch encoder equals B independent c=1 encoder sessions.

    For B=2/4 on identical-length audio, compare encoder output, mask, and the
    batch cross-KV handoff against B independent ``MoonshineCudaEncoderRuntime``
    sessions; then a batch decoder token window equals B independent c=1
    encoder+decoder sessions, and the reset/teardown parity closes the contract.
    """
    from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime
    from hipengine.runtime.moonshine_encoder_cuda_batch import (
        MoonshineCudaBatchEncoderRuntime,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=runtime
    )
    try:
        audio, mask = _batch_audio(runtime, batch_size, mask_row=1)
        seeds = list(range(1, batch_size + 1))

        # ---- B independent c=1 encoders ------------------------------------
        c1 = []
        for b in range(batch_size):
            c1.append(_run_c1_encoder(runtime, loaded, audio[b : b + 1], mask[b : b + 1]))

        # ---- batch encoder --------------------------------------------------
        benc = MoonshineCudaBatchEncoderRuntime(
            max_batch=batch_size,
            audio_samples=_AUDIO_SAMPLES,
            loaded_model=loaded,
            owns_weights=False,
        )
        benc.prepare_encoder_kernels()
        bdec = MoonshineCudaBatchRuntime(
            max_batch=batch_size,
            encoder_frames=40,
            loaded_model=loaded,
            owns_weights=False,
        )
        bdec.prepare_decoder_kernels()
        try:
            benc.encode(audio, mask)
            batch_out = _tensor_to_host(runtime, benc.encoder_output())
            batch_mask = _tensor_to_host(runtime, benc.attention_mask(), dtype=np.int32)
            for b in range(batch_size):
                ref_out, ref_mask = c1[b]
                assert np.array_equal(batch_out[b], ref_out[0]), (
                    f"B={batch_size} row {b} encoder output diverged"
                )
                assert np.array_equal(batch_mask[b], ref_mask[0]), (
                    f"B={batch_size} row {b} encoder mask diverged"
                )

            # ---- batch cross-KV handoff vs B c=1 handoffs -------------------
            benc.handoff_to(bdec)
            ref_caches = []
            for b in range(batch_size):
                from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime
                from hipengine.runtime.moonshine_encoder_cuda import (
                    MoonshineCudaEncoderRuntime,
                )

                enc = MoonshineCudaEncoderRuntime(
                    audio_samples=_AUDIO_SAMPLES,
                    loaded_model=loaded,
                    owns_weights=False,
                )
                enc.prepare_encoder_kernels()
                dec = MoonshineCudaResidentRuntime(
                    encoder_frames=40, loaded_model=loaded, owns_weights=False
                )
                dec.prepare_decoder_kernels()
                try:
                    enc.encode(audio[b : b + 1], mask[b : b + 1])
                    enc.handoff_to(dec)
                    caches = []
                    for layer in range(8):
                        cache = dec.cross_cache(layer)
                        caches.append(
                            (
                                _tensor_to_host(runtime, cache.key)[0],
                                _tensor_to_host(runtime, cache.value)[0],
                            )
                        )
                    ref_caches.append(caches)
                finally:
                    enc.close()
                    dec.close()
            for b in range(batch_size):
                for layer in range(8):
                    cache = bdec.cross_cache(layer)
                    got_key = _tensor_to_host(runtime, cache.key)[b]
                    got_value = _tensor_to_host(runtime, cache.value)[b]
                    ref_key, ref_value = ref_caches[b][layer]
                    assert np.array_equal(got_key, ref_key), (
                        f"B={batch_size} row {b} layer {layer} cross key diverged"
                    )
                    assert np.array_equal(got_value, ref_value), (
                        f"B={batch_size} row {b} layer {layer} cross value diverged"
                    )

            # ---- batch decoder token window vs B c=1 sessions ----------------
            c1_tokens = [
                _run_c1_decode_window(runtime, loaded, audio[b : b + 1], mask[b : b + 1], seeds[b], 6)
                for b in range(batch_size)
            ]
            batch_tokens = [[] for _ in range(batch_size)]
            toks = np.asarray(seeds, dtype=np.int64)
            for position in range(6):
                bdec.set_batch_decode_state(tokens=toks.tolist(), position=position)
                bdec.batch_token_step()
                toks = bdec.read_tokens()
                for b in range(batch_size):
                    batch_tokens[b].append(int(toks[b]))
            for b in range(batch_size):
                assert batch_tokens[b] == c1_tokens[b], (
                    f"B={batch_size} row {b} token window diverged: "
                    f"batch={batch_tokens[b]} c1={c1_tokens[b]}"
                )

            # reset: a fresh decode without reloading the cross cache must
            # reproduce the same window (fixed addresses, state zeroed).
            bdec.reset_generation(clear_cross_cache=False)
            again = [[] for _ in range(batch_size)]
            toks = np.asarray(seeds, dtype=np.int64)
            for position in range(6):
                bdec.set_batch_decode_state(tokens=toks.tolist(), position=position)
                bdec.batch_token_step()
                toks = bdec.read_tokens()
                for b in range(batch_size):
                    again[b].append(int(toks[b]))
            for b in range(batch_size):
                assert again[b] == batch_tokens[b], (
                    f"B={batch_size} row {b} reset window diverged"
                )

            contract = bdec.allocation_contract()
            assert contract["max_batch"] == batch_size
        finally:
            # Close in reverse construction order so each runtime's teardown
            # leak check runs while the other's workspace is already freed.
            bdec.close()
            benc.close()
            assert bdec.teardown_returned_to_baseline is True
            assert benc.teardown_returned_to_baseline is True
    finally:
        loaded.weights.free(runtime=runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _snapshot_available(),
    reason="CUDA sm_120a gate or snapshot is not available",
)
def test_moonshine_cuda_batch_encoder_rejects_bad_batch() -> None:
    """Batch encoder validates batch size, length, and finite inputs."""
    from hipengine.runtime.moonshine_encoder_cuda_batch import (
        MoonshineCudaBatchEncoderRuntime,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=runtime
    )
    try:
        with pytest.raises(ValueError, match="max_batch"):
            MoonshineCudaBatchEncoderRuntime(
                max_batch=0, audio_samples=_AUDIO_SAMPLES, loaded_model=loaded
            )
        benc = MoonshineCudaBatchEncoderRuntime(
            max_batch=2, audio_samples=_AUDIO_SAMPLES, loaded_model=loaded,
            owns_weights=False,
        )
        try:
            with pytest.raises(ValueError, match="real_samples"):
                benc.upload_input(
                    np.zeros((2, _AUDIO_SAMPLES // 2), dtype=np.float16)
                )
            with pytest.raises(ValueError, match="shape"):
                benc.upload_input(np.zeros((3, _AUDIO_SAMPLES), dtype=np.float16))
            with pytest.raises(ValueError, match="finite"):
                benc.upload_input(
                    np.full((2, _AUDIO_SAMPLES), np.nan, dtype=np.float16)
                )
            with pytest.raises(RuntimeError, match="not uploaded"):
                benc.run_encode()
        finally:
            benc.close()
            assert benc.teardown_returned_to_baseline is True
    finally:
        loaded.weights.free(runtime=runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _snapshot_available(),
    reason="CUDA sm_120a gate or snapshot is not available",
)
def test_moonshine_cuda_batch_encoder_handoff_contract() -> None:
    """Batch encoder handoff validates the decoder is a batch runtime."""
    from hipengine.runtime.moonshine_encoder_cuda_batch import (
        MoonshineCudaBatchEncoderRuntime,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=runtime
    )
    try:
        benc = MoonshineCudaBatchEncoderRuntime(
            max_batch=1, audio_samples=_AUDIO_SAMPLES, loaded_model=loaded,
            owns_weights=False,
        )
        benc.prepare_encoder_kernels()
        try:
            with pytest.raises(TypeError, match="MoonshineCudaBatchRuntime"):
                benc.handoff_to(object())
            with pytest.raises(TypeError, match="MoonshineCudaBatchRuntime"):
                benc.handoff_to(None)
        finally:
            benc.close()
    finally:
        loaded.weights.free(runtime=runtime)
