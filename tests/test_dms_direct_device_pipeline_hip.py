"""Integrated direct-pointer pack/append RED gate for external DMS."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kvcache.dms import create_dms_bf16_backend
from hipengine.kvcache.dms_device import DMSExternalLinearDeviceProjector
from tests.test_dms_external_runtime import _source
from tests.test_dms_streaming_pack_hip import _bf16_bits


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _admit(backend, *, request_id: int, tokens: int, max_new_tokens: int) -> None:
    request = SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tuple(range(tokens)),
        max_new_tokens=max_new_tokens,
    )
    backend.reserve(
        backend.estimate(
            request,
            None,
            {
                "kind": "admission",
                "tokens": tokens,
                "max_new_tokens": max_new_tokens,
            },
        )
    )


def _compare_metadata(host_backend, device_backend, request_id: int) -> None:
    host = host_backend.state_for_request(request_id)
    device = device_backend.state_for_request(request_id)
    np.testing.assert_array_equal(device.base_offsets, host.base_offsets)
    np.testing.assert_array_equal(device.range_capacity, host.range_capacity)
    np.testing.assert_array_equal(device.live_counts, host.live_counts)
    np.testing.assert_array_equal(device.token_positions, host.token_positions)
    np.testing.assert_array_equal(device.evict_mask, host.evict_mask)
    assert device.logical_tokens == host.logical_tokens


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_external_dms_direct_device_pack_append_has_no_host_kv_shadow() -> None:
    source = _source()
    layers = source.config.num_layers
    heads = source.config.num_kv_heads
    dim = source.config.head_dim
    hidden_size = int(source.config.hidden_size)
    tokens = 6
    max_new = 3
    request_id = 41
    host_backend = create_dms_bf16_backend(
        retrofit=source.config,
        slots_per_layer=128,
        max_request_rows=1,
        max_pack_rows=16,
        device_payloads=False,
    )
    device_backend = create_dms_bf16_backend(
        retrofit=source.config,
        slots_per_layer=128,
        max_request_rows=1,
        max_pack_rows=16,
        device_payloads=True,
        device_backend="hip_gfx1151",
    )
    projector = DMSExternalLinearDeviceProjector(source, backend="hip_gfx1151")
    buffers = []
    try:
        _admit(
            host_backend,
            request_id=request_id,
            tokens=tokens,
            max_new_tokens=max_new,
        )
        _admit(
            device_backend,
            request_id=request_id,
            tokens=tokens,
            max_new_tokens=max_new,
        )
        rng = np.random.default_rng(4141)
        hidden = rng.standard_normal((tokens, layers, hidden_size)).astype(np.float32)
        k = rng.standard_normal((tokens, layers, heads, dim)).astype(np.float32)
        v = rng.standard_normal(k.shape).astype(np.float32)
        decisions_layer_major = np.empty((layers, tokens, heads), dtype=np.uint8)
        decisions_buf = malloc(decisions_layer_major.nbytes)
        logits_buf = malloc(tokens * heads * np.dtype(np.float32).itemsize)
        buffers.extend((decisions_buf, logits_buf))

        cpu_decisions = np.empty((tokens, layers, heads), dtype=np.bool_)
        for layer in range(layers):
            hidden_bits = _bf16_bits(hidden[:, layer, :])
            k_bits = _bf16_bits(k[:, layer, :, :])
            v_bits = _bf16_bits(v[:, layer, :, :])
            hidden_buf = malloc(hidden_bits.nbytes)
            k_buf = malloc(k_bits.nbytes)
            v_buf = malloc(v_bits.nbytes)
            buffers.extend((hidden_buf, k_buf, v_buf))
            copy_host_to_device(hidden_buf, host_array_ptr(hidden_bits), hidden_bits.nbytes)
            copy_host_to_device(k_buf, host_array_ptr(k_bits), k_bits.nbytes)
            copy_host_to_device(v_buf, host_array_ptr(v_bits), v_bits.nbytes)
            decision_offset = layer * tokens * heads
            projector.project(
                hidden_ptr=hidden_buf.ptr,
                compact_layer_index=layer,
                tokens=tokens,
                logits_ptr=logits_buf.ptr,
                evict_ptr=decisions_buf.ptr + decision_offset,
            )
            device_backend.device_streaming_pack_layer(
                request_id,
                layer,
                k_ptr=k_buf.ptr,
                v_ptr=v_buf.ptr,
                evict_ptr=decisions_buf.ptr + decision_offset,
                tokens=tokens,
            )
            _, cpu_decisions[:, layer, :] = source.project(
                hidden_bits,
                compact_layer_index=layer,
            )
        copy_device_to_host(
            host_array_ptr(decisions_layer_major),
            decisions_buf,
            decisions_layer_major.nbytes,
        )
        decisions_host = decisions_layer_major.transpose(1, 0, 2)
        np.testing.assert_array_equal(decisions_host.astype(np.bool_), cpu_decisions)
        device_backend.finalize_device_streaming_pack(
            request_id,
            eviction=decisions_host.astype(np.bool_),
            tokens=tokens,
        )
        host_backend.streaming_pack(request_id, k, v, cpu_decisions)
        _compare_metadata(host_backend, device_backend, request_id)
        device_state = device_backend.state_for_request(request_id)
        assert not device_state.k_payload
        assert not device_state.v_payload

        hidden_new = rng.standard_normal((layers, hidden_size)).astype(np.float32)
        k_new = rng.standard_normal((layers, heads, dim)).astype(np.float32)
        v_new = rng.standard_normal(k_new.shape).astype(np.float32)
        decisions_new = np.empty((layers, heads), dtype=np.uint8)
        decisions_new_buf = malloc(decisions_new.nbytes)
        append_logits = malloc(heads * np.dtype(np.float32).itemsize)
        buffers.extend((decisions_new_buf, append_logits))
        cpu_new = np.empty((layers, heads), dtype=np.bool_)
        for layer in range(layers):
            hidden_bits = _bf16_bits(hidden_new[layer : layer + 1])
            k_bits = _bf16_bits(k_new[layer])
            v_bits = _bf16_bits(v_new[layer])
            hidden_buf = malloc(hidden_bits.nbytes)
            k_buf = malloc(k_bits.nbytes)
            v_buf = malloc(v_bits.nbytes)
            buffers.extend((hidden_buf, k_buf, v_buf))
            copy_host_to_device(hidden_buf, host_array_ptr(hidden_bits), hidden_bits.nbytes)
            copy_host_to_device(k_buf, host_array_ptr(k_bits), k_bits.nbytes)
            copy_host_to_device(v_buf, host_array_ptr(v_bits), v_bits.nbytes)
            projector.project(
                hidden_ptr=hidden_buf.ptr,
                compact_layer_index=layer,
                tokens=1,
                logits_ptr=append_logits.ptr,
                evict_ptr=decisions_new_buf.ptr + layer * heads,
            )
            device_backend.device_append_layer(
                request_id,
                layer,
                k_ptr=k_buf.ptr,
                v_ptr=v_buf.ptr,
                evict_ptr=decisions_new_buf.ptr + layer * heads,
                position=tokens,
            )
            _, cpu_layer = source.project(hidden_bits, compact_layer_index=layer)
            cpu_new[layer] = cpu_layer[0]
        copy_device_to_host(
            host_array_ptr(decisions_new), decisions_new_buf, decisions_new.nbytes
        )
        np.testing.assert_array_equal(decisions_new.astype(np.bool_), cpu_new)
        device_backend.finalize_device_append(
            request_id,
            eviction=decisions_new.astype(np.bool_),
            position=tokens,
        )
        host_backend.append_decode(
            request_id,
            k_new,
            v_new,
            cpu_new,
            position=tokens,
        )
        _compare_metadata(host_backend, device_backend, request_id)
        assert not device_backend.state_for_request(request_id).k_payload
        assert not device_backend.state_for_request(request_id).v_payload
    finally:
        for buffer in reversed(buffers):
            free(buffer)
        projector.close()
        host_backend.close()
        device_backend.close()
