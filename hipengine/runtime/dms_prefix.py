"""Owned full-prompt DMS snapshots for resident request reuse.

A pruned prompt cannot stand in for a shorter boundary: exact-budget selection
and dense prompt attention depend on the complete prompt. Cache lookup therefore
requires equal prompt tokens, not just a common radix prefix.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import copy

import numpy as np

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, host_array_ptr
from hipengine.kvcache.dms import create_dms_bf16_backend
from hipengine.kvcache.dms_prefix import DMSPrefixSnapshot, capture_prefix, restore_prefix


def _buffers(session, include_logits):
    scratch = session.scratch
    result = [buffer for pair in zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
              for buffer in pair if buffer is not None]
    if include_logits:
        if session._logits_buf is None:
            raise RuntimeError("DMS prefix sampling snapshot requires resident logits")
        result.append(session._logits_buf)
    return result


@dataclass(frozen=True)
class DMSPromptSnapshot:
    runner: object
    backend_factory: object
    policy: str
    tokens: tuple[int, ...]
    sampled: bool
    compact: DMSPrefixSnapshot
    recurrent_and_logits: tuple[np.ndarray, ...]
    result: object

    @property
    def nbytes(self):
        return (self.compact.nbytes + sum(a.nbytes for a in self.recurrent_and_logits)
                + int(getattr(getattr(self.result, "logits", None), "nbytes", 0)))

    def matches(self, session, tokens, sampled):
        return (self.runner is session.runner and self.backend_factory is session.dms_backend_factory
                and self.policy == session._dms_source.config.fingerprint
                and self.tokens == tuple(tokens) and self.sampled == sampled)


def capture_prompt(session, tokens, result, *, sampled):
    if session.position != len(tokens):
        raise ValueError("DMS prompt snapshot must be captured at the prompt boundary")
    arrays = []
    for buffer in _buffers(session, sampled):
        data = np.empty(buffer.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(data), buffer, data.nbytes)
        data.flags.writeable = False
        arrays.append(data)
    return DMSPromptSnapshot(session.runner, session.dms_backend_factory,
                             session._dms_source.config.fingerprint,
                             tuple(tokens), sampled, capture_prefix(session._dms_backend, 0),
                             tuple(arrays), copy.deepcopy(result))


def restore_prompt(session, snapshot):
    if not snapshot.matches(session, snapshot.tokens, snapshot.sampled):
        raise ValueError("DMS prompt snapshot model or policy changed")
    if session.position != 0 or session._dms_backend is not None:
        raise ValueError("DMS prompt restore requires a fresh session")
    if len(snapshot.tokens) >= session.scratch.max_positions:
        raise ValueError("DMS prompt snapshot leaves no continuation capacity")
    buffers = _buffers(session, snapshot.sampled)
    if len(buffers) != len(snapshot.recurrent_and_logits) or any(
        b.nbytes != a.nbytes for b, a in zip(buffers, snapshot.recurrent_and_logits)
    ):
        raise ValueError("DMS prefix recurrent/logit buffer geometry mismatch")
    config = session._dms_source.config
    capacity = max(1, int(snapshot.compact.live_counts.max())) + int(session.dms_max_new_tokens)
    factory = session.dms_backend_factory or create_dms_bf16_backend
    backend = factory(retrofit=config, slots_per_layer=config.num_kv_heads * capacity,
                      max_request_rows=1, max_pack_rows=max(1, min(len(snapshot.tokens), 4096)),
                      device_payloads=True, device_backend=session.backend)
    try:
        request = SimpleNamespace(request_id=0, prompt_tokens=(), max_new_tokens=session.dms_max_new_tokens)
        backend.reserve(backend.estimate(request, None, {
            "kind": "admission", "tokens": 0, "max_new_tokens": session.dms_max_new_tokens,
            "per_head_slots": capacity, "logical_prompt_tokens": len(snapshot.tokens),
        }))
        restore_prefix(backend, 0, snapshot.compact)
        for buffer, data in zip(buffers, snapshot.recurrent_and_logits, strict=True):
            copy_host_to_device(buffer, host_array_ptr(data), data.nbytes)
        pool = session._dms_dense_prefill_pool
        if pool is not None:
            allocation = session.unbind_device_kv_allocation()
            pool.release(allocation.request_id)
            pool.close()
            session._dms_dense_prefill_pool = None
    except BaseException:
        backend.close()
        raise
    session._dms_backend = backend
    session.runner.__dict__["_dms_decode_owner"] = session
    session._position = len(snapshot.tokens)
    return copy.deepcopy(snapshot.result)
