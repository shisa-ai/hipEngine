"""Owned DMS prompt and block-boundary snapshots for resident reuse.

Complete hits restore compact KV. Extensions and branches restore cold dense
prompt KV and recurrence, then reselect retention over the combined prompt.
"""
from dataclasses import dataclass, replace
from collections import OrderedDict
from types import SimpleNamespace
import copy

import numpy as np

from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, host_array_ptr
from hipengine.kvcache.dms import create_dms_bf16_backend
from hipengine.kvcache.dms_prefix import DMSPrefixSnapshot, capture_prefix, restore_prefix


class DMSPrefixCache:
    """LRU of independently owned prompt boundaries, bounded by host bytes."""

    def __init__(self, max_bytes, max_entries=16):
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.entries = OrderedDict()

    @property
    def nbytes(self):
        return sum(snapshot.nbytes for snapshot in self.entries.values())

    def lookup(self, session, tokens, sampled):
        candidates = []
        for source in self.entries.values():
            if not source.compatible(session):
                continue
            if source.matches(session, tokens, sampled) or (
                source.cold is not None and source.cold.reused_tokens(tokens)
            ):
                candidates.append((source, source))
                continue
            if source.cold is None:
                continue
            common = 0
            for a, b in zip(source.tokens, tokens):
                if a != b:
                    break
                common += 1
            points = [end for end in source.cold.boundaries if end <= common and end < len(tokens)]
            if points:
                end = max(points)
                width = source.cold.layers[0][0].nbytes // len(source.tokens)
                cold = replace(source.cold, tokens=source.tokens[:end],
                               layers=tuple(tuple(a[:end * width] for a in pair)
                                            for pair in source.cold.layers),
                               predictor_logits=(None if source.cold.predictor_logits is None else
                                                 source.cold.predictor_logits[:, :end]),
                               predictor_decisions=source.cold.predictor_decisions[:, :end],
                               hidden=source.cold.hidden[:end], boundaries={})
                prefix = replace(source, tokens=source.tokens[:end], compact=None,
                                 recurrent_and_logits=source.cold.boundaries[end],
                                 result=None, cold=cold, complete=False,
                                 checkpoint_key=(source.checkpoint_key[0], end))
                candidates.append((source, prefix))
        if not candidates:
            return None
        source, snapshot = max(candidates, key=lambda item: len(item[1].tokens))
        self.entries.move_to_end(source.tokens)
        return snapshot

    def put(self, snapshot):
        evicted = []
        previous = self.entries.pop(snapshot.tokens, None)
        if previous is not None:
            evicted.append(previous)
        if snapshot.nbytes > self.max_bytes:
            evicted.append(snapshot)
            snapshot = _fitting_prefix(snapshot, self.max_bytes)
        if snapshot is not None:
            previous = self.entries.pop(snapshot.tokens, None)
            if previous is not None:
                evicted.append(previous)
            self.entries[snapshot.tokens] = snapshot
        while self.entries and (self.nbytes > self.max_bytes or len(self.entries) > self.max_entries):
            evicted.append(self.entries.popitem(last=False)[1])
        return tuple(evicted)


def _fitting_prefix(source, budget):
    """Own a restorable boundary when the complete prompt exceeds the budget.

    Views are useful for sizing candidates but cannot be retained: their bases
    would keep the entire oversized prompt alive while underreporting its bytes.
    Keep earlier boundaries too, so sibling requests can still share a prefix.
    """
    if source.cold is None:
        return None
    for end in sorted(source.cold.boundaries, reverse=True):
        cold = source.cold
        layers = tuple(tuple(a[:end * (a.size // len(source.tokens))] for a in pair)
                       for pair in cold.layers)
        candidate = replace(
            cold, tokens=source.tokens[:end], layers=layers,
            predictor_logits=None if cold.predictor_logits is None else cold.predictor_logits[:, :end],
            predictor_decisions=cold.predictor_decisions[:, :end], hidden=cold.hidden[:end],
            boundaries={n: arrays for n, arrays in cold.boundaries.items() if n < end},
        )
        recurrent = cold.boundaries[end]
        if candidate.nbytes + sum(a.nbytes for a in recurrent) > budget:
            continue

        def own(array):
            result = array.copy()
            result.flags.writeable = False
            return result

        candidate = replace(
            candidate, layers=tuple(tuple(own(a) for a in pair) for pair in candidate.layers),
            predictor_logits=None if candidate.predictor_logits is None else own(candidate.predictor_logits),
            predictor_decisions=own(candidate.predictor_decisions), hidden=own(candidate.hidden),
            boundaries={n: tuple(own(a) for a in arrays) for n, arrays in candidate.boundaries.items()},
        )
        return replace(source, tokens=candidate.tokens, compact=None, result=None,
                       recurrent_and_logits=tuple(own(a) for a in recurrent), cold=candidate,
                       complete=False, checkpoint_key=(source.checkpoint_key[0], end))
    return None


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
    compact: DMSPrefixSnapshot | None
    recurrent_and_logits: tuple[np.ndarray, ...]
    result: object
    cold: object | None = None
    checkpoint_key: tuple[int, ...] = ()
    complete: bool = True

    @property
    def nbytes(self):
        return ((0 if self.compact is None else self.compact.nbytes) + sum(a.nbytes for a in self.recurrent_and_logits)
                + int(getattr(getattr(self.result, "logits", None), "nbytes", 0))
                + (0 if self.cold is None else self.cold.nbytes))

    def compatible(self, session):
        return (self.runner is session.runner and self.backend_factory is session.dms_backend_factory
                and self.policy == session._dms_source.config.fingerprint)

    def matches(self, session, tokens, sampled):
        return (self.complete and self.compatible(session)
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
                             tuple(arrays), copy.deepcopy(result),
                             getattr(session, "_dms_cold_prefill", None),
                             (getattr(session, "_dms_cache_identity", id(session)), len(tokens)))


def restore_recurrent_prefix(session, snapshot):
    """Restore target recurrence before computing only an uncached suffix."""
    buffers = _buffers(session, False)
    for buffer, data in zip(buffers, snapshot.recurrent_and_logits[:len(buffers)], strict=True):
        if buffer.nbytes != data.nbytes:
            raise ValueError("DMS prefix recurrent buffer geometry mismatch")
        copy_host_array_to_device(buffer, data, data.nbytes)


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
            copy_host_array_to_device(buffer, data, data.nbytes)
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
