"""Cold pre-compaction state needed to extend a DMS prompt without replay.

Compact decode KV alone cannot reproduce dense prompt attention or recompute
full-prompt exact-budget selection. These host-owned planes are cache resources,
not a second device decode store, and must be accounted separately.
"""
from dataclasses import dataclass

import numpy as np

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_array_to_device, host_array_ptr


def read_prefix(buffer, nbytes):
    if nbytes > buffer.nbytes:
        raise ValueError("DMS cold prefix exceeds source buffer")
    data = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(data), DeviceBuffer(buffer.ptr, nbytes), nbytes)
    data.flags.writeable = False
    return data


def write_prefix(buffer, data):
    if data.nbytes > buffer.nbytes:
        raise ValueError("DMS cold prefix exceeds destination buffer")
    copy_host_array_to_device(DeviceBuffer(buffer.ptr, data.nbytes), data, data.nbytes)


@dataclass(frozen=True)
class DMSColdPrefill:
    tokens: tuple[int, ...]
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    predictor_logits: np.ndarray | None
    predictor_decisions: np.ndarray
    hidden: np.ndarray
    boundaries: dict[int, tuple[np.ndarray, ...]]

    @property
    def nbytes(self):
        return (sum(a.nbytes for pair in self.layers for a in pair)
                + self.predictor_decisions.nbytes + self.hidden.nbytes
                + sum(a.nbytes for values in self.boundaries.values() for a in values)
                + (0 if self.predictor_logits is None else self.predictor_logits.nbytes))

    def reused_tokens(self, tokens):
        return len(self.tokens) if (len(tokens) > len(self.tokens)
                                    and tuple(tokens[:len(self.tokens)]) == self.tokens) else 0


def capture_cold_prefill(session, collector, tokens):
    config = session._dms_source.config
    nbytes = len(tokens) * config.num_kv_heads * config.head_dim * 2
    # dense_pool owns one contiguous prompt plane per physical attention layer.
    if session.dms_prefill_mode == "layer_outer":
        layers = tuple(session._dms_cache_layers[layer] for layer in config.physical_layer_ids)
    else:
        layers = tuple(tuple(read_prefix(buffer, nbytes) for buffer in session.scratch.full_cache(layer))
                       for layer in config.physical_layer_ids)
    shape = (config.num_layers, len(tokens), config.num_kv_heads)
    decisions = read_prefix(collector._decisions, int(np.prod(shape))).reshape(shape)
    logits = (None if collector._logits is None else
              read_prefix(collector._logits, int(np.prod(shape)) * 4).view(np.float32).reshape(shape))
    return DMSColdPrefill(tuple(tokens), layers, logits, decisions,
                          session._dms_cache_hidden.copy(),
                          completed_boundaries(session))


def cache_boundary_points(session):
    tokens = getattr(session, "_dms_cache_capture_tokens", None)
    if tokens is None:
        return ()
    state_bytes = sum(buffer.nbytes for pair in zip(session.scratch.layer_conv_states,
                                                    session.scratch.layer_recurrent_states)
                      for buffer in pair if buffer is not None)
    budget = int(getattr(session, "_dms_cache_byte_budget", 1 << 30)) // 2
    count = min(4, budget // max(1, state_bytes))
    blocks = (len(tokens) - 1) // 256
    fallback = (np.linspace(1, blocks, min(count, blocks), dtype=int) * 256).tolist()
    # Layer-outer prefill captures recurrence only at natural chunk ends.
    # Arbitrary block-grid points otherwise produce no restorable state at all.
    chunk_size = session._linear_prefill_layer_chunk_size(len(tokens))
    hidden = getattr(session, "_prefill_hidden_a", None)
    if hidden is not None and hidden.nbytes // (session.runner.hidden_size * 2) < len(tokens):
        # Chunk-outer execution uses the shared scratch width, not the
        # layer-outer linear chunk setting (these can differ substantially).
        chunk_size = min(session._prefill_scratch_rows(len(tokens)),
                         int(session._bulk_prefill_scratch.rows))
    reuse = getattr(session, "_dms_prefill_reuse", None)
    start = 0 if reuse is None else len(reuse.tokens)
    minimum = int(session.runner.weights.config.ssm_conv_kernel)
    from hipengine.runtime.qwen35_gguf_runner import _chunk_ranges
    remaining = len(tokens) - start
    ranges = (_chunk_ranges(remaining, chunk_size, min_chunk_size=minimum)
              if remaining > 0 else ())
    natural = [start + end for _, end in ranges if start + end < len(tokens)]
    if reuse is not None:
        natural = sorted({*natural, start, *reuse.cold.boundaries})
    selected = [natural[i] for i in np.linspace(0, len(natural) - 1,
                                               min(count, len(natural)), dtype=int)]
    selected.extend(end for end in fallback if end not in selected)
    return tuple(sorted(selected[:count]))


def capture_recurrent_boundary(session, layer, end):
    if end not in cache_boundary_points(session):
        return
    states = session.__dict__.setdefault("_dms_cache_boundaries", {})
    boundary = states.setdefault(end, {})
    boundary[layer] = tuple(read_prefix(buffer, buffer.nbytes) for buffer in (
        session.scratch.layer_conv_states[layer], session.scratch.layer_recurrent_states[layer]
    ) if buffer is not None)


def completed_boundaries(session):
    linear_layers = [i for i, pair in enumerate(zip(session.scratch.layer_conv_states,
                                                   session.scratch.layer_recurrent_states))
                     if any(buffer is not None for buffer in pair)]
    result = {end: tuple(array for layer in linear_layers for array in states[layer])
              for end, states in getattr(session, "_dms_cache_boundaries", {}).items()
              if all(layer in states for layer in linear_layers)}
    reuse = getattr(session, "_dms_prefill_reuse", None)
    if reuse is not None:
        selected = cache_boundary_points(session)
        result.update({end: arrays for end, arrays in reuse.cold.boundaries.items() if end in selected})
        if len(reuse.tokens) in selected:
            count = sum(len(tuple(buffer for buffer in pair if buffer is not None))
                        for pair in zip(session.scratch.layer_conv_states,
                                        session.scratch.layer_recurrent_states))
            result[len(reuse.tokens)] = reuse.recurrent_and_logits[:count]
    return result


def capture_hidden_chunk(session, ptr, start, rows, total):
    """Own raw trunk rows so an AR source can prime NextN without target replay."""
    if getattr(session, "_dms_cache_capture_tokens", None) is None:
        return
    width = int(session.runner.hidden_size) * 2
    if start == 0 or getattr(session, "_dms_cache_hidden", None) is None:
        session._dms_cache_hidden = np.empty((total, width), dtype=np.uint8)
        reuse = getattr(session, "_dms_prefill_reuse", None)
        if reuse is not None:
            session._dms_cache_hidden[:len(reuse.tokens)] = reuse.cold.hidden
    session._dms_cache_hidden[start:start + rows] = read_prefix(
        DeviceBuffer(ptr, rows * width), rows * width
    ).reshape(rows, width)


def prime_cached_hidden(session, snapshot, sink, request_id):
    """Feed a fresh NextN sink from cached target rows, without target prefill."""
    from hipengine.core.memory import malloc, free
    data = np.ascontiguousarray(snapshot.cold.hidden)
    buffer = malloc(data.nbytes, runtime=session.runtime)
    try:
        write_prefix(buffer, data)
        sink.consume(request_id=request_id, chunk_start=0, hidden_ptr=buffer.ptr,
                     rows=len(snapshot.tokens), stream=0)
        session.runtime.device_synchronize()
    finally:
        free(buffer, runtime=session.runtime)


def seed_collector(collector, cold):
    """Seed raw prefix scores; finalize reselects over the combined prompt."""
    count = len(cold.tokens)
    heads = collector.num_kv_heads
    if count >= collector.token_count:
        raise ValueError("DMS collector seed requires a nonempty suffix")
    for layer in range(len(cold.layers)):
        offset = layer * collector.token_count * heads
        data = np.ascontiguousarray(cold.predictor_decisions[layer])
        write_prefix(DeviceBuffer(collector._decisions.ptr + offset, data.nbytes), data)
        if cold.predictor_logits is not None:
            if collector._logits is None:
                raise ValueError("DMS collector lacks cached predictor score storage")
            data = np.ascontiguousarray(cold.predictor_logits[layer])
            write_prefix(DeviceBuffer(collector._logits.ptr + offset * 4, data.nbytes), data)
    collector._next[:] = count
