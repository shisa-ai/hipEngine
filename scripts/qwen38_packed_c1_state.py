"""Exact resident-state isolation checks for deferred packed C1 verification.

This checks publication before acceptance, not numerical equivalence of the
selected state, provider repair, or full lifecycle qualification.
"""
from __future__ import annotations

import hashlib
from typing import Any
from types import SimpleNamespace

import numpy as np

from scripts.gguf_packed_ar_state_oracle import _device_hash


def snapshot_committed_state(session: Any) -> dict[str, Any]:
    """Hash committed buffers and record destinations; ignore uncommitted KV tails."""
    scratch = session.scratch
    cfg = session.runner.weights.config
    position = int(session.position)
    if position < 0:
        raise ValueError("negative committed position")
    session.runtime.device_synchronize()
    buffers = {}

    def capture(name, buffer, nbytes=None):
        if buffer is None or int(buffer.nbytes) <= 0 or int(buffer.ptr) <= 0:
            raise ValueError(f"missing or empty committed buffer: {name}")
        size = int(buffer.nbytes) if nbytes is None else int(nbytes)
        if not 0 <= size <= int(buffer.nbytes):
            raise ValueError(f"committed buffer too short: {name}")
        buffers[name] = {
            "ptr": int(buffer.ptr), "allocation_nbytes": int(buffer.nbytes),
            "checked_nbytes": size,
            "blake2b_128": _device_hash(session, buffer, nbytes=nbytes),
        }

    for layer, (conv, recurrent) in enumerate(zip(
        scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True,
    )):
        if conv is None and recurrent is None:
            continue
        capture(f"conv:{layer}", conv)
        capture(f"recurrent:{layer}", recurrent)
    from scripts.qwen38_packed_c1_kv import resident_rows, hash_rows
    rows = resident_rows(session, 0, position)
    row_bytes = int(cfg.head_count_kv) * int(cfg.key_length) * 2

    def capture_kv(name, buffer):
        hashes = hash_rows(session, buffer, rows, row_bytes)
        buffers[name] = {
            "ptr": int(buffer.ptr), "allocation_nbytes": int(buffer.nbytes),
            "checked_nbytes": len(rows) * row_bytes,
            "physical_rows": rows, "blake2b_128": hashes,
        }
    for layer, (key, value) in enumerate(zip(
        scratch.full_key_caches, scratch.full_value_caches, strict=True,
    )):
        if key is None and value is None:
            continue
        capture_kv(f"key:{layer}", key)
        capture_kv(f"value:{layer}", value)
    capture("hidden_seed", scratch.hidden_seed_fp32)
    capture("position_device", scratch.position_buf)
    capture("context_device", scratch.context_buf)
    return {
        "position": position,
        "position_host": np.asarray(scratch.position_host).tolist(),
        "context_host": np.asarray(scratch.context_host).tolist(),
        "buffers": buffers,
    }


def selected_prefix(tokens, top1, remaining: int) -> int:
    """Independent greedy chain selection, reserving one correction/bonus token."""
    if not tokens or len(tokens) != len(top1) or remaining < 1:
        raise ValueError("invalid chain or remaining decode budget")
    accepted = 0
    while accepted < min(len(tokens) - 1, remaining - 1):
        if int(tokens[accepted + 1]) != int(top1[accepted]):
            break
        accepted += 1
    return accepted


def selected_state_sources(owner, session, *, selected_row: int) -> dict:
    """Hash a CPU-selected source row before the real device commit runs."""
    if selected_row < 0:
        raise ValueError("negative selected row")
    owner.runtime.device_synchronize()
    result = {}

    def capture(name, source, destination):
        if source is None or destination is None or int(destination.nbytes) <= 0:
            raise ValueError(f"missing selected state: {name}")
        size = int(destination.nbytes)
        offset = selected_row * size
        if offset + size > int(source.nbytes):
            raise ValueError(f"selected row exceeds source: {name}")
        view = SimpleNamespace(ptr=int(source.ptr) + offset, nbytes=size)
        result[name] = dict(ptr=int(destination.ptr), nbytes=size,
                            hash=_device_hash(owner, view))

    for layer, (conv, recurrent) in enumerate(zip(
        session.scratch.layer_conv_states, session.scratch.layer_recurrent_states, strict=True,
    )):
        if conv is None and recurrent is None:
            continue
        pair = owner._verify_linear_state_row_pair(layer)
        if pair is None:
            raise ValueError(f"missing selected linear rows: {layer}")
        capture(f"conv:{layer}", pair[0], conv)
        capture(f"recurrent:{layer}", pair[1], recurrent)
    capture("hidden_seed", owner._verify_hidden_seed_buf, session.scratch.hidden_seed_fp32)
    return result


def selected_aux_sources(session, result, *, accepted: int) -> dict:
    """Select the local BF16 provider row and independently encode next cursors."""
    from hipengine.core import DType

    source = result.pre_output_norm_hidden
    hidden = int(session.runner.hidden_size)
    if (source is None or source.dtype != DType.BF16 or len(source.shape) != 2
            or source.shape[1] != hidden or not 0 <= accepted < source.shape[0]):
        raise ValueError("invalid provider hidden source or selected row")
    size = hidden * 2
    destination = session._hidden_a
    if destination is None or int(destination.nbytes) < size:
        raise ValueError("invalid provider hidden destination")
    session.runtime.device_synchronize()
    view = SimpleNamespace(ptr=int(source.ptr) + accepted * size, nbytes=size)
    expected = {"provider_hidden": dict(ptr=int(destination.ptr), nbytes=size,
                                        hash=_device_hash(session, view))}
    if int(destination.nbytes) > size:
        tail = SimpleNamespace(ptr=int(destination.ptr) + size, nbytes=int(destination.nbytes) - size)
        expected["provider_tail"] = dict(ptr=tail.ptr, nbytes=tail.nbytes, hash=_device_hash(session, tail))
    for name, buffer, advance in (("position_device", session.scratch.position_buf, 1),
                                   ("context_device", session.scratch.context_buf, 2)):
        if buffer is None or int(buffer.nbytes) != 8:
            raise ValueError(f"invalid cursor destination: {name}")
        raw = np.array([int(result.start_position) + accepted + advance], dtype=np.int64)
        expected[name] = dict(ptr=int(buffer.ptr), nbytes=8,
                             hash=hashlib.blake2b(raw.tobytes(), digest_size=16).hexdigest())
    return expected


def assert_aux_commit(session, expected: dict) -> None:
    """Check cursor values, BF16 provider bytes, and the published provider pointer."""
    session.runtime.device_synchronize()
    hidden = session._hidden_a
    row = expected["provider_hidden"]
    tail = expected.get("provider_tail")
    allocation_size = row["nbytes"] + (tail["nbytes"] if tail else 0)
    if hidden is None or int(hidden.ptr) != row["ptr"] or int(hidden.nbytes) != allocation_size:
        raise ValueError("selected provider allocation changed")
    if tail:
        view = SimpleNamespace(ptr=int(hidden.ptr) + row["nbytes"], nbytes=tail["nbytes"])
        if view.ptr != tail["ptr"] or _device_hash(session, view) != tail["hash"]:
            raise ValueError("selected provider tail changed")
    hidden_row = SimpleNamespace(ptr=int(hidden.ptr), nbytes=row["nbytes"])
    for name, buffer in (("provider_hidden", hidden_row),
                          ("position_device", session.scratch.position_buf),
                          ("context_device", session.scratch.context_buf)):
        row = expected[name]
        if (buffer is None or int(buffer.ptr) != row["ptr"] or int(buffer.nbytes) != row["nbytes"]
                or _device_hash(session, buffer) != row["hash"]):
            raise ValueError(f"selected auxiliary commit mismatch: {name}")
    if int(session._last_target_hidden_ptr) != expected["provider_hidden"]["ptr"]:
        raise ValueError("selected provider hidden pointer was not published")


def assert_committed_state_unchanged(before: dict, after: dict) -> None:
    for field in sorted(set(before) | set(after)):
        if before.get(field) != after.get(field):
            raise ValueError(f"deferred packed verifier mutated committed {field}")
