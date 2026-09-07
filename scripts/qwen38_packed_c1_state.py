"""Exact resident-state isolation checks for deferred packed C1 verification.

This checks publication before acceptance, not numerical equivalence of the
selected state, provider repair, or full lifecycle qualification.
"""
from __future__ import annotations

from typing import Any

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
    live_nbytes = position * int(cfg.head_count_kv) * int(cfg.key_length) * 2
    for layer, (key, value) in enumerate(zip(
        scratch.full_key_caches, scratch.full_value_caches, strict=True,
    )):
        if key is None and value is None:
            continue
        capture(f"key:{layer}", key, live_nbytes)
        capture(f"value:{layer}", value, live_nbytes)
    capture("hidden_seed", scratch.hidden_seed_fp32)
    capture("position_device", scratch.position_buf)
    capture("context_device", scratch.context_buf)
    return {
        "position": position,
        "position_host": np.asarray(scratch.position_host).tolist(),
        "context_host": np.asarray(scratch.context_host).tolist(),
        "buffers": buffers,
    }


def assert_committed_state_unchanged(before: dict, after: dict) -> None:
    for field in sorted(set(before) | set(after)):
        if before.get(field) != after.get(field):
            raise ValueError(f"deferred packed verifier mutated committed {field}")
