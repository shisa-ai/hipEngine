"""P1 clearing command: does an integer window bound bind under CompactVarlen?

P1's sweep measured integer ``window_left`` / ``window_right`` only in the
non-varlen launch and found them top-left aligned. The parity test exercises
the varlen launch but only with both ``WindowValue`` sentinels, which carry no
bound and therefore reproduce plain causal. This probe runs the one cell
neither has touched: **integer bounds under ``CompactVarlen``**, at the
discriminating shape ``rows=512 keys=1536 window=128`` (``keys > rows`` so the
alignment frames differ, and a window narrower than the block so it binds).

Scoring is against two references rather than an expected answer:

1. **plain causal** -- if the flash output matches this, the integer bounds
   were ignored, i.e. the window did not bind. That is the third
   pre-registered outcome: the cell does not discriminate.
2. **bottom-right windowed causal** -- matching this is what would let P1's
   reverted plumbing be restored, and only after this flip is observed.

Anything matching neither is reported as such rather than coerced.

    .venv/bin/python scripts/gemma4_p1_window_alignment_probe.py
"""

from __future__ import annotations

import numpy as np

_ROWS = 512
_KEYS = 1536
_WINDOW = 128
_HEADS = 16
_KV_HEADS = 8
_HEAD_DIM = 256
_TOLERANCE = {"rtol": 2.0e-2, "atol": 1.0e-2}


def _bf16(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    )


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float32)


def _query_positions() -> np.ndarray:
    """Absolute position of each query row: the block is a suffix of the keys."""
    return np.arange(_ROWS, dtype=np.int64) + (_KEYS - _ROWS)


def _mask_causal() -> np.ndarray:
    queries = _query_positions()[:, None]
    keys = np.arange(_KEYS, dtype=np.int64)[None, :]
    return np.ascontiguousarray((keys <= queries).astype(np.uint8))


def _mask_bottom_right_window() -> np.ndarray:
    """Sliding window over absolute positions: keys in [q - W + 1, q]."""
    queries = _query_positions()[:, None]
    keys = np.arange(_KEYS, dtype=np.int64)[None, :]
    keep = (keys <= queries) & (keys > queries - _WINDOW)
    return np.ascontiguousarray(keep.astype(np.uint8))


def _mask_block_local_causal() -> np.ndarray:
    """Causal in block-local coordinates: key j <= query row i."""
    rows = np.arange(_ROWS, dtype=np.int64)[:, None]
    keys = np.arange(_KEYS, dtype=np.int64)[None, :]
    return np.ascontiguousarray((keys <= rows).astype(np.uint8))


def _mask_causal_window_left_block_local() -> np.ndarray:
    """Absolute causal with only the window's *left* edge block-local.

    Included as a near-miss control: it shares P1's reading of the left edge
    but omits the block-local right bound, so it must NOT match if the right
    bound is genuinely being cut at ``i``.
    """
    queries = _query_positions()[:, None]
    rows = np.arange(_ROWS, dtype=np.int64)[:, None]
    keys = np.arange(_KEYS, dtype=np.int64)[None, :]
    keep = (keys <= queries) & (keys > rows - _WINDOW)
    return np.ascontiguousarray(keep.astype(np.uint8))


def _mask_top_left_window() -> np.ndarray:
    """Both window bounds in block-local query coordinates: ``i - W + 1 <= j <= i``.

    P1's reading of the measured behaviour, now reproduced under
    ``CompactVarlen``: the integer bounds are interpreted relative to row ``i``
    of the *query block*, so a suffix block is cut at ``i`` instead of at
    ``start + i``. The absolute-causal conjunct is redundant here because
    ``i < start``, but it is kept so the mask states the full intent.
    """
    queries = _query_positions()[:, None]
    rows = np.arange(_ROWS, dtype=np.int64)[:, None]
    keys = np.arange(_KEYS, dtype=np.int64)[None, :]
    keep = (keys <= queries) & (keys > rows - _WINDOW) & (keys <= rows)
    return np.ascontiguousarray(keep.astype(np.uint8))


def _exact(mask: np.ndarray, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch,
        gemma4_attention_prefill_bf16,
    )

    runtime = get_hip_runtime()
    buffers = []
    scratch = Gemma4AttentionScratch()
    try:

        def upload(array: np.ndarray) -> object:
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
            return buf

        q_buf, k_buf, v_buf, m_buf = upload(query), upload(key), upload(value), upload(mask)
        out = np.zeros((_ROWS, _HEADS, _HEAD_DIM), dtype=np.uint16)
        out_buf = upload(out)
        gemma4_attention_prefill_bf16(
            q_buf.ptr, k_buf.ptr, v_buf.ptr, m_buf.ptr, out_buf.ptr,
            tokens=_ROWS, keys=_KEYS, num_heads=_HEADS, num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM, scale=1.0, scratch=scratch, runtime=runtime,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes, runtime=runtime)
        return _bf16_to_f32(out)
    finally:
        scratch.close()
        for buf in buffers:
            free(buf, runtime=runtime)


def _flash(query, key, value, **window):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch,
        gemma4_attention_prefill_aotriton,
    )

    runtime = get_hip_runtime()
    buffers = []
    scratch = Gemma4AttentionScratch()
    try:

        def upload(array: np.ndarray) -> object:
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
            return buf

        q_buf, k_buf, v_buf = upload(query), upload(key), upload(value)
        out = np.zeros((_ROWS, _HEADS, _HEAD_DIM), dtype=np.uint16)
        out_buf = upload(out)
        gemma4_attention_prefill_aotriton(
            q_buf.ptr, k_buf.ptr, v_buf.ptr, out_buf.ptr,
            tokens=_ROWS, keys=_KEYS, num_heads=_HEADS, num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM, scale=1.0, scratch=scratch, runtime=runtime, **window,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes, runtime=runtime)
        return _bf16_to_f32(out)
    finally:
        scratch.close()
        for buf in buffers:
            free(buf, runtime=runtime)


def _score(label: str, candidate: np.ndarray, reference: np.ndarray) -> dict:
    diff = np.abs(candidate - reference)
    return {
        "label": label,
        "maxabs": float(diff.max()),
        "meanabs": float(diff.mean()),
        "match": bool(np.allclose(candidate, reference, **_TOLERANCE)),
    }


def main() -> int:
    rng = np.random.default_rng(5)
    query = _bf16(rng.normal(0.0, 0.5, (_ROWS, _HEADS, _HEAD_DIM)))
    key = _bf16(rng.normal(0.0, 0.5, (_KEYS, _KV_HEADS, _HEAD_DIM)))
    value = _bf16(rng.normal(0.0, 0.5, (_KEYS, _KV_HEADS, _HEAD_DIM)))

    print(f"shape: rows={_ROWS} keys={_KEYS} window={_WINDOW} (suffix block start={_KEYS - _ROWS})")

    ref_causal = _exact(_mask_causal(), query, key, value)
    ref_window = _exact(_mask_bottom_right_window(), query, key, value)
    ref_block_causal = _exact(_mask_block_local_causal(), query, key, value)
    ref_top_left = _exact(_mask_top_left_window(), query, key, value)
    ref_near_miss = _exact(_mask_causal_window_left_block_local(), query, key, value)
    # The references must actually differ, or the discrimination is vacuous.
    ref_delta = _score("reference-delta", ref_causal, ref_window)
    print(f"references differ: maxabs={ref_delta['maxabs']:.6f} match={ref_delta['match']}")
    if ref_delta["match"]:
        print("ABORT: the two references are identical, so this shape cannot discriminate")
        return 2

    baseline = _flash(query, key, value)
    windowed = _flash(
        query, key, value, window_left=_WINDOW - 1, window_right=0
    )

    rows_out = [
        _score("baseline(sentinels) vs causal", baseline, ref_causal),
        _score("integer-window vs causal", windowed, ref_causal),
        _score("integer-window vs bottom-right window", windowed, ref_window),
        _score("integer-window vs block-local causal", windowed, ref_block_causal),
        _score("integer-window vs TOP-LEFT window", windowed, ref_top_left),
        _score("integer-window vs left-edge-only control", windowed, ref_near_miss),
    ]

    print()
    print(f"{'comparison':44s} {'maxabs':>12s} {'meanabs':>12s}  match")
    for row in rows_out:
        print(f"{row['label']:44s} {row['maxabs']:12.6f} {row['meanabs']:12.6f}  {row['match']}")

    causal_match = rows_out[1]["match"]
    matches = [r for r in rows_out[2:] if r["match"]]
    print()
    if causal_match:
        print("RESULT: non-binding -- the integer bounds were ignored and the output")
        print("        is plain causal. Third pre-registered outcome: the cell does")
        print("        not discriminate, and P1 closes on coverage.")
        return 1
    if matches:
        hit = matches[0]["label"]
        if "bottom-right window" in hit:
            print("RESULT: bottom-right -- integer bounds bind bottom-right under")
            print("        CompactVarlen. P1's reverted plumbing may be restored, after")
            print("        this flip is observed in the parity test as well.")
            return 0
        print(f"RESULT: windowed but NOT bottom-right -- exact match: {hit}")
        print("        The alignment does not flip under varlen: the integer bounds")
        print("        still cut a suffix block at block-local i rather than start + i.")
        print("        P1's blocker is confirmed structural, and P2 option (a) is this")
        print("        gap's only route.")
        return 4
    print("RESULT: matches no candidate. The candidate set is still incomplete;")
    print("        do not coerce this into a verdict.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())