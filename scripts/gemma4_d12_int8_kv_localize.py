#!/usr/bin/env python3
"""Gemma 4 D12: localize the INT8-vs-BF16 KV numerical difference at attention.

This is a bounded diagnostic, not a promotion and not a performance campaign.
The committed screening harness
(``scripts/gemma4_int8_kv_teacher_forced_eval.py``) established a deterministic
full-model numerical difference between the BF16 KV arm and the
``int8_per_token_head`` arm on the real UD-Q4_K_XL artifact. It did not localize
the cause. This unit does three things, on the already-frozen short English
token chain (``prose_en_short``: 96 tokens, prefill 63, first scored decode
position 63):

1. **Whole-model boundary localization.** Run both arms for the single forced
   decode position and capture, at the real attention boundary of every layer,
   the layer's hidden input and its BF16 query / written K / written V / context
   output. The first layer whose attention *output* diverges while its *input* is
   still identical is where the difference is injected; the first layer whose
   input diverges is where it has propagated.

2. **Fixed-input consumer control.** At the selected layers replay one captured
   common boundary input -- the BF16 arm's prefill + decode K/V and the decode
   query -- through (a) the production BF16 attention output, (b) the production
   INT8 writer plus consumer including its FP32-to-BF16 context narrowing, and
   (c) one common independent FP32 CPU attention algorithm evaluated separately
   on the original BF16 values and on ``float32(int8) * float32(stored scale)``
   over identical live spans and mask.

3. **Declared tolerances and literal controls.** Comparisons and tolerances are
   frozen in :data:`TOLERANCES` before any measurement. The frozen BF16 consumer
   tolerance is reported as it falls, not relaxed; the BF16 production output is
   additionally compared to the BF16 rounding of the CPU control by literal raw
   bytes, which is a stricter statement than equal scalar error floors.

Scope of the CPU comparison. ``common_cpu_attention`` reuses
``_attention_head_row`` and the mask loop from
``hipengine.kernels.cpu_reference.gemma4_int8``. It is therefore a **shared
function-consistency check**: it shows the GPU consumer and the CPU path agree
when both run the same function, not that an independent second oracle exists.
The quantized-oracle self-check is the same kind of statement.

If the two whole-model arms already carry different Q/K/V at a later layer,
their whole-model outputs are not a fixed-input comparison; the replay above
supplies the fixed-input comparison instead, and the distinction is recorded
explicitly rather than collapsed.

Full captured arrays and the raw log stay outside the repository under a unique
directory; only the compact artifact is written into the tree. No production
source, kernel, runtime, flag, admission or default changes. This script reads
existing registered kernels and uses temporary diagnostic device buffers; every
interception is restored and every allocation released in ``finally``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

from scripts.gemma4_teacher_forced_gate import chain_sha256, sha256_file, _provenance

#: The committed screening artifact that froze the short English chain. Reading
#: the chain from it is the point: the prompt ids and prefill must be the frozen
#: ones, not a fresh or tuned prompt.
FROZEN_ARTIFACT = Path(
    "benchmarks/results/2026-10-02-gemma4-d12-int8-kv-teacher-forced.json"
)
FROZEN_CASE = "prose_en_short"

#: This diagnostic's frozen-chain invariants, pinned independently of the artifact
#: it reads. The loader recomputes the hash from the ids and also requires it to
#: equal :data:`FROZEN_CHAIN_SHA256`, so a tampered prompt array whose supplied
#: hash was recomputed to match is still refused, and the prefill/scored_rows/
#: prompt-token metadata is pinned rather than trusted. This pins a diagnostic
#: input invariant; it is not runtime identity admission and gates no execution.
FROZEN_CHAIN_SHA256 = "ac020238439bfd683dc7ec9a7200440b182277f7931cad3c83ceeae5a80d9e32"
FROZEN_PROMPT_TOKENS = 96
FROZEN_PREFILL = 63
FROZEN_SCORED_ROWS = 32

#: Comparisons and tolerances, frozen before measurement. ``cpu_shared_function``
#: is the common CPU algorithm against the existing quantized oracle on the
#: quantized representation (a shared-function consistency check, not an
#: independent oracle). The INT8 consumer emits FP32 (no BF16 floor), so its
#: tolerance is tight; the BF16 production output is read back BF16-narrowed, so
#: its tolerance is a frozen threshold that the BF16 readback floor can exceed.
#: These values are never relaxed after a measurement.
TOLERANCES: dict[str, float] = {
    "cpu_shared_function_self_check_rel_l2": 1e-6,
    "int8_consumer_vs_cpu_rel_l2": 1e-4,
    "int8_consumer_vs_cpu_max_abs": 1e-4,
    "bf16_consumer_vs_cpu_rel_l2": 5e-3,
    "bf16_consumer_vs_cpu_max_abs": 5e-3,
}

#: Arithmetic-order differences between the CPU control and the GPU consumers,
#: recorded so a tolerance judgement names them rather than hiding them.
ARITHMETIC_ORDER_NOTES = (
    "CPU control: keys @ q via numpy (BLAS order), np.exp softmax with a max "
    "subtraction, weighted V summed in slot order, all FP32. GPU INT8 consumer: "
    "per-warp strided partial dot products reduced by a shuffle tree, expf "
    "softmax with a block max, weighted V summed in slot order, all FP32. GPU "
    "BF16 consumer (decode kernel): the same products and stride-tree order as "
    "the exact prefill kernel, output narrowed to BF16. The CPU and GPU orders "
    "differ in reduction order and exp implementation; the frozen tolerances "
    "are evaluated as declared and may fail when BF16 rounding exceeds them."
)


@dataclass(frozen=True)
class FrozenChain:
    """The frozen short English chain and its attention geometry."""

    name: str
    prompt_ids: tuple[int, ...]
    prefill: int
    scored_rows: int
    prompt_tokens: int
    chain_sha256: str
    attention_geometry: tuple[dict[str, Any], ...]


def load_frozen_chain(
    path: Path = FROZEN_ARTIFACT, case_name: str = FROZEN_CASE
) -> FrozenChain:
    """Load the frozen chain and geometry from the committed screening artifact.

    The recorded ``chain_sha256`` is recomputed from the ids and must match, so a
    hand-edited prompt array cannot be passed off as the frozen chain. The
    recomputed hash must also equal the independently pinned
    :data:`FROZEN_CHAIN_SHA256`, and ``prompt_tokens`` / ``prefill`` /
    ``scored_rows`` must equal their pinned values, so neither a recomputed
    supplied hash nor altered prefill metadata is accepted. This is a diagnostic
    input invariant; it is not runtime identity admission.
    """

    data = json.loads(Path(path).read_text())
    workload = data.get("workload") or {}
    cases = workload.get("cases") or []
    match = next((case for case in cases if case.get("name") == case_name), None)
    if match is None:
        raise ValueError(f"artifact {path} has no case {case_name!r}")
    ids = tuple(int(token) for token in match["prompt_ids"])
    recorded = str(match.get("chain_sha256"))
    recomputed = chain_sha256(ids)
    if recorded != recomputed:
        raise ValueError(
            f"chain hash mismatch for {case_name}: recorded {recorded}, "
            f"recomputed {recomputed}"
        )
    if recomputed != FROZEN_CHAIN_SHA256:
        raise ValueError(
            f"frozen chain hash mismatch: recomputed {recomputed}, pinned "
            f"{FROZEN_CHAIN_SHA256}"
        )
    if len(ids) != FROZEN_PROMPT_TOKENS:
        raise ValueError(
            f"frozen chain prompt length {len(ids)} != {FROZEN_PROMPT_TOKENS}"
        )
    if int(match["prompt_tokens"]) != FROZEN_PROMPT_TOKENS:
        raise ValueError(
            f"frozen chain prompt_tokens {match['prompt_tokens']} != "
            f"{FROZEN_PROMPT_TOKENS}"
        )
    if int(match["prefill"]) != FROZEN_PREFILL:
        raise ValueError(
            f"frozen chain prefill {match['prefill']} != {FROZEN_PREFILL}"
        )
    if int(match["scored_rows"]) != FROZEN_SCORED_ROWS:
        raise ValueError(
            f"frozen chain scored_rows {match['scored_rows']} != {FROZEN_SCORED_ROWS}"
        )
    geometry = tuple(
        {
            "layer_type": str(entry["layer_type"]),
            "num_heads": int(entry["num_heads"]),
            "num_kv_heads": int(entry["num_kv_heads"]),
            "head_dim": int(entry["head_dim"]),
            "sliding_window": (
                None if entry.get("sliding_window") is None else int(entry["sliding_window"])
            ),
            "k_eq_v": bool(entry["k_eq_v"]),
        }
        for entry in data["attention_geometry"]
    )
    return FrozenChain(
        name=str(match["name"]),
        prompt_ids=ids,
        prefill=int(match["prefill"]),
        scored_rows=int(match["scored_rows"]),
        prompt_tokens=int(match["prompt_tokens"]),
        chain_sha256=recomputed,
        attention_geometry=geometry,
    )


def float_to_bf16_bits(values: Any) -> np.ndarray:
    """Round float values to BF16 bits with the same rule as the loader.

    Round-half-to-even, matching ``hipengine.loading.materialize``; the unit tier
    asserts equality with that implementation.
    """

    f32 = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    bits = f32.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float(bits: Any) -> np.ndarray:
    """Decode BF16 bit patterns to float32."""

    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )


def relative_l2(a: Any, b: Any) -> float:
    """Relative L2 distance ``||a - b|| / max(||a||, eps)`` over flattened arrays."""

    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(x))
    numerator = float(np.linalg.norm(x - y))
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def max_abs_diff(a: Any, b: Any) -> float:
    """Maximum absolute elementwise difference."""

    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x - y)))


def literal_comparison(a: Any, b: Any) -> dict[str, Any]:
    """Literal shape, dtype and raw-byte comparison of two arrays.

    A stricter statement than equal scalar error summaries: ``raw_bytes_equal``
    is true only when the two arrays are the same shape and dtype and their bytes
    are identical.
    """

    x = np.asarray(a)
    y = np.asarray(b)
    shape_match = x.shape == y.shape
    dtype_match = x.dtype == y.dtype
    return {
        "a_shape": list(x.shape),
        "b_shape": list(y.shape),
        "a_dtype": str(x.dtype),
        "b_dtype": str(y.dtype),
        "shape_match": bool(shape_match),
        "dtype_match": bool(dtype_match),
        "raw_bytes_equal": bool(
            shape_match and dtype_match and x.tobytes() == y.tobytes()
        ),
    }


def common_cpu_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    positions: np.ndarray,
    query_position: int,
    sliding_window: int | None,
    scale: float,
    num_kv_heads: int,
) -> np.ndarray:
    """One independent FP32 CPU attention row over gathered K/V.

    The single algorithm used for both representations in the fixed-consumer
    control. ``query`` is ``[num_heads, head_dim]``, ``keys``/``values`` are
    ``[context, num_kv_heads, head_dim]``. The keep mask and the per-head row are
    taken from the existing quantized oracle
    (``hipengine.kernels.cpu_reference.gemma4_int8``): a slot is visible when its
    absolute position is non-negative, no later than the query position, and --
    on a sliding layer -- strictly inside the window. The mask is not
    redefined here, so the comparison is a shared-function consistency check.
    """

    from hipengine.kernels.cpu_reference.gemma4_int8 import _attention_head_row

    q = np.asarray(query, dtype=np.float32)
    key = np.asarray(keys, dtype=np.float32)
    value = np.asarray(values, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("query must be [num_heads, head_dim]")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("keys and values must both be [context, num_kv_heads, head_dim]")
    num_heads, head_dim = q.shape
    if key.shape[2] != head_dim:
        raise ValueError("query head_dim must match key head_dim")
    if num_heads % int(num_kv_heads) != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    context = int(key.shape[0])
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.size != context:
        raise ValueError("positions must have one entry per key slot")
    window = None if sliding_window is None or int(sliding_window) <= 0 else int(sliding_window)
    keep = np.ones(context, dtype=bool)
    for slot in range(context):
        visible = int(pos[slot]) >= 0 and int(pos[slot]) <= int(query_position)
        if window is not None:
            visible = visible and int(pos[slot]) > int(query_position) - window
        keep[slot] = visible
    kv_group = num_heads // int(num_kv_heads)
    out = np.empty((num_heads, head_dim), dtype=np.float32)
    for head in range(num_heads):
        kv_head = head // kv_group
        row = _attention_head_row(q[head], key[:, kv_head], value[:, kv_head], keep, scale)
        out[head] = np.nan if row is None else row
    return out


# ---------------------------------------------------------------------------
# Validation: no silently skipped layers, no invalid shapes or non-finite values
# ---------------------------------------------------------------------------


def validate_replay_layers(replay_layers: Sequence[int], num_layers: int) -> list[int]:
    """Return the replay layer list, refusing an empty, duplicate or out-of-range set."""

    layers = [int(layer) for layer in replay_layers]
    if not layers:
        raise ValueError(
            "at least one replay layer is required; an empty configuration must "
            "fail rather than vacuously pass"
        )
    if len(set(layers)) != len(layers):
        raise ValueError(f"duplicate replay layers: {layers}")
    for layer in layers:
        if layer < 0 or layer >= int(num_layers):
            raise ValueError(f"replay layer {layer} outside [0, {num_layers})")
    return layers


def _expected_field_widths(entry: dict[str, Any], rows: int, hidden_size: int) -> dict[str, int]:
    q_width = int(entry["num_heads"]) * int(entry["head_dim"])
    kv_width = int(entry["num_kv_heads"]) * int(entry["head_dim"])
    return {
        "q_rot": rows * q_width,
        "k_rot": rows * kv_width,
        "v": rows * kv_width,
        "context": rows * q_width,
        "hidden_in": rows * int(hidden_size),
    }


def _validate_record(
    record: dict[str, Any],
    entry: dict[str, Any],
    *,
    rows: int,
    hidden_size: int,
    label: str,
) -> None:
    if int(record.get("rows", -1)) != int(rows):
        raise ValueError(f"{label}: rows {record.get('rows')} != {rows}")
    for field, width in _expected_field_widths(entry, rows, hidden_size).items():
        if field not in record:
            raise ValueError(f"{label}: missing field {field!r}")
        array = np.asarray(record[field])
        if array.size != width:
            raise ValueError(
                f"{label}: field {field!r} has {array.size} elements, expected {width}"
            )
        if array.dtype != np.uint16:
            raise ValueError(f"{label}: field {field!r} dtype {array.dtype} is not uint16")
        if not np.all(np.isfinite(bf16_bits_to_float(array))):
            raise ValueError(f"{label}: field {field!r} holds a non-finite value")
    if "context_f32" in record:
        context_f32 = np.asarray(record["context_f32"])
        width = _expected_field_widths(entry, rows, hidden_size)["context"]
        if context_f32.size != width or context_f32.dtype != np.float32:
            raise ValueError(f"{label}: context_f32 shape/dtype is invalid")
        if not np.all(np.isfinite(context_f32)):
            raise ValueError(f"{label}: context_f32 holds a non-finite value")


def validate_decode_records(
    records: dict[int, dict[str, Any]],
    geometry: Sequence[dict[str, Any]],
    *,
    hidden_size: int,
    require_context_f32: bool,
) -> None:
    """Require one valid, finite record for every decode layer and no extras."""

    num_layers = len(geometry)
    expected = set(range(num_layers))
    present = {int(key) for key in records}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        raise ValueError(f"decode capture is missing layers {missing}")
    if extra:
        raise ValueError(f"decode capture has unexpected layers {extra}")
    for layer in range(num_layers):
        record = records[layer]
        _validate_record(
            record, geometry[layer], rows=1, hidden_size=hidden_size,
            label=f"decode layer {layer}",
        )
        if require_context_f32 and "context_f32" not in record:
            raise ValueError(f"decode layer {layer}: missing INT8 context_f32 scratch")


def validate_prefill_records(
    records: dict[int, dict[str, Any]],
    geometry: Sequence[dict[str, Any]],
    *,
    hidden_size: int,
    prefill_rows: int,
    required_layers: Sequence[int],
) -> None:
    """Require a valid, finite prefill record for exactly the replay layers."""

    expected = set(int(layer) for layer in required_layers)
    present = {int(key) for key in records}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        raise ValueError(f"prefill capture is missing layers {missing}")
    if extra:
        raise ValueError(f"prefill capture has unexpected layers {extra}")
    for layer in sorted(expected):
        _validate_record(
            records[layer], geometry[layer], rows=prefill_rows, hidden_size=hidden_size,
            label=f"prefill layer {layer}",
        )


# ---------------------------------------------------------------------------
# Comparison and classification
# ---------------------------------------------------------------------------


def compare_layer_boundaries(
    bf16_layers: dict[int, dict[str, Any]],
    int8_layers: dict[int, dict[str, Any]],
    *,
    num_layers: int,
) -> dict[str, Any]:
    """Localize the first divergent attention input and output across layers.

    Every array in a record is raw BF16 bits, so input equality is exact bit
    equality. ``context`` (the attention output) is compared both exactly and in
    decoded floats. The first layer whose output diverges while its input is
    identical is the injection point; the first layer whose input diverges is
    where the difference has propagated through the residual stream. Every layer
    must be present: a missing layer is an error, not a skip.
    """

    layers: dict[int, Any] = {}
    first_output: int | None = None
    first_input: int | None = None
    for layer in range(int(num_layers)):
        if layer not in bf16_layers or layer not in int8_layers:
            raise ValueError(f"layer {layer} is missing from a capture")
        b = bf16_layers[layer]
        i = int8_layers[layer]
        hidden_equal = np.array_equal(b["hidden_in"], i["hidden_in"])
        q_equal = np.array_equal(b["q_rot"], i["q_rot"])
        k_equal = np.array_equal(b["k_rot"], i["k_rot"])
        v_equal = np.array_equal(b["v"], i["v"])
        context_equal = np.array_equal(b["context"], i["context"])
        ctx_b = bf16_bits_to_float(b["context"]).astype(np.float64)
        ctx_i = bf16_bits_to_float(i["context"]).astype(np.float64)
        changed = int(np.count_nonzero(np.asarray(b["context"]) != np.asarray(i["context"])))
        record = {
            "hidden_input_equal": bool(hidden_equal),
            "q_equal": bool(q_equal),
            "k_equal": bool(k_equal),
            "v_equal": bool(v_equal),
            "context_equal": bool(context_equal),
            "context_changed_elements": changed,
            "context_elements": int(np.asarray(b["context"]).size),
            "context_max_abs_diff": max_abs_diff(ctx_b, ctx_i),
            "context_rel_l2": relative_l2(ctx_b, ctx_i),
            "bf16_route": b.get("bf16_route"),
            "int8_route": i.get("int8_route"),
        }
        layers[layer] = record
        if not context_equal and first_output is None:
            first_output = layer
        if not (hidden_equal and q_equal and k_equal and v_equal) and first_input is None:
            first_input = layer
    return {
        "num_layers_compared": len(layers),
        "layers": layers,
        "first_divergent_output_layer": first_output,
        "first_divergent_input_layer": first_input,
    }


def compare_kv_reconstruction(
    keys_bf16: np.ndarray,
    values_bf16: np.ndarray,
    keys_int8: np.ndarray,
    values_int8: np.ndarray,
) -> dict[str, Any]:
    """Original-versus-reconstructed source comparison for a complete K/V sequence.

    ``*_bf16`` are the original BF16 values decoded to FP32; ``*_int8`` are
    ``float32(int8) * float32(stored scale)``. The comparison covers every slot
    of the complete prefill-plus-decode sequence, so it is a statement about the
    whole written K/V rather than the decode row alone.
    """

    result: dict[str, Any] = {"slots": int(np.asarray(keys_bf16).shape[0])}
    for name, original, reconstructed in (
        ("key", keys_bf16, keys_int8),
        ("value", values_bf16, values_int8),
    ):
        original = np.asarray(original, dtype=np.float32)
        reconstructed = np.asarray(reconstructed, dtype=np.float32)
        result[name] = {
            "shape": list(original.shape),
            "reconstructed_shape": list(reconstructed.shape),
            "original_vs_reconstructed_rel_l2": relative_l2(original, reconstructed),
            "original_vs_reconstructed_max_abs": max_abs_diff(original, reconstructed),
            "exact_equal_elements": int(np.count_nonzero(original == reconstructed)),
            "elements": int(original.size),
            "original_bf16_vs_reconstructed_literal": literal_comparison(
                original, reconstructed
            ),
        }
    return result


def compare_prefill_kv_equality(
    bf16_prefill: dict[str, Any], int8_prefill: dict[str, Any]
) -> dict[str, bool]:
    """Bit-equality of the two arms' written prefill K and V at one layer.

    The decode-boundary classification must include the whole written history,
    not just the current decode row: the replay writes the BF16 arm's captured
    prefill K/V, so its match to the whole-model INT8 output depends on the two
    arms' prefill K/V being identical. Raw BF16 bits, exact comparison.
    """

    return {
        "k_equal": bool(np.array_equal(bf16_prefill["k_rot"], int8_prefill["k_rot"])),
        "v_equal": bool(np.array_equal(bf16_prefill["v"], int8_prefill["v"])),
    }


def wholemodel_fixed_input_verdict(
    decode_row_inputs_equal: bool, prefill_kv_equal: bool
) -> dict[str, bool]:
    """Classify whether the whole-model arms are a fixed-input comparison.

    True only when the current decode row's hidden/query/written K/written V are
    bit-identical *and* the two arms' written prefill K/V over all history are
    bit-identical. Either alone is insufficient.
    """

    return {
        "decode_row_inputs_equal": bool(decode_row_inputs_equal),
        "prefill_kv_equal": bool(prefill_kv_equal),
        "fixed_input": bool(decode_row_inputs_equal and prefill_kv_equal),
    }


def judge_replay(report: dict[str, Any], tolerances: dict[str, float]) -> dict[str, Any]:
    """Classify a fixed-input replay against the frozen tolerances.

    ``bf16_consumer_faithful`` / ``int8_consumer_faithful`` say whether each GPU
    consumer matches the common CPU algorithm on the *same* input, judged by the
    frozen tolerances. These are never relaxed: a measured BF16 max-absolute
    error above the frozen threshold makes ``bf16_consumer_faithful`` false and
    is reported as a failure. The quantization error and the fixed-input
    consumer difference are reported as values, not pass/fail.
    """

    failed: list[str] = []
    bf16_faithful = (
        float(report["bf16_consumer_vs_cpu_rel_l2"]) <= tolerances["bf16_consumer_vs_cpu_rel_l2"]
        and float(report["bf16_consumer_vs_cpu_max_abs"]) <= tolerances["bf16_consumer_vs_cpu_max_abs"]
    )
    int8_faithful = (
        float(report["int8_consumer_vs_cpu_rel_l2"]) <= tolerances["int8_consumer_vs_cpu_rel_l2"]
        and float(report["int8_consumer_vs_cpu_max_abs"]) <= tolerances["int8_consumer_vs_cpu_max_abs"]
    )
    if not bf16_faithful:
        failed.append("bf16_consumer_vs_cpu")
    if not int8_faithful:
        failed.append("int8_consumer_vs_cpu")
    return {
        "bf16_consumer_faithful": bf16_faithful,
        "int8_consumer_faithful": int8_faithful,
        "failed": failed,
        "passed": not failed,
    }


def overall_verdict(
    *,
    cpu_self_check_passed: bool,
    replay_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Combine every control into one explicit diagnostic verdict.

    The controls are the shared-function CPU self-check and, for every replay
    layer, the frozen consumer verdicts (which include the literal BF16
    production-versus-rounded-CPU raw-byte comparison). A failed control makes
    the overall verdict fail and the process exit non-zero; the artifact is still
    written, and nothing here gates the runtime.
    """

    failed: list[str] = []
    if not cpu_self_check_passed:
        failed.append("cpu_shared_function_self_check")
    for layer, report in replay_reports.items():
        if not report["verdict"]["passed"]:
            failed.append(f"replay_consumer_control:layer{layer}")
        # Structural invariant: the BF16 readback must be the rounding of the
        # FP32 consumer output. The literal BF16-production-versus-rounded-CPU
        # comparison is reported as a proof beside the verdict, not as a control:
        # on a full-attention layer the GPU reduction order can round to a
        # neighbour, so raw-byte equality is not required.
        if not report["replay_context_bf16_vs_f32_rounding_literal"]["raw_bytes_equal"]:
            failed.append(f"replay_bf16_is_f32_rounding:layer{layer}")
    return {"failed": failed, "passed": not failed}


# ---------------------------------------------------------------------------
# Device capture and replay (imports stay inside these functions)
# ---------------------------------------------------------------------------


def _copy_d2h(source: Any, host: np.ndarray, runtime: Any) -> np.ndarray:
    """Copy device memory (a buffer or a raw pointer) into a host array."""

    from hipengine.core.hip import MemcpyKind
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    if hasattr(source, "nbytes"):
        copy_device_to_host(host_array_ptr(host), source, host.nbytes, runtime=runtime)
    else:
        runtime.memcpy(
            host_array_ptr(host), int(source), host.nbytes, MemcpyKind.DEVICE_TO_HOST
        )
    return host


def _download_bf16(ptr: Any, elements: int, runtime: Any) -> np.ndarray:
    host = np.empty(int(elements), dtype=np.uint16)
    return _copy_d2h(ptr, host, runtime)


def _download_f32(ptr: Any, elements: int, runtime: Any) -> np.ndarray:
    host = np.empty(int(elements), dtype=np.float32)
    return _copy_d2h(ptr, host, runtime)


def _upload_bf16(bits: np.ndarray, runtime: Any, buffers: list) -> Any:
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    contig = np.ascontiguousarray(bits, dtype=np.uint16)
    buffer = malloc(contig.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contig), contig.nbytes, runtime=runtime)
    return buffer


@contextmanager
def _layer_counter_reset(runner: Any, state: dict[str, int]) -> Iterator[None]:
    """Reset the per-forward layer index counter on every ``runner.forward``."""

    original = runner.forward

    def wrapped(tokens: Any, **kwargs: Any) -> Any:
        state["index"] = 0
        return original(tokens, **kwargs)

    runner.forward = wrapped
    try:
        yield
    finally:
        runner.forward = original


@contextmanager
def _capture_layer_boundaries(
    *,
    runtime: Any,
    hidden_size: int,
    geometry: Sequence[dict[str, Any]],
    state: dict[str, int],
    target_layers: set[int],
    records: dict[int, dict[str, Any]],
) -> Iterator[None]:
    """Capture the attention boundary of every targeted layer on one forward.

    Intercepts the module-level layer forward the runner calls. The hidden input
    is read before the call; the BF16 query, written K, written V, context and
    (on the INT8 arm) the FP32 consumer scratch are read after it, because the
    layer forward leaves those scratch buffers intact. Originals are restored in
    ``finally``.
    """

    import hipengine.runtime.gemma4 as gemma4_runtime
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer_mod

    original = gemma4_runtime.gemma4_layer_forward_bf16

    def wrapped(hidden_ptr, cos_ptr, sin_ptr, keep_mask_ptr, layer, *, scratch, **kwargs):
        index = int(state["index"])
        state["index"] = index + 1
        rows = int(kwargs["rows"])
        capture = index in target_layers
        hidden_in = None
        if capture:
            runtime.device_synchronize()
            hidden_in = _download_bf16(hidden_ptr, rows * hidden_size, runtime)
        result = original(
            hidden_ptr, cos_ptr, sin_ptr, keep_mask_ptr, layer, scratch=scratch, **kwargs
        )
        if capture:
            runtime.device_synchronize()
            entry = geometry[index]
            q_width = entry["num_heads"] * entry["head_dim"]
            kv_width = entry["num_kv_heads"] * entry["head_dim"]
            record = {
                "rows": rows,
                "hidden_in": hidden_in,
                "q_rot": _download_bf16(scratch._by_name["q_rot"].ptr, rows * q_width, runtime),
                "k_rot": _download_bf16(scratch._by_name["k_rot"].ptr, rows * kv_width, runtime),
                "v": _download_bf16(scratch._by_name["v"].ptr, rows * kv_width, runtime),
                "context": _download_bf16(scratch._by_name["context"].ptr, rows * q_width, runtime),
                "bf16_route": layer_mod.last_prefill_attention_route(),
            }
            int8_kv = kwargs.get("int8_kv")
            if int8_kv is not None:
                record["context_f32"] = _download_f32(
                    int8_kv.context_f32, rows * q_width, runtime
                )
                record["int8_route"] = layer_mod.last_int8_kv_route()
            records[index] = record
        return result

    gemma4_runtime.gemma4_layer_forward_bf16 = wrapped
    try:
        yield
    finally:
        gemma4_runtime.gemma4_layer_forward_bf16 = original


def capture_arm(
    runner: Any,
    chain: FrozenChain,
    *,
    runtime: Any,
    hidden_size: int,
    geometry: Sequence[dict[str, Any]],
    prefill_layers: set[int],
    decode_layers: set[int],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Run prefill plus the first decode position and capture the boundaries.

    Only the first scored decode position is forwarded: the acceptance reuses the
    frozen chain's first position, which already carries a large captured KL
    difference, and a larger sweep is out of scope.
    """

    records: dict[str, dict[int, dict[str, Any]]] = {"prefill": {}, "decode": {}}
    state = {"index": 0}
    with _layer_counter_reset(runner, state):
        runner.reset()
        with _capture_layer_boundaries(
            runtime=runtime,
            hidden_size=hidden_size,
            geometry=geometry,
            state=state,
            target_layers=prefill_layers,
            records=records["prefill"],
        ):
            runner.forward(list(chain.prompt_ids[: chain.prefill]))
        with _capture_layer_boundaries(
            runtime=runtime,
            hidden_size=hidden_size,
            geometry=geometry,
            state=state,
            target_layers=decode_layers,
            records=records["decode"],
        ):
            runner.forward([chain.prompt_ids[chain.prefill]])
    return records


def _zero_layer_planes(owner: Any, layer: int, runtime: Any) -> None:
    """Zero one layer's payload and scale planes so unwritten slots stay finite."""

    from hipengine.core.memory import copy_host_to_device, host_array_ptr

    for buffer in (
        owner.key_caches[layer],
        owner.value_caches[layer],
        owner.k_scale_buffers[layer],
        owner.v_scale_buffers[layer],
    ):
        zeros = np.zeros(buffer.nbytes, dtype=np.uint8)
        copy_host_to_device(buffer, host_array_ptr(zeros), zeros.nbytes, runtime=runtime)


def _download_quantized(owner: Any, layer: int, runtime: Any) -> tuple[np.ndarray, ...]:
    """Download the writer's quantized cache in the oracle's layout."""

    block_size = owner.block_size
    blocks = owner.blocks
    _, num_kv_heads, head_dim = owner.attentions[layer]
    scale_np = np.float16 if owner.scale_dtype.itemsize == 2 else np.float32
    key_host = np.empty((blocks, block_size, num_kv_heads, head_dim), dtype=np.int8)
    value_host = np.empty_like(key_host)
    k_scale_host = np.empty((blocks, block_size, num_kv_heads), dtype=scale_np)
    v_scale_host = np.empty_like(k_scale_host)
    _download_raw(owner.key_caches[layer], key_host, runtime)
    _download_raw(owner.value_caches[layer], value_host, runtime)
    _download_raw(owner.k_scale_buffers[layer], k_scale_host, runtime)
    _download_raw(owner.v_scale_buffers[layer], v_scale_host, runtime)
    return key_host, value_host, k_scale_host, v_scale_host


def _download_raw(buffer: Any, host: np.ndarray, runtime: Any) -> np.ndarray:
    return _copy_d2h(buffer, host, runtime)


def _run_replay_block(
    owner: Any,
    layer: int,
    geometry_entry: dict[str, Any],
    *,
    record: dict[str, Any],
    rows: int,
    write_offset: int,
    runtime: Any,
    capture: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Append one block through the production writer+consumer and read back.

    Runs the exact registered functions the runner uses; no arithmetic is
    reimplemented and no production source changes. Temporary device buffers are
    freed in ``finally``.
    """

    from hipengine.core.memory import free as hip_free
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        Gemma4LayerGeometry,
        _run_int8_attention,
    )

    block = owner.begin_block(write_offset=write_offset, rows=rows, stream=0)
    layer_kv = owner.layer_kv(layer, block)
    geometry = Gemma4LayerGeometry(
        num_heads=geometry_entry["num_heads"],
        num_kv_heads=geometry_entry["num_kv_heads"],
        head_dim=geometry_entry["head_dim"],
        scale=1.0,
        k_eq_v=bool(geometry_entry["k_eq_v"]),
        sliding_window=geometry_entry["sliding_window"],
    )
    buffers: list = []
    try:
        query = _upload_bf16(record["q_rot"], runtime, buffers)
        key = _upload_bf16(record["k_rot"], runtime, buffers)
        value = _upload_bf16(record["v"], runtime, buffers)
        width = rows * geometry_entry["num_heads"] * geometry_entry["head_dim"]
        context = _upload_bf16(np.zeros(width, dtype=np.uint16), runtime, buffers)
        _run_int8_attention(
            layer_kv,
            query_bf16=query.ptr,
            key_bf16=key.ptr,
            value_bf16=value.ptr,
            context_bf16=context.ptr,
            rows=rows,
            geometry=geometry,
            stream=0,
        )
        runtime.device_synchronize()
        if not capture:
            return None, None
        context_f32 = _download_f32(layer_kv.context_f32, width, runtime)
        context_bf16 = _download_bf16(context, width, runtime)
        return context_f32, context_bf16
    finally:
        for buffer in buffers:
            hip_free(buffer, runtime=runtime)


def replay_fixed_input_layer(
    *,
    layer: int,
    geometry_entry: dict[str, Any],
    attentions: Sequence[tuple[int, int, int]],
    prefill_record: dict[str, Any],
    decode_record: dict[str, Any],
    runtime: Any,
    block_size: int,
    capacity: int,
    max_block: int,
    scale_dtype: Any,
) -> dict[str, Any]:
    """Replay one layer's captured BF16 K/V and query through both consumers.

    Builds a temporary INT8 owner, writes the captured BF16 prefill K/V and then
    the captured decode K/V through the registered writer, runs the registered
    consumer, and returns the FP32 and BF16 outputs plus the quantized cache.
    The owner and every temporary buffer are released in ``finally``.
    """

    from hipengine.runtime.gemma4_int8_kv import Gemma4Int8KVCache

    owner = Gemma4Int8KVCache(
        capacity=capacity,
        max_block=max_block,
        attentions=tuple(attentions),
        block_size=block_size,
        scale_dtype=scale_dtype,
    )
    try:
        _zero_layer_planes(owner, layer, runtime)
        prefill_rows = int(prefill_record["rows"])
        _run_replay_block(
            owner, layer, geometry_entry, record=prefill_record, rows=prefill_rows,
            write_offset=0, runtime=runtime, capture=False,
        )
        context_f32, context_bf16 = _run_replay_block(
            owner, layer, geometry_entry, record=decode_record, rows=1,
            write_offset=prefill_rows, runtime=runtime, capture=True,
        )
        key_cache, value_cache, k_scale, v_scale = _download_quantized(owner, layer, runtime)
        blocks = int(owner.blocks)
        owner_attentions = tuple(owner.attentions)
    finally:
        owner.close()
    return {
        "layer": layer,
        "prefill_rows": prefill_rows,
        "context_f32": context_f32,
        "context_bf16": context_bf16,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "blocks": blocks,
        "block_size": int(block_size),
        "num_kv_heads": int(geometry_entry["num_kv_heads"]),
        "head_dim": int(geometry_entry["head_dim"]),
        "num_heads": int(geometry_entry["num_heads"]),
        "sliding_window": geometry_entry["sliding_window"],
        "attentions": owner_attentions,
    }


def _gather_dequantized(
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    *,
    context: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather slots ``[0, context)`` from a dequantized identity-table cache."""

    from hipengine.kernels.cpu_reference.ops import dequantize_kv_int8_per_token_head

    key, value = dequantize_kv_int8_per_token_head(key_cache, value_cache, k_scale, v_scale)
    keys = np.stack([key[slot // block_size, slot % block_size] for slot in range(context)])
    values = np.stack([value[slot // block_size, slot % block_size] for slot in range(context)])
    return keys, values


def compare_replay(
    replay: dict[str, Any],
    *,
    bf16_prefill: dict[str, Any],
    bf16_decode: dict[str, Any],
    int8_decode_context_f32: np.ndarray | None,
) -> dict[str, Any]:
    """Compute the fixed-input comparison quantities for one layer.

    Three quantities on the same captured common input:

    1. the production BF16 attention output (captured);
    2. the production INT8 consumer output, FP32 and FP32-to-BF16 narrowed;
    3. one common CPU FP32 algorithm on the original BF16 values and on
       ``float32(int8) * float32(stored scale)``.

    The CPU algorithm is checked against the existing quantized oracle on the
    quantized representation as a shared-function consistency check, and the
    BF16 production output is compared to the BF16 rounding of the CPU control by
    literal raw bytes. Tolerances are frozen; nothing is relaxed here.
    """

    from hipengine.kernels.cpu_reference.gemma4_int8 import (
        gemma4_attention_decode_int8_per_token_head,
    )

    num_heads = int(replay["num_heads"])
    num_kv_heads = int(replay["num_kv_heads"])
    head_dim = int(replay["head_dim"])
    block_size = int(replay["block_size"])
    prefill_rows = int(replay["prefill_rows"])
    context = prefill_rows + 1
    window = replay["sliding_window"]

    q_f32 = bf16_bits_to_float(bf16_decode["q_rot"]).reshape(num_heads, head_dim)
    k_prefill = bf16_bits_to_float(bf16_prefill["k_rot"]).reshape(prefill_rows, num_kv_heads, head_dim)
    v_prefill = bf16_bits_to_float(bf16_prefill["v"]).reshape(prefill_rows, num_kv_heads, head_dim)
    k_decode = bf16_bits_to_float(bf16_decode["k_rot"]).reshape(1, num_kv_heads, head_dim)
    v_decode = bf16_bits_to_float(bf16_decode["v"]).reshape(1, num_kv_heads, head_dim)
    keys_bf16 = np.concatenate([k_prefill, k_decode], axis=0)
    values_bf16 = np.concatenate([v_prefill, v_decode], axis=0)

    keys_int8, values_int8 = _gather_dequantized(
        replay["key_cache"], replay["value_cache"], replay["k_scale"], replay["v_scale"],
        context=context, block_size=block_size,
    )
    positions = np.arange(context, dtype=np.int64)
    query_position = prefill_rows
    cpu_bf16 = common_cpu_attention(
        q_f32, keys_bf16, values_bf16,
        positions=positions, query_position=query_position,
        sliding_window=window, scale=1.0, num_kv_heads=num_kv_heads,
    )
    cpu_int8 = common_cpu_attention(
        q_f32, keys_int8, values_int8,
        positions=positions, query_position=query_position,
        sliding_window=window, scale=1.0, num_kv_heads=num_kv_heads,
    )

    # Shared-function consistency check: the common algorithm on the quantized
    # representation must reproduce the existing oracle, because it calls the
    # oracle's own head row. This is not an independent second oracle.
    oracle = gemma4_attention_decode_int8_per_token_head(
        q_f32,
        replay["key_cache"],
        replay["value_cache"],
        replay["k_scale"],
        replay["v_scale"],
        np.arange(replay["blocks"], dtype=np.int32),
        context,
        block_size=block_size,
        scale=1.0,
        token_positions=positions,
        row_position=query_position,
        sliding_window=window,
    )
    oracle_self = relative_l2(cpu_int8, oracle)

    prod_int8_f32 = np.asarray(replay["context_f32"], dtype=np.float32).reshape(num_heads, head_dim)
    prod_int8_bf16 = bf16_bits_to_float(replay["context_bf16"]).reshape(num_heads, head_dim)
    prod_bf16_bits = np.asarray(bf16_decode["context"], dtype=np.uint16).reshape(
        num_heads * head_dim
    )
    prod_bf16 = bf16_bits_to_float(prod_bf16_bits).reshape(num_heads, head_dim)

    # The BF16 production output is read back BF16-narrowed. The frozen tolerance
    # is reported as it falls; the rounding floor and the literal raw-byte
    # comparison against the BF16 rounding of the CPU control are recorded beside
    # it, so a failure is attributed rather than hidden.
    cpu_bf16_rounded_bits = float_to_bf16_bits(cpu_bf16)
    cpu_bf16_rounded = bf16_bits_to_float(cpu_bf16_rounded_bits).reshape(num_heads, head_dim)
    bf16_floor_rel = relative_l2(cpu_bf16_rounded, cpu_bf16)
    bf16_floor_max = max_abs_diff(cpu_bf16_rounded, cpu_bf16)
    bf16_observed_rel = relative_l2(prod_bf16, cpu_bf16)
    bf16_observed_max = max_abs_diff(prod_bf16, cpu_bf16)
    bf16_literal = literal_comparison(
        prod_bf16_bits, cpu_bf16_rounded_bits.reshape(-1)
    )

    replay_context_bf16_literal = literal_comparison(
        np.asarray(replay["context_bf16"], dtype=np.uint16),
        float_to_bf16_bits(replay["context_f32"]),
    )
    wholemodel_literal = (
        None
        if int8_decode_context_f32 is None
        else literal_comparison(replay["context_f32"], np.asarray(int8_decode_context_f32, dtype=np.float32))
    )

    report: dict[str, Any] = {
        "layer": int(replay["layer"]),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "context": context,
        "query_position": query_position,
        "sliding_window": window,
        "attention_scale": 1.0,
        "attention_logit_softcap": None,
        "cpu_shared_function_self_check_rel_l2": oracle_self,
        "cpu_shared_function_self_check_scope": (
            "the common CPU algorithm calls _attention_head_row and the mask loop "
            "from the existing quantized oracle, so this is a shared-function "
            "consistency check, not an independent second oracle"
        ),
        "bf16_consumer_vs_cpu_rel_l2": bf16_observed_rel,
        "bf16_consumer_vs_cpu_max_abs": bf16_observed_max,
        "bf16_readback_rounding_floor_rel_l2": bf16_floor_rel,
        "bf16_readback_rounding_floor_max_abs": bf16_floor_max,
        "bf16_readback_floor_comparison_allowance": 1e-6,
        "bf16_consumer_within_readback_floor": bool(
            bf16_observed_rel <= bf16_floor_rel + 1e-6
            and bf16_observed_max <= bf16_floor_max + 1e-6
        ),
        "bf16_consumer_matches_readback_floor_exactly": bool(
            bf16_observed_rel == bf16_floor_rel and bf16_observed_max == bf16_floor_max
        ),
        "bf16_production_matches_rounded_cpu_raw_bytes": bf16_literal,
        "int8_consumer_vs_cpu_rel_l2": relative_l2(prod_int8_f32, cpu_int8),
        "int8_consumer_vs_cpu_max_abs": max_abs_diff(prod_int8_f32, cpu_int8),
        "int8_consumer_bf16_vs_cpu_rel_l2": relative_l2(prod_int8_bf16, cpu_int8),
        "replay_context_bf16_vs_f32_rounding_literal": replay_context_bf16_literal,
        "int8_replay_vs_wholemodel_context_f32_rel_l2": (
            None
            if int8_decode_context_f32 is None
            else relative_l2(prod_int8_f32, np.asarray(int8_decode_context_f32, dtype=np.float32))
        ),
        "int8_replay_vs_wholemodel_context_f32_literal": wholemodel_literal,
        "kv_reconstruction": compare_kv_reconstruction(
            keys_bf16, values_bf16, keys_int8, values_int8
        ),
        "quantization_cpu_bf16_vs_cpu_int8_rel_l2": relative_l2(cpu_bf16, cpu_int8),
        "quantization_cpu_bf16_vs_cpu_int8_max_abs": max_abs_diff(cpu_bf16, cpu_int8),
        "fixed_input_bf16_vs_int8_bf16_rel_l2": relative_l2(prod_bf16, prod_int8_bf16),
        "fixed_input_bf16_vs_int8_bf16_max_abs": max_abs_diff(prod_bf16, prod_int8_bf16),
        "cpu_bf16_output_l2": float(np.linalg.norm(cpu_bf16.astype(np.float64))),
        "cpu_int8_output_l2": float(np.linalg.norm(cpu_int8.astype(np.float64))),
    }
    return report


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _provenance_row(artifact: Path, loading: dict[str, Any]) -> dict[str, Any]:
    provenance = _provenance(artifact, loading)
    provenance["wrapper_source_sha256"] = sha256_file(Path(__file__))
    provenance["command"] = " ".join([sys.executable, *sys.argv])
    provenance["physical_host"] = {
        "hostname": socket.gethostname(),
        "platform_node": os.uname().nodename,
    }
    provenance["gpu_binding"] = {
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    root = Path(__file__).resolve().parents[1]
    provenance["attention_source_sha256"] = {
        name: sha256_file(root / "hipengine/kernels/hip_gfx1100/gemma4" / name)
        for name in ("gemma4_attention.hip", "gemma4_attention.py", "gemma4_attention_int8.hip",
                     "gemma4_attention_int8.py", "gemma4_layer.py")
    }
    provenance["int8_source_sha256"] = {
        name: sha256_file(root / relative)
        for name, relative in (
            ("writer_hip", "hipengine/kernels/hip_gfx1100/attention/paged_kv_write.hip"),
            ("writer_py", "hipengine/kernels/hip_gfx1100/attention/paged_kv_write.py"),
            ("owner", "hipengine/runtime/gemma4_int8_kv.py"),
            ("runtime", "hipengine/runtime/gemma4.py"),
        )
    }
    # The CPU control and the dequantization it compares against are sources too,
    # so the shared-function scope is auditable.
    provenance["cpu_control_source_sha256"] = {
        name: sha256_file(root / relative)
        for name, relative in (
            ("cpu_oracle", "hipengine/kernels/cpu_reference/gemma4_int8.py"),
            ("cpu_dequant", "hipengine/kernels/cpu_reference/ops.py"),
        )
    }
    provenance["arithmetic_order_notes"] = ARITHMETIC_ORDER_NOTES
    return provenance


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _save_npz(path: Path, arrays: dict[str, np.ndarray], provenance: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["provenance"] = np.frombuffer(
        json.dumps(_json_safe(provenance), sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez(path, **payload)
    return sha256_file(path)


def _build_interpretation(
    localization: dict[str, Any],
    replay_reports: dict[str, dict[str, Any]],
    *,
    replay_layers: Sequence[int],
    prefill_kv_equality: dict[int, dict[str, bool]],
) -> dict[str, Any]:
    """Derive the interpretation from the measured localization and replay data.

    The whole-model fixed-input classification combines the current decode row's
    hidden/query/written-K/written-V equality with the measured bit-equality of
    the two arms' written prefill K/V over all history. The decode row alone is
    insufficient: the replay writes the BF16 arm's prefill K/V, so a differing
    prefill history makes the whole-model arms not a fixed-input comparison even
    when the decode row matches. Every note is built from measured booleans.
    """

    layers = localization["layers"]
    first_output = localization["first_divergent_output_layer"]
    first_input = localization["first_divergent_input_layer"]
    decode_row_inputs_equal = {
        str(layer): bool(
            layers[layer]["hidden_input_equal"]
            and layers[layer]["q_equal"]
            and layers[layer]["k_equal"]
            and layers[layer]["v_equal"]
        )
        for layer in replay_layers
    }
    prefill_kv_equal = {
        str(layer): bool(
            prefill_kv_equality[layer]["k_equal"] and prefill_kv_equality[layer]["v_equal"]
        )
        for layer in replay_layers
    }
    fixed_input = {
        str(layer): wholemodel_fixed_input_verdict(
            decode_row_inputs_equal[str(layer)], prefill_kv_equal[str(layer)]
        )
        for layer in replay_layers
    }
    notes: list[str] = []
    if first_output is None:
        notes.append("No decode-boundary attention output diverged at this position.")
    else:
        inputs_equal = bool(
            layers[first_output]["hidden_input_equal"]
            and layers[first_output]["q_equal"]
            and layers[first_output]["k_equal"]
            and layers[first_output]["v_equal"]
        )
        notes.append(
            f"First observed decode-boundary attention output divergence is layer "
            f"{first_output}; its whole-model decode-row inputs are "
            f"{'bit-identical' if inputs_equal else 'already different'}."
        )
    if first_input is None:
        notes.append("No decode-boundary attention input diverged at this position.")
    else:
        notes.append(
            f"First observed decode-boundary attention input divergence is layer "
            f"{first_input}."
        )
    for layer in replay_layers:
        report = replay_reports[str(layer)]
        verdict = report["verdict"]
        floor_relation = (
            "exactly equals the BF16 readback floor"
            if report["bf16_consumer_matches_readback_floor_exactly"]
            else (
                "does not exceed the recorded BF16 readback floor plus 1e-6, but does not equal the floor"
                if report["bf16_consumer_within_readback_floor"]
                else "exceeds the recorded BF16 readback floor plus 1e-6"
            )
        )
        notes.append(
            f"Layer {layer}: frozen INT8 consumer tolerance "
            f"{'passed' if verdict['int8_consumer_faithful'] else 'FAILED'}; frozen "
            f"BF16 consumer tolerance "
            f"{'passed' if verdict['bf16_consumer_faithful'] else 'FAILED'} "
            f"(observed max abs {report['bf16_consumer_vs_cpu_max_abs']:.6g} against "
            f"frozen {TOLERANCES['bf16_consumer_vs_cpu_max_abs']}); the observed BF16 "
            f"error {floor_relation}; BF16 production matches the BF16-rounded CPU "
            f"control by raw bytes: "
            f"{report['bf16_production_matches_rounded_cpu_raw_bytes']['raw_bytes_equal']}."
        )
        notes.append(
            f"Layer {layer}: whole-model decode-row inputs are "
            f"{'identical' if decode_row_inputs_equal[str(layer)] else 'already different'} "
            f"and written prefill K/V over all history are "
            f"{'identical' if prefill_kv_equal[str(layer)] else 'already different'}, "
            f"so its whole-model arms are "
            f"{'a fixed-input comparison' if fixed_input[str(layer)]['fixed_input'] else 'not a fixed-input comparison'}; "
            f"the replay INT8-vs-whole-model raw-byte equality is "
            f"{None if report['int8_replay_vs_wholemodel_context_f32_literal'] is None else report['int8_replay_vs_wholemodel_context_f32_literal']['raw_bytes_equal']}."
        )
    return {
        "first_observed_decode_output_divergence_layer": first_output,
        "first_observed_decode_input_divergence_layer": first_input,
        "earliest_prefill_divergence": None,
        "earliest_prefill_divergence_note": (
            "prefill hidden input and context were captured only at the replay "
            "layers, so the earliest prefill divergence across all layers was not "
            "captured; the localization above is a decode-boundary statement."
        ),
        "wholemodel_decode_row_inputs_equal_at_replayed_layers": decode_row_inputs_equal,
        "wholemodel_prefill_kv_equal_at_replayed_layers": prefill_kv_equal,
        "wholemodel_fixed_input_at_replayed_layers": {
            layer: verdict["fixed_input"] for layer, verdict in fixed_input.items()
        },
        "int8_consumer_faithful_on_fixed_input": all(
            value["verdict"]["int8_consumer_faithful"] for value in replay_reports.values()
        ),
        "bf16_consumer_frozen_tolerance_passed": all(
            value["verdict"]["bf16_consumer_faithful"] for value in replay_reports.values()
        ),
        "bf16_consumer_within_readback_floor": all(
            value["bf16_consumer_within_readback_floor"]
            for value in replay_reports.values()
        ),
        "bf16_consumer_matches_readback_floor_exactly_per_layer": {
            layer: bool(report["bf16_consumer_matches_readback_floor_exactly"])
            for layer, report in replay_reports.items()
        },
        "bf16_production_matches_rounded_cpu_raw_bytes": {
            layer: bool(
                report["bf16_production_matches_rounded_cpu_raw_bytes"]["raw_bytes_equal"]
            )
            for layer, report in replay_reports.items()
        },
        "notes": notes,
    }


def run(args: argparse.Namespace) -> int:
    from scripts.gemma4_campaign_bench import _resolve_generator
    from scripts.gemma4_int8_kv_teacher_forced_eval import _execution_identity

    artifact = Path(args.artifact)
    started = time.time()
    chain = load_frozen_chain(args.frozen_artifact, FROZEN_CASE)
    llm, runner0, loading = _resolve_generator(artifact, args.context)
    generator = llm._get_text_generator()
    arrays: dict[str, np.ndarray] = {}
    try:
        from hipengine.core.hip import get_hip_runtime

        runtime = get_hip_runtime()
        config = runner0.weights.config
        hidden_size = int(config.hidden_size)
        geometry = [
            {
                "layer_type": str(attention.layer_type),
                "num_heads": int(attention.num_heads),
                "num_kv_heads": int(attention.num_kv_heads),
                "head_dim": int(attention.head_dim),
                "sliding_window": (
                    None if attention.sliding_window is None else int(attention.sliding_window)
                ),
                "k_eq_v": bool(attention.k_eq_v),
            }
            for attention in config.attention
        ]
        attentions = tuple(
            (entry["num_heads"], entry["num_kv_heads"], entry["head_dim"]) for entry in geometry
        )
        if tuple(geometry) != chain.attention_geometry:
            raise RuntimeError("runner attention geometry does not match the frozen artifact")
        num_layers = len(geometry)
        # An empty, duplicate or out-of-range replay set must fail before any
        # capture rather than vacuously pass.
        replay_layers = validate_replay_layers(args.replay_layers, num_layers)

        # --- BF16 arm capture --------------------------------------------
        bf16_runner = generator._ensure_runner(storage="bf16")
        bf16_caps = capture_arm(
            bf16_runner, chain, runtime=runtime, hidden_size=hidden_size,
            geometry=geometry, prefill_layers=set(replay_layers), decode_layers=set(range(num_layers)),
        )
        # --- INT8 arm capture --------------------------------------------
        int8_runner = generator._ensure_runner(
            storage="int8_per_token_head", scale_dtype=args.scale_dtype,
            granularity="per_token_head",
        )
        assert int8_runner.uses_int8_kv, "the INT8 arm did not take the INT8 owner"
        int8_caps = capture_arm(
            int8_runner, chain, runtime=runtime, hidden_size=hidden_size,
            geometry=geometry, prefill_layers=set(replay_layers), decode_layers=set(range(num_layers)),
        )

        # Structural validation: every decode layer present and finite, the
        # replay prefill layers present and finite, nothing extra.
        validate_decode_records(
            bf16_caps["decode"], geometry, hidden_size=hidden_size, require_context_f32=False
        )
        validate_decode_records(
            int8_caps["decode"], geometry, hidden_size=hidden_size, require_context_f32=True
        )
        validate_prefill_records(
            bf16_caps["prefill"], geometry, hidden_size=hidden_size,
            prefill_rows=chain.prefill, required_layers=replay_layers,
        )
        validate_prefill_records(
            int8_caps["prefill"], geometry, hidden_size=hidden_size,
            prefill_rows=chain.prefill, required_layers=replay_layers,
        )

        localization = compare_layer_boundaries(
            bf16_caps["decode"], int8_caps["decode"], num_layers=num_layers
        )

        # Measured bit-equality of the two arms' written prefill K/V at every
        # replay layer. The decode-boundary classification must include the whole
        # written history, not just the current decode row.
        prefill_kv_equality = {
            layer: compare_prefill_kv_equality(
                bf16_caps["prefill"][layer], int8_caps["prefill"][layer]
            )
            for layer in replay_layers
        }

        # --- fixed-input replay at the selected layers -------------------
        replay_reports: dict[str, Any] = {}
        for layer in replay_layers:
            replay = replay_fixed_input_layer(
                layer=layer,
                geometry_entry=geometry[layer],
                attentions=attentions,
                prefill_record=bf16_caps["prefill"][layer],
                decode_record=bf16_caps["decode"][layer],
                runtime=runtime,
                block_size=args.int8_block_size,
                capacity=args.replay_capacity,
                max_block=args.replay_max_block,
                scale_dtype=args.scale_dtype,
            )
            wholemodel = int8_caps["decode"][layer].get("context_f32")
            report = compare_replay(
                replay,
                bf16_prefill=bf16_caps["prefill"][layer],
                bf16_decode=bf16_caps["decode"][layer],
                int8_decode_context_f32=wholemodel,
            )
            report["k_eq_v"] = bool(geometry[layer]["k_eq_v"])
            report["layer_type"] = geometry[layer]["layer_type"]
            report["bf16_route"] = bf16_caps["decode"][layer]["bf16_route"]
            report["int8_route"] = int8_caps["decode"][layer]["int8_route"]
            report["wholemodel_prefill_kv_equal"] = prefill_kv_equality[layer]
            report["verdict"] = judge_replay(report, TOLERANCES)
            replay_reports[str(layer)] = report

            prefix = f"replay.layer{layer}"
            arrays[f"{prefix}.prod_int8_context_f32"] = replay["context_f32"]
            arrays[f"{prefix}.prod_int8_context_bf16"] = replay["context_bf16"]
            arrays[f"{prefix}.key_cache"] = replay["key_cache"]
            arrays[f"{prefix}.value_cache"] = replay["value_cache"]
            arrays[f"{prefix}.k_scale"] = replay["k_scale"]
            arrays[f"{prefix}.v_scale"] = replay["v_scale"]
            arrays[f"{prefix}.bf16_prefill_k"] = bf16_caps["prefill"][layer]["k_rot"]
            arrays[f"{prefix}.bf16_prefill_v"] = bf16_caps["prefill"][layer]["v"]
            arrays[f"{prefix}.bf16_decode_q"] = bf16_caps["decode"][layer]["q_rot"]
            arrays[f"{prefix}.bf16_decode_k"] = bf16_caps["decode"][layer]["k_rot"]
            arrays[f"{prefix}.bf16_decode_v"] = bf16_caps["decode"][layer]["v"]
            arrays[f"{prefix}.bf16_decode_context"] = bf16_caps["decode"][layer]["context"]
            if wholemodel is not None:
                arrays[f"{prefix}.wholemodel_int8_context_f32"] = wholemodel

        # Every captured boundary array for both arms, so the parent can
        # recompute every equality and localization claim from the NPZ.
        for arm, caps in (("bf16", bf16_caps), ("int8", int8_caps)):
            for layer, record in caps["decode"].items():
                for field in ("q_rot", "k_rot", "v", "hidden_in", "context"):
                    arrays[f"{arm}.decode.layer{layer}.{field}"] = record[field]
                if "context_f32" in record:
                    arrays[f"{arm}.decode.layer{layer}.context_f32"] = record["context_f32"]
            for layer, record in caps["prefill"].items():
                for field in ("q_rot", "k_rot", "v", "hidden_in", "context"):
                    arrays[f"{arm}.prefill.layer{layer}.{field}"] = record[field]

        cpu_self_check = {
            key: value["cpu_shared_function_self_check_rel_l2"]
            for key, value in replay_reports.items()
        }
        cpu_self_check_passed = all(
            value <= TOLERANCES["cpu_shared_function_self_check_rel_l2"]
            for value in cpu_self_check.values()
        )
        all_replays_faithful = all(
            value["verdict"]["passed"] for value in replay_reports.values()
        )
        overall = overall_verdict(
            cpu_self_check_passed=cpu_self_check_passed, replay_reports=replay_reports
        )
        interpretation = _build_interpretation(
            localization, replay_reports, replay_layers=replay_layers,
            prefill_kv_equality=prefill_kv_equality,
        )

        provenance = _provenance_row(artifact, loading)
        provenance["execution_identity"] = _execution_identity(llm, generator, int8_runner)
        npz_sha = None
        if args.npz:
            npz_sha = _save_npz(Path(args.npz), arrays, provenance)

        record = {
            "kind": "gemma4_d12_int8_kv_attention_boundary_localization",
            "performance_claim": False,
            "diagnostic_only": True,
            "promotion_qualified": False,
            "frozen_chain": {
                "name": chain.name,
                "prompt_tokens": chain.prompt_tokens,
                "prefill": chain.prefill,
                "scored_rows": chain.scored_rows,
                "first_scored_decode_position": chain.prefill,
                "chain_sha256": chain.chain_sha256,
                "prompt_ids": list(chain.prompt_ids),
                "frozen_artifact": str(args.frozen_artifact),
            },
            "attention_geometry": geometry,
            "mask_semantics": (
                "causal (key <= query) plus the per-layer sliding window; the INT8 "
                "route reads no evict mask"
            ),
            "attention_scale": 1.0,
            "attention_logit_softcap": None,
            "arithmetic_order_notes": ARITHMETIC_ORDER_NOTES,
            "tolerances": dict(TOLERANCES),
            "replay_layers": list(replay_layers),
            "wholemodel_prefill_kv_equality": {
                str(layer): prefill_kv_equality[layer] for layer in replay_layers
            },
            "localization": localization,
            "replay": replay_reports,
            "cpu_shared_function_self_check_rel_l2": cpu_self_check,
            "cpu_shared_function_self_check_passed": cpu_self_check_passed,
            "all_replays_faithful": all_replays_faithful,
            "overall": overall,
            "interpretation": interpretation,
            "npz_path": str(args.npz) if args.npz else None,
            "npz_sha256": npz_sha,
            "run_seconds": round(time.time() - started, 1),
            "provenance": provenance,
        }
        text = json.dumps(_json_safe(record), indent=1, sort_keys=True)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        # A failed control is reported in the artifact and the exit status; it
        # never gates the runtime.
        return 0 if overall["passed"] else 1
    finally:
        llm.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="capture both arms and localize the difference")
    run_parser.add_argument(
        "--artifact", type=Path,
        default=Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
    )
    run_parser.add_argument("--context", type=int, default=8192)
    run_parser.add_argument(
        "--frozen-artifact", type=Path, default=FROZEN_ARTIFACT,
        help="committed screening artifact holding the frozen short English chain",
    )
    run_parser.add_argument(
        "--scale-dtype", choices=("fp16", "fp32"), default="fp16",
    )
    run_parser.add_argument(
        "--replay-layers", type=int, nargs="+", default=[0, 5],
        help="layers to replay through both consumers on the captured common input",
    )
    run_parser.add_argument("--int8-block-size", type=int, default=256)
    run_parser.add_argument("--replay-capacity", type=int, default=256)
    run_parser.add_argument("--replay-max-block", type=int, default=64)
    run_parser.add_argument("--out", type=Path, help="compact diagnostic JSON to write")
    run_parser.add_argument("--npz", type=Path, help="full captured arrays outside the repository")
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error("unknown command")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
