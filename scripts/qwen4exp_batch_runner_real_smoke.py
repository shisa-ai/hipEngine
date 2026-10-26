#!/usr/bin/env python3
"""Real-checkpoint smoke qualification of the N-row Qwen4Exp batch runner.

The synthetic GPU tests cover the batch runner's dispatch, ownership, and state
plumbing on a three-layer model. They cannot cover the production geometry the
actual artifact uses: 48 layers (36 GDN, 12 QSA), head_dim 256 attention with a
128-dim indexer at ratio 4, 512 routed experts with top-10, hidden 2560 with a
10240-wide residual, and a PLE layer at layer 1. This script runs that model.

What it does, in order:

1. **Plan and capacity arithmetic** (``--plan-only`` stops here): read the GGUF
   metadata only, build the residency plan, and report the device-resident
   weight bytes against the free device bytes. No weight is read or uploaded.
2. **One shared residency.** Weights are materialized exactly once, through the
   production load path, and the batch runner and every c1 reference share that
   immutable residency. No duplicate weight allocation.
3. **c1 prefill, then hand-off.** Each short prompt is prefilled by its own c1
   runner (``prefill_serial``), imported into one batch slot, and then advanced
   by both paths from that same state.
4. **Teacher-forced continuation.** Every step's token per row is the c1
   reference's own greedy id, so both paths see the identical token stream and
   the comparison isolates the batch path's arithmetic and state handling.
5. **Row-set schedule.** Rows are advanced as C1, then C2, then C4 row sets, so
   the run also crosses row widths and leaves slots idle in between.
6. **Control checks.** Positions and owner tokens per row, untouched slots,
   then a second pass with the row order reversed after ``reset_slot`` and a
   fresh re-import, comparing per-slot state digests and logits byte for byte.
7. **Memory accounting.** Device free/used before load, after load, after the
   batch owner is constructed, and at the end, plus the measured per-slot state
   bytes.

This is a smoke qualification on short contexts. It is **not** a speed
measurement and **not** a production-quality claim: the calibrated
production-profile gate in ``docs/EXECUTION-PROFILES.md`` (mean/p95/p99/max KL,
top-1 by category, determinism, isolation, task quality) is not run here. Only
the outer smoke floor from ``AGENTS.md`` is evaluated, and the report says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import plan_qwen4_exp_residency
from hipengine.models import resolve_model
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from scripts.qwen4_exp_compare_logits import compare_logits

GIB = 1024**3
MIB = 1024**2

# Short, distinct, representative prompts across the four mtp-bench categories.
# They exist to give four *different* request states, not to measure anything.
DEFAULT_PROMPTS: tuple[tuple[str, str], ...] = (
    ("code", "Write a Python function that reverses a linked list."),
    ("general_en", "Explain in two sentences why the sky appears blue."),
    ("general_ja", "富士山の高さは何メートルですか。"),
    ("mixed_ja_en", "Rust の ownership と borrow checker の関係を説明して。"),
)

# The outer smoke/safety floor from AGENTS.md. The calibrated production gate is
# a separate, stricter contract that this script does not evaluate.
SMOKE_KL_FLOOR = 0.05
SMOKE_TOP1_FLOOR = 0.90


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="GGUF model directory")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rows", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-control-pass", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def _log(enable: bool, message: str) -> None:
    if enable:
        print(message, flush=True)


def _device_bytes(runtime) -> tuple[int, int]:
    free, total = runtime.mem_get_info()
    return int(free), int(total)


def _plan_report(model_path: Path, context: int, chunk: int) -> dict[str, Any]:
    """Read metadata only and report the residency and admission arithmetic."""

    parts = discover_gguf_files(model_path)
    model_map = build_qwen4_exp_gguf_tensor_map(
        tuple(load_gguf_index(part) for part in parts)
    )
    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=chunk)
    config = model_map.config
    runtime = get_hip_runtime()
    free_bytes, total_bytes = _device_bytes(runtime)
    kv_bytes_per_slot = context * config.bf16_kv_bytes_per_token
    raw_index_bytes_per_slot = context * config.fp32_raw_index_bytes_per_token
    gdn_bytes_per_slot = config.layer_types.count("gdn") * (
        config.gdn_time_step_rank * config.gdn_state_size**2 * 4
        + (2 * config.gdn_group_count * config.gdn_state_size + config.gdn_inner_size)
        * config.gdn_conv_kernel
        * 4
    )
    ple_bytes_per_slot = (
        config.residual_branch_count
        * config.hidden_size
        * config.ple_conv_kernel
        * config.ple_ngram_size
        * 4
        if config.ple_layers
        else 0
    )
    per_slot = (
        kv_bytes_per_slot
        + raw_index_bytes_per_slot
        + gdn_bytes_per_slot
        + ple_bytes_per_slot
    )
    return {
        "model_root": str(model_path),
        "part_count": len(parts),
        "first_part": parts[0].name,
        "geometry": {
            "block_count": config.block_count,
            "layer_types": {
                "gdn": config.layer_types.count("gdn"),
                "qsa": config.qsa_layer_count,
            },
            "hidden_size": config.hidden_size,
            "residual_width": config.residual_width,
            "residual_branch_count": config.residual_branch_count,
            "residual_low_rank": config.residual_low_rank,
            "attention_head_count": config.attention_head_count,
            "attention_kv_head_count": config.attention_kv_head_count,
            "attention_key_length": config.attention_key_length,
            "indexer_head_count": config.indexer_head_count,
            "indexer_key_length": config.indexer_key_length,
            "qsa_compression_ratio": config.qsa_compression_ratio,
            "qsa_block_budget": config.qsa_block_budget,
            "qsa_dense_equivalent_max_tokens": config.qsa_dense_equivalent_max_tokens,
            "gdn_group_count": config.gdn_group_count,
            "gdn_time_step_rank": config.gdn_time_step_rank,
            "gdn_state_size": config.gdn_state_size,
            "gdn_inner_size": config.gdn_inner_size,
            "expert_count": config.expert_count,
            "expert_used_count": config.expert_used_count,
            "expert_feed_forward_length": config.expert_feed_forward_length,
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "ple_layers": list(config.ple_layers),
            "ple_ngram_size": config.ple_ngram_size,
            "ple_heads_per_ngram": config.ple_heads_per_ngram,
        },
        "residency": {
            "raw_payload_bytes": int(plan.raw_payload_bytes),
            "device_weight_bytes": int(plan.device_weight_bytes),
            "ple_mmap_bytes": int(plan.ple_mmap_bytes),
            "staging_row_capacity": int(plan.staging_row_capacity),
            "staging_bytes": int(plan.staging_bytes),
        },
        "device": {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "weight_fits": bool(plan.device_weight_bytes < free_bytes),
            "free_after_weights_bytes": int(free_bytes - plan.device_weight_bytes),
        },
        "context": context,
        "per_slot_state_bytes": {
            "kv": int(kv_bytes_per_slot),
            "raw_index": int(raw_index_bytes_per_slot),
            "gdn": int(gdn_bytes_per_slot),
            "ple": int(ple_bytes_per_slot),
            "total": int(per_slot),
        },
    }


def _encode(tokenizer, prompts: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for category, prompt in prompts:
        tokens = [int(token) for token in tokenizer.encode(prompt)]
        if not tokens:
            raise RuntimeError(f"prompt {prompt!r} encoded to no tokens")
        rows.append({"category": category, "prompt": prompt, "tokens": tokens})
    return rows


def _schedule(rows: int, steps: int) -> list[list[int]]:
    """Row sets per step: grow C1 -> C2 -> C4, then stay wide."""

    plan: list[list[int]] = []
    width = 1
    while len(plan) < steps:
        plan.append(list(range(min(width, rows))))
        if len(plan) % 2 == 0:
            width += 1
    return plan


def _download(buffer, nbytes: int, runtime) -> bytes:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    host = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(buffer.ptr, nbytes),
        nbytes,
        runtime=runtime,
    )
    return host.tobytes()


def _slot_digests(batch: Qwen4ExpGGUFBatchRunner, runtime) -> dict[int, dict[str, str]]:
    """Hash one slot's persistent state so a repeat run can be compared exactly."""

    # One hash over every QSA layer's paged K/V, so a defect that lets two
    # layers collide in one cache changes the digest.
    attention_states = batch.attention_states
    attention = attention_states[0]
    digests: dict[int, dict[str, str]] = {}
    row_bytes = attention.kv_heads * attention.head_dim * 2
    block_bytes = attention.block_size * row_bytes
    for slot in range(batch.state_slots):
        blocks = [int(value) for value in attention.block_host[slot]]
        entry: dict[str, str] = {}
        per_layer = []
        for layer_state in attention_states:
            keys = b"".join(
                _download(
                    type(layer_state.key_cache)(
                        layer_state.key_cache.ptr + block * block_bytes, block_bytes
                    ),
                    block_bytes,
                    runtime,
                )
                for block in blocks
            )
            values = b"".join(
                _download(
                    type(layer_state.value_cache)(
                        layer_state.value_cache.ptr + block * block_bytes,
                        block_bytes,
                    ),
                    block_bytes,
                    runtime,
                )
                for block in blocks
            )
            per_layer.append(hashlib.sha256(keys + values).digest())
        entry["kv_sha256"] = hashlib.sha256(b"".join(per_layer)).hexdigest()
        index_digests = []
        for index_state in batch.index_states:
            index_block_bytes = (
                index_state.block_size * index_state.index_dim * 4
            )
            raw = b"".join(
                _download(
                    type(index_state.raw_keys)(
                        index_state.raw_keys.ptr + block * index_block_bytes,
                        index_block_bytes,
                    ),
                    index_block_bytes,
                    runtime,
                )
                for block in blocks
            )
            index_digests.append(hashlib.sha256(raw).hexdigest())
        entry["index_sha256"] = hashlib.sha256(
            "".join(index_digests).encode()
        ).hexdigest()
        entry["index_cursors"] = [
            int(state.cursor_host[slot]) for state in batch.index_states
        ]
        gdn = []
        for gdn_state in batch.gdn_states:
            gdn.append(
                hashlib.sha256(
                    _download(
                        type(gdn_state.conv_state)(
                            gdn_state.conv_slot_ptr(slot),
                            gdn_state.conv_bytes_per_slot,
                        ),
                        gdn_state.conv_bytes_per_slot,
                        runtime,
                    )
                    + _download(
                        type(gdn_state.recurrent_state)(
                            gdn_state.recurrent_slot_ptr(slot),
                            gdn_state.recurrent_bytes_per_slot,
                        ),
                        gdn_state.recurrent_bytes_per_slot,
                        runtime,
                    )
                ).hexdigest()
            )
        entry["gdn_sha256"] = hashlib.sha256("".join(gdn).encode()).hexdigest()
        if batch.ple_state is not None:
            entry["ple_sha256"] = hashlib.sha256(
                _download(
                    type(batch.ple_state.history)(
                        batch.ple_state.history_slot_ptr(slot),
                        batch.ple_state.history_bytes_per_slot,
                    ),
                    batch.ple_state.history_bytes_per_slot,
                    runtime,
                )
            ).hexdigest()
        digests[slot] = entry
    return digests


def _run_pass(
    *,
    batch: Qwen4ExpGGUFBatchRunner,
    references: Sequence[Qwen4ExpGGUFResidentModelRunner],
    owners: Sequence[int],
    rows: int,
    steps: int,
    order: Sequence[int],
    logits_by_row: dict[int, list[np.ndarray]],
    progress: bool,
    reverse_rows: bool = False,
    uniform: bool = False,
) -> list[dict[str, Any]]:
    """Advance the batch and the c1 references in lockstep over one schedule.

    ``order`` maps row position to slot: row ``r`` advances ``order[r]``, and the
    c1 reference for that slot is advanced with the same teacher-forced token.

    ``reverse_rows`` reverses the row order *within* each step while keeping the
    same row set, so every slot still advances the same number of times. That is
    the composition-order invariance check: reversing the schedule itself would
    give slots different histories and compare unrelated states. ``uniform``
    advances every row in every step, which is the schedule the control pass
    needs so that each slot's history is independent of the row order.
    """

    records: list[dict[str, Any]] = []
    schedule = [tuple(range(rows))] * steps if uniform else list(
        _schedule(rows, steps)
    )
    for step, row_set in enumerate(schedule):
        slots = [int(order[index]) for index in row_set]
        if reverse_rows:
            slots = list(reversed(slots))
        tokens = []
        expected = []
        for slot in slots:
            reference = references[slot]
            token = int(np.argmax(reference.last_logits))
            tokens.append(token)
            result = reference.step(token)
            expected.append(result.logits)
        actual = batch.step(
            tokens,
            slots=slots,
            owners=[owners[slot] for slot in slots],
        )
        if actual.slot_ids != tuple(slots):
            raise RuntimeError(
                f"step {step} published slot order {actual.slot_ids}, expected "
                f"{tuple(slots)}"
            )
        for row, slot in enumerate(slots):
            reference = references[slot]
            if int(actual.positions[row]) != int(reference.position) - 1:
                raise RuntimeError(
                    f"step {step} slot {slot} advanced at position "
                    f"{int(actual.positions[row])}, c1 says "
                    f"{int(reference.position) - 1}"
                )
            if batch.owner_of(slot) != owners[slot]:
                raise RuntimeError(f"step {step} slot {slot} lost its owner token")
            if int(actual.tokens[row]) != int(np.argmax(expected[row])):
                raise RuntimeError(
                    f"step {step} slot {slot} greedy id {int(actual.tokens[row])} "
                    f"differs from its own logits' argmax"
                )
            metrics = compare_logits(expected[row], actual.logits[row])
            logits_by_row.setdefault(slot, []).append(
                np.asarray(actual.logits[row], dtype=np.float32).copy()
            )
            records.append(
                {
                    "step": step,
                    "row": row,
                    "slot": slot,
                    "position": int(actual.positions[row]),
                    "token": int(tokens[row]),
                    "c1_top1": int(metrics["teacher_top1"]),
                    "batch_top1": int(metrics["hipengine_top1"]),
                    "top1_agreement": bool(metrics["top1_agreement"]),
                    "kl_c1_to_batch": float(metrics["kl_teacher_to_hipengine"]),
                    "kl_batch_to_c1": float(
                        compare_logits(actual.logits[row], expected[row])[
                            "kl_teacher_to_hipengine"
                        ]
                    ),
                    "max_absolute_logit_error": float(
                        metrics["max_absolute_logit_error"]
                    ),
                    "mean_absolute_logit_error": float(
                        metrics["mean_absolute_logit_error"]
                    ),
                }
            )
        if progress:
            worst = max(
                record["max_absolute_logit_error"]
                for record in records
                if record["step"] == step
            )
            print(
                f"  step {step}: rows {slots} max|c1-batch| {worst:.3e}",
                flush=True,
            )
    return records


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    kl = np.asarray([record["kl_c1_to_batch"] for record in records], dtype=np.float64)
    kl_back = np.asarray(
        [record["kl_batch_to_c1"] for record in records], dtype=np.float64
    )
    top1 = np.asarray(
        [1.0 if record["top1_agreement"] else 0.0 for record in records],
        dtype=np.float64,
    )
    max_abs = np.asarray(
        [record["max_absolute_logit_error"] for record in records], dtype=np.float64
    )
    mean_abs = np.asarray(
        [record["mean_absolute_logit_error"] for record in records], dtype=np.float64
    )
    return {
        "rows": len(records),
        "kl_c1_to_batch": {
            "mean": float(kl.mean()),
            "p95": float(np.percentile(kl, 95)),
            "max": float(kl.max()),
        },
        "kl_batch_to_c1": {
            "mean": float(kl_back.mean()),
            "max": float(kl_back.max()),
        },
        "top1_agreement": float(top1.mean()),
        "max_absolute_logit_error": {
            "mean": float(max_abs.mean()),
            "max": float(max_abs.max()),
        },
        "mean_absolute_logit_error": {
            "mean": float(mean_abs.mean()),
            "max": float(mean_abs.max()),
        },
        "smoke_floor": {
            "kl_max": SMOKE_KL_FLOOR,
            "top1_min": SMOKE_TOP1_FLOOR,
            "kl_pass": bool(kl.max() <= SMOKE_KL_FLOOR),
            "top1_pass": bool(top1.mean() >= SMOKE_TOP1_FLOOR),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    model_path = args.model.expanduser().resolve()
    if args.context <= 0 or args.steps <= 0:
        raise ValueError("--context and --steps must be positive")
    started = perf_counter()
    report: dict[str, Any] = {"command": " ".join(sys.argv)}

    plan_report = _plan_report(model_path, args.context, args.prefill_chunk_size)
    report["plan"] = plan_report
    print(
        "plan: device weights "
        f"{plan_report['residency']['device_weight_bytes'] / GIB:.2f} GiB, "
        f"PLE mmap {plan_report['residency']['ple_mmap_bytes'] / GIB:.2f} GiB, "
        f"free {plan_report['device']['free_bytes'] / GIB:.2f} GiB, "
        f"fits={plan_report['device']['weight_fits']}",
        flush=True,
    )
    per_slot = plan_report["per_slot_state_bytes"]
    print(
        f"per-slot state at context {args.context}: "
        f"kv {per_slot['kv'] / MIB:.1f} MiB, raw index "
        f"{per_slot['raw_index'] / MIB:.1f} MiB, gdn {per_slot['gdn'] / MIB:.1f} MiB, "
        f"ple {per_slot['ple'] / MIB:.1f} MiB, total {per_slot['total'] / MIB:.1f} MiB",
        flush=True,
    )
    if args.plan_only:
        report["elapsed_seconds"] = perf_counter() - started
        _write(args.json_out, report)
        return 0
    if not plan_report["device"]["weight_fits"]:
        raise SystemExit(
            "refusing to materialize: device weights "
            f"{plan_report['residency']['device_weight_bytes'] / GIB:.2f} GiB "
            f"exceed free device bytes "
            f"{plan_report['device']['free_bytes'] / GIB:.2f} GiB"
        )

    # c1's decode is graph-captured as a single-request optimization; the batch
    # runner is eager. Captured and eager execution are different profiles, and
    # with capture on the control pass is not reproducible (measured: reset and
    # re-prefill did not restore the captured state, so pass B differed from
    # pass A on every slot). Off, the control pass is byte-identical. The MoE
    # section is pinned through each runner's explicit control (the retired
    # process env would no longer reach it); per-layer graphs keep their
    # diagnostic env lever.
    os.environ["HIPENGINE_QWEN4_EXP_LAYER_GRAPHS"] = "0"
    eager_moe_graph = False

    generator = None
    batch = None
    references: list[Qwen4ExpGGUFResidentModelRunner] = []
    try:
        index = load_gguf_index(discover_gguf_files(model_path)[0])
        plugin = resolve_model(index.architecture or "")
        runtime = get_hip_runtime()
        reset_memory_stats()
        load_started = perf_counter()
        print(
            "loading weights through the production load path "
            f"({plan_report['residency']['device_weight_bytes'] / GIB:.2f} GiB "
            "device-resident, one shared residency)",
            flush=True,
        )
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model_path,
            weight_index=index,
            model_plugin=plugin,
            backend=args.backend,
            max_sequence_length=args.context,
            resident_capacity=1,
            prefill_chunk_size=args.prefill_chunk_size,
            moe_graph_enabled=eager_moe_graph,
        )
        report["execution_profile_env"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("HIPENGINE_QWEN4_EXP_")
            or key.startswith("HIPENGINE_EXECUTION_PROFILE")
        }
        report["load_seconds"] = perf_counter() - load_started
        report["load_owned_bytes"] = memory_stats()
        free_after_load, _ = _device_bytes(runtime)
        report["device_after_load"] = {
            "free_bytes": free_after_load,
            "weight_bytes_measured": int(
                plan_report["device"]["free_bytes"] - free_after_load
            ),
        }
        print(
            f"loaded in {report['load_seconds']:.1f}s; free device now "
            f"{free_after_load / GIB:.2f} GiB",
            flush=True,
        )
        resident = generator.runner.resident
        rows = int(args.rows)
        prompts = _encode(generator.tokenizer, DEFAULT_PROMPTS[:rows])
        report["prompts"] = [
            {k: v for k, v in row.items() if k != "tokens"} | {
                "token_count": len(row["tokens"])
            }
            for row in prompts
        ]

        for _ in range(rows):
            references.append(
                Qwen4ExpGGUFResidentModelRunner(
                    resident,
                    max_sequence_length=args.context,
                    prefill_chunk_size=args.prefill_chunk_size,
                    backend=args.backend,
                    runtime=runtime,
                    moe_graph_enabled=eager_moe_graph,
                )
            )
        # last_logits is the c1 teacher's own greedy source between steps.
        for reference, row in zip(references, prompts):
            result = reference.prefill_serial(row["tokens"])
            reference.last_logits = np.asarray(result.logits, dtype=np.float32).copy()
            row["prefill_position"] = int(reference.position)

        batch = Qwen4ExpGGUFBatchRunner(
            resident,
            state_slots=rows,
            max_positions=args.context,
            backend=args.backend,
            runtime=runtime,
        )
        free_after_batch, _ = _device_bytes(runtime)
        report["device_after_batch_owner"] = {
            "free_bytes": free_after_batch,
            "batch_state_bytes_measured": int(free_after_load - free_after_batch),
        }
        print(
            f"batch owner for {rows} slots holds "
            f"{(free_after_load - free_after_batch) / MIB:.1f} MiB; free "
            f"{free_after_batch / GIB:.2f} GiB",
            flush=True,
        )

        owners = [batch.admit(slot) for slot in range(rows)]
        import_positions = []
        for slot, row in enumerate(prompts):
            import_positions.append(
                batch.import_prefilled(slot, owners[slot], source=references[slot])
            )
            if import_positions[-1] != row["prefill_position"]:
                raise RuntimeError(
                    f"slot {slot} imported position {import_positions[-1]}, c1 "
                    f"prefilled {row['prefill_position']}"
                )
        report["import"] = {
            "positions": import_positions,
            "poisoned_slots": list(batch.poisoned_slots),
            "owners": owners,
        }

        print("pass A: ascending row order", flush=True)
        logits_a: dict[int, list[np.ndarray]] = {}
        records_a = _run_pass(
            batch=batch,
            references=references,
            owners=owners,
            rows=rows,
            steps=args.steps,
            order=list(range(rows)),
            logits_by_row=logits_a,
            progress=args.progress,
        )
        digests_a = _slot_digests(batch, runtime)
        report["pass_a"] = {
            "records": records_a,
            "summary": _summary(records_a),
            "digests": {str(k): v for k, v in digests_a.items()},
        }
        print(
            "pass A summary: "
            f"kl max {report['pass_a']['summary']['kl_c1_to_batch']['max']:.3e}, "
            f"top1 {report['pass_a']['summary']['top1_agreement']:.3f}, "
            f"max|dlogit| "
            f"{report['pass_a']['summary']['max_absolute_logit_error']['max']:.3e}",
            flush=True,
        )

        if not args.skip_control_pass:
            # Both control passes advance every row in every step from a fresh
            # import, so each slot's history is identical between them and the
            # only difference is the row order inside each step. Comparing the
            # ladder schedule of pass A against a reversed ladder would compare
            # slots that advanced a different number of times.
            def _fresh_uniform_pass(
                *, reverse_rows: bool, label: str
            ) -> tuple[dict[int, dict[str, str]], dict[int, list[np.ndarray]]]:
                print(
                    f"control pass {label}: reset, re-import, "
                    f"{'reversed' if reverse_rows else 'ascending'} row order",
                    flush=True,
                )
                for slot, owner in enumerate(owners):
                    batch.reset_slot(slot, owner)
                    if batch.is_poisoned(slot):
                        raise RuntimeError(
                            f"slot {slot} stayed poisoned after reset"
                        )
                for reference, row in zip(references, prompts):
                    reference.reset()
                    result = reference.prefill_serial(row["tokens"])
                    reference.last_logits = np.asarray(
                        result.logits, dtype=np.float32
                    ).copy()
                for slot, row in enumerate(prompts):
                    position = batch.import_prefilled(
                        slot, owners[slot], source=references[slot]
                    )
                    if position != row["prefill_position"]:
                        raise RuntimeError(
                            f"control import of slot {slot} landed at {position}"
                        )
                logits: dict[int, list[np.ndarray]] = {}
                _run_pass(
                    batch=batch,
                    references=references,
                    owners=owners,
                    rows=rows,
                    steps=args.steps,
                    order=list(range(rows)),
                    logits_by_row=logits,
                    progress=args.progress,
                    reverse_rows=reverse_rows,
                    uniform=True,
                )
                return _slot_digests(batch, runtime), logits

            digests_c, logits_c = _fresh_uniform_pass(
                reverse_rows=False, label="A2"
            )
            digests_b, logits_b = _fresh_uniform_pass(
                reverse_rows=True, label="B"
            )
            identical = {
                str(slot): bool(
                    digests_c[slot] == digests_b[slot]
                    and all(
                        np.array_equal(first, second)
                        for first, second in zip(
                            logits_c[slot], logits_b[slot]
                        )
                    )
                )
                for slot in range(rows)
            }
            report["control_pass_b"] = {
                "row_order": "reversed within each step",
                "baseline_digests": {str(k): v for k, v in digests_c.items()},
                "digests": {str(k): v for k, v in digests_b.items()},
                "identical_to_ascending": identical,
            }
            if not all(identical.values()):
                raise RuntimeError(
                    "control pass diverged from the ascending uniform pass: "
                    + json.dumps(identical, indent=2)
                )
            print(
                "control pass B: every slot's state and logits are byte-identical "
                "to the ascending uniform pass",
                flush=True,
            )

        report["final"] = {
            "positions": [batch.position(slot) for slot in range(rows)],
            "poisoned_slots": list(batch.poisoned_slots),
            "owned": [batch.owner_of(slot) for slot in range(rows)],
            "free_slots": list(batch.free_slots),
        }
        free_end, _ = _device_bytes(runtime)
        report["device_end"] = {"free_bytes": free_end}
        report["owned_peak_bytes"] = memory_stats()
        report["max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report["elapsed_seconds"] = perf_counter() - started
        _write(args.json_out, report)
        print(
            f"done in {report['elapsed_seconds']:.1f}s; smoke floor kl_pass="
            f"{report['pass_a']['summary']['smoke_floor']['kl_pass']} "
            f"top1_pass={report['pass_a']['summary']['smoke_floor']['top1_pass']}",
            flush=True,
        )
        return 0
    finally:
        if batch is not None:
            batch.close()
        for reference in references:
            reference.close()
        if generator is not None:
            generator.close()


def _write(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
