#!/usr/bin/env python3
"""Attribute one Qwen3.8 GGUF server configuration's device memory on one GPU.

This is the capacity campaign's Packet 2 allocation ledger: it combines a
device-free planned weight census with in-process live snapshots so the fixed
intercept (weights, pool backing, workspace lease, graphs) separates from the
context-proportional and request-proportional parts.

Snapshots come from ``runner.kv_pool_memory_snapshot()`` (pool contract,
storage view, dynamic pool stats, packed-workspace lease, tracked allocator,
sampled HIP current/peak) plus whole-card sysfs sampling through the request.
The planned census groups aliases by source identity and reports canonical
versus alternate-layout bytes, so resident payload — not GGUF file size — is
the weight baseline.

Diagnostic harness: it changes no route and makes no throughput claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402

GIB = 1 << 30


def _card_unique_id(card: Any) -> str | None:
    try:
        return (card.sysfs_path / "unique_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _planned_census(model_path: Path, backend: str) -> dict[str, Any]:
    """Device-free planned weight census with production layout flags."""

    from hipengine.kernels.backends import backend_package_capability
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_materialize import (
        GGUF_SELECTIVE_WEIGHT_ARENA_MAX_ALLOCATION_BYTES,
        plan_qwen35_gguf_materialization,
    )
    from hipengine.loading.qwen35_gguf_residency import (
        census_qwen35_gguf_weight_specs,
    )

    reader = GGUFReader(model_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    file_type_name = getattr(reader.info, "file_type_name", None)
    raw_qmicro_file_types = backend_package_capability(
        backend,
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
        (),
    )
    qmicro_file_types = (
        frozenset(str(item) for item in raw_qmicro_file_types)
        if isinstance(raw_qmicro_file_types, (tuple, list, set, frozenset))
        else frozenset()
    )
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=None,
        dense_q4_t16=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q4_T16", False)
        ),
        dense_q4_qmicro_t16_gate_up=(
            bool(
                backend_package_capability(
                    backend, "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP", False
                )
            )
            and file_type_name in qmicro_file_types
        ),
        dense_q4_t16_attn_q_08b=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q4_T16_ATTN_Q_08B", False)
        ),
        dense_q5_t16_ssm_out=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q5_T16_SSM_OUT", False)
        ),
        dense_q5_raw_mmq_ssm_out=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q5_RAW_MMQ_SSM_OUT", False)
        ),
        dense_q5_qmicro_planar_ssm_out=bool(
            backend_package_capability(
                backend, "GGUF_DENSE_Q5_QMICRO_PLANAR_SSM_OUT", False
            )
        ),
        dense_q5_t16_ssm_out_08b=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q5_T16_SSM_OUT_08B", False)
        ),
        dense_q5_t16_qkv=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q5_T16_QKV", False)
        ),
        dense_q5_t16_h5120=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q5_T16_H5120", False)
        ),
        dense_q6_qmicro_planar=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q6_QMICRO_PLANAR", False)
        ),
    )
    census = census_qwen35_gguf_weight_specs(plan.specs)
    nextn_tensors = [
        tensor.name
        for tensor in reader.info.tensors
        if ".nextn." in str(tensor.name)
    ]
    nextn_bytes = sum(
        int(tensor.nbytes)
        for tensor in reader.info.tensors
        if ".nextn." in str(tensor.name)
    )
    return {
        "logical_tensor_count": census.logical_tensor_count,
        "alias_count": census.alias_count,
        "source_nbytes": census.source_nbytes,
        "resident_nbytes": census.resident_nbytes,
        "alternate_layout_nbytes": census.alternate_layout_nbytes,
        "issues": list(census.issues),
        "gguf_nextn_tensor_count": len(nextn_tensors),
        "gguf_nextn_source_nbytes": nextn_bytes,
        "gguf_file_bytes": model_path.stat().st_size,
    }


def _snapshot(runner: Any) -> dict[str, Any]:
    snapshot = runner.kv_pool_memory_snapshot()
    from hipengine.core.memory import live_allocation_histogram

    snapshot["large_allocations"] = live_allocation_histogram(min_bytes=64 << 20)
    return json.loads(json.dumps(snapshot, default=str))


def _sysfs_used_bytes(card: Any) -> int:
    return int(card.vram_used_path.read_text().strip())


def _kv_layout_audits(runner: Any) -> list[dict[str, Any]]:
    """Per-session persistent KV ownership audits (shadow-byte evidence)."""

    audits: list[dict[str, Any]] = []
    for session in runner._resident_sessions():
        audit_fn = getattr(session, "device_kv_layout_audit", None)
        if callable(audit_fn):
            audits.append(audit_fn())
    return audits


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model).resolve()
    card = select_card(pci_id=args.pci_id)
    # Bind the process to the target GPU before any HIP context exists.
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ["GPU_MAX_HW_QUEUES"] = "1"

    planned = _planned_census(model_path, args.backend)

    from hipengine.llm import LLM, SamplingParams

    sampler = VramSampler(card, interval_ms=20.0, keep_samples=True)

    llm = LLM(
        str(model_path),
        backend=args.backend,
        quant=args.quant,
        max_active_requests=args.max_active_requests,
        max_sequence_length=args.context,
        kv_storage=args.kv_storage,
        kv_scale_dtype=args.kv_scale_dtype,
        kv_scale_granularity=args.kv_scale_granularity,
    )
    adapter = llm._get_text_generator()
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        kv_storage=args.kv_storage,
        kv_scale_dtype=args.kv_scale_dtype,
        kv_scale_granularity=args.kv_scale_granularity,
    )
    llm.prepare(max_sequence_length=args.context, sampling_params=sampling)
    runner = adapter._runner

    load_snapshot = _snapshot(runner)
    load_sysfs = _sysfs_used_bytes(card)
    sampler.start()

    mtp_warmup_info: dict[str, Any] | None = None
    mtp_after_snapshot: dict[str, Any] | None = None
    mtp_after_sysfs: int | None = None
    if args.speculative_mtp_serving == "enabled":
        preparer = getattr(adapter, "prepare_request_scratch", None)
        if not callable(preparer):
            mtp_warmup_info = {
                "status": "unsupported",
                "reason": "generator has no prepare_request_scratch",
            }
        else:
            warmup_env = "HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP"
            previous = os.environ.get(warmup_env)
            os.environ[warmup_env] = "1"
            try:
                probe = preparer(
                    max_prompt_tokens=int(
                        min(int(args.mtp_probe_prompt_tokens), int(args.context))
                    ),
                    max_new_tokens=0,
                    sampling_params=sampling,
                    max_batch_size=max(1, int(args.max_active_requests)),
                    release_after_probe=True,
                )
            except Exception as exc:  # noqa: BLE001 - record and keep measuring
                mtp_warmup_info = {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:300],
                }
            else:
                mtp_warmup_info = {
                    "status": "warmed",
                    "probe": probe,
                    "packed_mtp_prefill_widths": list(
                        probe.get("packed_mtp_prefill_widths") or []
                    ),
                    "packed_mtp_verify_widths": list(
                        probe.get("packed_mtp_verify_widths") or []
                    ),
                    "packed_mtp_prefill_skipped": bool(
                        probe.get("packed_mtp_prefill_skipped")
                    ),
                    "packed_mtp_prefill_reason": probe.get(
                        "packed_mtp_prefill_reason"
                    ),
                }
            finally:
                if previous is None:
                    os.environ.pop(warmup_env, None)
                else:
                    os.environ[warmup_env] = previous
            if mtp_warmup_info.get("status") == "warmed":
                mtp_after_snapshot = _snapshot(runner)
                mtp_after_sysfs = _sysfs_used_bytes(card)
                # Functional note: at N=1 the packed MTP warmup is width-gated
                # and post-hoc draft acquisition on an AR-only load cannot work
                # (the materialization plan deliberately omitted NextN), so the
                # K0->K resident delta is measured server-side by the context
                # ceiling probe with --speculative-mtp-serving enabled (its
                # load plans NextN and the startup warmup engages it).
                mtp_warmup_info["k0_k_delta_vehicle"] = (
                    "gguf_context_ceiling_probe --speculative-mtp-serving enabled"
                )
                if mtp_warmup_info.get("status") == "warmed":
                    mtp_after_snapshot = _snapshot(runner)
                    mtp_after_sysfs = _sysfs_used_bytes(card)

    request_info: dict[str, Any] | None = None
    if args.run_request:
        tokenizer = runner.generator.tokenizer
        unit = tokenizer.encode("The harbourmaster records each arriving vessel.")
        target = int(args.context - args.max_tokens)
        filler_ids = (unit * (target // len(unit) + 1))[: max(1, target - 2)]
        prompt = tokenizer.decode(filler_ids, skip_special=False) + "\n\nAnswer: ledger"
        prompt_ids = tokenizer.encode(prompt)
        request_info = {"prompt_tokens": len(prompt_ids)}

        from hipengine.core.memory import live_allocation_histogram
        import threading

        peak_histogram: dict[str, int] = {"total_bytes": -1}
        stop = threading.Event()

        def _sample_histograms() -> None:
            while not stop.is_set():
                histogram = live_allocation_histogram(min_bytes=64 << 20)
                if histogram["total_bytes"] > peak_histogram["total_bytes"]:
                    peak_histogram.update(histogram)
                time.sleep(0.05)

        histogram_thread = threading.Thread(target=_sample_histograms, daemon=True)
        histogram_thread.start()
        try:
            started = time.perf_counter()
            outputs = llm.generate([prompt], sampling)
            elapsed = time.perf_counter() - started
        finally:
            stop.set()
            histogram_thread.join(timeout=1.0)
        request_info["elapsed_s"] = round(elapsed, 3)
        request_info["output_chars"] = len(outputs[0]) if outputs else 0
        request_info["peak_live_large_allocations"] = dict(peak_histogram)

    sampler.stop()
    request_sysfs_peak = sampler.result().peak_bytes
    after_snapshot = _snapshot(runner)
    after_sysfs = _sysfs_used_bytes(card)
    kv_layout_audits = _kv_layout_audits(runner)

    return {
        "schema": 1,
        "kind": "qwen38_xtx_allocation_ledger",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "performance_claim": False,
        "diagnostic": True,
        "config": {
            "model": str(model_path),
            "backend": args.backend,
            "quant": args.quant,
            "kv_storage": args.kv_storage,
            "kv_scale_dtype": None if args.kv_storage == "bf16" else args.kv_scale_dtype,
            "kv_scale_granularity": (
                None if args.kv_storage == "bf16" else args.kv_scale_granularity
            ),
            "context_tokens": args.context,
            "max_active_requests": args.max_active_requests,
            "max_tokens": args.max_tokens,
            "run_request": bool(args.run_request),
            "speculative_mtp_serving": str(args.speculative_mtp_serving),
        },
        "device": {
            "pci_id": card.pci_id,
            "drm_card": card.card_name,
            "unique_id": _card_unique_id(card),
            "vram_total_bytes": card.vram_total_bytes,
            "hip_visible_devices": str(args.gpu_index),
        },
        "planned_weight_census": planned,
        "snapshots": {
            "after_load": load_snapshot,
            "after_load_sysfs_used_bytes": load_sysfs,
            "after_mtp_warmup": mtp_after_snapshot,
            "after_mtp_warmup_sysfs_used_bytes": mtp_after_sysfs,
            "after_request": after_snapshot,
            "after_request_sysfs_used_bytes": after_sysfs,
            "request_sysfs_peak_bytes": request_sysfs_peak,
            "kv_layout_audits": kv_layout_audits,
        },
        "mtp_warmup": mtp_warmup_info,
        "request": request_info,
        "metric_note": (
            "tracked_allocator counts hipengine malloc/free bookkeeping; "
            "dynamic_pool is the shared KV pool's own accounting; sysfs is "
            "whole-card committed VRAM, which includes driver/runtime overhead "
            "the tracked allocator cannot see."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf", type=Path)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--kv-storage", default="bf16")
    parser.add_argument("--kv-scale-dtype", default="fp32")
    parser.add_argument("--kv-scale-granularity", default="per_token_head")
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--max-active-requests", default=1, type=int)
    parser.add_argument("--max-tokens", default=16, type=int)
    parser.add_argument("--run-request", action="store_true")
    parser.add_argument(
        "--speculative-mtp-serving",
        default="off",
        choices=("off", "enabled"),
        help=(
            "off measures the AR-only configuration (NextN unplanned, no draft "
            "assets); enabled warms the MTP serving route after the load "
            "snapshot (the server startup-warmup path) and records the K0->K "
            "delta: draft weights, runner/session and graph assets that stay "
            "pooled after engagement"
        ),
    )
    parser.add_argument("--mtp-probe-prompt-tokens", default=512, type=int)
    parser.add_argument("--pci-id", default="0000:10:00.0")
    parser.add_argument("--gpu-index", default=1, type=int)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "1")
    artifact = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2) + "\n")
    tracked = artifact["snapshots"]["after_load"]["tracked_allocator"]
    pool = artifact["snapshots"]["after_load"].get("dynamic_pool") or {}
    print(
        json.dumps(
            {
                "context": args.context,
                "kv_storage": args.kv_storage,
                "resident_weights_gib": round(
                    artifact["planned_weight_census"]["resident_nbytes"] / GIB, 3
                ),
                "tracked_active_gib": round(
                    tracked.get("active_allocated_bytes", 0) / GIB, 3
                ),
                "tracked_peak_gib": round(
                    tracked.get("peak_allocated_bytes", 0) / GIB, 3
                ),
                "pool_pages": pool.get("current_pages"),
                "workspace_lease_pages": artifact["snapshots"]["after_load"][
                    "packed_workspace_lease_pages"
                ],
                "load_sysfs_gib": round(
                    artifact["snapshots"]["after_load_sysfs_used_bytes"] / GIB, 3
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
