#!/usr/bin/env python3
"""A/B the Qwen4Exp steady-state decode synchronization boundary.

This is a diagnostic, not a promotion gate. The no-sync arm monkeypatches only
`runtime.device_synchronize` during the measured loop, restores it afterwards,
and requires identical generated IDs and logits hashes before the timing rows
can be interpreted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

SAFE43_DP4A_LAYERS = tuple([0, 2, 5, 6, 8, 9, 10, 11] + list(range(13, 48)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--pair-repetitions", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def configure_production_routes() -> None:
    os.environ.update(
        {
            "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": "35-47",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": "27-47",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS": "",
            "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS": "35-47",
            "HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1",
            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS": ",".join(map(str, SAFE43_DP4A_LAYERS)),
        }
    )


def main() -> None:
    args = build_parser().parse_args()
    configure_production_routes()

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root,
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151",
        max_sequence_length=args.max_sequence_length,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    rows: list[dict[str, Any]] = []
    try:
        runner = generator.runner
        runner.configure_mmq_prefill_resources()
        runtime = runner.runtime
        original_sync = runtime.device_synchronize

        def run(mode: str) -> dict[str, Any]:
            runtime.device_synchronize = original_sync
            result = runner.prefill([9707])
            for _ in range(args.warmup_steps):
                result = runner.step(int(result.token_id))
            original_sync()
            if mode == "no_redundant_sync":
                runtime.device_synchronize = lambda: None
            ids: list[int] = []
            logits_sha256: list[str] = []
            started = time.perf_counter()
            for _ in range(args.steps):
                result = runner.step(int(result.token_id))
                ids.append(int(result.token_id))
                logits_sha256.append(hashlib.sha256(result.logits.tobytes()).hexdigest())
            elapsed = time.perf_counter() - started
            runtime.device_synchronize = original_sync
            original_sync()
            row = {
                "mode": mode,
                "seconds": elapsed,
                "tok_s": args.steps / elapsed,
                "ids": ids,
                "logits_sha256": logits_sha256,
            }
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("mode", "seconds", "tok_s")}), flush=True)
            return row

        run("sync")
        run("no_redundant_sync")
        for _ in range(args.pair_repetitions):
            run("sync")
            run("no_redundant_sync")
            run("no_redundant_sync")
            run("sync")
    finally:
        generator.close()

    sync_seconds = [row["seconds"] for row in rows if row["mode"] == "sync"]
    nosync_seconds = [row["seconds"] for row in rows if row["mode"] == "no_redundant_sync"]
    report = {
        "schema": 1,
        "kind": "qwen4exp_decode_sync_ab",
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "sync_median_seconds": statistics.median(sync_seconds),
        "no_sync_median_seconds": statistics.median(nosync_seconds),
        "sync_tok_s": args.steps / statistics.median(sync_seconds),
        "no_sync_tok_s": args.steps / statistics.median(nosync_seconds),
        "generated_ids_repeat_exact": len({tuple(row["ids"]) for row in rows}) == 1,
        "logits_repeat_exact": len({tuple(row["logits_sha256"]) for row in rows}) == 1,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
