#!/usr/bin/env python3
"""StepFun split-GGUF resident load smoke for Strix Halo/GTT validation."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import HipError, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc, memory_stats, reset_memory_stats
from hipengine.loading.gguf import scan_gguf_splits
from hipengine.loading.stepfun_gguf import StepFunGGUFConfig, build_stepfun_gguf_tensor_map
from hipengine.loading.stepfun_gguf_materialize import (
    materialize_stepfun_gguf_weights,
    plan_stepfun_gguf_materialization,
)
from hipengine.runtime.stepfun_gguf_runner import (
    StepFunShortContextDecodePlanner,
    StepFunTextDecodeResourcePlan,
    stepfun_kv_cache_layer_nbytes,
    stepfun_kv_cache_nbytes as runtime_stepfun_kv_cache_nbytes,
)
from hipengine.tokenization.gguf import StepFunGGUFTokenizer

DEFAULT_MODEL_DIR = Path("/data/models/gguf")
DEFAULT_PATTERN = "Step-3.7-flash-Q3_K_L-*.gguf"
BOOT_CONFIG = Path("/etc/modprobe.d/amdgpu_llm_optimized.conf")


def _write_text_atomic(output: Path, text: str) -> None:
    """Atomically write text by replacing the destination with a flushed temp file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    _write_text_atomic(output, text)


def _stepfun_kv_cache_nbytes(
    config: StepFunGGUFConfig,
    *,
    context_pages: int,
    page_size: int,
) -> int:
    if context_pages <= 0:
        return 0
    return runtime_stepfun_kv_cache_nbytes(
        config,
        context_pages=context_pages,
        page_size=page_size,
    )


def _allocate_stepfun_kv_cache(
    config: StepFunGGUFConfig,
    *,
    context_pages: int,
    page_size: int,
    runtime,
) -> list[DeviceBuffer]:
    if context_pages <= 0:
        return []
    buffers: list[DeviceBuffer] = []
    try:
        for layer_id, (key_nbytes, value_nbytes) in enumerate(
            stepfun_kv_cache_layer_nbytes(
                config,
                context_pages=context_pages,
                page_size=page_size,
            )
        ):
            buffers.append(malloc(key_nbytes, runtime=runtime))
            buffers.append(malloc(value_nbytes, runtime=runtime))
            print(
                f"[kv_alloc] layer={layer_id} key={key_nbytes} value={value_nbytes}",
                file=sys.stderr,
                flush=True,
            )
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    return buffers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--selected-slot",
        action="append",
        default=None,
        help="Optional StepFun slot path to load; repeat for selected-slot smoke. Omit to load all weights.",
    )
    parser.add_argument(
        "--kv-context-pages",
        type=int,
        default=0,
        help="Optionally allocate a synthetic BF16 KV cache with this many pages after weight load.",
    )
    parser.add_argument(
        "--kv-page-size",
        type=int,
        default=512,
        help="Tokens per synthetic KV page when --kv-context-pages is non-zero.",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help="Scan metadata and print the materialization/resource plan without HIP allocation.",
    )
    parser.add_argument(
        "--kv-run-plan-message",
        default="hello",
        help="Chat user message used for the metadata-only KV decode run plan when KV planning is enabled.",
    )
    parser.add_argument(
        "--kv-run-plan-reasoning-effort",
        default="low",
        help="Reasoning-effort template value used for the metadata-only KV decode run plan.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write JSON output atomically to this path instead of stdout. Useful for "
            "refreshing resource artifacts without exposing truncated files to pollers."
        ),
    )
    args = parser.parse_args(argv)

    if args.kv_context_pages < 0:
        raise ValueError("--kv-context-pages must be non-negative")
    if args.kv_page_size <= 0:
        raise ValueError("--kv-page-size must be positive")

    paths = tuple(sorted(args.model_dir.glob(args.pattern)))
    if not paths:
        raise FileNotFoundError(f"no GGUF shards matching {args.model_dir / args.pattern}")

    started = time.perf_counter()
    snapshots: list[dict[str, object]] = []
    runtime = None

    def snap(label: str) -> dict[str, object]:
        if runtime is None:  # pragma: no cover - guarded by non-dry-run flow
            raise RuntimeError("HIP runtime is not initialized")
        free_bytes, total_bytes = runtime.mem_get_info()
        stats = memory_stats()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row = {
            "label": label,
            "hip_free_bytes": int(free_bytes),
            "hip_total_bytes": int(total_bytes),
            "hip_free_gib": free_bytes / 2**30,
            "hip_total_gib": total_bytes / 2**30,
            "hipengine_memory_stats": stats,
            "max_rss_kib": int(usage.ru_maxrss),
            "elapsed_s": time.perf_counter() - started,
        }
        snapshots.append(row)
        print(
            f"[{label}] hip_free={row['hip_free_gib']:.3f} GiB hip_total={row['hip_total_gib']:.3f} GiB",
            file=sys.stderr,
            flush=True,
        )
        return row

    if not args.dry_run_plan:
        runtime = get_hip_runtime()
        reset_memory_stats()
        snap("before_scan")
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    plan = plan_stepfun_gguf_materialization(model_map)
    if not args.dry_run_plan:
        snap("after_plan")

    selected_slots = None if args.selected_slot is None else tuple(args.selected_slot)
    weights = None
    kv_buffers: list[DeviceBuffer] = []
    text_decode_resource_plan = None
    kv_decode_run_plan = None
    if args.kv_context_pages:
        text_decode_resource_plan = StepFunTextDecodeResourcePlan.from_model_map(
            model_map,
            backend="hip_gfx1151",
            context_pages=args.kv_context_pages,
            page_size=args.kv_page_size,
        )
        decode_planner = StepFunShortContextDecodePlanner(
            info=info,
            model_map=model_map,
            tokenizer=StepFunGGUFTokenizer.from_gguf_info(info),
            backend="hip_gfx1151",
            max_context=args.kv_context_pages * args.kv_page_size,
            max_new_tokens=1,
        )
        kv_decode_run_plan = decode_planner.plan_kv_decode_chat(
            [{"role": "user", "content": args.kv_run_plan_message}],
            reasoning_effort=args.kv_run_plan_reasoning_effort,
            context_pages=args.kv_context_pages,
            page_size=args.kv_page_size,
        ).to_dict()
    kv_nbytes = _stepfun_kv_cache_nbytes(
        model_map.config,
        context_pages=args.kv_context_pages,
        page_size=args.kv_page_size,
    )
    if args.dry_run_plan:
        result = {
            "status": "planned",
            "error": None,
            "model_dir": str(args.model_dir),
            "pattern": args.pattern,
            "paths": [str(path) for path in paths],
            "split_count": info.split_count,
            "tensor_count": len(info.tensors),
            "plan_tensor_count": len(plan.specs),
            "plan_total_nbytes": plan.total_nbytes,
            "plan_total_gib": plan.total_nbytes / 2**30,
            "quant_counts": dict(plan.quant_counts),
            "selected_slots": selected_slots,
            "loaded_weight_count": 0,
            "loaded_nbytes": 0,
            "kv_context_pages": args.kv_context_pages,
            "kv_page_size": args.kv_page_size,
            "kv_buffer_count": 0 if not args.kv_context_pages else model_map.config.block_count * 2,
            "kv_nbytes": kv_nbytes,
            "kv_gib": kv_nbytes / 2**30,
            "text_decode_resource_plan": None
            if text_decode_resource_plan is None
            else text_decode_resource_plan.to_dict(),
            "kv_decode_run_plan": kv_decode_run_plan,
            "boot_config_path": str(BOOT_CONFIG),
            "boot_config_text": BOOT_CONFIG.read_text() if BOOT_CONFIG.exists() else None,
            "snapshots": snapshots,
        }
        _emit_json(result, pretty=args.pretty, output=args.output)
        return 0

    status = "unknown"
    error: dict[str, object] | None = None
    try:
        print(
            f"[load_start] tensors={len(plan.specs)} bytes={plan.total_nbytes} selected={selected_slots}",
            file=sys.stderr,
            flush=True,
        )
        weights = materialize_stepfun_gguf_weights(
            info,
            selected_slots=selected_slots,
            runtime=runtime,
        )
        snap("after_load")
        if args.kv_context_pages:
            kv_buffers = _allocate_stepfun_kv_cache(
                model_map.config,
                context_pages=args.kv_context_pages,
                page_size=args.kv_page_size,
                runtime=runtime,
            )
            runtime.device_synchronize()
            snap("after_kv_alloc")
            for buffer in reversed(kv_buffers):
                free(buffer, runtime=runtime)
            kv_buffers = []
            runtime.device_synchronize()
            snap("after_kv_free")
        status = "loaded"
    except HipError as exc:
        status = "hip_error"
        error = {"type": type(exc).__name__, "message": str(exc), "code": exc.code}
        snap("after_error")
    except Exception as exc:  # pragma: no cover - diagnostic path
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        snap("after_error")
    finally:
        for buffer in reversed(kv_buffers):
            free(buffer, runtime=runtime)
        kv_buffers = []
        if weights is not None:
            weights.free(runtime=runtime)
            runtime.device_synchronize()
            snap("after_free")

    result = {
        "status": status,
        "error": error,
        "model_dir": str(args.model_dir),
        "pattern": args.pattern,
        "paths": [str(path) for path in paths],
        "split_count": info.split_count,
        "tensor_count": len(info.tensors),
        "plan_tensor_count": len(plan.specs),
        "plan_total_nbytes": plan.total_nbytes,
        "plan_total_gib": plan.total_nbytes / 2**30,
        "quant_counts": dict(plan.quant_counts),
        "selected_slots": selected_slots,
        "loaded_weight_count": 0 if weights is None else len(weights.weights),
        "loaded_nbytes": 0 if weights is None else weights.allocated_nbytes,
        "kv_context_pages": args.kv_context_pages,
        "kv_page_size": args.kv_page_size,
        "kv_buffer_count": 0 if not args.kv_context_pages else model_map.config.block_count * 2,
        "kv_nbytes": kv_nbytes,
        "kv_gib": kv_nbytes / 2**30,
        "text_decode_resource_plan": None
        if text_decode_resource_plan is None
        else text_decode_resource_plan.to_dict(),
        "kv_decode_run_plan": kv_decode_run_plan,
        "boot_config_path": str(BOOT_CONFIG),
        "boot_config_text": BOOT_CONFIG.read_text() if BOOT_CONFIG.exists() else None,
        "snapshots": snapshots,
    }
    _emit_json(result, pretty=args.pretty, output=args.output)
    return 0 if status == "loaded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
