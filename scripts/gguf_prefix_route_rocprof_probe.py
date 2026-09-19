#!/usr/bin/env python3
"""Route-forced kernel comparison of the prefix-hit suffix-prefill paths.

A prefix-cache hit whose device-KV pages land contiguously keeps the
slot-local prefill route (per-slot AOTriton flash varlen attention).  A gapped
placement drops the packed slab to the native paged prefill kernel, which the
placement cost model measured at a large per-token multiple.  This probe
forces each route on identical prefix/suffix geometry through the production
resident runner so the gap can be attributed to kernel families under
rocprofv3 before committing to a kernel design.

Arms:

* ``hit-contiguous`` - radix hit, natural contiguous placement; the packed
  slab keeps the slot-local AOTriton route.
* ``hit-paged`` - same radix hit, but
  ``_gguf_device_kv_contiguous_base_row`` is patched to report the
  continuation session as non-contiguous, which is exactly the route decision
  a gapped placement produces (``device_kv_nonidentity_scatter`` -> the
  native paged prefill kernel).  The pages themselves stay wherever the
  allocator placed them; the paged kernel walks the block table either way.
* ``miss`` - ``prefix_cache=off`` full-prompt prefill: the hit's alternative.

Modes:

* ``--mode time`` runs the requested arms in-process with host timing only.
* ``--mode rocprof`` runs each requested arm as a child once under
  ``rocprofv3 --kernel-trace --hip-runtime-trace`` (after a non-profiled
  warmup child that prebuilds the JIT cache) and prints per-kernel-name
  tables plus the arm diff.  The profiled children run with
  ``HIPENGINE_REQUIRE_CACHED_BUILD=1`` and a pinned compiler-version file so
  no ``hipcc`` spawns under the profiler.
* ``--mode child`` is the single-arm payload the profiler wraps.

This is a diagnostic artifact only; no performance claim is retained from
this script.  The production benchmark for retained claims is
``scripts/gguf_prefix_reuse_bench.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
DEFAULT_COMPILER_VERSION_FILE = Path("/tmp/hipengine-hipcc-version.txt")
ARMS = ("hit-contiguous", "hit-paged", "miss")


def _request(prompt: tuple[int, ...], *, max_tokens: int) -> Any:
    from hipengine.generation.registry import GenerationRequest

    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _prefill_work(request_id: int, tokens: tuple[int, ...]) -> Any:
    from hipengine.dispatch import WorkItem, WorkKind

    return WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=(int(request_id),),
        row_to_request=(int(request_id),),
        token_rows=(tokens,),
    )


class _PagedRouteForce:
    """Patch the packed-slab route decision for chosen sessions only.

    ``hipengine.runtime.qwen35_gguf_runner._gguf_device_kv_contiguous_base_row``
    is consulted when the packed slab decides slot-local versus paged
    attention.  Returning ``None`` for an armed session reproduces the exact
    branch a gapped (non-contiguous) device-KV placement takes, without
    depending on the allocator actually fragmenting.
    """

    def __init__(self) -> None:
        import hipengine.runtime.qwen35_gguf_runner as runtime_module

        self._module = runtime_module
        self._original = runtime_module._gguf_device_kv_contiguous_base_row
        self._armed: set[int] = set()
        runtime_module._gguf_device_kv_contiguous_base_row = self._gated

    def _gated(self, session: Any, **kwargs: Any) -> int | None:
        if id(session) in self._armed:
            return None
        return self._original(session, **kwargs)

    def arm(self, session: Any) -> None:
        self._armed.add(id(session))

    def close(self) -> None:
        self._module._gguf_device_kv_contiguous_base_row = self._original


def _counter(snapshot: Mapping[str, Any], name: str) -> int:
    return int(snapshot.get("prefix_cache", {}).get(name, 0))


def _snapshot_counters(runner: Any) -> dict[str, int]:
    try:
        snapshot = runner.observability_snapshot()
    except Exception:
        return {}
    return {
        "usable_hits": _counter(snapshot, "usable_hits"),
        "admission_fallbacks": _counter(snapshot, "admission_fallbacks"),
        "contiguous_admissions": _counter(snapshot, "contiguous_admissions"),
        "gapped_admissions": _counter(snapshot, "gapped_admissions"),
    }


def _session_plan(runner: Any, session: Any) -> dict[str, Any]:
    """Read the last packed-prefill plan from the batch-owner session."""

    candidates = [session]
    try:
        owner = runner._packed_execution_owner(session)
        if owner is not None and owner is not session:
            candidates.insert(0, owner)
    except Exception:
        pass
    for candidate in candidates:
        plan = getattr(candidate, "last_packed_prefill_plan", None)
        if isinstance(plan, dict) and "device_kv_nonidentity_scatter" in plan:
            return {
                key: value
                for key, value in plan.items()
                if key
                in (
                    "device_kv_nonidentity_scatter",
                    "device_kv_contiguous_base_rows",
                    "device_kv_shifted_contiguous_rebase",
                    "gapped_slot_local_gather",
                    "gapped_gather_slots",
                    "executor_mode",
                )
            }
    return {}


def _model_config(runner: Any) -> dict[str, Any]:
    """Best-effort head/layer geometry for per-kernel normalization."""

    info: dict[str, Any] = {}
    try:
        cfg = runner._shared_runner.weights.config
        layer_types = list(cfg.layer_types)
        info.update(
            {
                "head_count": int(cfg.head_count),
                "head_count_kv": int(cfg.head_count_kv),
                "key_length": int(cfg.key_length),
                "hidden_size": int(runner._shared_runner.hidden_size),
                "layer_types": {
                    str(kind): int(layer_types.count(kind)) for kind in sorted(set(layer_types))
                },
            }
        )
    except Exception as exc:
        info["config_error"] = repr(exc)
    return info


def _run_arm(
    runner: Any,
    base_config: Any,
    *,
    arm: str,
    prefix: tuple[int, ...],
    suffix: tuple[int, ...],
    request_id_base: int,
    paged_force: _PagedRouteForce | None,
    kv_pool_initial_pages: int = 0,
    source_lifecycle: str = "completed",
) -> dict[str, Any]:
    """Run one source + continuation case for one arm; return a result row."""

    source_id = int(request_id_base)
    continuation_id = source_id + 1
    source_state = SimpleNamespace(request_id=source_id)
    continuation_state = SimpleNamespace(request_id=continuation_id)
    continued_prompt = (*prefix, *suffix)
    mode = "off" if arm == "miss" else "radix"
    config_updates: dict[str, Any] = {"prefix_cache": str(mode)}
    if int(kv_pool_initial_pages) > 0:
        # One big chunk with no growth window: mid-run pool growth is a
        # placement lottery that would silently flip the hit-contiguous arm
        # onto the paged route. Pin the arena so contiguity is a property of
        # the arm, not of allocator history.
        pages = int(kv_pool_initial_pages)
        config_updates.update(
            {
                "kv_pool_initial_pages": pages,
                "kv_pool_low_water_pages": pages,
                "kv_pool_high_water_pages": pages,
                "kv_pool_chunk_pages": pages,
            }
        )
    runner.configure_engine_loop(replace(base_config, **config_updates))
    runtime = runner._shared_runner.runtime
    if runtime is None:
        raise RuntimeError("probe requires a live HIP runtime")
    timings: dict[str, float] = {}
    row: dict[str, Any] = {"arm": str(arm), "source_lifecycle": str(source_lifecycle)}
    try:
        # Source: the shared prefix every continuation will hit.
        runner.register_batch((source_id,), _request(prefix, max_tokens=2), prompt_rows=(prefix,))
        runner.reserve_admission(source_state)
        runtime.device_synchronize()
        start = time.perf_counter()
        runner.prefill_batch(_prefill_work(source_id, prefix), commit=True)
        runtime.device_synchronize()
        timings["source_prefill_ms"] = (time.perf_counter() - start) * 1000.0
        source_row = runner._rows[source_id]
        if source_row.lease is None:
            raise RuntimeError("probe source did not become resident")
        row["source_route"] = _session_plan(runner, source_row.lease.session)
        row["source_block_ids"] = [int(b) for b in source_row.kv_allocation.block_ids]
        row["source_token_id"] = int(source_row.slot.prev_token)
        if source_lifecycle == "completed":
            # Match the served multi-turn flow: the prior turn finished and
            # only its retained prefix snapshot remains. Releasing the source
            # also frees its decode-tail page, which is what lets the
            # continuation's suffix land adjacent to the shared prefix.
            if mode == "radix":
                runner._release_row_resources(source_row, retain_prefix_snapshots=True)
                runner._rows.pop(source_id)
            else:
                runner.discard((source_id,))
            source_row = None

        before = _snapshot_counters(runner)
        runner.register_batch(
            (continuation_id,),
            _request(continued_prompt, max_tokens=1),
            prompt_rows=(continued_prompt,),
        )
        runtime.device_synchronize()
        total_start = time.perf_counter()
        runner.reserve_admission(continuation_state)
        runtime.device_synchronize()
        admission_end = time.perf_counter()
        continuation_row = runner._rows[continuation_id]
        if arm == "hit-paged":
            if paged_force is None:
                raise RuntimeError("hit-paged arm requires the route force")
            paged_force.arm(continuation_row.lease.session)

        runner.prefill_batch(_prefill_work(continuation_id, prefix), commit=True)
        runtime.device_synchronize()
        prefix_end = time.perf_counter()
        runner.prefill_batch(_prefill_work(continuation_id, suffix), commit=True)
        runtime.device_synchronize()
        suffix_end = time.perf_counter()

        timings["admission_ms"] = (admission_end - total_start) * 1000.0
        timings["continuation_prefix_prefill_ms"] = (prefix_end - admission_end) * 1000.0
        timings["continuation_suffix_prefill_ms"] = (suffix_end - prefix_end) * 1000.0
        timings["continuation_ttft_ms"] = (suffix_end - total_start) * 1000.0
        row["timings_ms"] = timings

        after = _snapshot_counters(runner)
        row["counters_delta"] = {key: after[key] - before.get(key, 0) for key in after}
        row["prefix_reused_tokens"] = int(continuation_row.prefix_reused_tokens)
        row["prefix_matched_tokens"] = int(continuation_row.prefix_matched_tokens)
        row["prefix_fallback_reason"] = continuation_row.prefix_fallback_reason
        row["continuation_token_id"] = int(continuation_row.slot.prev_token)
        row["config"] = _model_config(runner)
        allocation = getattr(continuation_row, "kv_allocation", None)
        if allocation is not None:
            block_ids = [int(block_id) for block_id in allocation.block_ids]
            row["continuation_block_ids"] = block_ids
            row["continuation_placement_contiguous"] = block_ids == list(
                range(block_ids[0], block_ids[0] + len(block_ids))
            ) if block_ids else False
        row["continuation_route"] = _session_plan(runner, continuation_row.lease.session)
        row["config"] = _model_config(runner)
        scatter = row["continuation_route"].get("device_kv_nonidentity_scatter")
        if arm == "hit-contiguous":
            if not row.get("continuation_placement_contiguous", False) or scatter:
                raise RuntimeError(
                    "hit-contiguous arm did not take the slot-local route "
                    f"(placement_contiguous={row.get('continuation_placement_contiguous')}, "
                    f"scatter={scatter}); probe needs a larger pinned pool"
                )
        if arm == "hit-paged":
            if scatter is not True:
                raise RuntimeError(
                    "hit-paged arm kept the slot-local route; probe route forcing is broken"
                )
    finally:
        remaining = tuple(rid for rid in (source_id, continuation_id) if rid in runner._rows)
        if remaining:
            runner.discard(remaining)
        runner._clear_prefix_snapshots()
    return row


def _pinned_pool_pages(args: argparse.Namespace) -> int:
    """Pool pages that cover both sessions plus the packed workspace lease."""

    if int(getattr(args, "kv_pool_initial_pages", 0) or 0) > 0:
        return int(args.kv_pool_initial_pages)
    # block_size 256; workspace lease scales with max context (~2 pages per
    # 256-token block of max positions); generous headroom avoids growth.
    positions = int(args.prefix_tokens) + int(args.suffix_tokens) + 64
    return max(64, (positions * 2) // 256 + 64)


def _child(args: argparse.Namespace) -> int:
    """Single-arm payload: run the case once and emit host-timing JSON."""

    from hipengine import LLM

    prefix = (int(args.prefix_token_id),) * int(args.prefix_tokens)
    suffix = (int(args.suffix_token_id),) * int(args.suffix_tokens)
    rows: list[dict[str, Any]] = []
    pinned_pages = _pinned_pool_pages(args)
    paged_force: _PagedRouteForce | None = None
    if args.arm == "hit-paged":
        paged_force = _PagedRouteForce()
    llm = LLM(
        str(args.model),
        backend=str(args.backend),
        quant=str(args.quant),
        max_active_requests=3,
        prefix_cache="radix" if args.arm != "miss" else "off",
    )
    try:
        llm.prepare(max_sequence_length=int(args.max_sequence_length))
        wrapper = llm._get_text_generator()
        runner = wrapper._runner
        base_config = wrapper._loop.config
        if args.warmup_cases > 0:
            for case in range(int(args.warmup_cases)):
                rows.append(
                    _run_arm(
                        runner,
                        base_config,
                        arm=args.arm,
                        prefix=prefix,
                        suffix=suffix,
                        request_id_base=30_000 + case * 10,
                        paged_force=paged_force,
                        kv_pool_initial_pages=pinned_pages,
                        source_lifecycle=str(args.source_lifecycle),
                    )
                )
        rows.append(
            _run_arm(
                runner,
                base_config,
                arm=args.arm,
                prefix=prefix,
                suffix=suffix,
                request_id_base=40_000,
                paged_force=paged_force,
                kv_pool_initial_pages=pinned_pages,
                source_lifecycle=str(args.source_lifecycle),
            )
        )
    finally:
        if paged_force is not None:
            paged_force.close()
        llm.close()
    payload = {
        "schema": 1,
        "status": "diagnostic",
        "date": date.today().isoformat(),
        "arm": str(args.arm),
        "model": str(args.model),
        "quant": str(args.quant),
        "backend": str(args.backend),
        "prefix_tokens": int(args.prefix_tokens),
        "suffix_tokens": int(args.suffix_tokens),
        "max_sequence_length": int(args.max_sequence_length),
        "rows": rows,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["rows"][-1], indent=2, sort_keys=True))
    return 0


def _child_command(args: argparse.Namespace, arm: str, json_out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "child",
        "--arm",
        arm,
        "--model",
        str(args.model),
        "--quant",
        str(args.quant),
        "--backend",
        str(args.backend),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--suffix-tokens",
        str(args.suffix_tokens),
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--prefix-token-id",
        str(args.prefix_token_id),
        "--suffix-token-id",
        str(args.suffix_token_id),
        "--warmup-cases",
        "1",
        "--kv-pool-initial-pages",
        str(_pinned_pool_pages(args)),
        "--source-lifecycle",
        str(args.source_lifecycle),
        "--json-out",
        str(json_out),
    ]


def _read_kernel_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            rows.append(
                {
                    "kernel": (
                        row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or ""
                    ).strip(),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "vgpr": _int_or_none(row.get("VGPR_Count")),
                    "lds": _int_or_none(row.get("LDS_Block_Size")),
                    "scratch": _int_or_none(row.get("Scratch_Size")),
                }
            )
    return rows


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _summarize_kernels(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_kernel: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_kernel.setdefault(
            row["kernel"],
            {"kernel": row["kernel"], "calls": 0, "total_ns": 0, "max_ns": 0, "lds_max": 0, "vgpr_max": 0},
        )
        entry["calls"] += 1
        entry["total_ns"] += row["duration_ns"]
        entry["max_ns"] = max(entry["max_ns"], row["duration_ns"])
        entry["lds_max"] = max(entry["lds_max"], row.get("lds") or 0)
        entry["vgpr_max"] = max(entry["vgpr_max"], row.get("vgpr") or 0)
    table = sorted(by_kernel.values(), key=lambda entry: -int(entry["total_ns"]))
    total_ns = sum(entry["total_ns"] for entry in table)
    for entry in table:
        entry["total_ms"] = entry["total_ns"] / 1e6
        entry["share"] = (entry["total_ns"] / total_ns) if total_ns else 0.0
    return table


def _rocprof(args: argparse.Namespace) -> int:
    args = argparse.Namespace(**vars(args))
    if args.rocprofv3 is None:
        args.rocprofv3 = shutil.which("rocprofv3") or "rocprofv3"
    if not Path(args.compiler_version_file).is_file():
        hipcc = shutil.which("hipcc")
        if hipcc is None:
            raise SystemExit("hipcc not found; provide --compiler-version-file")
        version = subprocess.run([hipcc, "--version"], capture_output=True, text=True, check=True)
        Path(args.compiler_version_file).write_text(version.stdout, encoding="utf-8")
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    arms = [arm.strip() for arm in str(args.arms).split(",") if arm.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected one of {ARMS}")

    child_env = os.environ.copy()
    child_env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    child_env["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    artifacts: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_root = raw_root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        warmup_json = arm_root / "warmup.json"
        print(f"[probe] warmup (prebuild JIT cache): {arm}", flush=True)
        subprocess.run(
            _child_command(args, arm, warmup_json),
            cwd=REPO_ROOT,
            env=child_env,
            check=True,
        )
        measured_json = arm_root / "measured.json"
        rocprof_cmd = [
            args.rocprofv3,
            "--kernel-trace",
            "--hip-runtime-trace",
            "--output-format",
            "csv",
            "-d",
            str(arm_root),
            "-o",
            f"probe-{arm}",
            "--",
            *_child_command(args, arm, measured_json),
        ]
        print(f"[probe] rocprofv3: {arm}", flush=True)
        log_path = arm_root / "rocprof.log"
        start = time.perf_counter()
        with log_path.open("w") as log_file:
            completed = subprocess.run(
                rocprof_cmd,
                cwd=REPO_ROOT,
                env=child_env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        wall_seconds = time.perf_counter() - start
        if completed.returncode != 0:
            tail = log_path.read_text().splitlines()[-40:]
            raise SystemExit(
                f"rocprofv3 failed for arm {arm} (exit {completed.returncode}); tail:\n"
                + "\n".join(tail)
            )
        csvs = sorted(arm_root.glob("*_kernel_trace.csv"))
        if len(csvs) != 1:
            raise SystemExit(f"expected exactly one kernel trace CSV for {arm}; found {csvs}")
        kernels = _read_kernel_csv(csvs[0])
        child_payload = json.loads(measured_json.read_text(encoding="utf-8"))
        artifacts[arm] = {
            "kernel_trace_csv": str(csvs[0]),
            "kernel_count": len(kernels),
            "kernel_total_ms": sum(row["duration_ns"] for row in kernels) / 1e6,
            "wall_seconds": wall_seconds,
            "child": child_payload,
            "top_kernels": _summarize_kernels(kernels)[: int(args.top)],
        }

    payload = {
        "schema": 1,
        "status": "diagnostic",
        "performance_claim": False,
        "date": date.today().isoformat(),
        "model": str(args.model),
        "quant": str(args.quant),
        "backend": str(args.backend),
        "prefix_tokens": int(args.prefix_tokens),
        "suffix_tokens": int(args.suffix_tokens),
        "arms": artifacts,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[probe] artifact: {out_path}")
    for arm, artifact in artifacts.items():
        suffix_ms = artifact["child"]["rows"][-1]["timings_ms"]["continuation_suffix_prefill_ms"]
        print(
            f"\n=== arm {arm}: suffix host {suffix_ms:.1f} ms, "
            f"{len(artifact['top_kernels'])} kernel families traced ==="
        )
        for entry in artifact["top_kernels"]:
            print(
                f"  {entry['total_ms']:>10.1f} ms  {entry['calls']:>6} calls  "
                f"{entry['kernel'][:100]}"
            )
    return 0


def _time_mode(args: argparse.Namespace) -> int:
    from hipengine import LLM

    prefix = (int(args.prefix_token_id),) * int(args.prefix_tokens)
    suffix = (int(args.suffix_token_id),) * int(args.suffix_tokens)
    arms = [arm.strip() for arm in str(args.arms).split(",") if arm.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected one of {ARMS}")
    paged_force = _PagedRouteForce()
    pinned_pages = _pinned_pool_pages(args)
    rows: list[dict[str, Any]] = []
    llm = LLM(
        str(args.model),
        backend=str(args.backend),
        quant=str(args.quant),
        max_active_requests=3,
        prefix_cache="radix",
    )
    try:
        llm.prepare(max_sequence_length=int(args.max_sequence_length))
        wrapper = llm._get_text_generator()
        runner = wrapper._runner
        base_config = wrapper._loop.config
        for arm in arms:
            for case in range(int(args.repetitions)):
                rows.append(
                    _run_arm(
                        runner,
                        base_config,
                        arm=arm,
                        prefix=prefix,
                        suffix=suffix,
                        request_id_base=50_000 + case * 10,
                        paged_force=paged_force,
                        kv_pool_initial_pages=pinned_pages,
                        source_lifecycle=str(args.source_lifecycle),
                    )
                )
    finally:
        paged_force.close()
        llm.close()
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("time", "rocprof", "child"), default="time")
    parser.add_argument("--arms", default="hit-contiguous,hit-paged,miss")
    parser.add_argument("--arm", choices=ARMS, default="hit-contiguous")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--prefix-token-id", type=int, default=9707)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-token-id", type=int, default=9708)
    parser.add_argument("--suffix-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-cases", type=int, default=0)
    parser.add_argument("--kv-pool-initial-pages", type=int, default=0)
    parser.add_argument("--source-lifecycle", choices=("active", "completed"), default="completed")
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/gguf-prefix-route-probe.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/gguf-prefix-route-probe-rocprof"))
    parser.add_argument("--compiler-version-file", type=Path, default=DEFAULT_COMPILER_VERSION_FILE)
    parser.add_argument("--rocprofv3", default=None)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    if int(args.max_sequence_length) <= 0:
        args.max_sequence_length = int(args.prefix_tokens) + int(args.suffix_tokens) + 64
    if int(args.max_sequence_length) < int(args.prefix_tokens) + int(args.suffix_tokens):
        raise SystemExit("--max-sequence-length must cover prefix+suffix")
    if args.mode == "child":
        return _child(args)
    if args.mode == "rocprof":
        return _rocprof(args)
    return _time_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
