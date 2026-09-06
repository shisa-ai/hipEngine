#!/usr/bin/env python3
"""Packet 5 K4 watchdog-bounded reproducer.

Runs the actual C1-C8 server bench at an explicitly unqualified (C8, K4)
diagnostic cell under a watchdog that dumps every host thread stack to a file
and exits before the historical 1200 s no-output hang window closes. The
probe exists to localize the K4 stall with evidence, not to produce a
performance claim: the K4 evidence row injected here is diagnostic-only,
`automatic_eligible=False`, and the screen opt-in refuses any automatic
widening by construction.

Mechanics:
- The candidate-depth bound (`MTP2_MAX_CANDIDATE_DEPTH`) is raised to 4 at
  runtime in every module that reads it, never by editing the source constant.
- A diagnostic K4 serving-evidence row is cloned from the registered C8/K3
  row (same model artifact identity) so static eligibility exists for the
  screening path; the row and the run output are stamped as non-claim.
- `faulthandler.dump_traceback_later(timeout, exit=True)` writes all-thread
  stacks (pinpointing the blocked HIP/runtime call) and terminates.
- A heartbeat thread prints elapsed time so the log shows where progress
  stopped even without a stack dump.

Usage:
  python scripts/qwen38_packet5_k4_watchdog_probe.py \
      --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
      --output-dir /tmp/he-bettermtp-raw/packet5
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_PROBE_LABEL = "packet5_k4_watchdog_probe_diagnostic_not_for_claims"


def _patch_depth_bound(depth: int) -> list[str]:
    """Raise the candidate-depth bound in every module that reads it."""

    import hipengine.generation.qwen35_gguf as gguf_mod
    import hipengine.generation.qwen35_gguf_mtp2 as mtp2_mod

    patched: list[str] = []
    for module, names in (
        (mtp2_mod, ("MTP2_MAX_CANDIDATE_DEPTH", "_MTP2_MAX_CANDIDATE_DEPTH")),
        (gguf_mod, ("MTP2_MAX_CANDIDATE_DEPTH",)),
    ):
        for name in names:
            setattr(module, name, int(depth))
            patched.append(f"{module.__name__}.{name}={depth}")
    return patched


def _inject_k4_evidence_row(width: int, budget: int) -> str:
    """Append a diagnostic K4 clone of the registered C8/K3 evidence row.

    The model plugin is a frozen dataclass instance registered at import
    time, so the patch must land on that instance (object.__setattr__); a
    class-attribute write is invisible to the registry's resolver.
    """

    import hipengine.models.qwen35 as models_mod

    base = None
    for row in models_mod._QWEN38_Q4KM_MTP_SERVING_EVIDENCE:
        if (
            row.backend == "hip_gfx1100"
            and row.realized_group_rows == int(width)
            and row.execution_profile == "production"
        ):
            base = row
            break
    if base is None:
        raise RuntimeError("no registered gfx1100 C8 K3 evidence row to clone")
    row = dataclasses.replace(
        base,
        evidence_key=f"qwen38-q4km-gfx1100-production-bf16-c{width}-k{budget}-d24-k4probe",
        candidate_budget=int(budget),
        reason=_PROBE_LABEL,
        evidence_artifacts=(
            "scripts/qwen38_packet5_k4_watchdog_probe.py (runtime-injected diagnostic row)",
        ),
        automatic_eligible=False,
    )
    plugin = models_mod.QWEN35_GGUF
    object.__setattr__(
        plugin,
        "speculative_mtp_serving_evidence",
        plugin.speculative_mtp_serving_evidence + (row,),
    )
    return row.evidence_key


def _slice_prompts(source: Path, count: int, destination: Path) -> Path:
    """Pass the canonical suite through unchanged when count is 0.

    The bench validates canonical prompt IDs/order, so any slice breaks
    admission; count=0 (default) keeps the full canonical file. A nonzero
    count only helps future probes that target a suite-relaxed entry point.
    """

    if int(count) <= 0:
        return source
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"prompt file {source} is empty")
    keep = lines[: int(count)]
    destination.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return destination


def _dump_asyncio_tasks(path: Path) -> None:
    """Dump every asyncio task's stack across all loops in the process."""

    import asyncio
    import gc

    lines: list[str] = []
    loops = {
        id(obj): obj
        for obj in gc.get_objects()
        if isinstance(obj, asyncio.AbstractEventLoop)
    }
    for loop in loops.values():
        try:
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        except RuntimeError:
            continue
        lines.append(f"loop {loop!r}: {len(tasks)} pending tasks")
        for task in tasks:
            lines.append(f"  task {task.get_name()!r} state={task._state}")
            for frame in task.get_stack(limit=25):
                lines.append(
                    f"    {frame.f_code.co_filename}:{frame.f_lineno}"
                    f" in {frame.f_code.co_name}"
                )
    # Finished tasks carry the exceptions that killed queue workers; the
    # batcher worker dying without finishing its items is the K4 deadlock
    # suspect.
    import traceback as _tb

    done_dumped = 0
    for obj in gc.get_objects():
        if not isinstance(obj, asyncio.Task) or not obj.done():
            continue
        if obj.cancelled():
            continue
        exc = obj.exception()
        if exc is None:
            continue
        done_dumped += 1
        lines.append(f"finished task {obj.get_name()!r} raised:")
        lines.extend(
            ("    " + line) for line in _tb.format_exception(type(exc), exc, exc.__traceback__)
        )
        if done_dumped >= 12:
            lines.append("    ... (more finished tasks truncated)")
            break
    if not done_dumped:
        lines.append("no finished task holds an unretrieved exception")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _heartbeat(deadline_s: float, stop: threading.Event) -> None:
    start = time.monotonic()
    while not stop.wait(15.0):
        elapsed = time.monotonic() - start
        remaining = deadline_s - elapsed
        print(
            f"[probe-heartbeat] elapsed={elapsed:.0f}s remaining={remaining:.0f}s "
            f"threads={threading.active_count()}",
            flush=True,
        )


def _run_direct_probe(args: argparse.Namespace) -> int:
    """Call the engine's speculative MTP runner directly at the probe cell.

    Bypasses the server batcher so a K4 failure surfaces as either a raw
    traceback (runner raises) or a generation-path hang caught by the
    watchdog with the blocking thread's exact frames.
    """

    import json as _json

    from hipengine.llm import LLM, SamplingParams

    os.environ["HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"] = str(int(args.budget))
    llm = LLM(
        str(args.model),
        backend="hip_gfx1100",
        execution_profile="production",
        max_active_requests=int(args.width),
        max_sequence_length=int(args.max_sequence_length),
        speculative_candidate_budget=int(args.budget),
    )
    llm.prepare(max_sequence_length=int(args.max_sequence_length))
    lines = args.prompts.read_text(encoding="utf-8").splitlines()
    prompt_payloads = [
        "\n".join(
            message["content"]
            for message in _json.loads(line)["messages"]
            if isinstance(message.get("content"), str)
        )
        for line in lines
    ]
    sampling = SamplingParams(max_tokens=int(args.max_tokens), temperature=0.0, top_p=1.0)

    result: dict[str, Any] = {}

    def _run() -> None:
        try:
            outputs = llm.generate_speculative_mtp_detailed(prompt_payloads, sampling)
            result["outputs"] = [str(getattr(o, "text", o))[:80] for o in outputs]
            result["status"] = "complete"
        except BaseException as exc:  # noqa: BLE001 - probe records and re-raises
            import traceback as _tb

            result["status"] = "raised"
            result["exception"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__)
            )

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=args.timeout)
    if worker.is_alive():
        result["status"] = "hung"
        result["note"] = (
            "runner still running at watchdog timeout; thread stacks in the"
            " faulthandler file pinpoint the blocked frame"
        )
    summary_path = args.output_dir / f"k4-w{args.width}-b{args.budget}-direct-summary.json"
    summary_path.write_text(_json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[probe-direct] status={result.get('status')} -> {summary_path}", flush=True)
    if result.get("traceback"):
        print(result["traceback"], flush=True)
    return 0 if result.get("status") == "complete" else 1


def _wrap_batcher_run_group(app: Any, out_dir: Path) -> None:
    """Log exceptions escaping ``_GenerationBatcher._run_group``.

    An exception in the batch-route pre-flight or post-processing kills the
    batcher worker; queued items then wait forever and the exception dies
    with the garbage-collected task. Wrapping it converts the deadlock into
    a recorded error while keeping the original behavior.
    """

    import types as _types
    import traceback as _tb

    batcher = app.state.hipengine_generation_batcher
    original = batcher._run_group
    original_generate = batcher._generate_prompts

    def _log_exception(where: str, exc: BaseException) -> None:
        path = out_dir / "batcher-run-group-exceptions.log"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"=== {time.strftime('%H:%M:%S')} {where} ===\n")
            handle.write(_tb.format_exception(type(exc), exc, exc.__traceback__))
            handle.write("\n")

    def logged_generate(*gen_args: Any, **gen_kwargs: Any):
        try:
            return original_generate(*gen_args, **gen_kwargs)
        except BaseException as exc:  # noqa: BLE001 - probe records and re-raises
            _log_exception("generate_prompts", exc)
            raise

    def logged(self: Any, group: Any, **kwargs: Any):
        try:
            return original(group, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - probe records and re-raises
            _log_exception(f"run_group size={len(group)}", exc)
            for item in group:
                try:
                    batcher._finish_queued_generation(item, exception=exc)
                except Exception:
                    pass
            raise

    batcher._generate_prompts = logged_generate  # type: ignore[method-assign]
    batcher._run_group = _types.MethodType(logged, batcher)  # type: ignore[method-assign]
    print("[probe] batcher._run_group/_generate_prompts wrapped for exception capture", flush=True)


def _trace_adapter_registration() -> None:
    """Log adapter register/release calls to localize C1 non-engagement."""

    import hipengine.generation.qwen35_gguf_mtp2 as m2

    original_register = m2.Qwen35GGUFMTP2Adapter.register_request
    original_release = m2.Qwen35GGUFMTP2Adapter.release_request
    original_capability = m2.Qwen35GGUFMTP2Adapter.capability

    def traced_capability(self: Any, request_semantics: Any) -> Any:
        result = original_capability(self, request_semantics)
        if result is None:
            ids = tuple(
                int(getattr(item, "request_id", -1)) for item in request_semantics
            )
            gates = []
            for rid in ids:
                try:
                    row = self.owner._row(rid)
                    gates.append(
                        f"rid={rid} greedy={row.native_greedy}"
                        f" first={row.first_token_emitted}"
                        f" lease={row.lease is not None}"
                        f" slot={row.slot is not None}"
                        f" state={rid in self._states}"
                        f" prompt_hidden={rid in self._prompt_hidden_rows}"
                        f" intent={self._intents.get(rid)}"
                        f" rowbudget={getattr(row, 'mtp2_candidate_budget', None)}"
                        f" fb={getattr(row, 'mtp2_prompt_fallback_reason', None)}"
                        f" prefix={getattr(row, 'prefix_reused_tokens', None)}"
                        f" failures={getattr(row, 'mtp2_failure_reasons', None)}"
                    )
                except Exception as exc:
                    gates.append(f"rid={rid} row_error={type(exc).__name__}")
            print(
                f"[probe-cap] capability None for {ids}: {'; '.join(gates)}",
                flush=True,
            )
        return result

    m2.Qwen35GGUFMTP2Adapter.capability = traced_capability  # type: ignore[method-assign]

    def traced_register(
        self: Any,
        request_id: int,
        candidate_budget: int,
        *,
        static_eligibility: Any = None,
    ) -> Any:
        print(
            f"[probe-reg] register rid={request_id} budget={candidate_budget}"
            f" elig={'yes' if static_eligibility is not None else 'NO'}",
            flush=True,
        )
        return original_register(
            self, request_id, candidate_budget, static_eligibility=static_eligibility
        )

    def traced_release(self: Any, request_id: int) -> Any:
        print(f"[probe-reg] release rid={request_id}", flush=True)
        return original_release(self, request_id)

    m2.Qwen35GGUFMTP2Adapter.register_request = traced_register  # type: ignore[method-assign]
    m2.Qwen35GGUFMTP2Adapter.release_request = traced_release  # type: ignore[method-assign]

    from hipengine.generation.qwen35_gguf import (
        Qwen35GGUFResidentModelRunner as _runner_cls,
    )

    original_begin = _runner_cls._begin_mtp2_prompt_streaming
    original_finish = _runner_cls._finish_mtp2_prompt_streaming

    def traced_begin(
        runner_self: Any, rows: Any
    ) -> Any:
        rids = tuple(int(row.request_id) for row in rows)
        budgets = {
            int(row.request_id): getattr(row, "mtp2_candidate_budget", None)
            for row in rows
        }
        result = original_begin(runner_self, rows)
        none_map = {
            int(row.request_id): (sink is None)
            for row, sink in zip(rows, result, strict=False)
        }
        print(
            f"[probe-stream] begin rids={rids} budgets={budgets} sinks_none={none_map}",
            flush=True,
        )
        return result

    def traced_finish(
        runner_self: Any, rows: Any, sinks: Any, **kwargs: Any
    ) -> Any:
        rids = tuple(int(row.request_id) for row in rows)
        result = original_finish(runner_self, rows, sinks, **kwargs)
        fallbacks = {
            int(row.request_id): getattr(row, "mtp2_prompt_fallback_reason", None)
            for row in rows
        }
        print(
            f"[probe-stream] finish rids={rids} fallbacks={fallbacks}",
            flush=True,
        )
        return result

    _runner_cls._begin_mtp2_prompt_streaming = traced_begin  # type: ignore[method-assign]
    _runner_cls._finish_mtp2_prompt_streaming = traced_finish  # type: ignore[method-assign]
    print("[probe] adapter registration tracing enabled", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=24,
                        help="24 matches every Qwen3.8 evidence row's output horizon")
    parser.add_argument("--prompt-count", type=int, default=0,
                        help="0 keeps the canonical suite (required by the bench gate)")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="bypass the server batcher and call the engine runner directly",
    )
    parser.add_argument(
        "--trace-registration",
        action="store_true",
        help="log adapter register/release calls",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.direct:
        faulthandler.enable()
        faulthandler.dump_traceback_later(
            args.timeout,
            repeat=False,
            exit=True,
            file=open(
                out_dir / f"k4-w{args.width}-b{args.budget}-watchdog-stacks.txt", "w"
            ),
        )
        return _run_direct_probe(args)
    bench_output = out_dir / f"k4-w{args.width}-b{args.budget}-probe.json"
    watchdog_file = out_dir / f"k4-w{args.width}-b{args.budget}-watchdog-stacks.txt"
    prompts_file = _slice_prompts(
        args.prompts,
        args.prompt_count,
        out_dir / f"k4-prompts-slice-{args.prompt_count}.jsonl",
    )

    patched = _patch_depth_bound(args.budget)
    evidence_key = _inject_k4_evidence_row(args.width, args.budget)
    os.environ["HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS"] = "1"

    print("[probe] patched:", ", ".join(patched), flush=True)
    print(f"[probe] injected diagnostic evidence row: {evidence_key}", flush=True)
    print(f"[probe] watchdog: {args.timeout:.0f}s -> {watchdog_file}", flush=True)

    faulthandler.enable()
    faulthandler.dump_traceback_later(
        args.timeout, repeat=False, exit=True, file=open(watchdog_file, "w")
    )

    def _watchdog_asyncio_dump() -> None:
        time.sleep(max(1.0, args.timeout - 5.0))
        try:
            _dump_asyncio_tasks(
                watchdog_file.with_name(watchdog_file.stem + "-asyncio.txt")
            )
        except BaseException:
            pass

    threading.Thread(target=_watchdog_asyncio_dump, daemon=True).start()
    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat, args=(args.timeout, stop), daemon=True
    )
    beat.start()

    from scripts.gguf_mtp_c1c8_server_bench import build_parser, run as bench_run

    import hipengine.server.api as _server_api
    import scripts.gguf_mtp_c1c8_server_bench as _bench_module

    _original_create_app = _server_api.create_app

    def _create_app_with_wrapper(*create_args: Any, **create_kwargs: Any):
        app = _original_create_app(*create_args, **create_kwargs)
        _wrap_batcher_run_group(app, out_dir)
        return app

    _server_api.create_app = _create_app_with_wrapper
    _bench_module.create_app = _create_app_with_wrapper
    if args.trace_registration:
        _trace_adapter_registration()

    argv = [
        "--model", str(args.model),
        "--backend", "hip_gfx1100",
        "--quant", "gguf_q4_k_m",
        "--execution-profile", "production",
        "--prompts", str(prompts_file),
        "--mtp-request-mode", "explicit",
        "--widths", str(args.width),
        # The injected evidence row inherits the registered row's
        # resident_capacity and the serving key compares it exactly: C1/C8
        # evidence was measured at capacity 8, C2 at capacity 2.
        "--resident-capacity", str(2 if int(args.width) == 2 else 8),
        "--expected-mtp-widths", str(args.width),
        "--max-tokens", str(args.max_tokens),
        "--candidate-budget", str(args.budget),
        "--batch-window-ms", "20",
        "--max-sequence-length", str(args.max_sequence_length),
        "--correctness-contract", "ar_exact",
        "--output", str(bench_output),
    ]
    bench_args = build_parser().parse_args(argv)
    started = time.monotonic()
    status = "exception"
    try:
        payload = bench_run(bench_args)
        status = str(payload.get("status"))
        failure_reasons = list(payload.get("failure_reasons", ()))
    except BaseException as exc:  # noqa: BLE001 - probe must record and re-raise
        failure_reasons = [f"{type(exc).__name__}: {exc}"]
        raise
    finally:
        stop.set()
        elapsed = time.monotonic() - started
        summary = {
            "kind": "packet5-k4-watchdog-probe",
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "watchdog_timeout_s": args.timeout,
            "watchdog_file": str(watchdog_file),
            "diagnostic_only": True,
            "failure_reasons": failure_reasons,
            "bench_output": str(bench_output),
        }
        (out_dir / f"k4-w{args.width}-b{args.budget}-probe-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[probe] done elapsed={elapsed:.0f}s status={status}", flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
