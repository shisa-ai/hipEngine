#!/usr/bin/env python3
"""Measure a decoding peer's inter-token latency while a long prefill runs.

RT-1 peer-fairness gate for the qwen4exp prefill chunk.

**Two different knobs, and only one of them is the -4.94%.** They are not
interchangeable and the ledger has conflated them:

``session.prefill_chunk_size``
    The *runner-internal* chunk. ``prefill_chunked`` picks the prefill
    workspace whose ``prefill_chunk_size`` fits the token count, swaps it in,
    and runs the whole prompt inside one call. Production binds this to 1024
    at ``qwen4_exp_profiles.py:_bind_default_chunk``. **This is the knob the
    measured -4.94% on ``code-p4096`` belongs to**, because
    ``qwen4exp_profile_gap.py`` calls ``runner.prefill(ids)`` directly with
    the whole prompt.

``HIPENGINE_MAX_PREFILL_CHUNK_TOKENS``
    The *engine-loop scheduling* cap (default 1024,
    ``engine_loop.DEFAULT_MAX_PREFILL_CHUNK_TOKENS``). It slices a row's
    prompt across loop ticks. It does **not** reach the runner:
    ``Qwen4ExpResidentServingRunner.prefill_batch`` only advances
    ``row.prefill_tokens_seen`` and calls ``row.runner.prefill(row.prompt_ids)``
    -- the *full* prompt -- once the last slice arrives. So this knob changes
    when the single prefill call fires, not how much GPU work it does.

This harness measures both, and records the executed chunk at each layer as
engagement evidence rather than assuming either knob took effect.

Arms (``--arms``):
  ``g1024_l1024``  today's production default
  ``g4096_l1024``  the -4.94% promotion, engine-loop cap unchanged
  ``g4096_l4096``  the promotion the ledger's corrected mechanism proposes

One child process per arm so no KV/session state carries between arms.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_FIXTURE = (
    _REPO_ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"
)

# (arm name, runner-internal chunk, engine-loop cap)
ARMS: dict[str, tuple[int, int]] = {
    "g1024_l1024": (1024, 1024),
    "g1024_l4096": (1024, 4096),
    "g4096_l1024": (4096, 1024),
    "g4096_l4096": (4096, 4096),
}


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _prompt_ids(fixture: dict[str, Any], case_id: str) -> tuple[int, ...]:
    for case in fixture["cases"]:
        if case["id"] == case_id:
            return tuple(int(token) for token in case["prompt_token_ids"])
    raise SystemExit(f"fixture has no case {case_id!r}")


def _jsonable(value: Any) -> Any:
    """Coerce a nested report structure into something json.dump accepts."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _session_runners(serving: Any) -> list[Any]:
    """Every runner-internal session this serving runner can hand a row to.

    ``Qwen4ExpResidentServingRunner`` seeds ``_all_runners`` with
    ``generator.runner`` and ``prepare()`` appends one more per extra capacity
    slot, copying the base runner's ``prefill_chunk_size`` at construction.  A
    c2 arm therefore has *two* sessions, and a prefill row can land on either,
    so every one of them has to agree on the chunk.
    """

    found: list[Any] = []
    for candidate in list(getattr(serving, "_all_runners", []) or []):
        if candidate is not None and hasattr(candidate, "prefill_chunk_size"):
            found.append(candidate)
    generator = getattr(serving, "generator", None)
    base = getattr(generator, "runner", None)
    if base is not None and hasattr(base, "prefill_chunk_size"):
        found.append(base)
    unique: list[Any] = []
    for item in found:
        if not any(item is other for other in unique):
            unique.append(item)
    if not unique:
        raise RuntimeError(
            f"could not locate a session with prefill_chunk_size on {type(serving).__name__}"
        )
    return unique


def _install_chunk_override(chunk: int) -> None:
    """Force the chunk at *construction* on the generator.

    ``Qwen4ExpGGUFTextGenerator.__init__`` is the single owner of the chunk:
    it sizes the PLE staging ring (``staging_token_capacity=prefill_chunk_size``)
    and hands the same value to the resident session, whose base prefill
    workspace is allocated in ``__init__``.  Patching only the session forces a
    4096-token chunk against a 1024-row PLE staging ring and fails with "PLE row
    request exceeds staging capacity"; patching the generator keeps them
    consistent.  A post-construction assignment cannot work, because
    ``_configure_prefill_workspace_selection`` rejects any extra workspace larger
    than the base one allocated at construction time.

    The generator patch alone is **not** sufficient.  ``LLM`` reaches the
    generator through the registry factory, which sets
    ``_implicit_profile_chunk_size = prefill_chunk_size is None`` - True whenever
    no chunk is passed, which is always here - and
    ``qwen4_exp_profiles._bind_default_chunk`` then runs *after* construction and
    resets ``generator.runner.prefill_chunk_size`` to 1024.  That silently
    reverted the override: the workspace stayed 1024 rows and the chunk-4096 arms
    died in ``copy_host_to_device`` with "copy size 32768 exceeds device buffer
    size 8192".  ``_bind_default_chunk`` is therefore neutralised too, so the
    chunk survives to the workspace allocation.
    """

    from hipengine.generation import qwen4_exp_profiles
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator

    original_init = Qwen4ExpGGUFTextGenerator.__init__

    def patched_init(self, *positional, **keyword):  # type: ignore[no-untyped-def]
        keyword["prefill_chunk_size"] = int(chunk)
        return original_init(self, *positional, **keyword)

    Qwen4ExpGGUFTextGenerator.__init__ = patched_init

    def patched_bind_default_chunk(generator, *, production):  # type: ignore[no-untyped-def]
        # Re-assert the requested chunk rather than the factory's implicit 1024.
        generator.runner.prefill_chunk_size = int(chunk)

    qwen4_exp_profiles._bind_default_chunk = patched_bind_default_chunk


def _install_instrumentation(trace: dict[str, list[dict[str, Any]]]) -> None:
    """Wrap the two prefill layers at class level so every object is covered."""

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpResidentServingRunner
    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner

    original_batch = Qwen4ExpResidentServingRunner.prefill_batch
    original_prefill = Qwen4ExpGGUFResidentModelRunner.prefill

    def prefill_batch(self, work, *, commit):  # type: ignore[no-untyped-def]
        rows = [len(tuple(row)) for row in work.token_rows]
        trace["loop_prefill_batch"].append(
            {"t": time.perf_counter(), "chunk_rows": rows, "total": sum(rows)}
        )
        return original_batch(self, work, commit=commit)

    def prefill(self, token_ids, **kwargs):  # type: ignore[no-untyped-def]
        workspaces = getattr(self, "_prefill_workspaces", None) or []
        trace["session_prefill"].append(
            {
                "t": time.perf_counter(),
                "n_tokens": len(token_ids),
                "prefill_chunk_size": int(self.prefill_chunk_size),
                # The workspace chunk is the value that actually drives
                # prefill_chunked's selection; the attribute alone is not proof.
                "workspace_chunks": [int(w.prefill_chunk_size) for w in workspaces],
            }
        )
        return original_prefill(self, token_ids, **kwargs)

    Qwen4ExpResidentServingRunner.prefill_batch = prefill_batch
    Qwen4ExpGGUFResidentModelRunner.prefill = prefill


def _run_child(args: argparse.Namespace) -> int:
    from hipengine import LLM, SamplingParams
    from hipengine.generation import GenerationRequest
    from hipengine.generation.engine_loop import DEFAULT_MAX_PREFILL_CHUNK_TOKENS

    fixture = _load_fixture(args.fixture)
    long_ids = _prompt_ids(fixture, args.prefill_case)
    short_ids = _prompt_ids(fixture, args.peer_case)[: args.peer_prompt_tokens]
    if len(long_ids) != 4096:
        raise SystemExit(f"--prefill-case must be 4096 tokens, got {len(long_ids)}")

    trace: dict[str, list[dict[str, Any]]] = {
        "loop_prefill_batch": [],
        "session_prefill": [],
    }

    env = {
        "HIPENGINE_MAX_ACTIVE_REQUESTS": "2",
        "HIPENGINE_PREFILL_DECODE_POLICY": "protect_decode",
        "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": str(args.loop_cap),
    }
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_prefill_peer_fairness_arm",
        "arm": args.arm,
        "order": args.order,
        "generator_chunk_requested": int(args.generator_chunk),
        "loop_cap_requested": int(args.loop_cap),
        "loop_cap_default": int(DEFAULT_MAX_PREFILL_CHUNK_TOKENS),
        "prefill_case": args.prefill_case,
        "peer_case": args.peer_case,
        "peer_prompt_tokens": len(short_ids),
        "prefill_prompt_tokens": len(long_ids),
        "peer_max_tokens": int(args.peer_max_tokens),
        "host": {
            "machine_id": Path("/etc/machine-id").read_text().strip(),
            "tuned_active": _tuned_active(),
            "cpu_governor": _cpu_governor(),
        },
        "env": env,
        "status": "started",
    }

    started = time.perf_counter()
    # Installed before construction: the base workspace is allocated in
    # __init__ from prefill_chunk_size, so a later assignment cannot enlarge it.
    _install_chunk_override(int(args.generator_chunk))
    with _temporary_env(env):
        llm = LLM(
            str(args.model_root),
            quant="gguf_ud_q4_k_xl",
            backend="hip_gfx1151",
        )
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=args.max_sequence_length,
                sampling_params=SamplingParams(max_tokens=args.peer_max_tokens),
            )
            serving = adapter._runner
            sessions = _session_runners(serving)
            # The profile binder re-asserts 1024 on the base runner after
            # construction; keep the attribute consistent with the workspace
            # that prefill_chunked will actually select.
            for session in sessions:
                session.prefill_chunk_size = int(args.generator_chunk)
            report["session_type"] = type(sessions[0]).__name__
            report["serving_type"] = type(serving).__name__
            report["session_count"] = len(sessions)
            report["session_chunks_after_set"] = [
                int(s.prefill_chunk_size) for s in sessions
            ]
            report["session_workspace_chunks"] = [
                [
                    int(w.prefill_chunk_size)
                    for w in (getattr(s, "_prefill_workspaces", None) or [])
                ]
                for s in sessions
            ]
            report["loop_prefill_chunk_size"] = int(adapter._loop.prefill_chunk_size)
            # The workspace is sized by min(prefill_chunk_size, max_sequence_length)
            # and context admission may clamp the requested context to fit device
            # memory, so a clamped context silently undersizes every prefill
            # input buffer.  Record what was actually admitted.
            admission = getattr(adapter, "context_admission", None)
            if admission is None:
                admission = getattr(getattr(adapter, "generator", None), "context_admission", None)
            report["context_admission"] = _jsonable(admission)
            report["session_max_sequence_length"] = [
                int(getattr(s, "max_sequence_length", -1)) for s in sessions
            ]
            report["prefill_token_ids_nbytes"] = [
                int(getattr(getattr(s, "prefill_token_ids", None), "nbytes", -1))
                for s in sessions
            ]
            # min(prefill_chunk_size, max_sequence_length) sizes every prefill
            # input buffer, so an admitted context below the prompt length means
            # this arm cannot run at all.  Report it instead of running a
            # 20-minute measurement that dies in copy_host_to_device.
            needed = len(long_ids) + len(short_ids) + 8
            admitted = min(report["session_max_sequence_length"], default=-1)
            if admitted < needed:
                report["status"] = "context_clamped"
                report["context_needed_tokens"] = needed
                return report

            _install_instrumentation(trace)

            def request(prompt: Sequence[int], max_tokens: int) -> GenerationRequest:
                return GenerationRequest(
                    prompts=(tuple(int(t) for t in prompt),),
                    max_tokens=int(max_tokens),
                    temperature=0.0,
                    top_p=1.0,
                    ignore_eos=True,
                )

            tokens: list[dict[str, Any]] = []
            work: list[dict[str, Any]] = []
            counts: dict[int, int] = {}
            peer_id: int | None = None
            prefill_id: int | None = None

            def drain(max_ticks: int = 1) -> bool:
                events = tuple(adapter.poll(max_ticks=max_ticks))
                now = time.perf_counter()
                for event in events:
                    if event.kind == "token":
                        rid = int(event.request_id)
                        tokens.append({"t": now, "request_id": rid})
                        counts[rid] = counts.get(rid, 0) + 1
                    elif event.kind == "work":
                        work.append(
                            {
                                "t": now,
                                "kind": str(getattr(event.work_kind, "value", event.work_kind)),
                                "request_ids": [int(r) for r in event.request_ids],
                            }
                        )
                return bool(events)

            def submit_peer() -> float:
                nonlocal peer_id
                submission = adapter.submit_detailed(
                    request(short_ids, args.peer_max_tokens)
                )
                peer_id = int(submission.request_ids[0])
                counts[peer_id] = 0
                return time.perf_counter()

            def submit_prefill() -> float:
                nonlocal prefill_id
                submission = adapter.submit_detailed(request(long_ids, 1))
                prefill_id = int(submission.request_ids[0])
                counts[prefill_id] = 0
                return time.perf_counter()

            # protect_decode gives an *already decoding* row strict priority, so
            # a peer submitted first finishes before the prefill is ever
            # scheduled.  The interleaving question is therefore a decode row
            # *arriving during* a long prefill, which is the ``prefill_first``
            # order: start the prefill, then land the peer on top of it.  The
            # submission itself has to be ordered, not just the polling -- both
            # rows present in the queue from tick one makes the two arms
            # identical.
            if args.order == "prefill_first":
                prefill_submit_t = submit_prefill()
                if not drain():
                    raise RuntimeError("loop stalled starting the prefill")
                peer_submit_t = submit_peer()
            elif args.order == "queued":
                # Both rows are in the queue before the first tick, long row
                # first, so the loop's own FIFO ordering decides and the two
                # arms stay comparable.  prefill_first cannot do this reliably:
                # poll() runs a whole tick synchronously, so whether the peer
                # lands during or after the prefill depends on how many ticks
                # the cap makes the prefill take.
                prefill_submit_t = submit_prefill()
                peer_submit_t = submit_peer()
            else:
                peer_submit_t = submit_peer()
                prime_deadline = time.perf_counter() + args.timeout_seconds
                while counts.get(peer_id, 0) < args.peer_primer_tokens:
                    if not drain():
                        raise RuntimeError("loop stalled while priming the peer")
                    if time.perf_counter() > prime_deadline:
                        raise RuntimeError("timed out priming the peer")
                prefill_submit_t = submit_prefill()

            assert peer_id is not None and prefill_id is not None
            deadline = time.perf_counter() + args.timeout_seconds
            while True:
                if not drain():
                    break
                if time.perf_counter() > deadline:
                    report["status"] = "timeout"
                    break
                if counts.get(peer_id, 0) >= args.peer_max_tokens and counts.get(prefill_id, 0) >= 1:
                    drain()
                    break

            if report["status"] == "started":
                report["status"] = "completed"
            report["elapsed_seconds"] = time.perf_counter() - started
            report["tokens"] = tokens
            report["work"] = work
            report["trace"] = trace
            report["peer_submit_t"] = peer_submit_t
            report["prefill_submit_t"] = prefill_submit_t
            report["peer_id"] = peer_id
            report["prefill_id"] = prefill_id
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


def _tuned_active() -> str | None:
    try:
        return subprocess.run(
            ["tuned-adm", "active"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return None


def _cpu_governor() -> str | None:
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except Exception:
        return None


def _summarize_arm(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce one arm's raw trace to the peer-fairness numbers."""

    peer_id = row["peer_id"]
    prefill_id = row["prefill_id"]
    peer_times = [t["t"] for t in row["tokens"] if t["request_id"] == peer_id]
    prefill_times = [t["t"] for t in row["tokens"] if t["request_id"] == prefill_id]
    peer_submit = row["peer_submit_t"]
    prefill_submit = row["prefill_submit_t"]

    gaps = [
        {"start": a, "end": b, "ms": (b - a) * 1000.0}
        for a, b in zip(peer_times, peer_times[1:])
    ]
    # The prefill window runs from the engine-loop tick that starts the *long*
    # row's prefill to that row's first generated token.  Taking the minimum of
    # all prefill ticks instead would fold in the peer's own short prompt and
    # widen the window to the whole run.
    long_tokens = row.get("prefill_prompt_tokens")
    batch_times = [
        c["t"]
        for c in row["trace"]["loop_prefill_batch"]
        if long_tokens is None or c["total"] >= int(long_tokens)
    ]
    window_start = min(batch_times) if batch_times else None
    window_end = prefill_times[0] if prefill_times else None

    def overlaps(gap: dict[str, Any]) -> bool:
        if window_start is None or window_end is None:
            return False
        return gap["end"] > window_start and gap["start"] < window_end

    during = [g for g in gaps if overlaps(g)]
    outside = [g for g in gaps if not overlaps(g)]

    def stats(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(values),
            "min_ms": ordered[0],
            "median_ms": statistics.median(ordered),
            "max_ms": ordered[-1],
            "mean_ms": statistics.fmean(ordered),
        }

    loop_chunks = [c["total"] for c in row["trace"]["loop_prefill_batch"]]
    session_calls = row["trace"]["session_prefill"]
    workspace_chunks = sorted(
        {
            int(chunk)
            for call in session_calls
            for chunk in call.get("workspace_chunks", [])
        }
    )
    return {
        "arm": row["arm"],
        "status": row["status"],
        "order": row.get("order"),
        "generator_chunk_requested": row["generator_chunk_requested"],
        "loop_cap_requested": row["loop_cap_requested"],
        "session_chunks_after_set": row.get("session_chunks_after_set"),
        "session_workspace_chunks": row.get("session_workspace_chunks"),
        "session_count": row.get("session_count"),
        "workspace_chunk_observed_at_prefill": workspace_chunks,
        "loop_prefill_chunk_size": row.get("loop_prefill_chunk_size"),
        "context_admission": row.get("context_admission"),
        "session_max_sequence_length": row.get("session_max_sequence_length"),
        "prefill_token_ids_nbytes": row.get("prefill_token_ids_nbytes"),
        "loop_prefill_batch_calls": len(loop_chunks),
        "loop_prefill_batch_chunks": loop_chunks,
        "session_prefill_calls": [
            {
                "n_tokens": c["n_tokens"],
                "prefill_chunk_size": c["prefill_chunk_size"],
                "workspace_chunks": c.get("workspace_chunks"),
            }
            for c in session_calls
        ],
        "peer_tokens": len(peer_times),
        "peer_ttft_s": (peer_times[0] - peer_submit) if peer_times else None,
        "prefill_wall_s": (prefill_times[0] - prefill_submit) if prefill_times else None,
        "prefill_window_s": (
            (window_end - window_start) if window_start and window_end else None
        ),
        "peer_itl_all": stats([g["ms"] for g in gaps]),
        "peer_itl_during_prefill": stats([g["ms"] for g in during]),
        "peer_itl_outside_prefill": stats([g["ms"] for g in outside]),
        "peer_max_gap_ms": max((g["ms"] for g in gaps), default=None),
        "work_kinds": _work_kind_runs(row["work"]),
        "env": row["env"],
        "host": row["host"],
        "elapsed_seconds": row["elapsed_seconds"],
    }


def _work_kind_runs(work: Sequence[dict[str, Any]]) -> list[str]:
    runs: list[str] = []
    for item in work:
        if not runs or runs[-1] != item["kind"]:
            runs.append(item["kind"])
    return runs


def _run_arm(
    arm: str,
    generator_chunk: int,
    loop_cap: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    directory = args.work_root / arm
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "child.json"
    if output.exists():
        output.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arm",
        arm,
        "--generator-chunk",
        str(generator_chunk),
        "--loop-cap",
        str(loop_cap),
        "--model-root",
        str(args.model_root),
        "--fixture",
        str(args.fixture),
        "--peer-max-tokens",
        str(args.peer_max_tokens),
        "--order",
        str(args.order),
        "--peer-primer-tokens",
        str(args.peer_primer_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--output",
        str(output),
    ]
    if args.compiler_version_file is not None:
        command += ["--compiler-version-file", str(args.compiler_version_file)]

    started = time.monotonic()
    log = directory / "process.log"
    with log.open("wb") as handle:
        completed = subprocess.run(
            command, cwd=_REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.monotonic() - started
    row: dict[str, Any] = {
        "arm": arm,
        "generator_chunk": generator_chunk,
        "loop_cap": loop_cap,
        "returncode": completed.returncode,
        "process_seconds": elapsed,
        "command": command,
        "log": str(log),
    }
    if not output.exists():
        row["status"] = "no_output"
        return row
    child = json.loads(output.read_text())
    row["status"] = "measured"
    row["summary"] = _summarize_arm(child)
    return row


def _run_parent(args: argparse.Namespace) -> int:
    arms = args.arms or list(ARMS)
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_prefill_peer_fairness",
        "performance_claim": False,
        "numerics_evaluated": False,
        "policy": "protect_decode",
        "model_root": str(args.model_root),
        "fixture": str(args.fixture),
        "host": {
            "machine_id": Path("/etc/machine-id").read_text().strip(),
            "tuned_active": _tuned_active(),
            "cpu_governor": _cpu_governor(),
        },
        "arms": [],
    }
    for name in arms:
        generator_chunk, loop_cap = ARMS[name]
        report["arms"].append(_run_arm(name, generator_chunk, loop_cap, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for arm in report["arms"]:
        summary = arm.get("summary")
        if summary is None:
            print(f"{arm['arm']:14s} status={arm['status']} rc={arm['returncode']}")
            continue
        during = summary["peer_itl_during_prefill"]
        print(
            f"{arm['arm']:14s} ws={summary['workspace_chunk_observed_at_prefill']} "
            f"loop_calls={summary['loop_prefill_batch_calls']} "
            f"prefill_wall={summary['prefill_wall_s']} "
            f"peer_ttft={summary['peer_ttft_s']} "
            f"peer_itl_during={during.get('median_ms')}/{during.get('max_ms')} ms"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", default="g1024_l1024")
    parser.add_argument("--generator-chunk", type=int, default=1024)
    parser.add_argument("--loop-cap", type=int, default=1024)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--prefill-case", default="code-p4096")
    parser.add_argument("--peer-case", default="code-p512")
    parser.add_argument(
        "--order",
        choices=("prefill_first", "peer_first", "queued"),
        default="queued",
        help=(
            "queued puts the long row and then the peer in the queue before the "
            "first tick, so the loop's own FIFO decides and both arms are "
            "comparable; peer_first shows that protect_decode gives an "
            "already-decoding row priority and starves the prefill; "
            "prefill_first is unreliable because poll() runs a whole tick "
            "synchronously"
        ),
    )
    parser.add_argument("--peer-prompt-tokens", type=int, default=16)
    parser.add_argument("--peer-max-tokens", type=int, default=64)
    parser.add_argument("--peer-primer-tokens", type=int, default=8)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        # The RT-3 packet sized this as max(768, prompt_tokens + 1), i.e. 4097 for
        # the p4096 case, and the workspace is min(prefill_chunk_size,
        # max_sequence_length) rows.  Requesting 8192 here made context admission
        # clamp back down, which then undersized every prefill input buffer and
        # killed the chunk-4096 arms in copy_host_to_device.  Match the protocol
        # that produced the -4.94% instead of inventing a larger context.
        default=4097,
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/qwen4exp-peer-fairness"))
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        return _run_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
