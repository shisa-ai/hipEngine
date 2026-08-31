#!/usr/bin/env python3
"""Measure direct runner-level serial versus packed multi-request prefill.

This diagnostic deliberately bypasses EngineLoop admission, server queueing,
and ``_try_prefill_native_work_batch``. It therefore answers only whether the
shared resident runner can execute a supplied session group efficiently and
exactly; it cannot establish which groups the shipping server forms. Serving
attribution must come from server route counters such as
``gguf_mtp_c1c8_server_bench.py --capture-prefill-attribution``.

One owner loads the model. Every peer session shares that owner's exact runtime
and runner. Four direct-runner arms are counterbalanced:

  mixed_serial -- one runner call per real unequal-length prompt
  mixed_packed -- one runner call carrying those same unequal-length prompts
  equal_serial -- one runner call per prompt at a shared reference length
  equal_packed -- one runner call carrying those same equal-length prompts

Correctness is paired only between routes with identical prompts. The artifact
is explicitly labelled runner-level and includes canonical source/model/host/
command provenance.

Usage:
    python3 scripts/gguf_packed_prefill_grouping_probe.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --lengths 35,36,36,39,39,43,46,48 --require-clean-source \
        --output /tmp/gguf-packed-prefill-runner-probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402


@contextmanager
def _temporary_environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_PRODUCTION_ENV = {
    "HIPENGINE_EXECUTION_PROFILE": "production",
}

_PROBE_SCOPE = {
    "level": "runner",
    "serving_path_claim_eligible": False,
    "serving_attribution_source": (
        "scripts/gguf_mtp_c1c8_server_bench.py --capture-prefill-attribution"
    ),
}


def _shared_peer_session_kwargs(owner: Any) -> dict[str, Any]:
    """Return identity-preserving ownership kwargs for every peer session."""

    runtime = getattr(owner, "runtime", None)
    runner = getattr(owner, "runner", None)
    if runtime is None or runner is None:
        raise RuntimeError("GGUF owner did not materialize a runtime and shared runner")
    return {"runtime": runtime, "shared_runner": runner}


def _invocation(argv: Sequence[str] | None) -> list[str]:
    """Return the exact executable/script/argument vector recorded in provenance."""

    if argv is None:
        return [sys.executable, *sys.argv]
    return [sys.executable, str(Path(__file__).resolve()), *map(str, argv)]


def _build_verdict(orders: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = tuple(orders)
    if len(labels) != 2:
        raise ValueError("runner probe requires exactly two counterbalanced orders")
    forward, reverse = (orders[label] for label in labels)

    def ratio(rows: dict[str, Any], serial: str, packed: str) -> float:
        return round(
            float(rows[serial]["wall_ms_median"])
            / float(rows[packed]["wall_ms_median"]),
            4,
        )

    def pair_exact(rows: dict[str, Any], serial: str, packed: str) -> bool:
        return bool(
            rows[serial]["token_ids_identical_across_reps"]
            and rows[packed]["token_ids_identical_across_reps"]
            and rows[serial]["token_ids"] == rows[packed]["token_ids"]
        )

    mixed_exact = pair_exact(forward, "mixed_serial", "mixed_packed") and pair_exact(
        reverse, "mixed_serial", "mixed_packed"
    )
    equal_exact = pair_exact(forward, "equal_serial", "equal_packed") and pair_exact(
        reverse, "equal_serial", "equal_packed"
    )
    return {
        "mixed_packed_vs_serial_forward_order": ratio(
            forward, "mixed_serial", "mixed_packed"
        ),
        "mixed_packed_vs_serial_reversed_order": ratio(
            reverse, "mixed_serial", "mixed_packed"
        ),
        "equal_packed_vs_serial_forward_order": ratio(
            forward, "equal_serial", "equal_packed"
        ),
        "equal_packed_vs_serial_reversed_order": ratio(
            reverse, "equal_serial", "equal_packed"
        ),
        "mixed_prompt_pair_exact": mixed_exact,
        "equal_prompt_pair_exact": equal_exact,
        "passed": bool(mixed_exact and equal_exact),
    }


def _prompt_rows(lengths: Sequence[int], token_id: int) -> tuple[tuple[int, ...], ...]:
    return tuple((int(token_id),) * int(length) for length in lengths)


def _route_once(
    sessions: Sequence[Any],
    prompts: Sequence[tuple[int, ...]],
    *,
    packed: bool,
) -> tuple[float, tuple[int, ...]]:
    """Time one prefill route and return (wall_seconds, per-prompt token ids)."""

    for session in sessions:
        session.reset()
    sessions[0].runtime.device_synchronize()
    start = time.perf_counter()
    tokens: list[int] = []
    if packed:
        results = sessions[0].prefill_batch_native(
            list(prompts),
            sessions=tuple(sessions[: len(prompts)]),
            return_logits=False,
        )
        tokens = [int(result.token_id) for result in results]
    else:
        for index, prompt in enumerate(prompts):
            result = sessions[index].prefill_batch_native(
                [prompt],
                sessions=(sessions[index],),
                return_logits=False,
            )
            tokens.append(int(result[0].token_id))
    # HIP launches are asynchronous: synchronize before reading the clock or the
    # wall measures enqueue latency instead of model completion.
    sessions[0].runtime.device_synchronize()
    wall = time.perf_counter() - start
    return wall, tuple(tokens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--backend", default="hip_gfx1100")
    ap.add_argument("--quant", default="gguf_q4_k_m")
    ap.add_argument(
        "--lengths",
        default="35,36,36,39,39,43,46,48",
        help="comma-separated prompt token counts, one per concurrent lane "
             "(default is the real mixed-length census minus the two >52 outliers)",
    )
    ap.add_argument("--equal-length", type=int, default=45)
    ap.add_argument("--prompt-token-id", type=int, default=9707)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--require-clean-source", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    lengths = tuple(int(value) for value in str(args.lengths).split(",") if value.strip())
    if not lengths:
        ap.error("--lengths must list at least one length")
    lanes = len(lengths)
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = Path(args.compiler_version_file).read_text(encoding="utf-8").splitlines()[0]

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    max_sequence_length = max(*lengths, int(args.equal_length)) + 8
    prompts = _prompt_rows(lengths, int(args.prompt_token_id))
    prompts_equal = _prompt_rows((int(args.equal_length),) * lanes, int(args.prompt_token_id))

    command = _invocation(argv)
    provenance = collect_artifact_provenance(
        repo_root=REPO,
        configured_backend=str(args.backend),
        resolved_backend=(
            None if str(args.backend) == "auto" else str(args.backend)
        ),
        model_path=args.model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_EXECUTION_PROFILE": "production",
            "HIPENGINE_REQUIRE_CACHED_BUILD": (
                "1"
                if bool(args.require_cached_build)
                else os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD")
            ),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        build_profile="production",
        timing_protocol=(
            "blocking wall around reset plus direct resident-runner prefill and "
            "device_synchronize; forward/reverse arm orders; median; no best-of"
        ),
        warmups=int(args.warmup),
        repetitions=int(args.reps),
        profiler={"enabled": False},
    )
    if bool(args.require_clean_source) and bool(provenance["dirty"]):
        raise RuntimeError("--require-clean-source requires a clean git worktree")

    results: dict[str, Any] = {
        "schema": 1,
        "kind": "gguf_runner_packed_prefill_grouping_probe",
        "generated": datetime.now(timezone.utc).isoformat(),
        "scope": dict(_PROBE_SCOPE),
        "provenance": provenance,
        "model": str(args.model),
        "backend": args.backend,
        "quant": args.quant,
        "lanes": lanes,
        "lengths": list(lengths),
        "total_rows": sum(lengths),
        "equal_length": int(args.equal_length),
        "warmup": int(args.warmup),
        "reps": int(args.reps),
        "orders": {},
    }

    stack = ExitStack()
    try:
        with _temporary_environment(_PRODUCTION_ENV):
            owner = stack.enter_context(
                Qwen35GGUFResidentSession(
                    args.model,
                    backend=args.backend,
                    compiler_version=compiler_version,
                    require_cached_build=bool(args.require_cached_build),
                    max_sequence_length=max_sequence_length,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                )
            )
            sessions = [owner]
            peer_kwargs = _shared_peer_session_kwargs(owner)
            for _ in range(lanes - 1):
                sessions.append(
                    stack.enter_context(
                        Qwen35GGUFResidentSession(
                            args.model,
                            backend=args.backend,
                            compiler_version=compiler_version,
                            require_cached_build=bool(args.require_cached_build),
                            max_sequence_length=max_sequence_length,
                            use_wmma_prefill=True,
                            use_gemv_decode=True,
                            **peer_kwargs,
                        )
                    )
                )
            ownership_ok = all(
                session.runtime is owner.runtime and session.runner is owner.runner
                for session in sessions
            )
            if not ownership_ok:
                raise RuntimeError("runner probe peer session ownership diverged")
            results["ownership"] = {
                "model_load_owners": 1,
                "sessions": lanes,
                "all_sessions_share_owner_runtime": True,
                "all_sessions_share_owner_runner": True,
            }
            print(
                f"LOADED lanes={lanes} max_sequence_length={max_sequence_length} "
                "shared_runtime_runner=True",
                flush=True,
            )

            routes = {
                "mixed_serial": (sessions, prompts, False),
                "mixed_packed": (sessions, prompts, True),
                "equal_serial": (sessions, prompts_equal, False),
                "equal_packed": (sessions, prompts_equal, True),
            }
            for order in (
                "mixed_serial,mixed_packed,equal_serial,equal_packed",
                "equal_packed,equal_serial,mixed_packed,mixed_serial",
            ):
                names = order.split(",")
                per_route: dict[str, Any] = {}
                for name in names:
                    sess, prompts_, packed = routes[name]
                    for _ in range(int(args.warmup)):
                        _route_once(sess, prompts_, packed=packed)
                    walls: list[float] = []
                    tokens_seen: list[tuple[int, ...]] = []
                    for _ in range(int(args.reps)):
                        wall, tokens = _route_once(sess, prompts_, packed=packed)
                        walls.append(wall)
                        tokens_seen.append(tokens)
                    rows = sum(len(p) for p in prompts_)
                    median_wall = statistics.median(walls)
                    per_route[name] = {
                        "wall_ms_samples": [round(value * 1000, 2) for value in walls],
                        "wall_ms_median": round(median_wall * 1000, 2),
                        "tok_s": round(rows / median_wall, 3),
                        "token_ids_identical_across_reps": len(set(tokens_seen)) == 1,
                        "token_ids": list(tokens_seen[0]),
                    }
                    print(
                        f"{order} :: {name}: wall={median_wall * 1000:.1f} ms "
                        f"tok_s={rows / median_wall:.1f} tokens={len(set(tokens_seen)) == 1}",
                        flush=True,
                    )
                results["orders"][order] = per_route
    finally:
        stack.close()

    verdict = _build_verdict(results["orders"])
    results["verdict"] = verdict
    results["status"] = "complete" if verdict["passed"] else "failed_correctness"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("VERDICT " + json.dumps(verdict), flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
