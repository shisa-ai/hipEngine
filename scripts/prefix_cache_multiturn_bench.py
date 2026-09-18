#!/usr/bin/env python3
"""Measure GGUF prefix caching on realistic multi-turn serving workloads.

Prefix reuse only pays when a request's prompt contains an earlier request's
prompt verbatim.  The two transcript styles real clients use behave
differently:

* ``cumulative`` - the client resends the whole transcript, including the
  assistant text the model itself produced (chat APIs, ShareGPT lanes).
* ``fixture`` - the client rebuilds the transcript with its own normalized
  assistant/tool-call text (agentic coding clients, the frozen A2 packet).

Lanes:

* ``sharegpt`` - pinned real ShareGPT conversations from
  ``benchmarks/prompts/qwen38-sharegpt-soak-v1.json``, run cumulatively.
* ``code_fixture`` - the checked-in synthetic repository workloads
  (``benchmarks/prompts/agentic-coding-v1.json``) driven exactly as the A2
  packet drives them: the transcript is rebuilt from the fixture each turn.
* ``code_cumulative`` - the same long repository prompts with the model's own
  output carried forward, which is what a normal coding client sends.

Every turn is a full ``generate_detailed`` call through the production resident
loop, rendered with the server's own chat renderer, so the prompt text is what
the OpenAI route would send.  Per-turn evidence: prompt/output token counts,
engine-attributed prefill and decode wall, and the runner's per-request prefix
telemetry (hit, matched/reused tokens, source, fallback reason).

This harness runs in process.  It measures the resident loop and scheduler, not
an HTTP socket.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hipengine  # noqa: E402

if Path(hipengine.__file__).resolve().parents[1] != REPO_ROOT:
    raise RuntimeError("prefix_cache_multiturn_bench imported hipengine from another checkout")

from hipengine import LLM, SamplingParams  # noqa: E402
from hipengine.benchmark.agentic import load_agentic_workload_suite  # noqa: E402
from hipengine.benchmark.agentic_live import (  # noqa: E402
    build_canonical_turn_messages,
    build_openai_tools,
    render_workload_prefix,
)
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.server.api import (  # noqa: E402
    ChatCompletionRequest,
    _render_prepared_chat_prompt_for_request,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_SHAREGPT = REPO_ROOT / "benchmarks/prompts/qwen38-sharegpt-soak-v1.json"
DEFAULT_AGENTIC = REPO_ROOT / "benchmarks/prompts/agentic-coding-v1.json"
_HARDWARE_LABELS = {
    "hip_gfx1100": "AMD Radeon Pro W7900 (gfx1100)",
    "hip_gfx1151": "AMD Radeon 8060S (gfx1151)",
}


@dataclass(frozen=True)
class LaneTurn:
    """One request: the messages to render and the output budget."""

    lane_id: str
    lane_kind: str
    turn_index: int
    messages: tuple[dict[str, Any], ...]
    max_tokens: int


@dataclass
class LaneResult:
    lane_id: str
    lane_kind: str
    turns: list[dict[str, Any]] = field(default_factory=list)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {"samples": [], "count": 0}
    return {
        "samples": samples,
        "count": len(samples),
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "stdev": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    }


def _sum(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values))


# ---------------------------------------------------------------------------
# Lane construction
# ---------------------------------------------------------------------------


def _sharegpt_lanes(path: Path, *, long_lane_response_tokens: int) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lanes: dict[str, dict[str, Any]] = {}
    for lane in payload["lanes"]:
        lane_id = f"sharegpt-{int(lane['lane'])}"
        user_turns = [str(turn) for turn in lane["user_turns"]]
        # A real mix: most chat replies are short, the two longest lanes ask
        # for a long answer so the workload is not uniformly small.
        long_lane = lane_id in {"sharegpt-4", "sharegpt-7"}
        lanes[lane_id] = {
            "kind": "sharegpt",
            "user_turns": user_turns,
            "max_tokens": long_lane_response_tokens if long_lane else 96,
            "system": "You are a helpful assistant. Answer the user's question directly.",
        }
    return lanes


def _agentic_lanes(
    path: Path,
    *,
    tokenize: Any,
    detokenize: Any,
    workload_ids: Sequence[str],
    max_tokens: int,
) -> dict[str, dict[str, Any]]:
    suite = load_agentic_workload_suite(path)
    tools = build_openai_tools(suite)
    lanes: dict[str, dict[str, Any]] = {}
    for workload_id in workload_ids:
        if workload_id not in suite.workloads:
            raise ValueError(f"unknown agentic workload {workload_id!r}")
        prefix = render_workload_prefix(
            suite,
            workload_id,
            tokenize=tokenize,
            detokenize=detokenize,
        )
        lanes[f"fixture-{workload_id}"] = {
            "kind": "code_fixture",
            "suite": suite,
            "workload_id": workload_id,
            "prefix_text": prefix.text,
            "prefix_tokens": prefix.target_tokens,
            "tools": tools,
            "max_tokens": max_tokens,
        }
        lanes[f"cumulative-{workload_id}"] = {
            "kind": "code_cumulative",
            "suite": suite,
            "workload_id": workload_id,
            "prefix_text": prefix.text,
            "prefix_tokens": prefix.target_tokens,
            "tools": tools,
            "max_tokens": max_tokens,
        }
    return lanes


def _agentic_turn_count(suite: Any, workload_id: str) -> int:
    return len(list(suite.workloads[workload_id]["turns"]))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    engine: Any,
    tools: Sequence[Mapping[str, Any]] | None,
    max_tokens: int,
) -> str:
    request = ChatCompletionRequest(
        model=None,
        messages=[dict(message) for message in messages],
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        tools=None if not tools else [dict(tool) for tool in tools],
    )
    prompt, _thinking, _prepared = _render_prepared_chat_prompt_for_request(
        request,
        chat_default_max_tokens=None,
        engine=engine,
    )
    return str(prompt)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _prefix_block(output: Any) -> dict[str, Any]:
    telemetry = getattr(output, "telemetry", None)
    diagnostics = None
    if telemetry is not None:
        diagnostics = getattr(telemetry, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        block = diagnostics.get("prefix_cache")
        if isinstance(block, Mapping):
            return dict(block)
    return {}


def _timing_block(output: Any) -> dict[str, float]:
    telemetry = getattr(output, "telemetry", None)
    timing = None if telemetry is None else getattr(telemetry, "timing", None)
    if not isinstance(timing, Mapping):
        return {}
    return {str(key): float(value) for key, value in timing.items()}


def _gtt_used_bytes() -> int | None:
    """Unified-memory (GTT) bytes held by the GPU on this host, when visible."""

    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used")):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _run_turn(
    llm: LLM,
    *,
    engine: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    max_tokens: int,
) -> dict[str, Any]:
    render_started = time.perf_counter()
    prompt = _render_messages(
        messages,
        engine=engine,
        tools=tools,
        max_tokens=max_tokens,
    )
    render_ms = (time.perf_counter() - render_started) * 1000.0
    sampling = SamplingParams(
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    prompt_tokens = None
    counter = getattr(llm, "count_tokens", None)
    if callable(counter):
        try:
            prompt_tokens = int(counter(prompt))
        except Exception:  # pragma: no cover - tokenizer diagnostic only
            prompt_tokens = None
    started = time.perf_counter()
    gtt_before = _gtt_used_bytes()
    output = llm.generate_detailed([prompt], sampling)[0]
    wall_ms = (time.perf_counter() - started) * 1000.0
    gtt_after = _gtt_used_bytes()
    generated = output.generated_token_ids or ()
    timing = _timing_block(output)
    prefix = _prefix_block(output)
    return {
        "render_ms": render_ms,
        "wall_ms": wall_ms,
        "prefill_ms": float(timing.get("prefill_ms", 0.0)),
        "decode_ms": float(timing.get("decode_ms", 0.0)),
        "prompt_tokens": int(
            prompt_tokens
            if prompt_tokens is not None
            else (output.prompt_tokens or 0)
        ),
        "generator_prompt_tokens": (
            None if output.prompt_tokens is None else int(output.prompt_tokens)
        ),
        "output_tokens": len(generated),
        "gtt_bytes_before": gtt_before,
        "gtt_bytes_after": gtt_after,
        "text": str(output.text),
        "prefix": prefix,
    }


def _run_lane(
    llm: LLM,
    *,
    engine: Any,
    lane_id: str,
    lane: Mapping[str, Any],
    turn_limit: int | None,
) -> LaneResult:
    kind = str(lane["kind"])
    result = LaneResult(lane_id=lane_id, lane_kind=kind)
    if kind == "sharegpt":
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": str(lane["system"])}
        ]
        user_turns = list(lane["user_turns"])
        if turn_limit is not None:
            user_turns = user_turns[: int(turn_limit)]
        for index, user_text in enumerate(user_turns):
            messages.append({"role": "user", "content": user_text})
            record = _run_turn(
                llm,
                engine=engine,
                messages=messages,
                tools=None,
                max_tokens=int(lane["max_tokens"]),
            )
            record.update(
                {"lane_id": lane_id, "lane_kind": kind, "turn_index": index}
            )
            result.turns.append(record)
            messages.append({"role": "assistant", "content": record["text"]})
        return result

    suite = lane["suite"]
    workload_id = str(lane["workload_id"])
    prefix_text = str(lane["prefix_text"])
    tools = lane["tools"]
    total_turns = _agentic_turn_count(suite, workload_id)
    if turn_limit is not None:
        total_turns = min(total_turns, int(turn_limit))
    if kind == "code_fixture":
        for index in range(total_turns):
            messages = build_canonical_turn_messages(
                suite,
                workload_id,
                turn_index=index,
                agent_id=f"bench-{workload_id}",
                prefix_text=prefix_text,
            )
            record = _run_turn(
                llm,
                engine=engine,
                messages=messages,
                tools=tools,
                max_tokens=int(lane["max_tokens"]),
            )
            record.update(
                {"lane_id": lane_id, "lane_kind": kind, "turn_index": index}
            )
            result.turns.append(record)
        return result

    # code_cumulative: a plain multi-turn coding conversation over the same
    # long repository prompt, with the model's own reply carried forward.
    workload_turns = list(suite.workloads[workload_id]["turns"])
    messages = [
        {
            "role": "system",
            "content": f"You are a coding assistant for the repository below.\n{prefix_text}",
        }
    ]
    for index in range(total_turns):
        messages.append({"role": "user", "content": str(workload_turns[index]["user"])})
        record = _run_turn(
            llm,
            engine=engine,
            messages=messages,
            tools=None,
            max_tokens=int(lane["max_tokens"]),
        )
        record.update({"lane_id": lane_id, "lane_kind": kind, "turn_index": index})
        result.turns.append(record)
        messages.append({"role": "assistant", "content": record["text"]})
    return result


def _summarize_mode(results: Sequence[LaneResult]) -> dict[str, Any]:
    turns = [turn for lane in results for turn in lane.turns]
    hits = [turn for turn in turns if bool(turn["prefix"].get("hit"))]
    lookups = [turn for turn in turns if bool(turn["prefix"].get("lookup"))]
    fallbacks: dict[str, int] = {}
    for turn in turns:
        reason = turn["prefix"].get("fallback_reason")
        if reason:
            fallbacks[str(reason)] = fallbacks.get(str(reason), 0) + 1
    output_tokens = _sum([turn["output_tokens"] for turn in turns])
    wall_ms = _sum([turn["wall_ms"] for turn in turns])
    by_kind: dict[str, dict[str, Any]] = {}
    for lane in results:
        bucket = by_kind.setdefault(
            lane.lane_kind,
            {"turns": 0, "hits": 0, "lookups": 0, "wall_ms": 0.0, "output_tokens": 0},
        )
        for turn in lane.turns:
            bucket["turns"] += 1
            bucket["hits"] += 1 if bool(turn["prefix"].get("hit")) else 0
            bucket["lookups"] += 1 if bool(turn["prefix"].get("lookup")) else 0
            bucket["wall_ms"] += float(turn["wall_ms"])
            bucket["output_tokens"] += int(turn["output_tokens"])
    return {
        "turns": len(turns),
        "hits": len(hits),
        "lookups": len(lookups),
        "hit_rate_of_lookups": (
            float(len(hits)) / float(len(lookups)) if lookups else 0.0
        ),
        "reused_tokens": _sum(
            [int(turn["prefix"].get("reused_tokens", 0)) for turn in turns]
        ),
        "prompt_tokens": _sum([int(turn["prompt_tokens"]) for turn in turns]),
        "output_tokens": output_tokens,
        "wall_ms": wall_ms,
        "output_tokens_per_second": (
            output_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
        ),
        "prefill_ms": _distribution([turn["prefill_ms"] for turn in turns]),
        "wall_ms_distribution": _distribution([turn["wall_ms"] for turn in turns]),
        "fallback_reasons": fallbacks,
        "by_kind": by_kind,
    }


def _repo_state() -> dict[str, Any]:
    import subprocess

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {"revision": revision, "tracked_status": status}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    quant = str(args.quant)
    modes = [str(mode) for mode in args.modes]
    if len(modes) != 1:
        raise SystemExit(
            "run exactly one mode per process: this model needs ~64 GiB of "
            "unified memory, so loading two modes in one process thrashes the "
            "host. Invoke once with --modes off and once with --modes radix, "
            "then pass the first artifact as --baseline-artifact."
        )
    output: dict[str, Any] = {
        "schema": 1,
        "kind": "gguf_prefix_cache_multiturn_bench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "label": args.hardware_label
            or _HARDWARE_LABELS.get(str(args.backend), str(args.backend)),
            "backend": str(args.backend),
            "hostname": platform.node(),
        },
        "model": {"path": str(model_path), "quant": quant},
        "protocol": {
            "scope": "in-process resident loop; server chat renderer; greedy ignore_eos",
            "lanes": args.lane_kinds,
            "turn_limit": args.turn_limit,
            "modes": modes,
            "repetitions": int(args.repetitions),
            "max_tokens": {
                "sharegpt_short": int(args.sharegpt_tokens),
                "sharegpt_long": int(args.sharegpt_long_tokens),
                "code": int(args.code_tokens),
            },
            "agentic_workloads": list(args.agentic_workloads),
        },
        "repo": _repo_state(),
        "modes": {},
        "notes": [],
    }
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        model_path=str(model_path),
        quant=quant,
        command=list(sys.argv),
        warmups=0,
        repetitions=int(args.repetitions),
        timing_protocol="in-process multi-turn, greedy ignore_eos, engine-attributed prefill/decode",
    )
    if provenance:
        output["provenance"] = provenance

    for mode in modes:
        print(f"=== mode {mode}: loading {model_path.name}", flush=True)
        llm = LLM(
            str(model_path),
            backend=str(args.backend),
            quant=quant,
            prefix_cache=mode,
            max_active_requests=int(args.max_active_requests),
            max_sequence_length=int(args.max_sequence_length),
        )
        engine = llm
        # Warm up the model + a short prefix so the first lane is not paying
        # graph capture and allocator growth.
        _run_turn(
            llm,
            engine=engine,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            tools=None,
            max_tokens=8,
        )
        lanes = _sharegpt_lanes(
            Path(args.sharegpt),
            long_lane_response_tokens=int(args.sharegpt_long_tokens),
        )
        lanes.update(
            _agentic_lanes(
                Path(args.agentic),
                tokenize=lambda text: llm.tokenize(text),
                detokenize=lambda ids: llm.detokenize(ids),
                workload_ids=list(args.agentic_workloads),
                max_tokens=int(args.code_tokens),
            )
        )
        if args.lane_kinds == "sharegpt":
            lanes = {k: v for k, v in lanes.items() if v["kind"] == "sharegpt"}
        elif args.lane_kinds == "code":
            lanes = {k: v for k, v in lanes.items() if v["kind"] != "sharegpt"}
        lane_ids = sorted(lanes)
        repetitions: list[dict[str, Any]] = []
        for repetition in range(int(args.repetitions)):
            ordered = lane_ids if repetition % 2 == 0 else list(reversed(lane_ids))
            results: list[LaneResult] = []
            for lane_id in ordered:
                print(f"    [{mode} r{repetition}] lane {lane_id}", flush=True)
                results.append(
                    _run_lane(
                        llm,
                        engine=engine,
                        lane_id=lane_id,
                        lane=lanes[lane_id],
                        turn_limit=args.turn_limit,
                    )
                )
            repetitions.append(
                {
                    "repetition": repetition,
                    "lane_order": list(ordered),
                    "summary": _summarize_mode(results),
                    "lanes": [
                        {
                            "lane_id": lane.lane_id,
                            "lane_kind": lane.lane_kind,
                            "turns": [
                                {
                                    key: value
                                    for key, value in turn.items()
                                    if key != "text"
                                }
                                for turn in lane.turns
                            ],
                        }
                        for lane in results
                    ],
                }
            )
        output["modes"][mode] = {
            "repetitions": repetitions,
            "live_loop_snapshot": llm.live_loop_snapshot(),
        }
        del llm

    output["comparison"] = _compare(
        output["modes"],
        baseline_artifact=args.baseline_artifact,
    )
    return output


def _compare(
    modes: Mapping[str, Any],
    *,
    baseline_artifact: Path | None = None,
) -> dict[str, Any]:
    if "off" in modes and "radix" in modes:
        off_payload = modes["off"]
        radix_payload = modes["radix"]
    elif baseline_artifact is not None and "radix" in modes:
        baseline = json.loads(Path(baseline_artifact).read_text(encoding="utf-8"))
        off_payload = baseline["modes"]["off"]
        radix_payload = modes["radix"]
    elif baseline_artifact is not None and "off" in modes:
        baseline = json.loads(Path(baseline_artifact).read_text(encoding="utf-8"))
        off_payload = modes["off"]
        radix_payload = baseline["modes"]["radix"]
    else:
        return {}
    comparison: dict[str, Any] = {
        "baseline_artifact": (
            None if baseline_artifact is None else str(baseline_artifact)
        ),
        "baseline_repo_revision": None,
    }
    if baseline_artifact is not None:
        baseline = json.loads(Path(baseline_artifact).read_text(encoding="utf-8"))
        comparison["baseline_repo_revision"] = baseline.get("repo", {}).get(
            "revision"
        )
    off_reps = list(off_payload["repetitions"])
    radix_reps = list(radix_payload["repetitions"])
    pairs = min(len(off_reps), len(radix_reps))
    for index in range(pairs):
        off = off_reps[index]["summary"]
        radix = radix_reps[index]["summary"]
        comparison[f"repetition_{index}"] = {
            "off_wall_ms": off["wall_ms"],
            "radix_wall_ms": radix["wall_ms"],
            "wall_delta_percent": (
                100.0 * (radix["wall_ms"] - off["wall_ms"]) / off["wall_ms"]
                if off["wall_ms"]
                else 0.0
            ),
            "off_output_tokens_per_second": off["output_tokens_per_second"],
            "radix_output_tokens_per_second": radix["output_tokens_per_second"],
            "off_hits": off["hits"],
            "radix_hits": radix["hits"],
            "radix_lookups": radix["lookups"],
            "radix_reused_tokens": radix["reused_tokens"],
            "radix_fallback_reasons": radix["fallback_reasons"],
            "off_by_kind": off["by_kind"],
            "radix_by_kind": radix["by_kind"],
        }
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--sharegpt", type=Path, default=DEFAULT_SHAREGPT)
    parser.add_argument("--agentic", type=Path, default=DEFAULT_AGENTIC)
    parser.add_argument(
        "--agentic-workloads",
        nargs="*",
        default=["small_repo", "growing_history"],
    )
    parser.add_argument(
        "--lane-kinds",
        choices=("all", "sharegpt", "code"),
        default="all",
    )
    parser.add_argument("--modes", nargs="*", default=["off"])
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=None,
        help="artifact from the other mode; adds a paired comparison block",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--turn-limit", type=int, default=None)
    parser.add_argument("--sharegpt-tokens", type=int, default=96)
    parser.add_argument("--sharegpt-long-tokens", type=int, default=384)
    parser.add_argument("--code-tokens", type=int, default=192)
    parser.add_argument("--max-active-requests", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=16384)
    parser.add_argument("--hardware-label")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run(args)
    text = json.dumps(output, indent=1, default=str)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    summary = {
        mode: payload["repetitions"][0]["summary"]
        for mode, payload in output["modes"].items()
        if payload.get("repetitions")
    }
    print(json.dumps({"summary": summary, "comparison": output["comparison"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
