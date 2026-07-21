#!/usr/bin/env python3
"""Collect natural automatic-tool quality from a live hipEngine server.

Unlike ``agentic_coding_live.py``, this A6 lane does not produce latency or
goodput rollups. Model failures are retained as scored quality observations and
can never set ``performance_claim=true``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic import (  # noqa: E402
    DEFAULT_AGENTIC_WORKLOADS,
    AgenticBenchmarkError,
    AgenticWorkloadSuite,
    load_agentic_workload_suite,
)
from hipengine.benchmark.agentic_live import (  # noqa: E402
    build_canonical_turn_messages,
    build_openai_tools,
    final_ownership_from_server,
    render_workload_prefix,
)
from hipengine.benchmark.agentic_quality import (  # noqa: E402
    AGENTIC_QUALITY_RECORDS_KIND,
    build_agentic_quality_artifact,
    normalize_chat_quality_turn,
)
from scripts.agentic_coding_live import LiveHTTPTransport  # noqa: E402


def _quality_chat_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "messages": list(messages),
        "tools": list(tools),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "enable_thinking": False,
        "stream": False,
    }


def _wait_for_final_ownership(
    transport: LiveHTTPTransport,
    *,
    cache_mode: str,
    timeout_s: float,
) -> dict[str, int]:
    deadline = time.monotonic() + float(timeout_s)
    last_error: AgenticBenchmarkError | None = None
    while True:
        try:
            return final_ownership_from_server(
                transport.ready(),
                transport.sessions(),
                cache_mode=cache_mode,
            )
        except AgenticBenchmarkError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise AgenticBenchmarkError(
                f"server did not reach final ownership after quality run: {last_error}"
            ) from last_error
        time.sleep(0.05)


def collect_live_quality_records(
    transport: LiveHTTPTransport,
    *,
    workloads_path: str | Path,
    workload_id: str,
    model: str,
    backend: str,
    concurrency: int,
    runs: int,
    max_tokens: int,
    cache_mode: str,
    idle_timeout_s: float,
) -> tuple[AgenticWorkloadSuite, dict[str, Any]]:
    """Collect independent natural auto-tool attempts over canonical valid histories."""

    if concurrency <= 0 or runs <= 0 or max_tokens <= 0:
        raise AgenticBenchmarkError("concurrency, runs, and max_tokens must be positive")
    if cache_mode != "off":
        raise AgenticBenchmarkError(
            "A6 quality collector supports cache_mode=off only until prefix quality gates exist"
        )
    suite = load_agentic_workload_suite(workloads_path)
    if workload_id not in suite.workloads:
        raise AgenticBenchmarkError(f"unknown workload_id {workload_id!r}")
    capabilities = transport.capabilities()
    cache = capabilities.get("cache")
    if isinstance(cache, Mapping) and cache.get("prefix_cache") not in {None, cache_mode}:
        raise AgenticBenchmarkError(
            f"server prefix cache is {cache.get('prefix_cache')!r}, expected {cache_mode!r}"
        )
    prefix = render_workload_prefix(
        suite,
        workload_id,
        tokenize=transport.tokenize,
        detokenize=transport.detokenize,
    )
    tools = build_openai_tools(suite)
    turns = suite.workloads[workload_id]["turns"]
    records: list[dict[str, Any]] = []

    for run_index in range(runs):
        run_id = f"run-{run_index}"
        for turn_index, _turn in enumerate(turns):
            prepared: list[tuple[str, str, list[int], dict[str, Any]]] = []
            for agent_index in range(concurrency):
                agent_id = f"agent-{agent_index}"
                session_id = f"session-{agent_index}"
                messages = build_canonical_turn_messages(
                    suite,
                    workload_id,
                    turn_index=turn_index,
                    agent_id=agent_id,
                    prefix_text=prefix.text,
                )
                prompt_ids = transport.rendered_prompt_ids(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                payload = _quality_chat_payload(
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
                prepared.append((agent_id, session_id, prompt_ids, payload))

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(transport.chat_json, payload)
                    for _agent_id, _session_id, _prompt_ids, payload in prepared
                ]
                responses = [future.result() for future in futures]
            for prepared_row, response in zip(prepared, responses, strict=True):
                agent_id, session_id, prompt_ids, _payload = prepared_row
                request_id = f"{run_id}-{agent_id}-turn-{turn_index}"
                records.append(
                    normalize_chat_quality_turn(
                        suite,
                        workload_id=workload_id,
                        turn_index=turn_index,
                        run_id=run_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        request_id=request_id,
                        prompt_token_ids=prompt_ids,
                        payload=response,
                    )
                )

    ownership = _wait_for_final_ownership(
        transport,
        cache_mode=cache_mode,
        timeout_s=idle_timeout_s,
    )
    return suite, {
        "kind": AGENTIC_QUALITY_RECORDS_KIND,
        "schema_version": 1,
        "configuration": {
            "id": f"live-quality-{workload_id}-c{concurrency}",
            "lane": "auto_tool",
            "concurrency": int(concurrency),
            "cache_mode": cache_mode,
            "backend": str(backend),
            "model": str(model),
            "require_complete_workloads": True,
            "performance_claim": False,
            "tool_choice": "auto",
            "collector": "real_http_blocking_natural_tool_quality",
            "history": "canonical_valid_fixture_transcript",
        },
        "turn_records": records,
        "final_ownership": ownership,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect the non-performance A6 automatic-tool quality lane."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--workloads", type=Path, default=DEFAULT_AGENTIC_WORKLOADS)
    parser.add_argument("--workload", default="small_repo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--cache-mode", choices=("off",), default="off")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--idle-timeout-s", type=float, default=30.0)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        transport = LiveHTTPTransport(
            args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
        )
        suite, records = collect_live_quality_records(
            transport,
            workloads_path=args.workloads,
            workload_id=args.workload,
            model=args.model,
            backend=args.backend,
            concurrency=args.concurrency,
            runs=args.runs,
            max_tokens=args.max_tokens,
            cache_mode=args.cache_mode,
            idle_timeout_s=args.idle_timeout_s,
        )
        artifact = build_agentic_quality_artifact(suite, records)
        args.records_json.parent.mkdir(parents=True, exist_ok=True)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.records_json.write_text(
            json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args.json.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (AgenticBenchmarkError, json.JSONDecodeError, OSError) as exc:
        print(f"live agentic quality benchmark rejected: {exc}", file=sys.stderr)
        return 2
    quality = artifact["quality"]
    print(
        f"A6 quality collected: {quality['successes']}/{quality['attempts']} successful "
        f"tool turns ({100.0 * quality['success_rate']:.1f}%) -> {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
