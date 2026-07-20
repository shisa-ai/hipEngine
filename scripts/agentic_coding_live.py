#!/usr/bin/env python3
"""Collect deterministic coding-agent tool turns from a live hipEngine server.

A1 connects to an already-running localhost server. Each measured SSE request is
matched against an independent non-streaming exact-token oracle built from the
same canonical fixture transcript. The output is passed through the fail-closed
A0 validator; it is diagnostic and cannot set ``performance_claim=true``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic import (  # noqa: E402
    AGENTIC_RECORDS_KIND,
    DEFAULT_AGENTIC_WORKLOADS,
    AgenticBenchmarkError,
    AgenticWorkloadSuite,
    build_agentic_benchmark_artifact,
    load_agentic_workload_suite,
)
from hipengine.benchmark.agentic_live import (  # noqa: E402
    ChatToolOracle,
    build_canonical_turn_messages,
    build_openai_tools,
    final_ownership_from_server,
    normalize_chat_oracle,
    normalize_chat_sse_turn,
    render_workload_prefix,
)


class LiveHTTPTransport:
    """Small thread-safe HTTP/SSE client; each call owns its connection."""

    def __init__(self, base_url: str, *, api_key: str | None = None, timeout_s: float = 600.0):
        parsed = urllib.parse.urlsplit(str(base_url).rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise AgenticBenchmarkError("base_url must be an http(s) URL")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._api_prefix = parsed.path.rstrip("/")
        self._root_prefix = (
            self._api_prefix[: -len("/v1")] if self._api_prefix.endswith("/v1") else ""
        )
        self._api_key = None if api_key is None else str(api_key)
        self._timeout_s = float(timeout_s)

    def _connection(self):
        owner = (
            http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        )
        return owner(self._host, self._port, timeout=self._timeout_s)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        api: bool = True,
    ) -> dict[str, Any]:
        prefix = self._api_prefix if api else self._root_prefix
        target = f"{prefix}/{path.lstrip('/')}"
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = self._connection()
        try:
            connection.request(method, target, body=body, headers=self._headers())
            response = connection.getresponse()
            raw = response.read()
        finally:
            connection.close()
        if response.status < 200 or response.status >= 300:
            raise AgenticBenchmarkError(
                f"{method} {target} failed with HTTP {response.status}: "
                f"{raw.decode('utf-8', errors='replace')[:1000]}"
            )
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise AgenticBenchmarkError(f"{method} {target} did not return a JSON object")
        return decoded

    def tokenize(self, text: str) -> list[int]:
        payload = self._json_request("POST", "hipengine/tokenize", {"text": str(text)})
        tokens = payload.get("token_ids")
        if not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
            raise AgenticBenchmarkError("tokenize response is missing exact token_ids")
        return [int(item) for item in tokens]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        payload = self._json_request(
            "POST",
            "hipengine/detokenize",
            {"token_ids": [int(item) for item in token_ids], "skip_special": False},
        )
        text = payload.get("text")
        if not isinstance(text, str):
            raise AgenticBenchmarkError("detokenize response is missing text")
        return text

    def rendered_prompt_ids(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: Mapping[str, Any],
    ) -> list[int]:
        rendered = self._json_request(
            "POST",
            "hipengine/count_tokens",
            {
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": dict(tool_choice),
                "enable_thinking": False,
            },
        )
        text = rendered.get("text")
        if not isinstance(text, str):
            raise AgenticBenchmarkError("count_tokens response is missing rendered text")
        tokens = self.tokenize(text)
        if rendered.get("token_count") != len(tokens):
            raise AgenticBenchmarkError("count_tokens and tokenize disagree on prompt length")
        return tokens

    def chat_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._json_request("POST", "chat/completions", payload)

    def chat_sse(
        self,
        payload: Mapping[str, Any],
        *,
        release: threading.Barrier | None = None,
    ) -> tuple[float, float, list[tuple[float, Mapping[str, Any] | str]]]:
        if release is not None:
            release.wait()
        target = f"{self._api_prefix}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = self._connection()
        submitted = time.perf_counter()
        events: list[tuple[float, Mapping[str, Any] | str]] = []
        try:
            connection.request("POST", target, body=body, headers=self._headers())
            response = connection.getresponse()
            if response.status < 200 or response.status >= 300:
                raw = response.read()
                raise AgenticBenchmarkError(
                    f"POST {target} failed with HTTP {response.status}: "
                    f"{raw.decode('utf-8', errors='replace')[:1000]}"
                )
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                observed = time.perf_counter()
                if data == "[DONE]":
                    events.append((observed, "[DONE]"))
                    continue
                decoded = json.loads(data)
                if not isinstance(decoded, dict):
                    raise AgenticBenchmarkError("SSE data event is not a JSON object")
                events.append((observed, decoded))
        finally:
            completed = time.perf_counter()
            connection.close()
        return submitted, completed, events

    def ready(self) -> dict[str, Any]:
        return self._json_request("GET", "ready", api=False)

    def sessions(self) -> dict[str, Any]:
        return self._json_request("GET", "hipengine/sessions")

    def capabilities(self) -> dict[str, Any]:
        return self._json_request("GET", "hipengine/capabilities")


def _tool_choice(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": str(name)}}


def _chat_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    expected_tool: str,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": str(model),
        "messages": list(messages),
        "tools": list(tools),
        "tool_choice": _tool_choice(expected_tool),
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "enable_thinking": False,
        "stream": bool(stream),
    }
    if stream:
        payload["stream_options"] = {"include_usage": True, "include_hipengine": True}
    return payload


def _wait_for_final_ownership(
    transport: LiveHTTPTransport,
    *,
    cache_mode: str,
    timeout_s: float,
) -> dict[str, int]:
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
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
                f"server did not reach final ownership before timeout: {last_error}"
            ) from last_error
        time.sleep(0.05)


def collect_live_records(
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
    """Collect cache-off deterministic tool rounds from a running server."""

    if concurrency <= 0 or runs <= 0 or max_tokens <= 0:
        raise AgenticBenchmarkError("concurrency, runs, and max_tokens must be positive")
    if cache_mode != "off":
        raise AgenticBenchmarkError(
            "A1 collector supports cache_mode=off only; radix telemetry is an A2 boundary"
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
        for turn_index, turn in enumerate(turns):
            expected_tool = str(turn["expected_tool"])
            prepared: list[tuple[str, str, list[int], dict[str, Any], ChatToolOracle]] = []
            for agent_index in range(concurrency):
                agent_id = f"agent-{agent_index}"
                messages = build_canonical_turn_messages(
                    suite,
                    workload_id,
                    turn_index=turn_index,
                    agent_id=agent_id,
                    prefix_text=prefix.text,
                )
                choice = _tool_choice(expected_tool)
                prompt_ids = transport.rendered_prompt_ids(
                    messages=messages,
                    tools=tools,
                    tool_choice=choice,
                )
                oracle_payload = _chat_payload(
                    model=model,
                    messages=messages,
                    tools=tools,
                    expected_tool=expected_tool,
                    max_tokens=max_tokens,
                    stream=False,
                )
                oracle = normalize_chat_oracle(
                    suite,
                    workload_id,
                    turn_index,
                    transport.chat_json(oracle_payload),
                )
                stream_payload = _chat_payload(
                    model=model,
                    messages=messages,
                    tools=tools,
                    expected_tool=expected_tool,
                    max_tokens=max_tokens,
                    stream=True,
                )
                prepared.append(
                    (agent_id, f"session-{agent_index}", prompt_ids, stream_payload, oracle)
                )

            release = threading.Barrier(concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(transport.chat_sse, stream_payload, release=release)
                    for _agent_id, _session_id, _prompt_ids, stream_payload, _oracle in prepared
                ]
                results = [future.result() for future in futures]
            for prepared_row, result in zip(prepared, results):
                agent_id, session_id, prompt_ids, _stream_payload, oracle = prepared_row
                submitted, completed, events = result
                request_id = f"{run_id}-{agent_id}-turn-{turn_index}"
                records.append(
                    normalize_chat_sse_turn(
                        suite,
                        workload_id=workload_id,
                        turn_index=turn_index,
                        run_id=run_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        request_id=request_id,
                        prompt_token_ids=prompt_ids,
                        submitted_at_s=submitted,
                        tool_result_submitted_at_s=completed,
                        oracle=oracle,
                        events=events,
                        cache_mode=cache_mode,
                    )
                )

    ownership = _wait_for_final_ownership(
        transport,
        cache_mode=cache_mode,
        timeout_s=idle_timeout_s,
    )
    payload = {
        "kind": AGENTIC_RECORDS_KIND,
        "schema_version": 1,
        "configuration": {
            "id": f"live-{workload_id}-c{concurrency}",
            "lane": "deterministic",
            "concurrency": int(concurrency),
            "cache_mode": cache_mode,
            "backend": str(backend),
            "model": str(model),
            "require_complete_workloads": True,
            "performance_claim": False,
            "collector": "real_http_sse_with_independent_nonstreaming_oracle",
            "token_timing_scope": "public_tool_fragments",
        },
        "turn_records": records,
        "final_ownership": ownership,
    }
    return suite, payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the A1 live coding-agent SSE diagnostic.")
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
        suite, records = collect_live_records(
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
        artifact = build_agentic_benchmark_artifact(suite, records)
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
        print(f"live agentic benchmark rejected: {exc}", file=sys.stderr)
        return 2
    coverage = artifact["coverage"]
    rollup = artifact["rollup"]
    print(
        f"A1 diagnostic passed: {coverage['runs']} runs x c{coverage['concurrency']}, "
        f"{coverage['turns']} turns, public TTFT p50 "
        f"{rollup['latency_ms']['ttft']['p50']:.3f} ms -> {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
