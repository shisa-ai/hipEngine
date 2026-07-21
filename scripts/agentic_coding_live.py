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
from dataclasses import dataclass
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
from hipengine.benchmark.prompts import file_sha256  # noqa: E402
from hipengine.tokenization.identity import token_ids_sha256  # noqa: E402


DEFAULT_A1_CONTROL = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a1-repeated-baseline.json"
)


@dataclass(frozen=True)
class _ControlOracleRecord:
    prompt_token_count: int
    prompt_token_ids_sha256: str
    oracle: ChatToolOracle


def _load_control_oracles(
    path: str | Path,
    *,
    workload_id: str,
    concurrency: int,
) -> tuple[dict[tuple[str, int], _ControlOracleRecord], dict[str, Any]]:
    control_path = Path(path)
    payload = json.loads(control_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "gfx1100_agentic_a1_repeated_baseline":
        raise AgenticBenchmarkError("cache-off control kind is unsupported")
    if payload.get("status") != "accepted_complete_baseline":
        raise AgenticBenchmarkError("cache-off control is not an accepted complete baseline")
    if payload.get("source_clean_and_pushed") is not True:
        raise AgenticBenchmarkError("cache-off control source is not clean and pushed")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping) or not all(
        acceptance.get(field) is True
        for field in (
            "all_correctness_gates_passed",
            "all_final_ownership_zero",
            "all_target_gpu0_exclusive",
        )
    ):
        raise AgenticBenchmarkError("cache-off control acceptance gates are incomplete")
    matches = [
        row
        for row in payload.get("configurations", ())
        if isinstance(row, Mapping)
        and row.get("workload") == str(workload_id)
        and row.get("logical_concurrency") == int(concurrency)
    ]
    if len(matches) != 1:
        raise AgenticBenchmarkError("cache-off control configuration is missing or ambiguous")
    configuration = matches[0]
    if not all(
        configuration.get(field) is True
        for field in (
            "all_correctness_gates_passed",
            "all_final_ownership_zero",
            "target_gpu0_exclusive",
            "variance_gate_passed",
        )
    ):
        raise AgenticBenchmarkError("cache-off control configuration failed a retained gate")
    command = configuration.get("server_command")
    if not isinstance(command, list) or not any(
        str(value) == "--prefix-cache"
        and index + 1 < len(command)
        and str(command[index + 1]) == "off"
        for index, value in enumerate(command)
    ):
        raise AgenticBenchmarkError(
            "cache-off control server command did not use prefix-cache off"
        )

    controls: dict[tuple[str, int], _ControlOracleRecord] = {}
    observations: dict[tuple[str, int], int] = {}
    samples = configuration.get("samples")
    if not isinstance(samples, list) or len(samples) != int(
        configuration.get("measured_runs", -1)
    ):
        raise AgenticBenchmarkError("cache-off control measured samples are incomplete")
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise AgenticBenchmarkError("cache-off control sample must be an object")
        if sample.get("target_gpu0_exclusive") is not True:
            raise AgenticBenchmarkError(
                "cache-off control sample is not target-GPU exclusive"
            )
        ownership = sample.get("final_ownership")
        if not isinstance(ownership, Mapping) or any(
            int(value) != 0 for value in ownership.values()
        ):
            raise AgenticBenchmarkError(
                "cache-off control sample did not drain ownership"
            )
        records = sample.get("turn_records")
        coverage = sample.get("coverage")
        if (
            not isinstance(records, list)
            or not isinstance(coverage, Mapping)
            or len(records) != int(coverage.get("turns", -1))
        ):
            raise AgenticBenchmarkError(
                "cache-off control turn records are incomplete"
            )
        for record in records:
            if (
                not isinstance(record, Mapping)
                or record.get("workload_id") != str(workload_id)
            ):
                raise AgenticBenchmarkError(
                    "cache-off control workload identity drifted"
                )
            agent_id = str(record.get("agent_id", ""))
            turn_index = int(record.get("turn_index", -1))
            if not agent_id or turn_index < 0:
                raise AgenticBenchmarkError(
                    "cache-off control request identity is invalid"
                )
            prompt = record.get("prompt")
            output = record.get("output")
            tool = record.get("tool")
            finish = record.get("finish")
            if not all(
                isinstance(value, Mapping)
                for value in (prompt, output, tool, finish)
            ):
                raise AgenticBenchmarkError(
                    "cache-off control record envelope is invalid"
                )
            if (
                output.get("generated_token_ids_source") != "response"
                or output.get("sse_exact_ids_observed") is not True
                or finish.get("reason") != "tool_calls"
                or tool.get("arguments_json_valid") is not True
                or tool.get("schema_valid") is not True
                or tool.get("result_linked") is not True
            ):
                raise AgenticBenchmarkError(
                    "cache-off control record failed exact response gates"
                )
            generated = tuple(
                int(token) for token in output.get("generated_token_ids", ())
            )
            if (
                not generated
                or output.get("generated_token_ids_sha256")
                != token_ids_sha256(generated)
            ):
                raise AgenticBenchmarkError(
                    "cache-off control generated-token identity is invalid"
                )
            candidate = _ControlOracleRecord(
                prompt_token_count=int(prompt.get("token_count", -1)),
                prompt_token_ids_sha256=str(
                    prompt.get("token_ids_sha256", "")
                ),
                oracle=ChatToolOracle(
                    generated_token_ids=generated,
                    name=str(tool.get("name", "")),
                    arguments=dict(tool.get("arguments", {})),
                    finish_reason="tool_calls",
                ),
            )
            key = (agent_id, turn_index)
            prior = controls.get(key)
            if prior is not None and prior != candidate:
                raise AgenticBenchmarkError(
                    "cache-off control changed across measured repeats"
                )
            controls[key] = candidate
            observations[key] = observations.get(key, 0) + 1
    agent_turns: dict[str, set[int]] = {}
    for agent_id, turn_index in controls:
        agent_turns.setdefault(agent_id, set()).add(turn_index)
    if set(agent_turns) != {
        f"agent-{agent_index}" for agent_index in range(int(concurrency))
    }:
        raise AgenticBenchmarkError(
            "cache-off control agent coverage is incomplete"
        )
    turn_sets = tuple(agent_turns.values())
    if not turn_sets or any(turns != turn_sets[0] for turns in turn_sets):
        raise AgenticBenchmarkError(
            "cache-off control turn coverage is inconsistent"
        )
    expected_turns = set(range(len(turn_sets[0])))
    if turn_sets[0] != expected_turns:
        raise AgenticBenchmarkError(
            "cache-off control turn coverage is not contiguous"
        )
    if any(count != len(samples) for count in observations.values()):
        raise AgenticBenchmarkError(
            "cache-off control repeat coverage is inconsistent"
        )
    return controls, {
        "path": str(control_path),
        "sha256": file_sha256(control_path),
        "source_revision": str(payload.get("source_revision")),
        "workload": str(workload_id),
        "concurrency": int(concurrency),
        "measured_repeats": len(samples),
    }


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
        tool_choice: Mapping[str, Any] | str,
    ) -> list[int]:
        rendered = self._json_request(
            "POST",
            "hipengine/count_tokens",
            {
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": (
                    dict(tool_choice) if isinstance(tool_choice, Mapping) else str(tool_choice)
                ),
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
    control_artifact: str | Path | None = None,
) -> tuple[AgenticWorkloadSuite, dict[str, Any]]:
    """Collect deterministic tool rounds from a running off/radix server."""

    if concurrency <= 0 or runs <= 0 or max_tokens <= 0:
        raise AgenticBenchmarkError("concurrency, runs, and max_tokens must be positive")
    if cache_mode not in {"off", "radix"}:
        raise AgenticBenchmarkError("cache_mode must be off or radix")
    controls: dict[tuple[str, int], _ControlOracleRecord] = {}
    control_provenance: dict[str, Any] | None = None
    if cache_mode == "radix":
        if control_artifact is None:
            raise AgenticBenchmarkError(
                "radix collection requires a retained cache-off control artifact"
            )
        controls, control_provenance = _load_control_oracles(
            control_artifact,
            workload_id=workload_id,
            concurrency=concurrency,
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
                if cache_mode == "radix":
                    control = controls.get((agent_id, turn_index))
                    if control is None:
                        raise AgenticBenchmarkError(
                            "retained cache-off control has no matching agent turn"
                        )
                    if (
                        len(prompt_ids) != control.prompt_token_count
                        or token_ids_sha256(prompt_ids)
                        != control.prompt_token_ids_sha256
                    ):
                        raise AgenticBenchmarkError(
                            "radix prompt identity differs from retained cache-off control"
                        )
                    oracle = control.oracle
                else:
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
            "collector": (
                "real_http_sse_with_retained_cache_off_control"
                if cache_mode == "radix"
                else "real_http_sse_with_independent_nonstreaming_oracle"
            ),
            "token_timing_scope": "public_tool_fragments",
            **(
                {}
                if control_provenance is None
                else {"cache_off_control": control_provenance}
            ),
        },
        "turn_records": records,
        "final_ownership": ownership,
    }
    return suite, payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect exact A1 cache-off or A2 radix coding-agent SSE rows."
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
    parser.add_argument("--cache-mode", choices=("off", "radix"), default="off")
    parser.add_argument(
        "--cache-off-control",
        type=Path,
        default=DEFAULT_A1_CONTROL,
        help="Retained A1 exact control for radix collection (no live oracle warmup)",
    )
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
            control_artifact=(
                args.cache_off_control if args.cache_mode == "radix" else None
            ),
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
