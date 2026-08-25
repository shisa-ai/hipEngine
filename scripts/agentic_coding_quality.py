#!/usr/bin/env python3
"""Collect natural automatic-tool quality from a live hipEngine server.

Unlike ``agentic_coding_live.py``, this quality lane does not produce latency
or goodput rollups. Model failures are retained as scored quality observations
and can never set ``performance_claim=true``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from scripts.agentic_coding_live import LiveHTTPTransport  # noqa: E402

_QUALITY_SYSTEM_POLICY = (
    "You are measuring automatic tool selection across repository, general, and "
    "multilingual tasks. Choose the appropriate tool from the declared set and call "
    "it exactly once. Return only the tool call. Do not expose reasoning or raw tool "
    "markers.\n\n"
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _quality_build_profile(backend: str) -> str:
    normalized = str(backend).strip()
    if not normalized:
        raise AgenticBenchmarkError("quality backend must be a non-empty string")
    return f"{normalized}_agentic_auto_tool_quality"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_live_capabilities(
    capabilities: Mapping[str, Any],
    *,
    model: str,
    backend: str,
    cache_mode: str,
) -> None:
    model_capability = capabilities.get("model")
    if not isinstance(model_capability, Mapping):
        raise AgenticBenchmarkError("server capabilities omit model identity")
    served_model = model_capability.get("id")
    if served_model != model:
        raise AgenticBenchmarkError(
            f"served model is {served_model!r}, expected {model!r}"
        )
    served_backend = model_capability.get("backend")
    if served_backend not in {backend, "auto", None}:
        raise AgenticBenchmarkError(
            f"served backend is {served_backend!r}, expected {backend!r}"
        )
    cache = capabilities.get("cache")
    if isinstance(cache, Mapping) and cache.get("prefix_cache") not in {None, cache_mode}:
        raise AgenticBenchmarkError(
            f"server prefix cache is {cache.get('prefix_cache')!r}, expected {cache_mode!r}"
        )
    tokenizer = capabilities.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or not all(
        tokenizer.get(name) is True for name in ("tokenize", "detokenize")
    ):
        raise AgenticBenchmarkError(
            "server capabilities must advertise tokenize and detokenize"
        )
    features = capabilities.get("features")
    tools = features.get("tools") if isinstance(features, Mapping) else None
    if not isinstance(tools, Mapping) or tools.get("enabled") is not True:
        raise AgenticBenchmarkError("server capabilities do not enable tools")
    if tools.get("strict_result_validation") is not True:
        raise AgenticBenchmarkError(
            "server capabilities do not advertise strict tool result validation"
        )


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
    workload_id: str | None = None,
    workload_ids: Sequence[str] | None = None,
    model: str,
    backend: str,
    concurrency: int,
    runs: int,
    max_tokens: int,
    cache_mode: str,
    idle_timeout_s: float,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[AgenticWorkloadSuite, dict[str, Any]]:
    """Collect independent natural auto-tool attempts over canonical valid histories."""

    if concurrency <= 0 or runs <= 0 or max_tokens <= 0:
        raise AgenticBenchmarkError("concurrency, runs, and max_tokens must be positive")
    if cache_mode != "off":
        raise AgenticBenchmarkError(
            "agentic quality collector supports cache_mode=off only until prefix "
            "quality gates exist"
        )
    suite = load_agentic_workload_suite(workloads_path)
    if workload_id is not None and workload_ids is not None:
        raise AgenticBenchmarkError("pass workload_id or workload_ids, not both")
    if workload_ids is not None:
        selected_workloads = tuple(str(value) for value in workload_ids)
        if not selected_workloads:
            selected_workloads = tuple(suite.workloads)
    elif workload_id is not None:
        selected_workloads = (str(workload_id),)
    else:
        selected_workloads = ("small_repo",)
    if len(set(selected_workloads)) != len(selected_workloads):
        raise AgenticBenchmarkError("quality workload selection contains duplicates")
    unknown = [value for value in selected_workloads if value not in suite.workloads]
    if unknown:
        raise AgenticBenchmarkError(f"unknown workload ids: {unknown}")
    capabilities = transport.capabilities()
    if not isinstance(capabilities, Mapping):
        raise AgenticBenchmarkError("server capabilities must be an object")
    _validate_live_capabilities(
        capabilities,
        model=model,
        backend=backend,
        cache_mode=cache_mode,
    )
    capabilities_payload = json.loads(
        json.dumps(capabilities, sort_keys=True, ensure_ascii=False)
    )
    tools = build_openai_tools(suite)
    records: list[dict[str, Any]] = []
    raw_turns: list[dict[str, Any]] = []
    total_turns = runs * concurrency * sum(
        len(suite.workloads[workload_id]["turns"])
        for workload_id in selected_workloads
    )

    def checkpoint(*, status: str, final_ownership: Mapping[str, Any] | None = None) -> None:
        if checkpoint_callback is None:
            return
        payload: dict[str, Any] = {
            "kind": "hipengine_agentic_coding_quality_checkpoint",
            "schema_version": 1,
            "status": status,
            "performance_claim": False,
            "configuration": {
                "model": str(model),
                "backend": str(backend),
                "cache_mode": cache_mode,
                "concurrency": int(concurrency),
                "repetitions": int(runs),
                "max_tokens": int(max_tokens),
                "workloads": list(selected_workloads),
                "server_capabilities_sha256": _canonical_sha256(capabilities_payload),
            },
            "progress": {
                "completed_turns": len(records),
                "total_turns": total_turns,
            },
            "server_capabilities": capabilities_payload,
            "turn_records": records,
            "raw_turns": raw_turns,
        }
        if final_ownership is not None:
            payload["final_ownership"] = dict(final_ownership)
        checkpoint_callback(payload)

    for run_index in range(runs):
        for selected_workload in selected_workloads:
            run_id = f"run-{run_index}-{selected_workload}"
            prefix = render_workload_prefix(
                suite,
                selected_workload,
                tokenize=transport.tokenize,
                detokenize=transport.detokenize,
            )
            turns = suite.workloads[selected_workload]["turns"]
            for turn_index, _turn in enumerate(turns):
                prepared: list[tuple[str, str, list[int], dict[str, Any]]] = []
                for agent_index in range(concurrency):
                    agent_id = f"agent-{agent_index}"
                    session_id = f"session-{selected_workload}-{agent_index}"
                    messages = build_canonical_turn_messages(
                        suite,
                        selected_workload,
                        turn_index=turn_index,
                        agent_id=agent_id,
                        prefix_text=prefix.text,
                        system_policy=_QUALITY_SYSTEM_POLICY,
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

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = [
                        executor.submit(transport.chat_json, payload)
                        for _agent_id, _session_id, _prompt_ids, payload in prepared
                    ]
                    responses = [future.result() for future in futures]
                for prepared_row, response in zip(prepared, responses, strict=True):
                    agent_id, session_id, prompt_ids, _payload = prepared_row
                    request_id = f"{run_id}-{agent_id}-turn-{turn_index}"
                    record = normalize_chat_quality_turn(
                        suite,
                        workload_id=selected_workload,
                        turn_index=turn_index,
                        run_id=run_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        request_id=request_id,
                        prompt_token_ids=prompt_ids,
                        payload=response,
                    )
                    records.append(record)
                    raw_turns.append(
                        {
                            "workload_id": selected_workload,
                            "run_id": run_id,
                            "agent_id": agent_id,
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "request_id": request_id,
                            "prompt_token_ids": list(prompt_ids),
                            "response": response,
                        }
                    )
                    checkpoint(status="in_progress")
                    print(
                        f"quality progress {len(records)}/{total_turns}: "
                        f"{selected_workload} run={run_index} turn={turn_index} "
                        f"outcome={record['quality']['outcome']}",
                        flush=True,
                    )

    ownership = _wait_for_final_ownership(
        transport,
        cache_mode=cache_mode,
        timeout_s=idle_timeout_s,
    )
    checkpoint(status="complete", final_ownership=ownership)
    return suite, {
        "kind": AGENTIC_QUALITY_RECORDS_KIND,
        "schema_version": 1,
        "configuration": {
            "id": (
                f"live-quality-{selected_workloads[0]}-c{concurrency}"
                if len(selected_workloads) == 1
                else f"live-quality-broad-{len(selected_workloads)}-c{concurrency}"
            ),
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
            "workloads": list(selected_workloads),
            "quality_system_policy": "automatic_selection_without_expected_tool_name_hint",
            "repetitions": int(runs),
            "max_tokens": int(max_tokens),
            "server_capabilities_sha256": _canonical_sha256(capabilities_payload),
            "server_capabilities": capabilities_payload,
        },
        "turn_records": records,
        "final_ownership": ownership,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect the non-performance automatic-tool quality lane."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--workloads", type=Path, default=DEFAULT_AGENTIC_WORKLOADS)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--all-workloads", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--target-arch")
    parser.add_argument("--device-name")
    parser.add_argument("--quant")
    parser.add_argument("--kv-dtype")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-clean-provenance", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--cache-mode", choices=("off",), default="off")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--idle-timeout-s", type=float, default=30.0)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-json",
        type=Path,
        help="Atomic per-turn raw-response checkpoint (default: RECORDS.checkpoint.json)",
    )
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
        if args.all_workloads and args.workload:
            raise AgenticBenchmarkError("--all-workloads cannot be combined with --workload")
        selected_workloads: tuple[str, ...]
        if args.all_workloads:
            selected_workloads = ()
        else:
            selected_workloads = tuple(args.workload or ("small_repo",))
        checkpoint_json = args.checkpoint_json or args.records_json.with_name(
            f"{args.records_json.stem}.checkpoint.json"
        )
        suite, records = collect_live_quality_records(
            transport,
            workloads_path=args.workloads,
            workload_ids=selected_workloads,
            model=args.model,
            backend=args.backend,
            concurrency=args.concurrency,
            runs=args.runs,
            max_tokens=args.max_tokens,
            cache_mode=args.cache_mode,
            idle_timeout_s=args.idle_timeout_s,
            checkpoint_callback=lambda payload: _atomic_write_json(
                checkpoint_json,
                payload,
            ),
        )
        provenance = None
        if args.model_path is not None:
            hipcc_version = None
            if args.compiler_version_file is not None:
                hipcc_version = args.compiler_version_file.read_text(encoding="utf-8")
            provenance = collect_artifact_provenance(
                repo_root=REPO_ROOT,
                configured_backend=args.backend,
                resolved_backend=args.backend,
                detected_arches=(() if args.target_arch is None else (args.target_arch,)),
                target_arch=args.target_arch,
                device_name=args.device_name,
                model_path=args.model_path,
                quant=args.quant,
                kv_dtype=args.kv_dtype,
                command=(sys.executable, *sys.argv),
                environment={
                    key: os.environ.get(key)
                    for key in (
                        "HIPENGINE_BACKEND",
                        "HIPENGINE_HIP_ARCH",
                        "HIPENGINE_COMPILER_VERSION_FILE",
                        "HIPENGINE_PREFIX_CACHE",
                        "HIPENGINE_QWEN35_NATIVE_SAMPLER",
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                        "GPU_MAX_HW_QUEUES",
                    )
                },
                build_profile=_quality_build_profile(args.backend),
                timing_protocol=(
                    "real localhost blocking OpenAI chat; response-owned IDs; "
                    "external result/patch/test oracles; no latency or throughput fields"
                ),
                warmups=0,
                repetitions=args.runs,
                profiler={"used": False, "reason": "non-performance quality lane"},
                hipcc_version=hipcc_version,
            )
            if args.require_clean_provenance and provenance["dirty"]:
                raise AgenticBenchmarkError("quality provenance must be clean")
        elif args.require_clean_provenance:
            raise AgenticBenchmarkError(
                "--require-clean-provenance requires --model-path"
            )
        artifact = build_agentic_quality_artifact(
            suite,
            records,
            provenance=provenance,
        )
        _atomic_write_json(args.records_json, records)
        _atomic_write_json(args.json, artifact)
    except (AgenticBenchmarkError, json.JSONDecodeError, OSError) as exc:
        print(f"live agentic quality benchmark rejected: {exc}", file=sys.stderr)
        return 2
    quality = artifact["quality"]
    print(
        f"Agentic quality collected: {quality['successes']}/{quality['attempts']} successful "
        f"tool turns ({100.0 * quality['success_rate']:.1f}%) -> {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
