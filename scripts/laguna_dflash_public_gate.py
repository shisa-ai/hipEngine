#!/usr/bin/env python3
"""Validate explicit Laguna DFlash through public blocking/streaming routes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from fastapi.testclient import TestClient

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.chat.poolside_v1 import render_poolside_v1_chat
from hipengine.core.memory import memory_stats
from hipengine.generation.laguna_dflash import (
    LAGUNA_DFLASH_CANDIDATE_BUDGET,
    LAGUNA_DFLASH_DRAFTER_REVISION,
    LAGUNA_DFLASH_DRAFTER_SHA256,
    LAGUNA_DFLASH_ECONOMICS_EVIDENCE,
    LAGUNA_DFLASH_FALLBACK_REASON,
    LAGUNA_DFLASH_TARGET_SHA256,
)
from hipengine.server import ServerConfig, create_app
from scripts.laguna_dflash_category_bench import DEFAULT_DRAFTER, DEFAULT_HELDOUT_IDS
from scripts.laguna_target_ar_bench import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    EXPECTED_CATEGORIES,
    EXPECTED_PROMPT_COUNT,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AR_ORACLE = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-"
    "laguna-dflash-category-economics-post-prefill.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-"
    "laguna-dflash-public-e2e.json"
)
SERVED_MODEL = "laguna-dflash-public-e2e"
DEFAULT_MAX_TOKENS = 32


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("drafter", nargs="?", type=Path, default=DEFAULT_DRAFTER)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--ar-oracle", type=Path, default=DEFAULT_AR_ORACLE)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_ar_oracle(
    path: Path,
    *,
    expected_prompt_count: int = EXPECTED_PROMPT_COUNT,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("prompt_runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Laguna AR oracle has no prompt_runs")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Laguna AR oracle prompt row must be an object")
        prompt = row.get("prompt")
        ar = row.get("ar")
        if not isinstance(prompt, Mapping) or not isinstance(ar, Mapping):
            raise ValueError("Laguna AR oracle row lacks prompt/ar metadata")
        prompt_id = str(prompt.get("id", "")).strip()
        if not prompt_id:
            raise ValueError("Laguna AR oracle row has no prompt id")
        grouped.setdefault(prompt_id, []).append(row)
    if len(grouped) != int(expected_prompt_count):
        raise ValueError(
            f"Laguna AR oracle requires {expected_prompt_count} prompts, got {len(grouped)}"
        )
    result: dict[str, dict[str, Any]] = {}
    for prompt_id, prompt_rows in grouped.items():
        id_rows = {
            tuple(int(token) for token in row["ar"]["generated_ids"])
            for row in prompt_rows
        }
        if len(id_rows) != 1:
            raise ValueError(f"Laguna AR oracle is nondeterministic for {prompt_id}")
        first = prompt_rows[0]
        prompt = first["prompt"]
        result[prompt_id] = {
            "category": str(prompt["category"]),
            "prompt_tokens": int(prompt["prompt_tokens"]),
            "prompt_ids_sha256": str(prompt["prompt_ids_sha256"]),
            "fixed_horizon_ids": next(iter(id_rows)),
            "fixed_horizon_ids_sha256": _sha256_json(next(iter(id_rows))),
            "repetitions": len(prompt_rows),
        }
    return result


def _public_ids_until_stop(
    token_ids: Sequence[int],
    stop_token_ids: Sequence[int],
) -> tuple[int, ...]:
    stops = {int(token) for token in stop_token_ids}
    visible: list[int] = []
    for raw_token in token_ids:
        token = int(raw_token)
        visible.append(token)
        if token in stops:
            break
    return tuple(visible)


def _load_prompts(
    path: Path,
    llm: LLM,
    oracle: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    categories: set[str] = set()
    ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw_line.strip():
            continue
        source = json.loads(raw_line)
        prompt_id = str(source.get("id", "")).strip()
        category = str(source.get("category", "")).strip()
        messages = source.get("messages")
        if not prompt_id or prompt_id in ids:
            raise ValueError(f"prompt line {line_number} has a blank/duplicate id")
        if prompt_id not in oracle:
            raise ValueError(f"prompt {prompt_id!r} is missing from the AR oracle")
        if not category or not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {prompt_id!r} lacks category/messages")
        rendered = render_poolside_v1_chat(
            messages,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        token_ids = tuple(int(token) for token in llm.tokenize(rendered))
        prompt_hash = _sha256_json(token_ids)
        expected = oracle[prompt_id]
        if prompt_hash != str(expected["prompt_ids_sha256"]):
            raise ValueError(f"prompt {prompt_id!r} token IDs differ from the AR oracle")
        ids.add(prompt_id)
        categories.add(category)
        prompts.append(
            {
                "id": prompt_id,
                "category": category,
                "split": "heldout" if prompt_id in DEFAULT_HELDOUT_IDS else "train",
                "prompt_tokens": len(token_ids),
                "token_ids": token_ids,
                "token_ids_sha256": prompt_hash,
            }
        )
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        raise ValueError(
            f"Laguna public gate requires {EXPECTED_PROMPT_COUNT} prompts, got {len(prompts)}"
        )
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"Laguna public categories differ: {sorted(categories)}"
        )
    if not DEFAULT_HELDOUT_IDS < ids:
        raise ValueError("Laguna public heldout IDs are incomplete")
    return prompts


def _blocking_result(response) -> dict[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(
            f"blocking public request failed HTTP {response.status_code}: {response.text}"
        )
    body = response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("blocking public response must contain exactly one choice")
    choice = choices[0]
    hipengine = choice.get("hipengine")
    if not isinstance(hipengine, Mapping):
        raise RuntimeError("blocking public response lacks hipengine metadata")
    generated_ids = hipengine.get("generated_token_ids")
    if not isinstance(generated_ids, list):
        raise RuntimeError("blocking public response lacks exact generated token IDs")
    return {
        "text": str(choice.get("text", "")),
        "generated_ids": tuple(int(token) for token in generated_ids),
        "finish_reason": choice.get("finish_reason"),
        "finish_details": choice.get("finish_details"),
        "execution_path": (
            hipengine.get("decode_state", {}).get("execution_path")
            if isinstance(hipengine.get("decode_state"), Mapping)
            else None
        ),
        "diagnostics": (
            dict(hipengine["diagnostics"])
            if isinstance(hipengine.get("diagnostics"), Mapping)
            else {}
        ),
        "generation_shape": body.get("hipengine", {}).get("generation_shape"),
    }


def _stream_result(response) -> dict[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(
            f"streaming public request failed HTTP {response.status_code}: {response.text}"
        )
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    text_parts: list[str] = []
    terminal: Mapping[str, Any] | None = None
    for payload in payloads:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        text_parts.append(str(choice.get("text", "")))
        if choice.get("finish_reason") is not None:
            terminal = choice
    if terminal is None:
        raise RuntimeError("streaming public response lacks a terminal choice")
    hipengine = terminal.get("hipengine")
    if not isinstance(hipengine, Mapping):
        raise RuntimeError("streaming terminal choice lacks hipengine metadata")
    generated_ids = hipengine.get("generated_token_ids")
    if not isinstance(generated_ids, list):
        raise RuntimeError("streaming terminal choice lacks cumulative generated IDs")
    return {
        "text": "".join(text_parts),
        "generated_ids": tuple(int(token) for token in generated_ids),
        "finish_reason": terminal.get("finish_reason"),
        "finish_details": terminal.get("finish_details"),
        "execution_path": (
            hipengine.get("decode_state", {}).get("execution_path")
            if isinstance(hipengine.get("decode_state"), Mapping)
            else None
        ),
        "diagnostics": (
            dict(hipengine["diagnostics"])
            if isinstance(hipengine.get("diagnostics"), Mapping)
            else {}
        ),
        "done": "data: [DONE]" in response.text,
        "event_count": len(payloads),
    }


def _provider_state(llm: LLM) -> dict[str, Any]:
    wrapper = llm._text_generator
    inner = None if wrapper is None else getattr(wrapper, "inner", None)
    provider = None if inner is None else getattr(inner, "_speculative_provider", None)
    target = None if provider is None else getattr(provider, "_target_session", None)
    drafter = None if provider is None else getattr(provider, "_drafter", None)
    cycle = None if provider is None else getattr(provider, "_cycle", None)
    return {
        "provider_present": provider is not None,
        "provider_closed": bool(getattr(provider, "_closed", False)) if provider else None,
        "target_present": target is not None,
        "target_position": None if target is None else int(target.position),
        "target_closed": bool(getattr(target, "closed", False)) if target else None,
        "drafter_present": drafter is not None,
        "drafter_context_tokens": (
            None if drafter is None else int(drafter.committed_context_tokens)
        ),
        "drafter_closed": bool(getattr(drafter, "_closed", False)) if drafter else None,
        "cycle_present": cycle is not None,
        "owner_ids": {
            "target": None if target is None else id(target),
            "drafter": None if drafter is None else id(drafter),
            "cycle": None if cycle is None else id(cycle),
        },
    }


def _state_is_reset(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("provider_present")
        and state.get("target_present")
        and state.get("drafter_present")
        and state.get("cycle_present")
        and state.get("target_position") == -1
        and state.get("drafter_context_tokens") == 0
        and state.get("target_closed") is False
        and state.get("drafter_closed") is False
        and state.get("provider_closed") is False
    )


def _request_body(
    prompt_ids: Sequence[int],
    *,
    max_tokens: int,
    speculative: bool,
    stream: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": SERVED_MODEL,
        "prompt": [int(token) for token in prompt_ids],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "stream": bool(stream),
    }
    if speculative:
        body["speculative"] = {
            "enabled": True,
            "provider": "dflash",
            "candidate_budget": LAGUNA_DFLASH_CANDIDATE_BUDGET,
        }
    if stream:
        body["stream_options"] = {
            "include_hipengine": True,
            "include_usage": True,
        }
    return body


def _capability_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    speculative = payload.get("sampling", {}).get("speculative", {})
    if not isinstance(speculative, Mapping):
        speculative = {}
    target = speculative.get("target")
    drafter = speculative.get("drafter")
    return {
        "serving_route": speculative.get("serving_route") is True,
        "configured": speculative.get("configured") is True,
        "provider": speculative.get("provider") == "dflash",
        "configured_provider": speculative.get("configured_provider") == "dflash",
        "request_field": speculative.get("request_field") == "speculative",
        "policy": speculative.get("policy") == "explicit_only",
        "default_disabled": speculative.get("default_enabled") is False,
        "streaming": speculative.get("streaming_compatible") is True,
        "candidate_budget": speculative.get("candidate_budget")
        == LAGUNA_DFLASH_CANDIDATE_BUDGET,
        "exactness": speculative.get("exactness_mode")
        == "target_corrected_greedy",
        "processed_target_verification": speculative.get(
            "processed_target_verification"
        )
        is False,
        "target_sha256": isinstance(target, Mapping)
        and target.get("sha256") == LAGUNA_DFLASH_TARGET_SHA256,
        "drafter_sha256": isinstance(drafter, Mapping)
        and drafter.get("sha256") == LAGUNA_DFLASH_DRAFTER_SHA256,
        "drafter_revision": isinstance(drafter, Mapping)
        and drafter.get("revision") == LAGUNA_DFLASH_DRAFTER_REVISION,
        "fallback_reason": speculative.get("fallback_reason")
        == LAGUNA_DFLASH_FALLBACK_REASON,
        "economics_evidence": speculative.get("economics_evidence")
        == LAGUNA_DFLASH_ECONOMICS_EVIDENCE,
        "no_performance_claim": speculative.get("performance_claim") is False,
    }


def _prompt_checks(
    *,
    expected_ids: Sequence[int],
    ar: Mapping[str, Any],
    blocking: Mapping[str, Any],
    streaming: Mapping[str, Any],
    state_after_blocking: Mapping[str, Any],
    state_after_streaming: Mapping[str, Any],
    retained_owner_ids: Mapping[str, Any] | None,
) -> dict[str, bool]:
    expected = tuple(int(token) for token in expected_ids)
    ar_ids = tuple(ar["generated_ids"])
    blocking_ids = tuple(blocking["generated_ids"])
    streaming_ids = tuple(streaming["generated_ids"])
    owner_ids = state_after_streaming.get("owner_ids")
    return {
        "ar_matches_fixed_oracle_stop_policy": ar_ids == expected,
        "blocking_matches_true_ar": blocking_ids == ar_ids,
        "streaming_matches_true_ar": streaming_ids == ar_ids,
        "blocking_streaming_ids_equal": blocking_ids == streaming_ids,
        "blocking_streaming_text_equal": blocking.get("text") == streaming.get("text"),
        "eot_markup_absent": "</assistant>" not in str(blocking.get("text", ""))
        and "</assistant>" not in str(streaming.get("text", "")),
        "blocking_route": isinstance(blocking.get("generation_shape"), Mapping)
        and blocking["generation_shape"].get("route") == "speculative",
        "blocking_execution_path": blocking.get("execution_path")
        == "laguna_dflash_b4_c1",
        "streaming_execution_path": streaming.get("execution_path")
        == "laguna_dflash_b4_c1",
        "stream_done": streaming.get("done") is True,
        "blocking_state_reset": _state_is_reset(state_after_blocking),
        "streaming_state_reset": _state_is_reset(state_after_streaming),
        "owners_retained": retained_owner_ids is None or owner_ids == retained_owner_ids,
    }


def run_gate(
    model: Path,
    drafter: Path,
    *,
    prompts_path: Path,
    ar_oracle_path: Path,
    backend: str,
    quant: str,
    max_tokens: int,
) -> dict[str, Any]:
    if int(max_tokens) <= 0:
        raise ValueError("max_tokens must be positive")
    oracle = _load_ar_oracle(ar_oracle_path)
    tracked_before = memory_stats()
    started = time.perf_counter()
    llm = LLM(
        str(model),
        backend=backend,
        quant=quant,
        max_active_requests=1,
        speculative_provider="dflash",
        draft_model=str(drafter),
        speculative_candidate_budget=LAGUNA_DFLASH_CANDIDATE_BUDGET,
    )
    config = ServerConfig(
        model=str(model),
        backend=backend,
        quant=quant,
        served_model_name=SERVED_MODEL,
        eager_load=False,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
        max_context_tokens=4_096,
        kv_storage="bf16",
        max_active_requests=1,
        speculative_provider="dflash",
        draft_model=str(drafter),
        speculative_candidate_budget=LAGUNA_DFLASH_CANDIDATE_BUDGET,
    )
    app = create_app(config, llm=llm)
    rows: list[dict[str, Any]] = []
    capabilities: dict[str, Any] = {}
    capability_checks: dict[str, bool] = {}
    model_capability = False
    retained_owner_ids: Mapping[str, Any] | None = None
    cancellation: dict[str, Any] = {}
    provider_ref: Any | None = None
    try:
        with TestClient(app) as client:
            capability_response = client.get("/v1/hipengine/capabilities")
            if capability_response.status_code != 200:
                raise RuntimeError("Laguna public capability request failed")
            capabilities = capability_response.json()
            capability_checks = _capability_checks(capabilities)
            models_response = client.get("/v1/models")
            model_capability = bool(
                models_response.json()["data"][0]["hipengine"]["capabilities"].get(
                    "speculative"
                )
            )
            prompts = _load_prompts(prompts_path, llm, oracle)
            wrapper = llm._text_generator
            inner = None if wrapper is None else getattr(wrapper, "inner", None)
            tokenizer = None if inner is None else getattr(inner, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Laguna public generator tokenizer is unavailable")
            stop_ids = tuple(int(token) for token in tokenizer.stop_token_ids)
            for index, prompt in enumerate(prompts, 1):
                prompt_started = time.perf_counter()
                expected_ids = _public_ids_until_stop(
                    oracle[prompt["id"]]["fixed_horizon_ids"][:max_tokens],
                    stop_ids,
                )
                ar_started = time.perf_counter()
                ar = _blocking_result(
                    client.post(
                        "/v1/completions",
                        json=_request_body(
                            prompt["token_ids"],
                            max_tokens=max_tokens,
                            speculative=False,
                        ),
                    )
                )
                ar_seconds = time.perf_counter() - ar_started
                blocking_started = time.perf_counter()
                blocking = _blocking_result(
                    client.post(
                        "/v1/completions",
                        json=_request_body(
                            prompt["token_ids"],
                            max_tokens=max_tokens,
                            speculative=True,
                        ),
                    )
                )
                blocking_seconds = time.perf_counter() - blocking_started
                state_after_blocking = _provider_state(llm)
                if retained_owner_ids is None:
                    retained_owner_ids = state_after_blocking["owner_ids"]
                streaming_started = time.perf_counter()
                streaming = _stream_result(
                    client.post(
                        "/v1/completions",
                        json=_request_body(
                            prompt["token_ids"],
                            max_tokens=max_tokens,
                            speculative=True,
                            stream=True,
                        ),
                    )
                )
                streaming_seconds = time.perf_counter() - streaming_started
                state_after_streaming = _provider_state(llm)
                checks = _prompt_checks(
                    expected_ids=expected_ids,
                    ar=ar,
                    blocking=blocking,
                    streaming=streaming,
                    state_after_blocking=state_after_blocking,
                    state_after_streaming=state_after_streaming,
                    retained_owner_ids=retained_owner_ids,
                )
                rows.append(
                    {
                        "id": prompt["id"],
                        "category": prompt["category"],
                        "split": prompt["split"],
                        "prompt_tokens": prompt["prompt_tokens"],
                        "prompt_ids_sha256": prompt["token_ids_sha256"],
                        "expected_ids_sha256": _sha256_json(expected_ids),
                        "ar_ids_sha256": _sha256_json(ar["generated_ids"]),
                        "blocking_ids_sha256": _sha256_json(
                            blocking["generated_ids"]
                        ),
                        "streaming_ids_sha256": _sha256_json(
                            streaming["generated_ids"]
                        ),
                        "output_tokens": len(expected_ids),
                        "finish": {
                            "ar": ar["finish_details"],
                            "blocking": blocking["finish_details"],
                            "streaming": streaming["finish_details"],
                        },
                        "timing_seconds": {
                            "ar": ar_seconds,
                            "blocking": blocking_seconds,
                            "streaming": streaming_seconds,
                            "total": time.perf_counter() - prompt_started,
                        },
                        "provider_diagnostics": blocking["diagnostics"],
                        "checks": checks,
                        "pass": all(checks.values()),
                    }
                )
                print(
                    f"{index}/{len(prompts)} {prompt['id']} "
                    f"ar={ar_seconds:.3f}s block={blocking_seconds:.3f}s "
                    f"stream={streaming_seconds:.3f}s pass={all(checks.values())}",
                    file=sys.stderr,
                    flush=True,
                )
            cancel_iterator = llm.stream_speculative_detailed(
                prompts[0]["token_ids"],
                SamplingParams(max_tokens=max_tokens, kv_storage="bf16"),
            )
            cancel_started = time.perf_counter()
            first_chunk = next(cancel_iterator)
            cancel_iterator.close()
            cancellation_state = _provider_state(llm)
            cancellation = {
                "first_chunk_text_nonempty": bool(first_chunk.text),
                "first_chunk_generated_ids": list(
                    first_chunk.generated_token_ids or ()
                ),
                "state": cancellation_state,
                "state_reset": _state_is_reset(cancellation_state),
                "elapsed_seconds": time.perf_counter() - cancel_started,
            }
            wrapper = llm._text_generator
            inner = None if wrapper is None else getattr(wrapper, "inner", None)
            provider_ref = (
                None if inner is None else getattr(inner, "_speculative_provider", None)
            )
    finally:
        llm.close()
    tracked_after = memory_stats()
    provider_closed = bool(getattr(provider_ref, "_closed", False))
    provider_owners_released = bool(
        provider_ref is not None
        and getattr(provider_ref, "_target_session", None) is None
        and getattr(provider_ref, "_drafter", None) is None
        and getattr(provider_ref, "_cycle", None) is None
    )
    lifecycle_checks = {
        "tracked_bytes": tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"],
        "tracked_allocations": tracked_after["active_allocations"]
        == tracked_before["active_allocations"],
        "provider_closed": provider_closed,
        "provider_owners_released": provider_owners_released,
    }
    split_checks = {
        split: all(row["pass"] for row in rows if row["split"] == split)
        for split in ("train", "heldout")
    }
    category_checks = {
        category: all(row["pass"] for row in rows if row["category"] == category)
        for category in sorted(EXPECTED_CATEGORIES)
    }
    protocol_eligible = int(max_tokens) == DEFAULT_MAX_TOKENS
    passed = bool(
        protocol_eligible
        and len(rows) == EXPECTED_PROMPT_COUNT
        and all(row["pass"] for row in rows)
        and all(capability_checks.values())
        and model_capability
        and cancellation.get("state_reset") is True
        and all(lifecycle_checks.values())
        and all(split_checks.values())
        and all(category_checks.values())
    )
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=backend,
        resolved_backend=llm.resolved_backend,
        target_arch="gfx1151",
        model_path=model,
        quant=quant,
        kv_dtype="bf16",
        command=[sys.executable, *sys.argv],
        environment={
            key: os.environ.get(key)
            for key in (
                "HIPENGINE_HIP_ARCH",
                "GPU_MAX_HW_QUEUES",
                "HIPENGINE_LAGUNA_PREFILL_CHUNK_SIZE",
            )
        },
        timing_protocol="public OpenAI c=1 AR/blocking-DFlash/streaming-DFlash wall",
        warmups=0,
        repetitions=1,
    )
    return {
        "schema": 1,
        "status": "accepted" if passed else "failed",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "scope": "Laguna DFlash explicit public blocking/streaming correctness",
        "performance_claim": False,
        "support_decision": "opt_in_supported" if passed else "blocked",
        "default_route": "ar",
        "model": {
            "path": str(model.resolve()),
            "sha256": LAGUNA_DFLASH_TARGET_SHA256,
            "backend": backend,
            "quant": quant,
        },
        "drafter": {
            "path": str(drafter.resolve()),
            "revision": LAGUNA_DFLASH_DRAFTER_REVISION,
            "sha256": LAGUNA_DFLASH_DRAFTER_SHA256,
            "candidate_budget": LAGUNA_DFLASH_CANDIDATE_BUDGET,
        },
        "workload": {
            "prompt_suite": str(prompts_path.resolve()),
            "prompt_suite_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
            "ar_oracle": str(ar_oracle_path.resolve()),
            "ar_oracle_sha256": hashlib.sha256(ar_oracle_path.read_bytes()).hexdigest(),
            "prompt_count": len(rows),
            "train_prompt_ids": sorted(
                row["id"] for row in rows if row["split"] == "train"
            ),
            "heldout_prompt_ids": sorted(
                row["id"] for row in rows if row["split"] == "heldout"
            ),
            "categories": sorted(EXPECTED_CATEGORIES),
            "max_tokens": int(max_tokens),
            "protocol_eligible": protocol_eligible,
            "routes_per_prompt": ["true_ar", "dflash_blocking", "dflash_streaming"],
        },
        "capability_checks": capability_checks,
        "model_capability": model_capability,
        "prompt_rows": rows,
        "split_checks": split_checks,
        "category_checks": category_checks,
        "cancellation": cancellation,
        "lifecycle": {
            "before": tracked_before,
            "after": tracked_after,
            "checks": lifecycle_checks,
        },
        "economics": {
            "default_eligible": False,
            "performance_claim": False,
            "fallback_reason": LAGUNA_DFLASH_FALLBACK_REASON,
            "evidence": LAGUNA_DFLASH_ECONOMICS_EVIDENCE,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "pass": passed,
    }


def main() -> int:
    args = _parser().parse_args()
    result = run_gate(
        args.model,
        args.drafter,
        prompts_path=args.prompts,
        ar_oracle_path=args.ar_oracle,
        backend=str(args.backend),
        quant=str(args.quant),
        max_tokens=int(args.max_tokens),
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
