#!/usr/bin/env python3
"""Build and run the canonical Qwen4Exp exact-token AR comparison.

The fixture contains one code, English, Japanese, and mixed Japanese/English
prompt at each of 512, 1024, and 4096 input tokens. Benchmark modes consume the
same token-ID arrays in hipEngine or a llama.cpp-compatible server. The first
sampled output belongs to prefill; decode timing covers exactly 128 subsequent
autoregressive transitions by requesting 129 visible outputs from llama.cpp.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PROMPTS = ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
DEFAULT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"
CANONICAL_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
CANONICAL_SHAPES = (512, 1024, 4096)
DEFAULT_TRANSITIONS = 128


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _read_prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or "id" not in payload or "category" not in payload:
            raise ValueError(f"{path}:{line_number}: expected id and category")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_number}: expected non-empty messages")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{path} did not contain prompts")
    return rows


def _category_material(rows: Sequence[Mapping[str, Any]], category: str) -> tuple[str, list[str]]:
    selected = [row for row in rows if str(row.get("category")) == category]
    if not selected:
        raise ValueError(f"missing canonical categories: {category}")
    sections: list[str] = []
    source_ids: list[str] = []
    for row in selected:
        source_id = str(row["id"])
        source_ids.append(source_id)
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"source prompt {source_id!r} has invalid messages")
        rendered: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError(f"source prompt {source_id!r} has an invalid message")
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        sections.append(f"[{source_id}]\n" + "\n".join(rendered))
    return "\n\n".join(sections) + "\n\n", source_ids


def _exact_case_tokens(
    tokenizer: Any,
    *,
    category_material: str,
    target_tokens: int,
) -> list[int]:
    prefix = tokenizer.encode("<|im_start|>user\n")
    suffix = tokenizer.encode("\n<|im_end|>\n<|im_start|>assistant\n")
    body = tokenizer.encode(category_material)
    if not body:
        raise ValueError("category material tokenized to an empty sequence")
    available = int(target_tokens) - len(prefix) - len(suffix)
    if available <= 0:
        raise ValueError(
            f"target {target_tokens} is too short for the canonical chat boundary "
            f"({len(prefix) + len(suffix)} tokens)"
        )
    repeats = (available + len(body) - 1) // len(body)
    token_ids = [*prefix, *(body * repeats)[:available], *suffix]
    if len(token_ids) != int(target_tokens):
        raise AssertionError("exact-token fixture construction produced the wrong length")
    return [int(token_id) for token_id in token_ids]


def build_fixture(
    *,
    tokenizer: Any,
    source_rows: Sequence[Mapping[str, Any]],
    shapes: Sequence[int],
    source_path: Path,
    source_sha256: str,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    present = {str(row.get("category")) for row in source_rows}
    missing = [category for category in CANONICAL_CATEGORIES if category not in present]
    if missing:
        raise ValueError("missing canonical categories: " + ", ".join(missing))
    normalized_shapes = sorted({int(shape) for shape in shapes})
    if not normalized_shapes or any(shape <= 0 for shape in normalized_shapes):
        raise ValueError("fixture shapes must be positive")

    cases: list[dict[str, Any]] = []
    for category in CANONICAL_CATEGORIES:
        material, source_ids = _category_material(source_rows, category)
        material_sha256 = hashlib.sha256(material.encode("utf-8")).hexdigest()
        for shape in normalized_shapes:
            token_ids = _exact_case_tokens(
                tokenizer,
                category_material=material,
                target_tokens=shape,
            )
            cases.append(
                {
                    "id": f"{category}-p{shape}",
                    "category": category,
                    "prompt_tokens": shape,
                    "prompt_token_ids": token_ids,
                    "prompt_token_ids_sha256": token_ids_sha256(token_ids),
                    "source_prompt_ids": source_ids,
                    "category_material_sha256": material_sha256,
                }
            )

    return {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_exact_token_fixture",
        "name": "Qwen3.8-Flash-Next canonical p512/p1024/p4096 AR comparison",
        "shapes": normalized_shapes,
        "categories": list(CANONICAL_CATEGORIES),
        "decode_transitions": DEFAULT_TRANSITIONS,
        "source": {
            "path": str(source_path),
            "sha256": str(source_sha256),
            "row_count": len(source_rows),
        },
        "model": dict(model_identity),
        "construction": (
            "Per-category source messages are rendered with role labels, repeated at the "
            "token-ID level between fixed Qwen chat boundaries, and truncated to the exact "
            "target length. Every engine consumes the committed token arrays directly."
        ),
        "cases": cases,
    }


def validate_fixture(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema", -1)) != 1:
        raise ValueError("unsupported canonical fixture schema")
    shapes = [int(shape) for shape in payload.get("shapes", ())]
    categories = [str(category) for category in payload.get("categories", ())]
    if categories != list(CANONICAL_CATEGORIES):
        raise ValueError("fixture categories do not match the canonical category order")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("fixture cases must be a list")
    expected = {(category, shape) for category in categories for shape in shapes}
    observed: set[tuple[str, int]] = set()
    ids: set[str] = set()
    for row in cases:
        if not isinstance(row, Mapping):
            raise ValueError("fixture case must be an object")
        case_id = str(row.get("id"))
        if case_id in ids:
            raise ValueError(f"duplicate fixture case id {case_id!r}")
        ids.add(case_id)
        category = str(row.get("category"))
        prompt_tokens = int(row.get("prompt_tokens", -1))
        token_ids = row.get("prompt_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != prompt_tokens:
            raise ValueError(f"fixture case {case_id!r} has an invalid token count")
        if token_ids_sha256(token_ids) != str(row.get("prompt_token_ids_sha256")):
            raise ValueError(f"fixture case {case_id!r} has an invalid token hash")
        observed.add((category, prompt_tokens))
    if observed != expected:
        raise ValueError("fixture case matrix does not match categories x shapes")


def load_fixture(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("canonical fixture must be a JSON object")
    validate_fixture(payload)
    return payload, sha256_path(path)


def hip_available() -> bool:
    """Whether the HIP runtime can be loaded in this process."""
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _forced_logits(generator, token_ids, steps, forced=None):
    """Teacher-forced prefill + ``steps`` decode; returns stacked logits.

    With ``forced`` (the incumbent arm's token chain) every arm consumes the
    SAME token sequence, so each row is a same-context measurement of pure
    route drift (the calibration definition) - a flip cannot cascade into
    incomparable contexts.
    """
    first = generator.runner.prefill(token_ids)
    rows = [np.ascontiguousarray(first.logits, dtype=np.float32)]
    token = int(first.token_id)
    chain = [token]
    for i in range(steps):
        if forced is not None:
            token = forced[i]
        nxt = generator.runner.step(token)
        generator.runner.runtime.device_synchronize()
        token = int(nxt.token_id)
        chain.append(token)
        rows.append(np.ascontiguousarray(nxt.logits, dtype=np.float32))
    return np.stack(rows), chain


def _kl(ref: np.ndarray, cand: np.ndarray) -> float:
    """KL divergence between two logit rows, computed in float64."""
    r = ref.astype(np.float64)
    c = cand.astype(np.float64)
    pr = np.exp(r - r.max())
    pr /= pr.sum()
    pc = np.exp(c - c.max())
    pc /= pc.sum()
    return float(np.sum(pc * (np.log(pc) - np.log(pr))))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _stats(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.mean(normalized)
    stdev = statistics.stdev(normalized) if len(normalized) > 1 else 0.0
    return {
        "samples": normalized,
        "median": float(statistics.median(normalized)),
        "mean": float(mean),
        "p95": float(_percentile(normalized, 0.95)),
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "stdev": float(stdev),
        "coefficient_of_variation": float(stdev / mean) if mean else 0.0,
    }


def _sample_group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    prefill_ms = sum(float(row["prefill_ms"]) for row in rows)
    transitions = sum(int(row["decode_transitions"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    summary = {
        "sample_count": len(rows),
        "case_count": len({str(row["case_id"]) for row in rows}),
        "prefill_tok_s_weighted": 1000.0 * prompt_tokens / prefill_ms,
        "decode_tok_s_weighted": 1000.0 * transitions / decode_ms,
        "client_wall_s_total": sum(float(row["client_wall_s"]) for row in rows),
        "prefill_tok_s": _stats([float(row["prefill_tok_s"]) for row in rows]),
        "decode_tok_s": _stats([float(row["decode_tok_s"]) for row in rows]),
    }
    # The host/device split only exists for hipEngine rows measured after the
    # in-bench split was added; llama.cpp rows and older artifacts omit it.
    if rows and all("decode_host_enqueue_ms" in row for row in rows):
        decode_host = sum(float(row["decode_host_enqueue_ms"]) for row in rows)
        decode_drain = sum(float(row["decode_device_drain_ms"]) for row in rows)
        prefill_host = sum(float(row["prefill_host_enqueue_ms"]) for row in rows)
        prefill_drain = sum(float(row["prefill_device_drain_ms"]) for row in rows)
        summary.update(
            {
                "decode_host_enqueue_ms_per_step": decode_host / transitions,
                "decode_device_drain_ms_per_step": decode_drain / transitions,
                "decode_host_enqueue_share": decode_host / decode_ms,
                "decode_device_drain_share": decode_drain / decode_ms,
                "prefill_host_enqueue_ms": prefill_host,
                "prefill_device_drain_ms": prefill_drain,
                "prefill_host_enqueue_share": prefill_host / prefill_ms,
            }
        )
    return summary


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("benchmark produced no measured samples")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_shape: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        by_case[str(row["case_id"])].append(row)
        by_shape[int(row["prompt_tokens"])].append(row)
        by_category[str(row["category"])].append(row)
    cases: dict[str, Any] = {}
    for case_id, rows in sorted(by_case.items()):
        digest_set = {str(row["output_token_ids_sha256"]) for row in rows}
        cases[case_id] = {
            **_sample_group_summary(rows),
            "category": str(rows[0]["category"]),
            "prompt_tokens": int(rows[0]["prompt_tokens"]),
            "deterministic": len(digest_set) == 1,
            "output_token_ids_sha256": sorted(digest_set),
        }
    return {
        "sample_count": len(samples),
        "all_cases_deterministic": all(row["deterministic"] for row in cases.values()),
        "cases": cases,
        "shapes": {
            str(shape): _sample_group_summary(rows)
            for shape, rows in sorted(by_shape.items())
        },
        "categories": {
            category: _sample_group_summary(rows)
            for category, rows in sorted(by_category.items())
        },
    }


def llamacpp_response_sample(
    *,
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    client_wall_s: float,
    repetition: int,
    expected_transitions: int,
) -> dict[str, Any]:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        raise ValueError("llama.cpp response omitted timings")
    prompt_tokens = int(case["prompt_tokens"])
    prompt_n = int(timings.get("prompt_n") or response.get("tokens_evaluated") or 0)
    if prompt_n != prompt_tokens:
        raise ValueError(
            f"llama.cpp evaluated {prompt_n} prompt tokens; expected {prompt_tokens}"
        )
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    predicted_n = int(timings.get("predicted_n") or 0)
    expected_outputs = int(expected_transitions) + 1
    token_ids = response.get("tokens")
    if not isinstance(token_ids, list) or len(token_ids) != expected_outputs:
        raise ValueError(f"llama.cpp response must contain {expected_outputs} output token IDs")
    if predicted_n != expected_outputs:
        raise ValueError(
            f"llama.cpp predicted_n={predicted_n}; expected {expected_outputs} visible outputs"
        )
    if prompt_ms <= 0.0 or predicted_ms <= 0.0:
        raise ValueError("llama.cpp response timings must be positive")
    normalized_ids = [int(token_id) for token_id in token_ids]
    return {
        "case_id": str(case["id"]),
        "category": str(case["category"]),
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids_sha256": str(case["prompt_token_ids_sha256"]),
        "repetition": int(repetition),
        "prefill_ms": prompt_ms,
        "prefill_tok_s": 1000.0 * prompt_tokens / prompt_ms,
        "decode_ms": predicted_ms,
        "decode_transitions": int(expected_transitions),
        "decode_tok_s": 1000.0 * int(expected_transitions) / predicted_ms,
        "client_wall_s": float(client_wall_s),
        "output_token_count": len(normalized_ids),
        "output_token_ids": normalized_ids,
        "output_token_ids_sha256": token_ids_sha256(normalized_ids),
        "native_predicted_n": predicted_n,
        "native_predicted_per_second": timings.get("predicted_per_second"),
        "stop_type": response.get("stop_type"),
        "truncated": response.get("truncated"),
    }


def _measurement_order(cases: Sequence[Mapping[str, Any]], repetition: int) -> list[Mapping[str, Any]]:
    ordered = sorted(cases, key=lambda row: (int(row["prompt_tokens"]), str(row["category"])))
    if repetition % 2:
        ordered.reverse()
    if ordered:
        shift = repetition % len(ordered)
        ordered = ordered[shift:] + ordered[:shift]
    return ordered


def _post_json(host: str, port: int, path: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("server response must be a JSON object")
    return parsed


def _wait_for_health(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2.0) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30.0)


def _git_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path.resolve()), "head": None, "tracked_clean": None}
    return {"path": str(path.resolve()), "head": head, "tracked_clean": not bool(status.strip())}


def _rocm_platform_version() -> str | None:
    try:
        return importlib.metadata.version("rocm")
    except importlib.metadata.PackageNotFoundError:
        try:
            result = subprocess.run(
                ["rocm-sdk", "version"], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        value = result.stdout.strip() or result.stderr.strip()
        return value or None


def _host_metadata() -> dict[str, Any]:
    tuned = subprocess.run(
        ["tuned-adm", "active"], capture_output=True, text=True, check=False
    )
    compiler_version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    if compiler_version_file:
        hipcc_version = Path(compiler_version_file).read_text().strip()
    else:
        hipcc = subprocess.run(
            ["hipcc", "--version"], capture_output=True, text=True, check=False
        )
        hipcc_version = hipcc.stdout.strip() or hipcc.stderr.strip()
    governors: dict[str, int] = defaultdict(int)
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            governors[path.read_text().strip()] += 1
        except OSError:
            continue
    gpu_clock_policies: dict[str, str] = {}
    for path in Path("/sys/class/drm").glob(
        "card*/device/power_dpm_force_performance_level"
    ):
        try:
            gpu_clock_policies[str(path)] = path.read_text().strip()
        except OSError:
            continue
    return {
        "hostname": socket.gethostname(),
        "machine_id": Path("/etc/machine-id").read_text().strip(),
        "kernel": platform.release(),
        "tuned_active": tuned.stdout.strip() or tuned.stderr.strip(),
        "cpu_governors": dict(governors),
        "gpu_clock_policies": gpu_clock_policies,
        "rocm_platform": _rocm_platform_version(),
        "hipcc_version": hipcc_version,
    }


def _completion_payload(case: Mapping[str, Any], transitions: int) -> dict[str, Any]:
    return {
        "prompt": [int(token_id) for token_id in case["prompt_token_ids"]],
        "n_predict": int(transitions) + 1,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 12345,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
        "return_tokens": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_llamacpp(args: argparse.Namespace) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(args.fixture)
    cases = fixture["cases"]
    transitions = int(fixture["decode_transitions"])
    server_bin = args.server_bin.resolve()
    model = args.model.resolve()
    log_path = args.server_log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server_bin),
        "-m", str(model),
        "--host", args.host,
        "--port", str(args.port),
        "--parallel", "1",
        "--no-webui",
        *args.server_arg,
    ]
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_engine_run",
        "engine": args.engine_label,
        "surface": "llamacpp_completion_server",
        "status": "running",
        "host": _host_metadata(),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": fixture_sha256,
        "model": str(model),
        "server_binary": str(server_bin),
        "server_binary_sha256": sha256_path(server_bin),
        "source": _git_metadata(args.source_root),
        "command": command,
        "protocol": {
            "warmups_per_case": int(args.warmups),
            "measured_repetitions": int(args.repetitions),
            "decode_transitions": transitions,
            "visible_output_tokens": transitions + 1,
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "cache_prompt": False,
            "timing_boundary": (
                "llama.cpp prompt_ms for exact prefill; request N+1 visible outputs and "
                "normalize predicted_ms to N post-first-output AR transitions"
            ),
        },
        "warmups": [],
        "samples": [],
    }
    _write_json(args.output, artifact)
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            _wait_for_health(args.host, args.port, args.startup_timeout)
            for warmup in range(args.warmups):
                for case in _measurement_order(cases, warmup):
                    started = time.perf_counter()
                    response = _post_json(
                        args.host,
                        args.port,
                        "/completion",
                        _completion_payload(case, transitions),
                        args.request_timeout,
                    )
                    row = llamacpp_response_sample(
                        case=case,
                        response=response,
                        client_wall_s=time.perf_counter() - started,
                        repetition=warmup,
                        expected_transitions=transitions,
                    )
                    artifact["warmups"].append(
                        {
                            "case_id": row["case_id"],
                            "repetition": warmup,
                            "output_token_ids_sha256": row["output_token_ids_sha256"],
                        }
                    )
                    print(
                        f"[warmup] {args.engine_label} {row['case_id']} "
                        f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                        flush=True,
                    )
            for repetition in range(args.repetitions):
                for case in _measurement_order(cases, repetition):
                    started = time.perf_counter()
                    response = _post_json(
                        args.host,
                        args.port,
                        "/completion",
                        _completion_payload(case, transitions),
                        args.request_timeout,
                    )
                    row = llamacpp_response_sample(
                        case=case,
                        response=response,
                        client_wall_s=time.perf_counter() - started,
                        repetition=repetition,
                        expected_transitions=transitions,
                    )
                    artifact["samples"].append(row)
                    artifact["summary"] = summarize_samples(artifact["samples"])
                    _write_json(args.output, artifact)
                    print(
                        f"[measure {repetition}] {args.engine_label} {row['case_id']} "
                        f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                        flush=True,
                    )
            artifact["status"] = "completed"
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _terminate(process)
            server_log.flush()
            artifact["server_returncode"] = process.returncode
            artifact["server_log"] = str(log_path)
            artifact["server_log_sha256"] = sha256_path(log_path)
            _write_json(args.output, artifact)
    artifact["server_returncode"] = process.returncode
    artifact["server_log"] = str(log_path)
    artifact["server_log_sha256"] = sha256_path(log_path)
    artifact["summary"] = summarize_samples(artifact["samples"])
    _write_json(args.output, artifact)
    return artifact


OPENAI_ADAPTERS = ("gufo", "kyojin")
OPENAI_COMPLETIONS_PATH = "/v1/completions"


def _openai_case_text(tokenizer: Any, case: Mapping[str, Any]) -> tuple[str, bool]:
    """Detokenize a canonical case and verify the tokenizer round-trip.

    Text-driven engines receive the raw detokenized prompt with no chat
    template. A case whose re-tokenization does not reproduce the fixture ids
    is disqualified for that engine and recorded; it is never silently
    re-aligned.
    """
    token_ids = [int(token_id) for token_id in case["prompt_token_ids"]]
    text = tokenizer.decode(token_ids)
    retokenized = tokenizer.encode(text, add_special_tokens=False)
    return text, retokenized == token_ids


def openai_completions_payload(adapter: str, text: str, transitions: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": text,
        "max_tokens": transitions + 1,
        "temperature": 0.0,
    }
    if adapter == "gufo":
        return payload
    if adapter == "kyojin":
        # The AR protocol launches kyojin with --draft-policy off, so no
        # speculative request switch is sent here. cache_prompt False matches
        # the llamacpp leg's cold-prompt discipline; return_token_ids gives
        # exact output ids without a text round-trip.
        payload.update(
            {
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "cache_prompt": False,
                "return_token_ids": True,
            }
        )
        return payload
    raise ValueError(f"unknown OpenAI-completions adapter: {adapter!r}")


def _post_openai_completion(
    host: str,
    port: int,
    payload: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    try:
        return _post_json(host, port, OPENAI_COMPLETIONS_PATH, payload, timeout)
    except urllib.error.HTTPError as exc:
        # gufo rejects unknown request fields; its server default sampling is
        # already greedy (temperature 0.0), so a rejected optional field falls
        # back to the minimal payload rather than failing the case.
        if exc.code != 400:
            raise
        body = exc.read().decode("utf-8", errors="replace")
        if "'temperature'" not in body and "temperature" not in body:
            raise
        minimal = {key: value for key, value in payload.items() if key in ("prompt", "max_tokens")}
        return _post_json(host, port, OPENAI_COMPLETIONS_PATH, minimal, timeout)


def openai_response_sample(
    *,
    adapter: str,
    tokenizer: Any,
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    client_wall_s: float,
    repetition: int,
    expected_transitions: int,
) -> dict[str, Any]:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        raise ValueError(f"{adapter} response omitted timings")
    prompt_tokens = int(case["prompt_tokens"])
    prompt_n = int(timings.get("prompt_n") or 0)
    usage = response.get("usage")
    if prompt_n == 0 and isinstance(usage, Mapping):
        prompt_n = int(usage.get("prompt_tokens") or 0)
    if prompt_n != prompt_tokens:
        raise ValueError(
            f"{adapter} evaluated {prompt_n} prompt tokens; expected {prompt_tokens}"
        )
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    predicted_n = int(timings.get("predicted_n") or 0)
    cached_n = int(timings.get("cache_n") or 0)
    if prompt_ms <= 0.0:
        raise ValueError(f"{adapter} response prompt_ms must be positive")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{adapter} response omitted choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError(f"{adapter} response choice must be an object")
    ids_source = "text_retokenized"
    raw_ids = response.get("token_ids")
    if not isinstance(raw_ids, list):
        raw_ids = choice.get("token_ids")
    if isinstance(raw_ids, list) and raw_ids:
        normalized_ids = [int(token_id) for token_id in raw_ids]
        ids_source = "return_token_ids"
    else:
        text = choice.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{adapter} response omitted completion text")
        normalized_ids = tokenizer.encode(text, add_special_tokens=False)
    transitions_actual = max(len(normalized_ids) - 1, 0)
    if transitions_actual:
        decode_tok_s = 1000.0 * transitions_actual / predicted_ms if predicted_ms > 0.0 else 0.0
    else:
        decode_tok_s = 0.0
    return {
        "case_id": str(case["id"]),
        "category": str(case["category"]),
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids_sha256": str(case["prompt_token_ids_sha256"]),
        "repetition": int(repetition),
        "prefill_ms": prompt_ms,
        "prefill_tok_s": 1000.0 * prompt_tokens / prompt_ms,
        "decode_ms": predicted_ms,
        "decode_transitions": transitions_actual,
        "decode_tok_s": decode_tok_s,
        "client_wall_s": float(client_wall_s),
        "output_token_count": len(normalized_ids),
        "output_token_ids": normalized_ids,
        "output_token_ids_sha256": token_ids_sha256(normalized_ids),
        "output_ids_source": ids_source,
        "cached_prompt_tokens": cached_n,
        "native_predicted_n": predicted_n,
        "native_predicted_per_second": timings.get("predicted_per_second"),
        "finish_reason": choice.get("finish_reason"),
        "decode_truncated": transitions_actual < int(expected_transitions),
    }


def run_openai(args: argparse.Namespace) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(args.fixture)
    cases = fixture["cases"]
    transitions = int(fixture["decode_transitions"])
    adapter = str(args.adapter)
    if adapter not in OPENAI_ADAPTERS:
        raise SystemExit(f"unsupported OpenAI-completions adapter: {adapter!r}")

    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer

    first_part = discover_gguf_files(args.model_root)[0]
    index = load_gguf_index(first_part)
    tokenizer = Qwen4ExpGGUFTokenizer.from_gguf_info(index)

    case_texts: dict[str, str] = {}
    disqualified: list[dict[str, Any]] = []
    for case in cases:
        text, exact_round_trip = _openai_case_text(tokenizer, case)
        if exact_round_trip:
            case_texts[str(case["id"])] = text
        else:
            disqualified.append(
                {"case_id": str(case["id"]), "reason": "detokenize/retokenize round-trip mismatch"}
            )
    qualified_cases = [case for case in cases if str(case["id"]) in case_texts]
    if not qualified_cases:
        raise SystemExit("no canonical case survived tokenizer round-trip verification")

    server_bin = args.server_bin.resolve()
    log_path = args.server_log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(server_bin), *args.server_arg]
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_engine_run",
        "engine": args.engine_label,
        "surface": "openai_completions",
        "adapter": adapter,
        "status": "running",
        "host": _host_metadata(),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": fixture_sha256,
        "model": str(args.model_root.resolve()),
        "server_binary": str(server_bin),
        "server_binary_sha256": sha256_path(server_bin),
        "source": _git_metadata(args.source_root),
        "command": command,
        "protocol": {
            "warmups_per_case": int(args.warmups),
            "measured_repetitions": int(args.repetitions),
            "decode_transitions": transitions,
            "visible_output_tokens": transitions + 1,
            "temperature": 0.0,
            "input_modality": "detokenized_raw_text",
            "chat_template": False,
            "prompt_cache": "cache_prompt=false where supported; measured rows must report cache_n=0",
            "tokenizer_round_trip_verified_cases": len(qualified_cases),
            "timing_boundary": (
                "engine-reported timings.prompt_ms for exact prefill; decode is the "
                "engine-reported predicted_ms over the actual generated tokens, so an "
                "early EOS truncates the decode leg (recorded per row) and the decode "
                "leg is not comparable to the ignore_eos llamacpp leg"
            ),
            "output_ids": (
                "return_token_ids where the engine returns ids; otherwise the completion "
                "text re-tokenized with the fixture tokenizer (recorded per row as "
                "output_ids_source)"
            ),
        },
        "disqualified_cases": disqualified,
        "warmups": [],
        "samples": [],
    }
    _write_json(args.output, artifact)
    server_cwd = args.server_cwd.resolve() if args.server_cwd else None
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            cwd=server_cwd,
        )
        try:
            _wait_for_health(args.host, args.port, args.startup_timeout)
            for warmup in range(args.warmups):
                for case in _measurement_order(qualified_cases, warmup):
                    started = time.perf_counter()
                    response = _post_openai_completion(
                        args.host,
                        args.port,
                        openai_completions_payload(adapter, case_texts[str(case["id"])], transitions),
                        args.request_timeout,
                    )
                    row = openai_response_sample(
                        adapter=adapter,
                        tokenizer=tokenizer,
                        case=case,
                        response=response,
                        client_wall_s=time.perf_counter() - started,
                        repetition=warmup,
                        expected_transitions=transitions,
                    )
                    artifact["warmups"].append(
                        {
                            "case_id": row["case_id"],
                            "repetition": warmup,
                            "output_token_ids_sha256": row["output_token_ids_sha256"],
                        }
                    )
                    print(
                        f"[warmup] {args.engine_label} {row['case_id']} "
                        f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                        flush=True,
                    )
            for repetition in range(args.repetitions):
                for case in _measurement_order(qualified_cases, repetition):
                    started = time.perf_counter()
                    response = _post_openai_completion(
                        args.host,
                        args.port,
                        openai_completions_payload(adapter, case_texts[str(case["id"])], transitions),
                        args.request_timeout,
                    )
                    row = openai_response_sample(
                        adapter=adapter,
                        tokenizer=tokenizer,
                        case=case,
                        response=response,
                        client_wall_s=time.perf_counter() - started,
                        repetition=repetition,
                        expected_transitions=transitions,
                    )
                    if int(row["cached_prompt_tokens"]) != 0:
                        raise ValueError(
                            f"{adapter} served case {row['case_id']} from the prompt cache "
                            f"(cache_n={row['cached_prompt_tokens']}); the measured prefill "
                            "is not a cold prompt evaluation"
                        )
                    artifact["samples"].append(row)
                    artifact["summary"] = summarize_samples(artifact["samples"])
                    _write_json(args.output, artifact)
                    print(
                        f"[measure {repetition}] {args.engine_label} {row['case_id']} "
                        f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                        flush=True,
                    )
            artifact["status"] = "completed"
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _terminate(process)
            server_log.flush()
            artifact["server_returncode"] = process.returncode
            artifact["server_log"] = str(log_path)
            artifact["server_log_sha256"] = sha256_path(log_path)
            _write_json(args.output, artifact)
    artifact["server_returncode"] = process.returncode
    artifact["server_log"] = str(log_path)
    artifact["server_log_sha256"] = sha256_path(log_path)
    artifact["summary"] = summarize_samples(artifact["samples"])
    _write_json(args.output, artifact)
    return artifact


def _proc_kib_fields(path: Path) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts and parts[0].isdigit():
            value = int(parts[0])
            fields[name] = value * 1024 if len(parts) > 1 and parts[1] == "kB" else value
    return fields


def _process_memory_snapshot() -> dict[str, int]:
    status = _proc_kib_fields(Path("/proc/self/status"))
    meminfo = _proc_kib_fields(Path("/proc/meminfo"))
    io = _proc_kib_fields(Path("/proc/self/io"))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "rss_bytes": status.get("VmRSS", 0),
        "rss_anon_bytes": status.get("RssAnon", 0),
        "rss_file_bytes": status.get("RssFile", 0),
        "host_mem_available_bytes": meminfo.get("MemAvailable", 0),
        "host_mem_free_bytes": meminfo.get("MemFree", 0),
        "host_swap_free_bytes": meminfo.get("SwapFree", 0),
        "process_read_bytes": io.get("read_bytes", 0),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
    }


def _hipengine_case_sample(
    runner: Any,
    *,
    case: Mapping[str, Any],
    repetition: int,
    transitions: int,
    prefill_kwargs: Mapping[str, Any] | None = None,
    step_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = runner.runtime
    token_ids = [int(token_id) for token_id in case["prompt_token_ids"]]
    prefill_call_kwargs = dict(prefill_kwargs or {})
    step_call_kwargs = dict(step_kwargs or {})
    memory_before = _process_memory_snapshot()
    runtime.device_synchronize()
    prefill_started = time.perf_counter_ns()
    result = runner.prefill(token_ids, **prefill_call_kwargs)
    prefill_enqueued = time.perf_counter_ns()
    runtime.device_synchronize()
    prefill_ended = time.perf_counter_ns()
    prefill_ms = (prefill_ended - prefill_started) / 1_000_000.0
    output_ids = [int(result.token_id)]
    current = int(result.token_id)
    decode_started = time.perf_counter_ns()
    for _ in range(int(transitions)):
        result = runner.step(current, **step_call_kwargs)
        current = int(result.token_id)
        output_ids.append(current)
    decode_enqueued = time.perf_counter_ns()
    runtime.device_synchronize()
    decode_ended = time.perf_counter_ns()
    decode_ms = (decode_ended - decode_started) / 1_000_000.0
    # `decode_ms` is max(host enqueue, device). Splitting the loop at the single
    # trailing synchronize separates the two windows, but note that the split
    # alone cannot attribute them: `runner.step` ends with a blocking
    # device-to-host copy of the argmax result (`_finish_lm_head`), so the host
    # window already contains the device wait and the trailing drain is zero by
    # construction. A zero drain therefore does NOT mean the step is host-bound.
    # The host-vs-device question needs the readback removed; see
    # `scripts/qwen4exp_decode_host_bound_probe.py`.
    prefill_host_enqueue_ms = (prefill_enqueued - prefill_started) / 1_000_000.0
    prefill_device_drain_ms = (prefill_ended - prefill_enqueued) / 1_000_000.0
    decode_host_enqueue_ms = (decode_enqueued - decode_started) / 1_000_000.0
    decode_device_drain_ms = (decode_ended - decode_enqueued) / 1_000_000.0
    client_wall_s = (prefill_ms + decode_ms) / 1000.0
    memory_after = _process_memory_snapshot()
    memory_delta = {
        key: memory_after[key] - memory_before[key]
        for key in ("process_read_bytes", "minor_faults", "major_faults")
    }
    return {
        "case_id": str(case["id"]),
        "category": str(case["category"]),
        "prompt_tokens": len(token_ids),
        "prompt_token_ids_sha256": str(case["prompt_token_ids_sha256"]),
        "repetition": int(repetition),
        "prefill_ms": prefill_ms,
        "prefill_tok_s": 1000.0 * len(token_ids) / prefill_ms,
        "decode_ms": decode_ms,
        "prefill_host_enqueue_ms": prefill_host_enqueue_ms,
        "prefill_device_drain_ms": prefill_device_drain_ms,
        "decode_host_enqueue_ms": decode_host_enqueue_ms,
        "decode_device_drain_ms": decode_device_drain_ms,
        "decode_host_enqueue_ms_per_step": decode_host_enqueue_ms / int(transitions),
        "decode_device_drain_ms_per_step": decode_device_drain_ms / int(transitions),
        "decode_transitions": int(transitions),
        "decode_tok_s": 1000.0 * int(transitions) / decode_ms,
        "client_wall_s": client_wall_s,
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_ids_sha256": token_ids_sha256(output_ids),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_delta": memory_delta,
        "phase_windows_ns": {
            "prefill": [prefill_started, prefill_ended],
            "decode": [decode_started, decode_ended],
        },
    }


def _capture_kwargs(mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the (prefill, step) capture kwargs for one capture mode.

    ``full`` is the runner default and what this bench has always measured:
    ``capture_logits=True`` copies the whole vocabulary to the host and takes a
    host argmax, which also forces a ``device_synchronize`` inside every step.

    ``compact`` is exactly what the production text generator passes for an
    uncontrolled greedy request (``_qwen4_exp_compact_*_kwargs``), so it is the
    configuration a serving client actually runs.
    """
    from hipengine.generation.qwen4_exp_gguf import (
        _qwen4_exp_compact_prefill_kwargs,
        _qwen4_exp_compact_step_kwargs,
    )

    if mode == "full":
        return {}, {}
    if mode == "compact":
        return (
            dict(_qwen4_exp_compact_prefill_kwargs()),
            dict(_qwen4_exp_compact_step_kwargs()),
        )
    raise ValueError(f"unknown capture mode {mode!r}")


def run_hipengine(args: argparse.Namespace) -> dict[str, Any]:
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    fixture, fixture_sha256 = load_fixture(args.fixture)
    cases = fixture["cases"]
    if args.case_id:
        selected = set(str(case_id) for case_id in args.case_id)
        cases = [row for row in cases if str(row["id"]) in selected]
        if {str(row["id"]) for row in cases} != selected:
            raise ValueError("unknown canonical case id in --case-id")
    transitions = int(fixture["decode_transitions"])
    model_root = args.model_root.resolve()
    max_sequence_length = max(int(row["prompt_tokens"]) for row in cases) + transitions + 8

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(str(args.execution_profile)),
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    prefill_kwargs, step_kwargs = _capture_kwargs(str(args.capture_mode))
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_engine_run",
        "engine": f"hipengine_{args.execution_profile}",
        "surface": "synchronized_direct_runner",
        "status": "running",
        "host": _host_metadata(),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": fixture_sha256,
        "model_root": str(model_root),
        "source": _git_metadata(ROOT),
        "profile": {
            "requested": str(args.execution_profile),
            "manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "protocol": {
            "case_ids": [str(row["id"]) for row in cases],
            "capture_mode": str(args.capture_mode),
            "prefill_kwargs": {k: bool(v) for k, v in sorted(prefill_kwargs.items())},
            "step_kwargs": {k: bool(v) for k, v in sorted(step_kwargs.items())},
            "ple_cache_mode": str(args.ple_cache_mode),
            "ple_cache_scope": "per_layer_token_embd.weight file range only",
            "ple_telemetry": bool(args.ple_telemetry),
            "ple_random_access": str(args.ple_random_access),
            "warmups_per_case": int(args.warmups),
            "measured_repetitions": int(args.repetitions),
            "decode_transitions": transitions,
            "visible_output_tokens": transitions + 1,
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "timing_boundary": (
                "synchronized runner.prefill including the first greedy output, followed by "
                "exactly N synchronized autoregressive runner.step transitions"
            ),
        },
        "warmups": [],
        "samples": [],
        "ple_cache_advice": [],
    }
    _write_json(args.output, artifact)
    generator = resolved.construct_generator(factory)
    try:
        if generator._resident is None:
            raise RuntimeError("Qwen4Exp canonical cache protocol needs resident PLE ownership")
        ple_table = generator._resident.ple_table
        ple_table.configure_random_access(args.ple_random_access)
        if args.ple_cache_mode == "warm":
            artifact["ple_cache_advice"].append(
                {"phase": "initial", **ple_table.advise_cache("warm")}
            )
        for warmup in range(args.warmups):
            for case in _measurement_order(cases, warmup):
                if args.ple_cache_mode == "cold":
                    artifact["ple_cache_advice"].append(
                        {"phase": "warmup", "case_id": str(case["id"]), **ple_table.advise_cache("cold")}
                    )
                if args.ple_telemetry:
                    ple_table.enable_telemetry()
                row = _hipengine_case_sample(
                    generator.runner,
                    case=case,
                    repetition=warmup,
                    transitions=transitions,
                    prefill_kwargs=prefill_kwargs,
                    step_kwargs=step_kwargs,
                )
                warmup_row = {
                    "case_id": row["case_id"],
                    "repetition": warmup,
                    "output_token_ids_sha256": row["output_token_ids_sha256"],
                }
                if args.ple_telemetry:
                    warmup_row["ple_telemetry"] = ple_table.telemetry()
                artifact["warmups"].append(warmup_row)
                print(
                    f"[warmup] hipengine {row['case_id']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        for repetition in range(args.repetitions):
            for case in _measurement_order(cases, repetition):
                if args.ple_cache_mode == "cold":
                    artifact["ple_cache_advice"].append(
                        {"phase": "measure", "case_id": str(case["id"]), "repetition": repetition, **ple_table.advise_cache("cold")}
                    )
                if args.ple_telemetry:
                    ple_table.enable_telemetry()
                row = _hipengine_case_sample(
                    generator.runner,
                    case=case,
                    repetition=repetition,
                    transitions=transitions,
                    prefill_kwargs=prefill_kwargs,
                    step_kwargs=step_kwargs,
                )
                if args.ple_telemetry:
                    row["ple_telemetry"] = ple_table.telemetry()
                artifact["samples"].append(row)
                artifact["summary"] = summarize_samples(artifact["samples"])
                _write_json(args.output, artifact)
                print(
                    f"[measure {repetition}] hipengine {row['case_id']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        if args.host_profile:
            artifact.setdefault("host_profiles", []).append(
                _hipengine_host_profile(
                    generator.runner,
                    case=cases[0],
                    transitions=transitions,
                    prefill_kwargs=prefill_kwargs,
                    step_kwargs=step_kwargs,
                )
            )
            _write_json(args.output, artifact)
            for entry in artifact["host_profiles"][-1]["entries"][:12]:
                print(
                    f"[host-profile] {entry['share'] * 100:5.1f}% {entry['ncalls']:>9d}  {entry['function']}",
                    flush=True,
                )
        artifact["status"] = "completed"
        artifact["memory_before_close"] = memory_stats()
    finally:
        generator.close()
    artifact["memory_after_close"] = memory_stats()
    artifact["summary"] = summarize_samples(artifact["samples"])
    _write_json(args.output, artifact)
    return artifact


def _hipengine_host_profile(
    runner: Any,
    *,
    case: Mapping[str, Any],
    transitions: int,
    prefill_kwargs: Mapping[str, Any] | None = None,
    step_kwargs: Mapping[str, Any] | None = None,
    top: int = 30,
) -> dict[str, Any]:
    """Attribute the host side of the bench's own decode loop.

    `wall - kernel_sum` says how much of a decode step is not device kernel time
    but not what it is. This profiles the bench's own loop (not a second
    harness's) so the attribution belongs to the measured configuration.

    cProfile inflates absolute times, so the returned `total_s` and the per-entry
    `tottime_s` are not a timing measurement - read the ordering and the shares.
    The pass is extra: `runner.reset()` restores the position first, so it does
    not grow the sequence and does not perturb the measured samples.
    """

    import cProfile
    import pstats

    token_ids = [int(token_id) for token_id in case["prompt_token_ids"]]
    prefill_call_kwargs = dict(prefill_kwargs or {})
    step_call_kwargs = dict(step_kwargs or {})
    runner.reset()
    result = runner.prefill(token_ids, **prefill_call_kwargs)
    current = int(result.token_id)
    runner.runtime.device_synchronize()

    def _loop() -> None:
        nonlocal current
        for _ in range(int(transitions)):
            current = int(runner.step(current, **step_call_kwargs).token_id)

    profiler = cProfile.Profile()
    profiler.enable()
    _loop()
    profiler.disable()
    runner.runtime.device_synchronize()
    stats = pstats.Stats(profiler)
    total = stats.total_tt
    callers_of = {key: value[4] for key, value in stats.stats.items()}
    entries = []
    for key, (_cc, ncalls, tottime, _ct, _callers) in sorted(
        stats.stats.items(), key=lambda kv: kv[1][2], reverse=True
    )[: int(top)]:
        callers = [
            {
                "function": caller[2],
                "file": caller[0],
                "line": caller[1],
                "ncalls": int(data[1]),
                "tottime_s": round(float(data[3]), 6),
            }
            for caller, data in sorted(
                callers_of.get(key, {}).items(), key=lambda kv: -kv[1][3]
            )[:5]
        ]
        filename, lineno, func = key
        entries.append(
            {
                "function": func,
                "file": filename,
                "line": lineno,
                "ncalls": int(ncalls),
                "tottime_s": round(float(tottime), 6),
                "share": round(float(tottime) / total, 6) if total else 0.0,
                "callers": callers,
            }
        )
    return {
        "case_id": str(case["id"]),
        "transitions": int(transitions),
        "total_s": round(float(total), 6),
        "entries": entries,
    }


def compare_engine_artifacts(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("compare mode requires at least two engine artifacts")
    engines: dict[str, dict[str, Any]] = {}
    fixture_hashes: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"engine artifact is not completed: {path}")
        engine = str(payload["engine"])
        if engine in engines:
            raise ValueError(f"duplicate engine label {engine!r}")
        fixture_hashes.add(str(payload["fixture_sha256"]))
        engines[engine] = payload
    if len(fixture_hashes) != 1:
        raise ValueError("engine artifacts used different exact-token fixtures")

    case_sets = [
        {str(row["case_id"]) for row in payload["samples"]}
        for payload in engines.values()
    ]
    if not case_sets[0] or any(case_set != case_sets[0] for case_set in case_sets[1:]):
        raise ValueError("engine artifacts do not contain identical non-empty case sets")
    case_ids = case_sets[0]
    correctness: dict[str, Any] = {}
    for case_id in sorted(case_ids):
        hashes: dict[str, list[str]] = {}
        for engine, payload in engines.items():
            hashes[engine] = sorted(
                {
                    str(row["output_token_ids_sha256"])
                    for row in payload["samples"]
                    if str(row["case_id"]) == case_id
                }
            )
        correctness[case_id] = {
            "all_engines_exact": len({digest for rows in hashes.values() for digest in rows}) == 1,
            "engine_hashes": hashes,
        }

    return {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_comparison",
        "fixture_sha256": next(iter(fixture_hashes)),
        "engine_artifacts": {
            engine: {"path": str(path.resolve()), "sha256": sha256_path(path)}
            for engine, path in zip(engines, paths, strict=True)
        },
        "engines": {
            engine: {
                "source": payload.get("source"),
                "profile": payload.get("profile"),
                "summary": payload["summary"],
            }
            for engine, payload in engines.items()
        },
        "cross_engine_outputs": {
            "all_cases_exact": all(row["all_engines_exact"] for row in correctness.values()),
            "exact_case_count": sum(row["all_engines_exact"] for row in correctness.values()),
            "case_count": len(correctness),
            "cases": correctness,
        },
    }


def _parse_shapes(value: str) -> tuple[int, ...]:
    try:
        shapes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shapes must be comma-separated integers") from exc
    if not shapes or any(shape <= 0 for shape in shapes):
        raise argparse.ArgumentTypeError("shapes must be positive")
    return shapes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    fixture_parser = subparsers.add_parser("fixture", help="Build the committed exact-token fixture")
    fixture_parser.add_argument("--model-root", type=Path, required=True)
    fixture_parser.add_argument("--source-prompts", type=Path, default=DEFAULT_SOURCE_PROMPTS)
    fixture_parser.add_argument("--shapes", type=_parse_shapes, default=CANONICAL_SHAPES)
    fixture_parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)

    hip_parser = subparsers.add_parser("hipengine", help="Run hipEngine production")
    hip_parser.add_argument("--model-root", type=Path, required=True)
    hip_parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    hip_parser.add_argument("--output", type=Path, required=True)
    hip_parser.add_argument("--prefill-chunk-size", type=int, default=512)
    hip_parser.add_argument(
        "--execution-profile", choices=("strict", "production"), default="production"
    )
    hip_parser.add_argument(
        "--case-id", nargs="+",
        help="Measure only these IDs after validating the complete canonical fixture",
    )
    hip_parser.add_argument(
        "--capture-mode", choices=("full", "compact"), default="full",
        help=(
            "full = runner defaults (capture_logits=True, host argmax, per-step synchronize); "
            "compact = the production generator's uncontrolled greedy kwargs (device argmax)"
        ),
    )
    hip_parser.add_argument(
        "--ple-cache-mode", choices=("warm", "cold"), default="warm",
        help="File-scoped PLE cache protocol; cold evicts only the PLE tensor range before each request",
    )
    hip_parser.add_argument(
        "--ple-random-access", choices=("off", "auto", "on"), default="off",
        help="Sparse mmap advice paired with page-aligned merged WILLNEED row prefetch",
    )
    hip_parser.add_argument(
        "--ple-telemetry", action="store_true",
        help="Record opt-in PLE row/page locality, fault proxies, and staging/H2D wall per request",
    )
    hip_parser.add_argument(
        "--host-profile", action="store_true",
        help=(
            "After the measured repetitions, profile the bench's own decode loop with "
            "cProfile and record the top functions by tottime. Absolute times are "
            "inflated by the profiler; read the ordering and the shares. The profiled "
            "pass is extra and resets the sequence first, so the samples are unaffected."
        ),
    )
    hip_parser.add_argument("--warmups", type=int, default=1)
    hip_parser.add_argument("--repetitions", type=int, default=3)
    hip_parser.add_argument("--compiler-version-file", type=Path)
    hip_parser.add_argument("--require-cached-build", action="store_true")

    llama_parser = subparsers.add_parser("llamacpp", help="Run a llama.cpp-compatible server")
    llama_parser.add_argument("--server-bin", type=Path, required=True)
    llama_parser.add_argument("--model", type=Path, required=True)
    llama_parser.add_argument("--engine-label", required=True)
    llama_parser.add_argument("--source-root", type=Path)
    llama_parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    llama_parser.add_argument("--output", type=Path, required=True)
    llama_parser.add_argument("--server-log", type=Path, required=True)
    llama_parser.add_argument("--server-arg", action="append", default=[])
    llama_parser.add_argument("--host", default="127.0.0.1")
    llama_parser.add_argument("--port", type=int, default=18080)
    llama_parser.add_argument("--warmups", type=int, default=1)
    llama_parser.add_argument("--repetitions", type=int, default=3)
    llama_parser.add_argument("--startup-timeout", type=float, default=1800.0)
    llama_parser.add_argument("--request-timeout", type=float, default=1800.0)

    openai_parser = subparsers.add_parser(
        "openai", help="Run an OpenAI-completions server (gufo, kyojin) on the canonical fixture"
    )
    openai_parser.add_argument("--adapter", choices=OPENAI_ADAPTERS, required=True)
    openai_parser.add_argument("--server-bin", type=Path, required=True)
    openai_parser.add_argument("--server-arg", action="append", default=[])
    openai_parser.add_argument("--server-cwd", type=Path)
    openai_parser.add_argument("--model-root", type=Path, required=True, help="GGUF root supplying the fixture tokenizer")
    openai_parser.add_argument("--engine-label", required=True)
    openai_parser.add_argument("--source-root", type=Path)
    openai_parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    openai_parser.add_argument("--output", type=Path, required=True)
    openai_parser.add_argument("--server-log", type=Path, required=True)
    openai_parser.add_argument("--host", default="127.0.0.1")
    openai_parser.add_argument("--port", type=int, default=18080)
    openai_parser.add_argument("--warmups", type=int, default=1)
    openai_parser.add_argument("--repetitions", type=int, default=3)
    openai_parser.add_argument("--startup-timeout", type=float, default=1800.0)
    openai_parser.add_argument("--request-timeout", type=float, default=1800.0)

    compare_parser = subparsers.add_parser("compare", help="Join completed engine artifacts")
    compare_parser.add_argument("artifacts", nargs="+", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "warmups", 1) < 0 or getattr(args, "repetitions", 1) <= 0:
        raise SystemExit("warmups must be non-negative and repetitions must be positive")
    if args.mode == "fixture":
        from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
        from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer

        first_part = discover_gguf_files(args.model_root)[0]
        index = load_gguf_index(first_part)
        tokenizer = Qwen4ExpGGUFTokenizer.from_gguf_info(index)
        source_rows = _read_prompt_rows(args.source_prompts)
        try:
            source_path = args.source_prompts.resolve().relative_to(ROOT)
        except ValueError:
            source_path = args.source_prompts.resolve()
        payload = build_fixture(
            tokenizer=tokenizer,
            source_rows=source_rows,
            shapes=args.shapes,
            source_path=source_path,
            source_sha256=sha256_path(args.source_prompts),
            model_identity={
                "architecture": index.architecture,
                "model_root": str(args.model_root),
                "first_part": first_part.name,
                "first_part_sha256": sha256_path(first_part),
                "tokenizer_backend": tokenizer.encoder_backend,
            },
        )
        _write_json(args.output, payload)
        print(json.dumps({"output": str(args.output), "sha256": sha256_path(args.output), "cases": len(payload["cases"])}, indent=2))
        return 0
    if args.mode == "hipengine":
        artifact = run_hipengine(args)
    elif args.mode == "llamacpp":
        if not args.server_bin.is_file() or not args.model.is_file():
            raise SystemExit("server binary and model first part must exist")
        artifact = run_llamacpp(args)
    elif args.mode == "openai":
        if not args.server_bin.is_file():
            raise SystemExit("server binary must exist")
        artifact = run_openai(args)
    else:
        artifact = compare_engine_artifacts(args.artifacts)
        _write_json(args.output, artifact)
    print(json.dumps({"kind": artifact["kind"], "status": artifact.get("status", "completed"), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
