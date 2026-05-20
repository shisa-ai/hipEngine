#!/usr/bin/env python3
"""GGUF resident decode graph/eager correctness and replay timing smoke."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from hipengine.loading.gguf import scan_gguf

DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        help="Optional rocprofv3 kernel-trace CSV to validate against the active decode graph symbol groups.",
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        help="Optional previous graph-smoke JSON whose graph_bucket.active_symbol_groups supplies coverage expectations.",
    )
    parser.add_argument(
        "--expected-symbol-groups",
        help="Comma-separated active symbol groups for --coverage-csv validation.",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only validate --coverage-csv symbol coverage; do not run the model.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = run_coverage(args) if args.coverage_only else run(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")
    return 0 if result["passed"] else 1


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = json.loads(args.fixture.read_text())
    model = Path(args.model or fixture["model"]["path"])
    max_new_tokens = int(args.max_new_tokens or fixture["sampling"]["max_new_tokens"])
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids = [int(item) for item in fixture["prompt_ids"]]
    expected_ids = [int(item) for item in fixture["expected_generated_token_ids"][:max_new_tokens]]
    compiler_version = args.compiler_version_file.read_text() if args.compiler_version_file else None

    eager = _run_eager(
        model,
        prompt_ids,
        max_new_tokens,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    graph = _run_graph(
        model,
        prompt_ids,
        max_new_tokens,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )

    logits_ref = eager["final_logits"]
    logits_graph = graph["final_logits"]
    max_abs = float(np.max(np.abs(logits_ref - logits_graph)))
    mean_abs = float(np.mean(np.abs(logits_ref - logits_graph)))
    kl = _kl_divergence(logits_ref.reshape(-1), logits_graph.reshape(-1))
    eager_top1 = int(np.argmax(logits_ref.reshape(-1)))
    graph_top1 = int(np.argmax(logits_graph.reshape(-1)))

    info = scan_gguf(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    graph_text = tokenizer.decode(graph["generated_ids"])
    eager_text = tokenizer.decode(eager["generated_ids"])

    finite_logits = bool(np.all(np.isfinite(logits_graph)) and np.all(np.isfinite(logits_ref)))
    ids_match = graph["generated_ids"] == eager["generated_ids"] == expected_ids
    top1_equal = eager_top1 == graph_top1
    passed = bool(
        ids_match
        and top1_equal
        and max_abs <= 1.0e-5
        and kl <= 0.05
        and finite_logits
    )
    result = {
        "schema": 1,
        "mode": "gguf_decode_graph_replay_correctness",
        "model": str(model),
        "prompt_ids": prompt_ids,
        "max_new_tokens": max_new_tokens,
        "expected_generated_token_ids": expected_ids,
        "eager_generated_token_ids": eager["generated_ids"],
        "graph_generated_token_ids": graph["generated_ids"],
        "eager_text": eager_text,
        "graph_text": graph_text,
        "ids_match": bool(ids_match),
        "final_logits": {
            "shape": list(logits_graph.shape),
            "finite": finite_logits,
            "eager_top1": eager_top1,
            "graph_top1": graph_top1,
            "top1_equal": bool(top1_equal),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "kl_eager_to_graph": kl,
        },
        "timing_seconds": {
            "eager_prefill": eager["prefill_seconds"],
            "eager_decode": eager["decode_seconds"],
            "graph_prefill": graph["prefill_seconds"],
            "graph_capture": graph["capture_seconds"],
            "graph_replay_decode_excludes_capture": graph["replay_seconds"],
        },
        "graph_bucket": graph["graph_bucket"],
        "notes": [
            "graph_capture is reported separately and is excluded from graph_replay_decode_excludes_capture",
            "graph replay consumes the device lm-head argmax token and advances device position/context inside the captured graph",
            "graph_bucket.active_symbol_groups enumerates the active decode kernel symbols expected in a rocprof graph-replay trace",
        ],
        "passed": passed,
    }
    if args.coverage_csv is not None:
        graph_bucket = result.get("graph_bucket") or {}
        expected = _expected_symbol_groups_from_args(
            args,
            default=tuple(graph_bucket.get("active_symbol_groups", ())),
        )
        coverage = validate_decode_graph_symbol_coverage(
            read_kernel_names_from_rocprof_csv(args.coverage_csv),
            expected_groups=expected,
        )
        result["decode_graph_symbol_coverage"] = coverage
        result["passed"] = bool(result["passed"] and coverage["passed"])
    return result


def _run_eager(
    model: Path,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    compiler_version: str | None,
    require_cached_build: bool,
) -> dict[str, Any]:
    with Qwen35GGUFResidentSession(
        model,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
    ) as session:
        t0 = time.perf_counter()
        result = session.prefill(prompt_ids)
        prefill_seconds = time.perf_counter() - t0
        generated = [int(result.token_id)]
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            result = session.step(result.token_id)
            generated.append(int(result.token_id))
        decode_seconds = time.perf_counter() - t1
        final_logits = result.logits.copy()
    return {
        "generated_ids": generated,
        "final_logits": final_logits,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
    }


def _run_graph(
    model: Path,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    compiler_version: str | None,
    require_cached_build: bool,
) -> dict[str, Any]:
    with Qwen35GGUFResidentSession(
        model,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
    ) as session:
        t0 = time.perf_counter()
        result = session.prefill(prompt_ids)
        prefill_seconds = time.perf_counter() - t0
        generated = [int(result.token_id)]
        final = result
        remaining = max_new_tokens - 1
        capture_seconds = 0.0
        replay_seconds = 0.0
        graph_bucket: dict[str, Any] | None = None
        if remaining > 0:
            t_capture = time.perf_counter()
            graph = session.capture_decode_graph(
                position=len(prompt_ids),
                steps_per_replay=1,
                max_replay_steps=remaining,
                record_steps=remaining,
            )
            graph_bucket = None if graph.bucket_key is None else graph.bucket_key.as_dict()
            capture_seconds = time.perf_counter() - t_capture
            try:
                t_replay = time.perf_counter()
                graph.replay(remaining)
                replay_seconds = time.perf_counter() - t_replay
                generated.extend(graph.read_generated_token_ids(remaining))
                final = graph.read_sample()
            finally:
                graph.close()
        final_logits = final.logits.copy()
    return {
        "generated_ids": generated,
        "final_logits": final_logits,
        "prefill_seconds": prefill_seconds,
        "capture_seconds": capture_seconds,
        "replay_seconds": replay_seconds,
        "graph_bucket": graph_bucket,
    }


_DECODE_GRAPH_SYMBOL_GROUP_REGEX: dict[str, tuple[str, ...]] = {
    "moe_q4_k_selected_dual": (
        r"q4_k_t16_selected_dual.*gemv",
        r"gguf_q4_k_selected_dual_pack8_gemv_decode",
    ),
    "moe_q5_k_selected": (
        r"qk_t16_selected.*<[^>]*,\s*5>",
        r"gguf_k_selected_pack8_gemv_decode.*<[^>]*,\s*5>",
    ),
    "moe_q6_k_selected": (
        r"qk_t16_selected.*<[^>]*,\s*6>",
        r"gguf_k_selected_pack8_gemv_decode.*<[^>]*,\s*6>",
    ),
    "dense_q8_0_single": (
        r"q8_0_t16_gemv_kernel",
        r"gguf_q8_0_pack8_gemv_decode",
    ),
    "dense_q8_0_dual": (
        r"q8_0_t16_dual_gemv_kernel",
        r"gguf_q8_0_pack8_dual_gate_up_gemv_decode",
    ),
    "dense_q4_k": (r"gguf_q4_k_pack8_gemv_decode",),
    "dense_q6_k_lm_head": (
        r"q6_k_t16_gemv_kernel",
        r"gguf_q6_k_pack8_gemv_decode",
    ),
    "gdn_decode": (r"qwen35_gdn_recurrent",),
    "paged_kv_write": (r"qwen35_write_paged_kv",),
    "paged_full_attention_decode": (r"qwen35_paged_full_attn_decode",),
}


def run_coverage(args: argparse.Namespace) -> dict[str, Any]:
    if args.coverage_csv is None:
        raise ValueError("--coverage-only requires --coverage-csv")
    expected = _expected_symbol_groups_from_args(args)
    coverage = validate_decode_graph_symbol_coverage(
        read_kernel_names_from_rocprof_csv(args.coverage_csv),
        expected_groups=expected,
    )
    return {
        "schema": 1,
        "mode": "gguf_decode_graph_symbol_coverage",
        "coverage_csv": str(args.coverage_csv),
        "expected_symbol_groups_source": _expected_symbol_groups_source(args),
        "decode_graph_symbol_coverage": coverage,
        "passed": bool(coverage["passed"]),
    }


def read_kernel_names_from_rocprof_csv(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or "").strip()
            if name:
                names.append(name)
    return names


def validate_decode_graph_symbol_coverage(
    kernel_names: list[str] | tuple[str, ...],
    *,
    expected_groups: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    expected = tuple(dict.fromkeys(str(group) for group in expected_groups if str(group)))
    by_group: dict[str, list[str]] = {group: [] for group in expected}
    unknown_groups = sorted(group for group in expected if group not in _DECODE_GRAPH_SYMBOL_GROUP_REGEX)
    lowered = [(name, name.lower()) for name in kernel_names]
    for group in expected:
        for pattern in _DECODE_GRAPH_SYMBOL_GROUP_REGEX.get(group, ()):
            regex = re.compile(pattern)
            for original, lower in lowered:
                if regex.search(lower) and original not in by_group[group]:
                    by_group[group].append(original)
    missing = sorted(group for group, names in by_group.items() if not names)
    return {
        "passed": not missing and not unknown_groups,
        "expected_symbol_groups": list(expected),
        "observed_symbol_groups": sorted(group for group, names in by_group.items() if names),
        "missing_symbol_groups": missing,
        "unknown_symbol_groups": unknown_groups,
        "kernel_names_by_group": {group: sorted(names) for group, names in by_group.items()},
    }


def _expected_symbol_groups_from_args(
    args: argparse.Namespace,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if args.expected_symbol_groups:
        return tuple(item.strip() for item in args.expected_symbol_groups.split(",") if item.strip())
    if args.coverage_json is not None:
        payload = json.loads(args.coverage_json.read_text())
        try:
            groups = payload["graph_bucket"]["active_symbol_groups"]
        except KeyError as exc:
            raise ValueError("--coverage-json must contain graph_bucket.active_symbol_groups") from exc
        return tuple(str(item) for item in groups)
    if default is not None and default:
        return tuple(default)
    raise ValueError("coverage validation requires --expected-symbol-groups or --coverage-json")


def _expected_symbol_groups_source(args: argparse.Namespace) -> str:
    if args.expected_symbol_groups:
        return "--expected-symbol-groups"
    if args.coverage_json is not None:
        return str(args.coverage_json)
    return "graph_bucket.active_symbol_groups"


def _kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    ref_log_z = _logsumexp(ref)
    cand_log_z = _logsumexp(cand)
    log_p = ref - ref_log_z
    log_q = cand - cand_log_z
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def _logsumexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


if __name__ == "__main__":
    raise SystemExit(main())
