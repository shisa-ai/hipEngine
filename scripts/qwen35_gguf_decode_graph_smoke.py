#!/usr/bin/env python3
"""GGUF resident decode graph/eager correctness and replay timing smoke."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = run(args)
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
    return {
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
        "notes": [
            "graph_capture is reported separately and is excluded from graph_replay_decode_excludes_capture",
            "graph replay consumes the device lm-head argmax token and advances device position/context inside the captured graph",
        ],
        "passed": passed,
    }


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
        if remaining > 0:
            t_capture = time.perf_counter()
            graph = session.capture_decode_graph(
                position=len(prompt_ids),
                steps_per_replay=1,
                max_replay_steps=remaining,
                record_steps=remaining,
            )
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
    }


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
