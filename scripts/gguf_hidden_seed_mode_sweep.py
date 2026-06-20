#!/usr/bin/env python3
"""Sweep hipEngine hidden-seed capture modes against a llama.cpp seed oracle."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType  # noqa: E402
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr  # noqa: E402
from scripts.llamacpp_mtp_compare_hidden_seed import (  # noqa: E402
    compare_capture_vectors,
    parse_prompt_tokens,
    redact_values,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    pack_float32,
    sha256_bytes,
    summarize_capture,
    summarize_floats,
    top_abs_diff_entries,
    unpack_float32,
)

DEFAULT_LLAMACPP_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter303-hidden-seed-compare.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter304-hidden-seed-mode-sweep.json"
)
DEFAULT_MODES = "prefill-bulk,prefill-native,prefill-serial,step-serial"

ModeCaptureFn = Callable[[Path, tuple[int, ...], int, int | None, str], dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-artifact", type=Path, default=DEFAULT_LLAMACPP_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exact-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=304)
    args = parser.parse_args()

    artifact = build_mode_sweep_artifact(
        llamacpp_artifact_path=args.llamacpp_artifact,
        model_path=args.model,
        prompt_tokens=parse_prompt_tokens(args.prompt_tokens),
        position=args.position,
        modes=parse_modes(args.modes),
        max_sequence_length=args.max_sequence_length,
        exact_atol=args.exact_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "best_mode": artifact["ranking"].get("best_mode"),
                "hip_modes_match": artifact["ranking"].get("hip_modes_match"),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_mode_sweep_artifact(
    *,
    llamacpp_artifact_path: Path,
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    modes: Sequence[str],
    max_sequence_length: int | None = None,
    exact_atol: float = 0.0,
    iteration: int = 304,
    mode_capture_fn: ModeCaptureFn | None = None,
) -> dict[str, Any]:
    if int(position) != len(prompt_tokens) - 1:
        raise ValueError("mode sweep currently targets the final prompt token")
    prior = json.loads(llamacpp_artifact_path.read_text())
    llama_capture = summarize_capture(
        binary_path=Path(prior["llamacpp_capture"]["binary_path"]),
        meta_path=Path(prior["llamacpp_capture"]["metadata_path"]),
    )
    capture_fn = mode_capture_fn or capture_hipengine_hidden_seed_mode
    mode_results = []
    raw_captures: dict[str, dict[str, Any]] = {}
    for mode in modes:
        capture = capture_fn(
            model_path,
            prompt_tokens,
            int(position),
            max_sequence_length,
            str(mode),
        )
        raw_captures[str(mode)] = capture
        delta = compare_capture_vectors(
            llamacpp_capture=llama_capture,
            hipengine_capture=capture,
            exact_atol=exact_atol,
        )
        mode_results.append(
            {
                "mode": str(mode),
                "status": "matched" if delta.get("exact_match") else "mismatched",
                "hipengine_capture": redact_values(capture),
                "numeric_delta": delta,
            }
        )
    pairwise = compare_hipengine_modes(raw_captures, exact_atol=exact_atol)
    ranking = rank_mode_results(mode_results, pairwise)
    conclusion = conclude(ranking=ranking, mode_results=mode_results)
    return {
        "schema": 1,
        "kind": "hipengine_hidden_seed_mode_sweep",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "matched" if ranking.get("any_exact_match") else "mismatched",
        "llamacpp_artifact_path": str(llamacpp_artifact_path),
        "llamacpp_capture": llama_capture,
        "model": str(model_path),
        "prompt_tokens": list(prompt_tokens),
        "position": int(position),
        "token_id": int(prompt_tokens[position]),
        "modes": list(modes),
        "max_sequence_length": max_sequence_length,
        "exact_atol": float(exact_atol),
        "mode_results": mode_results,
        "hipengine_pairwise": pairwise,
        "ranking": ranking,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion),
    }


def capture_hipengine_hidden_seed_mode(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    max_sequence_length: int | None,
    mode: str,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "mode": mode, "values": []}
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        if mode == "prefill-bulk":
            result = session.prefill(
                list(prompt_tokens),
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
        elif mode == "prefill-native":
            result = session.prefill(
                list(prompt_tokens),
                use_bulk=True,
                bulk_attention_mode="native",
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
        elif mode == "prefill-serial":
            result = session.prefill(
                list(prompt_tokens),
                use_bulk=False,
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
        elif mode == "step-serial":
            result = None
            for index, token_id in enumerate(prompt_tokens):
                result = session.step(
                    int(token_id),
                    position=index,
                    return_logits=False,
                    capture_hidden_seed_fp32=index == int(position),
                )
            assert result is not None
        else:
            raise ValueError(f"unsupported hidden seed capture mode: {mode}")
        return copy_session_fp32_seed(
            session=session,
            mode=mode,
            position=position,
            token_id=int(prompt_tokens[position]),
            next_token_id=int(result.token_id),
        )


def copy_session_fp32_seed(
    *,
    session: Any,
    mode: str,
    position: int,
    token_id: int,
    next_token_id: int,
) -> dict[str, Any]:
    contract = session.fp32_hidden_seed_contract(rows=1).as_dict()
    ptr = session.fp32_hidden_seed_ptr()
    runtime = session.runtime
    if runtime is None:
        raise RuntimeError("GGUF resident session runtime is unavailable")
    values = np.empty((session.runner.hidden_size,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(values),
        DeviceBuffer(int(ptr), values.nbytes),
        values.nbytes,
        runtime=runtime,
    )
    runtime.device_synchronize()
    return {
        "status": "captured",
        "mode": mode,
        "position": int(position),
        "token_id": int(token_id),
        "next_token_id": int(next_token_id),
        "contract": contract,
        "dtype": DType.FP32.name,
        "values": [float(value) for value in values.tolist()],
    }


def compare_hipengine_modes(
    captures: Mapping[str, dict[str, Any]], *, exact_atol: float
) -> dict[str, Any]:
    rows = []
    items = list(captures.items())
    for i, (left_name, left_capture) in enumerate(items):
        for right_name, right_capture in items[i + 1 :]:
            rows.append(
                compare_named_vectors(
                    left_name,
                    left_capture,
                    right_name,
                    right_capture,
                    exact_atol=exact_atol,
                )
            )
    return {
        "available": bool(rows),
        "all_exact": all(row.get("exact_match") for row in rows) if rows else False,
        "pairs": rows,
    }


def compare_named_vectors(
    left_name: str,
    left_capture: dict[str, Any],
    right_name: str,
    right_capture: dict[str, Any],
    *,
    exact_atol: float,
) -> dict[str, Any]:
    if left_capture.get("status") != "captured" or right_capture.get("status") != "captured":
        return {"left": left_name, "right": right_name, "available": False}
    left = [float(value) for value in left_capture["values"]]
    right = [float(value) for value in right_capture["values"]]
    if len(left) != len(right):
        return {
            "left": left_name,
            "right": right_name,
            "available": True,
            "shape_match": False,
            "left_count": len(left),
            "right_count": len(right),
        }
    diffs = [a - b for a, b in zip(left, right)]
    abs_diffs = [abs(value) for value in diffs]
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    return {
        "left": left_name,
        "right": right_name,
        "available": True,
        "shape_match": True,
        "left_sha256": sha256_bytes(pack_float32(left)),
        "right_sha256": sha256_bytes(pack_float32(right)),
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs))
        if diffs
        else 0.0,
        "top_abs_diff": top_abs_diff_entries(left, right, limit=4),
        "exact_match": max_abs <= exact_atol,
    }


def rank_mode_results(
    mode_results: Sequence[dict[str, Any]], pairwise: dict[str, Any]
) -> dict[str, Any]:
    comparable = [row for row in mode_results if row["numeric_delta"].get("shape_match")]
    by_rmse = sorted(comparable, key=lambda row: row["numeric_delta"].get("rmse", float("inf")))
    return {
        "any_exact_match": any(row["status"] == "matched" for row in mode_results),
        "best_mode": compact_mode_row(by_rmse[0]) if by_rmse else None,
        "by_rmse": [compact_mode_row(row) for row in by_rmse],
        "hip_modes_match": bool(pairwise.get("all_exact")),
    }


def compact_mode_row(row: dict[str, Any]) -> dict[str, Any]:
    delta = row["numeric_delta"]
    return {
        "mode": row["mode"],
        "status": row["status"],
        "rmse": delta.get("rmse"),
        "max_abs_diff": delta.get("max_abs_diff"),
        "mean_abs_diff": delta.get("mean_abs_diff"),
        "sha256": row["hipengine_capture"].get("summary", {}).get("sha256"),
    }


def conclude(*, ranking: dict[str, Any], mode_results: Sequence[dict[str, Any]]) -> str:
    if ranking.get("any_exact_match"):
        return "some_hipengine_seed_mode_matches_llamacpp"
    if ranking.get("hip_modes_match"):
        return "hipengine_seed_modes_agree_but_mismatch_llamacpp"
    if any(row["mode"] == "prefill-bulk" and row["status"] == "mismatched" for row in mode_results):
        return "hipengine_seed_modes_diverge_and_mismatch_llamacpp"
    return "hidden_seed_mode_sweep_inconclusive"


def next_action(conclusion: str) -> str:
    if conclusion == "some_hipengine_seed_mode_matches_llamacpp":
        return "route_mtp_seed_capture_through_matching_mode_or_fix_default"
    if conclusion == "hipengine_seed_modes_agree_but_mismatch_llamacpp":
        return "audit_shared_bf16_activation_or_output_norm_precision"
    if conclusion == "hipengine_seed_modes_diverge_and_mismatch_llamacpp":
        return "bisect_prefill_bulk_native_serial_seed_path"
    return "inspect_hidden_seed_mode_sweep_failure"


def parse_modes(text: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in text.split(",") if item.strip())
    if not modes:
        raise ValueError("mode list is empty")
    return modes


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
