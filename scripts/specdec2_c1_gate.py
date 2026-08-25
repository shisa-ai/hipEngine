#!/usr/bin/env python3
"""Exact gfx1151 SPECDEC2 c1/c2/c4 staged/direct/AR qualification gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence



def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def _choice_ids(response: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(token)
        for token in response["choices"][0]["hipengine"]["generated_token_ids"]
    )


def _choice_rows(response: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token) for token in choice["hipengine"]["generated_token_ids"])
        for choice in response["choices"]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from hipengine.generation.qwen35_gguf import _gguf_mtp_required_tensor_names
    from hipengine.generation.registry import GenerationRequest
    from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
    from hipengine.server import ServerConfig, create_app

    provider_fingerprints: list[dict[str, Any]] = []
    original_release_request = Qwen35GGUFNextNDraftProvider.release_request
    if args.provider_fingerprint:
        def release_with_fingerprint(provider, request_id):
            fingerprint = provider.executor.request_state_fingerprint(request_id)
            fingerprint["capture_index"] = len(provider_fingerprints)
            provider_fingerprints.append(fingerprint)
            return original_release_request(provider, request_id)

        Qwen35GGUFNextNDraftProvider.release_request = release_with_fingerprint

    os.environ["HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"] = str(args.budget)
    if not args.allow_fp16_state:
        os.environ["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] = "0"
    model = args.model.resolve()
    served_name = "qwen38-specdec2-s3"
    config = ServerConfig(
        model=str(model),
        served_model_name=served_name,
        eager_load=True,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
        speculative_mtp_serving="auto",
        execution_profile=args.execution_profile,
        max_active_requests=int(args.concurrency),
        generation_batch_window_ms=0.0,
    )
    started = time.perf_counter()
    app = create_app(config)
    try:
        with TestClient(app) as client:
            prompts = (args.prompt,) * int(args.concurrency)
            base = {
                "model": served_name,
                "prompt": args.prompt if args.concurrency == 1 else list(prompts),
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            ar_started = time.perf_counter()
            ar_response = client.post("/v1/completions", json=base)
            ar_wall = time.perf_counter() - ar_started
            ar_response.raise_for_status()
            staged_started = time.perf_counter()
            staged_response = client.post(
                "/v1/completions", json={**base, "speculative_mtp": True}
            )
            staged_wall = time.perf_counter() - staged_started
            staged_response.raise_for_status()
            warm_started = time.perf_counter()
            warm_response = client.post(
                "/v1/completions", json={**base, "speculative_mtp": True}
            )
            warm_wall = time.perf_counter() - warm_started
            warm_response.raise_for_status()
            llm = app.state.hipengine_llm
            adapter = llm._text_generator
            resident_snapshot = adapter.live_loop_snapshot()
            direct_generator = adapter.inner
            direct_config, _block_id, _required = _gguf_mtp_required_tensor_names(
                direct_generator.weight_index
            )
            direct_request = GenerationRequest(
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=False,
            )
            direct_started = time.perf_counter()
            direct_outputs = direct_generator._generate_dense_speculative_mtp_detailed(
                direct_request,
                config=direct_config,
            )
            direct_wall = time.perf_counter() - direct_started
    finally:
        Qwen35GGUFNextNDraftProvider.release_request = original_release_request

    ar = ar_response.json()
    staged = staged_response.json()
    warm = warm_response.json()
    direct_ids = tuple(
        tuple(int(token) for token in output.generated_token_ids or ())
        for output in direct_outputs
    )
    ar_ids = _choice_rows(ar)
    staged_ids = _choice_rows(staged)
    warm_ids = _choice_rows(warm)
    recent = resident_snapshot.get("runner", {}).get("routes", {}).get(
        "recent_completed", []
    )
    staged_rows = [row for row in recent if row.get("specdec2_mtp2_used")]
    normalized_provider_fingerprints = [
        {
            key: value
            for key, value in fingerprint.items()
            if key not in {"request_id", "slot", "capture_index"}
        }
        for fingerprint in provider_fingerprints
    ]
    staged_capture_count = 2 * int(args.concurrency)
    staged_provider_fingerprints_equal = bool(
        len(normalized_provider_fingerprints) >= staged_capture_count
        and len(
            {
                json.dumps(fingerprint, sort_keys=True)
                for fingerprint in normalized_provider_fingerprints[
                    :staged_capture_count
                ]
            }
        )
        == 1
    )
    provider_gate_passed = bool(
        not args.provider_fingerprint
        or (
            staged_provider_fingerprints_equal
            and normalized_provider_fingerprints[0]["visible_kv_bytes"] > 0
        )
    )
    passed = bool(
        ar_ids == staged_ids == warm_ids == direct_ids
        and len(staged_rows) >= 2 * int(args.concurrency)
        and all(
            int(row["specdec2_mtp2_candidate_counts"][0]) == int(args.budget)
            for row in staged_rows[-int(args.concurrency) :]
        )
        and (
            int(args.concurrency) == 1
            or all(
                int(row["specdec2_mtp2_proposal_batch_calls"]) > 0
                and int(row["specdec2_mtp2_target_batch_calls"]) > 0
                and int(row["specdec2_mtp2_candidate_device_handoffs"]) > 0
                and int(row["specdec2_mtp2_candidate_d2h_after_target"])
                == int(row["specdec2_mtp2_candidate_device_handoffs"])
                and int(row["specdec2_mtp2_device_accept_calls"])
                == int(row["specdec2_mtp2_target_batch_calls"])
                and int(row["specdec2_mtp2_selected_commit_batch_calls"])
                == int(row["specdec2_mtp2_target_batch_calls"])
                for row in staged_rows[-int(args.concurrency) :]
            )
        )
        and resident_snapshot.get("loop", {}).get("requests", {}).get("active") == 0
        and provider_gate_passed
    )
    payload = {
        "schema": 1,
        "kind": f"specdec2_gfx1151_c{int(args.concurrency)}_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "command": [sys.executable, *sys.argv],
        "repo": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(
                _git(["status", "--porcelain", "--untracked-files=no"])
            ),
            "shared_untracked_files_excluded": True,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "backend": os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151"),
            "gpu": "AMD Radeon 8060S Graphics",
        },
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "quant": "Q4_K_S",
            "kv": "bf16",
            "recurrent_state": (
                "fp16" if args.allow_fp16_state else "fp32_strict"
            ),
            "execution_profile": args.execution_profile,
            "execution_profile_manifest_sha256": getattr(
                llm, "execution_profile_manifest_sha256", None
            ),
            "execution_profile_strict_manifest_sha256": getattr(
                llm, "execution_profile_strict_manifest_sha256", None
            ),
        },
        "workload": {
            "prompt": args.prompt,
            "concurrency": int(args.concurrency),
            "max_tokens": int(args.max_tokens),
            "candidate_budget": int(args.budget),
            "sampling": "greedy",
        },
        "ids": {
            "ar": list(ar_ids[0]) if args.concurrency == 1 else [list(row) for row in ar_ids],
            "staged_cold": list(staged_ids[0]) if args.concurrency == 1 else [list(row) for row in staged_ids],
            "staged_warm": list(warm_ids[0]) if args.concurrency == 1 else [list(row) for row in warm_ids],
            "direct_exact": list(direct_ids[0]) if args.concurrency == 1 else [list(row) for row in direct_ids],
            "all_equal": ar_ids == staged_ids == warm_ids == direct_ids,
        },
        "wall_seconds": {
            "ar": ar_wall,
            "staged_cold": staged_wall,
            "staged_warm": warm_wall,
            "direct_exact": direct_wall,
            "warm_staged_over_direct": (
                warm_wall / direct_wall if direct_wall > 0.0 else None
            ),
        },
        "staged_rows": staged_rows,
        "provider_fingerprints": provider_fingerprints,
        "provider_fingerprint_gate": {
            "enabled": bool(args.provider_fingerprint),
            "captures": len(provider_fingerprints),
            "staged_repeat_equal": staged_provider_fingerprints_equal,
            "direct_equal_to_staged": bool(
                len(normalized_provider_fingerprints) >= 3 * int(args.concurrency)
                and normalized_provider_fingerprints[staged_capture_count]
                == normalized_provider_fingerprints[staged_capture_count - 1]
            ),
            "passed": provider_gate_passed,
        },
        "resident_snapshot": resident_snapshot,
        "total_wall_seconds": time.perf_counter() - started,
        "passed": passed,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
    )
    parser.add_argument("--budget", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--prompt", default="Write one short greeting.")
    parser.add_argument("--allow-fp16-state", action="store_true")
    parser.add_argument(
        "--execution-profile",
        choices=("strict", "production", "batch_invariant"),
        default="strict",
    )
    parser.add_argument("--provider-fingerprint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "budget": args.budget,
                "concurrency": args.concurrency,
                "passed": payload["passed"],
                "wall_seconds": payload["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
