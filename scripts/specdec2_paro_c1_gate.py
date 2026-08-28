#!/usr/bin/env python3
"""gfx1100 packed-PARO SPECDEC2 C1/K1 staged-vs-AR gate."""

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
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _choice_ids(response: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(token)
        for token in response["choices"][0]["hipengine"]["generated_token_ids"]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from hipengine.server import ServerConfig, create_app

    model = args.model.resolve()
    served_name = "qwen36-paro-specdec2-s7"
    os.environ["HIPENGINE_VERIFY_GPU_ACCEPT"] = "validate"
    config = ServerConfig(
        model=str(model),
        served_model_name=served_name,
        eager_load=True,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
        speculative_mtp_serving="enabled",
        execution_profile=args.execution_profile,
        max_active_requests=1,
        generation_batch_window_ms=0.0,
    )
    started = time.perf_counter()
    app = create_app(config)
    with TestClient(app) as client:
        base = {
            "model": served_name,
            "prompt": args.prompt,
            "max_tokens": int(args.max_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
        }
        ar_started = time.perf_counter()
        ar_response = client.post("/v1/completions", json=base)
        ar_wall = time.perf_counter() - ar_started
        ar_response.raise_for_status()
        cold_started = time.perf_counter()
        cold_response = client.post(
            "/v1/completions", json={**base, "speculative_mtp": True}
        )
        cold_wall = time.perf_counter() - cold_started
        cold_response.raise_for_status()
        warm_started = time.perf_counter()
        warm_response = client.post(
            "/v1/completions", json={**base, "speculative_mtp": True}
        )
        warm_wall = time.perf_counter() - warm_started
        warm_response.raise_for_status()
        llm = app.state.hipengine_llm
        adapter = llm._text_generator
        snapshot = adapter.live_loop_snapshot()
        engine_capability = {
            "llm_supports_speculative_mtp": bool(llm.supports_speculative_mtp),
            "generator_type": type(adapter).__name__,
            "generator_supports_speculative_mtp": bool(
                getattr(adapter, "supports_speculative_mtp", False)
            ),
            "generator_supports_staged_mtp": bool(
                getattr(adapter, "_supports_staged_speculative_mtp", False)
            ),
        }

    ar = ar_response.json()
    cold = cold_response.json()
    warm = warm_response.json()
    ar_ids = _choice_ids(ar)
    cold_ids = _choice_ids(cold)
    warm_ids = _choice_ids(warm)
    recent = snapshot.get("runner", {}).get("routes", {}).get(
        "recent_completed", []
    )
    staged_rows = [row for row in recent if row.get("specdec2_mtp2_used")]
    passed = bool(
        ar_ids == cold_ids == warm_ids
        and len(staged_rows) >= 2
        and all(
            list(row.get("specdec2_mtp2_candidate_counts", []))
            and set(row["specdec2_mtp2_candidate_counts"]) == {1}
            and set(row.get("specdec2_mtp2_execution_routes", [])) == {"eager"}
            for row in staged_rows[-2:]
        )
        and snapshot.get("loop", {}).get("requests", {}).get("active") == 0
    )
    return {
        "schema": 1,
        "kind": "specdec2_gfx1100_paro_c1_k1_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "command": [sys.executable, *sys.argv],
        "repo": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "backend": "hip_gfx1100",
            "gpu": "AMD Radeon Pro W7900",
        },
        "model": {
            "path": str(model),
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
            "max_tokens": int(args.max_tokens),
            "candidate_budget": 1,
            "sampling": "greedy",
        },
        "ids": {
            "ar": list(ar_ids),
            "staged_cold": list(cold_ids),
            "staged_warm": list(warm_ids),
            "all_equal": ar_ids == cold_ids == warm_ids,
        },
        "wall_seconds": {
            "ar": ar_wall,
            "staged_cold": cold_wall,
            "staged_warm": warm_wall,
        },
        "staged_rows": staged_rows,
        "engine_capability": engine_capability,
        "api_route": {
            "cold": cold.get("hipengine", {}).get("speculative_mtp"),
            "warm": warm.get("hipengine", {}).get("speculative_mtp"),
        },
        "resident_snapshot": snapshot,
        "total_wall_seconds": time.perf_counter() - started,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Write one short greeting.")
    parser.add_argument(
        "--execution-profile", choices=("strict", "production"), default="production"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "profile": args.execution_profile,
                "wall_seconds": payload["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
