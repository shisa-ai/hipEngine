#!/usr/bin/env python3
"""Compare PARO prefill state with AOTriton on stream 0 versus an isolated stream.

This is a differential correctness gate, not a throughput benchmark. It runs
the same prompt and production math in separate resident sessions, changing
only ``HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM``. The final hidden
row, sampled seed, every linear Conv/GDN state, and every live full-attention
K/V prefix must be byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_shrinking_correctness import _device_sha256, _slot_state_snapshot, _state_snapshot_mismatches
from scripts.qwen35_paro_bench import _prompt_tokens, _read_compiler_version

DEFAULT_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/"
    "models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/"
    "snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1"
)
ISOLATION_ENV = "HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM"


@contextmanager
def _isolation_mode(enabled: bool) -> Iterator[None]:
    previous = os.environ.get(ISOLATION_ENV)
    os.environ[ISOLATION_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ISOLATION_ENV, None)
        else:
            os.environ[ISOLATION_ENV] = previous


def comparison_mismatches(isolated: dict[str, Any], default_stream: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if isolated.get("seed_token_id") != default_stream.get("seed_token_id"):
        mismatches.append("seed_token_id")
    if isolated.get("final_hidden_sha256") != default_stream.get("final_hidden_sha256"):
        mismatches.append("final_hidden_sha256")
    mismatches.extend(
        f"state.{path}"
        for path in _state_snapshot_mismatches(
            isolated.get("state", {}),
            default_stream.get("state", {}),
        )
    )
    return mismatches


def _run_mode(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: Sequence[int],
    *,
    isolated: bool,
    compiler_version: str | None,
    require_cached_build: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    with _isolation_mode(isolated):
        with Qwen35ParoResidentSession(
            runner,
            max_sequence_length=len(prompt_tokens) + 1,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
        ) as session:
            loaded = time.perf_counter()
            session.reset()
            result = session.prefill_native(prompt_tokens, sample=True)
            if result is None:
                raise RuntimeError("sampled exactness pass returned no seed token")
            session.runtime.device_synchronize()
            completed = time.perf_counter()
            final_hidden_sha256 = _device_sha256(session, session.hidden.ptr, session.hidden_nbytes)
            state = _slot_state_snapshot(session, slot=0, live_count=len(prompt_tokens))
            execution = dict(session.last_prefill_execution)
    return {
        "isolated_stream": bool(isolated),
        "seed_token_id": int(result.token_id),
        "final_hidden_sha256": final_hidden_sha256,
        "state": state,
        "execution": execution,
        "diagnostic_seconds": {
            "session_load": float(loaded - started),
            "prefill_and_sample": float(completed - loaded),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1151")
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")

    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None
    runner = Qwen35ParoNextTokenRunner(
        args.model,
        shared_expert_format="packed_paro_w4",
        backend=args.backend,
    )
    prompt_tokens = _prompt_tokens(args.model, "Hello", args.token_id, args.prompt_length)
    isolated = _run_mode(
        runner,
        prompt_tokens,
        isolated=True,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    default_stream = _run_mode(
        runner,
        prompt_tokens,
        isolated=False,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    mismatches = comparison_mismatches(isolated, default_stream)
    payload = {
        "schema": 1,
        "status": "accepted" if not mismatches else "rejected_correctness",
        "performance_claim": False,
        "correctness_claim": not mismatches,
        "mode": "paro_prefill_aotriton_stream_exactness",
        "model": str(args.model.resolve()),
        "quant": "w4_paro",
        "workload": {
            "prompt_length": int(args.prompt_length),
            "token_id": int(args.token_id),
            "attn_aotriton_min_tokens": 512,
        },
        "isolated": isolated,
        "default_stream": default_stream,
        "comparison": {
            "passed": not mismatches,
            "mismatches": mismatches,
            "isolated_state_aggregate_sha256": isolated["state"]["aggregate_sha256"],
            "default_state_aggregate_sha256": default_stream["state"]["aggregate_sha256"],
        },
        "provenance": collect_artifact_provenance(
            repo_root=REPO_ROOT,
            configured_backend=args.backend,
            resolved_backend=runner.backend,
            target_arch=runner.target_arch,
            model_path=args.model,
            quant="w4_paro",
            kv_dtype="bf16",
            command=["python3", "scripts/paro_prefill_aotriton_stream_exactness.py", *(sys.argv[1:] if argv is None else argv)],
            environment={
                "comparison_env": ISOLATION_ENV,
                "tuned_profile": "accelerator-performance",
            },
            build_profile="production PARO prefill differential state gate",
            timing_protocol="correctness only; isolated first, default-stream control second; timings are diagnostic",
            warmups=0,
            repetitions=1,
            profiler={"enabled": False, "reason": "byte-state differential gate"},
            hipcc_version=compiler_version,
        ),
        "notes": [
            "Both legs use identical model math, prompt, chunks, AOTriton kernels, and sampling; only the AOTriton HIP stream changes.",
            "The gate compares final hidden bytes, sampled seed, 30 Conv/GDN state families, and 10 live K/V families.",
            "Diagnostic wall times include no balanced performance protocol and must not be used as throughput evidence.",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_content_sha256_without_self"] = hashlib.sha256(canonical).hexdigest()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
