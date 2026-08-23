#!/usr/bin/env python3
"""Run one marked steady decode transition through the production GGUF owner.

This child intentionally uses ``LLM.generate_detailed`` so work flows through
``EngineService`` and ``Qwen35GGUFResidentModelRunner._step_native_rows``.  It is
used after an unprofiled cache warmup; the profiled invocation marks one later
physical-cN transition after graph capture/replay has already started.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine import LLM, SamplingParams  # noqa: E402
from scripts.gguf_packed_ar_rocprof import _Roctx  # noqa: E402

MARKER_PREFIX = "hipengine_c2_production_owner_c"


def marker_name(concurrency: int, index: int) -> str:
    return f"{MARKER_PREFIX}{int(concurrency)}_decode_transition_{int(index)}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    if args.concurrency <= 0 or args.decode_tokens <= args.marker_index:
        raise ValueError("decode-tokens must exceed the positive marker-index")

    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1100")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file.expanduser().resolve()
        )

    params = SamplingParams(max_tokens=args.decode_tokens, ignore_eos=True)
    llm = LLM(
        str(model),
        backend=args.backend,
        quant=args.quant,
        max_active_requests=args.concurrency,
        max_sequence_length=args.prompt_length + args.decode_tokens + 8,
    )
    adapter = llm._get_text_generator()
    llm.prepare(
        max_sequence_length=args.prompt_length + args.decode_tokens + 8,
        sampling_params=params,
    )
    runner = adapter._runner
    runtime = runner._shared_runner.runtime
    roctx = _Roctx() if args.profile else None
    original = runner._step_native_rows
    observed_c = 0
    marked = False
    marked_wall_ms: float | None = None
    marked_plan: dict[str, Any] | None = None
    marked_manifest: dict[str, Any] | None = None

    def wrapped(self, rows, *, work=None):
        nonlocal observed_c, marked, marked_wall_ms, marked_plan, marked_manifest
        should_count = len(rows) == int(args.concurrency)
        if should_count:
            observed_c += 1
        if should_count and observed_c == int(args.marker_index):
            runtime.device_synchronize()
            if roctx is not None:
                roctx.push(marker_name(args.concurrency, args.marker_index))
            start = time.perf_counter()
            try:
                result = original(rows, work=work)
                runtime.device_synchronize()
            finally:
                marked_wall_ms = (time.perf_counter() - start) * 1000.0
                if roctx is not None:
                    roctx.pop()
            marked = True
            marked_plan = copy.deepcopy(self._last_physical_group_plan)
            marked_manifest = copy.deepcopy(self._last_execution_manifest)
            return result
        return original(rows, work=work)

    runner._step_native_rows = MethodType(wrapped, runner)
    prompt = tuple([int(args.prompt_token_id)] * int(args.prompt_length))
    prompts = tuple(prompt for _ in range(int(args.concurrency)))
    try:
        outputs = llm.generate_detailed(prompts, params)
        runtime.device_synchronize()
        snapshot = adapter.live_loop_snapshot()
    finally:
        runner._step_native_rows = original
        llm.close()

    generated = [list(output.generated_token_ids or ()) for output in outputs]
    if len(generated) != args.concurrency:
        raise RuntimeError("production owner returned the wrong output count")
    if any(len(tokens) != args.decode_tokens for tokens in generated):
        raise RuntimeError("production owner returned the wrong token count")
    if not marked:
        raise RuntimeError(
            f"physical c{args.concurrency} transition {args.marker_index} was not observed; "
            f"observed {observed_c}"
        )
    routes = snapshot["runner"]["routes"]
    loop = snapshot["loop"]
    return {
        "schema": 1,
        "kind": "gguf_c2_production_owner_profile_child",
        "profile": bool(args.profile),
        "model": str(model),
        "backend": args.backend,
        "quant": args.quant,
        "concurrency": args.concurrency,
        "prompt_length": args.prompt_length,
        "decode_tokens": args.decode_tokens,
        "marker_name": marker_name(args.concurrency, args.marker_index),
        "marked_wall_ms": marked_wall_ms,
        "observed_physical_c_transitions": observed_c,
        "generated_token_ids": generated,
        "all_rows_equal": all(tokens == generated[0] for tokens in generated),
        "marked_physical_group_plan": marked_plan,
        "marked_execution_manifest": marked_manifest,
        "route_counts": routes["counts"],
        "fallback_reasons": routes["fallback_reasons"],
        "final_requests": loop["requests"],
        "final_physical_bucket": loop["physical_bucket"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--marker-index", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
