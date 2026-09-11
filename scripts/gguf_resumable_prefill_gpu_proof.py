#!/usr/bin/env python3
"""GPU proof that the resumable layer-outer INT8 prefill really yields.

Roadmap P6 / reviewer item 2 (P6e). P6a made the compact INT8 prefill return
control to the service driver between bounded layer segments, P6b gave the
suspended request its own state so interleaved packed decode cannot overwrite it,
and P6c pinned the checkpoint contract. **Every one of those tests is a CPU test
with a monkeypatched recorder.** Nothing had verified on hardware that a segment
does real GPU work, that a suspended prefill survives a real interleaved decode,
or that the interleaved decode gap stays bounded.

This harness proves those three things in one process, without the HTTP layer:

* **Arm A (one-shot)** prefills a multi-round prompt through the one-shot packed
  entry and records the sampled token.
* **Arm B (resumable)** prefills the same prompt through
  ``prefill_batch_native_layer_outer_resumable`` in bounded layer segments. A
  third session in the same resident batch runs one decode step between every
  pair of segments.

Gates, all of which must pass for the artifact to report a pass:

1. **Exact continuation** - Arm B's sampled token equals Arm A's.
2. **Real yield** - the resumable arm needs more than one segment, and every
   segment does measurable GPU work rather than advancing bookkeeping.
3. **Bounded decode gap** - the decode steps taken while a prefill is suspended
   stay within a declared factor of the standalone decode step on the same
   session, at both p95 and max.
4. **Cleanup** - the suspended buffers are released once the prefill completes.

The route is eager, greedy, prefix-off, MTP-off, and artifact-scoped. Nothing
here is a throughput claim: it is a control-and-liveness proof, and the artifact
sets ``performance_claim: false``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core import build as _hipengine_build  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_for_backend  # noqa: E402

ARTIFACT_KIND = "w7900_p6e_resumable_prefill_gpu_proof"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_COMPILER_VERSION_FILE = Path("/tmp/hipengine-hipcc-version.txt")

REQUIRE_CACHED_BUILD_ENV = _hipengine_build._ENV_REQUIRE_CACHED_BUILD
COMPILER_VERSION_FILE_ENV = "HIPENGINE_COMPILER_VERSION_FILE"

# Declared before measuring, so the gate cannot be fitted to the result.
DECLARED_DECODE_GAP_P95_FACTOR = 2.0
DECLARED_DECODE_GAP_MAX_FACTOR = 3.0
# A segment that advances bookkeeping without doing GPU work takes microseconds.
DECLARED_SEGMENT_WORK_FLOOR_MS = 1.0


def _prompt_tokens(rows: int) -> list[int]:
    """The parity manifest's deterministic generator (rng 20260909)."""

    import numpy as np

    rng = np.random.default_rng(20260909)
    return [int(t) for t in rng.integers(1000, 4096, size=int(rows))]


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    index = max(0, int(round(0.95 * len(ordered))) - 1)
    return {
        "count": len(ordered),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[index], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def _build_environment(compiler_version_file: Path, backend: str) -> dict[str, str]:
    compiler_version = compiler_version_file.read_text(encoding="utf-8").strip()
    if not compiler_version:
        raise ValueError(f"compiler version file is empty: {compiler_version_file}")
    environment = dict(os.environ)
    environment.pop("ROCR_VISIBLE_DEVICES", None)
    environment.setdefault("HIP_VISIBLE_DEVICES", "0")
    environment["HIPENGINE_HIP_ARCH"] = hip_target_arch_for_backend(backend)
    environment[COMPILER_VERSION_FILE_ENV] = str(compiler_version_file)
    environment[REQUIRE_CACHED_BUILD_ENV] = "1"
    return environment


def _session_kwargs(
    *,
    max_sequence_length: int,
    backend: str,
    runtime: Any,
    shared_runner: Any,
    prefill_config: Any,
    kv_policy: Any,
) -> dict[str, Any]:
    """Constructor kwargs for one session of the resident INT8-direct batch."""

    kwargs: dict[str, Any] = {
        "max_sequence_length": int(max_sequence_length),
        "backend": str(backend),
        "prefill_config": prefill_config,
        "kv_policy": kv_policy.create_policy(),
        "kv_scale_dtype": "fp32",
        "kv_scale_granularity": kv_policy.scale_granularity,
        "use_wmma_prefill": True,
        "use_gemv_decode": True,
    }
    if runtime is not None:
        kwargs["runtime"] = runtime
    if shared_runner is not None:
        kwargs["shared_runner"] = shared_runner
    return kwargs


def evaluate_gates(
    *,
    arm_a_token: int | None,
    arm_b_token: int | None,
    segments: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gap: Mapping[str, Any],
    scratch_released: bool,
    peak_suspended_bytes: int,
) -> dict[str, Any]:
    """Evaluate the four P6e gates from measurements.

    Split out of the run so the gates can be tested against synthetic
    measurements: a gate that cannot fail is not evidence, and the failure paths
    never occur on a healthy GPU run.
    """

    gates: dict[str, Any] = {}
    gates["exact_continuation"] = {
        "passed": arm_b_token is not None and arm_b_token == arm_a_token,
        "arm_a_token_id": arm_a_token,
        "arm_b_token_id": arm_b_token,
        "detail": (
            "both arms prefill the same prompt greedily and must sample the same token"
        ),
    }
    gates["real_yield"] = {
        "passed": int(segments.get("count", 0)) > 1
        and float(segments.get("min_ms", 0.0)) >= DECLARED_SEGMENT_WORK_FLOOR_MS,
        "segments": segments.get("count"),
        "segment_min_ms": segments.get("min_ms"),
        "declared_floor_ms": DECLARED_SEGMENT_WORK_FLOOR_MS,
        "detail": (
            "more than one segment proves control returned mid-prefill; every "
            "segment above the floor proves each segment did GPU work rather "
            "than advancing bookkeeping"
        ),
    }
    p95_limit = float(baseline.get("p95_ms", 0.0)) * DECLARED_DECODE_GAP_P95_FACTOR
    max_limit = float(baseline.get("max_ms", 0.0)) * DECLARED_DECODE_GAP_MAX_FACTOR
    gates["bounded_decode_gap"] = {
        "passed": bool(gap)
        and float(gap.get("p95_ms", float("inf"))) <= p95_limit
        and float(gap.get("max_ms", float("inf"))) <= max_limit,
        "interleaved_p95_ms": gap.get("p95_ms"),
        "interleaved_max_ms": gap.get("max_ms"),
        "p95_limit_ms": round(p95_limit, 3),
        "max_limit_ms": round(max_limit, 3),
        "declared_p95_factor": DECLARED_DECODE_GAP_P95_FACTOR,
        "declared_max_factor": DECLARED_DECODE_GAP_MAX_FACTOR,
        "detail": (
            "decode steps taken while a prefill is suspended must stay within the "
            "declared factors of the standalone decode step on the same session"
        ),
    }
    gates["cleanup"] = {
        "passed": bool(scratch_released),
        "peak_suspended_bytes": int(peak_suspended_bytes),
        "detail": (
            "the suspended-state buffers must not outlive the completing "
            "checkpoint; peak_suspended_bytes proves suspension actually happened"
        ),
    }
    return gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    backend = str(args.backend)
    environment = _build_environment(
        args.compiler_version_file.expanduser().resolve(), backend
    )
    # The child-free path: this harness is the process, so apply the cache-only
    # guard to the interpreter before any kernel build can happen.
    for key, value in environment.items():
        os.environ[key] = value

    policy = resolve_kv_policy("int8_per_token_head", scale_dtype="fp32")
    prefill_config = PrefillConfig()
    max_sequence_length = int(args.max_sequence_length)

    def new_session(**overrides: Any) -> Any:
        return Qwen35GGUFResidentSession(
            model,
            **_session_kwargs(
                max_sequence_length=max_sequence_length,
                backend=backend,
                prefill_config=prefill_config,
                kv_policy=policy,
                **overrides,
            ),
        )

    stack = ExitStack()
    try:
        owner = stack.enter_context(
            new_session(runtime=None, shared_runner=None)
        )
        if owner.runner is None:
            raise RuntimeError("resident session runner is closed")
        # One session per role: one-shot target, resumable target, interleaved
        # decoder. They share the runner, which is what makes them one resident
        # batch and what makes the suspended-state question real.
        one_shot = stack.enter_context(
            new_session(runtime=owner.runtime, shared_runner=owner.runner)
        )
        resumable = stack.enter_context(
            new_session(runtime=owner.runtime, shared_runner=owner.runner)
        )
        decoder = stack.enter_context(
            new_session(runtime=owner.runtime, shared_runner=owner.runner)
        )

        # Read the real row capacity instead of assuming one: the resumable entry
        # raises NotImplementedError on a single-round prompt, and the capacity is
        # derived from max_sequence_length and the dense row cap.
        owner._ensure_bulk_prefill_workspace()  # type: ignore[attr-defined]
        scratch = owner._bulk_prefill_scratch  # type: ignore[attr-defined]
        if scratch is None:
            raise RuntimeError("bulk prefill workspace was not created")
        row_capacity = int(scratch.rows)
        if row_capacity <= 0:
            raise RuntimeError(f"unusable bulk prefill row capacity: {row_capacity}")

        rounds = max(2, int(args.prompt_rounds))
        prompt_rows = row_capacity * rounds
        headroom = max_sequence_length - prompt_rows - int(args.decode_tokens) - 8
        if headroom < 0:
            raise ValueError(
                f"prompt of {prompt_rows} rows does not fit in a sequence of "
                f"{max_sequence_length}; lower --prompt-rounds or raise "
                "--max-sequence-length"
            )
        prompt = _prompt_tokens(prompt_rows)

        # --- decoder session gets its own short context to decode from --------
        decoder_prompt = _prompt_tokens(max(8, row_capacity // 8))
        decoder_prefill = decoder.prefill(
            decoder_prompt, use_bulk=True, return_logits=False
        )
        decoder_token = int(decoder_prefill.token_id)

        def standalone_decode_walls(count: int) -> tuple[list[float], int]:
            walls: list[float] = []
            token = decoder_token
            for _ in range(count):
                started = time.perf_counter()
                step = decoder.step(token, return_logits=False)
                walls.append((time.perf_counter() - started) * 1e3)
                token = int(step.token_id)
            return walls, token

        baseline_walls, decoder_token = standalone_decode_walls(int(args.decode_tokens))

        # --- Arm A: one-shot packed prefill ----------------------------------
        started = time.perf_counter()
        arm_a = one_shot.prefill_batch_native(
            [prompt], sessions=[one_shot], return_logits=False
        )[0]
        arm_a_wall = time.perf_counter() - started
        arm_a_token = int(arm_a.token_id)

        # --- Arm B: resumable segmented prefill with interleaved decode ------
        layer_budget = int(args.layer_budget)
        segments: list[float] = []
        interleaved: list[float] = []
        state = None
        last_state = None
        peak_suspended_bytes = 0
        arm_b_token: int | None = None
        arm_b_started = time.perf_counter()
        segment_index = 0
        while True:
            started = time.perf_counter()
            result = resumable.prefill_batch_native_layer_outer_resumable(
                [prompt] if state is None else None,
                sessions=[resumable] if state is None else None,
                state=state,
                layer_budget=layer_budget,
            )
            segments.append((time.perf_counter() - started) * 1e3)
            segment_index += 1
            if isinstance(result, list):
                arm_b_token = int(result[0].token_id) if result[0] is not None else None
                break
            state = result
            last_state = state
            plan = getattr(resumable, "last_packed_prefill_plan", None) or {}
            peak_suspended_bytes = max(
                peak_suspended_bytes, int(plan.get("layer_outer_suspended_bytes", 0) or 0)
            )
            if segment_index > 4096:
                raise RuntimeError("resumable prefill did not terminate")
            # Interleaved decode while the prefill is suspended.
            started = time.perf_counter()
            step = decoder.step(decoder_token, return_logits=False)
            interleaved.append((time.perf_counter() - started) * 1e3)
            decoder_token = int(step.token_id)
        arm_b_wall = time.perf_counter() - arm_b_started

        # The P6b contract, checked the same way the CPU test checks it: the
        # suspended buffers must not outlive the completing checkpoint.
        scratch_released = last_state is not None and last_state.scratch is None
    finally:
        stack.close()

    baseline = _percentiles(baseline_walls)
    gap = _percentiles(interleaved)
    segment_stats = _percentiles(segments)

    gates = evaluate_gates(
        arm_a_token=arm_a_token,
        arm_b_token=arm_b_token,
        segments=segment_stats,
        baseline=baseline,
        gap=gap,
        scratch_released=scratch_released,
        peak_suspended_bytes=peak_suspended_bytes,
    )

    return {
        "schema": 1,
        "kind": ARTIFACT_KIND,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "passed": all(gate["passed"] for gate in gates.values()),
        "host": {
            "target_arch": hip_target_arch_for_backend(backend),
            "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES", ""),
        },
        "model": {
            "path": str(model),
            "quant": "gguf_q4_k_m",
            "kv": "int8_per_token_head + fp32 scales",
            "route": "int8_direct layer-outer resumable",
        },
        "workload": {
            "prompt_rows": prompt_rows,
            "bulk_prefill_row_capacity": row_capacity,
            "planned_rounds": rounds,
            "layer_budget": layer_budget,
            "decode_tokens": int(args.decode_tokens),
            "max_sequence_length": max_sequence_length,
            "execution_mode": "eager, greedy, prefix-off, MTP-off",
        },
        "gates": gates,
        "measurements": {
            "arm_a_one_shot_wall_s": round(arm_a_wall, 4),
            "arm_b_resumable_wall_s": round(arm_b_wall, 4),
            "segments": segment_stats,
            "peak_suspended_bytes": peak_suspended_bytes,
            "standalone_decode": baseline,
            "interleaved_decode": gap,
            "interleaved_steps": len(interleaved),
        },
        "command": " ".join(
            shlex.quote(part)
            for part in [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "trace_environment": {
            "HIP_VISIBLE_DEVICES": environment.get("HIP_VISIBLE_DEVICES", ""),
            "HIPENGINE_HIP_ARCH": environment["HIPENGINE_HIP_ARCH"],
            COMPILER_VERSION_FILE_ENV: environment[COMPILER_VERSION_FILE_ENV],
            REQUIRE_CACHED_BUILD_ENV: environment[REQUIRE_CACHED_BUILD_ENV],
        },
        "notes": [
            "This is a control-and-liveness proof, not a throughput claim; it sets "
            "performance_claim false and no benchmark scoreboard row changes.",
            "The decode-gap factors and the segment work floor were declared before "
            "measuring, so the gate could not be fitted to the result.",
            "Arm A and Arm B prefill the same prompt on different sessions of one "
            "resident batch; equal sampled tokens is the continuation check.",
            "The interleaved decoder runs while the prefill is suspended, which is "
            "the case P6b's dedicated suspended buffers exist to protect.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument(
        "--prompt-rounds",
        type=int,
        default=3,
        help="prompt length as a multiple of the bulk prefill row capacity",
    )
    parser.add_argument("--layer-budget", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument(
        "--compiler-version-file", type=Path, default=DEFAULT_COMPILER_VERSION_FILE
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
