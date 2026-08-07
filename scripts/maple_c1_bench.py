#!/usr/bin/env python3
"""Qualify Maple exact c1 decode with paired category/heldout A/B timing.

The router comparison uses selector-unset production versus the exact
``HIPENGINE_MAPLE_ROUTER_SINGLE_DISPATCH=0`` rollback. The affine4 comparison
uses the default-off exact wave32 candidate versus the production group64 head.
Two resident runners
start from byte-identical native-prefill state, advance in lockstep, and
alternate execution order. The artifact records exact token/top-logit parity,
final hidden/logit/live-KV/span hashes, router-counter reset, paired timing, and
tracked lifecycle over the complete natural plus category-heldout prompt suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import (  # noqa: E402
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
    memory_stats,
    reset_memory_stats,
)
from hipengine.loading.maple import load_maple_checkpoint  # noqa: E402
from hipengine.runtime.maple import PREFILL_CHUNK, MapleRunner  # noqa: E402
from hipengine.tokenization.maple import MapleTokenizer  # noqa: E402

DEFAULT_MODEL = "deepgrove/maple-preview-2bit-mlx"
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_HELDOUT = REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
PINNED_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"
ROUTER_SELECTOR = "HIPENGINE_MAPLE_ROUTER_SINGLE_DISPATCH"
AFFINE4_SELECTOR = "HIPENGINE_MAPLE_AFFINE4_WAVE32_EXACT"
COMPARISONS = ("router", "affine4_wave32")
PRODUCTION_SELECTORS = (
    ROUTER_SELECTOR,
    AFFINE4_SELECTOR,
    "HIPENGINE_MAPLE_GRAPH",
    "HIPENGINE_MAPLE_FUSE_MOE",
    "HIPENGINE_MAPLE_FUSE_QKATTN",
    "HIPENGINE_MAPLE_PREFILL_GROUPED_MOE",
    "HIPENGINE_MAPLE_PREFILL_GQA4",
)


def _capture(command: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": shlex.join(command),
        "returncode": int(completed.returncode),
        "output": (completed.stdout + completed.stderr).strip(),
    }


def _load_prompts(path: Path, *, heldout: bool) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        users = [
            message["content"]
            for message in row.get("messages", [])
            if message.get("role") == "user"
        ]
        if len(users) != 1:
            raise ValueError(f"{path}: prompt {row.get('id')!r} must have one user message")
        category = str(row["category"])
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"{path}: unsupported category {category!r}")
        prompts.append(
            {
                "id": str(row["id"]),
                "category": category,
                "text": str(users[0]),
                "heldout": bool(heldout),
            }
        )
    return prompts


def _tokenizer(checkpoint) -> MapleTokenizer:
    spec = checkpoint.spec
    return MapleTokenizer.from_model_path(
        checkpoint.index.model_path,
        model_vocab_size=spec.vocab_size,
        eos_token_id=spec.eos_token_id,
        bos_token_id=spec.bos_token_id,
    )


@contextmanager
def _production_environment() -> Iterator[None]:
    """Force selector-unset production defaults and restore the caller state."""

    saved = {name: os.environ.get(name) for name in PRODUCTION_SELECTORS}
    try:
        for name in PRODUCTION_SELECTORS:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _set_comparison_mode(mode: str, comparison: str) -> None:
    if mode not in ("candidate", "control"):
        raise ValueError(f"unsupported comparison mode {mode!r}")
    if comparison == "router":
        os.environ.pop(AFFINE4_SELECTOR, None)
        if mode == "candidate":
            os.environ.pop(ROUTER_SELECTOR, None)
        else:
            os.environ[ROUTER_SELECTOR] = "0"
        return
    if comparison == "affine4_wave32":
        os.environ.pop(ROUTER_SELECTOR, None)
        if mode == "candidate":
            os.environ[AFFINE4_SELECTOR] = "1"
        else:
            os.environ.pop(AFFINE4_SELECTOR, None)
        return
    raise ValueError(f"unsupported comparison {comparison!r}")


def _copy_device_bytes(
    runner: MapleRunner,
    source: DeviceBuffer,
    *,
    offset: int = 0,
    nbytes: int | None = None,
) -> np.ndarray:
    size = source.nbytes - int(offset) if nbytes is None else int(nbytes)
    if offset < 0 or size < 0 or offset + size > source.nbytes:
        raise ValueError("Maple state copy exceeds device buffer bounds")
    host = np.empty(size, dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(ptr=source.ptr + int(offset), nbytes=size),
        nbytes=size,
        runtime=runner.runtime,
    )
    return host


def _router_counter(runner: MapleRunner) -> int:
    host = np.empty(1, dtype=np.uint32)
    copy_device_to_host(
        host_array_ptr(host),
        runner.buffers.router_counter,
        runtime=runner.runtime,
    )
    return int(host[0])


def _state_gate(
    candidate: MapleRunner,
    control: MapleRunner,
    *,
    phase: str,
) -> dict[str, Any]:
    """Hash meaningful c1 continuation state without reading stale KV capacity."""

    if candidate.checkpoint.spec != control.checkpoint.spec:
        raise ValueError("Maple state comparison requires identical model specs")
    if candidate.position != control.position:
        return {
            "passed": False,
            "component_count": 0,
            "mismatches": ["position"],
            "candidate_sha256": None,
            "control_sha256": None,
        }
    spec = candidate.checkpoint.spec
    candidate_digest = hashlib.sha256()
    control_digest = hashlib.sha256()
    mismatches: list[str] = []
    components = 0

    def compare(
        name: str,
        candidate_buffer: DeviceBuffer,
        control_buffer: DeviceBuffer,
        *,
        candidate_offset: int = 0,
        control_offset: int = 0,
        nbytes: int | None = None,
    ) -> None:
        nonlocal components
        candidate_bytes = _copy_device_bytes(
            candidate,
            candidate_buffer,
            offset=candidate_offset,
            nbytes=nbytes,
        )
        control_bytes = _copy_device_bytes(
            control,
            control_buffer,
            offset=control_offset,
            nbytes=nbytes,
        )
        label = name.encode("utf-8") + b"\0"
        candidate_digest.update(label + candidate_bytes.tobytes())
        control_digest.update(label + control_bytes.tobytes())
        components += 1
        if not np.array_equal(candidate_bytes, control_bytes):
            mismatches.append(name)

    hidden_bytes = spec.hidden_size * np.dtype(np.uint16).itemsize
    if phase == "native_prefill":
        final_row = (candidate.position - 1) % PREFILL_CHUNK
        compare(
            "final_hidden",
            candidate.buffers.pf.hidden,
            control.buffers.pf.hidden,
            candidate_offset=final_row * hidden_bytes,
            control_offset=final_row * hidden_bytes,
            nbytes=hidden_bytes,
        )
    elif phase == "decode":
        compare(
            "final_hidden",
            candidate.buffers.hidden,
            control.buffers.hidden,
            nbytes=hidden_bytes,
        )
        for name in (
            "residual",
            "selected_ids",
            "routing_weights",
            "router_logits",
            "router_counter",
        ):
            compare(name, getattr(candidate.buffers, name), getattr(control.buffers, name))
    else:
        raise ValueError(f"unsupported state phase {phase!r}")

    for name in ("normalized", "logits", "argmax_index", "argmax_value"):
        compare(name, getattr(candidate.buffers, name), getattr(control.buffers, name))

    position = int(candidate.position)
    for layer_id, (candidate_layer, control_layer) in enumerate(
        zip(candidate.buffers.layers, control.buffers.layers, strict=True)
    ):
        live_rows = min(position, int(candidate_layer.spans.max_live_count))
        live_bytes = live_rows * spec.kv_size * np.dtype(np.uint16).itemsize
        compare(
            f"layer_{layer_id}.key_cache",
            candidate_layer.key_cache,
            control_layer.key_cache,
            nbytes=live_bytes,
        )
        compare(
            f"layer_{layer_id}.value_cache",
            candidate_layer.value_cache,
            control_layer.value_cache,
            nbytes=live_bytes,
        )

    for owner_name in ("sliding_span_owner", "global_span_owner"):
        candidate_owner = getattr(candidate.buffers, owner_name)
        control_owner = getattr(control.buffers, owner_name)
        for field in (
            "base_offsets",
            "live_counts",
            "token_positions",
            "evict_mask",
            "row_positions",
        ):
            compare(
                f"{owner_name}.{field}",
                getattr(candidate_owner, field),
                getattr(control_owner, field),
            )

    return {
        "passed": not mismatches,
        "component_count": components,
        "mismatches": mismatches,
        "candidate_sha256": candidate_digest.hexdigest(),
        "control_sha256": control_digest.hexdigest(),
    }


def _float32_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def _digest_ints(values: list[int], *, dtype: np.dtype[Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def _summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "aggregate_tokens_per_second": len(values) * 1_000.0 / sum(values),
    }


def _run_prompt_pair(
    candidate: MapleRunner,
    control: MapleRunner,
    *,
    prompt: dict[str, Any],
    tokens: tuple[int, ...],
    warmup_steps: int,
    measured_steps: int,
    repetitions: int,
    prompt_index: int,
    comparison: str,
) -> dict[str, Any]:
    repetitions_out: list[dict[str, Any]] = []
    prompt_passed = True
    for repetition in range(repetitions):
        candidate.reset()
        control.reset()
        prefill_results: dict[str, Any] = {}
        prefill_order = (
            ("candidate", "control")
            if (prompt_index + repetition) % 2 == 0
            else ("control", "candidate")
        )
        runners = {"candidate": candidate, "control": control}
        for mode in prefill_order:
            _set_comparison_mode(mode, comparison)
            prefill_results[mode] = runners[mode].prefill_native(tokens)
        prefill_state = _state_gate(candidate, control, phase="native_prefill")
        prefill_equal = (
            prefill_results["candidate"].token_id == prefill_results["control"].token_id
            and _float32_bits(prefill_results["candidate"].top_logit)
            == _float32_bits(prefill_results["control"].top_logit)
        )

        next_tokens = {
            mode: int(prefill_results[mode].token_id)
            for mode in ("candidate", "control")
        }
        measured_candidate_ms: list[float] = []
        measured_control_ms: list[float] = []
        paired_deltas_ms: list[float] = []
        candidate_tokens: list[int] = []
        control_tokens: list[int] = []
        candidate_top_bits: list[int] = []
        control_top_bits: list[int] = []
        token_matches = 0
        top_logit_matches = 0
        counter_checks = 0
        counter_violations: list[dict[str, int | str]] = []
        order_counts: Counter[str] = Counter()

        total_steps = warmup_steps + measured_steps
        for step_index in range(total_steps):
            order = (
                ("candidate", "control")
                if (prompt_index + repetition + step_index) % 2 == 0
                else ("control", "candidate")
            )
            order_counts["candidate_first" if order[0] == "candidate" else "control_first"] += 1
            results: dict[str, Any] = {}
            elapsed_ms: dict[str, float] = {}
            for mode in order:
                _set_comparison_mode(mode, comparison)
                started = time.perf_counter()
                results[mode] = runners[mode].step(next_tokens[mode])
                elapsed_ms[mode] = (time.perf_counter() - started) * 1_000.0
                counter = _router_counter(runners[mode])
                counter_checks += 1
                if counter != 0:
                    counter_violations.append(
                        {"mode": mode, "step": step_index, "value": counter}
                    )

            candidate_token = int(results["candidate"].token_id)
            control_token = int(results["control"].token_id)
            candidate_bits = _float32_bits(results["candidate"].top_logit)
            control_bits = _float32_bits(results["control"].top_logit)
            candidate_tokens.append(candidate_token)
            control_tokens.append(control_token)
            candidate_top_bits.append(candidate_bits)
            control_top_bits.append(control_bits)
            token_matches += candidate_token == control_token
            top_logit_matches += candidate_bits == control_bits
            next_tokens = {"candidate": candidate_token, "control": control_token}
            if step_index >= warmup_steps:
                measured_candidate_ms.append(elapsed_ms["candidate"])
                measured_control_ms.append(elapsed_ms["control"])
                paired_deltas_ms.append(elapsed_ms["control"] - elapsed_ms["candidate"])

        final_state = _state_gate(candidate, control, phase="decode")
        repetition_passed = (
            prefill_equal
            and prefill_state["passed"]
            and token_matches == total_steps
            and top_logit_matches == total_steps
            and not counter_violations
            and final_state["passed"]
        )
        prompt_passed = prompt_passed and repetition_passed
        repetitions_out.append(
            {
                "repetition": repetition,
                "prefill_order": list(prefill_order),
                "prefill_token_equal": prefill_results["candidate"].token_id
                == prefill_results["control"].token_id,
                "prefill_top_logit_equal": _float32_bits(
                    prefill_results["candidate"].top_logit
                )
                == _float32_bits(prefill_results["control"].top_logit),
                "prefill_state": prefill_state,
                "decode_positions": total_steps,
                "token_matches": token_matches,
                "top_logit_matches": top_logit_matches,
                "candidate_tokens_sha256": _digest_ints(candidate_tokens, dtype=np.dtype(np.int64)),
                "control_tokens_sha256": _digest_ints(control_tokens, dtype=np.dtype(np.int64)),
                "candidate_top_logits_sha256": _digest_ints(
                    candidate_top_bits, dtype=np.dtype(np.uint32)
                ),
                "control_top_logits_sha256": _digest_ints(
                    control_top_bits, dtype=np.dtype(np.uint32)
                ),
                "counter_checks": counter_checks,
                "counter_violations": counter_violations,
                "order_counts": dict(sorted(order_counts.items())),
                "candidate_timing": _summarize_ms(measured_candidate_ms),
                "control_timing": _summarize_ms(measured_control_ms),
                "candidate_step_ms": measured_candidate_ms,
                "control_step_ms": measured_control_ms,
                "paired_median_saving_ms": statistics.median(paired_deltas_ms),
                "paired_mean_saving_ms": statistics.fmean(paired_deltas_ms),
                "candidate_wins": sum(delta > 0.0 for delta in paired_deltas_ms),
                "measured_pairs": len(paired_deltas_ms),
                "paired_savings_ms": paired_deltas_ms,
                "final_state": final_state,
                "passed": repetition_passed,
            }
        )
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "heldout": prompt["heldout"],
        "prompt_tokens": len(tokens),
        "repetitions": repetitions_out,
        "passed": prompt_passed,
    }


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_ms: list[float] = []
    control_ms: list[float] = []
    paired_ms: list[float] = []
    order_counts: Counter[str] = Counter()
    category_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"candidate": [], "control": [], "paired": []}
    )
    position_count = 0
    token_matches = 0
    top_logit_matches = 0
    counter_checks = 0
    counter_violations = 0
    prefill_state_matches = 0
    final_state_matches = 0
    repetition_count = 0
    for row in rows:
        for repetition in row["repetitions"]:
            repetition_count += 1
            position_count += repetition["decode_positions"]
            token_matches += repetition["token_matches"]
            top_logit_matches += repetition["top_logit_matches"]
            counter_checks += repetition["counter_checks"]
            counter_violations += len(repetition["counter_violations"])
            prefill_state_matches += bool(repetition["prefill_state"]["passed"])
            final_state_matches += bool(repetition["final_state"]["passed"])
            order_counts.update(repetition["order_counts"])
            candidate_steps = repetition["candidate_step_ms"]
            control_steps = repetition["control_step_ms"]
            paired = repetition["paired_savings_ms"]
            candidate_ms.extend(candidate_steps)
            control_ms.extend(control_steps)
            paired_ms.extend(paired)
            category_samples[row["category"]]["candidate"].extend(candidate_steps)
            category_samples[row["category"]]["control"].extend(control_steps)
            category_samples[row["category"]]["paired"].extend(paired)

    candidate_total_ms = sum(candidate_ms)
    control_total_ms = sum(control_ms)
    speedup = control_total_ms / candidate_total_ms
    performance = {
        "measured_pairs": len(paired_ms),
        "candidate": _summarize_ms(candidate_ms),
        "control": _summarize_ms(control_ms),
        "speedup": speedup,
        "delta_percent": (speedup - 1.0) * 100.0,
        "paired_median_saving_ms": statistics.median(paired_ms),
        "paired_mean_saving_ms": statistics.fmean(paired_ms),
        "candidate_wins": sum(delta > 0.0 for delta in paired_ms),
        "order_counts": dict(sorted(order_counts.items())),
        "categories": {
            category: {
                "samples": len(values["paired"]),
                "candidate_aggregate_tokens_per_second": len(values["candidate"])
                * 1_000.0
                / sum(values["candidate"]),
                "control_aggregate_tokens_per_second": len(values["control"])
                * 1_000.0
                / sum(values["control"]),
                "paired_median_saving_ms": statistics.median(values["paired"]),
                "candidate_wins": sum(delta > 0.0 for delta in values["paired"]),
            }
            for category, values in sorted(category_samples.items())
        },
    }
    correctness = {
        "prompt_count": len(rows),
        "prompt_matches": sum(row["passed"] for row in rows),
        "repetition_count": repetition_count,
        "prefill_state_matches": prefill_state_matches,
        "final_state_matches": final_state_matches,
        "decode_position_count": position_count,
        "token_matches": token_matches,
        "top_logit_matches": top_logit_matches,
        "router_counter_checks": counter_checks,
        "router_counter_violations": counter_violations,
    }
    return correctness, performance


def _git_context() -> dict[str, Any]:
    head = _capture(["git", "rev-parse", "HEAD"])
    status = _capture(["git", "status", "--short", "--untracked-files=all"])
    return {
        "head": head["output"],
        "status": status["output"],
        "worktree_clean": status["returncode"] == 0 and not status["output"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--comparison", choices=COMPARISONS, default="router")
    parser.add_argument("--steps", type=int, default=32, help="measured decode steps per prompt")
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0 or args.repetitions <= 0:
        raise ValueError("steps/repetitions must be positive and warmup steps non-negative")

    git = _git_context()
    checkpoint = load_maple_checkpoint(args.model)
    tokenizer = _tokenizer(checkpoint)
    prompts = _load_prompts(args.suite, heldout=False) + _load_prompts(
        args.heldout, heldout=True
    )
    tokenized = [(prompt, tuple(tokenizer.encode_chat(prompt["text"]))) for prompt in prompts]
    if any(not tokens or len(tokens) > 512 for _, tokens in tokenized):
        raise ValueError("all qualification prompts must contain 1-512 native-prefill tokens")
    max_context = max(len(tokens) for _, tokens in tokenized) + args.warmup_steps + args.steps

    reset_memory_stats()
    rows: list[dict[str, Any]] = []
    with _production_environment():
        candidate = MapleRunner.load(
            checkpoint,
            backend=args.backend,
            max_context=max_context,
        )
        control = MapleRunner.load(
            checkpoint,
            backend=args.backend,
            max_context=max_context,
        )
        resident = memory_stats()
        try:
            for prompt_index, (prompt, tokens) in enumerate(tokenized):
                rows.append(
                    _run_prompt_pair(
                        candidate,
                        control,
                        prompt=prompt,
                        tokens=tokens,
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.steps,
                        repetitions=args.repetitions,
                        prompt_index=prompt_index,
                        comparison=args.comparison,
                    )
                )
        finally:
            control.close()
            candidate.close()
    after_close = memory_stats()

    correctness, performance = _aggregate(rows)
    category_counts = Counter(row["category"] for row in rows)
    heldout_counts = Counter(row["category"] for row in rows if row["heldout"])
    suite_qualified = (
        len(rows) == 18
        and all(category_counts[category] > 0 for category in REQUIRED_CATEGORIES)
        and all(heldout_counts[category] > 0 for category in REQUIRED_CATEGORIES)
    )
    correctness_passed = (
        suite_qualified
        and correctness["prompt_matches"] == len(rows)
        and correctness["prefill_state_matches"] == correctness["repetition_count"]
        and correctness["final_state_matches"] == correctness["repetition_count"]
        and correctness["token_matches"] == correctness["decode_position_count"]
        and correctness["top_logit_matches"] == correctness["decode_position_count"]
        and correctness["router_counter_violations"] == 0
    )
    lifecycle_passed = (
        after_close["current_allocated_bytes"] == 0
        and after_close["active_allocations"] == 0
    )
    protocol_qualified = (
        args.steps >= 32 and args.warmup_steps >= 4 and args.repetitions >= 2
    )
    performance_passed = performance["speedup"] > 1.0
    status = (
        "accepted"
        if correctness_passed
        and lifecycle_passed
        and protocol_qualified
        and performance_passed
        and git["worktree_clean"]
        else "rejected"
    )

    rocminfo = _capture(
        ["bash", "-lc", "rocminfo | grep -E 'Name:|Marketing Name:|gfx' | head -8"]
    )
    rocm_smi = _capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"])
    hipcc = _capture(["hipcc", "--version"])
    comparison_notes = {
        "router": (
            "The candidate uses the selector-unset production router; the control uses the exact two-dispatch rollback."
        ),
        "affine4_wave32": (
            "The candidate uses the exact wave32 affine4 head; the control uses the production 128-thread group64 head."
        ),
    }
    artifact = {
        "schema_version": 1,
        "date": date.today().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": f"maple_d0_c1_{args.comparison}_qualification",
        "status": status,
        "performance_claim": status == "accepted",
        "model": {
            "id": args.model,
            "revision": PINNED_REVISION,
            "resolved_path": str(Path(checkpoint.index.model_path).resolve()),
            "quant": "maple_ternary2",
            "exact_weight_bytes": checkpoint.validation.exact_weight_bytes,
        },
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
            "host": platform.node(),
            "rocminfo": rocminfo,
            "rocm_smi": rocm_smi,
            "hipcc": hipcc,
        },
        "software": {"python": platform.python_version(), "git": git},
        "protocol": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": {
                "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                "HIPENGINE_COMPILER_VERSION_FILE": os.environ.get(
                    "HIPENGINE_COMPILER_VERSION_FILE"
                ),
                "HIPENGINE_REQUIRE_CACHED_BUILD": os.environ.get(
                    "HIPENGINE_REQUIRE_CACHED_BUILD"
                ),
                ROUTER_SELECTOR: (
                    "unset candidate; 0 control"
                    if args.comparison == "router"
                    else "unset both"
                ),
                AFFINE4_SELECTOR: (
                    "1 candidate; unset control"
                    if args.comparison == "affine4_wave32"
                    else "unset both"
                ),
                "other_maple_experimental_selectors": "unset",
            },
            "backend": args.backend,
            "comparison": args.comparison,
            "suite": str(args.suite),
            "heldout": str(args.heldout),
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "repetitions": args.repetitions,
            "prompt_count": len(rows),
            "max_context": max_context,
            "timing_scope": "perf_counter around resident MapleRunner.step; native prefill and warmup excluded",
            "pairing": "two simultaneous resident runners; execution order alternates by prompt/repetition/step",
            "qualified_protocol": protocol_qualified,
        },
        "correctness": {
            **correctness,
            "category_counts": dict(sorted(category_counts.items())),
            "heldout_category_counts": dict(sorted(heldout_counts.items())),
            "suite_qualified": suite_qualified,
            "passed": correctness_passed,
        },
        "performance": {**performance, "passed": performance_passed},
        "memory": {
            "two_runner_resident_tracked": resident,
            "after_close": after_close,
            "lifecycle_passed": lifecycle_passed,
            "scope": "hipEngine-owned device allocations; excludes HIP runtime internals",
        },
        "rows": rows,
        "notes": [
            comparison_notes[args.comparison],
            "Every repetition begins from independently computed, byte-identical native-prefill state and advances both runners in lockstep.",
            "Timing is paired and counterbalanced; model load, native prefill, warmup, state copies, and counter checks are outside measured step windows.",
            "Final hashes cover hidden/normalized/full logits, router outputs/counter, live K/V bytes, and KVLiveSpans metadata.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "correctness": {
                    key: artifact["correctness"][key]
                    for key in (
                        "prompt_count",
                        "prompt_matches",
                        "decode_position_count",
                        "token_matches",
                        "top_logit_matches",
                        "prefill_state_matches",
                        "final_state_matches",
                        "router_counter_violations",
                        "passed",
                    )
                },
                "performance": {
                    key: artifact["performance"][key]
                    for key in (
                        "measured_pairs",
                        "speedup",
                        "delta_percent",
                        "paired_median_saving_ms",
                        "candidate_wins",
                        "passed",
                    )
                }
                | {
                    "candidate_tokens_per_second": artifact["performance"]["candidate"][
                        "aggregate_tokens_per_second"
                    ],
                    "control_tokens_per_second": artifact["performance"]["control"][
                        "aggregate_tokens_per_second"
                    ],
                },
                "two_runner_resident_bytes": resident["current_allocated_bytes"],
                "lifecycle_passed": lifecycle_passed,
                "protocol_qualified": protocol_qualified,
                "worktree_clean": git["worktree_clean"],
                "artifact": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
