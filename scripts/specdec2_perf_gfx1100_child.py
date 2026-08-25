#!/usr/bin/env python3
"""Emit the loaded packed-PARO AR/staged half of the gfx1100 P1 bridge.

Dense GGUF uses the shared ``scripts/specdec2_perf_bridge.py`` harness. Packed
PARO cannot keep its separately qualified direct MTP owner resident beside the
Generation-2 model on W7900, so this child measures true AR and staged SPECDEC2
in one load. The parent later attaches a counterbalanced direct-control child
and validates the complete common row contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_category_bench import load_prompt_rows
from scripts.specdec2_perf_gfx1100_bridge import (
    CANONICAL_PROMPTS,
    REQUIRED_TOP_LEVEL_STAGES,
    atomic_write_json,
    build_execution_plan,
)

DEFAULT_MODEL = Path(
    "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"
)
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
LOADED_ARMS = ("true_ar", "staged")


def validate_child_scope(*, lane: str, profile: str, candidate_budget: int) -> None:
    if str(lane) != "paro":
        raise ValueError("loaded PARO child lane must be paro; dense uses the shared bridge")
    if str(profile) not in {"strict", "production"}:
        raise ValueError("PARO bridge profile must be strict or production")
    if int(candidate_budget) != 1:
        raise ValueError("PARO staged bridge is K1-only")


def validate_loaded_arm_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
) -> None:
    """Fail closed when strict loaded AR/staged rows diverge."""

    if str(profile) != "strict":
        return
    cells: dict[tuple[str, int], dict[str, tuple[int, ...]]] = {}
    for row in rows:
        arm = str(row.get("arm") or "")
        if arm not in LOADED_ARMS:
            continue
        key = (str(row.get("prompt_id") or ""), int(row.get("run_index", -1)))
        cells.setdefault(key, {})[arm] = tuple(
            int(value) for value in row.get("generated_token_ids", ())
        )
    for (prompt_id, run_index), arms in cells.items():
        if set(arms) != set(LOADED_ARMS):
            continue
        if arms["true_ar"] != arms["staged"]:
            mismatch = next(
                (
                    index
                    for index, (ar_token, staged_token) in enumerate(
                        zip(arms["true_ar"], arms["staged"], strict=False)
                    )
                    if ar_token != staged_token
                ),
                min(len(arms["true_ar"]), len(arms["staged"])),
            )
            raise RuntimeError(
                "strict AR/staged generated IDs diverged for "
                f"prompt={prompt_id} run={run_index} token_index={mismatch}"
            )


def _seconds_from_ms(mapping: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        raw = mapping.get(name)
        if raw is None:
            continue
        value = float(raw) / 1000.0
        if math.isfinite(value) and value >= 0.0:
            return value
    return None


def _nonnegative_seconds(mapping: Mapping[str, Any], name: str) -> float | None:
    raw = mapping.get(name)
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def resolve_arm_timing(
    *,
    complete_request_seconds: float,
    output_timing: Mapping[str, Any],
    scheduler_observability: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain scheduler prefill/decode once and expose all unknown wall as residual."""

    complete = float(complete_request_seconds)
    if not math.isfinite(complete) or complete <= 0.0:
        raise ValueError("complete_request_seconds must be finite and positive")
    prefill = _nonnegative_seconds(scheduler_observability, "prefill_seconds")
    if prefill is None:
        prefill = _seconds_from_ms(output_timing, "prefill_ms", "backend_prefill_ms")
    prefill = 0.0 if prefill is None else prefill
    tokenize = _seconds_from_ms(output_timing, "tokenize_ms") or 0.0
    provider_open = (
        _seconds_from_ms(output_timing, "specdec2_mtp2_provider_open_ms") or 0.0
    )
    provider_prime = (
        _seconds_from_ms(output_timing, "specdec2_mtp2_prompt_prime_ms") or 0.0
    )
    provider_total = provider_open + provider_prime
    if provider_total > prefill + max(1e-9, prefill * 1e-6):
        raise ValueError("provider open/prime timing exceeds scheduler prefill wall")
    target_prefill = max(0.0, prefill - provider_total)
    decode = _nonnegative_seconds(scheduler_observability, "decode_seconds")
    if decode is None:
        decode = _seconds_from_ms(output_timing, "decode_ms", "backend_decode_ms")
    decode = complete if decode is None or decode <= 0.0 else decode
    known = tokenize + prefill + decode
    tolerance = max(1e-9, complete * 1e-6)
    if known > complete + tolerance:
        raise ValueError(
            "tokenize/prefill/decode timing exceeds complete request wall: "
            f"tokenize={tokenize:.9f}s prefill={prefill:.9f}s "
            f"decode={decode:.9f}s complete={complete:.9f}s"
        )
    stages = {name: 0.0 for name in REQUIRED_TOP_LEVEL_STAGES}
    stages["tokenize"] = tokenize
    stages["target_prefill"] = target_prefill
    stages["provider_open"] = provider_open
    stages["provider_prompt_prime"] = provider_prime
    stages["cycle_total"] = decode
    detail_mapping = {
        "proposal": "specdec2_mtp2_proposal_ms",
        "target_verify": "specdec2_mtp2_target_ms",
        "provider_update": "specdec2_mtp2_provider_update_ms",
        "accept_device": "specdec2_mtp2_accept_ms",
        "selected_target_commit": "specdec2_mtp2_selected_commit_ms",
        "candidate_or_target_readback": "specdec2_mtp2_candidate_readback_ms",
    }
    return {
        "complete_request_seconds": complete,
        "decode_only_seconds": decode,
        "top_level_stage_seconds": stages,
        "unattributed_seconds": max(0.0, complete - known),
        "cycle_detail_seconds": {
            name: float(output_timing[source]) / 1000.0
            for name, source in detail_mapping.items()
            if output_timing.get(source) is not None
            and float(output_timing[source]) >= 0.0
        },
        "scheduler_observability": dict(scheduler_observability),
    }


def _prompt_contract(prompt_id: str) -> tuple[str, str]:
    for identity, category, split in CANONICAL_PROMPTS:
        if identity == str(prompt_id):
            return category, split
    raise ValueError(f"unknown canonical prompt: {prompt_id}")


def build_bridge_row(
    *,
    lane: str,
    arm: str,
    profile: str,
    prompt_id: str,
    run_index: int,
    order_index: int,
    candidate_budget: int,
    max_tokens: int,
    generated_token_ids: Sequence[int],
    timing: Mapping[str, Any],
    selected_manifest_sha256: str,
    strict_manifest_sha256: str,
    commit: str,
    physical_target_rows: Sequence[int],
    physical_proposal_widths: Sequence[int],
    route_name: str,
) -> dict[str, Any]:
    category, split = _prompt_contract(prompt_id)
    selected_arm = str(arm)
    return {
        "schema": 1,
        "lane": str(lane),
        "arm": selected_arm,
        "profile": str(profile),
        "prompt_id": str(prompt_id),
        "category": category,
        "split": split,
        "run_index": int(run_index),
        "order_index": int(order_index),
        "concurrency": 1,
        "candidate_budget": int(candidate_budget),
        "realized_candidate_budget": 0 if selected_arm == "true_ar" else int(candidate_budget),
        "max_tokens": int(max_tokens),
        "generated_token_ids": [int(token) for token in generated_token_ids],
        "timing": {
            **dict(timing),
            "timing_owner_id": (
                f"{lane}:{profile}:{int(run_index)}:{prompt_id}:{selected_arm}:"
                f"c1:k{int(candidate_budget)}"
            ),
            "timing_owner": True,
            "timing_scope": "request",
        },
        "route": {
            "realized": str(route_name),
            "true_autoregressive_path": selected_arm == "true_ar",
            "staged_generation2": selected_arm == "staged",
            "direct_control": selected_arm == "direct",
            "physical_proposal_widths": [int(value) for value in physical_proposal_widths],
            "physical_target_rows": [int(value) for value in physical_target_rows],
        },
        "manifests": {
            "selected_sha256": str(selected_manifest_sha256),
            "strict_sha256": str(strict_manifest_sha256),
        },
        "provenance": {
            "commit": str(commit),
            "staged_dirty": False,
            "unstaged_dirty": False,
            "untracked_dirty": False,
        },
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _repo_provenance() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "staged_dirty": bool(_git("diff", "--cached", "--name-only")),
        "unstaged_dirty": bool(_git("diff", "--name-only")),
        "untracked_dirty": bool(_git("ls-files", "--others", "--exclude-standard")),
    }


def _output_timing(output: Any) -> dict[str, float]:
    telemetry = getattr(output, "telemetry", None)
    raw = None if telemetry is None else getattr(telemetry, "timing", None)
    if not isinstance(raw, Mapping):
        return {}
    return {str(name): float(value) for name, value in raw.items()}


def _output_execution_path(output: Any) -> str | None:
    telemetry = getattr(output, "telemetry", None)
    state = None if telemetry is None else getattr(telemetry, "decode_state", None)
    path = None if state is None else getattr(state, "execution_path", None)
    return None if path is None else str(path)


def _latest_scheduler_observability(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    loop = snapshot.get("loop") if isinstance(snapshot, Mapping) else None
    recent = loop.get("recent_completed") if isinstance(loop, Mapping) else None
    if not isinstance(recent, list) or not recent or not isinstance(recent[-1], Mapping):
        return {}
    observability = recent[-1].get("observability")
    return dict(observability) if isinstance(observability, Mapping) else {}


def _latest_route_metadata(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    runner = snapshot.get("runner") if isinstance(snapshot, Mapping) else None
    routes = runner.get("routes") if isinstance(runner, Mapping) else None
    recent = routes.get("recent_completed") if isinstance(routes, Mapping) else None
    if not isinstance(recent, list) or not recent or not isinstance(recent[-1], Mapping):
        return {}
    return dict(recent[-1])


def _sampling_params(max_tokens: int):
    from hipengine import SamplingParams

    return SamplingParams(
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _run_loaded_arm(
    llm: Any,
    *,
    arm: str,
    prompt: str,
    max_tokens: int,
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    if arm == "true_ar":
        output = llm.generate_detailed((prompt,), _sampling_params(max_tokens))[0]
    elif arm == "staged":
        output = llm.generate_speculative_mtp_detailed(
            (prompt,), _sampling_params(max_tokens)
        )[0]
    else:
        raise ValueError(f"unsupported loaded PARO arm: {arm}")
    complete = time.perf_counter() - started
    snapshot = llm.live_loop_snapshot()
    return (
        output,
        complete,
        _latest_scheduler_observability(snapshot),
        _latest_route_metadata(snapshot),
    )


def _warm_loaded_arms(llm: Any, *, max_tokens: int) -> None:
    for arm in LOADED_ARMS:
        _run_loaded_arm(
            llm,
            arm=arm,
            prompt="Write one short greeting.",
            max_tokens=min(int(max_tokens), 8),
        )


def run_loaded_packet(args: argparse.Namespace) -> dict[str, Any]:
    validate_child_scope(
        lane=args.lane,
        profile=args.profile,
        candidate_budget=args.candidate_budget,
    )
    provenance = _repo_provenance()
    if any(bool(provenance[name]) for name in ("staged_dirty", "unstaged_dirty", "untracked_dirty")):
        raise RuntimeError("loaded bridge execution requires a clean committed worktree")
    model = Path(args.model).resolve()
    prompt_path = Path(args.prompts).resolve()
    prompt_rows = load_prompt_rows(prompt_path)
    selected_ids = tuple(args.prompt_ids or ())
    if selected_ids:
        by_id = {str(row["id"]): row for row in prompt_rows}
        missing = [identity for identity in selected_ids if identity not in by_id]
        if missing:
            raise ValueError(f"unknown prompt ids: {', '.join(missing)}")
        prompt_rows = [by_id[identity] for identity in selected_ids]
    if args.limit is not None:
        prompt_rows = prompt_rows[: int(args.limit)]
    if not prompt_rows:
        raise ValueError("loaded bridge prompt selection is empty")
    prompt_ids = tuple(str(row["id"]) for row in prompt_rows)

    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1100"
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            Path(args.compiler_version_file).resolve()
        )
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine import LLM

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    llm = LLM(
        str(model),
        backend="hip_gfx1100",
        execution_profile=str(args.profile),
        max_active_requests=1,
        max_sequence_length=args.max_sequence_length,
    )
    try:
        llm.prepare(max_sequence_length=args.max_sequence_length)
        if args.warmup:
            _warm_loaded_arms(llm, max_tokens=args.max_tokens)
        selected_manifest = str(getattr(llm, "execution_profile_manifest_sha256", "") or "")
        strict_manifest = str(getattr(llm, "execution_profile_strict_manifest_sha256", "") or "")
        if len(selected_manifest) != 64 or len(strict_manifest) != 64:
            raise RuntimeError("loaded bridge did not resolve selected/strict manifests")
        full_plan = build_execution_plan(
            lane="paro",
            profiles=(args.profile,),
            candidate_budgets=(1,),
            runs=args.runs,
            prompt_ids=tuple(item[0] for item in CANONICAL_PROMPTS),
        )
        selected_plan = [row for row in full_plan if row["prompt_id"] in set(prompt_ids)]
        prompt_by_id = {str(row["id"]): row for row in prompt_rows}
        for plan_row in selected_plan:
            prompt_id = str(plan_row["prompt_id"])
            prompt = str(prompt_by_id[prompt_id]["prompt"])
            loaded_order = [arm for arm in plan_row["arm_order"] if arm in LOADED_ARMS]
            for order_index, arm in enumerate(plan_row["arm_order"]):
                if arm not in LOADED_ARMS:
                    continue
                print(
                    f"[specdec2-gfx1100-paro] profile={args.profile} "
                    f"run={plan_row['run_index']} prompt={prompt_id} arm={arm}",
                    flush=True,
                )
                output, complete, scheduler, route_metadata = _run_loaded_arm(
                    llm,
                    arm=arm,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                )
                generated_ids = getattr(output, "generated_token_ids", None)
                if not generated_ids:
                    raise RuntimeError("loaded PARO arm returned no exact generated IDs")
                output_timing = _output_timing(output)
                timing = resolve_arm_timing(
                    complete_request_seconds=complete,
                    output_timing=output_timing,
                    scheduler_observability=scheduler,
                )
                route_name = (
                    "true_ar"
                    if arm == "true_ar"
                    else (
                        (route_metadata.get("specdec2_mtp2_execution_routes") or [None])[-1]
                        or _output_execution_path(output)
                        or "staged_generation2"
                    )
                )
                row = build_bridge_row(
                    lane="paro",
                    arm=arm,
                    profile=args.profile,
                    prompt_id=prompt_id,
                    run_index=int(plan_row["run_index"]),
                    order_index=int(order_index),
                    candidate_budget=1,
                    max_tokens=args.max_tokens,
                    generated_token_ids=generated_ids,
                    timing=timing,
                    selected_manifest_sha256=selected_manifest,
                    strict_manifest_sha256=strict_manifest,
                    commit=str(provenance["commit"]),
                    physical_target_rows=(1 if arm == "true_ar" else 2,),
                    physical_proposal_widths=(() if arm == "true_ar" else (1,)),
                    route_name=str(route_name),
                )
                row["route_metadata"] = route_metadata
                row["output_timing"] = output_timing
                row["reload_boundary"] = {
                    "loaded_process": "paro_generation2_target",
                    "arms_in_process": list(LOADED_ARMS),
                    "physical_arm_order": loaded_order,
                    "direct_attachment_required": True,
                }
                rows.append(row)
                validate_loaded_arm_ids(rows, profile=args.profile)
                atomic_write_json(
                    args.output,
                    {
                        "schema": 1,
                        "kind": "specdec2_perf_gfx1100_paro_loaded_child",
                        "status": "in_progress",
                        "rows": rows,
                    },
                )
    finally:
        llm.close()
    return {
        "schema": 1,
        "kind": "specdec2_perf_gfx1100_paro_loaded_child",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "lane": "paro",
        "profile": args.profile,
        "candidate_budget": 1,
        "model": str(model),
        "prompt_file": str(prompt_path),
        "prompt_ids": list(prompt_ids),
        "runs": int(args.runs),
        "max_tokens": int(args.max_tokens),
        "arms": list(LOADED_ARMS),
        "provenance": provenance,
        "total_wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def _csv_text(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("paro",), default="paro")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--profile", choices=("strict", "production"), required=True)
    parser.add_argument("--candidate-budget", type=int, default=1)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-ids", type=_csv_text, default=())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_loaded_packet(args)
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "profile": payload["profile"],
                "rows": len(payload["rows"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
