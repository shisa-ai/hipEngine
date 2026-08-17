#!/usr/bin/env python3
"""PN1 no-change strict/production control-capture GPU smoke.

Runs a small teacher-forced c1 schedule on a Qwen3.6 GGUF model for both the
``strict`` and ``production`` execution profiles, emits the standardized
actual-control capture from live resident-session state, builds the independent
expected-control fixtures from the schedule spec, writes the RunCapture
manifests + variant manifests + task results, and evaluates everything through
``execution_profile_gate.py``.

This is a plumbing smoke, not a benchmark: it validates that the profile plans
resolve, the control producer emits schema-valid ground truth that agrees with
the frozen fixtures, and the gate produces a passed verdict for a no-change
run. The full 18-prompt task artifact, long-context/dynamic shapes, and BF16
attachment remain separate PN1/PN2 deliverables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.control_capture import (
    RowControlPrimitives,
    build_control_capture,
    build_control_fixture,
    derive_control_record,
    schedule_c1_control_records,
)
from hipengine.benchmark.execution_profiles import (
    EXECUTION_PROFILE_CAPTURE_KIND,
    EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION,
    RowDescriptor,
    manifest_sha256,
    validate_variant_manifest,
)
from hipengine.execution_profiles import (
    ExecutionProfile,
    resolve_runtime_profile,
)
from hipengine.generation import register_builtin_generators
from scripts.gguf_mtp_bench import build_chat_prompt

DEFAULT_GDN_MODE = "chain_lds32_direct_nonvolatile"
DEFAULT_BULK_ATTENTION_MODE = "bulk"
DEFAULT_ARITHMETIC_CLASS = "T2"
SMOKE_TASK_NAME = "greedy_ids_strict_aligned"
ISOLATION_SCENARIO_SUFFIX = "-isolation"

# Frozen c1 route environments (matches the retained router evidence): the
# cooperative/persistent F32 router is the production route; strict disables it.
_ROUTER_COOP_ENV = "HIPENGINE_GGUF_ROUTER_F32W_COOP"
_ROUTER_PERSISTENT_ENV = "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER"
_ROWTILE_ALL_ENV = "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"
_STRICT_ROUTE_ENV = {
    _ROUTER_COOP_ENV: "0",
    _ROUTER_PERSISTENT_ENV: "0",
    _ROWTILE_ALL_ENV: "0",
}
_PRODUCTION_ROUTE_ENV = {
    _ROUTER_COOP_ENV: "1",
    _ROUTER_PERSISTENT_ENV: "1",
    # rowtile floor (rows >= 4) = env unset; no-op for a single-row smoke.
    _ROWTILE_ALL_ENV: None,
}


@contextmanager
def _route_env(values: Mapping[str, str | None]) -> Iterator[None]:
    """Apply one frozen c1 route environment and restore the caller exactly."""

    keys = (_ROUTER_COOP_ENV, _ROUTER_PERSISTENT_ENV, _ROWTILE_ALL_ENV)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            value = values.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_rows(prompts_path: Path, *, limit: int) -> list[dict]:
    rows: list[dict] = []
    with prompts_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt_id = str(raw["id"])
            has_prompt = "prompt" in raw
            has_messages = "messages" in raw
            if has_prompt and has_messages:
                raise SystemExit(f"{prompt_id}: expected exactly one of prompt or messages[]")
            if has_prompt:
                prompt_text = str(raw["prompt"])
            else:
                messages = raw.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise SystemExit(f"{prompt_id}: expected prompt or messages[]")
                user_parts = [
                    str(message["content"])
                    for message in messages
                    if message.get("role") == "user" and message.get("content")
                ]
                prompt_text = "\n\n".join(user_parts)
            if not prompt_text or not prompt_text.strip():
                raise SystemExit(f"{prompt_id}: prompt text is empty")
            rows.append(
                {
                    "id": prompt_id,
                    "prompt": prompt_text,
                    "category": str(raw.get("category", "general_en")),
                }
            )
            if len(rows) >= limit:
                break
    if not rows:
        raise SystemExit("no prompts loaded")
    return rows


def _trajectory_with_controls(
    session,
    *,
    prompt_ids: list[int],
    forced_input_ids: list[int] | None,
    decode_steps: int,
    gdn_mode: str,
    bulk_attention_mode: str,
    scenario_id: str,
    request_id: str,
    route_top_k: int,
    graph_bucket: str,
    rng_seed: int,
    route_env: Mapping[str, str | None] | None = None,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Run prefill + forced decode under a route env and return live primitives."""

    from scripts.gguf_gdn_trajectory_gate import _gdn_mode

    @contextmanager
    def _env() -> Iterator[None]:
        if route_env is None:
            yield
        else:
            with _route_env(route_env):
                yield

    with _env():
        session.reset()
        with _gdn_mode(gdn_mode):
            result = session.prefill(
                [int(token) for token in prompt_ids],
                use_bulk=True,
                bulk_attention_mode=bulk_attention_mode,
                return_logits=True,
                capture_hidden_seed_fp32=False,
            )
        prompt_len = len(prompt_ids)
        logits_rows = [np.ascontiguousarray(result.logits, dtype=np.float32)]
        controls = [
            {
                "scenario_id": scenario_id,
                "scenario_step": 0,
                "request_id": request_id,
                "input_token_id": int(prompt_ids[-1]),
                "position": prompt_len - 1,
                "context_length": prompt_len,
                "route_top_k": route_top_k,
                "graph_bucket": graph_bucket,
                "rng_seed": rng_seed,
            }
        ]
        row_specs = [
            {
                "scenario_step": 0,
                "request_id": request_id,
                "teacher_step": 0,
                "category": "smoke",
                "shape": "prefill_last",
                "transition": "prefill_to_c1",
                "teacher_token_id": int(result.token_id),
            }
        ]
        if forced_input_ids is None:
            # Greedy: run decode_steps decode steps, feeding the previous
            # generated token each time (mirrors _run_logits_trajectory).
            previous = int(result.token_id)
            for step in range(1, int(decode_steps) + 1):
                position = session.position
                probe = session.step(previous, return_logits=True)
                logits_rows.append(np.ascontiguousarray(probe.logits, dtype=np.float32))
                controls.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_step": step,
                        "request_id": request_id,
                        "input_token_id": previous,
                        "position": int(position),
                        "context_length": int(position) + 1,
                        "route_top_k": route_top_k,
                        "graph_bucket": graph_bucket,
                        "rng_seed": rng_seed,
                    }
                )
                row_specs.append(
                    {
                        "scenario_step": step,
                        "request_id": request_id,
                        "teacher_step": step,
                        "category": "smoke",
                        "shape": "c1",
                        "transition": "steady",
                        "teacher_token_id": int(probe.token_id),
                    }
                )
                previous = int(probe.token_id)
        else:
            consume = [int(token) for token in forced_input_ids]
            for step, input_token_id in enumerate(consume, start=1):
                position = session.position
                probe = session.step(input_token_id, return_logits=True)
                logits_rows.append(np.ascontiguousarray(probe.logits, dtype=np.float32))
                controls.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_step": step,
                        "request_id": request_id,
                        "input_token_id": int(input_token_id),
                        "position": int(position),
                        "context_length": int(position) + 1,
                        "route_top_k": route_top_k,
                        "graph_bucket": graph_bucket,
                        "rng_seed": rng_seed,
                    }
                )
                row_specs.append(
                    {
                        "scenario_step": step,
                        "request_id": request_id,
                        "teacher_step": step,
                        "category": "smoke",
                        "shape": "c1",
                        "transition": "steady",
                        "teacher_token_id": int(probe.token_id),
                    }
                )
        logits = np.stack(logits_rows, axis=0)
        logits = np.reshape(logits, (int(logits.shape[0]), -1))
        return logits, controls, row_specs


def _assemble_capture(
    *,
    output_dir: Path,
    run_id: str,
    execution_profile: str,
    scenario_id: str,
    variant_manifest: dict,
    segments: list[tuple[np.ndarray, list[dict], list[dict]]],
    repeat_index: int,
) -> Path:
    logits = np.concatenate([segment[0] for segment in segments], axis=0)
    controls = [
        derive_control_record(RowControlPrimitives(**record)).to_dict()
        for segment in segments
        for record in segment[1]
    ]
    row_specs = [spec for segment in segments for spec in segment[2]]
    logits_path = output_dir / f"{run_id}-logits.npy"
    np.save(logits_path, logits)
    rows = tuple(
        RowDescriptor(
            scenario_id=scenario_id,
            scenario_step=int(spec["scenario_step"]),
            request_id=spec["request_id"],
            teacher_step=int(spec["teacher_step"]),
            category=spec["category"],
            shape=spec["shape"],
            transition=spec["transition"],
            teacher_token_id=int(spec["teacher_token_id"]),
        )
        for spec in row_specs
    )
    selected = [int(np.argmax(logits[index])) for index in range(len(rows))]
    payload = {
        "kind": EXECUTION_PROFILE_CAPTURE_KIND,
        "schema_version": EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION,
        "execution_profile": execution_profile,
        "scenario_id": scenario_id,
        "run_id": run_id,
        "variant_manifest_sha256": manifest_sha256(validate_variant_manifest(variant_manifest)),
        "repeat_index": repeat_index,
        "logits_path": logits_path.name,
        "logits_sha256": _sha256_file(logits_path),
        "rows": [row.to_dict() for row in rows],
        "selected_token_ids": selected,
        "controls": controls,
    }
    capture_path = output_dir / f"{run_id}-capture.json"
    capture_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return capture_path


def _write_fixture(
    *,
    output_dir: Path,
    name: str,
    scenario_id: str,
    run_id: str,
    records: Sequence[object],
) -> Path:
    payload = build_control_fixture(scenario_id=scenario_id, records=records)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _selected_ids(capture_path: Path) -> list[int]:
    return list(json.loads(capture_path.read_text(encoding="utf-8"))["selected_token_ids"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--scenario-id", default="qwen36_zbook_c1_smoke")
    parser.add_argument("--run-id", default="pn1-smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--gdn-mode", default=DEFAULT_GDN_MODE)
    parser.add_argument("--bulk-attention-mode", default=DEFAULT_BULK_ATTENTION_MODE)
    parser.add_argument("--arithmetic-class", default=DEFAULT_ARITHMETIC_CLASS,
                        choices=("T0", "T1", "T2", "T3"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.generation.qwen36_gguf_profiles import (
        QWEN36_GGUF_BACKEND,
        QWEN36_GGUF_MODEL,
        QWEN36_GGUF_QUANT,
    )

    register_builtin_generators()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _prompt_rows(args.prompts, limit=args.limit)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): build_chat_prompt(tokenizer, row["prompt"])
        for row in rows
    }
    max_sequence_length = (
        max(len(tokens) for tokens in prompt_tokens.values())
        + int(args.decode_steps)
        + 4
    )
    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )

    strict_plan = resolve_runtime_profile(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    )
    production_plan = resolve_runtime_profile(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
    )
    strict_manifest = validate_variant_manifest(strict_plan.manifest)
    production_manifest = validate_variant_manifest(production_plan.manifest)
    for path, manifest in (
        (output_dir / "strict-variant-manifest.json", strict_manifest),
        (output_dir / "production-variant-manifest.json", production_manifest),
    ):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    strict_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    production_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    repeat_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    isolation_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    fixture_records_by_prompt: dict[str, tuple[object, ...]] = {}
    teacher_by_prompt: dict[str, list[int]] = {}
    greedy_aligned = True

    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=1),
    ) as session:
        if session.runner is None:
            raise SystemExit("resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        print(
            f"resolved backend={resolved_backend} arch={target_arch} "
            f"strict_plan={strict_plan.profile.value} "
            f"production_plan={production_plan.profile.value}",
            flush=True,
        )

        for prompt_row in rows:
            prompt_id = prompt_row["id"]
            request_id = f"prompt-{prompt_id}"
            tokens = prompt_tokens[prompt_id]

            strict_logits, strict_controls, strict_row_specs = _trajectory_with_controls(
                session,
                prompt_ids=tokens,
                forced_input_ids=None,
                decode_steps=int(args.decode_steps),
                gdn_mode=args.gdn_mode,
                bulk_attention_mode=args.bulk_attention_mode,
                scenario_id=args.scenario_id,
                request_id=request_id,
                route_top_k=int(args.top_k),
                graph_bucket="c1",
                rng_seed=int(args.rng_seed),
                route_env=_STRICT_ROUTE_ENV,
            )
            teacher = [spec["teacher_token_id"] for spec in strict_row_specs[:-1]]
            teacher_by_prompt[prompt_id] = teacher
            strict_segments.append((strict_logits, strict_controls, strict_row_specs))

            def _production(
                *,
                scenario_id: str,
                run_forced: list[int],
                run_index: int,
            ) -> tuple[np.ndarray, list[dict], list[dict]]:
                return _trajectory_with_controls(
                    session,
                    prompt_ids=tokens,
                    forced_input_ids=run_forced,
                    decode_steps=int(args.decode_steps),
                    gdn_mode=args.gdn_mode,
                    bulk_attention_mode=args.bulk_attention_mode,
                    scenario_id=scenario_id,
                    request_id=request_id,
                    route_top_k=int(args.top_k),
                    graph_bucket="c1",
                    rng_seed=int(args.rng_seed),
                    route_env=_PRODUCTION_ROUTE_ENV,
                )

            production_segments.append(
                _production(scenario_id=args.scenario_id, run_forced=teacher, run_index=0)
            )
            repeat_segments.append(
                _production(
                    scenario_id=args.scenario_id,
                    run_forced=teacher,
                    run_index=1,
                )
            )
            isolation_segments.append(
                _production(
                    scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
                    run_forced=teacher,
                    run_index=1,
                )
            )
            fixture_records_by_prompt[prompt_id] = schedule_c1_control_records(
                scenario_id=args.scenario_id,
                request_id=request_id,
                prompt_ids=tokens,
                teacher_token_ids=teacher,
                route_top_k=int(args.top_k),
                graph_bucket="c1",
                rng_seed=int(args.rng_seed),
            )
            print(
                f"{prompt_id}: strict + production + repeat + isolation runs, "
                f"{len(teacher)} forced steps",
                flush=True,
            )

    strict_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-strict",
        execution_profile=ExecutionProfile.STRICT.value,
        scenario_id=args.scenario_id,
        variant_manifest=strict_manifest,
        segments=strict_segments,
        repeat_index=0,
    )
    production_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-production",
        execution_profile=ExecutionProfile.PRODUCTION.value,
        scenario_id=args.scenario_id,
        variant_manifest=production_manifest,
        segments=production_segments,
        repeat_index=0,
    )
    repeat_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-production-repeat",
        execution_profile=ExecutionProfile.PRODUCTION.value,
        scenario_id=args.scenario_id,
        variant_manifest=production_manifest,
        segments=repeat_segments,
        repeat_index=1,
    )
    isolation_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-isolation",
        execution_profile=ExecutionProfile.PRODUCTION.value,
        scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
        variant_manifest=production_manifest,
        segments=isolation_segments,
        repeat_index=1,
    )
    strict_expected_records: tuple[object, ...] = tuple(
        record
        for prompt_id in [row["id"] for row in rows]
        for record in fixture_records_by_prompt[prompt_id]
    )
    strict_fixture = _write_fixture(
        output_dir=output_dir,
        name=f"{args.run_id}-strict-expected-controls",
        scenario_id=args.scenario_id,
        run_id=f"{args.run_id}-strict",
        records=strict_expected_records,
    )
    production_fixture = _write_fixture(
        output_dir=output_dir,
        name=f"{args.run_id}-production-expected-controls",
        scenario_id=args.scenario_id,
        run_id=f"{args.run_id}-production",
        records=strict_expected_records,
    )
    isolation_expected_records: tuple[object, ...] = tuple(
        record
        for prompt_id in [row["id"] for row in rows]
        for record in schedule_c1_control_records(
            scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
            request_id=f"prompt-{prompt_id}",
            prompt_ids=prompt_tokens[prompt_id],
            teacher_token_ids=teacher_by_prompt[prompt_id],
            route_top_k=int(args.top_k),
            graph_bucket="c1",
            rng_seed=int(args.rng_seed),
        )
    )
    isolation_fixture = _write_fixture(
        output_dir=output_dir,
        name=f"{args.run_id}-isolation-expected-controls",
        scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
        run_id=f"{args.run_id}-isolation",
        records=isolation_expected_records,
    )
    greedy_aligned = bool(
        _selected_ids(strict_capture) == _selected_ids(production_capture)
    )

    task_results = {SMOKE_TASK_NAME: greedy_aligned}
    task_path = output_dir / "task-results.json"
    task_path.write_text(json.dumps(task_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_note = {
        "scenario_id": args.scenario_id,
        "run_id": args.run_id,
        "backend": resolved_backend,
        "target_arch": target_arch,
        "limit": args.limit,
        "decode_steps": args.decode_steps,
        "gdn_mode": args.gdn_mode,
        "bulk_attention_mode": args.bulk_attention_mode,
        "arithmetic_class": args.arithmetic_class,
        "task_results": task_results,
        "smoke_only": True,
    }
    (output_dir / "smoke-env.json").write_text(
        json.dumps(env_note, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote captures/fixtures/manifests/task-results to {output_dir}; "
        f"greedy_aligned={greedy_aligned}",
        flush=True,
    )
    if args.skip_gate:
        return 0 if greedy_aligned else 1

    verdict_path = output_dir / "gate-verdict.json"
    gate_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "execution_profile_gate.py"),
        "--variant-manifest", str(output_dir / "production-variant-manifest.json"),
        "--strict-manifest", str(output_dir / "strict-variant-manifest.json"),
        "--strict-capture", str(strict_capture),
        "--candidate-capture", str(production_capture),
        "--expected-controls", str(production_fixture),
        "--strict-expected-controls", str(strict_fixture),
        "--comparison-controls", str(isolation_fixture),
        "--repeat-capture", str(repeat_capture),
        "--isolation-capture", str(isolation_capture),
        "--task-results", str(task_path),
        "--arithmetic-class", args.arithmetic_class,
        "--output", str(verdict_path),
    ]
    print("invoking gate:", " ".join(gate_cmd), flush=True)
    gate_result = subprocess.run(gate_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if gate_result.stdout.strip():
        print("gate stdout:", gate_result.stdout.strip(), flush=True)
    if gate_result.stderr.strip():
        print("gate stderr:", gate_result.stderr.strip()[-800:], flush=True)
    if not verdict_path.is_file():
        print("gate produced no verdict file; rc=", gate_result.returncode, flush=True)
        return 1
    result = json.loads(verdict_path.read_text(encoding="utf-8"))
    print(
        f"gate verdict: execution_profile={result.get('execution_profile')} "
        f"status={result.get('decision', {}).get('status')} "
        f"automatic={result.get('decision', {}).get('eligible_for_automatic_admission')} "
        f"controls={result.get('control_semantics', {}).get('passed')} "
        f"determinism={result.get('determinism', {}).get('passed')} "
        f"isolation={result.get('isolation', {}).get('passed')} "
        f"tasks={result.get('task_quality', {}).get('passed')} "
        f"generated={result.get('generated_id_equality', {}).get('all_equal')}",
        flush=True,
    )
    return 0 if result.get("decision", {}).get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
