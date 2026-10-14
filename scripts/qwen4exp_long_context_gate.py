#!/usr/bin/env python3
"""Long-prompt production gate for Qwen4Exp: round-off floor plus retrieval.

The calibrated production envelope (mean/p95/p99/max KL against ``strict``)
was fitted on prompts of tens to hundreds of tokens. At 2,000 tokens and
beyond, ANY perturbation of round-off size moves the distribution by about
2e-3 mean KL against ``strict`` (an F32 summation-order change in one
projection does), so the short-prompt limits cannot tell a defect from
round-off there. This gate replaces fixed limits with two measurements that
stay meaningful at length (EXECUTION-PROFILES section 6.1):

1. **Round-off floor (numerical axis).** On the long numerical prompts,
   ``strict`` is rerun with a seeded perturbation of one BF16 ulp in one
   element of one prompt embedding row: the smallest change a single rounding
   flip in the BF16 residual stream can make. Each seed is a control arm.
   The candidate passes when its mean and p95 KL against ``strict`` do not
   exceed the median control's, its max KL does not exceed the largest
   control max, and its top-1 agreement is not below the lowest control's.
   In words: the candidate must diverge no more than a typical single
   rounding flip does. (Calibration on 2026-10-14: one flip gives all-prompt
   mean KL 4.5e-3 to 7.7e-3 on the four numerical prompts, more than the
   complete F16 route composition's 2.1e-3.)
2. **Retrieval (task axis).** On 8K and 16K retrieval prompts with answers
   stored in the suite, every arm is teacher-forced through the gold answer
   text. An answer counts as correct when the arm's argmax reproduces every
   token of that answer's number. The candidate must answer at least as many
   as ``strict``. The gold-answer NLL delta is reported alongside.

Also binding: candidate repeats are bit-identical, controls are effective (a
control that changes nothing measures no floor) and below a sanity cap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_canonical_ar_bench import (
    _git_metadata,
    _host_metadata,
    _write_json,
    sha256_path,
)
from scripts.qwen4exp_layer2_profile_gate import (
    CANDIDATES,
    GateError,
    _candidate_trajectory,
    _make_generator,
    _post_construct_hook,
    _strict_trajectory,
)

KIND = "qwen4exp_long_context_profile_gate"
DEFAULT_NUMERIC = ROOT / "benchmarks" / "prompts" / "qwen4exp-long-context-gate.jsonl"
DEFAULT_RETRIEVAL = ROOT / "benchmarks" / "prompts" / "qwen4exp-long-retrieval-v1.jsonl"
# A control whose mean KL exceeds this is no longer a round-off-sized floor.
CONTROL_MEAN_KL_CAP = 1e-2


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _prompt_text(row: Mapping[str, Any]) -> str:
    return str(row["messages"][0]["content"])


def _embedding_bits(runner: Any, token_id: int) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, host_array_ptr
    from hipengine.kernels.registry import resolve

    cfg = runner.config
    weight = runner.resident.weight("root.token_embedding")
    lookup = resolve(backend=runner.backend, layer="embedding",
                     quant=weight.spec.quant_key, variant="lookup_bf16_out")
    host = np.asarray([int(token_id)], dtype=np.int32)
    copy_host_to_device(runner.token_id_buffer, host_array_ptr(host), runtime=runner.runtime)
    lookup(runner.token_id_buffer.ptr, weight.allocation("raw").tensor.ptr,
           runner.embedding_buffer.ptr, 1, cfg.hidden_size, cfg.vocab_size,
           runtime=runner.runtime)
    bits = np.zeros(cfg.hidden_size, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(bits), runner.embedding_buffer, bits.nbytes,
                        runtime=runner.runtime)
    return bits


def roundoff_override(runner: Any, prompt_ids: Sequence[int], seed: int
                      ) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """One BF16 ulp, up or down, in one element of one prompt embedding row."""

    rng = np.random.default_rng(int(seed))
    position = int(rng.integers(len(prompt_ids) - 1))
    bits = _embedding_bits(runner, int(prompt_ids[position])).copy()
    element = int(rng.integers(bits.size))
    step = 1 if int(rng.integers(2)) else -1
    bits[element] = np.uint16((int(bits[element]) + step) & 0xFFFF)
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    return {position: values}, {"position": position, "element": element, "ulp_step": step}


def _forced_trajectory(runner: Any, prompt_ids: Sequence[int], forced: Sequence[int],
                       overrides: Mapping[int, np.ndarray] | None = None
                       ) -> tuple[dict[str, Any], ...]:
    if overrides is None:
        return _candidate_trajectory(runner, prompt_ids, forced)
    runner.reset()
    result = runner.prefill([int(t) for t in prompt_ids], embedding_overrides=overrides)
    out = [{"token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32)}]
    for token in forced:
        result = runner.step(int(token))
        out.append({"token_id": int(result.token_id),
                    "logits": np.ascontiguousarray(result.logits, dtype=np.float32)})
    return tuple(out)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - values.max()
    return values - np.log(np.exp(values).sum())


def gold_answer_tokens(tokenizer: Any, answers: Sequence[int]) -> tuple[list[int], list[list[int]]]:
    """Gold answer text ``1: <n>\\n2: <n>...`` and the token indices of each number."""

    ids: list[int] = []
    spans: list[list[int]] = []
    for index, answer in enumerate(answers, start=1):
        prefix = tokenizer.encode(f"{index}: " if index == 1 else f"\n{index}: ")
        ids.extend(prefix)
        number = tokenizer.encode(str(int(answer)))
        spans.append(list(range(len(ids), len(ids) + len(number))))
        ids.extend(number)
    return ids, spans


def score_retrieval(trajectory: Sequence[Mapping[str, Any]], gold: Sequence[int],
                    spans: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Row ``i`` of a gold-forced trajectory predicts gold token ``i``."""

    nll = [-float(_log_softmax(trajectory[i]["logits"])[int(token)])
           for i, token in enumerate(gold)]
    hits = [int(np.argmax(trajectory[i]["logits"])) == int(gold[i]) for i in range(len(gold))]
    correct = [all(hits[i] for i in span) for span in spans]
    return {"correct": correct, "correct_count": int(sum(correct)),
            "gold_nll_mean": float(np.mean(nll)),
            "answer_nll_mean": float(np.mean([nll[i] for span in spans for i in span]))}


def _summary(strict: Mapping[str, Sequence[Mapping[str, Any]]],
             arm: Mapping[str, Sequence[Mapping[str, Any]]],
             categories: Mapping[str, str], name: str) -> dict[str, Any]:
    rows: list[RowDescriptor] = []
    left, right = [], []
    for prompt_id, teacher in strict.items():
        for step, (s_row, c_row) in enumerate(zip(teacher, arm[prompt_id])):
            rows.append(RowDescriptor(
                scenario_id=name, scenario_step=len(rows), request_id=prompt_id,
                teacher_step=step, category=categories[prompt_id],
                shape="prefill_last" if step == 0 else "c1",
                transition="prefill_to_c1" if step == 0 else "steady",
                teacher_token_id=int(s_row["token_id"])))
            left.append(s_row["logits"])
            right.append(c_row["logits"])
    result = compare_profile_logits(np.stack(left), np.stack(right), rows)
    return dict(result["summary"])


def decide(candidate: Mapping[str, Any], controls: Sequence[Mapping[str, Any]],
           candidate_retrieval: int, strict_retrieval: int, *, deterministic: bool
           ) -> dict[str, Any]:
    """The binding rules, separated from measurement so they can be unit-tested."""

    floor = {
        "kl_mean": float(np.median([float(c["kl_mean"]) for c in controls])),
        "kl_p95": float(np.median([float(c["kl_p95"]) for c in controls])),
        "kl_max": max(float(c["kl_max"]) for c in controls),
        "top1_matches": min(int(c["top1_matches"]) for c in controls),
    }
    checks = {
        "controls_effective": all(float(c["kl_max"]) > 0.0 for c in controls),
        "controls_round_off_sized": floor["kl_mean"] <= CONTROL_MEAN_KL_CAP,
        "mean_within_floor": float(candidate["kl_mean"]) <= floor["kl_mean"],
        "p95_within_floor": float(candidate["kl_p95"]) <= floor["kl_p95"],
        "max_within_floor": float(candidate["kl_max"]) <= floor["kl_max"],
        "top1_within_floor": int(candidate["top1_matches"]) >= floor["top1_matches"],
        "retrieval_not_below_strict": int(candidate_retrieval) >= int(strict_retrieval),
        "candidate_repeats_identical": bool(deterministic),
    }
    valid = checks["controls_effective"] and checks["controls_round_off_sized"]
    passed = valid and all(checks.values())
    return {"floor": floor, "checks": checks, "measurement_valid": valid,
            "status": "passed" if passed else "failed" if valid else "invalid"}


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not _git_metadata(ROOT)["tracked_clean"]:
        raise GateError("profile qualification requires a clean committed tracked tree")
    if args.controls < 2 or args.candidate_repeats < 2:
        raise GateError("the floor needs at least two controls; determinism two repeats")
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        register_gguf_q4_k_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_gguf_q4_k_selected_prefill_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    spec = CANDIDATES[str(args.candidate)]
    numeric = _load(args.numeric_prompts)
    retrieval = _load(args.retrieval_prompts)
    categories = {str(r["id"]): str(r["category"]) for r in numeric}
    started = time.time()

    prompt_ids: dict[str, list[int]] = {}
    strict_numeric: dict[str, tuple] = {}
    control_runs: list[dict[str, tuple]] = [{} for _ in range(args.controls)]
    control_meta: list[dict[str, Any]] = [{} for _ in range(args.controls)]
    gold: dict[str, tuple[list[int], list[list[int]]]] = {}
    strict_retrieval: dict[str, dict[str, Any]] = {}
    strict_gen, strict_profile, _ = _make_generator(args, "strict")
    try:
        runner, tokenizer = strict_gen.runner, strict_gen.tokenizer
        for row in numeric:
            pid = str(row["id"])
            ids = [int(t) for t in build_chat_prompt(tokenizer, _prompt_text(row))]
            prompt_ids[pid] = ids
            teacher = _strict_trajectory(runner, ids, int(args.decode_steps))
            strict_numeric[pid] = teacher
            forced = [int(step["token_id"]) for step in teacher[:-1]]
            for index in range(args.controls):
                override, meta = roundoff_override(runner, ids, args.control_seed + index)
                control_runs[index][pid] = _forced_trajectory(runner, ids, forced, override)
                control_meta[index][pid] = meta
            print(f"strict+controls {pid} ({len(ids)} tokens)", flush=True)
        for row in retrieval:
            pid = str(row["id"])
            ids = [int(t) for t in build_chat_prompt(tokenizer, _prompt_text(row))]
            prompt_ids[pid] = ids
            gold[pid] = gold_answer_tokens(tokenizer, row["answers"])
            trajectory = _forced_trajectory(runner, ids, gold[pid][0][:-1])
            strict_retrieval[pid] = {"trajectory": trajectory,
                                     **score_retrieval(trajectory, *gold[pid])}
            print(f"strict retrieval {pid}: {strict_retrieval[pid]['correct']}", flush=True)
    finally:
        strict_gen.close()

    candidate_numeric: list[dict[str, tuple]] = [{} for _ in range(args.candidate_repeats)]
    candidate_retrieval: list[dict[str, dict[str, Any]]] = [
        {} for _ in range(args.candidate_repeats)]
    saved = {key: os.environ.get(key) for key in spec.environment}
    gen, candidate_profile, _ = _make_generator(args, spec.base_profile, spec.quant)
    _post_construct_hook(gen, args)
    os.environ.update(spec.environment)
    try:
        runner = gen.runner
        for repeat in range(args.candidate_repeats):
            for pid, teacher in strict_numeric.items():
                forced = [int(step["token_id"]) for step in teacher[:-1]]
                candidate_numeric[repeat][pid] = _forced_trajectory(runner, prompt_ids[pid], forced)
            for row in retrieval:
                pid = str(row["id"])
                trajectory = _forced_trajectory(runner, prompt_ids[pid], gold[pid][0][:-1])
                candidate_retrieval[repeat][pid] = {
                    "trajectory": trajectory, **score_retrieval(trajectory, *gold[pid])}
            print(f"candidate repeat {repeat + 1}/{args.candidate_repeats}", flush=True)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        gen.close()

    def same(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> bool:
        return all(np.array_equal(x["logits"], y["logits"]) for x, y in zip(a, b))

    deterministic = all(
        same(candidate_numeric[0][pid], run[pid])
        for run in candidate_numeric[1:] for pid in strict_numeric
    ) and all(
        same(candidate_retrieval[0][pid]["trajectory"], run[pid]["trajectory"])
        for run in candidate_retrieval[1:] for pid in strict_retrieval
    )
    controls = [_summary(strict_numeric, runs, categories, f"control_{i}")
                for i, runs in enumerate(control_runs)]
    candidate = _summary(strict_numeric, candidate_numeric[0], categories, spec.name)
    retrieval_categories = {str(r["id"]): str(r["category"]) for r in retrieval}
    retrieval_kl = _summary(
        {pid: v["trajectory"] for pid, v in strict_retrieval.items()},
        {pid: v["trajectory"] for pid, v in candidate_retrieval[0].items()},
        retrieval_categories, spec.name + "_retrieval")
    strict_correct = sum(v["correct_count"] for v in strict_retrieval.values())
    candidate_correct = sum(v["correct_count"] for v in candidate_retrieval[0].values())
    decision = decide(candidate, controls, candidate_correct, strict_correct,
                      deterministic=deterministic)
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        decision["status"] = "invalid"
        decision["measurement_valid"] = False

    def retrieval_rows(table: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {pid: {k: v for k, v in row.items() if k != "trajectory"}
                for pid, row in table.items()}

    return {
        "schema": 1,
        "kind": KIND,
        "status": decision["status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - started, 1),
        "performance_claim": False,
        "source": source,
        "host": _host_metadata(),
        "command": list(command),
        "candidate": {"id": str(args.candidate), "name": spec.name,
                      "base_profile": spec.base_profile,
                      "environment": dict(spec.environment),
                      "manifest_sha256": candidate_profile.manifest_sha256},
        "strict_manifest_sha256": strict_profile.manifest_sha256,
        "protocol": {
            "numeric_prompts": str(args.numeric_prompts),
            "numeric_prompts_sha256": sha256_path(args.numeric_prompts),
            "retrieval_prompts": str(args.retrieval_prompts),
            "retrieval_prompts_sha256": sha256_path(args.retrieval_prompts),
            "prompt_tokens": {pid: len(ids) for pid, ids in prompt_ids.items()},
            "decode_steps": int(args.decode_steps),
            "controls": int(args.controls),
            "control_seed": int(args.control_seed),
            "control_perturbation": "one BF16 ulp in one element of one prompt embedding row",
            "candidate_repeats": int(args.candidate_repeats),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "max_sequence_length": int(args.max_sequence_length),
            "control_mean_kl_cap": CONTROL_MEAN_KL_CAP,
        },
        "numeric": {"candidate": candidate, "controls": controls,
                    "control_perturbations": control_meta},
        "retrieval": {
            "strict_correct": strict_correct,
            "candidate_correct": candidate_correct,
            "answers": sum(len(r["answers"]) for r in retrieval),
            "strict": retrieval_rows(strict_retrieval),
            "candidate": retrieval_rows(candidate_retrieval[0]),
            "gold_forced_kl": retrieval_kl,
        },
        "decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--candidate", default="production_baseline", choices=tuple(CANDIDATES))
    parser.add_argument("--numeric-prompts", type=Path, default=DEFAULT_NUMERIC)
    parser.add_argument("--retrieval-prompts", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--control-seed", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--candidate-repeats", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=16640)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args, command=[Path(sys.argv[0]).name, *(argv or sys.argv[1:])])
    except GateError as error:
        parser.error(str(error))
    _write_json(args.output, result)
    print(json.dumps({"status": result["status"], "checks": result["decision"]["checks"],
                      "floor": result["decision"]["floor"],
                      "candidate": {k: result["numeric"]["candidate"][k]
                                    for k in ("kl_mean", "kl_p95", "kl_max", "top1_matches")},
                      "retrieval": [result["retrieval"]["strict_correct"],
                                    result["retrieval"]["candidate_correct"]]}, indent=1))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
