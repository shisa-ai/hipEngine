#!/usr/bin/env python3
"""Capture actual packed-C1 logits and replay their contexts under strict AR.

This is a conditional-logit diagnostic, not full profile qualification. Inputs
come from real candidate/rejection trajectories, not a strict-teacher schedule.
Raw NPZ files are local-only. No timing or performance claims are produced.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from scripts.quant_quality.metrics import per_row_metrics


def validate_capture(*, prefix, tokens, position, logical_rows, logits) -> np.ndarray:
    """Validate ownership coordinates and return active rows, never padding."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("full-vocabulary logits must be a nonempty matrix")
    if len(prefix) != int(position) or not prefix:
        raise ValueError("prefix length must equal the warm target position")
    if values.shape[0] != len(tokens) or not 1 <= int(logical_rows) <= len(tokens):
        raise ValueError("logical/padded row mapping does not align with token inputs")
    if any(int(t) < 0 or int(t) >= values.shape[1] for t in (*prefix, *tokens)):
        raise ValueError("context or input token outside vocabulary")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite logits (including inactive rows)")
    return np.ascontiguousarray(values[:int(logical_rows)], dtype=np.float32)


def compare_logits(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref, cand = np.asarray(reference), np.asarray(candidate)
    if ref.ndim != 2 or ref.shape != cand.shape or min(ref.shape) <= 0:
        raise ValueError("strict/candidate full-vocabulary shape mismatch")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("nonfinite strict/candidate logits")
    metrics = per_row_metrics(ref, cand, np.argmax(ref, axis=1), top_k=min(5, ref.shape[1]))
    kl = metrics["kl_nats"]
    limits = EvaluationThresholds()
    mean, p95, p99, maximum = float(kl.mean()), *map(float, np.percentile(kl, [95, 99])), float(kl.max())
    top1 = float(metrics["top1_equal"].mean())
    review = np.flatnonzero(kl > limits.review_kl).tolist()
    return {
        "rows": len(kl), "mean_kl": mean, "p95_kl": p95, "p99_kl": p99,
        "max_kl": maximum, "top1": top1, "review_rows": review,
        "review_details": [
            {"row": i, "kl": float(kl[i]),
             "topk_overlap": float(metrics["topk_set_overlap"][i]),
             "strict_top1": int(np.argmax(ref[i])), "candidate_top1": int(np.argmax(cand[i])),
             "strict_margin": (float(np.diff(np.partition(ref[i], -2)[-2:])[0])
                               if ref.shape[1] > 1 else None)}
            for i in review
        ],
        "max_abs_logit_delta": float(metrics["max_abs_logit_delta"].max()),
        "numerical_envelope_passed": bool(
            mean <= limits.mean_kl_max and p95 <= limits.p95_kl_max
            and p99 <= limits.p99_kl_max and maximum <= limits.max_kl_max
            and top1 >= limits.top1_min and not review
        ),
        "full_profile_qualification": False,
    }


def _read_device(ptr: int, shape: tuple[int, ...], dtype, runtime) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
    result = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(result), DeviceBuffer(int(ptr), result.nbytes),
                        result.nbytes, runtime=runtime)
    return result


class PackedC1Capture:
    """Instrument the real adapter and its target; never replace its execution."""

    def __init__(self, directory: Path, *, check_state: bool = False, check_commit: bool = False):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.check_state = check_state or check_commit
        self.check_commit = check_commit
        self.current = ContextVar("packed_c1_capture", default=None)
        self.records: list[dict[str, Any]] = []
        self.prompt: dict[str, Any] | None = None
        self.restores: list[tuple[Any, str, Any]] = []

    def _patch(self, owner, name, value):
        self.restores.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def install(self) -> None:
        from scripts import gguf_mtp_c1c8_server_bench as bench
        from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

        suite = bench.load_prompt_suite(ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl")
        prompts = {r["rendered_prompt"]: r for r in suite}
        arm = bench._run_arm
        execute = Qwen35GGUFMTP2Adapter._execute_target_frontier_batch
        verify = Qwen35GGUFResidentSession.verify_target_blocks_batch
        enqueue = Qwen35GGUFResidentSession._enqueue_target_block_rows_from_hidden

        def measured_arm(*args, **kwargs):
            if kwargs["width"] != 1:
                raise ValueError("conditional capture supports sequential C1 arms only")
            self.prompt = prompts[kwargs["prompt"]]
            try:
                return arm(*args, **kwargs)
            finally:
                self.prompt = None

        def target(adapter, plan, *args, **kwargs):
            if len(plan.speculative_request_ids) != 1 or self.prompt is None:
                raise ValueError("capture requires one scoped physical request")
            rid = plan.speculative_request_ids[0]
            row = adapter.owner._row(rid)
            generated = tuple(row.slot.generated_ids)
            if not generated or not adapter._physical_c1_request(rid):
                raise ValueError("capture did not reach evidence-owned packed C1")
            artifact = adapter.generator._kv_model_artifact_identity()
            if not artifact.content_verified or not artifact.sha256:
                raise ValueError("candidate model content is not verified")
            context = {
                "model_sha256": artifact.sha256,
                "prompt_length": len(row.prompt_ids),
                "prefix": tuple(row.prompt_ids) + generated[:-1],
                "root": int(generated[-1]), "fresh_logits": False,
                "logical_rows": 1 + int(plan.candidate_counts[plan.request_ids.index(rid)]),
                "request_id": int(rid), "slot": int(plan.resident_slots[plan.request_ids.index(rid)]),
                "capacity": int(adapter.owner.capacity), "cycle_id": int(plan.cycle_id),
                "prompt_id": self.prompt["id"], "category": self.prompt["category"],
                "profile": str(adapter.generator.execution_profile),
                "manifest_sha256": str(adapter.generator.execution_profile_manifest_sha256),
            }
            if self.check_commit:
                context["remaining_decode"] = int(row.request.max_tokens) - len(generated)
            token = self.current.set(context)
            try:
                output = execute(adapter, plan, *args, **kwargs)
                if self.check_commit:
                    index = context.get("record_index")
                    if index is None or not self.records[index].get("selected_commit", {}).get("passed"):
                        raise ValueError("target execution omitted checked selected commit")
                return output
            finally:
                self.current.reset(token)

        def head(owner, hidden_ptr, rows, **kwargs):
            result = enqueue(owner, hidden_ptr, rows, **kwargs)
            context = self.current.get()
            if context is not None:
                # Allocation alone is not evidence: direct-top1 can leave it stale.
                path = owner._last_packed_lm_head_decode_path
                if path not in {"q6_rowtile_f32_logits", "row_linear_f32_logits"}:
                    raise ValueError(f"actual head did not produce fresh full logits: {path}")
                context["fresh_logits"] = True
                context["head_path"] = path
            return result

        def packed(owner, jobs, **kwargs):
            context = self.current.get()
            if context is None:
                return verify(owner, jobs, **kwargs)
            if len(jobs) != 1 or not kwargs.get("device_result"):
                raise ValueError("capture requires one staged device-result target job")
            job = jobs[0]
            tokens = list(job["input_token_ids"])
            device = job.get("candidate_token_ids_device")
            if device is not None:
                tokens[1:] = _read_device(device.ptr, (len(tokens)-1,), np.int32, owner.runtime).tolist()
            position = int(job["session"].position)
            if int(tokens[0]) != context["root"]:
                raise ValueError("target root does not match authoritative generated tail")
            context["fresh_logits"] = False
            before_state = None
            if self.check_state:
                from scripts.qwen38_packed_c1_state import (
                    snapshot_committed_state, assert_committed_state_unchanged,
                )
                if not all(job.get(k) for k in (
                    "capture_linear_state_rows", "defer_linear_state_commit", "defer_state_scatter",
                )):
                    raise ValueError("state isolation requires deferred packed verification")
                before_state = snapshot_committed_state(job["session"])
            result = verify(owner, jobs, **kwargs)
            if before_state is not None:
                after_state = snapshot_committed_state(job["session"])
                assert_committed_state_unchanged(before_state, after_state)
                context["pre_accept_state_isolation"] = {
                    "passed": True, "checked_buffers": len(before_state["buffers"]),
                    "checked_bytes": sum(b["checked_nbytes"] for b in before_state["buffers"].values()),
                    "position": before_state["position"],
                }
            if not context["fresh_logits"] or owner._verify_logits_buf is None:
                raise ValueError("packed target did not write fresh full logits")
            all_logits = _read_device(owner._verify_logits_buf.ptr,
                                      (len(tokens), owner.runner.vocab_size), np.float32, owner.runtime)
            logits = validate_capture(prefix=context["prefix"], tokens=tokens, position=position,
                                      logical_rows=context["logical_rows"], logits=all_logits)
            index = len(self.records)
            name = f"cycle-{index:05d}.npz"
            np.savez(self.directory / name, logits=logits)
            self.records.append({
                **{k: v for k, v in context.items() if k not in {"fresh_logits", "root"}},
                "position": position, "tokens": tokens[:context["logical_rows"]],
                "physical_rows": len(tokens), "logits_file": name,
                "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            })
            context["record_index"] = index
            return result

        if self.check_commit:
            commit = Qwen35GGUFResidentSession._commit_deferred_packed_verify_states_batch_device

            def checked_commit(owner, results, sessions, *, accept_buffers, **kwargs):
                context = self.current.get()
                if context is None:
                    return commit(owner, results, sessions, accept_buffers=accept_buffers, **kwargs)
                from scripts.qwen38_packed_c1_state import (
                    selected_prefix, selected_state_sources, snapshot_committed_state,
                    selected_aux_sources, assert_aux_commit,
                )
                if len(results) != 1 or len(sessions) != 1:
                    raise ValueError("selected commit requires physical C1")
                record = self.records[context["record_index"]]
                with np.load(self.directory / record["logits_file"]) as data:
                    accepted = selected_prefix(record["tokens"], data["logits"].argmax(axis=1), context["remaining_decode"])
                result = results[0]
                if int(result.start_position) != record["position"]:
                    raise ValueError("selected commit start position changed")
                for name, expected in (("accepted_counts", accepted),
                                       ("commit_positions", record["position"] + accepted)):
                    actual = _read_device(getattr(accept_buffers, name).ptr, (1,), np.int32, owner.runtime)
                    if int(actual[0]) != expected:
                        raise ValueError(f"device acceptance differs from CPU oracle: {name}")
                expected_state = selected_state_sources(owner, sessions[0], selected_row=int(result.row_start) + accepted)
                expected_aux = selected_aux_sources(sessions[0], result, accepted=accepted)
                from scripts.qwen38_packed_c1_kv import selected_kv_sources, assert_kv_commit
                try:
                    expected_kv = selected_kv_sources(sessions[0], result, accepted=accepted)
                except Exception as error:
                    record['kv_error'] = f'{type(error).__name__}: {error}'
                    raise
                output = commit(owner, results, sessions, accept_buffers=accept_buffers, **kwargs)
                assert_aux_commit(sessions[0], expected_aux)
                try:
                    assert_kv_commit(sessions[0], expected_kv)
                except Exception as error:
                    record['kv_error'] = f'{type(error).__name__}: {error}'
                    raise
                after = snapshot_committed_state(sessions[0])
                for name, expected in expected_state.items():
                    actual = after["buffers"][name]
                    if (actual["ptr"], actual["allocation_nbytes"], actual["blake2b_128"]) != (expected["ptr"], expected["nbytes"], expected["hash"]):
                        raise ValueError(f"selected commit state mismatch: {name}")
                record["selected_commit"] = dict(passed=True, accepted=accepted, checked_buffers=len(expected_state))
                record["aux_commit"] = dict(passed=True, checked_buffers=len(expected_aux))
                record["kv_commit"] = dict(passed=True, checked_buffers=len(expected_kv['buffers']))
                return output

            self._patch(Qwen35GGUFResidentSession, "_commit_deferred_packed_verify_states_batch_device", checked_commit)

        self._patch(bench, "_run_arm", measured_arm)
        self._patch(Qwen35GGUFMTP2Adapter, "_execute_target_frontier_batch", target)
        self._patch(Qwen35GGUFResidentSession, "verify_target_blocks_batch", packed)
        self._patch(Qwen35GGUFResidentSession, "_enqueue_target_block_rows_from_hidden", head)

    def close(self, *, success: bool) -> None:
        for owner, name, original in reversed(self.restores):
            setattr(owner, name, original)
        (self.directory / "capture.json").write_text(json.dumps({
            "kind": "packed_c1_conditional_logits", "full_profile_qualification": False,
            "complete": bool(success and self.records), "records": self.records,
            "limitation": "Actual candidate contexts, not strict-teacher-driven state/task qualification",
        }, indent=2) + "\n")


def capture(args) -> None:
    recorder = PackedC1Capture(args.directory, check_state=args.check_state, check_commit=args.check_commit)
    success = False
    previous = sys.argv
    try:
        recorder.install()
        sys.argv = ["check_packed_c1.py", "--model", str(args.model),
                    "--capacity", str(args.capacity), "--budget", str(args.budget),
                    "--output", str(args.directory / "route.json")]
        runpy.run_path(str(ROOT / "campaign-artifacts/review/check_packed_c1.py"), run_name="__main__")
        success = True
    finally:
        sys.argv = previous
        recorder.close(success=success)


def replay_strict(session, record) -> np.ndarray:
    """Preserve prompt/decode boundaries in the independent strict trajectory."""
    prefix = tuple(record["prefix"])
    prompt_length = int(record["prompt_length"])
    if not 1 <= prompt_length <= len(prefix) or len(prefix) != record["position"]:
        raise ValueError("strict replay prompt/prefix coordinates are invalid")
    session.reset()
    session.prefill(prefix[:prompt_length], return_logits=False)
    for token in prefix[prompt_length:]:
        session.step(int(token), return_logits=False)
    if int(session.position) != record["position"]:
        raise ValueError("strict replay prefix cursor mismatch")
    rows = []
    for token in record["tokens"]:
        logits = np.asarray(session.step(int(token), return_logits=True).logits)
        if logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 1 or logits.size == 0:
            raise ValueError("strict scalar step must return one full-vocabulary row")
        rows.append(logits.copy())
    result = np.stack(rows)
    if int(session.position) != record["position"] + len(record["tokens"]):
        raise ValueError("strict replay final cursor mismatch")
    return result


def reference(args) -> None:
    from hipengine import LLM

    payload = json.loads((args.directory / "capture.json").read_text())
    if not payload["complete"] or not payload["records"]:
        raise ValueError("refusing partial/empty actual-route capture")
    llm = LLM(str(args.model), backend="hip_gfx1100", execution_profile="strict",
              max_sequence_length=1024, max_active_requests=1)
    comparisons = []
    scope_arrays: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    try:
        llm.prepare(max_sequence_length=1024)
        generator = llm._get_text_generator()
        if str(generator.execution_profile) != "strict":
            raise ValueError("independent reference is not strict-profile")
        artifact = generator._kv_model_artifact_identity()
        if not artifact.content_verified or {r["model_sha256"] for r in payload["records"]} != {artifact.sha256}:
            raise ValueError("strict and candidate model content differs")
        with generator._resident_session_scope(
            shared_runner=generator._get_shared_runner(), pool_name="packed_c1_strict_reference",
        ) as (session, _reused):
            for record in payload["records"]:
                with np.load(args.directory / record["logits_file"], allow_pickle=False) as raw:
                    candidate = raw["logits"]
                if hashlib.sha256(candidate.tobytes()).hexdigest() != record["logits_sha256"]:
                    raise ValueError("capture logits digest mismatch")
                strict = replay_strict(session, record)
                comparison = compare_logits(strict, candidate)
                comparisons.append({"file": record["logits_file"], "prompt_id": record["prompt_id"],
                                    **comparison})
                for scope in ("all", "category:" + record["category"], "prompt:" + record["prompt_id"]):
                    scope_arrays.setdefault(scope, []).append((strict, candidate))
                print(json.dumps({"cycle": len(comparisons), **comparison}), flush=True)
        scopes = {name: compare_logits(np.concatenate([r for r, _ in pairs]),
                                        np.concatenate([c for _, c in pairs]))
                  for name, pairs in scope_arrays.items()}
        result = {
            "kind": "packed_c1_conditional_logit_comparison", "full_profile_qualification": False,
            "model": str(args.model), "strict_manifest_sha256": llm.execution_profile_manifest_sha256,
            "candidate_manifests": sorted({r["manifest_sha256"] for r in payload["records"]}),
            "thresholds": asdict(EvaluationThresholds()), "cycles": comparisons, "scopes": scopes,
            "numerical_envelope_passed": all(r["numerical_envelope_passed"] for r in scopes.values()),
            "limitation": "Not a strict-teacher schedule, state/lifecycle/task or determinism certificate",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not result["numerical_envelope_passed"]:
            raise SystemExit(1)
    finally:
        llm.close()


def compare_captures(directories: list[Path]) -> dict[str, Any]:
    """Compare complete independent captures, checking arrays rather than hashes alone.

    Distinct directories prevent accidental self-comparison, but process independence
    still requires the recorded capture commands. This checks only captured surfaces.
    """
    paths = [Path(p).resolve() for p in directories]
    if len(paths) < 3 or len(set(paths)) != len(paths):
        raise ValueError("at least three distinct capture directories are required")
    payloads = [json.loads((p / "capture.json").read_text()) for p in paths]
    if any(not p.get("complete") or not p.get("records") for p in payloads):
        raise ValueError("refusing partial/empty repeat captures")
    records = payloads[0]["records"]
    if any(len(p["records"]) != len(records) for p in payloads):
        raise ValueError("repeat cycle counts differ")
    rows = 0
    for index, record in enumerate(records):
        baseline = None
        for path, payload in zip(paths, payloads):
            other = payload["records"][index]
            with np.load(path / other["logits_file"], allow_pickle=False) as raw:
                values = raw["logits"]
            if (values.ndim != 2 or values.shape[0] != other["logical_rows"]
                    or values.shape[0] <= 0 or values.shape[1] <= 0
                    or values.dtype != np.float32 or not np.isfinite(values).all()):
                raise ValueError("invalid repeat full-logit array")
            data = values.tobytes()
            if hashlib.sha256(data).hexdigest() != other["logits_sha256"]:
                raise ValueError("repeat logits digest mismatch")
            if baseline is None:
                baseline = (values.shape, data)
            elif (values.shape, data) != baseline:
                raise ValueError("repeat full-logit bytes differ")
            if other != record:
                raise ValueError("repeat capture metadata differs")
        rows += record["logical_rows"]
    return {
        "runs": len(paths), "cycles_per_run": len(records), "active_rows_per_run": rows,
        "prompts": len({r["prompt_id"] for r in records}), "metadata_exact": True,
        "full_logit_bytes_exact": True, "finite": True, "digests_verified": True,
        "full_profile_qualification": False, "directories": [str(p) for p in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "reference", "compare"))
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--check-commit", action="store_true",
                        help="Check GPU acceptance and selected Conv/GDN/hidden state against CPU-selected rows")
    parser.add_argument("--check-state", action="store_true",
                        help="Check committed resident state is unchanged before acceptance")
    parser.add_argument("--repeat-directory", type=Path, action="append", default=[],
                        help="Additional independent capture; repeat for each run in compare mode")
    parser.add_argument("--capacity", type=int, choices=(1, 2, 8), default=1)
    parser.add_argument("--budget", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--output", type=Path, default=Path("/tmp/packed-c1-conditional-numerics.json"))
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare_captures([args.directory, *args.repeat_directory])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        (capture if args.mode == "capture" else reference)(args)


if __name__ == "__main__":
    main()
