"""Gemma MTP attribution on the exact no-thinking HTTP request.

Run public AR/MTP, then compare full-vocabulary logits at identical committed
prefixes and replay each verifier block as scalar steps from its original KV
schedule. Inspect rewind metadata and live-prefix bytes separately from numeric
differences. This is a diagnostic, not a throughput or profile-promotion gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gemma4_teacher_forced_gate import row_kl_divergence


def paired_summary(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.ndim != 2 or a.shape != b.shape or not a.size:
        raise ValueError("paired logits must have equal nonempty (rows, vocab) shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite paired logits")
    kl = np.array([row_kl_divergence(x, y) for x, y in zip(a, b)])
    flips = np.flatnonzero(a.argmax(1) != b.argmax(1)).tolist()
    return {
        "rows": len(a), "bit_equal": bool(np.array_equal(a, b)),
        "kl_mean": float(kl.mean()), "kl_p95": float(np.quantile(kl, .95)),
        "kl_median": float(np.median(kl)), "kl_p90": float(np.quantile(kl, .90)),
        "kl_above_0_05_rate": float(np.mean(kl > .05)),
        "top1_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
        "kl_p99": float(np.quantile(kl, .99)), "kl_max": float(kl.max()),
        "top1_flips": flips,
        "flip_rows": [{"row": i, "reference_id": int(a[i].argmax()),
                       "candidate_id": int(b[i].argmax()), "kl": float(kl[i]),
                       "reference_logit_gap": float(np.sort(a[i])[-1] - np.sort(a[i])[-2])}
                      for i in flips],
    }


def committed_rows(prompt, generated, trace):
    """Check exact token ownership/cursors and align next-token prediction rows."""
    if not trace or trace[0]["tokens"] != list(prompt) or trace[0]["start"] != 0:
        raise ValueError("trace does not begin with the requested prompt")
    history = list(prompt)
    rows = [np.asarray(trace[0]["logits"]).reshape(-1)]
    for block in trace[1:]:
        start, keep = block["start"], block["keep"]
        if start != len(history) or not start < keep <= start + len(block["tokens"]):
            raise ValueError("invalid verifier cursor/rewind")
        history.extend(block["tokens"])
        del history[keep:]
        if history != list(prompt) + list(generated[:keep - len(prompt)]):
            raise ValueError("committed verifier input differs from published token history")
        logits = np.asarray(block["logits"]).reshape(len(block["tokens"]), -1)
        rows.extend(logits[:keep - start])
    if len(history) != len(prompt) + len(generated) - 1 or len(rows) != len(generated):
        raise ValueError("final pending-token/row accounting differs")
    result = np.stack(rows)
    if result.argmax(1).tolist() != list(generated):
        raise ValueError("published greedy tokens do not match their own target rows")
    return result


def kv_prefix_snapshot(runner, position):
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    result = []
    for layer in range(runner.layer_count):
        view = runner.shared_kv(layer)
        count = position - view.base
        if not 0 <= count <= view.capacity:
            raise ValueError("live KV prefix is outside owned capacity")
        size = count * view.num_kv_heads * view.head_dim * 2
        hashes = []
        for ptr in (view.key_cache, view.value_cache):
            host = np.empty(size, dtype=np.uint8)
            if size:
                runtime.memcpy(host.ctypes.data, ptr, size, HipMemcpyKind.DEVICE_TO_HOST)
            hashes.append(hashlib.sha256(host.tobytes()).hexdigest())
        result.append((view.key_cache, view.value_cache, view.base, count, *hashes))
    return result


@contextmanager
def capture(runner):
    trace = []
    forward, rewind = runner.forward, runner.rewind

    def observed_forward(tokens, **kwargs):
        start = runner.position
        logits = forward(tokens, **kwargs)
        if runner.position != start + len(tokens):
            raise ValueError("forward cursor does not match consumed tokens")
        trace.append({"start": start, "tokens": list(tokens), "kwargs": kwargs,
                      "logits": np.asarray(logits).copy()})
        return logits

    def observed_rewind(position):
        before = kv_prefix_snapshot(runner, position)
        rewind(position)
        after = kv_prefix_snapshot(runner, position)
        if before != after or runner.position != position:
            raise ValueError("rewind changed retained KV bytes, ownership, or cursor")
        if any(kv.write_offset != position for kv in runner._kv):
            raise ValueError("rewind did not publish every layer's write cursor")
        trace[-1].update(keep=position, kv_prefix_unchanged=True)

    runner.forward, runner.rewind = observed_forward, observed_rewind
    try:
        yield trace
    finally:
        runner.forward, runner.rewind = forward, rewind


def serial_teacher(runner, prompt, generated):
    runner.reset()
    rows = [runner.forward(prompt).copy()]
    for token in generated[:-1]:
        rows.append(runner.forward([token]).copy())
    return np.stack(rows)


def local_scalar_replays(runner, trace):
    """Each arm begins with identical preceding multi-row blocks and rewinds."""
    scalar, batched = [], []
    for index, block in enumerate(trace[1:], 1):
        runner.reset()
        for previous in trace[:index]:
            runner.forward(previous["tokens"], **previous["kwargs"])
            if "keep" in previous:
                runner.rewind(previous["keep"])
        if runner.position != block["start"]:
            raise ValueError("same-schedule replay reached a different cursor")
        for token in block["tokens"]:
            scalar.append(runner.forward([token]).copy())
        batched.extend(np.asarray(block["logits"]).reshape(len(block["tokens"]), -1))
    return np.stack(scalar), np.stack(batched)


def teacher_schedule(runner, prompt, chain, trace):
    """Force the strict chain while preserving the captured verify/rewind shapes."""
    runner.reset()
    rows = [runner.forward(prompt).copy()]
    for block in trace[1:]:
        offset = block["start"] - len(prompt)
        tokens = chain[offset:offset + len(block["tokens"])]
        if runner.position != block["start"] or len(tokens) != len(block["tokens"]):
            raise ValueError("teacher chain cannot fill the captured verifier shape")
        logits = np.asarray(runner.forward(tokens, **block["kwargs"])).reshape(len(tokens), -1)
        rows.extend(logits[:block["keep"] - block["start"]].copy())
        runner.rewind(block["keep"])
    if len(rows) != len(chain):
        raise ValueError("teacher rows are not aligned with every strict position")
    return np.stack(rows)


def cross_profile_teacher(args, profiles, *, prefill_variants=None):
    from hipengine import LLM
    production, strict = profiles
    chain, prompt = strict["mtp_ids"], strict["prompt_ids"]
    if production["prompt_ids"] != prompt:
        raise ValueError("profiles tokenized different prompts")
    baseline = np.load(strict["raw_logits"])["serial_teacher"]
    llm = LLM(args.model, backend=args.backend, execution_profile="production",
              max_sequence_length=args.context, prefix_cache="off",
              speculative_provider="gemma4_mtp", draft_model=args.draft,
              speculative_candidate_budget=4)
    try:
        wrapped = llm._get_text_generator()
        generator = getattr(wrapped, "_inner", wrapped)
        if prefill_variants is not None:
            generator.prefill_attention_variants = tuple(prefill_variants)
        runner = generator._ensure_runner()
        ar = serial_teacher(runner, prompt, chain)
        mtp = teacher_schedule(runner, prompt, chain, production["trace"])
        raw = args.directory / (
            "strict-prefill-cross-profile.npz" if prefill_variants is not None
            else "strict-teacher-cross-profile.npz"
        )
        np.savez_compressed(raw, strict=baseline, production_ar=ar, production_mtp=mtp)
        return {"teacher_ids": chain, "prefill_variants_override": prefill_variants,
                "strict_vs_production_ar": paired_summary(baseline, ar),
                "strict_vs_production_mtp": paired_summary(baseline, mtp),
                "production_ar_vs_mtp": paired_summary(ar, mtp),
                "raw_logits": str(raw), "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
    finally:
        llm.close()


def run_profile(args, profile, request):
    from hipengine import LLM, SamplingParams
    llm = LLM(args.model, backend=args.backend, execution_profile=profile,
              max_sequence_length=args.context, prefix_cache="off",
              speculative_provider="gemma4_mtp", draft_model=args.draft,
              speculative_candidate_budget=4)
    try:
        wrapped = llm._get_text_generator()
        generator = getattr(wrapped, "_inner", wrapped)
        prompt = list(generator.tokenize(generator.render_chat_prompt(
            request["messages"], enable_thinking=False)))
        params = SamplingParams(max_tokens=request["max_tokens"], temperature=0, seed=0)
        plain = llm.generate_detailed([prompt], params)[0]
        runner = generator._ensure_runner()
        with capture(runner) as trace:
            output = llm.generate_speculative_mtp_detailed([prompt], params)[0]
        ids, ar = list(output.generated_token_ids), list(plain.generated_token_ids)
        actual = committed_rows(prompt, ids, trace)
        serial = serial_teacher(runner, prompt, ids)
        local, batch = local_scalar_replays(runner, trace)
        # Same public request, same schedule, three MTP executions in total,
        # plus one replay after a different request on the same runner.
        repeats = [list(llm.generate_speculative_mtp_detailed([prompt], params)[0].generated_token_ids)
                   == ids for _ in range(2)]
        llm.generate(["Return the word unrelated."], SamplingParams(max_tokens=8, temperature=0))
        isolated = list(llm.generate_speculative_mtp_detailed([prompt], params)[0].generated_token_ids) == ids
        raw = args.directory / f"{profile}-logits.npz"
        np.savez_compressed(raw, mtp=actual, serial_teacher=serial, local_scalar=local, local_batch=batch)
        differences = [i for i, (a, b) in enumerate(zip(ar, ids)) if a != b]
        result = {"profile": profile, "manifest": llm.execution_profile_manifest,
                  "verify_dense_exact": runner.verify_dense_exact,
                  "prompt_ids": prompt, "ar_ids": ar, "mtp_ids": ids,
                  "ar_text": plain.text, "mtp_text": output.text,
                  "first_generated_difference": differences[0] if differences else None,
                  "generated_ids_equal": ar == ids,
                  "same_prefix_serial_vs_mtp": paired_summary(serial, actual),
                  "same_schedule_prefix_local_scalar_vs_batch": paired_summary(local, batch),
                  "ownership_checks_passed": True, "repeat_equal": repeats,
                  "after_unrelated_equal": isolated,
                  "trace": [{k: v for k, v in row.items() if k != "logits"} for row in trace],
                  "raw_logits": str(raw), "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
        if not all(repeats) or not isolated or (profile == "strict" and ar != ids):
            raise ValueError(f"{profile}: repeatability, isolation, or strict equality failed")
        return result
    finally:
        llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--source", type=Path, default=ROOT / "benchmarks/results/2026-10-17-gemma-mtp-http-controls.json")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    for model_path in (args.model, args.draft):
        if not Path(model_path).is_file():
            parser.error(f"missing model or assistant file: {model_path}")
    args.directory.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_text())
    request = next(row["request"] for row in source["rows"] if row["label"] == "thinking_off_mtp")
    from hipengine.benchmark.provenance import collect_artifact_provenance, collect_model_identity
    report = {"kind": "gemma_no_thinking_mtp_attribution", "performance_claim": False,
              "scope": "one observed request; diagnostic, not multicategory promotion evidence",
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "request": request, "profiles": [],
              "provenance": collect_artifact_provenance(repo_root=ROOT, model_path=args.model,
                    configured_backend=args.backend, kv_dtype="bf16", command=[sys.executable, *sys.argv]),
              "draft_identity": collect_model_identity(args.draft)}
    for profile in ("production", "strict"):
        print(f"profile={profile} starting", flush=True)
        result = run_profile(args, profile, request)
        report["profiles"].append(result)
        report["provenance"]["quant"] = result["manifest"]["quant"]
        (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k in (
            "profile", "first_generated_difference", "generated_ids_equal",
            "same_prefix_serial_vs_mtp", "same_schedule_prefix_local_scalar_vs_batch",
            "repeat_equal", "after_unrelated_equal")}), flush=True)
    print("strict-teacher cross-profile replay starting", flush=True)
    report["strict_teacher"] = cross_profile_teacher(args, report["profiles"])
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["strict_teacher"]), flush=True)
    print("strict-prefill ablation starting", flush=True)
    report["strict_prefill_ablation"] = cross_profile_teacher(
        args, report["profiles"], prefill_variants=("gemma4_plain",))
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["strict_prefill_ablation"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
