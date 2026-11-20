#!/usr/bin/env python3
"""Compare public AR/MTP/EAGLE/DSpine streaming on one host, without tuning.

All methods use the same raw user-message text, greedy fixed output count, BF16
KV and a declared execution profile (production by default). Prefix caching is off. Three resident LLM owners
remain loaded; AR and MTP share one owner. Calls are sequential, with rotating
method order. First-use requests after prepare are recorded separately, not
called process-cold: OS and JIT caches are not cleared.

Primary rate pools output tokens / external request wall. Arrival-decode rate
pools (outputs - 1) / (last token arrival - first token arrival). This measures
public delivery, not kernel-only decode. Internal timers are diagnostic only.
Each live chunk must correspond to one token; final metadata may arrive alone.
Raw IDs, arrival samples, routes, profiles and per-request results are preserved.
Same-method repeats must be exact. Cross-route ID equality is diagnostic for
production and binding for this harness's strict comparison. This harness alone
does not establish production numerical or task-quality equivalence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ("ar", "mtp", "eagle4", "dspine")


def method_order(index):
    offset = index % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def consume_stream(stream, *, expected_tokens, clock=time.perf_counter):
    started = clock()
    events, terminal = [], None
    try:
        for chunk in stream:
            elapsed = clock() - started
            if terminal is not None:
                raise ValueError("stream emitted data after terminal chunk")
            events.append((elapsed, chunk))
            if chunk.finish_details is not None:
                terminal = chunk
        wall = clock() - started
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if terminal is None or terminal.generated_token_ids is None:
        raise ValueError("stream has no terminal generated IDs")
    ids = list(terminal.generated_token_ids)
    if len(ids) != expected_tokens or terminal.finish_details.reason != "length":
        raise ValueError("stream has incomplete output or unexpected finish")
    # AR attaches finish to its final token. Speculative service adds a separate
    # empty terminal chunk. Empty nonterminal chunks still represent token bytes.
    if len(events) == len(ids) + 1 and events[-1][1].text == "":
        token_events = events[:-1]
    elif len(events) == len(ids):
        token_events = events
    else:
        raise ValueError("stream is buffered or token chunk accounting differs")
    arrivals = [t for t, _ in token_events]
    decode = arrivals[-1] - arrivals[0]
    if len(ids) < 2 or not math.isfinite(wall) or wall <= 0 or decode <= 0:
        raise ValueError("invalid external timing interval")
    return dict(
        output_ids=ids, text="".join(c.text for _, c in events), wall_s=wall,
        first_token_s=arrivals[0], first_text_s=next((t for t, c in events if c.text), None),
        token_arrival_s=arrivals, arrival_decode_s=decode, transition_tokens=len(ids) - 1,
        finish_reason=terminal.finish_details.reason,
        telemetry=asdict(terminal.telemetry) if terminal.telemetry is not None else None,
    )


def validate_group(rows, previous, prompt_id, *, expected_tokens, execution_profile="strict"):
    if execution_profile not in {"strict", "production"}:
        raise ValueError("unsupported comparison profile")
    if {r["method"] for r in rows} != set(METHODS) or len(rows) != len(METHODS):
        raise ValueError("incomplete method group")
    ar = next(r["output_ids"] for r in rows if r["method"] == "ar")
    for row in rows:
        ids = row["output_ids"]
        teacher = previous.setdefault((prompt_id, row["method"]), list(ids))
        row["repeat_exact"] = len(ids) == expected_tokens and ids == teacher
        row["exact"] = len(ids) == expected_tokens and ids == ar
        row["first_difference_vs_ar"] = next(
            (i for i, (a, b) in enumerate(zip(ids, ar)) if a != b),
            None if len(ids) == len(ar) else min(len(ids), len(ar)),
        )
    if not all(row["repeat_exact"] for row in rows):
        raise ValueError(f"public output length or repeat exactness mismatch: {prompt_id}")
    if execution_profile == "strict" and not all(row["exact"] for row in rows):
        raise ValueError(f"strict cross-method exactness mismatch: {prompt_id}")


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], "all"].append(row)
        grouped[row["method"], row["category"]].append(row)
    result = defaultdict(dict)
    for (method, category), group in sorted(grouped.items()):
        tokens = sum(len(r["output_ids"]) for r in group)
        wall = sum(r["wall_s"] for r in group)
        transitions = sum(r["transition_tokens"] for r in group)
        decode = sum(r["arrival_decode_s"] for r in group)
        text_times = [r["first_text_s"] for r in group if r["first_text_s"] is not None]
        result[method][category] = dict(
            requests=len(group), output_tokens=tokens, request_wall_s=wall,
            transition_tokens=transitions, arrival_decode_s=decode,
            request_tok_s=tokens / wall, arrival_transition_tok_s=transitions / decode,
            first_token_s_median=statistics.median(r["first_token_s"] for r in group),
            first_text_s_median=statistics.median(text_times) if text_times else None,
            exact_all=all(r.get("exact", False) for r in group),
            repeat_exact_all=all(r.get("repeat_exact", False) for r in group),
            cross_route_different_requests=sum(not r.get("exact", False) for r in group),
        )
    return dict(result)


def hardware_snapshot():
    paths = []
    for device in Path("/sys/class/drm").glob("card[0-9]*/device"):
        paths.extend(device.glob("pp_dpm_sclk"))
        paths.extend(device.glob("gpu_busy_percent"))
        for pattern in ("temp*_input", "power*_average", "freq*_input"):
            paths.extend(device.glob(f"hwmon/hwmon*/{pattern}"))
    result = {}
    for path in paths:
        try:
            result[str(path)] = path.read_text().strip()
        except OSError:
            pass
    return result


def run_one(llm, method, prompt, tokens):
    from hipengine import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=tokens, ignore_eos=True)
    before = hardware_snapshot()
    if method == "ar":
        stream = llm.stream_detailed(prompt, params)
    elif method == "mtp":
        stream = llm.stream_speculative_mtp_detailed([prompt], params)
    else:
        stream = llm.stream_speculative_detailed(prompt, params)
    result = consume_stream(stream, expected_tokens=tokens)
    result.update(method=method, hardware_before=before, hardware_after=hardware_snapshot())
    telemetry = result["telemetry"] or {}
    diagnostics = telemetry.get("diagnostics") or {}
    route = (telemetry.get("decode_state") or {}).get("execution_path")
    mtp = diagnostics.get("specdec2_mtp2") or {}
    if method == "mtp":
        engaged = (route == "gguf_specdec2_mtp2" and mtp.get("cycles", 0) > 0
                   and mtp.get("generated_draft_tokens", 0) > 0
                   and mtp.get("candidate_budget") == 3
                   and bool(mtp.get("selected_depth_histogram"))
                   and set(mtp["selected_depth_histogram"]) <= {"1", "2", "3"})
    elif method in ("eagle4", "dspine"):
        engaged = (str(route).startswith(f"qwen35_gguf_{method}_")
                   and diagnostics.get("draft_forwards", 0) > 0
                   and diagnostics.get("graph_verify_cycles") == diagnostics["draft_forwards"])
    else:
        engaged = (route == "gguf_packed_ar_server_decode"
                   and not mtp.get("cycles", 0) and not mtp.get("generated_draft_tokens", 0))
    result["intended_route_ran"] = engaged
    return result


def owner_details(llm):
    owner = llm._get_text_generator()
    provider = getattr(owner, "_speculative_provider", None)
    native = getattr(provider, "_native", None)
    pools = dict(getattr(owner, "_shared_session_pool", {}))
    runner = getattr(owner, "_resident_model_runner", None)
    if runner is not None:
        pools["resident_runner_leases"] = runner._resident_sessions()
    sessions = []
    for key, pool in pools.items():
        for session in pool:
            sessions.append(dict(
                pool=str(key), target_verifier_production_q4_rowtile=session.target_verifier_production_q4_rowtile,
                fp16_recurrent_state=session.runner.fp16_recurrent_state,
                short_c1_attn_variant=getattr(session.runner, "target_short_c1_attn_variant", None),
                use_wmma_prefill=session.use_wmma_prefill, use_gemv_decode=session.use_gemv_decode,
                kv_storage_dtype=str(session.kv_storage_dtype),
            ))
    return dict(profile=str(owner.execution_profile), manifest=llm.execution_profile_manifest,
                manifest_sha256=llm.execution_profile_manifest_sha256,
                capabilities=llm.speculative_capabilities,
                candidate_budget=llm.speculative_candidate_budget,
                q8_projection_names=sorted(getattr(native, "_q8_tiles", {})), sessions=sessions)


def run(args):
    from hipengine import LLM
    from hipengine.benchmark.provenance import collect_artifact_provenance, collect_model_identity
    from scripts.gguf_mtp_category_bench import load_prompt_rows
    from hipengine.speculative import native_cycle_graph  # noqa: F401

    if args.repetitions < 1 or args.tokens < 2 or args.limit < 0:
        raise ValueError("require positive repetitions, tokens >= 2 and limit >= 0")
    prompts = []
    for path in args.prompts:
        for row in load_prompt_rows(path):
            prompts.append(dict(id=row["id"], category=row["category"], text=row["prompt"], source=str(path)))
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("prompt IDs must be unique across files")
    if args.limit:
        prompts = prompts[:args.limit]
    owners, rows, first_use = {}, [], []
    report = dict(schema=2, kind="qwen38_public_stream_comparison", completed=False,
                  status="same_host_diagnostic", performance_claim=False, rows=rows, first_use=first_use,
                  protocol=dict(tokens=args.tokens, repetitions=args.repetitions,
                                execution_profile=args.execution_profile,
                                correctness="same-method repeat exactness; production cross-route IDs diagnostic",
                                quality_equivalence="not established by this timing harness",
                                order="rotate AR/MTP/EAGLE/DSpine each prompt/repetition group",
                                prompt_format="raw user content; no chat template", prefix_cache="off",
                                warmup="one 33-token first-use request per method after prepare",
                                first_use_scope="post-prepare; shared AR/MTP target; OS/JIT caches not cleared",
                                timing="external request wall and first-to-last token arrival",
                                resident_owners="three targets; AR/MTP share one; sequential requests",
                                full_suite=not args.limit, concurrency=1, kv_storage="bf16",
                                candidate_budgets=dict(mtp=3, eagle4=3, dspine=5)),
                  prompts=prompts, prepare_s={}, owners={})
    report["provenance"] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend=args.backend, model_path=args.model,
        quant="Q4_K_M", kv_dtype="bf16", command=[sys.executable, *sys.argv],
        environment={k: v for k, v in os.environ.items() if k.startswith("HIPENGINE_")},
        warmups=1, repetitions=args.repetitions, timing_protocol=report["protocol"]["timing"],
        profiler={"enabled": False},
    )
    report["drafter_identity"] = {m: collect_model_identity(getattr(args, m)) for m in ("eagle4", "dspine")}
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), *args.prompts]}

    def save():
        report["summary"] = summarize(rows)
        report["exact_all"] = (report["completed"] and bool(rows)
                               and all(r.get("exact", False) and r["intended_route_ran"]
                                       for r in rows + first_use))
        report["repeat_exact_all"] = (report["completed"] and bool(rows)
                                      and all(r.get("repeat_exact", False) for r in rows))
        report["intended_routes_all"] = bool(rows) and all(r["intended_route_ran"] for r in rows + first_use)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    try:
        for method in ("ar", "eagle4", "dspine"):
            started = time.perf_counter()
            kwargs = {} if method == "ar" else dict(speculative_provider=method, draft_model=str(getattr(args, method)))
            llm = LLM(args.model, backend=args.backend, max_active_requests=1,
                      max_sequence_length=args.context, execution_profile=args.execution_profile, prefix_cache="off",
                      speculative_candidate_budget=5 if method == "dspine" else 3, **kwargs)
            owners[method] = llm
            llm.prepare(max_sequence_length=args.context)
            report["prepare_s"][method] = time.perf_counter() - started
            print(f"prepared {method}", flush=True)
        owners["mtp"] = owners["ar"]
        tokenizer = owners["ar"]._get_text_generator()
        for prompt in prompts:
            prompt["ids"] = list(tokenizer.tokenize(prompt["text"]))
            if len(prompt["ids"]) + max(args.tokens, 33) + 32 > args.context:
                raise ValueError("prompt/output/workspace exceeds configured context")
        warm_ids = tuple(tokenizer.tokenize("Explain why the sky is blue in one sentence."))
        for method in METHODS:
            row = run_one(owners[method], method, warm_ids, 33)
            first_use.append(row)
            report["owners"][method] = owner_details(owners[method])
            print(f"first use {method}: {row['wall_s']:.3f}s", flush=True)
            save()
            if not row["intended_route_ran"]:
                raise ValueError(f"intended {method} route did not run; see first_use telemetry")
        validate_group(first_use, {}, "first_use", expected_tokens=33, execution_profile=args.execution_profile)
        previous = {}
        for rep in range(args.repetitions):
            for index, prompt in enumerate(prompts):
                group = []
                for method in method_order(rep * len(prompts) + index):
                    row = run_one(owners[method], method, tuple(prompt["ids"]), args.tokens)
                    row.update(prompt_id=prompt["id"], category=prompt["category"], repetition=rep,
                               prompt_tokens=len(prompt["ids"]), order_index=len(group))
                    rows.append(row)
                    group.append(row)
                    print(f"rep {rep} {prompt['id']} {method}: {row['wall_s']:.3f}s", flush=True)
                    save()
                    if not row["intended_route_ran"]:
                        raise ValueError(f"intended {method} route did not run; see last row telemetry")
                validate_group(group, previous, prompt["id"], expected_tokens=args.tokens,
                               execution_profile=args.execution_profile)
                save()
        report["completed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            for method in ("ar", "eagle4", "dspine"):
                if method in owners:
                    owners[method].close()
        finally:
            save()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--execution-profile", choices=("production", "strict"), default="production")
    parser.add_argument("--eagle4", type=Path, required=True)
    parser.add_argument("--dspine", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, nargs="+", default=[
        ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"])
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="smoke only; 0 runs the full suite")
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
