#!/usr/bin/env python3
"""DP-2 native cast-sharing gate: exact logits, state, full KV and engagement.

Heldouts are uniformly repeated/truncated to 1024/4096 rows for ownership and
state coverage, not scored as task-quality inputs. Counters wrap native scopes;
no diagnostic GEMM replaces a runtime consumer.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def expected_pairs(tokens: int, chunk: int = 4096) -> int:
    if tokens <= 0 or chunk <= 0:
        raise ValueError("positive tokens and chunk required")
    full, tail = divmod(tokens, chunk)
    # layers13..47: 35 shared pairs and 26 GDN pairs.
    return 61 * (full * (chunk in {1024, 4096}) + int(tail in {1024, 4096}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--paired-repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--decode-steps", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.decode_steps <= 128:
        parser.error("decode steps must be1..128")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    import numpy as np
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.memory import memory_stats, copy_device_to_host, host_array_ptr
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.runtime import gguf_linear as linear, qwen4_exp_runner as runner
    from scripts import qwen4exp_row4_state_gate as gate
    from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
    from scripts.qwen4exp_layer2_profile_gate import _state_summary
    from scripts.gguf_gdn_semantic_gate import _load_suites
    from scripts.gguf_mtp_bench import build_chat_prompt

    check_host()
    os.environ.pop(linear.WIDE_F16_PAIR_ENV, None)
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    production = gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL,
        backend=gate.QWEN4_EXP_BACKEND, quant=gate.QWEN4_EXP_QUANTS[1], profile=gate.ExecutionProfile.PRODUCTION)
    fixture, digest = gate.load_fixture(gate.DEFAULT_FIXTURE)
    report = dict(schema=1, status="running", performance_claim=False, command=sys.argv,
        host=gate._host_metadata(), source=gate._git_metadata(ROOT), model_identity=model_identity(args.model_root),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), fixture_sha256=digest,
        profile="production", manifest_sha256=production.manifest_sha256,
        diagnostic_env=linear.WIDE_F16_PAIR_ENV, chunk4096=True, context4352=True,
        paired_repeats=args.paired_repeats, decode_steps=args.decode_steps, cases=[])
    report["hipengine_artifact_provenance"] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", model_path=args.model_root, model_revision=report["model_identity"]["revision"],
        quant="UD-Q4_K_XL", kv_dtype="bf16", command=sys.argv, build_profile="cached production",
        timing_protocol="native exact state and ownership gate; no performance claim", warmups=0,
        repetitions=args.paired_repeats)
    generator = None
    original_scope = runner.wide_f16_activation_pair
    original_cast = wide.f32_to_f16
    counts = dict(pairs=0, accepted=0, casts=0)
    shapes = {}

    @contextlib.contextmanager
    def counted_scope(*a, **kw):
        with original_scope(*a, **kw) as loan:
            yield loan
            if loan is not None:
                assert loan.accepted == 2, "admitted pair not consumed exactly twice"
                roles = tuple(str(c.weight.spec.slot_path).split(".")[-1] for c in loan.consumers)
                assert roles in {("attn_qkv", "attn_gate"), ("shared_gate", "shared_up")}
                assert kw["rows"] in {1024, 4096} and kw["in_features"] == 2560
                counts["pairs"] += 1
                counts["accepted"] += loan.accepted
                shape = str((roles, kw["rows"], kw["in_features"],
                             kw["out_features_a"], kw["out_features_b"]))
                shapes[shape] = shapes.get(shape, 0) + 1

    def counted_cast(*a, **kw):
        if int(a[2]) in {1024 * 2560, 4096 * 2560}:
            counts["casts"] += 1
        return original_cast(*a, **kw)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
            generator = production.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
                model_path=args.model_root, weight_index=index, model_plugin=gate.resolve_model(index.architecture or ""),
                backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
            runner.wide_f16_activation_pair = counted_scope
            wide.f32_to_f16 = counted_cast
            cases = list(fixture["cases"])
            heldout = ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
            for row in _load_suites([heldout]):
                tokens = list(build_chat_prompt(generator.tokenizer, str(row["prompt"])))
                if not tokens:
                    raise ValueError("empty heldout")
                for rows in (1024, 4096):
                    padded = (tokens * ((rows - 1) // len(tokens) + 1))[:rows]
                    cases.append(dict(id=f"{row['id']}-p{rows}", category=row["category"],
                                      prompt_tokens=rows, prompt_token_ids=padded))
            report["heldout_suite"] = dict(path=str(heldout),
                sha256=hashlib.sha256(heldout.read_bytes()).hexdigest(),
                policy="uniform repeat/truncate to1024/4096; ownership/state only, not task scores")
            if args.case_id:
                known = {c["id"] for c in cases}
                if set(args.case_id) - known:
                    raise ValueError(f"unknown cases: {sorted(set(args.case_id) - known)}")
                cases = [c for c in cases if c["id"] in args.case_id]
            for case in cases:
                captures = []
                modes = ("candidate", "parent", "parent", "candidate", "candidate", "parent")[:2 * args.paired_repeats]
                for mode in modes:
                    os.environ[linear.WIDE_F16_PAIR_ENV] = "1" if mode == "candidate" else "0"
                    before = dict(counts)
                    first = generator.runner.prefill(case["prompt_token_ids"])
                    prefill = {k: counts[k] - before[k] for k in counts}
                    state_before = _state_summary(generator.runner)
                    logits, ids = [first.logits.copy()], [int(first.token_id)]
                    for _ in range(args.decode_steps):
                        step = generator.runner.step(ids[-1])
                        ids.append(int(step.token_id)); logits.append(step.logits.copy())
                    generator.runner.runtime.device_synchronize()
                    state = _state_summary(generator.runner)
                    kv = hashlib.sha256()
                    for attention in generator.runner.attention_states:
                        for buffer in (attention.key_cache, attention.value_cache):
                            raw = np.empty(buffer.nbytes, dtype=np.uint8)
                            copy_device_to_host(host_array_ptr(raw), buffer, runtime=generator.runner.runtime)
                            kv.update(raw)
                    state["full_kv_sha256"] = kv.hexdigest()
                    total = {k: counts[k] - before[k] for k in counts}
                    expected = expected_pairs(case["prompt_tokens"])
                    assert prefill == total, "pair or cast selected in decode"
                    assert total["pairs"] == (expected if mode == "candidate" else 0), total
                    assert total["accepted"] == 2 * total["pairs"], total
                    stack = np.stack(logits)
                    assert np.isfinite(stack).all()
                    capture = dict(mode=mode, calls=total, decode_calls=0, prefill_state=state_before,
                        state=state, output_token_ids=ids, logits_sha256=hashlib.sha256(stack.tobytes()).hexdigest())
                    if captures:
                        for key in ("prefill_state", "state", "output_token_ids", "logits_sha256"):
                            assert capture[key] == captures[0][key], (case["id"], mode, key)
                    captures.append(capture)
                parent_casts = {r["calls"]["casts"] for r in captures if r["mode"] == "parent"}
                candidate_casts = {r["calls"]["casts"] for r in captures if r["mode"] == "candidate"}
                assert len(parent_casts) == len(candidate_casts) == 1
                assert next(iter(parent_casts)) - next(iter(candidate_casts)) == expected
                report["cases"].append(dict(id=case["id"], category=case["category"],
                    rows=case["prompt_tokens"], expected_pairs=expected, exact=True, runs=captures))
                save()
                print(case["id"], "exact;", expected, "casts eliminated per prefill", flush=True)
            report["engagement_shapes"] = shapes
            report["status"] = "passed"
        except BaseException as error:
            report["status"] = "failed"; report["error"] = repr(error)
            raise
        finally:
            runner.wide_f16_activation_pair = original_scope
            wide.f32_to_f16 = original_cast
            if generator is not None:
                generator.close()
            report["memory_after_close"] = memory_stats()
            if any(report["memory_after_close"][k] != 0 for k in ("active_allocations", "current_allocated_bytes")):
                report["status"] = "failed_teardown"
            save()
    assert report["status"] == "passed", report["status"]


if __name__ == "__main__":
    main()
