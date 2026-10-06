"""DP-5 native publication engagement, state and full-KV parent comparison.

Registry wrappers only count native launches. Heldouts are uniformly extended
for ownership coverage; generated text is not scored on those extended inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[],
                        help="Run only these fixture/heldout cases for a bounded binder smoke")
    parser.add_argument("--paired-repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--gr-k320-full-rows", action="store_true",
                        help="Enable the default-off K320/full-row GR DPP sibling")
    args = parser.parse_args()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    import numpy as np
    from hipengine.core.memory import memory_stats, copy_device_to_host, host_array_ptr
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve, register
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.runtime import gguf_linear as linear
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata
    from scripts.qwen4exp_layer2_profile_gate import _state_summary
    from scripts.gguf_gdn_semantic_gate import _load_suites
    from scripts.gguf_mtp_bench import build_chat_prompt
    from scripts.qwen4exp_gdn_narrow_native_smoke import profile_record
    from scripts.qwen4exp_norm_f16_gate_support import (
        ENV, PRODUCER_KEY, expected_calls, heldout_tokens, validate_producer_call,
    )

    os.environ.pop(ENV, None)
    os.environ.pop("HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS", None)
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production, strict = [resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl", profile=p) for p in (ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT)]
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    report = dict(schema=1, status="running", kind="native_norm_f16_publication_smoke", performance_claim=False,
        host=_host_metadata(), source=_git_metadata(ROOT), commands={"argv": sys.argv},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model_root=str(args.model_root),
        fixture_sha256=digest, profile=profile_record(production, strict),
        kv_dtype="BF16", prefill_chunk_size=4096, max_sequence_length=4352, decode_steps=16, cases=[])
    generator = None
    original_producer = original_cast = original_consume = None
    try:
        index = load_gguf_index(discover_gguf_files(args.model_root)[0])
        generator = production.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root, weight_index=index, model_plugin=resolve_model(index.architecture or ""),
            backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
        assert os.environ[ENV] == "1", "native production binder did not enable publication"
        if args.gr_k320_full_rows:
            os.environ["HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS"] = "1"
        report["native_binder_value"] = os.environ[ENV]
        counts = dict(publisher=0, cast=0, consumed=0)
        key = PRODUCER_KEY
        original_producer = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
        def counted_producer(*a, **kw):
            validate_producer_call(a)
            counts["publisher"] += 1
            return original_producer(*a, **kw)
        register(key, counted_producer, replace=True)
        original_cast = wide.f32_to_f16
        def counted_cast(*a, **kw):
            if int(a[2]) == 4096 * 10240:
                counts["cast"] += 1
            return original_cast(*a, **kw)
        wide.f32_to_f16 = counted_cast
        original_consume = linear._consume_wide_f16_publication
        def counted_consume(*a, **kw):
            ptr = original_consume(*a, **kw)
            counts["consumed"] += bool(ptr)
            return ptr
        linear._consume_wide_f16_publication = counted_consume
        cases = [c for c in fixture["cases"] if c["prompt_tokens"] == 4096
                 or c["id"] in ("code-p512", "code-p1024")]
        heldout_path = ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
        for row in _load_suites([heldout_path]):
            tokens = heldout_tokens(list(build_chat_prompt(generator.tokenizer, str(row["prompt"]))))
            cases.append(dict(id=row["id"], category=row["category"], prompt_tokens=len(tokens), prompt_token_ids=tokens))
        report["heldout_suite"] = dict(path=str(heldout_path), sha256=hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
            token_policy="uniform repeat/truncate to4096 for state/ownership; not task quality")
        if args.case_id:
            known = {case["id"] for case in cases}
            if set(args.case_id) - known:
                raise ValueError(f"unknown case IDs: {sorted(set(args.case_id) - known)}")
            cases = [case for case in cases if case["id"] in args.case_id]
        report["paired_repeats"] = args.paired_repeats
        modes = ("candidate", "parent", "parent", "candidate", "candidate", "parent")[:2 * args.paired_repeats]
        first_native = True
        for case in cases:
            captures = []
            for mode in modes:
                if first_native:
                    # The first request observes the binder, without an arm override.
                    assert mode == "candidate" and os.environ[ENV] == "1"
                    first_native = False
                else:
                    os.environ[ENV] = report["native_binder_value"] if mode == "candidate" else "0"
                    if args.gr_k320_full_rows:
                        os.environ["HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS"] = "1" if mode == "candidate" else "0"
                before = dict(counts)
                first = generator.runner.prefill(case["prompt_token_ids"])
                prefill_state = _state_summary(generator.runner)
                logits, token = [first.logits.copy()], int(first.token_id)
                ids = [token]
                prefill_calls = {k: counts[k] - before[k] for k in counts}
                for _ in range(16):
                    step = generator.runner.step(token)
                    token = int(step.token_id)
                    ids.append(token)
                    logits.append(step.logits.copy())
                generator.runner.runtime.device_synchronize()
                state = _state_summary(generator.runner)
                kv_hash = hashlib.sha256()
                for attention in generator.runner.attention_states:
                    for buffer in (attention.key_cache, attention.value_cache):
                        raw = np.empty(buffer.nbytes, dtype=np.uint8)
                        copy_device_to_host(host_array_ptr(raw), buffer, runtime=generator.runner.runtime)
                        kv_hash.update(raw)
                state["full_kv_sha256"] = kv_hash.hexdigest()
                total_calls = {k: counts[k] - before[k] for k in counts}
                expected = expected_calls(case["prompt_tokens"], 4096)
                assert prefill_calls == total_calls, "publication/cast selected in decode"
                assert total_calls == dict(publisher=expected if mode == "candidate" else 0,
                    consumed=expected if mode == "candidate" else 0, cast=expected if mode == "parent" else 0), total_calls
                stack = np.stack(logits)
                assert np.isfinite(stack).all()
                capture = dict(mode=mode, prefill_calls=prefill_calls, decode_calls=0,
                    logits_sha256=hashlib.sha256(stack.tobytes()).hexdigest(), prefill_state=prefill_state,
                    state=state, output_token_ids=ids)
                if captures:
                    for k in ("logits_sha256", "prefill_state", "state", "output_token_ids"):
                        assert capture[k] == captures[0][k], (case["id"], mode, k)
                captures.append(capture)
            report["cases"].append(dict(case_id=case["id"], category=case["category"], prompt_tokens=case["prompt_tokens"], runs=captures, exact=True))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(case["id"], args.paired_repeats, "paired repeats exact; expected", expected, flush=True)
        report["status"] = "passed"
    finally:
        if original_producer is not None:
            register(PRODUCER_KEY, original_producer, replace=True)
        if original_cast is not None:
            wide.f32_to_f16 = original_cast
        if original_consume is not None:
            linear._consume_wide_f16_publication = original_consume
        if generator is not None:
            generator.close()
        report["memory_after_close"] = memory_stats()
        if any(report["memory_after_close"][k] != 0 for k in ("active_allocations", "current_allocated_bytes")):
            report["status"] = "failed_teardown"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed", report["status"]
    print("PASSED", len(report["cases"]), "cases; teardown", report["memory_after_close"], flush=True)


if __name__ == "__main__":
    main()
