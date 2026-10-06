"""Verify native checked-index GDN selection, parent parity and teardown.

No performance claim: logging-only registry wrappers observe the leaf selected
by the binder. Native/parent/native/parent runs compare complete model state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profile_record(production, strict):
    """Keep immutable runtime manifests JSON-safe without changing identity."""
    from hipengine.execution_profiles import validate_variant_manifest
    return {
        "manifest_sha256": production.manifest_sha256,
        "strict_manifest_sha256": strict.manifest_sha256,
        "manifest": validate_variant_manifest(production.manifest),
        "strict_manifest": validate_variant_manifest(strict.manifest),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    from hipengine.kernels.registry import KernelKey, resolve, register
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata
    from scripts.qwen4exp_layer2_profile_gate import _state_summary
    from scripts.gguf_gdn_semantic_gate import _load_suites
    from scripts.gguf_mtp_bench import build_chat_prompt
    from scripts.qwen4exp_gdn_narrow_gate_support import validate_call, expected_calls

    env = "HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT"
    wide = "qwen4exp_sigmoid_wave_norm_prefill"
    narrow = "qwen4exp_sigmoid_wave_norm_narrow_prefill"
    os.environ.pop(env, None)
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profiles = {p: resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
                quant="gguf_ud_q4_k_xl", profile=p) for p in (ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT)}
    resolved, strict = profiles[ExecutionProfile.PRODUCTION], profiles[ExecutionProfile.STRICT]
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    report = dict(schema=1, status="running", kind="native_binder_route_smoke", performance_claim=False,
        host=_host_metadata(), source=_git_metadata(ROOT), commands={"argv": sys.argv},
        source_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--", "hipengine", "scripts", "tests"])).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model_root=str(args.model_root),
        fixture_sha256=digest, profile=profile_record(resolved, strict),
        kv_dtype="BF16", prefill_chunk_size=4096, max_sequence_length=4352, decode_steps=16, cases=[])
    generator = None
    originals = {}
    try:
        index = load_gguf_index(discover_gguf_files(args.model_root)[0])
        generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root, weight_index=index, model_plugin=resolve_model(index.architecture or ""),
            backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
        assert os.environ[env] == narrow
        report["native_binder_value"] = os.environ[env]
        counts = {wide: 0, narrow: 0}
        for variant in (wide, narrow):
            key = KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state", variant)
            fn = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=variant)
            originals[key] = fn
            def counted(*call_args, _fn=fn, _variant=variant, **kwargs):
                validate_call(call_args, max_rows=4096)
                counts[_variant] += 1
                return _fn(*call_args, **kwargs)
            register(key, counted, replace=True)
        cases = [c for c in fixture["cases"] if c["id"] in ("code-p512", "code-p1024", "code-p4096", "general_ja-p512")]
        heldout_path = ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
        row = next(c for c in _load_suites([heldout_path]) if c["category"] == "general_ja")
        tokens = list(build_chat_prompt(generator.tokenizer, str(row["prompt"])))
        cases.append(dict(id=row["id"], category=row["category"], prompt_tokens=len(tokens), prompt_token_ids=tokens))
        report["heldout_suite"] = dict(path=str(heldout_path), sha256=hashlib.sha256(heldout_path.read_bytes()).hexdigest())
        for case in cases:
            captures = []
            for mode in ("native", "parent", "native", "parent"):
                # The first native arm uses untouched binder output; later
                # arms restore exactly that named value after parent comparison.
                if mode == "parent":
                    os.environ[env] = wide
                elif os.environ[env] != narrow:
                    os.environ[env] = report["native_binder_value"]
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
                assert prefill_calls == total_calls, "GDN prefill selected in decode"
                assert total_calls == {wide: expected if mode == "parent" else 0, narrow: expected if mode == "native" else 0}, total_calls
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
            print(case["id"], "native/parent/native/parent exact; expected", expected, flush=True)
        report["status"] = "passed"
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)
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
