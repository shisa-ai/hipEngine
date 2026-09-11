"""Compare short-context execution at dense-equivalent and native capacities."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,load_fixture,_git_metadata,_host_metadata,_hipengine_case_sample)
from scripts.qwen4exp_halo_box_campaign_ab import arm_sequence,summarize_campaign_ab


def select_cases(cases, baseline, candidate, transitions):
    selected = [c for c in cases if c["prompt_tokens"] in (512,1024)]
    expected = {f"{cat}-p{n}" for cat in ("code","general_en","general_ja","mixed_ja_en") for n in (512,1024)}
    if (len(selected)!=8 or {c["id"] for c in selected}!=expected or
        any(c["prompt_tokens"]+transitions>min(baseline,candidate) for c in selected)):
        raise ValueError("capacity gate requires all eight p512/p1024 category cases")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--baseline-capacity",type=int,default=2051)
    p.add_argument("--candidate-capacity",type=int)
    a = p.parse_args()
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.execution_profiles import ExecutionProfile,resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,QWEN4_EXP_MODEL,QWEN4_EXP_BACKEND,QWEN4_EXP_QUANTS)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files,load_gguf_index
    from hipengine.loading.qwen4_exp_materialize import plan_qwen4_exp_memory_admission
    from hipengine.models import resolve_model
    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
    from scripts.qwen4exp_framework_family_refresh import check_host
    check_host()
    if not _git_metadata(ROOT)["tracked_clean"]:
        raise RuntimeError("commit the validated capacity harness before measurement")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    free_before,total = runtime.mem_get_info()
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    fixture,fixture_sha256 = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(a.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    candidate_capacity = a.candidate_capacity or int(plugin.native_context_length)
    transitions = int(fixture["decode_transitions"])
    cases = select_cases(fixture["cases"],a.baseline_capacity,candidate_capacity,transitions)
    resolved = resolve_runtime_profile(model=QWEN4_EXP_MODEL,backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],profile=ExecutionProfile.PRODUCTION)
    report = dict(schema=1,status="running",source=_git_metadata(ROOT),host=_host_metadata(),
        fixture_sha256=fixture_sha256,command=sys.argv,model="Qwen3.8-Flash-Next UD-Q4_K_XL",
        manifest_sha256=resolved.manifest_sha256,kv_dtype="BF16",
        capacities=dict(before=a.baseline_capacity,after=candidate_capacity),
        protocol="One shared weight residency, two request-owned runners; alternating six-slot order; one warmup per arm/case; tg128",
        samples=[],warmups=[],device_free_before=free_before,device_total=total)
    report["hipengine_artifact_provenance"] = collect_artifact_provenance(
        repo_root=ROOT,configured_backend="hip_gfx1151",resolved_backend="hip_gfx1151",
        target_arch="gfx1151",device_name="AMD Radeon 8060S",model_path=a.model_root,
        model_revision="8bdc666649440e9bdc97e16f3f75782c98478ff5",quant="UD-Q4_K_XL",
        kv_dtype="BF16",command=[sys.executable,*sys.argv],
        timing_protocol=report["protocol"],warmups=1,repetitions=3)
    start = time.perf_counter()
    generator = None
    large = None
    try:
        generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
            model_path=a.model_root,weight_index=index,model_plugin=plugin,backend="hip_gfx1151",
            max_sequence_length=a.baseline_capacity,prefill_chunk_size=512))
        report["baseline_setup_seconds"] = time.perf_counter()-start
        admission = plan_qwen4_exp_memory_admission(generator._resident.plan,
            available_device_bytes=free_before,context_tokens=candidate_capacity,
            resident_capacity=2,scratch_bytes=8*1024**3,reserve_bytes=4*1024**3)
        report["conservative_two_large_runner_admission"] = asdict(admission)
        if not admission.passed:
            raise MemoryError("insufficient memory for the two-runner capacity A/B")
        start = time.perf_counter()
        large = Qwen4ExpGGUFResidentModelRunner(generator._resident,
            max_sequence_length=candidate_capacity,prefill_chunk_size=512,
            backend="hip_gfx1151",runtime=runtime)
        runtime.device_synchronize()
        report["candidate_setup_seconds"] = time.perf_counter()-start
        runners = {"before":generator.runner,"after":large}
        for number,case in enumerate(cases):
            for mode in (("before","after") if number%2==0 else ("after","before")):
                row = _hipengine_case_sample(runners[mode],case=case,repetition=-1,transitions=transitions)
                row["mode"] = mode
                report["warmups"].append(row)
            for slot,mode in enumerate(arm_sequence(number)):
                row = _hipengine_case_sample(runners[mode],case=case,repetition=slot//2,transitions=transitions)
                row.update(mode=mode,sequence_slot=slot,allocated_capacity=runners[mode].max_sequence_length)
                report["samples"].append(row)
                print(case["id"],mode,slot,"pp",row["prefill_tok_s"],"tg",row["decode_tok_s"],flush=True)
                a.output.write_text(json.dumps(report,indent=2)+"\n")
        report["summary"] = summarize_campaign_ab(report["samples"],repetitions_per_mode=3)
        if not report["summary"]["correctness"]["cross_mode_output_exact"]:
            raise AssertionError("capacity changes generated output")
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        if large is not None:
            large.close()
        if generator is not None:
            generator.close()
        report["memory_after_close"] = memory_stats()
        if report["memory_after_close"]["active_allocations"] != 0:
            report["status"] = "failed"
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        if report["memory_after_close"]["active_allocations"] != 0:
            raise RuntimeError("capacity A/B leaked tracked device ownership")


if __name__=="__main__":
    main()
