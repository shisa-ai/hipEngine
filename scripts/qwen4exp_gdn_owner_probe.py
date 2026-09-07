"""Pair GDN owners on identical model-produced inputs; not throughput timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from hipengine.core.memory import malloc, free, memory_stats
from hipengine.kernels.registry import KernelKey, register, resolve
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from tests.test_qwen4_exp_gdn_hip import _download
from types import SimpleNamespace


def validate_geometry(args):
    if len(args) != 14 or tuple(args[10:14]) != (16,48,128,128) or not 2 <= args[9] <= 1024:
        raise ValueError("requires serial-prefix Hk16/Hv48/D128,2..1024 rows")


def pair_order(index):
    return (0,1) if index % 2 == 0 else (1,0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--case-id",action="append")
    args=p.parse_args()
    check_host()
    identity=model_identity(args.model_root)
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"]=str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"]="1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile=gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL,backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1],profile=gate.ExecutionProfile.PRODUCTION)
    fixture,digest=gate.load_fixture(gate.DEFAULT_FIXTURE)
    ids=args.case_id or ["code-p512","code-p1024","code-p4096","general_en-p4096","general_ja-p4096","mixed_ja_en-p4096"]
    cases=[next(c for c in fixture["cases"] if c["id"]==name) for name in ids]
    index=gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
    generator=profile.construct_generator(lambda:gate.Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root,weight_index=index,model_plugin=gate.resolve_model(index.architecture or ""),
        backend="hip_gfx1151",max_sequence_length=4352,prefill_chunk_size=1024))
    runtime=generator.runner.runtime
    # This diagnostic intercepts the parent even after the candidate is promoted.
    os.environ["HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM"]="0"
    key=KernelKey("hip_gfx1151","gdn_recurrence_norm_gate","f32_state","qwen4exp_sigmoid_register_prefill")
    parent=resolve(backend=key.backend,layer=key.layer,quant=key.quant,variant=key.variant)
    candidate=resolve(backend=key.backend,layer=key.layer,quant=key.quant,variant="qwen4exp_sigmoid_wave_norm_prefill")
    saved=malloc(48*128*128*4,runtime=runtime)
    start,stop=runtime.event_create(),runtime.event_create()
    report=dict(source=gate._git_metadata(ROOT),host=gate._host_metadata(),model_identity=identity,
        command=sys.argv,fixture_sha256=digest,chunk_size=1024,cases=[],status="running",
        protocol="Every serial call: restore starting state outside timing,1 warmup pair+2 measured counterbalanced pairs; compare complete output/state hashes after every pair. Only HIP-event kernel intervals timed.",
        performance_claim=False)
    captures=[]

    def hooked(*a,**kw):
        validate_geometry(a)
        if int(kw.get("stream",0))!=0:
            raise ValueError("owner probe only supports default stream")
        runtime.device_synchronize()
        runtime.memcpy(saved.ptr,a[7],saved.nbytes,3)
        samples=[[],[]]
        for rep in range(3):
            hashes={}
            for arm in pair_order(rep+len(captures)):
                runtime.memcpy(a[7],saved.ptr,saved.nbytes,3)
                runtime.device_synchronize()
                runtime.event_record(start)
                (parent if arm==0 else candidate)(*a,**kw)
                runtime.event_record(stop)
                runtime.event_synchronize(stop)
                if rep:
                    samples[arm].append(runtime.event_elapsed_time_ms(start,stop))
                h=hashlib.sha256()
                for ptr,shape in ((a[7],(48,128,128)),(a[8],(a[9],48,128))):
                    data=_download(SimpleNamespace(ptr=ptr,nbytes=int(np.prod(shape))*4),
                                   shape,np.float32,runtime)
                    if not np.isfinite(data).all():
                        raise AssertionError("nonfinite model GDN output/state")
                    h.update(data.tobytes())
                hashes[arm]=h.hexdigest()
            if hashes[0]!=hashes[1]:
                raise AssertionError("GDN model-input output/state mismatch")
        captures.append(dict(tokens=a[9],samples_ms=samples,exact=True))

    try:
        register(key,hooked,replace=True)
        for case in cases:
            captures.clear()
            token=generator.runner.prefill(case["prompt_token_ids"])
            expected=gate.gdn_wave_norm_expected_calls(case["prompt_tokens"],1024)
            if len(captures)!=expected:
                raise AssertionError((case["id"],len(captures),expected))
            totals=[sum(statistics.mean(c["samples_ms"][i]) for c in captures) for i in (0,1)]
            report["cases"].append(dict(id=case["id"],calls=len(captures),first_token=token.token_id,
                parent_ms=totals[0],candidate_ms=totals[1],speedup=totals[0]/totals[1],
                captures=list(captures)))
            print(case["id"],totals,totals[0]/totals[1],flush=True)
        report["status"]="passed"
    except Exception as error:
        report["status"]="failed"
        report["error"]=repr(error)
        raise
    finally:
        register(key,parent,replace=True)
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        free(saved,runtime=runtime)
        generator.close()
        report["memory_after_close"]=memory_stats()
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
