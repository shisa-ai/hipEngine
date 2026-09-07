"""Fixed-root decode flag isolation; not normal request throughput."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host,model_identity
from hipengine.kernels.registry import KernelKey,resolve,register
from hipengine.core.memory import memory_stats

FLAG="HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR"


def orders(pairs):
    if pairs<2 or pairs%2:
        raise ValueError("even pair count>=2 required")
    return [(0,1) if i%2==0 else (1,0) for i in range(pairs)]

def phase_flags(vary_prefill, arm):
    if arm not in (0,1):
        raise ValueError("invalid arm")
    return (str(arm),"0") if vary_prefill else ("0",str(arm))


def telemetry():
    paths=list(Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk"))
    paths+=list(Path("/sys/class/hwmon").glob("hwmon*/temp1_input"))
    result={}
    for path in paths:
        try:
            result[str(path)]=path.read_text().strip()
        except OSError:
            pass
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pairs",type=int,default=4)
    p.add_argument("--steps",type=int,default=16)
    p.add_argument("--vary-prefill",action="store_true",
                   help="Re-prefill per arm with selected variant; decode flag always0")
    a=p.parse_args()
    schedule=orders(a.pairs)
    if not 1<=a.steps<=128:
        p.error("steps must be1..128")
    check_host()
    identity=model_identity(a.model_root)
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"]=str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"]="1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile=gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL,backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1],profile=gate.ExecutionProfile.PRODUCTION)
    fixture,digest=gate.load_fixture(gate.DEFAULT_FIXTURE)
    index=gate.load_gguf_index(gate.discover_gguf_files(a.model_root)[0])
    generator=profile.construct_generator(lambda:gate.Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root,weight_index=index,model_plugin=gate.resolve_model(index.architecture or ""),
        backend="hip_gfx1151",max_sequence_length=4352,prefill_chunk_size=1024))
    runner=generator.runner
    key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv","strict_h256_head_pair_rows_spans")
    original=resolve(backend=key.backend,layer=key.layer,quant=key.quant,variant=key.variant)
    calls=[0]
    def counted(*args,**kwargs):
        calls[0]+=1
        return original(*args,**kwargs)
    report=dict(status="running",source=gate._git_metadata(ROOT),host=gate._host_metadata(),
        model_identity=identity,fixture_sha256=digest,command=sys.argv,cases=[],
        steps=a.steps,pairs=a.pairs,vary_prefill=a.vary_prefill,
        protocol="Parent prefill once per case; restore identical root before each arm,one warmup per flag,balanced measured pairs; host hashes and telemetry outside timing",
        limits="Snapshot restore and inter-arm inspection change cache/power history. This isolates flag-at-decode only,not the effects of preceding candidate prefill. No clock changes.")
    if a.vary_prefill:
        report.update(
            protocol="Fresh prefill per arm,flag toggles only prefill;decode flag always0,one warmup per arm,balanced pairs. No snapshot restore or host hashes between prefill and decode.",
            limits="Bounded two-case phase diagnostic with telemetry/step clocks,not full-suite throughput or thermal causal proof. No clock changes.")
    previous=os.environ.get(FLAG)
    try:
        register(key,counted,replace=True)
        for name in ("code-p4096","mixed_ja_en-p4096"):
            case=next(c for c in fixture["cases"] if c["id"]==name)
            os.environ[FLAG]="0"
            if not a.vary_prefill:
                root=runner.prefill(case["prompt_token_ids"])
                snapshot=runner.snapshot()
            rows=[]
            expected=None
            for pair,order in enumerate([(0,1),*schedule]):
                for arm in order:
                    before=telemetry()
                    prefill_flag,decode_flag=phase_flags(a.vary_prefill,arm)
                    prefill_seconds=None
                    if a.vary_prefill:
                        os.environ[FLAG]=prefill_flag
                        start_calls=calls[0]
                        start=time.perf_counter()
                        root=runner.prefill(case["prompt_token_ids"],capture_logits=False,
                                            capture_target_hidden=False)
                        runner.runtime.device_synchronize()
                        prefill_seconds=time.perf_counter()-start
                        if calls[0]-start_calls!=(24 if arm else 0):
                            raise AssertionError("prefill engagement mismatch")
                    else:
                        runner.restore(snapshot)
                    os.environ[FLAG]=decode_flag
                    initial_calls=calls[0]
                    token=root.token_id
                    tokens=[]
                    step_seconds=[]
                    runner.runtime.device_synchronize()
                    start=time.perf_counter()
                    for _ in range(a.steps):
                        step_start=time.perf_counter()
                        token=runner.step(token,capture_logits=False,capture_target_hidden=False).token_id
                        step_seconds.append(time.perf_counter()-step_start)
                        tokens.append(token)
                    runner.runtime.device_synchronize()
                    elapsed=time.perf_counter()-start
                    state=gate._state_summary(runner)
                    result=(tokens,state["state_sha256"],state["layout_sha256"])
                    if expected is None:
                        expected=result
                    if result!=expected or not state["finite"] or calls[0]!=initial_calls:
                        raise AssertionError("decode flag altered state/output or engaged candidate")
                    if pair:
                        rows.append(dict(pair=pair-1,flag=arm,seconds=elapsed,
                            prefill_flag=prefill_flag,decode_flag=decode_flag,
                            prefill_seconds=prefill_seconds,step_seconds=step_seconds,
                            state_sha256=state["state_sha256"],tokens=tokens,candidate_calls=0,
                            telemetry_before=before,telemetry_after=telemetry()))
            means=[statistics.mean(r["seconds"] for r in rows if r["flag"]==i) for i in (0,1)]
            report["cases"].append(dict(id=name,rows=rows,off_on_wall_ratio=means[0]/means[1],
                order_ratios=[statistics.mean(r["seconds"] for r in rows if r["flag"]==0 and r["pair"]%2==i)/
                              statistics.mean(r["seconds"] for r in rows if r["flag"]==1 and r["pair"]%2==i)
                              for i in (0,1)]))
            print(name,means,means[0]/means[1],flush=True)
        report["status"]="passed"
    except Exception as error:
        report.update(status="failed",error=repr(error))
        raise
    finally:
        register(key,original,replace=True)
        if previous is None:
            os.environ.pop(FLAG,None)
        else:
            os.environ[FLAG]=previous
        generator.close()
        report["memory_after_close"]=memory_stats()
        a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
