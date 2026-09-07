"""Fixed-root decode flag isolation; not normal request throughput."""
import argparse
import gc
import hashlib
import json
import math
import os
import resource
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

def cpu_counters():
    usage=resource.getrusage(resource.RUSAGE_THREAD)
    return dict(thread_seconds=time.thread_time(),user_seconds=usage.ru_utime,
        system_seconds=usage.ru_stime,minor_faults=usage.ru_minflt,
        major_faults=usage.ru_majflt,voluntary_switches=usage.ru_nvcsw,
        involuntary_switches=usage.ru_nivcsw)


def counter_delta(before,after):
    return {key:after[key]-value for key,value in before.items()}

def wait_runtime_identity():
    """Record loaded binaries and env; env alone does not prove activation."""
    libraries={}
    try:
        lines=Path("/proc/self/maps").read_text().splitlines()
        paths={line.split(maxsplit=5)[5] for line in lines if len(line.split(maxsplit=5))==6}
        for value in paths:
            path=Path(value)
            if path.name.startswith(("libamdhip64.so","libhsa-runtime64.so")) and path.is_file():
                with path.open("rb") as source:
                    libraries[str(path)]=hashlib.file_digest(source,"sha256").hexdigest()
    except OSError as error:
        libraries["error"]=str(error)
    return dict(libraries=libraries,environment={name:os.environ.get(name) for name in
        ("HSA_ENABLE_MWAITX","HSA_ENABLE_INTERRUPT","ROC_ACTIVE_WAIT_TIMEOUT")},
        caveat="Requested environment and binary identity only; not wait-path engagement proof")


def orders(pairs):
    if pairs<2 or pairs%2:
        raise ValueError("even pair count>=2 required")
    return [(0,1) if i%2==0 else (1,0) for i in range(pairs)]

def phase_flags(vary_prefill, arm):
    if arm not in (0,1):
        raise ValueError("invalid arm")
    return (str(arm),"0") if vary_prefill else ("0",str(arm))

def active_transition(milliseconds, clock=time.perf_counter):
    """Diagnostic busy interval; caller must charge its full wall time."""
    if not math.isfinite(milliseconds) or not 0 <= milliseconds <= 500:
        raise ValueError("active transition must be finite and0..500ms")
    start=clock()
    end=start
    while end-start < milliseconds/1000:
        end=clock()
    return end-start

def step_marker(case, pair, arm, step):
    return f"qsa_phase:{case}:pair{pair}:arm{arm}:step{step}"


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
    p.add_argument("--trace-markers",action="store_true")
    p.add_argument("--cpu-accounting",action="store_true",
                   help="Read thread CPU/fault/switch deltas and observe GC without disabling it")
    p.add_argument("--thread-perf",action="store_true",
                   help="Count calling-thread user cycles/instructions only during decode")
    p.add_argument("--cpu-frequency",action="store_true",
                   help="Read CPUFreq feedback/reported frequency and CPU identity at step boundaries")
    p.add_argument("--pin-cpu",type=int,
                   help="Pin calling thread after model load; restore affinity at exit")
    p.add_argument("--cpu-min-khz",type=int,
                   help="Temporary requested minimum on pinned CPU policy; restored at exit")
    p.add_argument("--hip-wait",choices=("auto","spin","yield","blocking"),
                   help="Disposable-process device scheduling flag before model allocation")
    p.add_argument("--transition-active-ms",type=float,
                   help="Both arms candidate-prefill; arm1 adds charged CPU-active interval (0..500ms)")
    p.add_argument("--case-id",action="append",choices=("code-p4096","mixed_ja_en-p4096"))
    a=p.parse_args()
    schedule=orders(a.pairs)
    if a.transition_active_ms is not None:
        if not a.vary_prefill or not math.isfinite(a.transition_active_ms) or not 0 < a.transition_active_ms <= 500:
            p.error("transition-active-ms requires vary-prefill and finite0<ms<=500")
    if a.cpu_min_khz is not None and a.pin_cpu is None:
        p.error("--cpu-min-khz requires --pin-cpu")
    frequency=None
    if a.cpu_frequency:
        from scripts.qwen4exp_cpu_frequency import CpuFrequency
        frequency=CpuFrequency()
    perf=None
    if a.thread_perf:
        from scripts.qwen4exp_thread_perf import ThreadPerf
        # Fail permissions/capability checks before loading the model.
        perf=ThreadPerf()
        perf.close()
        perf=None
    marker=None
    if a.trace_markers:
        from scripts.qwen4exp_profile_gap import Roctx
        marker=Roctx()
    if not 1<=a.steps<=128:
        p.error("steps must be1..128")
    check_host()
    wait_policy=None
    if a.hip_wait:
        from scripts.qwen4exp_hip_wait_policy import configure
        wait_policy=configure(a.hip_wait)
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
        hip_wait_policy=wait_policy,
        wait_runtime=wait_runtime_identity(),
        model_identity=identity,fixture_sha256=digest,command=sys.argv,cases=[],
        steps=a.steps,pairs=a.pairs,vary_prefill=a.vary_prefill,trace_markers=a.trace_markers,
        cpu_accounting=a.cpu_accounting,thread_perf=a.thread_perf,cpu_frequency=a.cpu_frequency,
        protocol="Parent prefill once per case; restore identical root before each arm,one warmup per flag,balanced measured pairs; host hashes and telemetry outside timing",
        limits="Snapshot restore and inter-arm inspection change cache/power history. This isolates flag-at-decode only,not the effects of preceding candidate prefill. No clock changes.")
    if a.vary_prefill:
        report.update(
            protocol="Fresh prefill per arm,flag toggles only prefill;decode flag always0,one warmup per arm,balanced pairs. No snapshot restore or host hashes between prefill and decode.",
            limits="Bounded two-case phase diagnostic with telemetry/step clocks,not full-suite throughput or thermal causal proof. No clock changes.")
    if a.transition_active_ms is not None:
        report.update(transition_active_ms=a.transition_active_ms,
            protocol="Both arms fresh candidate-prefill,decode flag0;arm0 direct transition,arm1 bounded CPU-active interval. Charge interval in transition_plus_decode and request totals.",
            limits="Diagnostic busy work is not production code. Lower decode time alone is not a win; all added transition wall must be counted. No power-efficiency claim.")
    previous=os.environ.get(FLAG)
    from scripts.qwen4exp_thread_affinity import ThreadAffinity
    affinity=ThreadAffinity(a.pin_cpu)
    from scripts.qwen4exp_cpu_floor import CpuFloor
    floor=CpuFloor(a.pin_cpu,a.cpu_min_khz)
    gc_events=[]
    gc_start=[None]
    active_step=[None]
    def on_gc(phase,info):
        if phase=="start":
            gc_start[0]=(time.perf_counter(),active_step[0])
        elif gc_start[0] is not None:
            start,step=gc_start[0]
            if step is not None:
                gc_events.append(dict(step=step,seconds=time.perf_counter()-start,**info))
            gc_start[0]=None
    try:
        report["thread_affinity"]=affinity.enter()
        report["cpu_floor"]=floor.enter()
        if a.thread_perf:
            perf=ThreadPerf()
        if a.cpu_accounting:
            gc.callbacks.append(on_gc)
        register(key,counted,replace=True)
        for name in a.case_id or ("code-p4096","mixed_ja_en-p4096"):
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
                    if a.transition_active_ms is not None:
                        prefill_flag,decode_flag="1","0"
                    prefill_seconds=None
                    if a.vary_prefill:
                        os.environ[FLAG]=prefill_flag
                        start_calls=calls[0]
                        start=time.perf_counter()
                        root=runner.prefill(case["prompt_token_ids"],capture_logits=False,
                                            capture_target_hidden=False)
                        runner.runtime.device_synchronize()
                        prefill_seconds=time.perf_counter()-start
                        if calls[0]-start_calls!=(24 if prefill_flag=="1" else 0):
                            raise AssertionError("prefill engagement mismatch")
                    else:
                        runner.restore(snapshot)
                    os.environ[FLAG]=decode_flag
                    initial_calls=calls[0]
                    token=root.token_id
                    tokens=[]
                    step_seconds=[]
                    cpu_steps=[]
                    frequency_steps=[]
                    gc_events.clear()
                    runner.runtime.device_synchronize()
                    transition_seconds=active_transition(
                        a.transition_active_ms if a.transition_active_ms is not None and arm else 0)
                    if perf:
                        perf.start()
                    start=time.perf_counter()
                    for step in range(a.steps):
                        frequency_before=frequency.sample() if frequency else None
                        if marker:
                            marker.push(step_marker(name,pair-1,arm,step))
                        step_start=time.perf_counter()
                        cpu_before=cpu_counters() if a.cpu_accounting else None
                        active_step[0]=step
                        try:
                            token=runner.step(token,capture_logits=False,capture_target_hidden=False).token_id
                        finally:
                            active_step[0]=None
                            if marker:
                                marker.pop()
                        if cpu_before is not None:
                            cpu_steps.append(counter_delta(cpu_before,cpu_counters()))
                        step_seconds.append(time.perf_counter()-step_start)
                        if frequency:
                            frequency_steps.append(dict(before=frequency_before,after=frequency.sample()))
                        tokens.append(token)
                    runner.runtime.device_synchronize()
                    elapsed=time.perf_counter()-start
                    hardware=perf.stop() if perf else None
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
                            transition_seconds=transition_seconds,
                            transition_plus_decode_seconds=transition_seconds+elapsed,
                            request_seconds=(prefill_seconds+transition_seconds+elapsed
                                             if prefill_seconds is not None else None),
                            cpu_steps=cpu_steps,gc_events=list(gc_events),
                            hardware_counters=hardware,
                            frequency_steps=frequency_steps,
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
        try:
            if perf:
                perf.close()
            if on_gc in gc.callbacks:
                gc.callbacks.remove(on_gc)
            register(key,original,replace=True)
            if previous is None:
                os.environ.pop(FLAG,None)
            else:
                os.environ[FLAG]=previous
            generator.close()
        finally:
            try:
                report["restored_cpu_min_khz"]=floor.close()
            finally:
                report["restored_thread_affinity"]=affinity.close()
                report["memory_after_close"]=memory_stats()
                a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
