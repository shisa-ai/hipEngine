"""Diagnostic request-owned full GDN graphs in advancing decode; no default change."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host,model_identity
from hipengine.runtime import qwen4_exp_runner as runner_module
from hipengine.core.memory import memory_stats


def mutable_regions(residual,kw):
    if kw["rows"]!=1 or (kw["num_k_heads"],kw["num_v_heads"],kw["head_dim"])!=(16,48,128):
        raise ValueError("requires single-row Hk16/Hv48/D128")
    return ((residual,kw["branches"]*kw["hidden"]*2),
            (kw["conv_state_ptr"],(2*16*128+48*128)*kw["conv_kernel"]*4),
            (kw["recurrent_state_ptr"],48*128*128*4))


class LayerGraphs:
    def __init__(self,rt,original):
        self.rt,self.original=rt,original
        self.stream=rt.stream_create(nonblocking=True)
        self.entries={}
        self.stats=dict(capture=0,replay=0,setup_seconds=0.0)
        self.enabled=False

    def read(self,ptr,size):
        out=np.empty(size,np.uint8)
        self.rt.memcpy(out.ctypes.data,ptr,size,2)
        return out

    def write(self,regions,values):
        for (ptr,size),value in zip(regions,values,strict=True):
            self.rt.memcpy(ptr,value.ctypes.data,size,1)

    def __call__(self,residual,weights,**kw):
        if not self.enabled or kw["rows"]!=1:
            return self.original(residual,weights,**kw)
        if kw.get("stream",0)!=0 or kw["runtime"] is not self.rt:
            raise ValueError("probe requires default stream and one runtime owner")
        regions=mutable_regions(residual,kw)
        cache=kw.get("moe_graph_cache")
        if cache is None or not cache.enabled:
            raise ValueError("production MoE graph baseline required")
        key=(kw["moe_graph_key"],id(weights),id(kw["scratch"]),regions)
        if key in self.entries:
            graph,executable,out=self.entries[key]
            self.rt.graph_launch(executable,0)
            self.stats["replay"]+=1
            return out
        started=time.perf_counter()
        self.rt.device_synchronize()
        before=[self.read(*region) for region in regions]
        out=self.original(residual,weights,**kw)
        self.rt.device_synchronize()
        final_regions=(*regions,(out.ptr,kw["branches"]*kw["hidden"]*2))
        reference=[self.read(*region) for region in final_regions]
        graph=executable=0
        capturing=False
        try:
            self.write(regions,before)
            capture_kw=dict(kw,stream=self.stream,moe_graph_cache=None,moe_graph_key=None)
            self.rt.stream_begin_capture(self.stream,2)
            capturing=True
            captured_out=self.original(residual,weights,**capture_kw)
            graph=self.rt.stream_end_capture(self.stream)
            capturing=False
            if captured_out.ptr!=out.ptr:
                raise AssertionError("graph output pointer changed")
            if any(not np.array_equal(self.read(*region),value)
                   for region,value in zip(regions,before,strict=True)):
                raise AssertionError("capture executed recurrent update")
            executable=self.rt.graph_instantiate(graph)
            self.rt.graph_launch(executable,0)
            self.rt.device_synchronize()
            if any(not np.array_equal(self.read(*region),value)
                   for region,value in zip(final_regions,reference,strict=True)):
                raise AssertionError("graph state/output differs from production layer")
        except Exception:
            if capturing:
                try:
                    graph=self.rt.stream_end_capture(self.stream)
                except Exception:
                    pass
            if executable:self.rt.graph_exec_destroy(executable)
            if graph:self.rt.graph_destroy(graph)
            # Leave the original successful transition in place, but fail the probe.
            self.write(final_regions,reference)
            raise
        self.entries[key]=(graph,executable,out)
        self.stats["capture"]+=1
        self.stats["setup_seconds"]+=time.perf_counter()-started
        return out

    def close(self):
        self.rt.device_synchronize()
        for graph,executable,_ in self.entries.values():
            self.rt.graph_exec_destroy(executable)
            self.rt.graph_destroy(graph)
        self.entries.clear()
        self.rt.stream_destroy(self.stream)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--case-id",default="code-p4096")
    p.add_argument("--steps",type=int,default=16)
    a=p.parse_args()
    if not 4<=a.steps<=128:p.error("steps must be4..128")
    check_host()
    identity=model_identity(a.model_root)
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"]=str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"]="1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile=gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL,backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1],profile=gate.ExecutionProfile.PRODUCTION)
    fixture,digest=gate.load_fixture(gate.DEFAULT_FIXTURE)
    case=next(c for c in fixture["cases"] if c["id"]==a.case_id)
    index=gate.load_gguf_index(gate.discover_gguf_files(a.model_root)[0])
    gen=profile.construct_generator(lambda:gate.Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root,weight_index=index,model_plugin=gate.resolve_model(index.architecture or ""),
        backend="hip_gfx1151",max_sequence_length=4352,prefill_chunk_size=1024))
    original=runner_module.run_qwen4_exp_gdn_layer
    graphs=LayerGraphs(gen.runner.runtime,original)
    report=dict(status="running",source=gate._git_metadata(ROOT),host=gate._host_metadata(),
        model_identity=identity,fixture_sha256=digest,command=sys.argv,case_id=a.case_id,
        runs=[],limits="Single request owner,c1 only,diagnostic wrapper,off/on/on/off order. First graph arm includes setup/validation;not canonical full-suite throughput.")
    try:
        root=gen.runner.prefill(case["prompt_token_ids"])
        snapshot=gen.runner.snapshot()
        runner_module.run_qwen4_exp_gdn_layer=graphs
        expected=None
        for enabled in (False,True,True,False):
            gen.runner.restore(snapshot)
            graphs.enabled=enabled
            before=dict(graphs.stats)
            token=root.token_id
            tokens=[]
            timings=[]
            for _ in range(a.steps):
                start=time.perf_counter()
                token=gen.runner.step(token,capture_logits=False,capture_target_hidden=False).token_id
                gen.runner.runtime.device_synchronize()
                timings.append(time.perf_counter()-start)
                tokens.append(token)
            state=gate._state_summary(gen.runner)
            result=(tokens,state["state_sha256"],state["layout_sha256"])
            if expected is None:expected=result
            if result!=expected or not state["finite"]:
                raise AssertionError("advancing decode state/token mismatch")
            delta={k:v-before[k] for k,v in graphs.stats.items()}
            if enabled and delta["capture"]+delta["replay"]!=36*a.steps:
                raise AssertionError("GDN graph engagement mismatch")
            report["runs"].append(dict(enabled=enabled,tokens=tokens,state=state,
                step_seconds=timings,total_seconds=sum(timings),graph_delta=delta))
            print(enabled,sum(timings),delta,flush=True)
        report["status"]="passed"
    except Exception as error:
        report.update(status="failed",error=repr(error))
        raise
    finally:
        runner_module.run_qwen4_exp_gdn_layer=original
        try:graphs.close()
        finally:gen.close()
        report["memory_after_close"]=memory_stats()
        a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
