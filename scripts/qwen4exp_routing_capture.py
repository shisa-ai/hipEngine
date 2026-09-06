"""Capture actual expert counts at exact Q4/Q5_1 pair-prefill boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def validate_replay_identity(packet, *, fixture_sha256):
    from scripts.qwen4exp_framework_family_refresh import HOST_ID, MODEL_FINGERPRINT
    if (packet.get("host", {}).get("machine_id") != HOST_ID or
        packet.get("fixture_sha256") != fixture_sha256 or
        packet.get("model_identity", {}).get("fingerprint", {}).get("value")
            != MODEL_FINGERPRINT):
        raise ValueError("routing capture host/model/fixture identity mismatch")


def summarize_routing(packet):
    groups = {}
    for row in packet["records"]:
        groups.setdefault((row["case_id"], row["family"]), []).append(row["counts"])
    result = []
    for (case_id, family), boundaries in sorted(groups.items()):
        counts = np.asarray(boundaries, dtype=np.int64).reshape(-1)
        active = counts[counts > 0]
        total = int(counts.sum())
        result.append(dict(
            case_id=case_id, family=family, boundaries=len(boundaries),
            active_expert_instances=int(len(active)), compact_rows=total,
            active_rows_percentiles=dict(zip(
                ("p50", "p90", "p99", "max"),
                np.percentile(active, [50, 90, 99, 100]).tolist())),
            rows_in_experts_gt8_share=float(counts[counts > 8].sum() / total),
            weight_passes_by_row_batch={
                str(rb): int(np.sum((counts + rb - 1) // rb)) for rb in (8, 16, 32)}))
    return result


def routing_record(starts, compact_rows, experts):
    starts = np.asarray(starts)
    if (starts.dtype.kind not in "iu" or starts.shape != (experts+1,) or
        starts[0] != 0 or starts[-1] != compact_rows or
        np.any(starts[1:] < starts[:-1])):
        raise ValueError("invalid compact expert boundaries")
    counts = np.diff(starts.astype(np.int64))
    active = counts[counts>0]
    return dict(counts=counts.tolist(),active_experts=int(len(active)),
        median_active_rows=float(np.median(active)) if len(active) else 0.,
        max_rows=int(counts.max(initial=0)),
        weight_passes_by_row_batch={str(r):int(np.sum((counts+r-1)//r)) for r in (8,16,32)})


def select_routing(packet, *, case_id, layer, chunk, tokens):
    if packet.get("status") != "captured_exact":
        raise ValueError("routing capture has not passed instrumentation parity")
    matches = [r for r in packet["records"] if
        r["case_id"]==case_id and r["family"]=="q4" and
        r["tensor"]==f"blk.{layer}.ffn_gate_exps.weight" and r["chunk_index"]==chunk]
    if len(matches)!=1:
        raise ValueError("routing selection must identify exactly one captured boundary")
    row = matches[0]
    counts = np.asarray(row["counts"])
    if (counts.dtype.kind not in "iu" or
        counts.shape!=(row["num_experts"],) or np.any(counts<0) or
        int(counts.sum())!=tokens*10 or row["compact_rows"]!=tokens*10 or
        (row["in_features"],row["out_features"])!=(2560,640)):
        raise ValueError("routing capture shape does not match the microbenchmark")
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path)
    p.add_argument("--compiler-version-file",type=Path)
    p.add_argument("--case-id",action="append")
    p.add_argument("--summarize",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture
        reports = []
        for path in a.summarize:
            packet = json.loads(path.read_text())
            validate_replay_identity(packet, fixture_sha256=load_fixture(DEFAULT_FIXTURE)[1])
            if packet["status"] != "captured_exact":
                raise ValueError("cannot summarize a failed or unfinished capture")
            reports.append({
                **{k: v for k, v in packet.items() if k not in ("records", "cases")},
                "cases": [
                    dict(id=c["id"], exact=c["exact"],
                         prefill_logits_sha256=c["captures"][0]["prefill_logits"],
                         next_logits_sha256=c["captures"][0]["next_logits"],
                         state_sha256=c["captures"][0]["state"]["state_sha256"])
                    for c in packet["cases"]],
                "raw_path": str(path), "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "summary": summarize_routing(packet)})
        a.output.write_text(json.dumps(dict(
            schema=1, kind="qwen4exp_real_routing", performance_claim=False,
            summary_command=sys.argv,
            limitations="Counts at exact Q4/Q5_1 pair boundaries only; no timed speedup. "
                         "Replay activations remain synthetic. Row-batch passes are arithmetic estimates, "
                         "not measured memory traffic. State digest does not hash full KV contents.",
            captures=reports), indent=2)+"\n")
        return
    if not a.model_root or not a.compiler_version_file or not a.case_id:
        p.error("capture requires model-root, compiler-version-file and case-id")
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr, memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,QWEN4_EXP_MODEL,QWEN4_EXP_BACKEND,QWEN4_EXP_QUANTS)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, register, resolve
    from hipengine.loading.gguf import discover_gguf_files,load_gguf_index
    from hipengine.models import resolve_model
    from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE,load_fixture,_git_metadata,_host_metadata
    from scripts.qwen4exp_framework_family_refresh import check_host,model_identity
    from scripts.qwen4exp_layer2_profile_gate import _state_summary
    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    fixture,digest = load_fixture(DEFAULT_FIXTURE)
    cases = [c for c in fixture["cases"] if c["id"] in a.case_id]
    if len(cases)!=len(set(a.case_id)):
        p.error("unknown case id")
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(model=QWEN4_EXP_MODEL,backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],profile=ExecutionProfile.PRODUCTION)
    index = load_gguf_index(discover_gguf_files(a.model_root)[0])
    report = dict(schema=1,status="running",performance_claim=False,source=_git_metadata(ROOT),
        host=_host_metadata(),model_identity=model_identity(a.model_root),model_root=str(a.model_root.resolve()),
        fixture_sha256=digest,command=sys.argv,manifest_sha256=profile.manifest_sha256,
        scope="Exact Q4/Q5_1 pair-prefill calls only, not WMMA/Q5_K/Q8-down routes",
        records=[],cases=[])
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root,weight_index=index,model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151",max_sequence_length=4352,prefill_chunk_size=512))
    originals = {}
    context = dict(enabled=False,case_id="",sequence={})
    try:
        runner = generator.runner
        names = {}
        for w in runner.resident.device_weights.values():
            if "raw" in w.allocations:
                names[int(w.allocation("raw").tensor.ptr)] = w.spec.source.name
        for family,quant,variant in (
            ("q4","gguf_q4_k","selected_dual_grouped_pair2_bf16_bf16_out"),
            ("q51","gguf_q5_1","selected_grouped_prefill_pair2_bf16_bf16_out")):
            key = KernelKey("hip_gfx1151","moe_linear",quant,variant)
            original = resolve(backend=key.backend,layer=key.layer,quant=key.quant,variant=key.variant)
            originals[key] = original
            def wrapped(*args,_family=family,_original=original,**kwargs):
                if context["enabled"]:
                    offset = 6 if _family=="q4" else 4
                    rows,experts,k,n = map(int,args[offset:offset+4])
                    tensor = names[int(args[2])]
                    sequence_key = (_family,tensor)
                    chunk = context["sequence"].get(sequence_key,0)
                    context["sequence"][sequence_key] = chunk+1
                    starts = np.empty(experts+1,np.int64)
                    runner.runtime.device_synchronize()
                    copy_device_to_host(host_array_ptr(starts),DeviceBuffer(int(args[1]),starts.nbytes),
                                        runtime=runner.runtime)
                    report["records"].append(dict(
                        case_id=context["case_id"],family=_family,tensor=tensor,chunk_index=chunk,
                        compact_rows=rows,num_experts=experts,in_features=k,out_features=n,
                        **routing_record(starts,rows,experts)))
                return _original(*args,**kwargs)
            register(key,wrapped,replace=True)
        for case in cases:
            captures = []
            begin = len(report["records"])
            for enabled in (False,True,False):
                context.update(enabled=enabled,case_id=case["id"],sequence={})
                first = runner.prefill(case["prompt_token_ids"])
                first_hash = hashlib.sha256(first.logits.tobytes()).hexdigest()
                nxt = runner.step(int(first.token_id))
                runner.runtime.device_synchronize()
                captures.append(dict(prefill_logits=first_hash,next_logits=hashlib.sha256(nxt.logits.tobytes()).hexdigest(),
                                     state=_state_summary(runner)))
            if not captures[0]==captures[1]==captures[2]:
                raise AssertionError("routing instrumentation changes logits/state")
            if len(report["records"])==begin:
                raise AssertionError("routing capture did not engage")
            report["cases"].append(dict(id=case["id"],exact=True,captures=captures))
            print(case["id"],len(report["records"])-begin,"routing records; exact",flush=True)
            a.output.write_text(json.dumps(report,indent=2)+"\n")
        report["status"] = "captured_exact"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        for key,fn in originals.items():
            register(key,fn,replace=True)
        generator.close()
        report["memory_after_close"] = memory_stats()
        if report["memory_after_close"]["active_allocations"] != 0:
            report["status"] = "failed"
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        if report["memory_after_close"]["active_allocations"] != 0:
            raise RuntimeError("routing capture leaked tracked allocations")


if __name__=="__main__":
    main()
