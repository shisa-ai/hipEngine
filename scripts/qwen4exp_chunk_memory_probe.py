"""Measure native-context/c2 allocation against the existing scratch allowance."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def allocation_margins(plan, allocated_bytes):
    components = sum(plan[k] for k in (
        "device_weight_bytes","staging_bytes","kv_bytes","index_bytes","runtime_state_bytes"))
    return {
        "explicit_components_bytes":components,
        "observed_remainder_bytes":allocated_bytes-components,
        "scratch_allowance_bytes":plan["scratch_bytes"],
        "scratch_margin_bytes":components+plan["scratch_bytes"]-allocated_bytes,
        "including_reserve_margin_bytes":plan["required_bytes"]-allocated_bytes,
    }


def device_allocation_margins(plan, allocated_bytes):
    components = sum(plan[k] for k in (
        "device_weight_bytes", "kv_bytes", "index_bytes", "runtime_state_bytes"))
    return {
        "device_components_bytes": components,
        "host_staging_reservation_bytes": plan["staging_bytes"],
        "observed_device_scratch_bytes": allocated_bytes - components,
        "device_scratch_margin_bytes": components + plan["scratch_bytes"] - allocated_bytes,
    }


def resolve_context_length(requested, native):
    value = int(native) if requested is None else int(requested)
    if value <= 0:
        raise ValueError("positive context length required")
    return value


def prepare_lazy_group_risk(runner):
    """Reserve each shared repair queue for the largest gate/up/down output."""
    cfg = runner.config
    rows = min(runner.prefill_chunk_size, runner.max_sequence_length)
    compact = rows * cfg.expert_used_count
    width = max(cfg.hidden_size, 2 * cfg.expert_feed_forward_length)
    records = []
    for name in ("gdn_prefill_scratch", "qsa_prefill_scratch"):
        count, indices = getattr(runner, name).moe.ensure_group_risk_buffers(
            compact_rows=compact, out_features_total=width)
        records.append(dict(owner=name, rows=rows, compact_rows=compact,
                            output_width=width, nbytes=count.nbytes + indices.nbytes))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--chunk-size",type=int,default=1024)
    p.add_argument("--capacity",type=int,choices=(1,2),default=2)
    p.add_argument("--context-length", type=int,
                   help="Explicit context for bounded chunk screens; default is model native context")
    args = p.parse_args()
    if args.chunk_size < 1:
        p.error("positive chunk size required")
    if args.context_length is not None and args.context_length < 1:
        p.error("positive context length required")
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile,resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,QWEN4_EXP_MODEL,QWEN4_EXP_BACKEND,QWEN4_EXP_QUANTS)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files,load_gguf_index
    from hipengine.models import resolve_model
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata,_host_metadata
    from scripts.qwen4exp_framework_family_refresh import check_host,model_identity
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        raise ValueError("memory probe requires a clean tracked revision")
    identity = model_identity(args.model_root)
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(model=QWEN4_EXP_MODEL,backend=QWEN4_EXP_BACKEND,
                                      quant=QWEN4_EXP_QUANTS[1],profile=ExecutionProfile.PRODUCTION)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    context_length = resolve_context_length(args.context_length, plugin.native_context_length)
    report = dict(schema=3,status="running",performance_claim=False,source=source,
                  host=_host_metadata(),model_identity=identity,command=sys.argv,
                  chunk_size=args.chunk_size,resident_capacity=args.capacity,
                  requested_context_length=context_length,
                  manifest_sha256=profile.manifest_sha256,
                  limits="Allocation/admission only; constructor and worst-case grouped repair queues. "
                         "No inference, hidden-seed export, graph capture or driver-owned scratch certificate. "
                         "Existing reserve remains.")
    generator = None
    start = time.monotonic()
    try:
        generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root,weight_index=index,model_plugin=plugin,
            backend="hip_gfx1151",max_sequence_length=context_length,
            resident_capacity=args.capacity,prefill_chunk_size=args.chunk_size))
        report["admission"] = generator.context_admission
        report["first_runner_memory"] = memory_stats()
        serving = generator.create_resident_model_runner(capacity=args.capacity)
        context = serving.prepare()
        assert context == context_length
        assert len(serving._all_runners) == args.capacity
        report["before_lazy_memory"] = memory_stats()
        report["lazy_group_risk"] = [
            dict(runner_index=i, queues=prepare_lazy_group_risk(runner))
            for i, runner in enumerate(serving._all_runners)]
        report["prepared_context"] = context
        report["prepared_runners"] = len(serving._all_runners)
        report["prepared_memory"] = memory_stats()
        plan = report["admission"]["plan"]
        report["allocation_margins"] = allocation_margins(
            plan,report["prepared_memory"]["current_allocated_bytes"])
        report["device_allocation_margins"] = device_allocation_margins(
            plan, report["prepared_memory"]["current_allocated_bytes"])
        if report["device_allocation_margins"]["device_scratch_margin_bytes"] < 0:
            raise AssertionError("device allocation exceeds device components plus scratch allowance")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if generator is not None:
                generator.close()
        except Exception as error:
            report["status"] = "failed"
            report["close_error"] = f"{type(error).__name__}: {error}"
        report["memory_after_close"] = memory_stats()
        report["elapsed_seconds"] = time.monotonic()-start
        if report["memory_after_close"]["active_allocations"]:
            report["status"] = "failed"
            report["ownership_error"] = "tracked allocations remain after close"
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
