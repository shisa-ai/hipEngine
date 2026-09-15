"""Check a prefill-only owner without allocating a donor decoder/KV cache."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown
from scripts.qwen4exp_chunk_boundary_gate import trajectory
from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk
from scripts.qwen4exp_chunk_workspace import capture_workspace, use_workspace
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_layer2_profile_gate import _make_generator
from scripts.qwen4exp_q8_repair_depth_gate import resolve_allocation_profile, validate_chunk_allocation

PREFILL_OWNERS = (
    "gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
    "qsa_prefill_metadata", "_prefill_buffers",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--allocation-evidence", type=Path, required=True)
    parser.add_argument("--reference-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        parser.error("owned-workspace capture requires clean tracked source")
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host, model = _host_metadata(), model_identity(args.model_root)
        allocation = json.loads(args.allocation_evidence.read_bytes())
        resolved = resolve_allocation_profile()
        validate_chunk_allocation(allocation, chunk=4096, context=4352,
                                  manifest=resolved.manifest_sha256, host=host, model=model)
        frozen = json.loads(args.reference_capture.read_bytes())
        if (frozen["status"] != "passed" or not frozen["source"]["tracked_clean"]
                or frozen["model"] != model or frozen["host"]["machine_id"] != host["machine_id"]
                or frozen["manifest"] != resolved.manifest_sha256
                or frozen["protocol"]["chunk"] != 4096
                or frozen["protocol"]["capacity"] != 4352):
            raise ValueError("incompatible frozen isolated reference")
        fixture, digest = load_fixture(DEFAULT_FIXTURE)
        sources = {case["id"]: case["prompt_token_ids"] for case in fixture["cases"]}
        prompts = {"a": sources["code-p4096"][:2052],
                   "b": sources["general_ja-p4096"] + [sources["general_ja-p4096"][-1]],
                   "c": sources["mixed_ja_en-p4096"][:2049]}
        report = dict(status="running", source=source, host=host, model=model,
                      command=sys.argv, fixture_sha256=digest, cases=[],
                      reference_sha256=hashlib.sha256(args.reference_capture.read_bytes()).hexdigest(),
                      performance_claim=False, promotion_claim=False,
                      limits="Prefill-only owner preparation/borrowing; automatic selection is not enabled.")
        arm_args = SimpleNamespace(**vars(args), max_sequence_length=4352, prefill_chunk_size=4096)
        generator, profile, _ = _make_generator(arm_args, "production")
        report["manifest"] = profile.manifest_sha256
        runner = generator.runner
        try:
            prepare_lazy_group_risk(runner)
            before = memory_stats()["current_allocated_bytes"]
            plan = qwen4_exp_scratch_breakdown(
                runner.config, context_tokens=4352, prefill_chunk_size=1024)
            extra_bytes = sum(plan[key] for key in PREFILL_OWNERS)
            free, _ = runner.runtime.mem_get_info()
            if free < extra_bytes + (4 << 30):
                raise MemoryError("extra workspace would consume reserve")
            state_owner, attention_owners = runner.state, runner.attention_states
            extra = runner._allocate_extra_prefill_workspace(1024)
            prepare_lazy_group_risk(extra)
            actual_extra = memory_stats()["current_allocated_bytes"] - before
            if actual_extra != extra_bytes:
                raise ValueError("extra workspace allocation does not match its prefill-only budget")
            if runner.state is not state_owner or runner.attention_states is not attention_owners:
                raise ValueError("prefill preparation replaced decode state")
            free_after, _ = runner.runtime.mem_get_info()
            if free_after < 4 << 30:
                raise MemoryError("extra workspace preparation consumed reserve")
            report["extra_workspace_bytes"] = actual_extra
            report["free_before_extra"] = free
            report["free_after_extra"] = free_after
            report["reserve_bytes"] = 4 << 30
            report["owner_count"] = len(runner._prefill_workspaces)
            view = capture_workspace(extra)
            for name, prompt in prompts.items():
                logits, tokens, baseline = trajectory(runner, prompt, steps=8, chunk=4096)
                expected = frozen["references"][name][8]
                if (baseline["final"] != expected["state"]
                        or hashlib.sha256(logits[-1].tobytes()).hexdigest() != expected["logits_sha256"]):
                    raise ValueError("native workspace changed frozen model output/state")
                for repeat in range(3):
                    with use_workspace(runner, view):
                        actual, _, payload = trajectory(
                            runner, prompt, steps=8, chunk=1024, teacher=tokens)
                    if (not np.array_equal(actual, logits)
                            or any(payload[phase] != baseline[phase] for phase in ("prefill", "final"))):
                        raise ValueError("owned borrowed workspace changed model output/state")
                    report["cases"].append(dict(
                        role=name, repeat=repeat, rows=len(logits), exact=True,
                        native_chunks=baseline["chunks"], borrowed_chunks=payload["chunks"],
                        final_state=payload["final"]))
                    print("owned workspace", name, repeat, "passed", flush=True)
            report["status"] = "passed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            generator.close()
            report["memory_after_close"] = memory_stats()
            if (_git_metadata(ROOT) != source or report["memory_after_close"]["current_allocated_bytes"]):
                report["status"] = "invalid_source_or_lifecycle"
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
