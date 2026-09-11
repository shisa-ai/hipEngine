"""Validate and compact the three-engine Framework screening capture."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, summarize_samples, token_ids_sha256,
)
from scripts.qwen4exp_framework_family_refresh import HOST_ID, MODEL_REVISION, PIN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_samples(rows):
    fixture, _ = load_fixture(DEFAULT_FIXTURE)
    cases = {c["id"]: c for c in fixture["cases"]}
    expected = {(case, rep) for case in cases for rep in range(3)}
    if len(rows) != len(expected) or {(r["case_id"], r["repetition"]) for r in rows} != expected:
        raise ValueError("requires all 12 cases with three distinct repetitions")
    for row in rows:
        case = cases[row["case_id"]]
        if (row["category"] != case["category"]
                or row["prompt_tokens"] != case["prompt_tokens"]
                or row["prompt_token_ids_sha256"] != token_ids_sha256(case["prompt_token_ids"])):
            raise ValueError("sample does not match canonical prompt")
        if (row["decode_transitions"] != 128 or row["output_token_count"] != 129
                or len(row["output_token_ids"]) != 129
                or row["output_token_ids_sha256"] != token_ids_sha256(row["output_token_ids"])):
            raise ValueError("output count/hash or decode boundary mismatch")
        for key in ("prefill_ms", "decode_ms", "prefill_tok_s", "decode_tok_s", "client_wall_s"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("non-finite or non-positive timing")
        if not math.isclose(row["prefill_tok_s"], row["prompt_tokens"]*1000/row["prefill_ms"],
                            rel_tol=1e-6):
            raise ValueError("prefill rate/timing mismatch")
        if not math.isclose(row["decode_tok_s"], 128000/row["decode_ms"], rel_tol=1e-6):
            raise ValueError("decode rate/timing mismatch")
    summary = summarize_samples(rows)
    if not summary["all_cases_deterministic"]:
        raise ValueError("within-engine outputs are not repeatable")
    return summary


def variance_report(rows, summary):
    """Describe within-case spread and repetition drift without pooling workloads."""
    report = {}
    for metric in ("prefill_tok_s", "decode_tok_s"):
        by_case = {}
        for case, stats in summary["cases"].items():
            ordered = sorted((r for r in rows if r["case_id"] == case),
                             key=lambda r: r["repetition"])
            values = [r[metric] for r in ordered]
            by_case[case] = {
                **stats[metric],
                "n": len(values),
                "range": max(values)-min(values),
                "last_vs_first_percent": 100*(values[-1]/values[0]-1),
                "repetition_order": [r["repetition"] for r in ordered],
            }
        cvs = [v["coefficient_of_variation"] for v in by_case.values()]
        report[metric] = dict(
            cases=by_case, median_within_case_cv=statistics.median(cvs),
            max_within_case_cv=max(cvs),
            cases_above_two_percent_cv=[k for k, v in by_case.items()
                                       if v["coefficient_of_variation"] > .02],
        )
    report["interpretation"] = (
        "Sample SD uses n-1. CV is SD/mean within one engine/case, not spread across workloads. "
        "Last/first is descriptive repetition-order drift, not a causal trend estimate. "
        "Three samples cannot establish stable tails or independent-sample confidence; no significance claim.")
    return report


def compact_capture(path):
    controller = json.loads(path.read_text())
    if controller["status"] != "captured" or not controller["source"]["tracked_clean"]:
        raise ValueError("requires completed clean-source controller")
    if controller["host"]["machine_id"] != HOST_ID:
        raise ValueError("physical host mismatch")
    stages = controller["stages"]
    if [s["engine"] for s in stages] != ["hipengine", "halo-box-vulkan", "halo-box-hip"]:
        raise ValueError("requires all three serial engine stages")
    _, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    engines = []
    for stage in stages:
        if digest(stage["path"]) != stage["sha256"]:
            raise ValueError("child artifact hash mismatch")
        raw = json.loads(Path(stage["path"]).read_text())
        if (raw["status"] != "completed" or raw["fixture_sha256"] != fixture_hash
                or raw["host"]["machine_id"] != HOST_ID or not raw["source"]["tracked_clean"]):
            raise ValueError("child status/fixture/host/source mismatch")
        summary = validate_samples(raw["samples"])
        if stage["engine"] == "hipengine":
            if raw["source"] != controller["source"] or raw["profile"]["requested"] != "production":
                raise ValueError("hipEngine source/profile mismatch")
            if raw["profile"]["fell_back_to_strict"]:
                raise ValueError("production fell back to strict")
            if any(raw["memory_after_close"][k] for k in
                   ("active_allocations", "current_allocated_bytes")):
                raise ValueError("hipEngine ownership did not close")
        else:
            if raw["source"]["head"] != PIN or raw["server_returncode"] != 0:
                raise ValueError("comparator source/teardown mismatch")
            if raw["server_binary_sha256"] not in controller["binary_hashes"].values():
                raise ValueError("comparator binary mismatch")
            for flag in ("-ctk", "-ctv"):
                if raw["command"][raw["command"].index(flag)+1] != "bf16":
                    raise ValueError("comparator KV mismatch")
        engine = {k: raw[k] for k in (
            "engine", "source", "host", "protocol", "fixture_sha256",
        )}
        for key in ("profile", "model_root", "model", "command", "server_binary",
                    "server_binary_sha256", "server_returncode", "memory_after_close"):
            if key in raw:
                engine[key] = raw[key]
        engine.update(
            stage=stage["engine"], summary=summary,
            variance=variance_report(raw["samples"], summary),
            max_prefill_cv=max(c["prefill_tok_s"]["coefficient_of_variation"]
                               for c in summary["cases"].values()),
            max_decode_cv=max(c["decode_tok_s"]["coefficient_of_variation"]
                              for c in summary["cases"].values()),
            samples=[{k: r[k] for k in (
                "case_id", "repetition", "prefill_ms", "decode_ms", "client_wall_s",
                "prompt_token_ids_sha256", "output_token_ids_sha256",
            )} for r in raw["samples"]],
        )
        engines.append(engine)
    he, vk = engines[:2]
    ratios = {
        shape: {
            metric: vk["summary"]["shapes"][shape][metric] / he["summary"]["shapes"][shape][metric]
            for metric in ("prefill_tok_s_weighted", "decode_tok_s_weighted")
        }
        for shape in ("512", "1024", "4096")
    }
    return dict(
        schema=1, kind="qwen4exp_framework_baseline_refresh",
        status="captured_screening_not_statistical_closure", performance_claim=True,
        controller=controller, controller_sha256=digest(path),
        elapsed_seconds=sum(s["elapsed_seconds"] for s in stages),
        engines=engines, vulkan_over_hipengine=ratios,
        limitations=[
            "Sequential full suites on one exclusive host, not inter-engine counterbalanced experiments.",
            "One warmup and three repetitions per case; report all per-case variance, not statistical closure.",
            "Profiler/logger off for throughput; semantic family captures use separate instruments.",
            "Within-engine repeatability is required; cross-engine generated-ID equality is diagnostic only.",
        ],
    )


def render_table(packet):
    lines = [
        "| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG | Max PP / TG CV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for engine in packet["engines"]:
        cells = [
            f'{engine["summary"]["shapes"][s]["prefill_tok_s_weighted"]:.2f} / '
            f'{engine["summary"]["shapes"][s]["decode_tok_s_weighted"]:.2f}'
            for s in ("512", "1024", "4096")
        ]
        lines.append("| " + engine["stage"] + " | " + " | ".join(cells) +
                     f' | {100*engine["max_prefill_cv"]:.2f}% / {100*engine["max_decode_cv"]:.2f}% |')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    packet = compact_capture(args.input)
    he = packet["engines"][0]
    packet["hipengine_artifact_provenance"] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", device_name="AMD Radeon 8060S",
        model_path=he["model_root"], model_revision=MODEL_REVISION,
        quant="UD-Q4_K_XL", kv_dtype="BF16",
        command=packet["controller"]["stages"][0]["command"],
        timing_protocol=he["protocol"]["timing_boundary"], warmups=1, repetitions=3,
        hipcc_version=he["host"]["hipcc_version"])
    packet["provenance_note"] = (
        "Canonical provenance collected during packaging; controller retains the clean measurement "
        "revision and command/binary hashes. Each engine has independent source and protocol metadata.")
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(render_table(packet))
    print(render_table(packet), end="")


if __name__ == "__main__":
    main()
