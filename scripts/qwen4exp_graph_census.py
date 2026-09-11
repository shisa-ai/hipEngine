"""Summarize graph coverage and interval-safe gaps from a family capture."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def union_ns(intervals):
    total = 0
    right = None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative interval")
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def bounds(row):
    return int(row["Start_Timestamp"]), int(row["End_Timestamp"])


def clipped(row, window):
    start, end = bounds(row)
    start, end = max(start, window[0]), min(end, window[1])
    return (start, end) if end > start else None


def analyze_window(apis, kernels, window):
    if window[1] <= window[0]:
        raise ValueError("empty measurement window")
    selected_apis = [r for r in apis if clipped(r, window)]
    selected_kernels = [r for r in kernels if clipped(r, window)]
    # Correlation identity is a string: avoid lossy float parsing of timestamps/IDs.
    functions = {r["Correlation_Id"]: r["Function"] for r in selected_apis}
    graph_apis = [r for r in selected_apis if r["Function"].startswith("hipGraphLaunch")]
    graph_ids = {r["Correlation_Id"] for r in graph_apis}
    kernel_intervals = [clipped(r, window) for r in selected_kernels]
    graph_intervals = [clipped(r, window) for r in graph_apis]
    graph_groups = defaultdict(list)
    for r in selected_kernels:
        if r["Correlation_Id"] in graph_ids:
            graph_groups[r["Correlation_Id"]].append(clipped(r, window))
    intra_graph_gap = sum(
        max(e for _, e in v) - min(s for s, _ in v) - union_ns(v)
        for v in graph_groups.values())
    kernel_union = union_ns(kernel_intervals)
    return dict(
        window_ns=window[1] - window[0],
        api_calls=dict(Counter(r["Function"] for r in selected_apis)),
        api_duration_sum_ns=dict(
            (name, sum(clipped(r, window)[1] - clipped(r, window)[0]
                       for r in selected_apis if r["Function"] == name))
            for name in sorted({r["Function"] for r in selected_apis})),
        kernel_rows=len(selected_kernels), kernel_union_ns=kernel_union,
        non_kernel_window_upper_bound_ns=window[1] - window[0] - kernel_union,
        graph_launches=len(graph_apis),
        graph_kernel_rows=sum(len(v) for v in graph_groups.values()),
        graph_launches_without_kernel=len(graph_ids - graph_groups.keys()),
        graph_kernel_union_ns=union_ns([i for v in graph_groups.values() for i in v]),
        graph_api_non_kernel_ns=union_ns(kernel_intervals + graph_intervals) - kernel_union,
        intra_graph_gap_ns=intra_graph_gap,
        kernel_rows_without_api=sum(r["Correlation_Id"] not in functions for r in selected_kernels),
        boundary_crossing_kernel_rows=sum(bounds(r) != clipped(r, window) for r in selected_kernels))


def read_one(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern} in {directory}")
    with paths[0].open(newline="") as stream:
        return list(csv.DictReader(stream)), dict(
            path=str(paths[0]), sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())


def summarize(capture):
    if capture["status"] != "captured":
        raise ValueError("requires completed family capture")
    if not capture["source"].get("tracked_clean"):
        raise ValueError("requires clean captured runtime source")
    result = dict(
        schema=1, kind="qwen4exp_graph_census",
        measurement_class="diagnostic_profiled_not_performance",
        source=capture["source"], host=capture["host"],
        model_identity=capture["model_identity"], quant=capture["quant"],
        kv_dtype=capture["kv_dtype"], fixture_sha256=capture["fixture_sha256"],
        limitations=[
            "Non-kernel time includes copies, waits, Python and profiling; not all removable dispatch.",
            "API durations overlap kernels and each other; never add API sum to kernel sum.",
            "Intra-graph gaps are observed gaps, not a measured PM4 speedup or guaranteed ceiling.",
            "Kernel-only graph metadata is structural eligibility, not PM4 ABI/lifecycle qualification.",
            "External PM4 is disabled by activity profiling; this is the current HIP/AQL baseline."],
        cases=[])
    for case in capture["cases"]:
        child_path = Path(case["raw_path"])
        if hashlib.sha256(child_path.read_bytes()).hexdigest() != case["raw_sha256"]:
            raise ValueError("child capture hash mismatch")
        if json.loads(child_path.read_text()) != case["raw"]:
            raise ValueError("embedded child differs from hashed file")
        directory = child_path.parent / "trace" / capture["host"]["hostname"]
        apis, api_source = read_one(directory, "*_hip_api_trace.csv")
        kernels, kernel_source = read_one(directory, "*_kernel_trace.csv")
        markers, marker_source = read_one(directory, "*_marker_api_trace.csv")
        prefix = (f"qwen4exp_prefill_p{case['prompt_tokens']}_" if case["phase"] == "prefill"
                  else f"qwen4exp_decode_live{case['live_count']}_rep")
        windows = [r for r in markers if r["Function"].startswith(prefix)]
        if len(windows) != (1 if case["phase"] == "prefill" else 3):
            raise ValueError("unexpected measured window count")
        child = case["raw"]
        closed = child.get("memory_after_close", child.get("lifecycle", {}).get("after_close", {}))
        if closed.get("active_allocations") != 0 or closed.get("current_allocated_bytes") != 0:
            raise ValueError("capture ownership did not close")
        graph_info = {}
        if case["phase"] == "decode":
            context = child["contexts"][0]
            graph_info = {k: context[k] for k in ("graph_before", "graph_after", "summary")}
        result["cases"].append(dict(
            id=case["id"], phase=case["phase"], command=case["command"],
            raw_path=str(child_path), raw_sha256=case["raw_sha256"],
            lifecycle=child.get("lifecycle", {}),
            memory_after_close=closed,
            profile=case["profile"],
            correctness=(dict(logits_sha256=child["logits_sha256"],
                              token_id=child["token_id"])
                         if case["phase"] == "prefill" else child["status"]),
            capacity=(None if case["phase"] == "prefill"
                      else child.get("protocol", {}).get("allocated_capacity")),
            capacity_note=("Prefill child does not emit capacity; resolve its recorded command "
                           "against captured helper revision. Canonical default is max(768,prompt+1)."
                           if case["phase"] == "prefill" else "Explicit child protocol capacity."),
            trace_sources=[api_source, kernel_source, marker_source],
            graph_info=graph_info,
            windows=[dict(marker=r["Function"], **analyze_window(apis, kernels, bounds(r)))
                     for r in windows]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = summarize(json.loads(args.capture.read_text()))
    packet["capture_path"] = str(args.capture)
    packet["capture_sha256"] = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
