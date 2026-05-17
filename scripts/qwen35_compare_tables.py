#!/usr/bin/env python3
"""Print hardcoded Qwen/PARO comparison tables.

The values here are intentionally static: they summarize the current retained
resident-runner hipENGINE diagnostics and the external comparison rows we use
for quick status checks.  They are not a benchmark runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Row:
    workload: str
    prefill_tok_s: float | None
    decode_tok_s: float | None
    peak_gib: float | None


@dataclass(frozen=True)
class Series:
    key: str
    display: str
    source: str
    notes: str
    rows: tuple[Row, ...]


QWEN35_SOURCE = "benchmarks/results/2026-05-17-hipengine-qwen35-d31-d33-grouped-gqa-long-context-diagnostic.json"
SHISA_SOURCE = "benchmarks/results/2026-05-17-hipengine-qwen36-shisa-packed-vs-legacy-refresh-diagnostic.json"

TARGETS: dict[str, Series] = {
    "qwen35-current": Series(
        key="qwen35-current",
        display="hipENGINE Qwen3.5 current",
        source=QWEN35_SOURCE,
        notes=(
            "Qwen3.5-35B-A3B-PARO w4_paro resident-runner diagnostic with current defaults: "
            "AOTriton prefill threshold 512, graph-replay decode, Marlin-K decode, and D3.1-D3.3 "
            "grouped-GQA long-context decode. Long rows use parent-style chunk flags."
        ),
        rows=(
            Row("512/128", 2177.649, 115.627, 18.176),
            Row("4K/128", 2449.055, 116.263, 20.047),
            Row("32K/128", 1964.345, 99.560, 20.320),
            Row("128K/128", 1015.761, 63.368, 23.288),
        ),
    ),
    "shisa-packed": Series(
        key="shisa-packed",
        display="hipENGINE shisa Qwen3.6 packed PARO",
        source=SHISA_SOURCE,
        notes=(
            "shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5 unstripped checkpoint forced to "
            "shared_expert_format=packed_paro_w4; packed is the default A-side for shisa comparisons. "
            "512/4K rows are no-chunk short rows; 32K/128K rows use parent-style chunk flags."
        ),
        rows=(
            Row("512/128", 2518.836, 111.738, 18.123),
            Row("4K/128", 2711.013, 113.231, 19.995),
            Row("32K/128", 2130.562, 97.779, 20.267),
            Row("128K/128", 1048.543, 62.014, 23.235),
        ),
    ),
    "shisa-legacy": Series(
        key="shisa-legacy",
        display="hipENGINE shisa Qwen3.6 legacy shared expert",
        source=SHISA_SOURCE,
        notes=(
            "same shisa unstripped checkpoint forced to shared_expert_format=legacy_fp16. "
            "Use --target shisa-legacy to make legacy the A-side, or --against-target with no value "
            "to compare packed A against legacy B."
        ),
        rows=(
            Row("512/128", 2272.088, 115.324, 18.176),
            Row("4K/128", 2487.298, 116.688, 20.047),
            Row("32K/128", 1974.833, 99.746, 20.320),
            Row("128K/128", 1002.841, 63.190, 23.288),
        ),
    ),
}

BASELINES: dict[str, Series] = {
    "nano-vllm-amd": Series(
        key="nano-vllm-amd",
        display="nano-vllm-amd parent",
        source="~/amd-gpu-tuning/docs/OPTIMAL.md Latest Results plus local 2026-05-13 reruns",
        notes="Qwen3.5-35B-A3B-PARO parent compact-WMMA + graph-replay rows, graph/step true.",
        rows=(
            Row("512/128", 2696.4, 116.05, 18.80),
            Row("4K/128", 2741.5, 113.05, 21.64),
            Row("32K/128", 1880.0, 98.8, 21.37),
            Row("128K/128", 914.0, 62.6, 27.42),
        ),
    ),
    "llama.cpp-hip": Series(
        key="llama.cpp-hip",
        display="llama.cpp HIP",
        source="~/amd-gpu-tuning/PLAN-LONGCONTEXT.md split rows",
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF, f16 KV, split pp/tg rows with decode depth. "
            "Peak GiB from benchmarks/results/2026-05-17-llamacpp-hip-qwen36-peak.json."
        ),
        rows=(
            Row("512/128", 2436.049, 85.487, 21.125),
            Row("4K/128", 2176.905, 87.375, 21.197),
            Row("32K/128", 1496.409, 76.994, 21.738),
            Row("128K/128", 710.213, 57.341, 23.605),
        ),
    ),
    "llama.cpp-vulkan": Series(
        key="llama.cpp-vulkan",
        display="llama.cpp Vulkan",
        source="~/amd-gpu-tuning/PLAN-LONGCONTEXT.md split rows",
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF, f16 KV, split pp/tg rows with decode depth. "
            "Peak GiB from benchmarks/results/2026-05-17-llamacpp-vulkan-qwen36-peak.json."
        ),
        rows=(
            Row("512/128", 1816.927, 127.515, 20.844),
            Row("4K/128", 1705.093, 120.163, 20.969),
            Row("32K/128", 1128.554, 98.073, 21.533),
            Row("128K/128", 480.539, 64.478, 23.596),
        ),
    ),
}

BASELINE_ALIASES = {
    "nano": "nano-vllm-amd",
    "nano-vllm": "nano-vllm-amd",
    "nano-vllm-amd": "nano-vllm-amd",
    "parent": "nano-vllm-amd",
    "llama.cpp hip": "llama.cpp-hip",
    "llama.cpp-hip": "llama.cpp-hip",
    "llamacpp-hip": "llama.cpp-hip",
    "hip": "llama.cpp-hip",
    "llama.cpp vulkan": "llama.cpp-vulkan",
    "llama.cpp-vulkan": "llama.cpp-vulkan",
    "llamacpp-vulkan": "llama.cpp-vulkan",
    "vulkan": "llama.cpp-vulkan",
}

TARGET_ALIASES = {
    "qwen35": "qwen35-current",
    "qwen3.5": "qwen35-current",
    "qwen35-current": "qwen35-current",
    "current": "qwen35-current",
    "hipengine": "qwen35-current",
    "shisa": "shisa-packed",
    "qwen36": "shisa-packed",
    "qwen3.6": "shisa-packed",
    "packed": "shisa-packed",
    "packed-paro": "shisa-packed",
    "packed-paro-w4": "shisa-packed",
    "packed_paro_w4": "shisa-packed",
    "shisa-packed": "shisa-packed",
    "legacy": "shisa-legacy",
    "legacy-fp16": "shisa-legacy",
    "legacy_fp16": "shisa-legacy",
    "unpacked": "shisa-legacy",
    "shisa-legacy": "shisa-legacy",
}


def _normalized_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", "-").split())


def _normalize_baseline(value: str) -> str:
    key = _normalized_key(value)
    key = BASELINE_ALIASES.get(key, key)
    if key not in BASELINES and key != "all":
        valid = ", ".join(["nano-vllm-amd", "llama.cpp-hip", "llama.cpp-vulkan", "all"])
        raise SystemExit(f"unknown baseline {value!r}; choose one of: {valid}")
    return key


def _normalize_target(value: str) -> str:
    key = _normalized_key(value)
    key = TARGET_ALIASES.get(key, key)
    if key not in TARGETS:
        valid = ", ".join(["qwen35-current", "shisa-packed", "shisa-legacy"])
        raise SystemExit(f"unknown target {value!r}; choose one of: {valid}")
    return key


def _auto_compare_target(target_key: str) -> str:
    return "shisa-packed" if target_key == "shisa-legacy" else "shisa-legacy"


def _row_map(rows: Iterable[Row]) -> dict[str, Row]:
    return {row.workload: row for row in rows}


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_gib(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_pct(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None or baseline == 0:
        return "—"
    return f"{(current / baseline - 1.0) * 100.0:+.1f}%"


def _fmt_gib_delta(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "—"
    return f"{current - baseline:+.2f} GiB"


def _print_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    print(f"### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" if idx == 0 else "---:" for idx, _ in enumerate(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def _shared_workloads(left: Series, right: Series) -> list[str]:
    right_workloads = set(_row_map(right.rows))
    return [row.workload for row in left.rows if row.workload in right_workloads]


def print_comparison(target: Series, baseline: Series) -> None:
    left = _row_map(target.rows)
    right = _row_map(baseline.rows)
    workloads = _shared_workloads(target, baseline)

    print(f"## {target.display} vs {baseline.display}\n")
    print(f"A target source: {target.source}")
    print(f"B baseline source: {baseline.source}")
    print(f"notes: {target.notes} {baseline.notes}\n")

    _print_table(
        "Prefill",
        ("Workload", f"{target.display} tok/s", f"{baseline.display} tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].prefill_tok_s),
                _fmt_rate(right[workload].prefill_tok_s),
                _fmt_pct(left[workload].prefill_tok_s, right[workload].prefill_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Decode",
        ("Workload", f"{target.display} tok/s", f"{baseline.display} tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].decode_tok_s),
                _fmt_rate(right[workload].decode_tok_s),
                _fmt_pct(left[workload].decode_tok_s, right[workload].decode_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Memory / peak GiB",
        ("Workload", f"{target.display} peak GiB", f"{baseline.display} peak GiB", "Delta A vs B"),
        [
            (
                workload,
                _fmt_gib(left[workload].peak_gib),
                _fmt_gib(right[workload].peak_gib),
                _fmt_gib_delta(left[workload].peak_gib, right[workload].peak_gib),
            )
            for workload in workloads
        ],
    )


def print_target_comparison(target: Series, compare_target: Series) -> None:
    left = _row_map(target.rows)
    right = _row_map(compare_target.rows)
    workloads = _shared_workloads(target, compare_target)

    print(f"## {target.display} (A) vs {compare_target.display} (B)\n")
    print(f"A source: {target.source}")
    print(f"B source: {compare_target.source}")
    print(f"A notes: {target.notes}")
    print(f"B notes: {compare_target.notes}\n")

    _print_table(
        "Prefill",
        ("Workload", "A tok/s", "B tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].prefill_tok_s),
                _fmt_rate(right[workload].prefill_tok_s),
                _fmt_pct(left[workload].prefill_tok_s, right[workload].prefill_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Decode",
        ("Workload", "A tok/s", "B tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].decode_tok_s),
                _fmt_rate(right[workload].decode_tok_s),
                _fmt_pct(left[workload].decode_tok_s, right[workload].decode_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Memory / peak GiB",
        ("Workload", "A peak GiB", "B peak GiB", "Delta A vs B"),
        [
            (
                workload,
                _fmt_gib(left[workload].peak_gib),
                _fmt_gib(right[workload].peak_gib),
                _fmt_gib_delta(left[workload].peak_gib, right[workload].peak_gib),
            )
            for workload in workloads
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline",
        nargs="?",
        default="all",
        help=(
            "Comparison baseline: nano-vllm-amd, llama.cpp-hip, llama.cpp-vulkan, or all. "
            "Ignored when --against-target is set. Default: all."
        ),
    )
    parser.add_argument(
        "--target",
        default="qwen35-current",
        help=(
            "A-side hipENGINE target: qwen35-current, shisa-packed, or shisa-legacy. "
            "Aliases: qwen35, shisa/packed, legacy/unpacked. Default: qwen35-current."
        ),
    )
    parser.add_argument(
        "--against-target",
        "--compare-target",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Compare the A-side target against another hipENGINE target instead of external baselines. "
            "With no value, uses legacy when A is packed/shisa and packed when A is legacy."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_key = _normalize_target(args.target)
    target = TARGETS[target_key]

    if args.against_target is not None:
        compare_key = _auto_compare_target(target_key) if args.against_target == "auto" else _normalize_target(args.against_target)
        print_target_comparison(target, TARGETS[compare_key])
        return

    key = _normalize_baseline(args.baseline)
    if key == "all":
        for index, baseline in enumerate(BASELINES.values()):
            if index:
                print("---\n")
            print_comparison(target, baseline)
    else:
        print_comparison(target, BASELINES[key])


if __name__ == "__main__":
    main()
