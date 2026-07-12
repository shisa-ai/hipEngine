#!/usr/bin/env python3
"""Generate a compact MTP acceptance-rate status report.

This intentionally records current diagnostic status in one artifact so future
questions do not need to chase changelog prose plus scattered benchmark JSONs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_COMPARE_SUMMARY = Path(
    "benchmarks/results/2026-06-15-gfx1151-mtp-compare-20260615-060801-summary.json"
)
DEFAULT_DIAGNOSTICS_SUMMARY = Path(
    "benchmarks/results/2026-06-15-gfx1151-mtp-diagnostics-20260615-081020-summary.json"
)
DEFAULT_B2_SWEEP = Path("benchmarks/results/mtp-bench-1781845000-b2-sweep-summary.json")
DEFAULT_B2_COUNT = Path(
    "benchmarks/results/mtp-bench-1781844300-b2-count-prompt-visible-output.json"
)
DEFAULT_B2_MINIMAL = Path(
    "benchmarks/results/mtp-bench-1781843600-b2-minimal-visible-output.json"
)
DEFAULT_B1_VISIBLE = Path(
    "benchmarks/results/mtp-bench-1781842600-b1-cycles20-visible-output.json"
)
DEFAULT_GREETING_COMPARISON = Path(
    "benchmarks/results/mtp-bench-1781845600-b2-greeting-native-vs-llamacpp-topk-comparison.json"
)
DEFAULT_LLAMACPP_HIP = Path(
    "/tmp/hipengine-mtp-gfx1151-runs/20260615-060801/"
    "llamacpp-hip-ud-q4km-mtp-d32/summary.json"
)
DEFAULT_LLAMACPP_VULKAN = Path(
    "/tmp/hipengine-mtp-gfx1151-runs/20260615-060801/"
    "llamacpp-vulkan-ud-q4km-mtp-d32/summary.json"
)
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/mtp-acceptance-status-2026-06-20.json")
DEFAULT_MARKDOWN_OUTPUT = Path("benchmarks/results/mtp-acceptance-status-2026-06-20.md")


Json = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-summary", type=Path, default=DEFAULT_COMPARE_SUMMARY)
    parser.add_argument(
        "--diagnostics-summary",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_SUMMARY,
    )
    parser.add_argument("--b2-sweep", type=Path, default=DEFAULT_B2_SWEEP)
    parser.add_argument("--b2-count", type=Path, default=DEFAULT_B2_COUNT)
    parser.add_argument("--b2-minimal", type=Path, default=DEFAULT_B2_MINIMAL)
    parser.add_argument("--b1-visible", type=Path, default=DEFAULT_B1_VISIBLE)
    parser.add_argument(
        "--greeting-comparison",
        type=Path,
        default=DEFAULT_GREETING_COMPARISON,
    )
    parser.add_argument("--llamacpp-hip", type=Path, default=DEFAULT_LLAMACPP_HIP)
    parser.add_argument("--llamacpp-vulkan", type=Path, default=DEFAULT_LLAMACPP_VULKAN)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    args = parser.parse_args()

    report = build_report(
        compare_summary_path=args.compare_summary,
        diagnostics_summary_path=args.diagnostics_summary,
        b2_sweep_path=args.b2_sweep,
        b2_count_path=args.b2_count,
        b2_minimal_path=args.b2_minimal,
        b1_visible_path=args.b1_visible,
        greeting_comparison_path=args.greeting_comparison,
        llamacpp_hip_path=args.llamacpp_hip,
        llamacpp_vulkan_path=args.llamacpp_vulkan,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report) + "\n")
    print(json.dumps(report["headline"], indent=2))


def build_report(
    *,
    compare_summary_path: Path,
    diagnostics_summary_path: Path,
    b2_sweep_path: Path,
    b2_count_path: Path,
    b2_minimal_path: Path,
    b1_visible_path: Path,
    greeting_comparison_path: Path,
    llamacpp_hip_path: Path,
    llamacpp_vulkan_path: Path,
) -> Json:
    compare = load_json(compare_summary_path)
    diagnostics = load_json(diagnostics_summary_path)
    b2_sweep = load_json(b2_sweep_path)
    b2_count = load_json(b2_count_path)
    b2_minimal = load_json(b2_minimal_path)
    b1_visible = load_json(b1_visible_path)
    greeting = load_json(greeting_comparison_path)
    llama_hip = load_json(llamacpp_hip_path)
    llama_vulkan = load_json(llamacpp_vulkan_path)

    hipengine_current = current_hipengine_summary(compare)
    llama_hip_b4 = llama_row_summary(llama_hip, "b4", llamacpp_hip_path)
    llama_vulkan_b4 = llama_row_summary(llama_vulkan, "b4", llamacpp_vulkan_path)
    native_gguf = native_gguf_summary(
        b2_sweep=b2_sweep,
        b2_count=b2_count,
        b2_minimal=b2_minimal,
        b1_visible=b1_visible,
        greeting=greeting,
        paths={
            "b2_sweep": b2_sweep_path,
            "b2_count": b2_count_path,
            "b2_minimal": b2_minimal_path,
            "b1_visible": b1_visible_path,
            "greeting_comparison": greeting_comparison_path,
        },
    )
    diagnostic_budgets = diagnostic_budget_summary(diagnostics, diagnostics_summary_path)

    broad = {
        "source_artifact": str(compare_summary_path),
        "workload": "gfx1151 D32, 9 prompts, Qwen3.6-35B-A3B UD-Q4_K_M MTP GGUF",
        "hipengine_paro_mtp_b1": hipengine_current,
        "llamacpp_hip_b4": llama_hip_b4,
        "llamacpp_vulkan_b4": llama_vulkan_b4,
        "gaps_vs_llamacpp_b4": {
            "hip": compare_gap(hipengine_current, llama_hip_b4),
            "vulkan": compare_gap(hipengine_current, llama_vulkan_b4),
        },
    }

    report = {
        "schema": 1,
        "kind": "mtp_acceptance_status_report",
        "date": "2026-06-20",
        "status": "diagnostic_recorded",
        "performance_claim": False,
        "headline": headline(broad, native_gguf),
        "denominator_notes": {
            "accept_per_draft": "accepted draft tokens / proposed draft tokens",
            "accepted_per_output": "accepted draft tokens / visible output tokens",
            "llamacpp_accept_rate": "llama.cpp accepted drafts / proposed drafts",
            "decode_tok_s": "prompt-mean tok/s unless field name says wall_tps",
        },
        "current_broad_comparison": broad,
        "native_gguf_diagnostics": native_gguf,
        "hipengine_budget_diagnostics": diagnostic_budgets,
        "sources": {
            "compare_summary": str(compare_summary_path),
            "diagnostics_summary": str(diagnostics_summary_path),
            "llamacpp_hip_summary": str(llamacpp_hip_path),
            "llamacpp_vulkan_summary": str(llamacpp_vulkan_path),
            "b2_sweep": str(b2_sweep_path),
            "b2_count": str(b2_count_path),
            "b2_minimal": str(b2_minimal_path),
            "b1_visible": str(b1_visible_path),
            "greeting_comparison": str(greeting_comparison_path),
        },
        "next_actions": [
            "Continue correctness-first GGUF MTP parity bisection through layer-17 MoE.",
            (
                "Do not promote acceptance-rate claims until native GGUF "
                "candidate sets match llama.cpp."
            ),
            "Use accepted_per_output for hipEngine-vs-llama.cpp status comparisons.",
        ],
    }
    return report


def load_json(path: Path) -> Json:
    if not path.exists():
        raise FileNotFoundError(f"missing required source artifact: {path}")
    return json.loads(path.read_text())


def current_hipengine_summary(compare: Mapping[str, Any]) -> Json:
    item = compare["results"]["hipengine_paro_mtp_b1"]
    return {
        "artifact": item.get("artifact"),
        "budget": item["budget"],
        "prompts": item["prompts"],
        "exact_vs_ar": item["all_exact_ar_match"],
        "accepted_per_output": item["accepted_per_output"],
        "accepted_tokens_per_cycle": item["accepted_tokens_per_cycle"],
        "visible_tokens_per_cycle": item["visible_tokens_per_cycle"],
        "mtp_decode_tok_s_prompt_mean": item["mtp_decode_tok_s_prompt_mean"],
        "ar_decode_tok_s_prompt_mean": item["ar_decode_tok_s_prompt_mean"],
        "actual_decode_speedup_vs_ar_prompt_mean": item[
            "actual_decode_speedup_vs_ar_prompt_mean"
        ],
        "actual_decode_speedup_vs_ar_total_time": item[
            "actual_decode_speedup_vs_ar_total_time"
        ],
        "cycle_wall_ms_per_cycle": item["cycle_wall_ms_per_cycle"],
        "verify_ms_per_cycle": item["verify_ms_per_cycle"],
        "proposal_update_ms_per_cycle": item["proposal_update_ms_per_cycle"],
        "decode_tokens": item["decode_tokens"],
    }


def llama_row_summary(summary: Mapping[str, Any], mode: str, source: Path) -> Json:
    row = row_by_mode(summary, mode)
    base = row_by_mode(summary, "base")
    return {
        "source_artifact": str(source),
        "mode": row["mode"],
        "total_predicted": row["total_predicted"],
        "mean_tps": row["mean_tps"],
        "median_tps": row["median_tps"],
        "wall_tps": row["wall_tps"],
        "base_mean_tps": base["mean_tps"],
        "base_wall_tps": base["wall_tps"],
        "total_draft": row["total_draft"],
        "total_accepted": row["total_accepted"],
        "accept_rate": row["accept_rate"],
        "accepted_per_output": row["accepted_per_output"],
        "draft_per_output": row["draft_per_output"],
        "speedup_mean_vs_base": row["speedup_mean_vs_base"],
        "speedup_wall_vs_base": row["speedup_wall_vs_base"],
    }


def row_by_mode(summary: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    for row in summary["rows"]:
        if row.get("mode") == mode:
            return row
    raise KeyError(f"mode {mode!r} not found")


def compare_gap(hipengine: Mapping[str, Any], llama: Mapping[str, Any]) -> Json:
    accepted_gap = hipengine["accepted_per_output"] - llama["accepted_per_output"]
    decode_gap = hipengine["mtp_decode_tok_s_prompt_mean"] - llama["mean_tps"]
    return {
        "accepted_per_output_abs_gap": accepted_gap,
        "accepted_per_output_ratio": safe_ratio(
            hipengine["accepted_per_output"],
            llama["accepted_per_output"],
        ),
        "decode_tok_s_abs_gap": decode_gap,
        "decode_tok_s_ratio": safe_ratio(
            hipengine["mtp_decode_tok_s_prompt_mean"],
            llama["mean_tps"],
        ),
        "classification": "behind_llamacpp_b4"
        if accepted_gap < 0 and decode_gap < 0
        else "mixed_or_ahead",
    }


def native_gguf_summary(
    *,
    b2_sweep: Mapping[str, Any],
    b2_count: Mapping[str, Any],
    b2_minimal: Mapping[str, Any],
    b1_visible: Mapping[str, Any],
    greeting: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Json:
    prompts = list(b2_sweep["prompts"])
    total_accepted = int(b2_sweep["aggregate"]["total_accepted_draft_tokens"])
    total_drafts = int(b2_sweep["aggregate"]["total_generated_draft_tokens"])
    cycles = int(b2_sweep["cycles_per_prompt"]) * len(prompts)
    visible_outputs = cycles + total_accepted
    best_prompt = max(prompts, key=lambda item: item["warm_speedup_vs_ar_visible"])
    return {
        "status": "candidate_set_parity_incomplete",
        "b2_three_prompt_sweep": {
            "source_artifact": str(paths["b2_sweep"]),
            "total_accepted_draft_tokens": total_accepted,
            "total_generated_draft_tokens": total_drafts,
            "accept_per_draft": safe_ratio(total_accepted, total_drafts),
            "accepted_per_output": safe_ratio(total_accepted, visible_outputs),
            "mean_warm_speedup_vs_ar_visible": b2_sweep["aggregate"][
                "mean_warm_speedup_vs_ar_visible"
            ],
            "mean_cold_speedup_vs_ar_visible": b2_sweep["aggregate"][
                "mean_cold_speedup_vs_ar_visible"
            ],
            "warm_positive_prompts": b2_sweep["aggregate"]["warm_positive_prompts"],
            "depth2_accept_prompt_count": b2_sweep["aggregate"][
                "depth2_accept_prompt_count"
            ],
            "best_warm_prompt": {
                "slug": best_prompt["slug"],
                "warm_speedup_vs_ar_visible": best_prompt[
                    "warm_speedup_vs_ar_visible"
                ],
                "warm_accept_per_draft": best_prompt["warm_accept_per_draft"],
                "accepted_per_output": best_prompt["accepted_per_output"],
            },
        },
        "b2_count_prompt": metrics_summary(paths["b2_count"], b2_count["metrics"]),
        "b2_minimal_prompt": metrics_summary(
            paths["b2_minimal"],
            b2_minimal["metrics"],
        ),
        "b1_visible_output": metrics_summary(paths["b1_visible"], b1_visible["metrics"]),
        "greeting_blocker": {
            "source_artifact": str(paths["greeting_comparison"]),
            "finding": greeting["finding"],
            "native_artifact": greeting["native_artifact"],
            "llamacpp_trace": greeting["llamacpp_trace"],
            "llamacpp_draft_acceptance": greeting["llamacpp_summary"][
                "draft_acceptance"
            ],
            "llamacpp_draft_n_accepted": greeting["llamacpp_summary"][
                "draft_n_accepted"
            ],
            "llamacpp_response_reasoning_content": greeting[
                "llamacpp_response_reasoning_content"
            ],
        },
    }


def metrics_summary(source: Path, metrics: Mapping[str, Any]) -> Json:
    warm = metrics.get("warm_excluding_cycle0") or {}
    return {
        "source_artifact": str(source),
        "accept_per_draft": metrics["accept_per_draft"],
        "accepted_per_output": metrics["accepted_per_output"],
        "tokens_per_sec": metrics["tokens_per_sec"],
        "speedup_vs_ar_visible": metrics["speedup_vs_ar_visible"],
        "total_accepted": metrics["total_accepted"],
        "total_drafts": metrics["total_drafts"],
        "warm_excluding_cycle0": {
            "accept_per_draft": warm.get("accept_per_draft"),
            "accepted_per_output": warm.get("accepted_per_output"),
            "tokens_per_sec": warm.get("tokens_per_sec"),
            "speedup_vs_ar_visible": warm.get("speedup_vs_ar_visible"),
            "total_accepted": warm.get("total_accepted"),
            "total_drafts": warm.get("total_drafts"),
        },
    }


def diagnostic_budget_summary(diagnostics: Mapping[str, Any], source: Path) -> Json:
    hip = diagnostics["hipengine"]
    keys = ["exact_b1_decode_batched", "exact_b3_decode_batched", "exact_b1_c1_loop"]
    out = {"source_artifact": str(source)}
    for key in keys:
        item = hip[key]
        out[key] = {
            "mtp_decode_tok_s_prompt_mean": item["mtp_decode_tok_s_prompt_mean"],
            "ar_decode_tok_s_prompt_mean": item["ar_decode_tok_s_prompt_mean"],
            "speedup_prompt_mean": item["speedup_prompt_mean"],
            "accepted_total": item["accepted_total"],
            "accepted_per_output_aggregate": item[
                "accepted_per_output_aggregate"
            ],
            "accept_per_active_budget_aggregate": item[
                "accept_per_active_budget_aggregate"
            ],
            "accepted_tokens_per_cycle_aggregate": item[
                "accepted_tokens_per_cycle_aggregate"
            ],
        }
    return out


def headline(broad: Mapping[str, Any], native_gguf: Mapping[str, Any]) -> Json:
    hip = broad["hipengine_paro_mtp_b1"]
    llama_hip = broad["llamacpp_hip_b4"]
    llama_vk = broad["llamacpp_vulkan_b4"]
    sweep = native_gguf["b2_three_prompt_sweep"]
    return {
        "summary": (
            "hipEngine MTP acceptance is recorded but still behind llama.cpp: "
            "the broad gfx1151 B1 row is exact yet slower than AR and far below "
            "llama.cpp B4 accepted/output; native GGUF B2 remains parity-blocked."
        ),
        "current_status": "behind_llamacpp_and_candidate_set_parity_incomplete",
        "hipengine_b1_accepted_per_output": hip["accepted_per_output"],
        "llamacpp_hip_b4_accepted_per_output": llama_hip["accepted_per_output"],
        "llamacpp_vulkan_b4_accepted_per_output": llama_vk["accepted_per_output"],
        "hipengine_b1_decode_tok_s": hip["mtp_decode_tok_s_prompt_mean"],
        "llamacpp_hip_b4_mean_tps": llama_hip["mean_tps"],
        "llamacpp_vulkan_b4_mean_tps": llama_vk["mean_tps"],
        "native_gguf_b2_sweep_accept_per_draft": sweep["accept_per_draft"],
        "native_gguf_b2_sweep_accepted_per_output": sweep["accepted_per_output"],
        "native_gguf_b2_mean_warm_speedup_vs_ar": sweep[
            "mean_warm_speedup_vs_ar_visible"
        ],
        "native_gguf_b2_mean_cold_speedup_vs_ar": sweep[
            "mean_cold_speedup_vs_ar_visible"
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    head = report["headline"]
    broad = report["current_broad_comparison"]
    native = report["native_gguf_diagnostics"]
    gaps = broad["gaps_vs_llamacpp_b4"]
    lines = [
        "# MTP acceptance status — 2026-06-20",
        "",
        head["summary"],
        "",
        "## Current broad comparison",
        "",
        "| Engine | Budget | accepted/output | decode tok/s | speedup vs base/AR | Source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        broad_engine_row(
            "hipEngine PARO MTP",
            "B1",
            broad["hipengine_paro_mtp_b1"]["accepted_per_output"],
            broad["hipengine_paro_mtp_b1"]["mtp_decode_tok_s_prompt_mean"],
            broad["hipengine_paro_mtp_b1"][
                "actual_decode_speedup_vs_ar_prompt_mean"
            ],
            broad["source_artifact"],
        ),
        broad_engine_row(
            "llama.cpp HIP UD-Q4_K_M",
            "B4",
            broad["llamacpp_hip_b4"]["accepted_per_output"],
            broad["llamacpp_hip_b4"]["mean_tps"],
            broad["llamacpp_hip_b4"]["speedup_mean_vs_base"],
            broad["llamacpp_hip_b4"]["source_artifact"],
        ),
        broad_engine_row(
            "llama.cpp Vulkan UD-Q4_K_M",
            "B4",
            broad["llamacpp_vulkan_b4"]["accepted_per_output"],
            broad["llamacpp_vulkan_b4"]["mean_tps"],
            broad["llamacpp_vulkan_b4"]["speedup_mean_vs_base"],
            broad["llamacpp_vulkan_b4"]["source_artifact"],
        ),
        "",
        "Gaps: hipEngine accepted/output is "
        f"{gaps['hip']['accepted_per_output_ratio']:.3f}× llama.cpp HIP B4 and "
        f"{gaps['vulkan']['accepted_per_output_ratio']:.3f}× llama.cpp Vulkan B4; "
        f"decode tok/s is {gaps['hip']['decode_tok_s_ratio']:.3f}× HIP and "
        f"{gaps['vulkan']['decode_tok_s_ratio']:.3f}× Vulkan.",
        "",
        "## Native GGUF diagnostics",
        "",
        "| Row | accept/draft | accepted/output | warm speedup | cold speedup | Source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        native_row("B2 3-prompt sweep", native["b2_three_prompt_sweep"]),
        metrics_row("B2 count prompt", native["b2_count_prompt"]),
        metrics_row("B2 minimal prompt", native["b2_minimal_prompt"]),
        metrics_row("B1 visible-output", native["b1_visible_output"]),
        "",
        "Greeting blocker: llama.cpp accepted 3/4 traced drafts on the greeting "
        "prompt, while native depth-1 top-k omitted the target token; comparison "
        f"artifact: `{native['greeting_blocker']['source_artifact']}`.",
        "",
        "## Interpretation",
        "",
        "- The broad hipEngine row is exact but below AR (`0.912x` prompt-mean) and "
        "behind llama.cpp B4 on both accepted/output and decode tok/s.",
        "- The native GGUF B2 path shows isolated warm wins, but aggregate acceptance "
        "is too low and greeting remains candidate-set/parity blocked.",
        "- Use `accepted_per_output` for engine-to-engine status; `accept_per_draft` "
        "is useful within a fixed budget but not comparable across B1/B2/B4 alone.",
    ]
    return "\n".join(lines)


def broad_engine_row(
    name: str,
    budget: str,
    accepted_per_output: float,
    decode_tps: float,
    speedup: float,
    source: str,
) -> str:
    return (
        f"| {name} | {budget} | {accepted_per_output:.3f} | "
        f"{decode_tps:.2f} | {speedup:.3f}× | `{source}` |"
    )


def native_row(label: str, item: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {item['accept_per_draft']:.3f} | "
        f"{item['accepted_per_output']:.3f} | "
        f"{item['mean_warm_speedup_vs_ar_visible']:.3f}× | "
        f"{item['mean_cold_speedup_vs_ar_visible']:.3f}× | "
        f"`{item['source_artifact']}` |"
    )


def metrics_row(label: str, item: Mapping[str, Any]) -> str:
    warm = item["warm_excluding_cycle0"]
    return (
        f"| {label} | {item['accept_per_draft']:.3f} | "
        f"{item['accepted_per_output']:.3f} | "
        f"{warm['speedup_vs_ar_visible']:.3f}× | "
        f"{item['speedup_vs_ar_visible']:.3f}× | `{item['source_artifact']}` |"
    )


def safe_ratio(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


if __name__ == "__main__":
    main()
