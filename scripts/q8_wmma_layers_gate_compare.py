"""Compare Q8 dense WMMA prefill layer-scope gate arms on a common basis.

Arms that ran different case sets are not directly comparable: a scope run on
four categories and a scope run on ``code`` alone differ in both scope and
prompt mix, so any headline difference confounds the two.  This tool reports
only the categories the arms share, and separates the two things a KL delta can
mean -- a larger per-row perturbation, or the same perturbation landing on more
near-tie rows.

Usage::

    python3 scripts/q8_wmma_layers_gate_compare.py ARTIFACT [ARTIFACT ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

CATEGORY_METRICS = (
    ("kl_mean", "mean KL"),
    ("kl_p95", "p95 KL"),
    ("kl_max", "max KL"),
    ("top1_agreement", "top-1"),
    ("max_abs_logit_delta", "max |dlogit|"),
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("quality", {})
    return {
        "path": path,
        "layers": payload.get("route", {}).get("candidate_layers", ""),
        "label": _scope_label(payload.get("route", {}).get("candidate_layers", "")),
        "summary": quality.get("summary", {}),
        "category": quality.get("by_scope", {}).get("category", {}),
        "shape": quality.get("by_scope", {}).get("shape", {}),
        "mismatches": quality.get("top1_mismatch_rows", []),
        "provenance": payload.get("provenance", {}),
        "measurement_valid": payload.get("measurement_valid"),
    }


def _scope_label(layers: str) -> str:
    values = [int(part) for part in layers.split(",") if part.strip()]
    if not values:
        return "(none)"
    if values == list(range(min(values), max(values) + 1)):
        return f"{min(values)}-{max(values)}"
    return layers


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value and abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _shared_categories(arms: Sequence[dict[str, Any]]) -> list[str]:
    shared = set(arms[0]["category"])
    for arm in arms[1:]:
        shared &= set(arm["category"])
    return sorted(shared)


def render(arms: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Layer-scope arms\n")
    for arm in arms:
        summary = arm["summary"]
        lines.append(
            f"- layers {arm['label']}: {summary.get('rows', 0)} rows, "
            f"{len(arm['category'])} categories "
            f"({', '.join(sorted(arm['category'])) or 'none'}), "
            f"measurement_valid={arm['measurement_valid']}"
        )
    shared = _shared_categories(arms)
    if not shared:
        lines.append("\nNo shared category: these arms are not comparable.")
        return "\n".join(lines)

    lines.append(
        f"\n# Controlled comparison on the shared categories: {', '.join(shared)}\n"
    )
    header = ["metric", "category", *[f"layers {arm['label']}" for arm in arms]]
    if len(arms) == 2:
        header.append("ratio")
    rows: list[list[str]] = []
    for key, title in CATEGORY_METRICS:
        for category in shared:
            values = [arm["category"][category].get(key) for arm in arms]
            row = [title, category, *[_fmt(value) for value in values]]
            if len(arms) == 2:
                first, second = values
                row.append(
                    f"{second / first:.1f}x"
                    if isinstance(first, float) and first
                    else "-"
                )
            rows.append(row)
    lines.append(_table(header, rows))

    lines.append("\n# What a KL change is made of\n")
    for arm in arms:
        merged = _merge_categories(arm, shared)
        flip = (
            f"{merged['flip_rows']}/{merged['rows']} rows"
            if merged["flip_recorded"]
            else "not recorded by this arm"
        )
        lines.append(
            f"- layers {arm['label']}: max |dlogit| {_fmt(merged['max_abs'])}, "
            f"flip-eligible {flip}, top-1 misses {merged['misses']}"
        )
    lines.append(
        "\nA larger max |dlogit| means the route perturbs each row more. A larger\n"
        "flip-eligible share at a similar |dlogit| means the same perturbation is\n"
        "landing on more near-tie rows, which is a prompt-mix effect rather than\n"
        "an arithmetic one."
    )

    prefill = [arm for arm in arms if "prefill_last" in arm["shape"]]
    if prefill:
        lines.append("\n# Prefill row versus decode rows\n")
        for arm in prefill:
            shapes = arm["shape"]
            last = shapes["prefill_last"]
            rest = shapes.get("c1", {})
            lines.append(
                f"- layers {arm['label']}: prefill_last margin_min "
                f"{_fmt(last.get('strict_margin_min'))} with max |dlogit| "
                f"{_fmt(last.get('max_abs_logit_delta'))}; c1 margin_min "
                f"{_fmt(rest.get('strict_margin_min'))} with max |dlogit| "
                f"{_fmt(rest.get('max_abs_logit_delta'))}"
            )
        lines.append(
            "\nA near-zero prefill KL beside a non-zero prefill |dlogit| is a "
            "confidence\nartifact, not evidence that prefill skipped the "
            "candidate arithmetic: a\nbypassed route returns a zero delta."
        )
    return "\n".join(lines)


def _merge_categories(arm: dict[str, Any], shared: Sequence[str]) -> dict[str, Any]:
    rows = sum(arm["category"][name].get("rows", 0) for name in shared)
    misses = sum(
        arm["category"][name].get("rows", 0)
        - arm["category"][name].get("top1_matches", 0)
        for name in shared
    )
    max_abs = max(
        arm["category"][name].get("max_abs_logit_delta", 0.0) for name in shared
    )
    flip_rows = sum(
        arm["category"][name].get("flip_eligible_rows", 0) for name in shared
    )
    has_flip = all("flip_eligible_rows" in arm["category"][name] for name in shared)
    return {
        "rows": rows,
        "misses": misses,
        "max_abs": max_abs,
        "flip_rows": flip_rows,
        "flip_recorded": has_flip,
    }


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(header[index])), *(len(str(row[index])) for row in rows))
        for index in range(len(header))
    ]
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(
            str(cell).ljust(widths[index]) for index, cell in enumerate(cells)
        ) + " |"
    divider = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(header), divider, *(line(row) for row in rows)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    arms = [_load(path) for path in args.artifacts]
    arms.sort(key=lambda arm: len(arm["layers"]))
    print(render(arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
