"""Kernels and the four-axis registry.

`AGENTS.md` keys kernels by `(backend, layer, quant, variant)` and requires a
registered strict fallback for every fused composite. This extractor pairs each
kernel source file with the registry keys, Python launchers, and tests that
mention it, so a file nothing dispatches to becomes visible.

It does not resolve dispatch. `fusion.plan()` decides what actually runs; a row
here says "nothing names this file", which is a reason to look, not a verdict.
"""

from __future__ import annotations

import re

from ..core import REPO_ROOT, Row
from . import corpus, register

KERNEL_ROOT = REPO_ROOT / "hipengine" / "kernels"
KERNEL_KEY = re.compile(r"""KernelKey\(\s*["']([^"']+)["']\s*,\s*["']([^"']+)["']""")
GLOBAL_FN = re.compile(r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)")


@register("kernels")
def extract() -> tuple[list[Row], dict]:
    code = corpus()
    rows: list[Row] = []

    keys: dict[tuple[str, str], list[str]] = {}
    for path, text in code.items():
        for backend, layer in KERNEL_KEY.findall(text):
            keys.setdefault((backend, layer), []).append(path)

    sources = sorted(
        p for p in KERNEL_ROOT.rglob("*")
        if p.suffix in (".hip", ".py") and "__pycache__" not in p.parts and p.name != "__init__.py"
    ) if KERNEL_ROOT.exists() else []

    for path in sources:
        rel = path.relative_to(REPO_ROOT).as_posix()
        stem = path.stem
        text = code.get(rel, "")
        backend = next((part for part in path.parts if part.startswith(("hip_", "cuda_", "cpu_"))), "unknown")
        signals: list[str] = []

        referrers = [
            other for other, body in code.items()
            if other != rel and (stem in body or path.name in body)
        ]
        runtime_referrers = [r for r in referrers if r.startswith("hipengine/") and "/kernels/" not in r]
        test_referrers = [r for r in referrers if r.startswith("tests/")]

        if not referrers:
            signals.append("no other file in the tree names this module")
        if not runtime_referrers:
            signals.append("nothing outside hipengine/kernels/ names it — no runtime caller found")
        if not test_referrers:
            signals.append("no test file names it")
        if path.suffix == ".hip":
            entries = GLOBAL_FN.findall(text)
            if not entries:
                signals.append("no __global__ entry point found")
            launched = [e for e in entries if any(e in body for other, body in code.items() if other != rel)]
            if entries and not launched:
                signals.append(f"{len(entries)} __global__ entry point(s), none named elsewhere")
        if "KernelKey" not in text and not any("KernelKey" in code.get(r, "") for r in referrers):
            signals.append("neither it nor its referrers mention KernelKey")

        rows.append(Row(
            kind="kernel",
            key=rel[len("hipengine/kernels/"):] if rel.startswith("hipengine/kernels/") else rel,
            title=rel,
            location=rel,
            evidence={
                "backend": backend,
                "suffix": path.suffix,
                "lines": text.count("\n") + 1 if text else 0,
                "referrers": len(referrers),
                "runtime_referrers": runtime_referrers[:6],
                "test_referrers": test_referrers[:4],
            },
            signals=signals,
        ))

    return rows, {
        "sources": len(rows),
        "registry_keys": len(keys),
        "backends": sorted({b for b, _ in keys}),
    }
