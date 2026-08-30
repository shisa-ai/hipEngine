#!/usr/bin/env python3
"""Fail when a published benchmark artifact's provenance fields cannot be true.

Motivation is a measured failure, not a hypothesis. On 2026-08-30 an artifact was committed whose
`model` named a quant this run never used - and the invented quant was baked into its filename - whose
`hardware` claimed a VRAM figure the host does not have, whose `host` cited a kernel and RAM value
read from nothing, and whose `supersedes` named an artifact file that has never existed. The
measurements inside it were fine. No gate looked at any of those fields, and the evidence policy in
AGENTS.md is built entirely out of them: model + quant + workload + host + hardware + command.

This is parse-level and side-effect free - no import, no GPU, no model load. Scope follows the
command-drift gate: the artifacts published by `benchmarks/README.md`, because the full history is not
a fixable backlog and a published row is the one a reader repeats.

Fail-tier rules
---------------
DANGLING-SUPERSEDES   a `supersedes` / `superseded_by` / `links` value cites an artifact json that is
                      not in benchmarks/results/. The cheapest lie to catch, and the one that shipped.
QUANT-MODEL-CONFLICT  `model` names a quant-like token and the artifact declares `quant`, and the two
                      disagree. This is the shape of today's error: a Q4_K_S label on a Q4_K_M run.
                      Both the internal key form (`gguf_q4_k_m`) and the display form (`Q4_K_M`) are
                      canonicalised, so the rule reads disagreement rather than spelling.
BAD-JSON              the artifact does not parse.
MISSING-ARTIFACT      the README cites an artifact that is not in benchmarks/results/.

Warning tier (reported, never fatal - new hosts and new models are legitimate)
-----------------------------------------------------------------------------
UNIQUE-HARDWARE       this artifact's plain-string `hardware` appears in no other published artifact.
UNIQUE-HOST           the same for `host`. Structured (dict) values are skipped: the repr of a dict is
                      unique by construction, and that noise would hide the signal.

A filename-vs-model rule lived here first and was dropped for the same reason: `...q4km...` in a path
partial-matches "q4" and then reads as a conflict with `Q4_K_M`. Aliasing q4km / Q4_K_M / IQ4_XS by
regex is guesswork, so the gate does not guess; it checks only what an artifact states against what the
same artifact cites.

Usage:
    .venv/bin/python scripts/check_artifact_provenance.py [--repo .] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SUPERSESSION_KEYS = ("supersedes", "superseded_by", "links", "supersedes_detail")
JSON_TOKEN = re.compile(r"([A-Za-z0-9._-]+\.json)")
# Q4_K_M, Q4_K_S, Q8_0, IQ4_XS, UD-Q4_K_M, gguf_q4_k_m, Q6_K, BF16, FP16 ...
QUANT_SHAPE = re.compile(
    r"((?:gguf_|UD-|IQ)?Q\d(?:_[A-Za-z0-9]{1,3}){0,3}|BF16|FP16|FP8|F16)", re.IGNORECASE
)
READ_CITATION = re.compile(r"([A-Za-z0-9._-]+\.json)(?!l)")

# Provenance problems owned by another lane: key -> dated reason. A stale entry fails the gate, which
# is what prunes them.
EXCEPTIONS: dict[str, str] = {}


def _norm_quant(text: str) -> str:
    """Canonical quant: drop gguf_/UD-/IQ decoration and separators, upper-case.

    Declared quants arrive both as `Q4_K_M` and as the internal quant key `gguf_q4_k_m`. Those are the
    same fact, and reading them as a conflict would make the gate cry wolf on real published rows.
    """
    out = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    for prefix in ("GGUF", "UD", "IQ"):
        if out.startswith(prefix):
            return out[len(prefix):]
    return out


def quant_tokens(text: str) -> set[str]:
    return {m.group(1) for m in QUANT_SHAPE.finditer(text or "")}


def exception_key(artifact: str, problem: str, detail: str) -> str:
    return f"{artifact}::{problem}::{detail}"


def _strings(payload: Any, key: str | None = None):
    """Yield (dict_key, string) for every string anywhere in a JSON payload."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            yield from _strings(v, key=k)
    elif isinstance(payload, list):
        for v in payload:
            yield from _strings(v, key=key)
    elif isinstance(payload, str):
        yield (key or "", payload)


def _uniqueness_warnings(payloads: dict[str, Any]) -> list[dict[str, str]]:
    """Flag hardware/host strings that no other published artifact uses. Plain strings only."""
    fields = ("hardware", "host")
    counts: dict[str, dict[str, int]] = {field: {} for field in fields}
    for payload in payloads.values():
        for field in fields:
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                counts[field][value] = counts[field].get(value, 0) + 1
    warnings: list[dict[str, str]] = []
    for name, payload in payloads.items():
        for field in fields:
            value = payload.get(field)
            if isinstance(value, str) and value.strip() and counts[field][value] == 1:
                warnings.append(
                    {
                        "artifact": name,
                        "problem": f"UNIQUE-{field.upper()}",
                        "detail": value[:120],
                    }
                )
    return warnings


def check_repo(repo: Path, exceptions: dict[str, str] | None = None) -> dict[str, Any]:
    """Audit the provenance fields of every artifact published by benchmarks/README.md."""
    repo = Path(repo)
    allow = EXCEPTIONS if exceptions is None else exceptions
    readme = repo / "benchmarks" / "README.md"
    results = repo / "benchmarks" / "results"
    if not readme.is_file():
        raise FileNotFoundError(f"no benchmarks/README.md under {repo}")
    cited = sorted(set(READ_CITATION.findall(readme.read_text())))
    existing = {p.name for p in results.glob("*.json")} if results.is_dir() else set()

    payloads: dict[str, Any] = {}
    violations: list[dict[str, str]] = []
    for name in cited:
        path = results / name
        if not path.is_file():
            violations.append({"artifact": name, "problem": "MISSING-ARTIFACT", "detail": name})
            continue
        try:
            payloads[name] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(
                {"artifact": name, "problem": "BAD-JSON", "detail": str(exc)[:160]}
            )

    for name, payload in payloads.items():
        for key, text in _strings(payload):
            if key not in SUPERSESSION_KEYS:
                continue
            for target in JSON_TOKEN.findall(text):
                if target not in existing:
                    violations.append(
                        {
                            "artifact": name,
                            "problem": "DANGLING-SUPERSEDES",
                            "detail": f"{key} cites {target}",
                        }
                    )
        model = str(payload.get("model") or "")
        quant = str(payload.get("quant") or "")
        if model and quant:
            named = {_norm_quant(t) for t in quant_tokens(model)}
            if named and _norm_quant(quant) not in named:
                violations.append(
                    {
                        "artifact": name,
                        "problem": "QUANT-MODEL-CONFLICT",
                        "detail": f"quant={quant!r} but model names {sorted(quant_tokens(model))}",
                    }
                )

    matched: list[str] = []
    kept: list[dict[str, str]] = []
    for violation in violations:
        key = exception_key(violation["artifact"], violation["problem"], violation["detail"])
        if key in allow:
            matched.append(key)
        else:
            kept.append(violation)
    matched_set = set(matched)
    return {
        "schema": "artifact_provenance.v1",
        "repo": str(repo),
        "artifacts_cited": len(cited),
        "artifacts_checked": len(payloads),
        "violations": kept,
        "warnings": _uniqueness_warnings(payloads),
        "exceptions_matched": sorted(matched_set),
        "exceptions_unmatched": sorted(set(allow) - matched_set),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument("--json", default=None, type=Path)
    parser.add_argument("--show-warnings", action="store_true")
    args = parser.parse_args(argv)
    report = check_repo(args.repo)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    problems = report["violations"]
    print(
        f"artifact provenance: {report['artifacts_checked']}/{report['artifacts_cited']} cited "
        f"artifacts parsed, {len(problems)} violation(s), {len(report['warnings'])} warning(s), "
        f"{len(report['exceptions_matched'])} exception(s) matched"
    )
    for violation in problems[:40]:
        print(f"  FAIL {violation['artifact']}: {violation['problem']} - {violation['detail']}")
    if args.show_warnings:
        for warning in report["warnings"][:40]:
            print(f"  warn {warning['artifact']}: {warning['problem']} - {warning['detail']}")
    for stale in report["exceptions_unmatched"][:10]:
        print(f"  stale exception (prune it): {stale}")
    return 1 if problems or report["exceptions_unmatched"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
