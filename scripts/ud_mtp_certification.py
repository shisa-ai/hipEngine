#!/usr/bin/env python3
"""U6: evaluate whether a UD artifact may be granted the MTP scope.

UD AR admission and UD MTP admission are separate certification units.  AR
certificates are pinned with scope ``("ar",)``; MTP admission is granted only by
an entry in ``_UD_MTP_PRESET_FINGERPRINTS``.  This script runs the qualification
that an entry must be based on, WITHOUT writing that entry: it grants the scope
in-process ("candidate mode") so the run is reproducible, and reports whether
the candidate meets the gates.

It refuses to run unless the enabling conditions hold, so a "pass" cannot come
from relaxed validation:

* the quant-agnostic speculative accept chain resolves for the UD session
  identity;
* the artifact's pinned NextN draft dtype manifest resolves through the
  admitted preset identity and the strict draft build passes with it.

The run itself uses the repository's dense-GGUF suite against the full
mtp-bench category suite (code / general_en / general_ja / mixed_ja_en) with a
true no-MTP autoregressive denominator measured under the same protocol.

This is necessary but not sufficient for the pin: the complete U6 scope also
covers the declared backend/profile/context/width envelope and the c>1 serving
paths.  Read the verdict, not just the exit status.

Usage:
    python3 scripts/ud_mtp_certification.py --artifact ud-q4-k-m --output /tmp/u6-km.json
    python3 scripts/ud_mtp_certification.py --artifact both --runs 2 --output /tmp/u6.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# artifact name -> (GGUF path, session quant identity)
ARTIFACTS = {
    "ud-q4-k-m": (
        Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"),
        "gguf_ud_q4_k_m",
    ),
    "ud-q4-k-s": (
        Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf"),
        "gguf_ud_q4_k_s",
    ),
}


def _preflight(path: Path, quant: str) -> dict[str, object]:
    """Verify the enabling conditions; raise if any is missing."""

    # Registration path for the speculative kernels: the MTP generation module
    # imports the speculative package (which registers the accept chain at
    # import time), so preflight pulls in the same package rather than reaching
    # for the registrar directly.
    import hipengine.kernels.hip_gfx1100.speculative  # noqa: F401

    from hipengine.kernels.registry import KernelKey, registered_keys
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_admission import (
        resolve_qwen35_gguf_artifact_preset,
        resolve_qwen35_gguf_nextn_draft_qtypes,
    )
    from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map

    accept_key = KernelKey("hip_gfx1100", "dflash_accept_chain", quant, "i32")
    if accept_key not in registered_keys():
        raise SystemExit(
            f"U6 preflight failed: accept-chain kernel not registered for {quant!r}; "
            "MTP verification cannot resolve its accept kernel for this session identity"
        )

    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    structural = build_qwen35_gguf_nextn_tensor_map(reader.info, strict=False)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=structural, file_type_stamp=reader.info.file_type_name
    )
    if preset is None:
        raise SystemExit(f"U6 preflight failed: {path} resolves no artifact preset")
    pin = resolve_qwen35_gguf_nextn_draft_qtypes(preset)
    if pin is None:
        raise SystemExit(
            f"U6 preflight failed: preset {preset.preset_key!r} carries no pinned "
            "NextN draft dtype manifest"
        )
    # Strict build with the pin: this is the same call draft materialization
    # makes, so a dtype mismatch here would fail the run anyway.
    build_qwen35_gguf_nextn_tensor_map(reader.info, pinned_qtypes=pin)
    return {
        "preset_key": preset.preset_key,
        "manifest_fingerprint": preset.manifest_fingerprint,
        "scopes": list(preset.scopes),
        "draft_dtypes": {slot: qtype.name for slot, qtype in pin.items()},
    }


def _grant_mtp_scope_in_process(preset_keys: set[str]) -> None:
    """Candidate mode: add the MTP scope for these presets, in this process only."""

    from hipengine.loading import qwen35_gguf_admission as admission

    original = admission.resolve_qwen35_gguf_artifact_preset

    def patched(*args, **kwargs):
        preset = original(*args, **kwargs)
        if preset is not None and preset.preset_key in preset_keys:
            return admission.replace(
                preset,
                scopes=(*preset.scopes, admission.GGUF_PRESET_SCOPE_MTP),
                note=f"{preset.note} [U6 candidate: MTP scope granted in-process]".strip(),
            )
        return preset

    admission.resolve_qwen35_gguf_artifact_preset = patched


def _category_rows(payload: dict) -> dict[str, dict[str, object]]:
    """Per-category exactness and acceptance for the MTP candidate."""

    import collections

    prompt_categories: dict[str, str] = {}
    prompts_path = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
    if prompts_path.exists():
        for line in prompts_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                prompt_categories[str(record.get("id"))] = str(record.get("category", "?"))

    rows: dict[str, dict[str, object]] = collections.defaultdict(
        lambda: {"runs": 0, "exact_runs": 0, "accept_match_runs": 0, "accepted": 0, "proposed": 0}
    )
    for budget_rows in payload.get("rows", {}).get("mtp", {}).values():
        for row in budget_rows:
            category = prompt_categories.get(str(row.get("id")), "?")
            bucket = rows[category]
            bucket["runs"] += 1
            bucket["exact_runs"] += 1 if row.get("exact_greedy_match") else 0
            bucket["accept_match_runs"] += 1 if row.get("gpu_accept_match_cpu") else 0
            bucket["accepted"] += int(row.get("accepted_draft_tokens", 0) or 0)
            bucket["proposed"] += int(row.get("proposed_draft_tokens", 0) or 0)
    for bucket in rows.values():
        proposed = bucket["proposed"]
        bucket["acceptance"] = (bucket["accepted"] / proposed) if proposed else 0.0
    return dict(rows)


def _verdict(payload: dict, runs: int) -> dict[str, object]:
    correctness = payload.get("correctness", {})
    summary = payload.get("summary", {})
    true_ar = summary.get("true_ar", {}).get("full", {})
    budgets = summary.get("mtp", {})
    best_budget = None
    best_ratio = 0.0
    for budget, block in budgets.items():
        ratio = float(block.get("full", {}).get("mtp_vs_true_ar", 0.0) or 0.0)
        if ratio > best_ratio:
            best_budget, best_ratio = budget, ratio
    gates = {
        "status_complete_exact": payload.get("status") == "complete_exact",
        "all_exact_greedy": bool(correctness.get("all_exact_greedy")),
        "all_gpu_accept_match_cpu": bool(correctness.get("all_gpu_accept_match_cpu")),
        "true_ar_denominator_present": bool(true_ar.get("decode_tok_s_weighted")),
        "faster_than_true_ar": best_ratio > 1.0,
    }
    return {
        "gates": gates,
        "gates_passed": all(gates.values()),
        "runs": runs,
        "true_ar_tok_s": true_ar.get("decode_tok_s_weighted"),
        "best_candidate_budget": best_budget,
        "best_mtp_tok_s": (
            None
            if best_budget is None
            else budgets[best_budget].get("full", {}).get("decode_tok_s_weighted")
        ),
        "best_mtp_vs_true_ar": best_ratio,
        "categories": _category_rows(payload),
    }


def _run_one(
    *,
    artifact: str,
    path: Path,
    quant: str,
    output: Path,
    runs: int,
    candidate_budgets: tuple[int, ...],
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, object]:
    preflight = _preflight(path, quant)
    _grant_mtp_scope_in_process({str(preflight["preset_key"])})

    from scripts import qwen36_dense_gguf_suite as suite

    argv = [
        "qwen36_dense_gguf_suite.py",
        "--model",
        str(path),
        "--quant",
        quant,
        "--candidate-budgets",
        ",".join(str(budget) for budget in candidate_budgets),
        "--runs",
        str(runs),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output",
        str(output),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    original_argv = sys.argv
    try:
        sys.argv = argv
        exit_code = suite.main()
    finally:
        sys.argv = original_argv
    # A non-zero exit means the suite's status was not ``complete_exact``.
    # That is a result about the candidate, not a harness failure, so record it
    # and let the gate verdict carry the outcome.
    if not output.exists():
        raise SystemExit(
            f"U6 candidate run for {artifact} produced no payload (suite exit {exit_code})"
        )

    payload = json.loads(output.read_text())
    return {
        "artifact": artifact,
        "model": str(path),
        "quant": quant,
        "command": " ".join(argv),
        "suite_exit_code": exit_code,
        "preflight": preflight,
        "evidence": _verdict(payload, runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        choices=(*ARTIFACTS, "both"),
        default="both",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--candidate-budgets", default="3")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="prompt limit passthrough (diagnostic runs); default is the full suite",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    budgets = tuple(int(part) for part in str(args.candidate_budgets).split(",") if part)
    names = list(ARTIFACTS) if args.artifact == "both" else [args.artifact]

    report: dict[str, object] = {
        "unit": "U6-ud-mtp-certification",
        "candidate_mode": (
            "MTP scope granted in-process; no admission pin was written. A pass "
            "is evidence for a pin, not a pin."
        ),
        "protocol": {
            "suite": "scripts/qwen36_dense_gguf_suite.py",
            "prompts": "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
            "candidate_budgets": list(budgets),
            "runs": args.runs,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
        },
        "artifacts": [],
    }
    for name in names:
        path, quant = ARTIFACTS[name]
        if not path.exists():
            raise SystemExit(f"pinned artifact missing: {path}")
        raw = args.raw_dir / f"u6-{name}-raw.json"
        print(f"[u6] {name}: {path} (quant={quant}) -> {raw}", flush=True)
        report["artifacts"].append(
            _run_one(
                artifact=name,
                path=path,
                quant=quant,
                output=raw,
                runs=args.runs,
                candidate_budgets=budgets,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
            )
        )

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[u6] wrote {args.output}")
    failed = False
    for entry in report["artifacts"]:
        evidence = entry["evidence"]
        ar_rate = evidence["true_ar_tok_s"]
        mtp_rate = evidence["best_mtp_tok_s"]
        print(
            f"[u6] {entry['artifact']}: gates_passed={evidence['gates_passed']} "
            f"AR={ar_rate:.3f} MTP={mtp_rate:.3f} "
            f"ratio={evidence['best_mtp_vs_true_ar']:.4f}"
        )
        for gate, ok in evidence["gates"].items():
            if not ok:
                failed = True
                print(f"[u6]   GATE FAILED: {gate}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
