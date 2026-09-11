#!/usr/bin/env python3
"""Paired UD/plain GGUF MTP measurement under one enforced protocol.

Task #22 needs a UD-vs-plain comparison in which every dimension except the
artifact is identical: host/GPU, model protocol, prompt suite, output horizon,
candidate budget, KV layout, AR baseline definition, and the correctness and
lifecycle gates. This driver makes that identity structural rather than
hopeful:

* one ``PAIRED_PROTOCOL`` mapping supplies every flag, so no pair can carry a
  per-artifact override;
* every pair runs the same ``qwen36_dense_gguf_suite.py`` entry point, so the
  AR baseline is the suite's own true no-MTP greedy path for each artifact and
  the KV layout comes from the same resident-session construction;
* the run refuses to start unless the U6 certification unit is complete for
  the UD artifacts, which is the "only after U6 certification passes" gate.

The gate is deliberately not bypassable by a flag. While U6 is incomplete the
useful action is ``--dry-run``, which resolves and prints the exact commands
and the certification state without touching the GPU.

Usage:
    python3 scripts/ud_mtp_paired.py --dry-run
    python3 scripts/ud_mtp_paired.py --runs 2 --output benchmarks/results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Every dimension the task names, in one place.  The pairs below differ only in
# (model, quant); any other difference would have to be added here and would
# therefore apply to every pair.
PAIRED_PROTOCOL = {
    "suite": "scripts/qwen36_dense_gguf_suite.py",
    "prompts": "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
    "max_new_tokens": 25,
    "candidate_budgets": (3,),
    "runs": 2,
    "target_verify_mode": "native",
    "draft_hidden_variant": "pre_output_norm",
    "warmup": True,
    "kv_layout": "resident session default (Qwen35GGUFResidentSession, wmma prefill, gemv decode)",
    "ar_baseline": "suite true no-MTP single-row greedy per artifact",
    "correctness_gates": (
        "status_complete_exact",
        "all_exact_greedy",
        "all_gpu_accept_match_cpu",
        "true_ar_denominator_present",
        "faster_than_true_ar",
    ),
}

# label -> (gguf path, session quant identity, family)
PAIRS = {
    "ud-q4-k-m": (
        Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"),
        "gguf_ud_q4_k_m",
        "ud",
    ),
    "ud-q4-k-s": (
        Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf"),
        "gguf_ud_q4_k_s",
        "ud",
    ),
    "plain-q4-k-m": (
        Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"),
        "gguf_q4_k_m",
        "plain",
    ),
    "plain-q4-k-s": (
        Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
        "gguf_q4_k_s",
        "plain",
    ),
}

# UD artifacts whose certification unit gates the run.  The plain controls are
# not UD presets and are not gated here.
UD_GATED_LABELS = tuple(label for label, (_, _, family) in PAIRS.items() if family == "ud")


def u6_gate_state() -> dict[str, object]:
    """Return the U6 certification state for the gated UD artifacts."""

    from hipengine.loading.qwen35_gguf_admission import (
        _UD_MTP_CERTIFICATIONS,
        _UD_MTP_PRESET_FINGERPRINTS,
    )

    by_preset = {cert.preset_key: cert for cert in _UD_MTP_CERTIFICATIONS.values()}
    state: dict[str, object] = {"gated": {}, "pin_fingerprints": sorted(_UD_MTP_PRESET_FINGERPRINTS)}
    blocked_any = False
    for label in UD_GATED_LABELS:
        _, quant, _ = PAIRS[label]
        cert = by_preset.get(quant)
        if cert is None:
            state["gated"][label] = {"preset_key": quant, "certified": False, "reason": "no U6 record"}
            blocked_any = True
            continue
        blocked = list(cert.blocked_items)
        blocked_any = blocked_any or bool(blocked)
        state["gated"][label] = {
            "preset_key": quant,
            "certified": cert.is_complete(),
            "blocked_items": blocked,
            "blockers": {item.item: item.blocker for item in cert.items if item.blocker},
        }
    state["gate_passed"] = not blocked_any
    return state


def resolve_commands(*, runs: int, raw_dir: Path, limit: int | None) -> list[dict[str, object]]:
    """Resolve the exact suite command for every pair under PAIRED_PROTOCOL."""

    protocol = dict(PAIRED_PROTOCOL)
    protocol["runs"] = int(runs)
    commands: list[dict[str, object]] = []
    for label, (path, quant, family) in PAIRS.items():
        argv = [
            protocol["suite"],
            "--model",
            str(path),
            "--quant",
            quant,
            "--prompts",
            protocol["prompts"],
            "--candidate-budgets",
            ",".join(str(budget) for budget in protocol["candidate_budgets"]),
            "--runs",
            str(protocol["runs"]),
            "--max-new-tokens",
            str(protocol["max_new_tokens"]),
            "--target-verify-mode",
            protocol["target_verify_mode"],
            "--draft-hidden-variant",
            protocol["draft_hidden_variant"],
            "--output",
            str(raw_dir / f"paired-{label}.json"),
        ]
        if not protocol["warmup"]:
            argv.append("--no-warmup")
        if limit is not None:
            argv.extend(["--limit", str(limit)])
        commands.append(
            {
                "label": label,
                "family": family,
                "model": str(path),
                "quant": quant,
                "argv": argv,
            }
        )
    return commands


def _grant_mtp_scope_in_process(preset_keys: set[str]) -> None:
    """Candidate-mode scope grant: in this process only, never a written pin."""

    from hipengine.loading import qwen35_gguf_admission as admission

    original = admission.resolve_qwen35_gguf_artifact_preset

    def patched(*args, **kwargs):
        preset = original(*args, **kwargs)
        if preset is not None and preset.preset_key in preset_keys:
            return admission.replace(
                preset,
                scopes=(*preset.scopes, admission.GGUF_PRESET_SCOPE_MTP),
                note=f"{preset.note} [paired run: MTP scope granted in-process]".strip(),
            )
        return preset

    admission.resolve_qwen35_gguf_artifact_preset = patched


def _run_pair(argv: list[str], output: Path) -> dict[str, object]:
    from scripts import qwen36_dense_gguf_suite as suite

    original_argv = sys.argv
    try:
        sys.argv = list(argv)
        exit_code = suite.main()
    finally:
        sys.argv = original_argv
    if not output.exists():
        raise SystemExit(f"paired run produced no payload (suite exit {exit_code}): {output}")
    return json.loads(output.read_text())


def _verdict(payload: dict, runs: int) -> dict[str, object]:
    correctness = payload.get("correctness", {})
    summary = payload.get("summary", {})
    true_ar = summary.get("true_ar", {}).get("full", {})
    budgets = summary.get("mtp", {})
    best_budget, best_ratio = None, 0.0
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
        "runs": runs,
        "gates": gates,
        "gates_passed": all(gates.values()),
        "true_ar_tok_s": true_ar.get("decode_tok_s_weighted"),
        "best_candidate_budget": best_budget,
        "best_mtp_tok_s": (
            None if best_budget is None else budgets[best_budget].get("full", {}).get("decode_tok_s_weighted")
        ),
        "best_mtp_vs_true_ar": best_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=int(PAIRED_PROTOCOL["runs"]))
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--limit", type=int, default=None, help="prompt limit (diagnostic runs only)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved commands and the U6 gate state; run nothing",
    )
    args = parser.parse_args()

    gate = u6_gate_state()
    commands = resolve_commands(runs=args.runs, raw_dir=args.raw_dir, limit=args.limit)

    if args.dry_run:
        print(json.dumps({"protocol": PAIRED_PROTOCOL, "u6_gate": gate, "commands": commands}, indent=2))
        return 0 if gate["gate_passed"] else 1

    if not gate["gate_passed"]:
        print(
            "refusing to run: the U6 certification unit is incomplete for the gated UD "
            "artifacts, and this measurement is only valid after U6 passes.",
            file=sys.stderr,
        )
        for label, entry in gate["gated"].items():
            if entry.get("certified"):
                continue
            print(f"  {label} ({entry['preset_key']}): blocked={entry.get('blocked_items')}", file=sys.stderr)
            for item, blocker in (entry.get("blockers") or {}).items():
                print(f"      {item}: {blocker}", file=sys.stderr)
        print("use --dry-run to inspect the resolved protocol and commands.", file=sys.stderr)
        return 1

    if args.output is None:
        parser.error("--output is required unless --dry-run is given")

    _grant_mtp_scope_in_process({quant for _, quant, family in PAIRS.values() if family == "ud"})
    report: dict[str, object] = {
        "unit": "paired-ud-plain-mtp",
        "protocol": PAIRED_PROTOCOL,
        "u6_gate": gate,
        "host_note": (
            "single host/GPU by construction; no concurrent GPU work is enforced by "
            "the operator, not by this script"
        ),
        "pairs": [],
    }
    for command in commands:
        label = str(command["label"])
        output = Path(str(command["argv"][command["argv"].index("--output") + 1]))
        print(f"[paired] {label}: {command['model']} (quant={command['quant']}) -> {output}", flush=True)
        payload = _run_pair([str(part) for part in command["argv"]], output)
        report["pairs"].append(
            {
                "label": label,
                "family": command["family"],
                "model": command["model"],
                "quant": command["quant"],
                "command": " ".join(str(part) for part in command["argv"]),
                "evidence": _verdict(payload, args.runs),
            }
        )

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[paired] wrote {args.output}")
    failed = False
    for entry in report["pairs"]:
        evidence = entry["evidence"]
        ar_rate = evidence["true_ar_tok_s"] or 0.0
        mtp_rate = evidence["best_mtp_tok_s"] or 0.0
        print(
            f"[paired] {entry['label']}: gates_passed={evidence['gates_passed']} "
            f"AR={ar_rate:.3f} MTP={mtp_rate:.3f} ratio={evidence['best_mtp_vs_true_ar']:.4f}"
        )
        for gate_name, ok in evidence["gates"].items():
            if not ok:
                failed = True
                print(f"[paired]   GATE FAILED: {gate_name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
