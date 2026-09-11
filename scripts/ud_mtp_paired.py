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
* the run refuses to start unless every U6 *pre-measurement* item is
  qualified for the UD artifacts: the structural and control items that can be
  verified without the run and whose violation would make its numbers
  meaningless.  The two items the run itself establishes (control/determinism
  evidence and the backend/profile/context/width envelope) cannot gate the run
  without circularity, and the admission pin still requires both.

While the pre-measurement items are open the useful action is ``--dry-run``,
which resolves and prints the exact commands and the certification state
without touching the GPU.

Usage:
    python3 scripts/ud_mtp_paired.py --dry-run
    python3 scripts/ud_mtp_paired.py --runs 2 --output benchmarks/results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
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
    "correctness_gates": {
        "binding": (
            "true_ar_denominator_present",
            "all_gpu_accept_match_cpu",
            "deterministic_repeats",
            "faster_than_true_ar",
        ),
        "recorded_not_binding": (
            "status_complete_exact",
            "all_exact_greedy",
        ),
        "note": (
            "docs/EXECUTION-PROFILES.md section 6: free-running generated-ID equality "
            "is recorded but is not the denominator, and section 4.1 permits logits "
            "and generated IDs to differ at near ties under the production profile. "
            "The control-plane gates are binding in every profile."
        ),
    },
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
            state["gated"][label] = {
                "preset_key": quant,
                "measurement_ready": False,
                "pin_complete": False,
                "reason": "no U6 record",
            }
            blocked_any = True
            continue
        blocked = [item.item for item in cert.items if not item.qualified]
        pre_blocked = list(cert.measurement_blockers)
        blocked_any = blocked_any or bool(pre_blocked)
        state["gated"][label] = {
            "preset_key": quant,
            "measurement_ready": cert.measurement_ready(),
            "pin_complete": cert.is_complete(),
            "measurement_blockers": pre_blocked,
            "open_items": blocked,
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


def _grant_mtp_scope_in_process(preset_keys: set[str]) -> object:
    """Candidate-mode scope grant: in this process only, never a written pin.

    Returns the function it replaced so the caller can restore it; use
    :func:`mtp_scope_granted` rather than calling this directly.
    """

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
    return original


@contextmanager
def mtp_scope_granted(preset_keys: set[str]):
    """Scope-grant context manager, so the patch never outlives the run.

    ``main()`` is called in-process by the tests; leaving the admission module
    patched would silently change every later admission assertion in the same
    interpreter.
    """

    from hipengine.loading import qwen35_gguf_admission as admission

    original = _grant_mtp_scope_in_process(preset_keys)
    try:
        yield
    finally:
        admission.resolve_qwen35_gguf_artifact_preset = original


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


def _provenance_summary(payload: dict) -> dict[str, object]:
    """Host identity and claim eligibility, so the artifact is self-contained."""

    provenance = payload.get("provenance") or {}
    model = payload.get("model") or {}
    workload = payload.get("workload") or {}
    return {
        "host_name": provenance.get("host_name"),
        "device_name": provenance.get("device_name"),
        "resolved_backend": provenance.get("resolved_backend"),
        "target_arch": provenance.get("target_arch"),
        "hipengine_commit": provenance.get("hipengine_commit"),
        "git_branch": provenance.get("git_branch"),
        "dirty": provenance.get("dirty"),
        "untracked_count": provenance.get("untracked_count"),
        "speed_claim_eligible": payload.get("speed_claim_eligible"),
        "suite_status": payload.get("status"),
        "model_path": model.get("path"),
        "model_size_bytes": model.get("size_bytes"),
        "model_file_type": model.get("file_type"),
        "prompt_file": workload.get("prompt_file"),
        "prompt_file_sha256": workload.get("prompt_file_sha256"),
    }


def expected_prompt_ids(prompts_path: Path) -> tuple[str, ...]:
    """The prompt ids the suite will run, in file order."""

    ids: list[str] = []
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(str(json.loads(line)["id"]))
    return tuple(ids)


def _evidence_completeness(
    payload: dict,
    *,
    command: dict[str, object],
    runs: int,
    prompt_ids: tuple[str, ...],
) -> dict[str, object]:
    """Whether the payload is the complete evidence the resolved command asked for.

    Without this, a truncated payload passes every downstream gate trivially: a
    single-run file gives each prompt one token hash, so ``_determinism`` finds
    no disagreement and reports deterministic repeats.
    """

    problems: list[str] = []
    if payload.get("schema") != 1:
        problems.append(f"schema {payload.get('schema')!r} != 1")
    if payload.get("kind") != "qwen36_dense_gguf_ar_mtp_suite":
        problems.append(f"kind {payload.get('kind')!r}")

    model = payload.get("model") or {}
    if str(model.get("path")) != str(command["model"]):
        problems.append(f"model.path {model.get('path')!r} != {command['model']!r}")

    workload = payload.get("workload") or {}
    if Path(str(workload.get("prompt_file"))).resolve() != Path(
        str(PAIRED_PROTOCOL["prompts"])
    ).resolve():
        problems.append(f"workload.prompt_file {workload.get('prompt_file')!r}")
    if tuple(str(p) for p in (workload.get("prompt_ids") or ())) != prompt_ids:
        problems.append("workload.prompt_ids does not match the prompt file")
    if int(workload.get("prompt_count") or 0) != len(prompt_ids):
        problems.append(f"workload.prompt_count {workload.get('prompt_count')!r}")
    if int(workload.get("max_new_tokens_visible") or 0) != int(PAIRED_PROTOCOL["max_new_tokens"]):
        problems.append(f"workload.max_new_tokens_visible {workload.get('max_new_tokens_visible')!r}")
    if tuple(int(b) for b in (workload.get("candidate_budgets") or ())) != tuple(
        int(b) for b in PAIRED_PROTOCOL["candidate_budgets"]
    ):
        problems.append(f"workload.candidate_budgets {workload.get('candidate_budgets')!r}")

    provenance = payload.get("provenance") or {}
    for field in ("host_name", "device_name", "resolved_backend", "target_arch", "hipengine_commit"):
        if not provenance.get(field):
            problems.append(f"provenance.{field} missing")

    rows = payload.get("rows") or {}
    groups: dict[str, list[dict]] = {"true_ar": list(rows.get("true_ar") or [])}
    for budget, budget_rows in (rows.get("mtp") or {}).items():
        groups[f"mtp_B{budget}"] = list(budget_rows)
    if "true_ar" not in rows:
        problems.append("rows.true_ar missing")
    expected_budgets = {str(int(b)) for b in PAIRED_PROTOCOL["candidate_budgets"]}
    if {str(b) for b in (rows.get("mtp") or {})} != expected_budgets:
        problems.append(f"rows.mtp budgets {sorted(rows.get('mtp') or {})} != {sorted(expected_budgets)}")

    for label, group in groups.items():
        expected_rows = len(prompt_ids) * int(runs)
        if len(group) != expected_rows:
            problems.append(f"{label}: {len(group)} rows != {expected_rows} expected")
            continue
        seen: dict[str, set[int]] = {}
        duplicates: set[tuple[str, int]] = set()
        for row in group:
            prompt = str(row.get("id"))
            run = int(row.get("run", -1))
            if run in seen.setdefault(prompt, set()):
                duplicates.add((prompt, run))
            seen[prompt].add(run)
            if not row.get("token_sha256_i64"):
                problems.append(f"{label}: {prompt} run {run} has no token_sha256_i64")
                break
        if duplicates:
            problems.append(f"{label}: duplicate (prompt, run) {sorted(duplicates)[:4]}")
        missing = [p for p in prompt_ids if p not in seen]
        if missing:
            problems.append(f"{label}: missing prompts {missing[:4]}")
        for prompt, seen_runs in seen.items():
            if seen_runs != set(range(int(runs))):
                problems.append(f"{label}: {prompt} runs {sorted(seen_runs)} != 0..{int(runs) - 1}")
                break
    return {"complete": not problems, "problems": problems, "expected_rows_per_group": len(prompt_ids) * int(runs)}


def _determinism(payload: dict, runs: int) -> dict[str, object]:
    """Whether every prompt repeats bit-identically across runs, AR and MTP."""

    rows = payload.get("rows", {})
    groups: dict[str, list[dict]] = {"true_ar": list(rows.get("true_ar", []))}
    for budget, budget_rows in (rows.get("mtp") or {}).items():
        groups[f"mtp_B{budget}"] = list(budget_rows)
    unstable: dict[str, list[str]] = {}
    for label, group in groups.items():
        by_prompt: dict[str, set[str]] = {}
        for row in group:
            by_prompt.setdefault(str(row["id"]), set()).add(str(row["token_sha256_i64"]))
        differing = sorted(prompt for prompt, hashes in by_prompt.items() if len(hashes) > 1)
        if differing:
            unstable[label] = differing
    return {
        "runs": int(runs),
        "deterministic": not unstable,
        "unstable": unstable,
    }


def _exactness(payload: dict) -> dict[str, object]:
    """Free-running ID agreement, recorded but not binding (docs section 6)."""

    rows = payload.get("rows", {})
    divergent: list[str] = []
    total = 0
    for group in (rows.get("mtp") or {}).values():
        for row in group:
            total += 1
            if not bool(row.get("exact_greedy_match")):
                divergent.append(f"{row['id']}#run{row.get('run')}")
    return {
        "rows": total,
        "divergent_rows": sorted(divergent),
        "divergent_count": len(divergent),
        "binding": False,
        "note": (
            "docs/EXECUTION-PROFILES.md section 6: free-running generated-ID equality "
            "is recorded but is not the denominator; section 4.1 permits logits and "
            "generated IDs to differ at near ties in the production profile."
        ),
    }


def _verdict(
    payload: dict,
    runs: int,
    *,
    command: dict[str, object],
    prompt_ids: tuple[str, ...],
) -> dict[str, object]:
    """Binding control gates plus the recorded-not-binding exactness evidence."""

    summary = payload.get("summary", {})
    true_ar = summary.get("true_ar", {}).get("full", {})
    budgets = summary.get("mtp", {})
    best_budget, best_ratio = None, 0.0
    for budget, block in budgets.items():
        ratio = float(block.get("full", {}).get("mtp_vs_true_ar", 0.0) or 0.0)
        if ratio > best_ratio:
            best_budget, best_ratio = budget, ratio
    completeness = _evidence_completeness(
        payload, command=command, runs=runs, prompt_ids=prompt_ids
    )
    determinism = _determinism(payload, runs)
    correctness = payload.get("correctness", {})
    binding_gates = {
        "evidence_complete": bool(completeness["complete"]),
        "true_ar_denominator_present": bool(true_ar.get("decode_tok_s_weighted")),
        "all_gpu_accept_match_cpu": bool(correctness.get("all_gpu_accept_match_cpu")),
        "deterministic_repeats": bool(completeness["complete"])
        and bool(determinism["deterministic"]),
        "faster_than_true_ar": best_ratio > 1.0,
    }
    return {
        "runs": runs,
        "binding_gates": binding_gates,
        "binding_passed": all(binding_gates.values()),
        "evidence_completeness": completeness,
        "recorded": {
            "suite_status": payload.get("status"),
            "all_exact_greedy": bool(correctness.get("all_exact_greedy")),
            "exactness": _exactness(payload),
        },
        "determinism": determinism,
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
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "re-derive the artifact from raw payloads already present in --raw-dir "
            "instead of running the GPU commands; the raw payloads are the evidence "
            "and are not modified"
        ),
    )
    args = parser.parse_args()

    gate = u6_gate_state()
    commands = resolve_commands(runs=args.runs, raw_dir=args.raw_dir, limit=args.limit)

    if args.dry_run:
        print(json.dumps({"protocol": PAIRED_PROTOCOL, "u6_gate": gate, "commands": commands}, indent=2))
        return 0 if gate["gate_passed"] else 1

    if not gate["gate_passed"]:
        print(
            "refusing to run: a U6 pre-measurement item is unqualified for a gated UD "
            "artifact. Those items are the ones that can be verified without the run, "
            "and a violation would make the measured numbers meaningless.",
            file=sys.stderr,
        )
        for label, entry in gate["gated"].items():
            if entry.get("measurement_ready"):
                continue
            print(
                f"  {label} ({entry['preset_key']}): blocked={entry.get('measurement_blockers')}",
                file=sys.stderr,
            )
            for item in entry.get("measurement_blockers") or ():
                print(f"      {item}: {(entry.get('blockers') or {}).get(item)}", file=sys.stderr)
        print("use --dry-run to inspect the resolved protocol and commands.", file=sys.stderr)
        return 1

    if args.output is None:
        parser.error("--output is required unless --dry-run is given")

    prompt_ids = expected_prompt_ids(REPO_ROOT / str(PAIRED_PROTOCOL["prompts"]))
    report: dict[str, object] = {
        "unit": "paired-ud-plain-mtp",
        "protocol": PAIRED_PROTOCOL,
        "u6_gate": gate,
        "candidate_mode": (
            "the U6 admission pin is empty (the two paired_run items are still open), so "
            "MTP scope is granted in-process and this artifact is diagnostic evidence "
            "for those items, not a retained admission."
        ),
        "host_note": (
            "single host/GPU by construction; no concurrent GPU work is enforced by "
            "the operator, not by this script"
        ),
        "pairs": [],
    }
    with mtp_scope_granted({quant for _, quant, family in PAIRS.values() if family == "ud"}):
        for command in commands:
            label = str(command["label"])
            output = Path(str(command["argv"][command["argv"].index("--output") + 1]))
            if args.from_raw:
                if not output.exists():
                    print(f"[paired] {label}: missing raw payload {output}", file=sys.stderr)
                    return 1
                print(f"[paired] {label}: re-deriving from {output}", flush=True)
                payload = json.loads(output.read_text())
            else:
                print(
                    f"[paired] {label}: {command['model']} (quant={command['quant']}) -> {output}",
                    flush=True,
                )
                payload = _run_pair([str(part) for part in command["argv"]], output)
            evidence = _verdict(
                payload, args.runs, command=command, prompt_ids=prompt_ids
            )
            if not evidence["evidence_completeness"]["complete"]:
                print(
                    f"[paired] {label}: incomplete evidence for the resolved command:",
                    file=sys.stderr,
                )
                for problem in evidence["evidence_completeness"]["problems"][:12]:
                    print(f"[paired]   {problem}", file=sys.stderr)
            report["pairs"].append(
                {
                    "label": label,
                    "family": command["family"],
                    "model": command["model"],
                    "quant": command["quant"],
                    "command": " ".join(str(part) for part in command["argv"]),
                    "raw_payload": str(output),
                    "provenance": _provenance_summary(payload),
                    "evidence": evidence,
                }
            )

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[paired] wrote {args.output}")
    failed = False
    for entry in report["pairs"]:
        evidence = entry["evidence"]
        ar_rate = evidence["true_ar_tok_s"] or 0.0
        mtp_rate = evidence["best_mtp_tok_s"] or 0.0
        recorded = evidence["recorded"]
        exact = recorded["exactness"]
        print(
            f"[paired] {entry['label']}: binding_passed={evidence['binding_passed']} "
            f"AR={ar_rate:.3f} MTP={mtp_rate:.3f} ratio={evidence['best_mtp_vs_true_ar']:.4f}"
        )
        print(
            f"[paired]   recorded (not binding): suite_status={recorded['suite_status']} "
            f"all_exact_greedy={recorded['all_exact_greedy']} "
            f"divergent_rows={exact['divergent_count']}/{exact['rows']} "
            f"{exact['divergent_rows']}"
        )
        for gate_name, ok in evidence["binding_gates"].items():
            if not ok:
                failed = True
                print(f"[paired]   BINDING GATE FAILED: {gate_name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
