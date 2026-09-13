#!/usr/bin/env python3
"""Compare the eager and graph true-AR decode paths on one artifact.

The true-AR performance denominator runs the production decode-graph replay
path. That path is fast because the host submits one graph instead of roughly
860 individual kernels per transition, but it is only a valid denominator if it
decodes the same tokens: a graph that produced a different sequence would be
measuring different work.

The suite's own ``exact_greedy_match`` compares the AR tokens against the MTP
verifier's, and both of those run through graphs, so it cannot see a fault
shared by the graph paths. This script is the independent check: it runs the
same model, prompt, and settings twice -- once with ``--ar-decode-mode eager``
(synchronous scalar ``step()`` submission) and once with ``graph`` -- and
compares the recorded token ids transition by transition.

This is an operator-invoked GPU check, not part of the default test run: it
loads the artifact twice. ``tests/test_qwen36_dense_gguf_suite.py`` pins the
wiring contract that both modes must satisfy; this pins the device behaviour.

Exit code is 0 when every transition matches and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ud_mtp_paired import device_selected, mtp_scope_granted  # noqa: E402

SUITE = "scripts/qwen36_dense_gguf_suite.py"
DEFAULT_PROMPTS = "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
AR_MODES = ("eager", "graph")

def ar_rows_by_key(payload: dict) -> dict[tuple[str, int], dict]:
    """Index a suite payload's true-AR rows by (prompt id, run index)."""

    rows = (payload.get("rows") or {}).get("true_ar") or []
    indexed: dict[tuple[str, int], dict] = {}
    for row in rows:
        indexed[(str(row["id"]), int(row.get("run", 0)))] = row
    return indexed

def compare_ar_token_sequences(eager: dict, graph: dict) -> dict:
    """Compare two suite payloads' true-AR token ids transition by transition.

    Returns a report naming every divergence, so a mismatch localizes to a
    prompt, a run, and a transition index rather than just failing.
    """

    eager_rows = ar_rows_by_key(eager)
    graph_rows = ar_rows_by_key(graph)
    problems: list[str] = []
    if not eager_rows or not graph_rows:
        problems.append(
            f"missing true-AR rows (eager {len(eager_rows)}, graph {len(graph_rows)})"
        )
    only_eager = sorted(set(eager_rows) - set(graph_rows))
    only_graph = sorted(set(graph_rows) - set(eager_rows))
    if only_eager:
        problems.append(f"rows only in eager: {only_eager[:4]}")
    if only_graph:
        problems.append(f"rows only in graph: {only_graph[:4]}")

    compared_transitions = 0
    divergences: list[dict] = []
    for key in sorted(set(eager_rows) & set(graph_rows)):
        eager_tokens = list(eager_rows[key].get("token_ids") or [])
        graph_tokens = list(graph_rows[key].get("token_ids") or [])
        if len(eager_tokens) != len(graph_tokens):
            problems.append(
                f"{key[0]}#run{key[1]}: length {len(graph_tokens)} != {len(eager_tokens)}"
            )
        for index, (left, right) in enumerate(zip(eager_tokens, graph_tokens)):
            compared_transitions += 1
            if int(left) != int(right):
                divergences.append(
                    {
                        "prompt": key[0],
                        "run": key[1],
                        "transition": index,
                        "eager_token": int(left),
                        "graph_token": int(right),
                    }
                )
    if divergences:
        first = divergences[0]
        problems.append(
            f"{len(divergences)} divergent transitions, first at "
            f"{first['prompt']}#run{first['run']} index {first['transition']}: "
            f"eager {first['eager_token']} vs graph {first['graph_token']}"
        )
    return {
        "matched": not problems,
        "problems": problems,
        "compared_rows": len(set(eager_rows) & set(graph_rows)),
        "compared_transitions": compared_transitions,
        "divergences": divergences[:32],
    }

def _suite_argv(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    """The suite's own argv, without an interpreter prefix.

    The suite runs in-process (see :func:`_run_suite`) so the MTP scope grant
    reaches it: the grant patches the admission module in this interpreter, and
    a subprocess would not inherit it. This mirrors ``ud_mtp_paired._run_pair``.
    """

    argv = [
        SUITE,
        "--model",
        str(args.model),
        "--quant",
        str(args.quant),
        "--prompts",
        str(args.prompts),
        "--candidate-budgets",
        "3",
        "--runs",
        "1",
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--target-verify-mode",
        "native",
        "--draft-hidden-variant",
        "pre_output_norm",
        "--ar-decode-mode",
        mode,
        "--output",
        str(output),
    ]
    if args.limit is not None:
        argv.extend(["--limit", str(int(args.limit))])
    if not args.warmup:
        argv.append("--no-warmup")
    return argv

def _run_suite(argv: list[str], output: Path) -> dict:
    """Run the suite in-process on ``argv`` and load the payload it wrote."""

    from scripts import qwen36_dense_gguf_suite as suite

    original_argv = sys.argv
    try:
        sys.argv = list(argv)
        suite.main()
    finally:
        sys.argv = original_argv
    if not output.exists():
        raise SystemExit(f"suite produced no payload: {output}")
    return json.loads(output.read_text())

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--prompts", type=Path, default=Path(DEFAULT_PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--device-index",
        type=int,
        required=True,
        help=(
            "HIP device index to run on. Required: this host has more than one "
            "gfx1100 card and the device is never inferred."
        ),
    )
    parser.add_argument(
        "--expect-device",
        default=None,
        help="substring the suite's reported device_name must contain",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass through to the suite; keep on for a representative comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/gguf-ar-eager-graph-equivalence"),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict] = {}
    with device_selected(int(args.device_index)):
        # The suite always materializes the NextN draft provider, so an AR-only
        # comparison still needs MTP admission. The U6 pin is empty, so the
        # scope is granted in this process only -- never written -- exactly as
        # the paired protocol does.
        with mtp_scope_granted({str(args.quant)}):
            for mode in AR_MODES:
                output = args.output_dir / f"ar-{mode}.json"
                argv = _suite_argv(args, mode, output)
                print(f"[equivalence] {mode}: {' '.join(argv)}", flush=True)
                payloads[mode] = _run_suite(argv, output)

    device_names = {
        mode: str((payload.get("provenance") or {}).get("device_name") or "")
        for mode, payload in payloads.items()
    }
    if len(set(device_names.values())) != 1:
        print(
            f"[equivalence] the two modes ran on different devices: {device_names}",
            file=sys.stderr,
        )
        return 1
    reported = next(iter(device_names.values()))
    if args.expect_device and args.expect_device not in reported:
        print(
            f"[equivalence] device mismatch: reported {reported!r} does not contain "
            f"{args.expect_device!r}",
            file=sys.stderr,
        )
        return 1

    report = compare_ar_token_sequences(payloads["eager"], payloads["graph"])
    report["device_name"] = reported
    report["device_index"] = int(args.device_index)
    report["model"] = str(args.model)
    report["quant"] = str(args.quant)
    report["max_new_tokens"] = int(args.max_new_tokens)
    report["limit"] = None if args.limit is None else int(args.limit)
    report["raw_payloads"] = {
        mode: str(args.output_dir / f"ar-{mode}.json") for mode in AR_MODES
    }

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[equivalence] wrote {args.json}")
    print(
        f"[equivalence] device={reported!r} rows={report['compared_rows']} "
        f"transitions={report['compared_transitions']} matched={report['matched']}"
    )
    for problem in report["problems"]:
        print(f"[equivalence]   {problem}", file=sys.stderr)
    return 0 if report["matched"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
