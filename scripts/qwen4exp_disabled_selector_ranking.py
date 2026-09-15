#!/usr/bin/env python3
"""Rank the disabled production arithmetic selectors by recoverable prefill time.

The gfx1151 ``gguf_ud_q4_k_xl`` production binder disables every selector in
``PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`` and then re-admits a smaller set. The
selectors that stay off are off as a group; nothing in the tree records what
each one is individually worth at the current HEAD, which is the number needed
to decide what to requalify first.

This runs one child per arm through ``scripts/qwen4exp_profile_gap.py`` with
``--override``, so every arm is the same binary, the same fixture, the same
case and the same chunk size, and differs only by the flag under test. The
child reports its own effective route environment, so an arm that silently
failed to take effect is visible rather than inferred.

Performance-only. It measures wall time and records the child's generated
token id; it does not evaluate numerics, and a faster arm here is not a
qualified arm. Arms whose layer companions are empty are skipped with a reason
instead of being reported as neutral.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.generation.qwen4_exp_profiles import (  # noqa: E402
    PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS,
    PRODUCTION_GDN_PEER_PREFILL_LAYERS,
    PRODUCTION_Q4_IU8_PREFILL_LAYERS,
    PRODUCTION_QSA_FLASH_PREFILL_LAYERS,
)

DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
CHILD = REPO_ROOT / "scripts" / "qwen4exp_profile_gap.py"

_L = lambda layers: ",".join(str(layer) for layer in layers)  # noqa: E731

# Arm name -> overrides. Each re-enables exactly one disabled selector, with the
# layer companion the production binder would have supplied for it, plus any
# flag that would otherwise shadow it.
#
# Shadowing notes, read from the dispatch rather than assumed:
# * ``Q4_IU8_PREFILL`` is unreachable while ``EXACT_GROUPED_Q4=1`` (the
#   production default), because the exact-risk+repair branch is an earlier arm
#   of the same ``if``/``elif`` chain in ``qwen4_exp_runner``. Enabling the fast
#   suffix therefore requires turning the exact route off, and an arm that only
#   sets ``Q4_IU8_PREFILL=1`` measures the exact route twice.
ARMS: dict[str, dict[str, str]] = {
    "baseline": {},
    "moe": {"HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL": "1"},
    "q4_iu8_fast": {
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": _L(PRODUCTION_Q4_IU8_PREFILL_LAYERS),
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "0",
    },
    "q8_mmq": {"HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1"},
    "q8_iu8_wmm": {"HIPENGINE_QWEN4_EXP_Q8_IU8_WMM": "1"},
    "gr_iu8": {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"},
    "gr_iu8_down": {"HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1"},
    "gdn_peer": {
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS": _L(
            PRODUCTION_GDN_PEER_PREFILL_LAYERS
        ),
    },
    "gdn_colwarps": {
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": _L(
            PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS
        ),
    },
    "qsa_flash": {
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS": _L(
            PRODUCTION_QSA_FLASH_PREFILL_LAYERS
        ),
    },
    "gr_both": {
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1",
    },
    "all_prefill": {
        "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": _L(PRODUCTION_Q4_IU8_PREFILL_LAYERS),
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_Q8_IU8_WMM": "1",
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS": _L(
            PRODUCTION_GDN_PEER_PREFILL_LAYERS
        ),
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": _L(
            PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS
        ),
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS": _L(
            PRODUCTION_QSA_FLASH_PREFILL_LAYERS
        ),
    },
}


def _run_arm(
    name: str,
    overrides: dict[str, str],
    *,
    model_root: Path,
    case_id: str,
    fixture: Path,
    chunk_size: int,
    repetitions: int,
    compiler_version_file: Path,
    work_root: Path,
) -> dict:
    directory = work_root / name
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "child.json"
    if output.exists():
        output.unlink()
    command = [
        sys.executable,
        str(CHILD),
        "--model-root",
        str(model_root),
        "--mode",
        "prefill",
        "--fixture",
        str(fixture),
        "--case-id",
        case_id,
        "--prefill-chunk-size",
        str(chunk_size),
        "--repetitions",
        str(repetitions),
        "--compiler-version-file",
        str(compiler_version_file),
        "--require-cached-build",
        "--output",
        str(output),
    ]
    for key, value in overrides.items():
        command += ["--override", f"{key}={value}"]

    started = time.monotonic()
    log = directory / "process.log"
    with log.open("wb") as handle:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.monotonic() - started
    row: dict = {
        "arm": name,
        "overrides": dict(overrides),
        "returncode": completed.returncode,
        "process_seconds": elapsed,
        "command": command,
    }
    if not output.exists():
        row["status"] = "no_output"
        return row
    child = json.loads(output.read_text())
    wall = child.get("wall_summary") or {}
    route = child.get("route_env") or {}
    bound = child.get("bound_route_env") or {}
    row.update(
        {
            "status": "measured",
            "tok_s": wall.get("tok_s"),
            "wall_seconds": wall.get("wall_seconds"),
            "token_id": child.get("token_id"),
            "configuration_class": child.get("configuration_class"),
            "named_profile_intact": child.get("named_profile_intact"),
            "fell_back_to_strict": child.get("fell_back_to_strict"),
            "override_stage": child.get("override_stage"),
            "overrides_recorded": child.get("overrides"),
            "effective_route": {
                key: route.get(key)
                for key in sorted(overrides)
                if key in route
            },
            "effective_route_missing_keys": [
                key for key in sorted(overrides) if key not in route
            ],
            "baseline_route": {
                key: bound.get(key)
                for key in sorted(ARMS["all_prefill"])
            },
        }
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=REPO_ROOT
        / "benchmarks"
        / "fixtures"
        / "qwen4exp_canonical_ar_p512_p1024_p4096.json",
    )
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        help=f"Subset of arms to run (available: {', '.join(ARMS)})",
    )
    args = parser.parse_args(argv)

    unknown = [name for name in args.arms if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    args.work_root.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_disabled_selector_ranking",
        "performance_claim": False,
        "numerics_evaluated": False,
        "case_id": args.case_id,
        "prefill_chunk_size": args.prefill_chunk_size,
        "repetitions": args.repetitions,
        "fixture": str(args.fixture),
        "model_root": str(args.model_root),
        "arms": [],
    }
    baseline = None
    for name in args.arms:
        row = _run_arm(
            name,
            ARMS[name],
            model_root=args.model_root,
            case_id=args.case_id,
            fixture=args.fixture,
            chunk_size=args.prefill_chunk_size,
            repetitions=args.repetitions,
            compiler_version_file=args.compiler_version_file,
            work_root=args.work_root,
        )
        if name == "baseline" and row.get("tok_s"):
            baseline = row["tok_s"]
        if baseline and row.get("tok_s"):
            row["ratio_vs_baseline"] = row["tok_s"] / baseline
            row["delta_pct_vs_baseline"] = 100.0 * (row["tok_s"] / baseline - 1.0)
        report["arms"].append(row)
        report["baseline_tok_s"] = baseline
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(
            f"{name:16s} rc={row['returncode']} "
            f"tok_s={row.get('tok_s')} "
            f"ratio={row.get('ratio_vs_baseline')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
