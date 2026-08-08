#!/usr/bin/env python3
"""Gate Moonshine's exact gfx1151 LM-head candidate on six real-audio fixtures.

The gate runs every retained real fixture through both ``wave8_argmax`` and
``wave8_top1`` under eager dispatch and four-bucket HIP graph replay.  Child
runs use prebuilt code objects only.  Promotion requires exact generation
through EOS, fixture state/logit gates, zero timed allocation, complete
teardown, four graph captures with 194 replays, and paired route outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance

SIX_REAL_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)
LM_HEAD_ROUTES = ("wave8_argmax", "wave8_top1")
TOKEN_ROUTES = ("eager", "graph")
_EXPECTED_GRAPH_BUCKETS = (
    "position_0",
    "position_1",
    "positions_2_3",
    "positions_4_193",
)
_PAIRED_EXACT_FIELDS = (
    "boundary_capture",
    "encoder_frames",
    "source_frames",
    "failures",
    "first_eos_position",
    "generation_tokens_exact",
    "selected_tokens_exact",
    "post_eos_unselected_token_mismatches",
    "positions",
    "max_abs",
    "max_relative_l2",
    "logit_kl_mean",
    "logit_kl_max",
    "logit_top1_agreement",
    "logit_gate_passed",
    "timed_step_allocations",
    "token_route",
    "teardown_current_bytes",
    "teardown_active_allocations",
    "quantization",
)
_GRAPH_EXACT_FIELDS = (
    "captured",
    "graph_count",
    "buckets",
    "capture_positions",
    "replay_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_equal(
    failures: list[str],
    report: Mapping[str, Any],
    field: str,
    expected: object,
) -> None:
    observed = report.get(field)
    if observed != expected:
        failures.append(f"{field}={observed!r}, expected {expected!r}")


def smoke_report_failures(
    report: Mapping[str, Any],
    *,
    expected_route: str,
    expected_token_route: str,
) -> list[str]:
    """Return all admission failures in one decoder-smoke report."""

    if expected_route not in LM_HEAD_ROUTES:
        raise ValueError(f"unknown LM-head route {expected_route!r}")
    if expected_token_route not in TOKEN_ROUTES:
        raise ValueError(f"unknown token route {expected_token_route!r}")

    failures: list[str] = []
    _expect_equal(failures, report, "all_passed", True)
    _expect_equal(failures, report, "failures", [])
    _expect_equal(failures, report, "generation_tokens_exact", True)
    _expect_equal(failures, report, "selected_tokens_exact", True)
    _expect_equal(failures, report, "timed_step_allocations", 0)
    _expect_equal(failures, report, "token_route", expected_token_route)
    _expect_equal(failures, report, "logit_gate_passed", True)
    _expect_equal(failures, report, "teardown_current_bytes", 0)
    _expect_equal(failures, report, "teardown_active_allocations", 0)
    _expect_equal(
        failures,
        report,
        "boundary_capture",
        "all_layer_boundaries" if expected_token_route == "eager" else "final_hidden_only",
    )

    top1 = report.get("logit_top1_agreement")
    if not isinstance(top1, (int, float)) or float(top1) < 0.9:
        failures.append(f"logit_top1_agreement={top1!r}, expected >= 0.9")
    kl_max = report.get("logit_kl_max")
    if not isinstance(kl_max, (int, float)) or float(kl_max) > 0.05:
        failures.append(f"logit_kl_max={kl_max!r}, expected <= 0.05")

    lm_head = report.get("lm_head")
    if not isinstance(lm_head, Mapping):
        failures.append("lm_head is not an object")
    else:
        expected_partials = 4_608 if expected_route == "wave8_top1" else 0
        expected_dtype = "fp16" if expected_partials else None
        expected_index_dtype = "int64" if expected_partials else None
        for field, expected in (
            ("route", expected_route),
            ("materializes_full_fp16_logits", True),
            ("stable_lowest_id_top1", True),
            ("partial_count", expected_partials),
            ("partial_value_dtype", expected_dtype),
            ("partial_index_dtype", expected_index_dtype),
            ("fallback", "wave8_argmax"),
        ):
            observed = lm_head.get(field)
            if observed != expected:
                failures.append(f"lm_head.{field}={observed!r}, expected {expected!r}")

    token_graph = report.get("token_graph")
    if expected_token_route == "eager":
        if token_graph is not None:
            failures.append("token_graph must be null for eager dispatch")
    elif not isinstance(token_graph, Mapping):
        failures.append("token_graph is not an object for graph dispatch")
    else:
        for field, expected in (
            ("captured", True),
            ("graph_count", 4),
            ("buckets", list(_EXPECTED_GRAPH_BUCKETS)),
            ("capture_positions", [0, 1, 2, 4]),
            ("replay_count", 194),
        ):
            observed = token_graph.get(field)
            if observed != expected:
                failures.append(f"token_graph.{field}={observed!r}, expected {expected!r}")
    return failures


def _matrix_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        return (
            str(row["fixture"]),
            str(row["lm_head_route"]),
            str(row["token_route"]),
        )
    except KeyError as error:
        raise ValueError(f"matrix row is missing {error.args[0]}") from error


def build_gate_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_fixtures: Sequence[str] = SIX_REAL_FIXTURES,
) -> dict[str, Any]:
    """Validate a complete route matrix and return its promotion decision."""

    expected_keys = {
        (fixture, route, token_route)
        for fixture in expected_fixtures
        for route in LM_HEAD_ROUTES
        for token_route in TOKEN_ROUTES
    }
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _matrix_key(row)
        if key in by_key:
            raise ValueError(f"duplicate matrix row {key}")
        by_key[key] = row
    observed_keys = set(by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(f"incomplete route matrix: missing={missing}, extra={extra}")

    failures: list[str] = []
    for key in sorted(by_key):
        row = by_key[key]
        fixture, route, token_route = key
        returncode = row.get("returncode")
        if returncode != 0:
            failures.append(f"{fixture}/{token_route}/{route}: returncode={returncode!r}")
        report = row.get("report")
        if not isinstance(report, Mapping):
            failures.append(f"{fixture}/{token_route}/{route}: report missing")
            continue
        failures.extend(
            f"{fixture}/{token_route}/{route}: {failure}"
            for failure in smoke_report_failures(
                report,
                expected_route=route,
                expected_token_route=token_route,
            )
        )

    paired_exact = True
    extra_resident_bytes: set[int] = set()
    for fixture in expected_fixtures:
        for token_route in TOKEN_ROUTES:
            fallback = by_key[(fixture, "wave8_argmax", token_route)].get("report")
            candidate = by_key[(fixture, "wave8_top1", token_route)].get("report")
            if not isinstance(fallback, Mapping) or not isinstance(candidate, Mapping):
                paired_exact = False
                continue
            for field in _PAIRED_EXACT_FIELDS:
                if fallback.get(field) != candidate.get(field):
                    paired_exact = False
                    failures.append(
                        f"{fixture}/{token_route}: paired {field} differs: "
                        f"{fallback.get(field)!r} != {candidate.get(field)!r}"
                    )
            if token_route == "graph":
                fallback_graph = fallback.get("token_graph")
                candidate_graph = candidate.get("token_graph")
                if not isinstance(fallback_graph, Mapping) or not isinstance(
                    candidate_graph, Mapping
                ):
                    paired_exact = False
                else:
                    for field in _GRAPH_EXACT_FIELDS:
                        if fallback_graph.get(field) != candidate_graph.get(field):
                            paired_exact = False
                            failures.append(
                                f"{fixture}/{token_route}: paired token_graph.{field} differs"
                            )
            fallback_nbytes = fallback.get("resident_nbytes")
            candidate_nbytes = candidate.get("resident_nbytes")
            if isinstance(fallback_nbytes, int) and isinstance(candidate_nbytes, int):
                difference = candidate_nbytes - fallback_nbytes
                extra_resident_bytes.add(difference)
                if difference != 46_080:
                    failures.append(
                        f"{fixture}/{token_route}: candidate resident delta "
                        f"{difference}, expected 46080"
                    )
            else:
                failures.append(f"{fixture}/{token_route}: resident_nbytes missing")

    unique_failures = sorted(set(failures))
    return {
        "passed": not unique_failures,
        "matrix_rows": len(rows),
        "fixture_count": len(expected_fixtures),
        "routes": list(LM_HEAD_ROUTES),
        "token_routes": list(TOKEN_ROUTES),
        "all_transcripts_exact_through_eos": all(
            isinstance(row.get("report"), Mapping)
            and row["report"].get("generation_tokens_exact") is True
            for row in rows
        ),
        "all_selected_fixture_tokens_exact": all(
            isinstance(row.get("report"), Mapping)
            and row["report"].get("selected_tokens_exact") is True
            for row in rows
        ),
        "all_logit_gates_passed": all(
            isinstance(row.get("report"), Mapping)
            and row["report"].get("logit_gate_passed") is True
            for row in rows
        ),
        "all_timed_allocations_zero": all(
            isinstance(row.get("report"), Mapping)
            and row["report"].get("timed_step_allocations") == 0
            for row in rows
        ),
        "all_teardowns_clean": all(
            isinstance(row.get("report"), Mapping)
            and row["report"].get("teardown_current_bytes") == 0
            and row["report"].get("teardown_active_allocations") == 0
            for row in rows
        ),
        "all_graph_runs_four_bucket_194_replay": all(
            isinstance(row.get("report"), Mapping)
            and isinstance(row["report"].get("token_graph"), Mapping)
            and row["report"]["token_graph"].get("graph_count") == 4
            and row["report"]["token_graph"].get("replay_count") == 194
            for row in rows
            if row.get("token_route") == "graph"
        ),
        "paired_route_outcomes_exact": paired_exact,
        "candidate_extra_resident_bytes": sorted(extra_resident_bytes),
        "failures": unique_failures,
    }


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "all_passed",
        "boundary_capture",
        "encoder_frames",
        "source_frames",
        "failures",
        "first_eos_position",
        "generation_tokens_exact",
        "selected_tokens_exact",
        "post_eos_unselected_token_mismatches",
        "positions",
        "max_abs",
        "max_relative_l2",
        "logit_kl_mean",
        "logit_kl_max",
        "logit_top1_agreement",
        "logit_gate_passed",
        "timed_step_allocations",
        "token_route",
        "lm_head",
        "resident_nbytes",
        "teardown_current_bytes",
        "teardown_active_allocations",
        "token_graph",
        "quantization",
    )
    return {field: report.get(field) for field in fields}


def _run_child(
    *,
    python: Path,
    smoke_script: Path,
    model_path: Path,
    fixture_path: Path,
    compiler_version_file: Path,
    route: str,
    token_route: str,
    raw_dir: Path,
) -> dict[str, Any]:
    stem = f"{fixture_path.stem}-{token_route}-{route}"
    raw_report = raw_dir / f"{stem}.json"
    stdout_path = raw_dir / f"{stem}.stdout.txt"
    stderr_path = raw_dir / f"{stem}.stderr.txt"
    command = [
        str(python),
        str(smoke_script),
        "--compiler-version-file",
        str(compiler_version_file),
        "--model-path",
        str(model_path),
        "--fixture",
        str(fixture_path),
        "--require-cached-build",
        "--pad-to-certified-bucket",
        "--lm-head-route",
        route,
        "--token-route",
        token_route,
        "--json",
        str(raw_report),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=dict(os.environ),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    payload: dict[str, Any] | None = None
    if raw_report.is_file():
        payload = json.loads(raw_report.read_text(encoding="utf-8"))
    return {
        "fixture": fixture_path.stem,
        "lm_head_route": route,
        "token_route": token_route,
        "returncode": completed.returncode,
        "command": command,
        "report": payload,
        "raw_report": (
            {"path": str(raw_report), "sha256": _sha256(raw_report)}
            if raw_report.is_file()
            else None
        ),
        "stdout": {"path": str(stdout_path), "sha256": _sha256(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": _sha256(stderr_path)},
        "stderr_tail": completed.stderr[-2000:],
    }


def _fixture_identity(fixture_dir: Path) -> dict[str, Any]:
    fixtures: dict[str, Any] = {}
    producers: list[Mapping[str, Any]] = []
    for fixture in SIX_REAL_FIXTURES:
        sidecar = fixture_dir / f"{fixture}.json"
        archive = fixture_dir / f"{fixture}.npz"
        if not sidecar.is_file() or not archive.is_file():
            raise FileNotFoundError(f"missing fixture bundle for {fixture}")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        producer = manifest["producer"]
        producers.append(
            {
                key: value
                for key, value in producer.items()
                if key not in {"first_eos_position", "encoder_mask_source"}
            }
        )
        first_eos_position = int(producer["first_eos_position"])
        fixtures[fixture] = {
            "sidecar": {"bytes": sidecar.stat().st_size, "sha256": _sha256(sidecar)},
            "archive": {"bytes": archive.stat().st_size, "sha256": _sha256(archive)},
            "input": manifest["input"],
            "first_eos_position": first_eos_position,
            "generated_ids_through_eos": manifest["decoder"]["token_ids"][
                : first_eos_position + 2
            ],
            "positions": manifest["decoder"]["positions"],
        }
    if any(producer != producers[0] for producer in producers[1:]):
        raise ValueError("six-fixture producer identities differ")
    summary = fixture_dir.parent / "moonshine-fixtures-six-summary.json"
    return {
        "root": str(fixture_dir),
        "producer": dict(producers[0]),
        "summary": (
            {"path": str(summary), "bytes": summary.stat().st_size, "sha256": _sha256(summary)}
            if summary.is_file()
            else None
        ),
        "fixtures": fixtures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_path = args.model_path.expanduser().resolve()
    fixture_dir = args.fixture_dir.expanduser().resolve()
    compiler_version_file = args.compiler_version_file.expanduser().resolve()
    python = args.python.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, label in (
        (model_path, "model path"),
        (fixture_dir, "fixture directory"),
        (compiler_version_file, "compiler version file"),
        (python, "Python interpreter"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    fixture_identity = _fixture_identity(fixture_dir)
    hipcc_version = compiler_version_file.read_text(encoding="utf-8")
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1151",
        resolved_backend="hip_gfx1151",
        target_arch="gfx1151",
        model_path=model_path,
        model_revision="cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
        quant="fp16",
        kv_dtype="fp16",
        command=sys.argv,
        build_profile="decode",
        timing_protocol="correctness_only_6_fixtures_x_2_routes_x_eager_graph",
        warmups=0,
        repetitions=1,
        profiler={"cached_build_required": True, "new_trace_required": False},
        hipcc_version=hipcc_version,
    )
    if args.require_clean and provenance["dirty"]:
        raise RuntimeError("refusing to publish the G1 route gate from a dirty worktree")

    raw_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    smoke_script = REPO_ROOT / "scripts/moonshine_decoder_smoke.py"
    rows = [
        _run_child(
            python=python,
            smoke_script=smoke_script,
            model_path=model_path,
            fixture_path=fixture_dir / f"{fixture}.npz",
            compiler_version_file=compiler_version_file,
            route=route,
            token_route=token_route,
            raw_dir=raw_dir,
        )
        for fixture in SIX_REAL_FIXTURES
        for token_route in TOKEN_ROUTES
        for route in LM_HEAD_ROUTES
    ]
    summary = build_gate_summary(rows)
    source_paths = (
        Path(__file__),
        smoke_script,
        REPO_ROOT / "hipengine/runtime/moonshine.py",
        REPO_ROOT / "hipengine/kernels/hip_gfx1100/linear/moonshine_projection.py",
        REPO_ROOT / "hipengine/kernels/hip_gfx1100/linear/moonshine_projection.hip",
    )
    artifact = {
        "schema_version": 1,
        "kind": "hipengine_moonshine_lm_head_route_admission",
        "status": "accepted_runtime_default" if summary["passed"] else "rejected",
        "scope": "six_real_audio_fixtures_eager_and_four_bucket_graph_correctness",
        "performance_claim": False,
        "model": {
            "id": "shisa-ai/shisa-realtime-asr-0.92b",
            "revision": "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
        },
        "fixture_collection": fixture_identity,
        "gate": summary,
        "decision": {
            "promote_wave8_top1": summary["passed"],
            "runtime_default": "wave8_top1" if summary["passed"] else "wave8_argmax",
            "fallback": "wave8_argmax",
            "reason": (
                "all 24 real-audio eager/graph route rows passed transcript, state, "
                "allocation, graph, paired-outcome, and teardown gates"
                if summary["passed"]
                else "one or more G1 real-audio route admission checks failed"
            ),
        },
        "rows": [
            {
                **{key: value for key, value in row.items() if key != "report"},
                "report": (
                    _compact_report(row["report"])
                    if isinstance(row.get("report"), Mapping)
                    else None
                ),
            }
            for row in rows
        ],
        "provenance": provenance,
        "source_files": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths
        },
        "raw_reports_committed": False,
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": artifact["status"], **summary}, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
