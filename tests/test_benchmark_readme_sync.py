from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_root_readme_benchmark_blocks_match_canonical_scoreboard() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/sync_benchmark_readme.py", "--check"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_gfx1151_model_topline_is_accepted_and_published_from_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "benchmarks/results"
    artifact_path = (
        results_dir
        / "2026-07-11-gfx1151-readme-refresh-"
        "20260711-d1231ee0-summary.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    paro_recovery = json.loads(
        (results_dir / "2026-07-12-gfx1151-paro-prefill-recovery.json").read_text(
            encoding="utf-8"
        )
    )
    paro_isolation = json.loads(
        (
            results_dir
            / "2026-07-12-gfx1151-paro-aotriton-stream-isolation.json"
        ).read_text(encoding="utf-8")
    )
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    root_readme = (repo_root / "README.md").read_text(encoding="utf-8")
    canonical_values = canonical.replace("**", "")
    root_values = root_readme.replace("**", "")

    assert artifact["schema"] == 1
    assert artifact["status"] == "accepted_topline"
    assert artifact["performance_claim"] is True
    assert artifact["measured_hipengine_commit"] == (
        "d1231ee081d9cc6799f59632a8e8db96de4c61c3"
    )
    assert artifact["assembly"]["hipengine_commit"] == (
        "7e9aad21d92b6ed2c6cf7ff83aa7b5896a74d15b"
    )
    assert artifact["assembly"]["dirty"] is False
    assert artifact["gates"]["all_passed"] is True
    assert all(artifact["gates"].values())
    assert paro_recovery["status"] == "accepted"
    assert paro_recovery["performance_claim"] is True
    assert paro_recovery["correctness_claim"] is True
    assert paro_recovery["provenance"]["hipengine_commit"].startswith("9944e481")
    assert paro_recovery["provenance"]["dirty"] is False
    assert paro_isolation["status"] == "accepted"
    assert paro_isolation["performance_claim"] is True
    assert paro_isolation["correctness_claim"] is True
    assert paro_isolation["measured_revision"].startswith("01e2cec5")
    assert paro_isolation["provenance"]["dirty"] is False
    assert paro_isolation["correctness"]["mismatch_paths"] == []
    assert paro_isolation["scope"]["refresh_pending"] == []
    assert set(paro_isolation["shape_refresh"]["retained_results"]) == {
        "32K/128",
        "64K/128",
        "128K/128",
    }
    assert all(
        not result["mismatch_paths"]
        for result in paro_isolation["shape_refresh"]["correctness"].values()
    )
    assert (
        paro_isolation["shape_refresh"]["negative_control"][
            "isolation_branch_entered"
        ]
        is False
    )

    workloads = ["512/128", "1K/128", "4K/128", "32K/128", "64K/128", "128K/128"]
    columns = [
        ("hipengine_paro", "hipEngine PARO"),
        ("hipengine_gguf", "hipEngine GGUF"),
        ("llamacpp_hip", "llama.cpp HIP"),
        ("llamacpp_vulkan", "llama.cpp Vulkan"),
    ]
    assert artifact["workloads"] == workloads
    assert [
        (column["key"], column["label"]) for column in artifact["columns"]
    ] == columns

    headings = {
        "prefill_tok_s": "Prefill tok/s",
        "decode_tok_s": "Decode tok/s",
        "peak_gib": "Peak memory GiB",
    }
    table_header = (
        "| Workload | hipEngine PARO | hipEngine GGUF | "
        "llama.cpp HIP | llama.cpp Vulkan |"
    )
    recovery_rows = {row["workload"]: row for row in paro_recovery["results"]}
    paro_result_keys = {
        "prefill_tok_s": "candidate_prefill_tok_s",
        "decode_tok_s": "candidate_decode_tok_s",
        "peak_gib": "candidate_peak_gib",
    }
    for table_key, heading in headings.items():
        rows = artifact["tables"][table_key]
        assert [row["workload"] for row in rows] == workloads
        assert f"#### {heading}" in canonical
        assert f"#### {heading}" in root_readme
        assert table_header in canonical
        assert table_header in root_readme
        for row in rows:
            paro_value = recovery_rows[row["workload"]][paro_result_keys[table_key]]
            if row["workload"] == "4K/128":
                isolated = paro_isolation["performance"]["candidate_isolated_stream"]
                isolation_keys = {
                    "prefill_tok_s": ("prefill_tok_s", "median"),
                    "decode_tok_s": ("decode_tok_s", "median"),
                    "peak_gib": ("tracked_peak_allocated_gib",),
                }
                path = isolation_keys[table_key]
                paro_value = isolated[path[0]]
                if len(path) == 2:
                    paro_value = paro_value[path[1]]
            elif row["workload"] in paro_isolation["shape_refresh"][
                "retained_results"
            ]:
                isolated = paro_isolation["shape_refresh"]["retained_results"][
                    row["workload"]
                ]["candidate"]
                if table_key == "prefill_tok_s":
                    paro_value = isolated["prefill_tok_s"]["median"]
                elif table_key == "decode_tok_s":
                    paro_value = isolated["decode_tok_s"]["median"]
                else:
                    paro_value = isolated["tracked_peak_allocated_gib"]
            published_row = (
                f"| {row['workload']} | {paro_value:.3f} | "
                f"{row['hipengine_gguf']:.3f} | {row['llamacpp_hip']:.3f} | "
                f"{row['llamacpp_vulkan']:.3f} |"
            )
            assert published_row in canonical_values
            assert published_row in root_values

    assert "The 1K follow-up shared a max-32K session" in canonical
    assert "does not replace the existing right-sized 1K row" in canonical

    for name, component in artifact["components"].items():
        assert (repo_root / "benchmarks/results" / component["name"]).is_file(), name
        provenance = artifact["component_provenance"][name]
        assert provenance["hipengine_commit"] == artifact["measured_hipengine_commit"]
        assert provenance["target_arch"] == "gfx1151"
        assert provenance["dirty"] is False

    for correctness_path in artifact["linked_correctness"].values():
        assert (repo_root / correctness_path).is_file()


def test_gfx1151_mtp_topline_separates_exact_compat_and_llamacpp() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = repo_root / "benchmarks/results"
    exact_fixed = json.loads(
        (results / "2026-07-02-ar-mtp-default-parallelattn-full.json").read_text(
            encoding="utf-8"
        )
    )
    exact_natural = json.loads(
        (
            results / "2026-07-03-ar-mtp-default-natural24-budget-sweep-c1.json"
        ).read_text(encoding="utf-8")
    )
    compat = json.loads(
        (
            results
            / "2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-"
            "cyclecap24-f32head-full.json"
        ).read_text(encoding="utf-8")
    )
    llamacpp = json.loads(
        (
            results / "2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json"
        ).read_text(encoding="utf-8")
    )
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    root_readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert exact_fixed["apple_to_apple_ok"] is True
    assert exact_natural["apple_to_apple_ok"] is True
    assert compat["apple_to_apple_ok"] is True
    assert llamacpp["status"] == "diagnostic_retained"
    assert llamacpp["performance_claim"] is False

    exact_b5 = exact_fixed["mtp_by_budget"]["b5"]
    exact_b2 = exact_natural["mtp_by_budget"]["b2"]
    compat_b2 = compat["mtp_by_budget"]["b2"]
    llama_summary = llamacpp["summary"]["natural"]
    llama_timing = llamacpp["stage_timing_summary"]["measured_excluding_first_task"]
    expected_rows = [
        (
            f"| Headline MTP decode | {exact_b5['decode_tok_s_weighted']:.2f} "
            f"tok/s ({exact_b5['vs_ar_ratio']:.4f}x own AR) | "
            f"{compat_b2['decode_tok_s_weighted']:.2f} tok/s "
            f"({compat_b2['vs_ar_ratio']:.4f}x own AR) | "
            f"{llama_summary['mtp_weighted_predicted_per_second']:.2f} tok/s "
            f"({llama_summary['speedup']:.4f}x own AR) |"
        ),
        (
            f"| Matched natural24 B2 MTP decode | "
            f"{exact_b2['decode_tok_s_weighted']:.2f} tok/s (diagnostic) | "
            f"{compat_b2['decode_tok_s_weighted']:.2f} tok/s | "
            f"{llama_summary['mtp_weighted_predicted_per_second']:.2f} tok/s |"
        ),
        (
            f"| Matched natural24 own AR | "
            f"{exact_natural['ar']['decode_tok_s_weighted']:.2f} tok/s | "
            f"{compat['ar']['decode_tok_s_weighted']:.2f} tok/s | "
            f"{llama_summary['base_weighted_predicted_per_second']:.2f} tok/s |"
        ),
        (
            f"| Matched natural24 cycle wall/output | "
            f"{exact_b2['cycle_wall_ms_per_output']:.3f} ms | "
            f"{compat_b2['cycle_wall_ms_per_output']:.3f} ms | "
            f"{llama_timing['cycle_wall_ms_per_output']:.3f} ms |"
        ),
    ]
    header = (
        "| Metric | hipEngine GGUF exact/default | "
        "hipEngine GGUF `llama-compat` | llama.cpp HIP |"
    )
    for readme in (canonical, root_readme):
        assert header in readme
        assert "`llama-compat` is the closer 1:1 performance comparison" in readme
        assert "(`performance_claim=false`)" in readme
        for expected_row in expected_rows:
            assert expected_row in readme


def test_gfx1151_legacy_paro_diagnostic_is_linked_but_not_published() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (
            repo_root
            / "benchmarks/results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json"
        ).read_text(encoding="utf-8")
    )
    scoreboard = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")

    assert artifact["performance_claim"] is False
    reference = artifact["rows"]["1"]
    expected_reference = (
        f"| 1 | {reference['decode_tok_s_aggregate_median']:.3f} | "
        f"{reference['decode_tok_s_per_request_median']:.3f} | "
        f"{reference['decode_step_ms_median_of_run_medians']:.3f} |"
    )
    assert expected_reference not in scoreboard
    assert "No eligible native-batch timing row" in scoreboard
    assert "2026-07-10...current-diagnostic-summary.json" in scoreboard
    assert reference["status"] == "diagnostic_reference"
    assert reference["same_fixture_as_cn"] is False
    assert reference["source"]["hipengine_commit"].startswith("4175dabf")
    assert reference["source"]["tracked_source_dirty"] is False
    assert reference["source"]["untracked_files_present"] is False

    for rows in range(2, 9):
        result = artifact["rows"][str(rows)]
        expected = (
            f"| {rows} | {result['decode_tok_s_aggregate_median']:.3f} | "
            f"{result['decode_tok_s_per_request_median']:.3f} | "
            f"{result['decode_step_ms_median_of_run_medians']:.3f} |"
        )
        assert expected not in scoreboard
        assert result["status"] == "diagnostic_legacy_batch_oracle"
        assert result["generated_token_equality"] is True
        assert result["primitive_correctness"] is True

    for rows in (3, 5, 7):
        result = artifact["rows"][str(rows)]
        assert result["source"]["hipengine_commit"].startswith("02aec604")
        assert result["source"]["tracked_source_dirty"] is False

    assert artifact["native_batch_width_profile"]["native_widths"] == list(range(2, 9))
    assert artifact["native_batch_width_profile"]["routing_eligible"] is False
    assert artifact["native_batch_width_profile"]["oracle_status"] == "invalid_batch_shaped_c1_reference"

    true_c1 = json.loads(
        (
            repo_root
            / "benchmarks/results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json"
        ).read_text(encoding="utf-8")
    )
    assert true_c1["performance_claim"] is False
    assert true_c1["software"]["hipengine_commit"].startswith("0c184517")
    assert true_c1["software"]["hipengine_dirty"] is False
    assert true_c1["results"]["serial_c1_decode_bridge"]["row_equality"] == [True] * 8
    assert true_c1["results"]["native_batch_decode"]["row_equality"] == [False] * 8
    assert true_c1["results"]["native_batch_decode"]["first_mismatch"] == {
        "live_width": 8,
        "generated_token_index": 2,
        "all_rows": True,
        "native_token_id": 17,
        "c1_token_id": 220,
    }
    assert true_c1["decision"]["routing_eligible"] is False
