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
    artifact_path = (
        repo_root
        / "benchmarks/results/2026-07-11-gfx1151-readme-refresh-"
        "20260711-d1231ee0-summary.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    root_readme = (repo_root / "README.md").read_text(encoding="utf-8")

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
    for table_key, heading in headings.items():
        rows = artifact["tables"][table_key]
        assert [row["workload"] for row in rows] == workloads
        assert f"#### {heading}" in canonical
        assert f"#### {heading}" in root_readme
        assert table_header in canonical
        assert table_header in root_readme
        for row in rows:
            published_row = (
                f"| {row['workload']} | {row['hipengine_paro']:.3f} | "
                f"{row['hipengine_gguf']:.3f} | {row['llamacpp_hip']:.3f} | "
                f"{row['llamacpp_vulkan']:.3f} |"
            )
            assert published_row in canonical
            assert published_row in root_readme

    for name, component in artifact["components"].items():
        assert (repo_root / "benchmarks/results" / component["name"]).is_file(), name
        provenance = artifact["component_provenance"][name]
        assert provenance["hipengine_commit"] == artifact["measured_hipengine_commit"]
        assert provenance["target_arch"] == "gfx1151"
        assert provenance["dirty"] is False

    for correctness_path in artifact["linked_correctness"].values():
        assert (repo_root / correctness_path).is_file()


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
