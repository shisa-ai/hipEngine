from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hipengine.benchmark.provenance import validate_artifact_provenance


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


def test_gfx1100_mtp_topline_publishes_graph_ar_correction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (
            repo_root
            / "benchmarks/results/2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json"
        ).read_text(encoding="utf-8")
    )
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    root_readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert artifact["status"] == "accepted"
    assert artifact["performance_claim"] is True
    assert artifact["correctness_claim"] is True
    assert artifact["hardware"]["target_arch"] == "gfx1100"
    assert artifact["hardware"]["device"] == "AMD Radeon Pro W7900"
    gate = artifact["graph_correctness_and_break_even"]
    assert gate["checked_transitions"] == 24
    assert gate["hidden_gdn_kv_token_exact"] is True
    assert gate["minimum_admitted_transitions"] == 24

    natural = artifact["matched_natural24"]
    graph = natural["state_bound_graph"]
    llamacpp = natural["llamacpp_base"]
    exact = artifact["mtp_reclassification"]["exact_default_fixed_10_cycles"]
    compat = artifact["mtp_reclassification"]["llama_compat_natural24"]
    assert natural["all_graph_outputs_match_prior_eager_preview_and_tail"] is True
    assert graph["tok_s"] > llamacpp["transition_normalized_tok_s"]
    assert exact["apple_to_apple_ok"] is True
    assert compat["apple_to_apple_ok"] is True
    assert exact["mtp_vs_ar"] < 1.0
    assert compat["mtp_vs_ar"] < 1.0
    assert llamacpp["performance_claim"] is False

    expected_main_rows = [
        (
            f"| Decode | **{exact['true_ar_tok_s']:.2f} tok/s fixed / "
            f"{graph['tok_s']:.2f} tok/s natural24** | "
            f"{exact['mtp_tok_s']:.2f} tok/s | "
            f"{compat['mtp_tok_s']:.2f} tok/s | "
            f"{llamacpp['transition_normalized_tok_s']:.2f} tok/s transition-normalized |"
        ),
        (
            f"| MTP / own AR | 1.0000x | **{exact['mtp_vs_ar']:.4f}x** | "
            f"**{compat['mtp_vs_ar']:.4f}x** | n/a |"
        ),
    ]
    expected_split_rows = [
        (
            f"| Train | 6 | **{compat['splits']['train']['ar_tok_s']:.2f}** | "
            f"{compat['splits']['train']['mtp_tok_s']:.2f} | "
            f"**{compat['splits']['train']['ratio']:.4f}x** | "
            f"**{100 * compat['splits']['train']['draft_acceptance']:.2f}%**"
        ),
        (
            f"| Heldout | 4 | **{compat['splits']['heldout']['ar_tok_s']:.2f}** | "
            f"{compat['splits']['heldout']['mtp_tok_s']:.2f} | "
            f"**{compat['splits']['heldout']['ratio']:.4f}x** | "
            f"**{100 * compat['splits']['heldout']['draft_acceptance']:.2f}%**"
        ),
    ]
    header = (
        "| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | "
        "hipEngine GGUF `llama-compat` | llama.cpp HIP base AR |"
    )
    for readme in (canonical, root_readme):
        assert "#### GGUF MTP comparison, Radeon Pro W7900/gfx1100" in readme
        assert header in readme
        assert "##### W7900 `llama-compat` full-suite gate against graph AR" in readme
        assert "hipEngine is **93.30 versus 78.29 tok/s (+19.19%)**" in readme
        for expected in expected_main_rows:
            assert expected in readme
        for expected in expected_split_rows:
            assert expected in readme


def test_gfx1151_mtp_topline_separates_exact_compat_and_llamacpp() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = repo_root / "benchmarks/results"
    artifact = json.loads(
        (results / "2026-07-12-gfx1151-gguf-mtp-refresh.json").read_text(
            encoding="utf-8"
        )
    )
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    root_readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert artifact["status"] == "accepted_compat_with_exact_negative_and_external_diagnostic"
    assert artifact["performance_claim"] is True
    assert artifact["correctness_claim"] is True
    assert artifact["correctness"]["serial_prefix_state_exact"] is True
    exact = artifact["hipengine_exact_default"]
    compat = artifact["hipengine_llama_compat"]
    llamacpp = artifact["llamacpp_hip"]
    assert exact["apple_to_apple_ok"] is True
    assert exact["mtp_beats_ar"] is False
    assert compat["apple_to_apple_ok"] is True
    assert llamacpp["status"] == "diagnostic_retained"
    assert llamacpp["performance_claim"] is False
    assert artifact["timing_contract"]["llamacpp_cross_engine"].startswith(
        "request N+1 outputs"
    )
    assert llamacpp["transition_normalized"]["timed_decode_transitions_per_prompt"] == 24
    assert llamacpp["transition_normalized"]["base_decode_tok_s"] == (
        240_000 / llamacpp["transition_normalized"]["base_decode_ms_total"]
    )
    assert llamacpp["transition_normalized"]["mtp_decode_tok_s"] == (
        240_000 / llamacpp["transition_normalized"]["mtp_decode_ms_total"]
    )
    assert compat["cross_engine_cycle_wall"]["decode_tok_s"] == (
        1000 / compat["full"]["cycle_wall_ms_per_output"]
    )

    exact_b5 = exact["budgets"]["b5"]
    full = compat["full"]
    llama_native = llamacpp["native_reported"]
    llama_transition = llamacpp["transition_normalized"]
    expected_rows = [
        (
            f"| Canonical/native MTP decode | {exact_b5['decode_tok_s']:.2f} "
            f"tok/s ({exact_b5['vs_true_ar']:.4f}x own AR) | "
            f"**{full['decode_tok_s']:.2f} tok/s ({full['vs_true_ar']:.4f}x own AR)** | "
            f"{llama_native['mtp_decode_tok_s']:.2f} tok/s native "
            f"({llama_native['mtp_vs_base']:.4f}x own AR; not cross-engine comparable) |"
        ),
        (
            f"| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | "
            f"**{compat['cross_engine_cycle_wall']['decode_tok_s']:.2f} tok/s** | "
            f"{llama_transition['mtp_decode_tok_s']:.2f} tok/s |"
        ),
        (
            f"| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | "
            f"**{compat['true_ar']['decode_tok_s']:.2f} tok/s** | "
            f"{llama_transition['base_decode_tok_s']:.2f} tok/s |"
        ),
        (
            f"| Full | {full['prompts']} | {compat['true_ar']['decode_tok_s']:.2f} | "
            f"**{full['decode_tok_s']:.2f}** | **{full['vs_true_ar']:.4f}x** | "
            f"{100 * full['draft_acceptance']:.2f}% | "
            f"{100 * full['accepted_per_output']:.2f}% | "
            f"{full['cycle_wall_ms_per_output']:.3f} ms |"
        ),
    ]
    split_rows = {
        "Full": (compat["full"], compat["true_ar"]["decode_tok_s"]),
        "Train": (compat["train"], compat["train"]["true_ar_decode_tok_s"]),
        "Heldout": (compat["heldout"], compat["heldout"]["true_ar_decode_tok_s"]),
        "`code`": (
            compat["categories"]["code"],
            compat["categories"]["code"]["true_ar_decode_tok_s"],
        ),
        "`general_en`": (
            compat["categories"]["general_en"],
            compat["categories"]["general_en"]["true_ar_decode_tok_s"],
        ),
        "`general_ja`": (
            compat["categories"]["general_ja"],
            compat["categories"]["general_ja"]["true_ar_decode_tok_s"],
        ),
        "`mixed_ja_en`": (
            compat["categories"]["mixed_ja_en"],
            compat["categories"]["mixed_ja_en"]["true_ar_decode_tok_s"],
        ),
    }
    expected_split_rows = [
        (
            f"| {label} | {row['prompts']} | {ar_tok_s:.2f} | "
            f"**{row['decode_tok_s']:.2f}** | **{row['vs_true_ar']:.4f}x** | "
            f"{'**' if label in {'Train', 'Heldout'} else ''}"
            f"{100 * row['draft_acceptance']:.2f}%"
            f"{'**' if label in {'Train', 'Heldout'} else ''} | "
            f"{100 * row['accepted_per_output']:.2f}% | "
            f"{row['cycle_wall_ms_per_output']:.3f} ms |"
        )
        for label, (row, ar_tok_s) in split_rows.items()
    ]
    header = (
        "| Metric | hipEngine GGUF exact/default | "
        "hipEngine GGUF `llama-compat` | llama.cpp HIP |"
    )
    for readme in (canonical, root_readme):
        assert header in readme
        assert "##### gfx1151 `llama-compat` full-suite gate" in readme
        assert "one-untimed-token numerator" in readme
        assert "`performance_claim=false`" in readme
        for expected_row in expected_rows:
            assert expected_row in readme
        for expected_row in expected_split_rows:
            assert expected_row in readme


def test_llamacpp_benchmark_patchset_manifest_is_complete() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    patch_root = repo_root / "benchmarks/llama.cpp"
    manifest = json.loads((patch_root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["upstream"]["base_commit"] == (
        "6e9007ae61f4e994c27484759caac6ef2aa32b30"
    )
    assert manifest["instrumented_source"]["head_commit"] == (
        "1ebf790cda38d827559548f67b0469189690cc8c"
    )
    assert len(manifest["instrumented_source"]["local_commits"]) == 7
    assert manifest["validation"]["fresh_clone_git_apply_check"] is True
    assert manifest["validation"]["combined_diff_matches_captured_working_tree"] is True
    for patch in manifest["patches"]:
        path = patch_root / patch["path"]
        assert path.is_file()
        assert path.stat().st_size == patch["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == patch["sha256"]


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
