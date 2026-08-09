from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_LOCAL_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _assert_markdown_tables_are_rectangular(text: str) -> None:
    lines = text.splitlines()
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not _TABLE_SEPARATOR_RE.fullmatch(line):
            continue
        assert index > 0 and lines[index - 1].lstrip().startswith("|")
        expected_pipes = lines[index - 1].count("|")
        row = index
        while row < len(lines) and lines[row].lstrip().startswith("|"):
            assert lines[row].count("|") == expected_pipes, (
                f"malformed Markdown table at line {row + 1}: "
                f"expected {expected_pipes} pipes"
            )
            row += 1


def _heading_slug(heading: str) -> str:
    plain = re.sub(r"[`*_\[\]]", "", heading).strip().lower()
    plain = re.sub(r"[^\w\s-]", "", plain)
    return re.sub(r"\s+", "-", plain)


def _assert_local_markdown_links_exist(text: str, repo_root: Path) -> None:
    for target in _LOCAL_LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local_path, _, anchor = target.partition("#")
        path = repo_root / local_path
        assert path.exists(), f"broken README link: {target}"
        if not anchor or not path.is_file():
            continue
        heading_slugs = {
            _heading_slug(match.group(1))
            for line in path.read_text(encoding="utf-8").splitlines()
            if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
        }
        assert anchor in heading_slugs, f"broken README anchor: {target}"


def test_root_readme_is_compact_model_first_and_synced() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/sync_benchmark_readme.py", "--check"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) < 400
    assert readme.index("## Supported models") < readme.index("## Performance highlights")
    assert readme.index("## Performance highlights") < readme.index("## Status")
    for model_url in (
        "https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF",
        "https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed",
        "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF",
        "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "https://huggingface.co/poolside/Laguna-S-2.1-GGUF",
        "https://huggingface.co/deepgrove/maple-preview-2bit-mlx",
    ):
        assert model_url in readme
    assert readme.count("<!-- BEGIN TOPLINE:") == 1
    assert "<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->" in readme
    assert "## Memory Usage" not in readme
    assert "## Speculative decode (DFlash / MTP)" not in readme
    assert "H8C-H8Q" not in readme
    assert "SH14-C1" not in readme
    assert "**122.67 tok/s**" in readme
    assert "**214.788**" in readme
    assert "**Current release: v0.4.0 alpha.**" in readme
    assert "NVIDIA Blackwell (`sm_120a`)" in readme
    assert "Current development is\ntherefore focused on GGUF compatibility" in readme
    for internal_phrase in ("source-pinned", "physical c8", "packet reaches"):
        assert internal_phrase not in readme
    _assert_markdown_tables_are_rectangular(readme)
    _assert_local_markdown_links_exist(readme, repo_root)


def test_gfx1151_model_topline_is_accepted_and_published_from_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "benchmarks/results"
    artifact_path = (
        results_dir / "2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    head_major_path = (
        results_dir / "2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json"
    )
    head_major = json.loads(head_major_path.read_text(encoding="utf-8"))
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    canonical_values = canonical.replace("**", "")

    assert artifact["schema"] == 1
    assert artifact["status"] == (
        "accepted_current_topline_gguf_512_64k_128k_lifecycle_blocked"
    )
    assert artifact["performance_claim"] is True
    assert artifact["correctness_claim"] is True
    assert artifact["software"]["measurement_commit"] == (
        "2edbb2ee3ca74d7757500b5eafe737d43748489c"
    )
    assert artifact["software"]["tracked_dirty"] is False
    assert artifact["hardware"]["amd_iommu"] == "off"
    assert artifact["hardware"]["iommu_groups_after_boot"] == 0
    assert "NPU unavailable" in artifact["hardware"]["xdna_side_effect"]
    assert "not a causal IOMMU-only A/B" in artifact["measurement_qualification"]
    assert head_major["status"] == "accepted_default"
    assert head_major["performance_claim"] is True
    assert head_major["topline_eligible"] is True

    correctness = artifact["correctness"]
    assert correctness["paro"]["passed"] is True
    assert correctness["gguf_512_64k"]["passed"] is True
    assert correctness["gguf_512_64k"]["all_ids_are_9707"] is True
    assert correctness["gguf_512_64k"][
        "max_prefill_stdev_over_median_percent"
    ] < 5.0
    assert correctness["gguf_512_64k"][
        "max_decode_stdev_over_median_percent"
    ] < 5.0
    blocked = artifact["gguf_128k_blocker"]
    assert blocked["status"] == "blocked_no_current_topline"
    assert blocked["process_exit_code"] == 124
    assert len(blocked["completed_before_stall"]) == 2
    assert blocked["post_timeout_kfd_clients"] == 0

    comparison = artifact["comparison_to_previous_iommu_on_publication"]
    assert comparison["hipengine_combined"]["prefill"]["eligible_cells"] == 11
    assert comparison["hipengine_combined"]["prefill"][
        "arithmetic_mean_delta_percent"
    ] == 4.604194241819479
    assert comparison["hipengine_combined"]["decode"][
        "arithmetic_mean_delta_percent"
    ] == 6.201083397123748

    workloads = ["512/128", "1K/128", "4K/128", "32K/128", "64K/128", "128K/128"]
    topline = artifact["current_topline"]
    table_header = (
        "| Workload | hipEngine PARO | hipEngine GGUF | "
        "llama.cpp HIP | llama.cpp Vulkan |"
    )
    assert table_header in canonical

    metrics = {
        "prefill_tok_s": "Prefill tok/s",
        "decode_tok_s": "Decode tok/s",
        "tracked_peak_allocated_gib": "Peak memory GiB",
    }
    for metric, heading in metrics.items():
        assert f"#### {heading}" in canonical
        for workload in workloads:
            paro_value = topline["paro"][workload][metric]["median"]
            if metric == "prefill_tok_s":
                llama_key = "prefill_tok_s"
            elif metric == "decode_tok_s":
                llama_key = "decode_tok_s"
            else:
                llama_key = "peak_vram_gib"
            llama_hip = topline["llamacpp_hip"][workload][llama_key]
            llama_vulkan = topline["llamacpp_vulkan"][workload][llama_key]
            gguf = topline["gguf"].get(workload)
            if gguf is None:
                assert workload == "128K/128"
                row = (
                    f"| {workload} | {paro_value:.3f} | — (blocked) | "
                    f"{llama_hip:.3f} | {llama_vulkan:.3f} |"
                )
            else:
                head_major_row = head_major["end_to_end"].get(workload)
                if head_major_row is None:
                    gguf_value = gguf[metric]["median"]
                elif metric == "prefill_tok_s":
                    gguf_value = head_major_row["candidate_prefill_tok_s_median"]
                elif metric == "decode_tok_s":
                    gguf_value = head_major_row["candidate_decode_tok_s"]
                else:
                    gguf_value = head_major_row["candidate_peak_gib"]
                row = (
                    f"| {workload} | {paro_value:.3f} | {gguf_value:.3f} | "
                    f"{llama_hip:.3f} | {llama_vulkan:.3f} |"
                )
            assert row in canonical_values

    assert "**+4.60% prefill / +6.20% decode**" in canonical
    assert "same-commit reboot A/B" in canonical
    assert "128K/128 | 498.101 | — (blocked)" in canonical
    assert artifact_path.name in canonical
    assert head_major_path.name in canonical

    for source in artifact["source_artifacts"].values():
        if source is None:
            continue
        if "sha256" in source:
            assert len(source["sha256"]) == 64
        else:
            assert all(len(component["sha256"]) == 64 for component in source.values())


def test_gfx1100_mtp_topline_publishes_graph_ar_correction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = repo_root / "benchmarks/results"
    exact_artifact = json.loads(
        (results / "2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json").read_text(
            encoding="utf-8"
        )
    )
    native = json.loads(
        (results / "2026-07-19-w7900-llama-compat-reusable-native-cycle.json").read_text(
            encoding="utf-8"
        )
    )
    llamacpp = json.loads(
        (results / "2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json").read_text(
            encoding="utf-8"
        )
    )
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")

    assert exact_artifact["status"] == "accepted"
    assert native["status"] == "retained"
    assert native["speed_claim_eligible"] is True
    assert native["correctness"]["full_suite_semantic_oracle"]["exact_output_ids"] is True
    assert llamacpp["performance_claim"] is False

    exact = exact_artifact["mtp_reclassification"]["exact_default_fixed_10_cycles"]
    compat = native["results"]["conservative_r2"]
    llama_results = llamacpp["results"]
    expected_main_rows = [
        (
            f"| Decode | **{exact['true_ar_tok_s']:.2f} tok/s fixed / "
            f"{compat['full']['true_ar_tok_s']:.2f} tok/s natural24** | "
            f"{exact['mtp_tok_s']:.2f} tok/s | "
            f"**{compat['full']['mtp_tok_s']:.2f} tok/s** | "
            f"{llama_results['base']['transition_normalized_tok_s']:.2f} tok/s "
            f"transition-normalized | "
            f"{llama_results['mtp']['transition_normalized_tok_s']:.2f} tok/s "
            f"transition-normalized |"
        ),
        (
            f"| MTP / own AR | 1.0000x | **{exact['mtp_vs_ar']:.4f}x** | "
            f"**{compat['full']['mtp_vs_true_ar']:.4f}x** | n/a | "
            f"**{llama_results['transition_speedup']:.4f}x** |"
        ),
    ]
    expected_split_rows = [
        (
            f"| {label} | {row['prompts']} | {row['true_ar_tok_s']:.2f} | "
            f"**{row['mtp_tok_s']:.2f}** | **{row['mtp_vs_true_ar']:.4f}x** | "
            f"{'**' if label in {'Train', 'Heldout'} else ''}"
            f"{100 * row['draft_acceptance']:.2f}%"
            f"{'**' if label in {'Train', 'Heldout'} else ''} | "
            f"{100 * row['accepted_per_output']:.2f}% | "
            f"{row['cycle_wall_ms_per_output']:.3f} ms |"
        )
        for label, row in {
            "Full": compat["full"],
            "Train": compat["train"],
            "Heldout": compat["heldout"],
            "`code`": compat["categories"]["code"],
            "`general_en`": compat["categories"]["general_en"],
            "`general_ja`": compat["categories"]["general_ja"],
            "`mixed_ja_en`": compat["categories"]["mixed_ja_en"],
        }.items()
    ]
    header = (
        "| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | "
        "hipEngine GGUF `llama-compat` | llama.cpp HIP base AR |"
    )
    assert "#### GGUF MTP comparison, Radeon Pro W7900/gfx1100" in canonical
    assert header in canonical
    assert "##### W7900 reusable-native `llama-compat` full-suite gate" in canonical
    assert "**6.26% faster** than llama.cpp" in canonical
    for expected in expected_main_rows:
        assert expected in canonical
    for expected in expected_split_rows:
        assert expected in canonical


def test_gfx1151_mtp_topline_separates_exact_compat_and_llamacpp() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = repo_root / "benchmarks/results"
    prior_path = results / "2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json"
    transfer_path = results / "2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    canonical = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")

    assert prior["status"] == "accepted_compat_with_exact_negative_and_external_diagnostic"
    assert prior["correctness"]["teacher_forced_all_steps_passed"] is True
    assert prior["correctness"]["production_exact_external_match"] is True
    assert transfer["status"] == "retained"
    assert transfer["performance_claim"] is True
    assert transfer["correctness_claim"] is True
    assert transfer["correctness"]["full_suite"]["output_ids_compared"] == 240
    assert transfer["correctness"]["full_suite"]["cycles_compared"] == 97
    assert transfer["correctness"]["full_suite"]["cycle_semantics_exact"] is True

    exact = prior["hipengine_exact_default"]
    llamacpp = prior["llamacpp_hip"]
    exact_b5 = exact["budgets"]["b5"]
    n3 = transfer["results"]["n3_complete_cycle"]
    full = transfer["n3_splits"]["full"]
    llama_native = llamacpp["native_reported"]
    llama_transition = llamacpp["transition_normalized"]
    assert exact["mtp_beats_ar"] is False
    assert full["mtp_vs_true_ar"] > 1.0
    assert transfer["n3_splits"]["heldout"]["mtp_vs_true_ar"] > 1.0
    assert all(
        row["mtp_vs_true_ar"] > 1.0
        for row in transfer["n3_splits"]["categories"].values()
    )
    assert llamacpp["performance_claim"] is False

    expected_rows = [
        (
            f"| Canonical/native MTP decode | "
            f"{exact_b5['decode_tok_s_weighted']:.2f} tok/s "
            f"({exact_b5['vs_ar_ratio']:.4f}x own AR) | "
            f"**{n3['mtp_tok_s']:.2f} tok/s "
            f"({n3['mtp_vs_true_ar']:.4f}x own AR)** | "
            f"{llama_native['mtp_decode_tok_s']:.2f} tok/s native "
            f"({llama_native['mtp_vs_base']:.4f}x own AR; not cross-engine comparable) |"
        ),
        (
            f"| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | "
            f"**{n3['mtp_tok_s']:.2f} tok/s** | "
            f"{llama_transition['mtp_decode_tok_s']:.2f} tok/s |"
        ),
        (
            f"| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | "
            f"**{n3['true_ar_tok_s']:.2f} tok/s** | "
            f"{llama_transition['base_decode_tok_s']:.2f} tok/s |"
        ),
    ]
    split_rows = {
        "Full": full,
        "Train": transfer["n3_splits"]["train"],
        "Heldout": transfer["n3_splits"]["heldout"],
        "`code`": transfer["n3_splits"]["categories"]["code"],
        "`general_en`": transfer["n3_splits"]["categories"]["general_en"],
        "`general_ja`": transfer["n3_splits"]["categories"]["general_ja"],
        "`mixed_ja_en`": transfer["n3_splits"]["categories"]["mixed_ja_en"],
    }
    expected_split_rows = [
        (
            f"| {label} | {row['prompts']} | "
            f"{row['true_ar_tok_s']:.2f} | **{row['mtp_tok_s']:.2f}** | "
            f"**{row['mtp_vs_true_ar']:.4f}x** | "
            f"{'**' if label in {'Train', 'Heldout'} else ''}"
            f"{100 * row['draft_acceptance']:.2f}%"
            f"{'**' if label in {'Train', 'Heldout'} else ''} | "
            f"{100 * row['accepted_per_output']:.2f}% | "
            f"{row['cycle_wall_ms_per_output']:.3f} ms |"
        )
        for label, row in split_rows.items()
    ]
    header = (
        "| Metric | hipEngine GGUF exact/default | "
        "hipEngine GGUF `llama-compat` | llama.cpp HIP |"
    )
    assert header in canonical
    assert "##### gfx1151 NativeSpecCycle N3 `llama-compat` full-suite gate" in canonical
    assert "`performance_claim=false`" in canonical
    assert prior_path.name in canonical
    assert transfer_path.name in canonical
    for expected_row in expected_rows:
        assert expected_row in canonical
    for expected_row in expected_split_rows:
        assert expected_row in canonical


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
