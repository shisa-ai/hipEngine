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


def _assert_local_markdown_links_exist(text: str, base_dir: Path) -> None:
    for target in _LOCAL_LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local_path, _, anchor = target.partition("#")
        path = base_dir / local_path
        assert path.exists(), f"broken README link: {target}"
        if not anchor or not path.is_file():
            continue
        heading_slugs = {
            _heading_slug(match.group(1))
            for line in path.read_text(encoding="utf-8").splitlines()
            if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
        }
        assert anchor in heading_slugs, f"broken README anchor: {target}"


def _run_sync_check(
    tmp_path: Path,
    block_body: str,
    *,
    document_prefix: str = "# Benchmarks\n\n",
) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "source.md"
    target = tmp_path / "target.md"
    document = (
        document_prefix
        + "<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->\n"
        + f"{block_body.rstrip()}\n"
        + "<!-- END TOPLINE:README_HIGHLIGHTS -->\n"
    )
    source.write_text(document, encoding="utf-8")
    target.write_text(document, encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/sync_benchmark_readme.py",
            "--check",
            "--source",
            str(source),
            "--target",
            str(target),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def test_sync_accepts_compact_public_benchmark_summary(tmp_path: Path) -> None:
    result = _run_sync_check(
        tmp_path,
        "### GPU\n\n"
        "| Model | Throughput |\n"
        "| --- | ---: |\n"
        "| Example | **100 tok/s** |\n\n"
        "Compare rows only when their protocols match.",
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_sync_rejects_worklog_style_exported_prose(tmp_path: Path) -> None:
    diary = " ".join(["implementation-detail"] * 45)
    result = _run_sync_check(
        tmp_path,
        f"### GPU\n\n{diary}\n\n{diary}\n\n{diary}",
    )
    assert result.returncode == 2
    assert "public README prose budget" in result.stderr


def test_sync_rejects_oversized_exported_block(tmp_path: Path) -> None:
    rows = "\n".join(f"| model-{index} | {index} |" for index in range(78))
    result = _run_sync_check(
        tmp_path,
        "### GPU\n\n| Model | Throughput |\n| --- | ---: |\n" + rows,
    )
    assert result.returncode == 2
    assert "public README line budget" in result.stderr


def test_sync_rejects_oversized_public_readme(tmp_path: Path) -> None:
    prefix = "# Benchmarks\n\n" + "\n".join(
        f"Public project detail {index}." for index in range(320)
    ) + "\n\n"
    result = _run_sync_check(
        tmp_path,
        "### GPU\n\nBrief user-visible note.",
        document_prefix=prefix,
    )
    assert result.returncode == 2
    assert "public README document budget" in result.stderr


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
    assert len(readme.splitlines()) <= 325
    assert readme.index("## Supported models") < readme.index("## Performance highlights")
    assert readme.index("## Performance highlights") < readme.index("## Status")
    for model_url in (
        "https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed",
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
    assert "**93.644 tok/s public**" in readme
    assert "**214.788**" in readme
    assert "**Current release: v0.4.0 alpha.**" in readme
    assert "NVIDIA Blackwell (`sm_120a`)" in readme
    assert "Current development is\ntherefore focused on GGUF compatibility" in readme
    for internal_phrase in ("source-pinned", "physical c8", "packet reaches"):
        assert internal_phrase not in readme
    _assert_markdown_tables_are_rectangular(readme)
    _assert_local_markdown_links_exist(readme, repo_root)


def test_benchmark_readme_is_a_compact_current_scoreboard() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scoreboard_path = repo_root / "benchmarks/README.md"
    scoreboard = scoreboard_path.read_text(encoding="utf-8")

    assert len(scoreboard.splitlines()) < 460
    assert len(scoreboard.encode("utf-8")) < 30_000
    assert scoreboard.count("<!-- BEGIN TOPLINE:") == 1
    assert scoreboard.count("results/") < 40
    assert "## Platform Index" not in scoreboard
    assert "## Blocked and Diagnostic Benchmark Attempts" not in scoreboard
    assert "## README Sweep Test Procedure" not in scoreboard
    assert "## Maintenance contract" in scoreboard
    assert "It is not an optimization journal." in scoreboard
    assert "git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112" in scoreboard

    artifact_path = (
        repo_root
        / "benchmarks/results/2026-08-06-gfx1151-gguf-sh14-c1-cumulative-completion-gate.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["source"]["tracked_source_clean"] is True
    assert artifact["topline_eligible"] is False
    assert "Current snapshot" in scoreboard
    assert artifact_path.name in scoreboard

    labels = {
        "512/128": "512/128",
        "4K/128": "4K/128",
        "32K/128": "32K/128",
        "64K/128": "64K/128",
    }
    for artifact_label, table_label in labels.items():
        row = artifact["rows"][artifact_label]["hipengine_current"]
        expected = (
            f"| {table_label} | **{row['prefill_tok_s_median']:.3f} tok/s** | "
            f"**{row['decode_tok_s_median']:.3f} tok/s** | "
            f"{row['tracked_peak_gib']:.3f} GiB | "
            f"{row['whole_gtt_peak_gib']:.3f} GiB |"
        )
        assert expected in scoreboard
    assert "| 128K/128 | — | — | — | — |" in scoreboard

    _assert_markdown_tables_are_rectangular(scoreboard)
    _assert_local_markdown_links_exist(scoreboard, scoreboard_path.parent)


def test_qwen38_c1c8_current_rollup_matches_scoreboard() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scoreboard = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")
    artifact = json.loads(
        (
            repo_root
            / "benchmarks/results/2026-09-02-w7900-qwen38-q4km-c1c8-current-scoreboard.json"
        ).read_text(encoding="utf-8")
    )

    headings = {
        "**True AR decode**": "true_ar",
        "**Explicit K3 MTP decode diagnostic**": "explicit_k3",
        "**Prefill**": "prefill",
    }
    labels = {
        "hipEngine": "hipengine",
        "llama.cpp current HIP": "llama_cpp_current_hip",
        "llama.cpp Laurent HIP": "llama_cpp_laurent_hip",
    }
    markers = list(headings)
    for index, marker in enumerate(markers):
        begin = scoreboard.index(marker)
        end = (
            scoreboard.index(markers[index + 1], begin)
            if index + 1 < len(markers)
            else scoreboard.index("The ten-prompt suite", begin)
        )
        section = scoreboard[begin:end]
        for label, role in labels.items():
            line = next(
                row for row in section.splitlines() if row.startswith(f"| {label} |")
            )
            cells = [cell.strip().strip("*") for cell in line.strip("|").split("|")[1:]]
            assert [float(cell) for cell in cells] == artifact["rows_tok_s"][headings[marker]][role]

    assert artifact["performance_claim"] is False
    for source in artifact["source_artifacts"]:
        assert (repo_root / source["path"]).is_file()
    assert artifact["rows_tok_s"]["explicit_k3_divided_by_published_true_ar"] == [
        round(mtp / ar, 4)
        for mtp, ar in zip(
            artifact["rows_tok_s"]["explicit_k3"]["hipengine"],
            artifact["rows_tok_s"]["true_ar"]["hipengine"],
        )
    ]


def test_compact_mtp_scoreboard_uses_true_ar_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = repo_root / "benchmarks/results"
    scoreboard = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")

    dense_path = results / "2026-08-23-w7900-qwen36-27b-current-default-publication.json"
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    assert dense["performance_claim"] is True
    assert dense["performance_claim_class"] == "current_snapshot"
    assert dense["correctness"]["passed"] is True
    dense_natural = dense["natural25"]
    assert dense_natural["correctness"]["all_exact_greedy"] is True
    assert dense_natural["correctness"]["all_gpu_accept_match_cpu"] is True
    dense_ar = dense_natural["summary"]["true_ar"]["full"]
    dense_b3 = dense_natural["summary"]["mtp"]["3"]["full"]
    expected_dense = (
        "| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Exact/default natural25 B3 | "
        f"{dense_ar['decode_tok_s_weighted']:.3f} | "
        f"**{dense_b3['decode_tok_s_weighted']:.3f}** | "
        f"**{dense_b3['mtp_vs_true_ar']:.4f}x** |"
    )
    assert expected_dense in scoreboard
    assert dense_path.name in scoreboard

    w7900_path = results / "2026-07-19-w7900-llama-compat-reusable-native-cycle.json"
    w7900 = json.loads(w7900_path.read_text(encoding="utf-8"))
    w7900_full = w7900["results"]["conservative_r2"]["full"]
    expected_w7900 = (
        "| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | "
        "`llama-compat` MTP-2 natural suite | "
        f"{w7900_full['true_ar_tok_s']:.2f} | **{w7900_full['mtp_tok_s']:.2f}** | "
        f"**{w7900_full['mtp_vs_true_ar']:.4f}x** |"
    )
    assert expected_w7900 in scoreboard
    assert w7900["status"] == "retained"
    assert w7900["speed_claim_eligible"] is True
    assert w7900["correctness"]["full_suite_semantic_oracle"]["exact_output_ids"] is True
    assert w7900_path.name in scoreboard

    gfx1151_path = results / "2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json"
    gfx1151 = json.loads(gfx1151_path.read_text(encoding="utf-8"))
    gfx1151_n3 = gfx1151["results"]["n3_complete_cycle"]
    expected_gfx1151 = (
        "| Radeon 8060S / Qwen3.6-35B-A3B `UD-Q4_K_M` | "
        "`llama-compat` MTP-2 natural suite | "
        f"{gfx1151_n3['true_ar_tok_s']:.2f} | **{gfx1151_n3['mtp_tok_s']:.2f}** | "
        f"**{gfx1151_n3['mtp_vs_true_ar']:.4f}x** |"
    )
    assert expected_gfx1151 in scoreboard
    assert gfx1151["status"] == "retained"
    assert gfx1151["performance_claim"] is True
    assert gfx1151["correctness_claim"] is True
    assert gfx1151["correctness"]["full_suite"]["output_ids_compared"] == 240
    assert gfx1151["correctness"]["full_suite"]["cycles_compared"] == 97
    assert gfx1151["correctness"]["full_suite"]["cycle_semantics_exact"] is True
    assert gfx1151["n3_splits"]["heldout"]["mtp_vs_true_ar"] > 1.0
    assert all(
        row["mtp_vs_true_ar"] > 1.0
        for row in gfx1151["n3_splits"]["categories"].values()
    )
    assert gfx1151_path.name in scoreboard

    assert "full-suite gate" not in scoreboard
    assert "2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json" not in scoreboard
    assert "Verifier\n`off`/`B0` diagnostics are not speedup denominators." in scoreboard


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


def test_gfx1151_legacy_paro_diagnostic_is_not_published() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_path = (
        repo_root
        / "benchmarks/results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    scoreboard = (repo_root / "benchmarks/README.md").read_text(encoding="utf-8")

    assert artifact["performance_claim"] is False
    assert artifact_path.name not in scoreboard
    assert "No eligible native-batch timing row" not in scoreboard
    assert "[`HISTORY.md`](HISTORY.md)" in scoreboard

    for rows in range(1, 9):
        result = artifact["rows"][str(rows)]
        expected = (
            f"| {rows} | {result['decode_tok_s_aggregate_median']:.3f} | "
            f"{result['decode_tok_s_per_request_median']:.3f} | "
            f"{result['decode_step_ms_median_of_run_medians']:.3f} |"
        )
        assert expected not in scoreboard
        assert result["status"].startswith("diagnostic_")
        if rows > 1:
            assert result["generated_token_equality"] is True
            assert result["primitive_correctness"] is True

    assert artifact["native_batch_width_profile"]["native_widths"] == list(range(2, 9))
    assert artifact["native_batch_width_profile"]["routing_eligible"] is False
    assert artifact["native_batch_width_profile"]["oracle_status"] == (
        "invalid_batch_shaped_c1_reference"
    )
