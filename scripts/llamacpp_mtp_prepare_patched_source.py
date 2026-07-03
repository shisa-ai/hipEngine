#!/usr/bin/env python3
"""Prepare a temporary patched llama.cpp source tree for MTP checkpoint capture.

The external llama.cpp checkout remains read-only.  This helper uses `git archive`
from the reference checkout, extracts it into an isolated output directory, applies
the committed Qwen35MoE layer-input patch artifact, and writes a compact manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_layer_input_patch import (
    LAYER_INPUT_ASSIGNMENT,
    REFERENCE_COMMIT,
    TARGET_RELATIVE_PATH,
)

DEFAULT_LLAMA_CPP_ROOT = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_PATCH = Path(
    "benchmarks/results/mtp-gguf-iter284-llamacpp-qwen35moe-layer-input.patch"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/hipengine-llamacpp-mtp-iter285-qwen35moe-layer-input"
)
DEFAULT_MANIFEST = Path(
    "benchmarks/results/mtp-gguf-iter285-llamacpp-patched-source-manifest.json"
)
SENTINEL = ".hipengine_llamacpp_mtp_temp_source"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--iteration", type=int, default=285)
    args = parser.parse_args()

    manifest = prepare_patched_source(
        llamacpp_root=args.llamacpp_root,
        patch_path=args.patch,
        output_dir=args.output_dir,
        replace=args.replace,
        iteration=args.iteration,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "commit": manifest["reference_basis"]["observed_commit"],
                "output_dir": manifest["output_dir"],
                "assignment_line": manifest["target_validation"]["assignment_line"],
                "next_action": manifest["next_action"],
            },
            indent=2,
        )
    )


def prepare_patched_source(
    *,
    llamacpp_root: Path,
    patch_path: Path,
    output_dir: Path,
    replace: bool = False,
    iteration: int = 285,
) -> dict[str, Any]:
    root = llamacpp_root.resolve()
    patch = patch_path.resolve()
    out = output_dir.resolve()
    before_status = _git_status_target(root)
    commit = _git_output(root, "rev-parse", "HEAD")
    patch_sha = sha256_file(patch)

    ensure_output_dir_ready(out, replace=replace)
    _extract_git_archive(root, out)
    patch_result = _apply_patch(out, patch)
    _write_sentinel(out, root=root, commit=commit, patch_sha=patch_sha)

    target = out / TARGET_RELATIVE_PATH
    validation = validate_patched_target(target)
    after_status = _git_status_target(root)
    return {
        "schema": 1,
        "kind": "llamacpp_patched_source_manifest",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "prepared" if validation["patch_applied"] else "invalid",
        "llamacpp_root": str(root),
        "output_dir": str(out),
        "target_relative_path": str(TARGET_RELATIVE_PATH),
        "reference_basis": {
            "expected_commit": REFERENCE_COMMIT,
            "observed_commit": commit,
            "commit_matches_expected": commit == REFERENCE_COMMIT,
            "source_is_read_only_reference": True,
            "external_checkout_modified": before_status != after_status,
            "target_status_before": before_status,
            "target_status_after": after_status,
        },
        "patch": {
            "path": str(patch),
            "sha256": patch_sha,
            "bytes": patch.stat().st_size,
            "apply_stdout_tail": tail_text(patch_result.stdout),
            "apply_stderr_tail": tail_text(patch_result.stderr),
        },
        "target_validation": validation,
        "output_tree": summarize_output_tree(out),
        "next_action": "build_temp_llamacpp_and_capture_hidden_in_with_layer_input_api",
    }


def ensure_output_dir_ready(output_dir: Path, *, replace: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not replace:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    sentinel = output_dir / SENTINEL
    if not sentinel.exists():
        raise FileExistsError(
            f"refusing to replace output directory without {SENTINEL}: {output_dir}"
        )
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def validate_patched_target(target: Path) -> dict[str, Any]:
    text = target.read_text() if target.exists() else ""
    assignment_count = text.count(LAYER_INPUT_ASSIGNMENT)
    return {
        "target_exists": target.exists(),
        "assignment_count": assignment_count,
        "assignment_line": find_line(text, LAYER_INPUT_ASSIGNMENT),
        "patch_applied": target.exists() and assignment_count == 1,
        "anchor_after_assignment_present": (
            LAYER_INPUT_ASSIGNMENT + "\n        ggml_tensor * inpSA = inpL;"
        )
        in text,
    }


def summarize_output_tree(output_dir: Path) -> dict[str, Any]:
    file_count = 0
    dir_count = 0
    for path in output_dir.rglob("*"):
        if path.is_file():
            file_count += 1
        elif path.is_dir():
            dir_count += 1
    return {
        "sentinel": str(output_dir / SENTINEL),
        "sentinel_exists": (output_dir / SENTINEL).exists(),
        "file_count": file_count,
        "dir_count": dir_count,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def tail_text(text: str, *, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _extract_git_archive(root: Path, output_dir: Path) -> None:
    archive = subprocess.Popen(
        ["git", "-C", str(root), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar = subprocess.run(
        ["tar", "-x", "-C", str(output_dir)],
        stdin=archive.stdout,
        capture_output=True,
        text=False,
        check=False,
    )
    if archive.stdout is not None:
        archive.stdout.close()
    _, archive_stderr = archive.communicate()
    if archive.returncode or tar.returncode:
        raise RuntimeError(
            "git archive extraction failed: "
            f"git_rc={archive.returncode} tar_rc={tar.returncode} "
            f"git_stderr={archive_stderr.decode(errors='replace')} "
            f"tar_stderr={tar.stderr.decode(errors='replace')}"
        )


def _apply_patch(output_dir: Path, patch_path: Path) -> subprocess.CompletedProcess[str]:
    patch_text = patch_path.read_text()
    result = subprocess.run(
        ["patch", "-p1"],
        cwd=output_dir,
        input=patch_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "patch apply failed: "
            f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}"
        )
    return result


def _write_sentinel(output_dir: Path, *, root: Path, commit: str, patch_sha: str) -> None:
    (output_dir / SENTINEL).write_text(
        json.dumps(
            {
                "kind": "hipengine_llamacpp_mtp_temp_source",
                "root": str(root),
                "commit": commit,
                "patch_sha256": patch_sha,
            },
            indent=2,
        )
        + "\n"
    )


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def _git_status_target(root: Path) -> str:
    return _git_output(root, "status", "--porcelain", "--", str(TARGET_RELATIVE_PATH))


if __name__ == "__main__":
    main()
