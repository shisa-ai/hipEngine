from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.llamacpp_mtp_layer_input_patch import (
    LAYER_INPUT_ASSIGNMENT,
    build_patch_artifact,
)
from scripts.llamacpp_mtp_prepare_patched_source import (
    SENTINEL,
    ensure_output_dir_ready,
    prepare_patched_source,
    sha256_file,
    validate_patched_target,
)
from tests.test_llamacpp_mtp_layer_input_patch import _qwen35moe_source


def test_prepare_patched_source_from_git_archive(tmp_path: Path) -> None:
    root = _make_tiny_llamacpp_git_repo(tmp_path)
    patch_path = tmp_path / "layer-input.patch"
    build_patch_artifact(llamacpp_root=root, patch_output=patch_path)
    output_dir = tmp_path / "patched-source"

    manifest = prepare_patched_source(
        llamacpp_root=root,
        patch_path=patch_path,
        output_dir=output_dir,
    )

    target = output_dir / "src" / "models" / "qwen35moe.cpp"
    assert manifest["status"] == "prepared"
    assert manifest["reference_basis"]["external_checkout_modified"] is False
    assert manifest["patch"]["sha256"] == sha256_file(patch_path)
    assert manifest["target_validation"]["patch_applied"] is True
    assert manifest["target_validation"]["assignment_line"] == 4
    assert (output_dir / SENTINEL).exists()
    assert LAYER_INPUT_ASSIGNMENT in target.read_text()
    json.dumps(manifest)


def test_replace_requires_owned_sentinel(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "not-ours.txt").write_text("keep me")

    with pytest.raises(FileExistsError, match="without"):
        ensure_output_dir_ready(output_dir, replace=True)


def test_replace_owned_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / SENTINEL).write_text("{}\n")
    (output_dir / "old.txt").write_text("remove me")

    ensure_output_dir_ready(output_dir, replace=True)

    assert output_dir.exists()
    assert not (output_dir / "old.txt").exists()


def test_validate_patched_target_counts_assignment(tmp_path: Path) -> None:
    target = tmp_path / "qwen35moe.cpp"
    target.write_text(_qwen35moe_source(wired=True))

    validation = validate_patched_target(target)

    assert validation["target_exists"] is True
    assert validation["assignment_count"] == 1
    assert validation["patch_applied"] is True
    assert validation["anchor_after_assignment_present"] is True


def _make_tiny_llamacpp_git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp-hip"
    target = root / "src" / "models" / "qwen35moe.cpp"
    target.parent.mkdir(parents=True)
    target.write_text(_qwen35moe_source(wired=False))
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "src/models/qwen35moe.cpp"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=hipEngine Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "seed qwen35moe source",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root
