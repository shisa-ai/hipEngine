from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hipengine.benchmark.provenance import (
    collect_artifact_provenance,
    collect_model_identity,
    collect_repo_state,
    validate_artifact_provenance,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@hipengine.invalid")
    _git(repo, "config", "user.name", "hipEngine Tests")
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_repo_state_separates_staged_unstaged_and_untracked_axes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    clean = collect_repo_state(repo)
    assert clean["staged_dirty"] is False
    assert clean["unstaged_dirty"] is False
    assert clean["untracked_dirty"] is False
    assert clean["untracked_count"] == 0
    assert clean["dirty"] is False

    (repo / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
    unstaged = collect_repo_state(repo)
    assert unstaged["staged_dirty"] is False
    assert unstaged["unstaged_dirty"] is True
    assert unstaged["untracked_dirty"] is False

    _git(repo, "add", "tracked.txt")
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    mixed = collect_repo_state(repo)
    assert mixed["staged_dirty"] is True
    assert mixed["unstaged_dirty"] is False
    assert mixed["untracked_dirty"] is True
    assert mixed["untracked_count"] == 1
    assert mixed["dirty"] is True


def test_model_identity_is_content_derived_and_infers_snapshot_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--name" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    model = snapshot / "model.gguf"
    model.write_bytes(b"gguf-model-v1")

    first = collect_model_identity(model)
    assert first["path"] == str(model.resolve())
    assert first["revision"] == "a" * 40
    assert first["fingerprint"]["algorithm"] == "sha256-full-v1"
    assert first["fingerprint"]["size_bytes"] == len(b"gguf-model-v1")

    model.write_bytes(b"gguf-model-v2")
    second = collect_model_identity(model)
    assert second["fingerprint"]["value"] != first["fingerprint"]["value"]


def test_artifact_provenance_resolves_auto_backend_and_validates_schema(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    model = repo / "model.gguf"
    model.write_bytes(b"tiny-model")

    provenance = collect_artifact_provenance(
        repo_root=repo,
        configured_backend="auto",
        detected_arches=("gfx1151",),
        device_name="AMD Radeon 8060S",
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=("python3", "scripts/example.py", "--rows", "2"),
        environment={"HIPENGINE_HIP_ARCH": "gfx1151"},
        build_profile="decode",
        timing_protocol="client_makespan",
        warmups=2,
        repetitions=5,
        rocm_version="7.1-test",
        hipcc_version="hipcc test",
    )

    assert validate_artifact_provenance(provenance, require_model=True) == provenance
    assert provenance["kind"] == "hipengine_artifact_provenance"
    assert provenance["schema_version"] == 1
    assert provenance["configured_backend"] == "auto"
    assert provenance["resolved_backend"] == "hip_gfx1151"
    assert provenance["target_arch"] == "gfx1151"
    assert provenance["device_name"] == "AMD Radeon 8060S"
    assert provenance["model_path"] == str(model.resolve())
    assert provenance["model_revision"] is None
    assert provenance["model_fingerprint"]["value"]
    assert provenance["hipengine_commit"]
    assert provenance["staged_dirty"] is False
    assert provenance["unstaged_dirty"] is False
    assert provenance["untracked_dirty"] is True
    assert provenance["untracked_count"] == 1
    assert provenance["command"] == ["python3", "scripts/example.py", "--rows", "2"]


def test_artifact_provenance_captures_hip_hardware_queue_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "1")

    provenance = collect_artifact_provenance(
        repo_root=repo,
        configured_backend="hip_gfx1151",
        detected_arches=("gfx1151",),
        command=("python3", "bench.py"),
        rocm_version="7.1-test",
        hipcc_version="hipcc test",
    )

    assert provenance["environment"]["GPU_MAX_HW_QUEUES"] == "1"


def test_artifact_provenance_uses_explicit_target_when_device_probe_is_disabled(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)

    provenance = collect_artifact_provenance(
        repo_root=repo,
        configured_backend="auto",
        detected_arches=(),
        target_arch="gfx1151",
        device_name="AMD Radeon 8060S",
        command=("python3", "bench.py"),
        rocm_version="7.1-test",
        hipcc_version="hipcc test",
    )

    assert provenance["resolved_backend"] == "hip_gfx1151"
    assert provenance["target_arch"] == "gfx1151"


def test_artifact_provenance_validation_rejects_selector_or_missing_model(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    provenance = collect_artifact_provenance(
        repo_root=repo,
        configured_backend="cpu_reference",
        resolved_backend="cpu_reference",
        command=("python3", "bench.py"),
        rocm_version=None,
        hipcc_version=None,
    )

    selector = dict(provenance)
    selector["resolved_backend"] = "auto"
    with pytest.raises(ValueError, match="resolved_backend"):
        validate_artifact_provenance(selector)
    with pytest.raises(ValueError, match="model_path"):
        validate_artifact_provenance(provenance, require_model=True)


def test_json_schema_tracks_the_canonical_provenance_contract() -> None:
    schema = json.loads(
        Path("benchmarks/schemas/artifact-provenance.schema.json").read_text(encoding="utf-8")
    )

    assert schema["properties"]["kind"] == {"const": "hipengine_artifact_provenance"}
    assert schema["properties"]["schema_version"] == {"const": 1}
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
