from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import hipengine.benchmark.provenance as provenance_module
from hipengine.benchmark.provenance import (
    _detect_rocm_version,
    collect_artifact_provenance,
    collect_model_identity,
    collect_repo_state,
    is_execution_affecting,
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


def test_repo_state_names_the_dirty_paths_it_counts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    clean = collect_repo_state(repo)
    assert clean["dirty_paths"] == []
    assert clean["dirty_path_count"] == 0
    assert clean["untracked_paths"] == []
    assert clean["execution_affecting_dirty"] is False
    assert clean["execution_affecting_paths"] == []

    (repo / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    dirty = collect_repo_state(repo)
    # A bare boolean leaves a measurement unreproducible; the artifact has to
    # say which file was modified.
    assert dirty["dirty_paths"] == ["tracked.txt"]
    assert dirty["dirty_path_count"] == 1
    assert dirty["untracked_paths"] == ["untracked.txt"]
    assert dirty["execution_affecting_paths"] == ["tracked.txt", "untracked.txt"]


def test_repo_state_excuses_a_worktree_dirty_only_in_prose_trees(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("scratch\n", encoding="utf-8")
    (repo / "worklog" / "entries").mkdir(parents=True)
    (repo / "worklog" / "entries" / "e.md").write_text("entry\n", encoding="utf-8")

    state = collect_repo_state(repo)
    assert state["dirty"] is True
    assert state["untracked_count"] == 2
    # Documentation cannot change dispatch or arithmetic, so another agent's
    # scratch notes must not invalidate a measurement.
    assert state["execution_affecting_dirty"] is False
    assert state["execution_affecting_paths"] == []

    (repo / "kernel.hip").write_text("__global__ void k() {}\n", encoding="utf-8")
    affected = collect_repo_state(repo)
    assert affected["execution_affecting_dirty"] is True
    assert affected["execution_affecting_paths"] == ["kernel.hip"]


def test_execution_affecting_classification_defaults_to_conservative() -> None:
    for inert in (
        "docs/PLAN.md",
        "docs/superpowers/plans/x.md",
        "worklog/entries/e.md",
        "benchmarks/results/run.json",
        "CLAUDE.md",
        "README.md",
    ):
        assert is_execution_affecting(inert) is False, inert
    for affecting in (
        "hipengine/runtime/gguf_linear.py",
        "kernels/hip_gfx1151/q8.hip",
        "scripts/gate.py",
        "benchmarks/fixtures/canonical.json",
        "pyproject.toml",
        # An unrecognised path is assumed to matter rather than assumed inert.
        "unknown_tree/thing.bin",
    ):
        assert is_execution_affecting(affecting) is True, affecting


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


def test_rocm_version_prefers_active_hipcc_over_host_opt_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: "host-rocm-7.2.4")

    assert _detect_rocm_version(
        "HIP version: 7.15.0-0000000\nAMD clang version 23.0.0git"
    ) == "HIP version: 7.15.0-0000000"


def test_artifact_provenance_resolves_auto_backend_and_validates_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_HIP_ARCH", raising=False)
    repo = _repo(tmp_path)
    monkeypatch.setattr(provenance_module.socket, "gethostname", lambda: "zbook-test")
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
    assert provenance["schema_version"] == 3
    assert provenance["host_name"] == "zbook-test"
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
        device_name="AMD Radeon 8060S",
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
    assert schema["properties"]["schema_version"] == {"enum": [1, 2, 3]}
    assert "host_name" in schema["properties"]
    assert "host_name" not in schema["required"]
    assert schema["allOf"] == [
        {
            "if": {"properties": {"schema_version": {"minimum": 2}}},
            "then": {"required": ["host_name"]},
        },
        {
            "if": {"properties": {"schema_version": {"const": 3}}},
            "then": {
                "required": [
                    "dirty_paths",
                    "dirty_path_count",
                    "untracked_paths",
                    "execution_affecting_dirty",
                    "execution_affecting_paths",
                ]
            },
        },
    ]
    assert schema["additionalProperties"] is False
