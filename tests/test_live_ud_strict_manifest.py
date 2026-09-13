"""The artifact-scoped strict manifest names real owners and matches its artifact.

The U6 certification artifact claims rates and exactness for a named GGUF file.
This test holds the manifest that scopes that claim to account:

* every manifest validates under the same ``validate_variant_manifest`` the
  execution-profile gate uses, so the artifact cannot carry a manifest the gate
  would reject;
* every named variant resolves to a real registry key, so the manifest cannot
  describe a path that does not exist; and
* the manifest's declared GGML types and in-scope record count are re-derived
  here from the artifact's own GGUF bytes and the admission table, so a stale
  manifest fails instead of quietly scoping the wrong thing.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "benchmarks" / "results"
CERTIFICATION = RESULTS / "ud-mtp-certification-u6.json"
ARTIFACTS = ("ud-q4-k-m", "ud-q4-k-s")


def _require_hip() -> None:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:  # pragma: no cover - no-ROCm CI
        pytest.skip("HIP runtime is not available")


def _manifest_path(artifact: str) -> pathlib.Path:
    return RESULTS / f"{artifact}.strict-manifest.json"


@pytest.fixture(scope="module")
def built():
    _require_hip()
    from scripts.ud_strict_manifest import _register_kernels, build

    _register_kernels()
    return {artifact: build(artifact) for artifact in ARTIFACTS}


def test_manifest_files_exist_and_match_a_fresh_build(built):
    """A checked-in manifest must equal what the builder produces now."""

    for artifact in ARTIFACTS:
        path = _manifest_path(artifact)
        assert path.exists(), f"{path} is missing; run scripts/ud_strict_manifest.py"
        on_disk = json.loads(path.read_text())
        assert on_disk == built[artifact], (
            f"{path} is stale relative to the artifact's GGUF bytes or the "
            "admission table; re-run scripts/ud_strict_manifest.py"
        )


def test_every_manifest_validates_under_the_gate_validator(built):
    from hipengine.execution_profiles import manifest_sha256, validate_variant_manifest

    for artifact, record in built.items():
        assert record["strict_manifests"], artifact
        for entry in record["strict_manifests"]:
            manifest = entry["strict_manifest"]
            normalized = validate_variant_manifest(manifest)
            assert normalized == manifest, artifact
            assert manifest["execution_profile"] == "strict"
            assert manifest["backend"] == "hip_gfx1100"
            assert manifest_sha256(manifest) == entry["strict_manifest_sha256"]
            assert manifest["selections"], (artifact, entry["registry_quant"])


def test_every_named_variant_resolves_to_a_registry_key(built):
    from hipengine.kernels.registry import KernelKey, registered_keys
    from scripts.ud_strict_manifest import _register_kernels

    # The registry is process-global and other tests clear it, so register here
    # rather than relying on fixture ordering.
    _register_kernels()
    registered = set(registered_keys())
    checked = 0
    for artifact, record in built.items():
        for entry in record["strict_manifests"]:
            for selection in entry["strict_manifest"]["selections"]:
                key = KernelKey(
                    "hip_gfx1100",
                    selection["layer"],
                    selection["registry_quant"],
                    selection["selected_variant"],
                )
                assert key in registered, f"{artifact}: {key.display()} is not registered"
                assert (
                    selection["strict_fallback_variant"]
                    == selection["selected_variant"]
                ), "a strict selection is its own fallback"
                checked += 1
    assert checked > 200, f"too few selections verified: {checked}"


def test_scope_is_rederived_from_the_artifact_bytes(built):
    """The manifest's scope must follow from the GGUF file, not from itself."""

    from scripts.ud_strict_manifest import (
        _artifacts,
        _present_ggml_types,
        _scoped_records,
    )

    table = _artifacts()
    for artifact, record in built.items():
        path, quant_identity = table[artifact]
        present, tensor_count = _present_ggml_types(path)
        assert record["gguf_path"] == str(path)
        assert record["quant_identity"] == quant_identity
        assert record["scope"]["ggml_types_present"] == sorted(present)
        assert record["scope"]["tensors"] == tensor_count

        registry_records, consumer_records = _scoped_records(present)
        assert (
            record["scope"]["coverage_records_named_by_registry_key"]
            == len(registry_records)
        )
        assert (
            record["scope"]["coverage_records_named_by_consumer_module"]
            == len(consumer_records)
        )
        assert record["scope"]["coverage_records_in_scope"] == len(
            registry_records
        ) + len(consumer_records)


def test_certification_carries_the_manifest_bundle(built):
    cert = json.loads(CERTIFICATION.read_text())
    entries = {str(entry["artifact"]): entry for entry in cert["artifacts"]}
    for artifact, record in built.items():
        entry = entries[artifact]
        assert (
            entry["strict_manifest_bundle_sha256"]
            == record["strict_manifest_bundle_sha256"]
        ), f"{artifact}: certification carries a stale manifest hash"
        assert entry["strict_manifests"] == record["strict_manifests"]
        assert entry["strict_manifest_scope"] == record["scope"]
