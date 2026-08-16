from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hipengine.models.kv_capabilities import (
    KVCapabilityKey,
    ModelArtifactIdentity,
    model_artifact_identity,
)
from hipengine.models.qwen35 import Qwen35GGUFModel


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PASS_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
_REJECT_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"


def _key(
    *,
    sha256: str,
    size_bytes: int,
    backend: str,
    target_arch: str | None = None,
    scale_dtype: str = "fp32",
) -> KVCapabilityKey:
    return KVCapabilityKey(
        artifact_sha256=sha256,
        artifact_size_bytes=size_bytes,
        backend=backend,
        target_arch=target_arch or backend.removeprefix("hip_"),
        weight_quant="gguf_q4_k_m",
        kv_storage="int8_per_token_head",
        storage_layout="uniform",
        scale_dtype=scale_dtype,
        scale_granularity="per_token_head",
    )


def _artifact(*, sha256: str, size_bytes: int) -> ModelArtifactIdentity:
    return ModelArtifactIdentity(
        path="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        size_bytes=size_bytes,
        sha256=sha256,
        content_verified=True,
    )


def test_registered_capabilities_match_retained_artifact_model_identities() -> None:
    passing_path = (
        _REPO_ROOT
        / "benchmarks/results/2026-08-16-qwen38-27b-actual-context-quality-w7900.json"
    )
    rejected_path = (
        _REPO_ROOT
        / "benchmarks/results/2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json"
    )
    passing_artifact = json.loads(passing_path.read_text())
    rejected_artifact = json.loads(rejected_path.read_text())
    passing = passing_artifact["model"]
    rejected = rejected_artifact["model"]

    assert (passing["size_bytes"], passing["sha256"], passing["quant"]) == (
        17_106_773_984,
        _PASS_SHA256,
        "gguf_q4_k_m",
    )
    assert (rejected["size_bytes"], rejected["sha256"], rejected["quant"]) == (
        17_106_775_008,
        _REJECT_SHA256,
        "gguf_q4_k_m",
    )
    assert (
        passing_artifact["source"]["backend"],
        passing_artifact["source"]["target_arch"],
        passing_artifact["kv"]["storage"],
        passing_artifact["kv"]["scale_dtype"],
        passing_artifact["kv"]["scale_granularity"],
    ) == (
        "hip_gfx1100",
        "gfx1100",
        "int8_per_token_head",
        "fp32",
        "per_token_head",
    )
    assert (
        rejected_artifact["software"]["backend"],
        rejected_artifact["hardware"]["arch"],
        rejected_artifact["protocol"]["native_invocations"][0]["storage"],
        rejected_artifact["protocol"]["native_invocations"][0]["scale_dtype"],
    ) == ("hip_gfx1151", "gfx1151", "int8_per_token_head", "fp32")


def test_qwen38_gfx1100_exact_artifact_int8_capability_is_qualified() -> None:
    plugin = Qwen35GGUFModel()
    resolution = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )

    payload = resolution.as_dict()
    assert payload["capability_id"] == (
        "705a637209c4d2ecbad20934eec5770287e992b2fb2523dded69a2b0487ba778"
    )
    assert payload["status"] == "qualified"
    assert payload["runtime_action"] == "admit"
    assert payload["promotion_eligible"] is True
    assert payload["effective_kv_storage"] == "int8_per_token_head"
    assert payload["evidence"]["max_direct_rows"] == 1
    assert payload["evidence"]["max_serial_resident_rows"] == 4
    assert payload["evidence"]["persistent_bf16_mirror"] is False
    assert payload["evidence"]["quality_artifact"].endswith(
        "2026-08-16-qwen38-27b-actual-context-quality-w7900.json"
    )


def test_qwen38_gfx1151_exact_artifact_int8_capability_remains_rejected() -> None:
    plugin = Qwen35GGUFModel()
    resolution = plugin.resolve_kv_capability(
        key=_key(
            sha256=_REJECT_SHA256,
            size_bytes=17_106_775_008,
            backend="hip_gfx1151",
        ),
        artifact=_artifact(sha256=_REJECT_SHA256, size_bytes=17_106_775_008),
    )

    payload = resolution.as_dict()
    assert payload["status"] == "rejected"
    assert payload["runtime_action"] == "fallback_bf16"
    assert payload["promotion_eligible"] is False
    assert payload["effective_kv_storage"] == "bf16"
    assert "0.7778" in payload["reason"]


def test_same_filename_or_geometry_does_not_admit_unknown_artifact_or_scale() -> None:
    plugin = Qwen35GGUFModel()
    unknown_sha = "f" * 64
    unknown = plugin.resolve_kv_capability(
        key=_key(
            sha256=unknown_sha,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
        ),
        artifact=_artifact(sha256=unknown_sha, size_bytes=17_106_773_984),
    )
    wrong_scale = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            scale_dtype="fp16",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )
    wrong_target = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            target_arch="gfx1151",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )

    assert unknown.status == "unknown"
    assert unknown.effective_kv_storage == "bf16"
    assert unknown.promotion_eligible is False
    assert wrong_scale.status == "unknown"
    assert "contract is unqualified" in wrong_scale.reason
    assert wrong_target.status == "unknown"


def test_model_artifact_identity_hashes_content_and_invalidates_on_change(tmp_path: Path) -> None:
    path = tmp_path / "same-name.gguf"
    path.write_bytes(b"first-artifact")

    first = model_artifact_identity(path)
    repeated = model_artifact_identity(path)
    assert first == repeated
    assert first.content_verified is True
    assert first.sha256 == hashlib.sha256(b"first-artifact").hexdigest()

    path.write_bytes(b"different-artifact")
    changed = model_artifact_identity(path)
    assert changed.content_verified is True
    assert changed.sha256 == hashlib.sha256(b"different-artifact").hexdigest()
    assert changed.sha256 != first.sha256


def test_missing_model_artifact_identity_is_unverified(tmp_path: Path) -> None:
    identity = model_artifact_identity(tmp_path / "missing.gguf")

    assert identity.content_verified is False
    assert identity.sha256 is None
    assert identity.size_bytes is None
    assert "FileNotFoundError" in str(identity.error)
