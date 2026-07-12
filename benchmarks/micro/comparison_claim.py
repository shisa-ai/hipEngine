"""Shared provenance gate for joint HIP/Vulkan microbenchmark wrappers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _device_fingerprint(name: Any) -> str:
    text = re.sub(r"\([^)]*\)", "", str(name).lower()).strip()
    text = re.sub(r"^amd\s+", "", text)
    match = re.search(r"radeon\s+(\d+s)", text)
    if match:
        return f"radeon_{match.group(1)}"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _raw_result_evidence(
    raw_results: dict[str, Any],
    expected_run_tags: dict[str, str],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    identity_pass = True
    for backend in ("hip", "vulkan"):
        records = [
            raw
            for key, raw in raw_results.items()
            if key == backend or key.startswith(f"{backend}:")
        ]
        device_names: set[str] = set()
        device_fingerprints: set[str] = set()
        gfx_arches: set[str] = set()
        source_hashes: set[str] = set()
        backend_identity_pass = bool(records)
        for raw in records:
            if not isinstance(raw, dict):
                backend_identity_pass = False
                continue
            if (
                raw.get("backend") != backend
                or raw.get("run_tag") != expected_run_tags[backend]
                or raw.get("status") != "diagnostic"
            ):
                backend_identity_pass = False
            hardware = raw.get("hardware")
            if not isinstance(hardware, dict):
                backend_identity_pass = False
                continue
            device_name = str(hardware.get("device_name") or "").strip()
            fingerprint = _device_fingerprint(device_name)
            if not device_name or not fingerprint or fingerprint == "unknown":
                backend_identity_pass = False
            else:
                device_names.add(device_name)
                device_fingerprints.add(fingerprint)
            gfx_arch = str(hardware.get("gcn_arch_name") or "").strip()
            if gfx_arch:
                gfx_arches.add(gfx_arch)
            source = raw.get("source")
            if isinstance(source, dict):
                source_hash = str(source.get("source_hash") or "").strip()
                if source_hash:
                    source_hashes.add(source_hash)
            source_hash = str(raw.get("source_hash") or "").strip()
            if source_hash:
                source_hashes.add(source_hash)
        if len(device_names) != 1 or len(device_fingerprints) != 1:
            backend_identity_pass = False
        if backend == "hip" and len(gfx_arches) != 1:
            backend_identity_pass = False
        identity_pass = identity_pass and backend_identity_pass
        evidence[backend] = {
            "raw_result_count": len(records),
            "identity_pass": backend_identity_pass,
            "device_names": sorted(device_names),
            "device_fingerprints": sorted(device_fingerprints),
            "gfx_arches": sorted(gfx_arches),
            "source_hashes": sorted(source_hashes),
        }
    evidence["identity_pass"] = identity_pass
    return evidence


def build_joint_claim_evidence(
    *,
    source: dict[str, Any],
    source_paths: list[Path],
    repo_root: Path,
    raw_results: dict[str, Any],
    expected_run_tags: dict[str, str],
    configured_arch: str,
    fallback_gpu: str,
    correctness_pass: bool,
    matrix: dict[str, Any],
    matrix_complete: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_evidence = _raw_result_evidence(raw_results, expected_run_tags)
    hip = raw_evidence["hip"]
    vulkan = raw_evidence["vulkan"]
    hip_fingerprint = (
        hip["device_fingerprints"][0] if len(hip["device_fingerprints"]) == 1 else ""
    )
    vulkan_fingerprint = (
        vulkan["device_fingerprints"][0]
        if len(vulkan["device_fingerprints"]) == 1
        else ""
    )
    device_match = bool(hip_fingerprint) and hip_fingerprint == vulkan_fingerprint
    hip_arch = hip["gfx_arches"][0] if len(hip["gfx_arches"]) == 1 else ""
    configured_arch = configured_arch.strip()
    gfx_arch_match = bool(hip_arch) and (
        not configured_arch or configured_arch == hip_arch
    )
    source_hash_present = bool(source.get("source_hash"))
    same_commit = bool(source.get("commit"))
    clean_source = not bool(source.get("dirty"))
    performance_claim = all(
        (
            source_hash_present,
            same_commit,
            clean_source,
            bool(raw_evidence["identity_pass"]),
            device_match,
            gfx_arch_match,
            correctness_pass,
            matrix_complete,
        )
    )
    blocking_reasons = []
    for passed, reason in (
        (source_hash_present, "source_hash_missing"),
        (same_commit, "commit_missing"),
        (clean_source, "dirty_source"),
        (bool(raw_evidence["identity_pass"]), "raw_result_identity_mismatch_or_missing"),
        (device_match, "device_identity_mismatch_or_missing"),
        (gfx_arch_match, "gfx_arch_mismatch_or_missing"),
        (correctness_pass, "correctness_not_passed"),
        (matrix_complete, "comparison_matrix_incomplete"),
    ):
        if not passed:
            blocking_reasons.append(reason)
    hardware = {
        "hip": {
            "gpu_name": (
                hip["device_names"][0]
                if len(hip["device_names"]) == 1
                else fallback_gpu or "unknown"
            ),
            "gfx_arch": hip_arch or configured_arch or "unknown",
        },
        "vulkan": {
            "gpu_name": (
                vulkan["device_names"][0]
                if len(vulkan["device_names"]) == 1
                else fallback_gpu or "unknown"
            ),
            "gfx_arch": hip_arch or configured_arch or "unknown",
        },
    }
    source_coverage = {
        "scope": "single_joint_wrapper_invocation",
        "combined_source_hash": source.get("source_hash", ""),
        "combined_hash_files": [
            str(path.relative_to(repo_root)) for path in source_paths
        ],
        "combined_hash_backends": ["hip", "vulkan"],
        "backend_source_hashes": {
            "hip": hip["source_hashes"],
            "vulkan": vulkan["source_hashes"],
        },
        "backend_hash_status": {
            backend: (
                "raw_explicit"
                if raw_evidence[backend]["source_hashes"]
                else "covered_by_combined_source_hash"
            )
            for backend in ("hip", "vulkan")
        },
    }
    claim_gate = {
        "status": "pass" if performance_claim else "blocked",
        "performance_claim": performance_claim,
        "source_scope": "single_joint_wrapper_invocation",
        "same_commit": same_commit,
        "clean_source": clean_source,
        "source_hash_present": source_hash_present,
        "raw_result_identity_pass": raw_evidence["identity_pass"],
        "device_match": device_match,
        "hip_device_fingerprint": hip_fingerprint,
        "vulkan_device_fingerprint": vulkan_fingerprint,
        "gfx_arch_match": gfx_arch_match,
        "correctness_pass": correctness_pass,
        "matrix_complete": matrix_complete,
        "matrix": matrix,
        "raw_evidence": raw_evidence,
        "blocking_reasons": blocking_reasons,
    }
    return hardware, source_coverage, claim_gate
