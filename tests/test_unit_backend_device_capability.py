"""Execution-selection refusal for an explicit HIP backend the device cannot run.

The live INT8 MTP prefix-reuse module pinned the ``hip_gfx1151`` backend on a
gfx1100 host; every JIT kernel was compiled for ``gfx1151`` and HIP aborted the
process with a segmentation fault at the first launch. These tests pin the
replacement contract: an explicit HIP backend whose declared target arch is not
on the selected visible device is refused with a named error before any model
load or kernel launch, while compatible execution, automatic selection, and
offline cross-compilation are untouched.

``resolve_backend`` stays a pure key resolver, because registry helpers such as
``resolve_vibevoice_kernels`` resolve a backend for a dispatch table without
touching a device. The capability check runs at the execution seam that the
public ``LLM`` and the server both use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.build import plan_hip_build
from hipengine.kernels.backends import (
    CPU_BACKEND,
    BackendDeviceCapabilityError,
    hip_target_arch_environment,
    require_backend_device_compatibility,
    resolve_backend,
    selected_device_arches,
)


def test_explicit_incompatible_hip_backend_raises_named_error() -> None:
    with pytest.raises(BackendDeviceCapabilityError) as excinfo:
        require_backend_device_compatibility("hip_gfx1151", detected_arches=["gfx1100"], env={})

    message = str(excinfo.value)
    assert "hip_gfx1151" in message
    assert "gfx1151" in message
    assert "gfx1100" in message


def test_explicit_compatible_hip_backend_is_not_refused() -> None:
    require_backend_device_compatibility("hip_gfx1151", detected_arches=["gfx1151"], env={})
    require_backend_device_compatibility("hip_gfx1100", detected_arches=["gfx1100"], env={})

    assert resolve_backend("hip_gfx1151", detected_arches=["gfx1151"], env={}) == "hip_gfx1151"
    assert resolve_backend("hip_gfx1100", detected_arches=["gfx1100"], env={}) == "hip_gfx1100"


def test_auto_selection_is_preserved() -> None:
    assert resolve_backend("auto", detected_arches=["gfx1100"], env={}) == "hip_gfx1100"
    assert resolve_backend("auto", detected_arches=["gfx1151"], env={}) == "hip_gfx1151"


def test_auto_selection_with_unknown_arch_falls_back_without_capability_error() -> None:
    with pytest.warns(RuntimeWarning, match="gfx1102"):
        assert resolve_backend("auto", detected_arches=["gfx1102"], env={}) == CPU_BACKEND


def test_unknown_device_detection_does_not_block_an_explicit_backend() -> None:
    # No device could be resolved, so incompatibility is not proven.
    require_backend_device_compatibility("hip_gfx1151", detected_arches=[], env={})


def test_non_hip_and_unregistered_backends_are_not_gated() -> None:
    require_backend_device_compatibility("cpu_reference", detected_arches=["gfx1100"], env={})
    require_backend_device_compatibility("custom_backend", detected_arches=["gfx1100"], env={})

    assert resolve_backend("cpu_reference", detected_arches=["gfx1100"], env={}) == CPU_BACKEND
    assert (
        resolve_backend("custom_backend", detected_arches=["gfx1100"], env={})
        == "custom_backend"
    )


def test_selected_visible_device_is_considered_on_a_mixed_host() -> None:
    host = ["gfx1100", "gfx1151"]

    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=host, env={"ROCR_VISIBLE_DEVICES": "1"}
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151", detected_arches=host, env={"ROCR_VISIBLE_DEVICES": "0"}
        )
    # HIP_VISIBLE_DEVICES wins over ROCR_VISIBLE_DEVICES, matching the server.
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=host,
            env={"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "1"},
        )


def test_opaque_or_unresolvable_visible_selection_does_not_block() -> None:
    # A UUID handle cannot be mapped to an arch, so the check stays permissive.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=["gfx1100"], env={"ROCR_VISIBLE_DEVICES": "GPU-abc"}
    )
    # An out-of-range index cannot be resolved either.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=["gfx1100"], env={"ROCR_VISIBLE_DEVICES": "5"}
    )


def test_hipengine_hip_arch_is_not_hardware_evidence() -> None:
    # HIPENGINE_HIP_ARCH is a build-time cross-compile override, not a device.
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=["gfx1100"],
            env={"HIPENGINE_HIP_ARCH": "gfx1151"},
        )


def test_selected_device_arches_reports_the_visible_subset() -> None:
    assert selected_device_arches(detected_arches=["gfx1100", "gfx1151"], env={}) == (
        "gfx1100",
        "gfx1151",
    )
    assert selected_device_arches(
        detected_arches=["gfx1100", "gfx1151"], env={"ROCR_VISIBLE_DEVICES": "1"}
    ) == ("gfx1151",)


def test_offline_cross_compilation_is_not_gated() -> None:
    source = Path("hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_embedding.hip")
    with hip_target_arch_environment("gfx1151"):
        artifact = plan_hip_build(
            sources=[source],
            family="gguf_q6_k_embedding",
            profile="decode",
            compiler_version="test-compiler",
        )

    assert artifact.target_arch == "gfx1151"
    assert "--offload-arch=gfx1151" in artifact.flags


def _llm_with_detected_host(monkeypatch: pytest.MonkeyPatch, backend: str):
    from hipengine import LLM
    from hipengine.kernels import backends

    monkeypatch.setattr(backends, "detect_hip_target_arches", lambda: ("gfx1100",))
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    return LLM("/nonexistent/incompatible-model.gguf", backend=backend)


def test_public_llm_seam_refuses_before_loading_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm_with_detected_host(monkeypatch, "hip_gfx1151")

    with pytest.raises(BackendDeviceCapabilityError):
        _ = llm.resolved_backend


def test_public_llm_seam_validates_an_env_backend_force(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_BACKEND", "hip_gfx1151")
    llm = _llm_with_detected_host(monkeypatch, "auto")

    with pytest.raises(BackendDeviceCapabilityError):
        _ = llm.resolved_backend


def test_public_llm_seam_keeps_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm_with_detected_host(monkeypatch, "auto")

    assert llm.resolved_backend == "hip_gfx1100"
