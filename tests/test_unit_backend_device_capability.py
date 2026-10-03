"""Execution-selection refusal for an explicit HIP backend the device cannot run.

The live INT8 MTP prefix-reuse module pinned the ``hip_gfx1151`` backend on a
gfx1100 host; every JIT kernel was compiled for ``gfx1151`` and HIP aborted the
process with a segmentation fault at the first launch. These tests pin the
replacement contract:

- The probe result is a *per-device* architecture list in device order, with
  duplicates preserved, so an index names the device it actually names.
- ``ROCR_VISIBLE_DEVICES`` filters the physical list and ``HIP_VISIBLE_DEVICES``
  then indexes the result. The two compose; neither overrides the other.
- The check validates the *current logical device*, not "the target arch appears
  somewhere in the host": a gfx1151 on device 1 does not make gfx1151 kernels
  runnable on the default device 0.
- An explicit HIP backend the current device cannot execute is refused with a
  named error before any generator, weight load, or device work.
- Compatible execution, automatic selection, and offline cross-compilation are
  untouched.

``resolve_backend`` stays a pure key resolver, because registry helpers such as
``resolve_vibevoice_kernels`` resolve a backend for a dispatch table without
touching a device. The capability check runs at the execution seam that the
public ``LLM`` and the server both use.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.core.build import plan_hip_build
from hipengine.kernels.backends import (
    CPU_BACKEND,
    BackendDeviceCapabilityError,
    _current_hip_device_ordinal,
    hip_target_arch_environment,
    require_backend_device_compatibility,
    resolve_backend,
    selected_device_arches,
)

MIXED_HOST = ("gfx1100", "gfx1151")
TWO_CARD_GFX1100 = ("gfx1100", "gfx1100")


# --- per-device indexing ----------------------------------------------------


def test_duplicate_arch_devices_keep_their_own_index() -> None:
    """``(gfx1100, gfx1100, gfx1151)`` index 1 is a gfx1100, not a gfx1151.

    Deduplicating the host list first collapses the two gfx1100 cards, so index 1
    lands on the gfx1151 and the request is wrongly admitted.
    """

    host = ("gfx1100", "gfx1100", "gfx1151")

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=host,
            env={"ROCR_VISIBLE_DEVICES": "1"},
            current_device=0,
        )

    # Index 2 really is the gfx1151 card.
    require_backend_device_compatibility(
        "hip_gfx1151",
        detected_arches=host,
        env={"ROCR_VISIBLE_DEVICES": "2"},
        current_device=0,
    )


def test_same_arch_devices_stay_resolvable() -> None:
    """A second card of the same arch must not become an out-of-range index."""

    assert selected_device_arches(
        detected_arches=TWO_CARD_GFX1100,
        env={"ROCR_VISIBLE_DEVICES": "1"},
        current_device=0,
    ) == ("gfx1100",)

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=TWO_CARD_GFX1100,
            env={"ROCR_VISIBLE_DEVICES": "1"},
            current_device=0,
        )


# --- current device, not host membership ------------------------------------


def test_mixed_host_current_device_decides() -> None:
    """A gfx1151 elsewhere on the host does not excuse kernels on device 0."""

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151", detected_arches=MIXED_HOST, env={}, current_device=0
        )
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={}, current_device=1
    )

    require_backend_device_compatibility(
        "hip_gfx1100", detected_arches=MIXED_HOST, env={}, current_device=0
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1100", detected_arches=MIXED_HOST, env={}, current_device=1
        )


def test_mixed_host_defaults_to_logical_device_zero() -> None:
    """With no ordinal supplied, the engine default (device 0) is validated."""

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151", detected_arches=MIXED_HOST, env={}
        )


# --- visibility composition -------------------------------------------------


def test_visibility_filters_compose_instead_of_overriding() -> None:
    """ROCR=1 then HIP=0 selects physical device 1, so logical 0 is the gfx1151.

    Treating ``HIP_VISIBLE_DEVICES`` as taking precedence over
    ``ROCR_VISIBLE_DEVICES`` would report gfx1100 here and wrongly refuse
    gfx1151.
    """

    env = {"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"}

    assert selected_device_arches(
        detected_arches=MIXED_HOST, env=env, current_device=0
    ) == ("gfx1151",)
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env=env, current_device=0
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1100", detected_arches=MIXED_HOST, env=env, current_device=0
        )


def test_each_visibility_filter_remaps_on_its_own() -> None:
    assert selected_device_arches(
        detected_arches=MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": "1"}, current_device=0
    ) == ("gfx1151",)
    assert selected_device_arches(
        detected_arches=MIXED_HOST, env={"HIP_VISIBLE_DEVICES": "1"}, current_device=0
    ) == ("gfx1151",)
    # HIP indexes the ROCR-visible list, not the physical list.
    assert selected_device_arches(
        detected_arches=("gfx1100", "gfx1151", "gfx1100"),
        env={"ROCR_VISIBLE_DEVICES": "2,1", "HIP_VISIBLE_DEVICES": "1"},
        current_device=0,
    ) == ("gfx1151",)


# --- permissive only when genuinely unavailable -----------------------------


def test_unavailable_detection_is_permissive() -> None:
    # No probe result at all: incompatibility is not proven.
    require_backend_device_compatibility("hip_gfx1151", detected_arches=[], env={})
    assert selected_device_arches(detected_arches=[], env={}) == ()

    # An opaque device handle cannot be mapped to an architecture.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": "GPU-abc"}
    )
    # An empty visibility list cannot be mapped either.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": ""}
    )
    # An ordinal beyond the visible set is inconsistent, so it is not a refusal.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={}, current_device=7
    )


def test_missing_rocm_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_runtime(*args, **kwargs):
        raise OSError("libamdhip64.so not found")

    monkeypatch.setattr("ctypes.CDLL", _no_runtime)
    assert _current_hip_device_ordinal() is None


def test_non_hip_and_unregistered_backends_are_not_gated() -> None:
    require_backend_device_compatibility("cpu_reference", detected_arches=MIXED_HOST, env={})
    require_backend_device_compatibility("custom_backend", detected_arches=MIXED_HOST, env={})

    assert resolve_backend("cpu_reference", detected_arches=MIXED_HOST, env={}) == CPU_BACKEND
    assert (
        resolve_backend("custom_backend", detected_arches=MIXED_HOST, env={})
        == "custom_backend"
    )


def test_hipengine_hip_arch_is_not_hardware_evidence() -> None:
    # HIPENGINE_HIP_ARCH is a build-time cross-compile override, not a device.
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=("gfx1100",),
            env={"HIPENGINE_HIP_ARCH": "gfx1151"},
        )


# --- compatible execution and offline compilation ---------------------------


def test_compatible_explicit_backend_is_not_refused() -> None:
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=("gfx1151",), env={}, current_device=0
    )
    require_backend_device_compatibility(
        "hip_gfx1100", detected_arches=("gfx1100",), env={}, current_device=0
    )
    assert resolve_backend("hip_gfx1151", detected_arches=("gfx1151",), env={}) == "hip_gfx1151"


def test_auto_selection_is_preserved() -> None:
    assert resolve_backend("auto", detected_arches=("gfx1100",), env={}) == "hip_gfx1100"
    assert resolve_backend("auto", detected_arches=("gfx1151",), env={}) == "hip_gfx1151"

    with pytest.warns(RuntimeWarning, match="gfx1102"):
        assert resolve_backend("auto", detected_arches=("gfx1102",), env={}) == CPU_BACKEND


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


# --- public execution seam --------------------------------------------------


def _llm(monkeypatch: pytest.MonkeyPatch, backend: str, host: tuple[str, ...]):
    from hipengine import LLM
    from hipengine.kernels import backends

    monkeypatch.setattr(backends, "detect_hip_target_arches", lambda: host)
    monkeypatch.setattr(backends, "detect_hip_physical_device_arches", lambda: host)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    return LLM("/nonexistent/incompatible-model.gguf", backend=backend)


def test_homogeneous_host_does_not_query_the_hip_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-arch host must not initialize HIP just to run the check."""

    from hipengine.kernels import backends

    def _forbidden() -> int:
        raise AssertionError("hipGetDevice ran on a homogeneous host")

    monkeypatch.setattr(backends, "_current_hip_device_ordinal", _forbidden)
    assert selected_device_arches(detected_arches=("gfx1100", "gfx1100"), env={}) == ("gfx1100",)


def test_public_llm_seam_refuses_before_generator_or_device_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal lands before any generator, weight load, or device work.

    ``_get_text_generator`` reads the GGUF index (a metadata read of the file
    header and tensor directory, not weights) and then resolves the backend.
    Reading that metadata is allowed and is stubbed here; everything after the
    backend resolution is sentineled so it cannot run unnoticed.
    """

    import hipengine.generation as generation
    from hipengine import LLM
    from hipengine.kernels import backends

    metadata_read = {"count": 0}

    def _stub_metadata(self):
        metadata_read["count"] += 1
        return None, SimpleNamespace(name="stub-model")

    def _forbidden(*args, **kwargs):
        raise AssertionError("device work ran before the capability refusal")

    monkeypatch.setattr(LLM, "_load_model_metadata", _stub_metadata)
    monkeypatch.setattr(generation, "resolve_text_generator", _forbidden)
    monkeypatch.setattr(backends, "load_backend_kernel_package", _forbidden)
    monkeypatch.setattr(backends, "_current_hip_device_ordinal", _forbidden)

    llm = _llm(monkeypatch, "hip_gfx1151", ("gfx1100", "gfx1100"))

    with pytest.raises(BackendDeviceCapabilityError):
        llm._get_text_generator()

    # The metadata read is the only thing that ran before the refusal.
    assert metadata_read["count"] == 1
    assert llm._text_generator is None


def test_public_llm_seam_validates_an_env_backend_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_BACKEND", "hip_gfx1151")
    llm = _llm(monkeypatch, "auto", ("gfx1100",))

    with pytest.raises(BackendDeviceCapabilityError):
        _ = llm.resolved_backend


def test_public_llm_seam_keeps_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm(monkeypatch, "auto", ("gfx1100",))

    assert llm.resolved_backend == "hip_gfx1100"
