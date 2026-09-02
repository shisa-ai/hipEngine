"""B5 contracts for selective dense planar-Q6 integer MMQ."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock


def test_b5_q6_integer_mmq_profile_default_and_env_override() -> None:
    import hipengine.runtime.gguf_linear as gguf_linear

    resolver = getattr(gguf_linear, "q6_integer_mmq_for", None)
    assert callable(resolver)
    env = "HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL"
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(env, None)
        assert resolver("production") is True
        assert resolver("strict") is False
        assert resolver(None) is False
        assert resolver("production", profile_fell_back_to_strict=True) is False
        os.environ[env] = "1"
        assert resolver("production") is True
        assert resolver("strict") is True
        os.environ[env] = "0"
        assert resolver("production") is False


def test_b5_q6_integer_mmq_generator_configures_resident_session() -> None:
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator.execution_profile = "production"
    generator.execution_profile_fell_back_to_strict = False
    session = SimpleNamespace()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL", None)
        generator._configure_session(session)
    assert session.use_q6_integer_mmq is True
    generator.execution_profile = "strict"
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL", None)
        generator._configure_session(session)
    assert session.use_q6_integer_mmq is False
    generator.execution_profile = "production"
    with mock.patch.dict(
        os.environ,
        {"HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL": "0"},
        clear=False,
    ):
        generator._configure_session(session)
    assert session.use_q6_integer_mmq is False


def test_b5_q6_integer_mmq_workspace_is_bounded_and_restored() -> None:
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q4_k_q8_1_selected_prefill as candidate,
    )

    context = getattr(candidate, "q6_dense_integer_mmq_session", None)
    current = getattr(candidate, "q6_dense_integer_mmq_workspace", None)
    nbytes = getattr(candidate, "q6_dense_integer_mmq_nbytes", None)
    assert callable(context)
    assert callable(current)
    assert callable(nbytes)
    assert current() is None
    required = nbytes(48, 17_408)
    assert required > 0
    with context(
        True,
        workspace_ptr=0x1200,
        workspace_nbytes=required,
        library="candidate-library",
    ):
        owner = current()
        assert owner is not None
        assert owner.ptr == 0x1200
        assert owner.nbytes == required
        assert owner.library == "candidate-library"
    assert current() is None


def test_b5_q6_integer_mmq_registered_only_on_gfx1151() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered

    variant = "t16_q8_1_planar_integer_mmq64x64_bf16_bf16_out"
    load_backend_kernel_package("hip_gfx1151")
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            variant,
        )
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            variant,
        )
    )


def test_b5_q6_integer_mmq_dispatch_is_exactly_shape_and_row_bounded() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey
    import hipengine.runtime.gguf_linear as gguf_linear

    candidate = __import__(
        "hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill",
        fromlist=["q6_dense_integer_mmq_session"],
    )
    context = getattr(candidate, "q6_dense_integer_mmq_session", None)
    rewrite = getattr(gguf_linear, "_q6_integer_mmq_prefill_dispatch", None)
    assert callable(context)
    assert callable(rewrite)
    load_backend_kernel_package("hip_gfx1151")
    base = gguf_linear.GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_wmma_prefill_bf16_bf16_out",
        ),
        "t16",
    )
    with context(
        True,
        workspace_ptr=0x4000,
        workspace_nbytes=8 << 20,
        library="candidate-library",
    ):
        for rows in (17, 20, 28, 48):
            for shape in ((17_408, 5_120), (5_120, 1_024)):
                selected = rewrite(
                    base,
                    rows=rows,
                    in_features=shape[0],
                    out_features=shape[1],
                )
                assert selected.key.variant == (
                    "t16_q8_1_planar_integer_mmq64x64_bf16_bf16_out"
                )
        for rows in (16, 49, 512):
            assert rewrite(
                base,
                rows=rows,
                in_features=17_408,
                out_features=5_120,
            ) == base
        assert rewrite(
            base,
            rows=28,
            in_features=5_120,
            out_features=10_240,
        ) == base
    assert rewrite(
        base,
        rows=28,
        in_features=17_408,
        out_features=5_120,
    ) == base


def test_b5_q6_integer_mmq_resident_context_reuses_session_owned_workspace(
    monkeypatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q4_k_q8_1_selected_prefill as candidate,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    current = getattr(candidate, "q6_dense_integer_mmq_workspace", None)
    assert callable(current)
    allocation = SimpleNamespace(ptr=0x9000, nbytes=1024 * 17_408 * 2)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = "runtime"
    session._buffers = (allocation,)
    session._prefill_f16_staging_buf = allocation
    session._q6_integer_mmq_library = None
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=5_120,
                feed_forward_length=17_408,
                ssm_inner_size=6_144,
            )
        )
    )
    session.use_prefill_f16_staging = True
    session.use_q6_integer_mmq = True
    monkeypatch.setattr(
        runner_module,
        "build_gguf_q4_k_q8_1_selected_prefill",
        lambda **kwargs: "candidate-library",
        raising=False,
    )
    with mock.patch.dict(
        os.environ,
        {
            "HIPENGINE_GGUF_PREFILL_F16_STAGING": "1",
            "HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL": "1",
        },
        clear=False,
    ):
        with session._prefill_f16_staging_context():
            with session._q6_integer_mmq_context():
                owner = current()
                assert owner is not None
                assert owner.ptr == allocation.ptr
                assert owner.nbytes == allocation.nbytes
                assert owner.library == "candidate-library"
    assert current() is None
    assert session._buffers == (allocation,)


def test_b5_packed_target_verifier_enters_integer_mmq_context() -> None:
    import inspect

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    source = inspect.getsource(Qwen35GGUFResidentSession.verify_target_blocks_batch)
    assert "self._q6_integer_mmq_context()" in source


def test_b5_q6_integer_mmq_composite_launch_uses_owner_workspace(monkeypatch) -> None:
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q4_k_q8_1_selected_prefill as candidate,
    )

    context = getattr(candidate, "q6_dense_integer_mmq_session", None)
    launch = getattr(
        candidate,
        "gguf_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64_bf16_bf16_out",
        None,
    )
    assert callable(context)
    assert callable(launch)
    calls = []
    monkeypatch.setattr(
        candidate,
        "gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3",
        lambda *args, **kwargs: calls.append(("pack", args, kwargs)),
    )
    monkeypatch.setattr(
        candidate,
        "_launch_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64",
        lambda *args, **kwargs: calls.append(("mmq", args, kwargs)),
    )
    with context(
        True,
        workspace_ptr=0xA000,
        workspace_nbytes=8 << 20,
        library="candidate-library",
    ):
        launch(0x1000, 0x2000, 0x3000, 28, 17_408, 5_120, runtime="rt")
    assert [call[0] for call in calls] == ["pack", "mmq"]
    assert calls[0][1][1] == 0xA000
    assert calls[1][1][:3] == (0xA000, 0x2000, 0x3000)


def test_b5_dense_specialization_matches_selected_parent_on_gpu() -> None:
    import ctypes

    import numpy as np
    import pytest

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q4_k_q8_1_selected_prefill as candidate,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from tests.test_gguf_k_t16_selected_wmma_prefill import (
        _build_compact_t16_fixture,
    )

    fixture = _build_compact_t16_fixture(
        quant="gguf_q6_k_t16_v1",
        counts=[17],
        in_features=512,
        out_features=64,
        dtype="bf16",
        seed=2026,
    )
    tiles = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(fixture.qweight).tiles
    )
    metadata = (
        fixture.x_host,
        np.asarray([0, 17], dtype=np.int64),
        np.asarray([0, 64], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        tiles,
    )
    runtime = get_hip_runtime()
    library = candidate.build_gguf_q4_k_q8_1_selected_prefill(load=True)
    buffers = []
    try:
        for values in metadata:
            buffer = malloc(values.nbytes, runtime=runtime)
            copy_host_to_device(
                buffer,
                host_array_ptr(np.ascontiguousarray(values)),
                runtime=runtime,
            )
            buffers.append(buffer)
        q8 = malloc(candidate.q6_dense_integer_mmq_nbytes(17, 512), runtime=runtime)
        selected = malloc(17 * 64 * 2, runtime=runtime)
        dense = malloc(17 * 64 * 2, runtime=runtime)
        buffers.extend((q8, selected, dense))
        candidate.gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
            buffers[0].ptr,
            q8.ptr,
            17,
            512,
            residual_passes=1,
            q6_half_sums=True,
            library=library,
            runtime=runtime,
        )
        candidate.gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
            q8.ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            selected.ptr,
            17,
            512,
            64,
            1,
            64,
            residual_passes=1,
            rowvec=True,
            tile_rows=64,
            qmicro=True,
            compact_activation=True,
            half_row_activation=True,
            skip_padded_activation=True,
            qmicro_planar=True,
            integer_wmma=True,
            wmma_hoist_activation=True,
            wmma_prefetch_weight=True,
            wmma_prefetch_activation=True,
            precomputed_activation_sums=True,
            library=library,
            runtime=runtime,
        )
        with candidate.q6_dense_integer_mmq_session(
            True,
            workspace_ptr=q8.ptr,
            workspace_nbytes=q8.nbytes,
            library=library,
        ):
            candidate.gguf_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64_bf16_bf16_out(
                buffers[0].ptr,
                buffers[4].ptr,
                dense.ptr,
                17,
                512,
                64,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        selected_host = np.empty((17, 64), dtype=np.uint16)
        dense_host = np.empty_like(selected_host)
        copy_device_to_host(host_array_ptr(selected_host), selected, runtime=runtime)
        copy_device_to_host(host_array_ptr(dense_host), dense, runtime=runtime)
        runtime.device_synchronize()
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    assert np.array_equal(dense_host, selected_host)
    actual = (dense_host.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(actual).all()
    reference = np.asarray(fixture.reference, dtype=np.float32)
    relative_l2 = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
    assert relative_l2 <= 1.0e-2
