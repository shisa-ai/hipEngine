from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


def _buf(ptr: int) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr, nbytes=4096)


def _session() -> qgr.Qwen35GGUFResidentSession:
    session = object.__new__(qgr.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        hidden_size=4,
        linear_qkv_width=3,
        weights=SimpleNamespace(
            config=SimpleNamespace(
                ssm_inner_size=2,
                ssm_time_step_rank=1,
                is_moe=False,
            )
        ),
    )
    return session


def _scratch() -> SimpleNamespace:
    return SimpleNamespace(
        norm=_buf(10),
        linear_qkv=_buf(20),
        linear_qkv_f32=_buf(30),
        linear_z=_buf(40),
        linear_z_f32=_buf(50),
        linear_alpha=_buf(60),
        linear_alpha_f32=_buf(70),
        linear_beta=_buf(80),
        linear_beta_f32=_buf(90),
        conv_out=_buf(100),
        recurrent_out=_buf(110),
        attn_out=_buf(120),
        residual=_buf(130),
        post_norm=_buf(140),
    )


def test_verify_f32_linear_projections_flag_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS", raising=False)
    assert qgr._gguf_verify_f32_linear_projections_enabled() is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS", "1")
    assert qgr._gguf_verify_f32_linear_projections_enabled() is True


def test_verify_layer_boundary_capture_uses_f32_projection_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS", "1")

    def copy_bf16(ptr: int, elements: int, *, runtime) -> np.ndarray:
        return np.full((elements,), float(ptr), dtype=np.float32)

    def copy_f32(ptr: int, elements: int, *, runtime) -> np.ndarray:
        return np.full((elements,), -float(ptr), dtype=np.float32)

    monkeypatch.setattr(qgr, "_copy_bf16_ptr_to_host_f32", copy_bf16)
    monkeypatch.setattr(qgr, "_copy_f32_ptr_to_host", copy_f32)

    arrays = _session()._capture_verify_layer_boundary_rows(
        0,
        qgr.LINEAR_ATTENTION,
        hidden_in_ptr=1,
        hidden_in_f32_ptr=2,
        layer_out_ptr=3,
        layer_out_f32_ptr=None,
        scratch=_scratch(),
        rows=2,
        runtime=SimpleNamespace(),
    )

    np.testing.assert_array_equal(arrays["linear_qkv"], np.full((2, 3), -30.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["linear_z"], np.full((2, 2), -50.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["ssm_alpha"], np.full((2, 1), -70.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["ssm_beta"], np.full((2, 1), -90.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["linear_qkv_bf16_mirror"], np.full((2, 3), 20.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["linear_z_bf16_mirror"], np.full((2, 2), 40.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["ssm_alpha_bf16_mirror"], np.full((2, 1), 60.0, dtype=np.float32))
    np.testing.assert_array_equal(arrays["ssm_beta_bf16_mirror"], np.full((2, 1), 80.0, dtype=np.float32))
