from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_chain_conv_decode_bf16_tloop,
    qwen35_linear_attn_conv_decode_bf16,
    register_qwen35_linear_attn_conv_kernels,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.registry import KernelKey, register, resolve, unregister

_CHAIN_CONV_VARIANT = "bf16_c1_exact_state_rows_tloop"
_CHAIN_GDN_VARIANT = "bf16_c1_exact_state_rows_tloop"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


class _Buf:
    def __init__(self, nbytes: int) -> None:
        self.buffer = malloc(int(nbytes))

    @property
    def ptr(self) -> int:
        return int(self.buffer.ptr)

    @property
    def nbytes(self) -> int:
        return int(self.buffer.nbytes)

    def free(self) -> None:
        if self.buffer is not None:
            free(self.buffer)
            self.buffer = None


def _to_device(array: np.ndarray) -> _Buf:
    array = np.ascontiguousarray(array)
    buf = _Buf(array.nbytes)
    copy_host_to_device(buf.buffer, host_array_ptr(array), array.nbytes)
    return buf


def _from_device(buf: _Buf, shape: tuple[int, ...], dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buf.buffer, out.nbytes)
    return out


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = values + np.uint32(0x7FFF) + ((values >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.uint16))


def _weight(ptr: int):
    tensor = SimpleNamespace(ptr=int(ptr))
    return SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=tensor))


def test_chain_journal_exact_producers_are_registered() -> None:
    register_qwen35_linear_attn_conv_kernels()
    register_qwen35_linear_attn_gdn_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_chain_conv_decode",
            quant="gguf_qwen35",
            variant=_CHAIN_CONV_VARIANT,
        )
        is qwen35_linear_attn_chain_conv_decode_bf16_tloop
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_chain_recurrent_rmsnorm_gate",
            quant="gguf_qwen35",
            variant=_CHAIN_GDN_VARIANT,
        )
        is qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16
    )


def test_chain_journal_route_is_fail_closed_and_preserves_commit_ownership() -> None:
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    backend = "test_chain_journal"
    conv_key = KernelKey(
        backend,
        "linear_attn_chain_conv_decode",
        "gguf_qwen35",
        _CHAIN_CONV_VARIANT,
    )
    gdn_key = KernelKey(
        backend,
        "gdn_chain_recurrent_rmsnorm_gate",
        "gguf_qwen35",
        _CHAIN_GDN_VARIANT,
    )
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def conv(*args, **kwargs):
        calls.append(("conv", args, kwargs))

    def gdn(*args, **kwargs):
        calls.append(("gdn", args, kwargs))

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[object, ...]] = []

        def memcpy_async(self, *args) -> None:
            self.copies.append(args)

    runtime = FakeRuntime()
    runner = object.__new__(Qwen35GGUFFullStackRunner)
    runner.backend = backend
    runner.runtime = runtime
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(
            ssm_group_count=2,
            ssm_state_size=64,
            ssm_inner_size=256,
            ssm_time_step_rank=4,
            ssm_conv_kernel=4,
            rms_norm_eps=1.0e-6,
        )
    )
    layer = SimpleNamespace(
        weight=lambda name: _weight(
            {
                "ssm_conv1d": 0x6000,
                "ssm_dt_bias": 0x6100,
                "ssm_a": 0x6200,
                "ssm_norm": 0x6300,
            }[name]
        )
    )
    scratch = SimpleNamespace(
        linear_qkv=SimpleNamespace(ptr=0x1000),
        conv_out=SimpleNamespace(ptr=0x2000),
        linear_z=SimpleNamespace(ptr=0x3000),
        linear_alpha=SimpleNamespace(ptr=0x4000),
        linear_beta=SimpleNamespace(ptr=0x5000),
        recurrent_out=SimpleNamespace(ptr=0x7000),
    )
    conv_state = SimpleNamespace(ptr=0x8000, nbytes=80)
    recurrent_state = SimpleNamespace(ptr=0x9000, nbytes=96)
    conv_rows = SimpleNamespace(ptr=0xA000)
    recurrent_rows = SimpleNamespace(ptr=0xB000)

    register(conv_key, conv, replace=True)
    register(gdn_key, gdn, replace=True)
    try:
        assert runner._try_run_linear_attention_chain_journal_rows_exact(
            layer,
            scratch,
            conv_state,
            recurrent_state,
            rows=4,
            linear_state_rows=(conv_rows, recurrent_rows),
            commit_final_linear_state=False,
            stream=17,
            runtime=runtime,
        )
        assert [name for name, _args, _kwargs in calls] == ["conv", "gdn"]
        assert runtime.copies == []
        assert calls[0][1][:5] == (0x1000, 0x8000, 0xA000, 0x6000, 0x2000)
        assert calls[1][1][:8] == (
            0x2000,
            0x3000,
            0x4000,
            0x5000,
            0x6100,
            0x6200,
            0x6300,
            0x9000,
        )

        calls.clear()
        assert runner._try_run_linear_attention_chain_journal_rows_exact(
            layer,
            scratch,
            conv_state,
            recurrent_state,
            rows=4,
            linear_state_rows=(conv_rows, recurrent_rows),
            commit_final_linear_state=True,
            stream=19,
            runtime=runtime,
        )
        assert [name for name, _args, _kwargs in calls] == ["conv", "gdn"]
        assert runtime.copies == [
            (
                0x8000,
                0xA000 + 3 * 80,
                80,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                19,
            ),
            (
                0x9000,
                0xB000 + 3 * 96,
                96,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                19,
            ),
        ]

        calls.clear()
        runtime.copies.clear()
        unregister(gdn_key)
        assert not runner._try_run_linear_attention_chain_journal_rows_exact(
            layer,
            scratch,
            conv_state,
            recurrent_state,
            rows=4,
            linear_state_rows=(conv_rows, recurrent_rows),
            commit_final_linear_state=False,
            stream=23,
            runtime=runtime,
        )
        assert calls == []
        assert runtime.copies == []
    finally:
        unregister(conv_key)
        unregister(gdn_key)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 4])
def test_chain_journal_production_shape_is_byte_exact_to_scalar_decode(rows: int) -> None:
    """The direct journal path must preserve every c1 Conv/GDN output and state bit."""

    rng = np.random.default_rng(80427 + rows)
    runtime = get_hip_runtime()
    kernel_size = 4
    channels = 10240
    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = 128
    head_v_dim = 128
    conv_stride = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    assert conv_stride == channels

    hidden = _f32_to_bf16_u16(rng.normal(0.0, 0.08, (rows, channels)))
    conv_initial = rng.normal(0.0, 0.04, (channels, kernel_size)).astype(np.float32)
    conv_weight = rng.normal(0.0, 0.06, (channels, kernel_size)).astype(np.float32)
    gate = _f32_to_bf16_u16(rng.normal(0.0, 0.08, (rows, num_v_heads, head_v_dim)))
    alpha = _f32_to_bf16_u16(rng.normal(-0.1, 0.03, (rows, num_v_heads)))
    beta = _f32_to_bf16_u16(rng.normal(0.0, 0.05, (rows, num_v_heads)))
    dt_bias = rng.normal(-0.2, 0.03, (num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-1.0, 0.04, (num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.02, (head_v_dim,)).astype(np.float32)
    recurrent_shape = (num_v_heads, head_k_dim, head_v_dim)
    recurrent_initial = rng.normal(0.0, 0.01, recurrent_shape).astype(np.float32)

    buffers: list[_Buf] = []

    def device(array: np.ndarray) -> _Buf:
        buf = _to_device(array)
        buffers.append(buf)
        return buf

    def empty(nbytes: int) -> _Buf:
        buf = _Buf(nbytes)
        buffers.append(buf)
        return buf

    hidden_d = device(hidden)
    conv_weight_d = device(conv_weight)
    conv_scalar_d = device(conv_initial)
    conv_chain_d = device(conv_initial)
    conv_out_scalar_d = empty(rows * channels * np.dtype(np.float32).itemsize)
    conv_out_chain_d = empty(rows * channels * np.dtype(np.float32).itemsize)
    conv_rows_scalar_d = empty(rows * conv_initial.nbytes)
    conv_rows_chain_d = empty(rows * conv_initial.nbytes)
    gate_d = device(gate)
    alpha_d = device(alpha)
    beta_d = device(beta)
    dt_bias_d = device(dt_bias)
    a_log_d = device(a_log)
    norm_weight_d = device(norm_weight)
    recurrent_scalar_d = device(recurrent_initial)
    recurrent_chain_d = device(recurrent_initial)
    recurrent_out_scalar_d = empty(rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize)
    recurrent_out_chain_d = empty(rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize)
    recurrent_rows_scalar_d = empty(rows * recurrent_initial.nbytes)
    recurrent_rows_chain_d = empty(rows * recurrent_initial.nbytes)
    acc_d = empty(rows * num_v_heads * head_v_dim * np.dtype(np.float32).itemsize)

    try:
        hidden_row_nbytes = channels * np.dtype(np.uint16).itemsize
        conv_out_row_nbytes = channels * np.dtype(np.float32).itemsize
        gate_row_nbytes = num_v_heads * head_v_dim * np.dtype(np.uint16).itemsize
        ab_row_nbytes = num_v_heads * np.dtype(np.uint16).itemsize
        recurrent_out_row_nbytes = num_v_heads * head_v_dim * np.dtype(np.float32).itemsize
        for row in range(rows):
            qwen35_linear_attn_conv_decode_bf16(
                hidden_d.ptr + row * hidden_row_nbytes,
                conv_scalar_d.ptr,
                conv_weight_d.ptr,
                conv_out_scalar_d.ptr + row * conv_out_row_nbytes,
                channels,
                kernel_size,
                runtime=runtime,
            )
            runtime.memcpy_async(
                conv_rows_scalar_d.ptr + row * conv_initial.nbytes,
                conv_scalar_d.ptr,
                conv_initial.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
        qwen35_linear_attn_chain_conv_decode_bf16_tloop(
            hidden_d.ptr,
            conv_chain_d.ptr,
            conv_rows_chain_d.ptr,
            conv_weight_d.ptr,
            conv_out_chain_d.ptr,
            rows,
            channels,
            kernel_size,
            runtime=runtime,
        )

        for row in range(rows):
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
                conv_out_scalar_d.ptr + row * conv_out_row_nbytes,
                gate_d.ptr + row * gate_row_nbytes,
                alpha_d.ptr + row * ab_row_nbytes,
                beta_d.ptr + row * ab_row_nbytes,
                dt_bias_d.ptr,
                a_log_d.ptr,
                norm_weight_d.ptr,
                recurrent_scalar_d.ptr,
                recurrent_out_scalar_d.ptr + row * recurrent_out_row_nbytes,
                1.0e-6,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                runtime=runtime,
            )
            runtime.memcpy_async(
                recurrent_rows_scalar_d.ptr + row * recurrent_initial.nbytes,
                recurrent_scalar_d.ptr,
                recurrent_initial.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16(
            conv_out_chain_d.ptr,
            gate_d.ptr,
            alpha_d.ptr,
            beta_d.ptr,
            dt_bias_d.ptr,
            a_log_d.ptr,
            norm_weight_d.ptr,
            recurrent_chain_d.ptr,
            recurrent_rows_chain_d.ptr,
            acc_d.ptr,
            recurrent_out_chain_d.ptr,
            1.0e-6,
            rows,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            runtime=runtime,
        )
        runtime.device_synchronize()

        conv_out_scalar = _from_device(conv_out_scalar_d, (rows, channels), np.float32)
        conv_out_chain = _from_device(conv_out_chain_d, (rows, channels), np.float32)
        conv_rows_scalar = _from_device(
            conv_rows_scalar_d,
            (rows, channels, kernel_size),
            np.float32,
        )
        conv_rows_chain = _from_device(
            conv_rows_chain_d,
            (rows, channels, kernel_size),
            np.float32,
        )
        recurrent_out_scalar = _from_device(
            recurrent_out_scalar_d,
            (rows, num_v_heads, head_v_dim),
            np.float32,
        )
        recurrent_out_chain = _from_device(
            recurrent_out_chain_d,
            (rows, num_v_heads, head_v_dim),
            np.float32,
        )
        recurrent_rows_scalar = _from_device(
            recurrent_rows_scalar_d,
            (rows, *recurrent_shape),
            np.float32,
        )
        recurrent_rows_chain = _from_device(
            recurrent_rows_chain_d,
            (rows, *recurrent_shape),
            np.float32,
        )
        conv_chain_base = _from_device(conv_chain_d, conv_initial.shape, np.float32)
        recurrent_chain_base = _from_device(
            recurrent_chain_d,
            recurrent_initial.shape,
            np.float32,
        )

        assert np.array_equal(conv_out_chain.view(np.uint32), conv_out_scalar.view(np.uint32))
        assert np.array_equal(conv_rows_chain.view(np.uint32), conv_rows_scalar.view(np.uint32))
        assert np.array_equal(
            recurrent_out_chain.view(np.uint32),
            recurrent_out_scalar.view(np.uint32),
        )
        assert np.array_equal(
            recurrent_rows_chain.view(np.uint32),
            recurrent_rows_scalar.view(np.uint32),
        )
        assert np.array_equal(conv_chain_base.view(np.uint32), conv_initial.view(np.uint32))
        assert np.array_equal(
            recurrent_chain_base.view(np.uint32),
            recurrent_initial.view(np.uint32),
        )
    finally:
        for buf in reversed(buffers):
            buf.free()
