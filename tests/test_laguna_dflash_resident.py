from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.tensor import Tensor
from hipengine.speculative import laguna_dflash as laguna_dflash_module
from hipengine.speculative.laguna_dflash import (
    LagunaDFlashCaptureOwner,
    LagunaDFlashResidentDrafter,
)


CAPTURE_DEPTHS = (2, 11, 20, 30, 39, 48)


class _FakeRuntime:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.next_ptr = 0x100000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.calls = 0
        self.fail_at = fail_at

    def malloc(self, nbytes: int) -> int:
        self.calls += 1
        if self.calls == self.fail_at:
            raise MemoryError("synthetic capture allocation failure")
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        del dst, src, count, kind


def test_laguna_dflash_capture_owner_allocates_exact_depth_rows_and_frees() -> None:
    runtime = _FakeRuntime()

    owner = LagunaDFlashCaptureOwner.allocate(
        depths=CAPTURE_DEPTHS,
        hidden_size=3072,
        rows=3,
        runtime=runtime,  # type: ignore[arg-type]
        device=Device("hip", 0),
    )

    assert owner.depths == CAPTURE_DEPTHS
    assert owner.targets.rows == 3
    assert tuple(owner.targets.buffers) == CAPTURE_DEPTHS
    assert len(owner.tensors) == 6
    assert all(tensor.shape == (3, 3072) for tensor in owner.tensors)
    assert owner.nbytes == 6 * 3 * 3072 * 2
    allocated = set(runtime.allocations)
    owner.free()
    assert set(runtime.freed) == allocated
    assert runtime.allocations == {}
    owner.free()


def test_laguna_dflash_capture_owner_cleans_partial_failure() -> None:
    runtime = _FakeRuntime(fail_at=4)

    with pytest.raises(MemoryError, match="synthetic"):
        LagunaDFlashCaptureOwner.allocate(
            depths=CAPTURE_DEPTHS,
            hidden_size=3072,
            rows=1,
            runtime=runtime,  # type: ignore[arg-type]
        )

    assert runtime.allocations == {}
    assert len(runtime.freed) == 3


def test_laguna_dflash_capture_owner_rejects_duplicate_depths() -> None:
    with pytest.raises(ValueError, match="unique"):
        LagunaDFlashCaptureOwner.allocate(
            depths=(2, 2),
            hidden_size=3072,
            rows=1,
            runtime=_FakeRuntime(),  # type: ignore[arg-type]
        )


def test_laguna_dflash_query_uses_bf16_norm_rotary_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    device = Device("hip", 0)
    next_ptr = iter(range(0x1000, 0x10000, 0x100))

    def tensor(shape: tuple[int, ...], dtype: DType) -> Tensor:
        return Tensor.from_handle(next(next_ptr), shape, dtype, device)

    rows, hidden, q_heads, kv_heads, head_dim, intermediate = 2, 4, 2, 1, 4, 8
    q_features = q_heads * head_dim
    kv_features = kv_heads * head_dim
    drafter = object.__new__(LagunaDFlashResidentDrafter)
    drafter.config = SimpleNamespace(
        hidden_size=hidden,
        q_features=q_features,
        kv_features=kv_features,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rms_norm_eps=1.0e-6,
    )
    drafter.query_rows = rows
    drafter.runtime = object()
    drafter.device = device
    drafter.backend = "hip_gfx1151"
    drafter._draft_library = object()
    drafter.weights = SimpleNamespace(tensor=lambda _name: tensor((head_dim,), DType.BF16))
    drafter._f32_norm_weights = {
        "layers.0.input_layernorm.weight": tensor((hidden,), DType.FP32),
        "layers.0.post_attention_layernorm.weight": tensor((hidden,), DType.FP32),
    }
    drafter.rope = SimpleNamespace(
        cos=SimpleNamespace(tensor=tensor((64, head_dim), DType.FP32)),
        sin=SimpleNamespace(tensor=tensor((64, head_dim), DType.FP32)),
    )
    drafter.kv_cache = SimpleNamespace(attend_prefill=lambda *_args, **_kwargs: None)
    drafter.target = SimpleNamespace(
        libraries=SimpleNamespace(
            kv_attention=object(),
            attention_gate=object(),
            gguf_ops=object(),
        ),
        kernel_plan=SimpleNamespace(attention_gate=object()),
    )
    drafter.query_norm = tensor((rows, hidden), DType.BF16)
    drafter.query_raw = tensor((rows, q_features), DType.FP32)
    drafter.key_raw = tensor((rows, kv_features), DType.FP32)
    drafter.value_raw = tensor((rows, kv_features), DType.FP32)
    drafter.query_rotated = tensor((rows, q_features), DType.FP32)
    drafter.key_rotated = tensor((rows, kv_features), DType.FP32)
    drafter.query_positions = tensor((rows,), DType.INT32)
    drafter.attention_context = tensor((rows, q_features), DType.FP32)
    drafter.gate_logits = tensor((rows, q_heads), DType.FP32)
    drafter.gated_context = tensor((rows, q_features), DType.BF16)
    drafter.attention_output = tensor((rows, hidden), DType.BF16)
    drafter.post_attention = tensor((rows, hidden), DType.FP32)
    drafter.ffn_norm = tensor((rows, hidden), DType.BF16)
    drafter.ffn_gate = tensor((rows, intermediate), DType.BF16)
    drafter.ffn_up = tensor((rows, intermediate), DType.BF16)
    drafter.ffn_intermediate = tensor((rows, intermediate), DType.BF16)
    drafter.ffn_output = tensor((rows, hidden), DType.BF16)

    monkeypatch.setattr(laguna_dflash_module, "gguf_rmsnorm_f32_f32_weight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        laguna_dflash_module,
        "gguf_add_rmsnorm_f32_bf16_f32_weight",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(laguna_dflash_module, "gguf_f32_bf16_add_out_f32", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(laguna_dflash_module, "project_dflash_bf16_to_f32", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(laguna_dflash_module, "project_dflash_bf16_to_bf16", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(laguna_dflash_module, "gate_laguna_dflash_attention_bf16", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(laguna_dflash_module, "dflash_silu_mul_bf16", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hipengine.runtime.laguna_rope.launch_laguna_head_rmsnorm_rope",
        lambda *_args, **_kwargs: pytest.fail("target F32-weight rotary kernel must not receive BF16 DFlash norms"),
    )
    calls: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    def capture_rotary(query: Tensor, key: Tensor, *_args, **_kwargs) -> None:
        calls.append((query, key, _args[-2], _args[-1]))

    monkeypatch.setattr(laguna_dflash_module, "dflash_head_rmsnorm_rotary_f32", capture_rotary)
    query_in = tensor((rows, hidden), DType.FP32)
    query_out = tensor((rows, hidden), DType.FP32)

    drafter._run_query_layer(0, query_in=query_in, query_out=query_out, stream=0)

    assert len(calls) == 1
    query, key, query_out_view, key_out_view = calls[0]
    assert query.shape == (1, rows, q_heads, head_dim)
    assert key.shape == (1, rows, kv_heads, head_dim)
    assert query_out_view.shape == query.shape
    assert key_out_view.shape == key.shape
