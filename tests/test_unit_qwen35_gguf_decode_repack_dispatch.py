from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipengine.runtime.qwen35_gguf_runner as qwen_runtime
from hipengine.core.dtype import DType
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner


class _Tensor:
    def __init__(self, ptr: int, *, numel: int = 1) -> None:
        self.ptr = ptr
        self.numel = numel


class _Weight:
    def __init__(self, ptr: int) -> None:
        self._allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))

    def allocation(self):
        return self._allocation


class _Layer:
    def __init__(self) -> None:
        self._weights = {name: _Weight(0x1000 + index * 0x10) for index, name in enumerate(_WEIGHT_NAMES)}

    def weight(self, name: str) -> _Weight:
        return self._weights[name]


_WEIGHT_NAMES = (
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_q_norm",
    "attn_k_norm",
    "attn_output",
)


def _runner(
    *,
    is_moe: bool = True,
    backend: str = "hip_gfx1100",
    head_count: int = 16,
    head_count_kv: int = 2,
    hidden_size: int = 2048,
    block_count: int = 40,
) -> Qwen35GGUFFullStackRunner:
    runner = object.__new__(Qwen35GGUFFullStackRunner)
    cfg = SimpleNamespace(
        is_moe=is_moe,
        rms_norm_eps=1.0e-6,
        head_count=head_count,
        head_count_kv=head_count_kv,
        key_length=256,
        value_length=256,
        rope_dimension_count=64,
        hidden_size=hidden_size,
        block_count=block_count,
    )
    layer = _Layer()
    runner.weights = SimpleNamespace(config=cfg, layer=lambda layer_id: layer)
    runner.runtime = object()
    runner.backend = backend
    runner.compiler_version = None
    runner.require_cached_build = False
    runner._cast_library = lambda: "cast-lib"
    runner._paged_kv_write_library = lambda: "kv-write-lib"
    runner._paged_attn_decode_library = lambda: "paged-attn-lib"
    return runner


def _scratch(
    *,
    position: int,
    max_positions: int,
    kv_storage_dtype: DType = DType.BF16,
    bf16_mirror: bool = False,
) -> SimpleNamespace:
    block_size = 256
    if kv_storage_dtype is DType.INT8_PER_TOKEN_HEAD:
        scale_metadata = SimpleNamespace(k_scale=_Tensor(0x2130), v_scale=_Tensor(0x2140))
        append_spans = SimpleNamespace(storage_dtype=DType.INT8_PER_TOKEN_HEAD, scale_metadata=scale_metadata)
        decode_spans = SimpleNamespace(storage_dtype=DType.INT8_PER_TOKEN_HEAD, scale_metadata=scale_metadata)
    else:
        append_spans = object()
        decode_spans = object()
    scratch = SimpleNamespace(
        position_host=np.array([position], dtype=np.int64),
        set_full_attention_position=lambda position, runtime: None,
        norm=_Tensor(0x2000),
        full_q=_Tensor(0x2010),
        full_k=_Tensor(0x2020),
        full_v=_Tensor(0x2030),
        full_query_raw=_Tensor(0x2040),
        full_gate=_Tensor(0x2050),
        full_key_raw=_Tensor(0x2060),
        cos_table=_Tensor(0x2070),
        sin_table=_Tensor(0x2080),
        position_tensor=_Tensor(0x2090),
        full_query=_Tensor(0x20A0),
        full_key=_Tensor(0x20B0),
        append_spans=append_spans,
        decode_spans=decode_spans,
        kv_storage_dtype=kv_storage_dtype,
        block_size=block_size,
        max_positions=max_positions,
        full_attn_context=_Tensor(0x20C0),
        full_attn_split_partial=_Tensor(0x20D0),
        full_attn_split_m=_Tensor(0x20E0),
        full_attn_split_l=_Tensor(0x20F0),
        full_attn_split_count=(max_positions + block_size - 1) // block_size,
        full_gated=_Tensor(0x2100),
    )
    key_cache = _Tensor(0x2110)
    value_cache = _Tensor(0x2120)
    mirror_key_cache = _Tensor(0x2150)
    mirror_value_cache = _Tensor(0x2160)
    scratch.full_cache = lambda layer_id: (key_cache, value_cache)
    if bf16_mirror:
        scratch.full_bf16_mirror_cache = lambda layer_id: (mirror_key_cache, mirror_value_cache)
    scratch.append_spans_for_layer = lambda layer_id: append_spans
    scratch.decode_spans_for_layer = lambda layer_id: decode_spans
    return scratch


def _patch_full_attention_primitives(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    def record(name: str, *, returns=None):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return returns

        return fake

    monkeypatch.setattr(qwen_runtime, "gguf_rmsnorm_bf16_f32_weight", record("rmsnorm"))
    monkeypatch.setattr(qwen_runtime, "launch_gguf_linear_triple", record("qkv_triple", returns=True))
    monkeypatch.setattr(qwen_runtime, "launch_gguf_linear_pair", record("kv_pair", returns=True))
    monkeypatch.setattr(qwen_runtime, "launch_gguf_linear", record("linear"))
    monkeypatch.setattr(
        qwen_runtime.Qwen35GGUFFullStackRunner,
        "_full_attn_qk_postprocess_fn",
        lambda _self: None,
    )
    monkeypatch.setattr(qwen_runtime, "qwen35_split_qgate_bf16", record("split_qgate"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", record("bf16_to_f32"))
    monkeypatch.setattr(
        qwen_runtime,
        "gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight",
        record("rope_key_f32"),
    )
    monkeypatch.setattr(qwen_runtime, "qwen35_write_paged_kv_mixed_value_bf16_spans", record("kv_write"))
    monkeypatch.setattr(qwen_runtime, "qwen35_write_paged_kv_int8_per_token_head_spans", record("kv_write_int8"))
    monkeypatch.setattr(qwen_runtime, "qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans", record("split_k_int8_gate"))
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans",
        record("split_k_gqa_gate"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans",
        record("split_k_gqa_parallel_reduce"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans",
        record("split_k_warp_gate"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_split_k_gate_bf16_spans",
        record("split_k_gate"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_context_bf16_spans",
        record("attention_context"),
    )

    def resolve_paged_attn_decode(**kwargs):
        calls.append(("resolve_attention_context_batch", (), kwargs))
        return record("attention_context_batch")

    monkeypatch.setattr(
        qwen_runtime,
        "resolve_paged_attn_decode",
        resolve_paged_attn_decode,
        raising=False,
    )
    monkeypatch.setattr(qwen_runtime, "qwen35_full_attn_gate_mul_bf16", record("attention_gate"))
    return calls


def test_long_context_routes_full_attention_through_split_k_gqa_gate(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(is_moe=True)
    scratch = _scratch(position=4095, max_positions=4096)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=4095, stream=5)

    names = [name for name, _, _ in calls]
    assert "rope_key_f32" in names
    assert "bf16_to_f32" in names
    assert "split_k_gqa_gate" in names
    assert "split_k_warp_gate" not in names
    assert "split_k_gate" not in names
    assert "attention_context" not in names
    assert "attention_gate" not in names

    split_args = next(args for name, args, _ in calls if name == "split_k_gqa_gate")
    assert split_args[:8] == (
        scratch.full_query.ptr,
        0x2110,
        0x2120,
        scratch.full_gate.ptr,
        scratch.full_gated.ptr,
        scratch.full_attn_split_partial.ptr,
        scratch.full_attn_split_m.ptr,
        scratch.full_attn_split_l.ptr,
    )
    assert split_args[9:18] == (256, scratch.full_attn_split_count, 256, 16, 2, 256, 256, 1, 256 ** -0.5)


def test_dense_24q4kv_grouped_gqa_is_gfx1151_4k_only(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(
        is_moe=False,
        backend="hip_gfx1151",
        head_count=24,
        head_count_kv=4,
        hidden_size=5120,
        block_count=64,
    )
    calls = _patch_full_attention_primitives(monkeypatch)

    below = _scratch(position=4094, max_positions=4096)
    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, below, position=4094, stream=5)
    below_names = [name for name, _, _ in calls]
    assert "split_k_gate" in below_names
    assert "split_k_gqa_gate" not in below_names

    calls.clear()
    admitted = _scratch(position=4095, max_positions=4096)
    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, admitted, position=4095, stream=5)
    admitted_names = [name for name, _, _ in calls]
    assert "split_k_gqa_gate" in admitted_names
    assert "split_k_gate" not in admitted_names
    split_args = next(args for name, args, _ in calls if name == "split_k_gqa_gate")
    assert split_args[9:18] == (
        256,
        admitted.full_attn_split_count,
        256,
        24,
        4,
        256,
        256,
        1,
        256 ** -0.5,
    )

    calls.clear()
    monkeypatch.setenv("HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX", "0")
    opt_out = _scratch(position=4095, max_positions=4096)
    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, opt_out, position=4095, stream=5)
    opt_out_names = [name for name, _, _ in calls]
    assert "split_k_gate" in opt_out_names
    assert "split_k_gqa_gate" not in opt_out_names
    assert "split_k_warp_gate" not in opt_out_names

    calls.clear()
    monkeypatch.setenv("HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX", "1")
    runner.backend = "hip_gfx1100"
    fallback = _scratch(position=4095, max_positions=4096)
    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, fallback, position=4095, stream=5)
    fallback_names = [name for name, _, _ in calls]
    assert "split_k_gate" in fallback_names
    assert "split_k_gqa_gate" not in fallback_names


def test_long_context_parallel_reduce_is_gfx1100_default(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(is_moe=True)
    scratch = _scratch(position=32767, max_positions=32768)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=32767, stream=5)

    names = [name for name, _, _ in calls]
    assert "split_k_gqa_parallel_reduce" in names
    assert "split_k_gqa_gate" not in names
    split_args = next(args for name, args, _ in calls if name == "split_k_gqa_parallel_reduce")
    assert split_args[9:18] == (256, scratch.full_attn_split_count, 256, 16, 2, 256, 256, 1, 256 ** -0.5)


def test_long_context_parallel_reduce_is_gfx1151_default_at_32k_only(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(is_moe=True, backend="hip_gfx1151")
    calls = _patch_full_attention_primitives(monkeypatch)

    below = _scratch(position=32766, max_positions=32768)
    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, below, position=32766, stream=5)
    below_names = [name for name, _, _ in calls]
    assert "split_k_gqa_gate" in below_names
    assert "split_k_gqa_parallel_reduce" not in below_names

    calls.clear()
    admitted = _scratch(position=32767, max_positions=32768)
    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, admitted, position=32767, stream=5)
    admitted_names = [name for name, _, _ in calls]
    assert "split_k_gqa_parallel_reduce" in admitted_names
    assert "split_k_gqa_gate" not in admitted_names


def test_long_context_parallel_reduce_keeps_explicit_opt_out(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    monkeypatch.setenv("HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE", "0")
    runner = _runner(is_moe=True)
    scratch = _scratch(position=32767, max_positions=32768)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=32767, stream=5)

    names = [name for name, _, _ in calls]
    assert "split_k_gqa_gate" in names
    assert "split_k_gqa_parallel_reduce" not in names


def test_int8_kv_routes_full_attention_through_int8_append_and_split_k(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1048576")
    runner = _runner(is_moe=True)
    scratch = _scratch(position=4095, max_positions=4096, kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=4095, stream=5)

    names = [name for name, _, _ in calls]
    assert "kv_write_int8" in names
    assert "kv_write" not in names
    assert "split_k_int8_gate" in names
    assert "split_k_gqa_gate" not in names
    assert "attention_context" not in names
    assert "attention_gate" not in names

    kv_args = next(args for name, args, _ in calls if name == "kv_write_int8")
    assert kv_args[:7] == (
        scratch.full_key.ptr,
        scratch.full_key_raw.ptr,
        0x2110,
        0x2120,
        0x2130,
        0x2140,
        scratch.append_spans,
    )
    split_args = next(args for name, args, _ in calls if name == "split_k_int8_gate")
    assert split_args[:10] == (
        scratch.full_query.ptr,
        0x2110,
        0x2120,
        0x2130,
        0x2140,
        scratch.full_gate.ptr,
        scratch.full_gated.ptr,
        scratch.full_attn_split_partial.ptr,
        scratch.full_attn_split_m.ptr,
        scratch.full_attn_split_l.ptr,
    )
    assert split_args[11:21] == (256, scratch.full_attn_split_count, 256, 16, 2, 256, 256, 1, 256 ** -0.5)


def test_int8_short_bf16_mirror_routes_decode_through_bf16_cache(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1048576")
    runner = _runner(is_moe=True)
    scratch = _scratch(
        position=4095,
        max_positions=4096,
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        bf16_mirror=True,
    )
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=4095, stream=5)

    names = [name for name, _, _ in calls]
    assert "kv_write_int8" in names
    assert "kv_write" in names
    assert "split_k_int8_gate" not in names
    assert "attention_context" in names
    assert "attention_gate" in names

    mirror_write_args = [args for name, args, _ in calls if name == "kv_write"][-1]
    assert mirror_write_args[:5] == (
        scratch.full_key.ptr,
        scratch.full_v.ptr,
        0x2150,
        0x2160,
        scratch.append_spans,
    )
    attn_args = next(args for name, args, _ in calls if name == "attention_context")
    assert attn_args[:5] == (
        scratch.full_query.ptr,
        0x2150,
        0x2160,
        scratch.full_attn_context.ptr,
        scratch.decode_spans,
    )


def test_short_context_keeps_unfused_full_attention_gate(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(is_moe=True)
    scratch = _scratch(position=511, max_positions=768)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=511, stream=5)

    names = [name for name, _, _ in calls]
    assert "rope_key_f32" in names
    assert "bf16_to_f32" in names
    assert "attention_context" in names
    assert "attention_context_batch" not in names
    assert "attention_gate" in names
    assert "split_k_gqa_gate" not in names
    assert "split_k_warp_gate" not in names
    assert "split_k_gate" not in names

    context_args = next(args for name, args, _ in calls if name == "attention_context")
    gate_args = next(args for name, args, _ in calls if name == "attention_gate")
    assert context_args[:5] == (scratch.full_query.ptr, 0x2110, 0x2120, scratch.full_attn_context.ptr, scratch.decode_spans)
    assert context_args[5:11] == (512, scratch.block_size, 16, 2, 256, 256 ** -0.5)
    assert gate_args[:4] == (scratch.full_attn_context.ptr, scratch.full_gate.ptr, scratch.full_gated.ptr, runner.q_width)


def test_gfx1151_short_context_uses_exact_fixed256_batch_leaf(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    monkeypatch.setenv("HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS", "256")
    runner = _runner(is_moe=True, backend="hip_gfx1151")
    scratch = _scratch(position=511, max_positions=768)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=511, stream=5)

    names = [name for name, _, _ in calls]
    assert "attention_context_batch" in names
    assert "attention_context" not in names
    assert "attention_gate" in names
    resolve_kwargs = next(kwargs for name, _, kwargs in calls if name == "resolve_attention_context_batch")
    assert resolve_kwargs == {
        "backend": "hip_gfx1151",
        "spans": scratch.decode_spans,
        "kind": "context_batch",
        "model_quant": "w4_paro",
    }
    context_args = next(args for name, args, _ in calls if name == "attention_context_batch")
    assert context_args[:5] == (
        scratch.full_query.ptr,
        0x2110,
        0x2120,
        scratch.full_attn_context.ptr,
        scratch.decode_spans,
    )
    assert context_args[5:12] == (1, 512, scratch.block_size, 16, 2, 256, 256 ** -0.5)


def test_gfx1151_fixed256_batch_leaf_stops_before_1024(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "8192")
    runner = _runner(is_moe=True, backend="hip_gfx1151")
    scratch = _scratch(position=1023, max_positions=1280)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=1023, stream=5)

    names = [name for name, _, _ in calls]
    assert "attention_context" in names
    assert "attention_context_batch" not in names
    assert "split_k_gqa_gate" not in names
    context_args = next(args for name, args, _ in calls if name == "attention_context")
    assert context_args[5] == 1024


def test_gfx1151_08b_graph_cap_routes_short_attention_through_split_k3(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(
        is_moe=False,
        backend="hip_gfx1151",
        head_count=8,
        hidden_size=1024,
        block_count=24,
    )
    scratch = _scratch(position=513, max_positions=768)
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(
        0,
        0x3000,
        0x4000,
        scratch,
        position=513,
        attention_max_context_len=641,
        stream=5,
    )

    names = [name for name, _, _ in calls]
    assert "split_k_gate" in names
    assert "attention_context" not in names
    assert "attention_context_batch" not in names
    assert "attention_gate" not in names
    split_args = next(args for name, args, _ in calls if name == "split_k_gate")
    assert split_args[9:18] == (256, 3, 256, 8, 2, 256, 256, 1, 256 ** -0.5)


def test_gfx1151_08b_short_split_policy_preserves_boundaries_and_fallbacks(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    runner = _runner(
        is_moe=False,
        backend="hip_gfx1151",
        head_count=8,
        hidden_size=1024,
        block_count=24,
    )
    cfg = runner.weights.config

    assert not qwen_runtime._use_gguf_short_full_attention_split_decode(
        cfg, backend="hip_gfx1151", block_size=256, active_context=513
    )
    assert qwen_runtime._use_gguf_short_full_attention_split_decode(
        cfg, backend="hip_gfx1151", block_size=256, active_context=514
    )
    assert qwen_runtime._use_gguf_short_full_attention_split_decode(
        cfg, backend="hip_gfx1151", block_size=256, active_context=641
    )
    assert not qwen_runtime._use_gguf_short_full_attention_split_decode(
        cfg, backend="hip_gfx1151", block_size=256, active_context=642
    )
    assert not qwen_runtime._use_gguf_short_full_attention_split_decode(
        cfg, backend="hip_gfx1100", block_size=256, active_context=576
    )

    runner.weights.config.head_count = 16
    assert not qwen_runtime._use_gguf_short_full_attention_split_decode(
        runner.weights.config,
        backend="hip_gfx1151",
        block_size=256,
        active_context=576,
    )

    runner.weights.config.head_count = 8
    runner.weights.config.value_length = 128
    assert not qwen_runtime._use_gguf_short_full_attention_split_decode(
        runner.weights.config,
        backend="hip_gfx1151",
        block_size=256,
        active_context=576,
    )

    runner.weights.config.value_length = 256
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "0")
    assert not qwen_runtime._use_gguf_short_full_attention_split_decode(
        runner.weights.config,
        backend="hip_gfx1151",
        block_size=256,
        active_context=576,
    )


def test_decode_repack_flag_does_not_change_split_k_routing(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    decisions: list[bool] = []

    for decode_repack in ("0", "1"):
        monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", decode_repack)
        runner = _runner(is_moe=True)
        scratch = _scratch(position=4095, max_positions=4096)
        calls = _patch_full_attention_primitives(monkeypatch)

        runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=4095, stream=5)

        names = [name for name, _, _ in calls]
        decisions.append("split_k_gqa_gate" in names)
        assert "attention_context" not in names
        assert "attention_gate" not in names

    assert decisions == [True, True]
