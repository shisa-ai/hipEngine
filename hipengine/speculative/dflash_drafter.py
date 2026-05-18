"""Torch-free native DFlash drafter root/query scaffolding.

The z-lab DFlash drafter consumes concatenated target hidden taps, projects them
through ``fc + hidden_norm``, evaluates draft root/query rows, then applies the
target lm-head to rows ``1:block_size``.  This module owns the torch-free ABI for
that path: fixed root/query request metadata, device projection helper, and
candidate-only ``DraftBatch`` emission from compact top-k outputs.  Draft
context-KV materialization and the full DFlash decoder block kernels are wired in
later phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_bf16
from hipengine.loading.dflash import DFlashDraftConfig, DFlashDrafterDeviceWeights
from hipengine.speculative.dflash import DFlashDraftRequest, compile_dflash_chain
from hipengine.speculative.interfaces import DraftBatch


@dataclass(frozen=True, slots=True)
class DFlashRootQueryRequest:
    """One live request at the DFlash root/query boundary."""

    request_id: int
    root_token: int
    root_position: int
    context_length: int
    target_hidden_rows: Tensor

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self.root_token < 0:
            raise ValueError("root_token must be non-negative")
        if self.root_position < 0:
            raise ValueError("root_position must be non-negative")
        if self.context_length < 0:
            raise ValueError("context_length must be non-negative")
        if self.target_hidden_rows.ndim != 2:
            raise ValueError("target_hidden_rows must have shape (context_length, target_hidden_concat_size)")
        if self.target_hidden_rows.shape[0] != self.context_length:
            raise ValueError("target_hidden_rows first dimension must match context_length")
        if self.target_hidden_rows.dtype != DType.BF16:
            raise ValueError("target_hidden_rows must be BF16 bits")


@dataclass(frozen=True, slots=True)
class DFlashRootQueryPlan:
    """Fixed-shape root + mask/query token plan for one DFlash batch."""

    request_ids: tuple[int, ...]
    root_tokens: tuple[int, ...]
    root_positions: tuple[int, ...]
    context_lengths: tuple[int, ...]
    block_size: int
    mask_token_id: int
    noise_token_ids: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    target_hidden_concat_size: int

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @classmethod
    def from_requests(
        cls,
        requests: Sequence[DFlashRootQueryRequest],
        *,
        config: DFlashDraftConfig,
    ) -> "DFlashRootQueryPlan":
        reqs = tuple(requests)
        if not reqs:
            raise ValueError("at least one DFlash root/query request is required")
        concat = int(config.target_hidden_concat_size)
        noise_rows: list[tuple[int, ...]] = []
        positions: list[tuple[int, ...]] = []
        for req in reqs:
            if req.target_hidden_rows.shape[1] != concat:
                raise ValueError(
                    f"target hidden concat size {req.target_hidden_rows.shape[1]} does not match config {concat}"
                )
            noise = [int(config.mask_token_id)] * int(config.block_size)
            noise[0] = int(req.root_token)
            noise_rows.append(tuple(noise))
            start = int(req.root_position)
            positions.append(tuple(range(start, start + int(config.block_size))))
        return cls(
            request_ids=tuple(int(req.request_id) for req in reqs),
            root_tokens=tuple(int(req.root_token) for req in reqs),
            root_positions=tuple(int(req.root_position) for req in reqs),
            context_lengths=tuple(int(req.context_length) for req in reqs),
            block_size=int(config.block_size),
            mask_token_id=int(config.mask_token_id),
            noise_token_ids=tuple(noise_rows),
            position_ids=tuple(positions),
            target_hidden_concat_size=concat,
        )


def prepare_dflash_noise_inputs_bf16(
    root_tokens: Tensor,
    root_positions: Tensor,
    embed_tokens: Tensor,
    noise_token_ids: Tensor,
    position_ids: Tensor,
    noise_embeddings: Tensor,
    *,
    block_size: int,
    mask_token_id: int,
    stream: int = 0,
    library: object | None = None,
    threads: int = 256,
) -> Tensor:
    """Materialize root+mask token ids, positions, and BF16 embeddings on device.

    ``root_tokens`` and ``root_positions`` are compact int32 vectors of length
    ``request_count``.  Outputs are request-major slabs with shape
    ``[request_count, block_size]`` for ids/positions and
    ``[request_count, block_size, hidden_size]`` for embeddings.  The first row
    per request uses the root token; the remaining rows use ``mask_token_id``.
    ``embed_tokens`` may be BF16 (copied as bits) or FP16 (converted to BF16),
    matching the current shisa packed target artifact.
    """

    request_count, hidden_size, vocab_size = _validate_noise_input_tensors(
        root_tokens,
        root_positions,
        embed_tokens,
        noise_token_ids,
        position_ids,
        noise_embeddings,
        block_size=block_size,
    )
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_prepare_noise_inputs_bf16_i32,
        dflash_prepare_noise_inputs_f16_to_bf16_i32,
    )

    prepare = dflash_prepare_noise_inputs_bf16_i32 if embed_tokens.dtype == DType.BF16 else dflash_prepare_noise_inputs_f16_to_bf16_i32
    prepare(
        root_tokens.ptr,
        root_positions.ptr,
        embed_tokens.ptr,
        noise_token_ids.ptr,
        position_ids.ptr,
        noise_embeddings.ptr,
        request_count,
        block_size,
        hidden_size,
        vocab_size,
        mask_token_id,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return noise_embeddings


def project_dflash_target_hidden_bf16(
    target_hidden_concat: Tensor,
    out_projected: Tensor,
    scratch: Tensor,
    weights: DFlashDrafterDeviceWeights,
    *,
    stream: int = 0,
    libraries: dict[str, object] | None = None,
    threads: int = 256,
) -> Tensor:
    """Run native DFlash ``fc + hidden_norm`` over target hidden taps.

    ``target_hidden_concat`` has shape ``[context_rows, len(target_layer_ids) *
    target_hidden_size]`` and BF16 storage.  ``scratch`` and ``out_projected``
    both have shape ``[context_rows, hidden_size]`` and BF16 storage.  The helper
    is intentionally only the projection boundary; draft context-KV
    materialization and decoder block execution are separate follow-up work.
    """

    config = weights.config
    rows = _validate_projection_tensors(target_hidden_concat, out_projected, scratch, config)
    dense_lib = None if libraries is None else libraries.get("dense")
    norm_lib = None if libraries is None else libraries.get("norm")
    dense_gemv_out_bf16(
        target_hidden_concat.ptr,
        weights.tensor("fc.weight").ptr,
        scratch.ptr,
        rows,
        config.target_hidden_concat_size,
        config.hidden_size,
        threads=threads,
        stream=stream,
        library=dense_lib,
    )
    dflash_rmsnorm_bf16(
        scratch,
        weights.tensor("hidden_norm.weight"),
        out_projected,
        eps=config.rms_norm_eps,
        stream=stream,
        library=norm_lib,
        threads=threads,
    )
    return out_projected


def dflash_add_bf16(a: Tensor, b: Tensor, out: Tensor, *, stream: int = 0, library: object | None = None, threads: int = 256) -> Tensor:
    """Elementwise BF16 residual add for DFlash block wiring."""

    elements = _validate_same_shape_bf16("dflash_add", a, b, out)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_add_bf16 as _launch_add

    _launch_add(a.ptr, b.ptr, out.ptr, elements, threads=threads, stream=stream, library=library)  # type: ignore[arg-type]
    return out


def dflash_concat_rows(context: Tensor, query: Tensor, out: Tensor, *, stream: int = 0, library: object | None = None, threads: int = 256) -> Tensor:
    """Concatenate context+query rows on device for DFlash K/V assembly."""

    batch, context_len, query_len, features, dtype = _validate_concat_tensors(context, query, out)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_concat_rows_bf16, dflash_concat_rows_f32

    launcher = dflash_concat_rows_bf16 if dtype == DType.BF16 else dflash_concat_rows_f32
    launcher(
        context.ptr,
        query.ptr,
        out.ptr,
        batch,
        context_len,
        query_len,
        features,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return out


def dflash_rmsnorm_bf16(
    hidden: Tensor,
    weight: Tensor,
    out: Tensor,
    *,
    eps: float = 1.0e-6,
    stream: int = 0,
    library: object | None = None,
    threads: int = 128,
) -> Tensor:
    """Apply standard DFlash RMSNorm with direct BF16 weight scaling."""

    rows, hidden_size = _validate_rmsnorm_tensors(hidden, weight, out)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_rmsnorm_bf16 as _launch_rmsnorm

    _launch_rmsnorm(
        hidden.ptr,
        weight.ptr,
        out.ptr,
        rows,
        hidden_size,
        eps=eps,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return out


def project_dflash_bf16_to_bf16(
    hidden: Tensor,
    weight: Tensor,
    out: Tensor,
    *,
    stream: int = 0,
    library: object | None = None,
    threads: int = 128,
) -> Tensor:
    """Run a BF16 drafter projection and keep BF16 output storage."""

    rows, in_features, out_features = _validate_dense_projection_tensors(hidden, weight, out, out_dtype=DType.BF16)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_dense_bf16_to_bf16

    dflash_dense_bf16_to_bf16(
        hidden.ptr,
        weight.ptr,
        out.ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return out


def project_dflash_bf16_to_f32(
    hidden: Tensor,
    weight: Tensor,
    out: Tensor,
    *,
    stream: int = 0,
    library: object | None = None,
    threads: int = 128,
) -> Tensor:
    """Run a BF16 drafter projection and keep FP32 logits/heads for downstream math."""

    rows, in_features, out_features = _validate_dense_projection_tensors(hidden, weight, out, out_dtype=DType.FP32)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_dense_bf16_to_f32

    dflash_dense_bf16_to_f32(
        hidden.ptr,
        weight.ptr,
        out.ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return out


def project_dflash_qkv_bf16_mixed(
    hidden: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    q_out: Tensor,
    k_out: Tensor,
    v_out: Tensor,
    *,
    stream: int = 0,
    library: object | None = None,
    threads: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run fused query-side DFlash Q/K/V projections.

    Equivalent to the unfused GPU sequence
    :func:`project_dflash_bf16_to_f32` for Q and K plus
    :func:`project_dflash_bf16_to_bf16` for V.  The fused wrapper exists only
    for the native drafter and preserves the unfused fallback contract.
    """

    rows, in_features, q_features, kv_features = _validate_qkv_projection_tensors(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        q_out,
        k_out,
        v_out,
    )
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_qkv_proj_bf16_mixed

    dflash_qkv_proj_bf16_mixed(
        hidden.ptr,
        q_weight.ptr,
        k_weight.ptr,
        v_weight.ptr,
        q_out.ptr,
        k_out.ptr,
        v_out.ptr,
        rows,
        in_features,
        q_features,
        kv_features,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return q_out, k_out, v_out


def dflash_head_rmsnorm_rotary_f32(
    query: Tensor,
    key: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_table: Tensor,
    sin_table: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    query_out: Tensor,
    key_out: Tensor,
    *,
    eps: float = 1.0e-6,
    stream: int = 0,
    library: object | None = None,
    threads: int = 128,
) -> tuple[Tensor, Tensor]:
    """Apply direct-weight Q/K head RMSNorm plus rotary for DFlash attention."""

    shape = _validate_head_rotary_tensors(
        query,
        key,
        q_weight,
        k_weight,
        cos_table,
        sin_table,
        query_positions,
        key_positions,
        query_out,
        key_out,
    )
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_head_rmsnorm_rotary_f32 as _launch

    _launch(
        query.ptr,
        key.ptr,
        q_weight.ptr,
        k_weight.ptr,
        cos_table.ptr,
        sin_table.ptr,
        query_positions.ptr,
        key_positions.ptr,
        query_out.ptr,
        key_out.ptr,
        *shape,
        eps=eps,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return query_out, key_out


def dflash_gqa_attention_bf16(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    *,
    scale: float | None = None,
    stream: int = 0,
    library: object | None = None,
    threads: int = 128,
) -> Tensor:
    """Run correctness-first non-causal DFlash GQA attention.

    ``query`` has shape ``[batch, query_len, q_heads, head_dim]`` with F32
    storage. ``key`` has shape ``[batch, kv_len, kv_heads, head_dim]`` with F32
    storage and ``value`` has the same tail shape with BF16 storage. ``out`` is
    BF16 ``[batch, query_len, q_heads, head_dim]``.
    """

    batch, query_len, kv_len, q_heads, kv_heads, head_dim = _validate_attention_tensors(query, key, value, out)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_gqa_attention_f32_bf16

    dflash_gqa_attention_f32_bf16(
        query.ptr,
        key.ptr,
        value.ptr,
        out.ptr,
        batch,
        query_len,
        kv_len,
        q_heads,
        kv_heads,
        head_dim,
        scale=scale,
        threads=threads,
        stream=stream,
        library=library,  # type: ignore[arg-type]
    )
    return out


def draft_batch_from_topk(
    plan: DFlashRootQueryPlan,
    topk_token_ids: Sequence[Sequence[int]],
    *,
    candidate_budget: int,
    topk_rank: int = 0,
    pad_token_id: int = 0,
) -> DraftBatch:
    """Compile candidate-only DFlash chain rows from compact top-k tokens.

    ``topk_token_ids`` is request-major over draft rows and excludes the root
    row, matching the DFlash lm-head rows ``hidden[1:block_size]``.  ``topk_rank``
    selects the greedy chain rank (normally 0).  Root rows remain absent here and
    are inserted only by ``TargetVerifyBatch.from_draft()``.
    """

    if len(topk_token_ids) != plan.batch_size:
        raise ValueError("topk_token_ids must have one row per request")
    requests: list[DFlashDraftRequest] = []
    for idx, rows in enumerate(topk_token_ids):
        row_tokens: list[int] = []
        for row in rows[:candidate_budget]:
            if isinstance(row, Sequence) and not isinstance(row, (bytes, bytearray)):
                if topk_rank < 0 or topk_rank >= len(row):
                    raise ValueError("topk_rank is outside a top-k row")
                token = int(row[topk_rank])
            else:
                token = int(row)  # type: ignore[arg-type]
            if token < 0:
                raise ValueError("draft token ids must be non-negative")
            row_tokens.append(token)
        requests.append(
            DFlashDraftRequest(
                request_id=plan.request_ids[idx],
                root_position=plan.root_positions[idx],
                candidate_tokens=tuple(row_tokens),
                active_count=len(row_tokens),
            )
        )
    return compile_dflash_chain(requests, candidate_budget=candidate_budget, pad_token_id=pad_token_id)


def _validate_same_shape_bf16(name: str, a: Tensor, b: Tensor, out: Tensor) -> int:
    if a.shape != b.shape or a.shape != out.shape:
        raise ValueError(f"{name} tensors must share shape")
    if a.dtype != DType.BF16 or b.dtype != DType.BF16 or out.dtype != DType.BF16:
        raise ValueError(f"{name} tensors must use BF16 storage")
    if b.device != a.device or out.device != a.device:
        raise ValueError(f"{name} tensors must live on the same device")
    elements = 1
    for dim in a.shape:
        elements *= int(dim)
    return elements


def _validate_concat_tensors(context: Tensor, query: Tensor, out: Tensor) -> tuple[int, int, int, int, DType]:
    if context.ndim != 3 or query.ndim != 3 or out.ndim != 3:
        raise ValueError("DFlash concat tensors must be rank-3 [batch, rows, features]")
    batch, context_len, features = context.shape
    q_batch, query_len, q_features = query.shape
    if (q_batch, q_features) != (batch, features):
        raise ValueError("query rows must match context batch/features")
    if out.shape != (batch, context_len + query_len, features):
        raise ValueError("concat output shape must be [batch, context_len + query_len, features]")
    if context.dtype != query.dtype or out.dtype != context.dtype:
        raise ValueError("concat tensors must share dtype")
    if context.dtype not in {DType.BF16, DType.FP32}:
        raise ValueError("concat dtype must be BF16 or FP32")
    if query.device != context.device or out.device != context.device:
        raise ValueError("concat tensors must live on the same device")
    return int(batch), int(context_len), int(query_len), int(features), context.dtype


def _validate_rmsnorm_tensors(hidden: Tensor, weight: Tensor, out: Tensor) -> tuple[int, int]:
    if hidden.ndim != 2 or out.ndim != 2 or weight.ndim != 1:
        raise ValueError("DFlash RMSNorm tensors must be rank-2 hidden/out plus rank-1 weight")
    rows, hidden_size = hidden.shape
    if weight.shape != (hidden_size,):
        raise ValueError("RMSNorm weight shape must match hidden_size")
    if out.shape != hidden.shape:
        raise ValueError("RMSNorm output shape must match hidden shape")
    for name, tensor in (("hidden", hidden), ("weight", weight), ("out", out)):
        if tensor.dtype != DType.BF16:
            raise ValueError(f"{name} must use BF16 storage")
        if tensor.device != hidden.device:
            raise ValueError(f"{name} must live on the hidden tensor device")
    return int(rows), int(hidden_size)


def dflash_silu_mul_bf16(gate: Tensor, up: Tensor, out: Tensor, *, stream: int = 0, library: object | None = None, threads: int = 256) -> Tensor:
    """Elementwise BF16 SiLU(gate) * up for DFlash MLP wiring."""

    elements = _validate_same_shape_bf16("dflash_silu_mul", gate, up, out)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_silu_mul_bf16 as _launch_silu

    _launch_silu(gate.ptr, up.ptr, out.ptr, elements, threads=threads, stream=stream, library=library)  # type: ignore[arg-type]
    return out


def _validate_dense_projection_tensors(hidden: Tensor, weight: Tensor, out: Tensor, *, out_dtype: DType) -> tuple[int, int, int]:
    if hidden.ndim != 2 or weight.ndim != 2 or out.ndim != 2:
        raise ValueError("DFlash dense projection tensors must be rank-2")
    rows, in_features = hidden.shape
    out_features, weight_in = weight.shape
    if weight_in != in_features:
        raise ValueError("projection weight input dimension must match hidden rows")
    if out.shape != (rows, out_features):
        raise ValueError(f"projection output must have shape {(rows, out_features)}")
    if hidden.dtype != DType.BF16 or weight.dtype != DType.BF16:
        raise ValueError("projection hidden and weight tensors must use BF16 storage")
    if out.dtype != out_dtype:
        raise ValueError(f"projection output must use {out_dtype.name} storage")
    if weight.device != hidden.device or out.device != hidden.device:
        raise ValueError("projection tensors must live on the same device")
    return int(rows), int(in_features), int(out_features)


def _validate_qkv_projection_tensors(
    hidden: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    q_out: Tensor,
    k_out: Tensor,
    v_out: Tensor,
) -> tuple[int, int, int, int]:
    if hidden.ndim != 2 or q_weight.ndim != 2 or k_weight.ndim != 2 or v_weight.ndim != 2:
        raise ValueError("DFlash fused QKV projection tensors must be rank-2")
    if q_out.ndim != 2 or k_out.ndim != 2 or v_out.ndim != 2:
        raise ValueError("DFlash fused QKV projection outputs must be rank-2")
    rows, in_features = hidden.shape
    q_features, q_in = q_weight.shape
    k_features, k_in = k_weight.shape
    v_features, v_in = v_weight.shape
    if q_in != in_features or k_in != in_features or v_in != in_features:
        raise ValueError("Q/K/V projection weight input dimension must match hidden rows")
    if k_features != v_features:
        raise ValueError("K and V projection output dimensions must match")
    if q_out.shape != (rows, q_features):
        raise ValueError(f"Q projection output must have shape {(rows, q_features)}")
    if k_out.shape != (rows, k_features):
        raise ValueError(f"K projection output must have shape {(rows, k_features)}")
    if v_out.shape != (rows, v_features):
        raise ValueError(f"V projection output must have shape {(rows, v_features)}")
    for name, tensor in (("hidden", hidden), ("q_weight", q_weight), ("k_weight", k_weight), ("v_weight", v_weight)):
        if tensor.dtype != DType.BF16:
            raise ValueError(f"{name} must use BF16 storage")
    if q_out.dtype != DType.FP32 or k_out.dtype != DType.FP32:
        raise ValueError("Q/K projection outputs must use FP32 storage")
    if v_out.dtype != DType.BF16:
        raise ValueError("V projection output must use BF16 storage")
    for name, tensor in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
        ("q_out", q_out),
        ("k_out", k_out),
        ("v_out", v_out),
    ):
        if tensor.device != hidden.device:
            raise ValueError(f"{name} must live on the hidden tensor device")
    return int(rows), int(in_features), int(q_features), int(k_features)


def _validate_head_rotary_tensors(
    query: Tensor,
    key: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_table: Tensor,
    sin_table: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    query_out: Tensor,
    key_out: Tensor,
) -> tuple[int, int, int, int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or query_out.ndim != 4 or key_out.ndim != 4:
        raise ValueError("DFlash head rotary Q/K tensors must be rank-4")
    if query.dtype != DType.FP32 or key.dtype != DType.FP32 or query_out.dtype != DType.FP32 or key_out.dtype != DType.FP32:
        raise ValueError("DFlash head rotary Q/K tensors must use FP32 storage")
    batch, query_len, q_heads, head_dim = query.shape
    key_batch, kv_len, kv_heads, key_dim = key.shape
    if (key_batch, key_dim) != (batch, head_dim):
        raise ValueError("key batch/head_dim must match query")
    if query_out.shape != query.shape or key_out.shape != key.shape:
        raise ValueError("head rotary outputs must match query/key input shapes")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError("q/k norm weights must have shape (head_dim,)")
    if q_weight.dtype != DType.BF16 or k_weight.dtype != DType.BF16:
        raise ValueError("q/k norm weights must use BF16 storage")
    if cos_table.ndim != 2 or sin_table.shape != cos_table.shape:
        raise ValueError("cos/sin tables must be matching rank-2 tensors")
    max_positions, rotary_dim = cos_table.shape
    if cos_table.dtype != DType.FP32 or sin_table.dtype != DType.FP32:
        raise ValueError("cos/sin tables must use FP32 storage")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    if query_positions.shape != (batch, query_len) or key_positions.shape != (batch, kv_len):
        raise ValueError("position tensors must match query/key row shapes")
    if query_positions.dtype != DType.INT32 or key_positions.dtype != DType.INT32:
        raise ValueError("position tensors must use int32 storage")
    for name, tensor in (
        ("key", key),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_table", cos_table),
        ("sin_table", sin_table),
        ("query_positions", query_positions),
        ("key_positions", key_positions),
        ("query_out", query_out),
        ("key_out", key_out),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must live on the query device")
    return int(batch), int(query_len), int(kv_len), int(q_heads), int(kv_heads), int(head_dim), int(rotary_dim), int(max_positions)


def _validate_attention_tensors(query: Tensor, key: Tensor, value: Tensor, out: Tensor) -> tuple[int, int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or out.ndim != 4:
        raise ValueError("DFlash attention tensors must be rank-4")
    if query.dtype != DType.FP32 or key.dtype != DType.FP32:
        raise ValueError("query and key must use FP32 storage")
    if value.dtype != DType.BF16 or out.dtype != DType.BF16:
        raise ValueError("value and output must use BF16 storage")
    batch, query_len, q_heads, head_dim = query.shape
    key_batch, kv_len, kv_heads, key_dim = key.shape
    if (key_batch, key_dim) != (batch, head_dim):
        raise ValueError("key batch/head_dim must match query")
    if value.shape != (batch, kv_len, kv_heads, head_dim):
        raise ValueError("value shape must match key shape")
    if out.shape != (batch, query_len, q_heads, head_dim):
        raise ValueError("out shape must match query shape")
    if q_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    for name, tensor in (("key", key), ("value", value), ("out", out)):
        if tensor.device != query.device:
            raise ValueError(f"{name} must live on the query device")
    return int(batch), int(query_len), int(kv_len), int(q_heads), int(kv_heads), int(head_dim)


def _validate_noise_input_tensors(
    root_tokens: Tensor,
    root_positions: Tensor,
    embed_tokens: Tensor,
    noise_token_ids: Tensor,
    position_ids: Tensor,
    noise_embeddings: Tensor,
    *,
    block_size: int,
) -> tuple[int, int, int]:
    if root_tokens.ndim != 1 or root_positions.ndim != 1:
        raise ValueError("root token and position tensors must be rank-1")
    if root_tokens.shape != root_positions.shape:
        raise ValueError("root token and position tensors must have matching shape")
    if root_tokens.dtype != DType.INT32 or root_positions.dtype != DType.INT32:
        raise ValueError("root token and position tensors must be int32")
    if embed_tokens.ndim != 2 or embed_tokens.dtype not in {DType.BF16, DType.FP16}:
        raise ValueError("embed_tokens must have BF16 or FP16 shape (vocab_size, hidden_size)")
    request_count = root_tokens.shape[0]
    vocab_size, hidden_size = embed_tokens.shape
    expected_ids = (request_count, int(block_size))
    expected_embeddings = (request_count, int(block_size), hidden_size)
    for name, tensor in (("noise_token_ids", noise_token_ids), ("position_ids", position_ids)):
        if tensor.shape != expected_ids:
            raise ValueError(f"{name} must have shape {expected_ids}")
        if tensor.dtype != DType.INT32:
            raise ValueError(f"{name} must be int32")
        if tensor.device != root_tokens.device:
            raise ValueError(f"{name} must live on the root token device")
    if noise_embeddings.shape != expected_embeddings:
        raise ValueError(f"noise_embeddings must have shape {expected_embeddings}")
    if noise_embeddings.dtype != DType.BF16:
        raise ValueError("noise_embeddings must use BF16 storage")
    for name, tensor in (("root_positions", root_positions), ("embed_tokens", embed_tokens), ("noise_embeddings", noise_embeddings)):
        if tensor.device != root_tokens.device:
            raise ValueError(f"{name} must live on the root token device")
    return int(request_count), int(hidden_size), int(vocab_size)


def _validate_projection_tensors(
    target_hidden_concat: Tensor,
    out_projected: Tensor,
    scratch: Tensor,
    config: DFlashDraftConfig,
) -> int:
    if target_hidden_concat.ndim != 2:
        raise ValueError("target_hidden_concat must be rank-2")
    rows, concat = target_hidden_concat.shape
    expected = (rows, config.hidden_size)
    if concat != config.target_hidden_concat_size:
        raise ValueError(
            f"target hidden concat size {concat} does not match config {config.target_hidden_concat_size}"
        )
    for name, tensor in (("target_hidden_concat", target_hidden_concat), ("out_projected", out_projected), ("scratch", scratch)):
        if tensor.dtype != DType.BF16:
            raise ValueError(f"{name} must use BF16 storage")
        if tensor.device != target_hidden_concat.device:
            raise ValueError(f"{name} must live on the same device as target_hidden_concat")
    if out_projected.shape != expected:
        raise ValueError(f"out_projected must have shape {expected}")
    if scratch.shape != expected:
        raise ValueError(f"scratch must have shape {expected}")
    return int(rows)


__all__ = [
    "DFlashRootQueryPlan",
    "DFlashRootQueryRequest",
    "dflash_add_bf16",
    "dflash_concat_rows",
    "dflash_gqa_attention_bf16",
    "dflash_head_rmsnorm_rotary_f32",
    "dflash_rmsnorm_bf16",
    "draft_batch_from_topk",
    "project_dflash_bf16_to_bf16",
    "project_dflash_bf16_to_f32",
    "project_dflash_qkv_bf16_mixed",
    "prepare_dflash_noise_inputs_bf16",
    "project_dflash_target_hidden_bf16",
    "dflash_silu_mul_bf16",
]
