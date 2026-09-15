"""Vocabulary-row sharding for the TP2 output head.

The head is the single largest per-token weight read on the control rank
(248,320 x 5,120 Q6_K = 1.04 GB at the Qwen3.8-27B artifact), so the
doc's Packet 3 "measure head cost, shard only if worthwhile" question is
decided by splitting the head's vocabulary rows across the group: each rank
runs its own contiguous block-aligned row range concurrently, the host
concatenates the two f32 rows and takes the global argmax.

Exactness: the head GEMV is row-independent, so a per-rank row shard
produces bit-identical logit values to the replicated head; the global
greedy tie-break stays the concatenated row's first maximum index, which is
what ``np.argmax`` on the concatenation returns. Full-logit diagnostics
(``teacher_forced_logits``) read both ranks' shards and concatenate the same
values.

The split must respect Q6_K's 256-value block (one 210-byte record): rows
per rank are a whole number of blocks, and both resident layouts
(``gguf_q6_k_t16_v1`` and the qmicro-planar variant) repack in groups of 16
rows, so the shard row count must also be a multiple of 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

Q6_K_BLOCK_VALUES = 256
Q6_K_BLOCK_BYTES = 210
Q6_K_T16_ROWS_PER_TILE = 16


def q6_k_row_bytes(hidden: int) -> int:
    """One Q6_K weight row's source bytes: whole 256-value blocks per row."""

    if hidden % Q6_K_BLOCK_VALUES:
        raise HeadShardError(
            f"hidden {hidden} is not a whole number of Q6_K blocks"
        )
    return (hidden // Q6_K_BLOCK_VALUES) * Q6_K_BLOCK_BYTES


class HeadShardError(ValueError):
    """The head cannot be split for the requested degree."""


@dataclass(frozen=True)
class HeadShardPlan:
    """The immutable geometry of one head split."""

    world_size: int
    vocab_rows: int
    hidden: int
    layout: str
    quant_key: str
    rows_per_rank: int
    blocks_per_rank: int
    source_row_bytes: int

    def rank_row_start(self, rank: int) -> int:
        return int(rank) * self.rows_per_rank


def resolve_head_shard_plan(
    model_path: str | Path,
    *,
    world_size: int,
    backend: str = "hip_gfx1100",
) -> tuple[HeadShardPlan, Any]:
    """Resolve the head split from the engine's own materialization plan.

    Returns ``(plan, materialization)``. The layout is never guessed: it is
    the spec the runner's own chain resolved for ``root.lm_head``.
    """

    from hipengine.distributed.shard_weights import resolve_mlp_shard_context

    materialization, _context, _plans, config = resolve_mlp_shard_context(
        model_path, world_size=int(world_size), backend=backend
    )
    spec = materialization.root_specs["lm_head"]
    vocab_rows, hidden = (int(dim) for dim in spec.source.shape)
    layout = str(spec.layout)
    row_bytes = q6_k_row_bytes(hidden)
    world_size = int(world_size)
    if vocab_rows % Q6_K_BLOCK_VALUES:
        raise HeadShardError(
            f"vocab rows {vocab_rows} are not a whole number of "
            f"Q6_K {Q6_K_BLOCK_VALUES}-value blocks"
        )
    blocks = vocab_rows // Q6_K_BLOCK_VALUES
    if blocks % world_size:
        raise HeadShardError(
            f"the head's {blocks} Q6_K blocks do not split across "
            f"{world_size} ranks"
        )
    blocks_per_rank = blocks // world_size
    rows_per_rank = blocks_per_rank * Q6_K_BLOCK_VALUES
    if rows_per_rank % Q6_K_T16_ROWS_PER_TILE:
        raise HeadShardError(
            f"rows per rank {rows_per_rank} are not a multiple of the t16 "
            f"{Q6_K_T16_ROWS_PER_TILE}-row tile"
        )
    if "q6_k_t16" not in layout:
        raise HeadShardError(
            f"resident head layout {layout!r} is not a t16 family layout; "
            "raw layouts cannot be sliced rank-locally by this path"
        )
    return (
        HeadShardPlan(
            world_size=world_size,
            vocab_rows=vocab_rows,
            hidden=hidden,
            layout=layout,
            quant_key=str(spec.quant_key),
            rows_per_rank=rows_per_rank,
            blocks_per_rank=blocks_per_rank,
            source_row_bytes=row_bytes,
        ),
        materialization,
    )


def materialize_head_shards(
    model_path: str | Path,
    *,
    world_size: int,
    backend: str = "hip_gfx1100",
) -> tuple[HeadShardPlan, dict[int, dict[int, Any]]]:
    """Materialize each rank's contiguous head row shard in its resident layout.

    Returns ``(plan, {rank: {"tiles": payload}})``. Every rank's payload is
    derived from one contiguous block-aligned byte range of the source
    tensor, repacked to the plan's resolved layout by the same repack family
    the MLP shards use.
    """

    import numpy as np

    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf_t16 import (
        repack_gguf_q6_k_tile16,
        repack_gguf_q6_k_tile16_qmicro_planar,
    )

    plan, materialization = resolve_head_shard_plan(
        model_path, world_size=int(world_size), backend=backend
    )
    reader = GGUFReader(str(model_path))
    info = reader.info.tensor("output.weight")
    source = np.memmap(
        reader.path, dtype=np.uint8, mode="r",
        offset=info.data_offset, shape=(plan.vocab_rows * plan.source_row_bytes,),
    )
    repack = (
        repack_gguf_q6_k_tile16_qmicro_planar
        if "qmicro_planar" in plan.layout
        else repack_gguf_q6_k_tile16
    )
    rank_payloads: dict[int, dict[int, Any]] = {}
    for rank in range(plan.world_size):
        start = plan.rank_row_start(rank) * plan.source_row_bytes
        stop = start + plan.rows_per_rank * plan.source_row_bytes
        local = np.ascontiguousarray(source[start:stop]).reshape(
            1, plan.rows_per_rank, plan.source_row_bytes
        )
        tiles = np.ascontiguousarray(np.asarray(repack(local).tiles))
        rank_payloads[rank] = {"tiles": tiles.reshape(-1)}
    return plan, rank_payloads


def head_shard_bytes(plan: HeadShardPlan) -> int:
    """One rank's resident head shard bytes (t16 is byte-neutral)."""

    return plan.rows_per_rank * plan.source_row_bytes
