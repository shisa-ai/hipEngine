"""Unit tests for the TP2 head shard plan geometry (host-only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.distributed.head_shard import (
    HeadShardError,
    head_shard_bytes,
)
from hipengine.distributed import head_shard as hs


def _fake_materialization(vocab: int, hidden: int, layout: str):
    spec = SimpleNamespace(
        source=(vocab, hidden), layout=layout, quant_key=layout
    )
    # The real spec carries an ndarray source; give the fake one a shape.
    spec.source = SimpleNamespace(shape=(vocab, hidden))
    return SimpleNamespace(root_specs={"lm_head": spec})


def _resolve(monkeypatch, vocab: int, hidden: int, layout: str, world_size: int):
    import hipengine.distributed.shard_weights as sw

    monkeypatch.setattr(
        sw,
        "resolve_mlp_shard_context",
        lambda *a, **k: (_fake_materialization(vocab, hidden, layout), {}, None, None),
    )
    return hs.resolve_head_shard_plan("fake.gguf", world_size=world_size)


def test_plan_splits_block_aligned_even_rows(monkeypatch) -> None:
    # 970 Q6_K blocks split exactly across two ranks: 485 blocks = 124,160
    # rows, itself a multiple of the t16 16-row tile.
    plan, _materialization = _resolve(
        monkeypatch, 248_320, 5_120, "gguf_q6_k_t16_qmicro_planar_v1", 2
    )
    assert plan.rows_per_rank == 124_160
    assert plan.blocks_per_rank == 485
    assert plan.rank_row_start(0) == 0
    assert plan.rank_row_start(1) == 124_160
    assert plan.rank_row_start(1) + plan.rows_per_rank == plan.vocab_rows
    # t16 is byte-neutral: one rank's shard is rows x source-row bytes.
    assert head_shard_bytes(plan) == 124_160 * 4200


def test_plan_refuses_unsplittable_or_unaligned_geometry(monkeypatch) -> None:
    # An odd block count cannot split across two ranks.
    with pytest.raises(HeadShardError, match="do not split"):
        _resolve(monkeypatch, 248_320 + 256, 5_120, "gguf_q6_k_t16_v1", 2)
    # A non-t16 resident layout cannot be sliced rank-locally by this path.
    with pytest.raises(HeadShardError, match="not a t16 family layout"):
        _resolve(monkeypatch, 248_320, 5_120, "raw", 2)


def test_plan_preserves_resolved_layout(monkeypatch) -> None:
    # The layout comes from the materialization plan, never a table here.
    plan, _materialization = _resolve(
        monkeypatch, 248_320, 5_120, "gguf_q6_k_t16_v1", 2
    )
    assert plan.layout == "gguf_q6_k_t16_v1"
    assert plan.quant_key == "gguf_q6_k_t16_v1"
