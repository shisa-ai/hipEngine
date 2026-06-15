from __future__ import annotations

from dataclasses import dataclass

import pytest

from hipengine.speculative.gguf_mtp import (
    Qwen35GGUFMTPContext,
    Qwen35GGUFMTPDraftBatch,
    Qwen35GGUFMTPDraftRow,
    Qwen35GGUFMTPSeedRow,
)


@dataclass(frozen=True)
class _Contract:
    ready_for_mtp: bool = True
    rows: int = 1
    hidden_size: int = 8


@dataclass(frozen=True)
class _Seed:
    token_id: int
    position: int
    hidden_ptr: int
    hidden_contract: _Contract = _Contract()


class _TargetSession:
    def __init__(self, seed: _Seed) -> None:
        self.seed = seed
        self.calls: list[tuple[int, int]] = []

    def mtp_draft_seed(self, *, token_id: int, position: int) -> _Seed:
        self.calls.append((token_id, position))
        return _Seed(
            token_id=token_id,
            position=position,
            hidden_ptr=self.seed.hidden_ptr,
            hidden_contract=self.seed.hidden_contract,
        )


def test_gguf_mtp_context_captures_target_seed_and_builds_b1_row() -> None:
    target = _TargetSession(_Seed(token_id=0, position=0, hidden_ptr=0x1000))

    context = Qwen35GGUFMTPContext.from_target_seed(
        target,
        token_id=42,
        position=17,
        mtp_block=object(),
    )
    batch = context.build_b1_draft_batch(request_id=3, token_id=99)

    assert target.calls == [(42, 17)]
    assert context.pending_seed == Qwen35GGUFMTPSeedRow(
        token_id=42,
        position=17,
        hidden_ptr=0x1000,
        hidden_size=8,
        source="target",
    )
    assert batch.request_ids == (3,)
    assert batch.token_ids == (99,)
    assert batch.embedding_seed_ptrs == (0x1000,)
    assert batch.as_dict()["rows"] == [
        {
            "request_id": 3,
            "token_id": 99,
            "position": 18,
            "draft_depth": 1,
            "embedding_seed_ptr": 0x1000,
            "embedding_hidden_size": 8,
            "parent_token_id": 42,
            "parent_position": 17,
        }
    ]


def test_gguf_mtp_context_builds_multi_depth_draft_batch_from_seed_rows() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8, source="target"),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8, source="draft[0]"),
        Qwen35GGUFMTPSeedRow(token_id=12, position=7, hidden_ptr=0x3000, hidden_size=8, source="draft[1]"),
    )

    batch = context.build_draft_batch(request_id=4, token_ids=(101, 102, 103), seed_rows=seeds)

    assert batch.request_ids == (4,)
    assert batch.token_ids == (101, 102, 103)
    assert batch.embedding_seed_ptrs == (0x1000, 0x2000, 0x3000)
    assert [row.draft_depth for row in batch.rows] == [1, 2, 3]
    assert [row.position for row in batch.rows] == [6, 7, 8]
    assert [row.parent_token_id for row in batch.rows] == [10, 11, 12]
    assert [row.parent_position for row in batch.rows] == [5, 6, 7]


def test_gguf_mtp_context_requires_explicit_seed_rows_for_multi_depth_batch() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))

    with pytest.raises(ValueError, match="multi-depth GGUF MTP draft batches require explicit seed_rows"):
        context.build_draft_batch(request_id=0, token_ids=(11, 12))
    with pytest.raises(ValueError, match="seed_rows length"):
        context.build_draft_batch(
            request_id=0,
            token_ids=(11, 12),
            seed_rows=(Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),),
        )
    with pytest.raises(ValueError, match="share hidden_size"):
        context.build_draft_batch(
            request_id=0,
            token_ids=(11, 12),
            seed_rows=(
                Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
                Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=16),
            ),
        )


def test_gguf_mtp_context_accept_reseeds_from_verify_row_min_accepted() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    verify = context.record_verify_seeds(
        [
            _Seed(token_id=11, position=6, hidden_ptr=0x2000),
            _Seed(token_id=12, position=7, hidden_ptr=0x3000),
            _Seed(token_id=13, position=8, hidden_ptr=0x4000),
        ]
    )

    selected = context.accept(99)

    assert verify[2].source == "verify[2]"
    assert selected == verify[2]
    assert context.pending_seed == verify[2]
    assert context.build_b1_draft_batch(request_id=0, token_id=20).rows[0].position == 9


def test_gguf_mtp_context_rejects_unready_or_multirow_seed_contract() -> None:
    with pytest.raises(ValueError, match="ready fp32 hidden contract"):
        Qwen35GGUFMTPSeedRow.from_seed(
            _Seed(token_id=1, position=1, hidden_ptr=1, hidden_contract=_Contract(ready_for_mtp=False))
        )
    with pytest.raises(ValueError, match="one hidden seed row"):
        Qwen35GGUFMTPSeedRow.from_seed(
            _Seed(token_id=1, position=1, hidden_ptr=1, hidden_contract=_Contract(rows=2))
        )


def test_gguf_mtp_context_requires_seed_before_b1_batch_or_accept() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())

    with pytest.raises(RuntimeError, match="pending GGUF MTP hidden seed"):
        context.build_b1_draft_batch(request_id=0, token_id=1)
    with pytest.raises(RuntimeError, match="record_verify_seeds"):
        context.accept(0)


def test_gguf_mtp_draft_batch_validates_embedding_seed_rows() -> None:
    with pytest.raises(ValueError, match="embedding_seed_ptr"):
        Qwen35GGUFMTPDraftRow(
            request_id=0,
            token_id=1,
            position=2,
            draft_depth=1,
            embedding_seed_ptr=0,
            embedding_hidden_size=8,
            parent_token_id=0,
            parent_position=1,
        )
    row = Qwen35GGUFMTPDraftRow(
        request_id=0,
        token_id=1,
        position=2,
        draft_depth=1,
        embedding_seed_ptr=1,
        embedding_hidden_size=8,
        parent_token_id=0,
        parent_position=1,
    )
    with pytest.raises(ValueError, match="duplicate request_id/draft_depth"):
        Qwen35GGUFMTPDraftBatch(rows=(row, row))
