"""GGUF-native MTP context scaffolding.

This module is intentionally metadata/state only.  It does not run kernels or
allocate KV buffers yet; it fixes the target-attached state machine needed before
the native GGUF NextN runtime can be wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class _HiddenSeedContractLike(Protocol):
    ready_for_mtp: bool
    rows: int
    hidden_size: int


class _DraftSeedLike(Protocol):
    token_id: int
    position: int
    hidden_ptr: int
    hidden_contract: _HiddenSeedContractLike


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPSeedRow:
    """Ready fp32 post-output_norm seed row for GGUF MTP draft work."""

    token_id: int
    position: int
    hidden_ptr: int
    hidden_size: int
    source: str = "target"

    @classmethod
    def from_seed(cls, seed: _DraftSeedLike, *, source: str = "target") -> "Qwen35GGUFMTPSeedRow":
        contract = seed.hidden_contract
        if not contract.ready_for_mtp:
            raise ValueError("GGUF MTP seed requires a ready fp32 hidden contract")
        if int(contract.rows) != 1:
            raise ValueError("GGUF MTP context currently expects one hidden seed row")
        return cls(
            token_id=int(seed.token_id),
            position=int(seed.position),
            hidden_ptr=int(seed.hidden_ptr),
            hidden_size=int(contract.hidden_size),
            source=str(source),
        )

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("seed token_id must be non-negative")
        if self.position < 0:
            raise ValueError("seed position must be non-negative")
        if self.hidden_ptr <= 0:
            raise ValueError("seed hidden_ptr must be a non-zero device pointer")
        if self.hidden_size <= 0:
            raise ValueError("seed hidden_size must be positive")
        if not self.source:
            raise ValueError("seed source must be non-empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "position": self.position,
            "hidden_ptr": self.hidden_ptr,
            "hidden_size": self.hidden_size,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPDraftRow:
    """One GGUF MTP draft row carrying both token id and embedding seed ptr."""

    request_id: int
    token_id: int
    position: int
    draft_depth: int
    embedding_seed_ptr: int
    embedding_hidden_size: int
    parent_token_id: int
    parent_position: int

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self.token_id < 0 or self.parent_token_id < 0:
            raise ValueError("token ids must be non-negative")
        if self.position < 0 or self.parent_position < 0:
            raise ValueError("positions must be non-negative")
        if self.position <= self.parent_position:
            raise ValueError("draft position must be after parent position")
        if self.draft_depth <= 0:
            raise ValueError("draft_depth must be positive")
        if self.embedding_seed_ptr <= 0:
            raise ValueError("embedding_seed_ptr must be a non-zero device pointer")
        if self.embedding_hidden_size <= 0:
            raise ValueError("embedding_hidden_size must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "token_id": self.token_id,
            "position": self.position,
            "draft_depth": self.draft_depth,
            "embedding_seed_ptr": self.embedding_seed_ptr,
            "embedding_hidden_size": self.embedding_hidden_size,
            "parent_token_id": self.parent_token_id,
            "parent_position": self.parent_position,
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPDraftBatch:
    """GGUF-specific draft rows before conversion to device verify buffers.

    The shared PARO ``DraftBatch`` only carries candidate token metadata.  GGUF
    MTP also needs each row to carry an embedding/hidden seed pointer, matching
    llama.cpp's ``embd_nextn`` contract.
    """

    rows: tuple[Qwen35GGUFMTPDraftRow, ...]
    mode: str = "gguf_mtp_nextn"

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("GGUF MTP draft batch must contain at least one row")
        if self.mode != "gguf_mtp_nextn":
            raise ValueError("GGUF MTP draft batch mode must be gguf_mtp_nextn")
        seen: set[tuple[int, int]] = set()
        for row in self.rows:
            key = (row.request_id, row.draft_depth)
            if key in seen:
                raise ValueError("duplicate request_id/draft_depth in GGUF MTP draft batch")
            seen.add(key)

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted({row.request_id for row in self.rows}))

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(row.token_id for row in self.rows)

    @property
    def embedding_seed_ptrs(self) -> tuple[int, ...]:
        return tuple(row.embedding_seed_ptr for row in self.rows)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "request_ids": self.request_ids,
            "rows": [row.as_dict() for row in self.rows],
        }


@dataclass(slots=True)
class Qwen35GGUFMTPContext:
    """Target-attached GGUF MTP state shell.

    The context references the target resident session and block map but does not
    duplicate weights.  It tracks the pending fp32 target hidden seed and the
    verify-hidden rows used by the accept/reseed state machine.
    """

    target_session: Any
    mtp_block: Any | None = None
    pending_seed: Qwen35GGUFMTPSeedRow | None = None
    verify_seeds: tuple[Qwen35GGUFMTPSeedRow, ...] = field(default_factory=tuple)

    @classmethod
    def from_target_seed(
        cls,
        target_session: Any,
        *,
        token_id: int,
        position: int,
        mtp_block: Any | None = None,
    ) -> "Qwen35GGUFMTPContext":
        context = cls(target_session=target_session, mtp_block=mtp_block)
        context.capture_pending_seed_from_target(token_id=token_id, position=position)
        return context

    def capture_pending_seed_from_target(self, *, token_id: int, position: int) -> Qwen35GGUFMTPSeedRow:
        if not hasattr(self.target_session, "mtp_draft_seed"):
            raise TypeError("target_session must expose mtp_draft_seed(token_id=..., position=...)")
        seed = self.target_session.mtp_draft_seed(token_id=int(token_id), position=int(position))
        return self.capture_pending_seed(seed, source="target")

    def capture_pending_seed(self, seed: _DraftSeedLike, *, source: str = "target") -> Qwen35GGUFMTPSeedRow:
        row = Qwen35GGUFMTPSeedRow.from_seed(seed, source=source)
        self.pending_seed = row
        return row

    def record_verify_seeds(self, seeds: Sequence[_DraftSeedLike]) -> tuple[Qwen35GGUFMTPSeedRow, ...]:
        if not seeds:
            raise ValueError("verify seeds must contain at least one row")
        rows = tuple(
            Qwen35GGUFMTPSeedRow.from_seed(seed, source=f"verify[{index}]")
            for index, seed in enumerate(seeds)
        )
        hidden_sizes = {row.hidden_size for row in rows}
        if self.pending_seed is not None:
            hidden_sizes.add(self.pending_seed.hidden_size)
        if len(hidden_sizes) != 1:
            raise ValueError("verify seed hidden sizes must match the pending seed")
        self.verify_seeds = rows
        return rows

    def accept(self, n_accepted: int) -> Qwen35GGUFMTPSeedRow:
        """Apply the GGUF MTP accept/reseed rule from docs/MTP-gguf.md.

        llama.cpp reseeds from ``verify_h[min(n_accepted, n_rows - 1)]``.  This
        method updates ``pending_seed`` to that selected verify row.
        """

        accepted = int(n_accepted)
        if accepted < 0:
            raise ValueError("n_accepted must be non-negative")
        if not self.verify_seeds:
            raise RuntimeError("record_verify_seeds() must be called before accept()")
        index = min(accepted, len(self.verify_seeds) - 1)
        self.pending_seed = self.verify_seeds[index]
        return self.pending_seed

    def build_b1_draft_batch(self, *, request_id: int, token_id: int) -> Qwen35GGUFMTPDraftBatch:
        return self.build_draft_batch(request_id=request_id, token_ids=(int(token_id),))

    def build_draft_batch(
        self,
        *,
        request_id: int,
        token_ids: Sequence[int],
        seed_rows: Sequence[Qwen35GGUFMTPSeedRow | _DraftSeedLike] | None = None,
    ) -> Qwen35GGUFMTPDraftBatch:
        """Build B1-B4 GGUF MTP draft rows with explicit embedding seeds.

        B1 can use the context's ``pending_seed``.  B2-B4 must pass one seed row
        per proposed token, because each deeper draft row consumes the previous
        NextN/verify hidden state rather than reusing the original target seed.
        """

        tokens = tuple(int(token_id) for token_id in token_ids)
        if not tokens:
            raise ValueError("token_ids must contain at least one draft token")
        if seed_rows is None:
            if len(tokens) != 1:
                raise ValueError("multi-depth GGUF MTP draft batches require explicit seed_rows")
            if self.pending_seed is None:
                raise RuntimeError("pending GGUF MTP hidden seed is not populated")
            seeds = (self.pending_seed,)
        else:
            seeds = tuple(self._coerce_seed_row(seed, source=f"draft_seed[{index}]") for index, seed in enumerate(seed_rows))
            if len(seeds) != len(tokens):
                raise ValueError("seed_rows length must match token_ids length")
        hidden_sizes = {seed.hidden_size for seed in seeds}
        if len(hidden_sizes) != 1:
            raise ValueError("all GGUF MTP draft seed rows must share hidden_size")
        rows = tuple(
            Qwen35GGUFMTPDraftRow(
                request_id=int(request_id),
                token_id=token_id,
                position=seed.position + 1,
                draft_depth=depth,
                embedding_seed_ptr=seed.hidden_ptr,
                embedding_hidden_size=seed.hidden_size,
                parent_token_id=seed.token_id,
                parent_position=seed.position,
            )
            for depth, (token_id, seed) in enumerate(zip(tokens, seeds, strict=True), 1)
        )
        return Qwen35GGUFMTPDraftBatch(rows=rows)

    @staticmethod
    def _coerce_seed_row(
        seed: Qwen35GGUFMTPSeedRow | _DraftSeedLike,
        *,
        source: str,
    ) -> Qwen35GGUFMTPSeedRow:
        if isinstance(seed, Qwen35GGUFMTPSeedRow):
            return seed
        return Qwen35GGUFMTPSeedRow.from_seed(seed, source=source)

    def as_dict(self) -> dict[str, object]:
        return {
            "has_target_session": self.target_session is not None,
            "has_mtp_block": self.mtp_block is not None,
            "pending_seed": None if self.pending_seed is None else self.pending_seed.as_dict(),
            "verify_seeds": [seed.as_dict() for seed in self.verify_seeds],
        }
