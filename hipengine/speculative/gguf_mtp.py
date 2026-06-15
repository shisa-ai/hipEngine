"""GGUF-native MTP context scaffolding.

This module is intentionally metadata/state only.  It does not run kernels or
allocate KV buffers yet; it fixes the target-attached state machine needed before
the native GGUF NextN runtime can be wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from hipengine.kernels.registry import resolve


DEFAULT_DRAFT_TOPK_KERNEL = ("cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h")
DEFAULT_DRAFT_TOPK = 10
DEFAULT_DRAFT_SELECTION = "greedy_top1_from_topk"


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
class Qwen35GGUFMTPDraftProposal:
    """Selected GGUF MTP draft tokens plus the top-k evidence that produced them."""

    batch: "Qwen35GGUFMTPDraftBatch"
    top_k_token_ids: tuple[tuple[int, ...], ...]
    top_k_logits: tuple[tuple[float, ...], ...]
    topk_kernel: tuple[str, str, str, str] = DEFAULT_DRAFT_TOPK_KERNEL
    selection: str = DEFAULT_DRAFT_SELECTION
    selected_index: int = 0

    def __post_init__(self) -> None:
        if not self.top_k_token_ids:
            raise ValueError("top_k_token_ids must contain at least one row")
        if len(self.top_k_token_ids) != len(self.batch.rows):
            raise ValueError("top-k rows must match draft batch rows")
        if len(self.top_k_logits) != len(self.top_k_token_ids):
            raise ValueError("top_k_logits rows must match top_k_token_ids rows")
        if len(self.topk_kernel) != 4:
            raise ValueError("topk_kernel must be a four-axis registry key")
        if self.selection != DEFAULT_DRAFT_SELECTION:
            raise ValueError("GGUF MTP currently supports greedy top-1-from-top-k selection only")
        for token_row, logit_row, batch_row in zip(self.top_k_token_ids, self.top_k_logits, self.batch.rows, strict=True):
            if not token_row:
                raise ValueError("top-k token rows must be non-empty")
            if len(logit_row) != len(token_row):
                raise ValueError("top-k logit rows must match token rows")
            if self.selected_index < 0 or self.selected_index >= len(token_row):
                raise ValueError("selected_index must be within every top-k row")
            if int(token_row[self.selected_index]) != int(batch_row.token_id):
                raise ValueError("draft batch token IDs must match selected top-k tokens")

    @property
    def proposed_token_ids(self) -> tuple[int, ...]:
        return self.batch.token_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "batch": self.batch.as_dict(),
            "topk_kernel": list(self.topk_kernel),
            "selection": self.selection,
            "selected_index": self.selected_index,
            "top_k_token_ids": [list(row) for row in self.top_k_token_ids],
            "top_k_logits": [list(row) for row in self.top_k_logits],
            "proposed_token_ids": list(self.proposed_token_ids),
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


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPKVLiveSpansPlan:
    """Metadata-only KVLiveSpans ABI plan for GGUF MTP draft rows.

    This does not allocate device tensors. It records the exact append/decode
    live-count arrays and dense block-table shape that the future MTP KV owner
    must materialize for the single-NextN-layer cache.
    """

    rows: int
    block_size: int
    logical_blocks: int
    base_offsets: tuple[tuple[int, ...], ...]
    append_live_counts: tuple[int, ...]
    decode_live_counts: tuple[int, ...]
    token_positions: tuple[int, ...]
    evict_mask: tuple[tuple[bool, ...], ...] | None = None
    spans_mode: str = "uniform"
    storage_dtype: str = "bf16"

    @classmethod
    def from_draft_batch(
        cls,
        batch: Qwen35GGUFMTPDraftBatch,
        *,
        block_size: int = 256,
        storage_dtype: str = "bf16",
    ) -> "Qwen35GGUFMTPKVLiveSpansPlan":
        block = int(block_size)
        if block <= 0:
            raise ValueError("block_size must be positive")
        positions = tuple(row.position for row in batch.rows)
        max_decode_live_count = max(position + 1 for position in positions)
        logical_blocks = max(1, (max_decode_live_count + block - 1) // block)
        base_offsets = tuple(tuple(range(logical_blocks)) for _ in positions)
        return cls(
            rows=len(positions),
            block_size=block,
            logical_blocks=logical_blocks,
            base_offsets=base_offsets,
            append_live_counts=positions,
            decode_live_counts=tuple(position + 1 for position in positions),
            token_positions=positions,
            storage_dtype=str(storage_dtype),
        )

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.logical_blocks <= 0:
            raise ValueError("logical_blocks must be positive")
        if self.spans_mode != "uniform":
            raise ValueError("GGUF MTP KV spans must use uniform mode")
        if not self.storage_dtype:
            raise ValueError("storage_dtype must be non-empty")
        if len(self.base_offsets) != self.rows:
            raise ValueError("base_offsets rows must match rows")
        for row in self.base_offsets:
            if len(row) != self.logical_blocks:
                raise ValueError("each base_offsets row must match logical_blocks")
            if any(offset < 0 for offset in row):
                raise ValueError("base_offsets must be non-negative")
        for name, values in (
            ("append_live_counts", self.append_live_counts),
            ("decode_live_counts", self.decode_live_counts),
            ("token_positions", self.token_positions),
        ):
            if len(values) != self.rows:
                raise ValueError(f"{name} length must match rows")
            if any(value < 0 for value in values):
                raise ValueError(f"{name} values must be non-negative")
        if any(
            decode < append
            for append, decode in zip(self.append_live_counts, self.decode_live_counts, strict=True)
        ):
            raise ValueError("decode_live_counts must be >= append_live_counts")
        if self.evict_mask is not None:
            if len(self.evict_mask) != self.rows:
                raise ValueError("evict_mask rows must match rows")
            for row in self.evict_mask:
                if len(row) != max(self.decode_live_counts):
                    raise ValueError("evict_mask width must match max decode live count")

    def cpu_reference_kwargs(self, *, role: str = "decode") -> dict[str, object]:
        if role == "append":
            live_counts = self.append_live_counts
        elif role == "decode":
            live_counts = self.decode_live_counts
        else:
            raise ValueError("role must be append or decode")
        return {
            "kv_base_offsets": [list(row) for row in self.base_offsets],
            "kv_live_counts": list(live_counts),
            "kv_token_positions": list(self.token_positions),
            "kv_evict_mask": None
            if self.evict_mask is None
            else [[bool(value) for value in row] for row in self.evict_mask],
            "block_size": self.block_size,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "spans_mode": self.spans_mode,
            "storage_dtype": self.storage_dtype,
            "rows": self.rows,
            "block_size": self.block_size,
            "logical_blocks": self.logical_blocks,
            "base_offsets": [list(row) for row in self.base_offsets],
            "append_live_counts": list(self.append_live_counts),
            "decode_live_counts": list(self.decode_live_counts),
            "token_positions": list(self.token_positions),
            "evict_mask": None
            if self.evict_mask is None
            else [[bool(value) for value in row] for row in self.evict_mask],
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPDraftExecutionPlan:
    """Torch-free draft proposal plus metadata-only KVLiveSpans contract."""

    proposal: Qwen35GGUFMTPDraftProposal
    kv_live_spans: Qwen35GGUFMTPKVLiveSpansPlan
    attention_layer: str = "mtp_nextn_attention"

    def __post_init__(self) -> None:
        if not self.attention_layer:
            raise ValueError("attention_layer must be non-empty")
        rows = self.proposal.batch.rows
        if len(rows) != self.kv_live_spans.rows:
            raise ValueError("proposal rows must match KVLiveSpans rows")
        positions = tuple(row.position for row in rows)
        if positions != self.kv_live_spans.token_positions:
            raise ValueError("proposal positions must match KVLiveSpans token_positions")

    @property
    def proposed_token_ids(self) -> tuple[int, ...]:
        return self.proposal.proposed_token_ids

    def cpu_reference_kwargs(self) -> dict[str, object]:
        return {
            "append": self.kv_live_spans.cpu_reference_kwargs(role="append"),
            "decode": self.kv_live_spans.cpu_reference_kwargs(role="decode"),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "attention_layer": self.attention_layer,
            "proposed_token_ids": list(self.proposed_token_ids),
            "proposal": self.proposal.as_dict(),
            "kv_live_spans": self.kv_live_spans.as_dict(),
            "cpu_reference_kwargs": self.cpu_reference_kwargs(),
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPVerificationResult:
    """Prefix-match result for GGUF MTP proposal verification."""

    proposed_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    n_accepted: int
    reseed: Qwen35GGUFMTPSeedRow
    verify_seed_count: int
    first_mismatch_index: int | None = None
    rejected_proposal_token_id: int | None = None
    target_token_id_at_mismatch: int | None = None

    def __post_init__(self) -> None:
        if not self.proposed_token_ids:
            raise ValueError("proposed_token_ids must be non-empty")
        if len(self.target_token_ids) < len(self.proposed_token_ids):
            raise ValueError("target_token_ids must cover every proposed token")
        if self.verify_seed_count <= len(self.proposed_token_ids):
            raise ValueError("verify_seed_count must include proposed rows plus the next target row")
        if self.n_accepted < 0 or self.n_accepted > len(self.proposed_token_ids):
            raise ValueError("n_accepted must be in 0..len(proposed_token_ids)")
        if self.first_mismatch_index is None:
            if self.n_accepted != len(self.proposed_token_ids):
                raise ValueError("missing first_mismatch_index for partial acceptance")
            if self.rejected_proposal_token_id is not None or self.target_token_id_at_mismatch is not None:
                raise ValueError("mismatch token ids must be empty when all drafts are accepted")
        else:
            if self.first_mismatch_index != self.n_accepted:
                raise ValueError("first_mismatch_index must equal n_accepted")
            if self.rejected_proposal_token_id is None or self.target_token_id_at_mismatch is None:
                raise ValueError("mismatch token ids must be present for partial acceptance")

    @property
    def accepted_token_ids(self) -> tuple[int, ...]:
        return self.proposed_token_ids[: self.n_accepted]

    @property
    def accepted_per_draft(self) -> float:
        return float(self.n_accepted) / float(len(self.proposed_token_ids))

    def as_dict(self) -> dict[str, object]:
        return {
            "proposed_token_ids": list(self.proposed_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "accepted_token_ids": list(self.accepted_token_ids),
            "n_accepted": self.n_accepted,
            "draft_count": len(self.proposed_token_ids),
            "accepted_per_draft": self.accepted_per_draft,
            "first_mismatch_index": self.first_mismatch_index,
            "rejected_proposal_token_id": self.rejected_proposal_token_id,
            "target_token_id_at_mismatch": self.target_token_id_at_mismatch,
            "verify_seed_count": self.verify_seed_count,
            "reseed": self.reseed.as_dict(),
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

    def record_verify_seeds(
        self,
        seeds: Sequence[Qwen35GGUFMTPSeedRow | _DraftSeedLike],
    ) -> tuple[Qwen35GGUFMTPSeedRow, ...]:
        if not seeds:
            raise ValueError("verify seeds must contain at least one row")
        rows = tuple(
            self._coerce_seed_row(seed, source=f"verify[{index}]")
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

    def build_draft_proposal_from_logits(
        self,
        *,
        request_id: int,
        logits: Any,
        seed_rows: Sequence[Qwen35GGUFMTPSeedRow | _DraftSeedLike] | None = None,
        top_k: int = DEFAULT_DRAFT_TOPK,
        selected_index: int = 0,
        topk_kernel: tuple[str, str, str, str] = DEFAULT_DRAFT_TOPK_KERNEL,
    ) -> Qwen35GGUFMTPDraftProposal:
        """Select draft tokens from top-k logits and build GGUF MTP draft rows.

        The method does not run the NextN block.  It bridges the future runtime
        output logits to the existing target-attached batch state by resolving
        the four-axis top-k fallback/device kernel and selecting index 0 from the
        returned top-k rows (llama.cpp's greedy top-1-from-top-k contract).
        """

        selected_index = int(selected_index)
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if selected_index < 0 or selected_index >= top_k:
            raise ValueError("selected_index must be within top_k")
        if len(topk_kernel) != 4:
            raise ValueError("topk_kernel must be a four-axis registry key")
        kernel = resolve(
            backend=topk_kernel[0],
            layer=topk_kernel[1],
            quant=topk_kernel[2],
            variant=topk_kernel[3],
        )
        token_ids, values = kernel(logits, k=top_k)
        top_k_token_ids = _nested_int_rows(token_ids)
        top_k_logits = _nested_float_rows(values)
        selected_token_ids = tuple(row[selected_index] for row in top_k_token_ids)
        batch = self.build_draft_batch(
            request_id=request_id,
            token_ids=selected_token_ids,
            seed_rows=seed_rows,
        )
        return Qwen35GGUFMTPDraftProposal(
            batch=batch,
            top_k_token_ids=top_k_token_ids,
            top_k_logits=top_k_logits,
            topk_kernel=topk_kernel,
            selected_index=selected_index,
        )

    def build_kvlivespans_plan(
        self,
        batch: Qwen35GGUFMTPDraftBatch,
        *,
        block_size: int = 256,
        storage_dtype: str = "bf16",
    ) -> Qwen35GGUFMTPKVLiveSpansPlan:
        return Qwen35GGUFMTPKVLiveSpansPlan.from_draft_batch(
            batch,
            block_size=block_size,
            storage_dtype=storage_dtype,
        )

    def build_draft_execution_plan_from_logits(
        self,
        *,
        request_id: int,
        logits: Any,
        seed_rows: Sequence[Qwen35GGUFMTPSeedRow | _DraftSeedLike] | None = None,
        top_k: int = DEFAULT_DRAFT_TOPK,
        selected_index: int = 0,
        topk_kernel: tuple[str, str, str, str] = DEFAULT_DRAFT_TOPK_KERNEL,
        block_size: int = 256,
        storage_dtype: str = "bf16",
    ) -> Qwen35GGUFMTPDraftExecutionPlan:
        proposal = self.build_draft_proposal_from_logits(
            request_id=request_id,
            logits=logits,
            seed_rows=seed_rows,
            top_k=top_k,
            selected_index=selected_index,
            topk_kernel=topk_kernel,
        )
        kv_live_spans = self.build_kvlivespans_plan(
            proposal.batch,
            block_size=block_size,
            storage_dtype=storage_dtype,
        )
        return Qwen35GGUFMTPDraftExecutionPlan(
            proposal=proposal,
            kv_live_spans=kv_live_spans,
        )

    def verify_draft_proposal(
        self,
        proposal: Qwen35GGUFMTPDraftProposal | Qwen35GGUFMTPDraftExecutionPlan,
        *,
        target_token_ids: Sequence[int],
        verify_seeds: Sequence[Qwen35GGUFMTPSeedRow | _DraftSeedLike],
    ) -> Qwen35GGUFMTPVerificationResult:
        """Compare proposed draft tokens to target tokens and reseed.

        The method applies the llama.cpp reseed rule already used by ``accept``:
        ``verify_h[min(n_accepted, n_rows - 1)]``.  Verification seeds must
        include one row per proposal plus the next target row, so full acceptance
        can reseed from the post-proposal target row.
        """

        draft_proposal = proposal.proposal if isinstance(proposal, Qwen35GGUFMTPDraftExecutionPlan) else proposal
        proposed = draft_proposal.proposed_token_ids
        targets = tuple(int(token_id) for token_id in target_token_ids)
        if len(targets) < len(proposed):
            raise ValueError("target_token_ids must cover every proposed token")
        if len(verify_seeds) <= len(proposed):
            raise ValueError("verify_seeds must include proposed rows plus the next target row")
        self.record_verify_seeds(verify_seeds)
        accepted = 0
        for proposed_token, target_token in zip(proposed, targets, strict=False):
            if proposed_token != target_token:
                break
            accepted += 1
            if accepted == len(proposed):
                break
        reseed = self.accept(accepted)
        if accepted == len(proposed):
            mismatch_index = None
            rejected_token = None
            target_mismatch_token = None
        else:
            mismatch_index = accepted
            rejected_token = proposed[mismatch_index]
            target_mismatch_token = targets[mismatch_index]
        return Qwen35GGUFMTPVerificationResult(
            proposed_token_ids=proposed,
            target_token_ids=targets,
            n_accepted=accepted,
            first_mismatch_index=mismatch_index,
            rejected_proposal_token_id=rejected_token,
            target_token_id_at_mismatch=target_mismatch_token,
            reseed=reseed,
            verify_seed_count=len(verify_seeds),
        )

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


def _nested_int_rows(values: Any) -> tuple[tuple[int, ...], ...]:
    rows = values.tolist() if hasattr(values, "tolist") else values
    return tuple(tuple(int(value) for value in row) for row in rows)


def _nested_float_rows(values: Any) -> tuple[tuple[float, ...], ...]:
    rows = values.tolist() if hasattr(values, "tolist") else values
    return tuple(tuple(float(value) for value in row) for row in rows)
