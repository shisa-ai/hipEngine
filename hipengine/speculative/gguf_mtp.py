"""GGUF-native MTP context scaffolding.

This module is intentionally metadata/state only.  It does not run kernels or
allocate KV buffers yet; it fixes the target-attached state machine needed before
the native GGUF NextN runtime can be wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kvcache.policy import KVTransaction
from hipengine.speculative.interfaces import DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetVerifyBatch


DEFAULT_DRAFT_TOPK_KERNEL = ("cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h")
DEFAULT_DRAFT_TOPK = 10
DEFAULT_DRAFT_SELECTION = "greedy_top1_from_topk"
DEFAULT_NEXTN_CPU_REFERENCE_KERNEL = ("cpu_reference", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits")
DEFAULT_NEXTN_LAYER = "mtp_nextn_layer"
DEFAULT_NEXTN_QUANT = "w4_gguf"
DEFAULT_NEXTN_VARIANT = "qwen35_dense_logits"
DEFAULT_NEXTN_KV_WRITE_LAYER = "paged_kv_write"
DEFAULT_NEXTN_KV_WRITE_VARIANT = "mixed_bf16_spans"
DEFAULT_NEXTN_ATTN_DECODE_LAYER = "paged_attn_decode"
DEFAULT_NEXTN_ATTN_DECODE_VARIANT = "bf16_context_spans"
DEFAULT_DRAFT_TOPK_DEVICE_VARIANT = "topk_device"
GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE = "full_requested_budget_exercised"
GGUF_MTP_PARTIAL_TRACE_BUDGET_COVERAGE = "partial_trace_did_not_exercise_full_budget"
GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE = "computed"
GGUF_MTP_ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE = "not_comparable_debug_trace_missing_generated_draft_count"
GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE = "computed"
GGUF_MTP_ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE = "not_comparable_debug_trace_missing_visible_output_count"
GGUF_MTP_METRICS_CONTRACT_READY = "ready"


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

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return ("token_id", "position", "hidden_ptr", "hidden_size", "source")

    @staticmethod
    def validator_name() -> str:
        return "Qwen35GGUFMTPSeedRow.validate_payload"

    @classmethod
    def contract(cls) -> dict[str, object]:
        return {
            "required_fields": list(cls.required_fields()),
            "validator": cls.validator_name(),
        }

    @classmethod
    def validate_payload(cls, payload: Mapping[str, object], *, field_name: str = "seed row") -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        missing = tuple(field for field in cls.required_fields() if field not in payload)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{field_name} missing fields: {joined}")
        source = payload["source"]
        if not isinstance(source, str):
            raise ValueError(f"{field_name} source must be a string")
        try:
            cls(
                token_id=int(payload["token_id"]),
                position=int(payload["position"]),
                hidden_ptr=int(payload["hidden_ptr"]),
                hidden_size=int(payload["hidden_size"]),
                source=source,
            )
        except (TypeError, ValueError) as exc:
            detail = str(exc)
            if detail.startswith("seed "):
                detail = detail[len("seed ") :]
            raise ValueError(f"{field_name} {detail}") from exc

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

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return (
            "request_id",
            "token_id",
            "position",
            "draft_depth",
            "embedding_seed_ptr",
            "embedding_hidden_size",
            "parent_token_id",
            "parent_position",
        )

    @staticmethod
    def validator_name() -> str:
        return "Qwen35GGUFMTPDraftRow.validate_payload"

    @classmethod
    def contract(cls) -> dict[str, object]:
        return {
            "required_fields": list(cls.required_fields()),
            "validator": cls.validator_name(),
        }

    @classmethod
    def validate_payload(
        cls,
        payload: Mapping[str, object],
        *,
        field_name: str = "draft row",
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        missing = tuple(field for field in cls.required_fields() if field not in payload)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{field_name} missing fields: {joined}")
        try:
            cls(
                request_id=int(payload["request_id"]),
                token_id=int(payload["token_id"]),
                position=int(payload["position"]),
                draft_depth=int(payload["draft_depth"]),
                embedding_seed_ptr=int(payload["embedding_seed_ptr"]),
                embedding_hidden_size=int(payload["embedding_hidden_size"]),
                parent_token_id=int(payload["parent_token_id"]),
                parent_position=int(payload["parent_position"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} {exc}") from exc

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

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return (
            "batch",
            "draft_batch_contract",
            "topk_kernel",
            "selection",
            "selected_index",
            "top_k_token_ids",
            "top_k_logits",
            "proposed_token_ids",
        )

    @staticmethod
    def validator_name() -> str:
        return "Qwen35GGUFMTPDraftProposal.validate_payload"

    @classmethod
    def contract(cls) -> dict[str, object]:
        return {
            "required_fields": list(cls.required_fields()),
            "validator": cls.validator_name(),
        }

    @classmethod
    def missing_required_fields(cls, payload: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(field for field in cls.required_fields() if field not in payload)

    @classmethod
    def validate_payload(
        cls,
        payload: Mapping[str, object],
        *,
        field_name: str = "draft proposal",
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        missing = cls.missing_required_fields(payload)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{field_name} missing required fields: {joined}")
        if payload.get("draft_batch_contract") != Qwen35GGUFMTPDraftBatch.contract():
            raise ValueError(f"{field_name} draft_batch_contract mismatch")
        batch_payload = payload.get("batch")
        if not isinstance(batch_payload, Mapping):
            raise ValueError(f"{field_name} batch must be a mapping")
        Qwen35GGUFMTPDraftBatch.validate_payload(batch_payload, field_name=f"{field_name} batch")
        rows = batch_payload.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"{field_name} batch rows must be a sequence")
        draft_rows: list[Qwen35GGUFMTPDraftRow] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"{field_name} batch row[{index}] must be a mapping")
            draft_rows.append(
                Qwen35GGUFMTPDraftRow(
                    request_id=int(row["request_id"]),
                    token_id=int(row["token_id"]),
                    position=int(row["position"]),
                    draft_depth=int(row["draft_depth"]),
                    embedding_seed_ptr=int(row["embedding_seed_ptr"]),
                    embedding_hidden_size=int(row["embedding_hidden_size"]),
                    parent_token_id=int(row["parent_token_id"]),
                    parent_position=int(row["parent_position"]),
                )
            )
        batch = Qwen35GGUFMTPDraftBatch(rows=tuple(draft_rows), mode=str(batch_payload["mode"]))
        topk_kernel = payload.get("topk_kernel")
        top_k_token_ids = payload.get("top_k_token_ids")
        top_k_logits = payload.get("top_k_logits")
        if not isinstance(topk_kernel, Sequence) or isinstance(topk_kernel, (str, bytes)):
            raise ValueError(f"{field_name} topk_kernel must be a sequence")
        if len(topk_kernel) != 4:
            raise ValueError(f"{field_name} topk_kernel must be a four-axis registry key")
        if not isinstance(top_k_token_ids, Sequence) or isinstance(top_k_token_ids, (str, bytes)):
            raise ValueError(f"{field_name} top_k_token_ids must be a sequence")
        if not isinstance(top_k_logits, Sequence) or isinstance(top_k_logits, (str, bytes)):
            raise ValueError(f"{field_name} top_k_logits must be a sequence")
        token_rows = tuple(tuple(int(item) for item in row) for row in top_k_token_ids)
        logit_rows = tuple(tuple(float(item) for item in row) for row in top_k_logits)
        proposal = cls(
            batch=batch,
            top_k_token_ids=token_rows,
            top_k_logits=logit_rows,
            topk_kernel=tuple(str(item) for item in topk_kernel),
            selection=str(payload["selection"]),
            selected_index=int(payload["selected_index"]),
        )
        proposed_token_ids = payload.get("proposed_token_ids")
        if not isinstance(proposed_token_ids, Sequence) or isinstance(proposed_token_ids, (str, bytes)):
            raise ValueError(f"{field_name} proposed_token_ids must be a sequence")
        if tuple(int(item) for item in proposed_token_ids) != proposal.proposed_token_ids:
            raise ValueError(f"{field_name} proposed_token_ids mismatch")

    @property
    def proposed_token_ids(self) -> tuple[int, ...]:
        return self.batch.token_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "batch": self.batch.as_dict(),
            "draft_batch_contract": Qwen35GGUFMTPDraftBatch.contract(),
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

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return ("mode", "request_ids", "row_count", "draft_row_contract", "rows")

    @staticmethod
    def validator_name() -> str:
        return "Qwen35GGUFMTPDraftBatch.validate_payload"

    @classmethod
    def contract(cls) -> dict[str, object]:
        return {
            "required_fields": list(cls.required_fields()),
            "validator": cls.validator_name(),
        }

    @classmethod
    def missing_required_fields(cls, payload: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(field for field in cls.required_fields() if field not in payload)

    @classmethod
    def validate_payload(
        cls,
        payload: Mapping[str, object],
        *,
        field_name: str = "draft batch",
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        missing = cls.missing_required_fields(payload)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{field_name} missing required fields: {joined}")
        if payload.get("mode") != "gguf_mtp_nextn":
            raise ValueError(f"{field_name} mode must be gguf_mtp_nextn")
        if payload.get("draft_row_contract") != Qwen35GGUFMTPDraftRow.contract():
            raise ValueError(f"{field_name} draft_row_contract mismatch")
        rows = payload.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"{field_name} rows must be a sequence")
        row_count = int(payload["row_count"])
        if row_count != len(rows):
            raise ValueError(f"{field_name} row_count mismatch")
        request_ids = payload.get("request_ids")
        if not isinstance(request_ids, Sequence) or isinstance(request_ids, (str, bytes)):
            raise ValueError(f"{field_name} request_ids must be a sequence")
        normalized_rows: list[Qwen35GGUFMTPDraftRow] = []
        for index, row in enumerate(rows):
            Qwen35GGUFMTPDraftRow.validate_payload(
                row,
                field_name=f"{field_name} row[{index}]",
            )
            if not isinstance(row, Mapping):
                raise ValueError(f"{field_name} row[{index}] must be a mapping")
            normalized_rows.append(
                Qwen35GGUFMTPDraftRow(
                    request_id=int(row["request_id"]),
                    token_id=int(row["token_id"]),
                    position=int(row["position"]),
                    draft_depth=int(row["draft_depth"]),
                    embedding_seed_ptr=int(row["embedding_seed_ptr"]),
                    embedding_hidden_size=int(row["embedding_hidden_size"]),
                    parent_token_id=int(row["parent_token_id"]),
                    parent_position=int(row["parent_position"]),
                )
            )
        expected_request_ids = tuple(sorted({row.request_id for row in normalized_rows}))
        if tuple(int(item) for item in request_ids) != expected_request_ids:
            raise ValueError(f"{field_name} request_ids mismatch")
        cls(rows=tuple(normalized_rows), mode=str(payload["mode"]))

    def to_shared_draft_batch(self, *, mode: str = "verify_chain") -> DraftBatch:
        """Project GGUF MTP rows into the shared target verifier ABI.

        GGUF MTP keeps embedding-seed pointers on ``Qwen35GGUFMTPDraftRow``.
        The shared ``DraftBatch`` is candidate-only, so this method deliberately
        exports only verifier-facing token/topology metadata while validating
        that each deeper draft row references an earlier row for the same
        request.
        """

        target_mode = str(mode)
        if target_mode != "verify_chain":
            raise ValueError("GGUF MTP shared draft batches must use verify_chain mode")
        parent_by_request_depth: dict[tuple[int, int], int] = {}
        tree_parents: list[int] = []
        for index, row in enumerate(self.rows):
            if row.draft_depth == 1:
                parent = -1
            else:
                parent_key = (row.request_id, row.draft_depth - 1)
                if parent_key not in parent_by_request_depth:
                    raise ValueError("GGUF MTP draft rows must be ordered parent before child")
                parent = parent_by_request_depth[parent_key]
                parent_row = self.rows[parent]
                if row.parent_token_id != parent_row.token_id:
                    raise ValueError("draft row parent_token_id must match the previous depth token_id")
                if row.parent_position != parent_row.position:
                    raise ValueError("draft row parent_position must match the previous depth position")
            parent_by_request_depth[(row.request_id, row.draft_depth)] = index
            tree_parents.append(parent)
        return DraftBatch(
            request_ids=self.request_ids,
            candidate_tokens=tuple(row.token_id for row in self.rows),
            parent_positions=tuple(row.parent_position for row in self.rows),
            draft_depths=tuple(row.draft_depth for row in self.rows),
            row_to_request=tuple(row.request_id for row in self.rows),
            tree_parents=tuple(tree_parents),
            active_mask=tuple(True for _ in self.rows),
            mode=target_mode,
        )

    def to_target_verify_batch(self, *, mode: str = "verify_chain") -> TargetVerifyBatch:
        """Build the shared target-verifier row layout for this GGUF MTP batch.

        Depth-1 GGUF rows already carry the committed root token/position in
        ``parent_token_id`` and ``parent_position``.  Deriving roots here keeps
        future GGUF verify/accept integration from passing duplicate root
        metadata that could drift from the draft-row embedding seed contract.
        """

        root_by_request: dict[int, tuple[int, int]] = {}
        for row in self.rows:
            if row.draft_depth == 1:
                root_by_request[row.request_id] = (row.parent_token_id, row.parent_position)
        missing = [request_id for request_id in self.request_ids if request_id not in root_by_request]
        if missing:
            raise ValueError("GGUF MTP target verify batch requires one depth-1 root row per request")
        root_tokens = tuple(root_by_request[request_id][0] for request_id in self.request_ids)
        root_positions = tuple(root_by_request[request_id][1] for request_id in self.request_ids)
        return TargetVerifyBatch.from_draft(
            self.to_shared_draft_batch(mode=mode),
            root_tokens=root_tokens,
            root_positions=root_positions,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "request_ids": self.request_ids,
            "row_count": len(self.rows),
            "draft_row_contract": Qwen35GGUFMTPDraftRow.contract(),
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

    def to_target_verify_batch(self, *, mode: str = "verify_chain") -> TargetVerifyBatch:
        return self.proposal.batch.to_target_verify_batch(mode=mode)

    def target_verify_transaction(self, transaction_id: int, *, mode: str = "verify_chain") -> KVTransaction:
        """Build scheduler-facing speculative KV transaction metadata.

        Native GGUF MTP KV writes are not wired yet, but the transaction shape
        must match the target-verification row layout: request ids, candidate-row
        count, per-request candidate counts, and verify-chain/tree role all come
        from the same ``TargetVerifyBatch`` used by accept/commit plumbing.
        """

        target_batch = self.to_target_verify_batch(mode=mode)
        return KVTransaction(
            transaction_id=int(transaction_id),
            request_ids=target_batch.request_ids,
            draft_rows=target_batch.candidate_count,
            role=target_batch.mode,
            candidate_counts=target_batch.candidate_counts,
        )

    def target_accept_summary_from_top1(
        self,
        target_top1: Sequence[int],
        *,
        transaction_id: int | None = None,
        remaining_decode: Sequence[int] | None = None,
        mode: str = "verify_chain",
    ) -> TargetAcceptSummary:
        """CPU oracle for the future GGUF MTP target accept-summary kernel.

        ``target_top1`` is indexed by the shared ``TargetVerifyBatch`` rows built
        from this execution plan.  The returned summary fixes accepted counts,
        commit rows/tokens/positions, full-accept flags, and correction tokens
        without allocating device buffers or mutating GGUF MTP hidden-seed state.
        """

        target_batch = self.to_target_verify_batch(mode=mode)
        accept_result = target_batch.accept_from_top1(
            target_top1,
            transaction_id=transaction_id,
            remaining_decode=remaining_decode,
        )
        return TargetAcceptSummary.from_accept_result(target_batch, accept_result)

    def target_commit_plan_from_summary(self, summary: TargetAcceptSummary, transaction) -> TargetCommitPlan:
        """Validate a GGUF MTP accept summary and build the shared commit plan.

        The GGUF runner does not own scheduler/KV transactions yet, but future
        native integration must commit the same target rows that this execution
        plan verified.  This bridge mirrors the provider-neutral scheduler
        checks before delegating to ``TargetCommitPlan.from_summary``.
        """

        target_batch = self.to_target_verify_batch(mode=summary.mode)
        if summary.request_ids != target_batch.request_ids:
            raise ValueError("accept summary request_ids must match GGUF MTP target batch")
        if summary.candidate_counts is not None and summary.candidate_counts != target_batch.candidate_counts:
            raise ValueError("accept summary candidate_counts must match GGUF MTP target batch")
        if summary.draft_depth is not None and summary.draft_depth != target_batch.draft_depth:
            raise ValueError("accept summary draft_depth must match GGUF MTP target batch")
        if summary.tree_shape is not None and summary.tree_shape != target_batch.tree_shape:
            raise ValueError("accept summary tree_shape must match GGUF MTP target batch")
        root_rows = set(target_batch.root_rows)
        candidate_rows = set(target_batch.candidate_rows)
        for request_id, count, row, token, position in zip(
            summary.request_ids,
            summary.accepted_counts,
            summary.commit_rows,
            summary.commit_tokens,
            summary.commit_positions,
            strict=True,
        ):
            if row < 0 or row >= target_batch.rows:
                raise ValueError("accept summary commit row must be in GGUF MTP target batch")
            if target_batch.row_to_request[row] != request_id:
                raise ValueError("accept summary commit row must belong to its request")
            if count == 0 and row not in root_rows:
                raise ValueError("zero accepted candidates must commit the request root row")
            if count > 0 and row not in candidate_rows:
                raise ValueError("accepted candidates must commit a candidate row")
            if target_batch.draft_depths[row] != count:
                raise ValueError("accept summary commit row depth must match accepted count")
            if target_batch.tokens[row] != token or target_batch.positions[row] != position:
                raise ValueError("accept summary commit token/position must match target row")
        return TargetCommitPlan.from_summary(summary, transaction)

    def target_commit_plan_from_top1(
        self,
        target_top1: Sequence[int],
        *,
        transaction_id: int,
        remaining_decode: Sequence[int] | None = None,
        mode: str = "verify_chain",
    ) -> TargetCommitPlan:
        """Build a validated commit plan directly from target top-1 rows.

        This is the metadata-only equivalent of the future native target verify
        handoff: derive the KV transaction shape, compute the shared accept
        summary from target top-1 rows, then validate/build the provider-neutral
        commit plan against the GGUF-derived verifier rows.
        """

        summary = self.target_accept_summary_from_top1(
            target_top1,
            transaction_id=int(transaction_id),
            remaining_decode=remaining_decode,
            mode=mode,
        )
        transaction = self.target_verify_transaction(transaction_id, mode=mode)
        return self.target_commit_plan_from_summary(summary, transaction)

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
class Qwen35GGUFMTPAcceptStep:
    """Metadata result for one GGUF MTP target verify/commit step."""

    commit_plan: TargetCommitPlan
    reseed: Qwen35GGUFMTPSeedRow

    def __iter__(self):
        yield self.commit_plan
        yield self.reseed

    def as_dict(self) -> dict[str, object]:
        return {
            "transaction_id": self.commit_plan.transaction_id,
            "request_ids": list(self.commit_plan.request_ids),
            "accepted_counts": list(self.commit_plan.accepted_counts),
            "commit_rows": list(self.commit_plan.commit_rows),
            "commit_tokens": list(self.commit_plan.commit_tokens),
            "commit_positions": list(self.commit_plan.commit_positions),
            "next_tokens": None if self.commit_plan.next_tokens is None else list(self.commit_plan.next_tokens),
            "candidate_counts": None
            if self.commit_plan.candidate_counts is None
            else list(self.commit_plan.candidate_counts),
            "draft_depth": self.commit_plan.draft_depth,
            "tree_shape": None if self.commit_plan.tree_shape is None else list(self.commit_plan.tree_shape),
            "mode": self.commit_plan.mode,
            "reseed": self.reseed.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPAcceptStepMetrics:
    """Aggregate denominator contract for target-top1 GGUF MTP accept steps."""

    steps: tuple[Qwen35GGUFMTPAcceptStep, ...]
    output_token_count: int

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[Qwen35GGUFMTPAcceptStep],
        *,
        output_token_count: int,
    ) -> Qwen35GGUFMTPAcceptStepMetrics:
        normalized = tuple(steps)
        if not normalized:
            raise ValueError("at least one GGUF MTP accept step is required")
        out = int(output_token_count)
        if out <= 0:
            raise ValueError("output_token_count must be positive")
        for index, step in enumerate(normalized):
            if step.commit_plan.candidate_counts is None:
                raise ValueError(f"accept step {index} must include candidate_counts")
        if sum(sum(step.commit_plan.candidate_counts or ()) for step in normalized) <= 0:
            raise ValueError("draft_token_count must be positive")
        accepted = sum(sum(step.commit_plan.accepted_counts) for step in normalized)
        if accepted > out:
            raise ValueError("accepted draft tokens cannot exceed visible output tokens")
        return cls(steps=normalized, output_token_count=out)

    @property
    def cycle_count(self) -> int:
        return len(self.steps)

    @property
    def draft_token_count(self) -> int:
        return sum(sum(step.commit_plan.candidate_counts or ()) for step in self.steps)

    @property
    def candidate_budget(self) -> int:
        return max(count for step in self.steps for count in (step.commit_plan.candidate_counts or ()))

    @property
    def budget_label(self) -> str:
        return f"B{self.candidate_budget}"

    @property
    def accepted_token_count(self) -> int:
        return sum(sum(step.commit_plan.accepted_counts) for step in self.steps)

    @property
    def step_transaction_ids(self) -> tuple[int, ...]:
        return tuple(step.commit_plan.transaction_id for step in self.steps)

    @property
    def step_candidate_token_counts(self) -> tuple[int, ...]:
        return tuple(sum(step.commit_plan.candidate_counts or ()) for step in self.steps)

    @property
    def step_accepted_token_counts(self) -> tuple[int, ...]:
        return tuple(sum(step.commit_plan.accepted_counts) for step in self.steps)

    def step_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "transaction_id": step.commit_plan.transaction_id,
                "request_ids": list(step.commit_plan.request_ids),
                "candidate_token_count": candidate_count,
                "accepted_token_count": accepted_count,
                "candidate_counts": None
                if step.commit_plan.candidate_counts is None
                else list(step.commit_plan.candidate_counts),
                "accepted_counts": list(step.commit_plan.accepted_counts),
            }
            for step, candidate_count, accepted_count in zip(
                self.steps,
                self.step_candidate_token_counts,
                self.step_accepted_token_counts,
                strict=True,
            )
        )

    @property
    def accepted_per_draft(self) -> float:
        return float(self.accepted_token_count) / float(self.draft_token_count)

    @property
    def accepted_per_output(self) -> float:
        return float(self.accepted_token_count) / float(self.output_token_count)

    @staticmethod
    def denominator_labels() -> dict[str, str]:
        return {
            "accepted_per_draft": "accepted_token_count / draft_token_count",
            "accepted_per_output": "accepted_token_count / output_token_count",
        }

    @staticmethod
    def artifact_labels() -> dict[str, object]:
        return {
            "schema": 1,
            "kind": "hipengine_gguf_mtp_accept_step_metrics",
            "source": "Qwen35GGUFMTPAcceptStepMetrics",
            "result_source": "Qwen35GGUFMTPAcceptStep",
        }

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return (
            "cycle_count",
            "candidate_budget",
            "budget_label",
            "draft_token_count",
            "accepted_token_count",
            "output_token_count",
            "step_transaction_ids",
            "step_candidate_token_counts",
            "step_accepted_token_counts",
            "step_rows",
            "seed_row_contract",
            "steps",
            "accepted_per_draft",
            "accepted_per_output",
        )

    @staticmethod
    def validator_name() -> str:
        return "Qwen35GGUFMTPAcceptStepMetrics.validate_payload"

    @classmethod
    def missing_required_fields(cls, payload: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(field for field in cls.required_fields() if field not in payload)

    @classmethod
    def blocked_contract(
        cls,
        *,
        candidate_budget: int,
        blocked_until: str = "native_gguf_mtp_runtime",
    ) -> dict[str, object]:
        budget = int(candidate_budget)
        if budget <= 0:
            raise ValueError("candidate_budget must be positive")
        return {
            "status": "not_run",
            "blocked_until": str(blocked_until),
            **cls.artifact_labels(),
            "draft_max": budget,
            "candidate_budget": budget,
            "budget_label": f"B{budget}",
            "required_fields": list(cls.required_fields()),
            "validator": cls.validator_name(),
            "denominators": cls.denominator_labels(),
        }

    @classmethod
    def validate_blocked_contract(
        cls,
        payload: Mapping[str, object],
        *,
        candidate_budget: int | None = None,
        blocked_until: str | None = "native_gguf_mtp_runtime",
    ) -> None:
        expected_fields = {
            "status",
            "blocked_until",
            *cls.artifact_labels().keys(),
            "draft_max",
            "candidate_budget",
            "budget_label",
            "required_fields",
            "validator",
            "denominators",
        }
        missing = tuple(field for field in expected_fields if field not in payload)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"blocked accept-step metrics contract missing fields: {joined}")
        if payload.get("status") != "not_run":
            raise ValueError("blocked accept-step metrics contract status mismatch")
        if blocked_until is not None and payload.get("blocked_until") != blocked_until:
            raise ValueError("blocked accept-step metrics contract blocked_until mismatch")
        for key, expected in cls.artifact_labels().items():
            if payload.get(key) != expected:
                raise ValueError(f"blocked accept-step metrics contract {key} mismatch")
        budget = int(payload["candidate_budget"])
        if budget <= 0:
            raise ValueError("blocked accept-step metrics contract candidate_budget must be positive")
        if candidate_budget is not None and budget != int(candidate_budget):
            raise ValueError("blocked accept-step metrics contract candidate_budget mismatch")
        if int(payload["draft_max"]) != budget:
            raise ValueError("blocked accept-step metrics contract draft_max mismatch")
        if payload.get("budget_label") != f"B{budget}":
            raise ValueError("blocked accept-step metrics contract budget_label mismatch")
        if tuple(payload.get("required_fields", ())) != cls.required_fields():
            raise ValueError("blocked accept-step metrics contract required_fields mismatch")
        if payload.get("validator") != cls.validator_name():
            raise ValueError("blocked accept-step metrics contract validator mismatch")
        if payload.get("denominators") != cls.denominator_labels():
            raise ValueError("blocked accept-step metrics contract denominator labels mismatch")

    @classmethod
    def validate_payload(cls, payload: Mapping[str, object]) -> None:
        def int_sequence(name: str) -> tuple[int, ...]:
            value = payload.get(name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"accept-step metrics artifact {name} must be a sequence")
            return tuple(int(item) for item in value)

        for key, expected in cls.artifact_labels().items():
            if payload.get(key) != expected:
                raise ValueError(f"accept-step metrics artifact {key} mismatch")
        missing = cls.missing_required_fields(payload)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"accept-step metrics artifact missing required fields: {joined}")
        required_fields = payload.get("required_fields")
        if not isinstance(required_fields, Sequence) or isinstance(required_fields, (str, bytes)):
            raise ValueError("accept-step metrics artifact required_fields must be a sequence")
        if tuple(required_fields) != cls.required_fields():
            raise ValueError("accept-step metrics artifact required_fields mismatch")
        if payload.get("validator") != cls.validator_name():
            raise ValueError("accept-step metrics artifact validator mismatch")
        if payload.get("denominators") != cls.denominator_labels():
            raise ValueError("accept-step metrics artifact denominator labels mismatch")
        if payload.get("seed_row_contract") != Qwen35GGUFMTPSeedRow.contract():
            raise ValueError("accept-step metrics artifact seed_row_contract mismatch")

        cycle_count = int(payload["cycle_count"])
        candidate_budget = int(payload["candidate_budget"])
        draft_token_count = int(payload["draft_token_count"])
        accepted_token_count = int(payload["accepted_token_count"])
        output_token_count = int(payload["output_token_count"])
        if cycle_count <= 0:
            raise ValueError("accept-step metrics artifact cycle_count must be positive")
        if candidate_budget <= 0:
            raise ValueError("accept-step metrics artifact candidate_budget must be positive")
        if draft_token_count <= 0:
            raise ValueError("accept-step metrics artifact draft_token_count must be positive")
        if accepted_token_count < 0:
            raise ValueError("accept-step metrics artifact accepted_token_count must be non-negative")
        if output_token_count <= 0:
            raise ValueError("accept-step metrics artifact output_token_count must be positive")
        if accepted_token_count > output_token_count:
            raise ValueError("accept-step metrics artifact accepted draft tokens cannot exceed visible output tokens")
        if payload.get("budget_label") != f"B{candidate_budget}":
            raise ValueError("accept-step metrics artifact budget_label mismatch")

        step_transaction_ids = int_sequence("step_transaction_ids")
        step_candidate_counts = int_sequence("step_candidate_token_counts")
        step_accepted_counts = int_sequence("step_accepted_token_counts")
        for name, values in (
            ("step_transaction_ids", step_transaction_ids),
            ("step_candidate_token_counts", step_candidate_counts),
            ("step_accepted_token_counts", step_accepted_counts),
        ):
            if len(values) != cycle_count:
                raise ValueError(f"accept-step metrics artifact {name} length mismatch")
        if draft_token_count != sum(step_candidate_counts):
            raise ValueError("accept-step metrics artifact draft_token_count mismatch")
        if accepted_token_count != sum(step_accepted_counts):
            raise ValueError("accept-step metrics artifact accepted_token_count mismatch")

        step_rows = payload.get("step_rows")
        if not isinstance(step_rows, Sequence) or isinstance(step_rows, (str, bytes)):
            raise ValueError("accept-step metrics artifact step_rows must be a sequence")
        if len(step_rows) != cycle_count:
            raise ValueError("accept-step metrics artifact step_rows length mismatch")
        max_row_candidate_budget = 0
        row_request_ids_by_step: list[tuple[int, ...]] = []
        row_candidate_counts_by_step: list[tuple[int, ...]] = []
        row_accepted_counts_by_step: list[tuple[int, ...]] = []
        for index, row in enumerate(step_rows):
            if not isinstance(row, Mapping):
                raise ValueError("accept-step metrics artifact step_rows entries must be mappings")
            request_ids = row.get("request_ids")
            candidate_counts = row.get("candidate_counts")
            accepted_counts = row.get("accepted_counts")
            if not isinstance(request_ids, Sequence) or isinstance(request_ids, (str, bytes)):
                raise ValueError("accept-step metrics artifact step_rows request_ids must be sequences")
            if not isinstance(candidate_counts, Sequence) or isinstance(candidate_counts, (str, bytes)):
                raise ValueError("accept-step metrics artifact step_rows candidate_counts must be sequences")
            if not isinstance(accepted_counts, Sequence) or isinstance(accepted_counts, (str, bytes)):
                raise ValueError("accept-step metrics artifact step_rows accepted_counts must be sequences")
            row_request_ids = tuple(int(item) for item in request_ids)
            row_candidate_counts = tuple(int(item) for item in candidate_counts)
            row_accepted_counts = tuple(int(item) for item in accepted_counts)
            if not row_candidate_counts:
                raise ValueError("accept-step metrics artifact step_rows candidate_counts must be non-empty")
            max_row_candidate_budget = max(max_row_candidate_budget, max(row_candidate_counts))
            if int(row.get("transaction_id")) != step_transaction_ids[index]:
                raise ValueError("accept-step metrics artifact step_rows transaction_id mismatch")
            if int(row.get("candidate_token_count")) != step_candidate_counts[index]:
                raise ValueError("accept-step metrics artifact step_rows candidate_token_count mismatch")
            if int(row.get("accepted_token_count")) != step_accepted_counts[index]:
                raise ValueError("accept-step metrics artifact step_rows accepted_token_count mismatch")
            if sum(row_candidate_counts) != step_candidate_counts[index]:
                raise ValueError("accept-step metrics artifact step_rows candidate_counts mismatch")
            if sum(row_accepted_counts) != step_accepted_counts[index]:
                raise ValueError("accept-step metrics artifact step_rows accepted_counts mismatch")
            row_request_ids_by_step.append(row_request_ids)
            row_candidate_counts_by_step.append(row_candidate_counts)
            row_accepted_counts_by_step.append(row_accepted_counts)
        if max_row_candidate_budget != candidate_budget:
            raise ValueError("accept-step metrics artifact candidate_budget mismatch")

        steps = payload.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError("accept-step metrics artifact steps must be a sequence")
        if len(steps) != cycle_count:
            raise ValueError("accept-step metrics artifact steps length mismatch")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError("accept-step metrics artifact steps entries must be mappings")
            request_ids = step.get("request_ids")
            candidate_counts = step.get("candidate_counts")
            accepted_counts = step.get("accepted_counts")
            if not isinstance(request_ids, Sequence) or isinstance(request_ids, (str, bytes)):
                raise ValueError("accept-step metrics artifact steps request_ids must be sequences")
            if not isinstance(candidate_counts, Sequence) or isinstance(candidate_counts, (str, bytes)):
                raise ValueError("accept-step metrics artifact steps candidate_counts must be sequences")
            if not isinstance(accepted_counts, Sequence) or isinstance(accepted_counts, (str, bytes)):
                raise ValueError("accept-step metrics artifact steps accepted_counts must be sequences")
            if int(step.get("transaction_id")) != step_transaction_ids[index]:
                raise ValueError("accept-step metrics artifact steps transaction_id mismatch")
            if tuple(int(item) for item in request_ids) != row_request_ids_by_step[index]:
                raise ValueError("accept-step metrics artifact steps request_ids mismatch")
            if tuple(int(item) for item in candidate_counts) != row_candidate_counts_by_step[index]:
                raise ValueError("accept-step metrics artifact steps candidate_counts mismatch")
            if tuple(int(item) for item in accepted_counts) != row_accepted_counts_by_step[index]:
                raise ValueError("accept-step metrics artifact steps accepted_counts mismatch")
            Qwen35GGUFMTPSeedRow.validate_payload(
                step.get("reseed"),
                field_name="accept-step metrics artifact steps reseed",
            )
        if abs(float(payload["accepted_per_draft"]) - (accepted_token_count / draft_token_count)) > 1e-12:
            raise ValueError("accept-step metrics artifact accepted_per_draft mismatch")
        if abs(float(payload["accepted_per_output"]) - (accepted_token_count / output_token_count)) > 1e-12:
            raise ValueError("accept-step metrics artifact accepted_per_output mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            **self.artifact_labels(),
            "cycle_count": self.cycle_count,
            "candidate_budget": self.candidate_budget,
            "budget_label": self.budget_label,
            "draft_token_count": self.draft_token_count,
            "accepted_token_count": self.accepted_token_count,
            "output_token_count": self.output_token_count,
            "step_transaction_ids": list(self.step_transaction_ids),
            "step_candidate_token_counts": list(self.step_candidate_token_counts),
            "step_accepted_token_counts": list(self.step_accepted_token_counts),
            "step_rows": list(self.step_rows()),
            "accepted_per_draft": self.accepted_per_draft,
            "accepted_per_output": self.accepted_per_output,
            "denominators": self.denominator_labels(),
            "required_fields": list(self.required_fields()),
            "validator": self.validator_name(),
            "seed_row_contract": Qwen35GGUFMTPSeedRow.contract(),
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPTop1AcceptSpec:
    """One target-top1 accept cycle for GGUF MTP metrics aggregation."""

    plan: Qwen35GGUFMTPDraftExecutionPlan
    target_top1: Sequence[int]
    transaction_id: int
    verify_seeds: Sequence[Qwen35GGUFMTPSeedRow] = ()
    remaining_decode: Sequence[int] | None = None
    request_id: int | None = None
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        target_top1 = tuple(int(token) for token in self.target_top1)
        if not target_top1:
            raise ValueError("target_top1 must be non-empty")
        transaction_id = int(self.transaction_id)
        if transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        remaining_decode = (
            None
            if self.remaining_decode is None
            else tuple(int(token) for token in self.remaining_decode)
        )
        request_id = None if self.request_id is None else int(self.request_id)
        object.__setattr__(self, "target_top1", target_top1)
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "verify_seeds", tuple(self.verify_seeds))
        object.__setattr__(self, "remaining_decode", remaining_decode)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "mode", str(self.mode))


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
        accepted_prefix = self.proposed_token_ids[: self.n_accepted]
        target_prefix = self.target_token_ids[: self.n_accepted]
        if accepted_prefix != target_prefix:
            raise ValueError("accepted prefix tokens must match target tokens")
        if self.first_mismatch_index is None:
            if self.n_accepted != len(self.proposed_token_ids):
                raise ValueError("missing first_mismatch_index for partial acceptance")
            if self.rejected_proposal_token_id is not None or self.target_token_id_at_mismatch is not None:
                raise ValueError("mismatch token ids must be empty when all drafts are accepted")
        else:
            if self.first_mismatch_index != self.n_accepted:
                raise ValueError("first_mismatch_index must equal n_accepted")
            if self.n_accepted == len(self.proposed_token_ids):
                raise ValueError("first_mismatch_index must be empty when all drafts are accepted")
            if self.rejected_proposal_token_id is None or self.target_token_id_at_mismatch is None:
                raise ValueError("mismatch token ids must be present for partial acceptance")
            rejected = self.proposed_token_ids[self.first_mismatch_index]
            target = self.target_token_ids[self.first_mismatch_index]
            if self.rejected_proposal_token_id != rejected or self.target_token_id_at_mismatch != target:
                raise ValueError("mismatch token ids must match proposed/target tokens")
            if rejected == target:
                raise ValueError("mismatch token ids must differ for partial acceptance")

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


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPRuntimeKernelCheck:
    """One exact four-axis registry check for GGUF MTP runtime readiness."""

    name: str
    key: tuple[str, str, str, str]
    registered: bool
    required_for: str
    missing_is_blocker: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("kernel check name must be non-empty")
        if len(self.key) != 4:
            raise ValueError("kernel check key must be a four-axis registry key")
        if not self.required_for:
            raise ValueError("required_for must be non-empty")

    @classmethod
    def from_key(
        cls,
        *,
        name: str,
        key: tuple[str, str, str, str],
        required_for: str,
        missing_is_blocker: bool,
    ) -> "Qwen35GGUFMTPRuntimeKernelCheck":
        normalized_key = tuple(str(part) for part in key)
        if len(normalized_key) != 4:
            raise ValueError("kernel check key must be a four-axis registry key")
        return cls(
            name=name,
            key=normalized_key,  # type: ignore[arg-type]
            registered=is_registered(KernelKey(*normalized_key)),
            required_for=required_for,
            missing_is_blocker=bool(missing_is_blocker),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "key": list(self.key),
            "registered": self.registered,
            "required_for": self.required_for,
            "missing_is_blocker": self.missing_is_blocker,
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPRuntimeKernelPlan:
    """Shared GGUF MTP runtime/oracle registry readiness contract."""

    backend: str
    checks: tuple[Qwen35GGUFMTPRuntimeKernelCheck, ...]

    def __post_init__(self) -> None:
        target = self.backend
        if not target:
            raise ValueError("backend must be non-empty")
        if not self.checks:
            raise ValueError("runtime kernel plan requires at least one check")

    @classmethod
    def from_registry(
        cls,
        *,
        backend: str,
        draft_topk_kernel: tuple[str, str, str, str] = DEFAULT_DRAFT_TOPK_KERNEL,
    ) -> "Qwen35GGUFMTPRuntimeKernelPlan":
        target = str(backend)
        if not target:
            raise ValueError("backend must be non-empty")
        topk_key = tuple(str(part) for part in draft_topk_kernel)
        if len(topk_key) != 4:
            raise ValueError("draft_topk_kernel must be a four-axis registry key")
        checks = (
            Qwen35GGUFMTPRuntimeKernelCheck.from_key(
                name="cpu_nextn_oracle",
                key=DEFAULT_NEXTN_CPU_REFERENCE_KERNEL,
                required_for="exactness",
                missing_is_blocker=True,
            ),
            Qwen35GGUFMTPRuntimeKernelCheck.from_key(
                name="draft_topk_fallback_oracle",
                key=topk_key,  # type: ignore[arg-type]
                required_for="exactness",
                missing_is_blocker=True,
            ),
            Qwen35GGUFMTPRuntimeKernelCheck.from_key(
                name="native_nextn_runtime",
                key=(target, DEFAULT_NEXTN_LAYER, DEFAULT_NEXTN_QUANT, DEFAULT_NEXTN_VARIANT),
                required_for="native_runtime",
                missing_is_blocker=False,
            ),
            Qwen35GGUFMTPRuntimeKernelCheck.from_key(
                name="native_nextn_paged_kv_write",
                key=(target, DEFAULT_NEXTN_KV_WRITE_LAYER, DEFAULT_NEXTN_QUANT, DEFAULT_NEXTN_KV_WRITE_VARIANT),
                required_for="native_runtime",
                missing_is_blocker=False,
            ),
            Qwen35GGUFMTPRuntimeKernelCheck.from_key(
                name="native_nextn_paged_attn_decode",
                key=(target, DEFAULT_NEXTN_ATTN_DECODE_LAYER, DEFAULT_NEXTN_QUANT, DEFAULT_NEXTN_ATTN_DECODE_VARIANT),
                required_for="native_runtime",
                missing_is_blocker=False,
            ),
            Qwen35GGUFMTPRuntimeKernelCheck.from_key(
                name="native_draft_topk_device",
                key=(target, "mtp_draft_topk", DEFAULT_NEXTN_QUANT, DEFAULT_DRAFT_TOPK_DEVICE_VARIANT),
                required_for="optimization",
                missing_is_blocker=False,
            ),
        )
        return cls(backend=target, checks=checks)

    @property
    def missing_exactness_oracle_keys(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            check.key for check in self.checks if check.missing_is_blocker and not check.registered
        )

    @property
    def missing_native_runtime_keys(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            check.key for check in self.checks if check.required_for == "native_runtime" and not check.registered
        )

    @property
    def missing_optimization_keys(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            check.key for check in self.checks if check.required_for == "optimization" and not check.registered
        )

    @property
    def exactness_oracles_ready(self) -> bool:
        return not self.missing_exactness_oracle_keys

    @property
    def native_runtime_kernels_ready(self) -> bool:
        return not self.missing_native_runtime_keys

    @property
    def optimization_kernels_ready(self) -> bool:
        return not self.missing_optimization_keys

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "exactness_oracles_ready": self.exactness_oracles_ready,
            "native_runtime_kernels_ready": self.native_runtime_kernels_ready,
            "optimization_kernels_ready": self.optimization_kernels_ready,
            "missing_exactness_oracle_keys": [list(key) for key in self.missing_exactness_oracle_keys],
            "missing_native_runtime_keys": [list(key) for key in self.missing_native_runtime_keys],
            "missing_optimization_keys": [list(key) for key in self.missing_optimization_keys],
            "checks": [check.as_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPVerificationMetrics:
    """Aggregate denominator contract for GGUF MTP parity metrics."""

    results: tuple[Qwen35GGUFMTPVerificationResult, ...]
    output_token_count: int

    @classmethod
    def from_results(
        cls,
        results: Sequence[Qwen35GGUFMTPVerificationResult],
        *,
        output_token_count: int,
    ) -> "Qwen35GGUFMTPVerificationMetrics":
        return cls(results=tuple(results), output_token_count=int(output_token_count))

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("verification metrics require at least one result")
        if self.output_token_count <= 0:
            raise ValueError("output_token_count must be positive")
        if self.accepted_token_count > self.output_token_count:
            raise ValueError("accepted draft tokens cannot exceed visible output tokens")

    @property
    def cycle_count(self) -> int:
        return len(self.results)

    @property
    def draft_token_count(self) -> int:
        return sum(len(result.proposed_token_ids) for result in self.results)

    @property
    def accepted_token_count(self) -> int:
        return sum(result.n_accepted for result in self.results)

    @property
    def candidate_budget(self) -> int:
        return max(len(result.proposed_token_ids) for result in self.results)

    @property
    def budget_label(self) -> str:
        return f"B{self.candidate_budget}"

    @property
    def result_draft_token_counts(self) -> tuple[int, ...]:
        return tuple(len(result.proposed_token_ids) for result in self.results)

    @property
    def result_accepted_token_counts(self) -> tuple[int, ...]:
        return tuple(result.n_accepted for result in self.results)

    @property
    def accepted_per_draft(self) -> float:
        return float(self.accepted_token_count) / float(self.draft_token_count)

    @property
    def accepted_per_output(self) -> float:
        return float(self.accepted_token_count) / float(self.output_token_count)

    @staticmethod
    def denominator_labels() -> dict[str, str]:
        return {
            "accepted_per_draft": "accepted_token_count / draft_token_count",
            "accepted_per_output": "accepted_token_count / output_token_count",
        }

    @staticmethod
    def artifact_labels() -> dict[str, object]:
        return {
            "schema": 1,
            "kind": "hipengine_gguf_mtp_verification_metrics",
            "source": "Qwen35GGUFMTPVerificationMetrics",
            "result_source": "Qwen35GGUFMTPVerificationResult",
        }

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return (
            "cycle_count",
            "candidate_budget",
            "budget_label",
            "draft_token_count",
            "accepted_token_count",
            "output_token_count",
            "result_draft_token_counts",
            "result_accepted_token_counts",
            "accepted_per_draft",
            "accepted_per_output",
            "denominators",
            "seed_row_contract",
            "results",
        )

    @staticmethod
    def validator_name() -> str:
        return "Qwen35GGUFMTPVerificationMetrics.validate_payload"

    @classmethod
    def missing_required_fields(cls, payload: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(field for field in cls.required_fields() if field not in payload)

    @classmethod
    def validate_payload(cls, payload: Mapping[str, object]) -> None:
        def int_sequence(name: str) -> tuple[int, ...]:
            value = payload.get(name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"verification metrics artifact {name} must be a sequence")
            return tuple(int(item) for item in value)

        for key, expected in cls.artifact_labels().items():
            if payload.get(key) != expected:
                raise ValueError(f"verification metrics artifact {key} mismatch")
        missing = cls.missing_required_fields(payload)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"verification metrics artifact missing required fields: {joined}")
        required_fields = payload.get("required_fields")
        if not isinstance(required_fields, Sequence) or isinstance(required_fields, (str, bytes)):
            raise ValueError("verification metrics artifact required_fields must be a sequence")
        if tuple(required_fields) != cls.required_fields():
            raise ValueError("verification metrics artifact required_fields mismatch")
        if payload.get("validator") != cls.validator_name():
            raise ValueError("verification metrics artifact validator mismatch")
        if payload.get("denominators") != cls.denominator_labels():
            raise ValueError("verification metrics artifact denominator labels mismatch")
        if payload.get("seed_row_contract") != Qwen35GGUFMTPSeedRow.contract():
            raise ValueError("verification metrics artifact seed_row_contract mismatch")

        cycle_count = int(payload["cycle_count"])
        candidate_budget = int(payload["candidate_budget"])
        draft_token_count = int(payload["draft_token_count"])
        accepted_token_count = int(payload["accepted_token_count"])
        output_token_count = int(payload["output_token_count"])
        if cycle_count <= 0:
            raise ValueError("verification metrics artifact cycle_count must be positive")
        if candidate_budget <= 0:
            raise ValueError("verification metrics artifact candidate_budget must be positive")
        if draft_token_count <= 0:
            raise ValueError("verification metrics artifact draft_token_count must be positive")
        if accepted_token_count < 0:
            raise ValueError("verification metrics artifact accepted_token_count must be non-negative")
        if output_token_count <= 0:
            raise ValueError("verification metrics artifact output_token_count must be positive")
        if accepted_token_count > output_token_count:
            raise ValueError("verification metrics artifact accepted draft tokens cannot exceed visible output tokens")
        if payload.get("budget_label") != f"B{candidate_budget}":
            raise ValueError("verification metrics artifact budget_label mismatch")

        result_draft_counts = int_sequence("result_draft_token_counts")
        result_accepted_counts = int_sequence("result_accepted_token_counts")
        for name, values in (
            ("result_draft_token_counts", result_draft_counts),
            ("result_accepted_token_counts", result_accepted_counts),
        ):
            if len(values) != cycle_count:
                raise ValueError(f"verification metrics artifact {name} length mismatch")
        if any(count <= 0 for count in result_draft_counts):
            raise ValueError("verification metrics artifact result_draft_token_counts must be positive")
        if any(count < 0 for count in result_accepted_counts):
            raise ValueError("verification metrics artifact result_accepted_token_counts must be non-negative")
        if any(accepted > draft for accepted, draft in zip(result_accepted_counts, result_draft_counts, strict=True)):
            raise ValueError("verification metrics artifact accepted counts cannot exceed draft counts")
        if draft_token_count != sum(result_draft_counts):
            raise ValueError("verification metrics artifact draft_token_count mismatch")
        if accepted_token_count != sum(result_accepted_counts):
            raise ValueError("verification metrics artifact accepted_token_count mismatch")
        if candidate_budget != max(result_draft_counts):
            raise ValueError("verification metrics artifact candidate_budget mismatch")

        results = payload.get("results")
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            raise ValueError("verification metrics artifact results must be a sequence")
        if len(results) != cycle_count:
            raise ValueError("verification metrics artifact results length mismatch")
        for index, result in enumerate(results):
            if not isinstance(result, Mapping):
                raise ValueError("verification metrics artifact results entries must be mappings")
            proposed = result.get("proposed_token_ids")
            target = result.get("target_token_ids")
            accepted = result.get("accepted_token_ids")
            if not isinstance(proposed, Sequence) or isinstance(proposed, (str, bytes)):
                raise ValueError("verification metrics artifact results proposed_token_ids must be sequences")
            if not isinstance(target, Sequence) or isinstance(target, (str, bytes)):
                raise ValueError("verification metrics artifact results target_token_ids must be sequences")
            if not isinstance(accepted, Sequence) or isinstance(accepted, (str, bytes)):
                raise ValueError("verification metrics artifact results accepted_token_ids must be sequences")
            proposed_tokens = tuple(int(item) for item in proposed)
            target_tokens = tuple(int(item) for item in target)
            accepted_tokens = tuple(int(item) for item in accepted)
            result_draft_count = result_draft_counts[index]
            result_accepted_count = result_accepted_counts[index]
            if len(proposed_tokens) != result_draft_count:
                raise ValueError("verification metrics artifact results draft count mismatch")
            if len(target_tokens) < result_draft_count:
                raise ValueError("verification metrics artifact results target_token_ids must cover draft tokens")
            if len(accepted_tokens) != result_accepted_count:
                raise ValueError("verification metrics artifact results accepted count mismatch")
            if accepted_tokens != proposed_tokens[:result_accepted_count]:
                raise ValueError("verification metrics artifact results accepted_token_ids mismatch")
            if proposed_tokens[:result_accepted_count] != target_tokens[:result_accepted_count]:
                raise ValueError("verification metrics artifact results accepted prefix mismatch")
            if int(result.get("n_accepted")) != result_accepted_count:
                raise ValueError("verification metrics artifact results n_accepted mismatch")
            if int(result.get("draft_count")) != result_draft_count:
                raise ValueError("verification metrics artifact results draft_count mismatch")
            if int(result.get("verify_seed_count")) <= result_draft_count:
                raise ValueError("verification metrics artifact results verify_seed_count must include the next target row")
            if abs(float(result.get("accepted_per_draft")) - (result_accepted_count / result_draft_count)) > 1e-12:
                raise ValueError("verification metrics artifact results accepted_per_draft mismatch")
            Qwen35GGUFMTPSeedRow.validate_payload(
                result.get("reseed"),
                field_name="verification metrics artifact results reseed",
            )

            first_mismatch_index = result.get("first_mismatch_index")
            rejected_proposal_token_id = result.get("rejected_proposal_token_id")
            target_token_id_at_mismatch = result.get("target_token_id_at_mismatch")
            if first_mismatch_index is None:
                if result_accepted_count != result_draft_count:
                    raise ValueError("verification metrics artifact results missing first_mismatch_index")
                if rejected_proposal_token_id is not None or target_token_id_at_mismatch is not None:
                    raise ValueError("verification metrics artifact results mismatch token ids must be empty")
            else:
                mismatch_index = int(first_mismatch_index)
                if mismatch_index != result_accepted_count:
                    raise ValueError("verification metrics artifact results first_mismatch_index mismatch")
                if result_accepted_count == result_draft_count:
                    raise ValueError("verification metrics artifact results first_mismatch_index must be empty")
                if rejected_proposal_token_id is None or target_token_id_at_mismatch is None:
                    raise ValueError("verification metrics artifact results mismatch token ids must be present")
                rejected = proposed_tokens[mismatch_index]
                mismatch_target = target_tokens[mismatch_index]
                if int(rejected_proposal_token_id) != rejected or int(target_token_id_at_mismatch) != mismatch_target:
                    raise ValueError("verification metrics artifact results mismatch token ids mismatch")
                if rejected == mismatch_target:
                    raise ValueError("verification metrics artifact results mismatch token ids must differ")

        if abs(float(payload["accepted_per_draft"]) - (accepted_token_count / draft_token_count)) > 1e-12:
            raise ValueError("verification metrics artifact accepted_per_draft mismatch")
        if abs(float(payload["accepted_per_output"]) - (accepted_token_count / output_token_count)) > 1e-12:
            raise ValueError("verification metrics artifact accepted_per_output mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            **self.artifact_labels(),
            "cycle_count": self.cycle_count,
            "candidate_budget": self.candidate_budget,
            "budget_label": self.budget_label,
            "draft_token_count": self.draft_token_count,
            "accepted_token_count": self.accepted_token_count,
            "output_token_count": self.output_token_count,
            "result_draft_token_counts": list(self.result_draft_token_counts),
            "result_accepted_token_counts": list(self.result_accepted_token_counts),
            "accepted_per_draft": self.accepted_per_draft,
            "accepted_per_output": self.accepted_per_output,
            "denominators": self.denominator_labels(),
            "required_fields": list(self.required_fields()),
            "validator": self.validator_name(),
            "seed_row_contract": Qwen35GGUFMTPSeedRow.contract(),
            "results": [result.as_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class Qwen35GGUFMTPPerformanceReadiness:
    """M6 readiness gate for GGUF MTP performance comparisons.

    This shared contract intentionally consumes only booleans/status strings from
    preflight/runtime artifacts.  It does not inspect backend names, quant names,
    tensors, or device buffers; runtime dispatch stays behind the four-axis
    kernel registry.
    """

    blockers: tuple[str, ...]

    @classmethod
    def from_gate_inputs(
        cls,
        *,
        parity_precheck: bool,
        draft_budget_precheck: bool,
        draft_sampling_contract_precheck: bool,
        hidden_seed_contract_precheck: bool,
        exactness_gate: str,
        kvlivespans_paged_cache_smoke: bool,
        llamacpp_trace_budget_coverage: str,
        accepted_per_draft_status: str,
        accepted_per_output_status: str,
        native_runtime_kernels_ready: bool,
        optimization_kernels_ready: bool,
        metrics_contract_status: str,
    ) -> "Qwen35GGUFMTPPerformanceReadiness":
        blockers: list[str] = []
        if not parity_precheck:
            blockers.append("parity_precheck_failed")
        if not draft_budget_precheck:
            blockers.append("draft_budget_precheck_failed")
        if not draft_sampling_contract_precheck:
            blockers.append("draft_sampling_contract_precheck_failed")
        if not hidden_seed_contract_precheck:
            blockers.append("hidden_seed_contract_precheck_failed")
        if exactness_gate != "passed":
            blockers.append("exactness_gate_failed")
        if not kvlivespans_paged_cache_smoke:
            blockers.append("kvlivespans_paged_cache_smoke_failed")
        if llamacpp_trace_budget_coverage != GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE:
            blockers.append("partial_llamacpp_trace_budget_coverage")
        if accepted_per_draft_status != GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE:
            blockers.append("accepted_draft_denominator_not_comparable")
        if accepted_per_output_status != GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE:
            blockers.append("accepted_output_denominator_not_comparable")
        if not native_runtime_kernels_ready:
            blockers.append("native_runtime_kernels_missing")
        if not optimization_kernels_ready:
            blockers.append("optimization_kernels_missing")
        if metrics_contract_status != GGUF_MTP_METRICS_CONTRACT_READY:
            blockers.append("hipengine_metrics_not_ready")
        return cls(blockers=tuple(blockers))

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
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

    def accept_target_summary(
        self,
        summary: TargetAcceptSummary,
        *,
        request_id: int | None = None,
    ) -> Qwen35GGUFMTPSeedRow:
        """Apply a shared target accept summary to GGUF hidden-seed state.

        The provider-neutral summary fixes the accepted count and commit row;
        GGUF MTP still reseeds from the recorded verify hidden rows using the
        llama.cpp rule ``verify_h[min(n_accepted, n_rows - 1)]``.  This context
        owns a single pending seed, so multi-request summaries must be split by
        the scheduler before calling this method.
        """

        if len(summary.request_ids) != 1:
            raise ValueError("GGUF MTP context accepts one request summary at a time")
        summary_request_id = summary.request_ids[0]
        if request_id is not None and int(request_id) != summary_request_id:
            raise ValueError("target accept summary request_id must match context request")
        if summary.candidate_counts is not None:
            candidate_count = summary.candidate_counts[0]
            if len(self.verify_seeds) <= candidate_count:
                raise RuntimeError("verify seeds must include candidate rows plus the next target row")
        return self.accept(summary.accepted_counts[0])

    def accept_target_commit_plan(
        self,
        commit_plan: TargetCommitPlan,
        *,
        request_id: int | None = None,
    ) -> Qwen35GGUFMTPSeedRow:
        """Apply a validated target commit plan to GGUF hidden-seed state.

        Future scheduler/KV integration will hand back a committed transaction
        contract rather than the pre-commit summary.  The GGUF reseed rule is
        still driven by the accepted count, but this method ties it to the same
        commit metadata that will be used for KV/state commit.
        """

        if len(commit_plan.request_ids) != 1:
            raise ValueError("GGUF MTP context accepts one request commit plan at a time")
        plan_request_id = commit_plan.request_ids[0]
        if request_id is not None and int(request_id) != plan_request_id:
            raise ValueError("target commit plan request_id must match context request")
        if commit_plan.candidate_counts is not None:
            candidate_count = commit_plan.candidate_counts[0]
            if len(self.verify_seeds) <= candidate_count:
                raise RuntimeError("verify seeds must include candidate rows plus the next target row")
        return self.accept(commit_plan.accepted_counts[0])

    def accept_target_top1(
        self,
        plan: Qwen35GGUFMTPDraftExecutionPlan,
        target_top1: Sequence[int],
        *,
        transaction_id: int,
        remaining_decode: Sequence[int] | None = None,
        request_id: int | None = None,
        mode: str = "verify_chain",
    ) -> Qwen35GGUFMTPAcceptStep:
        """Build/apply a validated target commit plan from top-1 verifier rows.

        This is the metadata-only boundary for the future native GGUF MTP target
        verify/commit loop: top-1 rows become a validated commit plan, then the
        context applies the committed accepted count to the llama.cpp hidden-seed
        reseed rule.
        """

        commit_plan = plan.target_commit_plan_from_top1(
            target_top1,
            transaction_id=transaction_id,
            remaining_decode=remaining_decode,
            mode=mode,
        )
        seed = self.accept_target_commit_plan(commit_plan, request_id=request_id)
        return Qwen35GGUFMTPAcceptStep(commit_plan=commit_plan, reseed=seed)

    def accept_target_top1_metrics(
        self,
        specs: Sequence[Qwen35GGUFMTPTop1AcceptSpec],
        *,
        output_token_count: int,
    ) -> Qwen35GGUFMTPAcceptStepMetrics:
        """Run a metadata-only sequence of target-top1 accept cycles.

        Each spec may include the verify hidden rows captured for that cycle;
        when present they replace the context's recorded verify seeds before the
        target-top1 commit/reseed step.  The returned metrics object is the
        compact artifact contract used for future llama.cpp parity rows.
        """

        steps: list[Qwen35GGUFMTPAcceptStep] = []
        for spec in specs:
            if spec.verify_seeds:
                self.record_verify_seeds(spec.verify_seeds)
            steps.append(
                self.accept_target_top1(
                    spec.plan,
                    spec.target_top1,
                    transaction_id=spec.transaction_id,
                    remaining_decode=spec.remaining_decode,
                    request_id=spec.request_id,
                    mode=spec.mode,
                )
            )
        return Qwen35GGUFMTPAcceptStepMetrics.from_steps(steps, output_token_count=output_token_count)

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
