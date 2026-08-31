"""Backend-neutral SPECDEC2 capability, plan, and frontier contracts.

These records describe one bounded scheduler-owned transition.  They contain no
backend dispatch policy and no framework tensors: a provider may publish host
candidate ids for tests/slow fallbacks or a torch-free ``Tensor`` descriptor for
normal device-resident lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from hipengine.core.tensor import Tensor
from hipengine.speculative.interfaces import DraftBatch, TargetVerifyBatch


def _required_text(value: object, label: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return text


def _unique_nonnegative(values: Sequence[int], label: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{label} must be non-negative")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    return normalized


def _positive_values(values: Sequence[int], label: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError(f"{label} must contain positive values")
    return normalized


def _positive_unique(values: Sequence[int], label: str) -> tuple[int, ...]:
    normalized = _positive_values(values, label)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    return normalized


class ProviderAttachment(StrEnum):
    """How provider state relates to the target model."""

    TARGET_ATTACHED = "target_attached"
    INDEPENDENT = "independent"
    HOST = "host"
    REMOTE = "remote"


class ProviderCatchupMode(StrEnum):
    """How provider state remains valid across target-only K=0 transitions."""

    TARGET_OUTPUT = "target_output"
    BOUNDED_RESEED = "bounded_reseed"
    ONE_WAY_AR = "one_way_ar"
    UNSUPPORTED = "unsupported"


class SpecTransactionMode(StrEnum):
    """Declared provisional state/KV ownership mechanism."""

    RESERVED_APPEND = "reserved_append"
    PACKED_SCRATCH = "packed_scratch"
    REVERSIBLE_JOURNAL = "reversible_journal"


class SpecK0Class(StrEnum):
    """Whether a K0 row is provider-free or keeps an eligible provider synced."""

    NOT_K0 = "not_k0"
    PURE = "pure_k0"
    TRANSITIONAL = "transitional_k0"


class SpecPlanReason(StrEnum):
    """Stable pre-mutation reason for a speculative or K=0 decision."""

    SPECULATIVE_QUALIFIED = "speculative_qualified"
    NO_PROVIDER = "no_provider"
    POLICY_SELECTED_AR = "policy_selected_ar"
    UNSUPPORTED_SAMPLING = "unsupported_sampling"
    TARGET_GRAPH_CONTEXT_BUCKET_MISS = "target_graph_context_bucket_miss"
    TARGET_GRAPH_OUTPUT_ROOM_MISS = "target_graph_output_room_miss"
    RESOURCE_CLAIM_MISS = "resource_claim_miss"
    TARGET_PHYSICAL_BUCKET_MISS = "target_physical_bucket_miss"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    PROVIDER_CATCHUP_UNAVAILABLE = "provider_catchup_unavailable"


@dataclass(frozen=True, slots=True)
class SpeculativeCapability:
    """One fully resolved provider/target execution capability.

    This is a composed cold-plan record, not a fifth registry axis.  Exact
    provider and target factories still resolve through their plugin keys.
    """

    capability_key: str
    target_key: str
    provider_key: str
    method_key: str
    policy_fingerprint: str
    execution_profile: str
    kv_backend_key: str
    attachment: ProviderAttachment
    catchup_mode: ProviderCatchupMode
    supported_modes: tuple[str, ...]
    supported_sampling_modes: tuple[str, ...]
    max_requests: int
    max_candidates_per_request: int
    max_frontier_rows: int
    proposal_widths: tuple[int, ...]
    target_row_buckets: tuple[int, ...]
    target_transaction_mode: SpecTransactionMode
    provider_transaction_mode: SpecTransactionMode
    graph_supported: bool
    eager_supported: bool
    strict_fallback_key: str
    max_context_tokens: int | None = None
    terminal_zero_accept_supported: bool = False

    def __post_init__(self) -> None:
        for field in (
            "capability_key",
            "target_key",
            "provider_key",
            "method_key",
            "policy_fingerprint",
            "execution_profile",
            "kv_backend_key",
            "strict_fallback_key",
        ):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        object.__setattr__(self, "attachment", ProviderAttachment(self.attachment))
        object.__setattr__(self, "catchup_mode", ProviderCatchupMode(self.catchup_mode))
        object.__setattr__(
            self,
            "target_transaction_mode",
            SpecTransactionMode(self.target_transaction_mode),
        )
        object.__setattr__(
            self,
            "provider_transaction_mode",
            SpecTransactionMode(self.provider_transaction_mode),
        )
        modes = tuple(str(mode) for mode in self.supported_modes)
        if not modes or len(modes) != len(set(modes)):
            raise ValueError("supported_modes must be non-empty and unique")
        if any(mode not in {"verify_chain", "verify_tree"} for mode in modes):
            raise ValueError("supported_modes may contain verify_chain/verify_tree only")
        sampling = tuple(str(mode).strip() for mode in self.supported_sampling_modes)
        if not sampling or len(sampling) != len(set(sampling)) or any(not mode for mode in sampling):
            raise ValueError("supported_sampling_modes must be non-empty and unique")
        object.__setattr__(self, "supported_modes", modes)
        object.__setattr__(self, "supported_sampling_modes", sampling)
        for field in ("max_requests", "max_candidates_per_request", "max_frontier_rows"):
            value = int(getattr(self, field))
            if value <= 0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        proposal_widths = _positive_unique(self.proposal_widths, "proposal_widths")
        if any(width > self.max_requests for width in proposal_widths):
            raise ValueError("proposal width exceeds max_requests")
        target_buckets = _positive_unique(self.target_row_buckets, "target_row_buckets")
        if any(rows > self.max_frontier_rows for rows in target_buckets):
            raise ValueError("target row bucket exceeds max_frontier_rows")
        object.__setattr__(self, "proposal_widths", proposal_widths)
        object.__setattr__(self, "target_row_buckets", target_buckets)
        if not self.graph_supported and not self.eager_supported:
            raise ValueError("capability must declare at least one execution route")
        if self.max_context_tokens is not None:
            max_context_tokens = int(self.max_context_tokens)
            if max_context_tokens <= 0:
                raise ValueError("max_context_tokens must be positive when set")
            object.__setattr__(self, "max_context_tokens", max_context_tokens)

    def supports_shape(
        self,
        *,
        request_count: int,
        candidate_counts: Sequence[int],
        mode: str,
    ) -> bool:
        count = int(request_count)
        candidates = tuple(int(value) for value in candidate_counts)
        if count <= 0 or count != len(candidates) or count > self.max_requests:
            return False
        if str(mode) not in self.supported_modes:
            return False
        if any(value < 0 or value > self.max_candidates_per_request for value in candidates):
            return False
        if not any(candidates):
            return False
        return count + sum(candidates) <= self.max_frontier_rows

    def supports_sampling(self, sampling_mode: str) -> bool:
        return str(sampling_mode) in self.supported_sampling_modes


@dataclass(frozen=True, slots=True)
class SpecRequestPlan:
    """Immutable pre-mutation K/K0 decision for one fairness group."""

    operation_id: str
    cycle_id: int
    request_ids: tuple[int, ...]
    resident_slots: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    reasons: tuple[SpecPlanReason, ...]
    k0_classes: tuple[SpecK0Class, ...]
    mode: str
    capability_key: str | None
    provider_key: str | None
    target_transaction_mode: SpecTransactionMode
    provider_transaction_mode: SpecTransactionMode | None
    proposal_widths: tuple[int, ...]
    target_row_decomposition: tuple[int, ...]
    context_bucket_size: int
    execution_route: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        cycle_id = int(self.cycle_id)
        if cycle_id < 0:
            raise ValueError("cycle_id must be non-negative")
        object.__setattr__(self, "cycle_id", cycle_id)
        request_ids = _unique_nonnegative(self.request_ids, "request_ids")
        if not request_ids:
            raise ValueError("request_ids must be non-empty")
        slots = _unique_nonnegative(self.resident_slots, "resident_slots")
        if len(slots) != len(request_ids):
            raise ValueError("resident_slots must align with request_ids")
        counts = tuple(int(value) for value in self.candidate_counts)
        if len(counts) != len(request_ids) or any(value < 0 for value in counts):
            raise ValueError("candidate_counts must be non-negative and align with request_ids")
        reasons = tuple(SpecPlanReason(reason) for reason in self.reasons)
        if len(reasons) != len(request_ids):
            raise ValueError("reasons must align with request_ids")
        k0_classes = tuple(SpecK0Class(value) for value in self.k0_classes)
        if len(k0_classes) != len(request_ids):
            raise ValueError("k0_classes must align with request_ids")
        for count, reason, k0_class in zip(
            counts,
            reasons,
            k0_classes,
            strict=True,
        ):
            if count > 0 and reason is not SpecPlanReason.SPECULATIVE_QUALIFIED:
                raise ValueError("positive candidate counts require speculative-qualified reason")
            if count == 0 and reason is SpecPlanReason.SPECULATIVE_QUALIFIED:
                raise ValueError("speculative-qualified reason requires positive candidate count")
            if count > 0 and k0_class is not SpecK0Class.NOT_K0:
                raise ValueError("speculative rows require not_k0 classification")
            if count == 0 and k0_class is SpecK0Class.NOT_K0:
                raise ValueError("K0 rows require pure or transitional classification")
        mode = str(self.mode)
        if mode not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("mode must be decode, verify_chain, or verify_tree")
        has_spec = any(counts)
        if has_spec:
            if mode == "decode":
                raise ValueError("speculative plans require a verify mode")
            capability_key = _required_text(self.capability_key, "capability_key")
            provider_key = _required_text(self.provider_key, "provider_key")
            if self.provider_transaction_mode is None:
                raise ValueError("speculative plans require provider_transaction_mode")
            provider_mode = SpecTransactionMode(self.provider_transaction_mode)
            proposal_widths = _positive_values(self.proposal_widths, "proposal_widths")
            if sum(proposal_widths) != sum(1 for count in counts if count > 0):
                raise ValueError("proposal_widths must cover each speculative request exactly once")
            if self.execution_route not in {"graph", "eager"}:
                raise ValueError("speculative execution_route must be graph or eager")
        else:
            if mode != "decode":
                raise ValueError("K0-only plans must use decode mode")
            if self.capability_key is not None or self.provider_key is not None:
                raise ValueError("K0-only plans cannot retain provider capability")
            if self.provider_transaction_mode is not None or self.proposal_widths:
                raise ValueError("K0-only plans cannot open provider ownership")
            capability_key = None
            provider_key = None
            provider_mode = None
            proposal_widths = ()
            if self.execution_route != "ar":
                raise ValueError("K0-only execution_route must be ar")
        target_rows = _positive_values(
            self.target_row_decomposition, "target_row_decomposition"
        )
        logical_rows = len(request_ids) + sum(counts)
        if sum(target_rows) != logical_rows:
            raise ValueError("target_row_decomposition must sum to logical frontier rows")
        context_bucket_size = int(self.context_bucket_size)
        if context_bucket_size <= 0:
            raise ValueError("context_bucket_size must be positive")
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "resident_slots", slots)
        object.__setattr__(self, "candidate_counts", counts)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "k0_classes", k0_classes)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "capability_key", capability_key)
        object.__setattr__(self, "provider_key", provider_key)
        object.__setattr__(
            self,
            "target_transaction_mode",
            SpecTransactionMode(self.target_transaction_mode),
        )
        object.__setattr__(self, "provider_transaction_mode", provider_mode)
        object.__setattr__(self, "proposal_widths", proposal_widths)
        object.__setattr__(self, "target_row_decomposition", target_rows)
        object.__setattr__(self, "context_bucket_size", context_bucket_size)

    @property
    def has_speculative_rows(self) -> bool:
        return any(self.candidate_counts)

    @property
    def is_ar_only(self) -> bool:
        return not self.has_speculative_rows

    @property
    def logical_frontier_rows(self) -> int:
        return len(self.request_ids) + sum(self.candidate_counts)

    @property
    def max_candidate_count(self) -> int:
        return max(self.candidate_counts, default=0)

    @property
    def speculative_request_ids(self) -> tuple[int, ...]:
        return tuple(
            request_id
            for request_id, count in zip(self.request_ids, self.candidate_counts, strict=True)
            if count > 0
        )


@dataclass(frozen=True, slots=True)
class CandidateGraph:
    """Provider-owned candidate topology with host or device token ids."""

    provider_key: str
    method_key: str
    policy_fingerprint: str
    cycle_id: int
    transaction_id: int
    request_ids: tuple[int, ...]
    resident_slots: tuple[int, ...]
    root_positions: tuple[int, ...]
    row_offsets: tuple[int, ...]
    row_to_request: tuple[int, ...]
    parent_candidate_rows: tuple[int, ...]
    draft_depths: tuple[int, ...]
    active_mask: tuple[bool, ...]
    candidate_tokens: tuple[int, ...] = ()
    token_ids: Tensor | None = None
    candidate_ids: tuple[int, ...] = ()
    mode: str = "verify_chain"
    provider_metadata: tuple[tuple[str, int | str], ...] = ()

    def __post_init__(self) -> None:
        for field in ("provider_key", "method_key", "policy_fingerprint"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        for field in ("cycle_id", "transaction_id"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        request_ids = _unique_nonnegative(self.request_ids, "request_ids")
        if not request_ids:
            raise ValueError("request_ids must be non-empty")
        slots = _unique_nonnegative(self.resident_slots, "resident_slots")
        roots = tuple(int(value) for value in self.root_positions)
        if len(slots) != len(request_ids) or len(roots) != len(request_ids):
            raise ValueError("resident_slots/root_positions must align with request_ids")
        if any(position < 0 for position in roots):
            raise ValueError("root_positions must be non-negative")
        offsets = tuple(int(value) for value in self.row_offsets)
        if len(offsets) != len(request_ids) + 1 or offsets[0] != 0:
            raise ValueError("row_offsets must start at zero and align with request_ids")
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError("row_offsets must be nondecreasing")
        rows = offsets[-1]
        if rows <= 0:
            raise ValueError("CandidateGraph must contain at least one candidate row")
        aligned = (
            len(self.row_to_request),
            len(self.parent_candidate_rows),
            len(self.draft_depths),
            len(self.active_mask),
        )
        if any(length != rows for length in aligned):
            raise ValueError("candidate topology fields must align with row_offsets")
        row_to_request = tuple(int(value) for value in self.row_to_request)
        parents = tuple(int(value) for value in self.parent_candidate_rows)
        depths = tuple(int(value) for value in self.draft_depths)
        active = tuple(bool(value) for value in self.active_mask)
        request_by_row: list[int] = []
        for request_id, start, end in zip(
            request_ids, offsets[:-1], offsets[1:], strict=True
        ):
            request_by_row.extend([request_id] * (end - start))
        if row_to_request != tuple(request_by_row):
            raise ValueError("row_to_request must match request-local row_offsets")
        for row, (parent, depth) in enumerate(zip(parents, depths, strict=True)):
            if parent < -1 or parent >= row:
                raise ValueError("candidate parent must be -1 or an earlier row")
            if parent < 0:
                if depth != 1:
                    raise ValueError("root candidates must have draft depth 1")
            else:
                if row_to_request[parent] != row_to_request[row]:
                    raise ValueError("candidate parent must belong to the same request")
                if depth != depths[parent] + 1:
                    raise ValueError("candidate depth must be parent depth plus one")
        host_tokens = tuple(int(token) for token in self.candidate_tokens)
        if host_tokens and (len(host_tokens) != rows or any(token < 0 for token in host_tokens)):
            raise ValueError("candidate_tokens must be non-negative and align with rows")
        if self.token_ids is not None and self.token_ids.shape != (rows,):
            raise ValueError("token_ids must have shape (candidate_rows,)")
        if not host_tokens and self.token_ids is None:
            raise ValueError("CandidateGraph requires host candidate_tokens or device token_ids")
        candidate_ids = tuple(int(value) for value in self.candidate_ids)
        if candidate_ids and (len(candidate_ids) != rows or any(value < 0 for value in candidate_ids)):
            raise ValueError("candidate_ids must be non-negative and align with rows")
        mode = str(self.mode)
        if mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")
        metadata = tuple((str(key), value) for key, value in self.provider_metadata)
        keys = tuple(key for key, _value in metadata)
        if len(keys) != len(set(keys)) or any(not key or key != key.strip() for key in keys):
            raise ValueError("provider_metadata keys must be unique non-empty strings")
        if any(not isinstance(value, (int, str)) for _key, value in metadata):
            raise TypeError("provider_metadata values must be int or str")
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "resident_slots", slots)
        object.__setattr__(self, "root_positions", roots)
        object.__setattr__(self, "row_offsets", offsets)
        object.__setattr__(self, "row_to_request", row_to_request)
        object.__setattr__(self, "parent_candidate_rows", parents)
        object.__setattr__(self, "draft_depths", depths)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "candidate_tokens", host_tokens)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "provider_metadata", metadata)

    @property
    def candidate_rows(self) -> int:
        return self.row_offsets[-1]

    @property
    def candidate_counts(self) -> tuple[int, ...]:
        return tuple(
            end - start
            for start, end in zip(
                self.row_offsets[:-1], self.row_offsets[1:], strict=True
            )
        )

    @property
    def device(self):
        return None if self.token_ids is None else self.token_ids.device

    def to_draft_batch(self) -> DraftBatch:
        """Project host-visible candidates into the retained oracle contract."""

        if not self.candidate_tokens:
            raise RuntimeError("host candidate tokens are required for DraftBatch projection")
        slot_by_request = dict(zip(self.request_ids, self.resident_slots, strict=True))
        root_by_request = dict(zip(self.request_ids, self.root_positions, strict=True))
        return DraftBatch(
            request_ids=self.request_ids,
            candidate_tokens=self.candidate_tokens,
            parent_positions=tuple(
                root_by_request[request_id] + depth - 1
                for request_id, depth in zip(
                    self.row_to_request, self.draft_depths, strict=True
                )
            ),
            draft_depths=self.draft_depths,
            row_to_request=self.row_to_request,
            tree_parents=self.parent_candidate_rows,
            active_mask=self.active_mask,
            mode=self.mode,
            cycle_id=self.cycle_id,
            resident_slots=tuple(slot_by_request[request_id] for request_id in self.row_to_request),
            candidate_ids=self.candidate_ids,
            provider_metadata=self.provider_metadata,
        )


def physical_group_pad_rows(
    admitted_counts: Sequence[int],
    request_count: int,
    candidate_rows: int,
    max_rows: int,
    *,
    exact_counts: Sequence[int] = (),
) -> int:
    """Return inactive pad rows lifting a physical group to admitted multiples.

    Backends that admit only one rowtile row count (gfx1100 rows6) qualify
    exactly that launch shape. Groups pad up to the next multiple of the
    smallest admitted count so a chunked dispatch can run every launch at the
    admitted shape; groups whose next multiple exceeds the accept-buffer row
    capacity stay unpadded (they keep the strict fallback route). Independently
    qualified exact row counts bypass padding without changing that fallback
    multiple for any other width.
    """

    physical = int(request_count) + int(candidate_rows)
    if physical <= 0:
        return 0
    exact = frozenset(int(value) for value in exact_counts if int(value) > 0)
    if physical in exact:
        return 0
    counts = tuple(int(value) for value in admitted_counts if int(value) > 0)
    if not counts:
        return 0
    step = min(counts)
    if physical % step == 0:
        return 0
    padded = ((physical // step) + 1) * step
    if padded > int(max_rows):
        return 0
    return padded - physical


def pad_candidate_graph_rows(
    graph: CandidateGraph,
    *,
    pad_rows: int,
    pad_token_id: int,
    token_ids: "Tensor | None" = None,
) -> CandidateGraph:
    """Append inactive candidate rows owned by the graph's last request.

    Physical target verification prefers the backend's admitted production
    rowtile row counts (gfx1100: rows6). Groups whose root+candidate total
    falls below that shape pay the shared-B padded-tile fallback instead.
    Padding appends ``pad_rows`` inactive candidate rows to the last request so
    the physical launch rides the qualified rowtile while accept, commit, and
    acceptance accounting stay driven by the active rows only. ``token_ids``
    optionally supplies a device tensor covering the padded row count; when
    omitted, host candidate tokens are extended with ``pad_token_id``.
    """

    pads = int(pad_rows)
    if pads <= 0:
        raise ValueError("pad_rows must be positive")
    if int(pad_token_id) < 0:
        raise ValueError("pad_token_id must be non-negative")
    if not graph.request_ids:
        raise ValueError("candidate graph requires at least one request")
    owner = int(graph.request_ids[-1])
    base_rows = graph.candidate_rows
    owner_rows = base_rows - int(graph.row_offsets[-2])
    parents: list[int] = list(graph.parent_candidate_rows)
    depths: list[int] = list(graph.draft_depths)
    previous_row = base_rows - 1 if owner_rows else -1
    previous_depth = depths[previous_row] if previous_row >= 0 else 0
    for pad_index in range(pads):
        parents.append(previous_row)
        previous_depth += 1
        depths.append(previous_depth)
        previous_row = base_rows + pad_index
    host_tokens: tuple[int, ...] = ()
    if graph.candidate_tokens:
        host_tokens = tuple(graph.candidate_tokens) + (int(pad_token_id),) * pads
    elif token_ids is None:
        raise ValueError("device-token graphs require a padded token_ids tensor")
    candidate_ids = graph.candidate_ids
    if candidate_ids:
        candidate_ids = tuple(candidate_ids) + (candidate_ids[-1],) * pads
    return CandidateGraph(
        provider_key=graph.provider_key,
        method_key=graph.method_key,
        policy_fingerprint=graph.policy_fingerprint,
        cycle_id=graph.cycle_id,
        transaction_id=graph.transaction_id,
        request_ids=graph.request_ids,
        resident_slots=graph.resident_slots,
        root_positions=graph.root_positions,
        row_offsets=(*graph.row_offsets[:-1], base_rows + pads),
        row_to_request=(*graph.row_to_request, *((owner,) * pads)),
        parent_candidate_rows=tuple(parents),
        draft_depths=tuple(depths),
        active_mask=(*graph.active_mask, *((False,) * pads)),
        candidate_tokens=host_tokens,
        token_ids=token_ids if token_ids is not None else graph.token_ids,
        candidate_ids=candidate_ids,
        mode=graph.mode,
        provider_metadata=graph.provider_metadata,
    )


@dataclass(frozen=True, slots=True)
class TargetFrontier:
    """Canonical root-only or root+candidate target work for one cycle."""

    operation_id: str
    cycle_id: int
    request_ids: tuple[int, ...]
    resident_slots: tuple[int, ...]
    root_tokens: tuple[int, ...]
    root_positions: tuple[int, ...]
    physical_row_decomposition: tuple[int, ...]
    transaction_mode: SpecTransactionMode
    kv_storage_view_key: str
    kv_live_spans_owner: str
    execution_route: str
    candidate_graph: CandidateGraph | None = None
    target_batch: TargetVerifyBatch | None = None
    provider_transaction_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        cycle_id = int(self.cycle_id)
        if cycle_id < 0:
            raise ValueError("cycle_id must be non-negative")
        request_ids = _unique_nonnegative(self.request_ids, "request_ids")
        if not request_ids:
            raise ValueError("request_ids must be non-empty")
        slots = _unique_nonnegative(self.resident_slots, "resident_slots")
        roots = tuple(int(value) for value in self.root_tokens)
        positions = tuple(int(value) for value in self.root_positions)
        if any(len(values) != len(request_ids) for values in (slots, roots, positions)):
            raise ValueError("resident/root fields must align with request_ids")
        if any(value < 0 for value in (*roots, *positions)):
            raise ValueError("root tokens/positions must be non-negative")
        decomposition = _positive_values(
            self.physical_row_decomposition, "physical_row_decomposition"
        )
        if sum(decomposition) != self.logical_rows:
            raise ValueError("physical_row_decomposition must sum to logical rows")
        if self.candidate_graph is not None:
            if self.candidate_graph.request_ids != request_ids:
                raise ValueError("candidate_graph request_ids must match frontier")
            if self.candidate_graph.resident_slots != slots:
                raise ValueError("candidate_graph resident_slots must match frontier")
            if self.candidate_graph.root_positions != positions:
                raise ValueError("candidate_graph root positions must match frontier")
            if self.provider_transaction_id != self.candidate_graph.transaction_id:
                raise ValueError("candidate frontier requires matching provider_transaction_id")
            if self.target_batch is not None:
                if self.target_batch.request_ids != request_ids:
                    raise ValueError("target_batch request_ids must match frontier")
                if tuple(self.target_batch.tokens[row] for row in self.target_batch.root_rows) != roots:
                    raise ValueError("target_batch root tokens must match frontier")
                if tuple(self.target_batch.positions[row] for row in self.target_batch.root_rows) != positions:
                    raise ValueError("target_batch root positions must match frontier")
                if self.target_batch.rows != len(request_ids) + self.candidate_graph.candidate_rows:
                    raise ValueError("target_batch rows must match candidate_graph")
        elif self.target_batch is not None:
            raise ValueError("target_batch requires candidate_graph ownership")
        elif self.provider_transaction_id is not None:
            raise ValueError("root-only frontier cannot retain provider transaction")
        if self.provider_transaction_id is not None and int(self.provider_transaction_id) < 0:
            raise ValueError("provider_transaction_id must be non-negative")
        route = str(self.execution_route)
        if route not in {"ar", "graph", "eager"}:
            raise ValueError("execution_route must be ar, graph, or eager")
        if self.candidate_graph is None and route != "ar":
            raise ValueError("root-only frontier must use ar execution_route")
        if self.candidate_graph is not None and route == "ar":
            raise ValueError("candidate frontier cannot use ar execution_route")
        object.__setattr__(self, "cycle_id", cycle_id)
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "resident_slots", slots)
        object.__setattr__(self, "root_tokens", roots)
        object.__setattr__(self, "root_positions", positions)
        object.__setattr__(self, "physical_row_decomposition", decomposition)
        object.__setattr__(self, "transaction_mode", SpecTransactionMode(self.transaction_mode))
        object.__setattr__(
            self, "kv_storage_view_key", _required_text(self.kv_storage_view_key, "kv_storage_view_key")
        )
        object.__setattr__(
            self, "kv_live_spans_owner", _required_text(self.kv_live_spans_owner, "kv_live_spans_owner")
        )
        object.__setattr__(self, "execution_route", route)
        if self.provider_transaction_id is not None:
            object.__setattr__(self, "provider_transaction_id", int(self.provider_transaction_id))

    @classmethod
    def from_ar_roots(
        cls,
        *,
        operation_id: str,
        cycle_id: int,
        request_ids: Sequence[int],
        resident_slots: Sequence[int],
        root_tokens: Sequence[int],
        root_positions: Sequence[int],
        physical_row_decomposition: Sequence[int],
        transaction_mode: SpecTransactionMode,
        kv_storage_view_key: str,
        kv_live_spans_owner: str,
        execution_route: str = "ar",
    ) -> "TargetFrontier":
        return cls(
            operation_id=operation_id,
            cycle_id=int(cycle_id),
            request_ids=tuple(int(value) for value in request_ids),
            resident_slots=tuple(int(value) for value in resident_slots),
            root_tokens=tuple(int(value) for value in root_tokens),
            root_positions=tuple(int(value) for value in root_positions),
            physical_row_decomposition=tuple(int(value) for value in physical_row_decomposition),
            transaction_mode=transaction_mode,
            kv_storage_view_key=kv_storage_view_key,
            kv_live_spans_owner=kv_live_spans_owner,
            execution_route=execution_route,
        )

    @classmethod
    def from_candidate_graph(
        cls,
        *,
        operation_id: str,
        candidate_graph: CandidateGraph,
        root_tokens: Sequence[int],
        physical_row_decomposition: Sequence[int],
        transaction_mode: SpecTransactionMode,
        kv_storage_view_key: str,
        kv_live_spans_owner: str,
        execution_route: str,
    ) -> "TargetFrontier":
        roots = tuple(int(value) for value in root_tokens)
        target = None
        if candidate_graph.candidate_tokens:
            target = TargetVerifyBatch.from_draft(
                candidate_graph.to_draft_batch(),
                root_tokens=roots,
                root_positions=candidate_graph.root_positions,
            )
        return cls(
            operation_id=operation_id,
            cycle_id=candidate_graph.cycle_id,
            request_ids=candidate_graph.request_ids,
            resident_slots=candidate_graph.resident_slots,
            root_tokens=roots,
            root_positions=candidate_graph.root_positions,
            physical_row_decomposition=tuple(int(value) for value in physical_row_decomposition),
            transaction_mode=transaction_mode,
            kv_storage_view_key=kv_storage_view_key,
            kv_live_spans_owner=kv_live_spans_owner,
            execution_route=execution_route,
            candidate_graph=candidate_graph,
            target_batch=target,
            provider_transaction_id=candidate_graph.transaction_id,
        )

    @property
    def logical_rows(self) -> int:
        if self.candidate_graph is None:
            return len(self.request_ids)
        return len(self.request_ids) + self.candidate_graph.candidate_rows

    @property
    def candidate_counts(self) -> tuple[int, ...]:
        if self.candidate_graph is None:
            return (0,) * len(self.request_ids)
        return self.candidate_graph.candidate_counts

    @property
    def is_ar_only(self) -> bool:
        return self.candidate_graph is None


__all__ = [
    "CandidateGraph",
    "ProviderAttachment",
    "ProviderCatchupMode",
    "SpecK0Class",
    "SpecPlanReason",
    "SpecRequestPlan",
    "SpecTransactionMode",
    "SpeculativeCapability",
    "TargetFrontier",
]
