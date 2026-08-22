"""SPEC-C2 verifier-specific cost records and continuous packing policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


def _text(value: object, label: str) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return result


@dataclass(frozen=True, slots=True)
class VerifierCostRecord:
    """Artifact-backed verifier cost; intentionally incompatible with AR D2."""

    target_key: str
    provider_key: str
    mode: str
    request_count: int
    verifier_rows: int
    tree_shape: tuple[int, ...]
    context_bucket: int
    transaction_mode: str
    execution_profile: str
    predicted_microseconds: float
    source: str

    def __post_init__(self) -> None:
        for field in (
            "target_key", "provider_key", "transaction_mode", "execution_profile", "source"
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        mode = str(self.mode)
        if mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("verifier cost mode must be verify_chain or verify_tree")
        if "ar_d2" in self.source.lower():
            raise ValueError("verifier cost source must not use AR D2 evidence")
        if min(int(self.request_count), int(self.verifier_rows), int(self.context_bucket)) <= 0:
            raise ValueError("verifier request/row/context dimensions must be positive")
        if int(self.verifier_rows) < int(self.request_count):
            raise ValueError("verifier rows must be at least request count")
        if float(self.predicted_microseconds) <= 0.0:
            raise ValueError("predicted_microseconds must be positive")
        if any(parent < 0 for parent in self.tree_shape):
            raise ValueError("tree_shape entries must be non-negative")
        object.__setattr__(self, "mode", mode)

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "verifier_cost_v1",
            self.target_key,
            self.provider_key,
            self.mode,
            int(self.request_count),
            int(self.verifier_rows),
            self.tree_shape,
            int(self.context_bucket),
            self.transaction_mode,
            self.execution_profile,
        )


class VerifierCostMap:
    """Exact verifier-cost registry keyed independently from AR width maps."""

    def __init__(self, records: Sequence[VerifierCostRecord] = ()) -> None:
        self._records: dict[tuple[object, ...], VerifierCostRecord] = {}
        for record in records:
            self.register(record)

    def register(self, record: VerifierCostRecord, *, replace: bool = False) -> None:
        if not isinstance(record, VerifierCostRecord):
            raise TypeError("verifier cost map accepts VerifierCostRecord")
        if record.identity in self._records and not replace:
            raise ValueError("duplicate verifier cost identity")
        self._records[record.identity] = record

    def resolve(self, identity: Sequence[object]) -> VerifierCostRecord:
        key = tuple(identity)
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"no exact verifier cost record for {key!r}") from exc

    def snapshot(self) -> tuple[VerifierCostRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records, key=repr))


@dataclass(frozen=True, slots=True)
class SpeculativePackingBudget:
    max_draft_rows_per_round: int
    max_verify_rows_per_round: int
    max_speculative_cycles_per_round: int
    max_spec_transaction_bytes: int
    max_spec_work_items_per_round: int
    deadline_guard_seconds: float = 0.0

    def __post_init__(self) -> None:
        fields = (
            "max_draft_rows_per_round", "max_verify_rows_per_round",
            "max_speculative_cycles_per_round", "max_spec_transaction_bytes",
            "max_spec_work_items_per_round",
        )
        if any(int(getattr(self, field)) <= 0 for field in fields):
            raise ValueError("speculative packing budgets must be positive")
        if float(self.deadline_guard_seconds) < 0.0:
            raise ValueError("deadline_guard_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class SpeculativePackingRequest:
    request_id: int
    target_key: str
    provider_key: str
    policy_fingerprint: str
    mode: str
    context_bucket: int
    transaction_mode: str
    execution_profile: str
    candidate_rows: int
    transaction_bytes: int
    deadline_at: float | None = None

    def __post_init__(self) -> None:
        if int(self.request_id) < 0:
            raise ValueError("request_id must be non-negative")
        for field in (
            "target_key", "provider_key", "policy_fingerprint",
            "transaction_mode", "execution_profile",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")
        if min(int(self.context_bucket), int(self.candidate_rows), int(self.transaction_bytes)) <= 0:
            raise ValueError("context/candidate/transaction dimensions must be positive")

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.target_key,
            self.provider_key,
            self.policy_fingerprint,
            self.mode,
            int(self.context_bucket),
            self.transaction_mode,
            self.execution_profile,
            int(self.candidate_rows),
        )


@dataclass(frozen=True, slots=True)
class SpeculativePackedGroup:
    request_ids: tuple[int, ...]
    compatibility_keys: tuple[tuple[object, ...], ...]
    draft_rows: int
    verify_rows: int
    transaction_bytes: int
    mode: str


@dataclass(frozen=True, slots=True)
class SpeculativePackingPlan:
    groups: tuple[SpeculativePackedGroup, ...]
    deferred_request_ids: tuple[int, ...]
    ar_fallbacks: Mapping[int, str]
    ar_due_request_ids: tuple[int, ...]
    charged_draft_rows: int
    charged_verify_rows: int
    charged_transaction_bytes: int
    provisional_mutations: int = 0


def pack_speculative_requests(
    requests: Sequence[SpeculativePackingRequest],
    *,
    budget: SpeculativePackingBudget,
    cycles_already_served: Mapping[int, int] | None = None,
    now: float | None = None,
    ar_due_request_ids: Sequence[int] = (),
) -> SpeculativePackingPlan:
    """Pack one fair speculative pass without provisional mutation.

    Requests are sorted by cycles already consumed and stable input order. Only
    identical compatibility keys share a physical work item. Any deadline or
    transaction-budget fallback is decided before a group is published.
    """

    served = {} if cycles_already_served is None else {
        int(key): int(value) for key, value in cycles_already_served.items()
    }
    indexed = list(enumerate(requests))
    if len({request.request_id for _index, request in indexed}) != len(indexed):
        raise ValueError("speculative packing request ids must be unique")
    indexed.sort(key=lambda item: (served.get(item[1].request_id, 0), item[0]))
    current_time = None if now is None else float(now)
    fallbacks: dict[int, str] = {}
    eligible: list[SpeculativePackingRequest] = []
    for _index, request in indexed:
        if request.transaction_bytes > budget.max_spec_transaction_bytes:
            fallbacks[request.request_id] = "spec_transaction_budget"
            continue
        if (
            current_time is not None
            and request.deadline_at is not None
            and float(request.deadline_at) - current_time <= budget.deadline_guard_seconds
        ):
            fallbacks[request.request_id] = "deadline_guard"
            continue
        eligible.append(request)

    groups: list[SpeculativePackedGroup] = []
    deferred: list[int] = []
    draft_used = verify_used = txn_used = cycles_used = 0
    consumed: set[int] = set()
    for request in eligible:
        if request.request_id in consumed:
            continue
        start = eligible.index(request)
        compatible: list[SpeculativePackingRequest] = []
        for peer in eligible[start:]:
            if peer.request_id in consumed:
                continue
            if peer.compatibility_key != request.compatibility_key:
                break
            compatible.append(peer)
        selected: list[SpeculativePackingRequest] = []
        for peer in compatible:
            rows = int(peer.candidate_rows)
            if cycles_used + 1 > budget.max_speculative_cycles_per_round:
                break
            if len(groups) + 1 > budget.max_spec_work_items_per_round:
                break
            if draft_used + rows > budget.max_draft_rows_per_round:
                break
            if verify_used + rows > budget.max_verify_rows_per_round:
                break
            if txn_used + peer.transaction_bytes > budget.max_spec_transaction_bytes:
                break
            selected.append(peer)
            consumed.add(peer.request_id)
            draft_used += rows
            verify_used += rows
            txn_used += peer.transaction_bytes
            cycles_used += 1
        if selected:
            groups.append(
                SpeculativePackedGroup(
                    request_ids=tuple(peer.request_id for peer in selected),
                    compatibility_keys=tuple(peer.compatibility_key for peer in selected),
                    draft_rows=sum(peer.candidate_rows for peer in selected),
                    verify_rows=sum(peer.candidate_rows for peer in selected),
                    transaction_bytes=sum(peer.transaction_bytes for peer in selected),
                    mode=request.mode,
                )
            )
    deferred.extend(
        request.request_id for request in eligible if request.request_id not in consumed
    )
    return SpeculativePackingPlan(
        groups=tuple(groups),
        deferred_request_ids=tuple(deferred),
        ar_fallbacks=fallbacks,
        ar_due_request_ids=tuple(int(request_id) for request_id in ar_due_request_ids),
        charged_draft_rows=draft_used,
        charged_verify_rows=verify_used,
        charged_transaction_bytes=txn_used,
        provisional_mutations=0,
    )
