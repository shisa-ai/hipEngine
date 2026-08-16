"""Dense BF16 and artifact-qualified no-mirror INT8 global KV backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from types import SimpleNamespace
from collections.abc import Callable
from typing import Any, Mapping, Sequence

from hipengine.core import DType, Device, Tensor
from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVBackendSpec,
    KVBatchView,
    KVLease,
    KVPoolPlan,
    KVPoolSpec,
    KVStorageView,
    ResourceChange,
    ResourceClaim,
    ResourceClaimSet,
    ResourceDelta,
)
from hipengine.kvcache.backend_prefix import (
    BackendRadixCache,
    KVSnapshotHandle,
    PrefixCompatibilityKey,
)
from hipengine.kvcache.global_pool import GlobalKVPoolSet, GlobalPageLease
from hipengine.kvcache.ledger import FitAwareAdmissionController, ResourceLedger
from hipengine.kvcache.spans import KVLiveSpans, KVScaleMetadata

_CPU = Device("cpu", 0)


@dataclass(frozen=True, slots=True)
class DenseKVArtifactQualification:
    artifact_fingerprint: str
    kl_divergence: float
    top1_agreement: float
    no_bf16_mirror: bool
    evidence_source: str

    def __post_init__(self) -> None:
        artifact = str(self.artifact_fingerprint)
        source = str(self.evidence_source)
        if not artifact or artifact != artifact.strip():
            raise ValueError("qualified KV artifact fingerprint must be non-empty")
        if not source or source != source.strip():
            raise ValueError("qualified KV artifact evidence_source must be non-empty")
        if float(self.kl_divergence) < 0 or float(self.kl_divergence) > 0.05:
            raise ValueError("qualified KV artifact requires KL divergence <= 0.05")
        if not 0.9 <= float(self.top1_agreement) <= 1.0:
            raise ValueError("qualified KV artifact requires top-1 agreement >= 0.90")
        if not self.no_bf16_mirror:
            raise ValueError("qualified INT8 KV artifact must prove no BF16 mirror")
        object.__setattr__(self, "artifact_fingerprint", artifact)
        object.__setattr__(self, "kl_divergence", float(self.kl_divergence))
        object.__setattr__(self, "top1_agreement", float(self.top1_agreement))
        object.__setattr__(self, "evidence_source", source)


@dataclass(frozen=True, slots=True)
class DenseKVOperation:
    operation_id: str
    lease: KVLease
    append_pages: int = 0


class DenseKVCacheBackend:
    """One lifecycle implementation shared by dense BF16 and INT8 codecs.

    The codec determines only the declared plane set, storage dtype, and kernel
    bundle. Allocation, credits, admission, reclaim, COW, and in-flight safety
    all remain common.
    """

    def __init__(
        self,
        *,
        codec: str,
        page_capacity: int,
        block_size: int,
        artifact_fingerprint: str,
        physical_widths: tuple[int, ...] = (1, 2, 4, 8),
        generation: int = 1,
        plane_page_pointers: dict[str, tuple[int, ...]] | None = None,
        pointer_table_pointers: dict[str, int] | None = None,
        int8_qualification: DenseKVArtifactQualification | None = None,
    ) -> None:
        if codec not in {"bf16", "int8_per_token_head"}:
            raise ValueError("dense codec must be bf16 or int8_per_token_head")
        capacity = int(page_capacity)
        block = int(block_size)
        if capacity <= 0 or block <= 0:
            raise ValueError("page_capacity and block_size must be positive")
        artifact = str(artifact_fingerprint)
        if not artifact or artifact != artifact.strip():
            raise ValueError("dense backend requires an artifact fingerprint")
        if codec == "int8_per_token_head":
            if int8_qualification is None:
                raise ValueError("INT8 KV backend requires artifact qualification")
            if int8_qualification.artifact_fingerprint != artifact:
                raise ValueError("INT8 qualification fingerprint does not match artifact")
        elif int8_qualification is not None:
            raise ValueError("BF16 KV backend does not accept INT8 artifact qualification")
        self.int8_qualification = int8_qualification
        self.codec = codec
        self.page_capacity = capacity
        self.block_size = block
        self.generation = int(generation)
        if self.generation <= 0:
            raise ValueError("generation must be positive")
        self.spec = KVBackendSpec(
            topology_key="paged_dense_global",
            hot_codec_key=codec,
            tier_key="device_only",
            layout_fingerprint=f"arbitrary-page-table:{codec}:v1",
            artifact_fingerprint=artifact,
            prefix_mode="immutable_pages",
            transaction_mode="journal",
            kernel_bundle_key=(
                "dense_bf16_global_pages"
                if codec == "bf16"
                else "dense_int8_no_mirror_global_pages"
            ),
            physical_widths=physical_widths,
            max_context_tokens=capacity * block,
        )
        self._plane_roles = (
            ("k_payload", "v_payload")
            if codec == "bf16"
            else ("k_payload", "v_payload", "k_scale", "v_scale")
        )
        self._plan = self._build_plan()
        if plane_page_pointers is None:
            plane_page_pointers = _synthetic_page_pointers(
                self._plane_roles,
                capacity,
                codec=codec,
            )
        if set(plane_page_pointers) != set(self._plane_roles):
            raise ValueError("dense storage planes do not match the resolved codec")
        if pointer_table_pointers is None:
            pointer_table_pointers = _synthetic_pointer_tables(self._plane_roles, codec=codec)
        if set(pointer_table_pointers) != set(self._plane_roles):
            raise ValueError("dense pointer tables do not match the resolved codec")
        self.pool = GlobalKVPoolSet(
            backend_fingerprint=self.spec.artifact_fingerprint,
            generation=self.generation,
            plane_page_pointers=plane_page_pointers,
            pointer_table_pointers=pointer_table_pointers,
        )
        self.ledger = ResourceLedger(self._plan)
        self._leases: dict[str, KVLease] = {}

    @property
    def storage_dtype(self) -> DType:
        return DType.BF16 if self.codec == "bf16" else DType.INT8_PER_TOKEN_HEAD

    @property
    def page_pool_ids(self) -> tuple[str, ...]:
        return tuple(f"kv.{role}.pages" for role in self._plane_roles)

    def plan_pools(self, load_plan: Any) -> KVPoolPlan:
        del load_plan
        return self._plan

    def estimate(self, request: Any, prefix: Any, stage: Any) -> ResourceClaimSet:
        request_id = int(getattr(request, "request_id"))
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        stage_map: Mapping[str, Any] = stage if isinstance(stage, Mapping) else {}
        kind = str(stage_map.get("kind", "admission"))
        if kind == "work_item":
            return ResourceClaimSet(
                claim_id=f"dense-work:{request_id}",
                request_id=request_id,
            )
        request_tokens = tuple(int(token) for token in getattr(request, "prompt_tokens", ()))
        prompt_tokens = int(stage_map.get("tokens", len(request_tokens)))
        if prompt_tokens < 0:
            raise ValueError("prompt token count must be non-negative")
        total_prompt_pages = ceil(prompt_tokens / self.block_size) if prompt_tokens else 0
        shared_page_ids: tuple[int, ...] = ()
        if prefix is not None:
            try:
                prefix_backend = str(prefix.backend_fingerprint)
                prefix_artifact = str(prefix.artifact_fingerprint)
                prefix_generation = int(prefix.generation)
                prefix_tokens = tuple(int(token) for token in prefix.matched_tokens)
                shared_page_ids = tuple(int(page_id) for page_id in prefix.page_ids)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError("dense prefix must be a KV snapshot handle") from exc
            if (
                prefix_backend != self.spec.fingerprint
                or prefix_artifact != self.spec.artifact_fingerprint
                or prefix_generation != self.generation
            ):
                raise ValueError("dense prefix snapshot is backend-incompatible or stale")
            if request_tokens[: len(prefix_tokens)] != prefix_tokens:
                raise ValueError("dense prefix snapshot tokens do not match request")
            if len(prefix_tokens) % self.block_size != 0:
                raise ValueError("dense prefix snapshot must end at a complete page")
            if len(shared_page_ids) != len(prefix_tokens) // self.block_size:
                raise ValueError("dense prefix snapshot pages do not match token boundary")
            if len(shared_page_ids) > total_prompt_pages:
                raise ValueError("dense prefix snapshot exceeds prompt page count")
        private_pages = total_prompt_pages - len(shared_page_ids)
        max_new_tokens = int(stage_map.get("max_new_tokens", getattr(request, "max_new_tokens", 1)))
        growth_credit_pages = int(
            stage_map.get("growth_credit_pages", 1 if max_new_tokens > 0 else 0)
        )
        if growth_credit_pages < 0:
            raise ValueError("growth_credit_pages must be non-negative")
        total_pages = private_pages + growth_credit_pages
        claims = [
            ResourceClaim(pool_id, total_pages, ClaimLifetime.LEASE)
            for pool_id in self.page_pool_ids
            if total_pages
        ]
        claims.append(ResourceClaim("kv.request_rows", 1, ClaimLifetime.LEASE))
        return ResourceClaimSet(
            claim_id=f"dense-admission:{request_id}:{private_pages}:{growth_credit_pages}",
            request_id=request_id,
            claims=tuple(claims),
            metadata=(
                ("growth_credit_pages", growth_credit_pages),
                ("private_pages", private_pages),
                ("shared_page_ids", ",".join(str(page_id) for page_id in shared_page_ids)),
            ),
        )

    def reserve(self, claims: ResourceClaimSet) -> KVLease:
        (
            _request_id,
            lease_id,
            private_pages,
            growth_credit_pages,
            shared_page_ids,
        ) = self._claim_pages(claims)
        reservation = self.ledger.reserve_provisional(claims)
        try:
            lease = self._materialize_lease(
                claims,
                private_pages=private_pages,
                growth_credit_pages=growth_credit_pages,
                shared_page_ids=shared_page_ids,
            )
            self.ledger.commit(reservation, owner_id=lease_id)
            return lease
        except Exception:
            if lease_id in self._leases:
                self.pool.release(lease_id)
                self._leases.pop(lease_id, None)
            self.ledger.rollback(reservation)
            raise

    def materialize_committed(self, claims: ResourceClaimSet) -> KVLease:
        """Bind physical pages after fit-aware admission committed the ledger."""

        (
            _request_id,
            lease_id,
            private_pages,
            growth_credit_pages,
            shared_page_ids,
        ) = self._claim_pages(claims)
        if not self.ledger.has_owner(lease_id):
            raise RuntimeError(f"dense resource owner {lease_id!r} is not committed")
        return self._materialize_lease(
            claims,
            private_pages=private_pages,
            growth_credit_pages=growth_credit_pages,
            shared_page_ids=shared_page_ids,
        )

    def page_lease(self, lease: KVLease | str) -> GlobalPageLease:
        return self.pool.lease(lease if isinstance(lease, str) else lease.lease_id)

    def storage_view(self, lease: KVLease | None = None) -> KVStorageView:
        if lease is not None:
            self._validate_lease(lease)
        return self.pool.storage_view()

    def append_page(self, lease: KVLease) -> int:
        self._validate_lease(lease)
        return self.pool.consume_growth_credit(lease.lease_id)

    def renew_growth_credit(self, lease: KVLease, *, pages: int) -> tuple[int, ...]:
        self._validate_lease(lease)
        count = int(pages)
        if count <= 0:
            raise ValueError("pages must be positive")
        delta = ResourceDelta(
            operation_id=f"growth-credit-renew:{lease.request_id}",
            lease_id=lease.lease_id,
            request_id=lease.request_id,
            changes=tuple(
                ResourceChange(pool_id, count, ClaimLifetime.LEASE)
                for pool_id in self.page_pool_ids
            ),
        )
        self.ledger.apply_delta(lease.lease_id, delta)
        try:
            return self.pool.add_growth_credit(lease.lease_id, count)
        except Exception:
            self.ledger.apply_delta(
                lease.lease_id,
                ResourceDelta(
                    operation_id=f"growth-credit-renew-rollback:{lease.request_id}",
                    lease_id=lease.lease_id,
                    request_id=lease.request_id,
                    changes=tuple(
                        ResourceChange(pool_id, -count, ClaimLifetime.LEASE)
                        for pool_id in self.page_pool_ids
                    ),
                ),
            )
            raise

    def prepare(self, work_item: Any) -> KVBatchView:
        request_ids = tuple(int(request_id) for request_id in getattr(work_item, "request_ids"))
        if not request_ids:
            raise ValueError("dense backend prepare requires request ids")
        context_lengths = tuple(
            int(length)
            for length in getattr(work_item, "context_lengths", (0,) * len(request_ids))
        )
        if len(context_lengths) != len(request_ids):
            raise ValueError("context lengths must align with request ids")
        for request_id in request_ids:
            if f"lease:{request_id}" not in self._leases:
                raise KeyError(f"request_id {request_id} has no dense KV lease")
        rows = len(request_ids)
        base = 0x7D000000 + self.generation * 0x100000
        request_tensor = Tensor.from_handle(base + 0x1000, (rows,), DType.INT64, _CPU)
        scale_metadata = None
        if self.codec == "int8_per_token_head":
            scale_metadata = KVScaleMetadata(
                k_scale=Tensor.from_handle(base + 0x5000, (rows, 1), DType.FP16, _CPU),
                v_scale=Tensor.from_handle(base + 0x6000, (rows, 1), DType.FP16, _CPU),
            )
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(base + 0x2000, (rows, 1), DType.INT32, _CPU),
            live_counts=Tensor.from_handle(base + 0x3000, (rows,), DType.INT32, _CPU),
            max_live_count=max(context_lengths, default=0),
            storage_dtype=self.storage_dtype,
            request_ids=request_tensor,
            row_positions=Tensor.from_handle(base + 0x4000, (rows,), DType.INT32, _CPU),
            span_role="decode",
            scale_metadata=scale_metadata,
        )
        return KVBatchView(
            live_spans=spans,
            storage_view=self.pool.storage_view(),
            kernel_bundle_key=self.spec.kernel_bundle_key,
            execution_compatibility_key=(*self.spec.compatibility_key, "decode"),
        )

    def begin_transaction(self, rows: Sequence[Any], draft: Any) -> DenseKVOperation:
        del draft
        if len(rows) != 1:
            raise ValueError("dense transaction helper expects one lease row")
        lease = getattr(rows[0], "lease", rows[0])
        if not isinstance(lease, KVLease):
            raise TypeError("dense transaction row must provide a KVLease")
        self._validate_lease(lease)
        return DenseKVOperation(f"transaction:{lease.request_id}", lease)

    def commit(self, operation: Any, result: Any) -> ResourceDelta:
        del result
        if not isinstance(operation, DenseKVOperation):
            raise TypeError("dense commit requires DenseKVOperation")
        for _ in range(operation.append_pages):
            self.append_page(operation.lease)
        return ResourceDelta(
            operation_id=f"commit:{operation.operation_id}",
            lease_id=operation.lease.lease_id,
            request_id=operation.lease.request_id,
        )

    def rollback(self, operation: Any) -> ResourceDelta:
        if not isinstance(operation, DenseKVOperation):
            raise TypeError("dense rollback requires DenseKVOperation")
        return ResourceDelta(
            operation_id=f"rollback:{operation.operation_id}",
            lease_id=operation.lease.lease_id,
            request_id=operation.lease.request_id,
        )

    def reclaim(self, lease: KVLease) -> ResourceDelta:
        self._validate_lease(lease)
        if not self.ledger.has_owner(lease.lease_id):
            raise RuntimeError(f"dense KV lease {lease.lease_id!r} lacks ledger ownership")
        self.pool.release(lease.lease_id)
        delta = self.ledger.release(
            lease.lease_id,
            operation_id=f"reclaim:{lease.request_id}",
        )
        self._leases.pop(lease.lease_id)
        return delta

    def prefix_lookup(self, tokens: Sequence[int]) -> Any:
        return SimpleNamespace(
            hit=False,
            matched_tokens=(),
            remaining_tokens=tuple(int(token) for token in tokens),
        )

    def maintenance(self, budget: Any) -> list[Any]:
        del budget
        return []

    def observability_snapshot(self) -> dict[str, Any]:
        return {
            "backend": {
                "topology": self.spec.topology_key,
                "codec": self.spec.hot_codec_key,
                "artifact_fingerprint": self.spec.artifact_fingerprint,
                "physical_widths": list(self.spec.physical_widths),
                "block_size": self.block_size,
                "qualification": (
                    None
                    if self.int8_qualification is None
                    else {
                        "kl_divergence": self.int8_qualification.kl_divergence,
                        "top1_agreement": self.int8_qualification.top1_agreement,
                        "no_bf16_mirror": self.int8_qualification.no_bf16_mirror,
                        "evidence_source": self.int8_qualification.evidence_source,
                    }
                ),
            },
            "ledger": self.ledger.snapshot(),
            "pool": self.pool.snapshot(),
        }

    def has_request(self, request_id: int) -> bool:
        return f"lease:{int(request_id)}" in self._leases

    def lease_for_request(self, request_id: int) -> KVLease:
        try:
            return self._leases[f"lease:{int(request_id)}"]
        except KeyError as exc:
            raise KeyError(f"request_id {request_id} has no dense KV lease") from exc

    def _build_plan(self) -> KVPoolPlan:
        page_lifetimes = (ClaimLifetime.LEASE, ClaimLifetime.CACHE)
        pools = tuple(
            KVPoolSpec(
                f"kv.{role}.pages",
                self.page_capacity,
                unit="pages",
                plane_role=role,
                lifetimes=page_lifetimes,
            )
            for role in self._plane_roles
        )
        pools += (
            KVPoolSpec(
                "kv.request_rows",
                self.page_capacity,
                unit="rows",
                plane_role="row_metadata",
                lifetimes=(ClaimLifetime.LEASE,),
            ),
        )
        return KVPoolPlan(
            backend_fingerprint=self.spec.fingerprint,
            generation=self.generation,
            pools=pools,
        )

    def _claim_pages(
        self,
        claims: ResourceClaimSet,
    ) -> tuple[int, str, int, int, tuple[int, ...]]:
        if claims.request_id is None:
            raise ValueError("dense backend reservations require request_id")
        request_id = int(claims.request_id)
        lease_id = f"lease:{request_id}"
        if lease_id in self._leases:
            raise ValueError(f"request_id {request_id} already owns a dense KV lease")
        metadata = claims.metadata_dict()
        try:
            private_pages = int(metadata["private_pages"])
            growth_credit_pages = int(metadata["growth_credit_pages"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dense reservation claims lack typed page metadata") from exc
        if private_pages < 0 or growth_credit_pages < 0:
            raise ValueError("dense reservation page counts must be non-negative")
        shared_text = str(metadata.get("shared_page_ids", ""))
        try:
            shared_page_ids = (
                ()
                if not shared_text
                else tuple(int(page_id) for page_id in shared_text.split(","))
            )
        except ValueError as exc:
            raise ValueError("dense shared prefix page IDs must be integers") from exc
        if len(shared_page_ids) != len(set(shared_page_ids)):
            raise ValueError("dense shared prefix page IDs must be unique")
        return (
            request_id,
            lease_id,
            private_pages,
            growth_credit_pages,
            shared_page_ids,
        )

    def _materialize_lease(
        self,
        claims: ResourceClaimSet,
        *,
        private_pages: int,
        growth_credit_pages: int,
        shared_page_ids: tuple[int, ...],
    ) -> KVLease:
        if claims.request_id is None:
            raise ValueError("dense backend reservations require request_id")
        request_id = int(claims.request_id)
        lease_id = f"lease:{request_id}"
        if lease_id in self._leases:
            raise ValueError(f"request_id {request_id} already owns a dense KV lease")
        page_lease = self.pool.allocate(
            lease_id,
            private_pages=private_pages,
            growth_credit_pages=growth_credit_pages,
            shared_page_ids=shared_page_ids,
        )
        try:
            growth_claims = ResourceClaimSet(
                claim_id=f"growth-credit:{request_id}",
                request_id=request_id,
                claims=tuple(
                    ResourceClaim(pool_id, growth_credit_pages, ClaimLifetime.LEASE)
                    for pool_id in self.page_pool_ids
                    if growth_credit_pages
                ),
                metadata=(("growth_credit_pages", growth_credit_pages),),
            )
            lease = KVLease(
                lease_id=lease_id,
                request_id=request_id,
                backend_fingerprint=self.spec.fingerprint,
                generation=self.generation,
                claims=claims.with_claim_id(f"dense-ownership:{request_id}"),
                shared_handles=tuple(
                    f"page:{page_id}" for page_id in page_lease.shared_page_ids
                ),
                private_handles=tuple(
                    f"page:{page_id}" for page_id in page_lease.private_page_ids
                ),
                writable_tail_handle=(
                    None
                    if not page_lease.private_page_ids
                    else f"tail-page:{page_lease.private_page_ids[-1]}"
                ),
                metadata_handles=(f"request-row:{request_id}",),
                growth_credits=growth_claims,
            )
        except Exception:
            self.pool.release(lease_id)
            raise
        self._leases[lease_id] = lease
        return lease

    def _validate_lease(self, lease: KVLease) -> None:
        current = self._leases.get(lease.lease_id)
        if current is None:
            raise KeyError(f"unknown dense KV lease {lease.lease_id!r}")
        if current.request_id != lease.request_id:
            raise ValueError("dense KV lease request identity mismatch")
        if lease.backend_fingerprint != self.spec.fingerprint:
            raise ValueError("dense KV lease backend fingerprint mismatch")
        if lease.generation != self.generation:
            raise ValueError("dense KV lease generation mismatch")


class DenseKVAdmissionManager:
    """Resident-scheduler adapter for one dense global-pool backend."""

    def __init__(
        self,
        backend: DenseKVCacheBackend,
        *,
        lookahead: int = 32,
        max_bypasses: int = 8,
        prefix_cache: BackendRadixCache | None = None,
        prefix_scope: PrefixCompatibilityKey | None = None,
        tenant_resolver: Callable[[Any], str] | None = None,
        reuse_eligibility: Callable[[Any], bool] | None = None,
    ) -> None:
        if (prefix_cache is None) != (prefix_scope is None):
            raise ValueError("prefix_cache and prefix_scope must be configured together")
        if prefix_cache is not None and prefix_cache.spec != backend.spec:
            raise ValueError("prefix cache backend does not match dense backend")
        if prefix_cache is not None and reuse_eligibility is None:
            raise ValueError(
                "prefix cache requires an explicit deterministic reuse eligibility policy"
            )
        self.backend = backend
        self.prefix_cache = prefix_cache
        self.prefix_scope = prefix_scope
        self.tenant_resolver = tenant_resolver or (
            lambda request: str(getattr(request, "tenant_id", "default"))
        )
        self.reuse_eligibility = reuse_eligibility or (lambda request: False)
        self.controller = FitAwareAdmissionController(
            backend.ledger,
            lookahead=lookahead,
            max_bypasses=max_bypasses,
        )
        self._pending_snapshots: dict[int, KVSnapshotHandle] = {}
        self._tenant_by_request: dict[int, str] = {}
        self._cacheable_by_request: dict[int, bool] = {}

    def plan_admission(
        self,
        pending_requests: Sequence[Any],
        *,
        max_items: int,
    ) -> tuple[int, ...]:
        pending_by_id = {
            int(request.request_id): request for request in pending_requests
        }
        for request_id in self.controller.pending_request_ids:
            if request_id not in pending_by_id:
                self.controller.cancel(request_id)
                self._pending_snapshots.pop(request_id, None)
                self._tenant_by_request.pop(request_id, None)
                self._cacheable_by_request.pop(request_id, None)
        known = set(self.controller.pending_request_ids)
        known.update(
            request_id
            for request_id in pending_by_id
            if self.backend.has_request(request_id)
        )
        for request_id, request in pending_by_id.items():
            if request_id in known:
                continue
            snapshot = None
            cacheable = self.reuse_eligibility(request)
            self._cacheable_by_request[request_id] = bool(cacheable)
            if (
                cacheable
                and self.prefix_cache is not None
                and self.prefix_scope is not None
            ):
                match = self.prefix_cache.lookup(
                    self.prefix_scope,
                    tuple(int(token) for token in request.prompt_tokens),
                )
                snapshot = match.snapshot
                if snapshot is not None:
                    self._pending_snapshots[request_id] = snapshot
                self._tenant_by_request[request_id] = self.tenant_resolver(request)
            claims = self.backend.estimate(
                request,
                snapshot,
                {"kind": "admission"},
            )
            self.controller.enqueue(
                request_id,
                claims,
                owner_id=f"lease:{request_id}",
            )
        grants = self.controller.admit(max_items=max_items)
        if not grants and self.prefix_cache is not None and self.controller.pending_count:
            required_pages = 0
            for request_id in self.controller.pending_request_ids:
                claims = self.controller.pending_state(request_id).claims
                required_pages = max(
                    required_pages,
                    max(
                        (claims.units_by_pool().get(pool_id, 0) for pool_id in self.backend.page_pool_ids),
                        default=0,
                    ),
                )
            if required_pages:
                self.prefix_cache.evict_for_pressure(required_pages)
                grants = self.controller.admit(max_items=max_items)
        materialized: list[int] = []
        for grant in grants:
            try:
                self.backend.materialize_committed(grant.reservation.claims)
            except Exception:
                if self.backend.ledger.has_owner(grant.owner_id):
                    self.backend.ledger.release(
                        grant.owner_id,
                        operation_id=f"materialize-rollback:{grant.request_id}",
                    )
                raise
            materialized.append(grant.request_id)
        return tuple(materialized)

    def reserve_admission(self, request: Any) -> Any:
        request_id = int(request.request_id)
        if not self.backend.has_request(request_id):
            raise RuntimeError(
                f"request_id {request_id} has no materialized dense KV lease"
            )
        snapshot = self._pending_snapshots.pop(request_id, None)
        if snapshot is None:
            return None
        return replace(request, next_prompt_index=snapshot.matched_token_count)

    def rollback_admission(self, request: Any) -> None:
        request_id = int(request.request_id)
        self._pending_snapshots.pop(request_id, None)
        self._tenant_by_request.pop(request_id, None)
        self._cacheable_by_request.pop(request_id, None)
        if self.backend.has_request(request_id):
            self.backend.reclaim(self.backend.lease_for_request(request_id))
        else:
            self.controller.cancel(request_id)

    def reclaim_request(self, request: Any) -> ResourceDelta | None:
        request_id = int(request.request_id)
        self._pending_snapshots.pop(request_id, None)
        if not self.backend.has_request(request_id):
            self.controller.cancel(request_id)
            self._tenant_by_request.pop(request_id, None)
            self._cacheable_by_request.pop(request_id, None)
            return None
        lease = self.backend.lease_for_request(request_id)
        finish_reason = str(getattr(request, "finish_reason", ""))
        cacheable = self._cacheable_by_request.get(request_id, False) and finish_reason not in {
            "cancel",
            "disconnect",
            "timeout",
            "error",
            "shutdown",
        }
        if (
            cacheable
            and self.prefix_cache is not None
            and self.prefix_scope is not None
        ):
            self.prefix_cache.publish(
                self.prefix_scope,
                lease,
                tuple(int(token) for token in getattr(request, "prompt_tokens", ())),
                tenant_id=self._tenant_by_request.get(request_id, "default"),
            )
        self._tenant_by_request.pop(request_id, None)
        self._cacheable_by_request.pop(request_id, None)
        return self.backend.reclaim(lease)

    def resource_observability_snapshot(self) -> dict[str, Any]:
        snapshot = self.backend.observability_snapshot()
        snapshot["admission"] = self.controller.snapshot()
        snapshot["prefix_cache"] = (
            None if self.prefix_cache is None else self.prefix_cache.snapshot()
        )
        return snapshot


class DenseKVResidentRunnerAdapter:
    """Require resident runner kernels to consume backend `KVBatchView` data."""

    def __init__(
        self,
        runner: Any,
        admission: DenseKVAdmissionManager,
    ) -> None:
        if not callable(getattr(runner, "prefill_batch_with_kv", None)):
            raise TypeError("dense KV runner requires prefill_batch_with_kv")
        if not callable(getattr(runner, "decode_batch_with_kv", None)):
            raise TypeError("dense KV runner requires decode_batch_with_kv")
        self.runner = runner
        self.admission = admission
        self.backend = admission.backend
        bundle_key = str(getattr(runner, "kv_kernel_bundle_key", ""))
        if bundle_key != self.backend.spec.kernel_bundle_key:
            raise ValueError(
                "dense KV runner kernel bundle does not match the resolved backend"
            )
        layout_keys = tuple(
            str(layout_key)
            for layout_key in getattr(runner, "kv_storage_layout_keys", ())
        )
        if self.backend.storage_view().layout_key not in layout_keys:
            raise ValueError(
                "dense KV runner does not register the backend storage layout"
            )
        self.capacity = int(getattr(runner, "capacity"))
        if self.capacity <= 0:
            raise ValueError("dense KV runner capacity must be positive")

    def plan_admission(
        self,
        pending_requests: Sequence[Any],
        *,
        max_items: int,
    ) -> tuple[int, ...]:
        return self.admission.plan_admission(
            pending_requests,
            max_items=max_items,
        )

    def reserve_admission(self, request: Any) -> Any:
        return self.admission.reserve_admission(request)

    def rollback_admission(self, request: Any) -> None:
        self.admission.rollback_admission(request)

    def prefill_batch(self, work: Any, *, commit: bool) -> Any:
        view = self.backend.prepare(work)
        return self.runner.prefill_batch_with_kv(
            work,
            kv_batch_view=view,
            commit=commit,
        )

    def decode_batch(self, work: Any, *, commit: bool) -> Any:
        view = self.backend.prepare(work)
        return self.runner.decode_batch_with_kv(
            work,
            kv_batch_view=view,
            commit=commit,
        )

    def compact_batch(self, moves: Any) -> Any:
        compact = getattr(self.runner, "compact_batch", None)
        return None if not callable(compact) else compact(moves)

    def reclaim(self, completed: Any) -> None:
        reclaim = getattr(self.runner, "reclaim", None)
        if callable(reclaim):
            reclaim(completed)
        self.admission.reclaim_request(completed)

    def resource_observability_snapshot(self) -> dict[str, Any]:
        return self.admission.resource_observability_snapshot()

    def close(self) -> None:
        close = getattr(self.runner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runner, name)


def create_dense_bf16_backend(
    *,
    page_capacity: int,
    block_size: int,
    backend_fingerprint: str,
    physical_widths: tuple[int, ...] = (1, 2, 4, 8),
    generation: int = 1,
    plane_page_pointers: dict[str, tuple[int, ...]] | None = None,
    pointer_table_pointers: dict[str, int] | None = None,
) -> DenseKVCacheBackend:
    return DenseKVCacheBackend(
        codec="bf16",
        page_capacity=page_capacity,
        block_size=block_size,
        artifact_fingerprint=backend_fingerprint,
        physical_widths=physical_widths,
        generation=generation,
        plane_page_pointers=plane_page_pointers,
        pointer_table_pointers=pointer_table_pointers,
    )


def create_dense_int8_backend(
    *,
    page_capacity: int,
    block_size: int,
    qualification: DenseKVArtifactQualification,
    physical_widths: tuple[int, ...] = (1, 2, 4, 8),
    generation: int = 1,
    plane_page_pointers: dict[str, tuple[int, ...]] | None = None,
    pointer_table_pointers: dict[str, int] | None = None,
) -> DenseKVCacheBackend:
    if not isinstance(qualification, DenseKVArtifactQualification):
        raise ValueError("INT8 KV backend requires artifact qualification")
    return DenseKVCacheBackend(
        codec="int8_per_token_head",
        page_capacity=page_capacity,
        block_size=block_size,
        artifact_fingerprint=qualification.artifact_fingerprint,
        physical_widths=physical_widths,
        generation=generation,
        plane_page_pointers=plane_page_pointers,
        pointer_table_pointers=pointer_table_pointers,
        int8_qualification=qualification,
    )


def _synthetic_page_pointers(
    roles: tuple[str, ...],
    capacity: int,
    *,
    codec: str,
) -> dict[str, tuple[int, ...]]:
    codec_offset = 0 if codec == "bf16" else 0x10000000
    result: dict[str, tuple[int, ...]] = {}
    for role_index, role in enumerate(roles):
        base = 0x81000000 + codec_offset + role_index * 0x01000000
        # The discontinuity every eight pages deliberately models multiple
        # allocations while keeping one stable logical page-ID table.
        result[role] = tuple(
            base + page_id * 0x1000 + (page_id // 8) * 0x100000
            for page_id in range(capacity)
        )
    return result


def _synthetic_pointer_tables(
    roles: tuple[str, ...],
    *,
    codec: str,
) -> dict[str, int]:
    codec_offset = 0 if codec == "bf16" else 0x10000000
    return {
        role: 0x71000000 + codec_offset + index * 0x10000
        for index, role in enumerate(roles)
    }


__all__ = [
    "DenseKVAdmissionManager",
    "DenseKVArtifactQualification",
    "DenseKVCacheBackend",
    "DenseKVOperation",
    "DenseKVResidentRunnerAdapter",
    "create_dense_bf16_backend",
    "create_dense_int8_backend",
]
