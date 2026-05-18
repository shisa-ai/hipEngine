"""DFlash verify graph bucket metadata and fallback policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from hipengine.speculative.interfaces import TargetVerifyBatch

SUPPORTED_DFLASH_CHAIN_DEPTHS = (2, 4, 8)
SUPPORTED_DFLASH_PAGE_BUCKETS = (128,)


@dataclass(frozen=True, slots=True)
class DFlashVerifyGraphBucketKey:
    backend: str
    active_c: int
    context_bucket: int
    page_bucket: int
    mode: str
    draft_depth: int
    tree_shape: tuple[int, ...]
    top_k: int
    experts_per_token: int
    replay_steps: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        for name, value in (
            ("active_c", self.active_c),
            ("context_bucket", self.context_bucket),
            ("page_bucket", self.page_bucket),
            ("draft_depth", self.draft_depth),
            ("top_k", self.top_k),
            ("experts_per_token", self.experts_per_token),
            ("replay_steps", self.replay_steps),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.replay_steps <= 0:
            raise ValueError("replay_steps must be positive")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")
        object.__setattr__(self, "tree_shape", tuple(int(item) for item in self.tree_shape))

    @classmethod
    def from_batch(
        cls,
        batch: TargetVerifyBatch,
        *,
        backend: str,
        context_bucket: int,
        page_bucket: int,
        top_k: int = 1,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> "DFlashVerifyGraphBucketKey":
        return cls(
            backend=backend,
            active_c=len(batch.request_ids),
            context_bucket=int(context_bucket),
            page_bucket=int(page_bucket),
            mode=batch.mode,
            draft_depth=batch.draft_depth,
            tree_shape=batch.tree_shape,
            top_k=int(top_k),
            experts_per_token=int(experts_per_token),
            replay_steps=int(replay_steps),
        )

    @property
    def supported(self) -> bool:
        return (
            self.mode == "verify_chain"
            and self.active_c > 0
            and self.draft_depth in SUPPORTED_DFLASH_CHAIN_DEPTHS
            and self.page_bucket in SUPPORTED_DFLASH_PAGE_BUCKETS
        )

    @property
    def fallback_reason(self) -> str | None:
        if self.mode != "verify_chain":
            return f"unsupported mode {self.mode}"
        if self.active_c <= 0:
            return "active_c must be positive"
        if self.draft_depth not in SUPPORTED_DFLASH_CHAIN_DEPTHS:
            return f"unsupported draft_depth {self.draft_depth}"
        if self.page_bucket not in SUPPORTED_DFLASH_PAGE_BUCKETS:
            return f"unsupported page_bucket {self.page_bucket}"
        if len(self.tree_shape) != self.active_c * self.draft_depth:
            return "tree_shape does not match active_c * draft_depth"
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "active_c": self.active_c,
            "context_bucket": self.context_bucket,
            "page_bucket": self.page_bucket,
            "mode": self.mode,
            "draft_depth": self.draft_depth,
            "tree_shape": list(self.tree_shape),
            "top_k": self.top_k,
            "experts_per_token": self.experts_per_token,
            "replay_steps": self.replay_steps,
        }


@dataclass(frozen=True, slots=True)
class DFlashVerifyGraphAddresses:
    addresses: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        normalized = tuple((str(name), int(ptr)) for name, ptr in self.addresses)
        if any(ptr <= 0 for _, ptr in normalized):
            raise ValueError("graph bucket fixed buffer addresses must be non-zero")
        object.__setattr__(self, "addresses", tuple(sorted(normalized)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> "DFlashVerifyGraphAddresses":
        return cls(tuple((name, int(ptr)) for name, ptr in mapping.items()))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, ptr in self.addresses:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(ptr).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def as_dict(self) -> dict[str, str]:
        return {name: hex(ptr) for name, ptr in self.addresses}


@dataclass(frozen=True, slots=True)
class DFlashVerifyGraphValidation:
    bucket_key: DFlashVerifyGraphBucketKey
    status: str
    replay_steps: int
    fixed_addresses: DFlashVerifyGraphAddresses | None = None
    direct_match: bool | None = None
    graph_validation_passed: bool | None = None
    fallback_reason: str | None = None
    direct_output_fingerprint: str | None = None
    graph_output_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"captured", "direct_fallback", "capture_failed"}:
            raise ValueError("status must be captured, direct_fallback, or capture_failed")
        if self.replay_steps <= 0:
            raise ValueError("replay_steps must be positive")
        if self.status == "captured" and self.fixed_addresses is None:
            raise ValueError("captured graph validation requires fixed addresses")
        if self.status != "captured" and self.fallback_reason is None:
            raise ValueError("fallback/capture-failed graph validation requires a reason")

    def as_artifact_row(self) -> dict[str, object]:
        return {
            "bucket_key": self.bucket_key.as_dict(),
            "status": self.status,
            "replay_steps": self.replay_steps,
            "fixed_buffer_addresses": None if self.fixed_addresses is None else self.fixed_addresses.as_dict(),
            "fixed_address_fingerprint": None if self.fixed_addresses is None else self.fixed_addresses.fingerprint,
            "direct_match": self.direct_match,
            "graph_validation_passed": self.graph_validation_passed,
            "fallback_reason": self.fallback_reason,
            "direct_output_fingerprint": self.direct_output_fingerprint,
            "graph_output_fingerprint": self.graph_output_fingerprint,
        }


def dflash_verify_graph_decision(key: DFlashVerifyGraphBucketKey) -> DFlashVerifyGraphValidation:
    reason = key.fallback_reason
    if reason is not None:
        return DFlashVerifyGraphValidation(
            bucket_key=key,
            status="direct_fallback",
            replay_steps=key.replay_steps,
            fallback_reason=reason,
            direct_match=True,
            graph_validation_passed=None,
        )
    return DFlashVerifyGraphValidation(
        bucket_key=key,
        status="captured",
        replay_steps=key.replay_steps,
        fixed_addresses=DFlashVerifyGraphAddresses((("placeholder", 1),)),
        direct_match=None,
        graph_validation_passed=None,
    )


def fingerprint_int_arrays(arrays: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        shape = getattr(array, "shape", None)
        dtype = getattr(array, "dtype", None)
        data = getattr(array, "tobytes", None)
        digest.update(str(shape).encode("ascii"))
        digest.update(str(dtype).encode("ascii"))
        digest.update(data() if data is not None else bytes(array))
    return digest.hexdigest()


__all__ = [
    "DFlashVerifyGraphAddresses",
    "DFlashVerifyGraphBucketKey",
    "DFlashVerifyGraphValidation",
    "SUPPORTED_DFLASH_CHAIN_DEPTHS",
    "SUPPORTED_DFLASH_PAGE_BUCKETS",
    "dflash_verify_graph_decision",
    "fingerprint_int_arrays",
]
