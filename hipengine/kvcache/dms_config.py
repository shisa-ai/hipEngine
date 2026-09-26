"""Public compact-retention configuration, independent of model identity."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DMSConfig:
    """Select an external DMS sidecar and its single-request prefill route."""

    metadata_path: str | Path
    prefill_mode: str = "dense_pool"
    decision_mode: str = "sidecar"

    def __post_init__(self) -> None:
        if not str(self.metadata_path).strip():
            raise ValueError("DMS metadata_path must not be empty")
        if self.prefill_mode not in {"dense_pool", "layer_outer"}:
            raise ValueError("DMS prefill_mode must be dense_pool or layer_outer")
        if self.decision_mode not in {"sidecar", "no_evict"}:
            raise ValueError("DMS decision_mode must be sidecar or no_evict")
        object.__setattr__(self, "metadata_path", str(self.metadata_path))
