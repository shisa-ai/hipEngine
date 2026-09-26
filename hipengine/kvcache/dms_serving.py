"""Configuration shared by public DMS entry points."""
from __future__ import annotations


def resolve_dms_serving_options(
    metadata_path: str | None,
    max_active_requests: int | None,
    prefix_cache: str | None,
) -> tuple[str | None, int | None, str | None]:
    """Resolve singleton defaults and reject unsupported ownership contracts."""
    if metadata_path is None:
        return None, max_active_requests, prefix_cache
    path = str(metadata_path).strip()
    if not path:
        raise ValueError("dms_metadata_path must be non-empty when set")
    if max_active_requests is not None and int(max_active_requests) != 1:
        raise ValueError("DMS resident sessions require max_active_requests=1; packed DMS ownership is not implemented")
    if prefix_cache is not None and str(prefix_cache) != "off":
        raise ValueError("DMS prefix-cache snapshots are not implemented; use prefix_cache='off'")
    return path, 1, "off"
