"""Select packed workspace arena backing without touching packed arithmetic."""
from __future__ import annotations


from hipengine.runtime.qwen35_gguf_workspace import StandalonePackedBackingLease


def acquire_standalone_packed_backing(*, config, runtime, layout, pages, pool, allocate):
    """Construct fallback KV behind the storage adapter, not packed arithmetic."""
    from hipengine.runtime.qwen35_gguf_kv import _qwen35_gguf_kv_page_bytes

    return StandalonePackedBackingLease.acquire(
        pages=pages, page_bytes=_qwen35_gguf_kv_page_bytes(config, layout), pool=pool,
        allocate=lambda: allocate(
            config, runtime=runtime, start_block_id=0, pages=pages, layout=layout,
        ),
    )


def acquire_packed_backing(
    *, config, runtime, layout, pages, pool, allocate, eager_key, lease_growth, on_shortfall,
):
    """Resolve arena or standalone storage and transfer one ownership result."""
    backing, page_ids, growth_key = select_packed_arena_backing(
        pool, total_pages=pages, layout=layout, eager_key=eager_key,
        lease_growth=lease_growth, on_shortfall=on_shortfall,
    )
    if backing is not None:
        return backing, page_ids, "pool_lease", None, growth_key
    lease = acquire_standalone_packed_backing(
        config=config, runtime=runtime, layout=layout, pages=pages,
        pool=pool, allocate=allocate,
    )
    return lease.backing, tuple(range(pages)), "private", lease, None


def select_packed_arena_backing(
    pool, *, total_pages: int, layout, eager_key: str, lease_growth, on_shortfall,
):
    """Return backing, local page map and owned growth key, or no arena.

    The eager reservation stays pool-owned. Any temporary growth lease is released
    on refusal or exception; successful ownership transfers to the caller.
    """
    if pool is None:
        return None, None, None
    growth_key = None
    try:
        workspace_pages = getattr(pool, "workspace_pages", None)
        pages = workspace_pages(eager_key) if callable(workspace_pages) else None
        if pages is None:
            grown = lease_growth(pool, needed_pages=total_pages)
            if grown is not None:
                growth_key, pages = grown
        if pages is not None and len(pages) < total_pages:
            on_shortfall(len(pages))
            grown = lease_growth(pool, needed_pages=total_pages - len(pages))
            if grown is None:
                pages = None
            else:
                growth_key, extra = grown
                pages = (*tuple(pages), *extra)
        backing = None
        local_pages = None
        if pages is not None:
            bind = getattr(pool, "workspace_backing_for_pages", None)
            if callable(bind):
                binding = bind(tuple(pages[:total_pages]))
                compatible = binding is not None
                if compatible:
                    backing, local_pages = binding
            else:
                root_fits = getattr(pool, "workspace_pages_fit_root_backing", None)
                compatible = not callable(root_fits) or root_fits(tuple(pages[:total_pages]))
            if not compatible:
                if growth_key is not None:
                    pool.release_workspace(growth_key)
                    growth_key = None
                pages = None
                if callable(bind):
                    replacement = lease_growth(pool, needed_pages=total_pages)
                    if replacement is not None:
                        growth_key, replacement_pages = replacement
                        binding = bind(tuple(replacement_pages))
                        if binding is None:
                            pool.release_workspace(growth_key)
                            growth_key = None
                        else:
                            pages = replacement_pages
                            backing, local_pages = binding
        if pages is None:
            return None, None, None
        if backing is None:
            backing = getattr(pool, "backing", None)
        if backing is None or getattr(backing, "layout", None) != layout:
            raise RuntimeError("packed workspace lease has no layout-compatible KV backing")
        page_ids = tuple(int(page) for page in (
            local_pages if local_pages is not None else pages[:total_pages]
        ))
        return backing, page_ids, growth_key
    except BaseException:
        if growth_key is not None:
            pool.release_workspace(growth_key)
        raise
