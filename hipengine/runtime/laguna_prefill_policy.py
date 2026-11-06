"""Resolved legacy cached-prefill choices, separate from package role maps."""

from dataclasses import dataclass

from hipengine.kernels.backends import backend_package_capability


@dataclass(frozen=True)
class LagunaCachedPrefillPolicy:
    kv_preappend: bool
    cached_meta: bool
    global_qrow6: bool
    dense_initial: bool


def resolve_laguna_cached_prefill_policy(
    backend: str, *, kv_preappend: bool | None = None,
    cached_meta: bool | None = None, global_qrow6: bool | None = None,
    dense_initial: bool | None = None,
) -> LagunaCachedPrefillPolicy:
    """Resolve independent legacy choices without modifying role substitutions."""
    requested = {
        "kv_preappend": kv_preappend, "cached_meta": cached_meta,
        "global_qrow6": global_qrow6, "dense_initial": dense_initial,
    }
    return LagunaCachedPrefillPolicy(**{
        name: bool(backend_package_capability(
            backend, "LAGUNA_PREFILL_" + name.upper(), False,
        ) if value is None else value)
        for name, value in requested.items()
    })
