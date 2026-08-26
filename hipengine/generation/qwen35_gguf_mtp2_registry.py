"""Cold-path model-plugin registry for staged GGUF MTP2 adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

GGUFMTP2AdapterFactory = Callable[..., Any]

_FACTORIES: dict[str, GGUFMTP2AdapterFactory] = {}
_BUILTINS_REGISTERED = False


def register_gguf_mtp2_adapter(
    key: str,
    factory: GGUFMTP2AdapterFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one model-plugin-selected adapter factory."""

    normalized = str(key).strip()
    if not normalized:
        raise ValueError("GGUF MTP2 adapter key must be non-empty")
    if not callable(factory):
        raise TypeError("GGUF MTP2 adapter factory must be callable")
    if normalized in _FACTORIES and not replace:
        raise KeyError(f"GGUF MTP2 adapter already registered: {normalized}")
    _FACTORIES[normalized] = factory


def unregister_gguf_mtp2_adapter(key: str) -> None:
    """Remove one test/experimental adapter registration when present."""

    _FACTORIES.pop(str(key).strip(), None)


def resolve_gguf_mtp2_adapter(key: str) -> GGUFMTP2AdapterFactory:
    """Resolve one exact adapter factory or fail closed."""

    normalized = str(key).strip()
    try:
        return _FACTORIES[normalized]
    except KeyError as exc:
        raise KeyError(f"unregistered GGUF MTP2 adapter: {normalized}") from exc


def registered_gguf_mtp2_adapters() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def register_builtin_gguf_mtp2_adapters() -> None:
    """Import built-in factories once without loading model weights."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

    register_gguf_mtp2_adapter("dense_nextn", Qwen35GGUFMTP2Adapter, replace=True)
    _BUILTINS_REGISTERED = True


__all__ = [
    "GGUFMTP2AdapterFactory",
    "register_builtin_gguf_mtp2_adapters",
    "register_gguf_mtp2_adapter",
    "registered_gguf_mtp2_adapters",
    "resolve_gguf_mtp2_adapter",
    "unregister_gguf_mtp2_adapter",
]
