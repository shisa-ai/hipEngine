"""Quant plugin registry."""

from __future__ import annotations

from hipengine.quant.base import QuantPlugin


class DuplicateQuantError(ValueError):
    pass


class MissingQuantError(LookupError):
    pass


_QUANTS: dict[str, QuantPlugin] = {}


def register_quant(plugin: QuantPlugin, *, replace: bool = False) -> QuantPlugin:
    if not plugin.name:
        raise ValueError("quant plugin name must be non-empty")
    if plugin.name in _QUANTS and not replace:
        raise DuplicateQuantError(f"quant plugin {plugin.name!r} already registered")
    _QUANTS[plugin.name] = plugin
    return plugin


def resolve_quant(name: str) -> QuantPlugin:
    try:
        return _QUANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_QUANTS)) or "<none>"
        raise MissingQuantError(f"quant plugin {name!r} not registered; known: {known}") from exc


def registered_quants() -> tuple[QuantPlugin, ...]:
    return tuple(_QUANTS[name] for name in sorted(_QUANTS))
