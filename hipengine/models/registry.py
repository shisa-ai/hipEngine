"""Model plugin registry."""

from __future__ import annotations

from hipengine.models.base import ModelPlugin


class DuplicateModelError(ValueError):
    pass


class MissingModelError(LookupError):
    pass


_MODELS_BY_ARCH: dict[str, ModelPlugin] = {}
_MODELS_BY_NAME: dict[str, ModelPlugin] = {}


def register_model(plugin: ModelPlugin, *, replace: bool = False) -> ModelPlugin:
    if not plugin.name:
        raise ValueError("model plugin name must be non-empty")
    if not plugin.architectures:
        raise ValueError("model plugin must declare at least one HF architecture string")

    collisions = [arch for arch in plugin.architectures if arch in _MODELS_BY_ARCH]
    if plugin.name in _MODELS_BY_NAME:
        collisions.append(plugin.name)
    if collisions and not replace:
        raise DuplicateModelError(f"model plugin collision for: {', '.join(sorted(collisions))}")

    _MODELS_BY_NAME[plugin.name] = plugin
    for arch in plugin.architectures:
        _MODELS_BY_ARCH[arch] = plugin
    return plugin


def resolve_model(architecture: str) -> ModelPlugin:
    plugin = _MODELS_BY_ARCH.get(architecture) or _MODELS_BY_NAME.get(architecture)
    if plugin is None:
        known = ", ".join(sorted((*_MODELS_BY_ARCH, *_MODELS_BY_NAME))) or "<none>"
        raise MissingModelError(
            f"model architecture {architecture!r} not registered; known: {known}"
        )
    return plugin


def registered_models() -> tuple[ModelPlugin, ...]:
    return tuple(sorted(set(_MODELS_BY_NAME.values()), key=lambda plugin: plugin.name))


def clear_registry_for_tests() -> None:
    _MODELS_BY_ARCH.clear()
    _MODELS_BY_NAME.clear()
