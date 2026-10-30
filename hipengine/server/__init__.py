"""Optional OpenAI-compatible server surface.

Install with ``pip install hipengine[server]`` to use the FastAPI app factory or
``python -m hipengine.server`` CLI.

The public exports are lazy PEP 562 attributes. Importing
``hipengine.server.schemas`` therefore stays a schema-only import: it does not
pull in the FastAPI application, the generation engine, or model/runtime
modules. The first access to ``ServerConfig``, ``create_app``, or
``render_chat_prompt`` imports ``hipengine.server.api`` and caches the resolved
object in package globals, so identity across repeat accesses is preserved.
"""

_LAZY_EXPORT_MODULES = {
    "ServerConfig": "hipengine.server.api",
    "create_app": "hipengine.server.api",
    "render_chat_prompt": "hipengine.server.api",
}

__all__ = ["ServerConfig", "create_app", "render_chat_prompt"]


def __getattr__(name: str):
    """Resolve the lazy server exports on first access (PEP 562)."""

    module_path = _LAZY_EXPORT_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    # Cache in package globals so repeat accesses skip the import machinery.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORT_MODULES))
