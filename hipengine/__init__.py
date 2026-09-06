"""Public hipEngine API.

Importing this module must remain torch-free. Optional torch interop belongs behind the
``hipengine[torch]`` extra and outside the runtime hot path.

Startup isolation: ``ExecutionProfile`` is imported eagerly because
``hipengine.execution_profiles`` is pure metadata (no device or kernel imports),
so a bare ``import hipengine`` stays CPU-safe. ``LLM`` and ``SamplingParams``
are lazy PEP 562 exports: importing ``hipengine`` alone no longer pulls
``hipengine.llm`` -- and through it the speculative package and the GPU kernel
backends -- so CPU-only metadata tooling (for example
``scripts/gguf_quant_route_audit.py``) can run where the HIP runtime is absent.
The first attribute access imports ``hipengine.llm`` and caches the value in
package globals, so ``from hipengine import LLM, SamplingParams``, identity
across repeat accesses, ``dir(hipengine)``, and the runtime kernel-registration
chain all behave exactly as they did when the imports were eager.
"""

from hipengine.execution_profiles import ExecutionProfile

_LAZY_EXPORT_MODULES = {"LLM": "hipengine.llm", "SamplingParams": "hipengine.llm"}

__all__ = ["ExecutionProfile", "LLM", "SamplingParams"]


def __getattr__(name: str):
    """Resolve the lazy LLM-surface exports on first access (PEP 562)."""

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
