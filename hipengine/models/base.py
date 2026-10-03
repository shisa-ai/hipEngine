"""Model plugin protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

#: The surface a model plugin generates by default: prompt-to-text.
DEFAULT_GENERATION_SURFACES: tuple[str, ...] = ("text",)


def generation_surfaces(plugin: object) -> tuple[str, ...]:
    """Return the generation surfaces a model plugin declares.

    A surface names what a caller can ask this model to produce -- ``text`` for
    prompt-to-text, ``song`` for the YuE2 lyrics-to-audio pipeline. Callers use
    the declaration to route a request to a path the model actually implements
    and to refuse one it does not, instead of keying on the model's name or
    artifact. Plugins that declare nothing are text models, which is what every
    plugin but YuE2 is.
    """

    declared = getattr(plugin, "generation_surfaces", DEFAULT_GENERATION_SURFACES)
    if isinstance(declared, str):
        raise TypeError("generation_surfaces must be a sequence of surface names, not a string")
    surfaces = tuple(str(surface).strip() for surface in declared)
    if not surfaces or any(not surface for surface in surfaces):
        raise ValueError("generation_surfaces must name at least one non-empty surface")
    return surfaces


@runtime_checkable
class ModelPlugin(Protocol):
    """Architecture-level layer sequence and metadata.

    Real model plugins will also own weight-name maps, chat templates, RoPE config, and
    architecture-specific layer parameters. The scaffold keeps only the layer sequence needed
    to validate registry + fusion planning.

    A plugin may also declare ``generation_surfaces``: the generation surfaces it
    implements. The attribute is optional, and reading it through
    :func:`generation_surfaces` is what keeps the declaration from becoming a
    required protocol member.
    """

    name: str
    architectures: tuple[str, ...]

    def layer_sequence(self) -> Sequence[str]:
        """Return primitive layer keys in execution order."""
