"""Compatibility graph aliases backed by distinct capture-identity slots."""
from __future__ import annotations


def native_target_graph_alias(session, budget: int, *, device_accept: bool, sampled: bool = False):
    """Read a handoff slot, with legacy object support for direct test adapters."""
    aliases = session.__dict__.get('_native_target_graph_aliases')
    if aliases is not None:
        return aliases.get((int(budget), bool(device_accept), bool(sampled)))
    suffix = '_n2' if device_accept else ''
    if sampled:
        suffix += '_sampled'
    return getattr(session, f'_native_spec_b{int(budget)}_target_graph{suffix}', None)


class NativeTargetGraphAlias:
    """Keep depth, device accept and sampler mode separate in one alias store.

    Context-bucket reuse has its own cache. These aliases identify the graph
    selected for direct device proposal handoff; they are not context lookup.
    """

    def __init__(self, budget: int, *, device_accept: bool, sampled: bool = False):
        self.key = (int(budget), bool(device_accept), bool(sampled))

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return instance.__dict__.get('_native_target_graph_aliases', {}).get(self.key)

    def __set__(self, instance, graph):
        aliases = instance.__dict__.setdefault('_native_target_graph_aliases', {})
        if graph is None:
            aliases.pop(self.key, None)
        else:
            aliases[self.key] = graph
