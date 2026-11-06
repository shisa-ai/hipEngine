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


def set_native_target_graph_alias(session, budget: int, graph, *, device_accept: bool, sampled: bool = False):
    """Publish a handoff slot without merging capture identities."""
    key = (int(budget), bool(device_accept), bool(sampled))
    aliases = session.__dict__.setdefault('_native_target_graph_aliases', {})
    if graph is None:
        aliases.pop(key, None)
    else:
        aliases[key] = graph
    suffix = '_n2' if device_accept else ''
    if sampled:
        suffix += '_sampled'
    name = f'_native_spec_b{int(budget)}_target_graph{suffix}'
    if not isinstance(getattr(type(session), name, None), NativeTargetGraphAlias):
        setattr(session, name, graph)


def clear_native_target_graph_aliases(session, graph):
    """Clear every alias to a retired graph, including legacy direct adapters."""
    aliases = session.__dict__.get('_native_target_graph_aliases', {})
    for key, value in tuple(aliases.items()):
        if value is graph:
            aliases.pop(key, None)
    for name, value in tuple(session.__dict__.items()):
        if name.startswith('_native_spec_b') and '_target_graph' in name and value is graph:
            setattr(session, name, None)


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
