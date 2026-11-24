"""Failed prefill quiescence must retain the complete shared-weight owner chain."""
from dataclasses import MISSING, fields
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf import (
    Qwen35GGUFBringupGenerator, Qwen35GGUFResidentModelRunner,
)
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner, Qwen35GGUFResidentSession,
)


def _unprepared(cls):
    instance = object.__new__(cls)
    for item in fields(cls):
        if item.default is not MISSING:
            setattr(instance, item.name, item.default)
        elif item.default_factory is not MISSING:
            setattr(instance, item.name, item.default_factory())
    return instance


@pytest.mark.parametrize('surface', ['model', 'generator', 'shared'])
@pytest.mark.parametrize('pooled', [False, True])
@pytest.mark.parametrize('owns_runner', [False, True])
def test_failed_drain_retains_shared_weights_and_public_close_is_retryable(surface, pooled, owns_runner):
    released = []
    fail = [True]

    def synchronize(stream):
        if fail[0]:
            raise RuntimeError('device did not quiesce')

    runtime = SimpleNamespace(stream_synchronize=synchronize)
    shared = _unprepared(Qwen35GGUFFullStackRunner)
    shared.runtime = runtime
    shared.weights = SimpleNamespace(free=lambda **kwargs: released.append('weights'))
    shared.owns_resident_weights = True
    session = _unprepared(Qwen35GGUFResidentSession)
    session.runtime = runtime
    session.runner = shared
    session._owns_runner = owns_runner
    session._moe_graph = None
    session.prefill_control = lambda: None
    with pytest.raises(RuntimeError):
        session._check_prefill_control()
    session._cleanup_prefill_resource(lambda: released.append('oracle'))

    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator._shared_runner = shared
    generator._ensure_shared_pools()
    generator._shared_session_pool['test'] = [session] if pooled else []
    model = object.__new__(Qwen35GGUFResidentModelRunner)
    model.generator = generator
    model._shared_runner = shared
    model._closed = False
    model._rows = {}
    model._outputs = {}
    model._completed_metadata = {}
    model._kv_pool = None
    model._mtp2_adapter = None
    model._flush_all_packed_owners = lambda: None
    model._clear_prefix_snapshots = lambda: None
    model._release_available_sessions = lambda: None
    owner = {'model': model, 'generator': generator, 'shared': shared}[surface]
    with pytest.raises(RuntimeError, match='did not quiesce'):
        owner.close()
    assert not released
    assert shared.weights is not None
    assert generator._shared_runner is shared
    assert not getattr(owner, '_closed', False)
    assert session in shared._unquiesced_prefill_sessions.values()

    fail[0] = False
    owner.close()
    assert released == ['oracle', 'weights']
    assert shared.weights is None
    owner.close()
    assert released == ['oracle', 'weights']
