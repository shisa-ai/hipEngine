"""Rowtile diagnostic ownership survives public service-thread execution."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as linear
from hipengine.runtime import qwen4_exp_runner as runner


def test_rowtile_owner_crosses_thread_and_restores():
    owner = SimpleNamespace(runner=SimpleNamespace())

    def execute():
        with runner._q4k_rowtile_owner_context(owner.runner):
            return linear._resolve_use_q4k_rowtile(None)

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(execute).result() is True
        with pytest.raises(RuntimeError):
            with runner.q4k_rowtile_enabled(owner, False):
                assert owner.q4k_rowtile_enabled is False
                assert pool.submit(execute).result() is False
                with runner.q4k_rowtile_enabled(owner, True):
                    assert pool.submit(execute).result() is True
                assert pool.submit(execute).result() is False
                raise RuntimeError('restore on failure')
        assert not hasattr(owner, 'q4k_rowtile_enabled')
        assert not hasattr(owner.runner, 'q4k_rowtile_enabled')
        assert pool.submit(execute).result() is True


def test_unbound_owner_preserves_local_override():
    with linear.q4k_rowtile_session(False):
        with runner._q4k_rowtile_owner_context(SimpleNamespace()):
            assert linear._resolve_use_q4k_rowtile(None) is False
