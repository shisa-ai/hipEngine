"""Selected-expert pair reuse width scopes must not leak between requests."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hipengine.runtime import qwen35_gguf_runner as runner


@pytest.mark.parametrize('stem', ['pairreuse', 'down_pairreuse', 'q6_down_pairreuse'])
def test_pairreuse_width_scope_is_thread_local_and_restores(monkeypatch, stem):
    env = {
        'pairreuse': 'HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE',
        'down_pairreuse': 'HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE',
        'q6_down_pairreuse': 'HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE',
    }[stem]
    monkeypatch.delenv(env, raising=False)
    scope = getattr(runner, f'_gguf_t16_selected_{stem}_min_rows_scope')
    enabled = getattr(runner, f'_gguf_t16_selected_{stem}_enabled')
    ready = Barrier(2)
    checked = Barrier(2)

    def request(floor):
        with scope(floor):
            ready.wait(timeout=5)
            actual = enabled()
            checked.wait(timeout=5)
        return actual

    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(request, 8)
        inactive = pool.submit(request, None)
        assert (active.result(), inactive.result()) == (True, False)
    assert enabled() is False
    with scope(8):
        with pytest.raises(RuntimeError, match='failed'):
            with scope(None):
                assert enabled() is False
                raise RuntimeError('failed')
        assert enabled() is True
    assert enabled() is False
