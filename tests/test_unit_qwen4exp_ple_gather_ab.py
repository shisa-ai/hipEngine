import mmap
from types import SimpleNamespace

import pytest

from scripts.qwen4exp_ple_gather_ab import select_gather_arm


def test_mapping_arms_reset_hint_after_each_cold_remap():
    calls = []

    class Table:
        def advise_cache(self, mode):
            calls.append(("remap", mode))
            self._raw = SimpleNamespace(_mmap=SimpleNamespace(
                madvise=lambda hint: calls.append(("advice", hint)),
            ))

    table = Table()
    original = lambda ids: ids
    for mode in ("after", "before"):
        select_gather_arm(table, original, mode=mode, method="mmap_random", cache_mode="cold")
        assert table.gather_rows is original
    assert calls == [
        ("remap", "cold"), ("advice", mmap.MADV_RANDOM),
        ("remap", "cold"), ("advice", mmap.MADV_NORMAL),
    ]


def test_gather_arm_rejects_unknown_mode():
    with pytest.raises(ValueError):
        select_gather_arm(None, None, mode="invalid", method="mmap_random", cache_mode="warm")
