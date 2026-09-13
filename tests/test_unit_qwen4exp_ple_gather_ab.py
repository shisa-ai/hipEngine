import mmap
from types import SimpleNamespace

import pytest

from scripts.qwen4exp_ple_gather_ab import pair_sequence, select_gather_arm


def test_mapping_arms_reset_hint_after_each_cold_remap():
    calls = []

    class Table:
        def advise_cache(self, mode):
            calls.append(("remap", mode))
            self._raw = SimpleNamespace(_mmap=SimpleNamespace(
                madvise=lambda hint: calls.append(("advice", hint)),
            ))

        def configure_mapping_access(self, mode):
            self._raw._mmap.madvise(mmap.MADV_RANDOM if mode == "random" else mmap.MADV_NORMAL)
            return True

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


def test_pair_sequence_preserves_original_order_and_extends_counterbalance():
    assert pair_sequence(0, 3) == ("before", "after", "after", "before", "before", "after")
    for index in (0, 1):
        sequence = pair_sequence(index, 5)
        assert sequence.count("before") == sequence.count("after") == 5
        assert all(set(sequence[i:i + 2]) == {"before", "after"} for i in range(0, 10, 2))
    with pytest.raises(ValueError):
        pair_sequence(0, 0)
