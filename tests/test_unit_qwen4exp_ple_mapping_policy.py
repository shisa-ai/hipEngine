"""An instance-owned mapping hint must survive remap without affecting data."""

import mmap
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.qwen4_exp_materialize import Qwen4ExpPLEMMapTable
from tests._qwen4_exp_ple_fixtures import _iq4_nl_rows, _ple_tensor


def make_table():
    values = _iq4_nl_rows((1., 2., 3.))
    mappings = []

    class Raw:
        shape = values.shape

        def __init__(self):
            self.advice = []
            self._mmap = SimpleNamespace(madvise=self.advice.append, close=lambda: None)
            mappings.append(self)

        def __getitem__(self, ids):
            return values[ids]

    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda _: Raw()), _ple_tensor(3), semantic_rows=3,
    )
    return table, mappings


def test_mapping_policy_is_instance_owned_and_survives_remap():
    table, mappings = make_table()
    other, other_mappings = make_table()
    expected = table.gather_rows([2, 0, 2])
    assert table.configure_mapping_access("random")
    table.advise_cache("cold")
    assert mappings[0].advice == [mmap.MADV_RANDOM, mmap.MADV_DONTNEED]
    assert mappings[1].advice == [mmap.MADV_RANDOM]
    assert not other_mappings[0].advice
    assert table.gather_rows([2, 0, 2]).tobytes() == expected.tobytes()
    table.enable_telemetry()
    assert table.telemetry()["mapping_access_default"] == "random"
    assert table.telemetry()["mapping_access_applied"] is True
    table.close()
    other.close()


def test_explicit_normal_fallback_and_invalid_policy():
    table, mappings = make_table()
    table.configure_mapping_access("random")
    table.configure_mapping_access("normal")
    with pytest.raises(ValueError):
        table.configure_mapping_access("invalid")
    table.advise_cache("cold")
    assert mappings[-1].advice == [mmap.MADV_NORMAL]


def test_full_cache_warm_temporarily_restores_sequential_access():
    table, mappings = make_table()
    table.configure_mapping_access("random")
    table.warm_page_cache(chunk_rows=2)
    assert mappings[0].advice == [mmap.MADV_RANDOM, mmap.MADV_SEQUENTIAL, mmap.MADV_RANDOM]


def test_missing_mapping_capability_is_a_safe_fallback():
    values = _iq4_nl_rows((1., 2., 3.))
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda _: values), _ple_tensor(3), semantic_rows=3,
    )
    assert table.configure_mapping_access("random") is False
    np.testing.assert_array_equal(table.gather_rows([0]), np.full((1, 160), -127., np.float32))
