"""Empty PLE lookups must not call block dequantizers with zero blocks."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.qwen4_exp_materialize import Qwen4ExpPLEMMapTable
from tests._qwen4_exp_ple_fixtures import _iq4_nl_rows, _ple_tensor


def test_empty_gather_preserves_shape_counters_and_closed_guard():
    values = _iq4_nl_rows((1., 2.))
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda _: values), _ple_tensor(2), semantic_rows=2,
    )
    table.enable_telemetry()
    output = table.gather_rows([])
    assert output.shape == (0, 160)
    assert output.dtype == np.float32
    assert table.rows_gathered == 0
    assert table.telemetry()["calls"] == 0
    table.close()
    with pytest.raises(RuntimeError):
        table.gather_rows([])
