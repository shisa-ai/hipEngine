"""Checkpoint physical KV layout, without a GPU dependency."""
import ctypes
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.gemma4 import Gemma4Runner, Gemma4LayerKV


def _runner(monkeypatch, position, bases):
    from hipengine.core import hip
    monkeypatch.setattr(hip, "get_hip_runtime", lambda: SimpleNamespace(
        device_synchronize=lambda: None,
        memcpy=lambda dst, src, size, kind: ctypes.memmove(dst, src, size),
    ))
    runner = object.__new__(Gemma4Runner)
    runner._closed = False
    runner._int8_kv = None
    runner.capacity = 16
    runner._position = position
    runner._kv_capacity = [16, 6]
    runner._kv_base = list(bases)
    arrays = [np.arange(cap * 4, dtype=np.uint8) for cap in (16, 16, 6, 6)]
    runner._caches = [SimpleNamespace(ptr=a.ctypes.data, nbytes=a.nbytes) for a in arrays]
    runner._kv = [Gemma4LayerKV(arrays[i*2].ctypes.data, arrays[i*2+1].ctypes.data,
                              cap, position-bases[i]) for i, cap in enumerate((16, 6))]
    monkeypatch.setattr(Gemma4Runner, "reset", lambda self: setattr(self, "_position", 0))
    return runner, arrays


@pytest.mark.parametrize("position,bases", [(0, [0, 0]), (3, [0, 0]), (11, [0, 7])])
def test_checkpoint_preserves_physical_live_rows(monkeypatch, tmp_path, position, bases):
    source, source_arrays = _runner(monkeypatch, position, bases)
    path = tmp_path / "state"
    logits = np.array([1., -2.], dtype=np.float32)
    source.save_kv_state(path, logits=logits)
    target, target_arrays = _runner(monkeypatch, 0, [0, 0])
    for array in target_arrays:
        array.fill(255)
    np.testing.assert_array_equal(target.restore_kv_state(path), logits)
    assert target._kv_base == bases
    for index, (actual, expected) in enumerate(zip(target_arrays, source_arrays)):
        size = (position - bases[index // 2]) * 4
        assert actual[:size].tobytes() == expected[:size].tobytes()
        assert np.all(actual[size:] == 255)
    assert [kv.capacity for kv in target._kv] == [16, 6]
    assert [kv.write_offset for kv in target._kv] == [position - base for base in bases]


@pytest.mark.parametrize("field,value", [("schema", 1), ("layer_capacities", [16, 5]),
                                         ("base_positions", [0, 12]),
                                         ("base_positions", [0, 0]),
                                         ("logits", -1), ("logits", 100)])
def test_invalid_layout_does_not_mutate(monkeypatch, tmp_path, field, value):
    runner, arrays = _runner(monkeypatch, 11, [0, 7])
    path = tmp_path / "state"
    runner.save_kv_state(path)
    header, payload = path.read_bytes().split(b"\n", 1)
    data = json.loads(header)
    data[field] = value
    path.write_bytes(json.dumps(data).encode() + b"\n" + payload)
    before = [a.tobytes() for a in arrays]
    with pytest.raises(ValueError):
        runner.restore_kv_state(path)
    assert runner.position == 11
    assert [a.tobytes() for a in arrays] == before
