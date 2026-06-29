from __future__ import annotations

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner


def test_write_kv_rows_from_device_seed_base_uses_d2d_hidden_rows(monkeypatch) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.memcpy_calls = []

        def memcpy(self, dst, src, nbytes, kind) -> None:
            self.memcpy_calls.append((int(dst), int(src), int(nbytes), int(kind)))

    runtime = Runtime()
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = runtime
    runner.hidden_size = 4
    runner.token_embd_f32 = np.arange(40, dtype=np.float32).reshape(10, 4)
    runner.seed_a = DeviceBuffer(0x1000, 16)
    runner.token_embed = DeviceBuffer(0x2000, 16)

    writes = []

    def write_one_kv(**kwargs) -> None:
        writes.append(
            (
                int(kwargs["dense_cache_len"]),
                np.asarray(kwargs["cos"]).copy(),
                np.asarray(kwargs["sin"]).copy(),
            )
        )

    runner._write_one_kv = write_one_kv

    result_len = runner.write_kv_rows_from_device_seed_base(
        0x5000,
        np.asarray([2, 3], dtype=np.int64),
        positions=np.asarray([1, 2], dtype=np.int64),
        rope_cos=np.arange(12, dtype=np.float32).reshape(3, 4),
        rope_sin=np.arange(12, 24, dtype=np.float32).reshape(3, 4),
        dense_key_cache=DeviceBuffer(0x6000, 128),
        dense_value_cache=DeviceBuffer(0x7000, 128),
        dense_cache_len=7,
    )

    assert result_len == 9
    d2d_calls = [
        call for call in runtime.memcpy_calls
        if call[3] == int(HipMemcpyKind.DEVICE_TO_DEVICE)
    ]
    assert d2d_calls == [
        (0x1000, 0x5000, 16, int(HipMemcpyKind.DEVICE_TO_DEVICE)),
        (0x1000, 0x5010, 16, int(HipMemcpyKind.DEVICE_TO_DEVICE)),
    ]
    assert [item[0] for item in writes] == [7, 8]
    np.testing.assert_array_equal(writes[0][1], np.asarray([[4, 5, 6, 7]], dtype=np.float32))
    np.testing.assert_array_equal(writes[1][1], np.asarray([[8, 9, 10, 11]], dtype=np.float32))
    np.testing.assert_array_equal(writes[0][2], np.asarray([[16, 17, 18, 19]], dtype=np.float32))
    np.testing.assert_array_equal(writes[1][2], np.asarray([[20, 21, 22, 23]], dtype=np.float32))
