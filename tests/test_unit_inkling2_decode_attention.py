"""CPU tests for the one-row Inkling 2 decode attention route (no GPU)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn
from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention


class _Runtime:
    def __init__(self):
        self.next = 0x100000

    def malloc(self, nbytes):
        self.next += (nbytes + 255) // 256 * 256 + 256
        return self.next - ((nbytes + 255) // 256 * 256 + 256)

    def free(self, ptr):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _attention(block_table, decode_table, capacity=64):
    return Inkling2DeviceAttention(runtime=_Runtime(), launch=None, write=None,
                                   geometry=[(capacity, 4, 2, 8, 32), (capacity, 4, 2, 8, 32)],
                                   block_table=block_table, decode_table=decode_table)


def test_one_resident_row_takes_the_decode_route_with_its_own_scratch():
    seen = {}
    attention = _attention(None, lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs))
    attention.step_block_resident(1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    args, kwargs = seen["args"], seen["kwargs"]
    # query, keys, values, rel, sink, out, spans, nq, nk, dim, extent, key, value, rows
    assert args[:6] == (1, attention.banks[1][0], attention.banks[1][1], 4, 5, 7)
    assert args[7:] == (4, 2, 8, 1, 2, 3, 1)  # eager: the bias extent is the one live key
    assert kwargs["gain"] == 6 and kwargs["table"] == 8 and kwargs["rel_width"] == 2
    assert kwargs["scores"] == attention.scores and kwargs["scores"] != 0
    assert kwargs["gain_out"] == attention.gain and kwargs["gain_out"] not in (0, kwargs["scores"])
    assert attention.positions[1] == 1
    # The scratch is allocated once.
    attention.step_block_resident(1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    assert seen["kwargs"]["scores"] == kwargs["scores"] and seen["kwargs"]["gain_out"] == kwargs["gain_out"]


def test_blocks_and_owners_without_the_decode_route_keep_the_block_kernel():
    block, decode = Mock(), Mock()
    attention = _attention(block, decode)
    attention.step_block_resident(1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 4)
    assert block.call_count == 1 and decode.call_count == 0 and attention.positions[1] == 4
    attention.step_block_resident(1, 4, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    assert block.call_count == 1 and decode.call_count == 1
    _attention(block, None).step_block_resident(1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    assert block.call_count == 2
    with pytest.raises(RuntimeError, match="lag-table kernel"):
        Inkling2DeviceAttention(runtime=_Runtime(), launch=None, write=None,
                                geometry=[(64, 4, 2, 8, 32)]).step_block_resident(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)


def test_decode_launcher_refuses_what_it_cannot_serve_before_touching_hip(monkeypatch):
    monkeypatch.setattr(attn, "launch_cache_write", Mock())
    monkeypatch.setattr(attn, "_build_decode_scores", Mock())
    spans = SimpleNamespace(max_live_count=64)
    with pytest.raises(ValueError, match="exactly one query row"):
        attn.launch_decode_table_attention(1, 2, 3, 4, 5, 6, spans, 4, 2, 8, 32, 2, 3, 2, gain=7, table=8,
                                          rel_width=2, scores=9, gain_out=10)
    for missing in ("scores", "gain_out", "table"):
        values = {"gain": 7, "table": 8, "rel_width": 2, "scores": 9, "gain_out": 10}
        values[missing] = 0
        with pytest.raises(ValueError, match="scratch or lag table"):
            attn.launch_decode_table_attention(1, 2, 3, 4, 5, 6, spans, 4, 2, 8, 32, 2, 3, 1, **values)
    attn.launch_cache_write.assert_not_called()
