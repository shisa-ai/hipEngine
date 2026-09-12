import struct
import pytest
from scripts.qwen4exp_thread_perf import decode_group


def test_decode_ids_and_exact_large_counters():
    large=2**55+3
    assert decode_group(struct.pack("=7Q",2,100,99,large,7,41,9))==(
        100,99,{7:large,9:41})


@pytest.mark.parametrize("values",[(1,100,99,1,7,2,9),(2,100,101,1,7,2,9),(2,100,99,1,7,2,7)])
def test_invalid_group(values):
    with pytest.raises(ValueError):
        decode_group(struct.pack("=7Q",*values))


def test_short_read_rejected():
    with pytest.raises(ValueError):
        decode_group(b"")
