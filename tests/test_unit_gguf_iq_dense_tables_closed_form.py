"""Pin the closed forms that replace IQ decode table lookups in the strict kernel.

`gguf_iq_dense.hip` decodes IQ4 codebook entries and IQ3_XXS/IQ2_XS sign
masks arithmetically instead of indexing `__device__ __constant__` arrays: a
lane-varying index into a `__constant__` array is global memory and compiles
to one `global_load` per element. These tests assert the arithmetic and the
published tables agree exactly, so the two cannot drift apart silently.
"""
import re
from pathlib import Path

import pytest

KERNEL_DIR = Path(__file__).resolve().parents[1] / 'hipengine/kernels/hip_gfx1100/quant'
TABLES = (KERNEL_DIR / 'gguf_iq_dense_tables.h').read_text()
KERNEL = (KERNEL_DIR / 'gguf_iq_dense.hip').read_text()


# The published iq_signs[128] table, verbatim from llama.cpp
# @17252c769a63c1cb650ce98ae309cf4de0da7778 ggml/src/ggml-common.h (MIT), the
# same lineage the kernel header cites. It lives here rather than in device
# code precisely because the kernel no longer indexes it: this is the
# reference the kernel's arithmetic is checked against.
PUBLISHED_IQ_SIGNS = (
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30,
    159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45,
    46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60,
    189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75,
    204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90,
    219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245,
    246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
)


def _packed_words(marker, count):
    """Read the first `count` uint32 immediates following a marker comment."""
    assert marker in KERNEL, f'{marker!r} not found in kernel source'
    tail = KERNEL[KERNEL.index(marker):]
    words = re.findall(r'0x([0-9A-Fa-f]{8})u', tail)[:count]
    assert len(words) == count, f'{marker}: found {len(words)} of {count} immediates'
    return [int(value, 16) for value in words]


def test_iq_signs_matches_popcount_closed_form():
    """iq_signs[i] == i with bit 7 set when i has odd population count."""
    assert len(PUBLISHED_IQ_SIGNS) == 128
    for index, value in enumerate(PUBLISHED_IQ_SIGNS):
        assert value == (index | ((bin(index).count('1') & 1) << 7)), index


def test_kernel_sign_helper_uses_the_verified_closed_form():
    """The device helper is the expression this file pins, not a rewrite."""
    match = re.search(r'iq_sign_bits\(int i\) \{\s*return ([^;]+);', KERNEL)
    assert match, 'iq_sign_bits not found'
    assert match.group(1).split() == ['i', '|', '((__popc(i)', '&', '1)', '<<', '7)']


def test_iq2_xs_magnitude_word_matches_the_published_ladder():
    """IQ2_XS code 0/1/2/3 select magnitudes 8/25/43/43."""
    word = _packed_words('IQ2_XS magnitudes', 1)[0]
    assert [(word >> (8 * code)) & 0xFF for code in range(4)] == [8, 25, 43, 43]


@pytest.mark.parametrize('symbol', ('iq_signs',))
def test_replaced_tables_are_no_longer_defined(symbol):
    """The arithmetic forms replace the arrays; a stale array would mask drift."""
    assert not re.search(rf'__constant__\s+\w+\s+{symbol}\s*\[', KERNEL + TABLES)
