"""Attention scope inventory spans request phases without unifying backends."""
from types import SimpleNamespace

import pytest

from hipengine.runtime.gemma4 import Gemma4Runner


@pytest.mark.parametrize('decode', [None, ('gemma4_flash_long',)])
@pytest.mark.parametrize('rows', [1, 5])
@pytest.mark.parametrize('verification,row_exact,decode_phase', [
    (False, False, False), (False, False, True),
    (True, False, False), (True, True, False),
])
def test_scope_precedence_for_prefill_tail_decode_and_verify(decode, rows, verification, row_exact, decode_phase):
    owner = SimpleNamespace(
        decode_attention_variants=decode,
        prefill_attention_variants=('gemma4_wmma_flash',),
        exact_prefill_attention_variants=('gemma4_staged',))
    actual = Gemma4Runner._attention_scope(
        owner, rows=rows, verification=verification,
        row_exact=row_exact, decode_phase=decode_phase)
    step = decode_phase or (verification and rows == 1)
    assert actual['prefill_attention_variants'] == (
        None if step and decode is not None
        else ('gemma4_plain',) if verification else owner.prefill_attention_variants)
    assert actual['decode_attention_variants'] == (
        decode if step or (verification and not row_exact) else None)
    assert actual['exact_prefill_attention_variants'] == (
        None if verification or decode_phase else owner.exact_prefill_attention_variants)
