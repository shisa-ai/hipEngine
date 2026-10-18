"""Native sampler gate request ownership excludes independently owned leases."""
import pytest
from scripts.gguf_native_sampler_gate import _released_pool_exact, build_parser, _native_request

@pytest.mark.parametrize("refs,pinned,leases,expected", [(0,0,0,True),(16,16,16,True),(17,16,16,False),(16,17,16,False),(17,17,16,False)])
def test_released_pool_requires_only_declared_workspace_pages(refs,pinned,leases,expected):
    assert _released_pool_exact({"refcounted_pages":refs,"pinned_pages":pinned},leases) is expected

def test_full_vocabulary_gate_request():
    args = build_parser().parse_args(["--top-k", "0"])
    request = _native_request(((10,11),), max_tokens=4, row_seeds=(17,), top_k=args.top_k)
    assert request.top_k == 0
    assert request.top_p == .82
