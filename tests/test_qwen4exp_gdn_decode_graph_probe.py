import pytest
from scripts.qwen4exp_gdn_decode_graph_probe import mutable_regions


def test_mutable_state_sizes():
    kw=dict(rows=1,num_k_heads=16,num_v_heads=48,head_dim=128,branches=4,
            hidden=2560,conv_kernel=4,conv_state_ptr=2,recurrent_state_ptr=3)
    assert mutable_regions(1,kw)==((1,20480),(2,163840),(3,3145728))
    with pytest.raises(ValueError):
        mutable_regions(1,dict(kw,rows=2))
    with pytest.raises(ValueError):
        mutable_regions(1,dict(kw,num_v_heads=32))
