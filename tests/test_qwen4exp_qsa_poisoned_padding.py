"""Unselected physical KV and selection padding must not affect QSA output."""
import numpy as np
import pytest

from hipengine.core.memory import copy_host_to_device, host_array_ptr
from tests.test_qwen4exp_qsa_h256_wave import Fixture, hip_available


def required_physical_rows(tables, selected, counts, live_count, block_size):
    required = set()
    for table, positions, count in zip(tables, selected, counts):
        for position in positions[:count]:
            if 0 <= position < live_count:
                required.add(int(table[position // block_size] * block_size + position % block_size))
    return required


def test_required_rows_include_neighbors_and_ignore_padding():
    tables=np.array([[1,0],[0,1]])
    selected=np.array([[0,2,3],[1,2,99]])
    assert required_physical_rows(tables,selected,[1,2],3,2)=={2,1}


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("variant",["parent","wave","page256"])
@pytest.mark.parametrize("poison",[0x7fc1,0x7f80])
def test_unselected_kv_and_selection_padding_are_inert(variant,poison):
    f=Fixture(3,33,edge=True,page256=variant=="page256")
    candidate=variant!="parent"
    try:
        f.run(candidate)
        baseline=f.download(candidate)
        assert np.isfinite(baseline).all()
        required=required_physical_rows(f.tables,f.selected,f.counts,4351,256)
        assert required and len(required)<4352
        mask=np.ones(4352,dtype=bool)
        mask[list(required)]=False
        key,value=f.key.copy(),f.value.copy()
        key[mask]=np.uint16(poison)
        value[mask]=np.uint16(poison)
        selected=f.selected.copy()
        for row,count in enumerate(f.counts):
            selected[row,count:]=np.iinfo(np.int64).max
        for _ in range(2):
            for device,array in ((f.dk,key),(f.dv,value),(f.ds,selected)):
                copy_host_to_device(device,host_array_ptr(array),runtime=f.runtime)
            f.run(candidate)
            np.testing.assert_array_equal(f.download(candidate).view(np.uint32),
                                          baseline.view(np.uint32))
            # Restore original bytes to exercise repeated use of the same allocations.
            for device,array in ((f.dk,f.key),(f.dv,f.value),(f.ds,f.selected)):
                copy_host_to_device(device,host_array_ptr(array),runtime=f.runtime)
            f.run(candidate)
            np.testing.assert_array_equal(f.download(candidate).view(np.uint32),
                                          baseline.view(np.uint32))
    finally:
        f.close()
