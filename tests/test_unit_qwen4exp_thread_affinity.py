import pytest
from scripts.qwen4exp_thread_affinity import ThreadAffinity


def test_pin_and_restore(monkeypatch):
    mask=[{1,17}]
    writes=[]
    monkeypatch.setattr("os.sched_getaffinity",lambda pid:set(mask[0]))
    def set_mask(pid,value):
        assert pid==0
        writes.append(set(value))
        mask[0]=set(value)
    monkeypatch.setattr("os.sched_setaffinity",set_mask)
    affinity=ThreadAffinity(17)
    try:
        assert affinity.enter()["active"]==[17]
        raise RuntimeError("simulated probe failure")
    except RuntimeError:
        pass
    finally:
        assert affinity.close()==[1,17]
    assert writes==[{17},{1,17}]


def test_outside_mask_rejected_without_writes(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity",lambda pid:{1,17})
    monkeypatch.setattr("os.sched_setaffinity",lambda *args:pytest.fail("unexpected write"))
    with pytest.raises(ValueError):
        ThreadAffinity(2).enter()
