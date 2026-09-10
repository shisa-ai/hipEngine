import pytest
from scripts.qwen4exp_cpu_floor import CpuFloor


def fixture(tmp_path):
    path=tmp_path/"cpu17"/"cpufreq"
    path.mkdir(parents=True)
    for name,value in (("scaling_min_freq","2000000"),("scaling_max_freq","5187500"),
                       ("affected_cpus","17")):
        (path/name).write_text(value)
    return path


def test_restores_on_body_failure(tmp_path):
    path=fixture(tmp_path)
    floor=CpuFloor(17,4000000,tmp_path,lambda p,v:p.write_text(str(v)))
    try:
        assert floor.enter()["observed_khz"]==4000000
        raise RuntimeError("body failed")
    except RuntimeError:
        pass
    finally:
        assert floor.close()==2000000
    assert path.joinpath("scaling_min_freq").read_text()=="2000000"


def test_invalid_floor_does_not_write(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError):
        CpuFloor(17,6000000,tmp_path,lambda *a:pytest.fail("write")).enter()
    with pytest.raises(ValueError):
        CpuFloor(None,4000000)
