from scripts.qwen4exp_cpu_frequency import CpuFrequency


def test_read_only_frequency_and_missing_fields(tmp_path):
    path=tmp_path/"cpu3"/"cpufreq"
    path.mkdir(parents=True)
    (path/"cpuinfo_avg_freq").write_text("1234567\n")
    (path/"scaling_driver").write_text("test-driver\n")
    probe=CpuFrequency(tmp_path,lambda:3)
    result=probe.sample()
    assert result["values"]["cpuinfo_avg_freq"]==1234567
    assert result["values"]["scaling_driver"]=="test-driver"
    assert "scaling_cur_freq" in result["errors"]
    assert result["same_cpu_at_sample_boundaries"]
    assert (path/"cpuinfo_avg_freq").read_text()=="1234567\n"


def test_migration_is_not_hidden(tmp_path):
    cpus=iter([0,1])
    result=CpuFrequency(tmp_path,lambda:next(cpus)).sample()
    assert not result["same_cpu_at_sample_boundaries"]
