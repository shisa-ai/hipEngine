from scripts.qwen4exp_qsa_phase_trace_summary import union_ns,step_summary


def test_union_does_not_double_count_overlap():
    assert union_ns([(1,5),(3,7),(9,10)])==7
    assert union_ns([])==0


def test_exact_integer_timestamps_and_residual():
    start=2059133758503074
    k=dict(Start_Timestamp=str(start+1),End_Timestamp=str(start+5),Kernel_Name="kernel")
    result=step_summary(start,start+10,[k])
    assert result["kernel_busy_ms"]==4/1e6
    assert result["kernel_count"]==1
    assert result["wall_ms"]==10/1e6
