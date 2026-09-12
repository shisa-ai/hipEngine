import pytest

from scripts.qwen4exp_graph_census import analyze_window, summarize, union_ns


def row(start, end, corr=1, **kwargs):
    return dict(Start_Timestamp=str(start), End_Timestamp=str(end),
                Correlation_Id=str(corr), **kwargs)


def test_interval_union_counts_overlap_once():
    assert union_ns([(0, 5), (3, 9), (12, 15)]) == 12
    assert union_ns([]) == 0
    with pytest.raises(ValueError):
        union_ns([(5, 4)])


def test_graph_census_does_not_treat_api_sum_as_exposed_cost():
    apis = [
        row(0, 10, Function="hipGraphLaunch"),
        row(20, 90, 2, Function="hipStreamSynchronize"),
    ]
    kernels = [row(5, 40), row(50, 80)]
    got = analyze_window(apis, kernels, (0, 100))
    assert got["kernel_union_ns"] == 65
    assert got["non_kernel_window_upper_bound_ns"] == 35
    assert got["graph_launches"] == 1
    assert got["graph_kernel_rows"] == 2
    assert got["intra_graph_gap_ns"] == 10
    assert got["graph_api_non_kernel_ns"] == 5
    assert got["kernel_rows_without_api"] == 0


def test_missing_correlations_remain_explicit_and_windows_clip():
    got = analyze_window(
        [row(-20, -10, Function="hipGraphLaunch")],
        [row(20, 40, 9), row(90, 110, 10)], (0, 100))
    assert got["graph_launches"] == 0
    assert got["kernel_rows_without_api"] == 2
    assert got["kernel_union_ns"] == 30
    assert got["boundary_crossing_kernel_rows"] == 1


def test_large_integer_timestamps_preserve_single_ns():
    n = 10**18
    got = analyze_window(
        [row(n, n + 1, Function="hipGraphLaunch")],
        [row(n + 2, n + 3)], (n, n + 4))
    assert got["kernel_union_ns"] == 1
    assert got["graph_api_non_kernel_ns"] == 1


def test_summary_rejects_incomplete_or_dirty_capture():
    with pytest.raises(ValueError, match="completed"):
        summarize({"status": "running"})
    with pytest.raises(ValueError, match="clean"):
        summarize({"status": "captured", "source": {"tracked_clean": False}})


def test_independent_graph_spans_are_not_one_large_graph_gap():
    apis = [row(0, 1, 1, Function="hipGraphLaunch"),
            row(50, 51, 2, Function="hipGraphLaunch")]
    kernels = [row(2, 5, 1), row(7, 10, 1), row(52, 55, 2)]
    got = analyze_window(apis, kernels, (0, 100))
    assert got["intra_graph_gap_ns"] == 2
    assert got["graph_launches"] == 2
    assert got["graph_kernel_union_ns"] == 9
