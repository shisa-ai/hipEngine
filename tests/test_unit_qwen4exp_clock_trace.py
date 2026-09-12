from types import SimpleNamespace

from scripts.qwen4exp_clock_trace import read_sensors, summarize_window


def test_sensor_sample_and_phase_boundary(tmp_path):
    (tmp_path / "freq1_input").write_text("2880000000\n")
    (tmp_path / "temp1_input").write_text("77000\n")
    (tmp_path / "power1_average").write_text("126053000\n")
    values = read_sensors(tmp_path)
    assert values == {"sclk_hz": 2880000000, "temperature_millic": 77000, "power_uw": 126053000}
    rows = [{"time_ns": t, **values} for t in (9, 10, 15, 20, 21)]
    summary = summarize_window(rows, 10, 20)
    assert summary["samples"] == 3
    assert summary["sclk_hz"]["mean"] == 2880000000
    assert summarize_window(rows, 22, 30)["samples"] == 0
    assert read_sensors(tmp_path / "missing") == {}


def test_sample_reuses_four_timestamps(monkeypatch):
    from scripts import qwen4exp_canonical_ar_bench as bench

    clock = iter((100_000_000, 120_000_000, 121_000_000, 130_000_000))
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(bench, "_process_memory_snapshot", lambda: {
        "process_read_bytes": 0, "minor_faults": 0, "major_faults": 0})
    runner = SimpleNamespace(
        runtime=SimpleNamespace(device_synchronize=lambda: None),
        prefill=lambda ids: SimpleNamespace(token_id=2),
        step=lambda token: SimpleNamespace(token_id=3),
    )
    case = {"id": "case", "category": "code", "prompt_token_ids": [1],
            "prompt_token_ids_sha256": "test"}
    row = bench._hipengine_case_sample(runner, case=case, repetition=0, transitions=1)
    assert row["prefill_ms"] == 20
    assert row["decode_ms"] == 9
    assert row["phase_windows_ns"] == {
        "prefill": [100_000_000, 120_000_000], "decode": [121_000_000, 130_000_000]}
