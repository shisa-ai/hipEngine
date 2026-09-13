"""CPU-only tests for scripts/tp_collective_bench.py statistics and encoding."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp_collective_bench.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp_collective_bench_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_percentile_linear_interpolation(mod) -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert mod.percentile(values, 0.0) == 1.0
    assert mod.percentile(values, 1.0) == 4.0
    assert mod.percentile(values, 0.5) == pytest.approx(2.5)
    assert mod.percentile(values, 0.25) == pytest.approx(1.75)
    assert mod.percentile([5.0], 0.9) == 5.0
    with pytest.raises(ValueError):
        mod.percentile([], 0.5)
    with pytest.raises(ValueError):
        mod.percentile([1.0], 1.5)


def test_summarize_samples_reports_tails(mod) -> None:
    summary = mod.summarize_samples([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["count"] == 5
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["p99_ms"] == pytest.approx(4.96)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 5.0
    with pytest.raises(ValueError):
        mod.summarize_samples([])


def test_bandwidth_and_bus_bandwidth(mod) -> None:
    # 20 MiB in 1 ms = 20.97 GB/s algorithm bandwidth.
    assert mod.bandwidth_gbs(payload_bytes=20 * (1 << 20), latency_ms=1.0) == pytest.approx(20.971, rel=1e-3)
    assert mod.bandwidth_gbs(payload_bytes=100, latency_ms=0.0) == 0.0
    # Two-rank all-reduce bus bandwidth is algorithm bandwidth.
    assert mod.bus_bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0, world_size=2) == pytest.approx(
        mod.bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0)
    )
    # Four-rank all-reduce bus bandwidth scales by 2*(n-1)/n = 1.5.
    assert mod.bus_bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0, world_size=4) == pytest.approx(
        1.5 * mod.bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0)
    )


def test_build_cases_orders_decode_then_prefill(mod) -> None:
    cases = mod.build_cases(
        hidden_size=5120,
        dtypes=("fp32", "bf16"),
        rows=(1, 2),
        prefill_rows=(128,),
        ops=("all_reduce",),
    )
    assert [(case.rows, case.dtype) for case in cases] == [
        (1, "fp32"),
        (2, "fp32"),
        (128, "fp32"),
        (1, "bf16"),
        (2, "bf16"),
        (128, "bf16"),
    ]
    assert cases[0].count == 5120
    assert cases[0].payload_bytes == 5120 * 4
    assert cases[3].payload_bytes == 5120 * 2
    with pytest.raises(ValueError):
        mod.build_cases(hidden_size=8, dtypes=("fp32",), rows=(0,), prefill_rows=(), ops=("all_reduce",))
    with pytest.raises(ValueError):
        mod.build_cases(hidden_size=8, dtypes=("fp32",), rows=(1,), prefill_rows=(), ops=("reduce_scatter",))


def test_encode_decode_round_trip_fp32(mod) -> None:
    values = [1.0, 2.0, -3.5, 0.0]
    encoded = mod.encode_values(values, "fp32")
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(mod.decode_values(encoded, "fp32"), np.array(values, dtype=np.float32))


def test_encode_decode_round_trip_fp16(mod) -> None:
    values = [1.0, 2.0, -3.5]
    encoded = mod.encode_values(values, "fp16")
    assert encoded.dtype == np.float16
    np.testing.assert_array_equal(mod.decode_values(encoded, "fp16"), np.array(values, dtype=np.float32))


def test_encode_decode_round_trip_bf16_exact_values(mod) -> None:
    """Integers up to 2^8 are exactly representable in bf16."""

    values = [1.0, 2.0, 3.0, 128.0]
    encoded = mod.encode_values(values, "bf16")
    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(mod.decode_values(encoded, "bf16"), np.array(values, dtype=np.float32))


def test_bf16_rounding_matches_expected_bit_pattern(mod) -> None:
    # 1.0 -> 0x3F80, 2.0 -> 0x4000, 1.0078125 (1 + 2^-7) is exact in bf16.
    encoded = mod.encode_values([1.0, 2.0, 1.0078125], "bf16")
    assert [int(bits) for bits in encoded] == [0x3F80, 0x4000, 0x3F81]


def test_bf16_halfway_values_round_to_even(mod) -> None:
    # 1 + 2^-8 is exactly halfway between 0x3F80 and 0x3F81; RNE keeps 0x3F80.
    # 1 + 3*2^-8 is halfway between 0x3F81 and 0x3F82; RNE moves up to 0x3F82.
    encoded = mod.encode_values([1.00390625, 1.01171875], "bf16")
    assert [int(bits) for bits in encoded] == [0x3F80, 0x3F82]


def test_wire_itemsize(mod) -> None:
    assert mod.wire_itemsize("fp32") == 4
    assert mod.wire_itemsize("fp16") == 2
    assert mod.wire_itemsize("bf16") == 2


def test_encode_rejects_unknown_dtype(mod) -> None:
    with pytest.raises(ValueError):
        mod.encode_values([1.0], "int8")
    with pytest.raises(ValueError):
        mod.decode_values(np.zeros(1, dtype=np.uint8), "int8")


def test_link_sampler_records_transitions(mod, tmp_path: pathlib.Path) -> None:
    device = tmp_path / "device"
    device.mkdir()
    (device / "current_link_width").write_text("16\n", encoding="utf-8")
    (device / "current_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    sampler = mod.LinkSampler({0: device}, interval_s=0.01)
    with sampler:
        import time

        time.sleep(0.03)
    payload = sampler.to_dict()
    assert payload["rank0"]["last_observed"] == {"current_width_lanes": 16, "current_speed": "16.0 GT/s PCIe"}
