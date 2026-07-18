"""Focused host tests for scripts/paro_live_server_bench.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "paro_live_server_bench.py"
    module_name = "_paro_live_server_bench_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_parse_configurations_keeps_explicit_native_and_serial_order() -> None:
    assert SCRIPT._parse_configurations("c1,native_c2,native_c8,serial_c8") == (
        "c1",
        "native_c2",
        "native_c8",
        "serial_c8",
    )
    with pytest.raises(Exception, match="unknown configurations"):
        SCRIPT._parse_configurations("native_c3")
    with pytest.raises(Exception, match="unique"):
        SCRIPT._parse_configurations("c1,c1")


def test_prompt_rows_change_only_the_final_token_and_roundtrip_text() -> None:
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        @staticmethod
        def decode(ids):
            return ",".join(str(token) for token in ids)

        @staticmethod
        def encode(text):
            return Encoding([int(token) for token in text.split(",")])

    rows = SCRIPT._prompt_rows(Tokenizer(), 5, 4, 9707)
    assert tuple(row["token_ids"] for row in rows) == (
        (9707, 9707, 9707, 9707),
        (9707, 9707, 9707, 9708),
        (9707, 9707, 9707, 9709),
        (9707, 9707, 9707, 9710),
        (9707, 9707, 9707, 9707),
    )
    assert all(row["roundtrip_exact"] for row in rows)
    assert rows[1]["text"] == "9707,9707,9707,9708"


def test_stats_and_latency_delta_are_recomputed_from_new_samples() -> None:
    summary = SCRIPT._stats([10.0, 12.0, 11.0])
    assert summary["median"] == 11.0
    assert summary["p95"] == 12.0
    assert summary["stdev"] == pytest.approx(1.0)
    before = {"queue": {"samples": [1.0]}, "service": {"samples": []}}
    after = {"queue": {"samples": [1.0, 2.0, 3.0]}, "service": {"samples": [4.0]}}
    assert SCRIPT._latency_delta(before, after) == {
        "queue": [2.0, 3.0],
        "service": [4.0],
    }


def test_sse_parser_accepts_payload_and_done() -> None:
    assert SCRIPT._parse_sse_line('data: {"choices": []}') == {"choices": []}
    assert SCRIPT._parse_sse_line(b"data: [DONE]") == "[DONE]"
    assert SCRIPT._parse_sse_line("event: ignored") is None


def test_counter_delta_includes_new_and_removed_keys() -> None:
    assert SCRIPT._counter_delta({"a": 2, "b": 1}, {"a": 5, "c": 4}) == {
        "a": 3,
        "b": -1,
        "c": 4,
    }
