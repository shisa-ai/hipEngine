from __future__ import annotations

from scripts.laguna_gguf_load_smoke import (
    _observed_cache_state,
    _process_delta,
)


def test_laguna_load_profile_classifies_observed_physical_reads() -> None:
    assert _observed_cache_state(95, 100) == "cold_streamed"
    assert _observed_cache_state(5, 100) == "warm_cached"
    assert _observed_cache_state(50, 100) == "partially_cached"
    assert _observed_cache_state(None, 100) == "unknown"
    assert _observed_cache_state(0, 0) == "unknown"


def test_laguna_load_profile_process_delta_preserves_missing_io_fields() -> None:
    before = {
        "minor_faults": 10,
        "major_faults": 2,
        "io": {"read_bytes": 100, "syscr": 4},
    }
    after = {
        "minor_faults": 17,
        "major_faults": 3,
        "io": {"read_bytes": 160, "write_bytes": 9},
    }

    assert _process_delta(before, after) == {
        "minor_faults": 7,
        "major_faults": 1,
        "io": {
            "read_bytes": 60,
            "syscr": None,
            "write_bytes": None,
        },
    }
