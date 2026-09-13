from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.qwen4_exp_materialize import Qwen4ExpPLEMMapTable
from scripts.qwen4exp_ple_gather_screen import PreadGather, gather_sorted_unique, measure, pread_exact
from tests._qwen4_exp_ple_fixtures import _iq4_nl_rows, _ple_tensor


@pytest.mark.parametrize("indices", [[], [2, 0, 2, 1], [0], [2, 1, 0]])
@pytest.mark.parametrize("method", ["sorted_unique", "copy_elision", "dedup_elision", "sampled_dedup_elision"])
def test_candidate_preserves_exact_order_and_independent_output(indices, method):
    raw = _iq4_nl_rows((1., 2., 3.))
    table = Qwen4ExpPLEMMapTable(SimpleNamespace(tensor_data=lambda _: raw), _ple_tensor(3), semantic_rows=3)
    expected = table.gather_rows(indices) if indices else np.empty((0, 160), dtype=np.float32)
    result = gather_sorted_unique(table, indices, method=method)
    assert result.tobytes() == expected.tobytes()
    assert result.dtype == np.float32
    assert result.shape == expected.shape
    assert not np.shares_memory(result, expected)
    if indices:
        result[0] = 0
        assert gather_sorted_unique(table, indices, method=method).tobytes() == expected.tobytes()
    table.close()
    with pytest.raises(RuntimeError):
        gather_sorted_unique(table, indices)


@pytest.mark.parametrize("indices,error", [([-1], IndexError), ([3], IndexError), ([[0]], ValueError)])
def test_candidate_rejects_invalid_rows(indices, error):
    raw = _iq4_nl_rows((1., 2., 3.))
    table = Qwen4ExpPLEMMapTable(SimpleNamespace(tensor_data=lambda _: raw), _ple_tensor(3), semantic_rows=3)
    with pytest.raises(error):
        gather_sorted_unique(table, indices)


@pytest.mark.parametrize("indices", [
    np.arange(1024)[::-1],
    np.tile(np.arange(32), 32),
    np.repeat(np.arange(64), 16),
])
def test_large_candidate_branches_match_reference(indices):
    raw = _iq4_nl_rows(tuple(np.linspace(0.01, 1., 1024)))
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda _: raw), _ple_tensor(1024), semantic_rows=1024,
    )
    expected = table.gather_rows(indices)
    for method in ("copy_elision", "dedup_elision", "sampled_dedup_elision"):
        actual = gather_sorted_unique(table, indices, method=method)
        assert actual.tobytes() == expected.tobytes()
        assert not np.shares_memory(actual, raw)


def test_pread_retries_interruptions_and_short_reads():
    events = iter([InterruptedError(), b"ab", b"c"])
    calls = []

    def read(fd, size, offset):
        calls.append((fd, size, offset))
        event = next(events)
        if isinstance(event, Exception):
            raise event
        return event

    assert pread_exact(8, 3, 100, read=read) == b"abc"
    assert calls == [(8, 3, 100), (8, 3, 100), (8, 1, 102)]
    with pytest.raises(EOFError):
        pread_exact(8, 3, 100, read=lambda *args: b"")


@pytest.mark.parametrize("workers", [0, 2])
def test_pread_scatter_offsets_tail_and_teardown(tmp_path, workers):
    from dataclasses import replace

    raw = _iq4_nl_rows((1., 2., 3.))
    path = tmp_path / "rows"
    path.write_bytes(b"x" * 17 + raw.tobytes())
    tensor = replace(_ple_tensor(3), data_offset=17)
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(path=path, tensor_data=lambda _: raw), tensor, semantic_rows=3,
    )
    reader = PreadGather(table, workers)
    try:
        ids = [2, 0, 2, 1]
        assert reader.gather(ids).tobytes() == table.gather_rows(ids).tobytes()
        assert reader.gather([]).shape == (0, 160)
        with pytest.raises(IndexError):
            reader.gather([3])
    finally:
        reader.close()
        reader.close()
    with pytest.raises(RuntimeError):
        reader.gather([0])


def test_mapping_only_advice_preserves_values_and_records_io(tmp_path):
    from dataclasses import replace

    raw = _iq4_nl_rows((1., 2., 3.))
    path = tmp_path / "mapped-rows"
    path.write_bytes(b"x" * 17 + raw.tobytes())
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(path=path, tensor_data=lambda _: np.memmap(
            path, dtype=np.uint8, mode="r", offset=17, shape=(3, 90),
        )),
        replace(_ple_tensor(3), data_offset=17), semantic_rows=3,
    )
    try:
        result = measure(table, np.array([2, 0, 2]), cache_mode="cold",
                         repetitions=2, method="mmap_random")
        assert result["bit_exact"]
        assert len(result["samples"]) == 4
        assert all(row["process_read_bytes"] >= 0 for row in result["samples"])
    finally:
        table.close()
