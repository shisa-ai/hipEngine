"""Q8T16 comparison choices belong to independent execution contexts."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hipengine.runtime import gguf_linear as gl


@pytest.mark.parametrize("control", ["floor", "all"])
def test_q8t16_independent_execution_contexts(monkeypatch, control):
    for name in ("HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE", "HIPENGINE_GGUF_Q8_T16_PAIR_COL8", "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"):
        monkeypatch.delenv(name, raising=False)
    entered = Barrier(2)
    observed = Barrier(2)

    def run(choice):
        scope = (gl.q8_t16_pair_rowtile_min_rows_session(choice) if control == "floor"
                 else gl.q8_t16_rowtile_all_session(choice))
        with scope:
            entered.wait(timeout=5)
            try:
                actual = (gl._resolve_use_q8_t16_pair_col8(rows=8) if control == "floor"
                          else gl._resolve_use_q8_t16_all_rowtile())
                return actual
            finally:
                observed.wait(timeout=5)

    choices = (8, 0) if control == "floor" else (True, False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, choice) for choice in choices]
        assert [f.result(timeout=10) for f in futures] == [True, False]
