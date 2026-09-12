import math
import pytest
from scripts.qwen4exp_screen_calibration import calibrate, observations, HOST


def packet():
    return dict(host=dict(machine_id=HOST),protocol=dict(prefill_chunk_size=1024),
        samples=[dict(case_id=f"case{i}",mode=("before","after","after","before","before","after")[slot],
            sequence_slot=slot,repetition=slot//2,prefill_ms=10,decode_ms=10,
            output_token_ids_sha256="same") for i in range(12) for slot in range(6)])


def test_balanced_identity():
    result=observations(packet())
    assert len(result)==24
    assert all(r["first_ratio"]==r["aggregate_ratio"]==1 for r in result)


@pytest.mark.parametrize("change",["host","chunk","count","nan","trajectory"])
def test_bad_packet_rejected(change):
    p=packet()
    if change=="host": p["host"]["machine_id"]="elsewhere"
    if change=="chunk": p["protocol"]["prefill_chunk_size"]=512
    if change=="count": p["samples"].pop()
    if change=="nan": p["samples"][0]["prefill_ms"]=float("nan")
    if change=="trajectory": p["samples"][0]["output_token_ids_sha256"]="different"
    with pytest.raises(ValueError):
        observations(p)


def test_heldout_cannot_inflate_training_bound():
    training=[dict(metric=m,log_error=.01) for m in ("prefill","request")]
    heldout=[dict(case_id="x",metric=m,log_error=.1,first_ratio=1.05,
                  aggregate_ratio=1.05/math.exp(.1)) for m in ("prefill","request")]
    result=calibrate([training],[heldout])
    assert result["log_error_envelope"]["prefill"]==.01
    assert result["heldout"][0]["violations"]==2
    assert result["heldout"][0]["wrong_signs"]==2
