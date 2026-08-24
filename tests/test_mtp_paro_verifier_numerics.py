from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mtp_paro_verifier_numerics
from scripts.mtp_paro_verifier_numerics import _summary


def test_verifier_numerical_summary_reports_binding_tail_and_top1() -> None:
    result = _summary(
        np.asarray([0.0, 1.0e-3, 2.0e-2, 5.0e-2], dtype=np.float64),
        np.asarray([True, True, False, True], dtype=np.bool_),
    )

    assert result["rows"] == 4
    assert result["mean_kl"] == pytest.approx(0.01775)
    assert result["max_kl"] == 0.05
    assert result["top1_agreement"] == 0.75
    assert result["p99_kl"] > result["p95_kl"]


def test_verifier_capture_reads_materialized_verify_logits(monkeypatch) -> None:
    copied: list[tuple[int, int]] = []

    def fake_copy(_host_ptr, buffer, nbytes, *, runtime):
        del runtime
        copied.append((int(buffer.ptr), int(nbytes)))

    monkeypatch.setattr(mtp_paro_verifier_numerics, "copy_device_to_host", fake_copy)
    session = SimpleNamespace(
        vocab_size=4,
        verify_lm_logits=SimpleNamespace(ptr=0xABC000),
        batch_lm_logits=SimpleNamespace(ptr=0xBAD000),
        runtime=object(),
    )
    logits = mtp_paro_verifier_numerics._copy_logits(session, rows=2)

    assert logits.shape == (2, 4)
    assert copied == [(0xABC000, 2 * 4 * 4)]
