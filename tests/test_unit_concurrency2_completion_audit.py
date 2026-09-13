from __future__ import annotations

from pathlib import Path

from scripts import concurrency2_completion_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_completion_audit_has_evidence_for_every_requirement() -> None:
    payload = audit.run(REPO_ROOT)

    assert payload["passed"] is True
    assert payload["goal_complete"] is False
    assert payload["missing_evidence"] == []
    assert payload["false_passes"] == []
    assert payload["status_counts"] == {
        "passed": 38,
        "blocked": 2,
        "unavailable": 1,
    }
    rows = payload["requirements"]
    assert len({row["requirement_id"] for row in rows}) == len(rows)
    assert all(row["evidence_complete"] for row in rows)


def test_completion_audit_names_only_real_product_blockers() -> None:
    payload = audit.run(REPO_ROOT)
    blockers = {
        row["requirement_id"]: row["status"]
        for row in payload["blockers"]
    }

    assert blockers == {
        "C2-6.external": "unavailable",
        "DMS.product": "blocked",
        "C2-S.C5": "blocked",
    }
    external = next(
        row for row in payload["blockers"]
        if row["requirement_id"] == "C2-6.external"
    )
    assert "Vulkan binary has no usable GPU backend" in external["blocker"]


def test_completion_document_closes_load_and_keeps_remaining_product_boxes_open() -> None:
    document = (REPO_ROOT / "docs/CONCURRENCY2.md").read_text(encoding="utf-8")

    assert "- [x] Qualify 4K/16K/32K" in document
    assert "- [x] Run fixed, ragged, burst, Poisson" in document
    assert "- [ ] Compare matched same-model/quant/hardware" in document
    assert "- [x] Port streaming no-shadow prefill pack" in document
    assert "The clean canonical packet is **accepted**" in document
    assert "Host/backend status:" in document
    assert "Implemented host status:" in document
