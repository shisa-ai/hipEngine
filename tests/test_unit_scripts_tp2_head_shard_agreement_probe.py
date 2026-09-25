"""Unit tier: the head-shard probe's reduction check.

The probe exists to say whether the head-sharded prefill's error belongs to the
split or to the reduction, and the reduction half of that answer is only worth
anything if it compares the consumed row against a host sum of the partials the
ranks actually wrote. Rank agreement cannot stand in for that: two ranks reading
the same wrong slot, or a doubled partial, agree with each other too. These
tests drive the check with synthetic dumps, including exactly that
agree-but-wrong case.

No device contact: the dumps are ``.npy`` files written to a temp directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tp2_head_shard_agreement_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp2_head_shard_agreement_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load()


def _write(path: Path, values: np.ndarray) -> np.ndarray:
    bits = probe._narrow_bf16_rne(values)
    np.save(path, bits)
    return bits


def _case(tmp_path: Path, *, reduced: str) -> dict:
    """Write two partials plus a reduced row and run the probe's check."""

    rng = np.random.default_rng(11)
    p0 = rng.standard_normal(256).astype("<f4")
    p1 = (rng.standard_normal(256) * 0.5).astype("<f4")
    _write(tmp_path / "sharded_partial_l3_d0.npy", p0)
    _write(tmp_path / "sharded_partial_l3_d1.npy", p1)
    total = probe._widen_bf16(np.load(tmp_path / "sharded_partial_l3_d0.npy")) + probe._widen_bf16(
        np.load(tmp_path / "sharded_partial_l3_d1.npy")
    )
    if reduced == "host-sum":
        row = probe._narrow_bf16_rne(total)
    elif reduced == "doubled-partial":
        # Both ranks consume rank 0's partial twice: they agree with each other
        # and the row is still wrong.
        row = probe._narrow_bf16_rne(
            2.0 * probe._widen_bf16(np.load(tmp_path / "sharded_partial_l3_d0.npy"))
        )
    elif reduced == "stale-slot":
        row = probe._narrow_bf16_rne(total * 0.25)
    else:  # pragma: no cover - guard against a typo in a caller
        raise AssertionError(reduced)
    np.save(tmp_path / "sharded_l3_d0.npy", row)
    np.save(tmp_path / "sharded_l3_d1.npy", row)
    report: dict = {}
    probe._reduction_check(tmp_path, 3, report)
    return report["reduction"]["3"]


def test_unit_reduction_check_accepts_the_host_sum_of_both_partials(tmp_path: Path) -> None:
    stats = _case(tmp_path, reduced="host-sum")
    assert stats["rank0_exact"] is True
    assert stats["rank1_exact"] is True
    assert stats["rank0_mismatched_elems"] == 0


def test_unit_reduction_check_rejects_rows_both_ranks_agree_on(tmp_path: Path) -> None:
    """The whole point of the check: agreement is not correctness."""

    for reduced in ("doubled-partial", "stale-slot"):
        stats = _case(tmp_path, reduced=reduced)
        assert stats["rank0_exact"] is False, reduced
        assert stats["rank1_exact"] is False, reduced
        assert stats["rank0_mismatched_elems"] > 0
        assert stats["rank0_rel_rms"] > 0.0
        # Both ranks hold the same row, so a peer-to-peer comparison would have
        # reported agreement on exactly this input.
        assert np.array_equal(
            np.load(tmp_path / "sharded_l3_d0.npy"), np.load(tmp_path / "sharded_l3_d1.npy")
        )


def test_unit_reduction_check_needs_both_partials_and_both_rows(tmp_path: Path) -> None:
    report: dict = {}
    probe._reduction_check(tmp_path, 3, report)
    assert "reduction" not in report


def test_unit_narrow_bf16_rne_rounds_half_to_even() -> None:
    """The narrowing must match the boundary cast the exchange narrows with."""

    # 0x3F808000 is exactly halfway between two bf16 values: the even one wins.
    halfway = np.array([0x3F808000, 0x3F818000], dtype="<u4").view("<f4")
    assert probe._narrow_bf16_rne(halfway).tolist() == [0x3F80, 0x3F82]
    values = np.array([1.0, -2.5, 0.0], dtype="<f4")
    assert probe._narrow_bf16_rne(values).tolist() == [0x3F80, 0xC020, 0x0000]


def test_unit_compare_writes_its_report_into_a_directory_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """A compare on its own is still worth the report, and it must not crash.

    The capture commands are separate invocations, so a compare can be the first
    thing to touch the output directory. Writing the report is the last step of a
    comparison whose printed results have already been read, so a missing parent
    there loses the machine-readable half of an expensive run.
    """

    fresh = tmp_path / "not-created-yet"
    assert probe._compare(fresh, (0,)) == 0
    assert (fresh / "compare.json").exists()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
