"""CPU-only tests for scripts/tp_break_even.py group model and guards.

The model under test is one *synchronized* TP2 group time:

    T2 = max_rank(fixed) + max_rank(rank weights) + collective

compared against the faster matched TP1 arm. The earlier version projected a
complete TP2 time per rank and compared each against its own TP1 row, which
counted one group as two independent results and let the faster arm's projection
ignore the slower rank it would wait for.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp_break_even.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp_break_even_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


VALID_KERNEL_JSON = {
    "kind": "shard-kernel-smoke",
    "kernels": [
        {"Kernel_Name": "gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out", "DurationNs": 971796639},
        {"Kernel_Name": "gguf_q6_k_t16_qmicro_planar_gemm_bf16_out", "DurationNs": 16331988},
    ],
}


@pytest.fixture
def evidence_root(tmp_path, monkeypatch):
    """A benchmarks/results/ tree whose cwd is the tmp repo root."""

    root = tmp_path / "benchmarks" / "results"
    root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return root


def _write_kernel_artifact(root, name="shard-kernel-smoke.json", payload=VALID_KERNEL_JSON):
    path = root / name
    path.write_text(json.dumps(payload) if isinstance(payload, dict) else payload)
    return str(path)


@pytest.fixture
def evidence_path(evidence_root):
    return _write_kernel_artifact(evidence_root)


# -- baseline parsing ---------------------------------------------------------


def test_parse_tp1_accepts_named_quadruples(mod) -> None:
    device = mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")
    assert device == {
        "name": "W7900",
        "tok_s": 27.9,
        "tp1_gib": 15.652,
        "rank_gib": 8.646,
        "protocol": "512/128/int8-kv",
    }


@pytest.mark.parametrize(
    "spec",
    [
        "W7900",  # no values
        "W7900=27.9:15.652",  # missing rank shard and protocol
        "W7900=27.9:15.652:8.646",  # protocol omitted: arms cannot be matched
        "W7900=27.9:15.652:8.646:",  # empty protocol
        "=27.9:15.652:8.646:512",  # no device name
        "W7900=0:15.652:8.646:512",  # non-positive rate
        "W7900=27.9:15.652:20.0:512",  # shard larger than the model
    ],
)
def test_parse_tp1_rejects_malformed_specs(mod, spec: str) -> None:
    with pytest.raises(ValueError):
        mod.parse_tp1(spec)


# -- one rank's contribution --------------------------------------------------


def test_rank_projection_reproduces_the_hand_computed_row(mod) -> None:
    device = mod.parse_tp1("W7900=25.0:16.0:8.0:512/128/int8-kv")
    row = mod.rank_projection(device, fixed_share=0.0)
    assert row["tp1_ms_per_token"] == pytest.approx(40.0)
    # 16 GiB in 40 ms is the bandwidth the TP1 row implies.
    assert row["implied_bandwidth_gbs"] == pytest.approx(400.0)
    assert row["rank_weight_ms_per_token"] == pytest.approx(20.0)
    assert row["fixed_ms_per_token"] == pytest.approx(0.0)


def test_implied_bandwidth_divides_by_the_weight_share(mod) -> None:
    """``bandwidth = weights / ((1 - fixed_share) * T1)``.

    The share divides. Multiplying it understated the bandwidth (and therefore
    overstated the rank-weight time) by the square of the share.
    """

    device = mod.parse_tp1("W7900=25.0:16.0:8.0:512/128/int8-kv")
    row = mod.rank_projection(device, fixed_share=0.25)
    # 16 GiB over 75% of a 40 ms token.
    assert row["implied_bandwidth_gbs"] == pytest.approx(16.0 / (0.75 * 0.040))
    assert row["implied_bandwidth_gbs"] == pytest.approx(533.3333, rel=1e-4)
    assert row["rank_weight_ms_per_token"] == pytest.approx(8.0 / 533.3333 * 1000.0)
    assert row["fixed_ms_per_token"] == pytest.approx(10.0)


# -- the synchronized group ---------------------------------------------------


def test_project_group_takes_maxima_and_the_faster_baseline(mod) -> None:
    """One group waits for its slowest rank and is one result, not two.

    W7900 is the slower arm but has the larger shard; XTX is the faster arm and
    the baseline. The group pays the W7900 shard and is compared against the XTX
    token time.
    """

    devices = [
        mod.parse_tp1("W7900=25.0:16.0:8.0:512/128/int8-kv"),
        mod.parse_tp1("XTX=40.0:16.0:7.0:512/128/int8-kv"),
    ]
    row = mod.project_group(devices, collective_ms=1.0, fixed_share=0.0)
    assert row["limiting_rank_weight"] == "W7900"
    assert row["baseline_device"] == "XTX"
    assert row["baseline_tp1_ms_per_token"] == pytest.approx(25.0)
    assert row["rank_weight_group_ms_per_token"] == pytest.approx(20.0)
    assert row["tp2_group_ms_per_token"] == pytest.approx(21.0)
    assert row["projected_speedup"] == pytest.approx(25.0 / 21.0)
    assert row["break_even_collective_ms"] == pytest.approx(5.0)
    assert [entry["device"] for entry in row["per_rank"]] == ["W7900", "XTX"]


def test_project_group_hits_break_even_exactly_at_the_budget(mod) -> None:
    devices = [
        mod.parse_tp1("W7900=25.0:16.0:8.0:512/128/int8-kv"),
        mod.parse_tp1("XTX=40.0:16.0:7.0:512/128/int8-kv"),
    ]
    budget = mod.project_group(devices, collective_ms=1.0, fixed_share=0.25)[
        "break_even_collective_ms"
    ]
    row = mod.project_group(devices, collective_ms=budget, fixed_share=0.25)
    assert row["projected_speedup"] == pytest.approx(1.0, abs=1e-9)


def test_project_group_moves_the_right_way(mod) -> None:
    devices = [mod.parse_tp1("W7900=25.0:16.0:8.0:512/128/int8-kv")]
    cheap = mod.project_group(devices, collective_ms=1.0, fixed_share=0.0)
    expensive = mod.project_group(devices, collective_ms=8.0, fixed_share=0.0)
    assert cheap["projected_speedup"] > expensive["projected_speedup"]
    more_fixed = mod.project_group(devices, collective_ms=1.0, fixed_share=0.4)
    assert more_fixed["projected_speedup"] < cheap["projected_speedup"]
    assert more_fixed["break_even_collective_ms"] < cheap["break_even_collective_ms"]
    assert cheap["required_improvement_factor"] == pytest.approx(1.0 / 20.0)


# -- measured collective input ------------------------------------------------


def _chain_artifact(path: pathlib.Path, *, marginal: float = 177.8, depends: bool = True) -> pathlib.Path:
    payload = {
        "collective": {
            "cases": {
                "all_reduce:rows1:fp32": {
                    "dependent_chain": {
                        "payload_bytes": 20480,
                        "world_size": 2,
                        "modes": {
                            "per_step": {
                                "group_boundary": "one native group per reduction",
                                "depends_on_every_step": depends,
                                "marginal": {"overall_us_per_step": marginal},
                            },
                            "per_chain": {
                                "group_boundary": "one native group for the whole chain",
                                "depends_on_every_step": False,
                                "marginal": {"overall_us_per_step": 36.1},
                            },
                        },
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dependent_chain_marginal_is_read_from_the_artifact(tmp_path: pathlib.Path, mod) -> None:
    source = _chain_artifact(tmp_path / "chain.json", marginal=177.8)
    record = mod.read_dependent_chain_marginal(source, case_key="all_reduce:rows1:fp32")
    assert record["marginal_us_per_step"] == pytest.approx(177.8)
    assert record["mode"] == "per_step"
    assert record["depends_on_every_step"] is True
    assert record["payload_bytes"] == 20480
    assert record["source"] == str(source)


def test_dependent_chain_marginal_rejects_a_failed_dependency_check(
    tmp_path: pathlib.Path, mod
) -> None:
    """A marginal whose chain did not consume its predecessor is not a cost."""

    source = _chain_artifact(tmp_path / "chain.json", depends=False)
    with pytest.raises(ValueError, match="dependency check"):
        mod.read_dependent_chain_marginal(source, case_key="all_reduce:rows1:fp32")


def test_dependent_chain_marginal_rejects_an_unknown_case(tmp_path: pathlib.Path, mod) -> None:
    source = _chain_artifact(tmp_path / "chain.json")
    with pytest.raises(ValueError, match="no case"):
        mod.read_dependent_chain_marginal(source, case_key="all_reduce:rows99:fp32")


def _native_ab_artifact(
    path: pathlib.Path,
    *,
    marginal: float = 20.5,
    bounded_exact: bool = True,
    sum_exact: bool = True,
    sum_informative: bool = True,
    saturation_expected: bool = False,
    provisional: bool = False,
) -> pathlib.Path:
    def verification() -> dict[str, object]:
        return {
            "bounded": {"exact": bounded_exact, "finite": True},
            "sum": {
                "informative": sum_informative,
                "exact": sum_exact,
                "finite": sum_informative,
                "saturation_expected": saturation_expected,
            },
        }

    payload = {
        "kind": "tp2-staged-exchange-native-ab",
        "payload_bytes": 20480,
        "protocol": "batched: both D2H before either wait, no return wait",
        "arms": {
            "native": {
                "marginal": {"overall_us_per_step": marginal},
                # Depth 128 is the saturating case: the timed recurrence's closed
                # form overflows fp32 there, which is reported rather than passed.
                "depths": {
                    "1": {"verification": verification()},
                    "128": {
                        "verification": {
                            "bounded": {"exact": bounded_exact, "finite": True},
                            "sum": {
                                "informative": sum_informative,
                                "exact": sum_exact,
                                "finite": sum_informative,
                                "saturation_expected": saturation_expected,
                            },
                        }
                    },
                },
            }
        },
        "provenance_match": {"balanced_repetitions": not provisional},
        "provenance": {"git_commit": "deadbeef"},
        "comparison": {
            "python_us_per_step": 40.0,
            "python_over_native": 40.0 / marginal,
            "provisional": provisional,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_native_ab_marginal_is_read_from_its_artifact(tmp_path: pathlib.Path, mod) -> None:
    source = _native_ab_artifact(tmp_path / "native.json", marginal=20.5)
    record = mod.read_native_ab_marginal(source)
    assert record["marginal_us_per_step"] == pytest.approx(20.5)
    assert record["depends_on_every_step"] is True
    assert record["python_arm_us_per_step"] == pytest.approx(40.0)
    assert record["payload_bytes"] == 20480
    assert record["comparison_provisional"] is False
    assert record["git_commit"] == "deadbeef"


def test_native_ab_marginal_requires_every_depth_to_verify(
    tmp_path: pathlib.Path, mod
) -> None:
    """An unverified depth means the chain may not have compounded."""

    source = _native_ab_artifact(tmp_path / "native.json", bounded_exact=False)
    with pytest.raises(ValueError, match="did not verify"):
        mod.read_native_ab_marginal(source)


def test_native_ab_marginal_rejects_a_failed_timed_recurrence(
    tmp_path: pathlib.Path, mod
) -> None:
    """The bounded check passing is not enough: the timed path must hold too."""

    source = _native_ab_artifact(
        tmp_path / "native.json", sum_exact=False, saturation_expected=False
    )
    with pytest.raises(ValueError, match="did not verify"):
        mod.read_native_ab_marginal(source)


def test_native_ab_marginal_accepts_an_honest_saturation(tmp_path: pathlib.Path, mod) -> None:
    """A depth whose closed form overflows fp32 is reported, not failed."""

    source = _native_ab_artifact(
        tmp_path / "native.json",
        sum_informative=False,
        sum_exact=False,
        saturation_expected=True,
    )
    record = mod.read_native_ab_marginal(source)
    assert record["marginal_us_per_step"] == pytest.approx(20.5)


def test_native_ab_marginal_carries_the_provisional_flag(tmp_path: pathlib.Path, mod) -> None:
    """An unmatched comparison must not be read as a matched one."""

    source = _native_ab_artifact(tmp_path / "native.json", provisional=True)
    record = mod.read_native_ab_marginal(source)
    assert record["comparison_provisional"] is True


def test_native_ab_marginal_rejects_another_artifact_kind(
    tmp_path: pathlib.Path, mod
) -> None:
    source = _chain_artifact(tmp_path / "chain.json")
    with pytest.raises(ValueError, match="not a native A/B artifact"):
        mod.read_native_ab_marginal(source)


def test_native_mode_is_not_readable_from_the_chain_artifact(mod) -> None:
    """The native marginal lives in the A/B artifact, not the chain artifact."""

    assert "native_staged_exchange_batched" in mod.DEPENDENT_CHAIN_MODES
    assert "native_staged_exchange_batched" in mod.NATIVE_AB_MODES


def test_native_ab_selects_the_native_arm(tmp_path: pathlib.Path, mod) -> None:
    source = _native_ab_artifact(tmp_path / "native.json", marginal=20.8)
    record = mod.read_native_ab_marginal(source)
    assert record["mode"] == "native_staged_exchange_batched"
    assert record["world_size"] == 2


def test_per_chain_mode_is_not_selectable(mod) -> None:
    """Only the per-step structure can carry a layer dependency."""

    with pytest.raises(SystemExit):
        mod.main(
            [
                "--tp1",
                "W7900=27.9:15.652:8.646:512/128/int8-kv",
                "--marginal-us",
                "30",
                "--dependent-chain-mode",
                "per_chain",
            ]
        )


# -- report guards ------------------------------------------------------------


def test_build_report_withholds_certification_on_mismatched_protocols(mod) -> None:
    """A 512-prompt INT8-KV arm and an 8192-token BF16 arm are not a pair."""

    devices = [
        mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv"),
        mod.parse_tp1("XTX=29.82:15.652:7.009:8192/8/bf16-kv"),
    ]
    report = mod.build_report(devices, collective_ms=(1.0,), fixed_shares=(0.0,))
    assert report["protocol_match"]["matched"] is False
    assert report["verdict"]["certified"] is False
    assert any("different protocols" in reason for reason in report["verdict"]["withheld_reasons"])


def test_build_report_withholds_certification_without_the_gate(mod, evidence_path, evidence_root) -> None:
    devices = [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")]
    report = mod.build_report(devices, collective_ms=(1.0,), fixed_shares=(0.0,))
    assert report["pre_packet3_gate"]["satisfied"] is False
    assert report["verdict"]["certified"] is False
    assert any("shard-kernel gate" in reason for reason in report["verdict"]["withheld_reasons"])

    with_evidence = mod.build_report(
        devices,
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[evidence_path],
        results_root=evidence_root,
    )
    assert with_evidence["pre_packet3_gate"]["satisfied"] is True
    assert with_evidence["verdict"]["certified"] is True
    assert with_evidence["verdict"]["passes_target_in_every_row"] is True


def test_build_report_records_the_collective_source(mod) -> None:
    devices = [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")]
    report = mod.build_report(
        devices,
        collective_ms=(22.758,),
        fixed_shares=(0.0,),
        reduction_points=128,
        collective_source={
            "marginal_us_per_step": 177.8,
            "source": "benchmarks/results/chain.json",
            "mode": "per_step",
            "depends_on_every_step": True,
        },
    )
    assert report["collective_source"]["source"] == "benchmarks/results/chain.json"
    assert report["collective_ms_expected_from_count"] == pytest.approx(22.758)
    assert report["errors"] == []


def test_build_report_flags_a_budget_that_contradicts_the_count(mod) -> None:
    """A budget inconsistent with the reduction count is an error, not a pass."""

    devices = [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")]
    source = {
        "marginal_us_per_step": 177.8,
        "source": "benchmarks/results/chain.json",
        "mode": "per_step",
        "depends_on_every_step": True,
    }
    # 3.66 ms is 128 reductions at the superseded 28.6 us marginal.
    wrong = mod.build_report(
        devices,
        collective_ms=(3.66,),
        reduction_points=128,
        collective_source=source,
    )
    assert wrong["errors"], "a budget built from the superseded marginal was accepted"
    assert "does not match" in wrong["errors"][0]

    right = mod.build_report(
        devices,
        collective_ms=(22.758,),
        reduction_points=128,
        collective_source=source,
    )
    assert right["errors"] == []


def test_build_report_reports_a_losing_projection(mod) -> None:
    """The measured dependent cost at the decode shape loses on the eager path."""

    devices = [
        mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv"),
        mod.parse_tp1("XTX=29.82:15.652:7.009:512/128/int8-kv"),
    ]
    report = mod.build_report(
        devices,
        collective_ms=(22.758,),
        fixed_shares=(0.0,),
        reduction_points=128,
        collective_source={
            "marginal_us_per_step": 177.8,
            "source": "benchmarks/results/chain.json",
            "mode": "per_step",
            "depends_on_every_step": True,
        },
    )
    assert report["verdict"]["passes_target_in_every_row"] is False
    assert report["verdict"]["certified"] is False
    row = report["rows"][0]
    assert row["projected_speedup"] < 1.0
    # A row at or below 1.0x is a genuine blocker, not an unmet aspiration.
    assert report["verdict"]["beats_faster_tp1_arm"] is False
    assert any("would not beat the faster TP1 arm" in r for r in report["verdict"]["withheld_reasons"])
    assert row["required_improvement_factor"] > 1.0
    assert json.loads(json.dumps(report))["kind"] == "tp2_break_even"


def test_build_report_is_json_serializable_across_the_share_range(mod, evidence_path, evidence_root) -> None:
    devices = [
        mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv"),
        mod.parse_tp1("XTX=29.82:15.652:7.009:512/128/int8-kv"),
    ]
    report = mod.build_report(
        devices,
        collective_ms=(1.3, 1.5),
        fixed_shares=(0.0, 0.2),
        shard_kernel_evidence=[evidence_path],
        results_root=evidence_root,
    )
    assert len(report["rows"]) == 4
    assert report["verdict"]["worst_case_speedup"] <= report["verdict"]["best_case_speedup"]
    assert json.loads(json.dumps(report))["verdict"]["certified"] is True


def test_build_report_pins_the_share_sensitivity_of_the_matched_pair(mod, evidence_path, evidence_root) -> None:
    devices = [
        mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv"),
        mod.parse_tp1("XTX=29.82:15.652:7.009:512/128/int8-kv"),
    ]
    evidence = [evidence_path]
    passing = mod.build_report(
        devices,
        collective_ms=(1.3,),
        fixed_shares=(0.0, 0.2),
        shard_kernel_evidence=evidence,
        results_root=evidence_root,
    )
    assert passing["verdict"]["passes_target_in_every_row"] is True
    dipping = mod.build_report(
        devices,
        collective_ms=(1.3,),
        fixed_shares=(0.3,),
        shard_kernel_evidence=evidence,
        results_root=evidence_root,
    )
    assert dipping["verdict"]["passes_target_in_every_row"] is False
    assert dipping["rows"][0]["projected_speedup"] == pytest.approx(1.294, abs=0.005)
    # Missing the aspiration is recorded, never withheld, and does not stop the
    # result being certified: the design accepts any qualified net improvement.
    assert dipping["verdict"]["beats_faster_tp1_arm"] is True
    assert dipping["verdict"]["meets_planning_aspiration"] is False
    assert dipping["verdict"]["certified"] is True
    assert dipping["verdict"]["withheld_reasons"] == []
    assert any("aspiration" in note for note in dipping["verdict"]["aspiration_notes"])
    assert not any(
        "target" in reason for reason in dipping["verdict"]["withheld_reasons"]
    )


def test_the_two_thresholds_are_reported_separately(mod, evidence_path, evidence_root) -> None:
    """A 1.1x-class result is a qualified win, not a failure."""

    devices = [
        mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv"),
        mod.parse_tp1("XTX=29.82:15.652:7.009:512/128/int8-kv"),
    ]
    devices = [
        mod.parse_tp1("W7900=29.575:15.652:8.646:512-128-int8kv-eager"),
        mod.parse_tp1("XTX=35.355:15.652:7.009:512-128-int8kv-eager"),
    ]
    report = mod.build_report(
        devices,
        collective_ms=(3.2,),
        fixed_shares=(0.0, 0.1, 0.2, 0.3),
        shard_kernel_evidence=[evidence_path],
        results_root=evidence_root,
    )
    verdict = report["verdict"]
    speedups = [row["projected_speedup"] for row in report["rows"]]
    assert max(speedups) < mod.TARGET_SPEEDUP, "this fixture must miss the aspiration"
    assert min(speedups) > mod.BEATS_TP1_SPEEDUP, "and clear the gate"
    assert verdict["beats_faster_tp1_arm"] is True
    assert verdict["meets_planning_aspiration"] is False
    # The gate decides certification; the aspiration only annotates it.
    assert verdict["certified"] is True
    assert verdict["withheld_reasons"] == []
    assert len(verdict["aspiration_notes"]) == 1
    # The legacy field still means the aspiration, so older readers are not
    # silently re-pointed at the gate.
    assert verdict["passes_target_in_every_row"] is verdict["meets_planning_aspiration"]
    assert verdict["aspiration_target_speedup"] == mod.TARGET_SPEEDUP


def test_an_unreachable_gate_is_withheld_but_an_unreachable_aspiration_is_not(mod) -> None:
    """A bound below 1.0x is structural; a bound below 1.3x only caps the win."""

    devices = [
        mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv"),
        mod.parse_tp1("XTX=29.82:15.652:7.009:512/128/int8-kv"),
    ]
    # A large rank-weight share with no shard kernel benefit: the optimistic
    # bound (free rank-weight reads) cannot reach the gate.
    devices = [
        mod.parse_tp1("W7900=29.575:15.652:8.646:512-128-int8kv-eager"),
        mod.parse_tp1("XTX=35.355:15.652:7.009:512-128-int8kv-eager"),
    ]
    # 30 ms of collective exceeds the faster arm's whole 28.28 ms token time, so
    # even with free rank-weight reads the group cannot beat it.
    report = mod.build_report(
        devices,
        collective_ms=(30.0,),
        fixed_shares=(0.0,),
    )
    verdict = report["verdict"]
    assert verdict["optimistic_bound_beats_tp1"] is False
    assert any("no shard kernel can make the group" in r for r in verdict["withheld_reasons"])
    assert verdict["certified"] is False
    # An unreachable gate is withheld; an unreachable aspiration is only noted.
    assert not any(
        "aspiration" in reason for reason in verdict["withheld_reasons"]
    )


def test_build_report_reports_an_optimistic_bound_that_clears_the_target(mod) -> None:
    """A cheap collective leaves room even with free rank-weight reads."""

    devices = [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")]
    report = mod.build_report(devices, collective_ms=(1.3,), fixed_shares=(0.0,))
    verdict = report["verdict"]
    # 35.84 ms baseline over a 1.3 ms collective.
    assert verdict["optimistic_bound_speedup"] == pytest.approx(35.842 / 1.3, rel=1e-4)
    assert verdict["optimistic_bound_clears_target"] is True


def test_optimistic_bound_decides_a_losing_projection_without_shard_kernels(mod) -> None:
    """The matched pair's lower bound misses the target, so no kernel rescues it.

    This is the plan's own stop rule. With free rank-weight reads the group would
    still pay the measured collective, and that alone already exceeds the faster
    TP1 token time divided by the target.
    """

    devices = [
        mod.parse_tp1("W7900=29.575:15.652:8.646:512-128-int8kv-eager"),
        mod.parse_tp1("XTX=35.355:15.652:7.009:512-128-int8kv-eager"),
    ]
    report = mod.build_report(
        devices,
        collective_ms=(22.758,),
        fixed_shares=(0.0,),
        reduction_points=128,
        collective_source={
            "marginal_us_per_step": 177.8,
            "source": "benchmarks/results/2026-09-14-w7900-tp2-dependent-reduction-chain.json",
            "mode": "per_step",
            "depends_on_every_step": True,
        },
    )
    verdict = report["verdict"]
    assert verdict["optimistic_bound_speedup"] == pytest.approx(28.284 / 22.758, rel=1e-4)
    assert verdict["optimistic_bound_clears_target"] is False
    # The bound clears the gate, so it bounds the win rather than blocking it.
    assert verdict["optimistic_bound_beats_tp1"] is True
    assert any("optimistic bound" in note for note in verdict["aspiration_notes"])
    assert not any("optimistic bound" in reason for reason in verdict["withheld_reasons"])
    # What does block it is the projection itself: no row beats the faster TP1 arm.
    assert verdict["beats_faster_tp1_arm"] is False
    assert any("would not beat the faster TP1 arm" in r for r in verdict["withheld_reasons"])
    assert verdict["certified"] is False
    assert report["errors"] == []


# -- CLI ----------------------------------------------------------------------


def test_main_writes_the_artifact(tmp_path: pathlib.Path, mod, capsys) -> None:
    output = tmp_path / "break_even.json"
    exit_code = mod.main(
        [
            "--tp1",
            "W7900=27.9:15.652:8.646:512/128/int8-kv",
            "--marginal-us",
            "177.8",
            "--reduction-points",
            "128",
            "--fixed-share",
            "0.2",
            "--json",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows"][0]["collective_ms_per_token"] == pytest.approx(22.758)
    assert payload["collective_source"]["source"] == "command line"
    assert "verdict" in capsys.readouterr().out


def test_main_requires_a_measured_collective_source(mod) -> None:
    """No default marginal: the input must name where it came from."""

    with pytest.raises(SystemExit):
        mod.main(
            [
                "--tp1",
                "W7900=27.9:15.652:8.646:512/128/int8-kv",
                "--reduction-points",
                "128",
            ]
        )


def test_main_reads_the_marginal_from_a_chain_artifact(tmp_path: pathlib.Path, mod) -> None:
    source = _chain_artifact(tmp_path / "chain.json", marginal=177.8)
    output = tmp_path / "break_even.json"
    exit_code = mod.main(
        [
            "--tp1",
            "W7900=27.9:15.652:8.646:512/128/int8-kv",
            "--dependent-chain-artifact",
            str(source),
            "--dependent-chain-case",
            "all_reduce:rows1:fp32",
            "--reduction-points",
            "128",
            "--json",
            str(output),
            "--quiet",
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["collective_ms_per_token"] == [pytest.approx(22.758)]
    assert payload["collective_source"]["mode"] == "per_step"
    assert payload["collective_source"]["case"] == "all_reduce:rows1:fp32"


# -- shard-kernel evidence validation -----------------------------------------


def test_a_dangling_evidence_path_does_not_certify(mod, evidence_root) -> None:
    """The old defect: any nonempty list satisfied the gate."""

    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=["benchmarks/results/shard-kernel-smoke.json"],
        results_root=evidence_root,
    )
    gate = report["pre_packet3_gate"]
    assert gate["satisfied"] is False
    assert gate["evidence"][0]["error"] == "not a file"
    assert any(
        "shard-kernel evidence" in reason for reason in report["verdict"]["withheld_reasons"]
    )
    assert report["verdict"]["certified"] is False


def test_evidence_with_no_kernel_record_does_not_certify(mod, evidence_root) -> None:
    """A real JSON file whose contents prove nothing must fail the same way."""

    path = _write_kernel_artifact(evidence_root, payload={"note": "smoke passed", "ok": True})
    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[path],
        results_root=evidence_root,
    )
    gate = report["pre_packet3_gate"]
    assert gate["satisfied"] is False
    assert "no kernel record" in gate["evidence"][0]["error"]
    assert report["verdict"]["certified"] is False


def test_a_zero_or_negative_duration_does_not_certify(mod, evidence_root) -> None:
    """A name with no runtime is not evidence that a kernel executed."""

    path = _write_kernel_artifact(
        evidence_root,
        payload={"kernels": [{"Kernel_Name": "k", "DurationNs": 0}]},
    )
    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[path],
        results_root=evidence_root,
    )
    assert report["pre_packet3_gate"]["satisfied"] is False
    assert report["verdict"]["certified"] is False


def test_an_artifact_outside_results_does_not_certify(mod, tmp_path) -> None:
    """The artifacts directory is part of the evidence convention."""

    outside = tmp_path / "scratch.json"
    outside.write_text(json.dumps(VALID_KERNEL_JSON))
    root = tmp_path / "benchmarks" / "results"
    root.mkdir(parents=True)
    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[str(outside)],
        results_root=root,
    )
    assert report["pre_packet3_gate"]["evidence"][0]["error"] == "not under benchmarks/results/"
    assert report["verdict"]["certified"] is False


def test_a_real_kernel_trace_artifact_certifies(mod, evidence_root) -> None:
    """The gate opens only on contents that show a kernel ran with a duration."""

    path = _write_kernel_artifact(evidence_root)
    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[path],
        results_root=evidence_root,
    )
    gate = report["pre_packet3_gate"]
    assert gate["satisfied"] is True
    assert gate["evidence"][0]["kernel_records"] == 2
    assert gate["evidence"][0]["example"]["duration_ns"] == 971796639
    assert report["verdict"]["certified"] is True


def test_a_rocprof_csv_trace_certifies_and_a_headerless_one_does_not(
    mod, evidence_root
) -> None:
    """rocprofv3 --kernel-trace writes CSV; that form must be accepted as-is."""

    csv_trace = (
        "Kernel_Name,DurationNs,Calls\n"
        '"gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out",971796639,4080\n'
    )
    good = _write_kernel_artifact(evidence_root, name="trace.csv", payload=csv_trace)
    bad = _write_kernel_artifact(
        evidence_root, name="empty.csv", payload="a,b\n1,2\n"
    )
    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[good, bad],
        results_root=evidence_root,
    )
    gate = report["pre_packet3_gate"]
    assert gate["satisfied"] is False, "one invalid artifact must fail the whole gate"
    by_path = {e["path"]: e for e in gate["evidence"]}
    assert by_path[good]["kernel_records"] == 1
    assert "no Kernel_Name/DurationNs" in by_path[bad]["error"]
    assert report["verdict"]["certified"] is False


def test_one_invalid_artifact_names_itself_in_the_verdict(mod, evidence_root) -> None:
    """Mixed evidence must be attributable, not merged into one blob."""

    good = _write_kernel_artifact(evidence_root, name="good.json")
    bad = _write_kernel_artifact(evidence_root, name="bad.json", payload={"kernels": []})
    report = mod.build_report(
        [mod.parse_tp1("W7900=27.9:15.652:8.646:512/128/int8-kv")],
        collective_ms=(1.0,),
        fixed_shares=(0.0,),
        shard_kernel_evidence=[good, bad],
        results_root=evidence_root,
    )
    reasons = report["verdict"]["withheld_reasons"]
    assert any("good.json" not in r and "bad.json" in r for r in reasons)
    assert not any("good.json" in r for r in reasons)
