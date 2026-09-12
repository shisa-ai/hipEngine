from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import specdec2_perf_bridge as bridge_module
from scripts.specdec2_perf_bridge import (
    FULL_PROMPT_IDS,
    BridgeContractError,
    _StageLedger,
    _close_preserving_primary,
    _decode_only_seconds,
    _install_stage_ledger,
    _run_arm,
    _run_legacy_native,
    _summarize,
    arm_order,
    atomic_write_json,
    bridge_service_capacity,
    bridge_speed_claim_eligible,
    load_prompt_suite,
    normalize_timing_payloads,
    parse_budgets,
    parse_concurrencies,
    resolve_platform,
    validate_bridge_artifact,
)


def _timing_payload(*, owner: bool = True) -> dict[str, object]:
    return {
        "timing_scope": "batch",
        "batch_id": "batch-1",
        "group_rows": 2,
        "timing_owner": owner,
        "timing": {"decode_batch_ms": 10.0, "prefill_ms": 20.0},
    }


def _arm(route: str, token: int = 7) -> dict[str, object]:
    return {
        "status": "complete",
        "realized_route": route,
        "complete_wall_seconds": 1.0,
        "decode_only_seconds": 0.5,
        "generated_token_ids": [[token, token + 1]],
        "generated_tokens": 2,
        "timing_payloads": [
            {
                "timing_scope": "request",
                "batch_id": None,
                "group_rows": 1,
                "timing_owner": True,
                "timing": {"decode_ms": 500.0},
            }
        ],
    }


def _artifact() -> dict[str, object]:
    cells = []
    for prompt_id in FULL_PROMPT_IDS:
        cells.append(
            {
                "prompt_id": prompt_id,
                "run": 0,
                "concurrency": 1,
                "candidate_budget": 2,
                "execution_order": ["true_ar", "legacy_native", "specdec2"],
                "arms": {
                    "true_ar": _arm("true_ar"),
                    "legacy_native": _arm("legacy_native"),
                    "specdec2": _arm("specdec2"),
                },
                "exact": True,
            }
        )
    return {
        "schema": 1,
        "kind": "specdec2_perf_bridge",
        "status": "complete",
        "provenance": {
            "staged_dirty": False,
            "unstaged_dirty": False,
            "unexpected_untracked": [],
        },
        "model": {
            "execution_profile": "strict",
            "recurrent_state": "fp32",
        },
        "workload": {
            "scope": "full",
            "prompt_ids": list(FULL_PROMPT_IDS),
            "concurrency": [1],
            "candidate_budgets": [2],
            "runs": 1,
        },
        "cells": cells,
    }


def test_bridge_passes_declared_budget_through_public_llm_owner() -> None:
    source = Path("scripts/specdec2_perf_bridge.py").read_text(encoding="utf-8")

    assert "speculative_candidate_budget=int(budget)" in source


def test_bridge_resolves_independent_backend_arch_quant_and_queue_policy() -> None:
    gfx1151 = resolve_platform(
        backend="hip_gfx1151",
        target_arch=None,
        quant_label="Q4_K_S",
        gpu_max_hw_queues=None,
        environ={},
    )
    gfx1100 = resolve_platform(
        backend="hip_gfx1100",
        target_arch=None,
        quant_label="Q4_K_M",
        gpu_max_hw_queues=None,
        environ={},
    )
    explicit = resolve_platform(
        backend="hip_gfx1100",
        target_arch="gfx1100",
        quant_label="Q4_K_M",
        gpu_max_hw_queues=1,
        environ={"GPU_MAX_HW_QUEUES": "8"},
    )

    assert gfx1151 == {
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "quant_label": "Q4_K_S",
        "gpu_max_hw_queues": "2",
        "queue_source": "gfx1151_campaign_default",
    }
    assert gfx1100 == {
        "backend": "hip_gfx1100",
        "target_arch": "gfx1100",
        "quant_label": "Q4_K_M",
        "gpu_max_hw_queues": None,
        "queue_source": "unset",
    }
    assert explicit["gpu_max_hw_queues"] == "1"
    assert explicit["queue_source"] == "explicit_cli"
    with pytest.raises(ValueError, match="does not match backend"):
        resolve_platform(
            backend="hip_gfx1100",
            target_arch="gfx1151",
            quant_label="Q4_K_M",
            gpu_max_hw_queues=None,
            environ={},
        )


def test_bridge_parses_only_supported_physical_cells() -> None:
    assert parse_concurrencies("8,1,7,2,6,3,5,4") == (8, 1, 7, 2, 6, 3, 5, 4)
    assert parse_budgets("3,1,2") == (3, 1, 2)

    with pytest.raises(ValueError, match="concurrency"):
        parse_concurrencies("1,9")
    with pytest.raises(ValueError, match="candidate budget"):
        parse_budgets("0,2")
    with pytest.raises(ValueError, match="duplicate"):
        parse_budgets("2,2")


def test_bridge_legacy_native_uses_model_owned_moe_route() -> None:
    calls = []
    generator = SimpleNamespace(
        generate_speculative_mtp_detailed=lambda request: calls.append(
            ("moe", request)
        )
        or ("output",),
        _generate_dense_speculative_mtp_detailed=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense route must not run for MoE")
        ),
    )
    request = bridge_module.GenerationRequest(
        prompts=("prompt",),
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )

    assert _run_legacy_native(
        generator,
        SimpleNamespace(is_moe=True),
        request,
    ) == ("output",)
    assert calls[0][0] == "moe"


def test_bridge_production_profile_skips_incompatible_legacy_c1_control() -> None:
    skipped = _run_arm(
        arm="legacy_native",
        service=None,
        direct_generator=None,
        direct_config=None,
        request=None,
        concurrency=1,
        ledger=SimpleNamespace(),
        legacy_native_supported=False,
    )
    assert skipped == {
        "status": "skipped",
        "reason": "dense_direct_legacy_requires_strict_fp32_state",
        "realized_route": None,
    }

    payload = _artifact()
    payload["model"] = {
        "execution_profile": "production",
        "recurrent_state": "fp16",
    }
    for cell in payload["cells"]:
        cell["arms"]["legacy_native"] = skipped
    validate_bridge_artifact(payload)


def test_bridge_production_cross_arm_ids_are_diagnostic_but_repeats_are_exact() -> None:
    payload = _artifact()
    payload["model"] = {
        "execution_profile": "production",
        "recurrent_state": "fp32",
    }
    payload["workload"]["runs"] = 2
    for cell in payload["cells"]:
        cell["arms"]["legacy_native"] = {
            "status": "skipped",
            "reason": "dense_direct_legacy_requires_strict_fp32_state",
            "realized_route": None,
        }
        cell["arms"]["specdec2"] = _arm("specdec2", token=9)
        cell["exact"] = False
    repeated = json.loads(json.dumps(payload["cells"]))
    for cell in repeated:
        cell["run"] = 1
    payload["cells"].extend(repeated)

    validate_bridge_artifact(payload)

    payload["cells"][-1]["arms"]["specdec2"] = _arm("specdec2", token=10)
    with pytest.raises(BridgeContractError, match="not repeatable"):
        validate_bridge_artifact(payload)


def test_bridge_requires_separate_c1_and_physical_service_capacities() -> None:
    assert bridge_service_capacity((1,)) == 1
    assert bridge_service_capacity((2, 3, 4)) == 4
    assert bridge_service_capacity((3,)) == 3
    assert bridge_service_capacity((2,)) == 2
    assert bridge_service_capacity((6,), requested_capacity=8) == 8
    assert bridge_service_capacity((7,), requested_capacity=8) == 8
    assert bridge_service_capacity((8,), requested_capacity=8) == 8
    with pytest.raises(ValueError, match="smaller than realized concurrency"):
        bridge_service_capacity((8,), requested_capacity=7)
    with pytest.raises(ValueError, match="separate bridge invocations"):
        bridge_service_capacity((1, 2, 3, 4))


def test_roctx_prefers_profiler_sdk_overlay(monkeypatch) -> None:
    loaded = []

    def fake_cdll(name):
        loaded.append(name)
        return SimpleNamespace(
            roctxRangePushA=SimpleNamespace(),
            roctxRangePop=SimpleNamespace(),
        )

    monkeypatch.setattr(bridge_module.ctypes, "CDLL", fake_cdll)

    bridge_module._Roctx(True)

    assert loaded == ["librocprofiler-sdk-roctx.so.1"]


def test_bridge_counterbalance_is_index_only_and_reverses_ar_spec_order() -> None:
    assert arm_order(0) == ("true_ar", "legacy_native", "specdec2")
    assert arm_order(1) == ("specdec2", "legacy_native", "true_ar")
    assert [arm_order(index) for index in range(10)].count(
        ("true_ar", "legacy_native", "specdec2")
    ) == 5


def test_bridge_loads_the_exact_canonical_prompt_contract() -> None:
    rows = load_prompt_suite(Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))

    assert tuple(row["id"] for row in rows) == FULL_PROMPT_IDS
    assert all(row["rendered_prompt"].endswith("<|im_start|>assistant\n") for row in rows)
    assert all(len(row["prompt_sha256"]) == 64 for row in rows)


def test_bridge_speed_claim_gate_accepts_planned_full_k2_shape() -> None:
    assert bridge_speed_claim_eligible(
        scope="full",
        prompt_ids=FULL_PROMPT_IDS,
        runs=3,
        concurrencies=(1, 2, 4),
        tracked_clean=True,
        unexpected_untracked=(),
        all_exact=True,
    )
    assert not bridge_speed_claim_eligible(
        scope="full",
        prompt_ids=FULL_PROMPT_IDS,
        runs=3,
        concurrencies=(1,),
        tracked_clean=True,
        unexpected_untracked=(),
        all_exact=True,
    )
    assert not bridge_speed_claim_eligible(
        scope="full",
        prompt_ids=FULL_PROMPT_IDS,
        runs=3,
        concurrencies=(1, 2, 4),
        tracked_clean=False,
        unexpected_untracked=(),
        all_exact=True,
    )


def test_bridge_timing_ownership_counts_one_batch_payload() -> None:
    normalized = normalize_timing_payloads(
        [_timing_payload(owner=True), _timing_payload(owner=False)]
    )

    assert normalized["owners"] == 1
    assert normalized["owned_totals_ms"] == {
        "decode_batch_ms": 10.0,
        "prefill_ms": 20.0,
    }
    assert normalized["ignored_nonowners"] == 1


def test_bridge_rejects_duplicate_or_missing_batch_timing_owner() -> None:
    with pytest.raises(BridgeContractError, match="exactly one timing owner"):
        normalize_timing_payloads(
            [_timing_payload(owner=True), _timing_payload(owner=True)]
        )
    malformed = _timing_payload(owner=True)
    malformed.pop("timing_owner")
    with pytest.raises(BridgeContractError, match="timing_owner"):
        normalize_timing_payloads([malformed, _timing_payload(owner=False)])


def test_bridge_rejects_invalid_true_ar_denominator() -> None:
    payload = _artifact()
    payload["cells"][0]["arms"]["true_ar"]["complete_wall_seconds"] = 0.0

    with pytest.raises(BridgeContractError, match="true AR denominator"):
        validate_bridge_artifact(payload)


def test_bridge_rejects_incomplete_canonical_prompt_grid() -> None:
    payload = _artifact()
    payload["cells"] = payload["cells"][:-1]

    with pytest.raises(BridgeContractError, match="grid"):
        validate_bridge_artifact(payload)


def test_bridge_rejects_tracked_dirty_provenance() -> None:
    payload = _artifact()
    payload["provenance"]["unstaged_dirty"] = True

    with pytest.raises(BridgeContractError, match="tracked-clean"):
        validate_bridge_artifact(payload)


def test_bridge_rejects_silent_route_substitution() -> None:
    payload = _artifact()
    payload["cells"][0]["arms"]["specdec2"]["realized_route"] = "legacy_native"

    with pytest.raises(BridgeContractError, match="realized route"):
        validate_bridge_artifact(payload)


def test_bridge_rejects_generated_id_or_decode_timing_drift() -> None:
    payload = _artifact()
    payload["cells"][0]["arms"]["specdec2"]["generated_token_ids"] = [[8, 9]]
    with pytest.raises(BridgeContractError, match="generated IDs"):
        validate_bridge_artifact(payload)

    payload = _artifact()
    payload["cells"][0]["arms"]["specdec2"]["decode_only_seconds"] = None
    with pytest.raises(BridgeContractError, match="decode-only"):
        validate_bridge_artifact(payload)


def test_bridge_stage_ledger_records_nested_cycle_owners_and_restores() -> None:
    class Owner:
        def cycle(self, value: int) -> int:
            return value + 1

    owner = Owner()
    original = owner.cycle
    ledger = _StageLedger(roctx=False)
    assert ledger.install(owner, "cycle", "cycle_total")
    with ledger.arm("specdec2"):
        assert owner.cycle(2) == 3
    snapshot = ledger.snapshot()
    assert snapshot["call_counts"] == {"arm_complete": 1, "cycle_total": 1}
    assert snapshot["totals_seconds"]["cycle_total"] >= 0.0
    assert snapshot["allocation_samples"]["cycle_total"] == [
        {
            "allocated_bytes": 0,
            "freed_bytes": 0,
            "active_delta": 0,
            "current_bytes_delta": 0,
        }
    ]
    assert _decode_only_seconds(
        "specdec2",
        snapshot,
        {"owned_totals_ms": {}},
    ) > 0.0
    ledger.close()
    assert owner.cycle == original


def test_bridge_legacy_moe_decode_owner_uses_total_minus_prefill() -> None:
    assert _decode_only_seconds(
        "legacy_native",
        {"totals_seconds": {}},
        {
            "owned_totals_ms": {
                "mtp_run_total_ms": 900.0,
                "prefill_ms": 300.0,
            }
        },
    ) == pytest.approx(0.6)


def test_bridge_installs_initial_k0_attachment_and_refill_owners() -> None:
    class Adapter:
        def _catch_up_provider(self) -> None:
            return None

        def _catch_up_provider_batch(self) -> None:
            return None

        def begin_prompt_streaming(self, *args, **kwargs):
            return None

    class Runner:
        def __init__(self) -> None:
            self.adapter = Adapter()

        def _resolved_mtp2_adapter(self):
            return self.adapter

        def prefill_batch(self) -> None:
            return None

        def decode_batch(self) -> None:
            return None

        def prepare_speculative_k0(self) -> None:
            return None

        def prepare_speculative_requests(self) -> None:
            return None

        def propose_speculative_batch(self) -> None:
            return None

        def execute_target_frontier(self) -> None:
            return None

        def _flush_row_owner(self) -> None:
            return None

        def _close_packed_decode_graphs(self) -> None:
            return None

        def reclaim(self) -> None:
            return None

    class Loop:
        def _run_staged_speculative_cycle(self) -> None:
            return None

    runner = Runner()
    driver = type("Driver", (), {"_runner": runner, "_loop": Loop()})()
    service = type("Service", (), {"inner": driver})()
    ledger = _StageLedger(roctx=False)

    installed = _install_stage_ledger(service, ledger)

    assert installed["provider_k0_attach"]
    assert installed["provider_streaming_open"]
    assert installed["provider_open"]
    assert installed["nextn_prompt_prime_c1"]
    assert installed["nextn_prompt_prime_batch"]
    ledger.close()


def test_bridge_teardown_does_not_mask_an_active_measurement_failure() -> None:
    def broken_close() -> None:
        raise TimeoutError("driver already stopped")

    assert _close_preserving_primary(
        broken_close,
        primary_failure_active=True,
    ) == {
        "type": "TimeoutError",
        "message": "driver already stopped",
    }
    with pytest.raises(TimeoutError, match="driver already stopped"):
        _close_preserving_primary(
            broken_close,
            primary_failure_active=False,
        )


def test_bridge_summary_uses_group_wall_once_and_preserves_ratios() -> None:
    payload = _artifact()
    first = payload["cells"][0]
    first["ratios"] = {
        "legacy_native_over_true_ar_wall": 0.9,
        "specdec2_over_true_ar_wall": 1.2,
    }

    summary = _summarize([first])

    assert summary["arms"]["c1_k2_true_ar"]["generated_tokens"] == 2
    assert summary["arms"]["c1_k2_true_ar"]["aggregate_generated_tok_s"] == 2.0
    assert summary["ratios"]["c1_k2_specdec2_over_true_ar_wall"]["median"] == 1.2


def test_bridge_atomic_checkpoint_replaces_complete_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "bridge.json"

    atomic_write_json(output, {"status": "running", "completed": 1})
    atomic_write_json(output, {"status": "complete", "completed": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "completed": 2,
        "status": "complete",
    }
    assert not output.with_suffix(".json.tmp").exists()
