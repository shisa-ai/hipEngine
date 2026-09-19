from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.qwen35_gguf_mtp2_registry import (
    register_gguf_mtp2_adapter,
    unregister_gguf_mtp2_adapter,
)
from hipengine.models.qwen35 import Qwen35GGUFModel, Qwen35MoeGGUFModel
import hipengine.generation.qwen35_gguf_mtp2 as mtp2_module
from hipengine.generation.qwen35_gguf_mtp2 import (
    Qwen35GGUFMTP2Adapter,
    _MTP2RequestState,
    _PhysicalTargetCommitError,
    _target_verify_mode_for_context,
)
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.runtime import qwen35_gguf_runner as runner_mod
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNBatchDeviceProposal,
    Qwen35GGUFNextNDeviceProposal,
)
from hipengine.speculative import (
    DraftBatch,
    MtpProposalContext,
    SpeculativeMTPStaticEligibility,
    SpeculativeMTPStaticState,
    SpeculativeRequestSemantics,
    TargetAcceptSummary,
    TargetVerifyBatch,
    TargetVerifyBuffers,
)
from hipengine.speculative.ngram_mod import NgramModConfig, RequestLocalNgramMod
from hipengine.speculative.accounting import speculative_output_accounting


class _AdapterDouble:
    def __init__(self) -> None:
        self.calls = []

    def register_request(
        self,
        request_id,
        candidate_budget,
        *,
        static_eligibility=None,
    ):
        self.calls.append(
            ("register", request_id, candidate_budget, static_eligibility)
        )

    def capability(self, semantics):
        self.calls.append(("capability", tuple(semantics)))
        return "capability"

    def claims_fit(self, plan):
        return plan == "plan"

    def component_claims(self, plan):
        return {"plan": plan}

    def reserve_claims(self, claims):
        return ("reservation", claims)

    def release_claims(self, reservation):
        self.calls.append(("release", reservation))

    def prepare_requests(self, plan, semantics, *, stream=None):
        self.calls.append(("prepare", plan, tuple(semantics), stream))

    def propose_batch(self, plan, semantics, *, stream=None):
        return ("proposal", plan, tuple(semantics), stream)

    def execute_target_frontier(self, *args, **kwargs):
        return (args, kwargs)

    def rollback_cycle(self, *args):
        self.calls.append(("rollback", args))


def test_gfx1100_target_mode_resolves_before_verifier_construction() -> None:
    from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY

    target = SimpleNamespace(
        runner=SimpleNamespace(weights=SimpleNamespace(
            geometry=QWEN35_DENSE_H5120_GEOMETRY, file_type_name="MOSTLY_Q4_K_M",
        )),
    )
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1100", end_position=95
    ) == "native"
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1100", end_position=96
    ) == "serial_exact"
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1100", end_position=16384, target=target,
    ) == "native"
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1151", end_position=96
    ) == "native"
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1151", end_position=65545
    ) == "serial_exact"


def test_backend_packages_expose_independently_qualified_adapter_scopes() -> None:
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_PHYSICAL", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS",
        {},
    ) == {
        "production": (
            *((width, depth) for width in range(1, 5) for depth in range(1, 4)),
        )
    }
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_PHYSICAL", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS", {}
    ) == {
        "production": ((1, 2), (1, 3), (2, 2), (2, 3), (8, 3)),
        "strict": ((2, 2),),
    }
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES",
        {},
    ) == {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "production"): (1, 2, 3, 4),
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_SPECDEC2_PROPOSAL_LM_HEAD_ROWTILE_POLICIES",
        frozenset(),
    ) == frozenset(
        {
            (5120, 248320, rows) for rows in (2, 3, 4, 5, 6, 7, 8)
        }
    )
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_SPECDEC2_PROPOSAL_LM_HEAD_ROWTILE_POLICIES",
        frozenset(),
    ) == frozenset()


def test_qwen38_production_prompt_streaming_policy_admits_only_physical_c1_c4() -> None:
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile="production",
        ),
        capacity=4,
        _shared_runner=SimpleNamespace(
            weights=SimpleNamespace(
                geometry=QWEN35_DENSE_H5120_GEOMETRY,
                file_type_name="MOSTLY_Q4_K_M",
            ),
        ),
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )

    assert adapter.physical_prompt_streaming_widths == (1, 2, 3, 4)
    assert adapter._physical_prompt_streaming_admitted(1) is True
    assert adapter._physical_prompt_streaming_admitted(2) is True
    assert adapter._physical_prompt_streaming_admitted(3) is True
    assert adapter._physical_prompt_streaming_admitted(4) is True
    assert adapter._physical_prompt_streaming_admitted(5) is False


def test_qwen_gguf_plugins_select_distinct_mtp2_adapters() -> None:
    assert Qwen35GGUFModel().speculative_mtp2_adapter == "dense_nextn"
    assert Qwen35MoeGGUFModel().speculative_mtp2_adapter == "moe_nextn"


def _width_bound_owner(*, profile: str, capacity: int) -> SimpleNamespace:
    return SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile=profile,
        ),
        capacity=capacity,
        _shared_runner=None,
    )


def _certified_width_depths() -> tuple[tuple[int, int], ...]:
    return tuple(
        (width, depth)
        for width in range(1, 5)
        for depth in range(1, 4)
    )


def test_physical_width_depth_policy_is_package_owned_and_capacity_clamped() -> None:
    import hipengine.generation.qwen35_gguf_mtp2 as module

    # An unlisted profile retains every previously certified C1-C4 depth.
    default_adapter = Qwen35GGUFMTP2Adapter(
        _width_bound_owner(profile="strict", capacity=8),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=3,
    )
    assert default_adapter.physical_width_depths == _certified_width_depths()
    assert default_adapter.physical_max_requests == 4
    assert default_adapter._max_physical_requests() == 4

    real = module.backend_package_capability
    policy = (*_certified_width_depths(), (8, 3))

    def fake(backend: str, name: str, default: object = None) -> object:
        if name == "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS":
            return {"production": policy}
        return real(backend, name, default)

    module.backend_package_capability = fake
    try:
        lifted = Qwen35GGUFMTP2Adapter(
            _width_bound_owner(profile="production", capacity=8),
            enabled=True,
            target_verify_mode="packed",
            candidate_budget=3,
        )
        assert lifted.physical_width_depths == policy
        assert lifted.physical_max_requests == 8
        assert lifted._max_physical_requests() == 8
        clamped = Qwen35GGUFMTP2Adapter(
            _width_bound_owner(profile="production", capacity=5),
            enabled=True,
            target_verify_mode="packed",
            candidate_budget=3,
        )
        # Capacity five cannot turn the hole in the policy into C5 admission.
        assert clamped._max_physical_requests() == 4
    finally:
        module.backend_package_capability = real


@pytest.mark.parametrize(
    "table",
    [
        "nope",
        {"production": 0},
        {"production": None},
        {"production": ((0, 3),)},
        {"production": ((8, 8),)},
        {"production": ((8,),)},
    ],
)
def test_physical_width_depth_policy_misconfiguration_fails_closed(
    table: object,
) -> None:
    owner = _width_bound_owner(profile="production", capacity=8)
    import hipengine.generation.qwen35_gguf_mtp2 as module

    real = module.backend_package_capability

    def fake(backend: str, name: str, default: object = None) -> object:
        if name == "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS":
            return table
        return real(backend, name, default)

    module.backend_package_capability = fake
    try:
        with pytest.raises(RuntimeError):
            Qwen35GGUFMTP2Adapter(
                owner,
                enabled=True,
                target_verify_mode="packed",
                candidate_budget=3,
            )
    finally:
        module.backend_package_capability = real


def test_physical_width_depth_policy_admits_only_listed_cells() -> None:
    # The gfx1151 C8-K3 cell was withdrawn on 2026-09-19: this backend's
    # prompt-streaming policy admits widths (1,2,3,4), so an eight-row group is
    # refused by the provider while the route still ran, and the cell measured
    # 0.936x of its own AR baseline with 0/10 cells reproducing the AR output.
    adapter = Qwen35GGUFMTP2Adapter(
        _width_bound_owner(profile="production", capacity=8),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=3,
    )

    assert adapter._max_physical_requests() == 4
    for width in (5, 6, 7, 8):
        assert adapter._physical_width_depth_admitted(width, 3) is False
    for depth in (1, 2):
        assert adapter._physical_width_depth_admitted(8, depth) is False
    max_rows = adapter._max_physical_requests() * (adapter.candidate_budget + 1)
    assert max_rows == 16


def test_physical_width_depth_policy_gates_capability_and_claims() -> None:
    target = SimpleNamespace(
        runner=SimpleNamespace(fp16_recurrent_state=False),
        _target_scratch_owner=SimpleNamespace(slot_count=8),
        target_layout=SimpleNamespace(max_sequence_length=1024),
        kv_storage_dtype="bf16",
    )
    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=True,
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(),
    )
    ids = tuple(range(1, 9))
    adapter = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                backend="hip_gfx1151",
                execution_profile="production",
            ),
            capacity=8,
            _shared_runner=None,
            _row=lambda rid: row,
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    adapter._intents = {rid: 3 for rid in ids}
    adapter._static_eligibility_by_request = {
        rid: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_physical_c8_k3",
            max_candidate_count=3,
            max_realized_group_rows=8,
            automatic_eligible=False,
            strict_fallback_key="gguf_target_ar",
            evidence_key=f"test-c8-k3-{rid}",
            evidence_fingerprint=f"sha256:test-c8-k3-{rid}",
        )
        for rid in ids
    }
    adapter._prompt_hidden_rows = {rid: object() for rid in ids}
    adapter._states = {}
    adapter._disabled_requests = set()
    adapter._active_claims = None

    def semantics(width: int) -> tuple[SpeculativeRequestSemantics, ...]:
        return tuple(
            SpeculativeRequestSemantics(rid, "greedy", "verify_chain", 32, 25)
            for rid in ids[:width]
        )

    for width in (5, 6, 7, 8):
        assert adapter.partition_max_requests(ids[:width]) == 0
        assert adapter.capability(semantics(width)) is None
    # A zero partition bound preserves one whole due group, and the group then
    # takes one full-batch AR step: the wide cell is no longer listed, so
    # per-row eligibility and capacity 8 cannot widen the group past C4.
    assert adapter.partition_max_requests(ids) == 0
    assert adapter.capability(semantics(8)) is None
    assert adapter.capability(semantics(4)) is not None

    for width in (5, 6, 7, 8):
        assert adapter.claims_fit(
            SimpleNamespace(
                request_ids=ids[:width],
                speculative_request_ids=ids[:width],
                candidate_counts=(3,) * width,
            )
        ) is False
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=ids[:4],
            speculative_request_ids=ids[:4],
            candidate_counts=(3,) * 4,
        )
    ) is True
    for depth in (1, 2, 3):
        assert adapter.claims_fit(
            SimpleNamespace(
                request_ids=ids[:4],
                speculative_request_ids=ids[:4],
                candidate_counts=(depth,) * 4,
            )
        ) is True
    # The listed depths stop at 3, and the rows' own candidate bound agrees.
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=ids[:4],
            speculative_request_ids=ids[:4],
            candidate_counts=(4,) * 4,
        )
    ) is False


def _gfx1100_screening_owner(*, capacity: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1100",
            execution_profile="production",
        ),
        capacity=capacity,
        _shared_runner=None,
        _row=lambda rid: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(
                session=SimpleNamespace(
                    runner=SimpleNamespace(fp16_recurrent_state=False),
                    target_layout=SimpleNamespace(max_sequence_length=1024),
                    kv_storage_dtype="bf16",
                )
            ),
            slot=SimpleNamespace(),
        ),
    )


def _explicit_only_eligibility(rid: int, *, rows: int = 8, depth: int = 3) -> SpeculativeMTPStaticEligibility:
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="screening_test_explicit_only_row",
        max_candidate_count=depth,
        max_realized_group_rows=rows,
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key=f"test-screening-{rid}",
        evidence_fingerprint=f"sha256:test-screening-{rid}",
    )


def test_screening_unqualified_cells_default_off_keeps_policy_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the screening env, an explicit-only unqualified cell stays K0."""

    monkeypatch.delenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", raising=False)
    adapter = Qwen35GGUFMTP2Adapter(
        _gfx1100_screening_owner(),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=3,
    )
    ids = (11, 12, 13, 14, 15)
    adapter._static_eligibility_by_request = {
        rid: _explicit_only_eligibility(rid) for rid in ids
    }

    assert adapter._physical_width_depth_admitted_for_group(5, 3, ids) is False
    assert adapter._last_screening_cell is None


def test_screening_unqualified_cells_admits_explicit_only_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The screening env opens unlisted cells only for explicit-only rows."""

    monkeypatch.setenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", "1")
    adapter = Qwen35GGUFMTP2Adapter(
        _gfx1100_screening_owner(),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=3,
    )
    ids = tuple(range(1, 6))
    adapter._intents = {rid: 3 for rid in ids}
    adapter._static_eligibility_by_request = {
        rid: _explicit_only_eligibility(rid) for rid in ids
    }
    adapter._prompt_hidden_rows = {rid: object() for rid in ids}
    adapter._states = {}
    adapter._disabled_requests = set()
    adapter._active_claims = None

    assert adapter._physical_width_depth_admitted_for_group(5, 3, ids) is True
    assert adapter._last_screening_cell == {
        "width": 5,
        "depth": 3,
        "request_ids": list(ids),
    }
    semantics = tuple(
        SpeculativeRequestSemantics(rid, "greedy", "verify_chain", 32, 25)
        for rid in ids
    )
    capability = adapter.capability(semantics)
    assert capability is not None
    # Depth-3 listed cells exist at widths 1, 2, and 8 (C2/K3 qualified by
    # the Packet 6 grid + retained reproduction), so the listed derivation
    # still owns the tuple; the screening group does not extend it.
    assert capability.proposal_widths == (1, 2, 8)
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=ids,
            speculative_request_ids=ids,
            candidate_counts=(3,) * len(ids),
        )
    ) is True


def test_screening_depth_without_listed_cell_keeps_nonempty_proposal_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A K1 screening group at C2 must still expose its proposal width.

    The gfx1100 production policy lists no depth-1 cell, so the listed
    proposal-width derivation is empty; the server rejects empty widths and
    the C2/K1 screening measurement 400s without this fallback.
    """

    monkeypatch.setenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", "1")
    adapter = Qwen35GGUFMTP2Adapter(
        _gfx1100_screening_owner(capacity=2),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=1,
    )
    ids = (41, 42)
    adapter._intents = {rid: 1 for rid in ids}
    adapter._static_eligibility_by_request = {
        rid: _explicit_only_eligibility(rid, rows=8, depth=1) for rid in ids
    }
    adapter._prompt_hidden_rows = {rid: object() for rid in ids}
    adapter._states = {}
    adapter._disabled_requests = set()
    adapter._active_claims = None

    semantics = tuple(
        SpeculativeRequestSemantics(rid, "greedy", "verify_chain", 32, 25)
        for rid in ids
    )
    capability = adapter.capability(semantics)
    assert capability is not None
    # The capability advertises the eligibility bound; the executed per-request
    # K comes from the plan. The K1 cell itself must fit and the proposal
    # width tuple must be non-empty (the server rejects empty widths).
    assert capability.proposal_widths == (2,)
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=ids,
            speculative_request_ids=ids,
            candidate_counts=(1, 1),
        )
    ) is True


def test_screening_unqualified_cells_never_widen_automatic_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3.6/gfx1151 automatic policy cannot inherit screening cells."""

    monkeypatch.setenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", "1")
    adapter = Qwen35GGUFMTP2Adapter(
        _gfx1100_screening_owner(),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=3,
    )
    ids = (21, 22, 23, 24, 25)
    adapter._static_eligibility_by_request = {
        rid: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="automatic_qualified_row",
            max_candidate_count=3,
            max_realized_group_rows=8,
            automatic_eligible=True,
            strict_fallback_key="gguf_target_ar",
            evidence_key=f"test-automatic-{rid}",
            evidence_fingerprint=f"sha256:test-automatic-{rid}",
        )
        for rid in ids
    }

    assert adapter._physical_width_depth_admitted_for_group(5, 3, ids) is False
    assert adapter._last_screening_cell is None


def test_screening_unqualified_cells_require_static_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request without a serving-evidence row cannot screen a cell open."""

    monkeypatch.setenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", "1")
    adapter = Qwen35GGUFMTP2Adapter(
        _gfx1100_screening_owner(capacity=2),
        enabled=True,
        target_verify_mode="packed",
        candidate_budget=3,
    )

    assert adapter._physical_width_depth_admitted_for_group(2, 4, (31, 32)) is False
    assert adapter._last_screening_cell is None


def test_unregistered_model_plugin_mtp2_adapter_fails_closed() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._mtp2_adapter = None
    runner._mtp2_adapter_resolved = False
    runner.capacity = 1
    runner.generator = SimpleNamespace(
        backend="hip_gfx1100",
        supports_speculative_mtp=True,
        model_plugin=SimpleNamespace(speculative_mtp2_adapter="not_registered"),
    )

    assert runner._resolved_mtp2_adapter() is None
    assert runner._mtp2_adapter_resolved is True


def test_resident_runner_resolves_model_plugin_mtp2_adapter_without_model_branch() -> None:
    key = "test_moe_nextn"
    calls = []

    def factory(owner, **kwargs):
        calls.append((owner, kwargs))
        return _AdapterDouble()

    register_gguf_mtp2_adapter(key, factory)
    try:
        runner = object.__new__(Qwen35GGUFResidentModelRunner)
        runner._mtp2_adapter = None
        runner._mtp2_adapter_resolved = False
        runner.capacity = 1
        runner.generator = SimpleNamespace(
            backend="hip_gfx1100",
            target_arch="gfx1100",
            supports_speculative_mtp=True,
            speculative_candidate_budget=2,
            model_plugin=SimpleNamespace(speculative_mtp2_adapter=key),
            _kv_weight_quant_key=lambda: "gguf_q4_k_m",
        )

        adapter = runner._resolved_mtp2_adapter()

        assert isinstance(adapter, _AdapterDouble)
        assert calls == [
            (
                runner,
                {
                    "enabled": True,
                    "target_verify_mode": "native",
                    "candidate_budget": 2,
                    "quant": "gguf_q4_k_m",
                },
            )
        ]
    finally:
        unregister_gguf_mtp2_adapter(key)


def _bare_runner() -> Qwen35GGUFResidentModelRunner:
    """A runner built without ``__init__``, carrying its cycle accounting."""

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._execution_mutation_window = "unknown"
    runner._execution_step_request_id = None
    runner._execution_stepped_request_ids = set()
    runner._execution_committed_request_ids = set()
    runner._execution_speculative_plan = None
    return runner


def test_resident_runner_delegates_staged_methods_without_backend_branches() -> None:
    runner = _bare_runner()
    adapter = _AdapterDouble()
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(target_arch="gfx1151")
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}

    runner.register_speculative_request(7, 3)
    assert runner._rows[7].mtp2_candidate_budget == 3
    assert adapter.calls == [("register", 7, 3, None)]
    assert runner.speculative_capability(("semantics",)) == "capability"
    assert runner.speculative_claims_fit("plan") is True
    assert runner.speculative_component_claims("plan") == {"plan": "plan"}
    reservation = runner.reserve_speculative_claims("claims")
    assert reservation == ("reservation", "claims")
    runner.release_speculative_claims(reservation)
    runner.prepare_speculative_requests("plan", ("s",), stream=9)
    assert runner.propose_speculative_batch("plan", ("s",), stream=4) == (
        "proposal",
        "plan",
        ("s",),
        4,
    )
    assert runner.speculative_kv_live_spans_owner(SimpleNamespace(operation_id="op"))


def test_resident_runner_bounds_cycle_intent_by_static_evidence_k() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    adapter = _AdapterDouble()
    adapter.candidate_budget = 3
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(target_arch="gfx1151")
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}
    eligibility = SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="qualified_test_k2",
        max_candidate_count=2,
        max_realized_group_rows=4,
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key="test-k2",
        evidence_fingerprint="sha256:test-k2",
    )

    runner.register_speculative_request(7, 3, static_eligibility=eligibility)

    assert runner._rows[7].mtp2_candidate_budget == 2
    assert adapter.calls == [("register", 7, 2, eligibility)]


def test_resident_runner_delegates_bounded_complete_cycle_when_plugin_selects_it() -> None:
    adapter = SimpleNamespace(
        staged_frontier=False,
        execute_cycle=lambda plan, commit: (plan, commit),
    )
    runner = _bare_runner()
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(target_arch="gfx1100")

    assert runner.speculative_frontier_available("plan") is False
    assert runner.execute_speculative_cycle("plan", commit=True) == ("plan", True)


def test_physical_specdec2_uses_qualified_eager_when_graph_is_uncached() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._mtp2_adapter = SimpleNamespace()
    runner._mtp2_adapter_resolved = True

    assert runner.speculative_graph_available(object()) is False


def test_physical_extra_rowtiles_are_production_and_backend_capability_scoped() -> None:
    production = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                execution_profile="production",
                backend="hip_gfx1100",
            )
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    strict = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                execution_profile="strict",
                backend="hip_gfx1100",
            )
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )

    # Prompt streaming is model/quant scoped by the newer package API, so an
    # owner without loaded-weight identity fails closed even though the backend's
    # production rowtile capabilities remain available.
    assert production.physical_prompt_streaming is False
    assert production.production_physical_extra_rowtiles is True
    assert production.production_physical_q5_rowtile is True
    assert production.production_physical_q6_rowtile is True
    assert production.production_physical_q6_mixed_rowtiles is True
    assert strict.physical_prompt_streaming is False
    assert strict.production_physical_extra_rowtiles is False
    assert strict.production_physical_q5_rowtile is False
    assert strict.production_physical_q6_rowtile is False
    assert strict.production_physical_q6_mixed_rowtiles is False
    production.close()
    strict.close()


def test_mixed_q6_target_rowtiles_are_default_on_with_rollback_and_profile_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", raising=False
    )
    production = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                execution_profile="production",
                backend="hip_gfx1100",
            )
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    strict = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                execution_profile="strict",
                backend="hip_gfx1100",
            )
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    peer = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                execution_profile="production",
                backend="hip_gfx1151",
            )
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )

    assert production.production_physical_q6_mixed_rowtiles is True
    assert strict.production_physical_q6_mixed_rowtiles is False
    assert peer.production_physical_q6_mixed_rowtiles is False
    production.close()
    strict.close()
    peer.close()

    monkeypatch.setenv(
        "HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", "0"
    )
    rollback = Qwen35GGUFMTP2Adapter(
        SimpleNamespace(
            generator=SimpleNamespace(
                execution_profile="production",
                backend="hip_gfx1100",
            )
        ),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert rollback.production_physical_q6_mixed_rowtiles is False
    rollback.close()


def test_exact_target_rows_are_production_defaults_with_rollbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            execution_profile="production",
            backend="hip_gfx1100",
        ),
        capacity=8,
        _shared_runner=SimpleNamespace(hidden_size=4),
    )
    monkeypatch.delenv(
        "HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", raising=False
    )
    monkeypatch.delenv(
        "HIPENGINE_GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS", raising=False
    )
    monkeypatch.delenv(
        "HIPENGINE_GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS", raising=False
    )
    default = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert default.production_exact_target_row_counts == (8, 28, 32)
    assert default._target_group_pad_rows(request_count=2, candidate_rows=6) == 0
    assert default._target_group_pad_rows(request_count=7, candidate_rows=21) == 0
    assert default._target_group_pad_rows(request_count=8, candidate_rows=24) == 0
    assert default._target_group_pad_rows(request_count=1, candidate_rows=3) == 2
    default.close()

    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS", "0")
    c8_rollback = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert c8_rollback.production_exact_target_row_counts == (8, 28)
    assert c8_rollback._target_group_pad_rows(
        request_count=8, candidate_rows=24
    ) == 4
    c8_rollback.close()

    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS", "0")
    c7_rollback = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert c7_rollback.production_exact_target_row_counts == (8,)
    assert c7_rollback._target_group_pad_rows(
        request_count=7, candidate_rows=21
    ) == 2
    c7_rollback.close()

    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", "0")
    rollback = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert rollback.production_exact_target_row_counts == ()
    assert rollback._target_group_pad_rows(request_count=2, candidate_rows=6) == 4
    rollback.close()

    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", "1")
    strict_owner = SimpleNamespace(
        generator=SimpleNamespace(
            execution_profile="strict",
            backend="hip_gfx1100",
        ),
        capacity=8,
        _shared_runner=SimpleNamespace(hidden_size=4),
    )
    strict = Qwen35GGUFMTP2Adapter(
        strict_owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert strict.production_exact_target_row_counts == ()
    strict.close()


def test_gfx1100_wide_physical_limit_has_same_build_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1100",
            execution_profile="production",
        ),
        capacity=8,
        _shared_runner=SimpleNamespace(hidden_size=4),
    )
    monkeypatch.delenv("HIPENGINE_GGUF_SPECDEC2_MTP2_MAX_REQUESTS", raising=False)
    wide = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert wide.physical_max_requests == 8
    assert wide.physical_accept_max_rows == 36
    wide_bound = wide.physical_max_requests
    wide_rows = wide.physical_accept_max_rows
    wide.close()

    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_MTP2_MAX_REQUESTS", "4")
    rollback = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    # The rollback env caps admitted width; the exact bound and its derived
    # frontier padding follow the package policy, so assert the contract
    # (a real reduction, consistent bound) rather than a policy-dependent
    # constant that changes whenever the width/depth table is requalified.
    expected_bound = max(
        width for width, _depth in rollback._physical_width_depth_policy()
    )
    assert expected_bound <= 4
    assert rollback.physical_max_requests == expected_bound
    assert rollback.physical_request_bound == expected_bound
    assert rollback.physical_max_requests < wide_bound
    assert rollback.physical_accept_max_rows <= wide_rows
    rollback.close()


def test_gfx1100_capability_owns_one_c8_k3_frontier() -> None:
    request_ids = tuple(range(10, 18))
    targets = {
        request_id: SimpleNamespace(
            runner=SimpleNamespace(fp16_recurrent_state=False),
            target_layout=SimpleNamespace(max_sequence_length=1024),
            kv_storage_dtype="bf16",
        )
        for request_id in request_ids
    }
    rows = {
        request_id: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(session=targets[request_id]),
            slot=SimpleNamespace(),
        )
        for request_id in request_ids
    }
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 3
    adapter.target_verify_mode = "native"
    adapter.quant = "gguf_q4_k_m"
    adapter.target_key = "qwen_dense_gguf"
    adapter.provider_key = "qwen_nextn_dense"
    adapter.policy_prefix = "dense-nextn"
    adapter.physical_prompt_streaming = True
    adapter.production_physical_extra_rowtiles = True
    adapter.production_physical_q5_rowtile = True
    adapter.production_physical_q6_rowtile = True
    adapter.physical_max_requests = 8
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1100",
        execution_profile="production",
    )
    adapter.owner = SimpleNamespace(
        capacity=8,
        _row=lambda request_id: rows[int(request_id)],
    )
    adapter._intents = {request_id: 3 for request_id in request_ids}
    adapter._sampling_mode_by_request = {}
    adapter._static_eligibility_by_request = {
        request_id: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_c8",
            max_candidate_count=3,
            max_realized_group_rows=8,
            automatic_eligible=False,
            strict_fallback_key="gguf_target_ar",
            evidence_key=f"test-c8-{request_id}",
            evidence_fingerprint=f"sha256:test-c8-{request_id}",
        )
        for request_id in request_ids
    }
    adapter._disabled_requests = set()
    adapter._prompt_hidden_rows = {}
    adapter._states = {
        request_id: _MTP2RequestState(
            request_id=request_id,
            provider=SimpleNamespace(),
            provider_pool_key=None,
            provider_group_key=request_ids,
            verifier=SimpleNamespace(target_verify_mode="native"),
            root_hidden_buffer=SimpleNamespace(ptr=request_id),
        )
        for request_id in request_ids
    }
    semantics = tuple(
        SpeculativeRequestSemantics(
            request_id,
            "greedy",
            "verify_chain",
            32,
            24,
        )
        for request_id in request_ids
    )

    capability = adapter.capability(semantics)

    assert capability is not None
    assert capability.max_requests == 8
    assert capability.max_frontier_rows == 32
    expected_widths = tuple(
        width
        for width in (1, 2, 4, 8)
        if (width, capability.max_candidates_per_request)
        in adapter._physical_width_depth_policy()
    )
    assert capability.proposal_widths == expected_widths
    assert capability.target_row_buckets[-1] == 32
    assert adapter.partition_max_requests(request_ids) == 8
    assert adapter.physical_width_contract()["last_partition"] == {
        "request_ids": list(request_ids),
        "static_max_realized_group_rows": [8] * 8,
        "owner_capacity": 8,
        "physical_max_requests": 8,
        "resolved_max_requests": 8,
    }
    adapter._active_claims = None
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=request_ids,
            speculative_request_ids=request_ids,
        )
    ) is True


def test_real_adapter_requires_ar_root_and_exact_prefill_hidden_rows() -> None:
    target = SimpleNamespace(
        target_layout=SimpleNamespace(max_sequence_length=4096),
        kv_storage_dtype="bf16",
    )
    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=False,
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(generated_ids=[99]),
        prompt_ids=(10, 11, 12),
    )
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile="strict",
        ),
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: row,
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    semantics = (
        SpeculativeRequestSemantics(
            request_id=7,
            sampling_mode="greedy",
            mode="verify_chain",
            context_tokens=4,
            remaining_decode=8,
        ),
    )
    adapter.register_request(7, 3)
    adapter.observe_prefill_result(
        7,
        row.prompt_ids,
        SimpleNamespace(hidden_seeds=np.zeros((3, 4), dtype=np.float32)),
    )

    assert adapter.capability(semantics) is None
    row.first_token_emitted = True
    capability = adapter.capability(semantics)
    assert capability is not None
    assert capability.max_requests == 4
    assert capability.max_candidates_per_request == 3
    assert capability.max_frontier_rows == 16
    assert capability.max_context_tokens == 1023

    adapter.register_request(
        7,
        3,
        static_eligibility=SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_k2",
            max_candidate_count=2,
            max_realized_group_rows=4,
            automatic_eligible=False,
            strict_fallback_key="gguf_target_ar",
            evidence_key="test-k2-capability",
            evidence_fingerprint="sha256:test-k2-capability",
        ),
    )
    bounded = adapter.capability(semantics)
    assert bounded is not None
    assert bounded.max_candidates_per_request == 2
    assert bounded.max_frontier_rows == 12

    target.runner = SimpleNamespace(fp16_recurrent_state=True)
    assert adapter.capability(semantics) is None

    owner.generator.execution_profile = "production"
    owner.generator.execution_profile_fell_back_to_strict = False
    owner.generator.execution_profile_manifest_sha256 = "production-manifest"
    owner.generator.execution_profile_manifest = {
        "selections": (
            {
                "layer": "gdn_chain_recurrent_rmsnorm_gate",
                "scope": "specdec2_mtp2_target_state_rows",
                "selected_variant": "bf16_c1_exact_state_rows_tloop_fp16state",
                "strict_fallback_variant": "bf16_c1_exact_state_rows_tloop",
            },
        )
    }
    assert adapter.capability(semantics) is not None

    owner.generator.execution_profile_fell_back_to_strict = True
    assert adapter.capability(semantics) is None


def test_fp16_target_disables_c1_device_proposal_graph() -> None:
    target = SimpleNamespace(runner=SimpleNamespace(fp16_recurrent_state=True))
    assert not Qwen35GGUFMTP2Adapter._target_graph_supported(target)
    target.runner.fp16_recurrent_state = False
    assert Qwen35GGUFMTP2Adapter._target_graph_supported(target)


def test_physical_adapter_returns_device_candidate_graph_before_target(
    monkeypatch,
) -> None:
    runtime_ptrs = iter((0x9000, 0xA000))
    runtime = SimpleNamespace(
        memcpy=lambda *args: None,
        malloc=lambda nbytes: next(runtime_ptrs),
        free=lambda ptr: None,
    )
    targets = (
        SimpleNamespace(
            position=5,
            last_target_hidden=Tensor.from_handle(
                0x1100, (1, 8), DType.BF16, Device("hip", 0)
            ),
            runtime=runtime,
        ),
        SimpleNamespace(
            position=8,
            last_target_hidden=Tensor.from_handle(
                0x1200, (1, 8), DType.BF16, Device("hip", 0)
            ),
            runtime=runtime,
        ),
    )
    device_draft = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(10, 20),
        root_tokens=(100, 200),
        root_positions=(5, 8),
        candidate_counts=(1, 2),
        token_ids=Tensor.from_handle(
            0x5000, (3,), DType.INT32, Device("hip", 0)
        ),
        hidden_rows=(
            (Tensor.from_handle(0x6000, (1, 8), DType.BF16, Device("hip", 0)),),
            (
                Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x8000, (1, 8), DType.BF16, Device("hip", 0)),
            ),
        ),
    )
    calls = []
    executor = SimpleNamespace(
        hidden_size=8,
        capture_request_checkpoint=lambda request_id: f"checkpoint-{request_id}",
    )
    provider = SimpleNamespace(
        executor=executor,
        propose_batch_device=lambda context, candidate_counts: (
            calls.append(("propose", tuple(candidate_counts))) or device_draft
        ),
    )
    rows = tuple(
        SimpleNamespace(
            lease=SimpleNamespace(session=target),
            slot=SimpleNamespace(
                generated_ids=[token],
                seq_position=int(target.position),
            ),
            mtp2_proposal_batch_calls=0,
            mtp2_proposal_physical_rows=[],
            mtp2_candidate_device_handoffs=0,
        )
        for target, token in zip(targets, (100, 200), strict=True)
    )
    owner = SimpleNamespace(
        capacity=2,
        _row=lambda request_id: rows[(10, 20).index(request_id)],
        _flush_row_owner=lambda row: None,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = owner
    adapter._states = {
        request_id: _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(10, 20),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
        )
        for request_id in (10, 20)
    }
    monkeypatch.setattr(
        mtp2_module,
        "malloc",
        lambda nbytes, runtime: SimpleNamespace(ptr=0x9000, nbytes=nbytes),
    )
    monkeypatch.setattr(mtp2_module, "free", lambda buffer, runtime: None)
    plan = SimpleNamespace(
        speculative_request_ids=(10, 20),
        request_ids=(10, 20),
        candidate_counts=(1, 2),
        provider_key="nextn",
        cycle_id=3,
        resident_slots=(0, 1),
    )
    semantics = (
        SpeculativeRequestSemantics(10, "greedy", "verify_chain", 6, 8),
        SpeculativeRequestSemantics(20, "greedy", "verify_chain", 9, 8),
    )

    graph = adapter.propose_batch(plan, semantics)

    assert graph.candidate_tokens == ()
    assert graph.token_ids is device_draft.token_ids
    assert graph.candidate_counts == (1, 2)
    assert graph.provider_metadata[0] == ("candidate_handoff", "device_i32")
    assert calls == [("propose", (1, 2))]
    assert all(row.mtp2_candidate_device_handoffs == 1 for row in rows)
    assert all(
        state.proposal_device_batch is device_draft
        for state in adapter._states.values()
    )


def test_c1_adapter_warms_budget_graph_before_device_handoff() -> None:
    calls: list[tuple[object, ...]] = []
    target = SimpleNamespace(
        position=5,
        last_target_hidden=Tensor.from_handle(
            0x1100, (1, 8), DType.BF16, Device("hip", 0)
        ),
        runtime=SimpleNamespace(memcpy=lambda *args: None),
    )
    row = SimpleNamespace(
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(generated_ids=[100], seq_position=5),
    )
    draft = DraftBatch(
        request_ids=(10,),
        candidate_tokens=(101, 102),
        parent_positions=(5, 6),
        draft_depths=(1, 2),
        row_to_request=(10, 10),
        tree_parents=(-1, 0),
        active_mask=(True, True),
    )
    provider = SimpleNamespace(
        executor=SimpleNamespace(
            hidden_size=8,
            capture_request_checkpoint=lambda request_id: "checkpoint",
        ),
        propose=lambda context, **kwargs: (
            calls.append(("propose", kwargs["allow_graph"])) or draft
        ),
    )
    state = _MTP2RequestState(
        request_id=10,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(10,),
        verifier=SimpleNamespace(
            device_proposal_ready=lambda budget, remaining_decode: True
        ),
        root_hidden_buffer=SimpleNamespace(ptr=1),
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        capacity=1,
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned: None,
    )
    adapter._states = {10: state}
    adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
        Tensor.from_handle(0x2000, (1, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0x3000, (1, 8), DType.BF16, Device("hip", 0)),
    )
    plan = SimpleNamespace(
        speculative_request_ids=(10,),
        request_ids=(10,),
        candidate_counts=(2,),
        provider_key="nextn",
        cycle_id=3,
        resident_slots=(0,),
    )

    graph = adapter.propose_batch(
        plan,
        (SpeculativeRequestSemantics(10, "greedy", "verify_chain", 6, 8),),
    )

    assert calls == [("propose", True)]
    assert graph.candidate_tokens == (101, 102)
    assert state.proposal_device is None


def test_c1_adapter_carries_cached_proposal_descriptor_without_materializing_ids() -> None:
    target = SimpleNamespace(
        position=5,
        last_target_hidden=Tensor.from_handle(
            0x1100, (1, 8), DType.BF16, Device("hip", 0)
        ),
        runtime=SimpleNamespace(memcpy=lambda *args: None),
    )
    row = SimpleNamespace(
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(generated_ids=[100], seq_position=5),
        mtp2_candidate_device_handoffs=0,
    )
    proposal = Qwen35GGUFNextNDeviceProposal(
        request_id=10,
        root_token=100,
        root_position=5,
        budget=2,
        result_ptr=0x5000,
        result_nbytes=16,
        completion_event=0x6000,
        stream=0x7000,
        final_hidden=Tensor.from_handle(
            0x8000, (1, 8), DType.BF16, Device("hip", 0)
        ),
        hidden_rows=Tensor.from_handle(
            0x9000, (2, 8), DType.BF16, Device("hip", 0)
        ),
    )
    calls: list[tuple[object, ...]] = []
    provider = SimpleNamespace(
        executor=SimpleNamespace(
            hidden_size=8,
            capture_request_checkpoint=lambda request_id: "checkpoint",
        ),
        launch_device_proposal=lambda context, candidate_budget: (
            calls.append(("device", int(candidate_budget))) or proposal
        ),
        propose=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached device proposal must not materialize candidates")
        ),
    )
    state = _MTP2RequestState(
        request_id=10,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(10,),
        verifier=SimpleNamespace(
            device_proposal_ready=lambda budget, remaining_decode: True
        ),
        root_hidden_buffer=SimpleNamespace(ptr=1),
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        capacity=1,
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned: None,
    )
    adapter._states = {10: state}
    adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
        Tensor.from_handle(0x2000, (1, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0x3000, (1, 8), DType.BF16, Device("hip", 0)),
    )
    plan = SimpleNamespace(
        speculative_request_ids=(10,),
        request_ids=(10,),
        candidate_counts=(2,),
        provider_key="nextn",
        cycle_id=3,
        resident_slots=(0,),
    )

    graph = adapter.propose_batch(
        plan,
        (SpeculativeRequestSemantics(10, "greedy", "verify_chain", 6, 8),),
    )

    assert calls == [("device", 2)]
    assert graph.candidate_tokens == ()
    assert graph.token_ids is not None
    assert graph.token_ids.ptr == proposal.result_ptr
    assert graph.token_ids.shape == (2,)
    assert graph.token_ids.strides == (2,)
    assert state.proposal_device is proposal


def test_packed_target_device_result_binds_identity_and_only_device_rows() -> None:
    result = runner_mod.Qwen35GGUFPackedVerifyDeviceResult(
        request_id=10,
        resident_slot=3,
        transaction_id=7,
        start_position=5,
        row_start=2,
        row_end=5,
        input_token_ids=Tensor.from_handle(
            0x1000, (3,), DType.INT64, Device("hip", 0)
        ),
        target_top1=Tensor.from_handle(
            0x2000, (3,), DType.INT32, Device("hip", 0)
        ),
        hidden_seeds=Tensor.from_handle(
            0x3000, (3, 8), DType.FP32, Device("hip", 0)
        ),
        deferred_packed_state=object(),
        pre_output_norm_hidden=Tensor.from_handle(
            0x4000, (3, 8), DType.BF16, Device("hip", 0)
        ),
    )

    assert result.rows == 3
    assert result.request_id == 10
    assert result.transaction_id == 7
    assert result.pre_output_norm_hidden is not None
    assert result.pre_output_norm_hidden.ptr == 0x4000
    assert not hasattr(result, "token_ids")


def test_device_chain_oracle_trace_preserves_per_request_proposal_and_target_rows() -> None:
    draft = DraftBatch(
        request_ids=(10, 20),
        candidate_tokens=(101, 102, 201),
        parent_positions=(5, 6, 8),
        draft_depths=(1, 2, 1),
        row_to_request=(10, 10, 20),
        tree_parents=(-1, 0, -1),
        active_mask=(True, True, True),
    )
    batch = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(100, 200),
        root_positions=(5, 8),
    )
    top1 = [0] * batch.rows
    candidate_by_id = {
        request_id: tuple(
            sorted(
                (
                    row
                    for row in batch.candidate_rows
                    if batch.row_to_request[row] == request_id
                ),
                key=lambda row: batch.draft_depths[row],
            )
        )
        for request_id in batch.request_ids
    }
    root_by_id = dict(zip(batch.request_ids, batch.root_rows, strict=True))
    for row, value in zip(
        (root_by_id[10], *candidate_by_id[10]),
        (101, 102, 999),
        strict=True,
    ):
        top1[row] = value
    for row, value in zip(
        (root_by_id[20], *candidate_by_id[20]),
        (201, 888),
        strict=True,
    ):
        top1[row] = value
    summary = TargetAcceptSummary(
        request_ids=(10, 20),
        accepted_counts=(2, 0),
        accepted_tokens=((101, 102), ()),
        commit_rows=(candidate_by_id[10][-1], root_by_id[20]),
        commit_tokens=(102, 200),
        commit_positions=(7, 8),
        full_accept=(True, False),
        next_tokens=(999, 201),
        candidate_counts=(2, 1),
        transaction_id=7,
    )

    traces = mtp2_module._device_chain_oracle_trace_rows(
        batch,
        top1,
        summary,
        cycle_id=3,
    )

    assert traces == (
        {
            "cycle_id": 3,
            "request_id": 10,
            "root_token": 100,
            "root_position": 5,
            "candidate_tokens": [101, 102],
            "target_top1": [101, 102, 999],
            "accepted_count": 2,
            "accepted_tokens": [101, 102],
            "next_token": 999,
        },
        {
            "cycle_id": 3,
            "request_id": 20,
            "root_token": 200,
            "root_position": 8,
            "candidate_tokens": [201],
            "target_top1": [201, 888],
            "accepted_count": 0,
            "accepted_tokens": [],
            "next_token": 201,
        },
    )


def test_physical_accept_enqueue_keeps_candidate_and_target_ids_on_device(
    monkeypatch,
) -> None:
    draft = DraftBatch(
        request_ids=(10, 20),
        candidate_tokens=(0, 0),
        parent_positions=(5, 8),
        draft_depths=(1, 1),
        row_to_request=(10, 20),
        tree_parents=(-1, -1),
        active_mask=(True, True),
    )
    batch = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(100, 200),
        root_positions=(5, 8),
    )
    proposal = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(10, 20),
        root_tokens=(100, 200),
        root_positions=(5, 8),
        candidate_counts=(1, 1),
        token_ids=Tensor.from_handle(
            0x5000, (2,), DType.INT32, Device("hip", 0)
        ),
        hidden_rows=(
            (Tensor.from_handle(0x6000, (1, 8), DType.BF16, Device("hip", 0)),),
            (Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0)),),
        ),
    )
    results = (
        runner_mod.Qwen35GGUFPackedVerifyDeviceResult(
            request_id=10,
            resident_slot=0,
            transaction_id=7,
            start_position=5,
            row_start=0,
            row_end=2,
            input_token_ids=Tensor.from_handle(
                0x8000, (2,), DType.INT64, Device("hip", 0)
            ),
            target_top1=Tensor.from_handle(
                0x9000, (2,), DType.INT32, Device("hip", 0)
            ),
            hidden_seeds=Tensor.from_handle(
                0xA000, (2, 8), DType.FP32, Device("hip", 0)
            ),
            deferred_packed_state=object(),
        ),
        runner_mod.Qwen35GGUFPackedVerifyDeviceResult(
            request_id=20,
            resident_slot=1,
            transaction_id=7,
            start_position=8,
            row_start=2,
            row_end=4,
            input_token_ids=Tensor.from_handle(
                0x8100, (2,), DType.INT64, Device("hip", 0)
            ),
            target_top1=Tensor.from_handle(
                0x9100, (2,), DType.INT32, Device("hip", 0)
            ),
            hidden_seeds=Tensor.from_handle(
                0xA100, (2, 8), DType.FP32, Device("hip", 0)
            ),
            deferred_packed_state=object(),
        ),
    )
    pointer = iter(range(0xB000, 0xD000, 0x100))

    def tensor(shape, dtype=DType.INT32):
        return Tensor.from_handle(next(pointer), shape, dtype, Device("hip", 0))

    buffers = TargetVerifyBuffers.for_batch(
        batch,
        token_ids=tensor((4,)),
        positions=tensor((4,)),
        parent_rows=tensor((4,)),
        draft_depths=tensor((4,)),
        row_to_request=tensor((4,)),
        active_mask=tensor((4,), DType.BOOL),
        target_top1=tensor((4,)),
        accepted_counts=tensor((2,)),
        commit_rows=tensor((2,)),
        commit_tokens=tensor((2,)),
        commit_positions=tensor((2,)),
        next_tokens=tensor((2,)),
        full_accept=tensor((2,), DType.BOOL),
        committed_output_ids=Tensor.from_handle(
            next(pointer),
            (2, 4),
            DType.INT32,
            Device("hip", 0),
            strides=(16, 1),
        ),
        committed_output_lengths=tensor((2,)),
        transaction_id=7,
    )
    owner = SimpleNamespace(bind=lambda bound, transaction_id: buffers)
    remaining = tensor((4,))
    payload = tensor((4, 7))
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        compiler_version=None,
        require_cached_build=True,
    )
    adapter._batch_accept_library = object()
    adapter._batch_accept_resources = lambda runtime: (owner, remaining, payload)
    uploads: list[tuple[int, tuple[int, ...]]] = []
    adapter._upload_accept_array = lambda tensor, values, runtime: uploads.append(
        (int(tensor.ptr), tuple(int(value) for value in np.asarray(values).reshape(-1)))
    )
    launches: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        mtp2_module,
        "dflash_accept_chain_i32_packed",
        lambda *args, **kwargs: launches.append(args),
    )

    class Runtime:
        def __init__(self) -> None:
            self.device_copies: list[tuple[int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.device_copies.append((int(dst), int(src), int(nbytes)))

        def device_synchronize(self):
            raise AssertionError("accept enqueue must not synchronize")

    runtime = Runtime()

    pending = adapter._enqueue_target_batch_accept(
        batch,
        proposal=proposal,
        target_results=results,
        remaining_decode=(3, 3),
        transaction_id=7,
        runtime=runtime,
    )

    assert pending.buffers is buffers
    assert len(launches) == 1
    assert proposal.token_ids.ptr in {src for _dst, src, _nbytes in runtime.device_copies}
    assert {result.target_top1.ptr for result in results}.issubset(
        {src for _dst, src, _nbytes in runtime.device_copies}
    )
    assert (buffers.token_ids.ptr, (100, 200)) in uploads
    assert all(ptr != buffers.target_top1.ptr for ptr, _values in uploads)


def test_physical_adapter_emits_one_gpu_accept_payload_for_the_group(
    monkeypatch,
) -> None:
    draft = DraftBatch(
        request_ids=(10, 20),
        candidate_tokens=(101, 201),
        parent_positions=(5, 8),
        draft_depths=(1, 1),
        row_to_request=(10, 20),
        tree_parents=(-1, -1),
        active_mask=(True, True),
    )
    batch = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(100, 200),
        root_positions=(5, 8),
    )
    target_top1 = (101, 999, 303, 404)
    remaining = (3, 3)
    expected_accept = batch.accept_from_top1(
        target_top1,
        transaction_id=7,
        remaining_decode=remaining,
    )
    expected = mtp2_module.TargetAcceptSummary.from_accept_result(
        batch,
        expected_accept,
    )
    payload_host = np.asarray(
        [
            [
                expected.accepted_counts[index],
                expected.commit_rows[index],
                expected.commit_tokens[index],
                expected.commit_positions[index],
                -1 if expected.next_tokens[index] is None else expected.next_tokens[index],
                int(expected.full_accept[index]),
                expected.accepted_counts[index] + 1,
            ]
            for index in range(2)
        ],
        dtype=np.int32,
    )
    pointer = iter(range(0x1000, 0x3000, 0x100))

    def tensor(shape, dtype=DType.INT32):
        return Tensor.from_handle(next(pointer), shape, dtype, Device("hip", 0))

    buffers = TargetVerifyBuffers.for_batch(
        batch,
        token_ids=tensor((4,)),
        positions=tensor((4,)),
        parent_rows=tensor((4,)),
        draft_depths=tensor((4,)),
        row_to_request=tensor((4,)),
        active_mask=tensor((4,), DType.BOOL),
        target_top1=tensor((4,)),
        accepted_counts=tensor((2,)),
        commit_rows=tensor((2,)),
        commit_tokens=tensor((2,)),
        commit_positions=tensor((2,)),
        next_tokens=tensor((2,)),
        full_accept=tensor((2,), DType.BOOL),
        committed_output_ids=tensor((2, 4)),
        committed_output_lengths=tensor((2,)),
        transaction_id=7,
    )
    owner = SimpleNamespace(bind=lambda bound, transaction_id: buffers)
    remaining_tensor = tensor((4,))
    payload_tensor = tensor((4, 7))
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        compiler_version=None,
        require_cached_build=True,
    )
    adapter._batch_accept_library = object()
    adapter._batch_accept_resources = lambda runtime: (
        owner,
        remaining_tensor,
        payload_tensor,
    )
    adapter._upload_accept_array = lambda tensor, values, runtime: None
    calls = []
    monkeypatch.setattr(
        mtp2_module,
        "dflash_accept_chain_i32_packed",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        mtp2_module,
        "copy_device_to_host",
        lambda destination, source, nbytes, *, runtime: ctypes.memmove(
            destination,
            payload_host.ctypes.data,
            nbytes,
        ),
    )
    runtime = SimpleNamespace(device_synchronize=lambda: calls.append(("sync",)))

    summary, actual_buffers = adapter._accept_target_batch_on_device(
        batch,
        target_top1,
        remaining,
        transaction_id=7,
        runtime=runtime,
    )

    assert actual_buffers is buffers
    assert summary.accepted_counts == (1, 0)
    assert summary.accepted_tokens == ((101,), ())
    assert summary.commit_rows == expected.commit_rows
    assert summary.next_tokens == expected.next_tokens
    assert len([call for call in calls if call != ("sync",)]) == 1


def test_packed_owner_commits_selected_linear_rows_once_for_the_group(
    monkeypatch,
) -> None:
    class Buffer:
        def __init__(self, ptr, nbytes):
            self.ptr = int(ptr)
            self.nbytes = int(nbytes)

    class PackedState:
        pass

    monkeypatch.setattr(runner_mod, "_GGUFPackedTargetState", PackedState)
    monkeypatch.setattr(runner_mod, "set_decode_position_i64", lambda *args, **kwargs: None)
    copies = []
    runtime = SimpleNamespace(memcpy_async=lambda *args: None)
    owner = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    weights = SimpleNamespace(
        config=SimpleNamespace(layer_types=(runner_mod.LINEAR_ATTENTION,))
    )
    owner.runner = SimpleNamespace(weights=weights, hidden_size=2)
    owner.runtime = runtime
    owner._verify_hidden_seed_buf = Buffer(0x1000, 6 * 8)
    owner._packed_verify_max_written_positions = (0, 0)
    owner._verify_linear_state_row_pair = lambda layer_id: (
        Buffer(0x2000, 6 * 16),
        Buffer(0x3000, 6 * 32),
    )
    owner._fused_linear_state_pair_copy = lambda entries, **kwargs: (
        copies.extend(entries) or True
    )
    sessions = []
    for index in range(2):
        session = SimpleNamespace(
            runner=owner.runner,
            scratch=SimpleNamespace(
                hidden_seed_fp32=Buffer(0x4000 + index * 0x100, 8),
                layer_conv_states=(Buffer(0x5000 + index * 0x100, 16),),
                layer_recurrent_states=(Buffer(0x6000 + index * 0x100, 32),),
                position_host=np.asarray([0], dtype=np.int64),
                context_host=np.asarray([1], dtype=np.int64),
                position_buf=Buffer(0x7000 + index * 0x100, 8),
                context_buf=Buffer(0x8000 + index * 0x100, 8),
            ),
            _runtime_state_library=object(),
            _verify_hidden_seed_buf=None,
            _ensure_verify_block_buffers=lambda rows, runtime, session_index=index: None,
            _verify_hidden_seed_rows_populated=0,
            _hidden_seed_fp32_populated=False,
            _position=0,
        )
        session._verify_hidden_seed_buf = Buffer(0x9000 + index * 0x100, 3 * 8)
        sessions.append(session)
    packed = PackedState()
    results = (
        SimpleNamespace(
            token_ids=[1, 2, 3],
            deferred_packed_state=SimpleNamespace(
                owner=owner,
                packed_state=packed,
                row_start=0,
                row_end=3,
                slot_index=0,
                start_position=5,
            ),
        ),
        SimpleNamespace(
            token_ids=[4, 5, 6],
            deferred_packed_state=SimpleNamespace(
                owner=owner,
                packed_state=packed,
                row_start=3,
                row_end=6,
                slot_index=1,
                start_position=8,
            ),
        ),
    )
    accept_buffers = SimpleNamespace(
        accepted_counts=Tensor.from_handle(
            0xA000, (2,), DType.INT32, Device("hip", 0)
        )
    )

    contract = owner._commit_deferred_packed_verify_states_batch(
        results,
        sessions,
        accepted_counts=(2, 0),
        accept_buffers=accept_buffers,
    )

    assert contract["requests"] == 2
    assert contract["fused_linear_state_commit"] is True
    assert len(copies) == 2
    assert copies[0][0] == 0x2000 + 2 * 16
    assert copies[0][2] == 0x3000 + 2 * 32
    assert copies[1][0] == 0x2000 + 3 * 16
    assert copies[1][2] == 0x3000 + 3 * 32
    assert [session._position for session in sessions] == [8, 9]


def test_packed_owner_device_commit_selects_from_accept_buffers_before_readback(
    monkeypatch,
) -> None:
    class PackedState:
        pass

    monkeypatch.setattr(runner_mod, "_GGUFPackedTargetState", PackedState)
    packed = PackedState()
    calls: list[tuple[object, ...]] = []
    owner = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(layer_types=(runner_mod.FULL_ATTENTION,))
        )
    )
    owner._packed_verify_max_written_positions = (0, 0)
    owner.runtime = SimpleNamespace()
    owner._copy_session_packed_kv_segments = (
        lambda session, packed_state, slot_index, layer_id, **kwargs: calls.append(
            (
                "kv",
                session.request_id,
                packed_state,
                int(slot_index),
                int(layer_id),
                int(kwargs["rows"]),
            )
        )
    )
    sessions = tuple(
        SimpleNamespace(
            request_id=request_id,
            runner=owner.runner,
            scratch=object(),
            _commit_external_verify_state_row_device=(
                lambda source_owner, *, request_id=request_id, **kwargs: calls.append(
                    (
                        "state",
                        request_id,
                        source_owner,
                        int(kwargs["row_start"]),
                        int(kwargs["rows"]),
                        int(kwargs["commit_row_i32_ptr"]),
                        int(kwargs["commit_position_i32_ptr"]),
                    )
                )
            ),
            _commit_external_pre_output_norm_hidden_row_device=(
                lambda source_rows, *, request_id=request_id, **kwargs: calls.append(
                    (
                        "hidden",
                        request_id,
                        int(source_rows.ptr),
                        int(kwargs["commit_row_i32_ptr"]),
                        int(kwargs["commit_position_i32_ptr"]),
                    )
                )
            ),
        )
        for request_id in (10, 20)
    )
    results = tuple(
        runner_mod.Qwen35GGUFPackedVerifyDeviceResult(
            request_id=request_id,
            resident_slot=index,
            transaction_id=7,
            start_position=start,
            row_start=index * 3,
            row_end=index * 3 + 3,
            input_token_ids=Tensor.from_handle(
                0x1000 + index * 0x100,
                (3,),
                DType.INT64,
                Device("hip", 0),
            ),
            target_top1=Tensor.from_handle(
                0x2000 + index * 0x100,
                (3,),
                DType.INT32,
                Device("hip", 0),
            ),
            hidden_seeds=Tensor.from_handle(
                0x3000 + index * 0x100,
                (3, 8),
                DType.FP32,
                Device("hip", 0),
            ),
            pre_output_norm_hidden=Tensor.from_handle(
                0x4000 + index * 0x100,
                (3, 8),
                DType.BF16,
                Device("hip", 0),
            ),
            deferred_packed_state=SimpleNamespace(
                owner=owner,
                packed_state=packed,
                slot_index=index,
                row_start=index * 3,
                row_end=index * 3 + 3,
                start_position=start,
                end_position=start + 3,
            ),
        )
        for index, (request_id, start) in enumerate(((10, 5), (20, 8)))
    )
    accept_buffers = SimpleNamespace(
        accepted_counts=Tensor.from_handle(
            0xA000, (2,), DType.INT32, Device("hip", 0)
        ),
        commit_positions=Tensor.from_handle(
            0xB000, (2,), DType.INT32, Device("hip", 0)
        ),
    )

    contract = owner._commit_deferred_packed_verify_states_batch_device(
        results,
        sessions,
        accept_buffers=accept_buffers,
    )

    assert contract["requests"] == 2
    assert contract["accepted_counts_device_ptr"] == 0xA000
    assert calls == [
        ("kv", 10, packed, 0, 0, 3),
        ("state", 10, owner, 0, 3, 0xA000, 0xB000),
        ("hidden", 10, 0x4000, 0xA000, 0xB000),
        ("kv", 20, packed, 1, 0, 3),
        ("state", 20, owner, 3, 3, 0xA004, 0xB004),
        ("hidden", 20, 0x4100, 0xA004, 0xB004),
    ]


def test_adapter_recovers_only_precommit_failure_with_canonical_target_cursors() -> None:
    rows = {
        10: SimpleNamespace(
            slot=SimpleNamespace(seq_position=7),
            lease=SimpleNamespace(session=SimpleNamespace(position=7)),
            mtp2_recoverable_failures=0,
            mtp2_failure_reasons=[],
        ),
        20: SimpleNamespace(
            slot=SimpleNamespace(seq_position=9),
            lease=SimpleNamespace(session=SimpleNamespace(position=9)),
            mtp2_recoverable_failures=0,
            mtp2_failure_reasons=[],
        ),
    }
    rebuilds = []
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: rows[request_id],
        restore_speculative_target_rows=lambda plan: rebuilds.append(plan) or True,
    )
    plan = SimpleNamespace(speculative_request_ids=(10, 20))

    assert adapter.recover_cycle_failure(plan, RuntimeError("injected")) is True
    assert all(row.mtp2_recoverable_failures == 1 for row in rows.values())
    assert all(
        row.mtp2_failure_reasons
        == ["precommit_failure_ar_fallback", "RuntimeError:injected"]
        for row in rows.values()
    )

    assert (
        adapter.recover_cycle_failure(
            plan,
            _PhysicalTargetCommitError("selected target state may be committed"),
        )
        is True
    )
    assert rebuilds == [plan]
    assert all(row.mtp2_recoverable_failures == 2 for row in rows.values())
    assert all(
        row.mtp2_failure_reasons[-2:] == [
            "postcommit_target_rebuild_ar_fallback",
            "_PhysicalTargetCommitError:selected target state may be committed",
        ]
        for row in rows.values()
    )

    rows[20].lease.session.position = 10
    assert adapter.recover_cycle_failure(plan, RuntimeError("late")) is False


def test_model_runner_rebuilds_postcommit_targets_from_canonical_tokens() -> None:
    calls = []

    class Session:
        def __init__(self) -> None:
            self.position = 99

        def reset(self) -> None:
            calls.append("reset")
            self.position = 0

    sessions = (Session(), Session())
    rows = {
        10: SimpleNamespace(
            request_id=10,
            prompt_ids=(1, 2),
            slot=SimpleNamespace(
                generated_ids=[101, 102],
                prev_token=102,
                seq_position=3,
            ),
            lease=SimpleNamespace(session=sessions[0]),
        ),
        20: SimpleNamespace(
            request_id=20,
            prompt_ids=(3, 4, 5),
            slot=SimpleNamespace(
                generated_ids=[201, 202],
                prev_token=202,
                seq_position=4,
            ),
            lease=SimpleNamespace(session=sessions[1]),
        ),
    }

    class PackedOwner:
        def prefill_batch_native(self, prompts, **kwargs):
            calls.append((tuple(tuple(row) for row in prompts), kwargs))
            for session, tokens in zip(kwargs["sessions"], prompts, strict=True):
                session.position = len(tokens)
            return (SimpleNamespace(token_id=102), SimpleNamespace(token_id=202))

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._rows = rows
    runner._flush_rows = lambda selected: calls.append(
        ("flush", tuple(row.request_id for row in selected))
    )
    runner._packed_execution_owner = lambda session: PackedOwner()
    plan = SimpleNamespace(speculative_request_ids=(10, 20))

    assert runner.restore_speculative_target_rows(plan) is True
    assert calls[0] == ("flush", (10, 20))
    assert calls[1:3] == ["reset", "reset"]
    prompts, kwargs = calls[3]
    assert prompts == ((1, 2, 101), (3, 4, 5, 201))
    assert kwargs["full_prompt_lengths"] == [3, 4]
    assert kwargs["return_logits"] is False
    assert kwargs["return_hidden_seeds"] is False
    assert tuple(session.position for session in sessions) == (3, 4)


def test_model_runner_rebuilds_postcommit_targets_on_the_strict_route_when_packed_refuses() -> None:
    """A packed shape refusal must not fail a cycle that already committed.

    The rebuild runs after the cycle published tokens to the client stream, so
    a ``NotImplementedError`` here used to end a live response with
    ``unsupported_parameter``. The rows must instead rebuild on the registered
    strict per-session route with the same tokens and cursors.
    """

    calls: list[tuple] = []

    class Session:
        def __init__(self, next_token: int) -> None:
            self.position = 99
            self.next_token = int(next_token)

        def reset(self) -> None:
            calls.append("reset")
            self.position = 0

        def prefill(self, tokens, *, return_logits=False):
            calls.append(("prefill", tuple(tokens), bool(return_logits)))
            self.position = len(tokens)
            return SimpleNamespace(token_id=self.next_token)

    sessions = (Session(102), Session(202))
    rows = {
        10: SimpleNamespace(
            request_id=10,
            prompt_ids=(1, 2),
            slot=SimpleNamespace(
                generated_ids=[101, 102],
                prev_token=102,
                seq_position=3,
            ),
            lease=SimpleNamespace(session=sessions[0]),
        ),
        20: SimpleNamespace(
            request_id=20,
            prompt_ids=(3, 4, 5),
            slot=SimpleNamespace(
                generated_ids=[201, 202],
                prev_token=202,
                seq_position=4,
            ),
            lease=SimpleNamespace(session=sessions[1]),
        ),
    }

    class PackedOwner:
        def prefill_batch_native(self, prompts, **kwargs):
            calls.append("prefill_batch_native")
            raise NotImplementedError(
                "packed paged AR prefill currently requires context < 1024"
            )

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._rows = rows
    runner._flush_rows = lambda selected: calls.append(
        ("flush", tuple(row.request_id for row in selected))
    )
    runner._packed_execution_owner = lambda session: PackedOwner()
    plan = SimpleNamespace(speculative_request_ids=(10, 20))

    assert runner.restore_speculative_target_rows(plan) is True
    assert calls[0] == ("flush", (10, 20))
    assert calls[1:3] == ["reset", "reset"]
    assert calls[3:] == [
        "prefill_batch_native",
        ("prefill", (1, 2, 101), False),
        ("prefill", (3, 4, 5, 201), False),
    ]
    assert tuple(session.position for session in sessions) == (3, 4)


def test_model_runner_postcommit_rebuild_stays_fail_closed_for_block_table_sessions() -> None:
    """Only the block-table-aware route can serve those sessions, so refuse."""

    calls: list[str] = []

    class Session:
        kv_attention_source = "int8_direct"
        position = 0

        def reset(self) -> None:
            calls.append("reset")

        def prefill(self, tokens, *, return_logits=False):
            calls.append("prefill")
            return SimpleNamespace(token_id=1)

    session = Session()
    rows = {
        10: SimpleNamespace(
            request_id=10,
            prompt_ids=(1, 2),
            slot=SimpleNamespace(
                generated_ids=[101, 102],
                prev_token=102,
                seq_position=3,
            ),
            lease=SimpleNamespace(session=session),
        ),
    }

    class PackedOwner:
        def prefill_batch_native(self, prompts, **kwargs):
            raise NotImplementedError(
                "packed paged AR prefill currently requires context < 1024"
            )

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._rows = rows
    runner._flush_rows = lambda selected: None
    runner._packed_execution_owner = lambda session: PackedOwner()
    plan = SimpleNamespace(speculative_request_ids=(10,))

    assert runner.restore_speculative_target_rows(plan) is False
    assert calls == ["reset"]


def test_model_runner_production_rebuild_keeps_scheduler_token_on_near_tie() -> None:
    session = SimpleNamespace(position=5, reset=lambda: None)
    row = SimpleNamespace(
        request_id=7,
        prompt_ids=(1, 2),
        slot=SimpleNamespace(
            generated_ids=[101, 102],
            prev_token=102,
            seq_position=3,
        ),
        lease=SimpleNamespace(session=session),
    )
    owner = SimpleNamespace(
        prefill_batch_native=lambda prompts, **kwargs: (
            setattr(session, "position", len(prompts[0]))
            or SimpleNamespace(token_id=999)
        ,),
    )
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._rows = {7: row}
    runner._flush_rows = lambda rows: None
    runner._packed_execution_owner = lambda selected: owner

    assert runner.restore_speculative_target_request_ids(
        (7,),
        require_token_match=False,
    ) is True
    assert row.slot.prev_token == 102
    assert session.position == row.slot.seq_position == 3


@pytest.mark.parametrize(
    ("accepted", "expected_inputs", "full_tail"),
    [
        (0, [(90, 10)], False),
        (1, [(90, 10), (101, 11)], False),
        (2, [(90, 10), (101, 11), (102, 12)], False),
        (3, [], True),
    ],
)
def test_provider_repair_restores_and_replays_only_committed_prefix(
    accepted,
    expected_inputs,
    full_tail,
) -> None:
    calls = []

    class Executor:
        def restore_request_checkpoint(self, checkpoint):
            calls.append(("restore", checkpoint))

        def advance_state_only(self, request_id, token_id, position, hidden):
            calls.append(("advance", request_id, token_id, position, hidden))

    results = tuple(
        SimpleNamespace(token_id=101 + index, hidden=f"h{index}")
        for index in range(3)
    )
    provider = SimpleNamespace(
        executor=Executor(),
        last_results={7: results},
        advance_full_accept_tail=lambda request_id, accepted_count: calls.append(
            ("full", request_id, accepted_count)
        ),
    )
    state = _MTP2RequestState(
        request_id=7,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=SimpleNamespace(),
        root_hidden_buffer=SimpleNamespace(ptr=1),
        proposal_checkpoint="checkpoint",
        proposal_context=MtpProposalContext(
            request_ids=(7,),
            root_tokens=(90,),
            root_positions=(10,),
            target_hidden=SimpleNamespace(ndim=2, shape=(1, 4)),
        ),
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)

    adapter._repair_provider_state(
        state,
        accepted_count=accepted,
        candidate_count=3,
    )

    if full_tail:
        assert calls == [("full", 7, 3)]
    else:
        assert calls[0] == ("restore", "checkpoint")
        actual_inputs = [(call[2], call[3]) for call in calls[1:]]
        assert actual_inputs == expected_inputs


@pytest.mark.parametrize(
    ("accepted", "expected"),
    [
        (0, [("restore", "checkpoint"), ("host", 90, 10)]),
        (
            1,
            [
                ("restore", "checkpoint"),
                ("host", 90, 10),
                ("device", 0x5000, 11, 0x6000),
            ],
        ),
        (2, [("device", 0x5008, 12, 0x6010)]),
    ],
)
def test_provider_c1_device_repair_uses_only_retained_device_rows(
    accepted,
    expected,
) -> None:
    calls: list[tuple[object, ...]] = []

    class Executor:
        def restore_request_checkpoint(self, checkpoint):
            calls.append(("restore", checkpoint))

        def advance_state_only(self, request_id, token_id, position, hidden):
            calls.append(("host", int(token_id), int(position)))

        def advance_state_only_device(self, request_id, token_id, position, hidden):
            calls.append(
                ("device", int(token_id.ptr), int(position), int(hidden.ptr))
            )

    proposal = Qwen35GGUFNextNDeviceProposal(
        request_id=7,
        root_token=90,
        root_position=10,
        budget=2,
        result_ptr=0x5000,
        result_nbytes=16,
        completion_event=0x7000,
        stream=0x8000,
        final_hidden=Tensor.from_handle(
            0x6020, (1, 8), DType.BF16, Device("hip", 0)
        ),
        hidden_rows=Tensor.from_handle(
            0x6000, (2, 8), DType.BF16, Device("hip", 0)
        ),
    )
    state = _MTP2RequestState(
        request_id=7,
        provider=SimpleNamespace(executor=Executor(), last_results={}),
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=SimpleNamespace(),
        root_hidden_buffer=SimpleNamespace(ptr=1),
        proposal_checkpoint="checkpoint",
        proposal_context=MtpProposalContext(
            request_ids=(7,),
            root_tokens=(90,),
            root_positions=(10,),
            target_hidden=Tensor.from_handle(
                0x9000, (1, 8), DType.BF16, Device("hip", 0)
            ),
        ),
        proposal_device=proposal,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)

    adapter._repair_provider_state_device(
        state,
        proposal,
        accepted_count=accepted,
    )

    assert calls == expected
    assert state.provider.last_results == {}


def test_provider_batch_repair_shares_full_accept_tail_and_rejected_root(
    monkeypatch,
) -> None:
    calls = []

    class Executor:
        hidden_size = 8
        _ptrs = iter((0x3000, 0x4000))
        runtime = SimpleNamespace(
            memcpy=lambda *args: None,
            malloc=lambda nbytes: next(Executor._ptrs),
            free=lambda ptr: None,
        )

        def restore_request_checkpoint(self, checkpoint):
            calls.append(("restore", checkpoint))

        def advance_state_batch_only(
            self,
            request_ids,
            token_ids,
            positions,
            target_hidden,
        ):
            calls.append(
                (
                    "batch",
                    tuple(request_ids),
                    tuple(token_ids),
                    tuple(positions),
                    target_hidden.shape,
                )
            )

    executor = Executor()
    results = {
        1: (
            SimpleNamespace(token_id=101, position=5, hidden=SimpleNamespace(ptr=1101)),
            SimpleNamespace(token_id=102, position=6, hidden=SimpleNamespace(ptr=1102)),
        ),
        2: (
            SimpleNamespace(token_id=201, position=8, hidden=SimpleNamespace(ptr=1201)),
            SimpleNamespace(token_id=202, position=9, hidden=SimpleNamespace(ptr=1202)),
        ),
    }
    provider = SimpleNamespace(executor=executor, last_results=results)
    states = tuple(
        _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(1, 2),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
            proposal_checkpoint=f"checkpoint-{request_id}",
            proposal_context=MtpProposalContext(
                request_ids=(request_id,),
                root_tokens=((90,) if request_id == 1 else (190,)),
                root_positions=((5,) if request_id == 1 else (8,)),
                target_hidden=SimpleNamespace(
                    ptr=2000 + request_id,
                    ndim=2,
                    shape=(1, 8),
                ),
            ),
        )
        for request_id in (1, 2)
    )
    monkeypatch.setattr(
        mtp2_module,
        "malloc",
        lambda nbytes, runtime: SimpleNamespace(ptr=3000, nbytes=nbytes),
    )
    monkeypatch.setattr(mtp2_module, "free", lambda buffer, runtime: None)
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=2)

    adapter._repair_provider_states_batch(
        states,
        accepted_counts=(2, 0),
        candidate_counts=(2, 2),
    )

    assert calls == [
        ("restore", "checkpoint-2"),
        ("batch", (1, 2), (102, 190), (7, 8), (2, 8)),
    ]


def test_provider_batch_device_repair_never_materializes_candidate_ids() -> None:
    calls: list[tuple[object, ...]] = []

    class Executor:
        hidden_size = 8
        runtime = SimpleNamespace(memcpy=lambda *args: None)

        def restore_request_checkpoint(self, checkpoint):
            calls.append(("restore", checkpoint))

        def advance_state_batch_only(self, request_ids, token_ids, positions, hidden):
            calls.append(
                (
                    "host",
                    tuple(request_ids),
                    tuple(token_ids),
                    tuple(positions),
                    hidden.shape,
                )
            )

        def advance_state_batch_only_device(
            self,
            request_ids,
            token_ids,
            positions,
            hidden,
        ):
            calls.append(
                (
                    "device",
                    tuple(request_ids),
                    tuple((token.ptr, token.shape) for token in token_ids),
                    tuple(positions),
                    hidden.shape,
                )
            )

    executor = Executor()
    provider = SimpleNamespace(executor=executor, last_results={})
    proposal = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(1, 2),
        root_tokens=(90, 190),
        root_positions=(5, 8),
        candidate_counts=(2, 2),
        token_ids=Tensor.from_handle(
            0x5000, (4,), DType.INT32, Device("hip", 0)
        ),
        hidden_rows=(
            (
                Tensor.from_handle(0x6000, (1, 8), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x6100, (1, 8), DType.BF16, Device("hip", 0)),
            ),
            (
                Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x7100, (1, 8), DType.BF16, Device("hip", 0)),
            ),
        ),
    )
    states = tuple(
        _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(1, 2),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
            proposal_checkpoint=f"checkpoint-{request_id}",
            proposal_context=MtpProposalContext(
                request_ids=(request_id,),
                root_tokens=((90,) if request_id == 1 else (190,)),
                root_positions=((5,) if request_id == 1 else (8,)),
                target_hidden=Tensor.from_handle(
                    0x8000 + request_id * 0x100,
                    (1, 8),
                    DType.BF16,
                    Device("hip", 0),
                ),
            ),
            proposal_device_batch=proposal,
        )
        for request_id in (1, 2)
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=2)
    adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
        Tensor.from_handle(0x9000, (2, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0xA000, (2, 8), DType.BF16, Device("hip", 0)),
    )

    adapter._repair_provider_states_batch_device(
        states,
        proposal,
        accepted_counts=(2, 0),
    )

    assert calls[0] == ("restore", "checkpoint-2")
    assert calls[1][:4] == ("host", (2,), (190,), (8,))
    assert calls[2][0] == "device"
    assert calls[2][1] == (1,)
    assert calls[2][2] == ((0x5004, (1,)),)
    assert calls[2][3] == (7,)
    assert provider.last_results == {}


def test_provider_batch_device_repair_uses_root_snapshot_and_kminus1_state() -> None:
    calls: list[tuple[object, ...]] = []

    class Executor:
        hidden_size = 8
        runtime = SimpleNamespace(memcpy=lambda *args: None)

        def restore_request_root_state(self, request_id):
            calls.append(("root_snapshot", int(request_id)))

        def restore_request_checkpoint(self, checkpoint):
            calls.append(("checkpoint", checkpoint))

        def advance_state_batch_only(self, request_ids, token_ids, positions, hidden):
            calls.append(("host", tuple(request_ids), tuple(token_ids), tuple(positions)))

        def advance_state_batch_only_device(self, request_ids, token_ids, positions, hidden):
            calls.append(
                (
                    "device",
                    tuple(request_ids),
                    tuple(token.ptr for token in token_ids),
                    tuple(positions),
                )
            )

    executor = Executor()
    provider = SimpleNamespace(executor=executor, last_results={})
    candidate_counts = (2, 2, 2, 3)
    proposal = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(1, 2, 3, 4),
        root_tokens=(90, 190, 290, 390),
        root_positions=(5, 8, 11, 14),
        candidate_counts=candidate_counts,
        token_ids=Tensor.from_handle(0x5000, (9,), DType.INT32, Device("hip", 0)),
        hidden_rows=tuple(
            tuple(
                Tensor.from_handle(
                    0x6000 + row * 0x1000 + depth * 0x100,
                    (1, 8),
                    DType.BF16,
                    Device("hip", 0),
                )
                for depth in range(count)
            )
            for row, count in enumerate(candidate_counts)
        ),
    )
    states = tuple(
        _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(1, 2, 3, 4),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
            proposal_checkpoint=f"checkpoint-{request_id}",
            proposal_context=MtpProposalContext(
                request_ids=(request_id,),
                root_tokens=(proposal.root_tokens[row],),
                root_positions=(proposal.root_positions[row],),
                target_hidden=Tensor.from_handle(
                    0xA000 + row * 0x100,
                    (1, 8),
                    DType.BF16,
                    Device("hip", 0),
                ),
            ),
            proposal_device_batch=proposal,
        )
        for row, request_id in enumerate(proposal.request_ids)
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=4)
    adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
        Tensor.from_handle(0xD000, (4, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0xE000, (4, 8), DType.BF16, Device("hip", 0)),
    )

    adapter._repair_provider_states_batch_device(
        states,
        proposal,
        accepted_counts=(2, 1, 0, 1),
    )

    assert calls == [
        ("root_snapshot", 3),
        ("checkpoint", "checkpoint-4"),
        ("host", (4,), (390,), (14,)),
        ("device", (1, 4), (0x5004, 0x5018), (7, 15)),
    ]


def test_k0_catchup_consumes_current_root_before_target_ar() -> None:
    calls = []
    executor = SimpleNamespace(
        advance_state_only=lambda request_id, token, position, hidden: calls.append(
            (request_id, token, position, hidden)
        )
    )
    provider = SimpleNamespace(executor=executor)
    state = _MTP2RequestState(
        request_id=7,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=SimpleNamespace(),
        root_hidden_buffer=SimpleNamespace(ptr=1),
    )
    row = SimpleNamespace(
        first_token_emitted=True,
        lease=SimpleNamespace(
            session=SimpleNamespace(position=15, last_target_hidden="pre-root-hidden")
        ),
        slot=SimpleNamespace(generated_ids=[90]),
        mtp2_k0_catchups=0,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {7: state}
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {7: np.zeros((1, 4), dtype=np.float32)}
    adapter._disabled_requests = set()
    adapter._post_reject_pending = set()
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned_row: None,
    )
    plan = SimpleNamespace(
        request_ids=(7,),
        reasons=(mtp2_module.SpecPlanReason.RESOURCE_CLAIM_MISS,),
        k0_classes=(mtp2_module.SpecK0Class.TRANSITIONAL,),
    )

    adapter.prepare_k0(plan, (), stream=None)

    assert calls == [(7, 90, 15, "pre-root-hidden")]
    assert row.mtp2_k0_catchups == 1


def _prepare_k0_adapter(row: SimpleNamespace) -> Qwen35GGUFMTP2Adapter:
    """Adapter wired to one resident row, with request 7 owning an intent."""

    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {
        7: _MTP2RequestState(
            request_id=7,
            provider=SimpleNamespace(executor=SimpleNamespace()),
            provider_pool_key=None,
            provider_group_key=(7,),
            verifier=SimpleNamespace(),
            root_hidden_buffer=SimpleNamespace(ptr=1),
        )
    }
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {}
    adapter._disabled_requests = set()
    adapter._post_reject_pending = set()
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned_row: None,
    )
    return adapter


def _ar_only_k0_plan() -> SimpleNamespace:
    return SimpleNamespace(
        request_ids=(7,),
        reasons=(mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS,),
        k0_classes=(mtp2_module.SpecK0Class.PURE,),
    )


def test_prepare_k0_records_no_output_before_the_decode_emits() -> None:
    """Preparation is not output: only the decode that emits may count a token.

    A mixed group is retried as its speculative subset, so the same row can be
    prepared twice while one AR decode emits one token. Counting at preparation
    would report two autoregressive outputs for it.
    """

    row = SimpleNamespace(
        first_token_emitted=False,
        lease=SimpleNamespace(
            session=SimpleNamespace(position=1022, last_target_hidden=None)
        ),
        slot=SimpleNamespace(generated_ids=[90]),
        mtp2_k0_catchups=0,
        mtp2_ar_step_reasons={},
    )
    adapter = _prepare_k0_adapter(row)

    adapter.prepare_k0(_ar_only_k0_plan(), (), stream=None)
    adapter.prepare_k0(_ar_only_k0_plan(), (), stream=None)

    # PURE K0 rows never touch provider state, and preparation alone attributes
    # nothing.
    assert row.mtp2_k0_catchups == 0
    assert row.mtp2_ar_step_reasons == {}
    assert getattr(row, "mtp2_ar_step_output_tokens", None) is None
    assert getattr(row, "mtp2_first_fallback_position", None) is None

    # The decode that emits the token attributes it, with the plan's reason.
    adapter.note_speculative_ar_commit(7, mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS)

    assert row.mtp2_ar_step_output_tokens == 1
    assert row.mtp2_ar_step_reasons == {"target_graph_context_bucket_miss": 1}
    # No speculative output yet, so this step is not a fallback.
    assert getattr(row, "mtp2_first_fallback_position", None) is None


def test_ar_commits_locate_the_first_fallback_after_speculative_output() -> None:
    """An AR-only cycle after MTP cycles reports where speculation stopped."""

    row = SimpleNamespace(
        first_token_emitted=True,
        lease=SimpleNamespace(
            session=SimpleNamespace(position=1022, last_target_hidden=None)
        ),
        slot=SimpleNamespace(generated_ids=[90, 91, 92]),
        mtp2_k0_catchups=0,
        mtp2_ar_step_reasons={},
        mtp2_requested_budget=3,
        mtp2_cycles=1,
        mtp2_candidate_counts=[3],
        mtp2_accepted_counts=[2],
        mtp2_mtp_output_tokens=2,
    )
    adapter = _prepare_k0_adapter(row)
    reason = mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS

    adapter.prepare_k0(_ar_only_k0_plan(), (), stream=None)
    # The decode appends the token before the loop attributes it, and the
    # row's output grows by one for each emitted token.
    row.slot.generated_ids.append(93)
    adapter.note_speculative_ar_commit(7, reason)
    row.slot.generated_ids.append(94)
    adapter.note_speculative_ar_commit(7, reason)

    # The first AR token after the speculative block lands at index 3 of the
    # row's output, and each emitted token is counted exactly once.
    assert row.mtp2_ar_step_output_tokens == 2
    assert row.mtp2_first_fallback_position == 3
    assert row.mtp2_ar_step_reasons == {"target_graph_context_bucket_miss": 2}
    accounting = speculative_output_accounting(row)
    assert accounting is not None
    assert accounting["mtp_output_tokens"] == 2
    assert accounting["ar_output_tokens_in_cycles"] == 2
    assert accounting["first_fallback_position"] == 3


def test_eager_cycle_commit_attributes_visible_tokens_to_their_execution_mode() -> None:
    row = SimpleNamespace(
        request=SimpleNamespace(eos_token_id=None, ignore_eos=False, max_tokens=32),
        slot=SimpleNamespace(
            generated_ids=[90],
            prev_token=90,
            seq_position=5,
            native_decode_steps=0,
            done=False,
        ),
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_mtp_output_tokens=0,
        mtp2_ar_output_tokens=0,
        mtp2_first_fallback_position=None,
        mtp2_ar_step_reasons={},
    )

    # Two depth-3 speculative cycles commit their accepted drafts plus the
    # target bonus token; the K0 row of the same plan then emits one AR token.
    assert (
        mtp2_module._commit_eager_cycle_row(
            row,
            visible=(101, 102, 103),
            accepted=2,
            candidate_count=3,
            plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            target_position=6,
        )
        is False
    )
    assert (
        mtp2_module._commit_eager_cycle_row(
            row,
            visible=(104, 105),
            accepted=1,
            candidate_count=3,
            plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            target_position=8,
        )
        is False
    )
    assert (
        mtp2_module._commit_eager_cycle_row(
            row,
            visible=(106,),
            accepted=0,
            candidate_count=0,
            plan_reason=mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS,
            target_position=9,
        )
        is False
    )

    assert row.slot.generated_ids == [90, 101, 102, 103, 104, 105, 106]
    assert row.slot.prev_token == 106
    assert row.slot.seq_position == 9
    assert row.mtp2_candidate_counts == [3, 3, 0]
    assert row.mtp2_accepted_counts == [2, 1, 0]
    accounting = speculative_output_accounting(row)
    assert accounting is not None
    assert accounting["cycles"] == 3
    assert accounting["generated_draft_tokens"] == 6
    assert accounting["accepted_draft_tokens"] == 3
    assert accounting["selected_depth_histogram"] == {"0": 1, "3": 2}
    # The prompt root token at index 0 is not part of this cycle record: only
    # the cycles' own visible tokens are attributed here.
    assert accounting["mtp_output_tokens"] == 5
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] == 6
    assert accounting["ar_step_reason_counts"] == {
        "target_graph_context_bucket_miss": 1
    }


def test_eager_cycle_commit_reports_eos_and_max_tokens_completion() -> None:
    row = SimpleNamespace(
        request=SimpleNamespace(eos_token_id=2, ignore_eos=False, max_tokens=3),
        slot=SimpleNamespace(
            generated_ids=[90],
            prev_token=90,
            seq_position=5,
            native_decode_steps=0,
            done=False,
        ),
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_mtp_output_tokens=0,
        mtp2_ar_output_tokens=0,
        mtp2_first_fallback_position=None,
        mtp2_ar_step_reasons={},
    )

    assert (
        mtp2_module._commit_eager_cycle_row(
            row,
            visible=(101,),
            accepted=0,
            candidate_count=2,
            plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            target_position=6,
        )
        is False
    )
    # A short tail: the last speculative cycle has no bonus token to append.
    assert (
        mtp2_module._commit_eager_cycle_row(
            row,
            visible=(102,),
            accepted=0,
            candidate_count=2,
            plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            target_position=7,
        )
        is False
    )
    assert row.slot.done is True
    assert row.slot.generated_ids == [90, 101, 102]

    eos_row = SimpleNamespace(
        request=SimpleNamespace(eos_token_id=2, ignore_eos=False, max_tokens=64),
        slot=SimpleNamespace(
            generated_ids=[90],
            prev_token=90,
            seq_position=5,
            native_decode_steps=0,
            done=False,
        ),
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_mtp_output_tokens=0,
        mtp2_ar_output_tokens=0,
        mtp2_first_fallback_position=None,
        mtp2_ar_step_reasons={},
    )
    assert (
        mtp2_module._commit_eager_cycle_row(
            eos_row,
            visible=(101, 2),
            accepted=1,
            candidate_count=3,
            plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            target_position=7,
        )
        is True
    )
    assert eos_row.slot.done is True


def _cycle_commit_row(**request_overrides) -> SimpleNamespace:
    request = {
        "eos_token_id": None,
        "ignore_eos": False,
        "max_tokens": 64,
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "min_tokens": 0,
    }
    request.update(request_overrides)
    return SimpleNamespace(
        request=SimpleNamespace(**request),
        slot=SimpleNamespace(
            generated_ids=[90],
            prev_token=90,
            seq_position=5,
            native_decode_steps=0,
            done=False,
        ),
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_mtp_output_tokens=0,
        mtp2_ar_output_tokens=0,
        mtp2_first_fallback_position=None,
        mtp2_ar_step_reasons={},
    )


def test_eager_cycle_commit_implements_only_the_greedy_chain_eos_finish_rule() -> None:
    """The cycle commit's finish rule is why stop-semantics fields are blockers.

    A committed cycle ends a row only when its last visible token is the row's
    EOS.  Token stops, multi-token stops, and a mid-cycle EOS are therefore not
    finishes here, which is exactly the divergence
    ``SAMPLED_MTP_UNSERVABLE_BLOCKERS`` keeps off the route: the greedy route
    avoids the mid-cycle case by bounding acceptance with
    ``greedy_chain_eos_limit``, and a stochastic accept has no such bound.  If
    this test starts failing, the commit path learned the autoregressive finish
    rule and the blockers are ready to be re-measured.
    """

    for overrides, visible in (
        ({"stop_token_ids": (2,)}, (101, 2, 102)),
        ({"stop_token_sequences": ((2, 3),)}, (101, 2, 3)),
        ({"eos_token_id": 2, "min_tokens": 8}, (101, 2, 102)),
    ):
        row = _cycle_commit_row(**overrides)
        assert (
            mtp2_module._commit_eager_cycle_row(
                row,
                visible=visible,
                accepted=len(visible) - 1,
                candidate_count=len(visible),
                plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
                target_position=6,
            )
            is False
        ), overrides
        assert row.slot.done is False, overrides
        # The token past the stop is committed, which is the behavior the
        # blocker prevents from reaching a client.
        assert row.slot.generated_ids == [90, *visible], overrides

    # The one finish the commit path does implement: EOS as the last visible
    # token, which the greedy chain bound makes the only reachable position.
    trailing = _cycle_commit_row(eos_token_id=2)
    assert (
        mtp2_module._commit_eager_cycle_row(
            trailing,
            visible=(101, 2),
            accepted=1,
            candidate_count=3,
            plan_reason=mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            target_position=7,
        )
        is True
    )


def test_refill_reuses_live_provider_group_before_opening_singleton() -> None:
    provider = SimpleNamespace(executor=SimpleNamespace(max_requests=2))
    group = mtp2_module._MTP2ProviderGroup(
        key=(0, 1),
        provider=provider,
        provider_pool_key="pool",
        request_ids={1},
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {}
    adapter._provider_groups = {group.key: group}
    calls = []
    adapter._attach_request_to_group = lambda request_id, selected: (
        calls.append((request_id, selected.key))
        or _MTP2RequestState(
            request_id=request_id,
            provider=selected.provider,
            provider_pool_key=selected.provider_pool_key,
            provider_group_key=selected.key,
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
        )
    )
    adapter._open_batch_requests = lambda ids: (_ for _ in ()).throw(
        AssertionError(f"unexpected singleton group open: {ids}")
    )
    adapter.owner = SimpleNamespace(capacity=2)

    adapter._ensure_request_states((2,))

    assert calls == [(2, (0, 1))]
    assert adapter._states[2].provider_group_key == (0, 1)


def test_context_bucket_k0_does_not_attach_or_mutate_provider() -> None:
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {}
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {7: np.zeros((1, 4), dtype=np.float32)}
    adapter._disabled_requests = set()
    calls = []
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: SimpleNamespace(
            first_token_emitted=True,
            lease=SimpleNamespace(
                session=SimpleNamespace(position=1023, last_target_hidden="hidden")
            ),
            slot=SimpleNamespace(generated_ids=[90]),
            mtp2_k0_catchups=0,
        ),
        _flush_row_owner=lambda row: calls.append("flush"),
    )
    adapter._ensure_request_states = lambda ids: calls.append(("attach", ids))

    adapter.prepare_k0(
        SimpleNamespace(
            request_ids=(7,),
            reasons=(mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS,),
        ),
        (),
        stream=None,
    )

    assert calls == []
    assert adapter._states == {}


def test_k0_does_not_advance_provider_before_prefill_root_is_published() -> None:
    calls = []
    state = _MTP2RequestState(
        request_id=7,
        provider=SimpleNamespace(
            executor=SimpleNamespace(
                advance_state_only=lambda *args: calls.append(args)
            )
        ),
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=None,
        root_hidden_buffer=SimpleNamespace(ptr=1),
    )
    row = SimpleNamespace(
        first_token_emitted=False,
        lease=SimpleNamespace(
            session=SimpleNamespace(position=15, last_target_hidden="prefill-hidden")
        ),
        slot=SimpleNamespace(generated_ids=[90]),
        mtp2_k0_catchups=0,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {7: state}
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {7: np.zeros((1, 4), dtype=np.float32)}
    adapter._disabled_requests = set()
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned_row: None,
    )

    adapter.prepare_k0(
        SimpleNamespace(
            request_ids=(7,),
            reasons=(mtp2_module.SpecPlanReason.NO_PROVIDER,),
        ),
        (),
        stream=None,
    )

    assert calls == []
    assert row.mtp2_k0_catchups == 0


def test_packed_prompt_hidden_sinks_preserve_ragged_request_offsets() -> None:
    calls: dict[int, list[tuple[str, int, int, int, int]]] = {7: [], 8: []}

    class Sink:
        hidden_size = 4

        def __init__(self, request_id: int, total_rows: int) -> None:
            self.request_id = request_id
            self.total_rows = total_rows

        def consume(self, **kwargs) -> None:
            calls[self.request_id].append(
                (
                    "consume",
                    int(kwargs["chunk_start"]),
                    int(kwargs["hidden_ptr"]),
                    int(kwargs["rows"]),
                    int(kwargs["stream"]),
                )
            )

        def finish(self, **kwargs) -> None:
            calls[self.request_id].append(
                (
                    "finish",
                    int(kwargs["total_rows"]),
                    0,
                    0,
                    int(kwargs["stream"]),
                )
            )

    runner_mod._consume_packed_target_hidden_sinks(
        sinks=(Sink(7, 9), Sink(8, 13)),
        request_ids=(7, 8),
        prompt_row_starts=(2, 5),
        packed_cu_seqlens=(0, 2, 5),
        hidden_base_ptr=0x1000,
        hidden_row_nbytes=8,
        stream=3,
        finish=False,
    )

    assert calls == {
        7: [("consume", 2, 0x1000, 2, 3)],
        8: [("consume", 5, 0x1010, 3, 3)],
    }


def test_packed_prompt_hidden_sinks_consume_post_output_norm_rows(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        runner_mod,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda src, weight, out, **kwargs: calls.append(
            (
                int(src),
                int(weight),
                int(out),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
                int(kwargs["stream"]),
            )
        ),
    )

    result = runner_mod._normalize_packed_target_hidden_for_sinks(
        sinks=(SimpleNamespace(), None, SimpleNamespace()),
        src_ptr=0x1000,
        out_ptr=0x2000,
        rows=7,
        hidden_size=5120,
        output_norm_weight_ptr=0x3000,
        eps=1e-6,
        stream=5,
        runtime=object(),
    )

    assert result == 0x2000
    assert calls == [(0x1000, 0x3000, 0x2000, 7, 5120, 5)]
    assert runner_mod._normalize_packed_target_hidden_for_sinks(
        sinks=(None, None),
        src_ptr=0x4000,
        out_ptr=0x5000,
        rows=2,
        hidden_size=5120,
        output_norm_weight_ptr=0x6000,
        eps=1e-6,
        stream=0,
        runtime=object(),
    ) == 0x4000


def test_mtp2_streaming_prompt_success_transfers_one_carried_row_per_request(
    monkeypatch,
) -> None:
    class Runtime:
        pass

    targets = {
        rid: SimpleNamespace(
            runtime=Runtime(),
            target_layout=SimpleNamespace(max_sequence_length=1024),
            runner=SimpleNamespace(
                hidden_size=4,
                weights=SimpleNamespace(
                    root=lambda name: SimpleNamespace(
                        allocation=lambda: SimpleNamespace(
                            tensor=SimpleNamespace(ptr=0x6000)
                        )
                    ),
                    config=SimpleNamespace(rms_norm_eps=1e-6),
                ),
            ),
            _prefill_hidden_a=DeviceBuffer(0x5000 + rid * 0x100, 16),
            _last_target_hidden_ptr=0,
        )
        for rid in (7, 8)
    }
    rows = {
        rid: SimpleNamespace(
            request_id=rid,
            prompt_ids=(11 + rid, 22 + rid),
            lease=SimpleNamespace(session=targets[rid]),
            prefix_reused_tokens=0,
            mtp2_prompt_streaming=False,
            mtp2_prompt_prime_rows=0,
            mtp2_prompt_carried_bytes=0,
            mtp2_prompt_fallback_reason=None,
        )
        for rid in (7, 8)
    }
    finish_calls: list[tuple[int, bool]] = []

    class Executor:
        hidden_size = 4
        max_requests = 4
        runtime = targets[7].runtime

        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            pass

        def finish_prompt_priming(self, request_id, *, stream, synchronize) -> None:
            finish_calls.append((int(request_id), bool(synchronize)))

    class Provider:
        executor = Executor()

        def __init__(self) -> None:
            self.reset: list[int] = []
            self.released: list[int] = []

        def reset_request(self, request_id) -> None:
            self.reset.append(int(request_id))

        def release_request(self, request_id) -> None:
            self.released.append(int(request_id))

    provider = Provider()
    released_pool: list[tuple[object, object]] = []
    generator = SimpleNamespace(
        backend="hip_gfx1151",
        execution_profile="strict",
        _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: (
            provider,
            "pool",
            False,
        ),
        _release_mtp_draft_runner=lambda key, owned: released_pool.append(
            (key, owned)
        ),
    )
    owner = SimpleNamespace(
        generator=generator,
        capacity=4,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: rows[int(request_id)],
    )
    carried = {
        7: DeviceBuffer(0x7000, 8),
        8: DeviceBuffer(0x8000, 8),
    }

    class Sink:
        hidden_size = 4

        def __init__(self, *, request_id, prompt_tokens, **kwargs) -> None:
            self.request_id = int(request_id)
            self.total_rows = len(tuple(prompt_tokens))
            self.closed = False
            self.transform_hidden_rows = kwargs.get("transform_hidden_rows")

        def take_final_pending_buffer(self):
            return carried[self.request_id]

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        mtp2_module,
        "_StreamingNextNPromptSink",
        Sink,
        raising=False,
    )
    norm_allocations = iter((DeviceBuffer(0x9000, 16), DeviceBuffer(0xA000, 16)))
    freed_norm: list[int] = []
    norm_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        mtp2_module,
        "malloc",
        lambda *args, **kwargs: next(norm_allocations),
    )
    monkeypatch.setattr(
        mtp2_module,
        "free",
        lambda buffer, **kwargs: freed_norm.append(int(buffer.ptr)),
    )
    monkeypatch.setattr(
        mtp2_module,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda src, weight, dst, **kwargs: norm_calls.append(
            (int(src), int(weight), int(dst), int(kwargs["rows"]), int(kwargs["stream"]))
        ),
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    adapter.register_request(7, 2)
    adapter.register_request(8, 2)
    adapter.physical_prompt_streaming = True

    sinks = adapter.begin_prompt_streaming((7, 8), checkpoints={})
    assert adapter._active_prompt_claims is not None
    assert sinks[0].transform_hidden_rows(0xB000, 2, 3) == 0x9000
    assert norm_calls == [(0xB000, 0x6000, 0x9000, 2, 3)]
    assert adapter._active_prompt_claims.units_by_pool() == {
        "gguf_mtp2.carried_hidden_rows": 2,
        "gguf_mtp2.prompt_rows": 4,
        "gguf_mtp2.provider_request_slots": 2,
    }
    adapter.finish_prompt_streaming((7, 8), success=True, stream=0)

    assert adapter._active_prompt_claims is None
    assert tuple(sink.request_id for sink in sinks) == (7, 8)
    assert provider.reset == [7, 8]
    assert finish_calls == [(7, False), (8, False)]
    assert adapter._prompt_hidden_rows == {}
    assert set(adapter._states) == {7, 8}
    assert adapter._states[7].root_hidden_buffer is carried[7]
    assert adapter._states[8].root_hidden_buffer is carried[8]
    assert adapter._states[7].provider_group_key == adapter._states[8].provider_group_key
    assert targets[7]._last_target_hidden_ptr == carried[7].ptr
    assert targets[8]._last_target_hidden_ptr == carried[8].ptr
    assert rows[7].mtp2_prompt_streaming and rows[8].mtp2_prompt_streaming
    assert rows[7].mtp2_prompt_prime_rows == 2
    assert rows[8].mtp2_prompt_carried_bytes == 8
    assert freed_norm == [0x9000, 0xA000]
    adapter.observe_prefill_result(7, rows[7].prompt_ids, SimpleNamespace(token_id=9))
    assert adapter._states[7].root_hidden_buffer is carried[7]
    assert released_pool == []


def test_mtp2_prompt_activation_in_flight_refuses_only_new_rows(monkeypatch) -> None:
    """A chunked prefill holds its claim across ticks; a second request is refused.

    The refusal must be reported on the refused row instead of failing the
    engine service, and it must not disturb the activation already streaming.
    """

    runtime = SimpleNamespace()
    targets = {
        rid: SimpleNamespace(
            runtime=runtime,
            target_layout=SimpleNamespace(max_sequence_length=1024),
            runner=SimpleNamespace(
                hidden_size=4,
                weights=SimpleNamespace(
                    root=lambda name: SimpleNamespace(
                        allocation=lambda: SimpleNamespace(
                            tensor=SimpleNamespace(ptr=0x6000)
                        )
                    ),
                    config=SimpleNamespace(rms_norm_eps=1e-6),
                ),
            ),
            _prefill_hidden_a=DeviceBuffer(0x5000 + rid * 0x100, 16),
        )
        for rid in (7, 8)
    }
    rows = {
        rid: SimpleNamespace(
            request_id=rid,
            prompt_ids=(11 + rid, 22 + rid),
            lease=SimpleNamespace(session=targets[rid]),
            prefix_reused_tokens=0,
            mtp2_candidate_budget=3,
            mtp2_prompt_streaming=False,
            mtp2_prompt_fallback_reason=None,
        )
        for rid in (7, 8)
    }
    acquisitions: list[str] = []

    class Executor:
        hidden_size = 4
        max_requests = 4
        runtime = targets[7].runtime

        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            pass

    class Provider:
        executor = Executor()

        def reset_request(self, request_id) -> None:
            pass

        def release_request(self, request_id) -> None:
            pass

    provider = Provider()
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile="strict",
            _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: (
                acquisitions.append("provider"),
                provider,
                "pool",
                False,
            )[1:],
            _release_mtp_draft_runner=lambda key, owned: None,
        ),
        capacity=4,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: rows[int(request_id)],
    )

    class Sink:
        hidden_size = 4

        def __init__(self, *, request_id, prompt_tokens, **kwargs) -> None:
            self.request_id = int(request_id)

        def take_final_pending_buffer(self):
            return DeviceBuffer(0x7000 + self.request_id * 0x100, 16)

        def close(self) -> None:
            pass

    monkeypatch.setattr(mtp2_module, "_StreamingNextNPromptSink", Sink, raising=False)
    monkeypatch.setattr(
        mtp2_module, "malloc", lambda *args, **kwargs: DeviceBuffer(0x9000, 16)
    )
    monkeypatch.setattr(mtp2_module, "free", lambda buffer, **kwargs: None)
    monkeypatch.setattr(
        mtp2_module, "gguf_rmsnorm_bf16_f32_weight", lambda *args, **kwargs: None
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    adapter.register_request(7, 3)
    adapter.register_request(8, 3)
    adapter.physical_prompt_streaming = True

    # Request 7 opens the activation; its prompt is still being prefilled.
    sinks = adapter.begin_prompt_streaming((7,), checkpoints={})
    assert sinks is not None and sinks[0] is not None
    claim = adapter._active_prompt_claims
    assert claim is not None

    # Request 8 arrives while 7's activation is open: refused, not fatal.
    assert adapter.begin_prompt_streaming((8,), checkpoints={}) is None
    assert rows[8].mtp2_prompt_fallback_reason == "prompt_activation_in_flight"
    assert rows[8].mtp2_candidate_budget == 0
    assert rows[7].mtp2_prompt_fallback_reason is None
    assert rows[7].mtp2_candidate_budget == 3
    assert adapter._active_prompt_claims is claim
    assert adapter._prompt_streaming_sinks[7] is sinks[0]
    assert 8 not in adapter._prompt_streaming_sinks
    assert acquisitions == ["provider"]

    # A mixed call keeps the streaming row and refuses only the new one.
    mixed = adapter.begin_prompt_streaming((7, 8), checkpoints={})
    assert mixed is not None and mixed[0] is sinks[0] and mixed[1] is None
    assert acquisitions == ["provider"]

    # Releasing the activation lets a later request activate normally.
    adapter.finish_prompt_streaming((7,), success=True, stream=0)
    assert adapter._active_prompt_claims is None
    rows[8].mtp2_candidate_budget = 3
    rows[8].mtp2_prompt_fallback_reason = None
    reopened = adapter.begin_prompt_streaming((8,), checkpoints={})
    assert reopened is not None and reopened[0] is not None
    # The released group's provider is reused rather than re-acquired.
    assert acquisitions == ["provider"]
    assert adapter._prompt_streaming_group_keys[8] == (7,)
    adapter.finish_prompt_streaming((8,), success=True, stream=0)


def test_mtp2_sequential_physical_admission_reuses_compatible_provider_group() -> None:
    group = SimpleNamespace(
        key=(7,),
        provider=SimpleNamespace(executor=SimpleNamespace(max_requests=4)),
        request_ids={7},
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=4)
    adapter._states = {7: SimpleNamespace(provider_group_key=group.key)}
    adapter._provider_groups = {group.key: group}
    calls: list[tuple[object, ...]] = []

    def attach(request_id, selected_group):
        calls.append(("attach", int(request_id), selected_group.key))
        selected_group.request_ids.add(int(request_id))
        return SimpleNamespace(provider_group_key=selected_group.key)

    adapter._attach_request_to_group = attach
    adapter._open_request = lambda request_id: calls.append(("open", int(request_id)))
    adapter._open_batch_requests = lambda request_ids: calls.append(
        ("open_batch", tuple(request_ids))
    )

    adapter._ensure_request_states((8,))

    assert calls == [("attach", 8, (7,))]
    assert adapter._states[8].provider_group_key == (7,)
    assert group.request_ids == {7, 8}


def test_physical_prompt_catchup_skips_full_vocabulary_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Executor:
        hidden_size = 4

        def run_step_batch(self, *args, **kwargs):
            raise AssertionError("prompt catch-up must not score the LM head")

        def run_step(self, request_id, token_id, position, hidden, *, return_logits):
            assert return_logits is False
            calls.append(((request_id,), (token_id,), (position,), hidden.shape))

        def advance_state_batch_only(
            self, request_ids, token_ids, positions, hidden
        ):
            calls.append(
                (
                    tuple(request_ids),
                    tuple(token_ids),
                    tuple(positions),
                    hidden.shape,
                )
            )

    allocations = iter(
        (
            DeviceBuffer(0x1000, 16),
            DeviceBuffer(0x2000, 8),
            DeviceBuffer(0x3000, 8),
        )
    )
    monkeypatch.setattr(
        mtp2_module, "malloc", lambda *args, **kwargs: next(allocations)
    )
    monkeypatch.setattr(mtp2_module, "free", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mtp2_module,
        "copy_host_to_device",
        lambda *args, **kwargs: None,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._prompt_hidden_rows = {
        7: np.arange(12, dtype=np.float32).reshape(3, 4),
        8: np.arange(8, dtype=np.float32).reshape(2, 4),
    }
    provider = SimpleNamespace(executor=Executor())
    rows = (
        SimpleNamespace(prompt_ids=(10, 11, 12)),
        SimpleNamespace(prompt_ids=(20, 21)),
    )
    runtime = SimpleNamespace()
    targets = (SimpleNamespace(runtime=runtime), SimpleNamespace(runtime=runtime))

    roots = adapter._catch_up_provider_batch(provider, (7, 8), rows, targets)

    assert set(roots) == {7, 8}
    assert calls == [
        ((7, 8), (10, 20), (0, 0), (2, 4)),
        ((7, 8), (11, 21), (1, 1), (2, 4)),
        ((7,), (12,), (2,), (1, 4)),
    ]


def test_mtp2_cycle_hidden_workspace_reuses_stable_distinct_slabs() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.next_ptr = 0x1000
            self.malloc_calls: list[int] = []
            self.free_calls: list[int] = []

        def malloc(self, nbytes: int) -> int:
            ptr = self.next_ptr
            self.next_ptr += 0x1000
            self.malloc_calls.append(int(nbytes))
            return ptr

        def free(self, ptr: int) -> None:
            self.free_calls.append(int(ptr))

    runtime = Runtime()
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=4)
    adapter._cycle_workspace = None
    adapter._cycle_proposal_hidden = None
    adapter._cycle_repair_hidden = None
    adapter._cycle_workspace_shape = None

    proposal_a, repair_a = adapter._cycle_hidden_tensors(
        runtime,
        hidden_size=8,
    )
    proposal_b, repair_b = adapter._cycle_hidden_tensors(
        runtime,
        hidden_size=8,
    )

    assert proposal_a is proposal_b
    assert repair_a is repair_b
    assert proposal_a.ptr != repair_a.ptr
    assert proposal_a.shape == repair_a.shape == (4, 8)
    assert runtime.malloc_calls == [64, 64]
    with pytest.raises(RuntimeError, match="shape changed"):
        adapter._cycle_hidden_tensors(runtime, hidden_size=16)

    adapter._close_cycle_workspace()

    assert runtime.free_calls == [repair_a.ptr, proposal_a.ptr]
    assert adapter._cycle_workspace is None


def test_mtp2_wide_cycle_and_accept_workspaces_own_c8_k3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.next_ptr = 0x1000
            self.malloc_calls: list[int] = []
            self.free_calls: list[int] = []

        def malloc(self, nbytes: int) -> int:
            ptr = self.next_ptr
            self.next_ptr += 0x1000
            self.malloc_calls.append(int(nbytes))
            return ptr

        def free(self, ptr: int) -> None:
            self.free_calls.append(int(ptr))

    runtime = Runtime()
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=8)
    adapter.generator = SimpleNamespace(backend="hip_gfx1100")
    adapter.candidate_budget = 3
    adapter.physical_width_depths = ((1, 2), (1, 3), (2, 2), (8, 3))
    adapter.physical_max_requests = 8
    adapter.physical_accept_max_rows = 36
    adapter._ngram = None
    adapter._cycle_workspace = None
    adapter._cycle_proposal_hidden = None
    adapter._cycle_repair_hidden = None
    adapter._cycle_ngram_tokens = None
    adapter._cycle_workspace_shape = None
    adapter._batch_accept_workspace = None
    adapter._batch_accept_owner = None
    adapter._batch_accept_remaining = None
    adapter._batch_accept_payload = None

    proposal, repair = adapter._cycle_hidden_tensors(runtime, hidden_size=8)
    assert proposal.shape == repair.shape == (8, 8)

    specs = []
    reservations = []

    class Workspace:
        def __init__(self, *, device, runtime):
            self.device = device
            self.runtime = runtime
            self.freed = False

        def reserve_tensor(self, name, shape, dtype):
            reservations.append((str(name), tuple(shape), dtype))
            return Tensor.from_handle(
                0xA000 + len(reservations) * 0x100,
                shape,
                dtype,
                self.device,
            )

        def free(self):
            self.freed = True

    owner = object()
    monkeypatch.setattr(mtp2_module, "RuntimeWorkspace", Workspace)
    monkeypatch.setattr(
        mtp2_module.TargetVerifyBufferOwner,
        "allocate",
        lambda spec, *, workspace: specs.append(spec) or owner,
    )

    actual_owner, remaining, payload = adapter._batch_accept_resources(runtime)

    assert actual_owner is owner
    assert specs[0].bucket == "gguf-mtp2-physical-r36-c8"
    assert specs[0].max_rows == 36
    assert specs[0].max_requests == 8
    assert remaining.shape == (8,)
    assert payload.shape == (8, mtp2_module.ACCEPT_PACKED_PAYLOAD_FIELDS)
    assert [shape for _name, shape, _dtype in reservations] == [
        (8,),
        (8, mtp2_module.ACCEPT_PACKED_PAYLOAD_FIELDS),
    ]


def test_mtp2_accept_workspace_allocation_failure_releases_partial_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = []

    class FailingWorkspace:
        def __init__(self, *, device, runtime):
            del runtime
            self.device = device
            self.reserve_calls = 0
            self.freed = False
            instances.append(self)

        def reserve_tensor(self, name, shape, dtype):
            del name
            self.reserve_calls += 1
            if self.reserve_calls == 2:
                raise RuntimeError("payload allocation failed")
            return Tensor.from_handle(0xB000, shape, dtype, self.device)

        def free(self):
            self.freed = True

    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(capacity=8)
    adapter.generator = SimpleNamespace(backend="hip_gfx1100")
    adapter.candidate_budget = 3
    adapter.physical_width_depths = ((1, 2), (1, 3), (2, 2), (8, 3))
    adapter.physical_max_requests = 8
    adapter.physical_accept_max_rows = 36
    adapter._batch_accept_workspace = None
    adapter._batch_accept_owner = None
    adapter._batch_accept_remaining = None
    adapter._batch_accept_payload = None
    monkeypatch.setattr(mtp2_module, "RuntimeWorkspace", FailingWorkspace)
    monkeypatch.setattr(
        mtp2_module.TargetVerifyBufferOwner,
        "allocate",
        lambda spec, *, workspace: object(),
    )

    with pytest.raises(RuntimeError, match="payload allocation failed"):
        adapter._batch_accept_resources(object())

    assert len(instances) == 1
    assert instances[0].freed is True
    assert adapter._batch_accept_workspace is None
    assert adapter._batch_accept_owner is None
    assert adapter._batch_accept_remaining is None
    assert adapter._batch_accept_payload is None


def test_mtp2_physical_prompt_streaming_is_rejected_before_provider_open() -> None:
    rows = {
        request_id: SimpleNamespace(
            prompt_ids=(11, 22),
            lease=SimpleNamespace(
                session=SimpleNamespace(
                    target_layout=SimpleNamespace(max_sequence_length=1024),
                    runtime=object(),
                )
            ),
            prefix_reused_tokens=0,
            mtp2_candidate_budget=2,
            mtp2_prompt_fallback_reason=None,
        )
        for request_id in (7, 8)
    }
    acquired: list[str] = []
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: acquired.append(
                "provider"
            )
        ),
        capacity=4,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: rows[int(request_id)],
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    for request_id in rows:
        adapter.register_request(request_id, 2)

    assert adapter.begin_prompt_streaming((7, 8), checkpoints={}) is None
    assert acquired == []
    assert all(
        row.mtp2_prompt_fallback_reason == "physical_streaming_category_rejected"
        for row in rows.values()
    )
    assert all(row.mtp2_candidate_budget == 2 for row in rows.values())


def test_mtp2_singleton_only_streaming_opens_one_slot_provider_under_wide_owner(
    monkeypatch,
) -> None:
    acquired: list[int] = []
    released: list[int] = []

    class Executor:
        hidden_size = 4
        max_requests = 1

        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            pass

        def finish_prompt_priming(self, request_id, *, stream, synchronize) -> None:
            pass

    class Provider:
        executor = Executor()

        def reset_request(self, request_id) -> None:
            pass

        def release_request(self, request_id) -> None:
            released.append(int(request_id))

    provider = Provider()
    target = SimpleNamespace(
        runner=SimpleNamespace(
            fp16_recurrent_state=False,
            hidden_size=4,
            weights=SimpleNamespace(
                root=lambda name: SimpleNamespace(
                    allocation=lambda: SimpleNamespace(
                        tensor=SimpleNamespace(ptr=0x4000)
                    )
                ),
                config=SimpleNamespace(rms_norm_eps=1e-6),
            ),
        ),
        target_layout=SimpleNamespace(max_sequence_length=1024),
        runtime=object(),
        _prefill_hidden_a=DeviceBuffer(0x5000, 16),
    )
    row = SimpleNamespace(
        prompt_ids=(11, 22),
        lease=SimpleNamespace(session=target),
        prefix_reused_tokens=0,
        mtp2_candidate_budget=3,
        mtp2_prompt_fallback_reason=None,
    )
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile="strict",
            _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: (
                acquired.append(int(kwargs["max_requests"])) or provider,
                "pool",
                False,
            ),
            _release_mtp_draft_runner=lambda *args: None,
        ),
        capacity=4,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: row,
    )

    class Sink:
        def __init__(self, *, request_id, **kwargs) -> None:
            self.request_id = int(request_id)

        def close(self) -> None:
            pass

    monkeypatch.setattr(mtp2_module, "_StreamingNextNPromptSink", Sink)
    monkeypatch.setattr(
        mtp2_module,
        "malloc",
        lambda *args, **kwargs: DeviceBuffer(0x6000, 16),
    )
    monkeypatch.setattr(mtp2_module, "free", lambda *args, **kwargs: None)
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    adapter.register_request(
        7,
        3,
        static_eligibility=SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_singleton",
            max_candidate_count=3,
            max_realized_group_rows=1,
            automatic_eligible=True,
            strict_fallback_key="gguf_target_ar",
            evidence_key="test-singleton",
            evidence_fingerprint="sha256:test-singleton",
        ),
    )

    sinks = adapter.begin_prompt_streaming((7,), checkpoints={})

    assert sinks is not None and len(sinks) == 1
    assert acquired == [1]
    assert row.mtp2_prompt_fallback_reason is None
    adapter.finish_prompt_streaming((7,), success=False, stream=0)
    assert released == [7]
    assert adapter._provider_groups == {}


def test_mtp2_static_capability_requires_its_qualified_realized_width() -> None:
    def target():
        return SimpleNamespace(
            runner=SimpleNamespace(fp16_recurrent_state=False),
            _target_scratch_owner=SimpleNamespace(slot_count=4),
            target_layout=SimpleNamespace(max_sequence_length=1024),
            kv_storage_dtype="bf16",
        )

    targets = {7: target(), 8: target()}
    rows = {
        rid: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(session=targets[rid]),
            slot=SimpleNamespace(),
        )
        for rid in targets
    }
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 3
    adapter.target_verify_mode = "native"
    adapter.quant = "gguf_q4_k_m"
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        execution_profile="strict",
    )
    adapter.owner = SimpleNamespace(capacity=4, _row=lambda rid: rows[int(rid)])
    adapter._intents = {7: 3, 8: 3}
    adapter._sampling_mode_by_request = {}
    adapter._static_eligibility_by_request = {
        rid: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_singleton",
            max_candidate_count=3,
            max_realized_group_rows=1,
            automatic_eligible=True,
            strict_fallback_key="gguf_target_ar",
            evidence_key=f"test-singleton-{rid}",
            evidence_fingerprint=f"sha256:test-singleton-{rid}",
        )
        for rid in (7, 8)
    }
    adapter._disabled_requests = set()
    adapter._prompt_hidden_rows = {}
    adapter._states = {
        rid: _MTP2RequestState(
            request_id=rid,
            provider=SimpleNamespace(),
            provider_pool_key=None,
            provider_group_key=(rid,),
            verifier=SimpleNamespace(target_verify_mode="native"),
            root_hidden_buffer=SimpleNamespace(ptr=rid),
        )
        for rid in targets
    }
    one = SpeculativeRequestSemantics(
        request_id=7,
        sampling_mode="greedy",
        mode="verify_chain",
        context_tokens=32,
        remaining_decode=25,
    )
    two = SpeculativeRequestSemantics(
        request_id=8,
        sampling_mode="greedy",
        mode="verify_chain",
        context_tokens=32,
        remaining_decode=25,
    )

    assert adapter.capability((one,)) is not None
    assert adapter.capability((one, two)) is None
    assert adapter.partition_max_requests((7, 8)) == 0

    adapter._static_eligibility_by_request = {
        rid: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_c2",
            max_candidate_count=2,
            max_realized_group_rows=2,
            automatic_eligible=True,
            strict_fallback_key="gguf_target_ar",
            evidence_key=f"test-c2-{rid}",
            evidence_fingerprint=f"sha256:test-c2-{rid}",
        )
        for rid in (7, 8)
    }
    assert adapter.capability((one,)) is not None
    assert adapter.capability((one, two)) is not None


def test_mtp2_wide_provider_lazily_adds_singleton_target_verifier(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeVerifier:
        def __init__(
            self,
            target,
            *,
            max_candidate_budget: int,
            quant: str,
            target_verify_mode: str,
        ) -> None:
            calls.append(
                (
                    "verifier",
                    target,
                    int(max_candidate_budget),
                    str(quant),
                    str(target_verify_mode),
                )
            )

    monkeypatch.setattr(
        mtp2_module,
        "Qwen35GGUFTransactionalVerifier",
        FakeVerifier,
    )
    targets = {
        request_id: SimpleNamespace(position=32)
        for request_id in (7, 8)
    }
    rows = {
        request_id: SimpleNamespace(lease=SimpleNamespace(session=targets[request_id]))
        for request_id in targets
    }
    states = {
        request_id: _MTP2RequestState(
            request_id=request_id,
            provider=SimpleNamespace(),
            provider_pool_key=None,
            provider_group_key=(7, 8),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=request_id),
        )
        for request_id in targets
    }
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        capacity=4,
        _row=lambda request_id: rows[int(request_id)],
        _flush_row_owner=lambda row: calls.append(("flush", row)),
    )
    adapter.generator = SimpleNamespace(backend="hip_gfx1151")
    adapter.candidate_budget = 3
    adapter.quant = "gguf_q4_k_m"
    adapter.target_verify_mode = "native"
    adapter._states = states
    adapter._ensure_request_states = lambda ids: calls.append(("ensure", tuple(ids)))

    adapter.prepare_requests(
        SimpleNamespace(speculative_request_ids=(7,)),
        (),
    )

    assert isinstance(states[7].verifier, FakeVerifier)
    assert states[8].verifier is None
    assert calls == [
        ("ensure", (7,)),
        ("verifier", targets[7], 3, "gguf_q4_k_m", "native"),
    ]

    # A neighbor still selects the existing physical C2 target branch; adding
    # the singleton verifier must not decompose or serialize the group.
    claims = object()
    adapter._active_claims = claims
    adapter._execute_target_frontier_batch = lambda *args, **kwargs: "packed-c2"
    result = adapter.execute_target_frontier(
        SimpleNamespace(speculative_request_ids=(7, 8)),
        SimpleNamespace(target_batch=object(), candidate_graph=None),
        claims,
        commit=True,
        cancelled_request_ids=lambda: (),
    )
    assert result == "packed-c2"


def test_mtp2_physical_intent_allows_c1_before_or_after_c2() -> None:
    target = SimpleNamespace(
        runner=SimpleNamespace(fp16_recurrent_state=False),
        _target_scratch_owner=SimpleNamespace(slot_count=4),
        target_layout=SimpleNamespace(max_sequence_length=1024),
        kv_storage_dtype="bf16",
    )
    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=True,
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(),
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 1
    adapter.target_verify_mode = "native"
    adapter.quant = "gguf_q4_k_m"
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        execution_profile="production",
        execution_profile_fell_back_to_strict=False,
        execution_profile_manifest_sha256="production-manifest",
        execution_profile_manifest={
            "selections": (
                {
                    "layer": "gdn_chain_recurrent_rmsnorm_gate",
                    "scope": "specdec2_mtp2_target_state_rows",
                    "selected_variant": "bf16_c1_exact_state_rows_tloop_fp16state",
                    "strict_fallback_variant": "bf16_c1_exact_state_rows_tloop",
                },
            )
        },
    )
    adapter.owner = SimpleNamespace(capacity=4, _row=lambda rid: row)
    adapter._intents = {7: 1}
    adapter._sampling_mode_by_request = {}
    adapter._static_eligibility_by_request = {
        7: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_physical_c4",
            max_candidate_count=1,
            max_realized_group_rows=4,
            automatic_eligible=False,
            strict_fallback_key="gguf_target_ar",
            evidence_key="test-physical-c4",
            evidence_fingerprint="sha256:test-physical-c4",
        )
    }
    adapter._disabled_requests = set()
    adapter._prompt_hidden_rows = {7: object()}
    adapter._states = {}
    semantics = (
        SpeculativeRequestSemantics(7, "greedy", "verify_chain", 32, 25),
    )

    assert adapter.capability(semantics) is not None
    assert adapter.partition_max_requests((7,)) == 4
    adapter._active_claims = None
    assert adapter.claims_fit(
        SimpleNamespace(request_ids=(7,), speculative_request_ids=(7,))
    ) is True
    assert adapter.claims_fit(
        SimpleNamespace(request_ids=(7, 8), speculative_request_ids=(7,))
    ) is False

    adapter._prompt_hidden_rows = {}
    adapter._states = {
        7: _MTP2RequestState(
            request_id=7,
            provider=SimpleNamespace(executor=SimpleNamespace(max_requests=4)),
            provider_pool_key=None,
            provider_group_key=(7, 8),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=7),
        )
    }
    assert adapter.capability(semantics) is not None


def test_mtp2_long_prompt_selects_k0_before_provider_streaming() -> None:
    target = SimpleNamespace(
        target_layout=SimpleNamespace(max_sequence_length=4096),
        runtime=object(),
    )
    row = SimpleNamespace(
        prompt_ids=tuple(range(1022)),
        lease=SimpleNamespace(session=target),
        prefix_reused_tokens=0,
        mtp2_candidate_budget=2,
        mtp2_prompt_fallback_reason=None,
    )
    acquired: list[str] = []
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: acquired.append(
                "provider"
            )
        ),
        capacity=1,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: row,
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    adapter.register_request(7, 2)

    assert adapter.begin_prompt_streaming((7,), checkpoints={}) is None
    assert row.mtp2_candidate_budget == 0
    assert row.mtp2_prompt_fallback_reason == "target_context_k0"
    assert acquired == []
    assert adapter._prompt_streaming_sinks == {}
    assert adapter._states == {}


def test_mtp2_context_window_defaults_to_the_qualified_1023(monkeypatch) -> None:
    monkeypatch.delenv(mtp2_module._MTP2_MAX_CONTEXT_ENV, raising=False)
    assert mtp2_module._mtp2_context_window() == 1023


def test_mtp2_context_window_accepts_an_experimental_raise(monkeypatch) -> None:
    monkeypatch.setenv(mtp2_module._MTP2_MAX_CONTEXT_ENV, "16384")
    assert mtp2_module._mtp2_context_window() == 16384


@pytest.mark.parametrize("value", ["0", "-1", "", "soon", "1.5"])
def test_mtp2_context_window_rejects_unusable_values(monkeypatch, value) -> None:
    monkeypatch.setenv(mtp2_module._MTP2_MAX_CONTEXT_ENV, value)
    with pytest.raises(RuntimeError):
        mtp2_module._mtp2_context_window()


def test_mtp2_raised_context_window_admits_a_prompt_the_default_refuses(
    monkeypatch,
) -> None:
    """The experimental window is what lets a long prompt reach the provider.

    The default run stops at the context gate with ``target_context_k0`` and
    never touches the draft provider. The raised run reaches provider
    acquisition, which this fixture signals with a distinctive error because it
    does not implement the streaming provider ABI.
    """

    class _ReachedProvider(Exception):
        pass

    def build() -> tuple[object, SimpleNamespace]:
        target = SimpleNamespace(
            target_layout=SimpleNamespace(max_sequence_length=16384),
            runtime=object(),
        )
        row = SimpleNamespace(
            prompt_ids=tuple(range(1022)),
            lease=SimpleNamespace(session=target),
            prefix_reused_tokens=0,
            mtp2_candidate_budget=2,
            mtp2_prompt_fallback_reason=None,
        )

        def acquire(*args, **kwargs):
            raise _ReachedProvider

        owner = SimpleNamespace(
            generator=SimpleNamespace(_acquire_dense_mtp_draft_provider=acquire),
            capacity=1,
            _shared_runner=SimpleNamespace(hidden_size=4),
            _row=lambda request_id: row,
        )
        adapter = Qwen35GGUFMTP2Adapter(
            owner,
            enabled=True,
            target_verify_mode="native",
            candidate_budget=2,
        )
        adapter.register_request(7, 2)
        return adapter, row

    monkeypatch.delenv(mtp2_module._MTP2_MAX_CONTEXT_ENV, raising=False)
    adapter, row = build()
    assert adapter.begin_prompt_streaming((7,), checkpoints={}) is None
    assert row.mtp2_prompt_fallback_reason == "target_context_k0"
    assert row.mtp2_candidate_budget == 0

    monkeypatch.setenv(mtp2_module._MTP2_MAX_CONTEXT_ENV, "16384")
    adapter, row = build()
    with pytest.raises(_ReachedProvider):
        adapter.begin_prompt_streaming((7,), checkpoints={})
    assert row.mtp2_prompt_fallback_reason != "target_context_k0"


def test_mtp2_streaming_prompt_failure_drains_provider_and_sink() -> None:
    events: list[tuple[object, ...]] = []

    class Sink:
        def close(self) -> None:
            events.append(("sink_close",))

    provider = SimpleNamespace(
        executor=SimpleNamespace(
            finish_prompt_priming=lambda request_id, *, stream, synchronize: events.append(
                ("finish", int(request_id), int(stream), bool(synchronize))
            )
        ),
        release_request=lambda request_id: events.append(
            ("release_request", int(request_id))
        ),
    )
    group = mtp2_module._MTP2ProviderGroup(
        key=(7,),
        provider=provider,
        provider_pool_key="pool",
        request_ids={7},
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._prompt_streaming_sinks = {7: Sink()}
    adapter._prompt_streaming_group_keys = {7: (7,)}
    adapter._active_prompt_claims = mtp2_module.ResourceClaimSet.from_mapping(
        "prompt",
        {"rows": 1},
        lifetime=mtp2_module.ClaimLifetime.WORK_ITEM,
    )
    adapter._provider_groups = {(7,): group}
    adapter.generator = SimpleNamespace(
        _release_mtp_draft_runner=lambda key, owned: events.append(
            ("release_group", key, owned)
        )
    )

    adapter.finish_prompt_streaming((7,), success=False, stream=5)

    assert events == [
        ("finish", 7, 5, True),
        ("release_request", 7),
        ("sink_close",),
        ("release_group", "pool", provider),
    ]
    assert adapter._prompt_streaming_sinks == {}
    assert adapter._prompt_streaming_group_keys == {}
    assert adapter._active_prompt_claims is None
    assert adapter._provider_groups == {}


def _ngram_replay_prompt(seed: int) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    prefix = tuple(range(seed, seed + 24))
    continuation = tuple(range(seed + 1_000, seed + 1_024))
    # The first generated root completes the second occurrence of ``prefix``.
    return (*prefix, *continuation, *prefix[:-1]), prefix[-1], continuation


def test_physical_adapter_uses_ngram_first_and_skips_mtp_proposal() -> None:
    prompts = tuple(_ngram_replay_prompt(seed) for seed in (100, 500))
    runtime = SimpleNamespace(memcpy=lambda *args: None)
    targets = tuple(
        SimpleNamespace(
            position=len(prompt),
            last_target_hidden=Tensor.from_handle(
                0x1000 + index * 0x100,
                (1, 8),
                DType.BF16,
                Device("hip", 0),
            ),
            runtime=runtime,
        )
        for index, (prompt, _root, _continuation) in enumerate(prompts)
    )
    provider = SimpleNamespace(
        executor=SimpleNamespace(
            hidden_size=8,
            capture_request_checkpoint=lambda request_id: f"checkpoint-{request_id}",
        ),
        propose_batch_device=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("n-gram hit must skip MTP proposal compute")
        ),
    )
    rows = tuple(
        SimpleNamespace(
            prompt_ids=prompt,
            lease=SimpleNamespace(session=target),
            slot=SimpleNamespace(generated_ids=[root], seq_position=target.position),
            mtp2_candidate_device_handoffs=0,
        )
        for (prompt, root, _continuation), target in zip(prompts, targets, strict=True)
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        capacity=2,
        _row=lambda request_id: rows[(10, 20).index(request_id)],
        _flush_row_owner=lambda row: None,
    )
    adapter._ngram = RequestLocalNgramMod(
        NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=24)
    )
    adapter._states = {
        request_id: _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(10, 20),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
        )
        for request_id in (10, 20)
    }
    adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
        Tensor.from_handle(0x4000, (2, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0x5000, (2, 8), DType.BF16, Device("hip", 0)),
    )
    adapter._stage_ngram_tokens = lambda tokens, runtime: Tensor.from_handle(
        0x6000,
        (len(tuple(tokens)),),
        DType.INT32,
        Device("hip", 0),
    )
    plan = SimpleNamespace(
        speculative_request_ids=(10, 20),
        request_ids=(10, 20),
        candidate_counts=(3, 3),
        provider_key="nextn",
        cycle_id=9,
        resident_slots=(0, 1),
    )
    semantics = tuple(
        SpeculativeRequestSemantics(
            request_id,
            "greedy",
            "verify_chain",
            len(prompt) + 1,
            8,
        )
        for request_id, (prompt, _root, _continuation) in zip(
            (10, 20), prompts, strict=True
        )
    )

    graph = adapter.propose_batch(plan, semantics)

    assert graph.method_key == "ngram_mod+mtp2"
    assert graph.candidate_tokens == tuple(
        token
        for _prompt, _root, continuation in prompts
        for token in continuation[:3]
    )
    assert graph.token_ids is not None and graph.token_ids.ptr == 0x6000
    assert dict(graph.provider_metadata)["proposal_source"] == "request_local_ngram_mod"
    assert all(state.proposal_source == "ngram_mod" for state in adapter._states.values())
    assert all(state.proposal_device_batch is None for state in adapter._states.values())
    assert [row.mtp2_ngram_cycles for row in rows] == [1, 1]


def test_ngram_mixed_group_hit_fails_closed_to_one_physical_mtp_source() -> None:
    hit_prompt, hit_root, _continuation = _ngram_replay_prompt(100)
    miss_prompt = tuple(range(1_000, 1_071))
    rows = (
        SimpleNamespace(
            prompt_ids=hit_prompt,
            slot=SimpleNamespace(generated_ids=[hit_root]),
            lease=SimpleNamespace(session=SimpleNamespace(runtime=object())),
        ),
        SimpleNamespace(
            prompt_ids=miss_prompt,
            slot=SimpleNamespace(generated_ids=[1_071]),
            lease=SimpleNamespace(session=SimpleNamespace(runtime=object())),
        ),
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._ngram = RequestLocalNgramMod(
        NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=24)
    )
    context = MtpProposalContext(
        request_ids=(10, 20),
        root_tokens=(hit_root, 1_071),
        root_positions=(len(hit_prompt), len(miss_prompt)),
        target_hidden=Tensor.from_handle(
            0x7000, (2, 8), DType.BF16, Device("hip", 0)
        ),
    )

    assert adapter._try_ngram_proposal(
        (10, 20), rows, (3, 3), context
    ) is None
    assert [row.mtp2_ngram_lookup_calls for row in rows] == [1, 1]
    assert [getattr(row, "mtp2_ngram_lookup_hits", 0) for row in rows] == [1, 0]
    assert [getattr(row, "mtp2_ngram_cycles", 0) for row in rows] == [0, 0]


def test_ngram_target_rows_catch_mtp_up_through_root_and_accepted_prefix() -> None:
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(
        memcpy=lambda dst, src, nbytes, kind: calls.append(
            ("copy", int(dst), int(src), int(nbytes), kind)
        )
    )
    executor = SimpleNamespace(
        hidden_size=8,
        runtime=runtime,
        restore_request_checkpoint=lambda checkpoint: calls.append(
            ("restore", checkpoint)
        ),
        advance_state_only=lambda request_id, token, position, hidden: calls.append(
            ("one", int(request_id), int(token), int(position), int(hidden.ptr))
        ),
        advance_state_batch_only=lambda ids, tokens, positions, hidden: calls.append(
            (
                "batch",
                tuple(ids),
                tuple(tokens),
                tuple(positions),
                int(hidden.ptr),
            )
        ),
    )
    provider = SimpleNamespace(executor=executor)
    states = tuple(
        _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(10, 20),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
            proposal_checkpoint=f"checkpoint-{request_id}",
            proposal_context=MtpProposalContext(
                request_ids=(request_id,),
                root_tokens=(root,),
                root_positions=(position,),
                target_hidden=Tensor.from_handle(
                    0x8000 + index * 0x100,
                    (1, 8),
                    DType.BF16,
                    Device("hip", 0),
                ),
            ),
            ngram_candidate_tokens=candidates,
            proposal_source="ngram_mod",
        )
        for index, (request_id, root, position, candidates) in enumerate(
            (
                (10, 100, 5, (101, 102, 103)),
                (20, 200, 8, (201, 202, 203)),
            )
        )
    )
    rows = {10: SimpleNamespace(), 20: SimpleNamespace()}
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(_row=lambda request_id: rows[int(request_id)])
    adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
        Tensor.from_handle(0xA000, (2, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0xB000, (2, 8), DType.BF16, Device("hip", 0)),
    )
    hidden_rows = (
        Tensor.from_handle(0xC000, (4, 8), DType.BF16, Device("hip", 0)),
        Tensor.from_handle(0xD000, (4, 8), DType.BF16, Device("hip", 0)),
    )

    adapter._repair_provider_states_from_ngram_target_rows(
        states,
        hidden_rows,
        accepted_counts=(2, 0),
    )

    assert calls[:2] == [
        ("restore", "checkpoint-10"),
        ("restore", "checkpoint-20"),
    ]
    assert (
        "batch",
        (10, 20),
        (100, 200),
        (5, 8),
        0xB000,
    ) in calls
    assert any(call[:4] == ("one", 10, 101, 6) for call in calls)
    assert any(call[:4] == ("one", 10, 102, 7) for call in calls)
    assert rows[10].mtp2_ngram_accepted_tokens == 2
    assert rows[20].mtp2_ngram_accepted_tokens == 0


def test_physical_accept_readback_uses_blocking_copy_dependency_not_global_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = DraftBatch(
        request_ids=(10, 20),
        candidate_tokens=(101, 201),
        parent_positions=(5, 8),
        draft_depths=(1, 1),
        row_to_request=(10, 20),
        tree_parents=(-1, -1),
        active_mask=(True, True),
    )
    batch = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(100, 200),
        root_positions=(5, 8),
    )
    pointer = iter(range(0x5000, 0x7000, 0x100))

    def tensor(shape, dtype=DType.INT32):
        return Tensor.from_handle(next(pointer), shape, dtype, Device("hip", 0))

    output = tensor((2, 4))
    buffers = TargetVerifyBuffers.for_batch(
        batch,
        token_ids=tensor((4,)),
        positions=tensor((4,)),
        parent_rows=tensor((4,)),
        draft_depths=tensor((4,)),
        row_to_request=tensor((4,)),
        active_mask=tensor((4,), DType.BOOL),
        target_top1=tensor((4,)),
        accepted_counts=tensor((2,)),
        commit_rows=tensor((2,)),
        commit_tokens=tensor((2,)),
        commit_positions=tensor((2,)),
        next_tokens=tensor((2,)),
        full_accept=tensor((2,), DType.BOOL),
        committed_output_ids=output,
        committed_output_lengths=tensor((2,)),
        transaction_id=7,
    )
    payload = tensor((2, mtp2_module.ACCEPT_PACKED_PAYLOAD_FIELDS))
    pending = mtp2_module._PhysicalAcceptPending(
        batch=batch,
        buffers=buffers,
        payload=payload,
        request_count=2,
        output_stride=4,
    )
    payload_host = np.asarray(
        [[1, 1, 101, 6, 999, 0, 2], [0, 2, 200, 9, 201, 0, 1]],
        dtype=np.int32,
    )
    committed = np.asarray(
        [[100, 101, -1, -1], [200, -1, -1, -1]],
        dtype=np.int32,
    )

    def fake_copy(destination, source, nbytes, *, runtime):
        if int(source.ptr) == int(payload.ptr):
            ctypes.memmove(destination, payload_host.ctypes.data, nbytes)
            return
        row = (int(source.ptr) - int(output.ptr)) // (4 * DType.INT32.itemsize)
        ctypes.memmove(destination, committed[row].ctypes.data, nbytes)

    monkeypatch.setattr(mtp2_module, "copy_device_to_host", fake_copy)
    runtime = SimpleNamespace(
        device_synchronize=lambda: (_ for _ in ()).throw(
            AssertionError("bounded blocking D2H must own the dependency")
        )
    )

    summary = Qwen35GGUFMTP2Adapter._read_target_batch_accept(
        pending,
        runtime=runtime,
    )

    assert summary.accepted_counts == (1, 0)
    assert summary.accepted_tokens == ((101,), ())
    assert summary.commit_rows == (1, 2)


def _streaming_owner(*, profile: str = "production", widths=None):
    """Owner double for prompt-streaming resolver tests."""

    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile=profile,
        ),
        capacity=4,
        _shared_runner=SimpleNamespace(
            weights=SimpleNamespace(
                geometry=QWEN35_DENSE_H5120_GEOMETRY,
                file_type_name="MOSTLY_Q4_K_M",
            ),
        ),
    )
    return owner


def test_prompt_streaming_resolver_accepts_width_one_and_rejects_out_of_range(
    monkeypatch,
) -> None:
    """Scaling-campaign M3: the validator admits a registered width-1 policy
    without broadening the unqualified upper bound; unregistered profiles and
    models keep the replay route (empty widths)."""

    import hipengine.kernels.hip_gfx1151 as gfx1151

    monkeypatch.setitem(
        gfx1151.GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES,
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "production"),
        (1, 2, 3),
    )
    adapter = Qwen35GGUFMTP2Adapter(
        _streaming_owner(),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert adapter.physical_prompt_streaming_widths == (1, 2, 3)
    assert adapter._physical_prompt_streaming_admitted(1) is True
    assert adapter._physical_prompt_streaming_admitted(4) is False

    # Strict profile stays on replay: no strict key is registered.
    strict = Qwen35GGUFMTP2Adapter(
        _streaming_owner(profile="strict"),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    assert strict.physical_prompt_streaming_widths == ()
    assert strict.physical_prompt_streaming is False
    assert strict._physical_prompt_streaming_admitted(1) is False

    # Out-of-range registered widths remain a hard error on both ends.
    for bad in ((0, 2), (2, 9)):
        monkeypatch.setitem(
            gfx1151.GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES,
            (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "production"),
            bad,
        )
        with pytest.raises(RuntimeError, match=r"within \[1, 8\]"):
            Qwen35GGUFMTP2Adapter(
                _streaming_owner(),
                enabled=True,
                target_verify_mode="native",
                candidate_budget=3,
            )


def test_mtp2_production_routes_full_batches_above_the_measured_bound_to_ar() -> None:
    # M5 whole-batch routing: sub-group interleaving measured 0.74-0.80x AR at
    # physical widths 5-8, so a due batch wider than the production economic
    # bound must partition to 0 (one full-batch AR decode) instead of
    # chaining MTP sub-groups.
    from hipengine.kernels.hip_gfx1151 import (
        GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS,
        GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS,
    )

    assert GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS["production"] == 4
    # The wide cell is withdrawn, so the whole C1-C4 policy sits below the
    # bound and no listed cell can reach a due group wider than it.
    assert GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS["production"][-1] == (4, 3)

    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=True,
        lease=SimpleNamespace(
            session=SimpleNamespace(
                runner=SimpleNamespace(fp16_recurrent_state=False),
                _target_scratch_owner=SimpleNamespace(slot_count=8),
                target_layout=SimpleNamespace(max_sequence_length=1024),
                kv_storage_dtype="bf16",
            )
        ),
        slot=SimpleNamespace(),
    )
    ids = tuple(range(1, 9))
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 3
    adapter.target_verify_mode = "native"
    adapter.quant = "gguf_q4_k_m"
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        execution_profile="production",
    )
    adapter.owner = SimpleNamespace(capacity=8, _row=lambda rid: row)
    adapter._intents = {rid: 3 for rid in ids}
    adapter._static_eligibility_by_request = {
        rid: SpeculativeMTPStaticEligibility(
            state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
            reason="qualified_test_physical_c4",
            max_candidate_count=3,
            max_realized_group_rows=4,
            automatic_eligible=False,
            strict_fallback_key="gguf_target_ar",
            evidence_key=f"test-route-{rid}",
            evidence_fingerprint=f"sha256:test-route-{rid}",
        )
        for rid in ids
    }
    adapter._disabled_requests = set()
    adapter._prompt_hidden_rows = {}
    adapter._states = {}

    # Within-bound batches keep the certified MTP cycle.
    assert adapter.partition_max_requests(ids[:4]) == 4
    # Over-width due batches route to a single full-batch AR decode.
    assert adapter.partition_max_requests(ids) == 0
    assert adapter.partition_max_requests(ids[:5]) == 0


def test_mtp2_candidate_budget_admission_derives_from_declared_depth() -> None:
    """Adapter admission must follow the declared implementation-depth limit.

    The K4 server hang traced to a hardcoded ``{1, 2, 3}`` admission literal:
    lifting the declared depth constant did not lift admission, and the
    resulting ValueError escaped the batcher worker and orphaned queued
    requests. Admission now derives from ``_MTP2_MAX_CANDIDATE_DEPTH``.
    """

    import hipengine.generation.qwen35_gguf_mtp2 as module

    assert module._MTP2_MAX_CANDIDATE_DEPTH == 7

    # At the declared depth, budget 8 remains rejected.
    with pytest.raises(ValueError, match="candidate budget"):
        Qwen35GGUFMTP2Adapter(
            _width_bound_owner(profile="production", capacity=8),
            enabled=True,
            target_verify_mode="packed",
            candidate_budget=8,
        )

    # Lifting the declared depth lifts admission coherently.
    original = module._MTP2_MAX_CANDIDATE_DEPTH
    module._MTP2_MAX_CANDIDATE_DEPTH = 8
    try:
        lifted = Qwen35GGUFMTP2Adapter(
            _width_bound_owner(profile="production", capacity=8),
            enabled=True,
            target_verify_mode="packed",
            candidate_budget=8,
        )
        assert lifted.candidate_budget == 8
    finally:
        module._MTP2_MAX_CANDIDATE_DEPTH = original


def test_c8_frontier_padded_boundaries_follow_the_campaign_table() -> None:
    """C8 K3-K7 logical/padded frontier rows must match the campaign table.

    K3 32→36, K4 40→42, K5 48→48 (already a rows6 multiple), K6 56→60,
    K7 64→66. The 66-row padded frontier is representable: row validity is
    a per-row uint8 array (batch.active_mask), not a u64 bitmask.
    """

    from hipengine.speculative.frontier import physical_group_pad_rows

    for depth, logical, padded in (
        (3, 32, 36),
        (4, 40, 42),
        (5, 48, 48),
        (6, 56, 60),
        (7, 64, 66),
    ):
        candidate_rows = 8 * depth
        pad = physical_group_pad_rows(
            (6,),
            8,
            candidate_rows,
            max_rows=1024,
        )
        assert 8 + candidate_rows == logical, depth
        assert logical + pad == padded, (depth, pad)

    # A group whose padded shape exceeds the accept-row capacity stays
    # unpadded (strict fallback route) rather than overrunning the buffer.
    assert physical_group_pad_rows((6,), 8, 8 * 7, max_rows=65) == 0


def test_adapter_frontier_geometry_derives_from_declared_budget() -> None:
    """Adapter frontier/accept geometry must scale with the declared budget."""

    import hipengine.generation.qwen35_gguf_mtp2 as module

    real = module.backend_package_capability
    policy = tuple(
        (width, depth) for width in range(1, 9) for depth in range(1, 8)
    )

    def fake(backend: str, name: str, default: object = None) -> object:
        if name == "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS":
            return {"production": policy}
        if name == "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS":
            return (6,)
        return real(backend, name, default)

    module.backend_package_capability = fake
    try:
        adapter = Qwen35GGUFMTP2Adapter(
            _width_bound_owner(profile="production", capacity=8),
            enabled=True,
            target_verify_mode="packed",
            candidate_budget=7,
        )
        assert adapter.physical_max_requests == 8
        assert adapter.candidate_budget == 7
        # Logical C8/K7 frontier: 8 requests × (7 candidates + 1 root).
        assert adapter.physical_accept_max_rows == 66
    finally:
        module.backend_package_capability = real


def test_ar_commit_survives_the_decode_that_retires_the_request() -> None:
    """The token that reaches max_new_tokens is still an attributed AR step."""

    row = SimpleNamespace(
        first_token_emitted=True,
        lease=SimpleNamespace(session=SimpleNamespace(position=8)),
        slot=SimpleNamespace(generated_ids=[90, 91]),
        mtp2_ar_step_reasons={},
    )
    adapter = _prepare_k0_adapter(row)
    reason = mtp2_module.SpecPlanReason.TARGET_GRAPH_OUTPUT_ROOM_MISS
    adapter.prepare_k0(_ar_only_k0_plan(), (), stream=None)
    row.slot.generated_ids.append(92)
    # The decode that emitted the last token already retired the request from
    # speculation before the loop observed the commit.
    adapter._disabled_requests.add(7)
    adapter._intents.pop(7, None)

    adapter.note_speculative_ar_commit(7, reason)

    assert row.mtp2_ar_step_output_tokens == 1
    assert row.mtp2_ar_step_reasons == {"target_graph_output_room_miss": 1}


def test_ar_commit_is_ignored_for_a_released_or_unknown_row() -> None:
    """A decode that emitted nothing must not attribute an autoregressive token."""

    released = SimpleNamespace(
        first_token_emitted=True,
        lease=None,
        slot=None,
        mtp2_ar_step_reasons={},
    )
    adapter = _prepare_k0_adapter(released)
    reason = mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS

    adapter.note_speculative_ar_commit(7, reason)

    assert released.mtp2_ar_step_reasons == {}
    assert getattr(released, "mtp2_ar_step_output_tokens", None) is None

    # A row the owner does not know about is a caller/owner disagreement.
    row = SimpleNamespace(
        first_token_emitted=True,
        lease=SimpleNamespace(session=SimpleNamespace(position=8)),
        slot=SimpleNamespace(generated_ids=[90]),
        mtp2_ar_step_reasons={},
    )
    adapter = _prepare_k0_adapter(row)
    adapter.owner._rows = {7: row}

    adapter.note_speculative_ar_commit(99, reason)

    assert row.mtp2_ar_step_reasons == {}


def test_prompt_activation_available_tracks_the_active_claim() -> None:
    """The scheduler probe is true exactly while no activation is in flight."""

    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._active_prompt_claims = None
    assert adapter.prompt_activation_available() is True

    adapter._active_prompt_claims = object()
    assert adapter.prompt_activation_available() is False


def test_prefill_activation_ready_defers_only_unstarted_speculative_rows() -> None:
    """A row forfeits its provider only if its first chunk runs claim-less.

    A row already mid-prompt, a row with no candidate budget, and a
    prefix-reused row are all ready: none of them needs the prompt sink that the
    held claim protects.
    """

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    claim_held = {"value": True}
    runner._mtp2_adapter = SimpleNamespace(
        prompt_activation_available=lambda: not claim_held["value"]
    )
    runner._mtp2_adapter_resolved = True
    rows = {
        rid: SimpleNamespace(
            request_id=rid,
            prefill_tokens_seen=0,
            mtp2_candidate_budget=3,
            prefix_reused_tokens=0,
        )
        for rid in (1, 2, 3, 4)
    }
    rows[2].prefill_tokens_seen = 256
    rows[3].mtp2_candidate_budget = 0
    rows[4].prefix_reused_tokens = 128
    runner._rows = rows

    assert runner.prefill_activation_ready(1) is False
    assert runner.prefill_activation_ready(2) is True
    assert runner.prefill_activation_ready(3) is True
    assert runner.prefill_activation_ready(4) is True

    claim_held["value"] = False
    assert runner.prefill_activation_ready(1) is True


def test_prefill_activation_ready_without_adapter_is_always_ready() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._mtp2_adapter = None
    runner._mtp2_adapter_resolved = True
    runner._rows = {
        1: SimpleNamespace(
            request_id=1,
            prefill_tokens_seen=0,
            mtp2_candidate_budget=3,
            prefix_reused_tokens=0,
        )
    }

    assert runner.prefill_activation_ready(1) is True


def test_prefill_needs_activation_marks_only_unstarted_speculative_rows() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._rows = {
        rid: SimpleNamespace(
            request_id=rid,
            prefill_tokens_seen=0,
            mtp2_candidate_budget=3,
            prefix_reused_tokens=0,
            slot=None,
        )
        for rid in (1, 2, 3, 4)
    }
    runner._rows[2].prefill_tokens_seen = 256
    runner._rows[3].mtp2_candidate_budget = 0
    runner._rows[4].prefix_reused_tokens = 128

    assert runner.prefill_needs_activation(1) is True
    assert runner.prefill_needs_activation(2) is False
    assert runner.prefill_needs_activation(3) is False
    assert runner.prefill_needs_activation(4) is False


def test_sampled_prefill_hands_the_row_to_its_mtp_provider(monkeypatch) -> None:
    """Both prefill finishes must open provider draft state for a due row.

    `observe_prefill_result` is what records a row's prompt hidden rows, and the
    provider's capability refuses a row that has neither hidden rows nor a
    request state. A sampled row finishes prefill through
    `_finish_sampled_prefill`, which never called the hook, so a sampled request
    that registered a candidate budget still had no provider on every cycle:
    the planner reported `no_provider` and the row decoded autoregressively even
    with the sampled route's evidence row in place.
    """

    import sys
    import time
    from collections import defaultdict

    import hipengine.generation.qwen35_gguf as gguf_module

    observed = []

    class _Provider:
        def observe_prefill_result(self, request_id, prompt_ids, result):
            observed.append((int(request_id), tuple(prompt_ids), result))

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._mtp2_adapter = _Provider()
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(tokenizer=object())
    runner._route_counts = defaultdict(int)
    runner._fallback_reasons = defaultdict(int)
    runner._refresh_prefix_cache = lambda row: None
    sampling_request = SimpleNamespace(max_tokens=8)
    runner._prepare_sampled_prefill = lambda row: (sampling_request, object())

    monkeypatch.setattr(
        gguf_module,
        "_select_from_gguf_logits",
        lambda result, request, state, tokenizer: SimpleNamespace(token_id=11),
    )
    monkeypatch.setattr(gguf_module, "_gguf_token_text", lambda tokenizer, sample: "x")
    monkeypatch.setattr(gguf_module, "_gguf_logits_d2h_metadata", lambda result: (False, 0))
    monkeypatch.setattr(
        gguf_module, "_gguf_queue_json_object_close_if_needed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gguf_module, "_gguf_finished", lambda *args, **kwargs: False)
    assert sys.modules[Qwen35GGUFResidentModelRunner.__module__] is gguf_module

    def _row(*, budget: int):
        return SimpleNamespace(
            request_id=7,
            prompt_ids=(1, 2, 3),
            slot=None,
            lease=SimpleNamespace(session=SimpleNamespace(position=5), pool_key="pool"),
            native_sampler=False,
            sampler_plan=SimpleNamespace(fallback_reason=None, mode=SimpleNamespace(value="host")),
            samples=[],
            tokenize_ms=0.0,
            prompt_encode_ms=0.0,
            render_ms=0.0,
            admission_prepare_ms=0.0,
            prefill_ms=0.0,
            prefill_chunk_count=0.0,
            submitted_at=time.perf_counter(),
            request=SimpleNamespace(max_tokens=8),
            mtp2_candidate_budget=budget,
        )

    result = SimpleNamespace(token_id=11)
    row = _row(budget=3)
    runner._finish_sampled_prefill(row, result, native_compact_prefill=False)
    assert row.slot is not None
    assert observed == [(7, (1, 2, 3), result)]

    # A row that registered no candidate budget owns no provider state.
    observed.clear()
    runner._finish_sampled_prefill(_row(budget=0), result, native_compact_prefill=False)
    assert observed == []

    # The hook is also a no-op without a resolved adapter.
    runner._mtp2_adapter = None
    runner._observe_mtp2_prefill(_row(budget=3), result)
    assert observed == []


def test_streaming_prompt_refusal_names_a_reused_prefix_row() -> None:
    """The gate that excludes a reused-prefix row is where its reason is set.

    Such a row is filtered out before any adapter gate sees it, so without this
    it would carry no activation reason at all and the served response would
    report only the planner's ``no_provider``.
    """

    row = SimpleNamespace(
        request_id=7,
        mtp2_candidate_budget=3,
        prefix_reused_tokens=256,
        mtp2_prompt_fallback_reason=None,
    )
    runner = SimpleNamespace()

    sinks = Qwen35GGUFResidentModelRunner._begin_mtp2_prompt_streaming(runner, (row,))

    assert sinks == (None,)
    assert row.mtp2_prompt_fallback_reason == "prefix_reuse_k0"

    # A row that already recorded a refusal keeps it: the first gate to refuse
    # a row is the one that explains it.
    row.mtp2_prompt_fallback_reason = "target_context_k0"
    Qwen35GGUFResidentModelRunner._begin_mtp2_prompt_streaming(runner, (row,))
    assert row.mtp2_prompt_fallback_reason == "target_context_k0"

    # A row that owns no candidate budget is not a refusal at all.
    quiet = SimpleNamespace(
        request_id=8,
        mtp2_candidate_budget=0,
        prefix_reused_tokens=256,
        mtp2_prompt_fallback_reason=None,
    )
    Qwen35GGUFResidentModelRunner._begin_mtp2_prompt_streaming(runner, (quiet,))
    assert quiet.mtp2_prompt_fallback_reason is None


def test_note_speculative_plan_publishes_the_group_decision_on_every_row() -> None:
    """The loop's plan is the only source of the realized group width.

    A row's response used to carry the route's ``k0_class`` and a serving-key
    width, so a row that decoded autoregressively for its whole life still read
    as route-qualified with no group-level decision beside it.
    """

    rows = {7: SimpleNamespace(), 8: SimpleNamespace()}
    plan = SimpleNamespace(
        request_ids=(7, 8),
        reasons=(
            mtp2_module.SpecPlanReason.SPECULATIVE_QUALIFIED,
            mtp2_module.SpecPlanReason.NO_PROVIDER,
        ),
        is_ar_only=False,
    )
    runner = SimpleNamespace(_row=lambda rid: rows.get(rid))

    Qwen35GGUFResidentModelRunner.note_speculative_plan(runner, plan)

    assert rows[7].mtp2_plan_group_rows == 2
    assert rows[8].mtp2_plan_group_rows == 2
    assert rows[7].mtp2_plan_ar_only is False
    assert rows[7].mtp2_plan_reason == "speculative_qualified"
    assert rows[8].mtp2_plan_reason == "no_provider"

    # A row that has already left the engine is skipped, not an error.
    plan.request_ids = (7, 9)
    plan.reasons = (
        mtp2_module.SpecPlanReason.POLICY_SELECTED_AR,
        mtp2_module.SpecPlanReason.POLICY_SELECTED_AR,
    )
    plan.is_ar_only = True
    Qwen35GGUFResidentModelRunner.note_speculative_plan(runner, plan)
    assert rows[7].mtp2_plan_ar_only is True
    assert rows[7].mtp2_plan_reason == "policy_selected_ar"


def test_sampled_prefill_streams_prompt_hidden_rows_for_a_due_row(monkeypatch) -> None:
    """A sampled row that owes a draft provider must open the hidden-row sink.

    A sampled row needs its first-token logits on the host, and a packed prefill
    refuses to return logits and host hidden rows together, so the only route
    that serves both is the streamed target-hidden sink the greedy path uses.
    The provider refuses a row with no hidden rows, so without the sink a due
    sampled row had no provider for its whole lifetime.
    """

    from collections import defaultdict

    import hipengine.generation.qwen35_gguf as gguf_module

    captured = []

    def _prefill_batch(
        chunks,
        sessions=None,
        full_prompt_lengths=None,
        return_logits=None,
        return_hidden_seeds=None,
        target_hidden_chunk_sinks=None,
        target_hidden_request_ids=None,
        target_hidden_chunk_starts=None,
        **kwargs,
    ):
        captured.append(
            (
                "batch",
                return_logits,
                return_hidden_seeds,
                None if target_hidden_chunk_sinks is None else len(tuple(target_hidden_chunk_sinks)),
                None if target_hidden_request_ids is None else tuple(target_hidden_request_ids),
            )
        )
        return [SimpleNamespace(token_id=1)]

    scalar_calls = []

    def _scalar_prefill(token_ids, **kwargs):
        scalar_calls.append((tuple(token_ids), kwargs.get("return_logits")))
        return SimpleNamespace(token_id=1)

    monkeypatch.setattr(
        gguf_module, "_gguf_single_row_block_table_prefill_required", lambda session: False
    )
    session = SimpleNamespace(prefill=_scalar_prefill)
    owner = SimpleNamespace(prefill_batch_native=_prefill_batch)
    finished = []

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._resident_batch_owner = None
    runner._route_counts = defaultdict(int)
    runner._packed_execution_owner = lambda lease_session: owner
    runner._prepare_sampled_prefill = lambda row: (SimpleNamespace(max_tokens=8), object())
    runner._finish_sampled_prefill = lambda row, result, **kwargs: captured.append(
        ("finish", row.request_id)
    )
    runner._finish_mtp2_prompt_streaming = (
        lambda rows, sinks, *, success, stream=0: finished.append(
            (tuple(int(row.request_id) for row in rows), success)
        )
    )
    admitted = [True]
    runner._begin_mtp2_prompt_streaming = lambda rows: (
        ("sink",) if admitted[0] else (None,)
    )

    def _row(budget: int):
        return SimpleNamespace(
            request_id=7,
            prompt_ids=(1, 2, 3),
            slot=None,
            lease=SimpleNamespace(session=session, pool_key="pool"),
            native_sampler=False,
            prefill_ms=0.0,
            prefill_chunk_count=0,
            mtp2_candidate_budget=budget,
        )

    # A due row with an admitted sink streams its hidden rows and keeps logits.
    runner._prefill_sampled_row(_row(budget=3))
    assert captured == [("batch", True, False, 1, (7,)), ("finish", 7)]
    assert finished == [((7,), True)]
    assert scalar_calls == []

    # A refused sink keeps the shipped scalar prefill: logits, no provider.
    captured.clear()
    finished.clear()
    admitted[0] = False
    runner._prefill_sampled_row(_row(budget=3))
    assert captured == [("finish", 7)]
    assert finished == []
    assert scalar_calls == [((1, 2, 3), True)]

    # No candidate budget never opens a sink at all.
    captured.clear()
    scalar_calls.clear()
    runner._begin_mtp2_prompt_streaming = lambda rows: pytest.fail(
        "a row with no candidate budget must not open a streaming sink"
    )
    runner._prefill_sampled_row(_row(budget=0))
    assert captured == [("finish", 7)]
    assert scalar_calls == [((1, 2, 3), True)]


def test_adapter_refuses_ar_fallback_for_a_committed_cycle() -> None:
    """A committed eager row cannot fall back: the scheduler is behind it."""

    rows = {
        10: SimpleNamespace(
            slot=SimpleNamespace(seq_position=9),
            lease=SimpleNamespace(session=SimpleNamespace(position=9)),
            mtp2_recoverable_failures=0,
            mtp2_failure_reasons=[],
        ),
    }
    committed: set[int] = set()
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: rows[request_id],
        execution_committed_request_ids=lambda: frozenset(committed),
    )
    plan = SimpleNamespace(speculative_request_ids=(10,))

    # Canonical cursors and no recorded commit: the cycle still falls back to AR.
    assert adapter.recover_cycle_failure(plan, RuntimeError("precommit")) is True
    assert rows[10].mtp2_failure_reasons == ["precommit_failure_ar_fallback", "RuntimeError:precommit"]

    committed.add(10)
    assert adapter.recover_cycle_failure(plan, RuntimeError("postcommit")) is False
    assert rows[10].mtp2_failure_reasons[-2:] == [
        "postcommit_eager_commit_ar_fallback_refused",
        "RuntimeError:postcommit",
    ]


def test_adapter_containment_evidence_maps_commit_and_cursor_state() -> None:
    rows = {
        10: SimpleNamespace(
            slot=SimpleNamespace(seq_position=9),
            lease=SimpleNamespace(session=SimpleNamespace(position=9)),
        ),
        20: SimpleNamespace(
            slot=SimpleNamespace(seq_position=9),
            lease=SimpleNamespace(session=SimpleNamespace(position=8)),
        ),
        30: SimpleNamespace(slot=None, lease=None),
    }
    committed: set[int] = set()
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: rows[request_id],
        execution_committed_request_ids=lambda: frozenset(committed),
    )
    plan = SimpleNamespace(speculative_request_ids=(10, 20, 30))

    # A row whose cursor moved without a commit, and a row with no device state
    # to prove, are both retirable; the canonical row is left unnamed.
    assert adapter.containment_evidence(plan, RuntimeError("late")) == (
        "partial",
        (20, 30),
    )
    # A recorded commit makes the cycle ``committed`` and names the committed
    # row alongside the rows that cannot be proven canonical.
    committed.add(10)
    assert adapter.containment_evidence(plan, RuntimeError("late")) == (
        "committed",
        (10, 20, 30),
    )
    # A physical commit that may or may not have landed leaves the window
    # unresolved: that stays fatal rather than becoming a claim.
    assert (
        adapter.containment_evidence(
            plan,
            mtp2_module._PhysicalTargetCommitError("selected target state may be committed"),
        )
        is None
    )
    # A cycle whose rows are all canonical is recovery's business, not
    # containment's.
    committed.clear()
    rows[20].lease.session.position = 9
    rows[30] = SimpleNamespace(
        slot=SimpleNamespace(seq_position=4),
        lease=SimpleNamespace(session=SimpleNamespace(position=4)),
    )
    assert adapter.containment_evidence(plan, RuntimeError("late")) is None
