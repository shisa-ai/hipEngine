from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.kernels.backends import backend_package_capability
from hipengine.speculative import SpeculativeRequestSemantics


class _AdapterDouble:
    def __init__(self) -> None:
        self.calls = []

    def register_request(self, request_id, candidate_budget):
        self.calls.append(("register", request_id, candidate_budget))

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


def test_gfx1151_package_exposes_only_the_s3_c1_adapter_scope() -> None:
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_C1", False
    ) is False


def test_resident_runner_delegates_staged_methods_without_backend_branches() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    adapter = _AdapterDouble()
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(target_arch="gfx1151")
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}

    runner.register_speculative_request(7, 3)
    assert runner._rows[7].mtp2_candidate_budget == 3
    assert adapter.calls == [("register", 7, 3)]
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
    assert capability.max_requests == 1
    assert capability.max_candidates_per_request == 3
    assert capability.max_frontier_rows == 4
    assert capability.max_context_tokens == 4096

    target.runner = SimpleNamespace(fp16_recurrent_state=True)
    assert adapter.capability(semantics) is None
