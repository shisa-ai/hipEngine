"""The Q8_0 expert-down grouped fallback key is owner state, not a process key.

``HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT`` used to name which registered owner the
grouped Q8_0 expert-down fallback resolved. The profile binder now publishes that
key on the owning generator and runner (``q8_down_variant``), every carrier
forwards it, and the dispatch reads only the owner: an unbound owner keeps the
unguarded WMMA parent, an explicit owner string wins over a stale ambient value,
and nothing reads the environment at launch.

What these tests pin:

* **Publication.** UD production publishes the guarded block-scale owner; every
  other plan publishes the unguarded WMMA parent, and no plan writes the process
  key. The restored-flags table keeps the value as a declaration only.
* **Ownership.** Two owners bound in either order keep independent values, and a
  caller-set ambient value is reported once and stays inert.
* **Selection.** An unbound owner takes the parent leaf, never the environment;
  an explicit owner string is returned unchanged, so an unknown name still fails
  loudly at the registry instead of being rewritten to a fallback.
* **Capability.** Only the guarded leaf takes the sparse risk buffers; the
  unguarded leaves keep the plain ABI.
* **Forwarding.** The decision reaches the MoE dispatch through the generator
  inherit, the AR batch carrier, the batch step and the shared layer FFN, and it
  is part of the captured-segment policy key.
* **Diagnostics.** The layer2 gate names the variant through its spec, the
  campaign and state gate bind the leaf they count, and the shared-graph
  allowlist names the owner field.

These are CPU contract tests: no GPU, no kernels, no performance claim.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import warnings
from types import SimpleNamespace

import pytest

from hipengine import execution_profiles as profiles_module
from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation import qwen4_exp_ar_batch, qwen4_exp_gguf
from hipengine.generation import qwen4_exp_profiles as profile_module
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    last_prebinder_conflicts,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_runner as runtime
from tests.test_unit_qwen4exp_batch_runner import (
    _fake_runner,
    _install_step_fakes,
)
from tests.test_unit_qwen4exp_q8_down_selector_retired import (
    _chain_arms,
    _q8_down_chain,
)

ENV = "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT"
WMMA = "selected_grouped_wmma_prefill_bf16_bf16_out"
BLOCKSCALE = "selected_grouped_blockscale_prefill_bf16_bf16_out"
GUARDED = "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out"
WMMA_GATE = "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"
UD = "gguf_ud_q4_k_xl"
Q4KM = "gguf_q4_k_m"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    profile_module._owner_binder_discarded.clear()
    profile_module._last_prebinder_conflicts.clear()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(WMMA_GATE, raising=False)
    yield
    profile_module._owner_binder_discarded.clear()
    profile_module._last_prebinder_conflicts.clear()
    clear_runtime_profile_registry_for_tests()


def _enable_historical_q4km_plan() -> None:
    key = profiles_module.RuntimeProfileKey(
        QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, Q4KM,
        ExecutionProfile.PRODUCTION,
    )
    plan = profiles_module._RUNTIME_PROFILE_PLANS[key]
    if plan.qualification_failure is None:
        return
    profiles_module.register_runtime_profile_plan(
        model=key.model, backend=key.backend, quant=key.quant,
        profile=key.profile,
        plan=dataclasses.replace(plan, qualification_failure=None), replace=True,
    )


def _resolve(profile: ExecutionProfile, *, quant: str = UD):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    if quant == Q4KM and profile is ExecutionProfile.PRODUCTION:
        _enable_historical_q4km_plan()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


def _bind(resolved):
    runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    return generator, runner


# --- publication ------------------------------------------------------------


@pytest.mark.parametrize("profile,quant,expected", [
    (ExecutionProfile.PRODUCTION, UD, GUARDED),
    (ExecutionProfile.STRICT, UD, WMMA),
    (ExecutionProfile.PRODUCTION, Q4KM, WMMA),
])
def test_binder_publishes_the_plan_value_and_writes_no_process_key(
        profile, quant, expected):
    generator, runner = _bind(_resolve(profile, quant=quant))

    assert generator.q8_down_variant == expected
    assert runner.q8_down_variant == expected
    assert ENV not in os.environ


def test_the_restored_table_keeps_the_declaration_without_the_env_write():
    # The value stays declared where the restoration decision lives; only its
    # transport changed.
    assert PRODUCTION_Q8_QSA_RESTORED_FLAGS["Q8_DOWN_VARIANT"] == GUARDED
    assert profile_module._Q8_DOWN_WMMA_VARIANT == WMMA
    assert profile_module._Q8_DOWN_VARIANT_ENV == ENV
    assert "Q8_DOWN_VARIANT" not in profile_module._PROCESS_RESTORED_Q8_QSA_FLAGS
    assert set(profile_module._PROCESS_RESTORED_Q8_QSA_FLAGS) == (
        set(PRODUCTION_Q8_QSA_RESTORED_FLAGS) - {"Q8_DOWN_VARIANT", "Q8_0_SELECTED_WMMA_DOWN"})


def test_the_two_modules_name_the_same_leaves():
    """Two literals in two modules would drift; the registry pins both."""

    register_gfx1151_kernels(replace=True)
    assert profile_module._Q8_DOWN_WMMA_VARIANT == runtime.Q8_DOWN_WMMA_VARIANT
    assert (
        PRODUCTION_Q8_QSA_RESTORED_FLAGS["Q8_DOWN_VARIANT"]
        == runtime.Q8_DOWN_GUARDED_VARIANT
    )
    for variant in (WMMA, BLOCKSCALE, GUARDED):
        resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0",
                variant=variant)


def test_two_owners_keep_independent_values_in_both_bind_orders():
    production = _resolve(ExecutionProfile.PRODUCTION, quant=UD)
    strict = _resolve(ExecutionProfile.STRICT, quant=UD)
    first_runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    second_runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    first = SimpleNamespace(runner=first_runner)
    second = SimpleNamespace(runner=second_runner)

    production.binder(first, production)
    strict.binder(second, strict)
    assert first.q8_down_variant == GUARDED
    assert first_runner.q8_down_variant == GUARDED
    assert second.q8_down_variant == WMMA
    assert second_runner.q8_down_variant == WMMA

    strict.binder(first, strict)
    assert first.q8_down_variant == WMMA
    assert second.q8_down_variant == WMMA
    production.binder(second, production)
    assert first.q8_down_variant == WMMA
    assert second.q8_down_variant == GUARDED


def test_a_caller_set_value_is_reported_once_and_stays_inert(monkeypatch):
    monkeypatch.setenv(ENV, WMMA)
    with pytest.warns(RuntimeWarning, match=ENV):
        generator, runner = _bind(_resolve(ExecutionProfile.PRODUCTION, quant=UD))

    assert os.environ[ENV] == WMMA
    assert generator.q8_down_variant == GUARDED
    assert runner.q8_down_variant == GUARDED
    assert last_prebinder_conflicts()[ENV] == (WMMA, GUARDED)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bind(_resolve(ExecutionProfile.PRODUCTION, quant=UD))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_a_matching_ambient_value_is_not_a_conflict(monkeypatch):
    monkeypatch.setenv(ENV, GUARDED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bind(_resolve(ExecutionProfile.PRODUCTION, quant=UD))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert last_prebinder_conflicts() == {}


# --- selection --------------------------------------------------------------


def test_an_unbound_owner_takes_the_parent_leaf_and_never_the_environment(
        monkeypatch):
    monkeypatch.setenv(ENV, BLOCKSCALE)

    assert runtime._qwen4_exp_q8_down_variant(None) == WMMA
    assert runtime.resolve_q8_down_variant(SimpleNamespace()) == WMMA
    assert runtime.resolve_q8_down_variant(
        SimpleNamespace(q8_down_variant=None)) == WMMA


def test_an_explicit_owner_string_beats_a_stale_environment(monkeypatch):
    monkeypatch.setenv(ENV, GUARDED)

    assert runtime._qwen4_exp_q8_down_variant(WMMA) == WMMA
    assert runtime._qwen4_exp_q8_down_variant(BLOCKSCALE) == BLOCKSCALE
    assert runtime.resolve_q8_down_variant(
        SimpleNamespace(q8_down_variant=WMMA)) == WMMA
    assert runtime.resolve_q8_down_variant(
        SimpleNamespace(q8_down_variant=GUARDED)) == GUARDED


def test_selection_does_not_read_the_environment(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("the Q8_0 down key is owner state now")
            return super().get(key, default)

    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())

    assert runtime._qwen4_exp_q8_down_variant(None) == WMMA
    assert runtime.resolve_q8_down_variant(SimpleNamespace()) == WMMA
    assert runtime._qwen4_exp_q8_down_variant(GUARDED) == GUARDED


def test_unknown_explicit_variant_still_fails_loudly():
    bogus = "selected_grouped_nope_bf16_bf16_out"

    # Selection never rewrites a name it does not recognise...
    assert runtime._qwen4_exp_q8_down_variant(bogus) == bogus
    assert runtime._qwen4_exp_q8_down_guarded(bogus) is False
    # ...so the backend registry is the only place that decides, and it carries
    # no such leaf. ``is_registered`` is the exact-key lookup, because the
    # resolve-style fallback picks up the cpu_reference catch-all.
    register_gfx1151_kernels(replace=True)
    assert not is_registered(
        KernelKey("hip_gfx1151", "linear", "gguf_q8_0", bogus))


# --- capability -------------------------------------------------------------


def test_only_the_guarded_leaf_takes_the_sparse_risk_buffers():
    assert runtime._qwen4_exp_q8_down_guarded(GUARDED) is True
    assert runtime._qwen4_exp_q8_down_guarded(WMMA) is False
    assert runtime._qwen4_exp_q8_down_guarded(BLOCKSCALE) is False


def test_the_launch_site_resolves_the_owner_and_keeps_the_risk_branch():
    source = inspect.getsource(runtime.run_qwen4_exp_moe)

    assert ENV not in source
    assert "q8_variant = _qwen4_exp_q8_down_variant(q8_down_variant)" in source
    assert "_qwen4_exp_q8_down_guarded(q8_variant)" in source
    assert '"selected_grouped_blockscale_guarded_prefill_bf16_bf16_out"' not in source


def test_the_guarded_leaf_requires_its_risk_capacity():
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill as module

    with pytest.raises(ValueError, match="risk capacity"):
        module.gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out(
            1, 2, 3, 4, 5, 6, 64, 4, 640, 320, 96,
            risk_count_ptr=7, risk_indices_ptr=8, risk_capacity=64 * 320 - 1)


# --- the selected owner's launch ABI ----------------------------------------
# The two tests above prove the decision and the wrapper. These drive the real
# ``run_qwen4_exp_moe`` on a fake scratch up to the Q8_0 down launch, so the
# guarded/unguarded split is observed where the dispatch makes it: a guarded
# selection must request the sparse risk buffers and pass them to the owner, an
# unguarded one must pass none, and neither may read the retired process key.


class _StopAtDownLaunch(Exception):
    """The Q8_0 down owner launch is the last thing the fake MoE runs."""


class _FakeMoERuntime:
    def __init__(self) -> None:
        self.memsets: list[tuple] = []

    def memset_async(self, ptr, value, nbytes, stream):
        self.memsets.append((int(ptr), int(value), int(nbytes), int(stream)))


def _fake_moe_weight(quant_key: str, slot: str, ptr: int):
    return SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(quant_key=quant_key, slot_path=slot),
        allocation=lambda kind, _ptr=ptr: SimpleNamespace(
            tensor=SimpleNamespace(ptr=_ptr)),
    )


def _fake_moe_scratch(runtime: _FakeMoERuntime, risk_requests: list[tuple]):
    def buffer(ptr: int, nbytes: int = 4096):
        return SimpleNamespace(ptr=ptr, nbytes=nbytes)

    def ensure_group_risk_buffers(*, compact_rows: int, out_features_total: int):
        risk_requests.append((int(compact_rows), int(out_features_total)))
        return buffer(0x7A00), buffer(0x7B00)

    return SimpleNamespace(
        closed=False,
        runtime=runtime,
        router_logits=buffer(0x1000),
        selected=buffer(0x1100),
        routing=buffer(0x1200),
        hidden_bf16=buffer(0x1300),
        group_counts=buffer(0x1400),
        group_padded_counts=buffer(0x1500),
        group_expert_start=buffer(0x1600),
        group_wmma_expert_start=buffer(0x1700),
        group_wmma_total=buffer(0x1800),
        group_scatter_offsets=buffer(0x1900),
        group_sorted_lanes=buffer(0x1A00),
        group_sorted_experts=buffer(0x1B00),
        group_sorted_weights=buffer(0x1C00),
        group_tile_expert=buffer(0x1D00, nbytes=8 * 64),
        group_gate_up=buffer(0x1E00),
        expert_gate=buffer(0x1F00),
        expert_up=buffer(0x2000),
        expert_intermediate=buffer(0x2100),
        expert_down=buffer(0x2200),
        output=buffer(0x2300),
        ensure_group_risk_buffers=ensure_group_risk_buffers,
    )


def _fake_moe_dispatch(
    monkeypatch: pytest.MonkeyPatch, *, bound_variant, rows: int = 64,
    top_k: int = 2, experts: int = 4, hidden: int = 256, ffn: int = 128,
) -> dict[str, object]:
    """Run the real MoE dispatch on a fake scratch up to the Q8_0 down leaf.

    The route is the shipped grouped fallback shape: a Q4_K gate/up pair with a
    Q8_0 expert down, the WMMA-down gate open, routed F16 closed and no grouped
    production owner, so a multi-row chunk reaches the ``gguf_q8_0`` arm. At
    ``rows == 1`` that arm is out of reach and the dispatch lands on the strict
    per-expert selected gemv instead; either way the Q8_0 down leaf is the last
    thing this fake runs.
    """

    register_gfx1151_kernels(replace=True)
    monkeypatch.setenv(WMMA_GATE, "1")
    for name in (
        "qwen35_moe_group_count",
        "qwen35_moe_group_prefix",
        "qwen35_moe_group_scatter_gather_lowp",
        "qwen35_moe_group_scatter",
        "gguf_q4_k_quantize_bf16_q8_1",
        "silu_mul_separate_out_bf16",
        "launch_gguf_linear",
    ):
        monkeypatch.setattr(runtime, name, lambda *a, **k: None)

    resolved: list[dict[str, object]] = []

    def resolve(**kwargs):
        resolved.append(dict(kwargs))
        if (
            kwargs.get("layer") == "linear"
            and kwargs.get("quant") == "gguf_q8_0"
        ):
            def stop(*_args, **_kwargs):
                raise _StopAtDownLaunch

            return stop
        return lambda *a, **k: None

    monkeypatch.setattr(runtime, "resolve", resolve)
    launches: list[dict[str, object]] = []

    def q8_down(**kwargs):
        launches.append(kwargs)
        raise _StopAtDownLaunch

    monkeypatch.setattr(runtime, "_run_qwen4_exp_q8_wmma_down", q8_down)

    runtime_owner = _FakeMoERuntime()
    risk_requests: list[tuple] = []
    scratch = _fake_moe_scratch(runtime_owner, risk_requests)
    weights = {
        "router": _fake_moe_weight("gguf_q4_k", "layers.4.router", 0x11),
        "expert_gate": _fake_moe_weight(
            "gguf_q4_k", "layers.4.expert_gate", 0x12),
        "expert_up": _fake_moe_weight(
            "gguf_q4_k", "layers.4.expert_up", 0x13),
        "expert_down": _fake_moe_weight(
            "gguf_q8_0", "layers.4.expert_down", 0x14),
        "shared_gate": _fake_moe_weight(
            "gguf_q4_k", "layers.4.shared_gate", 0x15),
        "shared_up": _fake_moe_weight(
            "gguf_q4_k", "layers.4.shared_up", 0x16),
        "shared_down": _fake_moe_weight(
            "gguf_q4_k", "layers.4.shared_down", 0x17),
        "shared_gate_weight": _fake_moe_weight(
            "gguf_q4_k", "layers.4.shared_gate_weight", 0x18),
    }

    with pytest.raises(_StopAtDownLaunch):
        runtime.run_qwen4_exp_moe(
            0x100, weights, scratch=scratch, rows=rows, hidden=hidden,
            ffn=ffn, experts=experts, top_k=top_k, stream=0,
            production_moe_prefill_enabled=False,
            routed_f16_enabled=False,
            q8_down_variant=bound_variant,
        )

    return {
        "launches": launches,
        "resolved": resolved,
        "risk_requests": risk_requests,
        "memsets": runtime_owner.memsets,
    }


def _q8_down_leaf(record: dict[str, object]) -> str:
    """The one ``gguf_q8_0`` owner the dispatch resolved."""

    leaves = [
        entry["variant"]
        for entry in record["resolved"]
        if entry.get("layer") == "linear"
        and entry.get("quant") == "gguf_q8_0"
    ]
    assert len(leaves) == 1, leaves
    return leaves[0]


def _q8_down_launch(
    monkeypatch: pytest.MonkeyPatch, *, bound_variant,
    rows: int = 64, top_k: int = 2, experts: int = 4, hidden: int = 256,
    ffn: int = 128,
) -> dict[str, object]:
    """Run the grouped fallback shape and record its Q8_0 down launch."""

    record = _fake_moe_dispatch(
        monkeypatch, bound_variant=bound_variant, rows=rows, top_k=top_k,
        experts=experts, hidden=hidden, ffn=ffn,
    )
    assert len(record["launches"]) == 1, record["launches"]
    return {
        "launch": record["launches"][0],
        "variant": _q8_down_leaf(record),
        "risk_requests": record["risk_requests"],
        "memsets": record["memsets"],
    }


def test_the_guarded_selection_passes_the_risk_buffers_to_the_owner(monkeypatch):
    record = _q8_down_launch(monkeypatch, bound_variant=GUARDED)

    assert record["variant"] == GUARDED
    assert record["risk_requests"] == [(64 * 2, 256)]
    assert record["launch"]["risk_kwargs"] == {
        "risk_count_ptr": 0x7A00,
        "risk_indices_ptr": 0x7B00,
        "risk_capacity": 64 * 2 * 256,
    }
    # The launch still names the registered leaf and the grouped tile capacity.
    assert record["launch"]["compact_rows"] == 64 * 2
    assert record["launch"]["in_features"] == 128
    assert record["launch"]["out_features"] == 256
    assert record["launch"]["owner"] is not None


def test_the_unguarded_selection_passes_no_risk_buffers(monkeypatch):
    record = _q8_down_launch(monkeypatch, bound_variant=WMMA)

    assert record["variant"] == WMMA
    assert record["risk_requests"] == []
    assert record["launch"]["risk_kwargs"] == {}


def test_the_dispatch_ignores_a_stale_ambient_key(monkeypatch):
    monkeypatch.setenv(ENV, GUARDED)

    record = _q8_down_launch(monkeypatch, bound_variant=None)

    # The unbound owner takes the WMMA parent and the stale key never decides.
    assert record["variant"] == WMMA
    assert record["risk_requests"] == []
    assert record["launch"]["risk_kwargs"] == {}


def test_a_c1_capture_cannot_depend_on_the_owner(monkeypatch):
    """The one graph shape a c1 MoE capture has is a rows==1 pass.

    ``run_qwen4_exp_moe`` refuses a graph cache on any other width, and every
    term of its ``grouped_prefill`` gate needs rows>=2 (routed F16 from rows 2,
    the exact grouped chains from rows 2, the production grouped owner from rows
    16), so the Q8_0 down owner cannot change what a warm c1 capture replays.
    All three owners launch the same strict per-expert selected gemv.

    That is why the c1 MoE graph key carries only the decode-live decisions. If
    the grouped chain ever becomes reachable on decode this test fails, and the
    key has to gain the owner.
    """

    leaves = set()
    for variant in (GUARDED, WMMA, BLOCKSCALE):
        record = _fake_moe_dispatch(monkeypatch, bound_variant=variant, rows=1)
        assert record["launches"] == [], variant
        assert record["risk_requests"] == [], variant
        leaves.add(_q8_down_leaf(record))

    assert leaves == {"selected_gemv_bf16_bf16_out"}


# --- the scope --------------------------------------------------------------


def test_the_scope_binds_the_owner_and_its_runner_and_restores():
    owner = SimpleNamespace(q8_down_variant=GUARDED,
                            runner=SimpleNamespace(q8_down_variant=GUARDED))

    with runtime.q8_down_variant(owner, WMMA):
        assert owner.q8_down_variant == WMMA
        assert owner.runner.q8_down_variant == WMMA
    assert owner.q8_down_variant == GUARDED
    assert owner.runner.q8_down_variant == GUARDED


def test_the_scope_restores_after_failure():
    owner = SimpleNamespace(q8_down_variant=GUARDED, runner=SimpleNamespace())

    with pytest.raises(RuntimeError, match="failed arm"):
        with runtime.q8_down_variant(owner, WMMA):
            assert owner.q8_down_variant == WMMA
            raise RuntimeError("failed arm")
    assert owner.q8_down_variant == GUARDED


def test_the_scope_restores_the_unbound_default():
    owner = SimpleNamespace(runner=SimpleNamespace())

    with runtime.q8_down_variant(owner, WMMA):
        assert owner.q8_down_variant == WMMA
    assert getattr(owner, "q8_down_variant", None) is None
    assert runtime.resolve_q8_down_variant(owner) == WMMA


# --- forwarding -------------------------------------------------------------


def test_generator_inheritance_carries_the_decision_to_later_runners():
    generator = SimpleNamespace(q8_down_variant=GUARDED)
    runner = SimpleNamespace()

    qwen4_exp_gguf._inherit_lane_decisions(generator, runner)

    assert runner.q8_down_variant == GUARDED


def test_the_ar_batch_carrier_forwards_the_published_variant(monkeypatch):
    def fake_batch(*args, **kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        batch_module, "Qwen4ExpGGUFBatchRunner", fake_batch)
    generator = SimpleNamespace(
        _resident=object(),
        runner=SimpleNamespace(max_sequence_length=8, runtime=None),
        backend="hip_gfx1151",
        q8_down_variant=GUARDED,
    )

    batch = qwen4_exp_ar_batch.make_qwen4_exp_ar_batch_runner(
        generator, state_slots=2)

    assert batch.q8_down_variant == GUARDED


def test_the_batch_step_forwards_the_variant_into_every_layer(monkeypatch):
    _install_step_fakes(monkeypatch)
    seen: list[tuple[str, object]] = []

    def layer(name):
        def run(residual_ptr, weights, *, rows, **kwargs):
            seen.append((name, kwargs.get("q8_down_variant")))
            return SimpleNamespace(ptr=residual_ptr + 0x10)

        return run

    monkeypatch.setattr(
        batch_module, "run_qwen4_exp_batch_gdn_layer", layer("gdn"))
    monkeypatch.setattr(
        batch_module, "run_qwen4_exp_batch_qsa_layer", layer("qsa"))
    runner = _fake_runner(slots=2)
    runner.q8_down_variant = GUARDED

    owners = [runner.admit(slot) for slot in range(2)]
    try:
        runner.step([1, 1], slots=[0, 1], owners=owners, capture_logits=False)
    finally:
        for slot, owner in enumerate(owners):
            runner.release(slot, owner)

    assert seen
    for name, variant in seen:
        assert variant == GUARDED, name


def test_the_shared_layer_ffn_forwards_the_variant_into_the_moe(monkeypatch):
    seen: list[dict[str, object]] = []

    monkeypatch.setattr(
        batch_module, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(
        batch_module, "run_qwen4_exp_gr_read",
        lambda *a, **k: SimpleNamespace(
            mixed=SimpleNamespace(ptr=0x4000),
            inject_logits=SimpleNamespace(ptr=0x5000),
        ),
    )
    monkeypatch.setattr(
        batch_module, "run_qwen4_exp_moe",
        lambda *a, **k: seen.append(k) or SimpleNamespace(
            output=SimpleNamespace(ptr=0x6000)),
    )
    monkeypatch.setattr(
        batch_module, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
    scratch = SimpleNamespace(
        attention_gr=SimpleNamespace(inject_logits=SimpleNamespace(ptr=0x1000)),
        after_attention=SimpleNamespace(ptr=0x2000),
        ffn_gr=object(),
        moe=object(),
        output=SimpleNamespace(ptr=0x3000),
    )
    weights = SimpleNamespace(
        ffn_gr=SimpleNamespace(
            norm_weight_ptr=0x10, down=0x11, up=0x12, inject=0x13),
        moe=object(),
    )

    batch_module._run_layer_ffn(
        0x9000, 0x9100, weights, scratch,
        q8_down_variant=GUARDED,
        production_moe_prefill_enabled=False,
        rows=4, branches=2, hidden=64, low_rank=16, ffn=128, experts=4,
        top_k=2, stream=0, runtime=SimpleNamespace(),
    )

    assert len(seen) == 1
    assert seen[0]["q8_down_variant"] == GUARDED


def test_the_mixed_target_forwards_the_batch_variant():
    """The shared FFN is behaviourally tested above; this hop is source-level.

    ``_verify_rows`` runs the whole mixed layer loop (QSA attention, GDN, PLE),
    so the only cheap check here is that its one ``_run_layer_ffn`` call hands
    the batch owner's field over instead of letting the dispatch default it.
    """

    from hipengine.runtime import qwen4_exp_mixed_target as mixed

    source = inspect.getsource(mixed)
    assert source.count("_run_layer_ffn(") == 1
    assert 'q8_down_variant=getattr(batch, "q8_down_variant", None)' in source


def test_the_captured_segment_key_carries_the_variant():
    runner = _fake_runner(slots=1)
    runner.q8_down_variant = WMMA
    wmma_key = runner._expert_down_policy()
    runner.q8_down_variant = GUARDED
    guarded_key = runner._expert_down_policy()

    assert WMMA in wmma_key
    assert GUARDED in guarded_key
    assert wmma_key != guarded_key


def test_the_c1_runner_carries_the_field_to_every_moe_call_site():
    signature = inspect.signature(runtime.run_qwen4_exp_moe)
    assert signature.parameters["q8_down_variant"].default is None

    source = inspect.getsource(runtime.Qwen4ExpGGUFResidentModelRunner)
    assert source.count("q8_down_variant=self.q8_down_variant") == 4
    for layer in (
        runtime.run_qwen4_exp_dense_qsa_layer,
        runtime.run_qwen4_exp_qsa_prefill_layer,
        runtime.run_qwen4_exp_gdn_layer,
    ):
        assert "q8_down_variant=q8_down_variant" in inspect.getsource(layer)
    # The class field is the unbound default, like the other owner fields.
    assert runtime.Qwen4ExpGGUFResidentModelRunner.q8_down_variant is None


# --- the surviving chain ----------------------------------------------------


def test_the_q8_down_chain_keeps_its_arms_and_terminal_strict_fallback():
    tests, terminal = _chain_arms(_q8_down_chain())

    assert len(tests) == 2, tests
    assert tests[0] == '_resolve_q8_selected_down(q8_selected_down_enabled)'
    assert "forkb_grouped_down" in tests[1]
    assert terminal.orelse, "the Q8_0 down chain has no terminal else"


# --- diagnostics ------------------------------------------------------------


def test_the_profile_gate_binds_the_variant_through_its_spec():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, CandidateSpec

    spec = {entry.name: entry for entry in dataclasses.fields(CandidateSpec)}
    assert spec["q8_down_variant"].default is None
    source = inspect.getsource(CandidateSpec)
    assert "q8_down_variant" in source

    for name, expected in (
        ("q8_blockscale", BLOCKSCALE),
        ("q8_blockscale_mmq", BLOCKSCALE),
        ("q8_blockscale_guarded", GUARDED),
        ("q8_blockscale_guarded_mmq", GUARDED),
        ("q8_blockscale_guarded_quad", GUARDED),
        ("q8_blockscale_guarded_restore", GUARDED),
        ("q8_blockscale_guarded_ordered", GUARDED),
    ):
        candidate = CANDIDATES[name]
        assert candidate.q8_down_variant == expected, name
        assert ENV not in candidate.environment, name
        assert candidate.q8_selected_down_enabled is True, name


def test_the_profile_gate_binds_the_field_after_the_binder():
    from scripts import qwen4exp_layer2_profile_gate as gate

    source = inspect.getsource(gate)
    assert "candidate_spec.q8_down_variant is not None" in source
    assert "enter_context(q8_down_variant(" in source
    assert "bind_candidate_q8_down_variant(" in source


def test_the_campaign_binds_the_leaf_its_counter_counts():
    from scripts.qwen4exp_halo_box_campaign_ab import (
        Q8_WMMA_DOWN_VARIANT,
        route_scope,
    )

    assert Q8_WMMA_DOWN_VARIANT == WMMA
    runner = SimpleNamespace(q8_down_variant=GUARDED)
    with route_scope("q8-wmma-down", "after", runner):
        assert runner.q8_down_variant == WMMA
    assert runner.q8_down_variant == GUARDED
    # The separate gate is also owner-local and follows the arm.
    runner = SimpleNamespace()
    with route_scope("q8-wmma-down", "before", runner):
        assert runner.q8_down_variant == WMMA
        assert runner.q8_selected_down_enabled is False


def test_the_state_gate_arm_control_binds_the_counted_leaf():
    from scripts.qwen4exp_row4_state_gate import arm_control

    runner = SimpleNamespace(q8_down_variant=GUARDED)
    env: dict[str, str] = {}
    with arm_control("q8-wmma-down", "1", runner, environment=env):
        assert runner.q8_down_variant == WMMA
        assert runner.q8_selected_down_enabled is True
        assert WMMA_GATE not in env
    assert runner.q8_down_variant == GUARDED


def test_the_shared_graph_allowlist_names_the_owner_field():
    from scripts import qwen4exp_prefill_restore_ab as restore

    assert ENV not in restore.SHARED_DECODE_SAFE_ENV
    assert WMMA_GATE not in restore.SHARED_DECODE_SAFE_ENV
    assert 'q8_selected_down_enabled' in restore.SHARED_DECODE_SAFE_OWNER_FIELDS
    # The GDN tile-16 prefill owner and the columnwarp admission gate are the
    # same kind of decision as the Q8_0 down owner: prefill-or-uncaptured
    # selectors, so graph identity ignores them and a candidate may bind them.
    assert restore.SHARED_DECODE_SAFE_OWNER_FIELDS == {
        "q8_down_variant", "gdn_tile16_variant", "gdn_colwarps_prefill",
        "qsa_ordered_decode", "q8_dense_wide_layers", "gdn_colwarps_layers",
        "qsa_flash_prefill_enabled", "q8_selected_down_enabled",
        "gr_iu8_up", "gr_iu8_down", "q4_iu8_prefill_enabled"}

    restore.validate_shared_graphs(
        [], [{"q8_down_variant": WMMA, "q8_selected_down_enabled": True}])
    with pytest.raises(ValueError, match="captured decode units"):
        restore.validate_shared_graphs([{WMMA_GATE: "1"}])
    restore.validate_shared_graphs(
        [], [{"gdn_tile16_variant": "qwen4exp_gdn_tiled16_multi_prefill"}])
    restore.validate_shared_graphs([], [{"gdn_colwarps_prefill": False}])
    with pytest.raises(ValueError, match="captured decode units"):
        restore.validate_shared_graphs([{"HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL": "1"}])
    with pytest.raises(ValueError, match="captured decode units"):
        restore.validate_shared_graphs([], [{"unrelated_owner_field": WMMA}])


def test_the_shared_graph_inventory_lists_only_actual_owner_changes():
    from scripts import qwen4exp_layer2_profile_gate as gate
    from scripts import qwen4exp_prefill_restore_ab as restore

    # A candidate that names an owner inventories that one field.
    assert gate.candidate_owner_changes(
        gate.CANDIDATES["q8_blockscale_guarded"]
    ) == {"q8_down_variant": GUARDED, "q8_selected_down_enabled": True}
    # A candidate that keeps the plan's published owner inventories nothing, so
    # the validator never sees a literal "unchanged" entry.
    assert gate.candidate_owner_changes(
        gate.CANDIDATES["production_baseline"]
    ) == {}
    # The conservative arm closes columnwarp, ordered decode and flash;
    # all real owner changes must be inventoried.
    assert gate.candidate_owner_changes(
        gate.CANDIDATES["production_conservative"]
    ) == {"gdn_colwarps_prefill": False, "qsa_ordered_decode": False,
          "qsa_flash_prefill_enabled": False, "q8_selected_down_enabled": False,
          "gr_iu8_up": False, "gr_iu8_down": False,
          "q4_iu8_prefill_enabled": False}
    # A ``None`` entry is not a change even when it is passed explicitly.
    restore.validate_shared_graphs([], [{"q8_down_variant": None}])
