"""Physical row policy freezes diagnostic inputs at owner construction."""

from dataclasses import FrozenInstanceError

import pytest

from hipengine.generation.physical_target_rows import resolve_physical_target_rows


def test_target_submission_shape_tracks_jobs_and_clears_on_new_cycle():
    from types import SimpleNamespace
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._last_target_shape_contract = {"reason": "padded_to_row_multiple"}
    jobs = [{"input_token_ids": (1, 2, 3)}, {"input_token_ids": (4, 5, 6)}]
    adapter._record_target_submission_shape(jobs, pad_rows=2)
    assert adapter.physical_width_contract()["last_target_shape"] == {
        "logical_rows": 4, "pad_rows": 2, "selected_rows": 6,
        "reason": "padded_to_row_multiple", "submitted_rows": 6, "request_count": 2,
    }
    adapter._states = {}
    # Stop at request lookup, after entering the real cycle boundary.
    with pytest.raises(KeyError):
        adapter._execute_target_frontier_batch(
            SimpleNamespace(speculative_request_ids=(7,)), None, None,
            cancelled_request_ids=lambda: (),
        )
    assert adapter.physical_width_contract()["last_target_shape"] == {}
    adapter._record_target_submission_shape(jobs[:1], pad_rows=0)
    assert adapter.physical_width_contract()["last_target_shape"] == {
        "logical_rows": 3, "pad_rows": 0, "selected_rows": 3,
        "reason": "padding_not_applicable", "submitted_rows": 3, "request_count": 1,
    }



@pytest.mark.parametrize("profile", ["strict", "production", "batch_invariant"])
def test_rows_plan_snapshots_rollbacks(monkeypatch, profile):
    import hipengine.generation.physical_target_rows as module

    def capability(backend, name, default):
        assert backend == "test_backend"
        return {
            "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS": (6,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS": (8,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS": (6,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q5_ROWTILE_ROWS": (6,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXTRA_ROWTILE_SHAPES": ((8, 8),),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS": {8: (6, 2)},
            "GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS_POLICY": {
                "rows": (28,), "enabled_env": "TEST_EXACT_ROWS", "enabled_default": True,
            },
        }.get(name, default)

    monkeypatch.setattr(module, "backend_package_capability", capability)
    monkeypatch.delenv("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", raising=False)
    monkeypatch.delenv("TEST_EXACT_ROWS", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", raising=False)
    initial = resolve_physical_target_rows("test_backend", profile)
    assert initial.exact_counts == ((8, 28) if profile == "production" else ())
    assert initial.pad_counts == ((6,) if profile == "production" else ())
    assert initial.extra_q4_rowtiles is (profile == "production")
    assert initial.q5_rowtiles is (profile == "production")
    assert initial.q6_rowtiles is (profile == "production")
    assert initial.mixed_q6_rowtiles is (profile == "production")
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", "0")
    assert not resolve_physical_target_rows("test_backend", profile).mixed_q6_rowtiles
    assert initial.mixed_q6_rowtiles is (profile == "production")
    monkeypatch.setenv("TEST_EXACT_ROWS", "0")
    subsequent = resolve_physical_target_rows("test_backend", profile)
    assert subsequent.exact_counts == ((8,) if profile == "production" else ())
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", "0")
    assert resolve_physical_target_rows("test_backend", profile).exact_counts == ()
    assert initial.exact_counts == ((8, 28) if profile == "production" else ())
    with pytest.raises(FrozenInstanceError):
        initial.exact_counts = ()


@pytest.mark.parametrize("raw", [None, "12", b"12", 3])
def test_prompt_policy_rejects_nonsequence_widths(monkeypatch, raw):
    import hipengine.generation.physical_target_rows as module

    monkeypatch.setattr(module, "backend_package_capability", lambda *args: {"key": raw})
    with pytest.raises(RuntimeError, match="widths must be a sequence"):
        module.resolve_physical_prompt_policies("test_backend")


@pytest.mark.parametrize("raw", [[None], ["bad"], [float("inf")]])
def test_prompt_policy_names_invalid_integer_widths(monkeypatch, raw):
    import hipengine.generation.physical_target_rows as module

    monkeypatch.setattr(module, "backend_package_capability", lambda *args: {"key": raw})
    with pytest.raises(RuntimeError, match="widths must be integers"):
        module.resolve_physical_prompt_policies("test_backend")


@pytest.mark.parametrize("raw", [[0], [9]])
def test_prompt_policy_rejects_out_of_bounds_widths(monkeypatch, raw):
    import hipengine.generation.physical_target_rows as module

    monkeypatch.setattr(module, "backend_package_capability", lambda *args: {"key": raw})
    with pytest.raises(RuntimeError, match="widths must be within"):
        module.resolve_physical_prompt_policies("test_backend")


def test_bound_plan_is_immutable_and_authoritative_for_widths(monkeypatch):
    from types import SimpleNamespace
    import hipengine.generation.physical_target_rows as module
    from hipengine.core.specdec2_scope import PhysicalTargetArithmeticPlan
    from hipengine.generation.qwen35_gguf_mtp2 import (
        _physical_width_depths, _physical_prompt_streaming_widths,
    )
    from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY

    monkeypatch.setattr(module, "backend_package_capability", lambda backend, name, default: (
        {(QWEN35_DENSE_H5120_GEOMETRY, None, "strict"): (1, 2)}
        if name == "GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES" else {}
    ))
    owner = SimpleNamespace()
    module.bind_physical_target_plan(
        owner, "", "strict", arithmetic=PhysicalTargetArithmeticPlan(),
    )
    plan = owner.physical_target_plan
    owner.physical_target_width_depths = ((8, 7),)
    assert _physical_width_depths(owner) == plan.width_depths
    assert owner.physical_target_rows is plan.rows
    assert owner.physical_target_arithmetic is plan.arithmetic
    owner.execution_profile = "production"
    adapter_owner = SimpleNamespace(_shared_runner=SimpleNamespace(weights=SimpleNamespace(
        geometry=QWEN35_DENSE_H5120_GEOMETRY, file_type_name="unknown",
    )))
    assert _physical_prompt_streaming_widths(adapter_owner, owner) == (1, 2)
    with pytest.raises(FrozenInstanceError):
        plan.profile = "production"


def test_prompt_policy_snapshot_is_deeply_immutable(monkeypatch):
    import hipengine.generation.physical_target_rows as module

    widths = [4, 1, 4]
    policies = {("geometry", None, "production"): widths}
    monkeypatch.setattr(module, "backend_package_capability", lambda *args: policies)
    snapshot = module.resolve_physical_prompt_policies("test_backend")
    widths.append(8)
    policies.clear()
    assert snapshot[("geometry", None, "production")] == (1, 4)
    with pytest.raises(TypeError):
        snapshot[("geometry", None, "production")] = (8,)
