"""Short attention is bound to the profile owner, not ambient thread controls."""
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation import qwen35_gguf as gguf
from hipengine.generation.qwen35_dense_gfx1151_profiles import (
    _strict_binder, _production_binder, _strict_selections, _production_selections,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import resolve
from hipengine.runtime import qwen35_gguf_runner as qgr

STRICT = "bf16_context_batch_c1_exact_spans"
PRODUCTION = "bf16_context_batch_threads1024_spans"


def owner(profile):
    generator = object.__new__(gguf.Qwen35GGUFBringupGenerator)
    resolved = SimpleNamespace(manifest={"backend": "hip_gfx1151"}, profile=profile)
    (_strict_binder if profile is ExecutionProfile.STRICT else _production_binder)(generator, resolved)
    generator.model_path = "unused.gguf"
    generator.backend = "hip_gfx1151"
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    generator._auto_context_estimate = None
    return generator


@pytest.mark.parametrize("order", [(ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION),
                                    (ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT)])
def test_profile_owns_short_attention_and_manifest_without_environment_writes(monkeypatch, order):
    name = "HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS"
    monkeypatch.setenv(name, "512")
    import os
    owners = {profile: owner(profile) for profile in order}
    for profile, variant, selections in (
        (ExecutionProfile.STRICT, STRICT, _strict_selections()),
        (ExecutionProfile.PRODUCTION, PRODUCTION, _production_selections()),
    ):
        assert owners[profile].target_short_c1_attn_variant == variant
        selection = next(s for s in selections if s.scope == "short_c1_attention")
        assert selection.layer == "paged_attn_decode"
        assert selection.registry_quant == "w4_paro"
        assert selection.selected_variant == variant
        assert selection.strict_fallback_variant == STRICT
    assert os.environ[name] == "512"


def test_strict_fallback_keeps_strict_attention_for_an_unregistered_profile():
    generator = object.__new__(gguf.Qwen35GGUFBringupGenerator)
    _strict_binder(generator, SimpleNamespace(manifest={"backend": "hip_gfx1151"},
                                             profile=ExecutionProfile.FAST))
    assert generator.target_short_c1_attn_variant == STRICT


@pytest.mark.parametrize("variant,threads", [(STRICT, None), (PRODUCTION, 1024)])
def test_bound_runner_resolves_registered_leaf_despite_hostile_environment(monkeypatch, variant, threads):
    register_gfx1151_kernels(replace=True)
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1151"
    runner.target_short_c1_attn_variant = variant
    monkeypatch.setenv("HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS", "512")
    leaf = runner._full_attn_decode_short_batch_fn(object())
    assert leaf is resolve(backend="hip_gfx1151", layer="paged_attn_decode", quant="w4_paro", variant=variant)
    if threads is not None:
        assert leaf.keywords["threads"] == threads
    monkeypatch.setenv("HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS", "256")
    assert runner._full_attn_decode_short_batch_fn(object()) is leaf


@pytest.mark.parametrize("profile,variant", [(ExecutionProfile.STRICT, STRICT),
                                             (ExecutionProfile.PRODUCTION, PRODUCTION)])
def test_profile_variant_reaches_shared_runner_and_both_session_lifecycles(monkeypatch, profile, variant):
    captured = []
    class FakeRunner:
        runtime = None
        def __init__(self, *args, **kwargs):
            captured.append(("runner", kwargs))
    class FakeSession:
        max_sequence_length = None
        def __init__(self, *args, **kwargs):
            captured.append(("session", kwargs))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(gguf, "Qwen35GGUFFullStackRunner", FakeRunner)
    monkeypatch.setattr(gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = owner(profile)
    runner = generator._get_shared_runner()
    generator._construct_shared_session(runner, max_sequence_length=None, max_batch_size=1,
                                        defer_kv_allocation=False, use_wmma_prefill=True, use_gemv_decode=True)
    monkeypatch.setattr(generator, "_configure_session", lambda session: None)
    with generator._resident_session_scope(shared_runner=None, pool_name="test"):
        pass
    assert [kind for kind, _ in captured] == ["runner", "session", "session"]
    assert all(kwargs["target_short_c1_attn_variant"] == variant for _, kwargs in captured)
    # Private sessions also accept the same owner value for their own runner.
    assert qgr.Qwen35GGUFResidentSession.__dataclass_fields__["target_short_c1_attn_variant"].default is None


@pytest.mark.parametrize("requested", [None, STRICT, PRODUCTION])
@pytest.mark.parametrize("live", [None, STRICT, PRODUCTION])
def test_shared_session_rejects_a_different_attention_owner(monkeypatch, requested, live):
    class ReachedScratchBoundary(Exception):
        pass
    shared = SimpleNamespace(backend="hip_gfx1151", runtime=object(),
                             target_short_c1_attn_variant=live, fp16_recurrent_state=False,
                             verifier_prefill_gdn_capture=True)
    monkeypatch.setattr(qgr, "resolve_backend", lambda backend: "hip_gfx1151")
    monkeypatch.setattr(qgr, "backend_package_capability", lambda backend, name, default=None: default)
    monkeypatch.setattr(qgr, "_resolve_gguf_private_c1_small_weight_arena", lambda **kw: (False, "test"))
    monkeypatch.setattr(qgr, "_resolve_gguf_private_c1_decode_scratch_arena", lambda **kw: (False, "test"))
    def stop_before_scratch(value):
        raise ReachedScratchBoundary
    monkeypatch.setattr(qgr, "_resolve_target_verifier_production_q4_rowtile", stop_before_scratch)
    mismatch = requested is not None and requested != live
    with pytest.raises(RuntimeError if mismatch else ReachedScratchBoundary) as caught:
        qgr.Qwen35GGUFResidentSession("unused", shared_runner=shared, runtime=object(),
                                     target_short_c1_attn_variant=requested)
    if mismatch:
        assert "short attention variant" in str(caught.value)
    assert shared.target_short_c1_attn_variant == live


def test_private_session_passes_variant_to_its_owned_runner(monkeypatch):
    captured = {}
    class ReachedRunner(Exception):
        pass
    def construct_runner(*args, **kwargs):
        captured.update(kwargs)
        raise ReachedRunner
    monkeypatch.setattr(qgr, "resolve_backend", lambda backend: "hip_gfx1151")
    monkeypatch.setattr(qgr, "backend_package_capability", lambda backend, name, default=None: default)
    monkeypatch.setattr(qgr, "_resolve_gguf_private_c1_small_weight_arena", lambda **kw: (False, "test"))
    monkeypatch.setattr(qgr, "_resolve_gguf_private_c1_decode_scratch_arena", lambda **kw: (False, "test"))
    monkeypatch.setattr(qgr, "Qwen35GGUFFullStackRunner", construct_runner)
    with pytest.raises(ReachedRunner):
        qgr.Qwen35GGUFResidentSession("unused", runtime=object(), target_short_c1_attn_variant=STRICT)
    assert captured["target_short_c1_attn_variant"] == STRICT


