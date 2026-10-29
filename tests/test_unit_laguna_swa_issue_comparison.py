"""Host contracts for historical SWA issue-schedule comparison routes."""

from types import SimpleNamespace
import sys

import pytest

from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.kernels.registry import KernelKey, is_registered
from scripts.laguna_swa_issue_diagnostic import swa_issue_comparison
from scripts import laguna_long_context_profile as profiler


class _Runtime:
    def __init__(self):
        self.allocations = {}
        self.next_ptr = 0x10000000

    def malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += nbytes + 4096
        self.allocations[ptr] = nbytes
        return ptr

    def free(self, ptr):
        del self.allocations[ptr]

    def memcpy(self, *args):
        pass

    def memset(self, *args):
        pass


@pytest.fixture
def owner():
    runtime = _Runtime()
    config = SimpleNamespace(
        block_count=4, layer_types=(FULL_ATTENTION, SLIDING_ATTENTION) * 2,
        head_counts=(48, 72) * 2, head_count_kv=8, key_length=128,
        value_length=128, sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(config, context_length=4096, backend="hip_gfx1151", runtime=runtime)
    session = object.__new__(LagunaGGUFResidentSession)
    session.kv_cache = cache
    try:
        yield session
    finally:
        cache.free()
        assert not runtime.allocations


@pytest.mark.parametrize("setter,expected", [
    ("set_decode_swa_assume_exp", ("gqa3_vstage64_vec16_direct_fixed512", "gqa3_vstage64_vec16_direct_assume_exp_fixed512")),
    ("set_decode_swa_mixed32", ("gqa3_vstage64_vec16_direct_assume_exp_fixed512", "mixed32_vstage64_vec16_direct_assume_exp_fixed512")),
    ("set_decode_swa_mixed32_exp4", ("mixed32_vstage64_vec16_direct_assume_exp_fixed512", "mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512")),
    ("set_decode_swa_mixed32_exp8", ("mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512", "mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512")),
    ("set_decode_swa_mixed32_exp16", ("mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512", "mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512")),
    ("set_decode_swa_mixed32_exp32", ("mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512", "mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512")),
])
@pytest.mark.parametrize("position", [510, 511, 777])
def test_named_swa_comparison_reaches_its_two_registered_routes(owner, setter, expected, position):
    cache = owner.kv_cache
    selected = []
    cache._resolve = lambda layer, variant: lambda *a, **kw: selected.append(variant)
    for step in range(position + 1):
        cache.prepare_position(step)
    args = SimpleNamespace(**{"compare_" + setter.removeprefix("set_decode_"): True})
    before = vars(cache).copy()
    with swa_issue_comparison(owner, args):
        for enabled in (False, True):
            getattr(owner, setter)(enabled)
            assert cache.attend(1, 10, 20, gate_ptr=30, gated_out_ptr=40)
    assert vars(cache) == before
    if position >= 511:
        assert selected == ["swa_context_fused_exact_gated_" + variant + "_spans" for variant in expected]
        assert all(is_registered(KernelKey("hip_gfx1151", "laguna_attention_decode", "bf16", variant)) for variant in selected)
    else:
        assert selected[0] == selected[1]
        assert not any("assume_exp" in name for name in selected)


@pytest.mark.parametrize("failure", [False, True])
def test_swa_issue_scope_restores_nested_policies_and_failed_body(owner, failure):
    cache = owner.kv_cache
    before = vars(cache).copy()
    with swa_issue_comparison(owner, SimpleNamespace(compare_swa_mixed32_exp16=True)):
        outer = vars(cache).copy()
        try:
            with swa_issue_comparison(owner, SimpleNamespace(compare_swa_assume_exp=True)):
                assert not cache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
                owner.set_decode_swa_assume_exp(True)
                if failure:
                    raise RuntimeError("synthetic comparison failure")
        except RuntimeError as error:
            assert failure and str(error) == "synthetic comparison failure"
        assert vars(cache) == outer
    assert vars(cache) == before


def test_unrequested_swa_diagnostic_does_not_need_or_change_cache():
    with swa_issue_comparison(object(), SimpleNamespace()):
        pass


@pytest.mark.parametrize("field,value", [
    ("split_gate_fusion", False), ("swa_split_wave_local", False),
    ("swa_split_min_live", None), ("swa_split_tile16_min_live", None),
])
def test_swa_issue_scope_rejects_unavailable_reducer_policy_before_mutation(owner, field, value):
    setattr(owner.kv_cache, field, value)
    before = vars(owner.kv_cache).copy()
    with pytest.raises(ValueError, match="requires split attention"):
        with swa_issue_comparison(owner, SimpleNamespace(compare_swa_assume_exp=True)):
            pytest.fail("invalid comparison entered")
    assert vars(owner.kv_cache) == before


def test_profile_enters_and_restores_swa_scope_before_owner_cleanup(owner, monkeypatch, tmp_path):
    model = tmp_path / "synthetic.gguf"
    model.touch()
    monkeypatch.setattr(sys, "argv", [
        "laguna_long_context_profile.py", str(model), "--lengths", "512",
        "--context-length", "512", "--allow-dirty", "--compare-swa-mixed32-exp8",
    ])
    args = profiler._parse_args()
    monkeypatch.setattr(profiler, "_repo_state", lambda: {"tracked_clean": True})
    monkeypatch.setattr(profiler, "get_hip_runtime", lambda: SimpleNamespace(mem_get_info=lambda: (10, 20)))
    monkeypatch.setattr(profiler, "collect_artifact_provenance", lambda **kw: {})
    monkeypatch.setattr(profiler, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(profiler.LagunaGGUFTokenizer, "from_gguf_info", lambda info: object())
    monkeypatch.setattr(profiler, "_load_prompts", lambda *a: [])
    monkeypatch.setattr(profiler, "_profile_token_stream", lambda *a: ([1] * 512, "synthetic"))
    monkeypatch.setattr(profiler, "memory_stats", lambda: {})
    monkeypatch.setattr(profiler, "_compiler_version", lambda path: "synthetic")
    cache = owner.kv_cache
    before = vars(cache).copy()
    closed = []

    class Session:
        kv_cache = cache

        @property
        def prefill_chunk_size(self):
            assert not cache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
            assert cache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
            raise RuntimeError("profile reached scoped policy")

        def close(self):
            assert vars(cache) == before
            closed.append(True)

    monkeypatch.setattr(profiler, "LagunaGGUFResidentSession", lambda *a, **kw: Session())
    with pytest.raises(RuntimeError, match="profile reached scoped policy"):
        profiler.run(args)
    assert closed == [True] and vars(cache) == before
