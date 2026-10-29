"""Host routing contracts for the three global exp/mixed32 comparisons."""

from types import SimpleNamespace
import sys

import pytest

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache
from scripts.laguna_global_issue_diagnostic import global_issue_comparison
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
def owner(request):
    runtime = _Runtime()
    config = SimpleNamespace(
        block_count=4, layer_types=(FULL_ATTENTION, SLIDING_ATTENTION) * 2,
        head_counts=(48, 72) * 2, head_count_kv=8, key_length=128,
        value_length=128, sliding_window=512,
    )
    context_length, backend = getattr(request, "param", (4096, "hip_gfx1151"))
    session = object.__new__(LagunaGGUFResidentSession)
    session.kv_cache = allocate_laguna_kv_cache(
        config, context_length=context_length, backend=backend, runtime=runtime,
    )
    try:
        yield session
    finally:
        session.kv_cache.free()
        assert not runtime.allocations


_PAIRS = (
    ("assume_exp", "gqa2_vstage64_vec16_direct_fixedshape", "gqa2_vstage64_vec16_direct_assume_exp_fixedshape"),
    ("exp32", "gqa2_vstage64_vec16_direct_assume_exp_fixedshape", "gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape"),
    ("mixed32", "gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape", "mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape"),
)


@pytest.mark.parametrize("name,control,candidate", _PAIRS)
@pytest.mark.parametrize("live_count", [126, 127, 576, 4000, 4001])
def test_named_global_comparison_reaches_its_registered_pair(owner, name, control, candidate, live_count):
    cache = owner.kv_cache
    selected = []
    cache._resolve = lambda layer, variant: lambda *a, **kw: selected.append(variant)
    for position in range(live_count):
        cache.prepare_position(position)
    before = vars(cache).copy()
    args = SimpleNamespace(**{"compare_global_" + name: True})
    with global_issue_comparison(owner, args):
        for enabled in (False, True):
            getattr(owner, "set_decode_global_" + name)(enabled)
            assert cache.attend(0, 10, 20, gate_ptr=30, gated_out_ptr=40) is (live_count >= 127)
    assert vars(cache) == before
    if 127 <= live_count <= 4000:
        assert selected == [
            "global_context_fused_exact_gated_" + variant + "_spans"
            for variant in (control, candidate)
        ]
    else:
        assert selected[0] == selected[1]
        assert "assume_exp" not in selected[0]
    assert all(is_registered(KernelKey("hip_gfx1151", "laguna_attention_decode", "bf16", variant)) for variant in selected)


@pytest.mark.parametrize("owner", [(4095, "hip_gfx1151"), (4097, "hip_gfx1151")], indirect=True)
@pytest.mark.parametrize("name,control,candidate", _PAIRS)
def test_global_scope_preserves_generic_capacity_route(owner, name, control, candidate):
    cache = owner.kv_cache
    selected = []
    cache._resolve = lambda layer, variant: lambda *a, **kw: selected.append(variant)
    for position in range(576):
        cache.prepare_position(position)
    before = vars(cache).copy()
    with global_issue_comparison(owner, SimpleNamespace(**{"compare_global_" + name: True})):
        for enabled in (False, True):
            getattr(owner, "set_decode_global_" + name)(enabled)
            assert cache.attend(0, 10, 20, gate_ptr=30, gated_out_ptr=40)
    assert vars(cache) == before
    assert selected[0] == selected[1]
    assert "assume_exp_fixedshape" not in selected[0]
    assert all(is_registered(KernelKey(cache.backend, "laguna_attention_decode", "bf16", variant)) for variant in selected)


@pytest.mark.parametrize("owner", [(4096, "hip_gfx1100"), (4096, "hip_gfx1151")], indirect=True)
def test_unrequested_scope_preserves_peer_package_policy(owner):
    before = vars(owner.kv_cache).copy()
    with global_issue_comparison(owner, SimpleNamespace()):
        assert vars(owner.kv_cache) == before
    assert vars(owner.kv_cache) == before


@pytest.mark.parametrize("failure", [False, True])
def test_global_scope_restores_nested_policies_and_failed_body(owner, failure):
    cache = owner.kv_cache
    before = vars(cache).copy()
    with global_issue_comparison(owner, SimpleNamespace(compare_global_mixed32=True)):
        outer = vars(cache).copy()
        try:
            with global_issue_comparison(owner, SimpleNamespace(compare_global_assume_exp=True)):
                assert not cache.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
                owner.set_decode_global_assume_exp(True)
                if failure:
                    raise RuntimeError("synthetic comparison failure")
        except RuntimeError as error:
            assert failure and str(error) == "synthetic comparison failure"
        assert vars(cache) == outer
    assert vars(cache) == before


@pytest.mark.parametrize("field,value", [("split_gate_fusion", False), ("global_split_min_live", None)])
def test_global_scope_rejects_unavailable_reducer_policy_before_mutation(owner, field, value):
    setattr(owner.kv_cache, field, value)
    before = vars(owner.kv_cache).copy()
    with pytest.raises(ValueError, match="requires gated split attention"):
        with global_issue_comparison(owner, SimpleNamespace(compare_global_exp32=True)):
            pytest.fail("invalid comparison entered")
    assert vars(owner.kv_cache) == before


def test_global_scope_rejects_missing_cache():
    with pytest.raises(ValueError, match="requires gated split attention"):
        with global_issue_comparison(SimpleNamespace(kv_cache=None), SimpleNamespace(compare_global_exp32=True)):
            pytest.fail("invalid comparison entered")


def test_unrequested_global_scope_does_not_require_a_session():
    with global_issue_comparison(object(), SimpleNamespace()):
        pass


def test_global_scope_rejects_multiple_pairs_before_mutation(owner):
    before = vars(owner.kv_cache).copy()
    args = SimpleNamespace(compare_global_exp32=True, compare_global_mixed32=True)
    with pytest.raises(ValueError, match="only one global issue comparison"):
        with global_issue_comparison(owner, args):
            pytest.fail("invalid comparison entered")
    assert vars(owner.kv_cache) == before


@pytest.mark.parametrize("name,control,candidate", _PAIRS)
def test_profiler_enters_global_scope_and_restores_before_owner_close(
    owner, name, control, candidate, monkeypatch, tmp_path,
):
    model = tmp_path / "synthetic.gguf"
    model.touch()
    monkeypatch.setattr(sys, "argv", [
        "laguna_long_context_profile.py", str(model), "--lengths", "512",
        "--context-length", "4096", "--allow-dirty", "--compare-global-" + name.replace("_", "-"),
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
    selected = []
    cache._resolve = lambda layer, variant: lambda *a, **kw: selected.append(variant)
    for position in range(576):
        cache.prepare_position(position)
    before = vars(cache).copy()
    closed = []

    class Session:
        kv_cache = cache

        @property
        def prefill_chunk_size(self):
            for enabled in (False, True):
                getattr(owner, "set_decode_global_" + name)(enabled)
                assert cache.attend(0, 10, 20, gate_ptr=30, gated_out_ptr=40)
            assert selected == [
                "global_context_fused_exact_gated_" + variant + "_spans"
                for variant in (control, candidate)
            ]
            raise RuntimeError("profile reached scoped policy")

        def close(self):
            assert vars(cache) == before
            closed.append(True)

    monkeypatch.setattr(profiler, "LagunaGGUFResidentSession", lambda *a, **kw: Session())
    with pytest.raises(RuntimeError, match="profile reached scoped policy"):
        profiler.run(args)
    assert closed == [True] and vars(cache) == before
