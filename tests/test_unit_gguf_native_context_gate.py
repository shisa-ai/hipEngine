from types import SimpleNamespace

import pytest

from scripts.gguf_native_context_gate import fixed_prompt, native_context_override


def test_native_context_override_restores_both_limits_after_failure():
    package = SimpleNamespace(
        GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT=95,
        GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT=95,
        GGUF_SPECDEC2_NATIVE_TARGET_CACHE_CAPACITY_POLICIES=frozenset({"dense"}),
    )
    with pytest.raises(RuntimeError, match="probe"):
        with native_context_override(package, 256) as prior:
            assert set(prior.values()) == {95}
            assert package.GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT == 256
            assert package.GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT == 256
            assert package.GGUF_SPECDEC2_NATIVE_TARGET_CACHE_CAPACITY_POLICIES == frozenset()
            raise RuntimeError("probe")
    assert package.GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT == 95
    assert package.GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT == 95
    assert package.GGUF_SPECDEC2_NATIVE_TARGET_CACHE_CAPACITY_POLICIES == frozenset({"dense"})


def test_unbounded_backend_override_does_not_install_a_permanent_limit():
    package = SimpleNamespace()
    with native_context_override(package, 95) as prior:
        assert set(prior.values()) == {None}
        assert package.GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT == 95
    assert vars(package) == {}
    with native_context_override(package, None):
        assert vars(package) == {}
    assert vars(package) == {}


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        with native_context_override(SimpleNamespace(), limit):
            pass


def test_fixed_prompt_has_exact_length_and_unchanged_prefix():
    assert fixed_prompt([1, 2, 3], 8) == [1, 2, 3, 1, 2, 3, 1, 2]
    assert fixed_prompt([1, 2, 3], 2) == [1, 2]
    assert fixed_prompt([1, 2, 3], None) == [1, 2, 3]
    with pytest.raises(ValueError):
        fixed_prompt([], 128)


def test_forced_acceptance_changes_only_first_rejected_candidate():
    from scripts.gguf_native_context_state_gate import make_candidates

    greedy = [5, 9, 2]
    assert make_candidates(greedy, 0, 10) == [6, 9, 2]
    assert make_candidates(greedy, 1, 10) == [5, 0, 2]
    assert make_candidates(greedy, 3, 10) == greedy
    assert greedy == [5, 9, 2]


def test_state_gate_rejects_cursor_or_component_mismatch():
    from scripts.gguf_native_context_state_gate import compare_state

    expected = {"position": 128, "layer_conv_states/0": "abc", "hidden": "def"}
    compare_state(dict(expected), expected)
    with pytest.raises(AssertionError, match="position"):
        compare_state({**expected, "position": 127}, expected)
    with pytest.raises(AssertionError, match="layer_conv_states/0"):
        compare_state({**expected, "layer_conv_states/0": "xyz"}, expected)


def test_foreign_gpu_monitor_is_device_and_pid_specific(tmp_path):
    from scripts.gguf_native_context_gate import foreign_gpu_allocations

    for pid, gpu, size in ((10, 33912, 2 << 30), (20, 47021, 3 << 30),
                           (30, 33912, 4 << 30), (40, 33912, 100)):
        folder = tmp_path / str(pid)
        folder.mkdir(exist_ok=True)
        (folder / f"vram_{gpu}").write_text(str(size))
    assert foreign_gpu_allocations(tmp_path, 33912, 10) == {30: 4 << 30}
