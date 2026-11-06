"""Laguna comparison controls belong to one resident owner."""
import pytest

from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


@pytest.mark.parametrize("bounded", [False, True])
def test_long_attention_lazy_owners_are_separate_and_reused(monkeypatch, bounded):
    from types import SimpleNamespace
    from hipengine.runtime import laguna_gguf_runner as runner

    allocations = []

    def allocate(**kwargs):
        route = SimpleNamespace(**kwargs)
        allocations.append(route)
        return route

    monkeypatch.setattr(runner, "LagunaAttentionHipblasLt", allocate)
    owner = object.__new__(LagunaGGUFResidentSession)
    owner.runtime = object()
    owner.context_length = 8_231
    owner.long_attention_hipblaslt = owner.block_attention_hipblaslt = None
    route = owner._ensure_block_attention_hipblaslt(256) if bounded else owner._ensure_long_attention_hipblaslt()
    again = owner._ensure_block_attention_hipblaslt(256) if bounded else owner._ensure_long_attention_hipblaslt()
    assert route is again
    assert len(allocations) == 1
    assert route.max_context == 8_192
    assert route.max_q_heads == 48
    assert route.packed_queries and route.wave_rows_softmax
    assert route.runtime is owner.runtime
    if bounded:
        assert route.block_context == 4_096 and route.query_rows == 256
        with pytest.raises(RuntimeError, match="query width changed"):
            owner._ensure_block_attention_hipblaslt(128)
        other = owner._ensure_long_attention_hipblaslt()
        assert not hasattr(other, "block_context")
    else:
        assert not hasattr(route, "block_context")
        other = owner._ensure_block_attention_hipblaslt(128)
        assert other.block_context == 4_096 and other.query_rows == 128
    assert route is not other
    assert len(allocations) == 2


def test_attention_comparison_scope_restores_after_nested_failure():
    owner = LagunaGGUFResidentSession.__new__(LagunaGGUFResidentSession)
    owner.prefill_swa_attention_hipblaslt = True
    owner.prefill_dense_contiguous_cache = False
    with owner.prefill_attention_diagnostic(swa=False):
        assert owner.prefill_swa_attention_hipblaslt is False
        with pytest.raises(RuntimeError):
            with owner.prefill_attention_diagnostic(dense_contiguous=True):
                assert owner.prefill_swa_attention_hipblaslt is False
                assert owner.prefill_dense_contiguous_cache is True
                raise RuntimeError('failed')
        assert owner.prefill_dense_contiguous_cache is False
    assert owner.prefill_swa_attention_hipblaslt is True
    assert owner.prefill_dense_contiguous_cache is False


@pytest.mark.parametrize("option, attr", [
    ("long_attention", "prefill_long_attention_hipblaslt"),
    ("block_attention", "prefill_block_attention_hipblaslt"),
])
def test_long_attention_scope_restores_nested_setup_failure(option, attr):
    owner = object.__new__(LagunaGGUFResidentSession)
    setattr(owner, attr, True)

    class InvalidChoice:
        def __bool__(self):
            raise RuntimeError("invalid comparison")

    with owner.prefill_attention_diagnostic(**{option: False}):
        assert getattr(owner, attr) is False
        with pytest.raises(RuntimeError, match="invalid comparison"):
            with owner.prefill_attention_diagnostic(**{option: InvalidChoice()}):
                pytest.fail("failed setup entered body")
        assert getattr(owner, attr) is False
    assert getattr(owner, attr) is True


@pytest.mark.parametrize("name", [
    "set_prefill_long_attention_hipblaslt",
    "set_prefill_block_attention_hipblaslt",
    "set_prefill_swa_attention_hipblaslt",
    "set_prefill_dense_contiguous_cache",
])
def test_persistent_attention_comparison_setters_are_retired(name):
    assert not hasattr(LagunaGGUFResidentSession, name)


def test_attention_comparison_setup_failure_restores_only_selected_owner():
    owner = LagunaGGUFResidentSession.__new__(LagunaGGUFResidentSession)
    sibling = LagunaGGUFResidentSession.__new__(LagunaGGUFResidentSession)
    for session in (owner, sibling):
        session.prefill_swa_attention_hipblaslt = True
        session.prefill_dense_contiguous_cache = False

    class InvalidChoice:
        def __bool__(self):
            owner.prefill_dense_contiguous_cache = True
            raise RuntimeError('configuration failed')

    with pytest.raises(RuntimeError, match='configuration failed'):
        with owner.prefill_attention_diagnostic(swa=False, dense_contiguous=InvalidChoice()):
            pytest.fail('failed setup must not enter the comparison')
    for session in (owner, sibling):
        assert session.prefill_swa_attention_hipblaslt is True
        assert session.prefill_dense_contiguous_cache is False
