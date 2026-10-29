"""The resident-session prefill route has a diagnostic override that actually reaches the sites.

Background: docs/REFACTOR.md records that both shipping call sites passed `use_wmma_prefill=True`
literally into `_acquire_shared_session`, and `HIPENGINE_GGUF_WMMA_PREFILL` is opt-in-only, so the
all-GEMV prefill arm could not be measured at all. These tests pin the escape hatch and, just as
importantly, pin that the call sites cannot silently go back to a literal.
"""
from __future__ import annotations

import pathlib

import pytest

MODULE = "hipengine.generation.qwen35_gguf"
ENV = "HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL"
SOURCE = pathlib.Path(__file__).resolve().parents[1] / "hipengine" / "generation" / "qwen35_gguf.py"


def _module():
    import importlib

    return importlib.import_module(MODULE)


@pytest.mark.parametrize("value", [None, "1", "true", "YES", " on "])
def test_unset_and_truthy_keep_the_production_route(monkeypatch, value):
    module = _module()
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert module._resident_session_wmma_prefill_default() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_falsy_values_take_the_wmma_route_away(monkeypatch, value):
    module = _module()
    monkeypatch.setenv(ENV, value)
    assert module._resident_session_wmma_prefill_default() is False


@pytest.mark.parametrize("value", ["", "  ", "garbage", "2", "none", "onoff"])
def test_unrecognised_values_fail_closed(monkeypatch, value):
    """A typo that reads as "route unchanged" turns a route A/B into an unexplained null."""
    module = _module()
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError) as exc:
        module._resident_session_wmma_prefill_default()
    assert ENV in str(exc.value)


def test_value_is_read_at_call_time_not_bound_at_import(monkeypatch):
    """The original bug bound a constant as a default argument, so import-time binding is banned.

    HIPENGINE_GGUF_WMMA_PREFILL could never override the shipping sites partly because the flag was
    captured at def time. If someone reintroduces that shape, the env stops working and the only
    symptom is a benchmark that measures the same route twice.
    """
    module = _module()
    monkeypatch.delenv(ENV, raising=False)
    assert module._resident_session_wmma_prefill_default() is True
    monkeypatch.setenv(ENV, "0")
    assert module._resident_session_wmma_prefill_default() is False


def test_resident_sessions_no_longer_pass_a_literal_route_flag():
    """No resident session acquire/construct may pin the route literally.

    The first version of this guard asserted exactly two sites and was wrong in a way that cost a
    2-minute trace to discover: the resident AR path opens its slots in `_open_ar_serving_slots` and
    `_reserve_sessions`, not in `prepare_request_scratch`, so wiring only the latter left shipping
    route pinned True and diagnostic inert. The assertion is now shaped like the requirement -
    AR and MTP-target pools must resolve - with one exception, plus an exact resolved-site
    count so silent unwiring shows up.
    """
    import re

    source = SOURCE.read_text()
    pinned = re.findall(r'pool_name="([a-z_0-9]+)",\s*\n\s*use_wmma_prefill=True,', source)
    # One deliberate exception: the dense speculative MTP route was never part of this seam.
    assert set(pinned) <= {"mtp_target_dense"}, f"unexpected literal-pinned pools: {pinned}"
    for pool in ("ar_batch", "continuous_ar_dynamic_kv", "mtp_target"):
        resolver = r"use_wmma_prefill=_resident_session_wmma_prefill_default\(\),"
        pattern = re.compile(rf'pool_name="{pool}",\s*\n\s*{resolver}')
        assert pattern.search(source), (
            f'the resident {pool} session must resolve its prefill route through the diagnostic '
            'resolver; a literal there cannot be overridden by any bench - the dead end this '
            'resolver exists to remove'
        )
    resolved = source.count("use_wmma_prefill=_resident_session_wmma_prefill_default(),")
    # 2 prepare_request_scratch + 4 mtp_target (both pooled/pool-off branches) + 2 ar_batch serving
    # slots + 2 continuous_ar_dynamic_kv reserve slots.
    assert resolved == 10, f"expected 10 resolved sites, found {resolved}"


def test_public_session_decode_preferences_have_one_owner():
    import ast

    tree = ast.parse(SOURCE.read_text())
    choices = [
        keyword.value
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        for keyword in node.keywords if keyword.arg == "use_gemv_decode"
    ]
    assert choices
    for choice in choices:
        assert isinstance(choice, ast.Name) and choice.id in {
            "_GGUF_PUBLIC_USE_GEMV_DECODE", "use_gemv_decode",
        }, ast.unparse(choice)
    module = _module()
    assert module._GGUF_PUBLIC_USE_GEMV_DECODE is True


def test_out_of_scope_literals_are_still_named():
    """Exactly one literal remains, on the dense speculative MTP route."""
    import re

    source = SOURCE.read_text()
    blocks = re.findall(r'pool_name="([a-z_0-9]+)",\s*\n\s*use_wmma_prefill=True,', source)
    assert blocks == ["mtp_target_dense"], blocks
    assert source.count("use_wmma_prefill=True,") == 1, (
        "the single remaining literal belongs to _generate_dense_speculative_mtp_detailed; if the "
        "count moved, a route was either wired (update the guard) or newly pinned"
    )


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("value,expected", (("0", False), ("1", True)))
def test_streaming_uses_the_same_prefill_selector_as_pooled_ar(monkeypatch, shared, value, expected):
    from types import SimpleNamespace
    from hipengine.generation.registry import GenerationRequest

    module = _module()
    cls = module.Qwen35GGUFBringupGenerator
    generator = object.__new__(cls)
    generator.model_path = "/tmp/fake.gguf"
    generator.backend = "hip_gfx1100"
    generator.weight_index = SimpleNamespace()
    generator.model_plugin = SimpleNamespace()
    generator.tokenizer = SimpleNamespace(encode=lambda text: [10, 11])
    generator._prepare_kv_policy = lambda request: None
    generator._prepared_session_kv_kwargs = lambda: {}
    runner = SimpleNamespace(runtime=object()) if shared else None
    generator._prepared_shared_runner = lambda: runner
    generator._configure_native_sampler = lambda session: None
    generator._stream_greedy = lambda *args, **kwargs: iter(())
    seen = []

    class Session:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setenv(ENV, value)
    list(generator.stream_detailed(GenerationRequest(
        prompts=("hello",), max_tokens=1, temperature=0, top_p=1, ignore_eos=True,
    )))
    assert len(seen) == 1
    assert seen[0]["use_wmma_prefill"] is expected
    assert seen[0]["use_gemv_decode"] is True
    assert seen[0].get("shared_runner") is runner


@pytest.mark.parametrize("shared", (False, True))
def test_fallback_session_scope_resolves_prefill_at_call_time(monkeypatch, shared):
    from contextlib import contextmanager
    from types import SimpleNamespace

    module = _module()
    cls = module.Qwen35GGUFBringupGenerator
    generator = object.__new__(cls)
    generator.model_path = "/tmp/fake.gguf"
    generator.backend = "hip_gfx1100"
    generator._prepared_session_kv_kwargs = lambda: {}
    generator._configure_session = lambda session: None
    generator._release_shared_session = lambda *args: None
    seen = []

    @contextmanager
    def session(*args, **kwargs):
        seen.append(kwargs)
        yield object()

    def acquire(runner, **kwargs):
        seen.append(kwargs)
        return object(), ("fixture",), False

    monkeypatch.setattr(module, "Qwen35GGUFResidentSession", session)
    generator._acquire_shared_session = acquire
    runner = object() if shared else None
    for value, expected in (("0", False), ("1", True)):
        monkeypatch.setenv(ENV, value)
        with generator._resident_session_scope(shared_runner=runner, pool_name="ar"):
            pass
        assert seen[-1]["use_wmma_prefill"] is expected
        assert seen[-1]["use_gemv_decode"] is True

    monkeypatch.setenv(ENV, "invalid")
    with generator._resident_session_scope(shared_runner=runner, pool_name="ar", use_wmma_prefill=False):
        pass
    assert seen[-1]["use_wmma_prefill"] is False


@pytest.mark.parametrize("pool_name", ["ar_batch", "continuous_ar_dynamic_kv", "mtp_target", "compact_dms_request"])
def test_resolved_prefill_choice_partitions_pool_and_reuses_matching_session(monkeypatch, pool_name):
    from types import SimpleNamespace

    module = _module()
    cls = module.Qwen35GGUFBringupGenerator
    generator = object.__new__(cls)
    generator._prepared_kv_signature = ("fixture",)
    generator._prepared_max_sequence_length = 64
    monkeypatch.setattr(cls, "_configure_session", lambda self, session: None)
    constructed = []

    def construct(self, runner, **kwargs):
        session = SimpleNamespace(
            max_sequence_length=kwargs["max_sequence_length"],
            use_wmma_prefill=kwargs["use_wmma_prefill"],
            reset=lambda: None,
        )
        constructed.append(session)
        return session

    monkeypatch.setattr(cls, "_construct_shared_session", construct)

    def acquire(value):
        monkeypatch.setenv(ENV, value)
        return generator._acquire_shared_session(
            object(), pool_name=pool_name,
            use_wmma_prefill=module._resident_session_wmma_prefill_default(),
            use_gemv_decode=True,
        )

    enabled, enabled_key, reused = acquire("1")
    assert not reused and enabled.use_wmma_prefill is True
    generator._release_shared_session(enabled_key, enabled)
    disabled, disabled_key, reused = acquire("0")
    assert not reused and disabled.use_wmma_prefill is False
    assert enabled_key != disabled_key
    assert enabled.use_wmma_prefill is True
    generator._release_shared_session(disabled_key, disabled)
    again, key, reused = acquire("true")
    assert reused and again is enabled and key == enabled_key
    assert len(constructed) == 2
