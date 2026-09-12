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


@pytest.mark.parametrize("value", ["garbage", "2", "none", "onoff"])
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
