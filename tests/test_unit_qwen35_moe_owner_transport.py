"""R0137 owner transport for the gfx1151 MoE router and Q8T16 rowtile policy.

``HIPENGINE_GGUF_ROUTER_F32W_COOP``, ``HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER``
and ``HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL`` used to be the carrier between the
Qwen3.5 MoE gfx1151 profile binder and its runtime: the binder rewrote the
process environment, so a strict and a production owner in one process
retargeted each other, and the strict rowtile ``0`` overrode the runner's own
packed-decode session.

These tests pin the replacement transport: the profile binds explicit
construction-owned values on the generator, every runner/session/NextN/draft
constructor carries them, and runtime consumption is explicit. The ambient
environment stays the unbound/direct diagnostic input, with its existing
per-call semantics, and the rowtile ambient override keeps its documented
precedence whenever no owner decision exists.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen35_moe_gfx1151_profiles import (
    QWEN36_GGUF_BACKEND,
    QWEN36_GGUF_MODEL,
    QWEN36_GGUF_QUANT,
    ROUTER_COOP_ENV,
    ROUTER_PERSISTENT_ENV,
    ROWTILE_ALL_ENV,
    register_qwen36_gguf_gfx1151_profiles,
)

# Register the gfx1151 package at collection time so the profile-resolution
# variant gate and the package-floor helper see the shipped capabilities.
from hipengine.kernels.backends import load_backend_kernel_package  # noqa: E402

load_backend_kernel_package(QWEN36_GGUF_BACKEND)

_POLICY_ENV = (ROUTER_COOP_ENV, ROUTER_PERSISTENT_ENV, ROWTILE_ALL_ENV)
_OWNER_FIELDS = (
    "target_router_f32w_coop",
    "target_router_f32w_persistent_counter",
    "target_q8_t16_rowtile_all",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    for key in _POLICY_ENV:
        monkeypatch.delenv(key, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()
    from hipengine.runtime import gguf_linear

    gguf_linear.set_q8_t16_rowtile_all_enabled(None)
    gguf_linear.set_q8_t16_pair_rowtile_min_rows(None)


def _new_generator():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    return object.__new__(Qwen35GGUFBringupGenerator)


def _bare_runner(
    *,
    coop: bool | None = None,
    persistent: bool | None = None,
    rowtile: bool | None = None,
    backend: str = QWEN36_GGUF_BACKEND,
):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    runner = object.__new__(runner_mod.Qwen35GGUFFullStackRunner)
    runner.backend = backend
    runner.router_f32w_coop = coop
    runner.router_f32w_persistent_counter = persistent
    runner.q8_t16_rowtile_all = rowtile
    return runner


def _resolve_rowtile(runner, rows: int, requested: bool | None) -> bool | None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    return runner_mod._owner_rowtile_all(runner, rows, requested)


def _register() -> None:
    assert register_qwen36_gguf_gfx1151_profiles() is True


def _generator(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=profile,
    ).construct_generator(_new_generator)


# ------------------------------------------------------------ profile binding


def test_moe_production_binds_router_and_rowtile_owners(monkeypatch) -> None:
    # Caller-provided diagnostic values must survive the binder: it no longer
    # owns the process environment.
    for key in _POLICY_ENV:
        monkeypatch.setenv(key, "caller")
    _register()

    generator = _generator(ExecutionProfile.PRODUCTION)

    assert generator.target_router_f32w_coop is True
    assert generator.target_router_f32w_persistent_counter is True
    assert generator.target_q8_t16_rowtile_all is True
    assert {key: os.environ.get(key) for key in _POLICY_ENV} == {
        key: "caller" for key in _POLICY_ENV
    }


def test_moe_strict_binds_exact_router_and_rowtile_owners(monkeypatch) -> None:
    for key in _POLICY_ENV:
        monkeypatch.setenv(key, "caller")
    _register()

    generator = _generator(ExecutionProfile.STRICT)

    assert generator.target_router_f32w_coop is False
    assert generator.target_router_f32w_persistent_counter is False
    assert generator.target_q8_t16_rowtile_all is False
    assert {key: os.environ.get(key) for key in _POLICY_ENV} == {
        key: "caller" for key in _POLICY_ENV
    }


def test_moe_owner_values_do_not_leak_between_construction_orders() -> None:
    _register()

    strict_first = _generator(ExecutionProfile.STRICT)
    production_second = _generator(ExecutionProfile.PRODUCTION)
    production_third = _generator(ExecutionProfile.PRODUCTION)
    strict_fourth = _generator(ExecutionProfile.STRICT)

    assert (
        strict_first.target_router_f32w_coop,
        strict_first.target_router_f32w_persistent_counter,
        strict_first.target_q8_t16_rowtile_all,
    ) == (False, False, False)
    assert (
        strict_fourth.target_router_f32w_coop,
        strict_fourth.target_router_f32w_persistent_counter,
        strict_fourth.target_q8_t16_rowtile_all,
    ) == (False, False, False)
    for generator in (production_second, production_third):
        assert (
            generator.target_router_f32w_coop,
            generator.target_router_f32w_persistent_counter,
            generator.target_q8_t16_rowtile_all,
        ) == (True, True, True)


# --------------------------------------------------------------- router reads


def test_router_flags_prefer_bound_owner_over_ambient(monkeypatch) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_gguf_router_f32w_coop_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("ambient read")),
    )
    monkeypatch.setattr(
        runner_mod,
        "_gguf_router_f32w_persistent_counter_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("ambient read")),
    )
    runner = _bare_runner(coop=False, persistent=False)
    flags = runner_mod._layer_ffn_flags(runner)

    assert flags.router_f32w_coop_enabled() is False
    assert flags.router_f32w_persistent_counter_enabled() is False

    runner.router_f32w_coop = True
    runner.router_f32w_persistent_counter = True
    flags = runner_mod._layer_ffn_flags(runner)
    assert flags.router_f32w_coop_enabled() is True
    assert flags.router_f32w_persistent_counter_enabled() is True


def test_router_flags_keep_ambient_reads_for_unbound_runner(monkeypatch) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    runner = _bare_runner(coop=None, persistent=None)
    monkeypatch.setenv(ROUTER_COOP_ENV, "0")
    monkeypatch.setenv(ROUTER_PERSISTENT_ENV, "0")
    flags = runner_mod._layer_ffn_flags(runner)
    assert flags.router_f32w_coop_enabled() is False
    assert flags.router_f32w_persistent_counter_enabled() is False

    # Unbound direct diagnostics keep the historical per-call environment read.
    monkeypatch.setenv(ROUTER_COOP_ENV, "1")
    monkeypatch.setenv(ROUTER_PERSISTENT_ENV, "1")
    assert flags.router_f32w_coop_enabled() is True
    assert flags.router_f32w_persistent_counter_enabled() is True


def test_router_owners_are_interleaved_order_independent(monkeypatch) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    strict = _bare_runner(coop=False, persistent=False)
    unbound = _bare_runner(coop=None, persistent=None)
    monkeypatch.setenv(ROUTER_COOP_ENV, "1")
    monkeypatch.setenv(ROUTER_PERSISTENT_ENV, "1")
    production = _bare_runner(coop=True, persistent=True)

    reads = (
        (strict, False),
        (unbound, True),
        (production, True),
    )
    # Interleaved consumption: every owner keeps its own decision even though
    # the ambient value would retarget the unbound owner.
    for _ in range(2):
        for runner, expected in reads:
            flags = runner_mod._layer_ffn_flags(runner)
            assert flags.router_f32w_coop_enabled() is expected
            assert flags.router_f32w_persistent_counter_enabled() is expected

    monkeypatch.setenv(ROUTER_COOP_ENV, "0")
    monkeypatch.setenv(ROUTER_PERSISTENT_ENV, "0")
    assert runner_mod._layer_ffn_flags(strict).router_f32w_coop_enabled() is False
    assert runner_mod._layer_ffn_flags(production).router_f32w_coop_enabled() is True
    assert runner_mod._layer_ffn_flags(unbound).router_f32w_coop_enabled() is False


# ------------------------------------------------------------ rowtile reads


def test_rowtile_unbound_diagnostic_precedence_is_untouched(monkeypatch) -> None:
    from hipengine.runtime import gguf_linear

    # The documented diagnostic escape hatch: an ambient ``0`` still overrides
    # the packed-decode session when no owner decision exists.
    monkeypatch.setenv(ROWTILE_ALL_ENV, "0")
    with gguf_linear.q8_t16_rowtile_all_session(True):
        assert gguf_linear._resolve_use_q8_t16_all_rowtile() is False
    monkeypatch.delenv(ROWTILE_ALL_ENV, raising=False)
    with gguf_linear.q8_t16_rowtile_all_session(True):
        assert gguf_linear._resolve_use_q8_t16_all_rowtile() is True


def test_rowtile_runner_owner_rows_policy() -> None:
    from hipengine.runtime import gguf_linear

    unbound = _bare_runner(rowtile=None)
    assert unbound._q8_t16_rowtile_all_owner_for_packed_rows(4) is None
    assert unbound._q8_t16_rowtile_all_owner_for_rows(4) is None

    exact = _bare_runner(rowtile=False)
    for rows in (1, 2, 4, 8):
        assert exact._q8_t16_rowtile_all_owner_for_packed_rows(rows) is False
        assert exact._q8_t16_rowtile_all_owner_for_rows(rows) is False

    floor = _bare_runner(rowtile=True)
    # gfx1151 package floor is rows >= 4; the packed-AR route admits it.
    for rows, expected in ((1, False), (2, False), (3, False), (4, True), (8, True)):
        assert floor._q8_t16_rowtile_all_owner_for_packed_rows(rows) is expected
    # The outside-packed default is exact for every width, independent of any
    # other runner's active legacy session.
    for rows in (2, 4, 8):
        assert floor._q8_t16_rowtile_all_owner_for_rows(rows) is False


def test_rowtile_floor_is_scoped_to_the_packed_ar_route(monkeypatch) -> None:
    """The production floor must not retarget verifier, native or prefill rows.

    The old profile published no broad-rowtile input outside the packed-AR
    step. A bound production owner therefore leaves the exact route selected
    everywhere else; a bound strict owner keeps forcing it. An unbound owner
    keeps the diagnostic ambient/session resolution.
    """

    from hipengine.runtime import gguf_linear

    production = _bare_runner(rowtile=True)
    strict = _bare_runner(rowtile=False)
    unbound = _bare_runner(rowtile=None)

    for rows in (2, 4, 8):
        assert production._q8_t16_rowtile_all_owner_for_rows(rows) is False
        assert strict._q8_t16_rowtile_all_owner_for_rows(rows) is False
        assert unbound._q8_t16_rowtile_all_owner_for_rows(rows) is None

    # The packed invocation resolves the floor and carries it explicitly; the
    # session is never consulted for a bound owner.
    for rows, expected in ((2, False), (4, True), (8, True)):
        packed = production._q8_t16_rowtile_all_owner_for_packed_rows(rows)
        assert _resolve_rowtile(production, rows, packed) is expected
        with gguf_linear.q8_t16_rowtile_all_session(True):
            assert _resolve_rowtile(production, rows, packed) is expected
            assert production._q8_t16_rowtile_all_owner_for_rows(rows) is False

    # The legacy session remains the unbound fallback only.
    monkeypatch.setenv(ROWTILE_ALL_ENV, "0")
    with gguf_linear.q8_t16_rowtile_all_session(True):
        assert _resolve_rowtile(unbound, 4, None) is None
        assert gguf_linear._resolve_use_q8_t16_all_rowtile() is False
    monkeypatch.delenv(ROWTILE_ALL_ENV, raising=False)
    with gguf_linear.q8_t16_rowtile_all_session(True):
        assert gguf_linear._resolve_use_q8_t16_all_rowtile() is True


def test_two_production_runners_keep_packed_and_outside_routes_separate() -> None:
    """A legacy session on runner A must not retarget runner B's routes."""

    from hipengine.runtime import gguf_linear

    runner_a = _bare_runner(rowtile=True)
    runner_b = _bare_runner(rowtile=True)
    strict = _bare_runner(rowtile=False)
    unbound = _bare_runner(rowtile=None)

    with gguf_linear.q8_t16_rowtile_all_session(True):
        # A's packed invocation is explicit True at c4.
        assert _resolve_rowtile(runner_a, 4, True) is True
        # B's verifier/prefill/native defaults are exact, not A's floor.
        assert _resolve_rowtile(runner_b, 4, None) is False
        assert runner_b._q8_t16_rowtile_all_owner_for_rows(4) is False
        assert _resolve_rowtile(strict, 4, None) is False
        assert strict._q8_t16_rowtile_all_owner_for_rows(4) is False
        # The unbound runner keeps the ambient/session diagnostic input.
        assert _resolve_rowtile(unbound, 4, None) is None
        assert gguf_linear._resolve_use_q8_t16_all_rowtile() is True
    assert _resolve_rowtile(runner_b, 4, None) is False


def test_use_q8_t16_all_rowtile_explicit_policy(monkeypatch) -> None:
    from hipengine.runtime import gguf_linear

    strict = _bare_runner(rowtile=False)
    production = _bare_runner(rowtile=True)

    # An ambient production input cannot retarget a strict owner on any width.
    monkeypatch.setenv(ROWTILE_ALL_ENV, "1")
    for rows in (1, 2, 4, 8):
        owner = strict._q8_t16_rowtile_all_owner_for_rows(rows)
        assert owner is False
        assert (
            gguf_linear._use_q8_t16_all_rowtile(
                rows=rows, in_features=2048, out_features=4096, rowtile_all=owner
            )
            is False
        )

    # The production floor is a packed-AR decision: the packed invocation
    # carries the floor explicitly; the outside default is exact.
    monkeypatch.setenv(ROWTILE_ALL_ENV, "0")
    for rows, expected in ((1, False), (2, False), (3, False), (4, True), (8, True)):
        packed = production._q8_t16_rowtile_all_owner_for_packed_rows(rows)
        assert (
            gguf_linear._use_q8_t16_all_rowtile(
                rows=rows, in_features=2048, out_features=4096, rowtile_all=packed
            )
            is expected
        )
    for rows in (2, 4, 8):
        owner = production._q8_t16_rowtile_all_owner_for_rows(rows)
        assert owner is False
        # The bound owner decision wins outside the packed-AR session; a
        # diagnostic ambient ``1`` cannot retarget it.
        monkeypatch.setenv(ROWTILE_ALL_ENV, "1")
        assert (
            gguf_linear._use_q8_t16_all_rowtile(
                rows=rows, in_features=2048, out_features=4096, rowtile_all=owner
            )
            is False
        )
        monkeypatch.delenv(ROWTILE_ALL_ENV, raising=False)

    # ``None`` keeps the unbound/direct ambient resolver.
    assert (
        gguf_linear._use_q8_t16_all_rowtile(
            rows=4, in_features=2048, out_features=4096, rowtile_all=None
        )
        is False
    )
    monkeypatch.setenv(ROWTILE_ALL_ENV, "1")
    assert (
        gguf_linear._use_q8_t16_all_rowtile(
            rows=4, in_features=2048, out_features=4096, rowtile_all=None
        )
        is True
    )

    # Unrelated input widths stay exact for every owner.
    assert (
        gguf_linear._use_q8_t16_all_rowtile(
            rows=4, in_features=5120, out_features=1024, rowtile_all=True
        )
        is False
    )


def test_use_q8_t16_pair_rowtile_explicit_policy(monkeypatch) -> None:
    from hipengine.runtime import gguf_linear

    gguf_linear.set_q8_t16_pair_rowtile_min_rows(None)
    shapes = dict(rows=4, in_features=2048, out_features_a=8192, out_features_b=4096)

    monkeypatch.setenv(ROWTILE_ALL_ENV, "1")
    assert gguf_linear._use_q8_t16_pair_rowtile(**shapes, rowtile_all=False) is False
    assert gguf_linear._use_q8_t16_pair_rowtile(**shapes, rowtile_all=None) is True

    monkeypatch.setenv(ROWTILE_ALL_ENV, "0")
    assert gguf_linear._use_q8_t16_pair_rowtile(**shapes, rowtile_all=True) is True
    assert gguf_linear._use_q8_t16_pair_rowtile(**shapes, rowtile_all=None) is False

    # The independent pair package floor is not the broad-policy input.
    gguf_linear.set_q8_t16_pair_rowtile_min_rows(8)
    assert (
        gguf_linear._use_q8_t16_pair_rowtile(
            **{**shapes, "rows": 8}, rowtile_all=False
        )
        is True
    )
    gguf_linear.set_q8_t16_pair_rowtile_min_rows(None)


def test_attention_norm_seam_carries_the_invocation_input(monkeypatch) -> None:
    """The packed explicit value reaches the actual selector, not the AST."""

    from hipengine.runtime import gguf_linear
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    captured = {}

    def fake_norm(**kwargs):
        captured["selectors"] = kwargs["selectors"]
        return None

    monkeypatch.setattr(
        runner_mod._layer_attn, "run_attention_norm_rows", fake_norm
    )
    runner = _bare_runner(rowtile=True)
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(hidden_size=2048)
    )

    with gguf_linear.q8_t16_rowtile_all_session(True):
        runner_mod.Qwen35GGUFFullStackRunner._run_attention_norm_rows(
            runner,
            hidden_ptr=1,
            hidden_f32_ptr=None,
            weight_ptr=2,
            out_ptr=3,
            rows=4,
            eps=1e-6,
            stream=0,
            runtime=object(),
            q8_t16_rowtile_all=True,
        )
        assert captured["selectors"].q8_t16_rowtile_all is True

        # No invocation input means the outside-packed default, even while a
        # legacy session on another runner is active.
        runner_mod.Qwen35GGUFFullStackRunner._run_attention_norm_rows(
            runner,
            hidden_ptr=1,
            hidden_f32_ptr=None,
            weight_ptr=2,
            out_ptr=3,
            rows=4,
            eps=1e-6,
            stream=0,
            runtime=object(),
        )
        assert captured["selectors"].q8_t16_rowtile_all is False


def test_attention_selectors_carry_owner_rowtile_policy() -> None:
    from hipengine.runtime import gguf_linear
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    strict = _bare_runner(rowtile=False)
    production = _bare_runner(rowtile=True)
    unbound = _bare_runner(rowtile=None)

    assert runner_mod._layer_attention_selectors(strict, rows=4).q8_t16_rowtile_all is False
    # The outside-packed default is exact for a bound production owner.
    assert (
        runner_mod._layer_attention_selectors(production, rows=4).q8_t16_rowtile_all
        is False
    )
    assert (
        runner_mod._layer_attention_selectors(production, rows=2).q8_t16_rowtile_all
        is False
    )
    assert runner_mod._layer_attention_selectors(unbound, rows=4).q8_t16_rowtile_all is None
    # The packed invocation passes the floor explicitly; a legacy session on
    # another runner cannot change it.
    with gguf_linear.q8_t16_rowtile_all_session(True):
        assert (
            runner_mod._layer_attention_selectors(
                production, rows=4, q8_t16_rowtile_all=True
            ).q8_t16_rowtile_all
            is True
        )
        assert (
            runner_mod._layer_attention_selectors(production, rows=4).q8_t16_rowtile_all
            is False
        )


def test_full_attention_selectors_carry_owner_rowtile_policy() -> None:
    from types import SimpleNamespace

    from hipengine.runtime import gguf_linear
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    production = _bare_runner(rowtile=True)
    strict = _bare_runner(rowtile=False)
    scratch = SimpleNamespace()

    # The outside-packed default is exact for prefill and the verifier.
    assert (
        runner_mod._decode_batch_attention_selectors(
            production, scratch, None, rows=4
        ).q8_t16_rowtile_all
        is False
    )
    assert (
        runner_mod._bulk_full_attention_selectors(
            production, scratch, object(), rows=4
        ).q8_t16_rowtile_all
        is False
    )
    assert (
        runner_mod._bulk_full_attention_selectors(
            strict, scratch, object(), rows=4
        ).q8_t16_rowtile_all
        is False
    )
    # The packed invocation passes the floor explicitly; a legacy session on
    # another runner cannot change it.
    with gguf_linear.q8_t16_rowtile_all_session(True):
        assert (
            runner_mod._decode_batch_attention_selectors(
                production,
                scratch,
                None,
                rows=4,
                q8_t16_rowtile_all=True,
            ).q8_t16_rowtile_all
            is True
        )
        assert (
            runner_mod._decode_batch_attention_selectors(
                production, scratch, None, rows=4
            ).q8_t16_rowtile_all
            is False
        )


# --------------------------------------------------------- generator transport


def test_generator_owner_session_kwargs() -> None:
    generator = _new_generator()
    for name in _OWNER_FIELDS:
        setattr(generator, name, None)
    assert generator._target_router_policy_session_kwargs() == {}
    assert generator._target_q8_t16_rowtile_all_session_kwargs() == {}

    generator.target_router_f32w_coop = False
    generator.target_router_f32w_persistent_counter = True
    generator.target_q8_t16_rowtile_all = True
    assert generator._target_router_policy_session_kwargs() == {
        "target_router_f32w_coop": False,
        "target_router_f32w_persistent_counter": True,
    }
    assert generator._target_q8_t16_rowtile_all_session_kwargs() == {
        "target_q8_t16_rowtile_all": True
    }


def test_generator_transports_owners_to_shared_runner(monkeypatch) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(gguf, "Qwen35GGUFFullStackRunner", FakeRunner)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = QWEN36_GGUF_BACKEND
    generator.native_batch_decode = False
    generator._serving_capacity_bound = None
    generator.target_router_f32w_coop = True
    generator.target_router_f32w_persistent_counter = True
    generator.target_q8_t16_rowtile_all = True

    generator._get_shared_runner()

    assert captured["target_router_f32w_coop"] is True
    assert captured["target_router_f32w_persistent_counter"] is True
    assert captured["target_q8_t16_rowtile_all"] is True


def test_generator_transports_owners_to_shared_session(monkeypatch) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeSession:
        max_sequence_length = None

        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = QWEN36_GGUF_BACKEND
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    generator._auto_context_estimate = None
    generator.target_router_f32w_coop = False
    generator.target_router_f32w_persistent_counter = False
    generator.target_q8_t16_rowtile_all = False

    generator._construct_shared_session(
        SimpleNamespace(runtime=None),
        max_sequence_length=None,
        max_batch_size=1,
        defer_kv_allocation=False,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert captured["target_router_f32w_coop"] is False
    assert captured["target_router_f32w_persistent_counter"] is False
    assert captured["target_q8_t16_rowtile_all"] is False


def test_generator_transports_owners_to_direct_session(monkeypatch) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeSession:
        runtime = None

        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    monkeypatch.setattr(gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = QWEN36_GGUF_BACKEND
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    generator.target_router_f32w_coop = True
    generator.target_router_f32w_persistent_counter = False
    generator.target_q8_t16_rowtile_all = True
    monkeypatch.setattr(generator, "_configure_session", lambda session: None)

    with generator._resident_session_scope(shared_runner=None, pool_name="test"):
        pass

    assert captured["target_router_f32w_coop"] is True
    assert captured["target_router_f32w_persistent_counter"] is False
    assert captured["target_q8_t16_rowtile_all"] is True


# ------------------------------------------------------------ NextN and draft


@pytest.mark.parametrize("requested", [True, False, None])
def test_nextn_transports_owners(monkeypatch, requested: bool | None) -> None:
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    runner_kwargs: dict[str, object] = {}
    session_kwargs: dict[str, object] = {}

    class StopNextN(Exception):
        pass

    class FakeWeights:
        backend = QWEN36_GGUF_BACKEND

        def as_full_stack_weights(self):
            return object()

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            runner_kwargs.update(kwargs)
            self.hidden_size = 4
            self.vocab_size = 8
            self.runtime = kwargs.get("runtime")

    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            session_kwargs.update(kwargs)
            raise StopNextN

    monkeypatch.setattr(nextn, "register_gguf_q3_k_gemv_kernels", lambda: None)
    monkeypatch.setattr(nextn, "Qwen35GGUFFullStackRunner", FakeRunner)
    monkeypatch.setattr(nextn, "Qwen35GGUFResidentSession", FakeSession)

    with pytest.raises(StopNextN):
        nextn.Qwen35GGUFNextNDraftProvider.from_model(
            "unused",
            max_positions=8,
            max_requests=1,
            runtime=object(),
            resident_weights=FakeWeights(),
            target_router_f32w_coop=requested,
            target_router_f32w_persistent_counter=requested,
            target_q8_t16_rowtile_all=requested,
        )

    for name in _OWNER_FIELDS:
        assert runner_kwargs[name] is requested
        assert session_kwargs[name] is requested


def _nextn_generator(**owner):
    generator = _new_generator()
    generator.model_path = "unused"
    for name in _OWNER_FIELDS:
        setattr(generator, name, owner.get(name))
    generator._mtp_hot_vocab_path = lambda: None
    generator._shared_dense_mtp_nextn_weights = lambda *args, **kwargs: object()
    return generator


def _fake_draft_provider(monkeypatch, calls: list[dict]):
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    class FakeProvider:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        @classmethod
        def from_model(cls, model, **kwargs):
            calls.append({"model": model, **kwargs})
            return cls()

    monkeypatch.setattr(nextn, "Qwen35GGUFNextNDraftProvider", FakeProvider)
    return FakeProvider


@pytest.mark.parametrize("requested", [True, False, None])
def test_draft_provider_transports_owners(monkeypatch, requested: bool | None) -> None:
    calls: list[dict] = []
    _fake_draft_provider(monkeypatch, calls)
    generator = _nextn_generator(
        target_router_f32w_coop=requested,
        target_router_f32w_persistent_counter=requested,
        target_q8_t16_rowtile_all=requested,
    )
    target = SimpleNamespace(
        runtime=object(), require_cached_build=False, runner=object()
    )

    provider, key, reused = generator._acquire_dense_mtp_draft_provider(
        target, max_positions=64, pool_enabled=False
    )

    assert provider is not None and reused is False and key is None
    for name in _OWNER_FIELDS:
        assert calls[0][name] is requested


def test_draft_provider_pool_reuse_keeps_owner(monkeypatch) -> None:
    calls: list[dict] = []
    _fake_draft_provider(monkeypatch, calls)
    generator = _nextn_generator(
        target_router_f32w_coop=True,
        target_router_f32w_persistent_counter=True,
        target_q8_t16_rowtile_all=True,
    )
    target = SimpleNamespace(
        runtime=object(), require_cached_build=False, runner=object()
    )

    first, key, reused = generator._acquire_dense_mtp_draft_provider(
        target, max_positions=64, pool_enabled=True
    )
    assert reused is False and key is not None
    generator._release_mtp_draft_runner(key, first)
    second, second_key, second_reused = generator._acquire_dense_mtp_draft_provider(
        target, max_positions=64, pool_enabled=True
    )

    assert second is first and second_reused is True and second_key == key
    assert len(calls) == 1
    assert calls[0]["target_router_f32w_coop"] is True
    assert calls[0]["target_router_f32w_persistent_counter"] is True
    assert calls[0]["target_q8_t16_rowtile_all"] is True


# ------------------------------------------------------- live owner validation


def test_configure_session_rejects_live_router_and_rowtile_change() -> None:
    generator = _new_generator()
    generator.backend = QWEN36_GGUF_BACKEND
    generator.target_router_f32w_coop = True
    generator.target_router_f32w_persistent_counter = True
    generator.target_q8_t16_rowtile_all = True
    session = SimpleNamespace(
        _native_sampler_workspace=None,
        runner=SimpleNamespace(
            router_f32w_coop=False,
            router_f32w_persistent_counter=False,
            q8_t16_rowtile_all=False,
        ),
    )

    with pytest.raises(RuntimeError, match="router policy"):
        generator._configure_session(session)


def test_configure_session_rejects_live_rowtile_change() -> None:
    generator = _new_generator()
    generator.backend = QWEN36_GGUF_BACKEND
    generator.target_router_f32w_coop = None
    generator.target_q8_t16_rowtile_all = True
    session = SimpleNamespace(
        _native_sampler_workspace=None,
        runner=SimpleNamespace(
            router_f32w_coop=None,
            q8_t16_rowtile_all=False,
        ),
    )

    with pytest.raises(RuntimeError, match="rowtile policy"):
        generator._configure_session(session)


class _ReachedScratchBoundary(Exception):
    pass


def _shared_session_boundary(monkeypatch, shared):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_backend", lambda backend: QWEN36_GGUF_BACKEND)
    monkeypatch.setattr(
        runner_mod,
        "backend_package_capability",
        lambda backend, name, default=None: default,
    )
    monkeypatch.setattr(
        runner_mod,
        "_resolve_gguf_private_c1_small_weight_arena",
        lambda **kwargs: (False, "test"),
    )
    monkeypatch.setattr(
        runner_mod,
        "_resolve_gguf_private_c1_decode_scratch_arena",
        lambda **kwargs: (False, "test"),
    )

    def stop_before_scratch(value):
        raise _ReachedScratchBoundary

    monkeypatch.setattr(
        runner_mod, "_resolve_target_verifier_production_q4_rowtile", stop_before_scratch
    )
    return runner_mod.Qwen35GGUFResidentSession(
        "unused",
        shared_runner=shared,
        runtime=object(),
        target_router_f32w_coop=True,
        target_router_f32w_persistent_counter=True,
        target_q8_t16_rowtile_all=True,
    )


@pytest.mark.parametrize(
    ("coop", "persistent", "rowtile"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_shared_session_rejects_mismatched_owner_policy(
    monkeypatch, coop: bool, persistent: bool, rowtile: bool
) -> None:
    shared = SimpleNamespace(
        backend=QWEN36_GGUF_BACKEND,
        runtime=object(),
        fp16_recurrent_state=False,
        verifier_prefill_gdn_capture=False,
        router_f32w_coop=coop,
        router_f32w_persistent_counter=persistent,
        q8_t16_rowtile_all=rowtile,
    )

    with pytest.raises(RuntimeError, match="owner request"):
        _shared_session_boundary(monkeypatch, shared)


def test_shared_session_accepts_matching_owner_policy(monkeypatch) -> None:
    shared = SimpleNamespace(
        backend=QWEN36_GGUF_BACKEND,
        runtime=object(),
        fp16_recurrent_state=False,
        verifier_prefill_gdn_capture=False,
        router_f32w_coop=True,
        router_f32w_persistent_counter=True,
        q8_t16_rowtile_all=True,
    )

    with pytest.raises(_ReachedScratchBoundary):
        _shared_session_boundary(monkeypatch, shared)


def test_shared_session_rejects_unbound_shared_runner_for_bound_owner(
    monkeypatch,
) -> None:
    shared = SimpleNamespace(
        backend=QWEN36_GGUF_BACKEND,
        runtime=object(),
        fp16_recurrent_state=False,
        verifier_prefill_gdn_capture=False,
        router_f32w_coop=None,
        router_f32w_persistent_counter=None,
        q8_t16_rowtile_all=None,
    )

    with pytest.raises(RuntimeError, match="owner request"):
        _shared_session_boundary(monkeypatch, shared)


# --------------------------------------------------------------- graph identity


def test_graph_identity_separates_router_and_rowtile_policy() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    common = dict(
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_prefill_gdn=False,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )
    strict = cycle_mod._native_target_configuration_key(
        **common,
        router_f32w_coop=False,
        router_f32w_persistent_counter=False,
        q8_t16_rowtile_all=False,
    )
    production = cycle_mod._native_target_configuration_key(
        **common,
        router_f32w_coop=True,
        router_f32w_persistent_counter=True,
        q8_t16_rowtile_all=True,
    )
    unbound = cycle_mod._native_target_configuration_key(**common)

    assert strict != production != unbound
    assert strict != unbound


def test_graph_identity_reads_the_owner_resolved_policy() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    bound = SimpleNamespace(
        runner=_bare_runner(coop=False, persistent=True, rowtile=True)
    )
    unbound = SimpleNamespace(runner=_bare_runner())

    assert cycle_mod._native_target_route_identity(bound) == {
        "router_f32w_coop": False,
        "router_f32w_persistent_counter": True,
        "q8_t16_rowtile_all": True,
    }
    assert cycle_mod._native_target_route_identity(unbound) == {
        "router_f32w_coop": None,
        "router_f32w_persistent_counter": None,
        "q8_t16_rowtile_all": None,
    }


def test_moe_rows_shared_ffn_launches_carry_the_owner_policy() -> None:
    """The c>1 shared-expert gate/up launches are real broad-rowtile seams."""

    import ast
    from pathlib import Path

    source = Path(
        "hipengine/runtime/qwen35_gguf_runner.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_try_run_post_attention_moe_rows_compact_wmma"
        ):
            owner = node
            break
    assert owner is not None

    calls = []
    for node in ast.walk(owner):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Name)
            and func.id
            in {
                "launch_gguf_linear",
                "launch_gguf_linear_pair",
                "launch_gguf_linear_triple",
            }
        ):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if func.id == "launch_gguf_linear_triple":
            calls.append(("rowtile_all", keywords.get("rowtile_all")))
        else:
            calls.append(("q8_t16_rowtile_all", keywords.get("q8_t16_rowtile_all")))

    # gate/up pair + two singleton fallbacks + shared down; every launch in
    # the seam carries the invocation input. Only the 2048-wide gate/up
    # launches can select the broad rowtile, but the down launch still receives
    # the route-resolved value so the policy is not re-read from ambient.
    assert len(calls) == 4
    for name, value in calls:
        assert value is not None, name
        assert isinstance(value, ast.Name)
        assert value.id == "q8_t16_rowtile_all"

    # The seam accepts the explicit input and only resolves the outside default
    # when the caller did not supply one.
    args = {arg.arg for arg in owner.args.args}
    args.update(arg.arg for arg in owner.args.kwonlyargs)
    assert "q8_t16_rowtile_all" in args
    resolves_default = False
    for node in ast.walk(owner):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "q8_t16_rowtile_all"
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "_q8_t16_rowtile_all_owner_for_rows"
            ):
                resolves_default = True
    assert resolves_default


def test_indexed_linear_attention_seam_carries_invocation_input(monkeypatch) -> None:
    """The packed AR width-4 path binds the real indexed method and forwards it.

    ``_run_linear_attention_decode_slot_rows_exact`` passes an explicit
    ``q8_t16_rowtile_all`` into ``_run_linear_attention_attn_rows_indexed_exact``.
    That seam must accept the keyword on its real signature (not only through a
    permissive fake) and carry the value to its first leaf call unchanged.
    """

    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    captured: dict[str, object] = {}

    class _StopAtLeaf(Exception):
        pass

    def fake_norm(**kwargs):
        captured["rowtile_all"] = kwargs["q8_t16_rowtile_all"]
        raise _StopAtLeaf

    def leaf(name):
        return SimpleNamespace(
            allocation=lambda: SimpleNamespace(
                tensor=SimpleNamespace(ptr=hash(name) & 0xFFFF)
            )
        )

    runner = _bare_runner(rowtile=True)
    runner._run_attention_norm_rows = fake_norm
    runner.runtime = object()
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        layer=lambda layer_id: SimpleNamespace(weight=leaf),
    )
    state = SimpleNamespace(ptr=11)
    decode_scratch = SimpleNamespace(
        layer_conv_states=[state],
        layer_recurrent_states=[state],
    )
    batch_plan = SimpleNamespace(available=True)

    def call(**kwargs):
        return runner_mod.Qwen35GGUFFullStackRunner._run_linear_attention_attn_rows_indexed_exact(
            runner,
            0,
            1,
            2,
            SimpleNamespace(norm=SimpleNamespace(ptr=3)),
            rows=4,
            decode_scratch=decode_scratch,
            batch_plan=batch_plan,
            gdn_cu_seqlens_ptr=5,
            state_indices_ptr=6,
            **kwargs,
        )

    with pytest.raises(_StopAtLeaf):
        call(q8_t16_rowtile_all=True)
    assert captured["rowtile_all"] is True

    # Omitted input resolves the outside-packed owner default (bound -> False),
    # not the packed invocation value.
    with pytest.raises(_StopAtLeaf):
        call()
    assert captured["rowtile_all"] is False


def test_rowtile_seams_accept_the_keyword_on_real_signatures() -> None:
    """Every new seam call must bind against the real callable signature."""

    import inspect

    from hipengine.runtime import gguf_linear
    from hipengine.runtime import qwen35_gguf_layer_ffn as layer_ffn
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    Runner = runner_mod.Qwen35GGUFFullStackRunner
    seams = {
        "launch_gguf_linear": gguf_linear.launch_gguf_linear,
        "launch_gguf_linear_pair": gguf_linear.launch_gguf_linear_pair,
        "run_post_attention_ffn_rows": layer_ffn.run_post_attention_ffn_rows,
        "run_post_attention_moe_rows": layer_ffn.run_post_attention_moe_rows,
        "_decode_batch_attention_selectors": runner_mod._decode_batch_attention_selectors,
        "_try_run_post_attention_moe_rows_compact_wmma": (
            runner_mod._try_run_post_attention_moe_rows_compact_wmma
        ),
        "_run_attention_norm_rows": Runner._run_attention_norm_rows,
        "_run_linear_attention_alpha_beta_rows": (
            Runner._run_linear_attention_alpha_beta_rows
        ),
        "_run_linear_attention_attn_only": Runner._run_linear_attention_attn_only,
        "_run_linear_attention_attn_rows_indexed_exact": (
            Runner._run_linear_attention_attn_rows_indexed_exact
        ),
        "_run_linear_attention_decode_slot_rows_exact": (
            Runner._run_linear_attention_decode_slot_rows_exact
        ),
        "_run_full_attention_decode_batch_layer_rows": (
            Runner._run_full_attention_decode_batch_layer_rows
        ),
        "_run_full_attention_output_rows": Runner._run_full_attention_output_rows,
        "_run_post_attention_ffn_rows": Runner._run_post_attention_ffn_rows,
    }
    for name, seam in seams.items():
        signature = inspect.signature(seam)
        assert "q8_t16_rowtile_all" in signature.parameters, name
        # bind_partial rejects an unexpected keyword, unlike a fake ``**kwargs``.
        signature.bind_partial(q8_t16_rowtile_all=True)

    # The triple launch names the same policy ``rowtile_all``.
    triple = inspect.signature(gguf_linear.launch_gguf_linear_triple)
    assert "rowtile_all" in triple.parameters
    triple.bind_partial(rowtile_all=True)
