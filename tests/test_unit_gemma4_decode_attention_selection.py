"""The scoped long-global decode-attention preference, on CPU.

The gfx1100 production profile may route the one-token global decode step to the
staged singleton attention candidate. The strict native/exact kernel holds one
FP32 logit per live key in shared memory and, past 15,616 keys at ``head_dim``
512, keeps the same arithmetic on owned global scratch (its ``global_class``
variant); the staged candidate reproduces that arithmetic in bounded shared
memory. This is a route preference, not a changed result.

The scope is resolved through the kernel registry, so the registration is
executable: the layer runs whatever callable the ``decode_attention`` key holds,
including the registered strict fallback. Four separate contracts are pinned:

* **The threshold.** The switch is the launcher's own resident-LDS bound, not a
  literal: 15,616 keys still fit and keep the native kernel, 15,617 do not and
  take the candidate.
* **The scope.** A multi-row block, another head geometry, a short context, a
  sliding layer at head_dim 512, or a caller-dropped prefix all keep the native
  kernel, so the preference is scoped to exactly the shape it was verified on.
* **The registry.** A replaced decode-attention callable is what the layer runs;
  the strict fallback is the registered one too.
* **The profile and the phase.** ``production`` selects the candidate and
  ``strict`` keeps the native kernel, and the runner forwards the preference
  only for a genuine decode step -- a one-row prefill tail stays native.

Everything here is host-only: no build, no device, no kernel launch.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.execution_profiles import (
    ExecutionProfile,
    resolve_runtime_profile,
)
from hipengine.generation.gemma4_profiles import (
    DECODE_ATTENTION_LAYER,
    DECODE_ATTENTION_PLAIN,
    DECODE_ATTENTION_QUANT,
    DECODE_ATTENTION_SCOPE,
    DECODE_ATTENTION_STAGED,
    GEMMA4_GGUF_BACKEND,
    GEMMA4_GGUF_MODEL,
    GEMMA4_GGUF_QUANT,
    MOE_PREFILL_ENV,
    gemma4_gguf_profiles_registered,
    register_gemma4_gguf_profiles,
)
from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer_module
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
    gemma4_attention_resident_overflows,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_staged import (
    gemma4_attention_staged_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    _select_exact_decode_attention,
    _select_prefill_route,
)
from hipengine.kernels.registry import KernelKey, register

# Importing the built-in generators registers the gfx1100 Gemma 4 kernels the
# profile names. It runs at collection time so the conftest kernel baseline
# (snapshotted after collection) includes them and the per-test registry restore
# cannot wipe them between the parametrized profile cases.
from hipengine.generation import register_builtin_generators

register_builtin_generators()
register_gemma4_gguf_profiles()

# Gemma 4's global geometry: 16 query heads over 2 KV heads, head_dim 512.
GLOBAL = dict(
    tokens=1,
    keys=20000,
    num_heads=16,
    num_kv_heads=2,
    head_dim=512,
    sliding_window=None,
    key_begin=0,
)

# The last resident key count at head_dim 512 and the first one that overflows.
LAST_RESIDENT_KEYS = 15616
FIRST_OVERFLOW_KEYS = 15617


def select(**overrides):
    base = dict(decode_attention_variants=(DECODE_ATTENTION_STAGED,), **GLOBAL)
    base.update(overrides)
    return _select_exact_decode_attention(**base)


def test_the_switch_is_the_resident_lds_bound() -> None:
    """15,616 keys fit; 15,617 do not. The bound is the launcher's own formula."""

    assert gemma4_attention_resident_overflows(
        head_dim=512, keys=LAST_RESIDENT_KEYS
    ) is False
    assert gemma4_attention_resident_overflows(
        head_dim=512, keys=FIRST_OVERFLOW_KEYS
    ) is True


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (LAST_RESIDENT_KEYS - 1, "exact"),
        (LAST_RESIDENT_KEYS, "exact"),
        (FIRST_OVERFLOW_KEYS, DECODE_ATTENTION_STAGED),
    ],
)
def test_the_threshold_switches_at_the_resident_bound(keys, expected) -> None:
    launcher, route = select(keys=keys)

    assert route == expected
    if expected == DECODE_ATTENTION_STAGED:
        assert launcher is gemma4_attention_staged_bf16
    else:
        assert launcher is gemma4_attention_prefill_bf16


@pytest.mark.parametrize("tokens", [2, 7, 512])
def test_a_multi_row_block_keeps_the_native_kernel(tokens) -> None:
    """A prefill block, or a speculative verify pass, is never a decode step."""

    launcher, route = select(tokens=tokens)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


@pytest.mark.parametrize(
    "geometry",
    [
        # The sliding geometry: head_dim 256, GQA ratio 2.
        dict(num_heads=16, num_kv_heads=8, head_dim=256),
        # The global head width with the wrong ratio.
        dict(num_heads=16, num_kv_heads=4, head_dim=512),
        # A ratio of 8 at a head width the candidate's score tree does not cover.
        dict(num_heads=8, num_kv_heads=1, head_dim=128),
        # A shape unrelated to either geometry.
        dict(num_heads=8, num_kv_heads=1, head_dim=64),
    ],
)
def test_another_geometry_keeps_the_native_kernel(geometry) -> None:
    launcher, route = select(**geometry)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


@pytest.mark.parametrize("keys", [1, 512, LAST_RESIDENT_KEYS])
def test_a_short_global_decode_keeps_the_native_kernel(keys) -> None:
    """Below the bound the native kernel is not the owned-global-logits route."""

    launcher, route = select(keys=keys)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


def test_a_one_row_prefill_tail_keeps_the_native_kernel() -> None:
    """The final single-token block of a dense prefill is not a long decode.

    It attends over the one row it holds (``keys == tokens == 1``), so it is far
    below the bound and keeps the native kernel.
    """

    launcher, route = select(tokens=1, keys=1)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


def test_a_head_dim_512_sliding_layer_keeps_the_native_kernel() -> None:
    """A sliding layer's decode is windowed, whatever its head geometry.

    The candidate was verified on the windowless global call. A layer carrying a
    sliding window of 20,003 at the same 512/8 geometry presents a bounded,
    per-row-trimmed walk, which is not that shape.
    """

    launcher, route = select(sliding_window=20003)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


def test_an_explicitly_trimmed_global_call_keeps_the_native_kernel() -> None:
    """A caller that dropped a prefix skips keys the candidate was not checked on."""

    launcher, route = select(key_begin=1)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


def test_no_preference_keeps_the_native_kernel() -> None:
    for variants in (None, (), (DECODE_ATTENTION_PLAIN,)):
        launcher, route = _select_exact_decode_attention(
            decode_attention_variants=variants, **GLOBAL
        )
        assert route == "exact"
        assert launcher is gemma4_attention_prefill_bf16


def test_the_staged_choice_reports_the_inner_variant() -> None:
    """The reporter shows which inner variant ran, not just the outer route."""

    launcher, route = select()

    assert route == DECODE_ATTENTION_STAGED
    assert launcher is gemma4_attention_staged_bf16


# --- the registry is executable ---------------------------------------------


def _decode_key(variant: str) -> KernelKey:
    return KernelKey(GEMMA4_GGUF_BACKEND, DECODE_ATTENTION_LAYER, DECODE_ATTENTION_QUANT, variant)


def test_a_replaced_staged_callable_is_the_one_the_selector_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registration is executable, not decorative."""

    def replacement(*args, **kwargs):
        return 0

    register(_decode_key(DECODE_ATTENTION_STAGED), replacement, replace=True)
    launcher, route = select()

    assert route == DECODE_ATTENTION_STAGED
    assert launcher is replacement


@pytest.mark.parametrize("variants", [(DECODE_ATTENTION_PLAIN,), (DECODE_ATTENTION_STAGED,)])
def test_the_strict_fallback_is_the_registered_plain_callable(
    monkeypatch: pytest.MonkeyPatch, variants,
) -> None:
    """A call outside the scope falls back to the *registered* plain entry."""

    def replacement(*args, **kwargs):
        return 0

    register(_decode_key(DECODE_ATTENTION_PLAIN), replacement, replace=True)
    launcher, route = select(tokens=2, decode_attention_variants=variants)

    assert route == "exact"
    assert launcher is replacement


# --- the layer runs the registered callable ---------------------------------


class _ReachedAttention(Exception):
    """Sentinel: the attention launcher ran, so the layer can stop there."""


_PRE_ATTENTION_LAUNCHERS = (
    "gemma4_rmsnorm_f32w_bf16",
    "gemma4_project",
    "gemma4_qkv_split_bf16",
    "gemma4_rmsnorm_weightless_bf16",
    "gemma4_head_rmsnorm_f32w_bf16",
    "gemma4_partial_rotary_bf16",
)


def _install_layer_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every launch before the attention block, so no device work runs."""

    monkeypatch.setattr(layer_module, "_moe_stream", lambda: 0)
    monkeypatch.setattr(layer_module, "_append_kv", lambda *a, **k: None)
    for name in _PRE_ATTENTION_LAUNCHERS:
        monkeypatch.setattr(layer_module, name, lambda *a, **k: None)


def _layer_stub() -> SimpleNamespace:
    return SimpleNamespace(
        input_layernorm=0,
        qkv_proj=None,
        q_proj=0,
        k_proj=0,
        v_proj=0,
        q_norm=0,
        k_norm=0,
    )


def _scratch_stub(*, tokens: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=tokens,
        hidden_size=16,
        geometry=SimpleNamespace(
            num_heads=16,
            num_kv_heads=2,
            head_dim=512,
            scale=1.0,
            k_eq_v=False,
            sliding_window=None,
        ),
        buffer=lambda name: SimpleNamespace(ptr=0),
        attention=SimpleNamespace(),
    )


def _kv_stub(write_offset: int) -> SimpleNamespace:
    return SimpleNamespace(key_cache=0, value_cache=0, capacity=20000, write_offset=write_offset)


def test_the_layer_invokes_the_registered_staged_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replaced ``gemma4_staged`` entry is what the layer's exact site calls."""

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    register(_decode_key(DECODE_ATTENTION_STAGED), replacement, replace=True)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0,
            0,
            0,
            0,
            _layer_stub(),
            scratch=_scratch_stub(tokens=1),
            kv=_kv_stub(LAST_RESIDENT_KEYS),
            rows=1,
            decode_attention_variants=(DECODE_ATTENTION_STAGED,),
        )
    assert seen, "the registered staged callable was not invoked"
    assert seen[0]["keys"] == FIRST_OVERFLOW_KEYS


def test_the_layer_invokes_the_registered_plain_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-row block keeps the native arithmetic -- the *registered* one."""

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    register(_decode_key(DECODE_ATTENTION_PLAIN), replacement, replace=True)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0,
            0,
            0,
            0,
            _layer_stub(),
            scratch=_scratch_stub(tokens=2),
            kv=_kv_stub(LAST_RESIDENT_KEYS),
            rows=2,
            decode_attention_variants=(DECODE_ATTENTION_STAGED,),
        )
    assert seen, "the registered plain fallback was not invoked"
    assert seen[0]["tokens"] == 2


# --- the profile contract ---------------------------------------------------


def _resolved(profile):
    register_gemma4_gguf_profiles()
    assert gemma4_gguf_profiles_registered()
    return resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=profile,
    )


def _decode_rows(resolved):
    return [
        selection
        for selection in resolved.manifest["selections"]
        if selection["layer"] == DECODE_ATTENTION_LAYER
    ]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (ExecutionProfile.PRODUCTION, DECODE_ATTENTION_STAGED),
        (ExecutionProfile.STRICT, DECODE_ATTENTION_PLAIN),
        # batch_invariant is deliberately unregistered: it falls back to strict.
        (ExecutionProfile.BATCH_INVARIANT, DECODE_ATTENTION_PLAIN),
    ],
)
def test_the_profile_selects_the_scoped_decode_route(profile, expected) -> None:
    resolved = _resolved(profile)
    rows = _decode_rows(resolved)

    assert len(rows) == 1
    assert rows[0]["scope"] == DECODE_ATTENTION_SCOPE
    assert rows[0]["selected_variant"] == expected
    assert rows[0]["strict_fallback_variant"] == DECODE_ATTENTION_PLAIN
    assert rows[0]["registry_quant"] == DECODE_ATTENTION_QUANT


def test_the_binder_binds_the_selected_decode_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binder carries the manifest's choice onto the generator verbatim."""

    # The binder pins ``HIPENGINE_GEMMA4_MOE_PREFILL`` as a process env var, so
    # pin it through monkeypatch first: teardown then restores whatever the run
    # started with instead of leaking the strict route into later tests.
    monkeypatch.setenv(MOE_PREFILL_ENV, "auto")
    for profile, expected in (
        (ExecutionProfile.PRODUCTION, (DECODE_ATTENTION_STAGED,)),
        (ExecutionProfile.STRICT, (DECODE_ATTENTION_PLAIN,)),
        (ExecutionProfile.BATCH_INVARIANT, (DECODE_ATTENTION_PLAIN,)),
    ):
        resolved = _resolved(profile)
        generator = SimpleNamespace()
        resolved.binder(generator, resolved)
        assert generator.decode_attention_variants == expected


def test_the_decode_variants_are_registered_under_their_own_layer() -> None:
    from hipengine.kernels.registry import resolve

    register_gemma4_gguf_profiles()

    assert resolve(
        backend=GEMMA4_GGUF_BACKEND,
        layer=DECODE_ATTENTION_LAYER,
        quant=DECODE_ATTENTION_QUANT,
        variant=DECODE_ATTENTION_PLAIN,
    ) is gemma4_attention_prefill_bf16
    assert resolve(
        backend=GEMMA4_GGUF_BACKEND,
        layer=DECODE_ATTENTION_LAYER,
        quant=DECODE_ATTENTION_QUANT,
        variant=DECODE_ATTENTION_STAGED,
    ) is gemma4_attention_staged_bf16


# --- the surrounding routing is unchanged -----------------------------------


def test_the_existing_prefill_routing_is_unaffected() -> None:
    """The decode preference does not touch the tiled/aotriton/exact split."""

    global_block = dict(
        rows=128, keys=128, num_heads=16, num_kv_heads=2, head_dim=512,
        sliding_window=None, mask_is_causal=True,
    )
    assert _select_prefill_route(**global_block) == "tiled"
    assert (
        _select_prefill_route(**{**global_block, "head_dim": 256, "num_kv_heads": 8})
        == "aotriton"
    )
    assert _select_prefill_route(**{**global_block, "head_dim": 128}) == "exact"


def test_the_runner_forwards_the_decode_preference_to_the_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner hands the layer the profile's decode preference unchanged."""

    from hipengine.runtime import gemma4 as gemma4_module
    from tests.test_unit_gemma4_sliding_kv_ownership import (
        _dummy_buffer,
        _make_runner,
        _tiny_config,
    )

    config = _tiny_config()
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    captured: list[dict] = []
    monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(
        gemma4_module, "gemma4_layer_forward_bf16", lambda *a, **k: captured.append(k)
    )
    try:
        runner.decode_attention_variants = (DECODE_ATTENTION_STAGED,)
        rope = config.attention[0].rope
        runner._launch_block(
            [1],
            tables={rope: (_dummy_buffer(1), _dummy_buffer(2))},
            masks={64: _dummy_buffer(3), None: _dummy_buffer(4)},
            kv_write_offset=100,
            key_begin_at=None,
            needs_logits=False,
            decode_phase=True,
        )
        assert captured, "the layer was never called"
        assert all(
            kwargs["decode_attention_variants"] == (DECODE_ATTENTION_STAGED,)
            for kwargs in captured
        )
    finally:
        runner.close()


# --- the semantic decode phase ----------------------------------------------


def _long_global_config() -> Gemma4TextConfig:
    """One global layer at the real 512/8 geometry, cheap everywhere else."""

    return Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=(
            Gemma4AttentionGeometry(
                layer_type="full_attention",
                num_heads=16,
                num_kv_heads=2,
                head_dim=512,
                rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=512, rope_angles=256),
                sliding_window=None,
                k_eq_v=False,
            ),
        ),
        vocab_size=64,
    )


def _drive_forward_with_stubs(monkeypatch: pytest.MonkeyPatch, runner, config) -> list[dict]:
    """Patch every device launch a CPU ``forward`` needs; return the layer kwargs."""

    from hipengine.runtime import gemma4 as gemma4_module

    captured: list[dict] = []
    dummy = SimpleNamespace(ptr=0)
    attention = config.attention[0]

    monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_rmsnorm_f32w_bf16", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "launch_gguf_linear", lambda *a, **k: None)
    monkeypatch.setattr(
        gemma4_module,
        "gemma4_layer_forward_bf16",
        lambda *a, **k: captured.append(k),
    )
    monkeypatch.setattr(
        runner,
        "_stage_block_content",
        lambda tokens, **k: ({attention.rope: (dummy, dummy)}, {attention.sliding_window: dummy}),
    )
    monkeypatch.setattr(runner, "_prepare_sliding_caches", lambda *a, **k: None)
    monkeypatch.setattr(
        runner, "_collect_block", lambda *a, **k: np.zeros((1, 64), dtype=np.float32)
    )
    monkeypatch.setattr(runner, "_q8_mmq_prefill_session", lambda: nullcontext())
    return captured


def test_a_one_row_prefill_tail_stays_native_while_a_decode_step_takes_the_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner's chunk boundary, not the selector helper.

    A 16,385-token prompt over 1,024-token blocks ends in a one-row block at a
    context past the resident bound. That block is a prefill tail, so it keeps
    the native kernel; a genuine one-row ``forward`` at the same context is a
    decode step and takes the preference.
    """

    from tests.test_unit_gemma4_sliding_kv_ownership import _make_runner

    config = _long_global_config()
    runner, _ = _make_runner(monkeypatch, config=config, capacity=16640, max_block=1024)
    try:
        runner.decode_attention_variants = (DECODE_ATTENTION_STAGED,)
        captured = _drive_forward_with_stubs(monkeypatch, runner, config)

        # Chunked prefill: 16 full blocks plus a one-row tail.
        runner.forward([i % 64 for i in range(16385)])
        assert captured, "the layer was never called"
        assert captured[-1]["rows"] == 1, "the chunk boundary was not exercised"
        assert all(kwargs["decode_attention_variants"] is None for kwargs in captured), (
            "a prefill block forwarded the decode preference"
        )

        # A genuine incremental decode step at the same long context.
        captured.clear()
        runner.forward([7])
        assert len(captured) == 1
        assert captured[0]["rows"] == 1
        assert captured[0]["decode_attention_variants"] == (DECODE_ATTENTION_STAGED,)
    finally:
        runner.close()


# --- the decode-graph preparation -------------------------------------------


def _run_graph_preparation(monkeypatch: pytest.MonkeyPatch, position: int):
    """Run one cold ``_capture`` preparation; return ``(builds, owner, session)``."""

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged
    from hipengine.runtime import gemma4_decode_graph as graph

    geometry = SimpleNamespace(
        head_dim=512, num_heads=16, num_kv_heads=2, sliding_window=None
    )
    capturing = False
    builds: list[dict] = []

    def allocate(nbytes, **kwargs):
        assert not capturing, "attention allocated inside graph capture"
        return DeviceBuffer(ptr=0x1000, nbytes=nbytes)

    def begin(*args, **kwargs):
        nonlocal capturing
        capturing = True

    def end(*args):
        nonlocal capturing
        capturing = False
        return 100

    runtime = SimpleNamespace(
        current_device=lambda: 0,
        stream_begin_capture=begin,
        stream_end_capture=end,
        graph_instantiate=lambda _: 200,
    )
    monkeypatch.setattr(attention, "malloc", allocate)
    monkeypatch.setattr(attention, "build_gemma4_attention", lambda **kwargs: object())
    monkeypatch.setattr(
        attention, "split_workspace_bytes", lambda t, h, d, k, s, **kw: t * h * k * 4
    )
    monkeypatch.setattr(
        attention, "flash_workspace_bytes", lambda t, h, d, s, **kw: t * h * d * s * 4
    )
    monkeypatch.setattr(graph, "get_hip_runtime", lambda: runtime)

    def build_staged(**kwargs):
        assert not capturing, "the staged library was built inside graph capture"
        builds.append(kwargs)
        return object()

    monkeypatch.setattr(staged, "build_gemma4_attention_staged", build_staged)

    owner = attention.Gemma4AttentionScratch()
    runner = SimpleNamespace(
        position=position,
        prefill_attention_variants=None,
        decode_attention_variants=(DECODE_ATTENTION_STAGED,),
        weights=SimpleNamespace(
            layers=[None],
            config=SimpleNamespace(geometry=lambda index: geometry),
        ),
        _scratches=[SimpleNamespace(attention=owner)],
    )
    session = object.__new__(graph.Gemma4DecodeGraphSession)
    session._runner_ref = lambda: runner
    session._stream, session._captures = 7, 0
    session._retire = lambda: None
    session._identify_appends = lambda *a, **kw: ()
    session._zero_pending_tail = lambda _: None
    bucket = graph._Bucket.for_position(position)

    def launch(*args, **kwargs):
        assert capturing
        owner.buffer(1, stream=kwargs["stream"], runtime=runtime)

    runner._launch_block = launch
    session._capture(bucket, {}, {}, ("test",), token=1)
    return builds, owner, session, bucket


@pytest.mark.parametrize(
    ("position", "expect_staged"),
    [
        # keys == 15616: the last resident count, so the native kernel runs and
        # no staged library is needed.
        (15615, False),
        # keys == 15680: past the resident bound, so the staged candidate runs and
        # its cold library must be loaded before capture.
        (15679, True),
    ],
)
def test_graph_preparation_preloads_the_staged_library_across_the_threshold(
    monkeypatch: pytest.MonkeyPatch, position: int, expect_staged: bool
) -> None:
    """A cold staged library is built before capture, exactly where it is chosen.

    The first staged launch cannot dlopen or compile inside a capture, so the
    preparation that reserves the candidate's workspace must also load its
    library -- outside the capture window, and only on the side of the resident
    bound where the candidate is selected.
    """

    builds, _owner, session, bucket = _run_graph_preparation(monkeypatch, position)

    assert session._captures == 1
    if expect_staged:
        assert bucket.end > LAST_RESIDENT_KEYS
        assert builds, "the staged library was not preloaded before capture"
    else:
        assert bucket.end <= LAST_RESIDENT_KEYS
        assert not builds, "the staged library was loaded below the resident bound"
