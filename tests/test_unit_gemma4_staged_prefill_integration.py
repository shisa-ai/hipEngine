"""The scoped multi-row sliding prefill-attention substitution, on CPU.

The gfx1100 production profile routes a sliding prefill block that the tiled and
AOTriton routes refuse -- a sliding layer whose presented key count exceeds its
window -- to the staged multi-row attention candidate. The native exact kernel
serves that block by walking all presented keys from column 0; the staged
candidate keeps the same arithmetic (same score tree, same ``j mod 256`` softmax
partition, same ascending-key PV chain) in a shared footprint that is constant in
``keys``.

This file pins four separate contracts, all host-only -- no build, no device, no
kernel launch:

* **The capability and the trim.** The staged plan serves the sliding prefill
  geometry, its LDS is constant in ``keys``, a sliding layer's presented range is
  window-bounded, and the per-row window trim is exact on a sliding-causal mask.
* **The scope.** A global block, a one-row block, an unsupported head width, a
  non-integral GQA ratio, or no requested scope all keep the native kernel; a
  supported ratio the candidate implements (including ratio 4 at head_dim 512)
  takes the candidate.
* **The registry and the profile.** A replaced callable under the exact-prefill
  key is what the layer runs, the strict fallback is registered too, and
  ``production`` selects the candidate while ``strict`` (and the unregistered
  ``batch_invariant`` fallback) keeps the native kernel. The binder carries the
  manifest's choice onto the generator verbatim.
* **The plumbing and the surrounding routes.** The layer's native/exact call site
  forwards the layer's ``window`` unchanged and a cache-relative ``row_offset``,
  the runner forwards the preference, and the tiled/AOTriton/exact split and the
  existing ``prefill_attention`` scope are unchanged.

Run:
    uv run pytest tests/test_unit_gemma4_staged_prefill_integration.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    resolve_runtime_profile,
)
from hipengine.generation.gemma4_profiles import (
    DECODE_ATTENTION_LAYER,
    DECODE_ATTENTION_PLAIN,
    DECODE_ATTENTION_QUANT,
    EXACT_PREFILL_ATTENTION_LAYER,
    EXACT_PREFILL_ATTENTION_PLAIN,
    EXACT_PREFILL_ATTENTION_QUANT,
    EXACT_PREFILL_ATTENTION_SCOPE,
    EXACT_PREFILL_ATTENTION_STAGED,
    GEMMA4_GGUF_BACKEND,
    GEMMA4_GGUF_MODEL,
    GEMMA4_GGUF_QUANT,
    MOE_PREFILL_ENV,
    gemma4_gguf_profiles_registered,
    register_gemma4_gguf_profiles,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer_module
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_staged import (
    gemma4_attention_staged_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    _select_exact_prefill_attention,
    _select_prefill_route,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.runtime.gemma4 import _sliding_read_range
from tests.test_unit_gemma4_attention_staged import staged_cpu_oracle

# Importing the built-in generators registers the gfx1100 Gemma 4 kernels the
# profile names. It runs at collection time so the conftest kernel baseline
# (snapshotted after collection) includes them and the per-test registry restore
# cannot wipe them between the parametrized profile cases.
from hipengine.generation import register_builtin_generators

register_builtin_generators()
register_gemma4_gguf_profiles()

# Gemma 4's two attention geometries, as ``staged_plan`` takes them.
SLIDING = dict(num_heads=16, num_kv_heads=8, head_dim=256)
GLOBAL = dict(num_heads=16, num_kv_heads=2, head_dim=512)
SLIDING_WINDOW = 1024

# The measured chunk shape from the 8K attribution: 1024 query rows over the
# window-bounded 2047 presented keys (window 1024 + rows 1024 - 1).
SLIDING_CHUNK_ROWS = 1024
SLIDING_CHUNK_KEYS = 2047


# --- the geometry the integration serves ------------------------------------


@pytest.mark.parametrize("keys", [SLIDING_CHUNK_KEYS, 1024, 4096, 16384])
def test_staged_plan_serves_the_sliding_prefill_geometry(keys: int) -> None:
    """The multi-row sliding block is a plan, not a capability miss."""

    plan = staged.staged_plan(tokens=SLIDING_CHUNK_ROWS, keys=keys, **SLIDING)

    assert plan.rows == SLIDING_CHUNK_ROWS * SLIDING["num_heads"]
    assert plan.pv_threads == staged.THREADS == 256  # multi-row keeps the 256-thread PV
    assert not plan.pv_singleton
    # The score and softmax stages keep the strict 256-lane shapes.
    assert plan.threads == staged.THREADS == 256
    # One score CTA holds every query head of a KV head, up to
    # SCORE_ROWS_PER_BLOCK of them, so the grid counts (token, KV head) groups.
    groups = -(-(SLIDING["num_heads"] // SLIDING["num_kv_heads"]) // staged.SCORE_ROWS_PER_BLOCK)
    assert plan.score_grid == (
        SLIDING_CHUNK_ROWS * SLIDING["num_kv_heads"] * groups,
        staged.staged_score_chunks(keys),
    )


def test_staged_plan_serves_the_global_prefill_geometry() -> None:
    """The 512-wide geometry is a plan too, even though the route is not scoped to it."""

    plan = staged.staged_plan(tokens=SLIDING_CHUNK_ROWS, keys=2048, **GLOBAL)

    assert plan.rows == SLIDING_CHUNK_ROWS * GLOBAL["num_heads"]
    assert plan.pv_grid == (SLIDING_CHUNK_ROWS, GLOBAL["num_kv_heads"], 1)


def test_staged_prefill_lds_does_not_grow_with_keys() -> None:
    """The candidate's reason to exist, restated for the multi-row shape."""

    small = staged.staged_plan(tokens=SLIDING_CHUNK_ROWS, keys=1024, **SLIDING)
    large = staged.staged_plan(tokens=SLIDING_CHUNK_ROWS, keys=262144, **SLIDING)

    assert small.lds_bytes == large.lds_bytes == staged.staged_lds_bytes(256)
    assert large.lds_bytes <= staged.LDS_BUDGET_BYTES


def test_staged_prefill_workspace_matches_the_python_formula() -> None:
    """Pin the sizing the integration's memory budget depends on."""

    tokens, heads, keys = SLIDING_CHUNK_ROWS, SLIDING["num_heads"], SLIDING_CHUNK_KEYS
    plan = staged.staged_plan(tokens=tokens, keys=keys, **SLIDING)
    rows = tokens * heads
    chunks = staged.staged_score_chunks(keys)

    assert plan.workspace.scores_floats == rows * keys
    assert plan.workspace.chunk_max_floats == rows * chunks
    assert plan.workspace.denominator_floats == rows
    assert plan.workspace_bytes == (rows * keys + rows * chunks + rows) * 4
    assert plan.workspace_bytes == staged.staged_workspace_bytes(tokens, heads, keys)


# --- the window-bounded presented range -------------------------------------


@pytest.mark.parametrize("start", [1024, 7168, 260000])
def test_a_sliding_layer_presents_a_window_bounded_key_count(start: int) -> None:
    """The runtime trims a sliding read to the window, so staged's workspace is bounded.

    ``_sliding_read_range`` returns the first cached key a sliding block must
    walk; the layer then presents ``start + rows - first`` keys. That count is
    ``window + rows - 1`` once the block is past the window, independent of the
    context, which is what keeps the candidate's ``tokens * heads * keys``
    workspace affordable at any prompt length.
    """

    attention = SimpleNamespace(sliding_window=SLIDING_WINDOW)
    first = _sliding_read_range(attention, start, SLIDING_CHUNK_ROWS)
    keys = start + SLIDING_CHUNK_ROWS - first

    assert keys == SLIDING_WINDOW + SLIDING_CHUNK_ROWS - 1 == SLIDING_CHUNK_KEYS

    plan = staged.staged_plan(tokens=SLIDING_CHUNK_ROWS, keys=keys, **SLIDING)
    # A per-layer workspace of ~128 MiB, and it does not grow with the context.
    assert plan.workspace_bytes < 192 * 1024 * 1024


def test_a_global_layer_presents_the_full_context() -> None:
    """The contrast: without a window the workspace would scale with the context.

    This is why the route is scoped to sliding blocks. A 1024-row global block at
    a 256K context would need a multi-gigabyte workspace, which the sliding route
    never does.
    """

    attention = SimpleNamespace(sliding_window=None)
    start, rows = 262144 - 1024, SLIDING_CHUNK_ROWS
    first = _sliding_read_range(attention, start, rows)
    keys = start + rows - first
    assert first == 0 and keys == start + rows

    plan = staged.staged_plan(tokens=rows, keys=keys, **GLOBAL)
    assert plan.workspace_bytes > 16 * 1024**3  # > 16 GiB


# --- the per-row sliding trim is exact --------------------------------------


def _sliding_causal_mask(tokens: int, keys: int, window: int, row_offset: int) -> np.ndarray:
    """The keep mask a sliding-causal layer builds over the presented columns."""

    positions = np.arange(keys)[None, :]
    own = row_offset + np.arange(tokens)[:, None]
    return ((positions <= own) & (positions > own - window)).astype(np.uint8)


@pytest.mark.parametrize("tokens", [2, 3, 17, 1024])
@pytest.mark.parametrize("window", [1, 4, 64, 1024])
def test_the_window_trim_changes_no_result_on_a_sliding_mask(
    tokens: int, window: int
) -> None:
    """Trimming a zero-weight column cannot move the output.

    The candidate walks ``[first_kept, walk_end)`` when given ``window`` and
    ``row_offset``; the strict block kernel does the same. The shape is a later
    sliding chunk -- presented keys ``window + rows - 1`` and block start
    ``window - 1`` -- which is exactly the geometry the 8K attribution measured.
    This checks the CPU oracle's trimmed and untrimmed reductions agree on a
    genuine sliding-causal mask, which is the mask the layer presents.
    """

    row_offset = window - 1
    keys = window + tokens - 1
    rng = np.random.default_rng(20261006 + tokens + window)
    query = rng.standard_normal((tokens, 4, 8), dtype=np.float32)
    key = rng.standard_normal((keys, 2, 8), dtype=np.float32)
    value = rng.standard_normal((keys, 2, 8), dtype=np.float32)
    mask = _sliding_causal_mask(tokens, keys, window, row_offset)

    trimmed = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=2, scale=1.0,
        window=window, row_offset=row_offset,
    )
    masked_only = staged_cpu_oracle(query, key, value, mask, num_kv_heads=2, scale=1.0)

    assert np.array_equal(trimmed, masked_only)


@pytest.mark.parametrize("tokens", [1, 4, 64])
def test_the_trim_is_a_noop_on_a_dense_causal_first_chunk(tokens: int) -> None:
    """The first chunk presents ``keys == rows``; the window does not bind."""

    window, row_offset, keys = 1024, 0, tokens
    rng = np.random.default_rng(11 + tokens)
    query = rng.standard_normal((tokens, 2, 8), dtype=np.float32)
    key = rng.standard_normal((keys, 2, 8), dtype=np.float32)
    value = rng.standard_normal((keys, 2, 8), dtype=np.float32)
    positions = np.arange(keys)[None, :]
    own = np.arange(tokens)[:, None]
    mask = (positions <= own).astype(np.uint8)

    trimmed = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=2, scale=1.0,
        window=window, row_offset=row_offset,
    )
    masked_only = staged_cpu_oracle(query, key, value, mask, num_kv_heads=2, scale=1.0)

    assert np.array_equal(trimmed, masked_only)


def test_the_trim_window_matches_the_strict_prefill_formula() -> None:
    """The candidate's first_kept/walk_end mirror the strict block kernel's.

    ``gemma4_attn_first_kept`` and the block kernel's ``walk_end`` are the strict
    definition; the staged kernel's ``gemma4_staged_first_kept``/``_walk_end``
    clamp the same expressions. The oracle applies the trim, so pinning it
    against the strict formula is what makes the GPU oracle comparison evidence
    about the kernel rather than about the oracle's own idea of the window.

    The mask is the layer's own sliding-causal mask, so the trim's promise (the
    mask is zero below ``first_kept`` and above ``walk_end``) holds by
    construction.
    """

    rng = np.random.default_rng(7)
    shapes = [
        # (tokens, window, row_offset, keys): a later chunk and a dense first one.
        *[(tokens, window, window - 1, window + tokens - 1)
          for tokens in (2, 5, 33) for window in (1, 4, 16, 1024)],
        (5, 1024, 0, 5),
        (33, 1024, 0, 33),
    ]
    for tokens, window, row_offset, keys in shapes:
        query = rng.standard_normal((tokens, 2, 4), dtype=np.float32)
        key = rng.standard_normal((keys, 1, 4), dtype=np.float32)
        value = rng.standard_normal((keys, 1, 4), dtype=np.float32)
        mask = _sliding_causal_mask(tokens, keys, window, row_offset)

        oracle = staged_cpu_oracle(
            query, key, value, mask, num_kv_heads=1, scale=1.0,
            window=window, row_offset=row_offset,
        )
        for token in range(tokens):
            own = row_offset + token
            # The strict block kernel's own bounds.
            first_kept = max(0, own - window + 1)
            walk_end = min(keys, max(0, own + 1))
            assert mask[token, first_kept:walk_end].all()
            assert not mask[token, :first_kept].any()
            assert not mask[token, walk_end:].any()
            logits = (
                key[:, 0, :].astype(np.float64)
                @ query[token, 0, :].astype(np.float64)
            )
            positions = np.arange(keys)
            kept = (positions >= first_kept) & (positions < walk_end)
            logits = np.where(kept, logits, -np.inf)
            weights = np.exp(logits - logits.max())
            expected = (weights @ value[:, 0, :].astype(np.float64)) / weights.sum()
            assert np.allclose(oracle[token, 0], expected, atol=1e-9)


def test_a_window_wider_than_the_presented_range_trims_nothing() -> None:
    """``window >= keys`` leaves ``first_kept`` at 0, so the walk is the full range."""

    tokens, keys, window = 4, 8, 4096
    rng = np.random.default_rng(3)
    query = rng.standard_normal((tokens, 1, 4), dtype=np.float32)
    key = rng.standard_normal((keys, 1, 4), dtype=np.float32)
    value = rng.standard_normal((keys, 1, 4), dtype=np.float32)
    # A causal mask: the trailing columns are already zero, so a window wider
    # than the range adds no trim the mask does not already make.
    positions = np.arange(keys)[None, :]
    own = np.arange(tokens)[:, None]
    mask = (positions <= own).astype(np.uint8)

    trimmed = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=1, scale=1.0,
        window=window, row_offset=0,
    )
    masked_only = staged_cpu_oracle(query, key, value, mask, num_kv_heads=1, scale=1.0)

    assert np.array_equal(trimmed, masked_only)


def test_the_staged_launcher_exposes_the_prefill_trim_keywords() -> None:
    """The exact call site can pass ``window``/``row_offset`` without an adapter."""

    import inspect

    signature = inspect.signature(staged.gemma4_attention_staged_bf16)
    assert "window" in signature.parameters
    assert "row_offset" in signature.parameters
    assert signature.parameters["window"].default == 0
    assert signature.parameters["row_offset"].default == 0


# --- the exact-prefill selector ---------------------------------------------


def select(**overrides):
    base = dict(
        exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_STAGED,),
        tokens=SLIDING_CHUNK_ROWS,
        keys=SLIDING_CHUNK_KEYS,
        num_heads=SLIDING["num_heads"],
        num_kv_heads=SLIDING["num_kv_heads"],
        head_dim=SLIDING["head_dim"],
        sliding_window=SLIDING_WINDOW,
    )
    base.update(overrides)
    return _select_exact_prefill_attention(**base)


def test_the_selector_routes_a_sliding_multi_row_block_to_staged() -> None:
    """The measured chunk shape is the scope: multi-row, sliding, supported."""

    launcher, route = select()

    assert route == EXACT_PREFILL_ATTENTION_STAGED
    assert launcher is gemma4_attention_staged_bf16


@pytest.mark.parametrize(
    "overrides",
    [
        dict(sliding_window=None),  # a global block: outside the sliding scope
        dict(tokens=1),             # a one-row prefill tail or decode step
        dict(exact_prefill_attention_variants=None),  # nothing asked for the scope
        dict(exact_prefill_attention_variants=()),    # an empty request
        dict(keys=0),               # a meaningless key count
    ],
)
def test_the_selector_declines_a_call_outside_the_scope(overrides) -> None:
    """Outside the scope the selector returns ``(None, None)``.

    That sentinel is the contract: the caller keeps whatever launcher the decode
    scope already resolved, so a global block or a one-row block never has its
    resolved callable overridden by a scope it is not part of.
    """

    launcher, route = select(**overrides)

    assert launcher is None
    assert route is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(head_dim=128),         # a width the score tree does not implement
        dict(num_heads=16, num_kv_heads=3, head_dim=256),  # a non-integral GQA ratio
        # A GQA ratio past ``MAX_GRID_DIM * MAX_ROWS_PER_BLOCK`` has no PV grid.
        dict(num_heads=16 * 60000, num_kv_heads=1, head_dim=256),
    ],
)
def test_the_selector_keeps_the_native_kernel_for_an_unsupported_shape(overrides) -> None:
    """In scope but unsupported: the registered strict fallback runs.

    The call is inside the scope (multi-row, sliding), so the prefill scope's own
    registration is the callable that runs -- not the module-level wrapper --
    which is what keeps the strict multi-row prefill registry executable.
    """

    launcher, route = select(**overrides)

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


def test_the_selector_accepts_a_supported_ratio_four_geometry() -> None:
    """Capability, not a Gemma shape: ratio 4 at head_dim 512 is a staged plan.

    The candidate's score tree covers head_dim 256 and 512 and its PV
    decomposition covers any integral GQA ratio, so a sliding layer with
    ``16q/4kv`` at head_dim 512 is inside the scope. The scope is the geometry
    the kernels declare, never an enumerated list of known-good shapes.
    """

    launcher, route = select(num_heads=16, num_kv_heads=4, head_dim=512)

    assert route == EXACT_PREFILL_ATTENTION_STAGED
    assert launcher is gemma4_attention_staged_bf16


def test_the_selector_only_requests_the_scope_it_was_asked_for() -> None:
    """Requesting only the plain variant keeps the native kernel, via the registry."""

    launcher, route = select(
        exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_PLAIN,)
    )

    assert route == "exact"
    assert launcher is gemma4_attention_prefill_bf16


# --- the registry is executable ---------------------------------------------


def _exact_prefill_key(variant: str) -> KernelKey:
    return KernelKey(
        GEMMA4_GGUF_BACKEND, EXACT_PREFILL_ATTENTION_LAYER,
        EXACT_PREFILL_ATTENTION_QUANT, variant,
    )


def _decode_key(variant: str) -> KernelKey:
    return KernelKey(
        GEMMA4_GGUF_BACKEND, DECODE_ATTENTION_LAYER, DECODE_ATTENTION_QUANT, variant
    )


def test_a_replaced_staged_callable_is_the_one_the_selector_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registration is executable, not decorative."""

    def replacement(*args, **kwargs):
        return 0

    register(_exact_prefill_key(EXACT_PREFILL_ATTENTION_STAGED), replacement, replace=True)
    launcher, route = select()

    assert route == EXACT_PREFILL_ATTENTION_STAGED
    assert launcher is replacement


@pytest.mark.parametrize(
    "variants",
    [(EXACT_PREFILL_ATTENTION_PLAIN,), (EXACT_PREFILL_ATTENTION_STAGED,)],
)
def test_the_strict_fallback_is_the_registered_plain_callable(
    monkeypatch: pytest.MonkeyPatch, variants,
) -> None:
    """An in-scope call the candidate cannot execute falls back to the *registered*
    plain entry -- the prefill scope's own registration, not the module wrapper."""

    def replacement(*args, **kwargs):
        return 0

    register(_exact_prefill_key(EXACT_PREFILL_ATTENTION_PLAIN), replacement, replace=True)
    # head_dim 128 is inside the scope (multi-row, sliding) but the score tree
    # does not implement it, so the scope keeps the native arithmetic.
    launcher, route = select(
        head_dim=128, exact_prefill_attention_variants=variants
    )

    assert route == "exact"
    assert launcher is replacement


def test_the_exact_prefill_variants_are_registered_under_their_own_layer() -> None:
    register_gemma4_gguf_profiles()

    assert resolve(
        backend=GEMMA4_GGUF_BACKEND,
        layer=EXACT_PREFILL_ATTENTION_LAYER,
        quant=EXACT_PREFILL_ATTENTION_QUANT,
        variant=EXACT_PREFILL_ATTENTION_PLAIN,
    ) is gemma4_attention_prefill_bf16
    assert resolve(
        backend=GEMMA4_GGUF_BACKEND,
        layer=EXACT_PREFILL_ATTENTION_LAYER,
        quant=EXACT_PREFILL_ATTENTION_QUANT,
        variant=EXACT_PREFILL_ATTENTION_STAGED,
    ) is gemma4_attention_staged_bf16


# --- the layer runs the registered callable and forwards the trim -----------


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
    monkeypatch.setattr(layer_module, "_moe_stream", lambda: 0)
    monkeypatch.setattr(layer_module, "_append_kv", lambda *a, **k: None)
    for name in _PRE_ATTENTION_LAUNCHERS:
        monkeypatch.setattr(layer_module, name, lambda *a, **k: None)


def _sliding_layer_stub() -> SimpleNamespace:
    return SimpleNamespace(
        input_layernorm=0, qkv_proj=None, q_proj=0, k_proj=0, v_proj=0,
        q_norm=0, k_norm=0,
    )


def _sliding_scratch_stub(*, tokens: int = SLIDING_CHUNK_ROWS) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=tokens,
        hidden_size=16,
        geometry=SimpleNamespace(
            num_heads=16, num_kv_heads=8, head_dim=256, scale=1.0,
            k_eq_v=False, sliding_window=SLIDING_WINDOW,
        ),
        buffer=lambda name: SimpleNamespace(ptr=0),
        attention=SimpleNamespace(),
    )


def _global_scratch_stub(*, tokens: int = 1) -> SimpleNamespace:
    """A global head_dim 512 layer, so the call site reaches the exact route."""

    return SimpleNamespace(
        tokens=tokens,
        hidden_size=16,
        geometry=SimpleNamespace(
            num_heads=16, num_kv_heads=2, head_dim=512, scale=1.0,
            k_eq_v=False, sliding_window=None,
        ),
        buffer=lambda name: SimpleNamespace(ptr=0),
        attention=SimpleNamespace(),
    )


def test_the_layer_invokes_the_registered_staged_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replaced exact-prefill staged entry is what the layer's exact site calls."""

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    register(_exact_prefill_key(EXACT_PREFILL_ATTENTION_STAGED), replacement, replace=True)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0, 0, 0, 0, _sliding_layer_stub(),
            scratch=_sliding_scratch_stub(),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=8192, write_offset=1024
            ),
            rows=SLIDING_CHUNK_ROWS,
            key_begin=1,
            attention_mask_is_causal=True,
            window=SLIDING_WINDOW,
            exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_STAGED,),
        )

    assert seen, "the registered staged callable was not invoked"
    assert seen[0]["keys"] == SLIDING_CHUNK_KEYS
    assert seen[0]["tokens"] == SLIDING_CHUNK_ROWS
    # The staged multi-row prefill is the one launcher that receives the trim.
    assert seen[0]["window"] == SLIDING_WINDOW
    assert seen[0]["row_offset"] == 1024 - 1


def test_the_layer_exact_site_forwards_the_sliding_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The integration passes the trim the candidate was verified with.

    The layer's ``window`` is forwarded unchanged, and ``row_offset`` is the
    block's start in the mask's column frame -- cache-relative, exactly as the
    key pointer already is.
    """

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    register(_exact_prefill_key(EXACT_PREFILL_ATTENTION_STAGED), replacement, replace=True)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0, 0, 0, 0, _sliding_layer_stub(),
            scratch=_sliding_scratch_stub(),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=8192, write_offset=1024
            ),
            rows=SLIDING_CHUNK_ROWS,
            key_begin=1,
            attention_mask_is_causal=True,
            window=SLIDING_WINDOW,
            exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_STAGED,),
        )

    assert seen, "the exact attention call site never ran"
    assert seen[0]["window"] == SLIDING_WINDOW
    assert seen[0]["row_offset"] == 1024 - 1


def test_the_layer_forwards_a_zero_window_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``key_begin_at`` capture passes window 0; the layer forwards it verbatim.

    The runner's graph-capture path sets ``window=0`` ("no promise") while the
    layer's geometry still selects the staged sliding prefill, so the staged
    launcher must receive the caller's 0 -- not a window reconstructed from
    ``geometry.sliding_window``. ``row_offset`` still carries the block's
    cache-relative start.
    """

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    register(_exact_prefill_key(EXACT_PREFILL_ATTENTION_STAGED), replacement, replace=True)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0, 0, 0, 0, _sliding_layer_stub(),
            scratch=_sliding_scratch_stub(),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=8192, write_offset=1024
            ),
            rows=SLIDING_CHUNK_ROWS,
            key_begin=1,
            attention_mask_is_causal=True,
            window=0,
            exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_STAGED,),
        )

    assert seen, "the staged exact attention call site never ran"
    assert seen[0]["window"] == 0
    assert seen[0]["row_offset"] == 1024 - 1


def test_the_layer_exact_site_keeps_native_when_no_scope_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no exact-prefill request the native entry runs, unmodified."""

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    # No exact-prefill scope requested, so the selector returns the module-level
    # native entry directly; patch it to observe the call.
    monkeypatch.setattr(layer_module, "gemma4_attention_prefill_bf16", replacement)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0, 0, 0, 0, _sliding_layer_stub(),
            scratch=_sliding_scratch_stub(),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=8192, write_offset=1024
            ),
            rows=SLIDING_CHUNK_ROWS,
            key_begin=1,
            attention_mask_is_causal=True,
            window=SLIDING_WINDOW,
        )

    assert seen, "the native attention call site never ran"
    # The native entry keeps the keyword set it has always been called with.
    assert "window" not in seen[0]
    assert "row_offset" not in seen[0]


def test_the_layer_strict_multirow_prefill_runs_the_registered_plain_without_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict multi-row sliding prefill is the prefill scope's own registration.

    The scope applies (multi-row, sliding), so the registered ``gemma4_plain``
    entry is what runs -- and it is called without the staged-only trim keywords.
    """

    _install_layer_stubs(monkeypatch)
    seen: list[dict] = []

    def replacement(*args, **kwargs):
        seen.append(kwargs)
        raise _ReachedAttention

    register(_exact_prefill_key(EXACT_PREFILL_ATTENTION_PLAIN), replacement, replace=True)
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0, 0, 0, 0, _sliding_layer_stub(),
            scratch=_sliding_scratch_stub(),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=8192, write_offset=1024
            ),
            rows=SLIDING_CHUNK_ROWS,
            key_begin=1,
            attention_mask_is_causal=True,
            window=SLIDING_WINDOW,
            exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_PLAIN,),
        )

    assert seen, "the registered strict plain entry was not invoked"
    assert "window" not in seen[0]
    assert "row_offset" not in seen[0]


def test_the_layer_keeps_the_registered_decode_plain_when_the_prefill_scope_is_out_of_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An out-of-scope call keeps the decode scope's resolved callable.

    The exact-prefill scope is present, but this is a one-row global block --
    outside the sliding multi-row scope -- so the prefill selector declines and
    the layer runs the *registered* decode plain entry, not the prefill scope's
    plain, and with no trim keywords.
    """

    _install_layer_stubs(monkeypatch)
    decode_seen: list[dict] = []
    prefill_seen: list[dict] = []

    def decode_replacement(*args, **kwargs):
        decode_seen.append(kwargs)
        raise _ReachedAttention

    def prefill_replacement(*args, **kwargs):
        prefill_seen.append(kwargs)
        raise _ReachedAttention

    register(_decode_key(DECODE_ATTENTION_PLAIN), decode_replacement, replace=True)
    register(
        _exact_prefill_key(EXACT_PREFILL_ATTENTION_PLAIN), prefill_replacement, replace=True
    )
    with pytest.raises(_ReachedAttention):
        layer_module.gemma4_layer_forward_bf16(
            0, 0, 0, 0, _sliding_layer_stub(),
            scratch=_global_scratch_stub(tokens=1),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=8192, write_offset=100
            ),
            rows=1,
            key_begin=0,
            attention_mask_is_causal=True,
            window=0,
            decode_attention_variants=(DECODE_ATTENTION_PLAIN,),
            exact_prefill_attention_variants=(EXACT_PREFILL_ATTENTION_STAGED,),
        )

    assert decode_seen, "the registered decode plain entry was not invoked"
    assert not prefill_seen, (
        "an out-of-scope call was routed to the exact-prefill scope's callable"
    )
    assert "window" not in decode_seen[0]
    assert "row_offset" not in decode_seen[0]


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


def _exact_prefill_rows(resolved):
    return [
        selection
        for selection in resolved.manifest["selections"]
        if selection["layer"] == EXACT_PREFILL_ATTENTION_LAYER
    ]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (ExecutionProfile.PRODUCTION, EXACT_PREFILL_ATTENTION_STAGED),
        (ExecutionProfile.STRICT, EXACT_PREFILL_ATTENTION_PLAIN),
        # batch_invariant is deliberately unregistered: it falls back to strict.
        (ExecutionProfile.BATCH_INVARIANT, EXACT_PREFILL_ATTENTION_PLAIN),
    ],
)
def test_the_profile_selects_the_scoped_prefill_route(profile, expected) -> None:
    resolved = _resolved(profile)
    rows = _exact_prefill_rows(resolved)

    assert len(rows) == 1
    assert rows[0]["scope"] == EXACT_PREFILL_ATTENTION_SCOPE
    assert rows[0]["selected_variant"] == expected
    assert rows[0]["strict_fallback_variant"] == EXACT_PREFILL_ATTENTION_PLAIN
    assert rows[0]["registry_quant"] == EXACT_PREFILL_ATTENTION_QUANT


def test_the_binder_binds_the_selected_prefill_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binder carries the manifest's choice onto the generator verbatim."""

    # The binder pins ``HIPENGINE_GEMMA4_MOE_PREFILL`` as a process env var, so
    # pin it through monkeypatch first: teardown then restores whatever the run
    # started with instead of leaking the strict route into later tests.
    monkeypatch.setenv(MOE_PREFILL_ENV, "auto")
    for profile, expected in (
        (ExecutionProfile.PRODUCTION, (EXACT_PREFILL_ATTENTION_STAGED,)),
        (ExecutionProfile.STRICT, (EXACT_PREFILL_ATTENTION_PLAIN,)),
        (ExecutionProfile.BATCH_INVARIANT, (EXACT_PREFILL_ATTENTION_PLAIN,)),
    ):
        resolved = _resolved(profile)
        generator = SimpleNamespace()
        resolved.binder(generator, resolved)
        assert generator.exact_prefill_attention_variants == expected


# --- the surrounding routing is unchanged -----------------------------------


def test_the_existing_prefill_routing_is_unaffected() -> None:
    """The exact-prefill preference does not touch the tiled/aotriton/exact split."""

    global_block = dict(
        rows=128, keys=128, num_heads=16, num_kv_heads=2, head_dim=512,
        sliding_window=None, mask_is_causal=True,
    )
    assert _select_prefill_route(**global_block) == "tiled"
    assert (
        _select_prefill_route(**{**global_block, "head_dim": 256, "num_kv_heads": 8})
        == "aotriton"
    )
    # A sliding block past its window is exactly the native/exact geometry the
    # new scope serves -- but the route split itself is unchanged.
    assert (
        _select_prefill_route(
            rows=SLIDING_CHUNK_ROWS, keys=SLIDING_CHUNK_KEYS,
            num_heads=16, num_kv_heads=8, head_dim=256,
            sliding_window=SLIDING_WINDOW, mask_is_causal=True,
        )
        == "exact"
    )
    assert _select_prefill_route(**{**global_block, "head_dim": 128}) == "exact"


def test_the_existing_prefill_attention_scope_is_untouched() -> None:
    """The new scope is a separate registry key; the old one still resolves."""

    register_gemma4_gguf_profiles()

    assert resolve(
        backend=GEMMA4_GGUF_BACKEND,
        layer="prefill_attention",
        quant="gguf_q4_k_m",
        variant="gemma4_staged",
    ) is gemma4_attention_staged_bf16
    # And the new key is distinct from it.
    assert EXACT_PREFILL_ATTENTION_LAYER != "prefill_attention"


# --- the runtime plumbing ---------------------------------------------------


def _forwarding_runner(monkeypatch: pytest.MonkeyPatch, *, capacity: int = 256,
                       max_block: int = 8):
    from tests.test_unit_gemma4_sliding_kv_ownership import (
        _dummy_buffer,
        _make_runner,
        _tiny_config,
    )

    config = _tiny_config()
    runner, _ = _make_runner(monkeypatch, config=config, capacity=capacity, max_block=max_block)
    captured: list[dict] = []
    from hipengine.runtime import gemma4 as gemma4_module

    monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(
        gemma4_module, "gemma4_layer_forward_bf16", lambda *a, **k: captured.append(k)
    )
    rope = config.attention[0].rope
    return runner, config, captured, _dummy_buffer, rope


def test_the_runner_forwards_the_exact_prefill_preference_to_the_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner hands the layer the profile's exact-prefill preference unchanged."""

    runner, config, captured, _dummy_buffer, rope = _forwarding_runner(monkeypatch)
    try:
        runner.exact_prefill_attention_variants = (EXACT_PREFILL_ATTENTION_STAGED,)
        runner._launch_block(
            [1, 2, 3, 4],
            tables={rope: (_dummy_buffer(1), _dummy_buffer(2))},
            masks={64: _dummy_buffer(3), None: _dummy_buffer(4)},
            kv_write_offset=100,
            key_begin_at=None,
            needs_logits=False,
            decode_phase=False,
        )
        assert captured, "the layer was never called"
        assert all(
            kwargs["exact_prefill_attention_variants"] == (EXACT_PREFILL_ATTENTION_STAGED,)
            for kwargs in captured
        )
    finally:
        runner.close()


def test_the_runner_forwards_a_zero_window_when_key_begin_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``key_begin_at`` is the graph-capture path: its zero window is intentional.

    A frozen key begin makes the layer's window 0 -- "no promise" -- and the
    layer then passes that 0 straight through, so a captured decode step keeps
    the full walk rather than a trim derived from a geometry it did not promise.
    """

    runner, config, captured, _dummy_buffer, rope = _forwarding_runner(monkeypatch)
    try:
        runner.exact_prefill_attention_variants = (EXACT_PREFILL_ATTENTION_STAGED,)
        runner._launch_block(
            [1, 2, 3, 4],
            tables={rope: (_dummy_buffer(1), _dummy_buffer(2))},
            masks={64: _dummy_buffer(3), None: _dummy_buffer(4)},
            kv_write_offset=100,
            key_begin_at=lambda attention: 0,
            needs_logits=False,
            decode_phase=False,
        )
        assert captured, "the layer was never called"
        assert all(kwargs["window"] == 0 for kwargs in captured)
        assert all(kwargs["key_begin"] == 0 for kwargs in captured)
    finally:
        runner.close()


def test_the_runner_clears_the_prefill_preference_for_a_verify_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verify block keeps the strict variant, exactly as the prefill scope does."""

    runner, config, captured, _dummy_buffer, rope = _forwarding_runner(monkeypatch)
    try:
        runner.exact_prefill_attention_variants = (EXACT_PREFILL_ATTENTION_STAGED,)
        runner._launch_block(
            [1, 2, 3, 4],
            tables={rope: (_dummy_buffer(1), _dummy_buffer(2))},
            masks={64: _dummy_buffer(3), None: _dummy_buffer(4)},
            kv_write_offset=100,
            key_begin_at=None,
            needs_logits=False,
            verification=True,
            decode_phase=False,
        )
        assert captured, "the layer was never called"
        assert all(
            kwargs["exact_prefill_attention_variants"] is None for kwargs in captured
        )
    finally:
        runner.close()


def test_the_runner_disables_the_prefill_preference_for_a_decode_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decode phase disables the exact-prefill scope outright.

    The prefill scope is never even offered to the layer on a genuine decode
    step, so it cannot win on a singleton decode; a one-row prefill tail
    (``decode_phase=False``) still forwards it and the layer's selector keeps it
    native.
    """

    runner, config, captured, _dummy_buffer, rope = _forwarding_runner(monkeypatch)
    try:
        runner.exact_prefill_attention_variants = (EXACT_PREFILL_ATTENTION_STAGED,)
        runner._launch_block(
            [1, 2, 3, 4],
            tables={rope: (_dummy_buffer(1), _dummy_buffer(2))},
            masks={64: _dummy_buffer(3), None: _dummy_buffer(4)},
            kv_write_offset=100,
            key_begin_at=None,
            needs_logits=False,
            decode_phase=True,
        )
        assert captured, "the layer was never called"
        assert all(
            kwargs["exact_prefill_attention_variants"] is None for kwargs in captured
        )
    finally:
        runner.close()


def test_the_selector_declines_a_one_row_call() -> None:
    """The phase boundary is the scope, not the row count alone.

    A genuine one-row decode step is the decode scope's; the exact-prefill scope
    requires ``tokens > 1``, so it declines a one-row call and the caller keeps
    the decode scope's resolved launcher.
    """

    launcher, route = select(tokens=1, keys=SLIDING_CHUNK_KEYS)

    assert launcher is None
    assert route is None
