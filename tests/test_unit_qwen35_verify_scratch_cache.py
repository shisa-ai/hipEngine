"""CPU lifetime contracts for cached verifier scratch objects."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_paro_runner as runner
from hipengine.runtime import workspace as workspace_module
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative.graph_stage import (
    GRAPH_STAGE_BINDING_MISS,
    SpecGraphBinding,
    SpecGraphStageCache,
)


_CLASSES = {
    "linear": runner.Qwen35ParoLinearAttentionScratch,
    "c1": runner.Qwen35ParoMoeScratch,
    "grouped": runner.Qwen35ParoGroupedMoeScratch,
    "dense": runner.Qwen35ParoDenseMlpScratch,
}
_NAMES = {
    "linear": ("linear_attn.attn_input", "linear_attn.qkv_z", "linear_attn.tree_recurrent_state"),
    "c1": ("moe.normed",), "grouped": ("moe.grouped.normed",),
    "dense": ("dense_mlp.normed",),
}


def _session(family):
    session = runner.Qwen35ParoResidentSession.__new__(runner.Qwen35ParoResidentSession)
    session._verify_linear_scratch_cache = {}
    session._verify_mlp_scratch_cache = {}
    session._verify_scratch_cache_generation = 0
    session._verify_mlp_scratch_policy = lambda rows: family
    reservations = []
    allocations = {}
    lookups = []

    def reserve(*, tokens, activation_dtype):
        assert activation_dtype == DType.FP16
        tensor = Tensor.from_handle(0x10000 + len(reservations) * 0x1000, (tokens, 8), activation_dtype, Device("hip", 0))
        scratch = _CLASSES[family](**{field.name: tensor for field in fields(_CLASSES[family])})
        reservations.append(scratch)
        for name in _NAMES[family]:
            allocations[name] = SimpleNamespace(tensor=tensor)
        return scratch

    def allocation(name):
        lookups.append(name)
        return allocations[name]

    state = SimpleNamespace(
        workspace=SimpleNamespace(allocation=allocation),
        reserve_linear_attention_scratch=reserve, reserve_dense_mlp_scratch=reserve,
        reserve_moe_grouped_prefill_scratch=reserve, reserve_moe_c1_scratch=reserve,
    )
    method = session._verify_linear_attention_scratch if family == "linear" else session._verify_mlp_scratch

    def get(rows):
        return method(2, state, rows=rows)

    return session, state, get, reservations, allocations, lookups


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("family", list(_CLASSES))
@pytest.mark.parametrize("rows", [2, 7, 19])
@pytest.mark.parametrize("stamp", [None, "0", "false", "1"])
def test_verifier_scratch_always_reuses_and_invalidates(monkeypatch, retired, family, rows, stamp):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_VERIFY_SCRATCH_CACHE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_CACHE", retired)
    if stamp is None:
        monkeypatch.delenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", stamp)
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(rows)
    assert get(rows) is first
    assert len(reservations) == 1
    assert lookups == []
    other_rows = get(rows + 1)
    assert other_rows is not first
    assert len(reservations) == 2
    # The wider reserve re-specified this layer's workspace names, so the
    # rows-wide object is a view of freed buffers and must not be handed back.
    # Cache hits use their generation, not a per-hit workspace lookup.
    rebuilt = get(rows)
    assert rebuilt is not first
    assert len(reservations) == 3
    assert lookups == []
    assert get(rows) is rebuilt
    session._clear_verify_scratch_caches()
    assert session._verify_scratch_cache_generation == 1
    assert session._verify_linear_scratch_cache == {}
    assert session._verify_mlp_scratch_cache == {}
    after = get(rows)
    assert after is not rebuilt and after is not other_rows
    assert get(rows) is after
    cache = session._verify_linear_scratch_cache if family == "linear" else session._verify_mlp_scratch_cache
    assert next(iter(cache.values())) == (1, after)


@pytest.mark.parametrize("family", list(_CLASSES))
@pytest.mark.parametrize("mismatch", ["missing", "pointer", "shape", "dtype", "device"])
def test_verifier_scratch_rebuilds_after_workspace_invalidation(monkeypatch, family, mismatch):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "0")
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    session._clear_verify_scratch_caches()
    name = _NAMES[family][-1]
    tensor = allocations[name].tensor
    if mismatch == "missing":
        del allocations[name]
    else:
        values = dict(ptr=tensor.ptr, shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if mismatch == "pointer":
            values["ptr"] += 0x8000
        elif mismatch == "shape":
            values["shape"] = (8, 4)
        elif mismatch == "dtype":
            values["dtype"] = DType.BF16
        if mismatch == "device":
            values["device"] = Device("hip", 1)
        allocations[name] = SimpleNamespace(tensor=Tensor.from_handle(**values))
    assert get(4) is not first
    assert len(reservations) == 2
    assert lookups == []


@pytest.mark.parametrize("family", list(_CLASSES))
@pytest.mark.parametrize("stamp", [None, "0", "false", "1"])
def test_verifier_scratch_rebuilds_on_generation_mismatch(monkeypatch, family, stamp):
    if stamp is None:
        monkeypatch.delenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", stamp)
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    session._verify_scratch_cache_generation += 1
    assert get(4) is not first
    assert len(reservations) == 2
    assert lookups == []


@pytest.mark.parametrize("family", list(_CLASSES))
def test_verifier_scratch_cache_keeps_layers_separate(monkeypatch, family):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    method = session._verify_linear_attention_scratch if family == "linear" else session._verify_mlp_scratch
    other = method(5, state, rows=4)
    assert other is not first
    assert method(5, state, rows=4) is other
    assert get(4) is first
    assert len(reservations) == 2


def test_verifier_mlp_cache_keeps_policy_keys_separate(monkeypatch):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    session, *_ = _session("c1")
    cached = {}
    for family in ("c1", "grouped", "dense"):
        _, state, *_ = _session(family)
        session._verify_mlp_scratch_policy = lambda rows, policy=family: policy
        cached[family] = session._verify_mlp_scratch(2, state, rows=4)
    assert set(session._verify_mlp_scratch_cache) == {
        (2, 4, "c1"), (2, 4, "grouped"), (2, 4, "dense"),
    }
    for family, scratch in cached.items():
        session._verify_mlp_scratch_policy = lambda rows, policy=family: policy
        assert session._verify_mlp_scratch(2, state, rows=4) is scratch


@pytest.mark.parametrize("family", list(_CLASSES))
@pytest.mark.parametrize("mismatch", ["class", "rows", "dtype"])
def test_verifier_scratch_stamp_still_checks_cached_layout(monkeypatch, family, mismatch):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "false")
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    if mismatch == "class":
        bad = SimpleNamespace()
    else:
        name = "attn_input" if family == "linear" else "normed"
        tensor = getattr(first, name)
        bad_tensor = Tensor.from_handle(
            tensor.ptr,
            (8, 4) if mismatch == "rows" else tensor.shape,
            DType.BF16 if mismatch == "dtype" else tensor.dtype,
            tensor.device,
        )
        bad = replace(first, **{name: bad_tensor})
    cache = session._verify_linear_scratch_cache if family == "linear" else session._verify_mlp_scratch_cache
    key = next(iter(cache))
    cache[key] = (session._verify_scratch_cache_generation, bad)
    assert get(4) is not first
    assert len(reservations) == 2
    assert lookups == []


@pytest.mark.parametrize("boundary", ["prefill_restore", "graph_invalidation"])
def test_verifier_scratch_owner_boundaries_advance_generation(monkeypatch, boundary):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "0")
    session, state, get, reservations, allocations, lookups = _session("c1")
    first = get(4)
    if boundary == "prefill_restore":
        session.config = SimpleNamespace(num_experts=128, layer_types=("linear_attention",))
        session.states = [state]
        session.moe_scratch = {}
        session.linear_scratch = {}
        releases = []
        session._release_prefill_workspace = lambda: releases.append("workspace")
        session._release_prefill_hidden_buffer = lambda: releases.append("hidden")
        session._restore_decode_scratch_after_prefill()
        assert releases == ["workspace", "hidden"]
        assert session.moe_scratch[0].normed.shape[0] == 1
    else:
        session._invalidate_verify_graph_cache()
    assert session._verify_scratch_cache_generation == 1
    assert not session._verify_mlp_scratch_cache
    assert not session._verify_linear_scratch_cache
    assert get(4) is not first
    assert lookups == []


# --- pointer liveness against a real RuntimeWorkspace ---------------------
#
# ``RuntimeWorkspace.reserve_tensor`` frees the previous buffer whenever a name
# is re-reserved with a different shape, so a verifier scratch object cached at
# an earlier width is a view of released memory. These tests drive the real
# workspace (real name table, real free-before-reallocate lifetime) against a
# tracked allocator, so a cache that keeps a dead view fails here instead of on
# a GPU.

_FAMILIES = ("linear", "c1", "grouped", "dense")


class _TrackedAllocator:
    """Stand-in for the workspace module's ``malloc``/``free``.

    Addresses are never reused, so returning to an earlier width always
    produces a different pointer and a stale view can never be mistaken for a
    live re-allocation.
    """

    def __init__(self, *, fail_on_malloc: int | None = None) -> None:
        self.live: dict[int, int] = {}
        self.freed: list[int] = []
        self.malloc_count = 0
        self.free_count = 0
        # ``fail_on_malloc`` is the successful-malloc count at which the next
        # allocation raises, which models a device OOM part-way through one
        # named reservation.
        self.fail_on_malloc = fail_on_malloc
        self._next_address = 0x100000

    def malloc(self, nbytes: int, *, runtime=None) -> DeviceBuffer:
        _ = runtime
        if self.fail_on_malloc is not None and self.malloc_count >= self.fail_on_malloc:
            raise MemoryError("tracked allocator: device allocation failure")
        size = int(nbytes)
        address = self._next_address
        self._next_address += max(size, 1) + 0x1000
        self.live[address] = size
        self.malloc_count += 1
        return DeviceBuffer(address, size)

    def free(self, buffer: DeviceBuffer, *, runtime=None) -> None:
        _ = runtime
        address = int(buffer.ptr)
        assert address in self.live, f"double free of 0x{address:x}"
        del self.live[address]
        self.freed.append(address)
        self.free_count += 1


def _install_allocator(monkeypatch, allocator: _TrackedAllocator) -> None:
    monkeypatch.setattr(workspace_module, "malloc", allocator.malloc)
    monkeypatch.setattr(workspace_module, "free", allocator.free)


def _config(*, num_experts: int = 8):
    return SimpleNamespace(
        hidden_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=6,
    )


def _family_config(family: str):
    return _config(num_experts=0 if family == "dense" else 8)


def _state(config, *, layer_id: int = 0):
    layer_weights = SimpleNamespace(
        layer_id=int(layer_id),
        config=config,
        weights=SimpleNamespace(tensors={}),
    )
    return runner.Qwen35ParoDecodeState(
        layer_weights=layer_weights,
        workspace=RuntimeWorkspace(runtime=None),
    )


def _liveness_session(config):
    session = runner.Qwen35ParoResidentSession.__new__(runner.Qwen35ParoResidentSession)
    session.config = config
    session._verify_scratch_cache_generation = 0
    session._verify_linear_scratch_cache = {}
    session._verify_mlp_scratch_cache = {}
    return session


def _reserve(session, state, family: str, rows: int):
    layer_id = int(state.layer_weights.layer_id)
    if family == "linear":
        return session._verify_linear_attention_scratch(layer_id, state, rows=rows)
    return session._verify_mlp_scratch(layer_id, state, rows=rows)


def _attach_prefill_workspace(session):
    """Give ``session`` a prefill-owned workspace, as production setup does."""

    session.prefill_workspace = RuntimeWorkspace(runtime=None)
    session._prefill_scratch_state = None
    session.prefill_full_scratch = None
    if getattr(session, "runtime", None) is None:
        session.runtime = SimpleNamespace(device_synchronize=lambda: None)
    session.backend = SimpleNamespace()
    return session


def _scratch_pointers(scratch) -> set[int]:
    pointers: set[int] = set()
    for field in fields(scratch):
        pointer = getattr(getattr(scratch, field.name), "ptr", None)
        if isinstance(pointer, int):
            pointers.add(pointer)
    return pointers


def _cached_pointers(session) -> dict[str, set[int]]:
    cached: dict[str, set[int]] = {}
    for key, (_generation, scratch) in session._verify_linear_scratch_cache.items():
        cached[f"linear{key}"] = _scratch_pointers(scratch)
    for key, (_generation, scratch) in session._verify_mlp_scratch_cache.items():
        cached[f"mlp{key}"] = _scratch_pointers(scratch)
    return cached


def _points_into_live_allocation(allocator: _TrackedAllocator, pointer: int) -> bool:
    """True when ``pointer`` addresses a live allocation, interior views included.

    Split views (``qkv``/``z``, ``a``/``b``) are offsets inside one workspace
    allocation, so base-address equality would report them as dead.
    """

    return any(base <= pointer < base + nbytes for base, nbytes in allocator.live.items())


def _assert_cached_views_are_live(allocator: _TrackedAllocator, session) -> None:
    for name, pointers in _cached_pointers(session).items():
        dead = sorted(
            pointer for pointer in pointers if not _points_into_live_allocation(allocator, pointer)
        )
        assert not dead, f"{name} caches views of freed buffers {[hex(p) for p in dead]}"


def _assert_workspace_holds_one_copy(allocator: _TrackedAllocator, state) -> None:
    """No per-width duplication: one live buffer per workspace name."""

    assert len(allocator.live) == len(state.workspace.names)
    assert allocator.malloc_count - allocator.free_count == len(state.workspace.names)


def _first_width(family: str) -> int:
    return 16 if family == "grouped" else 2


@pytest.mark.parametrize("family", _FAMILIES)
def test_verify_scratch_width_cycle_keeps_only_live_views(monkeypatch, family):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config(family)
    state = _state(config)
    session = _liveness_session(config)
    base = _first_width(family)
    if family != "linear":
        assert session._verify_mlp_scratch_policy(base) == family

    scratch = {rows: _reserve(session, state, family, rows) for rows in (base, base + 1, base + 2)}
    _assert_cached_views_are_live(allocator, session)
    mallocs = allocator.malloc_count
    assert _reserve(session, state, family, base + 2) is scratch[base + 2]
    assert allocator.malloc_count == mallocs

    # Widths base -> base+1 -> base+2 -> base: the base-wide buffers were freed
    # by the base+1 reserve, so the cached base-wide object cannot come back.
    assert _scratch_pointers(scratch[base]) & set(allocator.freed)
    rebuilt = _reserve(session, state, family, base)
    assert rebuilt is not scratch[base]
    _assert_cached_views_are_live(allocator, session)
    assert _reserve(session, state, family, base) is rebuilt
    _assert_workspace_holds_one_copy(allocator, state)

    # An unrelated width invalidates the same way.
    unrelated = base + 7
    wider = _reserve(session, state, family, unrelated)
    _assert_cached_views_are_live(allocator, session)
    after_unrelated = _reserve(session, state, family, base)
    assert after_unrelated is not rebuilt
    _assert_cached_views_are_live(allocator, session)
    # The unrelated width was evicted too, so coming back to it re-reserves.
    assert _reserve(session, state, family, unrelated) is not wider
    _assert_workspace_holds_one_copy(allocator, state)


@pytest.mark.parametrize("family", _FAMILIES)
def test_verify_scratch_eviction_is_scoped_to_one_layer(monkeypatch, family):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config(family)
    session = _liveness_session(config)
    base = _first_width(family)
    first_layer = _state(config, layer_id=0)
    second_layer = _state(config, layer_id=1)

    kept = _reserve(session, first_layer, family, base)
    _reserve(session, second_layer, family, base + 1)

    # Another layer owns another workspace, so its reserve freed nothing here.
    assert _reserve(session, first_layer, family, base) is kept
    _assert_cached_views_are_live(allocator, session)


def test_verify_scratch_eviction_is_scoped_to_one_family(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)

    linear = _reserve(session, state, "linear", 2)
    moe = _reserve(session, state, "c1", 2)

    # A linear reserve re-specifies linear_attn.* names only.
    _reserve(session, state, "linear", 3)
    assert _reserve(session, state, "c1", 2) is moe
    assert _reserve(session, state, "linear", 2) is not linear
    _assert_cached_views_are_live(allocator, session)


def test_verify_scratch_policy_transition_keeps_the_other_policy_live(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    assert session._verify_mlp_scratch_policy(4) == "c1"
    assert session._verify_mlp_scratch_policy(16) == "grouped"

    c1 = _reserve(session, state, "c1", 4)
    grouped = _reserve(session, state, "grouped", 16)

    # c1 owns moe.* and grouped owns moe.grouped.*, so neither reserve freed
    # the other's views.
    assert _reserve(session, state, "c1", 4) is c1
    assert _reserve(session, state, "grouped", 16) is grouped
    # A second c1 width frees moe.* only.
    _reserve(session, state, "c1", 5)
    assert _reserve(session, state, "grouped", 16) is grouped
    _assert_cached_views_are_live(allocator, session)


def test_verify_scratch_width_cycle_forces_a_verifier_graph_recapture(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    session.linear_scratch = {}
    session.moe_scratch = {}

    def reserve(rows: int) -> None:
        session.linear_scratch[0] = session._verify_linear_attention_scratch(0, state, rows=rows)
        session.moe_scratch[0] = session._verify_mlp_scratch(0, state, rows=rows)

    def binding(rows: int) -> SpecGraphBinding:
        return SpecGraphBinding.build(
            bucket=(rows,),
            addresses=session._verify_graph_addresses(),
            configuration=(),
        )

    reserve(2)
    captured = binding(2)
    reserve(3)
    reserve(4)
    reserve(2)

    # The live maps the fingerprint walks must only ever hold current
    # allocations, so a graph captured at an earlier width cannot replay into
    # freed memory: its addresses are dead and the binding drifts.
    _assert_cached_views_are_live(allocator, session)
    live = binding(2)
    assert all(
        _points_into_live_allocation(allocator, pointer) for _name, pointer in live.addresses
    )
    # The width-2 capture baked row-count-dependent addresses that the wider
    # reserves released (the row-count-independent barrier survives), so the
    # stale entry can no longer be replayed.
    released = [
        pointer
        for _name, pointer in captured.addresses
        if not _points_into_live_allocation(allocator, pointer)
    ]
    assert released
    assert captured.addresses != live.addresses
    assert captured.ineligibility_reason(live) == GRAPH_STAGE_BINDING_MISS


# --- reserve failure ------------------------------------------------------
#
# ``RuntimeWorkspace.reserve_tensor`` frees a name's old buffer before it
# allocates the replacement, so a reservation that raises part-way has already
# released buffers the cached views point at. Eviction therefore has to run
# before the reserve, not after it returns.


def _family_cache(session, family: str) -> dict:
    if family == "linear":
        return session._verify_linear_scratch_cache
    return session._verify_mlp_scratch_cache


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("fail_after", [0, 2])
def test_verify_scratch_failed_reserve_drops_the_old_width_entry(monkeypatch, family, fail_after):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config(family)
    state = _state(config)
    session = _liveness_session(config)
    base = _first_width(family)

    old = _reserve(session, state, family, base)
    frees_before = allocator.free_count
    # ``fail_after`` replacements succeed before the allocation that raises, so
    # the failed reservation frees one more old buffer than it replaced.
    allocator.fail_on_malloc = allocator.malloc_count + fail_after
    with pytest.raises(MemoryError):
        _reserve(session, state, family, base + 1)
    allocator.fail_on_malloc = None

    # The failed reservation released old buffers before it raised...
    assert allocator.free_count == frees_before + fail_after + 1
    # ...so the old width's entry must be gone, not merely unreachable.
    assert _family_cache(session, family) == {}
    _assert_cached_views_are_live(allocator, session)

    # A later lookup at the old width must reserve again instead of handing the
    # pre-failure object back, and what it rebuilds is live: the workspace drops
    # the record of the name whose replacement allocation raised, so nothing
    # still points at the buffer the failed reservation released.
    rebuilt = _reserve(session, state, family, base)
    assert rebuilt is not old
    _assert_cached_views_are_live(allocator, session)
    _assert_workspace_holds_one_copy(allocator, state)
    assert _reserve(session, state, family, base) is rebuilt


@pytest.mark.parametrize("family", _FAMILIES)
def test_verify_scratch_failed_reserve_keeps_unrelated_layer_entries(monkeypatch, family):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config(family)
    session = _liveness_session(config)
    base = _first_width(family)
    failing_layer = _state(config, layer_id=0)
    other_layer = _state(config, layer_id=1)

    _reserve(session, failing_layer, family, base)
    kept = _reserve(session, other_layer, family, base)
    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        _reserve(session, failing_layer, family, base + 1)
    allocator.fail_on_malloc = None

    # The other layer owns another workspace: the failure cannot touch it, and
    # the fail-closed path must not clear its cache entry or its generation.
    mallocs = allocator.malloc_count
    assert _reserve(session, other_layer, family, base) is kept
    assert allocator.malloc_count == mallocs
    _assert_cached_views_are_live(allocator, session)
    expected_key = (1, base) if family == "linear" else (1, base, family)
    assert set(_family_cache(session, family)) == {expected_key}


def test_verify_scratch_failed_reserve_keeps_unrelated_policy_entries(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)

    _reserve(session, state, "c1", 4)
    grouped = _reserve(session, state, "grouped", 16)
    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        _reserve(session, state, "c1", 5)
    allocator.fail_on_malloc = None

    # A failed c1 reservation freed moe.* names only; the grouped entry is live
    # and must still be a cache hit.
    assert _reserve(session, state, "grouped", 16) is grouped
    _assert_cached_views_are_live(allocator, session)
    assert set(session._verify_mlp_scratch_cache) == {(0, 16, "grouped")}


def test_verify_scratch_failed_reserve_retires_verifier_graphs(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    synced: list[str] = []
    session.runtime = SimpleNamespace(device_synchronize=lambda: synced.append("sync"))
    session._verify_graphs = SpecGraphStageCache(max_entries=4)
    session._verify_graph_partials = []
    session._verify_graph_partial_buffers = []
    session._verify_graph_last_status = "unobserved"
    session._verify_graph_last_error = None
    session.linear_scratch = {}
    session.moe_scratch = {}

    def reserve(rows: int) -> None:
        session.linear_scratch[0] = session._verify_linear_attention_scratch(0, state, rows=rows)
        session.moe_scratch[0] = session._verify_mlp_scratch(0, state, rows=rows)

    reserve(2)
    key = ("verify", 2)
    binding = SpecGraphBinding.build(
        bucket=(2,),
        addresses=session._verify_graph_addresses(),
        configuration=(),
    )
    released: list[str] = []
    graph, reason = session._verify_graphs.capture(key, binding, lambda: "graph", released.append)
    assert (graph, reason) == ("graph", None)
    generation = session._verify_scratch_cache_generation

    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        _reserve(session, state, "linear", 3)
    allocator.fail_on_malloc = None

    # The live maps still report the pre-failure addresses, so the fingerprint
    # can no longer vouch for a capture: the graphs are released before the
    # failure leaves the cycle. Live scratch entries keep their generation.
    assert synced == ["sync"]
    assert key not in session._verify_graphs
    assert released == ["graph"]
    assert session._verify_scratch_cache_generation == generation
    assert session._verify_graph_last_status == "invalidated"


@pytest.mark.parametrize("family", _FAMILIES)
def test_decode_batch_growth_after_failed_reserve_returns_live_pointers(monkeypatch, family):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config(family)
    state = _state(config)
    session = _liveness_session(config)
    session.states = [state]
    session.linear_scratch = {}
    session.moe_scratch = {}
    base = _first_width(family)

    # The verifier publishes its scratch in the live maps, which is what a
    # captured graph and the decode path both read.
    session.linear_scratch[0] = session._verify_linear_attention_scratch(0, state, rows=base)
    session.moe_scratch[0] = session._verify_mlp_scratch(0, state, rows=base)
    dropped_linear = session.linear_scratch[0]
    dropped_moe = session.moe_scratch[0]

    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        _reserve(session, state, family, base + 1)
    allocator.fail_on_malloc = None

    if family == "linear":
        dropped_map, dropped = session.linear_scratch, dropped_linear
        # The MoE family reserves different names, so its entry stays valid.
        assert session.moe_scratch[0] is dropped_moe
    else:
        dropped_map, dropped = session.moe_scratch, dropped_moe
        assert session.linear_scratch[0] is dropped_linear
    # The failed family's resident entry is gone: a direct consumer fails before
    # any kernel can dereference the released buffers.
    assert 0 not in dropped_map
    with pytest.raises(KeyError):
        dropped_map[0]
    _assert_cached_views_are_live(allocator, session)

    # Public decode growth at the dropped width recovers by reserving.
    if family == "linear":
        grown = session._ensure_linear_decode_batch_scratch(0, base)
    elif family == "c1":
        grown = session._ensure_moe_decode_batch_scratch(0, base, force_selected_c1_moe=True)
    else:
        grown = session._ensure_moe_decode_batch_scratch(0, base)
    assert grown is not dropped
    assert all(
        _points_into_live_allocation(allocator, pointer) for pointer in _scratch_pointers(grown)
    )
    _assert_workspace_holds_one_copy(allocator, state)
    _assert_cached_views_are_live(allocator, session)


def test_failed_family_reserve_keeps_a_resident_entry_of_another_policy(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    session.linear_scratch = {}
    session.moe_scratch = {}

    grouped = _reserve(session, state, "grouped", 16)
    session.moe_scratch[0] = grouped
    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        _reserve(session, state, "c1", 2)
    allocator.fail_on_malloc = None

    # A failed c1 reserve freed moe.* names only: the resident grouped entry is
    # another family and is still live, so it is not dropped.
    assert session.moe_scratch[0] is grouped
    _assert_cached_views_are_live(allocator, session)
    assert _reserve(session, state, "grouped", 16) is grouped


def test_canonicalize_decode_scratch_recovers_after_failed_family_reserve(monkeypatch):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    config.layer_types = ("linear_attention",)
    state = _state(config)
    session = _liveness_session(config)
    session.states = [state]
    session.linear_scratch = {}
    session.moe_scratch = {}

    session.linear_scratch[0] = session._verify_linear_attention_scratch(0, state, rows=2)
    session.moe_scratch[0] = session._verify_mlp_scratch(0, state, rows=2)
    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        _reserve(session, state, "linear", 3)
    allocator.fail_on_malloc = None
    assert 0 not in session.linear_scratch

    # Canonicalization already rebuilds a missing entry with ``.get`` + reserve;
    # the fail-closed drop leans on that, so it must still produce c=1 views.
    session._canonicalize_decode_scratch()

    assert session.linear_scratch[0].attn_input.shape[0] == 1
    assert session.moe_scratch[0].normed.shape[0] == 1
    for scratch in (session.linear_scratch[0], session.moe_scratch[0]):
        assert all(
            _points_into_live_allocation(allocator, pointer)
            for pointer in _scratch_pointers(scratch)
        )
    _assert_workspace_holds_one_copy(allocator, state)


# --- full-attention families ----------------------------------------------
#
# ``_ensure_full_decode_batch_scratch`` and ``_ensure_full_prefill_scratch``
# publish a multi-name ``reserve_full_attention_scratch`` result in a resident
# slot. A failure part-way through frees some names, so the old object must be
# dropped rather than returned when the old width is requested again.


@pytest.mark.parametrize("fail_after", [0, 2])
def test_full_decode_batch_growth_after_failed_reserve_returns_live_pointers(monkeypatch, fail_after):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    session.states = [state]
    session.max_splits = 1
    session.full_scratch = {}

    resident = state.reserve_full_attention_scratch(
        tokens=2,
        num_splits=1,
        activation_dtype=DType.FP16,
        gated_dtype=DType.FP16,
    )
    session.full_scratch[0] = resident
    frees_before = allocator.free_count
    allocator.fail_on_malloc = allocator.malloc_count + fail_after
    with pytest.raises(MemoryError):
        session._ensure_full_decode_batch_scratch(0, 3)
    allocator.fail_on_malloc = None

    # The failed reservation released old attention buffers before it raised.
    assert allocator.free_count == frees_before + fail_after + 1
    # The half-replaced object must not stay reachable.
    assert 0 not in session.full_scratch
    with pytest.raises(KeyError):
        session.full_scratch[0]

    # Retrying the old width re-reserves, and every pointer it returns is live.
    rebuilt = session._ensure_full_decode_batch_scratch(0, 2)
    assert rebuilt is not resident
    assert all(
        _points_into_live_allocation(allocator, pointer) for pointer in _scratch_pointers(rebuilt)
    )
    _assert_workspace_holds_one_copy(allocator, state)
    # Same-width reuse still holds after the recovery.
    mallocs = allocator.malloc_count
    assert session._ensure_full_decode_batch_scratch(0, 2) is rebuilt
    assert allocator.malloc_count == mallocs


@pytest.mark.parametrize("fail_after", [0, 2])
def test_full_prefill_scratch_after_failed_reserve_returns_live_pointers(monkeypatch, fail_after):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    session.states = [state]
    _attach_prefill_workspace(session)

    resident = session._ensure_full_prefill_scratch(tokens=2)
    assert session.prefill_full_scratch is resident
    frees_before = allocator.free_count
    allocator.fail_on_malloc = allocator.malloc_count + fail_after
    with pytest.raises(MemoryError):
        session._ensure_full_prefill_scratch(tokens=3)
    allocator.fail_on_malloc = None

    assert allocator.free_count == frees_before + fail_after + 1
    assert session.prefill_full_scratch is None

    rebuilt = session._ensure_full_prefill_scratch(tokens=2)
    assert rebuilt is not resident
    assert all(
        _points_into_live_allocation(allocator, pointer) for pointer in _scratch_pointers(rebuilt)
    )
    _assert_workspace_holds_one_copy(allocator, session._prefill_scratch_state)
    # The query dtype stays part of the layout identity, before and after.
    assert rebuilt.query.dtype == DType.FP32
    assert session._ensure_full_prefill_scratch(tokens=2, aotriton_attention=True).query.dtype == DType.BF16


@pytest.mark.parametrize("helper", ["decode_batch", "prefill"])
def test_full_scratch_failed_reserve_retires_verifier_graphs(monkeypatch, helper):
    allocator = _TrackedAllocator()
    _install_allocator(monkeypatch, allocator)
    config = _family_config("c1")
    state = _state(config)
    session = _liveness_session(config)
    synced: list[str] = []
    session.runtime = SimpleNamespace(device_synchronize=lambda: synced.append("sync"))
    session._verify_graphs = SpecGraphStageCache(max_entries=4)
    session._verify_graph_partials = []
    session._verify_graph_partial_buffers = []
    session._verify_graph_last_status = "unobserved"
    session._verify_graph_last_error = None
    session.states = [state]
    session.max_splits = 1
    session.full_scratch = {}
    session.linear_scratch = {}
    session.moe_scratch = {}
    _attach_prefill_workspace(session)

    if helper == "decode_batch":
        session._ensure_full_decode_batch_scratch(0, 2)

        def grow() -> None:
            session._ensure_full_decode_batch_scratch(0, 3)
    else:
        session._ensure_full_prefill_scratch(tokens=2)

        def grow() -> None:
            session._ensure_full_prefill_scratch(tokens=3)

    key = ("verify", 2)
    binding = SpecGraphBinding.build(
        bucket=(2,),
        addresses=session._verify_graph_addresses(),
        configuration=(),
    )
    released: list[str] = []
    graph, reason = session._verify_graphs.capture(key, binding, lambda: "graph", released.append)
    assert (graph, reason) == ("graph", None)
    generation = session._verify_scratch_cache_generation

    allocator.fail_on_malloc = allocator.malloc_count + 1
    with pytest.raises(MemoryError):
        grow()
    allocator.fail_on_malloc = None

    # An attention-name failure leaves the linear/MoE caches live, but the
    # fingerprint can no longer vouch for a capture: graphs go, caches stay.
    assert synced == ["sync"]
    assert key not in session._verify_graphs
    assert released == ["graph"]
    assert session._verify_graph_last_status == "invalidated"
    assert session._verify_scratch_cache_generation == generation
