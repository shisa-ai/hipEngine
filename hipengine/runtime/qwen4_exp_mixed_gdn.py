"""Depth-wave GDN and Conv state adapter for one mixed target row layout.

:mod:`hipengine.runtime.qwen4_exp_mixed_layout` names one shared target pass:
each request's root row, its prepared candidate rows, the physical request slot
that slot owns, and the physical position of every row.
:mod:`hipengine.runtime.qwen4_exp_batch_gdn` runs independent requests through
the request-indexed GDN and Conv owners, one dispatch per stage. Both primitives
require the state slots named in one dispatch to be **distinct**: row ``r`` reads
and writes slot ``indices[r]`` in place, so two rows naming one slot would
interleave their shifts and commit a state that is neither row's result.

A prepared chain therefore cannot be handed to those primitives as independent
rows: depth 1 extends depth 0's state inside the same request. This module is the
adapter between the two contracts. It consumes one mixed layout plus one set of
shared projected rows and runs the same two indexed primitives in **depth
waves**. Wave ``d`` carries each request's row at depth ``d``, at most one row per
request, so every slot in a dispatch is unique. Row ``(request, d)`` reads the
snapshot its parent row ``(request, d-1)`` produced, which is a different slot,
so the in-place update is the chain's sequential recurrence.

Provisional state and snapshots
-------------------------------

The working slab is one :class:`~hipengine.runtime.qwen4_exp_batch_gdn.Qwen4ExpBatchGdnState`
with one slot per **physical row** of the layout, so provisional slot ``p`` *is*
physical row ``p``. Before the first wave every request's root slot is copied
from its live slot in the caller's state owner; before wave ``d >= 1`` each row's
slot is copied from its parent row's slot. Nothing in the pass writes to the live
owner, so verification cannot mutate the state it is deciding about, and a
repeated pass starts from the same live bytes.

The slab is therefore also the snapshot store: physical row ``p`` holds the
recurrent and Conv state after consuming the rows of its chain up to ``p``'s
depth. :meth:`Qwen4ExpMixedGdnAdapter.commit` turns accepted-prefix counts into
the destination rows the layout already validates
(:meth:`~hipengine.runtime.qwen4_exp_mixed_layout.Qwen4ExpMixedLayout.commit_destinations`)
and copies exactly those two snapshot blocks per request into the live slot.

Admission epoch
---------------

A layout names the owner token each request was admitted under, because a
physical slot's live state belongs to one admission: the batch owner releases a
slot and later hands the same index to another request. A layout compiled before
that turnover still names the old token, and its rows would then read and commit
somebody else's state. Passing ``owner_of`` to
:meth:`Qwen4ExpMixedGdnAdapter.allocate` makes the adapter compare every request's
layout token against the token the slot holds now, before ``run`` copies a root
snapshot and before ``commit`` writes the live owner. Without the callback the
adapter trusts the caller's slot ownership; it cannot detect stale reuse.

One pass, one commit
--------------------

A verification pass is one transaction. ``run`` publishes
:attr:`Qwen4ExpMixedGdnAdapter.verification_pending` only after the last wave, and
``commit`` consumes it *before* its first copy. A commit that fails partway
through has already written some requests' live state, so the transaction is no
longer replayable: the next ``commit`` refuses until a fresh ``run`` rebuilds
every snapshot from the live owner. Validation failures (a rejected accepted
count, a stale owner token) raise before the flag is cleared, so a corrected
call is still the same transaction.

What this adapter does and does not do
--------------------------------------

It runs the recurrence, not the pass. The input projections (``attn_qkv``,
``attn_gate``, ``ssm_alpha``, ``ssm_beta``) are the caller's: it projects the
whole physical row set once, shares that result across every row, and hands the
four row buffers here. This module does not project, does not gather attention,
PLE or QSA state, and does not wire a runner. It closes the state-slot collision
between one chain and the indexed primitives; it does not make a mixed pass
shared, and nothing here claims that it does.

The gated recurrence output stays in physical row order in
:attr:`Qwen4ExpMixedGdnAdapter.core`, so the caller runs the stateless ``ssm_out``
projection over all rows at once, exactly as
:func:`~hipengine.runtime.qwen4_exp_batch_gdn.run_qwen4_exp_batch_gdn_output_projection`
does for an independent row set.

Arithmetic
----------

Both indexed owners run their per-row body per block, so a row's output and final
state are the same bytes as
:func:`~hipengine.runtime.qwen4_exp_batch_gdn.run_qwen4_exp_batch_gdn_token_mixer`
produces for the same row inputs and the same input state, whatever the wave
width. This module adds no numerical variant of its own: it reuses
:func:`~hipengine.kernels.hip_gfx1100.linear_attn.conv.qwen4_exp_linear_attn_conv_decode_indexed_f32`
and
:func:`~hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn.qwen4_exp_gdn_decode_rowsplit_indexed_f32`
unchanged.

Cost of the first coherent form
-------------------------------

Gather, scatter and snapshot copies are synchronous device-to-device copies, one
per row per operand: four operands into the wave, one result out, and two state
blocks per row. A pass over ``total_rows`` rows therefore issues about
``7 * total_rows`` copies and two launches per wave, and the state copies
dominate the bytes: ``2 * total_rows`` blocks of
``num_v_heads * 128 * 128 * 4`` bytes each, which for a four-deep chain at four
value heads is one mebibyte per request. That is the price of keeping every
accepted-prefix snapshot addressable. The buffer mapping
(:attr:`Qwen4ExpMixedGdnAdapter.core`, :attr:`~Qwen4ExpMixedGdnAdapter.wave`,
:attr:`~Qwen4ExpMixedGdnAdapter.working_state`, and the ``*_row_ptr`` /
``*_snapshot_ptr`` accessors) is what a later runner needs to replace those
copies with strided operands or a wave-shaped projection.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    free,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS,
    QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE,
    QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_conv_indexed_state_indices,
    qwen4_exp_linear_attn_conv_decode_indexed_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_gdn_decode_rowsplit_indexed_f32,
    qwen4_exp_gdn_indexed_state_indices,
)
from hipengine.runtime.qwen4_exp_batch_gdn import Qwen4ExpBatchGdnState
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedLayout
from hipengine.speculative.interfaces import TargetCommitPlan

__all__ = [
    "Qwen4ExpMixedGdnAdapter",
    "Qwen4ExpMixedGdnGeometry",
    "Qwen4ExpMixedGdnPlan",
    "Qwen4ExpMixedGdnProjectedRows",
    "Qwen4ExpMixedGdnWave",
    "Qwen4ExpMixedGdnWaveScratch",
    "plan_qwen4_exp_mixed_gdn",
]

# ``qwen4_exp_gdn_decode_rowsplit_indexed_f32`` is compiled for this head geometry
# only, and rejects anything else on the host before it launches.
_INDEXED_HEAD_DIM = 128
# The recurrence loads Conv rows as float4 and takes the Conv row stride in
# floats, so every row base it reads has to stay 16-byte aligned.
_FLOAT4_BYTES = 16
_WAVE_BUFFER_NAMES = ("qkv", "gate", "alpha", "beta", "conv", "core")


@dataclass(frozen=True)
class Qwen4ExpMixedGdnGeometry:
    """The GDN/Conv geometry one layer's state owner and this adapter share.

    Taken from the live state owner rather than restated by the caller, so the
    adapter cannot verify a chain against a different head geometry than the
    state it commits into.
    """

    num_k_heads: int
    num_v_heads: int
    head_dim: int
    conv_kernel: int

    def __post_init__(self) -> None:
        k_heads = _positive_int(self.num_k_heads, "num_k_heads")
        v_heads = _positive_int(self.num_v_heads, "num_v_heads")
        head_dim = _positive_int(self.head_dim, "head_dim")
        kernel = _positive_int(self.conv_kernel, "conv_kernel")
        object.__setattr__(self, "num_k_heads", k_heads)
        object.__setattr__(self, "num_v_heads", v_heads)
        object.__setattr__(self, "head_dim", head_dim)
        object.__setattr__(self, "conv_kernel", kernel)
        if v_heads % k_heads:
            raise ValueError("num_v_heads must be divisible by num_k_heads")
        if head_dim != _INDEXED_HEAD_DIM:
            raise ValueError(
                f"the indexed GDN decode owner requires head_dim "
                f"{_INDEXED_HEAD_DIM}, got {head_dim}"
            )
        if kernel > QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE:
            raise ValueError(
                f"conv_kernel {kernel} exceeds the indexed conv kernel bound "
                f"{QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE}"
            )
        qkv_width = self.qkv_width
        if qkv_width > QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS:
            raise ValueError(
                f"qkv_width {qkv_width} exceeds the indexed conv channel bound "
                f"{QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS}"
            )
        if qkv_width % 4:
            raise ValueError(
                "qkv_width must be a multiple of 4 floats so every Conv row base "
                "stays 16-byte aligned for the float4 Q/K loads"
            )

    @classmethod
    def from_state(cls, state: Qwen4ExpBatchGdnState) -> "Qwen4ExpMixedGdnGeometry":
        """Read the geometry of the state owner this adapter commits into."""

        if not isinstance(state, Qwen4ExpBatchGdnState):
            raise TypeError("state must be a Qwen4ExpBatchGdnState")
        return cls(
            num_k_heads=state.num_k_heads,
            num_v_heads=state.num_v_heads,
            head_dim=state.head_dim,
            conv_kernel=state.conv_kernel,
        )

    @property
    def qkv_width(self) -> int:
        return 2 * self.num_k_heads * self.head_dim + self.num_v_heads * self.head_dim

    @property
    def core_width(self) -> int:
        return self.num_v_heads * self.head_dim

    @property
    def scalar_width(self) -> int:
        return self.num_v_heads


@dataclass(frozen=True)
class Qwen4ExpMixedGdnWave:
    """One depth wave: at most one row per request, every slot distinct.

    ``rows`` is ascending physical row order, so the wave is a function of the
    layout's physical positions and not of the order requests were declared in.
    ``parent_rows[i]`` is the physical row whose snapshot feeds ``rows[i]``, and
    is ``-1`` at depth ``0``: a root row's input state comes from its request's
    live slot, not from another row of this pass.
    """

    depth: int
    request_ids: tuple[int, ...]
    rows: tuple[int, ...]
    parent_rows: tuple[int, ...]
    slots: tuple[int, ...]

    def __post_init__(self) -> None:
        depth = _non_negative_int(self.depth, "depth")
        object.__setattr__(self, "depth", depth)
        fields = {
            "request_ids": tuple(
                _non_negative_int(value, "request_id") for value in self.request_ids
            ),
            "rows": tuple(_non_negative_int(value, "physical row") for value in self.rows),
            "parent_rows": tuple(
                _int_index(value, "parent row") for value in self.parent_rows
            ),
            "slots": tuple(_non_negative_int(value, "slot") for value in self.slots),
        }
        for name, values in fields.items():
            object.__setattr__(self, name, values)
        widths = {len(values) for values in fields.values()}
        if len(widths) != 1:
            raise ValueError("a wave must hold one entry per row in every field")
        if not self.rows:
            raise ValueError("a wave requires at least one row")
        if len(set(self.rows)) != len(self.rows):
            raise ValueError(f"wave {depth} names a physical row twice")
        if len(set(self.slots)) != len(self.slots):
            raise ValueError(
                f"wave {depth} names a state slot twice; every row in one "
                "dispatch must own a distinct slot"
            )
        if list(self.rows) != sorted(self.rows):
            raise ValueError(f"wave {depth} rows must be in ascending physical order")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError(f"wave {depth} names a request twice")
        for parent in self.parent_rows:
            if depth == 0 and parent != -1:
                raise ValueError(
                    f"wave 0 rows are roots; their state comes from the live "
                    f"slot, not from physical row {parent}"
                )
            if depth and parent < 0:
                raise ValueError(
                    f"wave {depth} rows extend a parent row and must name one"
                )

    @property
    def width(self) -> int:
        """Rows in this dispatch."""

        return len(self.rows)

    def physical_row(self, request_id: int) -> int:
        for request, row in zip(self.request_ids, self.rows, strict=True):
            if request == request_id:
                return row
        raise KeyError(
            f"request {request_id} contributes no row at depth {self.depth}"
        )


@dataclass(frozen=True)
class Qwen4ExpMixedGdnPlan:
    """The depth-wave decomposition of one mixed layout, validated on the host.

    ``waves[d]`` carries every request whose chain reaches depth ``d``, so the
    waves partition the layout's physical rows exactly once each.
    ``live_state_slots`` is the slot width of the state owner the roots are
    copied from and the commit writes into.
    """

    layout: Qwen4ExpMixedLayout
    geometry: Qwen4ExpMixedGdnGeometry
    live_state_slots: int
    waves: tuple[Qwen4ExpMixedGdnWave, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, Qwen4ExpMixedLayout):
            raise TypeError("plan layout must be a Qwen4ExpMixedLayout")
        if not isinstance(self.geometry, Qwen4ExpMixedGdnGeometry):
            raise TypeError("plan geometry must be a Qwen4ExpMixedGdnGeometry")
        object.__setattr__(
            self, "live_state_slots", _positive_int(self.live_state_slots, "live_state_slots")
        )
        waves = tuple(self.waves)
        for wave in waves:
            if not isinstance(wave, Qwen4ExpMixedGdnWave):
                raise TypeError("plan waves must be Qwen4ExpMixedGdnWave values")
        object.__setattr__(self, "waves", waves)
        self._validate()

    def _validate(self) -> None:
        expected = _build_waves(self.layout)
        if self.waves != expected:
            raise ValueError(
                "the plan's waves do not match this layout's rows: "
                f"stored {self.waves!r} vs derived {expected!r}"
            )
        slots = self.state_slots
        if slots > QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS:
            raise ValueError(
                f"{slots} physical rows exceed the indexed GDN state slab width "
                f"{QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS}"
            )
        if slots > QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS:
            raise ValueError(
                f"{slots} physical rows exceed the indexed conv state slab width "
                f"{QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS}"
            )
        if self.live_state_slots > QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS:
            raise ValueError(
                f"live_state_slots {self.live_state_slots} exceeds the indexed conv "
                f"state slab width {QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS}"
            )
        for request in self.layout.requests:
            if request.slot >= self.live_state_slots:
                raise ValueError(
                    f"request {request.request_id} owns slot {request.slot}, which is "
                    f"outside the live state slab of {self.live_state_slots} slot(s)"
                )
        # The primitives' own admission helpers are the authoritative slot and row
        # bound, so the plan is admitted exactly where a wave dispatch would be.
        for wave in self.waves:
            qwen4_exp_gdn_indexed_state_indices(
                wave.slots, rows=wave.width, state_slots=slots
            )
            qwen4_exp_conv_indexed_state_indices(
                wave.slots, rows=wave.width, state_slots=slots
            )

    # -- accessors --------------------------------------------------------

    @property
    def state_slots(self) -> int:
        """Provisional slab width: one slot per physical row of the layout."""

        return self.layout.total_rows

    @property
    def total_rows(self) -> int:
        return self.layout.total_rows

    @property
    def request_width(self) -> int:
        return self.layout.physical_request_width

    @property
    def max_wave_rows(self) -> int:
        """Rows in the widest dispatch: wave 0, which holds every request."""

        return max(wave.width for wave in self.waves)

    def wave(self, depth: int) -> Qwen4ExpMixedGdnWave:
        parsed = _non_negative_int(depth, "depth")
        if parsed >= len(self.waves):
            raise KeyError(
                f"this pass prepares depths 0..{len(self.waves) - 1}, not {parsed}"
            )
        return self.waves[parsed]

    def live_slot(self, request_id: int) -> int:
        """The live state slot a request's root snapshot is copied from."""

        return self.layout.request(request_id).slot

    def snapshot_row(self, request_id: int, depth: int) -> int:
        """Physical row, and so provisional slot, holding one chain snapshot."""

        return self.layout.physical_row(request_id, depth)


@dataclass(frozen=True)
class Qwen4ExpMixedGdnProjectedRows:
    """The shared input projections, one packed row per physical row.

    Row ``p`` of each buffer is physical row ``p``'s projected operands, in the
    order the layout stores its rows. The adapter gathers the rows a wave needs
    out of these buffers; it never projects them.
    """

    qkv: DeviceBuffer
    gate: DeviceBuffer
    alpha: DeviceBuffer
    beta: DeviceBuffer

    def __post_init__(self) -> None:
        for name in ("qkv", "gate", "alpha", "beta"):
            buffer = getattr(self, name)
            if not isinstance(buffer, DeviceBuffer):
                raise TypeError(f"projected {name} rows must be a DeviceBuffer")


@dataclass
class Qwen4ExpMixedGdnWaveScratch:
    """One wave's contiguous operand rows, the shape the primitives read.

    Every buffer holds ``rows`` packed rows of its own width, so a wave's rows are
    gathered here in wave order and the primitives address them as rows
    ``0..rows-1``.
    """

    qkv: DeviceBuffer
    gate: DeviceBuffer
    alpha: DeviceBuffer
    beta: DeviceBuffer
    conv: DeviceBuffer
    core: DeviceBuffer
    rows: int
    qkv_width: int
    core_width: int
    scalar_width: int
    runtime: HipRuntime
    closed: bool = field(default=False)
    # Buffers still owned, in allocation order; ``close`` frees the last entry
    # first and drops it only after its free returns, so a retry cannot double-free.
    _pending_free: list[DeviceBuffer] = field(default_factory=list)

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        qkv_width: int,
        core_width: int,
        scalar_width: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpMixedGdnWaveScratch":
        count = _positive_int(rows, "rows")
        qkv = _positive_int(qkv_width, "qkv_width")
        core = _positive_int(core_width, "core_width")
        scalar = _positive_int(scalar_width, "scalar_width")
        active_runtime = runtime or get_hip_runtime()
        sizes = (
            count * qkv * 4,
            count * core * 4,
            count * scalar * 4,
            count * scalar * 4,
            count * qkv * 4,
            count * core * 4,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in sizes:
                buffers.append(malloc(size, runtime=active_runtime))
        except BaseException as error:
            try:
                _free_pending(buffers, active_runtime)
            except Exception as cleanup_error:  # pragma: no cover - defensive
                error.add_note(
                    f"cleanup after the failed allocation also failed: {cleanup_error!r}"
                )
            raise
        return cls(
            *buffers,
            rows=count,
            qkv_width=qkv,
            core_width=core,
            scalar_width=scalar,
            runtime=active_runtime,
            _pending_free=list(buffers),
        )

    # -- geometry ---------------------------------------------------------

    def width(self, name: str) -> int:
        """Row width in floats of one wave buffer."""

        return self._widths()[self._buffer(name)]

    def row_ptr(self, name: str, row: int) -> int:
        """Device address of one row inside a wave buffer."""

        buffer = self._buffer(name)
        parsed = _non_negative_int(row, "row")
        if parsed >= self.rows:
            raise ValueError(
                f"wave row {parsed} is outside [0, {self.rows})"
            )
        return buffer.ptr + parsed * self._widths()[buffer] * 4

    def _buffer(self, name: str) -> DeviceBuffer:
        if name not in _WAVE_BUFFER_NAMES:
            raise KeyError(
                f"unknown wave buffer {name!r}; this scratch holds "
                + ", ".join(_WAVE_BUFFER_NAMES)
            )
        return getattr(self, name)

    def _widths(self) -> dict[DeviceBuffer, int]:
        return {
            self.qkv: self.qkv_width,
            self.gate: self.core_width,
            self.alpha: self.scalar_width,
            self.beta: self.scalar_width,
            self.conv: self.qkv_width,
            self.core: self.core_width,
        }

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        pending = self._pending_free
        if not pending:
            # Ownership was not handed over by ``allocate`` (direct construction);
            # take it now so closing cannot silently leak.
            pending.extend(
                (self.qkv, self.gate, self.alpha, self.beta, self.conv, self.core)
            )
        _free_pending(pending, self.runtime)
        self.closed = True


@dataclass
class Qwen4ExpMixedGdnAdapter:
    """Runs one mixed layout's recurrence in depth waves over provisional state.

    :meth:`allocate` builds the provisional state, the wave scratch and the
    per-physical-row result buffer. :meth:`run` executes the pass without touching
    the live state owner; :meth:`commit` copies the accepted prefix's snapshots
    into it, once per completed pass.
    """

    plan: Qwen4ExpMixedGdnPlan
    live_state: Qwen4ExpBatchGdnState
    working_state: Qwen4ExpBatchGdnState
    wave: Qwen4ExpMixedGdnWaveScratch
    core: DeviceBuffer
    runtime: HipRuntime
    # Current owner token of one physical slot, or ``None`` to trust the layout.
    owner_of: Callable[[int], int] | None = field(default=None)
    # Set only by a pass that ran every wave; consumed by the first commit copy.
    verification_pending: bool = field(default=False)
    closed: bool = field(default=False)
    closing: bool = field(default=False)
    _core_owned: bool = field(default=True)

    # -- construction -----------------------------------------------------

    @classmethod
    def allocate(
        cls,
        *,
        layout: Qwen4ExpMixedLayout,
        live_state: Qwen4ExpBatchGdnState,
        owner_of: Callable[[int], int] | None = None,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpMixedGdnAdapter":
        """Allocate one adapter for ``layout`` against ``live_state``.

        ``owner_of`` answers "which owner token does this physical slot hold
        now?". When given, every request's layout token is checked against it
        here and again before ``run`` and ``commit`` touch the device, so a slot
        released and re-admitted to another request is refused instead of
        verifying against somebody else's state.

        Every layout, ownership and geometry check runs before the first byte is
        allocated, and a failed allocation releases whatever it had already taken.
        """

        if not isinstance(layout, Qwen4ExpMixedLayout):
            raise TypeError("layout must be a Qwen4ExpMixedLayout")
        if owner_of is not None and not callable(owner_of):
            raise TypeError("owner_of must be callable or None")
        _require_live_state(live_state)
        geometry = Qwen4ExpMixedGdnGeometry.from_state(live_state)
        plan = plan_qwen4_exp_mixed_gdn(
            layout, geometry=geometry, live_state_slots=live_state.state_slots
        )
        _require_owner_epoch(plan, owner_of)
        active_runtime = runtime or live_state.runtime
        if active_runtime is not live_state.runtime:
            raise ValueError(
                "runtime must be the state owner's runtime; the provisional slabs "
                "copy from and commit into that owner"
            )
        working_state = Qwen4ExpBatchGdnState.allocate(
            state_slots=plan.state_slots,
            num_k_heads=geometry.num_k_heads,
            num_v_heads=geometry.num_v_heads,
            head_dim=geometry.head_dim,
            conv_kernel=geometry.conv_kernel,
            runtime=active_runtime,
        )
        wave: Qwen4ExpMixedGdnWaveScratch | None = None
        core: DeviceBuffer | None = None
        try:
            wave = Qwen4ExpMixedGdnWaveScratch.allocate(
                rows=plan.max_wave_rows,
                qkv_width=geometry.qkv_width,
                core_width=geometry.core_width,
                scalar_width=geometry.scalar_width,
                runtime=active_runtime,
            )
            core = malloc(plan.total_rows * geometry.core_width * 4, runtime=active_runtime)
        except BaseException:
            for owner in (core, wave, working_state):
                if owner is None:
                    continue
                try:
                    releaser = getattr(owner, "close", None)
                    if releaser is None:
                        free(owner, runtime=active_runtime)
                    else:
                        releaser()
                except Exception:  # pragma: no cover - defensive
                    pass
            raise
        return cls(
            plan=plan,
            live_state=live_state,
            working_state=working_state,
            wave=wave,
            core=core,
            runtime=active_runtime,
            owner_of=owner_of,
        )

    # -- geometry ---------------------------------------------------------

    @property
    def geometry(self) -> Qwen4ExpMixedGdnGeometry:
        return self.plan.geometry

    def core_row_ptr(self, physical_row: int) -> int:
        """Device address of one physical row's gated recurrence output."""

        parsed = self._require_row(physical_row)
        return self.core.ptr + parsed * self.geometry.core_width * 4

    def recurrent_snapshot_ptr(self, physical_row: int) -> int:
        """Device address of one physical row's retained recurrent snapshot."""

        parsed = self._require_row(physical_row)
        return self.working_state.recurrent_slot_ptr(parsed)

    def conv_snapshot_ptr(self, physical_row: int) -> int:
        """Device address of one physical row's retained Conv snapshot."""

        parsed = self._require_row(physical_row)
        return self.working_state.conv_slot_ptr(parsed)

    # -- execution --------------------------------------------------------

    def run(
        self,
        projected: Qwen4ExpMixedGdnProjectedRows,
        *,
        conv_weight_ptr: int,
        dt_bias_ptr: int,
        a_ptr: int,
        norm_weight_ptr: int,
        stream: int = 0,
    ) -> DeviceBuffer:
        """Advance every physical row's provisional state in depth waves.

        Root snapshots are copied from the live owner, each deeper snapshot from
        its parent row, and each wave runs the two indexed owners once. Returns
        :attr:`core`: ``[total_rows, core_width]`` in physical row order, ready
        for the caller's shared output projection.

        Every input, ownership and geometry check runs before the first copy, so a
        rejected call leaves the provisional slabs, the live owner and
        :attr:`core` exactly as they were. The pass becomes committable only after
        the last wave, so a failed pass cannot leave older snapshots looking
        current.
        """

        self._require_open()
        _require_live_state(self.live_state)
        self.require_owner_epoch()
        stream_id = _non_negative_int(stream, "stream")
        geometry = self.geometry
        plan = self.plan
        _require_projected(projected, plan, geometry)
        for name, pointer in (
            ("conv_weight", conv_weight_ptr),
            ("dt_bias", dt_bias_ptr),
            ("a", a_ptr),
            ("norm_weight", norm_weight_ptr),
        ):
            _positive_int(pointer, f"{name}_ptr")
        if self.wave.conv.ptr % _FLOAT4_BYTES:
            raise ValueError(
                "the wave Conv scratch must be 16-byte aligned for the float4 Q/K loads"
            )
        if self.working_state.indices.ptr % 8:
            raise ValueError("state indices must be 8-byte aligned for int64 loads")

        # A pass is committable only once every wave has run, so a failure below
        # cannot leave the previous pass's snapshots looking current.
        self.verification_pending = False
        for request in plan.layout.requests:
            self._copy_snapshot(
                source_slot=plan.live_slot(request.request_id),
                destination_row=plan.snapshot_row(request.request_id, 0),
                source=self.live_state,
            )
        for wave in plan.waves:
            if wave.depth:
                for row, parent in zip(wave.rows, wave.parent_rows, strict=True):
                    self._copy_snapshot(
                        source_slot=parent,
                        destination_row=row,
                        source=self.working_state,
                    )
            self._gather_rows(projected, wave)
            active = self.working_state.prepare(wave.slots)
            qwen4_exp_linear_attn_conv_decode_indexed_f32(
                self.wave.qkv.ptr,
                self.working_state.conv_state.ptr,
                conv_weight_ptr,
                self.wave.conv.ptr,
                active.indices.ptr,
                rows=wave.width,
                state_slots=plan.state_slots,
                channels=geometry.qkv_width,
                kernel_size=geometry.conv_kernel,
                stream=stream_id,
                runtime=self.runtime,
            )
            qwen4_exp_gdn_decode_rowsplit_indexed_f32(
                self.wave.conv.ptr,
                self.wave.gate.ptr,
                self.wave.alpha.ptr,
                self.wave.beta.ptr,
                dt_bias_ptr,
                a_ptr,
                norm_weight_ptr,
                self.working_state.recurrent_state.ptr,
                self.wave.core.ptr,
                active.indices.ptr,
                rows=wave.width,
                state_slots=plan.state_slots,
                num_k_heads=geometry.num_k_heads,
                num_v_heads=geometry.num_v_heads,
                head_k_dim=geometry.head_dim,
                head_v_dim=geometry.head_dim,
                stream=stream_id,
                runtime=self.runtime,
            )
            self._scatter_core(wave)
        self.verification_pending = True
        return self.core

    def require_owner_epoch(self) -> None:
        """Refuse a layout whose slots have been re-admitted since it was built.

        A no-op without an ``owner_of`` callback: the adapter then has no view of
        the batch owner's admissions. Runs before every device copy in ``run`` and
        ``commit``, so stale reuse is reported instead of verifying against, or
        committing into, another request's state.
        """

        _require_owner_epoch(self.plan, self.owner_of)

    def commit(
        self,
        accepted_counts: Sequence[int],
        *,
        transaction_id: int = 0,
        next_tokens: Sequence[int | None] | None = None,
    ) -> TargetCommitPlan:
        """Commit each request's accepted prefix from its snapshot into live state.

        ``accepted_counts`` and the slot ownership are validated before any copy,
        and the completed pass is consumed *before* the first copy: a commit that
        fails partway through has already written some requests' live state, so it
        cannot be replayed. Recovery is a fresh :meth:`run`, which rebuilds every
        snapshot from the live owner. Returns the layout's
        :class:`~hipengine.speculative.interfaces.TargetCommitPlan`.
        """

        self._require_open()
        _require_live_state(self.live_state)
        self.require_owner_epoch()
        if not self.verification_pending:
            raise RuntimeError(
                "Qwen4Exp mixed GDN adapter has no completed verification pass to "
                "commit; run a pass first, and re-run after any failed commit"
            )
        destinations = self.plan.layout.commit_destinations(
            accepted_counts,
            transaction_id=transaction_id,
            next_tokens=next_tokens,
        )
        # Consume the transaction before the first write: a partially applied
        # commit must not be repeatable as if it were still pending.
        self.verification_pending = False
        for request, row in zip(
            self.plan.layout.requests, destinations.commit_rows, strict=True
        ):
            self._copy_snapshot(
                source_slot=row,
                destination_row=request.slot,
                source=self.working_state,
                destination=self.live_state,
            )
        return destinations

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Release the provisional state, wave scratch and result buffer.

        Each step is retry-safe: a failed free leaves that owner's remainder
        pending, so a second ``close`` releases exactly what is left. The live
        state owner is borrowed, never freed here.
        """

        if self.closed:
            return
        # Poison execution before the first free, as the state owner does: a
        # partial close can leave earlier buffers released and later ones owned.
        self.closing = True
        self._release_core()
        self.wave.close()
        self.working_state.close()
        self.closed = True

    def __enter__(self) -> "Qwen4ExpMixedGdnAdapter":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _release_core(self) -> None:
        if not self._core_owned:
            return
        free(self.core, runtime=self.runtime)
        self._core_owned = False

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp mixed GDN adapter is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp mixed GDN adapter is closing; its buffers may already be "
                "released"
            )
        if self.working_state.closed or self.working_state.closing:
            raise RuntimeError("Qwen4Exp mixed GDN provisional state is not open")
        if self.wave.closed:
            raise RuntimeError("Qwen4Exp mixed GDN wave scratch is closed")

    def _require_row(self, physical_row: object) -> int:
        self._require_open()
        parsed = _non_negative_int(physical_row, "physical_row")
        if parsed >= self.plan.total_rows:
            raise ValueError(
                f"physical row {parsed} is outside [0, {self.plan.total_rows})"
            )
        return parsed

    def _copy_snapshot(
        self,
        *,
        source_slot: int,
        destination_row: int,
        source: Qwen4ExpBatchGdnState,
        destination: Qwen4ExpBatchGdnState | None = None,
    ) -> None:
        """Copy one slot's recurrent and Conv blocks, recurrent first.

        The source and destination owners are named explicitly because a root row
        reads the live owner and writes the provisional one, while every deeper
        row reads and writes the provisional slab.
        """

        target = self.working_state if destination is None else destination
        _copy_device_range(
            self.runtime,
            target.recurrent_slot_ptr(destination_row),
            source.recurrent_slot_ptr(source_slot),
            target.recurrent_bytes_per_slot,
        )
        _copy_device_range(
            self.runtime,
            target.conv_slot_ptr(destination_row),
            source.conv_slot_ptr(source_slot),
            target.conv_bytes_per_slot,
        )

    def _gather_rows(
        self, projected: Qwen4ExpMixedGdnProjectedRows, wave: Qwen4ExpMixedGdnWave
    ) -> None:
        """Copy a wave's packed rows out of the shared projections, in wave order."""

        widths = (
            ("qkv", projected.qkv, self.geometry.qkv_width),
            ("gate", projected.gate, self.geometry.core_width),
            ("alpha", projected.alpha, self.geometry.scalar_width),
            ("beta", projected.beta, self.geometry.scalar_width),
        )
        for index, physical_row in enumerate(wave.rows):
            for name, source, width in widths:
                _copy_device_range(
                    self.runtime,
                    self.wave.row_ptr(name, index),
                    source.ptr + physical_row * width * 4,
                    width * 4,
                )

    def _scatter_core(self, wave: Qwen4ExpMixedGdnWave) -> None:
        """Copy a wave's gated recurrence output back to physical row order."""

        nbytes = self.geometry.core_width * 4
        for index, physical_row in enumerate(wave.rows):
            _copy_device_range(
                self.runtime,
                self.core_row_ptr(physical_row),
                self.wave.row_ptr("core", index),
                nbytes,
            )


def plan_qwen4_exp_mixed_gdn(
    layout: Qwen4ExpMixedLayout,
    *,
    geometry: Qwen4ExpMixedGdnGeometry,
    live_state_slots: int,
) -> Qwen4ExpMixedGdnPlan:
    """Decompose one mixed layout into depth waves, or refuse it.

    Pure host work: it allocates nothing and dispatches nothing, so a layout,
    ownership or geometry defect is reported before an adapter exists.
    """

    if not isinstance(layout, Qwen4ExpMixedLayout):
        raise TypeError("layout must be a Qwen4ExpMixedLayout")
    if not isinstance(geometry, Qwen4ExpMixedGdnGeometry):
        raise TypeError("geometry must be a Qwen4ExpMixedGdnGeometry")
    return Qwen4ExpMixedGdnPlan(
        layout=layout,
        geometry=geometry,
        live_state_slots=_positive_int(live_state_slots, "live_state_slots"),
        waves=_build_waves(layout),
    )


def _build_waves(layout: Qwen4ExpMixedLayout) -> tuple[Qwen4ExpMixedGdnWave, ...]:
    """One wave per prepared depth, each row ordered by physical position."""

    waves: list[Qwen4ExpMixedGdnWave] = []
    for depth in range(layout.max_chain_rows):
        entries: list[tuple[int, int, int, int]] = []
        for request in layout.requests:
            if request.rows <= depth:
                continue
            row = layout.physical_row(request.request_id, depth)
            parent = -1 if depth == 0 else layout.physical_row(request.request_id, depth - 1)
            entries.append((row, request.request_id, parent, row))
        entries.sort()
        waves.append(
            Qwen4ExpMixedGdnWave(
                depth=depth,
                request_ids=tuple(request for _, request, _, _ in entries),
                rows=tuple(row for row, _, _, _ in entries),
                parent_rows=tuple(parent for _, _, parent, _ in entries),
                slots=tuple(slot for _, _, _, slot in entries),
            )
        )
    return tuple(waves)


def _require_owner_epoch(
    plan: Qwen4ExpMixedGdnPlan, owner_of: Callable[[int], int] | None
) -> None:
    """Re-admit every request's layout owner token against the slot's token now.

    A physical slot's live state belongs to one admission. If the batch owner
    released slot ``s`` and handed the same index to another request after this
    layout was compiled, the layout's rows still name the old token: they would
    copy the new occupant's state as a root snapshot and commit over it. The
    callback is the only view the adapter has of that turnover, so a layout that
    no longer matches it is refused rather than resolved.
    """

    if owner_of is None:
        return
    for request in plan.layout.requests:
        current = _non_negative_int(owner_of(request.slot), "owner token")
        if current == request.owner_token:
            continue
        if current == 0:
            raise ValueError(
                f"slot {request.slot} is unowned now, but request "
                f"{request.request_id} was laid out under owner token "
                f"{request.owner_token}; the slot was released and its live state "
                "is not this request's"
            )
        raise ValueError(
            f"slot {request.slot} is owned by token {current}, but request "
            f"{request.request_id} was laid out under owner token "
            f"{request.owner_token}; the slot was released and re-admitted, so "
            "its live state is not this request's"
        )


def _require_live_state(state: object) -> None:
    if not isinstance(state, Qwen4ExpBatchGdnState):
        raise TypeError("live state must be a Qwen4ExpBatchGdnState")
    if state.closed:
        raise RuntimeError("the live Qwen4Exp batch GDN state is closed")
    if state.closing:
        raise RuntimeError(
            "the live Qwen4Exp batch GDN state is closing; its buffers may already "
            "be released"
        )


def _require_projected(
    projected: object,
    plan: Qwen4ExpMixedGdnPlan,
    geometry: Qwen4ExpMixedGdnGeometry,
) -> None:
    """Reject projections that cannot hold one packed row per physical row."""

    if not isinstance(projected, Qwen4ExpMixedGdnProjectedRows):
        raise TypeError("projected must be a Qwen4ExpMixedGdnProjectedRows")
    required = (
        ("qkv", geometry.qkv_width),
        ("gate", geometry.core_width),
        ("alpha", geometry.scalar_width),
        ("beta", geometry.scalar_width),
    )
    for name, width in required:
        buffer = getattr(projected, name)
        needed = plan.total_rows * width * 4
        if buffer.nbytes < needed:
            raise ValueError(
                f"projected {name} rows hold {buffer.nbytes} bytes, need {needed} "
                f"for {plan.total_rows} physical row(s) of width {width}"
            )


def _copy_device_range(
    runtime: HipRuntime, destination_ptr: int, source_ptr: int, nbytes: int
) -> None:
    """Copy one device range to another. Synchronous, as this adapter documents."""

    runtime.memcpy(destination_ptr, source_ptr, nbytes, MemcpyKind.DEVICE_TO_DEVICE)


def _free_pending(pending: list[DeviceBuffer], runtime: HipRuntime) -> None:
    """Free ``pending`` newest-first (last entry first), popping after each free.

    A failed ``free`` leaves that buffer and everything older in ``pending``, so
    retrying releases exactly the remainder instead of double-freeing the buffers
    that are already gone.
    """

    while pending:
        free(pending[-1], runtime=runtime)
        pending.pop()


def _non_negative_int(value: object, name: str) -> int:
    parsed = _int_index(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_int(value: object, name: str) -> int:
    parsed = _int_index(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _int_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
