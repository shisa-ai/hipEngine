"""Full-model AR decode batch for the Qwen4Exp resident runner.

Every plain autoregressive row is prefilled on a c1 staging runner, its state
is imported into one physical slot of a :class:`Qwen4ExpGGUFBatchRunner`, and
the staging runner goes back to the pool. From then on the row advances by one
shared N-row step per decode tick. The slots share one paged K/V and index-key
block pool sized from the device memory left after startup, and a request
holds exactly the blocks its prompt plus decode budget needs, so the number of
concurrent requests is bounded by live tokens rather than by one full-context
window per request.

A request whose extent exceeds a batch slot (``batch_slot_positions``) or the
whole pool decodes on its own c1 staging runner, as do MTP rows: a speculative
row verifies a candidate chain, which the one-token-per-slot batch step does
not express. An automatic MTP row is returned to the batch when other requests
arrive (see ``Qwen4ExpResidentServingRunner``).

Independent HTTP arrivals overlap because the generator declares the
``token_budget`` prefill/decode policy (``engine_loop_config_defaults``): prefill
chunks rotate across requests and every decoding row steps each round. An
explicit ``HIPENGINE_PREFILL_DECODE_POLICY`` still overrides the declaration.

Three properties are control, not arithmetic, and are asserted exactly:

* one physical slot per request, pinned at admission, imported once, and
  released on reclaim, discard, or a failed step;
* the slot never follows scheduler slot indices, so a scheduler compaction
  cannot move a row's state or silently re-bind it to another request;
* a failed or malformed step poisons its rows, so the owner releases every
  participating slot and refuses further work instead of stepping a stale owner
  token or attributing output from a row set it cannot trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpTokenResult

#: Rows the indexed GDN, conv, and PLE decode kernels accept in one step.
QWEN4_EXP_MAX_BATCH_ROWS = 64
#: Default concurrent requests when the operator sets no active-request count.
QWEN4_EXP_DEFAULT_REQUEST_CAPACITY = 8
#: c1 runners kept for prefill staging, MTP rows, and over-long requests.
QWEN4_EXP_STAGING_RUNNERS = 2
#: Largest extent one batch slot holds. Each slot keeps per-position index
#: slabs outside the shared pool, so a longer cap costs memory for every slot;
#: a longer request decodes on its c1 staging runner instead.
QWEN4_EXP_BATCH_SLOT_POSITIONS = 65536

def batch_slot_positions(max_sequence_length: int) -> int:
    """Per-slot extent of the batch for one serving context."""

    return max(1, min(int(max_sequence_length), QWEN4_EXP_BATCH_SLOT_POSITIONS))


def _require_integer(value: object, *, label: str) -> int:
    """Parse an integral handle, rejecting a bool and a non-integral value.

    ``int(3.7)`` silently truncates to ``3`` and ``int(True)`` becomes ``1``, so
    a caller's bug would turn into a wrong token, slot, or position. The batch
    runner repeats this check one layer down; doing it here refuses a malformed
    row before any slot is touched.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer, got a bool")
    try:
        return int(operator.index(value))
    except TypeError:
        raise ValueError(
            f"{label} must be an integer, got {type(value).__name__}"
        ) from None


class Qwen4ExpARBatchStepError(RuntimeError):
    """A shared AR batch step failed or returned an unusable row set.

    The rows cannot continue: the batch runner poisoned their state (or advanced
    it past a result we cannot attribute), and stepping the c1 runner from its
    prefill position instead would mix two arithmetic histories inside one
    request. The serving runner fails the tick and refuses new batch work.
    """

    def __init__(
        self,
        request_ids: Sequence[int],
        slots: Sequence[int],
        cause: BaseException,
    ) -> None:
        self.request_ids = tuple(int(value) for value in request_ids)
        self.slots = tuple(int(value) for value in slots)
        self.cause = cause
        super().__init__(
            "Qwen4Exp AR batch step failed for request(s) "
            f"{list(self.request_ids)} on slot(s) {list(self.slots)}: {cause}. "
            "Every participating slot was released and the batch decode path "
            "refuses further work until the serving runner is rebuilt."
        )


@dataclass
class _Lease:
    """One request's pinned physical slot, owner token, and import state."""

    slot: int
    owner: int
    positions: int
    imported: bool = False
    #: Physical widths this request's own decode steps ran at, keyed by row
    #: count. A request that shared a step with three peers reports ``{4: n}``,
    #: so a response can describe the work it actually took part in instead of
    #: the queue shape it was planned under.
    step_widths: dict[int, int] = field(default_factory=dict)


class Qwen4ExpARBatchOwner:
    """Own the shared batch runner and the request-to-physical-slot map.

    ``batch_factory`` is called once, lazily, and returns an object with the
    :class:`Qwen4ExpGGUFBatchRunner` surface used here (``free_slots``,
    ``admit``, ``import_prefilled``, ``step``, ``release``, ``close``). Keeping
    it a factory lets the host construct the owner before paying for the state
    slabs, and lets the unit tier drive the whole owner with a fake runner.
    """

    def __init__(
        self,
        *,
        batch_factory: Callable[[], Any],
        state_slots: int,
        max_positions: int,
        label: str = "qwen4exp_full_model_ar_batch",
    ) -> None:
        slots = _require_integer(state_slots, label="state_slots")
        positions = _require_integer(max_positions, label="max_positions")
        if slots <= 0:
            raise ValueError("Qwen4Exp AR batch state_slots must be positive")
        if positions <= 0:
            raise ValueError("Qwen4Exp AR batch max_positions must be positive")
        self._factory = batch_factory
        self.state_slots = slots
        self.max_positions = positions
        self.label = str(label)
        self._batch: Any | None = None
        self._leases: dict[int, _Lease] = {}
        self._failed_requests: set[int] = set()
        self._failures: list[str] = []
        # Guards the counters and the width histogram so a live telemetry read
        # cannot iterate a dict another thread is mutating.
        self._state_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._steps = 0
        self._imports = 0
        self._releases = 0
        self._widths: dict[int, int] = {}

    # -- construction -----------------------------------------------------

    @property
    def runner(self) -> Any | None:
        """The shared batch runner, or ``None`` before the first reservation."""

        return self._batch

    @property
    def failure(self) -> str | None:
        """Why this owner refuses work, or ``None`` while it is healthy."""

        with self._state_lock:
            return None if not self._failures else "; ".join(self._failures)

    @property
    def live_request_ids(self) -> tuple[int, ...]:
        return tuple(self._leases)

    @property
    def free_slot_count(self) -> int:
        """Slots still available for a new request without building the owner."""

        return self.state_slots - len(self._leases)

    @property
    def free_block_count(self) -> int:
        """Pool blocks a new reservation can still take."""

        batch = self._batch
        return 0 if batch is None else int(batch.free_block_count)

    @property
    def pool_blocks(self) -> int:
        batch = self._batch
        return 0 if batch is None else int(batch.pool_blocks)

    def blocks_for_positions(self, positions: int) -> int:
        batch = self.ensure()
        return int(batch.blocks_for_positions(positions))

    def fits(self, positions: int) -> bool:
        """Whether a request of ``positions`` can ever hold a batch slot.

        False for an extent past the slot cap or larger than the whole pool;
        such a request decodes on its own c1 runner instead of waiting forever.
        A failed owner still answers True, so the reservation refuses the
        request by name instead of quietly moving it to another decode path.
        """

        needed = int(positions)
        if needed <= 0 or needed > self.max_positions:
            return False
        if self._failures:
            return True
        batch = self.ensure()
        return int(batch.blocks_for_positions(needed)) <= int(batch.pool_blocks)

    def ensure(self) -> Any:
        """Return the batch runner, building it on first use.

        A build failure (for example a device allocation that does not fit)
        propagates to the caller, which is the admission path: the request is
        refused before any of its work runs. A close in progress refuses work
        too, so a retry cannot race the outstanding free.
        """

        if self._closing:
            raise RuntimeError(
                "Qwen4Exp AR batch owner is closing; no work is accepted until "
                "close() succeeds"
            )
        if self._closed:
            raise RuntimeError("Qwen4Exp AR batch owner is closed")
        if self._failures:
            raise RuntimeError(
                "Qwen4Exp AR batch decode is disabled after a failed call: "
                + self.failure
            )
        if self._batch is None:
            self._batch = self._factory()
        return self._batch

    # -- admission --------------------------------------------------------

    def reserve(self, request_id: int, *, positions: int) -> int:
        """Pin one physical slot to ``request_id`` and return the slot index.

        The position bound is checked here rather than after prefill so an
        oversized request fails closed at admission, before it consumes a
        prefill pass or a c1 runner.
        """

        rid = _require_integer(request_id, label="request_id")
        if self._closing:
            raise RuntimeError(
                "Qwen4Exp AR batch owner is closing; no work is accepted until "
                "close() succeeds"
            )
        if self._closed:
            raise RuntimeError("Qwen4Exp AR batch owner is closed")
        if rid in self._leases:
            raise ValueError(f"request {rid} already owns an AR batch slot")
        needed = _require_integer(positions, label="positions")
        if needed <= 0:
            raise ValueError("an AR batch reservation needs a positive position count")
        if needed > self.max_positions:
            raise ValueError(
                f"request {rid} needs {needed} positions but an AR batch slot "
                f"holds {self.max_positions}"
            )
        batch = self.ensure()
        free = tuple(batch.free_slots)
        if not free:
            raise RuntimeError(
                "Qwen4Exp AR batch owner has no free state slot "
                f"({self.state_slots} in use)"
            )
        slot = int(free[0])
        owner = int(batch.admit(slot, positions=needed))
        lease = _Lease(slot, owner, needed)
        with self._state_lock:
            collision = rid in self._leases
            if not collision:
                self._leases[rid] = lease
                self._failed_requests.discard(rid)
        if collision:
            # Only reachable if two admissions race for one request id. Hand
            # the just-admitted slot back rather than leaking it.
            batch.release(lease.slot, lease.owner)
            raise ValueError(f"request {rid} already owns an AR batch slot")
        return slot

    def owns(self, request_id: int) -> bool:
        """Whether ``request_id`` holds a pinned physical slot."""

        return int(request_id) in self._leases

    def imported(self, request_id: int) -> bool:
        """Whether ``request_id``'s prefilled c1 state was imported."""

        lease = self._leases.get(int(request_id))
        return lease is not None and lease.imported

    def failed(self, request_id: int) -> bool:
        """Whether a failed step released ``request_id``'s slot."""

        return int(request_id) in self._failed_requests

    def lease(self, request_id: int) -> tuple[int, int] | None:
        """``(slot, owner)`` of ``request_id``, for diagnostics and tests."""

        lease = self._leases.get(int(request_id))
        return None if lease is None else (lease.slot, lease.owner)

    def import_prefilled(self, request_id: int, source: object) -> int:
        """Copy ``source``'s prefilled c1 state into the request's slot once."""

        rid = _require_integer(request_id, label="request_id")
        batch = self.ensure()
        lease = self._require_lease(rid)
        if lease.imported:
            # A second import would overwrite the live KV/GDN/PLE history with
            # the prefill state, so the request would silently replay its
            # prompt. Refuse instead of resetting.
            raise RuntimeError(
                f"request {rid} already imported its prefilled state; a second "
                "import would reset the slot's live history"
            )
        try:
            position = int(
                batch.import_prefilled(lease.slot, lease.owner, source=source)
            )
        except BaseException as error:
            # A rejected import writes nothing; a failed copy poisons the slot.
            # Either way this request cannot decode here, so drop the lease and
            # stop the path instead of leaving a half-imported slot reachable.
            self._record_failure(f"import of request {rid} failed: {error}")
            self.release(rid)
            raise
        lease.imported = True
        with self._state_lock:
            self._imports += 1
        return position

    # -- decode -----------------------------------------------------------

    def step(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        capture_logits: bool,
    ) -> Mapping[int, Qwen4ExpTokenResult]:
        """Advance every row by one token in a single N-row batch call.

        ``rows`` is ``(request_id, token)`` in the caller's work-item order. The
        returned mapping is keyed by request id, so a row's result cannot be
        confused with its neighbour's even when the physical slots are ordered
        differently from the request order.

        The whole row set is parsed and validated before the call. The result is
        validated before any output is attributed: a step that advances state
        but returns an unusable row set fails its rows exactly like a failed
        call, because the state cannot be re-stepped or re-attributed.
        """

        if not rows:
            raise ValueError("an AR batch step needs at least one row")
        request_ids = tuple(
            _require_integer(request_id, label=f"row {index} request id")
            for index, (request_id, _) in enumerate(rows)
        )
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("a request appears twice in one AR batch step")
        tokens = tuple(
            _require_integer(token, label=f"row {index} token id")
            for index, (_, token) in enumerate(rows)
        )
        capture = bool(capture_logits)
        batch = self.ensure()
        leases = tuple(self._require_lease(rid) for rid in request_ids)
        for request_id, lease in zip(request_ids, leases):
            if not lease.imported:
                raise RuntimeError(
                    f"request {request_id} has no imported prefilled state; an "
                    "AR batch step would advance a slot that holds no request "
                    "history"
                )
        slots = tuple(lease.slot for lease in leases)
        try:
            result = batch.step(
                tokens,
                slots=slots,
                owners=tuple(lease.owner for lease in leases),
                capture_logits=capture,
            )
        except BaseException as error:
            self._fail_rows(request_ids, f"step failed: {error}")
            raise Qwen4ExpARBatchStepError(request_ids, slots, error) from error
        try:
            decoded = self._decode_rows(request_ids, slots, result, capture=capture)
        except BaseException as error:
            # The step already advanced device state, so a malformed result is
            # as unrecoverable as a failed call: fail the rows instead of
            # attributing output from a row set we cannot trust.
            self._fail_rows(
                request_ids, f"step returned a malformed result: {error}"
            )
            raise Qwen4ExpARBatchStepError(request_ids, slots, error) from error
        with self._state_lock:
            self._steps += 1
            width = len(request_ids)
            self._widths[width] = self._widths.get(width, 0) + 1
            for lease in leases:
                lease.step_widths[width] = lease.step_widths.get(width, 0) + 1
        return decoded

    def _decode_rows(
        self,
        request_ids: Sequence[int],
        slots: Sequence[int],
        result: object,
        *,
        capture: bool,
    ) -> dict[int, Qwen4ExpTokenResult]:
        """Validate one step's row set before any output is attributed."""

        rows = len(request_ids)
        returned = tuple(
            _require_integer(slot, label=f"result slot {index}")
            for index, slot in enumerate(result.slot_ids)
        )
        expected = tuple(int(slot) for slot in slots)
        if returned != expected:
            raise ValueError(
                f"the batch runner returned slots {returned} for requested "
                f"slots {expected}"
            )
        tokens = tuple(
            _require_integer(token, label=f"result token {index}")
            for index, token in enumerate(result.tokens)
        )
        if len(tokens) != rows:
            raise ValueError(
                f"the batch runner returned {len(tokens)} token(s) for {rows} "
                "row(s)"
            )
        for index, token in enumerate(tokens):
            if token < 0:
                raise ValueError(f"result token {index} is negative: {token}")
        logits = result.logits
        if capture:
            if logits is None:
                raise ValueError(
                    "the batch runner returned no logits for a row set that "
                    "needs the full readback"
                )
            array = np.asarray(logits)
            if array.ndim != 2 or array.shape[0] != rows or array.shape[1] <= 0:
                raise ValueError(
                    "the batch runner returned logits with shape "
                    f"{array.shape} for {rows} row(s)"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    "the batch runner returned non-finite logits for at least "
                    "one row"
                )
        decoded: dict[int, Qwen4ExpTokenResult] = {}
        for index, request_id in enumerate(request_ids):
            decoded[request_id] = Qwen4ExpTokenResult(
                token_id=tokens[index],
                logits=None if logits is None else np.asarray(logits[index]),
            )
        return decoded

    # -- release ----------------------------------------------------------

    def release(self, request_id: int) -> bool:
        """Drop ``request_id``'s slot. Idempotent, and safe after a failure."""
        rid = int(request_id)
        with self._state_lock:
            lease = self._leases.pop(rid, None)
        if lease is None:
            return False
        batch = self._batch
        if batch is None or self._closed or self._closing:
            # A close in progress owns the slabs; do not hand them a release.
            return True
        try:
            batch.release(lease.slot, lease.owner)
        except BaseException as error:
            # The lease is already gone, so nothing can step this slot again;
            # record the leak instead of raising through a teardown path.
            self._record_failure(f"release of slot {lease.slot} failed: {error}")
            return True
        with self._state_lock:
            self._releases += 1
        return True

    def close(self) -> None:
        """Close the shared runner, keeping a failed free retryable.

        The batch runner frees its slabs in reverse and pops each one only after
        its free returns, so a failed close leaves the remaining slabs reachable
        and a retry frees each exactly once. This method therefore keeps the
        runner reference and the leases until the close actually succeeds, and
        refuses new work while a close is outstanding.
        """

        if self._closed:
            return
        self._closing = True
        batch = self._batch
        if batch is not None:
            try:
                batch.close()
            except BaseException as error:
                self._record_failure(f"close failed: {error}")
                raise
        self._batch = None
        with self._state_lock:
            self._leases.clear()
        self._closing = False
        self._closed = True

    # -- reporting --------------------------------------------------------

    def request_telemetry(self, request_id: int) -> dict[str, Any] | None:
        """Widths this one request's decode steps ran at, before its release.

        Returns ``None`` for a request that never held a slot. The caller reads
        this at completion, while the lease still exists: releasing the slot
        discards the per-request history, exactly like the global histogram is
        discarded at close.
        """

        rid = int(request_id)
        with self._state_lock:
            lease = self._leases.get(rid)
            if lease is None:
                return None
            widths = dict(lease.step_widths)
        return {
            "execution_path": self.label,
            "steps": sum(widths.values()),
            "step_widths": {str(width): count for width, count in sorted(widths.items())},
            "max_width": max(widths) if widths else 0,
            "state_slots": self.state_slots,
        }

    def telemetry(self) -> dict[str, Any]:
        """Physical-width evidence for the scratch probe and worklog.

        A live HTTP caller may read this while the decode thread is stepping.
        The counters, width histogram, failure list, and lease count are read
        under one lock, and lease insert/remove/clear take that lock, so
        ``live_slots`` is the lease map's size at that instant rather than a
        later re-count. The result is still a snapshot, not a transaction
        against device state: a step can complete between the lock release and
        the HTTP response, so ``steps`` and ``live_slots`` may lag by one step.
        """

        with self._state_lock:
            widths = dict(self._widths)
            failure = None if not self._failures else "; ".join(self._failures)
            steps = self._steps
            imports = self._imports
            releases = self._releases
            live_slots = len(self._leases)
        return {
            "execution_path": self.label,
            "state_slots": self.state_slots,
            "max_positions": self.max_positions,
            "pool_blocks": self.pool_blocks,
            "free_blocks": self.free_block_count,
            "block_positions": (
                0 if self._batch is None else int(self._batch.block_size)
            ),
            "live_slots": live_slots,
            "steps": steps,
            "step_widths": {str(width): count for width, count in sorted(widths.items())},
            "imports": imports,
            "releases": releases,
            "failure": failure,
        }

    def _require_lease(self, request_id: int) -> _Lease:
        try:
            return self._leases[int(request_id)]
        except KeyError as exc:
            raise KeyError(
                f"request {request_id} does not own an AR batch slot"
            ) from exc

    def _fail_rows(self, request_ids: Sequence[int], message: str) -> None:
        """Mark a stepped row set unusable and release every one of its slots."""

        self._record_failure(f"requests {list(request_ids)}: {message}")
        for request_id in request_ids:
            self._failed_requests.add(request_id)
            self.release(request_id)

    def _record_failure(self, message: str) -> None:
        with self._state_lock:
            if message not in self._failures:
                self._failures.append(message)


def make_qwen4_exp_ar_batch_runner(
    generator: object, *, state_slots: int, max_positions: int | None = None
) -> Any:
    """Build the production batch runner over the generator's shared weights.

    The slots share one block pool sized from the device memory that is free
    when the runner is built, so it is built after every other serving owner.
    """

    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    resident = getattr(generator, "_resident", None)
    if resident is None:
        raise RuntimeError(
            "Qwen4Exp AR batch decode requires model-owned resident weights; "
            "this generator was built with an external runner"
        )
    runner = generator.runner
    batch = Qwen4ExpGGUFBatchRunner(
        resident,
        state_slots=int(state_slots),
        max_positions=int(
            batch_slot_positions(runner.max_sequence_length)
            if max_positions is None else max_positions
        ),
        backend=generator.backend,
        runtime=getattr(runner, "runtime", None),
        pool_blocks="auto",
    )
    # The expert-down rungs are request-local on the owning runner, so a batch
    # owner that never receives them would keep the parents on a rows>=64 pass.
    # The batch row cap is 64, so fold128 is reachable at exactly its own
    # threshold while the rows>=512 rungs stay on their parents; the Q5_1 down
    # M1 and PF-1 fork-b decisions select from rows>=2, so they are reachable
    # here too and must travel with the rest. The exact grouped Q5_1 down chain
    # gate is the chain all of those rungs sit inside; every plan publishes it
    # on, so an unbound carrier keeps it on rather than closing the chain. The
    # chain's non-expertgrid fallback rung is published on for the same reason:
    # an unbound carrier keeps the out8 owner, not the output1 base. The F16
    # WMMA down owner below that chain is published off, and a decode row
    # cannot reach the production admission that also selects it, so the
    # carrier has to hand over the plan's value rather than let the batch
    # owner default it. The tile pairs are the same kind of decision: the
    # binder publishes 16/16 for the compact F16 Q4_K gate/up owner and
    # (64, 32) for the Q8_0 dense WMMA prefill, and ``None`` means the
    # dispatch heuristic, so the carrier forwards the published value as-is.
    batch.q51_exact_grouped_down = bool(
        getattr(generator, "q51_exact_grouped_down", True))
    batch.moe_decode_warp = getattr(generator, "moe_decode_warp", frozenset())
    batch.routed_f16_enabled = bool(getattr(generator, "routed_f16_enabled", False))
    batch.q51_wmma_down = bool(getattr(generator, "q51_wmma_down", False))
    batch.q51_out8 = bool(getattr(generator, "q51_out8", True))
    batch.q4_exact_grouped_gate = bool(
        getattr(generator, "q4_exact_grouped_gate", True))
    batch.q4_exact_grouped_all = bool(
        getattr(generator, "q4_exact_grouped_all", True))
    batch.q4_out4 = bool(getattr(generator, "q4_out4", True))
    # The Q4 DP4A decode pair is request-local too. ``None`` on both fields is
    # the direct/unbound fallback; a bound pair travels as-is so a batch c1
    # segment never reads the ambient process keys the generator has replaced.
    batch.q4_dp4a_decode_enabled = getattr(
        generator, "q4_dp4a_decode_enabled", None)
    batch.q4_dp4a_decode_layers = getattr(
        generator, "q4_dp4a_decode_layers", None)
    batch.q4_tile_m = getattr(generator, "q4_tile_m", None)
    batch.q4_tile_n = getattr(generator, "q4_tile_n", None)
    batch.q8_wmma_tiles = getattr(generator, "q8_wmma_tiles", None)
    batch.q51_down_m1 = bool(getattr(generator, "q51_down_m1", False))
    batch.q51_fold128_prefill = bool(
        getattr(generator, "q51_fold128_prefill", False))
    batch.q51_fold_pair_prefill = bool(
        getattr(generator, "q51_fold_pair_prefill", False))
    batch.forkb_grouped_down = bool(
        getattr(generator, "forkb_grouped_down", False))
    batch.q5k_row4_prefill = bool(getattr(generator, "q5k_row4_prefill", False))
    batch.q8_down_row4_prefill = bool(
        getattr(generator, "q8_down_row4_prefill", False))
    batch.q8_down_bundle_prefill = bool(
        getattr(generator, "q8_down_bundle_prefill", False))
    return batch


__all__ = [
    "QWEN4_EXP_BATCH_SLOT_POSITIONS",
    "QWEN4_EXP_DEFAULT_REQUEST_CAPACITY",
    "QWEN4_EXP_MAX_BATCH_ROWS",
    "QWEN4_EXP_STAGING_RUNNERS",
    "Qwen4ExpARBatchOwner",
    "Qwen4ExpARBatchStepError",
    "batch_slot_positions",
    "make_qwen4_exp_ar_batch_runner",
]
