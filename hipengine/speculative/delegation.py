"""Shared runner-to-adapter staged-lifecycle delegation.

The dense Qwen GGUF and PARO resident runners forward the Generation-2 staged
lifecycle to a resolved :class:`~hipengine.speculative.provider.StagedTargetAdapter`
through exactly this shape.  Before this module the two runner classes carried
~150 duplicated lines of resolve-adapter / raise-if-missing / delegate methods
that differed only in their missing-adapter error string, whether each call ran
under the host's HIP target-arch environment, per-cycle bookkeeping, and one
recover hook.

The mixin implements the delegation once.  A runner provides:

* ``_staged_adapter()`` -- resolve the current adapter or return ``None``;
* ``_staged_adapter_missing_error()`` -- the runner's named error for a required
  stage reached without an adapter;
* ``_staged_adapter_arch_environment()`` -- optional context manager wrapped
  around every device-touching stage (cold plan/claim queries stay unwrapped);
* ``_begin_staged_adapter_cycle(plan)`` -- optional per-cycle bookkeeping fired
  before the mutating stages;
* ``_on_staged_adapter_cycle_recovered(plan, error, recovered)`` -- optional
  hook after ``recover_cycle_failure``.

Hooks are consulted only through ``getattr``-safe defaults, so a runner
without an override keeps working.  Owner-specific surfaces (opaque
``execute_speculative_cycle``, KV liveness-owner derivation, row telemetry,
graph availability, recovery rebuilds) stay in the owning runner class.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from hipengine.speculative.provider import (
    StagedTargetAdapter,
    validate_staged_speculative_runner,
)


class StagedAdapterDelegation:
    """Mixin forwarding the scheduler's staged lifecycle to a target adapter."""

    # ---------------------------------------------------------- overridables

    def _staged_adapter(self) -> StagedTargetAdapter | None:
        """Resolve the current staged adapter, or ``None`` when absent."""

        raise NotImplementedError("staged runner must resolve its adapter")

    def _staged_adapter_missing_error(self) -> str:
        """The named error a required stage raises without an adapter."""

        return "staged target adapter is unavailable"

    def _staged_adapter_arch_environment(self) -> Any:
        """Context manager around device-touching stages (default: none)."""

        return nullcontext(None)

    def _begin_staged_adapter_cycle(self, plan) -> None:
        """Per-cycle bookkeeping fired before the mutating stages."""

        return None

    def _on_staged_adapter_cycle_recovered(self, plan, error, recovered: bool) -> None:
        """Hook after ``recover_cycle_failure`` (default: no-op)."""

        return None

    # ------------------------------------------------------- required stages

    def _required_staged_adapter(self) -> StagedTargetAdapter:
        adapter = self._staged_adapter()
        if adapter is None:
            raise RuntimeError(self._staged_adapter_missing_error())
        return adapter

    def staged_speculative_owner(self):
        """Declare this runner's scheduler-facing staged speculative lifecycle.

        The Generation-2 loop resolves the owner through
        ``resolve_staged_speculative_runner`` and validates it before driving
        one bounded target/provider cycle, so an incomplete owner fails loudly
        instead of silently falling back to autoregressive decoding.
        """

        return validate_staged_speculative_runner(self)

    def speculative_capability(self, request_semantics):
        adapter = self._staged_adapter()
        return None if adapter is None else adapter.capability(request_semantics)

    def speculative_claims_fit(self, plan) -> bool:
        adapter = self._staged_adapter()
        return bool(adapter is not None and adapter.claims_fit(plan))

    def speculative_frontier_available(self, plan) -> bool:
        adapter = self._staged_adapter()
        return bool(
            adapter is not None
            and getattr(adapter, "staged_frontier", True)
        )

    def speculative_component_claims(self, plan):
        return self._required_staged_adapter().component_claims(plan)

    def reserve_speculative_claims(self, claims):
        return self._required_staged_adapter().reserve_claims(claims)

    def release_speculative_claims(self, reservation) -> None:
        adapter = self._required_staged_adapter()
        with self._staged_adapter_arch_environment():
            adapter.release_claims(reservation)

    def prepare_speculative_requests(self, plan, request_semantics, *, stream=None) -> None:
        self._begin_staged_adapter_cycle(plan)
        adapter = self._required_staged_adapter()
        with self._staged_adapter_arch_environment():
            adapter.prepare_requests(plan, request_semantics, stream=stream)

    def propose_speculative_batch(self, plan, request_semantics, *, stream=None):
        self._begin_staged_adapter_cycle(plan)
        adapter = self._required_staged_adapter()
        with self._staged_adapter_arch_environment():
            return adapter.propose_batch(plan, request_semantics, stream=stream)

    def execute_target_frontier(
        self,
        plan,
        frontier,
        complete_claims,
        *,
        commit: bool,
        cancelled_request_ids,
    ):
        self._begin_staged_adapter_cycle(plan)
        adapter = self._required_staged_adapter()
        with self._staged_adapter_arch_environment():
            return adapter.execute_target_frontier(
                plan,
                frontier,
                complete_claims,
                commit=commit,
                cancelled_request_ids=cancelled_request_ids,
            )

    # ------------------------------------------------- optional-stage forwarders

    def prepare_speculative_k0(self, plan, request_semantics, *, stream=None) -> None:
        """Best-effort K0 catch-up; an absent adapter is not an error."""

        self._begin_staged_adapter_cycle(plan)
        adapter = self._staged_adapter()
        if adapter is not None:
            with self._staged_adapter_arch_environment():
                adapter.prepare_k0(plan, request_semantics, stream=stream)

    def note_speculative_ar_commit(self, request_id, reason=None) -> None:
        """Forward one emitted autoregressive token to the resolved adapter."""

        adapter = self._staged_adapter()
        observe = (
            None
            if adapter is None
            else getattr(adapter, "note_speculative_ar_commit", None)
        )
        if callable(observe):
            with self._staged_adapter_arch_environment():
                observe(request_id, reason)

    def rollback_speculative_cycle(self, plan, candidate_graph, error) -> None:
        adapter = self._staged_adapter()
        if adapter is not None:
            with self._staged_adapter_arch_environment():
                adapter.rollback_cycle(plan, candidate_graph, error)

    def recover_speculative_cycle_failure(self, plan, error) -> bool:
        adapter = self._staged_adapter()
        if adapter is None:
            return False
        with self._staged_adapter_arch_environment():
            recovered = bool(adapter.recover_cycle_failure(plan, error))
        self._on_staged_adapter_cycle_recovered(plan, error, recovered)
        return recovered


__all__ = ["StagedAdapterDelegation"]
