#!/usr/bin/env python3
"""Containment proof: one row's in-step failure must not close the engine service.

The engine service owns one shared driver. Before this unit, any exception that
escaped a driver tick reached ``except BaseException: self._fail_all(exc)``:
every active request was aborted and the service closed, so later requests failed
with ``engine service is closed``. The repair adds a runner-classified
containment path (``contain_execution_failure`` -> ``ExecutionFailure``) that the
loop applies to a step's own pre-device validation failures.

This harness proves both halves on the real resident GGUF path, in-process:

1. **Reference** - one greedy request to completion, recording its token IDs.
2. **Containment** - stream a second request, and once its first chunk proves
   prefill finished, patch ``raise_if_generation_deadline_expired`` so that row's
   next *decode* prologue raises ``GenerationCancelled``. That is the class the
   classifier contains: the step's own validation, before any kernel launch for
   that step. The healthy co-resident must then complete with token IDs identical
   to the reference, the injected row must report a cancellation, the service
   must still report ``status == ok``, and a follow-up request must complete.
3. **Empty device phase** - the same fault in the decode step's own
   post-device check on a tick whose device phase has no packed work (every
   row's first token came from prefill).  Nothing in that tick can have
   advanced a row's device state, so the claim is ``none`` and names only the
   failing row: the co-resident must still complete with reference token IDs
   and the service must stay ``ok``.
4. **Post-device containment** - the same fault on a later call, in a decode
   step's own post-device check, so the failed step has already run its packed
   kernels.  The first decode after prefill does no device work (the row's
   first token comes from prefill), so the fault must land past it: the default
   call index is the second decode step's post-device check.  The claim is
   ``partial`` and names the packed step's rows; the service must still report
   ``status == ok``, a follow-up request must complete, and the co-resident
   must either be retired with the group or complete with reference token IDs -
   never continue with different ones.
5. **Control** - repeat the pre-device fault with ``contain_execution_failure``
   forced to return None and confirm the same fault closes the service and fails
   the co-resident. That is the pre-repair behaviour, so the run is a real
   RED/GREEN rather than a demonstration of a path that could not fail.

Not a throughput measurement: ``performance_claim`` is false. Greedy,
artifact-scoped, default host settings.

Usage:

    python3 scripts/gguf_execution_failure_containment_proof.py \
        --model /home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --backend hip_gfx1151 --out /tmp/containment-proof.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine import LLM, SamplingParams  # noqa: E402
from hipengine.generation import (  # noqa: E402
    EngineService,
    GenerationCancelled,
    GenerationExecutionFailed,
    GenerationRequest,
)


class _Fault:
    """Raise once in the armed row's Nth resident-step prologue call.

    A row's prefill prologue calls ``raise_if_generation_deadline_expired`` at
    most twice (once inside a packed prefill batch, once in the per-row loop),
    and every later call is a decode step's own check.  Firing on a call after
    those therefore lands in a decode step deterministically, without depending
    on stream timing.
    """

    def __init__(self) -> None:
        self.prompt: tuple[int, ...] | None = None
        self.armed = False
        self.matches_seen = 0
        self.fire_on_match = 3
        self.fired = 0
        self.fired_in: str | None = None

    def arm(self, prompt: Sequence[int], *, fire_on_match: int = 3) -> None:
        self.prompt = tuple(int(token) for token in prompt)
        self.matches_seen = 0
        self.fire_on_match = max(1, int(fire_on_match))
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def should_fire(self, request: Any) -> bool:
        if not self.armed or self.prompt is None:
            return False
        prompts = getattr(request, "prompts", None)
        if not prompts:
            return False
        if tuple(int(token) for token in prompts[0]) != self.prompt:
            return False
        self.matches_seen += 1
        return self.matches_seen >= self.fire_on_match


class _ClassifierProbe:
    """Record what the runner's failure classifier saw for one injected fault."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.quiesce_calls: list[dict[str, Any]] = []

    def install(self) -> dict[str, Any]:
        from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

        originals = {
            "contain_execution_failure": Qwen35GGUFResidentModelRunner.contain_execution_failure,
            "_quiesce_after_execution_failure": Qwen35GGUFResidentModelRunner._quiesce_after_execution_failure,
        }

        def contain(self, error, *, phase, request_ids, work_kind):
            record = {
                "error_type": type(error).__name__,
                "phase": phase,
                "request_ids": list(request_ids),
                "work_kind": work_kind,
                "mutation_window": getattr(self, "_execution_mutation_window", "<absent>"),
                "step_request_id": getattr(self, "_execution_step_request_id", "<absent>"),
                "row_present": [
                    int(request_id) in self._rows for request_id in request_ids
                ],
                "runtime_paths": self._runtime_paths(request_ids),
            }
            result = originals["contain_execution_failure"](
                self,
                error,
                phase=phase,
                request_ids=request_ids,
                work_kind=work_kind,
            )
            record["contained"] = result is not None
            record["contained_request_ids"] = (
                None if result is None else list(result.request_ids)
            )
            record["mutation"] = None if result is None else str(result.mutation)
            record["stepped_request_ids"] = sorted(
                int(request_id)
                for request_id in getattr(self, "_execution_stepped_request_ids", ())
            )
            self._probe.calls.append(record)
            return result

        def quiesce(self, request_ids):
            outcome = originals["_quiesce_after_execution_failure"](self, request_ids)
            self._probe.quiesce_calls.append(
                {
                    "request_ids": list(request_ids),
                    "runtimes": self._runtime_paths(request_ids),
                    "synchronized": bool(outcome),
                }
            )
            return outcome

        def runtime_paths(self, request_ids):
            found: dict[str, Any] = {}
            for request_id in request_ids:
                row = self._rows.get(int(request_id))
                if row is None:
                    found[str(request_id)] = "row-absent"
                    continue
                slot = getattr(row, "slot", None)
                session = getattr(slot, "session", None)
                lease = getattr(row, "lease", None)
                found[str(request_id)] = {
                    "row_type": type(row).__name__,
                    "slot_type": type(slot).__name__ if slot is not None else None,
                    "slot_attrs": [
                        name
                        for name in dir(slot)
                        if "runtime" in name.lower() or "session" in name.lower()
                    ]
                    if slot is not None
                    else [],
                    "session_type": type(session).__name__ if session is not None else None,
                    "session_runtime": type(getattr(session, "runtime", None)).__name__
                    if session is not None
                    else None,
                    "lease_session_runtime": type(
                        getattr(getattr(lease, "session", None), "runtime", None)
                    ).__name__,
                    "row_attrs": [
                        name
                        for name in dir(row)
                        if "runtime" in name.lower() or "session" in name.lower() or "lease" in name.lower()
                    ],
                }
            shared = getattr(self, "_shared_runner", None)
            found["shared_runner"] = {
                "type": type(shared).__name__ if shared is not None else None,
                "has_runtime": hasattr(shared, "_runtime"),
            }
            return found

        Qwen35GGUFResidentModelRunner.contain_execution_failure = contain
        Qwen35GGUFResidentModelRunner._quiesce_after_execution_failure = quiesce
        Qwen35GGUFResidentModelRunner._runtime_paths = runtime_paths
        Qwen35GGUFResidentModelRunner._probe = self
        return originals

    @staticmethod
    def restore(originals: Mapping[str, Any]) -> None:
        from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

        for name, value in originals.items():
            setattr(Qwen35GGUFResidentModelRunner, name, value)
        for name in ("_runtime_paths", "_probe"):
            if name in Qwen35GGUFResidentModelRunner.__dict__:
                delattr(Qwen35GGUFResidentModelRunner, name)


def _install_fault(fault: _Fault) -> Any:
    """Patch the resident step prologue's deadline/cancellation check."""

    from hipengine.generation import qwen35_gguf as module

    original = module.raise_if_generation_deadline_expired

    def patched(request_or_deadline: Any, **kwargs: Any) -> None:
        if fault.should_fire(request_or_deadline):
            fault.armed = False
            fault.fired += 1
            fault.fired_in = sys._getframe(1).f_code.co_name
            raise GenerationCancelled()
        return original(request_or_deadline, **kwargs)

    module.raise_if_generation_deadline_expired = patched
    return original


def _prompt_tokens(prompt_token: int) -> tuple[int, ...]:
    return (int(prompt_token),) * 8


def _request(prompt_token: int, *, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=(_prompt_tokens(prompt_token),),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _service(llm: LLM) -> EngineService:
    generator = llm._get_text_generator()
    if not isinstance(generator, EngineService):
        raise SystemExit("this proof requires the shared engine service driver")
    return generator


def _reference_tokens(llm: LLM, prompt_token: int, *, max_tokens: int) -> tuple[int, ...]:
    handle = _service(llm).submit_child(_request(prompt_token, max_tokens=max_tokens))
    output = handle.result(timeout=900.0)
    return tuple(int(token) for token in (output.generated_token_ids or ()))


def _describe(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "is_cancellation": isinstance(exc, GenerationCancelled),
        "is_execution_failure": isinstance(exc, GenerationExecutionFailed),
        "phase": getattr(exc, "phase", None),
        "mutation": getattr(exc, "mutation", None),
        "cause_type": getattr(exc, "cause_type", None),
    }


def _run_scenario(
    llm: LLM,
    *,
    prompt_token: int,
    neighbor_token: int,
    max_tokens: int,
    fire_on_match: int = 3,
    probe: _ClassifierProbe | None = None,
) -> dict[str, Any]:
    service = _service(llm)
    fault = _Fault()
    original = _install_fault(fault)
    originals = probe.install() if probe is not None else None
    result: dict[str, Any] = {}
    try:
        try:
            healthy = service.submit_child(_request(prompt_token, max_tokens=max_tokens))
            injected = service.submit_child(_request(neighbor_token, max_tokens=max_tokens))
        except BaseException as exc:
            # A previous scenario closed the service; report it instead of
            # aborting the run, because that outcome is the control's evidence.
            result["submit_error"] = _describe(exc)
            result["health_at_start"] = service.health()
            return result
        fault.arm(_prompt_tokens(neighbor_token), fire_on_match=fire_on_match)
        try:
            output = injected.result(timeout=900.0)
            result["injected_error"] = None
            result["injected_tokens"] = [int(t) for t in (output.generated_token_ids or ())]
        except BaseException as exc:
            result["injected_error"] = _describe(exc)
        result["fault_fired"] = fault.fired
        result["fault_fired_in"] = fault.fired_in
        result["fault_matches_seen"] = fault.matches_seen
        result["health_after_fault"] = service.health()
        try:
            output = healthy.result(timeout=900.0)
            result["healthy_tokens"] = [int(t) for t in (output.generated_token_ids or ())]
            result["healthy_error"] = None
        except BaseException as exc:
            result["healthy_tokens"] = []
            result["healthy_error"] = _describe(exc)
        try:
            follow_up = service.submit_child(_request(prompt_token, max_tokens=max_tokens))
            follow_up_output = follow_up.result(timeout=900.0)
            result["follow_up_tokens"] = [
                int(t) for t in (follow_up_output.generated_token_ids or ())
            ]
            result["follow_up_error"] = None
        except BaseException as exc:
            result["follow_up_tokens"] = []
            result["follow_up_error"] = _describe(exc)
        result["health_final"] = service.health()
        if probe is not None:
            result["classifier_calls"] = probe.calls
            result["quiesce_calls"] = probe.quiesce_calls
        return result
    finally:
        fault.disarm()
        from hipengine.generation import qwen35_gguf as module

        module.raise_if_generation_deadline_expired = original
        if originals is not None:
            _ClassifierProbe.restore(originals)


def _run_control(
    llm: LLM,
    *,
    prompt_token: int,
    neighbor_token: int,
    max_tokens: int,
    fire_on_match: int = 3,
) -> dict[str, Any]:
    """Pre-repair behaviour: the same fault must close the service."""

    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    original_contain = Qwen35GGUFResidentModelRunner.contain_execution_failure
    Qwen35GGUFResidentModelRunner.contain_execution_failure = (
        lambda self, error, *, phase, request_ids, work_kind: None
    )
    try:
        return _run_scenario(
            llm,
            prompt_token=prompt_token,
            neighbor_token=neighbor_token,
            max_tokens=max_tokens,
            fire_on_match=fire_on_match,
        )
    finally:
        Qwen35GGUFResidentModelRunner.contain_execution_failure = original_contain

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="auto")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--prompt-token", type=int, default=9707)
    parser.add_argument(
        "--fire-on-match",
        type=int,
        default=3,
        help="fire the fault on this matching prologue call (prefill uses at most two)",
    )
    parser.add_argument(
        "--empty-phase-match",
        type=int,
        default=4,
        help=(
            "fire the empty-device-phase scenario on this matching call; the "
            "first decode after prefill is the one whose device phase has no "
            "packed work"
        ),
    )
    parser.add_argument(
        "--post-device-match",
        type=int,
        default=6,
        help=(
            "fire the second scenario on this matching call; the first decode "
            "after prefill does no device work, so the second decode step's "
            "post-device check is the first marked post-device call"
        ),
    )
    parser.add_argument(
        "--skip-post-device",
        action="store_true",
        help="run only the pre-device containment and control scenarios",
    )
    parser.add_argument("--neighbor-token", type=int, default=374)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="run only the containment scenario (the control closes the service)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="record the runner's own classification inputs (mutation window, rows, runtimes)",
    )
    args = parser.parse_args(argv)

    sampling = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    started = time.perf_counter()
    llm = LLM(str(args.model), backend=str(args.backend), quant=str(args.quant))
    prepared = llm.prepare(
        max_sequence_length=int(args.max_context_tokens),
        sampling_params=sampling,
    )
    payload: dict[str, Any] = {
        "protocol": "execution-failure-containment-proof",
        "model": str(args.model),
        "backend": str(args.backend),
        "quant": str(args.quant),
        "prepared_context_tokens": prepared,
        "max_tokens": int(args.max_tokens),
        "performance_claim": False,
    }
    try:
        reference = _reference_tokens(
            llm, int(args.prompt_token), max_tokens=int(args.max_tokens)
        )
        payload["reference_tokens"] = list(reference)
        probe = _ClassifierProbe() if args.probe else None
        payload["contained"] = _run_scenario(
            llm,
            prompt_token=int(args.prompt_token),
            neighbor_token=int(args.neighbor_token),
            max_tokens=int(args.max_tokens),
            fire_on_match=int(args.fire_on_match),
            probe=probe,
        )
        print(
            json.dumps({"contained": payload["contained"]}, indent=2, sort_keys=True),
            flush=True,
        )
        if not args.skip_post_device:
            payload["contained_empty_device_phase"] = _run_scenario(
                llm,
                prompt_token=int(args.prompt_token),
                neighbor_token=int(args.neighbor_token),
                max_tokens=int(args.max_tokens),
                fire_on_match=int(args.empty_phase_match),
                probe=_ClassifierProbe(),
            )
            print(
                json.dumps(
                    {"contained_empty_device_phase": payload["contained_empty_device_phase"]},
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            payload["contained_post_device"] = _run_scenario(
                llm,
                prompt_token=int(args.prompt_token),
                neighbor_token=int(args.neighbor_token),
                max_tokens=int(args.max_tokens),
                fire_on_match=int(args.post_device_match),
                probe=_ClassifierProbe(),
            )
            print(
                json.dumps(
                    {"contained_post_device": payload["contained_post_device"]},
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        if not args.skip_control:
            payload["control_no_containment"] = _run_control(
                llm,
                prompt_token=int(args.prompt_token),
                neighbor_token=int(args.neighbor_token),
                max_tokens=int(args.max_tokens),
                fire_on_match=int(args.fire_on_match),
            )
            print(
                json.dumps(
                    {"control_no_containment": payload["control_no_containment"]},
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        try:
            llm._get_text_generator().close()
        except BaseException:
            pass
    payload["elapsed_s"] = round(time.perf_counter() - started, 3)

    contained = payload["contained"]
    control = payload.get("control_no_containment") or {}
    injected_error = contained.get("injected_error") or {}
    control_error = control.get("injected_error") or {}
    checks = {
        "fault_fired_in_decode": contained.get("fault_fired_in") == "decode_batch",
        "injected_row_reported": contained.get("injected_error") is not None,
        "injected_row_not_an_engine_fault": not bool(
            injected_error.get("is_execution_failure")
        ),
        "healthy_matches_reference": contained.get("healthy_tokens") == list(reference),
        "service_healthy_after_fault": contained.get("health_after_fault", {}).get(
            "status"
        )
        == "ok",
        "follow_up_completed": contained.get("follow_up_error") is None,
        "follow_up_matches_reference": contained.get("follow_up_tokens") == list(reference),
    }
    if not args.skip_control:
        checks["control_closed_service"] = (
            control.get("health_final", {}).get("status") == "unhealthy"
        )
        checks["control_failed_healthy_row"] = (
            control.get("healthy_error") is not None
            or control.get("submit_error") is not None
        )
        # Without containment the fault closes the service, and the refusal is
        # reported the way the profile contract requires: a fatal
        # ``GenerationExecutionFailed`` naming the phase, the affected rows and
        # the deepest cause, with an ``unknown`` mutation class.
        checks["control_reports_the_refused_scope"] = (
            bool(control_error.get("is_execution_failure"))
            and control_error.get("mutation") == "unknown"
            and control_error.get("phase") == "decode"
            and control_error.get("cause_type") == "GenerationCancelled"
        )
    if not args.skip_post_device:
        empty_phase = payload.get("contained_empty_device_phase") or {}
        empty_claims = [
            call
            for call in empty_phase.get("classifier_calls", ())
            if call.get("contained")
        ]
        empty_claim = empty_claims[-1] if empty_claims else {}
        checks["empty_phase_fault_fired_in_decode"] = (
            empty_phase.get("fault_fired_in") == "decode_batch"
        )
        checks["empty_phase_claim_is_none"] = empty_claim.get("mutation") == "none"
        checks["empty_phase_names_the_failing_row"] = list(
            empty_claim.get("contained_request_ids") or ()
        ) == [empty_claim.get("step_request_id")] and empty_claim.get(
            "step_request_id"
        ) is not None
        checks["empty_phase_healthy_matches_reference"] = empty_phase.get(
            "healthy_tokens"
        ) == list(reference)
        checks["service_healthy_after_empty_phase_fault"] = (
            empty_phase.get("health_after_fault", {}).get("status") == "ok"
        )
        checks["empty_phase_follow_up_completed"] = (
            empty_phase.get("follow_up_error") is None
        )
        checks["empty_phase_follow_up_matches_reference"] = (
            empty_phase.get("follow_up_tokens") == list(reference)
        )
        post_device = payload.get("contained_post_device") or {}
        claims = [
            call
            for call in post_device.get("classifier_calls", ())
            if call.get("contained")
        ]
        claim = claims[-1] if claims else {}
        scope = tuple(claim.get("contained_request_ids") or ())
        stepped = set(claim.get("stepped_request_ids") or ())
        checks["post_device_fault_fired_in_decode"] = (
            post_device.get("fault_fired_in") == "decode_batch"
        )
        checks["post_device_claim_is_partial"] = claim.get("mutation") == "partial"
        # The scope comes from the runner's own device accounting: every row it
        # names was marked as possibly having advanced before the failure.
        checks["post_device_scope_is_the_stepped_group"] = bool(scope) and set(
            scope
        ) <= stepped
        checks["service_healthy_after_post_device_fault"] = (
            post_device.get("health_after_fault", {}).get("status") == "ok"
        )
        checks["post_device_follow_up_completed"] = (
            post_device.get("follow_up_error") is None
        )
        checks["post_device_follow_up_matches_reference"] = (
            post_device.get("follow_up_tokens") == list(reference)
        )
        # A peer the claim named is retired with the group; a peer it left
        # unnamed must still match the reference.  Neither may silently
        # continue from an advanced position.
        checks["post_device_peer_outcome_is_consistent"] = bool(
            post_device.get("healthy_error") is not None
            or post_device.get("healthy_tokens") == list(reference)
        )
    payload["checks"] = checks
    payload["passed"] = all(checks.values())
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
