"""A/B the TP2 bulk-prefill source-F16 owner, in either of its two forms.

Two modes, because the candidate was measured in two different ways and only one
of them was ever committed:

``--mode owner``
    Toggles ``MlpTP2GenerationSession.use_t16_f16_rocblas_prefill`` on the real
    route. This is the integrated measurement, and the mode to use when quoting
    the +3.97% (490.0 -> 471.3 ms). It also reports the scratch the arm actually
    allocated and whether the owner actually built, so a run where the owner
    silently fell back cannot be mistaken for a run where it executed.

``--mode context``
    Injects a resident session's context managers around the per-rank attention
    call. This is the older probe, kept because it isolates which context carries
    the win (the answer is ``_q6_f16_rocblas_prefill_context``, and
    ``_q6_integer_mmq_context`` alone gives nothing). It is *not* the integrated
    path: it builds resident sessions and reaches into their contexts.

Three things this gets right that earlier throwaway probes did not.

**Per-rank entry.** Each context carries that rank's own f16 planes, so it must
be entered *inside* the per-device loop around that rank's work. Entering one
rank's context around both ranks' work makes rank 1 read rank 0's planes and
faults the GPU (``Memory access fault ... Page not present``). The wrapper here
reads the current device and enters only that device's context.

**Warm both arms.** Each arm is run once unmeasured before it is measured. A
cold arm pays first-call costs (rocBLAS handle, f16 plane allocation) that would
otherwise be charged to the arm itself.

**Counterbalanced order.** The arms alternate A/B/B/A rather than A/B/A/B, so a
monotone drift across the run cannot favour one arm. Both per-round paired
deltas and the raw samples are reported, and the drift between the first and
last measurement of each arm is the check on whether the ordering worked.

Usage::

    # all five contexts vs none (the full candidate)
    python scripts/tp2_prefill_context_ab.py --arm-b all5 --rounds 4

    # just the f16-rocBLAS context (the measured source of the win)
    python scripts/tp2_prefill_context_ab.py --arm-b _q6_f16_rocblas_prefill_context \\
        --rounds 4 --json /tmp/context-ab.json

    # the honest control: the same context list twice, to measure the noise floor
    python scripts/tp2_prefill_context_ab.py --arm-b all5 --arm-a all5 --rounds 3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The contexts the single-card bulk prefill installs and the TP2 path omits, in
# the order that route enters them.
CONTEXTS = (
    "_prefill_f16_staging_context",
    "_q6_integer_mmq_context",
    "_iq_dense_mmq_context",
    "_q8_mmq_prefill_context",
    "_q6_f16_rocblas_prefill_context",
)
ALL5 = "all5"
NONE = "none"


class _CountingContext:
    """Wrap the owner context so builds are counted, not assumed.

    A fallback is a normal outcome, so the flag being on is not evidence the
    owner executed. This counts the contexts that actually install an owner by
    reading the contextvar the context sets.
    """

    def __init__(self, inner: Any, module: Any, counter: list[int]) -> None:
        self._inner = inner
        self._module = module
        self._counter = counter

    def __enter__(self) -> Any:
        entered = self._inner.__enter__()
        if self._module._t16_f16_rocblas_prefill_session.get() is not None:
            self._counter[0] += 1
        return entered

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        default="owner",
        choices=("owner", "context"),
        help=(
            "owner = toggle the integrated TP2 flag (the measurement to quote); "
            "context = inject a resident session's contexts (the older probe)"
        ),
    )
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--logits-rows", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=4, help="A/B pairs; the total is 2x this")
    parser.add_argument(
        "--arm-a",
        default=NONE,
        help=f"context mode: {NONE}, {ALL5}, or a context name from {CONTEXTS}; "
        f"owner mode: {NONE} or {ALL5}",
    )
    parser.add_argument(
        "--arm-b",
        default=ALL5,
        help=f"{NONE}, {ALL5}, or one context name from: {', '.join(CONTEXTS)}",
    )
    parser.add_argument("--reduce-mode", default="device")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    def check_arm(spec: str, label: str) -> str:
        if spec in (NONE, ALL5):
            return spec
        if args.mode == "context" and spec in CONTEXTS:
            return spec
        if args.mode == "owner":
            raise SystemExit(
                f"--arm-{label.lower()} must be {NONE} or {ALL5} in owner mode; got {spec!r}. "
                "Individual context names are a context-mode arm."
            )
        raise SystemExit(
            f"--arm-{label.lower()} must be {NONE}, {ALL5}, or one of {CONTEXTS}; got {spec!r}"
        )

    arm_a_spec = check_arm(args.arm_a, "A")
    arm_b_spec = check_arm(args.arm_b, "B")
    if arm_a_spec == arm_b_spec:
        print(
            f"both arms are {arm_a_spec!r}: this is an A/A run measuring the protocol's "
            "noise floor, not an effect"
        )

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
    import hipengine.runtime.qwen35_gguf_runner as runner_module

    runtime = get_hip_runtime()
    devices = tuple(int(part) for part in str(args.devices).split(","))
    session = MlpTP2GenerationSession(
        args.model,
        devices=devices,
        mode="tp2",
        max_sequence_length=int(args.max_sequence_length),
        bulk_prefill=True,
        bulk_prefill_rows=int(args.prompt_tokens),
        reduce_mode=str(args.reduce_mode),
    )
    prompt = [9707] * int(args.prompt_tokens)
    residents: dict[int, Any] = {}
    state: dict[str, str] = {"arm": NONE}

    def arm_contexts(owner: Any, spec: str) -> list[Any]:
        if spec == NONE:
            return []
        if spec == ALL5:
            return [getattr(owner, name)() for name in CONTEXTS]
        if spec == "_q6_f16_rocblas_prefill_context":
            return [owner._q6_f16_rocblas_prefill_context(request_rows=int(args.prompt_tokens))]
        return [getattr(owner, spec)()]

    @contextlib.contextmanager
    def enter(device: int):
        with contextlib.ExitStack() as stack:
            for context in arm_contexts(residents[device], state["arm"]):
                stack.enter_context(context)
            yield

    # Wrap the two attention entry points; both are called per device, so the
    # current device selects whose context is entered.
    for name in (
        "_run_linear_attention_prefill_attn_rows",
        "_run_full_attention_prefill_attn_rows",
    ):
        original = getattr(runner_module.Qwen35GGUFFullStackRunner, name)

        def make(original: Any) -> Any:
            def wrapper(self: Any, *a: Any, **kw: Any) -> Any:
                device = runtime.get_device()
                if state["arm"] != NONE and device in residents:
                    with enter(device):
                        return original(self, *a, **kw)
                return original(self, *a, **kw)

            return wrapper

        setattr(runner_module.Qwen35GGUFFullStackRunner, name, make(original))

    owner_builds = [0]
    if args.mode == "owner":
        _original_owner_context = session._rank_f16_rocblas_owner_context

        def counting_owner_context(device: int, rows: int):
            # Count only contexts that actually carry an owner: a fallback
            # yields the exact T16 session and is not evidence the owner ran.
            import hipengine.runtime.gguf_linear as _gl

            ctx = _original_owner_context(device, rows)
            return _CountingContext(ctx, _gl, owner_builds)

        session._rank_f16_rocblas_owner_context = counting_owner_context

    def run() -> float:
        runtime.device_synchronize()
        started = time.perf_counter()
        session.bulk_prefill(prompt, logits_rows=int(args.logits_rows))
        runtime.device_synchronize()
        return (time.perf_counter() - started) * 1000.0

    evidence: dict[str, dict] = {}

    def apply_arm(spec: str) -> None:
        """Select one arm, and record what it actually allocated and built."""

        # Reset the build counter here, before the run, not after the snapshot.
        # Resetting after the snapshot let the warmup's builds be attributed to
        # the first measured arm, which reported 128 owner builds for an arm
        # whose owner was disabled.
        owner_builds[0] = 0
        if args.mode == "context":
            state["arm"] = spec
            return
        session.use_t16_f16_rocblas_prefill = spec != NONE
        # Drop the cached planes so this arm allocates its own, and so a
        # previous arm's allocation cannot be read as this one's.
        session._release_rank_f16_rocblas_planes()
        state["arm"] = NONE

    def snapshot_arm(label: str) -> None:
        if args.mode != "owner":
            return
        planes = [p for p in session._rank_f16_rocblas_planes.values() if p is not None]
        ready = [bool(v) for v in session._rank_f16_rocblas_ready.values()]
        evidence[label] = {
            "enabled": bool(session.use_t16_f16_rocblas_prefill),
            "planes_allocated_per_rank": [int(getattr(p, "rows", 0)) for p in planes],
            "scratch_mib_per_rank": round(
                sum(
                    int(getattr(getattr(p, name), "nbytes", 0))
                    for p in planes
                    for name in ("q6_f16_x", "q6_f16_weight", "q6_f16_out")
                )
                / len(planes)
                / 2**20,
                2,
            ) if planes else 0.0,
            "rank_ready": ready,
            "owner_builds": owner_builds[0],
            "handle_present": sorted(
                device for device, handle in session._rank_f16_rocblas.items() if handle is not None
            ),
        }

    try:
        session.bulk_prefill(prompt, logits_rows=int(args.logits_rows))  # graph/JIT warmup
        if args.mode == "context":
            for device in session.devices:
                with scoped_current_device(runtime, device):
                    residents[device] = runner_module.Qwen35GGUFResidentSession(
                        args.model,
                        runtime=runtime,
                        backend="hip_gfx1100",
                        max_sequence_length=int(args.max_sequence_length),
                        shared_runner=session._runners[device],
                    )

        # Warm each arm once unmeasured, so neither pays first-call setup
        # (rocBLAS handle, f16 plane allocation) inside a measured sample.
        for spec in {arm_a_spec, arm_b_spec}:
            apply_arm(spec)
            run()

        samples: dict[str, list[float]] = {"A": [], "B": []}
        # Counterbalanced: A/B/B/A, so monotone drift cannot favour one arm.
        for pair in range(int(args.rounds)):
            order = ("A", "B") if pair % 2 == 0 else ("B", "A")
            for label in order:
                apply_arm(arm_a_spec if label == "A" else arm_b_spec)
                samples[label].append(run())
                if pair == 0:
                    snapshot_arm(label)
    finally:
        for resident in residents.values():
            try:
                resident.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                pass
        session.close()

    a, b = samples["A"], samples["B"]
    median_a, median_b = statistics.median(a), statistics.median(b)
    paired = [100.0 * (x / y - 1.0) for x, y in zip(a, b)]
    report = {
        "schema": 1,
        "kind": "tp2_prefill_context_ab",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": args.model,
        "devices": list(devices),
        "prompt_tokens": int(args.prompt_tokens),
        "logits_rows": int(args.logits_rows),
        "reduce_mode": str(args.reduce_mode),
        "mode": args.mode,
        "arm_a": arm_a_spec,
        "arm_b": arm_b_spec,
        "arm_evidence": evidence,
        "identical_arms": arm_a_spec == arm_b_spec,
        "rounds": int(args.rounds),
        "arm_a_samples_ms": [round(v, 3) for v in a],
        "arm_b_samples_ms": [round(v, 3) for v in b],
        "arm_a_median_ms": round(median_a, 3),
        "arm_b_median_ms": round(median_b, 3),
        "throughput_delta_percent": round(100.0 * (median_a / median_b - 1.0), 3),
        "paired_deltas_percent": [round(v, 3) for v in paired],
        "paired_all_same_sign": all(v > 0 for v in paired) or all(v < 0 for v in paired),
        "drift_a_percent": round(100.0 * (a[0] / a[-1] - 1.0), 3) if len(a) > 1 else 0.0,
        "drift_b_percent": round(100.0 * (b[0] / b[-1] - 1.0), 3) if len(b) > 1 else 0.0,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")

    print(f"  A {arm_a_spec:<34} {[round(v, 1) for v in a]}  median {median_a:.1f} ms")
    print(f"  B {arm_b_spec:<34} {[round(v, 1) for v in b]}  median {median_b:.1f} ms")
    print(f"  B vs A: {report['throughput_delta_percent']:+.2f}% throughput")
    print(f"  paired per-round deltas: {report['paired_deltas_percent']}  "
          f"same sign: {report['paired_all_same_sign']}")
    print(f"  drift within arm A {report['drift_a_percent']:+.2f}%, "
          f"within arm B {report['drift_b_percent']:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
