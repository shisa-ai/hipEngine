#!/usr/bin/env python3
"""Live acceptance for forced tokens on the sampled MTP route.

A forced token is a caller-level override: the autoregressive route emits the
queue head instead of drawing from the row. A speculative cycle publishes
several tokens at once, so the override has to reach the accept walk as a row's
law -- a point mass on the forced token -- and the live queue may only be
consumed for the tokens a cycle actually published. The failures this gate
guards against are a route that ignores the queue (publishing the model's own
continuation), one that consumes the whole queue per cycle, one that consumes it
twice, and one that leaks the queue into another request.

Both arms run the same requests through ``hipengine.LLM`` -- the route is the
only difference -- and the autoregressive arm is the reference for every token
after the forced prefix.

Checks:

* ``forced_prefix_published`` -- the forced tokens appear at the head of the
  output, in order, exactly once, on the speculative arm.
* ``continuation_parity`` -- on a greedy case (``temperature=0``) every token the
  speculative arm published equals the autoregressive arm's token at the same
  position, so the forced prefix did not disturb the continuation and the queue
  was consumed exactly once. A sampled case cannot be compared token for token:
  the accept walk draws one uniform per verified row where the autoregressive
  route draws one per step, so the two arms consume the request's stream at
  different rates and publish different draws from the same law (the induced-law
  gate measures that law). Those cases require the forced prefix to match
  exactly and record the shared prefix length.
* ``force_sequence_completion`` -- a force-sequence whose first token the model
  is about to emit anyway is completed by the route (the model's own token, then
  the queued remainder), and the autoregressive arm publishes the same ids.
* ``queue_isolated`` -- an unforced request after the forced one publishes the
  same ids as the unforced seed request, so no queue state survived the request.
* ``route_ran`` -- the speculative arm's telemetry reports the MTP execution path
  with real cycles while the autoregressive arm reports a plain decode path, so
  no comparison is between two AR runs.

Usage:
    .venv/bin/python scripts/mtp_forced_token_gate.py \
        --model ~/models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --json benchmarks/results/<date>-<cell>-mtp-forced-tokens.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MTP_EXECUTION_PATH = "gguf_specdec2_mtp2"

# Repetitive text keeps the draft chains long, so a forced queue of two or three
# tokens has to be consumed across more than one row of the same cycle.
PROMPT = (
    "Count slowly, one number per word: one two three four five six seven "
    "eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty twenty-one twenty-two"
)
SEED_PROMPT = "Write one short sentence about rivers."


@dataclass(frozen=True)
class Case:
    """One forced-token request, expressed in text and encoded per arm."""

    name: str
    purpose: str
    forced_text: str = ""
    force_sequence_text: tuple[str, ...] = ()
    prompt: str = PROMPT
    max_tokens: int = 24
    temperature: float = 0.0
    seed: int = 3
    extras: dict[str, Any] = field(default_factory=dict)


CASES: tuple[Case, ...] = (
    Case(
        name="forced_two_tokens",
        purpose=(
            "A two-token queue must be emitted in order at the head of the "
            "output and consumed exactly once."
        ),
        forced_text=" zeta omega",
    ),
    Case(
        name="forced_three_tokens_greedy",
        purpose=(
            "A three-token queue is longer than a single verified chain, so the "
            "cycle consumes it across its rows and the next cycle continues."
        ),
        forced_text=" zeta omega kappa",
        max_tokens=32,
    ),
    Case(
        name="forced_tokens_sampled",
        purpose=(
            "The sampled branch (temperature > 0) applies the same override: the "
            "forced prefix is emitted and the continuation is the AR one."
        ),
        forced_text=" zeta omega",
        temperature=0.7,
        seed=11,
    ),
    Case(
        name="force_sequence_completion",
        purpose=(
            "A partially matched force-sequence queues its remainder from inside "
            "the cycle that emitted the matching token."
        ),
        # The first element is the case prompt's own first token, taken from an
        # unforced probe of that prompt, so the partial match is guaranteed to
        # trigger on the token the model is about to emit anyway.
        force_sequence_text=("{probe_token}", " kappa", " lambda"),
    ),
)


def _case_payload(case: Case, *, tokens: dict[str, Any]) -> dict[str, Any]:
    return {**case.extras, **tokens}


def _forced_ids(
    llm: Any, case: Case, probe_token_id: int | None
) -> dict[str, Any]:
    """Return the case's sampler fields with text encoded for this arm's tokenizer."""

    fields: dict[str, Any] = {}
    if case.forced_text:
        ids = tuple(int(token) for token in llm.tokenize(case.forced_text))
        if not ids:
            raise RuntimeError(f"case {case.name} forced text encoded to no tokens")
        fields["forced_tokens_pending"] = ids
        fields["forced_token_reason"] = "gate"
    if case.force_sequence_text:
        if probe_token_id is None:
            raise RuntimeError(
                f"case {case.name} needs the probe token before it can encode"
            )
        sequence: list[int] = [int(probe_token_id)]
        for text in case.force_sequence_text[1:]:
            sequence.extend(int(token) for token in llm.tokenize(text))
        fields["force_sequence_completion_token_sequences"] = (tuple(sequence),)
        fields["force_sequence_completion_reason"] = "gate"
    return fields


def _telemetry(output: Any) -> dict[str, Any]:
    telemetry = getattr(output, "telemetry", None)
    state = None if telemetry is None else getattr(telemetry, "decode_state", None)
    return {
        "execution_path": None if state is None else getattr(state, "execution_path", None),
        "sampler_mode": None if state is None else getattr(state, "sampler_mode", None),
        "forced_tokens_pending": (
            [] if state is None else list(getattr(state, "forced_tokens_pending", ()) or ())
        ),
        "forced_token_id": None if state is None else getattr(state, "forced_token_id", None),
        "forced_tokens_remaining": (
            None if state is None else getattr(state, "forced_tokens_remaining", None)
        ),
        "diagnostics": (
            None if telemetry is None else dict(getattr(telemetry, "diagnostics", None) or {})
        ),
    }


def _run(
    llm: Any, *, prompt: str, fields: dict[str, Any], case: Case, mtp: bool
) -> dict[str, Any]:
    """Run one request on this arm's route.

    The speculative arm goes through ``generate_speculative_mtp_detailed``, the
    entry point a caller uses to ask for the model-owned MTP route; the
    autoregressive arm goes through plain ``generate_detailed``. They are the
    same split the server makes per request.
    """

    from hipengine.llm import SamplingParams

    params = SamplingParams(
        max_tokens=int(case.max_tokens),
        temperature=float(case.temperature),
        seed=int(case.seed),
        **fields,
    )
    run = llm.generate_speculative_mtp_detailed if mtp else llm.generate_detailed
    outputs = run([prompt], params)
    if len(outputs) != 1:
        raise RuntimeError(f"expected one output, got {len(outputs)}")
    output = outputs[0]
    return {
        "text": output.text,
        "ids": [int(token) for token in (output.generated_token_ids or ())],
        "telemetry": _telemetry(output),
    }


MIN_FREE_GIB = 40.0


def _preflight_memory() -> float:
    """Return available GiB, refusing to load a model that will not fit.

    Loading this artifact takes about 17 GiB of GTT, which is host RAM. An
    unbounded KV context on top of it is what turns a gate run into a machine
    OOM: the automatic sizing picks the model's own maximum, and at BF16 that is
    tens of GiB. The check is a guard against that arithmetic, not a tuning
    knob.
    """

    available_kib = None
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break
    if available_kib is None:
        raise RuntimeError("could not read MemAvailable from /proc/meminfo")
    available_gib = available_kib / 1024**2
    if available_gib < MIN_FREE_GIB:
        raise RuntimeError(
            f"only {available_gib:.1f} GiB of RAM is available and this gate loads "
            f"a model of about 17 GiB per arm; free memory (another resident model "
            f"holding GTT counts) or lower --max-sequence-length. Minimum: "
            f"{MIN_FREE_GIB:.0f} GiB."
        )
    return available_gib


def _run_arm(
    model: str, *, serving: str, mtp: bool, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Run every case on one arm and return its per-case observations."""

    from hipengine.llm import LLM

    # Checked per arm: the two arms are separate model loads, and the first arm's
    # release is only observable as the second arm's headroom.
    available_gib = _preflight_memory()
    print(f"[gate] {serving}: {available_gib:.1f} GiB of RAM available", flush=True)
    llm = LLM(model, speculative_mtp_serving=serving, **kwargs)
    try:
        # The unforced seed run supplies the force-sequence trigger token and the
        # reference an unforced request must still match after a forced one.
        seed = _run(
            llm,
            prompt=SEED_PROMPT,
            fields={},
            case=Case(name="seed", purpose="", prompt=SEED_PROMPT, max_tokens=16),
            mtp=mtp,
        )
        seed_token_id = int(seed["ids"][0]) if seed["ids"] else None
        # The constructor's max_sequence_length is a declaration about the
        # resident session. Assert the session was sized to it: auto-selection
        # takes the largest context that fits, which at BF16 is tens of GiB of
        # host memory and can exhaust the machine.
        observations: dict[str, Any] = {
            "seed": seed,
            "cases": {},
            "declared_context_tokens": int(kwargs["max_sequence_length"]),
            "resident_context_tokens": _resident_context_tokens(llm),
        }
        for case in CASES:
            probe = None
            probe_token_id = None
            if case.force_sequence_text:
                # An unforced run of the case's own prompt: its first token is the
                # partial match the force-sequence is built to extend.
                probe = _run(
                    llm,
                    prompt=case.prompt,
                    fields={},
                    case=Case(
                        name=f"{case.name}_probe",
                        purpose="",
                        prompt=case.prompt,
                        max_tokens=16,
                    ),
                    mtp=mtp,
                )
                if not probe["ids"]:
                    raise RuntimeError(f"probe for {case.name} published no token")
                probe_token_id = int(probe["ids"][0])
            fields = _forced_ids(llm, case, probe_token_id)
            observations["cases"][case.name] = {
                # Token ids are tuples and stay lists; a reason is a string and
                # must not be split into characters.
                "fields": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in fields.items()
                },
                "probe": probe,
                "request": _run(
                    llm, prompt=case.prompt, fields=fields, case=case, mtp=mtp
                ),
            }
        # The unforced request after the forced ones must be unaffected.
        observations["after"] = _run(
            llm,
            prompt=SEED_PROMPT,
            fields={},
            case=Case(name="after", purpose="", prompt=SEED_PROMPT, max_tokens=16),
            mtp=mtp,
        )
        observations["tokenizer"] = {
            "seed_token_text": (
                None if seed_token_id is None else llm.detokenize([seed_token_id])
            )
        }
        return observations
    finally:
        llm.close()


def _check(observations: dict[str, Any], *, arm: str) -> dict[str, Any]:
    """Assert the per-arm invariants and return the arm's report."""

    seed = observations["seed"]
    assert (
        observations["resident_context_tokens"] == observations["declared_context_tokens"]
    ), {
        "the_resident_session_was_not_sized_to_the_declared_context": {
            "declared": observations["declared_context_tokens"],
            "resident": observations["resident_context_tokens"],
        }
    }
    if arm == "mtp":
        assert seed["telemetry"]["execution_path"] == MTP_EXECUTION_PATH, {
            "unforced_seed_did_not_speculate": seed["telemetry"]
        }
    else:
        assert seed["telemetry"]["execution_path"] != MTP_EXECUTION_PATH, {
            "autoregressive_arm_speculated": seed["telemetry"]
        }
    report: dict[str, Any] = {
        "seed": seed,
        "cases": {},
        # Recorded so the artifact shows the session was sized to the declaration
        # rather than to whatever automatic selection would have chosen.
        "declared_context_tokens": observations["declared_context_tokens"],
        "resident_context_tokens": observations["resident_context_tokens"],
    }
    for case in CASES:
        entry = observations["cases"][case.name]
        request = entry["request"]
        forced = list(entry["fields"].get("forced_tokens_pending") or [])
        published = [int(token) for token in request["ids"]]
        if forced:
            assert published[: len(forced)] == forced, {
                "forced_prefix_not_published": {
                    "case": case.name,
                    "forced": forced,
                    "published": published,
                }
            }
        report["cases"][case.name] = {
            "purpose": case.purpose,
            "fields": entry["fields"],
            "ids": published,
            "text": request["text"],
            "telemetry": request["telemetry"],
            "probe_ids": None if entry["probe"] is None else entry["probe"]["ids"],
        }
    report["after"] = observations["after"]
    report["tokenizer"] = observations["tokenizer"]
    return report


def _compare(arms: dict[str, Any]) -> dict[str, Any]:
    """Assert the cross-arm checks and return the gate's acceptance record."""

    mtp = arms["mtp"]
    ar = arms["ar"]
    acceptance: dict[str, Any] = {}
    assert mtp["seed"]["ids"] == ar["seed"]["ids"], {
        "unforced_seed_differs_between_arms": {
            "mtp": mtp["seed"]["ids"],
            "ar": ar["seed"]["ids"],
        }
    }
    assert mtp["after"]["ids"] == ar["after"]["ids"], {
        "unforced_request_after_a_forced_one_differs_between_arms": {
            "mtp": mtp["after"]["ids"],
            "ar": ar["after"]["ids"],
        }
    }
    assert mtp["after"]["ids"] == mtp["seed"]["ids"], {
        "queue_leaked_into_the_next_request": {
            "after": mtp["after"]["ids"],
            "seed": mtp["seed"]["ids"],
        }
    }
    acceptance["queue_isolated"] = (
        "The unforced request run after every forced case published the same ids "
        "as the unforced seed request on the speculative arm, and the same ids as "
        "the autoregressive arm."
    )

    parity_rows: list[dict[str, Any]] = []
    for case in CASES:
        mtp_case = mtp["cases"][case.name]
        ar_case = ar["cases"][case.name]
        shared = min(len(mtp_case["ids"]), len(ar_case["ids"]))
        assert shared > 0, {"case_published_nothing": {"case": case.name}}
        common = mtp_case["ids"][:shared] == ar_case["ids"][:shared]
        shared_prefix = next(
            (
                index
                for index in range(shared)
                if mtp_case["ids"][index] != ar_case["ids"][index]
            ),
            shared,
        )
        forced = list(mtp_case["fields"].get("forced_tokens_pending") or [])
        sequences = mtp_case["fields"].get("force_sequence_completion_token_sequences") or []
        if case.temperature <= 0.0:
            assert common, {
                "published_ids_differ_from_the_autoregressive_arm": {
                    "case": case.name,
                    "mtp": mtp_case["ids"],
                    "ar": ar_case["ids"],
                    "shared": shared,
                }
            }
        # A sampled case is a different draw from the same law, so its
        # continuation is not comparable. Its forced prefix still is, and the
        # per-arm checks below already require each arm to publish exactly the
        # queued tokens, which is the same statement position by position.
        if forced:
            assert mtp_case["ids"][: len(forced)] == forced, {
                "forced_prefix_not_published": {"case": case.name}
            }
        if mtp_case["probe_ids"] is not None:
            assert mtp_case["probe_ids"] == ar_case["probe_ids"], {
                "unforced_probe_differs_between_arms": {
                    "case": case.name,
                    "mtp": mtp_case["probe_ids"],
                    "ar": ar_case["probe_ids"],
                }
            }
        if sequences:
            sequence = [int(token) for token in sequences[0]]
            assert mtp_case["ids"][: len(sequence)] == sequence, {
                "force_sequence_not_completed": {
                    "case": case.name,
                    "sequence": sequence,
                    "published": mtp_case["ids"],
                }
            }
        parity_rows.append(
            {
                "case": case.name,
                "temperature": case.temperature,
                "forced_tokens": len(forced),
                "force_sequence_tokens": len(sequences[0]) if sequences else 0,
                "shared_prefix": shared_prefix,
                "identical_output": common,
                "published": len(mtp_case["ids"]),
                "mtp_execution_path": mtp_case["telemetry"]["execution_path"],
                "ar_execution_path": ar_case["telemetry"]["execution_path"],
            }
        )
        if case.name == "force_sequence_completion":
            acceptance["force_sequence_completion"] = (
                f"The model's own seed token plus the two queued tokens were "
                f"published as {sequence} on both arms, so the remainder was "
                "queued and emitted from inside the cycle."
            )
        assert mtp_case["telemetry"]["execution_path"] == MTP_EXECUTION_PATH, {
            "speculative_case_did_not_speculate": {
                "case": case.name,
                "telemetry": mtp_case["telemetry"],
            }
        }
        assert ar_case["telemetry"]["execution_path"] != MTP_EXECUTION_PATH, {
            "autoregressive_case_speculated": {
                "case": case.name,
                "telemetry": ar_case["telemetry"],
            }
        }
    acceptance["forced_prefix_published"] = (
        "Every forced case published its queued tokens at the head of the output, "
        "in order, on both arms."
    )
    greedy_rows = [row for row in parity_rows if row["temperature"] <= 0.0]
    sampled_rows = [row for row in parity_rows if row["temperature"] > 0.0]
    acceptance["continuation_parity"] = (
        f"Every greedy case published the autoregressive arm's exact ids "
        f"({len(greedy_rows)} cases), so the queue was consumed exactly once and "
        f"the continuation was not disturbed. The {len(sampled_rows)} sampled "
        "case(s) published the forced prefix on both arms; their continuations "
        "are different draws from the same law because the accept walk consumes "
        "the request's uniform stream per verified row."
    )
    acceptance["route_ran"] = (
        f"Every speculative case reported {MTP_EXECUTION_PATH} while every "
        "autoregressive case reported a plain decode path."
    )
    return {"acceptance": acceptance, "rows": parity_rows}


def _resident_context_tokens(llm: Any) -> int | None:
    """The context resident sizing chose for this engine.

    ``resident_capacity_estimate`` prices a session at a context the caller
    passes, so it reports the model's maximum when asked for the headroom. The
    sizing decision itself is the generator's: a caller's declaration when there
    is one, the automatic selection otherwise.
    """

    generator = getattr(llm, "_text_generator", None)
    if generator is None:
        getter = getattr(llm, "_get_text_generator", None)
        generator = getter() if callable(getter) else None
    resolved = getattr(generator, "resident_context_tokens", None)
    return None if resolved is None else int(resolved)


def _write_json(path_text: str | None, report: dict[str, Any]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[gate] wrote {path}", flush=True)


def _single_arm_report(args: Any, arm_report: dict[str, Any], *, arm: str) -> dict[str, Any]:
    """The artifact shape for a single-arm run: this arm's own invariants only."""

    return {
        "scope": (
            "forced tokens and force-sequence completion on the sampled MTP route "
            f"({arm} arm only)"
        ),
        "model": args.model,
        "kv_storage": args.kv_storage,
        "backend": args.backend,
        "serving": args.mtp_serving if arm == "mtp" else args.ar_serving,
        "arm": arm,
        "max_sequence_length": int(args.max_sequence_length),
        "cross_arm_comparison": "not run: this invocation loaded one arm",
        "cases": [
            {
                "name": case.name,
                "purpose": case.purpose,
                "prompt": case.prompt,
                "max_tokens": case.max_tokens,
                "temperature": case.temperature,
                "seed": case.seed,
            }
            for case in CASES
        ],
        "summary": {
            "cases": len(CASES),
            "forced_cases": sum(1 for case in CASES if case.forced_text),
            "force_sequence_cases": sum(1 for case in CASES if case.force_sequence_text),
            "seed_ids": len(arm_report["seed"]["ids"]),
        },
        "arm_report": arm_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--kv-storage", default="bf16")
    parser.add_argument("--json", default=None)
    parser.add_argument("--mtp-serving", default="auto")
    parser.add_argument("--ar-serving", default="off")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="resident KV context, pinned explicitly (default 4096)",
    )
    parser.add_argument(
        "--arm",
        choices=("both", "mtp", "ar"),
        default="both",
        help="run one arm only and report its own invariants (default both)",
    )
    args = parser.parse_args(argv)


    kwargs: dict[str, Any] = {
        "backend": args.backend,
        "kv_storage": args.kv_storage,
        "prefix_cache": "off",
        "max_sequence_length": int(args.max_sequence_length),
    }
    print(f"[gate] running the speculative arm (serving={args.mtp_serving})", flush=True)
    mtp_raw = _run_arm(args.model, serving=args.mtp_serving, mtp=True, kwargs=kwargs)
    mtp = _check(mtp_raw, arm="mtp")
    if args.arm == "mtp":
        _write_json(args.json, _single_arm_report(args, mtp, arm="mtp"))
        print("[gate] PASS (speculative arm only)", flush=True)
        return 0
    print(f"[gate] running the autoregressive arm (serving={args.ar_serving})", flush=True)
    ar_raw = _run_arm(args.model, serving=args.ar_serving, mtp=False, kwargs=kwargs)
    ar = _check(ar_raw, arm="ar")
    if args.arm == "ar":
        _write_json(args.json, _single_arm_report(args, ar, arm="ar"))
        print("[gate] PASS (autoregressive arm only)", flush=True)
        return 0
    comparison = _compare({"mtp": mtp, "ar": ar})

    for row in comparison["rows"]:
        print(
            f"[gate] {row['case']:28s} forced={row['forced_tokens']} "
            f"seq={row['force_sequence_tokens']} published={row['published']:3d} "
            f"shared={row['shared_prefix']:3d} mtp={row['mtp_execution_path']}",
            flush=True,
        )
    report = {
        "scope": "forced tokens and force-sequence completion on the sampled MTP route",
        "model": args.model,
        "kv_storage": args.kv_storage,
        "backend": args.backend,
        "mtp_serving": args.mtp_serving,
        "ar_serving": args.ar_serving,
        "passed": True,
        "cases": [
            {
                "name": case.name,
                "purpose": case.purpose,
                "prompt": case.prompt,
                "max_tokens": case.max_tokens,
                "temperature": case.temperature,
                "seed": case.seed,
            }
            for case in CASES
        ],
        "seed_prompt": SEED_PROMPT,
        "seed_token_text": mtp["tokenizer"]["seed_token_text"],
        "summary": {
            "cases": len(CASES),
            "forced_cases": sum(1 for case in CASES if case.forced_text),
            "force_sequence_cases": sum(1 for case in CASES if case.force_sequence_text),
            "mtp_seed_ids": len(mtp["seed"]["ids"]),
        },
        "acceptance": comparison["acceptance"],
        "rows": comparison["rows"],
        "arms": {"mtp": mtp, "ar": ar},
    }
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1) + "\n")
        print(f"[gate] wrote {path}", flush=True)
    print("[gate] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
