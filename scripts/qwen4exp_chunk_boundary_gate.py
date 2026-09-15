"""Canonical boundary numerics and full payload parity for larger chunks.

This is c1 runner reuse, not c2 serving isolation or a performance benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import fcntl

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata,
)
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_summary
from scripts.qwen4exp_q8_repair_depth_gate import (
    observe_prefill_chunks, resolve_allocation_profile, validate_chunk_allocation,
)
from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits

LENGTHS = (2047, 2048, 2049, 2051, 2052, 4095, 4097)


def boundary_cases(fixture):
    sources = {row["id"]: row for row in fixture["cases"]}
    cases = []
    for category in ("code", "general_ja"):
        source = sources[f"{category}-p4096"]["prompt_token_ids"]
        if len(source) != 4096:
            raise ValueError("boundary source must contain exactly4096 tokens")
        for length in LENGTHS:
            # One extra ordinary source token extends the last boundary.
            tokens = (list(source) + [source[-1]])[:length]
            cases.append(dict(id=f"{category}-boundary{length}", category=category,
                              prompt_token_ids=tokens, prompt_tokens=length))
    return cases


def full_state(runner):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    runner.runtime.device_synchronize()

    def read(buffer):
        raw = np.empty(buffer.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(raw), buffer, runtime=runner.runtime)
        return raw

    kv = hashlib.sha256()
    kv_bytes = 0
    kv_finite = True
    for attention in runner.attention_states:
        for buffer in (attention.key_cache, attention.value_cache):
            raw = read(buffer)
            kv.update(raw)
            kv_bytes += raw.nbytes
            # This runner's key/value buffers use BF16 storage.
            kv_finite &= bool(((raw.view(np.uint16) & 0x7F80) != 0x7F80).all())
    index = hashlib.sha256()
    index_bytes = 0
    index_finite = True
    for state in runner.index_states:
        raw = read(state.raw_keys).view(np.float32).reshape(state.capacity, state.index_dim)
        live = np.ascontiguousarray(raw[state.physical_positions_host[:state.count]])
        pooled = read(state.pooled_keys).view(np.float32).reshape(-1, state.index_dim)
        pooled = np.ascontiguousarray(pooled[:state.pooled_count])
        for values in (live, pooled):
            index.update(values.tobytes())
            index_bytes += values.nbytes
            index_finite &= bool(np.isfinite(values).all())
    return dict(recurrent=_state_summary(runner), full_kv_sha256=kv.hexdigest(),
                full_kv_bytes=kv_bytes, live_index_sha256=index.hexdigest(),
                full_kv_finite=kv_finite,
                live_index_bytes=index_bytes, live_index_finite=index_finite)


def trajectory(runner, prompt, *, steps, chunk, teacher=None):
    with observe_prefill_chunks(runner, tokens=len(prompt), size=chunk) as chunks:
        result = runner.prefill(prompt)
        prefill = full_state(runner)
        logits = [result.logits.copy()]
        tokens = [int(result.token_id)]
        for step in range(steps):
            result = runner.step(tokens[-1] if teacher is None else teacher[step])
            logits.append(result.logits.copy())
            tokens.append(int(result.token_id))
    return np.stack(logits), tokens, {
        "prefill": prefill, "final": full_state(runner), "chunks": chunks}


def control_state(payload):
    return {
        "recurrent": {key: value for key, value in payload["recurrent"].items()
                      if key not in ("state_sha256", "finite")},
        "full_kv_bytes": payload["full_kv_bytes"],
        "live_index_bytes": payload["live_index_bytes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--allocation-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--chunk-size", type=int, choices=(2048, 4096), default=2048)
    args = parser.parse_args()
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        parser.error("boundary capture requires clean tracked source")
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    from hipengine.core.memory import memory_stats

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
        cases = boundary_cases(fixture)
        if args.case_id:
            selected = set(args.case_id)
            if len(selected) != len(args.case_id) or not selected <= {c["id"] for c in cases}:
                raise ValueError("invalid boundary case selection")
            cases = [case for case in cases if case["id"] in selected]
        host, model = _host_metadata(), model_identity(args.model_root)
        allocation = json.loads(args.allocation_evidence.read_bytes())
        profile = resolve_allocation_profile()
        validate_chunk_allocation(allocation, chunk=args.chunk_size, context=4352,
                                  manifest=profile.manifest_sha256, host=host, model=model)
        report = dict(
            status="running", source=source, host=host, model=model, command=sys.argv,
            fixture_sha256=fixture_hash, allocation_evidence_sha256=hashlib.sha256(
                args.allocation_evidence.read_bytes()).hexdigest(),
            protocol=dict(steps=64, candidate_repeats=3, capacity=4352,
                          strict_chunk=1024, candidate_chunk=args.chunk_size, complete_matrix=not bool(args.case_id)),
            lifecycle={}, cases=[], performance_claim=False, promotion_claim=False,
            limitations=["c1 reuse only; not c2 isolation, cancellation, MTP or native-context admission.",
                         "Full KV buffers and live raw/pooled index keys are hashed; temporary index scores are not.",
                         "Payload exactness is recorded separately from the production numerical gate."])
        teachers, states, reference_logits, descriptors = {}, {}, [], []
        output_logits = []
        try:
            for arm, chunk in (("strict", 1024), ("production", args.chunk_size)):
                arm_args = SimpleNamespace(**vars(args), max_sequence_length=4352,
                                           prefill_chunk_size=chunk)
                generator, resolved, _ = _make_generator(arm_args, arm)
                report[arm + "_manifest"] = resolved.manifest_sha256
                try:
                    for case in cases:
                        prompt = case["prompt_token_ids"]
                        if arm == "strict":
                            logits, tokens, payload = trajectory(
                                generator.runner, prompt, steps=64, chunk=chunk)
                            teachers[case["id"]] = tokens
                            states[case["id"]] = payload
                            reference_logits.extend(logits)
                            for step, token in enumerate(tokens):
                                descriptors.append(RowDescriptor(
                                    scenario_id=f"chunk{args.chunk_size}-boundaries",
                                    scenario_step=len(descriptors), request_id=case["id"],
                                    teacher_step=step, category=case["category"],
                                    shape=f"p{len(prompt)}_" + ("prefill_last" if step == 0 else "c1"),
                                    transition="prefill_to_c1" if step == 0 else "steady",
                                    teacher_token_id=token))
                        else:
                            repeats, state_runs, intervening = [], [], []
                            for repeat in range(3):
                                if repeat:
                                    # Exercise prior-request scratch before reconstructing the target prefix.
                                    noise = list(reversed(prompt[:257]))
                                    with observe_prefill_chunks(
                                        generator.runner, tokens=len(noise), size=chunk,
                                    ) as noise_chunks:
                                        generator.runner.prefill(noise)
                                    intervening.append(noise_chunks)
                                logits, _, payload = trajectory(
                                    generator.runner, prompt, steps=64, chunk=chunk,
                                    teacher=teachers[case["id"]])
                                repeats.append(logits)
                                state_runs.append(payload)
                                print(arm, case["id"], repeat, flush=True)
                            output_logits.extend(repeats[0])
                            parity = all(
                                payload[phase] == states[case["id"]][phase]
                                for payload in state_runs for phase in ("prefill", "final"))
                            control_exact = all(
                                control_state(payload[phase]) == control_state(states[case["id"]][phase])
                                for payload in state_runs for phase in ("prefill", "final"))
                            finite = all(
                                payload[phase]["recurrent"]["finite"]
                                and payload[phase]["live_index_finite"]
                                and payload[phase]["full_kv_finite"]
                                for payload in [states[case["id"]], *state_runs]
                                for phase in ("prefill", "final"))
                            report["cases"].append(dict(
                                id=case["id"], prompt_tokens=len(prompt),
                                prompt_sha256=hashlib.sha256(
                                    json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
                                deterministic=all(np.array_equal(repeats[0], values)
                                                  for values in repeats[1:])
                                and all(payload == state_runs[0] for payload in state_runs[1:]),
                                payload_exact_vs_strict=parity,
                                control_exact=control_exact, state_finite=finite,
                                intervening_prefill_chunks=intervening,
                                strict_payload=states[case["id"]], candidate_payloads=state_runs))
                        if arm == "strict":
                            print(arm, case["id"], flush=True)
                finally:
                    generator.close()
                    report["lifecycle"][arm] = memory_stats()
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
            report["quality"] = compare_profile_logits(
                np.stack(reference_logits), np.stack(output_logits), descriptors)
            report["deterministic"] = all(case["deterministic"] for case in report["cases"])
            report["payload_exact_vs_strict"] = all(
                case["payload_exact_vs_strict"] for case in report["cases"])
            report["state_gate"] = {
                "passed": all(case["control_exact"] and case["state_finite"]
                              for case in report["cases"])}
            report["status"] = "completed"
            if (_git_metadata(ROOT) != source
                    or any(row["current_allocated_bytes"] for row in report["lifecycle"].values())):
                report["status"] = "invalid_source_or_lifecycle"
        except BaseException as error:
            report["status"] = "invalid_capture"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
