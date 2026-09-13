#!/usr/bin/env python3
"""Exact scalar/native context gate for logits, state, KV, commit and rollback."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_native_context_gate import fixed_prompt, native_context_override


def read_bytes(runtime, ptr, nbytes):
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    host = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(ptr, nbytes), nbytes, runtime=runtime)
    return host


def state_fingerprint(session):
    owner = session._target_scratch_owner
    runtime = session.runtime
    runtime.device_synchronize()
    hashes = {"position": session.position}
    for family in ("layer_conv_states", "layer_recurrent_states", "full_key_caches", "full_value_caches"):
        for layer, buffer in enumerate(getattr(owner, family)):
            if buffer is None:
                continue
            nbytes = buffer.nbytes
            if family.startswith("full_"):
                nbytes = session.position * (buffer.nbytes // owner.max_positions)
            hashes[f"{family}/{layer}"] = hashlib.sha256(read_bytes(runtime, buffer.ptr, nbytes)).hexdigest()
    hidden = session.last_target_hidden
    hashes["hidden"] = hashlib.sha256(read_bytes(runtime, hidden.ptr, hidden.shape[1] * 2)).hexdigest()
    return hashes


def compare_state(actual, expected):
    changed = [key for key in expected if actual.get(key) != expected[key]]
    if changed:
        raise AssertionError(f"state mismatch: {changed[:12]}")


def make_candidates(greedy, accepted, vocab_size):
    candidates = list(greedy)
    if accepted < len(candidates):
        candidates[accepted] = (candidates[accepted] + 1) % vocab_size
    return candidates


def run(args, rows):
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier, _StateJournal
    from hipengine.runtime.gguf_native_spec_cycle import (
        build_native_b2_target_batch, _native_target_graph_context_limit,
    )
    from hipengine.speculative import TargetCommitPlan
    from scripts.qwen36_dense_gguf_suite import (
        Qwen35GGUFTokenizer, load_gguf_index, load_prompt_rows, build_chat_prompt,
    )

    version = None if args.compiler_version_file is None else args.compiler_version_file.read_text()
    reset_memory_stats()
    package = importlib.import_module(f"hipengine.kernels.{args.backend}")
    with ExitStack() as stack:
        stack.enter_context(native_context_override(package, args.native_context_limit))
        target = stack.enter_context(Qwen35GGUFResidentSession(
            args.model, backend=args.backend, max_sequence_length=args.capacity,
            compiler_version=version, require_cached_build=args.require_cached_build,
            use_wmma_prefill=True, use_gemv_decode=True,
        ))
        target.select_prefill_quant(args.quant)
        reference = stack.enter_context(Qwen35GGUFResidentSession(
            args.model, backend=args.backend, max_sequence_length=args.capacity,
            shared_runner=target.runner, compiler_version=version,
            require_cached_build=args.require_cached_build,
            use_wmma_prefill=True, use_gemv_decode=True,
        ))
        verifier = stack.enter_context(Qwen35GGUFTransactionalVerifier(
            target, max_candidate_budget=max(args.budgets), quant=args.quant, target_verify_mode="native",
        ))
        target._ensure_verify_block_buffers(max(args.budgets) + 1, runtime=target.runtime)
        target._ensure_verify_linear_state_row_buffers(max(args.budgets) + 1, runtime=target.runtime)
        checkpoints = []
        for session in (target, reference):
            journal = _StateJournal.allocate(session, max_rows=1, initial_state_only=True)
            stack.callback(journal.close)
            checkpoints.append(journal)
        tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(args.model))
        natural = load_prompt_rows(args.prompts)
        prompt_sources = [("repeat", [9707])]
        prompt_sources += [(p["id"], list(build_chat_prompt(tokenizer, p["prompt"], reasoning="off")))
                           for p in natural[:args.prompt_limit]]
        transaction = 0

        def restore(session, journal, position):
            journal.restore_initial()
            session._target_scratch_owner.set_full_attention_position(position, session.runtime)
            session._position = position
            session.runtime.device_synchronize()

        for prompt_id, seed in prompt_sources:
            for length in args.contexts:
                prompt = fixed_prompt(seed, length)
                roots = []
                for session, journal in zip((target, reference), checkpoints):
                    session.reset()
                    roots.append(int(session.prefill(prompt, use_bulk=args.bulk_prefill).token_id))
                    journal.capture_initial(force_consumer_state=True)
                    session.runtime.device_synchronize()
                if roots[0] != roots[1]:
                    raise AssertionError("independent prefill roots differ")
                initial = state_fingerprint(reference)
                compare_state(state_fingerprint(target), initial)
                root = roots[0]
                greedy = []
                token = root
                for _ in range(max(args.budgets)):
                    token = int(reference.step(token).token_id)
                    greedy.append(token)
                restore(reference, checkpoints[1], length)

                for budget in args.budgets:
                    for accepted in sorted({0, budget // 2, budget}):
                        candidates = make_candidates(greedy[:budget], accepted, target.runner.vocab_size)
                        inputs = [root, *candidates]
                        oracle_logits = []
                        oracle_states = []
                        for token in inputs:
                            result = reference.step(token, return_logits=True)
                            if not np.isfinite(result.logits).all():
                                raise AssertionError("scalar reference logits are not finite")
                            oracle_logits.append(result.logits.copy())
                            oracle_states.append(state_fingerprint(reference))
                        restore(reference, checkpoints[1], length)
                        following_logits = None
                        following_state = None
                        next_token = int(np.argmax(oracle_logits[accepted]))
                        if args.following_step:
                            for token in inputs[:accepted + 1]:
                                reference.step(token)
                            following_logits = reference.step(next_token, return_logits=True).logits.copy()
                            following_state = state_fingerprint(reference)
                            restore(reference, checkpoints[1], length)

                        transports = ("graph", "eager") if args.graph_first else ("eager", "graph")
                        logit_modes = (False, True) if args.n2_first else (True, False)
                        for transport in transports:
                            for diagnostic_logits in logit_modes:
                                for repetition in range(args.repetitions):
                                    restore(target, checkpoints[0], length)
                                    batch = build_native_b2_target_batch(inputs, start_position=length, request_id=0)
                                    bucket = verifier.graph_bucket(("state-gate", budget), batch)
                                    expected_graph_extent = _native_target_graph_context_limit(target, rows=budget + 1)
                                    transaction += 1
                                    prepared = verifier.prepare(
                                        batch, transaction_id=transaction, graph_bucket=bucket,
                                        remaining_decode=(budget + 1,), return_logits=diagnostic_logits,
                                        allow_graph=transport == "graph",
                                    )
                                    required_native = args.native_context_limit or args.require_native_through
                                    if length + budget + 1 <= required_native:
                                        if prepared.target_verify_mode != "native":
                                            raise AssertionError("native coverage silently fell back")
                                        graph_required = (
                                            not args.allow_graph_transition_fallback
                                            or expected_graph_extent is not None
                                        )
                                        if transport == "graph" and graph_required and not prepared.native_graph_submitted:
                                            raise AssertionError("graph coverage silently fell back")
                                    if transport == "eager" and prepared.native_graph_submitted:
                                        raise AssertionError("eager verification reported a stale graph submission")
                                    if prepared.summary.accepted_counts != (accepted,):
                                        raise AssertionError("acceptance differs from forced scalar chain")
                                    if not prepared.gpu_accept_match_cpu:
                                        raise AssertionError("GPU acceptance differs from CPU oracle")
                                    logit_hash = None
                                    if diagnostic_logits:
                                        expected = np.concatenate(oracle_logits, axis=0)
                                        np.testing.assert_array_equal(prepared.target_logits, expected)
                                        logit_hash = hashlib.sha256(expected.tobytes()).hexdigest()
                                    summary = prepared.summary
                                    plan = TargetCommitPlan(
                                        transaction_id=transaction, request_ids=batch.request_ids,
                                        accepted_counts=summary.accepted_counts, commit_rows=summary.commit_rows,
                                        commit_tokens=summary.commit_tokens, commit_positions=summary.commit_positions,
                                        next_tokens=summary.next_tokens, candidate_counts=batch.candidate_counts,
                                        draft_depth=batch.draft_depth, tree_shape=batch.tree_shape, mode=batch.mode,
                                    )
                                    verifier.commit(prepared, plan)
                                    committed = state_fingerprint(target)
                                    compare_state(committed, oracle_states[accepted])
                                    if args.following_step:
                                        following = target.step(next_token, return_logits=True)
                                        np.testing.assert_array_equal(following.logits, following_logits)
                                        compare_state(state_fingerprint(target), following_state)
                                    # Exercise rollback even after a successful selected-state commit.
                                    verifier.rollback(prepared)
                                    compare_state(state_fingerprint(target), initial)
                                    rows.append(dict(
                                        prompt_id=prompt_id, prompt_ids=prompt, context=length, budget=budget,
                                        accepted=accepted, transport=transport, logits=diagnostic_logits,
                                        repetition=repetition, graph=prepared.native_graph_submitted,
                                        expected_graph_extent=expected_graph_extent,
                                        mode=prepared.target_verify_mode, logit_sha256=logit_hash,
                                        initial=initial, committed=committed, passed=True,
                                        following_step=args.following_step,
                                    ))
                        print(f"PASS {prompt_id} p{length} B{budget} accepted={accepted}", flush=True)
    return memory_stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--contexts", default="95,96,121,128,252,253,256")
    parser.add_argument("--budgets", default="1,2,3")
    parser.add_argument("--native-context-limit", type=int)
    parser.add_argument("--require-native-through", type=int, default=0)
    parser.add_argument("--allow-graph-transition-fallback", action="store_true")
    parser.add_argument("--bulk-prefill", action="store_true")
    parser.add_argument("--graph-first", action="store_true")
    parser.add_argument("--n2-first", action="store_true")
    parser.add_argument("--following-step", action="store_true")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--prompt-limit", type=int, default=1)
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.contexts = [int(x) for x in args.contexts.split(",")]
    args.budgets = [int(x) for x in args.budgets.split(",")]
    if (not args.contexts or not args.budgets or min(args.contexts) < 1 or min(args.budgets) < 1
            or max(args.budgets) > 7 or args.repetitions < 1 or args.prompt_limit < 0
            or args.capacity < max(args.contexts) + max(args.budgets) + 2):
        parser.error("invalid shape or insufficient capacity")
    if args.output.exists():
        parser.error("output already exists")
    ctypes.CDLL("libamdhip64.so")
    from hipengine.core.hip import get_hip_runtime
    if get_hip_runtime().mem_get_info()[0] < 20 * (1 << 30):
        raise RuntimeError("state gate requires an idle GPU with at least 20 GiB free")
    payload = {"schema": 1, "status": "running", "performance_claim": False,
               "command": [sys.executable, *sys.argv], "rows": []}
    try:
        payload["memory_after_close"] = run(args, payload["rows"])
        if payload["memory_after_close"]["active_allocations"] != 0:
            raise AssertionError("state gate leaked allocations")
        payload["status"] = "complete_exact"
    except Exception:
        payload["status"] = "failed"
        payload["error"] = traceback.format_exc()
        print(payload["error"], file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0 if payload["status"] == "complete_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
