"""Exercise chunk2048 in the real request-owned c2 pool.

Direct native pool work items are used, not HTTP/SSE or batched GPU kernels.
Compact production output paths remain enabled; diagnostics read buffers afterward.
"""

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.dispatch import RequestState, WorkItem, WorkKind
from hipengine.generation.registry import GenerationRequest, FinishDetails
from hipengine.generation.batch_scheduler import CompletedRequest, RequestObservability
from scripts.qwen4exp_chunk_boundary_gate import full_state
from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_layer2_profile_gate import _make_generator
from scripts.qwen4exp_q8_repair_depth_gate import resolve_allocation_profile, validate_chunk_allocation


def capture(runner):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    if runner.closed:
        raise ValueError("cannot inspect a released runner allocation")
    runner.runtime.device_synchronize()
    logits = np.empty(runner.config.vocab_size, dtype=np.float32)
    copy_device_to_host(host_array_ptr(logits), runner.logits_buffer, runtime=runner.runtime)
    state = full_state(runner)
    if (not np.isfinite(logits).all() or not state["recurrent"]["finite"]
            or not state["full_kv_finite"] or not state["live_index_finite"]):
        raise ValueError("nonfinite c2 state/logits")
    return dict(logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                next_token=int(np.argmax(logits)), state=state)


def work(kind, ids, token_rows=()):
    ids = tuple(ids)
    mapping = (tuple(rid for rid, tokens in zip(ids, token_rows, strict=True) for _ in tokens)
               if kind == WorkKind.PREFILL else ids)
    return WorkItem(kind=kind, request_ids=ids, row_to_request=mapping,
                    token_rows=tuple(tuple(row) for row in token_rows))


def register(pool, rid, tokens):
    tokens = tuple(tokens)
    request = GenerationRequest(prompts=(tokens,), max_tokens=16, temperature=0.,
                                top_p=1., ignore_eos=True)
    pool.register_batch((rid,), request, prompt_rows=(tokens,))
    state = RequestState(rid, tokens, 16)
    pool.reserve_admission(state)
    return state


def prefill(pool, rid, tokens):
    pool.prefill_batch(work(WorkKind.PREFILL, (rid,), (tokens,)), commit=True)


def cancel(pool, rid):
    row = pool._row(rid)
    finish = FinishDetails(reason="cancel", cancelled=True, sampler_mode="greedy")
    observation = RequestObservability(
        queue_seconds=0., time_to_first_token_seconds=None, inter_token_seconds=(),
        service_seconds=None, completion_seconds=0., prefill_seconds=0., decode_seconds=0.,
        kv_pages_owned=0, kv_pages_peak=0, bucket_key=None, admission_blocked_reason=None,
        finish_reason="cancel", finish_details=finish, submitted_timestamp=0.,
        admitted_timestamp=None, completion_timestamp=0.)
    generated = tuple(row.generated_ids)
    pool.reclaim(CompletedRequest(rid, row.prompt_ids, generated, True, "cancel", finish, observation))
    output = pool.take_outputs((rid,))[0]
    if output.generated_token_ids != generated or not output.finish_details.cancelled:
        raise ValueError("cancelled output accounting drift")
    return len(generated)


def owned_ranges(runner):
    buffers = list(runner.state.owned_buffers.values()) + [runner.logits_buffer]
    for state in runner.attention_states:
        buffers.extend((state.key_cache, state.value_cache, state.position, state.context))
    for state in runner.index_states:
        buffers.extend((state.raw_keys, state.pooled_keys, state.selected_positions,
                        state.selected_count))
    cfg = runner.config
    for field in ("gdn_prefill_scratch", "qsa_prefill_scratch"):
        buffers.extend(getattr(runner, field).moe.ensure_group_risk_buffers(
            compact_rows=runner.prefill_chunk_size * cfg.expert_used_count,
            out_features_total=max(cfg.hidden_size, 2 * cfg.expert_feed_forward_length)))
    return [(int(buffer.ptr), int(buffer.ptr) + buffer.nbytes) for buffer in buffers]


def disjoint(left, right):
    return all(a1 <= b0 or b1 <= a0 for a0, a1 in left for b0, b1 in right)


def exercise(pool, prompts, capture_fn=capture, *, checkpoint_each=True):
    references = {}
    for index, (name, prompt) in enumerate(prompts.items()):
        rid = index + 1
        register(pool, rid, prompt)
        prefill(pool, rid, prompt)
        samples = [capture_fn(pool._row(rid).runner)]
        for _ in range(8):
            token = pool.decode_batch(work(WorkKind.DECODE, (rid,)), commit=True)[0]
            if token.token_id != samples[-1]["next_token"] or token.finished:
                raise ValueError("isolated compact decode accounting drift")
            samples.append(capture_fn(pool._row(rid).runner))
        references[name] = samples
        cancel(pool, rid)
    results = []
    for repeat in range(3):
        a, b, c, d = (100 + repeat * 10 + value for value in range(4))
        # Alternate admission order so each request exercises both physical owners.
        order = ((a, "a"), (b, "b")) if repeat % 2 == 0 else ((b, "b"), (a, "a"))
        for rid, name in order:
            register(pool, rid, prompts[name])
        if pool._row(a).runner is pool._row(b).runner:
            raise ValueError("two live requests alias a runner")
        owners = {name: pool._all_runners.index(pool._row(rid).runner)
                  for rid, name in ((a, "a"), (b, "b"))}
        progress = {a: 0, b: 0, d: 0}
        names = {a: "a", b: "b", d: "c"}
        checks = 0

        def compare(rid):
            nonlocal checks
            if not checkpoint_each:
                return None
            actual = capture_fn(pool._row(rid).runner)
            if actual != references[names[rid]][progress[rid]]:
                raise ValueError(f"c2 request {names[rid]} differs at step{progress[rid]}")
            checks += 1
            return actual

        def advance(ids):
            expected = {rid: references[names[rid]][progress[rid]]["next_token"] for rid in ids}
            generated = pool.decode_batch(work(WorkKind.DECODE, ids), commit=True)
            if [item.request_id for item in generated] != list(ids):
                raise ValueError("decode output request mapping drift")
            for item in generated:
                if item.token_id != expected[item.request_id] or item.finished:
                    raise ValueError("interleaved compact decode accounting drift")
                progress[item.request_id] += 1
                compare(item.request_id)

        prefill(pool, a, prompts["a"][:1024])
        prefill(pool, b, prompts["b"][:1024])
        if pool._row(a).next_result is not None or pool._row(b).next_result is not None:
            raise ValueError("partial scheduler prefix unexpectedly completed model prefill")
        prefill(pool, a, prompts["a"][1024:])
        compare(a)
        advance((a,))
        advance((a,))
        prefill(pool, b, prompts["b"][1024:])
        compare(a)
        compare(b)
        for _ in range(2):
            advance((a, b) if repeat % 2 == 0 else (b, a))
        old_peer = pool._row(b).runner
        if cancel(pool, b) != 2:
            raise ValueError("peer cancellation lost generated tokens")
        compare(a)
        admission = register(pool, c, prompts["c"])
        pool.rollback_admission(admission)
        compare(a)
        pool.reserve_admission(admission)
        prefill(pool, c, prompts["c"][:1024])
        if pool._row(c).next_result is not None or cancel(pool, c) != 0:
            raise ValueError("partial-prefix cancellation published output")
        compare(a)
        register(pool, d, prompts["c"])
        if pool._row(d).runner is not old_peer:
            raise ValueError("replacement did not exercise the released physical runner")
        prefill(pool, d, prompts["c"])
        compare(a)
        compare(d)
        for _ in range(2):
            advance((d, a))
        while progress[a] < 8:
            advance((a,))
        compare(d)
        finished_a = pool._row(a).runner
        if cancel(pool, a) != 8:
            raise ValueError("active-request cancellation accounting drift")
        compare(d)
        while progress[d] < 8:
            advance((d,))
        if not checkpoint_each:
            for runner, name in ((finished_a, "a"), (pool._row(d).runner, "c")):
                if capture_fn(runner) != references[name][8]:
                    raise ValueError(f"deferred c2 state differs for {name}")
                checks += 1
        if cancel(pool, d) != 8:
            raise ValueError("replacement-request cancellation accounting drift")
        if pool.active_request_ids or len(pool._available) != 2 or pool._outputs:
            raise ValueError("pool ownership did not drain")
        results.append(dict(repeat=repeat, owners=owners, checkpoints=checks,
                            peer_cancel_tokens=2, partial_cancel_tokens=0,
                            active_cancel_tokens=8, replacement_cancel_tokens=8))
        print("c2 repeat", repeat, "passed", flush=True)
    if {row["owners"]["a"] for row in results} != {0, 1}:
        raise ValueError("active request did not exercise both physical owners")
    return dict(references=references, repeats=results)


def validate_traces(traces, prompts):
    expected = {index + 1: len(prompt) for index, prompt in enumerate(prompts.values())}
    for repeat in range(3):
        base = 100 + repeat * 10
        expected.update({base: len(prompts["a"]), base + 1: len(prompts["b"]),
                         base + 3: len(prompts["c"])})
    observed = {}
    for event in traces:
        observed.setdefault(event["request_id"], []).append(event["rows"])
    if set(observed) != set(expected):
        raise ValueError("unexpected model-prefill request set")
    for rid, length in expected.items():
        count, tail = divmod(length, 2048)
        if observed[rid] != [2048] * count + ([tail] if tail else []):
            raise ValueError("actual c2 model chunk coverage differs")
    return dict(model_prefills=len(observed), chunk_calls=len(traces),
                partial_cancellations_without_model_prefill=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--allocation-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--defer-inspection", action="store_true",
                        help="Inspect only final A/C state after the interleaved sequence")
    args = parser.parse_args()
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        parser.error("c2 capture requires clean tracked source")
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host, model = _host_metadata(), model_identity(args.model_root)
        allocation = json.loads(args.allocation_evidence.read_bytes())
        resolved = resolve_allocation_profile()
        validate_chunk_allocation(allocation, chunk=2048, context=4352,
                                  manifest=resolved.manifest_sha256, host=host, model=model)
        if allocation["prepared_runners"] != 2 or len(allocation["lazy_group_risk"]) != 2:
            raise ValueError("two prepared runner/queue owners required")
        fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
        sources = {case["id"]: case["prompt_token_ids"] for case in fixture["cases"]}
        prompts = {
            "a": sources["code-p4096"][:2052],
            "b": sources["general_ja-p4096"] + [sources["general_ja-p4096"][-1]],
            "c": sources["mixed_ja_en-p4096"][:2049],
        }
        report = dict(status="running", source=source, host=host, model=model,
                      command=sys.argv, fixture_sha256=fixture_hash,
                      protocol=dict(capacity=4352, chunk=2048, resident_runners=2,
                                    compact_outputs=True, repeats=3, decode_steps=8,
                                    inspection_mode="deferred" if args.defer_inspection else "each_checkpoint"),
                      performance_claim=False, promotion_claim=False,
                      limitations=["Native pool work items, not HTTP/SSE or concurrent GPU-prefill scheduling.",
                                   "Reference is the same production2048 arithmetic in isolation, not strict.",
                                   "No native-depth, MTP or performance claim."])
        arm_args = SimpleNamespace(**vars(args), max_sequence_length=4352, prefill_chunk_size=2048)
        generator, profile, _ = _make_generator(arm_args, "production")
        report["manifest"] = profile.manifest_sha256
        pool = None
        stack = ExitStack()
        traces = []
        try:
            pool = generator.create_resident_model_runner(capacity=2)
            pool.prepare()
            if len(pool._all_runners) != 2 or any(
                    runner.prefill_chunk_size != 2048 for runner in pool._all_runners):
                raise ValueError("c2 runner construction did not preserve chunk2048")
            report["repair_queues"] = [prepare_lazy_group_risk(runner) for runner in pool._all_runners]
            ranges = [owned_ranges(runner) for runner in pool._all_runners]
            if not disjoint(*ranges):
                raise ValueError("request-owned state/repair buffers overlap")
            report["disjoint_owner_ranges"] = True
            report["owner_range_counts"] = [len(values) for values in ranges]
            report["owner_range_scope"] = (
                "Recurrent, logits, attention K/V/cursors, live-index storage and repair queues; "
                "view counts may include aliases within one owner, never across owners.")
            for owner, runner in enumerate(pool._all_runners):
                original = runner._prefill_chunk

                def counted(tokens, _original=original, _owner=owner, **kwargs):
                    owner_runner = pool._all_runners[_owner]
                    ids = [row.request_id for row in pool._rows.values() if row.runner is owner_runner]
                    if len(ids) != 1:
                        raise ValueError("model prefill has no unique live request owner")
                    traces.append(dict(owner=_owner, request_id=ids[0], rows=len(tokens)))
                    return _original(tokens, **kwargs)

                runner._prefill_chunk = counted
                stack.callback(delattr, runner, "_prefill_chunk")
            report.update(exercise(pool, prompts, checkpoint_each=not args.defer_inspection))
            report["trace_gate"] = validate_traces(traces, prompts)
            report["status"] = "passed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            stack.close()
            if pool is None:
                generator.close()
            else:
                pool.close()
            report["chunk_traces"] = traces
            report["memory_after_close"] = memory_stats()
            if (_git_metadata(ROOT) != source or report["memory_after_close"]["current_allocated_bytes"]):
                report["status"] = "invalid_source_or_lifecycle"
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
