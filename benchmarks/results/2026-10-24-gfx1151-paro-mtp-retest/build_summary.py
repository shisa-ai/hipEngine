#!/usr/bin/env python3
"""Build the offline PARO MTP retest summary from the first sweep and its confirmation.

The builder reads the two measure.py artifacts, validates both against the
retest protocol, and publishes first-sweep metrics. The confirmation sweep is
drift evidence only; the builder never selects the faster sweep.

Exit status is 0 when both sweeps validate and the summary is written, 1 when
validation fails (no summary is written), and 2 on usage errors.

Usage:
    python3 build_summary.py --first first.json --confirmation confirmation.json --output summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

KIND = 'public_paro_paired_streaming_retest'
SCHEMA = 1
TOTAL_CASES = 21
SPLIT_COUNTS = {'category': 10, 'heldout': 8, 'control': 3}
CONTROL_LENGTHS = (512, 1024, 4096)
GENERATED_TOKENS = 128
DECODE_TOKENS = GENERATED_TOKENS - 1
WARMUPS = 2
MEASURED = 5
PAIRS = WARMUPS + MEASURED
ARMS = ('ar', 'mtp')
NATURAL_SPLITS = ('category', 'heldout')
NATURAL_DRIFT_LIMIT = 0.05
CONTROL_DRIFT_LIMIT = 0.10
HEX = set('0123456789abcdef')

EXPECTED_BACKEND = 'hip_gfx1151'
EXPECTED_RETEST = {
    'schema': SCHEMA,
    'case_count': TOTAL_CASES,
    'category_prompts': SPLIT_COUNTS['category'],
    'category_heldouts': SPLIT_COUNTS['heldout'],
    'synthetic_prompt_lengths': list(CONTROL_LENGTHS),
    'decode_tokens': GENERATED_TOKENS,
    'warmups_per_case_per_arm': WARMUPS,
    'samples_per_case_per_arm': MEASURED,
    'backend': EXPECTED_BACKEND,
    'quant': 'w4_paro',
    'kv_storage': 'bf16',
}

EXPECTED_EMBEDDED = {
    'backend': EXPECTED_BACKEND,
    'quant': 'w4_paro',
    'kv_storage': 'bf16',
    'max_active_requests': 1,
    'max_sequence_length': 4352,
    'prefix_cache': 'off',
    'temperature': 0,
    'ignore_eos': True,
    'output_tokens': GENERATED_TOKENS,
    'warmups_per_case_per_arm': WARMUPS,
    'samples_per_case_per_arm': MEASURED,
}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right, rel_tol=1e-9):
    return is_number(left) and is_number(right) and math.isclose(left, right, rel_tol=rel_tol, abs_tol=1e-12)


def token_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in HEX for char in value)


def normalized_command(command):
    if not isinstance(command, list):
        return None
    masked = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token == '--output':
            masked.append(token)
            masked.append('<output>')
            skip_next = True
        else:
            masked.append(token)
    return masked


class Sweep:
    def __init__(self, label, path):
        self.label = label
        self.path = Path(path).resolve()
        self.raw = self.path.read_bytes()
        self.size = len(self.raw)
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self.data = json.loads(self.raw)
        if not isinstance(self.data, dict):
            raise ValueError(f'{self.path} is not a JSON object')
        self.driver_path = None
        self.driver_sha256_now = None
        provenance = self.data.get('provenance')
        command = provenance.get('command') if isinstance(provenance, dict) else None
        if isinstance(command, list) and len(command) >= 2 and isinstance(command[1], str):
            self.driver_path = Path(command[1])
            if self.driver_path.is_file():
                self.driver_sha256_now = sha256_file(self.driver_path)


def emit(checks, prefix, name, problems, checked, note=''):
    ok = not problems
    if ok:
        detail = note or f'{checked} checked'
    else:
        counted = f'{len(problems)} of {checked} failed' if checked else f'{len(problems)} failed'
        detail = counted + ': ' + ' | '.join(problems[:5])
        if len(problems) > 5:
            detail += f' | (+{len(problems) - 5} more)'
    checks.append({'name': prefix + name, 'passed': ok, 'detail': detail})
    return ok


def validate_retest(retest):
    checks = []
    problems = []
    checked = 0
    for key, expected in EXPECTED_RETEST.items():
        checked += 1
        if retest.get(key) != expected:
            problems.append(f'{key}={retest.get(key)!r} expected {expected!r}')
    emit(checks, 'protocol.', 'retest_constants', problems, checked)

    problems = []
    commit = retest.get('measurement_commit')
    if not (isinstance(commit, str) and len(commit) == 40 and all(char in HEX for char in commit)):
        problems.append(f'measurement_commit={commit!r}')
    model = retest.get('model')
    if not (isinstance(model, str) and model.strip()):
        problems.append(f'model={model!r}')
    surface = retest.get('surface')
    if not (isinstance(surface, str) and 'stream_detailed' in surface and 'stream_speculative_mtp_detailed' in surface):
        problems.append(f'surface={surface!r}')
    emit(checks, 'protocol.', 'retest_identity', problems, 3)
    return checks


def validate_sweep(sweep, retest):
    checks = []
    prefix = sweep.label + '.'
    data = sweep.data

    problems = []
    if data.get('kind') != KIND:
        problems.append(f"kind={data.get('kind')!r}")
    if data.get('schema') != SCHEMA:
        problems.append(f"schema={data.get('schema')!r}")
    emit(checks, prefix, 'artifact.kind_and_schema', problems, 2)

    problems = []
    if data.get('passed') is not True:
        problems.append(f"passed={data.get('passed')!r}")
    if 'error' in data:
        problems.append('error field present: ' + str(data.get('error'))[:160])
    emit(checks, prefix, 'artifact.passed', problems, 2)

    memory = data.get('memory_after_close')
    problems = []
    if not isinstance(memory, dict):
        problems.append('memory_after_close missing')
    elif memory.get('active_allocations') != 0:
        problems.append(f"active_allocations={memory.get('active_allocations')!r}")
    emit(checks, prefix, 'artifact.memory_after_close', problems, 1)

    embedded = data.get('protocol')
    problems = []
    checked = 0
    if not isinstance(embedded, dict):
        problems.append('protocol missing')
    else:
        for key, expected in EXPECTED_EMBEDDED.items():
            checked += 1
            if embedded.get(key) != expected:
                problems.append(f'{key}={embedded.get(key)!r} expected {expected!r}')
        for embedded_key, retest_key in (
            ('output_tokens', 'decode_tokens'),
            ('warmups_per_case_per_arm', 'warmups_per_case_per_arm'),
            ('samples_per_case_per_arm', 'samples_per_case_per_arm'),
            ('backend', 'backend'),
            ('quant', 'quant'),
            ('kv_storage', 'kv_storage'),
            ('surface', 'surface'),
        ):
            checked += 1
            if embedded.get(embedded_key) != retest.get(retest_key):
                problems.append(
                    f'{embedded_key}={embedded.get(embedded_key)!r} != protocol.json {retest_key}={retest.get(retest_key)!r}'
                )
        checked += 1
        surface = embedded.get('surface')
        if not (isinstance(surface, str) and 'stream_detailed' in surface and 'stream_speculative_mtp_detailed' in surface):
            problems.append(f'surface={surface!r}')
    emit(checks, prefix, 'protocol.matches_retest', problems, checked)

    raw_cases = data.get('cases')
    problems = []
    cases = []
    if not isinstance(raw_cases, list):
        problems.append('cases is not a list')
    else:
        if len(raw_cases) != TOTAL_CASES:
            problems.append(f'{len(raw_cases)} cases')
        cases = [case for case in raw_cases if isinstance(case, dict)]
        if len(cases) != len(raw_cases):
            problems.append(f'{len(raw_cases) - len(cases)} non-object cases')
    emit(checks, prefix, 'cases.present', problems, TOTAL_CASES)
    if not cases:
        return checks

    problems = []
    if len(cases) != TOTAL_CASES:
        problems.append(f'{len(cases)} cases')
    counts = Counter(case.get('split') for case in cases)
    for split, expected in SPLIT_COUNTS.items():
        if counts.get(split, 0) != expected:
            problems.append(f'{split}={counts.get(split, 0)} expected {expected}')
    for split in counts:
        if split not in SPLIT_COUNTS:
            problems.append(f'unexpected split {split!r}')
    emit(checks, prefix, 'cases.count_and_splits', problems, TOTAL_CASES)

    ids = [case.get('id') for case in cases]
    problems = []
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        problems.append(f'duplicate ids {duplicates[:5]}')
    if not all(isinstance(value, str) and value for value in ids):
        problems.append('missing or non-string id')
    emit(checks, prefix, 'cases.unique_ids', problems, len(ids))

    problems = []
    controls = [case for case in cases if case.get('split') == 'control']
    lengths = sorted(case.get('prompt_tokens') for case in controls if isinstance(case.get('prompt_tokens'), int))
    if tuple(lengths) != CONTROL_LENGTHS:
        problems.append(f'control lengths {lengths}')
    for case in controls:
        if case.get('id') != f"p{case.get('prompt_tokens')}":
            problems.append(f"id {case.get('id')!r} != p{case.get('prompt_tokens')}")
        if case.get('category') != 'length_control':
            problems.append(f"{case.get('id')} category {case.get('category')!r}")
    emit(checks, prefix, 'cases.control_shapes', problems, len(controls))

    problems = []
    natural = [case for case in cases if case.get('split') in NATURAL_SPLITS]
    for case in natural:
        if not isinstance(case.get('prompt_tokens'), int) or case['prompt_tokens'] <= 0:
            problems.append(f"{case.get('id')} prompt_tokens")
        if not isinstance(case.get('category'), str) or not case['category']:
            problems.append(f"{case.get('id')} category")
    emit(checks, prefix, 'cases.natural_metadata', problems, len(natural))

    problems = []
    checked = 0
    for case in cases:
        for arm in ARMS:
            checked += 1
            reference = case.get(arm + '_ids')
            if not isinstance(reference, list) or len(reference) != GENERATED_TOKENS:
                length = len(reference) if isinstance(reference, list) else reference
                problems.append(f"{case.get('id')}/{arm}: reference ids {length!r}")
            elif not all(isinstance(token, int) and token >= 0 for token in reference):
                problems.append(f"{case.get('id')}/{arm}: non-integer reference ids")
    emit(checks, prefix, 'cases.reference_ids', problems, checked)

    problems = []
    checked = 0
    for case in cases:
        checked += 2
        prompt_hash = case.get('prompt_ids_sha256')
        if not is_sha256(prompt_hash):
            problems.append(f"{case.get('id')} prompt_ids_sha256={prompt_hash!r}")
        equal = case.get('ar_mtp_ids_equal')
        ar_ids = case.get('ar_ids')
        mtp_ids = case.get('mtp_ids')
        if not isinstance(equal, bool):
            problems.append(f"{case.get('id')} ar_mtp_ids_equal is not a bool")
        elif isinstance(ar_ids, list) and isinstance(mtp_ids, list) and equal != (ar_ids == mtp_ids):
            problems.append(f"{case.get('id')} ar_mtp_ids_equal inconsistent with ids")
    emit(checks, prefix, 'cases.identity_fields', problems, checked)

    expected_order = []
    for pair in range(PAIRS):
        order = ('ar', 'mtp') if pair % 2 == 0 else ('mtp', 'ar')
        for arm in order:
            expected_order.append((pair, arm))

    rule_problems = defaultdict(list)
    rule_checked = Counter()

    def rule(name, ok, message):
        rule_checked[name] += 1
        if not ok:
            rule_problems[name].append(message)

    for case in cases:
        case_id = case.get('id')
        samples = case.get('samples')
        rule('samples.present', isinstance(samples, list), f'{case_id}: samples missing')
        if not isinstance(samples, list):
            continue
        got_order = [(sample.get('pair'), sample.get('arm')) for sample in samples if isinstance(sample, dict)]
        rule(
            'samples.count_and_order',
            got_order == expected_order,
            f'{case_id}: sample sequence {got_order[:4]} != protocol order',
        )
        arm_hashes = {}
        for arm in ARMS:
            reference = case.get(arm + '_ids')
            arm_hashes[arm] = token_hash(reference) if isinstance(reference, list) else None
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            arm = sample.get('arm')
            pair = sample.get('pair')
            where = f'{case_id}/{arm}/pair{pair}'
            rule('samples.warmup_flag', sample.get('warmup') is (isinstance(pair, int) and pair < WARMUPS),
                 f'{where}: warmup={sample.get("warmup")!r}')
            tokens = sample.get('generated_tokens')
            rule('samples.generated_tokens', tokens == GENERATED_TOKENS, f'{where}: generated_tokens={tokens!r}')
            rule('samples.id_hash', sample.get('generated_ids_sha256') == arm_hashes.get(arm),
                 f'{where}: generated_ids_sha256 != case {arm}_ids hash')

            events = sample.get('token_events')
            counts = []
            times = []
            ok_events = isinstance(events, list) and len(events) >= 2
            if ok_events:
                counts = [event.get('tokens') for event in events]
                times = [event.get('seconds') for event in events]
                ok_events = (
                    all(isinstance(value, int) for value in counts)
                    and all(is_number(value) for value in times)
                    and counts[0] == 1
                    and counts[-1] == GENERATED_TOKENS
                    and all(left < right for left, right in zip(counts, counts[1:]))
                    and all(left <= right for left, right in zip(times, times[1:]))
                    and times[-1] > times[0]
                )
            rule('samples.token_events', ok_events, f'{where}: token_events invalid')
            if not ok_events:
                continue

            expected_decode_tokens = GENERATED_TOKENS - counts[0]
            decode_tokens = sample.get('decode_tokens')
            rule('samples.decode_tokens', decode_tokens == expected_decode_tokens == DECODE_TOKENS,
                 f'{where}: decode_tokens={decode_tokens!r} expected {expected_decode_tokens}')
            decode_seconds = sample.get('decode_seconds')
            rule('samples.decode_window',
                 is_number(decode_seconds) and decode_seconds > 0 and close(decode_seconds, times[-1] - times[0]),
                 f'{where}: decode_seconds={decode_seconds!r} != last-first event')
            rule('samples.first_token', close(sample.get('first_token_seconds'), times[0]),
                 f'{where}: first_token_seconds != first event')
            whole_call = sample.get('whole_call_seconds')
            rule('samples.whole_call_window',
                 is_number(whole_call) and whole_call > 0 and whole_call >= times[-1],
                 f'{where}: whole_call_seconds={whole_call!r} before last event')
            rate = sample.get('decode_tokens_per_second')
            rule('samples.decode_rate_arithmetic',
                 is_number(decode_tokens) and is_number(decode_seconds) and decode_seconds > 0
                 and is_number(rate) and close(rate, decode_tokens / decode_seconds),
                 f'{where}: decode_tokens_per_second={rate!r}')
            whole_rate = sample.get('whole_call_tokens_per_second')
            rule('samples.whole_call_rate_arithmetic',
                 is_number(tokens) and is_number(whole_call) and whole_call > 0
                 and is_number(whole_rate) and close(whole_rate, tokens / whole_call),
                 f'{where}: whole_call_tokens_per_second={whole_rate!r}')

            route = sample.get('route')
            rule('routes.present', isinstance(route, dict), f'{where}: route missing')
            if not isinstance(route, dict):
                continue
            used = route.get('specdec2_mtp2_used')
            cycles = route.get('specdec2_mtp2_cycles')
            candidates = route.get('specdec2_mtp2_candidate_counts')
            accepted = route.get('specdec2_mtp2_accepted_counts')
            execution_routes = route.get('specdec2_mtp2_execution_routes')
            if arm == 'ar':
                ok = used is False and cycles == 0 and candidates == [] and accepted == [] and execution_routes == []
                rule('routes.ar_no_speculation', ok,
                     f'{where}: AR route used={used!r} cycles={cycles!r} candidates={candidates!r} routes={execution_routes!r}')
            elif arm == 'mtp':
                ok_cycles = used is True and isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0
                rule('routes.mtp_real_cycles', ok_cycles, f'{where}: MTP used={used!r} cycles={cycles!r}')
                if ok_cycles:
                    ok_counts = (
                        isinstance(candidates, list) and isinstance(accepted, list)
                        and len(candidates) == cycles and len(accepted) == cycles
                        and set(candidates) == {1}
                        and all(isinstance(value, int) and not isinstance(value, bool) for value in accepted)
                        and all(0 <= value <= budget for value, budget in zip(accepted, candidates))
                        and sum(accepted) <= cycles
                    )
                    rule('routes.mtp_candidate_accounting', ok_counts,
                         f'{where}: cycles={cycles!r} candidates={candidates!r} accepted={accepted!r}')
                    rule('routes.mtp_eager', isinstance(execution_routes, list) and set(execution_routes) == {'eager'},
                         f'{where}: execution routes {execution_routes!r}')
                    fallback = sample.get('fallback_reasons')
                    rule('routes.mtp_no_recovery',
                         isinstance(fallback, dict)
                         and not any(str(key).startswith('specdec2_mtp2_recovered') for key in fallback),
                         f'{where}: speculative recovery fallback present')
            else:
                rule('samples.arm', False, f'{where}: unknown arm {arm!r}')

        for arm in ARMS:
            arm_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get('arm') == arm]
            warmup_count = sum(1 for sample in arm_samples if sample.get('warmup') is True)
            measured_count = sum(1 for sample in arm_samples if sample.get('warmup') is False)
            rule('samples.count_per_arm', warmup_count == WARMUPS and measured_count == MEASURED,
                 f'{case_id}/{arm}: {warmup_count} warmup {measured_count} measured')

        medians = case.get('medians')
        for arm in ARMS:
            recorded = medians.get(arm) if isinstance(medians, dict) else None
            arm_samples = [
                sample for sample in samples
                if isinstance(sample, dict) and sample.get('arm') == arm and sample.get('warmup') is False
            ]
            ok = isinstance(recorded, dict) and len(arm_samples) == MEASURED
            detail = f'{case_id}/{arm}: medians'
            if ok:
                for key in ('decode_tokens_per_second', 'first_token_seconds', 'whole_call_tokens_per_second'):
                    try:
                        expected_value = statistics.median(sample[key] for sample in arm_samples)
                    except (KeyError, TypeError):
                        ok = False
                        detail = f'{case_id}/{arm}: {key} missing'
                        break
                    if not close(recorded.get(key), expected_value):
                        ok = False
                        detail = f'{case_id}/{arm}: {key} recorded {recorded.get(key)!r} != recomputed {expected_value!r}'
                        break
            rule('cases.medians_match', ok, detail)

    for name in (
        'samples.present',
        'samples.count_and_order',
        'samples.warmup_flag',
        'samples.count_per_arm',
        'samples.arm',
        'samples.generated_tokens',
        'samples.id_hash',
        'samples.token_events',
        'samples.decode_tokens',
        'samples.decode_window',
        'samples.first_token',
        'samples.whole_call_window',
        'samples.decode_rate_arithmetic',
        'samples.whole_call_rate_arithmetic',
        'routes.present',
        'routes.ar_no_speculation',
        'routes.mtp_real_cycles',
        'routes.mtp_candidate_accounting',
        'routes.mtp_eager',
        'routes.mtp_no_recovery',
        'cases.medians_match',
    ):
        emit(checks, prefix, name, rule_problems[name], rule_checked[name])

    provenance = data.get('provenance')
    problems = []
    if not isinstance(provenance, dict):
        problems.append('provenance missing')
    else:
        if provenance.get('kind') != 'hipengine_artifact_provenance':
            problems.append(f"kind={provenance.get('kind')!r}")
        for key in ('dirty', 'staged_dirty', 'unstaged_dirty', 'untracked_dirty', 'execution_affecting_dirty'):
            if provenance.get(key) is not False:
                problems.append(f'{key}={provenance.get(key)!r}')
        if provenance.get('dirty_path_count') != 0:
            problems.append(f"dirty_path_count={provenance.get('dirty_path_count')!r}")
        if provenance.get('untracked_count') != 0:
            problems.append(f"untracked_count={provenance.get('untracked_count')!r}")
        if provenance.get('execution_affecting_paths'):
            problems.append(f"execution_affecting_paths={provenance.get('execution_affecting_paths')!r}")
    emit(checks, prefix, 'provenance.clean_checkout', problems, 7)

    problems = []
    if isinstance(provenance, dict):
        commit = provenance.get('hipengine_commit')
        if not (isinstance(commit, str) and len(commit) == 40 and all(char in HEX for char in commit)):
            problems.append(f'hipengine_commit={commit!r}')
        elif commit != retest.get('measurement_commit'):
            problems.append(
                f"hipengine_commit {commit} != protocol.json measurement_commit {retest.get('measurement_commit')}"
            )
        for key in ('configured_backend', 'resolved_backend'):
            if provenance.get(key) != retest.get('backend'):
                problems.append(f"{key}={provenance.get(key)!r} expected {retest.get('backend')!r}")
        if provenance.get('quant') != retest.get('quant'):
            problems.append(f"quant={provenance.get('quant')!r}")
        if provenance.get('kv_dtype') != retest.get('kv_storage'):
            problems.append(f"kv_dtype={provenance.get('kv_dtype')!r}")
        model = retest.get('model')
        if isinstance(model, str) and Path(provenance.get('model_path') or '').name != Path(model).name:
            problems.append(f"model_path={provenance.get('model_path')!r} does not match {model!r}")
        fingerprint = provenance.get('model_fingerprint')
        if not isinstance(fingerprint, dict):
            problems.append('model_fingerprint missing')
        else:
            if not is_sha256(fingerprint.get('value')):
                problems.append(f"model_fingerprint.value={fingerprint.get('value')!r}")
            if fingerprint.get('exists') is not True:
                problems.append(f"model_fingerprint.exists={fingerprint.get('exists')!r}")
        profiler = provenance.get('profiler')
        if not isinstance(profiler, dict) or profiler.get('enabled') is not False:
            problems.append(f'profiler={profiler!r}')
        for key in ('hipcc_version', 'rocm_version'):
            value = provenance.get(key)
            if not (isinstance(value, str) and value.strip()):
                problems.append(f'{key}={value!r}')
        command = provenance.get('command')
        if not (isinstance(command, list) and len(command) >= 2 and all(isinstance(token, str) for token in command)):
            problems.append(f'command={command!r}')
    emit(checks, prefix, 'provenance.identity', problems, 11)

    problems = []
    recorded = data.get('driver_sha256')
    driver_path = sweep.driver_path
    if not is_sha256(recorded):
        problems.append(f'driver_sha256={recorded!r}')
    elif driver_path is None:
        problems.append('driver path not recorded in provenance.command')
    elif not driver_path.is_file():
        problems.append(f'driver file missing: {driver_path}')
    elif sweep.driver_sha256_now != recorded:
        problems.append(f'{driver_path} sha256 {sweep.driver_sha256_now} != recorded {recorded}')
    emit(checks, prefix, 'provenance.driver_hash', problems, 1)

    problems = []
    checked = 0
    repo_root = Path(provenance.get('repo_root') or '') if isinstance(provenance, dict) else Path('')
    sources = embedded.get('sources') if isinstance(embedded, dict) else None
    if not isinstance(sources, list) or not sources:
        problems.append('protocol sources missing')
    else:
        seen = set()
        for source in sources:
            checked += 1
            if not isinstance(source, dict):
                problems.append('source entry is not an object')
                continue
            relative = source.get('path')
            digest = source.get('sha256')
            if not isinstance(relative, str) or not relative:
                problems.append(f'path={relative!r}')
                continue
            if relative in seen:
                problems.append(f'duplicate source {relative}')
            seen.add(relative)
            if not is_sha256(digest):
                problems.append(f'{relative} sha256={digest!r}')
                continue
            target = repo_root / relative
            if not target.is_file():
                problems.append(f'{relative} missing under {repo_root}')
                continue
            actual = sha256_file(target)
            if actual != digest:
                problems.append(f'{relative} sha256 {actual} != recorded {digest}')
    emit(checks, prefix, 'provenance.sources', problems, checked)

    return checks


def validate_pair(first, confirmation):
    checks = []

    problems = []
    if first.data.get('protocol') != confirmation.data.get('protocol'):
        problems.append('embedded protocol differs between sweeps')
    emit(checks, '', 'pair.protocol_identical', problems, 1)

    problems = []
    checked = 0
    if first.data.get('driver_sha256') != confirmation.data.get('driver_sha256'):
        problems.append('driver_sha256 differs between sweeps')
    checked += 1
    left = first.data.get('provenance') if isinstance(first.data.get('provenance'), dict) else {}
    right = confirmation.data.get('provenance') if isinstance(confirmation.data.get('provenance'), dict) else {}
    for key in (
        'hipengine_commit',
        'model_path',
        'quant',
        'kv_dtype',
        'configured_backend',
        'resolved_backend',
        'target_arch',
        'build_profile',
        'hipcc_version',
        'rocm_version',
        'warmups',
        'repetitions',
        'environment',
    ):
        checked += 1
        if left.get(key) != right.get(key):
            problems.append(f'{key} differs between sweeps')
    checked += 1
    if (left.get('model_fingerprint') or {}).get('value') != (right.get('model_fingerprint') or {}).get('value'):
        problems.append('model_fingerprint differs between sweeps')
    checked += 1
    if normalized_command(left.get('command')) != normalized_command(right.get('command')):
        problems.append('command differs between sweeps beyond --output')
    emit(checks, '', 'pair.provenance_identical', problems, checked)
    return checks


def measured_samples(case, arm=None):
    samples = case.get('samples') or []
    return [
        sample for sample in samples
        if isinstance(sample, dict) and sample.get('warmup') is False and (arm is None or sample.get('arm') == arm)
    ]


def pooled(samples):
    decode_tokens = sum(sample['decode_tokens'] for sample in samples)
    decode_seconds = sum(sample['decode_seconds'] for sample in samples)
    whole_call_tokens = sum(sample['generated_tokens'] for sample in samples)
    whole_call_seconds = sum(sample['whole_call_seconds'] for sample in samples)
    return {
        'samples': len(samples),
        'decode_tokens': decode_tokens,
        'decode_seconds': decode_seconds,
        'decode_tokens_per_second': decode_tokens / decode_seconds if decode_seconds else None,
        'whole_call_tokens': whole_call_tokens,
        'whole_call_seconds': whole_call_seconds,
        'whole_call_tokens_per_second': whole_call_tokens / whole_call_seconds if whole_call_seconds else None,
        'first_token_seconds_median': statistics.median(sample['first_token_seconds'] for sample in samples)
        if samples else None,
    }


def acceptance(samples):
    cycles = candidates = accepted = 0
    for sample in samples:
        route = sample['route']
        cycles += route['specdec2_mtp2_cycles']
        candidates += sum(route['specdec2_mtp2_candidate_counts'])
        accepted += sum(route['specdec2_mtp2_accepted_counts'])
    expected = DECODE_TOKENS * len(samples)
    return {
        'samples': len(samples),
        'cycles': cycles,
        'candidates': candidates,
        'accepted': accepted,
        'accepted_per_candidate': accepted / candidates if candidates else None,
        'tokens_from_cycles': cycles + accepted,
        'expected_decode_tokens': expected,
        'accounting_delta': cycles + accepted - expected,
    }


def arm_block(cases, arm):
    samples = [sample for case in cases for sample in measured_samples(case, arm)]
    block = pooled(samples)
    block['cases'] = len(cases)
    if arm == 'mtp':
        block['candidate_acceptance'] = acceptance(samples)
    return block


def identity(cases):
    equal = [case.get('id') for case in cases if case.get('ar_mtp_ids_equal') is True]
    differ = [case.get('id') for case in cases if case.get('ar_mtp_ids_equal') is not True]
    by_split = {}
    for split in ('category', 'heldout', 'control'):
        subset = [case for case in cases if case.get('split') == split]
        if subset:
            by_split[split] = {
                'cases': len(subset),
                'ar_mtp_ids_equal': sum(1 for case in subset if case.get('ar_mtp_ids_equal') is True),
            }
    return {
        'cases': len(cases),
        'ar_mtp_ids_equal': len(equal),
        'ar_mtp_ids_differ': differ,
        'by_split': by_split,
    }


def build_published(cases):
    natural = [case for case in cases if case.get('split') in NATURAL_SPLITS]
    controls = sorted(
        (case for case in cases if case.get('split') == 'control'),
        key=lambda case: case.get('prompt_tokens') or 0,
    )
    published = {
        'natural_suite': {
            'cases': len(natural),
            'ar': arm_block(natural, 'ar'),
            'mtp': arm_block(natural, 'mtp'),
        },
        'by_split': {},
        'by_category': [],
        'length_controls': {},
        'identity': identity(cases),
    }
    for split in NATURAL_SPLITS:
        subset = [case for case in cases if case.get('split') == split]
        published['by_split'][split] = {
            'cases': len(subset),
            'ar': arm_block(subset, 'ar'),
            'mtp': arm_block(subset, 'mtp'),
        }
    for category in sorted({case.get('category') for case in natural}):
        subset = [case for case in natural if case.get('category') == category]
        published['by_category'].append({
            'category': category,
            'cases': len(subset),
            'ar': arm_block(subset, 'ar'),
            'mtp': arm_block(subset, 'mtp'),
        })
    for case in controls:
        entry = {'prompt_tokens': case.get('prompt_tokens'), 'cases': 1}
        for arm in ARMS:
            samples = measured_samples(case, arm)
            block = {
                'samples': len(samples),
                'median_decode_tokens_per_second': statistics.median(
                    sample['decode_tokens_per_second'] for sample in samples
                ),
                'median_first_token_seconds': statistics.median(
                    sample['first_token_seconds'] for sample in samples
                ),
                'median_whole_call_tokens_per_second': statistics.median(
                    sample['whole_call_tokens_per_second'] for sample in samples
                ),
            }
            if arm == 'mtp':
                block['candidate_acceptance'] = acceptance(samples)
            entry[arm] = block
        published['length_controls'][case.get('id')] = entry
    return published


def drift_entry(first_value, confirmation_value, limit):
    entry = {'first': first_value, 'confirmation': confirmation_value}
    if is_number(first_value) and is_number(confirmation_value) and first_value != 0:
        difference = (confirmation_value - first_value) / first_value
        entry['relative_difference'] = difference
        entry['absolute_difference'] = confirmation_value - first_value
        flagged = limit is not None and abs(difference) > limit
    else:
        entry['relative_difference'] = None
        flagged = False
    if limit is not None:
        entry['limit'] = limit
    entry['flagged'] = flagged
    return entry


def build_drift(published, confirmation_published, first_cases, confirmation_cases):
    drift = {
        'published_sweep': 'first',
        'natural_suite': {},
        'by_category': [],
        'length_controls': {},
        'flags': [],
        'id_mismatches': [],
    }
    for arm in ARMS:
        first_block = published['natural_suite'][arm]
        confirmation_block = confirmation_published['natural_suite'][arm]
        decode = drift_entry(
            first_block['decode_tokens_per_second'],
            confirmation_block['decode_tokens_per_second'],
            NATURAL_DRIFT_LIMIT,
        )
        whole_call = drift_entry(
            first_block['whole_call_tokens_per_second'],
            confirmation_block['whole_call_tokens_per_second'],
            None,
        )
        drift['natural_suite'][arm] = {
            'decode_tokens_per_second': decode,
            'whole_call_tokens_per_second': whole_call,
        }
        if decode['flagged']:
            drift['flags'].append(
                f"natural_suite.{arm}.decode_tokens_per_second drift {decode['relative_difference']:+.2%} "
                f'exceeds {NATURAL_DRIFT_LIMIT:.0%}'
            )

    confirmation_categories = {block['category']: block for block in confirmation_published['by_category']}
    for first_block in published['by_category']:
        confirmation_block = confirmation_categories.get(first_block['category'])
        if confirmation_block is None:
            continue
        drift['by_category'].append({
            'category': first_block['category'],
            'ar': drift_entry(
                first_block['ar']['decode_tokens_per_second'],
                confirmation_block['ar']['decode_tokens_per_second'],
                None,
            ),
            'mtp': drift_entry(
                first_block['mtp']['decode_tokens_per_second'],
                confirmation_block['mtp']['decode_tokens_per_second'],
                None,
            ),
        })

    for case_id, first_block in published['length_controls'].items():
        confirmation_block = confirmation_published['length_controls'].get(case_id)
        if confirmation_block is None:
            continue
        entry = {'prompt_tokens': first_block['prompt_tokens']}
        for arm in ARMS:
            median_decode = drift_entry(
                first_block[arm]['median_decode_tokens_per_second'],
                confirmation_block[arm]['median_decode_tokens_per_second'],
                CONTROL_DRIFT_LIMIT,
            )
            entry[arm] = {
                'median_decode_tokens_per_second': median_decode,
                'median_first_token_seconds': drift_entry(
                    first_block[arm]['median_first_token_seconds'],
                    confirmation_block[arm]['median_first_token_seconds'],
                    None,
                ),
                'median_whole_call_tokens_per_second': drift_entry(
                    first_block[arm]['median_whole_call_tokens_per_second'],
                    confirmation_block[arm]['median_whole_call_tokens_per_second'],
                    None,
                ),
            }
            if median_decode['flagged']:
                drift['flags'].append(
                    f"length_controls.{case_id}.{arm}.median_decode_tokens_per_second drift "
                    f"{median_decode['relative_difference']:+.2%} exceeds {CONTROL_DRIFT_LIMIT:.0%}"
                )
        drift['length_controls'][case_id] = entry

    first_by_id = {case.get('id'): case for case in first_cases}
    for case in confirmation_cases:
        case_id = case.get('id')
        other = first_by_id.get(case_id)
        if other is None:
            drift['id_mismatches'].append({'case': case_id, 'reason': 'case missing in first sweep'})
            continue
        for arm in ARMS:
            left = other.get(arm + '_ids')
            right = case.get(arm + '_ids')
            if isinstance(left, list) and isinstance(right, list) and left != right:
                drift['id_mismatches'].append({
                    'case': case_id,
                    'arm': arm,
                    'reason': 'output ids differ between sweeps',
                })
    if drift['id_mismatches']:
        drift['flags'].append(
            f"{len(drift['id_mismatches'])} case/arm output id mismatches between sweeps "
            '(cross-sweep determinism, informational)'
        )
    return drift


def build_summary(first, confirmation, retest, protocol_path, checks):
    first_cases = first.data['cases']
    confirmation_cases = confirmation.data['cases']
    published = build_published(first_cases)
    confirmation_published = build_published(confirmation_cases)
    drift = build_drift(published, confirmation_published, first_cases, confirmation_cases)

    provenance = first.data.get('provenance') or {}
    sources = (first.data.get('protocol') or {}).get('sources') or []

    notes = [
        'First sweep is published. The confirmation sweep is drift evidence only; the faster sweep is never selected.',
        'Natural-suite decode throughput pools measured samples as sum(decode_tokens)/sum(decode_seconds). '
        f'decode_tokens is {DECODE_TOKENS} per sample (generated tokens minus the first observed cumulative count), '
        'so first-token latency, prefill, and MTP prompt priming are excluded.',
        'Whole-call throughput pools sum(generated_tokens)/sum(whole_call_seconds) and includes prefill and '
        'first-token latency, so it is not comparable to the decode rate.',
        'Per-length values are medians of per-sample rates, not ratios of pooled sums.',
        'Candidate acceptance is sum(accepted_counts)/sum(candidate_counts) over measured MTP samples; '
        'the candidate budget is one per cycle.',
        'AR/MTP id equality is a cross-route diagnostic, not a same-schedule repeatability check.',
        'Drift flags are reported for review and do not block publication; the summary never chooses between sweeps.',
    ]
    acceptance_block = published['natural_suite']['mtp']['candidate_acceptance']
    if acceptance_block['accounting_delta'] != 0:
        notes.append(
            'MTP route counters do not reconcile with the decode window: cycles plus accepted candidates '
            f"sum to {acceptance_block['tokens_from_cycles']} for {acceptance_block['samples']} measured natural-suite "
            f"samples, while the 127-per-sample post-first-token count implies {acceptance_block['expected_decode_tokens']}. "
            'Treat cycle and acceptance counts as route-reported and do not derive token throughput from them.'
        )

    return {
        'schema': SCHEMA,
        'kind': 'paro_mtp_retest_summary',
        'published_sweep': 'first',
        'publication_rule': 'first sweep published; confirmation sweep is drift evidence only',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'first': {'path': str(first.path), 'sha256': first.sha256, 'bytes': first.size},
            'confirmation': {
                'path': str(confirmation.path),
                'sha256': confirmation.sha256,
                'bytes': confirmation.size,
            },
            'protocol': {
                'path': str(Path(protocol_path).resolve()),
                'sha256': sha256_file(protocol_path),
                'bytes': Path(protocol_path).stat().st_size,
            },
            'driver': {
                'path': str(first.driver_path) if first.driver_path else None,
                'sha256_recorded': first.data.get('driver_sha256'),
                'sha256_now': first.driver_sha256_now,
                'confirmation_sha256_recorded': confirmation.data.get('driver_sha256'),
            },
            'sources': [
                {
                    'path': source.get('path'),
                    'sha256': source.get('sha256'),
                    'repo_root': provenance.get('repo_root'),
                }
                for source in sources
            ],
        },
        'provenance': {
            'hipengine_commit': provenance.get('hipengine_commit'),
            'repo_root': provenance.get('repo_root'),
            'model_path': provenance.get('model_path'),
            'model_fingerprint': provenance.get('model_fingerprint'),
            'backend': provenance.get('resolved_backend'),
            'configured_backend': provenance.get('configured_backend'),
            'quant': provenance.get('quant'),
            'kv_dtype': provenance.get('kv_dtype'),
            'build_profile': provenance.get('build_profile'),
            'rocm_version': provenance.get('rocm_version'),
            'hipcc_version': provenance.get('hipcc_version'),
            'command': provenance.get('command'),
            'confirmation_command': (confirmation.data.get('provenance') or {}).get('command'),
            'dirty': provenance.get('dirty'),
            'execution_affecting_dirty': provenance.get('execution_affecting_dirty'),
            'memory_after_close': first.data.get('memory_after_close'),
            'timing_protocol': provenance.get('timing_protocol'),
        },
        'validation': {
            'passed': True,
            'total': len(checks),
            'checks': checks,
        },
        'published': published,
        'confirmation_drift': drift,
        'notes': notes,
    }


def format_report(summary):
    published = summary['published']
    drift = summary['confirmation_drift']
    inputs = summary['inputs']
    lines = [
        'PARO MTP retest summary',
        f"published sweep: first  sha256 {inputs['first']['sha256']}",
        f"confirmation:    drift only  sha256 {inputs['confirmation']['sha256']}",
        f"validation:      PASS ({summary['validation']['total']} checks)",
        '',
        'Natural suite (category + heldout)',
        f"  {'arm':<4}{'decode tok/s':>14}{'decode tokens':>15}{'decode s':>11}"
        f"{'whole tok/s':>14}{'first-token med s':>19}{'samples':>9}",
    ]

    def row(name, block):
        return (
            f"  {name:<4}{block['decode_tokens_per_second']:>14.2f}{block['decode_tokens']:>15}"
            f"{block['decode_seconds']:>11.3f}{block['whole_call_tokens_per_second']:>14.2f}"
            f"{block['first_token_seconds_median']:>19.3f}{block['samples']:>9}"
        )

    for arm in ARMS:
        lines.append(row(arm.upper(), published['natural_suite'][arm]))
    acceptance_block = published['natural_suite']['mtp']['candidate_acceptance']
    lines.append(
        f"  MTP candidate acceptance: {acceptance_block['accepted']}/{acceptance_block['candidates']} = "
        f"{acceptance_block['accepted_per_candidate']:.2%} over {acceptance_block['cycles']} cycles"
    )
    lines.append('')
    lines.append('Per split (decode tok/s, AR / MTP)')
    for split, block in published['by_split'].items():
        lines.append(
            f"  {split:<9} cases {block['cases']:>2}  AR {block['ar']['decode_tokens_per_second']:>7.2f}"
            f"  MTP {block['mtp']['decode_tokens_per_second']:>7.2f}"
        )
    lines.append('')
    lines.append('Per category (decode tok/s, AR / MTP)')
    for block in published['by_category']:
        lines.append(
            f"  {block['category']:<12} cases {block['cases']:>2}  AR {block['ar']['decode_tokens_per_second']:>7.2f}"
            f"  MTP {block['mtp']['decode_tokens_per_second']:>7.2f}"
        )
    lines.append('')
    lines.append('Length controls (median decode tok/s, AR / MTP)')
    for case_id, block in published['length_controls'].items():
        lines.append(
            f"  {case_id:<6} prompt {block['prompt_tokens']:>5}  "
            f"AR {block['ar']['median_decode_tokens_per_second']:>7.2f}"
            f"  MTP {block['mtp']['median_decode_tokens_per_second']:>7.2f}"
        )
    lines.append('')
    identity_block = published['identity']
    lines.append(
        f"AR/MTP id equality: {identity_block['ar_mtp_ids_equal']}/{identity_block['cases']} cases"
    )
    lines.append('')
    lines.append('Confirmation drift (confirmation vs first)')
    for arm in ARMS:
        entry = drift['natural_suite'][arm]['decode_tokens_per_second']
        lines.append(
            f"  natural {arm.upper():<3} {entry['relative_difference']:+.2%} "
            f"(limit {NATURAL_DRIFT_LIMIT:.0%}, {'FLAGGED' if entry['flagged'] else 'within limit'})"
        )
    for case_id, block in drift['length_controls'].items():
        for arm in ARMS:
            entry = block[arm]['median_decode_tokens_per_second']
            lines.append(
                f"  {case_id:<6} {arm.upper():<3} {entry['relative_difference']:+.2%} "
                f"(limit {CONTROL_DRIFT_LIMIT:.0%}, {'FLAGGED' if entry['flagged'] else 'within limit'})"
            )
    if drift['flags']:
        lines.append('  flags:')
        for flag in drift['flags']:
            lines.append(f'    - {flag}')
    else:
        lines.append('  flags: none')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='build_summary.py',
        description='Validate the two PARO MTP retest sweeps and publish first-sweep metrics.',
    )
    parser.add_argument('--first', type=Path, required=True,
                        help='first measurement sweep written by measure.py')
    parser.add_argument('--confirmation', type=Path, required=True,
                        help='independent confirmation sweep written by measure.py')
    parser.add_argument('--output', type=Path, required=True,
                        help='summary JSON path to write')
    parser.add_argument('--protocol', type=Path, default=None,
                        help='retest protocol.json; defaults to the protocol.json beside --first')
    args = parser.parse_args(argv)

    protocol_path = args.protocol if args.protocol is not None else args.first.parent / 'protocol.json'
    try:
        retest = json.loads(Path(protocol_path).read_bytes())
        if not isinstance(retest, dict):
            raise ValueError(f'{protocol_path} is not a JSON object')
        first = Sweep('first', args.first)
        confirmation = Sweep('confirmation', args.confirmation)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    checks = []
    checks.extend(validate_retest(retest))
    checks.extend(validate_sweep(first, retest))
    checks.extend(validate_sweep(confirmation, retest))
    checks.extend(validate_pair(first, confirmation))
    failures = [check for check in checks if not check['passed']]
    if failures:
        print(f'validation failed: {len(failures)} of {len(checks)} checks', file=sys.stderr)
        for check in failures:
            print(f"  FAIL {check['name']}: {check['detail']}", file=sys.stderr)
        print('no summary written', file=sys.stderr)
        return 1

    summary = build_summary(first, confirmation, retest, protocol_path, checks)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    except OSError as exc:
        print(f'error writing {args.output}: {exc}', file=sys.stderr)
        return 1
    sys.stdout.write(format_report(summary) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
