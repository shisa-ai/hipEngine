#!/usr/bin/env python3
"""Tests for build_summary.py using synthetic fixtures.

Run with:
    python3 -m unittest test_build_summary -v
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_summary  # noqa: E402

COMMIT = 'a' * 40
MODEL = '/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16'
CATEGORIES = ('code', 'general_en', 'general_ja', 'mixed_ja_en')
CONTROL_LENGTHS = (512, 1024, 4096)
AR_BASE_SECONDS = 0.50
MTP_BASE_SECONDS = 0.25
MTP_CYCLES = 100
MTP_ACCEPTED = 27


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def token_hash(ids):
    return sha256_bytes(json.dumps(list(ids), separators=(',', ':')).encode())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def case_specs():
    specs = []
    for index in range(10):
        specs.append({'id': f'cat{index}', 'category': CATEGORIES[index % 4], 'split': 'category',
                      'prompt_tokens': 100 + index})
    for index in range(8):
        specs.append({'id': f'held{index}', 'category': CATEGORIES[index % 4], 'split': 'heldout',
                      'prompt_tokens': 200 + index})
    for length in CONTROL_LENGTHS:
        specs.append({'id': f'p{length}', 'category': 'length_control', 'split': 'control',
                      'prompt_tokens': length})
    return specs


def first_token_seconds(spec):
    if spec['split'] == 'control':
        return {512: 2.0, 1024: 4.0, 4096: 16.0}[spec['prompt_tokens']]
    return 1.0


def default_measured_seconds(case, arm, pair):
    base = AR_BASE_SECONDS if arm == 'ar' else MTP_BASE_SECONDS
    return base * (1.0 + 0.01 * pair)


def warmup_seconds(case, arm, pair):
    base = AR_BASE_SECONDS if arm == 'ar' else MTP_BASE_SECONDS
    return base * 5.0


def scaled_measured_seconds(natural_ar=1.0, natural_mtp=1.0, control=None):
    control = control or {}

    def seconds(case, arm, pair):
        base = AR_BASE_SECONDS if arm == 'ar' else MTP_BASE_SECONDS
        value = base * (1.0 + 0.01 * pair)
        if case['split'] == 'control':
            value *= control.get((case['prompt_tokens'], arm), 1.0)
        else:
            value *= natural_ar if arm == 'ar' else natural_mtp
        return value

    return seconds


def make_events(first_seconds, decode_seconds):
    events = []
    for index in range(128):
        if index == 0:
            seconds = first_seconds
        elif index == 127:
            seconds = first_seconds + decode_seconds
        else:
            seconds = first_seconds + decode_seconds * index / 127
        events.append({'tokens': index + 1, 'seconds': seconds})
    return events


def make_route(arm):
    route = {
        'request_id': 0,
        'native_greedy': True,
        'native_compact_prefill': True,
        'native_decode_steps': 0,
        'serial_decode_steps': 127,
        'serial_decode_fallback': arm == 'ar',
        'stable_model_slot': 0,
        'specdec2_mtp2_used': arm == 'mtp',
        'specdec2_mtp2_cycles': 0,
        'specdec2_mtp2_candidate_counts': [],
        'specdec2_mtp2_accepted_counts': [],
        'specdec2_mtp2_execution_routes': [],
        'specdec2_mtp2_provider_open_ms': 0.0,
        'specdec2_mtp2_prompt_prime_ms': 0.0,
        'specdec2_mtp2_candidate_device_handoffs': 0,
        'specdec2_mtp2_candidate_d2h_after_target': 0,
    }
    if arm == 'mtp':
        route.update({
            'serial_decode_fallback': False,
            'specdec2_mtp2_cycles': MTP_CYCLES,
            'specdec2_mtp2_candidate_counts': [1] * MTP_CYCLES,
            'specdec2_mtp2_accepted_counts': [1] * MTP_ACCEPTED + [0] * (MTP_CYCLES - MTP_ACCEPTED),
            'specdec2_mtp2_execution_routes': ['eager'] * MTP_CYCLES,
            'specdec2_mtp2_candidate_device_handoffs': MTP_CYCLES,
        })
    return route


def make_protocol(repo_root):
    prompts = []
    for name in ('a.jsonl', 'b.jsonl'):
        path = repo_root / 'benchmarks/prompts' / name
        prompts.append({'path': str(path.relative_to(repo_root)), 'sha256': sha256_bytes(path.read_bytes())})
    return {
        'schema': 1,
        'measurement_commit': COMMIT,
        'model': MODEL,
        'backend': 'hip_gfx1151',
        'quant': 'w4_paro',
        'kv_storage': 'bf16',
        'surface': 'LLM.stream_detailed for AR; LLM.stream_speculative_mtp_detailed for MTP',
        'arms': ['speculative_mtp=false', 'speculative_mtp=true, candidate budget 1'],
        'case_count': 21,
        'category_prompts': 10,
        'category_heldouts': 8,
        'synthetic_prompt_lengths': list(CONTROL_LENGTHS),
        'decode_tokens': 128,
        'warmups_per_case_per_arm': 2,
        'samples_per_case_per_arm': 5,
        'paired_order': 'AR/MTP for even pair indices; MTP/AR for odd pair indices',
        'confirmation': 'Repeat the complete command independently; publish the first sweep, never the faster one.',
        'confirmation_review': 'Investigate aggregate natural-suite rate differences above 5%, or '
                               'length-control median differences above 10%, before publishing. '
                               'Preserve both sweeps and all samples.',
        'sources': prompts,
    }


def embedded_protocol(protocol):
    return {
        'backend': protocol['backend'],
        'quant': protocol['quant'],
        'kv_storage': protocol['kv_storage'],
        'max_active_requests': 1,
        'max_sequence_length': 4352,
        'prefix_cache': 'off',
        'temperature': 0,
        'ignore_eos': True,
        'output_tokens': 128,
        'warmups_per_case_per_arm': 2,
        'samples_per_case_per_arm': 5,
        'order': 'AR then MTP on even pairs; MTP then AR on odd pairs',
        'surface': protocol['surface'],
        'decode_timing': 'Observed first-token to last-token wall; denominator is generated tokens minus '
                         'first observed cumulative count. Terminal duplicate chunks excluded.',
        'first_token_timing': 'Public call start to first token, including prefill and MTP prompt priming.',
        'whole_call_timing': 'Public call start through iterator exhaustion; model construction and snapshot '
                             'extraction excluded after warmup.',
        'profile': 'Shipping backend defaults; no gfx1151 PARO execution-profile plan resolves. '
                   'No arithmetic or variant override.',
        'cross_route_id_equality': 'Diagnostic, not a same-schedule repeatability check.',
        'sources': [dict(source) for source in protocol['sources']],
    }


def make_provenance(repo_root, driver_path, output_path):
    return {
        'kind': 'hipengine_artifact_provenance',
        'schema_version': 3,
        'collected_at': '2026-10-24T00:00:00+00:00',
        'host_name': 'gfx1151',
        'repo_root': str(repo_root),
        'hipengine_commit': COMMIT,
        'git_branch': None,
        'staged_dirty': False,
        'unstaged_dirty': False,
        'untracked_dirty': False,
        'dirty': False,
        'dirty_paths': [],
        'dirty_path_count': 0,
        'untracked_paths': [],
        'untracked_count': 0,
        'execution_affecting_dirty': False,
        'execution_affecting_paths': [],
        'configured_backend': 'hip_gfx1151',
        'resolved_backend': 'hip_gfx1151',
        'target_arch': 'gfx1151',
        'device_name': 'AMD Radeon 8060S Graphics',
        'model_path': MODEL,
        'model_revision': None,
        'model_fingerprint': {
            'algorithm': 'sha256-directory-manifest-v1',
            'value': 'b' * 64,
            'size_bytes': 100,
            'sampled_bytes': 10,
            'file_count': 2,
            'exists': True,
            'path_type': 'directory',
        },
        'quant': 'w4_paro',
        'kv_dtype': 'bf16',
        'command': [
            sys.executable,
            str(driver_path),
            '--model', MODEL,
            '--cases', 'all',
            '--warmups', '2',
            '--repetitions', '5',
            '--output', str(output_path),
        ],
        'environment': {'HIPENGINE_BACKEND': None, 'HIPENGINE_HIP_ARCH': 'gfx1151'},
        'rocm_version': 'HIP version: 7.15.26333-0000000',
        'hipcc_version': 'HIP version: 7.15.26333-0000000\nAMD clang version 23.0.0git',
        'build_profile': 'shipping public PARO staged adapter',
        'timing_protocol': 'Observed first-token to last-token wall; denominator is generated tokens minus '
                           'first observed cumulative count. Terminal duplicate chunks excluded.',
        'warmups': 2,
        'repetitions': 5,
        'profiler': {'enabled': False, 'reason': 'paired public latency measurement'},
    }


def make_sweep_data(protocol, provenance, driver_path, *, measured=None, ids_shift=0):
    measured = measured or default_measured_seconds
    data = {
        'schema': 1,
        'kind': 'public_paro_paired_streaming_retest',
        'protocol': embedded_protocol(protocol),
        'driver_sha256': sha256_bytes(Path(driver_path).read_bytes()),
        'cases': [],
        'passed': True,
        'memory_after_close': {'active_allocations': 0},
    }
    for spec in case_specs():
        ar_ids = [(index + ids_shift) % 4096 for index in range(128)]
        mtp_ids = list(ar_ids)
        record = {
            'id': spec['id'],
            'category': spec['category'],
            'split': spec['split'],
            'prompt_tokens': spec['prompt_tokens'],
            'prompt_ids_sha256': 'c' * 64,
            'ar_ids': ar_ids,
            'mtp_ids': mtp_ids,
            'ar_text': 'ar text',
            'mtp_text': 'mtp text',
            'samples': [],
        }
        hashes = {'ar': token_hash(ar_ids), 'mtp': token_hash(mtp_ids)}
        for pair in range(7):
            order = ('ar', 'mtp') if pair % 2 == 0 else ('mtp', 'ar')
            for arm in order:
                if pair < 2:
                    seconds = warmup_seconds(spec, arm, pair)
                else:
                    seconds = measured(spec, arm, pair)
                events = make_events(first_token_seconds(spec), seconds)
                decode_seconds = events[-1]['seconds'] - events[0]['seconds']
                whole_call_seconds = events[-1]['seconds'] + 0.05
                record['samples'].append({
                    'pair': pair,
                    'warmup': pair < 2,
                    'arm': arm,
                    'generated_tokens': 128,
                    'generated_ids_sha256': hashes[arm],
                    'first_token_seconds': events[0]['seconds'],
                    'decode_seconds': decode_seconds,
                    'decode_tokens': 127,
                    'decode_tokens_per_second': 127 / decode_seconds,
                    'whole_call_seconds': whole_call_seconds,
                    'whole_call_tokens_per_second': 128 / whole_call_seconds,
                    'token_events': events,
                    'route': make_route(arm),
                    'fallback_reasons': {},
                })
        record['ar_mtp_ids_equal'] = ar_ids == mtp_ids
        record['medians'] = {
            arm: {
                key: statistics.median(
                    sample[key] for sample in record['samples']
                    if sample['arm'] == arm and not sample['warmup']
                )
                for key in ('decode_tokens_per_second', 'first_token_seconds', 'whole_call_tokens_per_second')
            }
            for arm in ('ar', 'mtp')
        }
        data['cases'].append(record)
    data['provenance'] = provenance
    return data


def make_fixture(root, *, first_measured=None, confirmation_measured=None,
                 confirmation_ids_shift=0, confirmation_provenance_hook=None):
    repo = root / 'repo'
    prompts = repo / 'benchmarks/prompts'
    prompts.mkdir(parents=True)
    (prompts / 'a.jsonl').write_text('{"id": "cat0"}\n')
    (prompts / 'b.jsonl').write_text('{"id": "held0"}\n')
    driver = root / 'measure.py'
    driver.write_text('# synthetic measurement driver\n')
    protocol = make_protocol(repo)
    protocol_path = root / 'protocol.json'
    write_json(protocol_path, protocol)
    first_path = root / 'first.json'
    confirmation_path = root / 'confirmation.json'

    first_data = make_sweep_data(
        protocol,
        make_provenance(repo, driver, first_path),
        driver,
        measured=first_measured,
    )
    confirmation_data = make_sweep_data(
        protocol,
        make_provenance(repo, driver, confirmation_path),
        driver,
        measured=confirmation_measured,
        ids_shift=confirmation_ids_shift,
    )
    if confirmation_provenance_hook is not None:
        confirmation_provenance_hook(confirmation_data['provenance'])
    write_json(first_path, first_data)
    write_json(confirmation_path, confirmation_data)
    return {
        'root': root,
        'repo': repo,
        'driver': driver,
        'protocol': protocol_path,
        'first': first_path,
        'confirmation': confirmation_path,
        'first_data': first_data,
        'confirmation_data': confirmation_data,
    }


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = make_fixture(self.root)
        self.output = self.root / 'summary.json'

    def tearDown(self):
        self._tmp.cleanup()

    def run_builder(self, *, first=None, confirmation=None, output=None, protocol=None):
        output = output or self.output
        argv = [
            '--first', str(first or self.fixture['first']),
            '--confirmation', str(confirmation or self.fixture['confirmation']),
            '--output', str(output),
        ]
        if protocol is not None:
            argv.extend(['--protocol', str(protocol)])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = build_summary.main(argv)
        return code, stdout.getvalue(), stderr.getvalue(), output

    def rewrite_first(self, mutate):
        data = copy.deepcopy(self.fixture['first_data'])
        mutate(data)
        write_json(self.fixture['first'], data)
        return data

    def rewrite_confirmation(self, mutate):
        data = copy.deepcopy(self.fixture['confirmation_data'])
        mutate(data)
        write_json(self.fixture['confirmation'], data)
        return data

    def assert_rejected(self, code, stderr, output):
        self.assertEqual(code, 1, stderr)
        self.assertFalse(output.exists(), 'summary must not be written on rejection')
        self.assertIn('validation failed', stderr)
        self.assertIn('no summary written', stderr)

    def expect_rejection(self, **kwargs):
        code, _, stderr, output = self.run_builder(**kwargs)
        self.assert_rejected(code, stderr, output)

    def build_summary_json(self):
        code, stdout, stderr, output = self.run_builder()
        self.assertEqual(code, 0, stderr)
        self.assertTrue(output.exists())
        return json.loads(output.read_text()), stdout

    # ------------------------------------------------------------------
    # arithmetic

    def test_natural_pooled_rate_is_sum_over_sum(self):
        summary, _ = self.build_summary_json()
        data = json.loads(self.fixture['first'].read_text())
        natural = [case for case in data['cases'] if case['split'] in ('category', 'heldout')]
        for arm in ('ar', 'mtp'):
            samples = [sample for case in natural for sample in case['samples']
                       if sample['arm'] == arm and not sample['warmup']]
            expected = (sum(sample['decode_tokens'] for sample in samples)
                        / sum(sample['decode_seconds'] for sample in samples))
            block = summary['published']['natural_suite'][arm]
            self.assertEqual(block['samples'], 18 * 5)
            self.assertEqual(block['cases'], 18)
            self.assertAlmostEqual(block['decode_tokens_per_second'], expected, places=12)
            self.assertEqual(block['decode_tokens'], 127 * 90)

    def test_warmups_are_excluded_from_published_metrics(self):
        summary, _ = self.build_summary_json()
        data = json.loads(self.fixture['first'].read_text())
        natural = [case for case in data['cases'] if case['split'] in ('category', 'heldout')]
        for arm in ('ar', 'mtp'):
            all_samples = [sample for case in natural for sample in case['samples'] if sample['arm'] == arm]
            with_warmups = (sum(sample['decode_tokens'] for sample in all_samples)
                            / sum(sample['decode_seconds'] for sample in all_samples))
            published = summary['published']['natural_suite'][arm]['decode_tokens_per_second']
            self.assertFalse(abs(with_warmups - published) < 1e-6,
                             'warmup samples must change the pooled value if they were included')

    def test_per_split_and_category_rates(self):
        summary, _ = self.build_summary_json()
        data = json.loads(self.fixture['first'].read_text())
        for split in ('category', 'heldout'):
            cases = [case for case in data['cases'] if case['split'] == split]
            for arm in ('ar', 'mtp'):
                samples = [sample for case in cases for sample in case['samples']
                           if sample['arm'] == arm and not sample['warmup']]
                expected = (sum(sample['decode_tokens'] for sample in samples)
                            / sum(sample['decode_seconds'] for sample in samples))
                block = summary['published']['by_split'][split][arm]
                self.assertAlmostEqual(block['decode_tokens_per_second'], expected, places=12)
                self.assertEqual(block['cases'], len(cases))
        by_category = {block['category']: block for block in summary['published']['by_category']}
        natural = [case for case in data['cases'] if case['split'] in ('category', 'heldout')]
        for category in CATEGORIES:
            cases = [case for case in natural if case['category'] == category]
            for arm in ('ar', 'mtp'):
                samples = [sample for case in cases for sample in case['samples']
                           if sample['arm'] == arm and not sample['warmup']]
                expected = (sum(sample['decode_tokens'] for sample in samples)
                            / sum(sample['decode_seconds'] for sample in samples))
                self.assertAlmostEqual(by_category[category][arm]['decode_tokens_per_second'], expected, places=12)
                self.assertEqual(by_category[category]['cases'], len(cases))

    def test_whole_call_rate_is_sum_over_sum(self):
        summary, _ = self.build_summary_json()
        data = json.loads(self.fixture['first'].read_text())
        natural = [case for case in data['cases'] if case['split'] in ('category', 'heldout')]
        for arm in ('ar', 'mtp'):
            samples = [sample for case in natural for sample in case['samples']
                       if sample['arm'] == arm and not sample['warmup']]
            expected = (sum(sample['generated_tokens'] for sample in samples)
                        / sum(sample['whole_call_seconds'] for sample in samples))
            block = summary['published']['natural_suite'][arm]
            self.assertAlmostEqual(block['whole_call_tokens_per_second'], expected, places=12)
            self.assertEqual(block['whole_call_tokens'], 128 * 90)

    def test_candidate_acceptance_from_summed_counters(self):
        summary, _ = self.build_summary_json()
        acceptance = summary['published']['natural_suite']['mtp']['candidate_acceptance']
        self.assertEqual(acceptance['samples'], 90)
        self.assertEqual(acceptance['cycles'], MTP_CYCLES * 90)
        self.assertEqual(acceptance['candidates'], MTP_CYCLES * 90)
        self.assertEqual(acceptance['accepted'], MTP_ACCEPTED * 90)
        self.assertAlmostEqual(acceptance['accepted_per_candidate'], 0.27, places=12)
        self.assertEqual(acceptance['tokens_from_cycles'], 127 * 90)
        self.assertEqual(acceptance['accounting_delta'], 0)

    def test_length_control_medians(self):
        summary, _ = self.build_summary_json()
        data = json.loads(self.fixture['first'].read_text())
        controls = summary['published']['length_controls']
        self.assertEqual(sorted(controls), ['p1024', 'p4096', 'p512'])
        cases = {case['id']: case for case in data['cases']}
        for length in CONTROL_LENGTHS:
            entry = controls[f'p{length}']
            self.assertEqual(entry['prompt_tokens'], length)
            for arm in ('ar', 'mtp'):
                rates = [sample['decode_tokens_per_second'] for sample in cases[f'p{length}']['samples']
                         if sample['arm'] == arm and not sample['warmup']]
                self.assertEqual(len(rates), 5)
                self.assertAlmostEqual(entry[arm]['median_decode_tokens_per_second'],
                                       statistics.median(rates), places=9)
                self.assertEqual(entry[arm]['samples'], 5)

    def test_identity_counts(self):
        summary, _ = self.build_summary_json()
        identity = summary['published']['identity']
        self.assertEqual(identity['cases'], 21)
        self.assertEqual(identity['ar_mtp_ids_equal'], 21)
        self.assertEqual(identity['ar_mtp_ids_differ'], [])
        self.assertEqual(identity['by_split']['category']['cases'], 10)
        self.assertEqual(identity['by_split']['heldout']['cases'], 8)
        self.assertEqual(identity['by_split']['control']['cases'], 3)

    def test_inputs_link_paths_and_hashes(self):
        summary, _ = self.build_summary_json()
        inputs = summary['inputs']
        for key, path_key in (('first', 'first'), ('confirmation', 'confirmation'), ('protocol', 'protocol')):
            path = Path(self.fixture[path_key])
            self.assertEqual(Path(inputs[key]['path']), path.resolve())
            self.assertEqual(inputs[key]['sha256'], sha256_bytes(path.read_bytes()))
            self.assertEqual(inputs[key]['bytes'], len(path.read_bytes()))
        self.assertEqual(inputs['driver']['path'], str(self.fixture['driver']))
        self.assertEqual(inputs['driver']['sha256_recorded'], sha256_bytes(self.fixture['driver'].read_bytes()))
        self.assertEqual(inputs['driver']['sha256_now'], inputs['driver']['sha256_recorded'])
        self.assertEqual(len(inputs['sources']), 2)
        self.assertEqual(summary['published_sweep'], 'first')
        self.assertEqual(summary['provenance']['hipengine_commit'], COMMIT)

    def test_default_protocol_is_sibling_of_first(self):
        code, _, stderr, output = self.run_builder(protocol=None)
        self.assertEqual(code, 0, stderr)
        summary = json.loads(output.read_text())
        self.assertEqual(Path(summary['inputs']['protocol']['path']), self.fixture['protocol'].resolve())

    # ------------------------------------------------------------------
    # publication rule and drift flags

    def test_publishes_first_sweep_never_the_faster_one(self):
        faster = make_fixture(
            self.root / 'faster',
            confirmation_measured=scaled_measured_seconds(natural_ar=1 / 1.20, natural_mtp=1 / 1.20),
        )
        code, stdout, stderr, output = self.run_builder(
            first=faster['first'], confirmation=faster['confirmation'],
            protocol=faster['protocol'], output=self.root / 'faster-summary.json',
        )
        self.assertEqual(code, 0, stderr)
        summary = json.loads(output.read_text())
        first_data = json.loads(faster['first'].read_text())
        natural = [case for case in first_data['cases'] if case['split'] in ('category', 'heldout')]
        for arm in ('ar', 'mtp'):
            samples = [sample for case in natural for sample in case['samples']
                       if sample['arm'] == arm and not sample['warmup']]
            expected_first = (sum(sample['decode_tokens'] for sample in samples)
                              / sum(sample['decode_seconds'] for sample in samples))
            self.assertAlmostEqual(summary['published']['natural_suite'][arm]['decode_tokens_per_second'],
                                   expected_first, places=12)
            drift = summary['confirmation_drift']['natural_suite'][arm]['decode_tokens_per_second']
            self.assertGreater(drift['confirmation'], drift['first'])
            self.assertAlmostEqual(drift['relative_difference'], 0.20, places=6)
            self.assertTrue(drift['flagged'])

    def test_flags_natural_drift_above_five_percent(self):
        for scale, should_flag in ((1 / 1.06, True), (1 / 1.04, False)):
            with self.subTest(scale=scale):
                fixture = make_fixture(
                    self.root / f'drift-{scale:.4f}',
                    confirmation_measured=scaled_measured_seconds(natural_ar=scale, natural_mtp=scale),
                )
                output = self.root / f'drift-{scale:.4f}.json'
                code, _, stderr, _ = self.run_builder(
                    first=fixture['first'], confirmation=fixture['confirmation'],
                    protocol=fixture['protocol'], output=output,
                )
                self.assertEqual(code, 0, stderr)
                summary = json.loads(output.read_text())
                for arm in ('ar', 'mtp'):
                    drift = summary['confirmation_drift']['natural_suite'][arm]['decode_tokens_per_second']
                    self.assertEqual(drift['flagged'], should_flag)
                    self.assertAlmostEqual(abs(drift['relative_difference']), 0.06 if should_flag else 0.04, places=6)
                flags = summary['confirmation_drift']['flags']
                if should_flag:
                    self.assertTrue(any('natural_suite' in flag for flag in flags))
                else:
                    self.assertFalse(any('natural_suite' in flag for flag in flags))

    def test_flags_control_median_drift_above_ten_percent(self):
        for scale, should_flag in ((1 / 1.11, True), (1 / 1.09, False)):
            with self.subTest(scale=scale):
                fixture = make_fixture(
                    self.root / f'control-{scale:.4f}',
                    confirmation_measured=scaled_measured_seconds(control={(512, 'ar'): scale}),
                )
                output = self.root / f'control-{scale:.4f}.json'
                code, _, stderr, _ = self.run_builder(
                    first=fixture['first'], confirmation=fixture['confirmation'],
                    protocol=fixture['protocol'], output=output,
                )
                self.assertEqual(code, 0, stderr)
                summary = json.loads(output.read_text())
                ar_drift = summary['confirmation_drift']['length_controls']['p512']['ar'][
                    'median_decode_tokens_per_second']
                mtp_drift = summary['confirmation_drift']['length_controls']['p512']['mtp'][
                    'median_decode_tokens_per_second']
                self.assertEqual(ar_drift['flagged'], should_flag)
                self.assertFalse(mtp_drift['flagged'])
                self.assertAlmostEqual(abs(ar_drift['relative_difference']), 0.11 if should_flag else 0.09, places=6)
                flags = summary['confirmation_drift']['flags']
                self.assertEqual(any('length_controls.p512.ar' in flag for flag in flags), should_flag)

    def test_cross_sweep_id_mismatch_is_flagged_not_fatal(self):
        fixture = make_fixture(self.root / 'ids', confirmation_ids_shift=1)
        output = self.root / 'ids.json'
        code, _, stderr, _ = self.run_builder(
            first=fixture['first'], confirmation=fixture['confirmation'],
            protocol=fixture['protocol'], output=output,
        )
        self.assertEqual(code, 0, stderr)
        summary = json.loads(output.read_text())
        self.assertEqual(len(summary['confirmation_drift']['id_mismatches']), 42)
        self.assertTrue(any('id mismatches' in flag for flag in summary['confirmation_drift']['flags']))
        self.assertEqual(summary['published_sweep'], 'first')

    # ------------------------------------------------------------------
    # rejection: artifact shape

    def test_rejects_passed_false(self):
        self.rewrite_first(lambda data: data.update({'passed': False}))
        self.expect_rejection()

    def test_rejects_error_field(self):
        self.rewrite_first(lambda data: data.update({'error': 'AssertionError: boom'}))
        self.expect_rejection()

    def test_rejects_case_count(self):
        self.rewrite_first(lambda data: data['cases'].pop())
        self.expect_rejection()

    def test_rejects_duplicate_case_id(self):
        self.rewrite_first(lambda data: data['cases'][1].update({'id': data['cases'][0]['id']}))
        self.expect_rejection()

    def test_rejects_wrong_split_counts(self):
        self.rewrite_first(lambda data: data['cases'][0].update({'split': 'control'}))
        self.expect_rejection()

    def test_rejects_control_shape(self):
        self.rewrite_first(lambda data: data['cases'][-1].update({'prompt_tokens': 2048}))
        self.expect_rejection()

    def test_rejects_missing_measured_sample(self):
        def mutate(data):
            case = data['cases'][0]
            for index, sample in enumerate(case['samples']):
                if sample['arm'] == 'ar' and not sample['warmup']:
                    case['samples'].pop(index)
                    return
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_missing_warmup_sample(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'].pop(0))
        self.expect_rejection()

    def test_rejects_pair_order(self):
        def mutate(data):
            samples = data['cases'][0]['samples']
            samples[0], samples[1] = samples[1], samples[0]
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_extra_measured_sample(self):
        def mutate(data):
            case = data['cases'][0]
            duplicate = copy.deepcopy(case['samples'][2])
            duplicate['pair'] = 7
            duplicate['warmup'] = False
            case['samples'].append(duplicate)
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_generated_token_count(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'][0].update({'generated_tokens': 129}))
        self.expect_rejection()

    def test_rejects_id_hash_mismatch(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'][0].update({'generated_ids_sha256': 'd' * 64}))
        self.expect_rejection()

    def test_rejects_reference_id_length(self):
        self.rewrite_first(lambda data: data['cases'][0].update({'ar_ids': data['cases'][0]['ar_ids'][:64]}))
        self.expect_rejection()

    def test_rejects_decode_window_tamper(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'][0].update(
            {'decode_seconds': data['cases'][0]['samples'][0]['decode_seconds'] * 2}))
        self.expect_rejection()

    def test_rejects_decode_token_tamper(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'][0].update({'decode_tokens': 126}))
        self.expect_rejection()

    def test_rejects_medians_tamper(self):
        self.rewrite_first(lambda data: data['cases'][0]['medians']['ar'].update(
            {'decode_tokens_per_second': data['cases'][0]['medians']['ar']['decode_tokens_per_second'] * 1.5}))
        self.expect_rejection()

    def test_rejects_embedded_protocol_mismatch(self):
        self.rewrite_first(lambda data: data['protocol'].update({'samples_per_case_per_arm': 4}))
        self.expect_rejection()

    def test_rejects_retest_protocol_constants(self):
        protocol = json.loads(self.fixture['protocol'].read_text())
        protocol['decode_tokens'] = 64
        write_json(self.fixture['protocol'], protocol)
        self.expect_rejection()

    def test_rejects_retest_surface_without_mtp_entrypoint(self):
        protocol = json.loads(self.fixture['protocol'].read_text())
        protocol['surface'] = 'LLM.stream_detailed for both arms'
        write_json(self.fixture['protocol'], protocol)
        self.expect_rejection()

    # ------------------------------------------------------------------
    # rejection: routes

    def test_rejects_ar_with_speculative_cycles(self):
        def mutate(data):
            sample = data['cases'][0]['samples'][0]
            sample['route'].update({
                'specdec2_mtp2_used': True,
                'specdec2_mtp2_cycles': 5,
                'specdec2_mtp2_candidate_counts': [1] * 5,
                'specdec2_mtp2_accepted_counts': [0] * 5,
                'specdec2_mtp2_execution_routes': ['eager'] * 5,
            })
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_mtp_without_cycles(self):
        def mutate(data):
            sample = data['cases'][0]['samples'][1]
            sample['route'].update({
                'specdec2_mtp2_cycles': 0,
                'specdec2_mtp2_candidate_counts': [],
                'specdec2_mtp2_accepted_counts': [],
                'specdec2_mtp2_execution_routes': [],
            })
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_mtp_candidate_budget(self):
        def mutate(data):
            route = data['cases'][0]['samples'][1]['route']
            route['specdec2_mtp2_candidate_counts'] = [2] * route['specdec2_mtp2_cycles']
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_mtp_count_length_mismatch(self):
        def mutate(data):
            route = data['cases'][0]['samples'][1]['route']
            route['specdec2_mtp2_accepted_counts'] = route['specdec2_mtp2_accepted_counts'][:-1]
        self.rewrite_first(mutate)
        self.expect_rejection()

    def test_rejects_mtp_recovery_fallback(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'][1].update(
            {'fallback_reasons': {'specdec2_mtp2_recovered_eager': 3}}))
        self.expect_rejection()

    def test_rejects_mtp_non_eager_route(self):
        self.rewrite_first(lambda data: data['cases'][0]['samples'][1]['route'].update(
            {'specdec2_mtp2_execution_routes': ['eager'] * (MTP_CYCLES - 1) + ['graph']}))
        self.expect_rejection()

    # ------------------------------------------------------------------
    # rejection: provenance

    def test_rejects_dirty_provenance(self):
        self.rewrite_first(lambda data: data['provenance'].update(
            {'dirty': True, 'dirty_path_count': 1, 'execution_affecting_dirty': True,
             'execution_affecting_paths': ['hipengine/x.py']}))
        self.expect_rejection()

    def test_rejects_commit_mismatch(self):
        self.rewrite_first(lambda data: data['provenance'].update({'hipengine_commit': '1' * 40}))
        self.expect_rejection()

    def test_rejects_driver_hash_mismatch(self):
        self.rewrite_first(lambda data: data.update({'driver_sha256': 'e' * 64}))
        self.expect_rejection()

    def test_rejects_source_hash_mismatch(self):
        self.rewrite_first(lambda data: data['protocol']['sources'][0].update({'sha256': 'f' * 64}))
        self.expect_rejection()

    def test_rejects_missing_prompt_source(self):
        self.rewrite_first(lambda data: data['protocol']['sources'][0].update(
            {'path': 'benchmarks/prompts/missing.jsonl'}))
        self.expect_rejection()

    def test_rejects_live_allocations(self):
        self.rewrite_first(lambda data: data.update({'memory_after_close': {'active_allocations': 2}}))
        self.expect_rejection()

    def test_rejects_confirmation_commit_mismatch(self):
        self.rewrite_confirmation(lambda data: data['provenance'].update({'hipengine_commit': '2' * 40}))
        self.expect_rejection()

    def test_rejects_confirmation_command_mismatch(self):
        def mutate(data):
            command = data['provenance']['command']
            index = command.index('--repetitions')
            command[index + 1] = '3'
        self.rewrite_confirmation(mutate)
        self.expect_rejection()

    def test_rejects_missing_confirmation_file(self):
        code, _, stderr, output = self.run_builder(confirmation=self.root / 'absent.json')
        self.assertEqual(code, 1)
        self.assertIn('error:', stderr)
        self.assertFalse(output.exists())

    # ------------------------------------------------------------------
    # real incomplete smoke artifacts must be rejected, not accommodated

    def test_rejects_real_smoke_artifacts(self):
        results = HERE
        protocol = results / 'protocol.json'
        smoke_pairs = [('smoke.json', 'smoke-v2-failed.json'), ('smoke-v1-failed.json', 'smoke-v2-failed.json')]
        if not protocol.exists():
            self.skipTest('real protocol.json not present')
        for first_name, confirmation_name in smoke_pairs:
            first = results / first_name
            confirmation = results / confirmation_name
            if not first.exists() or not confirmation.exists():
                continue
            with self.subTest(first=first_name):
                output = self.root / f'smoke-{first_name}.json'
                code, _, stderr, _ = self.run_builder(
                    first=first, confirmation=confirmation, protocol=protocol, output=output,
                )
                self.assert_rejected(code, stderr, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
