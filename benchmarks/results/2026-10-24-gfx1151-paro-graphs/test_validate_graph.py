"""Negative controls for the independent graph measurement validator."""
import copy
import json
from pathlib import Path
import unittest
from validate_graph import validate


class GraphValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(Path(__file__).with_name('smoke-initial.json').read_text())

    def test_completed_smoke(self):
        self.assertEqual(validate(self.data, 3), 18)

    def test_rejects_corrupted_records(self):
        def sample(data, arm):
            return next(s for s in data['cases'][0]['samples'] if s['arm'] == arm)
        changes = {
            'passed': lambda d: d.update(passed=False),
            'attribution': lambda d: d.update(attribution=True),
            'leak': lambda d: d['memory_after_close'].update(active_allocations=1),
            'missing_case': lambda d: d['cases'].pop(),
            'missing_call': lambda d: d['cases'][0]['samples'].pop(),
            'wrong_ids': lambda d: d['cases'][0]['graph_ids'].__setitem__(0, 0),
            'hash': lambda d: sample(d, 'graph').update(generated_ids_sha256='bad'),
            'ttft': lambda d: sample(d, 'graph').update(first_token_seconds=0),
            'throughput': lambda d: sample(d, 'graph').update(decode_tokens_per_second=1e9),
            'whole_rate': lambda d: sample(d, 'graph').update(whole_call_tokens_per_second=1e9),
            'arm': lambda d: sample(d, 'graph').update(arm='ar'),
            'used': lambda d: sample(d, 'ar')['route'].update(specdec2_mtp2_used=True),
            'acceptance': lambda d: sample(d, 'graph')['route']['specdec2_mtp2_accepted_counts'].__setitem__(0, 99),
            'depth': lambda d: sample(d, 'graph')['route']['specdec2_mtp2_candidate_counts'].__setitem__(0, 3),
            'recovered': lambda d: sample(d, 'graph')['fallback_reasons'].update(specdec2_mtp2_recovered_fake=1),
            'eager_mislabeled': lambda d: sample(d, 'eager')['route']['specdec2_mtp2_execution_routes'].__setitem__(0, 'graph'),
            'no_replay': lambda d: [v['graph'].update(status='captured', replayed=False) for v in sample(d, 'graph')['verification_events']],
            'fallback': lambda d: sample(d, 'graph')['verification_events'][0]['graph'].update(status='capture_failed_fallback'),
            'synchronized': lambda d: sample(d, 'graph')['verification_events'][0].update(seconds=0.1),
            'median': lambda d: d['cases'][0]['medians']['graph'].update(decode_tokens_per_second=1e9),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(AssertionError): validate(data, 3)


if __name__ == '__main__': unittest.main()
