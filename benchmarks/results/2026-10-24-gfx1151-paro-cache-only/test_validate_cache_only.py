"""Mechanical regression checks for the paired measurement validator."""
import copy
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from validate_cache_only import digest, validate, main


def packet():
    ids=list(range(128))
    route={'specdec2_mtp2_used':True,'specdec2_mtp2_cycles':64,
           'specdec2_mtp2_accepted_counts':[1]*63+[0],
           'specdec2_mtp2_candidate_counts':[1]*64,
           'specdec2_mtp2_execution_routes':['eager']*64,
           'native_decode_steps':0,'serial_decode_steps':0}
    samples=[]
    for arm in ('full','cache_only'):
        samples.append({'arm':arm,'pair':0,'warmup':False,'generated_tokens':128,
                        'generated_ids_sha256':digest(ids),'decode_tokens':127,
                        'token_events':[{'tokens':1,'seconds':1.0},{'tokens':128,'seconds':3.0}],
                        'first_token_seconds':1.0,'decode_seconds':2.0,'whole_call_seconds':3.1,
                        'decode_tokens_per_second':63.5,'route':copy.deepcopy(route),'fallback_reasons':{}})
    return {'passed':True,'protocol':{'warmups_per_case_per_arm':0,'samples_per_case_per_arm':1},
            'memory_after_close':{'active_allocations':0,'current_allocated_bytes':0},
            'provenance':{'resolved_backend':'hip_gfx1151'},
            'cases':[{'id':'example','full_ids':ids,'cache_only_ids':ids.copy(),
                      'full_cache_only_ids_equal':True,'samples':samples}]}


class ValidatorTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate(packet(),expected_cases=1),2)

    def test_rejects_incomplete_or_inconsistent_measurements(self):
        changes=[
            lambda x:x.update(passed=False),
            lambda x:x['memory_after_close'].update(active_allocations=1),
            lambda x:x['memory_after_close'].update(current_allocated_bytes=1),
            lambda x:x['cases'][0]['cache_only_ids'].__setitem__(0,999),
            lambda x:x['cases'][0]['samples'].pop(),
            lambda x:x['cases'][0]['samples'][0].update(warmup=True),
            lambda x:x['cases'][0]['samples'][0].update(generated_ids_sha256='wrong'),
            lambda x:x['cases'][0]['samples'][0].update(decode_tokens_per_second=99),
            lambda x:x['cases'][0]['samples'][0].update(first_token_seconds=2),
            lambda x:x['cases'][0]['samples'][0].update(whole_call_seconds=2),
            lambda x:x['cases'][0]['samples'][0]['route'].update(specdec2_mtp2_used=False),
            lambda x:x['cases'][0]['samples'][0]['route'].update(serial_decode_steps=1),
            lambda x:x['cases'][0]['samples'][1]['route']['specdec2_mtp2_accepted_counts'].__setitem__(0,0),
            lambda x:x['cases'][0]['samples'][0]['fallback_reasons'].update(specdec2_mtp2_recovered_error=1),
        ]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                x=packet();change(x)
                with self.assertRaises(AssertionError):validate(x,expected_cases=1)

    def test_split_confirmation_preserves_interrupted_teardown_status(self):
        first=packet()
        base=first['cases'][0]
        base.update(split='category',prompt_ids_sha256='prompt')
        names=[f'case{i}' for i in range(19)]+['p1024','p4096']
        first['cases']=[{**copy.deepcopy(base),'id':name} for name in names]
        first['driver_sha256']='original'
        for key in ('hipengine_commit','model_fingerprint','device_name','host_name'):
            first['provenance'][key]='same'
        partial=copy.deepcopy(first)
        partial['cases']=partial['cases'][:19]
        partial['passed']=False
        partial.pop('memory_after_close');partial.pop('provenance')
        tail=copy.deepcopy(first)
        tail['cases']=tail['cases'][19:]
        tail['driver_sha256']='selector-copy'
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in ('first.json','partial.json','tail.json','summary.json')]
            for path,data in zip(paths,(first,partial,tail)):
                path.write_text(json.dumps(data))
            argv=['validate','--first',str(paths[0]),'--confirmation-partial',str(paths[1]),
                  '--confirmation-tail',str(paths[2]),'--output',str(paths[3])]
            with patch('sys.argv',argv),patch('builtins.print'):
                main()
            summary=json.loads(paths[3].read_text())
            self.assertEqual(summary['checked_calls'],84)
            self.assertIn('not observed',summary['confirmation_completion']['partial_process_teardown'])
            partial['cases'].pop()
            paths[1].write_text(json.dumps(partial))
            with patch('sys.argv',argv),self.assertRaises(AssertionError):
                main()

    def test_missing_case(self):
        with self.assertRaises(AssertionError):validate(packet())


if __name__=='__main__':
    unittest.main()
