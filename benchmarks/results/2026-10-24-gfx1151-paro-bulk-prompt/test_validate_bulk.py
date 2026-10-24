"""Negative controls for the bulk prompt measurement validator."""
import copy
import unittest
from validate_bulk import digest, validate


def packet():
    ids=list(range(128))
    route={'specdec2_mtp2_used':True,'specdec2_mtp2_cycles':64,
           'specdec2_mtp2_accepted_counts':[1]*63+[0],
           'specdec2_mtp2_candidate_counts':[1]*64,
           'specdec2_mtp2_execution_routes':['eager']*64,
           'native_decode_steps':0,'serial_decode_steps':0}
    samples=[]
    for arm in ('serial','bulk'):
        samples.append({'arm':arm,'pair':0,'warmup':False,'generated_tokens':128,
                        'generated_ids_sha256':digest(ids),'decode_tokens':127,
                        'token_events':[{'tokens':1,'seconds':1.0},{'tokens':128,'seconds':3.0}],
                        'first_token_seconds':1.0,'decode_seconds':2.0,'whole_call_seconds':3.1,
                        'decode_tokens_per_second':63.5,'route':copy.deepcopy(route),'fallback_reasons':{},
                        'capture_seconds':[]})
        samples[-1]['route'].update(specdec2_mtp2_bulk_prompt_rows=36 if arm=='bulk' else 0,
                                    specdec2_mtp2_bulk_prompt_chunks=3 if arm=='bulk' else 0)
    return {'passed':True,'attribution':False,'protocol':{'warmups_per_case_per_arm':0,'samples_per_case_per_arm':1},
            'memory_after_close':{'active_allocations':0,'current_allocated_bytes':0},
            'provenance':{'resolved_backend':'hip_gfx1151'},
            'cases':[{'id':'example','prompt_tokens':37,'ids':{'serial':ids,'bulk':ids.copy()},'samples':samples}]}


class ValidatorTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate(packet(),expected_cases=1),2)

    def test_rejects_invalid(self):
        mutations=[
            lambda x:x.update(passed=False),lambda x:x.update(attribution=True),
            lambda x:x['memory_after_close'].update(active_allocations=1),
            lambda x:x['memory_after_close'].update(current_allocated_bytes=1),
            lambda x:x['cases'][0]['ids']['bulk'].__setitem__(0,999),
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
            lambda x:x['cases'][0]['samples'][0]['capture_seconds'].append(1),
            lambda x:x['cases'][0]['samples'][1]['route'].update(specdec2_mtp2_bulk_prompt_rows=0),
            lambda x:x['cases'][0]['samples'][1]['route'].update(specdec2_mtp2_bulk_prompt_chunks=0),
        ]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                data=packet();mutation(data)
                with self.assertRaises(AssertionError):validate(data,expected_cases=1)

if __name__=='__main__':unittest.main()
