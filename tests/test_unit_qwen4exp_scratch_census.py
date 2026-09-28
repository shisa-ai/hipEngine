from scripts.qwen4exp_scratch_census import parse_notes, reconcile


def test_metadata_private_segment_parser():
    text = '.name: _Zfoo\n    .private_segment_fixed_size: 36\n.name: plain\n    .private_segment_fixed_size: 0\n'
    assert parse_notes(text) == [
        {'mangled': '_Zfoo', 'private_segment_fixed_size': 36},
        {'mangled': 'plain', 'private_segment_fixed_size': 0},
    ]
    assert parse_notes('') == []


def test_complete_match_requires_observed_metadata_equality():
    traces = [{'Kernel_Name': 'foo', 'Scratch_Size': '36'}]
    metadata = [{'symbol': 'foo', 'private_segment_fixed_size': 36}]
    assert reconcile(traces, metadata)['all_metadata_complete']
    assert not reconcile(traces, [])['project_metadata_complete']
    assert not reconcile(traces, metadata * 2)['project_metadata_complete']
    assert not reconcile(traces + [{'Kernel_Name': 'foo', 'Scratch_Size': '0'}], metadata)['project_metadata_complete']
    assert not reconcile(traces, [{'symbol': 'foo', 'private_segment_fixed_size': 0}])['project_metadata_complete']


def test_empty_and_runtime_only_are_not_complete():
    assert not reconcile([], [])['project_metadata_complete']
    trace = [{'Kernel_Name': '__amd_rocclr_copyBuffer', 'Scratch_Size': '0'}]
    assert not reconcile(trace, [])['project_metadata_complete']
    trace.append({'Kernel_Name': 'foo', 'Scratch_Size': '0'})
    report = reconcile(trace, [{'symbol': 'foo', 'private_segment_fixed_size': 0}])
    assert report['project_metadata_complete']
    assert not report['all_metadata_complete']


def test_unknown_runtime_lookalike_fails_closed():
    report = reconcile([{'Kernel_Name': '__amd_unknown', 'Scratch_Size': '0'}], [])
    assert report['rows'][0]['status'] == 'missing'
    assert not report['project_metadata_complete']
