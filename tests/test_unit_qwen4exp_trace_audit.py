from __future__ import annotations

import csv

import pytest

from scripts.qwen4exp_trace_audit import audit, interval_union_ns, resource_census


def test_interval_union_does_not_double_count_overlap():
    assert interval_union_ns([(10, 20), (15, 30), (30, 40), (50, 55)]) == 35
    assert interval_union_ns([]) == 0
    with pytest.raises(ValueError):
        interval_union_ns([(3, 2)])


def test_resource_census_preserves_unknown_scratch_and_multiple_launch_shapes():
    rows = [
        {'Kernel_Name': 'a', 'Start_Timestamp': '10', 'End_Timestamp': '20',
         'Scratch_Size': '36', 'VGPR_Count': '64', 'LDS_Block_Size': '0'},
        {'Kernel_Name': 'a', 'Start_Timestamp': '20', 'End_Timestamp': '40',
         'Scratch_Size': '0', 'VGPR_Count': '32', 'LDS_Block_Size': '128'},
        {'Kernel_Name': 'b', 'Start_Timestamp': '40', 'End_Timestamp': '50'},
    ]
    report = resource_census(rows)
    assert report[0]['kernel'] == 'a'
    assert report[0]['launches'] == 2
    assert report[0]['scratch_bytes_observed'] == [0, 36]
    assert report[0]['scratch_unknown_launches'] == 0
    assert report[1]['scratch_bytes_observed'] == []
    assert report[1]['scratch_unknown_launches'] == 1


def test_audit_scopes_each_repeat_and_excludes_warmup(tmp_path):
    def write(suffix, rows):
        with (tmp_path / ('trace_' + suffix + '.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write('marker_api_trace', [
        {'Function': 'decode_rep0', 'Start_Timestamp': 100, 'End_Timestamp': 200, 'Thread_Id': 1},
        {'Function': 'decode_rep1', 'Start_Timestamp': 300, 'End_Timestamp': 400, 'Thread_Id': 1},
        {'Function': 'qwen4exp_role:boundary', 'Start_Timestamp': 100, 'End_Timestamp': 400, 'Thread_Id': 1},
    ])
    write('hip_api_trace', [
        {'Function': 'hipLaunchKernel', 'Start_Timestamp': t, 'End_Timestamp': t+1,
         'Correlation_Id': t, 'Thread_Id': 1} for t in [0, 110, 310]
    ])
    write('kernel_trace', [
        {'Kernel_Name': 'k', 'Start_Timestamp': t+10, 'End_Timestamp': t+20,
         'Correlation_Id': t, 'Scratch_Size': 0} for t in [0, 110, 310]
    ])
    report = audit(tmp_path, 'decode_rep')
    assert len(report['windows']) == 2
    for window in report['windows']:
        assert window['roles']['kernel_rows'] == 1
        assert window['roles']['unattributed_ms'] == 0
        assert window['kernel_union_ms'] == 0.00001
        assert window['outside_kernel_union_ms'] == 0.00009
    with pytest.raises(ValueError, match='no measured windows'):
        audit(tmp_path, 'missing')
    assert report['windows'][0]['device_copy_rows'] is None
    assert report['windows'][0]['kernel_and_copy_union_ms'] is None

    write('memory_copy_trace', [
        {'Direction': 'MEMORY_COPY_HOST_TO_DEVICE', 'Start_Timestamp': 125,
         'End_Timestamp': 140},
    ])
    write('scratch_memory_trace', [
        {'Operation': 'SCRATCH_MEMORY_ALLOC', 'Start_Timestamp': 20,
         'End_Timestamp': 30, 'Allocation_Size': 1024},
    ])
    report = audit(tmp_path, 'decode_rep')
    # Kernel [120,130] overlaps copy [125,140]; union, not duration sum.
    assert report['windows'][0]['kernel_and_copy_union_ms'] == 0.00002
    assert report['windows'][0]['outside_kernel_and_copy_union_ms'] == 0.00008
    assert report['windows'][1]['device_copy_rows'] == []
    assert report['windows'][0]['scratch_operation_rows'] == []
    assert len(report['whole_capture_scratch_operations']) == 1
