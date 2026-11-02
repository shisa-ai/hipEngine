"""Selector-ranking reports Q4 owner choices, not ambient process values."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_disabled_selector_ranking as ranking

Q4_PREFILL = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'
Q4_EXACT = 'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'


@pytest.mark.parametrize('selected', [False, True, None])
def test_ranking_uses_child_q4_owner_records(monkeypatch, tmp_path, selected):
    def run(command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({
            'wall_summary': {}, 'route_env': {Q4_PREFILL: '0', Q4_EXACT: '1'},
            'bound_route_env': {Q4_PREFILL: '0', Q4_EXACT: '1'},
            'q4_iu8_prefill_enabled': selected,
            'q4_iu8_exact_enabled': selected,
        }))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ranking.subprocess, 'run', run)
    row = ranking._run_arm('q4', {Q4_PREFILL: '1', Q4_EXACT: '0'}, {},
        model_root=tmp_path, case_id='tiny', fixture=tmp_path / 'fixture',
        chunk_size=17, repetitions=1, compiler_version_file=tmp_path / 'compiler',
        work_root=tmp_path)
    assert row['q4_iu8_prefill_enabled'] is selected
    assert row['q4_iu8_exact_enabled'] is selected
    assert Q4_PREFILL not in row['effective_route']
    assert Q4_EXACT not in row['effective_route']
    assert Q4_PREFILL not in row['baseline_route']
    assert Q4_EXACT not in row['baseline_route']
    assert row['effective_route_missing_keys'] == (
        sorted([Q4_PREFILL, Q4_EXACT]) if selected is None else [])
