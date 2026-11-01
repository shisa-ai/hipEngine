"""Ranking reports the flash owner pair rather than stale process inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_disabled_selector_ranking as ranking

E = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
L = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


@pytest.mark.parametrize('policy', [None, {'enabled': True, 'layers': '35-47'}])
def test_ranking_flash_owner_provenance(monkeypatch, tmp_path, policy):
    def run(command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'route_env': {E: '0', L: '2'},
            'bound_route_env': {E: '0', L: '2'}, 'qsa_flash_prefill_policy': policy}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ranking.subprocess, 'run', run)
    row = ranking._run_arm('qsa_flash', {E: '1', L: '35-47'}, {},
        model_root=tmp_path, case_id='x', fixture=tmp_path/'fixture.json', chunk_size=73,
        repetitions=1, compiler_version_file=tmp_path/'compiler.txt', work_root=tmp_path)
    assert row['qsa_flash_prefill_policy'] == policy
    assert row['effective_route'] == {}
    assert row['effective_route_missing_keys'] == ([] if policy is not None else [L, E])
    assert E not in row['baseline_route'] and L not in row['baseline_route']
