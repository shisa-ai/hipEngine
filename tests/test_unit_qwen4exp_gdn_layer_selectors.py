"""Independent diagnostics carry GDN layer scope to their measured owner."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_disabled_selector_ranking as ranking
from scripts import qwen4exp_decode_sync_ab as sync

ENV = 'HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS'


@pytest.mark.parametrize('arm', ['gdn_colwarps', 'all_prefill'])
def test_ranking_transports_layer_scope_and_reads_child_report(monkeypatch, tmp_path, arm):
    layers = ranking._L(ranking.PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS)
    assert ENV not in ranking.ARMS[arm]
    assert ranking.OWNER_ARMS[arm]['gdn_colwarps_layers'] == layers

    def run(command, **kwargs):
        from scripts.qwen4exp_profile_gap import build_parser
        parsed = build_parser().parse_args(command[2:])
        assert parsed.gdn_colwarps_layers == layers
        Path(parsed.output).write_text(json.dumps({
            'gdn_colwarps_layers': layers, 'gdn_colwarps_prefill': True,
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ranking.subprocess, 'run', run)
    result = ranking._run_arm(
        arm, ranking.ARMS[arm], ranking.OWNER_ARMS[arm], model_root=tmp_path,
        case_id='probe', fixture=tmp_path / 'fixture.json', chunk_size=256,
        repetitions=1, compiler_version_file=tmp_path / 'compiler.txt', work_root=tmp_path)
    assert result['gdn_colwarps_layers'] == layers
    assert result['gdn_colwarps_prefill'] is True


def test_decode_sync_binds_and_restores_its_scope_without_process_write(monkeypatch):
    monkeypatch.setenv(ENV, '999')
    import os
    assert os.environ[ENV] == '999'
    owner = SimpleNamespace(gdn_colwarps_layers='13-47', gdn_colwarps_prefill=False,
                            runner=SimpleNamespace(gdn_colwarps_layers='13-47',
                                                   gdn_colwarps_prefill=False))
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            selected = sync.bind_production_gdn(stack, owner)
            assert selected == '27-47'
            assert owner.gdn_colwarps_layers == owner.runner.gdn_colwarps_layers == selected
            assert owner.gdn_colwarps_prefill is owner.runner.gdn_colwarps_prefill is True
            raise RuntimeError('injected')
    assert owner.gdn_colwarps_layers == owner.runner.gdn_colwarps_layers == '13-47'
    assert owner.gdn_colwarps_prefill is owner.runner.gdn_colwarps_prefill is False
