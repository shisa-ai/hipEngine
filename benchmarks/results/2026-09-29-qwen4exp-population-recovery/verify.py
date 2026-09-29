#!/usr/bin/env python3
"""Validate the recovered synthetic-input router populations (CPU only)."""
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(Path(__file__).with_name('populations.json').read_text())
    script = root / artifact['capture_script']
    assert hashlib.sha256(script.read_bytes()).hexdigest() == artifact['capture_script_sha256']
    assert artifact['input']['kind'] == 'synthetic_token_id_sequence'
    assert artifact['input']['natural_code_prompt'] is False
    assert artifact['input']['stop_exclusive'] - artifact['input']['start_inclusive'] == 1024
    populations = artifact['populations']
    assert len(populations) == artifact['calls'] == 48
    for index, row in enumerate(populations):
        assert row['capture_index'] == index
        counts = row['counts']
        assert len(counts) == artifact['experts'] == 512
        assert all(type(value) is int and value >= 0 for value in counts)
        assert sum(counts) == 1024 * 10
        for tile in (1, 2, 4, 8, 16, 32, 64):
            assert sum((value + tile - 1) // tile for value in counts) == row['tile_sweep'][f'tile{tile}']
        assert row['tile_sweep']['tile16'] * 16 == row['wmma_total_rows_engine']
    assert mean(sum(value > 0 for value in row['counts']) for row in populations) == artifact['summary']['mean_active_experts']
    for tile in (16, 32):
        assert mean(row['tile_sweep'][f'tile{tile}'] for row in populations) == artifact['summary'][f'mean_tile{tile}']
    print('PASS: 48 populations, 512 experts each, 10240 assignments each; all tile maps agree')


if __name__ == '__main__':
    main()
