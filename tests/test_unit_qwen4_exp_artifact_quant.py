"""Cold artifact selection must not confuse UD storage with Q4_K_M metadata."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine import LLM
from hipengine.models.qwen4_exp import QWEN4_EXP_GGUF


def _shards(monkeypatch, *, mutation=None):
    import hipengine.loading as loading

    rows = json.loads((Path(__file__).parent / 'fixtures/qwen4exp_ud_q4_k_xl_inventory.json').read_text())['tensors']
    if mutation:
        mutation(rows)
    tensors = tuple(SimpleNamespace(name=n, ggml_type=t, shape=tuple(s)) for n, t, s in rows)
    paths = [Path('/renamed/model-1.gguf'), Path('/renamed/model-2.gguf')]
    infos = [SimpleNamespace(path=p, architecture='qwen4exp', metadata={'general.file_type': 15}, tensors=ts)
             for p, ts in zip(paths, ((), tensors))]
    monkeypatch.setattr(loading, 'discover_gguf_files', lambda path: paths)
    monkeypatch.setattr(loading, 'load_gguf_index', lambda path: infos[paths.index(path)])
    return infos[0], infos


def test_auto_quant_uses_all_shards_not_filename_or_file_type(monkeypatch):
    first, infos = _shards(monkeypatch)
    # Real split GGUFs may store architecture only in the empty first shard.
    infos[1].architecture = None
    llm = LLM('/renamed/model.gguf', backend='cpu_reference')
    llm._weight_index, llm._model_plugin = first, QWEN4_EXP_GGUF
    assert llm.resolved_quant == 'gguf_ud_q4_k_xl'
    assert llm.quant == 'auto'


@pytest.mark.parametrize('mutation', [
    lambda rows: rows.pop(),
    lambda rows: rows[0].__setitem__(1, 999),
    lambda rows: rows[0][2].__setitem__(0, 123),
    lambda rows: rows.append(rows[0]),
])
def test_auto_unknown_inventory_fails_closed(monkeypatch, mutation):
    first, _ = _shards(monkeypatch, mutation=mutation)
    llm = LLM('/renamed/model.gguf')
    llm._weight_index, llm._model_plugin = first, QWEN4_EXP_GGUF
    with pytest.raises(ValueError, match='explicit.*quant'):
        _ = llm.resolved_quant
    assert llm._resolved_quant is None


def test_explicit_quant_does_not_infer_or_override(monkeypatch):
    first, _ = _shards(monkeypatch, mutation=lambda rows: rows.clear())
    llm = LLM('/renamed/model.gguf', quant='gguf_q4_k_m')
    llm._weight_index, llm._model_plugin = first, QWEN4_EXP_GGUF
    assert llm.resolved_quant == 'gguf_q4_k_m'


def _assert_auto_and_explicit_ud_select_identical_profile_manifest(monkeypatch):
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles

    register_qwen4_exp_gfx1151_profiles()
    first, _ = _shards(monkeypatch)
    engines = [LLM('/renamed/model.gguf', backend='hip_gfx1151', quant=q)
               for q in ('auto', 'gguf_ud_q4_k_xl')]
    for llm in engines:
        llm._weight_index, llm._model_plugin = first, QWEN4_EXP_GGUF
    assert engines[0].resolved_execution_profile == 'production'
    assert engines[0].execution_profile_manifest == engines[1].execution_profile_manifest
    assert engines[0].execution_profile_manifest_sha256 == engines[1].execution_profile_manifest_sha256


def test_auto_and_explicit_ud_select_identical_profile_manifest():
    import subprocess
    import sys

    subprocess.run([sys.executable, '-c',
        "import runpy, pytest; ns = runpy.run_path(" + repr(__file__) + "); "
        "ns['_assert_auto_and_explicit_ud_select_identical_profile_manifest'](pytest.MonkeyPatch())"
    ], check=True, timeout=60)


def test_mixed_architecture_shards_fail_closed(monkeypatch):
    first, infos = _shards(monkeypatch)
    infos[1].architecture = 'other'
    with pytest.raises(ValueError, match='explicit.*quant'):
        QWEN4_EXP_GGUF.resolve_default_quant(first, model_path='/renamed/model.gguf')
