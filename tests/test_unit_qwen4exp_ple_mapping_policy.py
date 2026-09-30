"""An instance-owned mapping hint must survive remap without affecting data."""

import mmap
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.qwen4_exp_materialize import Qwen4ExpPLEMMapTable
from tests._qwen4_exp_ple_fixtures import _iq4_nl_rows, _ple_tensor


def make_table():
    values = _iq4_nl_rows((1., 2., 3.))
    mappings = []

    class Raw:
        shape = values.shape

        def __init__(self):
            self.advice = []
            self._mmap = SimpleNamespace(madvise=self.advice.append, close=lambda: None)
            mappings.append(self)

        def __getitem__(self, ids):
            return values[ids]

    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda _: Raw()), _ple_tensor(3), semantic_rows=3,
    )
    return table, mappings


def test_mapping_policy_is_instance_owned_and_survives_remap():
    table, mappings = make_table()
    other, other_mappings = make_table()
    expected = table.gather_rows([2, 0, 2])
    assert table.configure_mapping_access("random")
    table.advise_cache("cold")
    assert mappings[0].advice == [mmap.MADV_RANDOM, mmap.MADV_DONTNEED]
    assert mappings[1].advice == [mmap.MADV_RANDOM]
    assert not other_mappings[0].advice
    assert table.gather_rows([2, 0, 2]).tobytes() == expected.tobytes()
    table.enable_telemetry()
    assert table.telemetry()["mapping_access_default"] == "random"
    assert table.telemetry()["mapping_access_applied"] is True
    table.close()
    other.close()


def test_explicit_normal_fallback_and_invalid_policy():
    table, mappings = make_table()
    table.configure_mapping_access("random")
    table.configure_mapping_access("normal")
    with pytest.raises(ValueError):
        table.configure_mapping_access("invalid")
    table.advise_cache("cold")
    assert mappings[-1].advice == [mmap.MADV_NORMAL]


def test_full_cache_warm_temporarily_restores_sequential_access():
    table, mappings = make_table()
    table.configure_mapping_access("random")
    table.warm_page_cache(chunk_rows=2)
    assert mappings[0].advice == [mmap.MADV_RANDOM, mmap.MADV_SEQUENTIAL, mmap.MADV_RANDOM]


def test_missing_mapping_capability_is_a_safe_fallback():
    values = _iq4_nl_rows((1., 2., 3.))
    table = Qwen4ExpPLEMMapTable(
        SimpleNamespace(tensor_data=lambda _: values), _ple_tensor(3), semantic_rows=3,
    )
    assert table.configure_mapping_access("random") is False
    np.testing.assert_array_equal(table.gather_rows([0]), np.full((1, 160), -127., np.float32))


def test_production_binder_selects_random_with_strict_and_explicit_rollback(monkeypatch):
    from hipengine.generation.qwen4_exp_profiles import _bind
    import os

    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS", raising=False)
    calls = []
    table = SimpleNamespace(configure_mapping_access=lambda mode: calls.append(mode))
    generator = SimpleNamespace(_resident=SimpleNamespace(ple_table=table))
    profile = SimpleNamespace(manifest_sha256="test", manifest={"quant": "gguf_ud_q4_k_xl"})
    _bind(generator, profile, production=True)
    # The UD quant is the arithmetic-recovery profile, which closes the runner's
    # colwarps gate (GDN_COLWARPS_PREFILL is a recovery flag). The tile16 variant is
    # read only inside that gate, so binding a route here would name something that
    # can never dispatch. The multi-column route is bound on the profiles that keep
    # the gate open instead - see the q4_k_m case below.
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == ""
    _bind(generator, profile, production=False)
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == ""
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS", "normal")
    _bind(generator, profile, production=True)
    assert calls == ["random", "normal", "normal"]
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS")
    profile.manifest["quant"] = "gguf_q4_k_m"
    _bind(generator, profile, production=True)
    # The multi-column route supersedes the DPP route (measured 1.34x-1.68x at
    # Hk16/Hv48/Dk=Dv=128 on gfx1151, NMSE ~1.8e-14). This is the profile whose
    # colwarps gate is open, so this is where the binding has to appear. This test
    # pins the binding, so it moves with the default.
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == (
        "qwen4exp_gdn_tiled16_multi_prefill")
    assert calls[-1] == "normal"
