"""Rejected H7Y mirrors must not remain a runtime or kernel surface."""
from dataclasses import fields
from pathlib import Path

from hipengine.kernels import hip_gfx1100
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import registered_keys
from hipengine.runtime.laguna_kv import LagunaKVLayerState


def test_rejected_h7y_capability_and_mirror_fields_are_absent():
    assert not hasattr(hip_gfx1100, 'LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS')
    assert not any('lane_major' in f.name for f in fields(LagunaKVLayerState))


def test_rejected_h7y_kernel_keys_are_absent():
    load_backend_kernel_package('hip_gfx1100')
    load_backend_kernel_package('hip_gfx1151')
    assert not [k for k in registered_keys()
                if k.layer in {'laguna_kv_write', 'laguna_attention_prefill'}
                and 'lane_major' in k.variant]


def test_rejected_h7y_exports_are_absent_from_device_source():
    source = (Path(__file__).resolve().parents[1] /
              'hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip').read_text()
    assert 'laguna_swa_write_kv_rows_natural_lane_major' not in source
    assert 'laguna_swa_attention_prefill_qrow4_dense_initial_lane_major' not in source
